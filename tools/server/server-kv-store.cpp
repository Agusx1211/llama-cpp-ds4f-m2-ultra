#include "server-kv-store.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace server_kv_store {

struct lease_counter {
    std::atomic<size_t> value{ 0 };
};

namespace {

struct object_record {
    object_class                   storage_class = object_class::live;
    uint64_t                       charged_bytes = 0;
    uint64_t                       generation    = 0;
    uint64_t                       last_access   = 0;
    std::shared_ptr<lease_counter> leases        = std::make_shared<lease_counter>();
    llama_snapshot_identity        identity;
};

bool key_is_zero(const object_key & key) {
    return std::all_of(key.begin(), key.end(), [](uint8_t byte) { return byte == 0; });
}

bool is_lower_hex(const std::string & value, size_t size) {
    return value.size() == size && std::all_of(value.begin(), value.end(), [](char byte) {
               return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f');
           });
}

bool is_generation_name(const std::string & value) {
    return value.size() == 27 && value.rfind("generation-", 0) == 0 && is_lower_hex(value.substr(11), 16);
}

bool is_chunk_name(const std::string & value) {
    return value.size() == 19 && value.rfind("chunk-", 0) == 0 && value.compare(14, 5, ".pack") == 0 &&
           is_lower_hex(value.substr(6, 8), 8);
}

object_key parse_key(const std::string & text) {
    object_key key{};
    const auto nibble = [](char byte) -> uint8_t {
        return static_cast<uint8_t>(byte <= '9' ? byte - '0' : byte - 'a' + 10);
    };
    for (size_t index = 0; index < key.size(); ++index) {
        key[index] = static_cast<uint8_t>((nibble(text[index * 2]) << 4) | nibble(text[index * 2 + 1]));
    }
    return key;
}

const char * class_name(object_class storage_class) {
    return storage_class == object_class::live ? "live" : "prefix";
}

fs::path pool_path(const config & cfg) {
    return fs::u8path(cfg.snapshot.root_path);
}

fs::path class_path(const config & cfg, object_class storage_class) {
    return pool_path(cfg) / class_name(storage_class);
}

fs::path object_path(const config & cfg, object_class storage_class, const object_key & key) {
    return class_path(cfg, storage_class) / llama_snapshot_digest_hex(key);
}

llama_snapshot_store_config object_config(const config & cfg, object_class storage_class, const object_key & key) {
    llama_snapshot_store_config result = cfg.snapshot;
    result.root_path                   = object_path(cfg, storage_class, key).string();
    return result;
}

status fsync_directory(const fs::path & path, int * os_error = nullptr) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        if (os_error != nullptr) {
            *os_error = errno;
        }
        return status::io_error;
    }
    int error = 0;
    if (::fsync(descriptor) != 0) {
        error = errno;
    }
    if (::close(descriptor) != 0 && error == 0) {
        error = errno;
    }
    if (os_error != nullptr) {
        *os_error = error;
    }
    return error == 0 ? status::ok : status::io_error;
}

status ensure_directory(const fs::path & path) {
    std::error_code error;
    fs::create_directories(path, error);
    if (error) {
        return status::io_error;
    }
    const fs::file_status file_status = fs::symlink_status(path, error);
    return !error && fs::is_directory(file_status) && !fs::is_symlink(file_status) ? status::ok : status::io_error;
}

bool checked_add(uint64_t & target, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - target) {
        return false;
    }
    target += value;
    return true;
}

struct scanned_object {
    status                  store_status    = status::reconciliation_required;
    llama_snapshot_status   snapshot_status = llama_snapshot_status::invalid_argument;
    int                     os_error        = 0;
    uint64_t                charged_bytes   = 0;
    llama_snapshot_manifest manifest;
};

status count_generation_files(const fs::path & path, uint64_t & bytes, int & os_error) {
    std::error_code error;
    bool            has_manifest = false;
    for (fs::directory_iterator iterator(path, error), end; !error && iterator != end; iterator.increment(error)) {
        const fs::file_status file_status = iterator->symlink_status(error);
        if (error || fs::is_symlink(file_status) || !fs::is_regular_file(file_status)) {
            os_error = error.value();
            return status::reconciliation_required;
        }
        const std::string name = iterator->path().filename().string();
        if (name == "generation.manifest") {
            if (has_manifest) {
                return status::reconciliation_required;
            }
            has_manifest = true;
        } else if (!is_chunk_name(name)) {
            return status::reconciliation_required;
        }
        const uintmax_t file_bytes = iterator->file_size(error);
        if (error || file_bytes > std::numeric_limits<uint64_t>::max() ||
            !checked_add(bytes, static_cast<uint64_t>(file_bytes))) {
            os_error = error.value();
            return status::reconciliation_required;
        }
    }
    if (error) {
        os_error = error.value();
        return status::reconciliation_required;
    }
    return has_manifest ? status::ok : status::reconciliation_required;
}

scanned_object scan_object(const config & cfg, object_class storage_class, const object_key & key) {
    scanned_object                      result;
    const llama_snapshot_store_config   snapshot_cfg = object_config(cfg, storage_class, key);
    llama_snapshot_store                snapshot(snapshot_cfg);
    const llama_snapshot_cleanup_result cleanup = snapshot.cleanup_temporary_generations();
    if (cleanup.status != llama_snapshot_status::ok) {
        result.snapshot_status = cleanup.status;
        result.os_error        = cleanup.os_error;
        return result;
    }

    const fs::path  root = fs::u8path(snapshot_cfg.root_path);
    std::error_code error;
    bool            has_current = false;
    for (fs::directory_iterator iterator(root, error), end; !error && iterator != end; iterator.increment(error)) {
        const fs::file_status file_status = iterator->symlink_status(error);
        if (error || fs::is_symlink(file_status)) {
            result.os_error = error.value();
            return result;
        }
        const std::string name = iterator->path().filename().string();
        if (name == "current.manifest" && fs::is_regular_file(file_status)) {
            has_current                = true;
            const uintmax_t file_bytes = iterator->file_size(error);
            if (error || file_bytes > std::numeric_limits<uint64_t>::max() ||
                !checked_add(result.charged_bytes, static_cast<uint64_t>(file_bytes))) {
                result.os_error = error.value();
                return result;
            }
        } else if (is_generation_name(name) && fs::is_directory(file_status)) {
            const status counted = count_generation_files(iterator->path(), result.charged_bytes, result.os_error);
            if (counted != status::ok) {
                return result;
            }
        } else {
            return result;
        }
    }
    if (error) {
        result.os_error = error.value();
        return result;
    }
    if (!has_current) {
        result.snapshot_status = llama_snapshot_status::no_current_generation;
        return result;
    }
    const llama_snapshot_open_result opened = snapshot.inspect_current();
    if (opened.status != llama_snapshot_status::ok) {
        result.snapshot_status = opened.status;
        result.os_error        = opened.os_error;
        return result;
    }
    int                         validate_error = 0;
    const llama_snapshot_status validated      = snapshot.validate(opened.manifest, &validate_error);
    if (validated != llama_snapshot_status::ok) {
        result.snapshot_status = validated;
        result.os_error        = validate_error;
        return result;
    }
    result.store_status    = status::ok;
    result.snapshot_status = llama_snapshot_status::ok;
    result.manifest        = opened.manifest;
    return result;
}

class vector_source final : public llama_snapshot_chunk_source_i {
  public:
    explicit vector_source(const std::vector<uint8_t> & payload) : payload_(payload) {}

    llama_snapshot_chunk_source_result acquire(uint32_t, uint64_t offset, uint64_t size) noexcept override {
        if (offset > payload_.size() || size > payload_.size() - offset) {
            return { llama_snapshot_status::invalid_argument, nullptr, 0, 0 };
        }
        return { llama_snapshot_status::ok, payload_.data() + static_cast<size_t>(offset), size, 0 };
    }

    void release(uint32_t) noexcept override {}

  private:
    const std::vector<uint8_t> & payload_;
};

}  // namespace

struct shared_state {
    explicit shared_state(config value) : cfg(std::move(value)) {}

    config                               cfg;
    mutable std::mutex                   mutex;
    std::map<std::string, object_record> objects;
    stats                                counters;
    status                               ready        = status::reconciliation_required;
    uint64_t                             access_clock = 0;
};

namespace {

uint64_t committed_total(const stats & counters) {
    return counters.live_committed_bytes + counters.prefix_committed_bytes;
}

uint64_t reserved_total(const stats & counters) {
    return counters.live_reserved_bytes + counters.prefix_reserved_bytes;
}

void add_record(stats & counters, const object_record & record) {
    if (record.storage_class == object_class::live) {
        counters.live_committed_bytes += record.charged_bytes;
        ++counters.live_objects;
    } else {
        counters.prefix_committed_bytes += record.charged_bytes;
        ++counters.prefix_objects;
    }
}

void remove_record(stats & counters, const object_record & record) {
    if (record.storage_class == object_class::live) {
        counters.live_committed_bytes -= record.charged_bytes;
        --counters.live_objects;
    } else {
        counters.prefix_committed_bytes -= record.charged_bytes;
        --counters.prefix_objects;
    }
}

status validate_config(const config & cfg) {
    if (cfg.live_quota_bytes == 0 || cfg.prefix_quota_bytes > cfg.live_quota_bytes || cfg.max_live_objects == 0 ||
        cfg.max_prefix_objects == 0) {
        return status::invalid_config;
    }
    llama_snapshot_metadata metadata;
    metadata.snapshot_generation         = 1;
    metadata.request_generation          = 1;
    metadata.identity.architecture       = "validation";
    metadata.identity.target_kv_type     = "validation";
    metadata.identity.context_size       = 1;
    metadata.identity.raw_window         = 1;
    metadata.identity.c4_ratio           = 4;
    metadata.identity.hca_ratio          = 128;
    metadata.identity.dsv4_state_version = 1;
    metadata.identity.rollback_depth     = 1;
    metadata.identity.model_artifact_digest.fill(1);
    metadata.identity.tokenizer_digest.fill(1);
    metadata.identity.chat_template_digest.fill(1);
    metadata.identity.runtime_build_digest.fill(1);
    metadata.identity.target_kv_digest.fill(1);
    metadata.identity.rope_digest.fill(1);
    return llama_snapshot_estimate_storage(cfg.snapshot, metadata, 0).status == llama_snapshot_status::ok ?
               status::ok :
               status::invalid_config;
}

status erase_prefix_locked(shared_state &                                 state,
                           std::map<std::string, object_record>::iterator victim,
                           bool                                           inject_failure,
                           int *                                          os_error = nullptr) {
    if (inject_failure) {
        if (os_error != nullptr) {
            *os_error = EIO;
        }
        return status::io_error;
    }
    const object_key key  = parse_key(victim->first);
    const fs::path   path = object_path(state.cfg, object_class::prefix, key);
    std::error_code  error;
    fs::remove_all(path, error);
    if (error) {
        if (os_error != nullptr) {
            *os_error = error.value();
        }
        state.ready                          = status::reconciliation_required;
        state.counters.reconciliation_needed = true;
        return status::io_error;
    }
    int          sync_error = 0;
    const status synced     = fsync_directory(path.parent_path(), &sync_error);
    if (synced != status::ok) {
        if (os_error != nullptr) {
            *os_error = sync_error;
        }
        state.ready                          = status::reconciliation_required;
        state.counters.reconciliation_needed = true;
        return status::commit_uncertain;
    }
    remove_record(state.counters, victim->second);
    state.objects.erase(victim);
    ++state.counters.prefix_evictions;
    return status::ok;
}

std::map<std::string, object_record>::iterator oldest_disposable_prefix(shared_state &      state,
                                                                        const std::string & excluded_key) {
    auto best = state.objects.end();
    for (auto iterator = state.objects.begin(); iterator != state.objects.end(); ++iterator) {
        const object_record & candidate = iterator->second;
        if (iterator->first == excluded_key || candidate.storage_class != object_class::prefix ||
            candidate.leases->value.load(std::memory_order_acquire) != 0) {
            continue;
        }
        if (best == state.objects.end() || candidate.last_access < best->second.last_access ||
            (candidate.last_access == best->second.last_access && iterator->first < best->first)) {
            best = iterator;
        }
    }
    return best;
}

bool has_leased_prefix(const shared_state & state, const std::string & excluded_key) {
    for (const auto & item : state.objects) {
        if (item.first != excluded_key && item.second.storage_class == object_class::prefix &&
            item.second.leases->value.load(std::memory_order_acquire) != 0) {
            return true;
        }
    }
    return false;
}

void reserve(stats & counters, object_class storage_class, uint64_t bytes) {
    (storage_class == object_class::live ? counters.live_reserved_bytes : counters.prefix_reserved_bytes) += bytes;
}

void release(stats & counters, object_class storage_class, uint64_t bytes) {
    (storage_class == object_class::live ? counters.live_reserved_bytes : counters.prefix_reserved_bytes) -= bytes;
}

}  // namespace

const char * status_name(status value) {
    switch (value) {
        case status::ok:
            return "ok";
        case status::invalid_config:
            return "invalid_config";
        case status::invalid_argument:
            return "invalid_argument";
        case status::not_found:
            return "not_found";
        case status::class_conflict:
            return "class_conflict";
        case status::object_limit:
            return "object_limit";
        case status::stale_generation:
            return "stale_generation";
        case status::object_in_use:
            return "object_in_use";
        case status::live_quota_exceeded:
            return "live_quota_exceeded";
        case status::prefix_quota_exceeded:
            return "prefix_quota_exceeded";
        case status::blocked_by_prefix_lease:
            return "blocked_by_prefix_lease";
        case status::reconciliation_required:
            return "reconciliation_required";
        case status::io_error:
            return "io_error";
        case status::snapshot_error:
            return "snapshot_error";
        case status::commit_uncertain:
            return "commit_uncertain";
    }
    return "unknown";
}

read_lease::read_lease(std::shared_ptr<lease_counter> counter,
                       llama_snapshot_store_config    snapshot,
                       llama_snapshot_manifest        manifest) :
    counter_(std::move(counter)),
    snapshot_(std::move(snapshot)),
    manifest_(std::move(manifest)) {}

read_lease::~read_lease() {
    counter_->value.fetch_sub(1, std::memory_order_release);
}

const llama_snapshot_manifest & read_lease::manifest() const {
    return manifest_;
}

llama_snapshot_status read_lease::validate(int * os_error) const {
    return llama_snapshot_store(snapshot_).validate(manifest_, os_error);
}

llama_snapshot_read_result read_lease::read_all() const {
    return llama_snapshot_store(snapshot_).read_all(manifest_);
}

store::store(config cfg) : state_(std::make_shared<shared_state>(std::move(cfg))) {
    if (validate_config(state_->cfg) != status::ok) {
        state_->ready = status::invalid_config;
        return;
    }
    state_->ready = reconcile();
}

store::~store() = default;

status store::initialization_status() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->ready;
}

status store::reconcile() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (validate_config(state_->cfg) != status::ok) {
        state_->ready = status::invalid_config;
        return state_->ready;
    }
    for (const auto & item : state_->objects) {
        if (item.second.leases->value.load(std::memory_order_acquire) != 0) {
            return status::object_in_use;
        }
    }
    if (ensure_directory(pool_path(state_->cfg)) != status::ok ||
        ensure_directory(class_path(state_->cfg, object_class::live)) != status::ok ||
        ensure_directory(class_path(state_->cfg, object_class::prefix)) != status::ok) {
        state_->ready = status::io_error;
        return state_->ready;
    }

    std::error_code error;
    for (fs::directory_iterator iterator(pool_path(state_->cfg), error), end; !error && iterator != end;
         iterator.increment(error)) {
        const std::string     name        = iterator->path().filename().string();
        const fs::file_status file_status = iterator->symlink_status(error);
        if (error || fs::is_symlink(file_status) || !fs::is_directory(file_status) ||
            (name != "live" && name != "prefix")) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            return state_->ready;
        }
    }
    if (error) {
        state_->ready = status::io_error;
        return state_->ready;
    }

    std::map<std::string, object_record> scanned;
    stats                                counters;
    counters.prefix_evictions = state_->counters.prefix_evictions;
    uint64_t clock            = 0;
    for (object_class storage_class : { object_class::live, object_class::prefix }) {
        const fs::path root = class_path(state_->cfg, storage_class);
        for (fs::directory_iterator iterator(root, error), end; !error && iterator != end; iterator.increment(error)) {
            const std::string     name        = iterator->path().filename().string();
            const fs::file_status file_status = iterator->symlink_status(error);
            if (error || fs::is_symlink(file_status) || !fs::is_directory(file_status) || !is_lower_hex(name, 64)) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            const object_key key = parse_key(name);
            if (scanned.find(name) != scanned.end()) {
                state_->ready                          = status::class_conflict;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            const scanned_object object = scan_object(state_->cfg, storage_class, key);
            if (object.store_status != status::ok) {
                if (object.snapshot_status == llama_snapshot_status::no_current_generation &&
                    object.charged_bytes == 0) {
                    std::error_code remove_error;
                    fs::remove(iterator->path(), remove_error);
                    if (!remove_error && fsync_directory(root) == status::ok) {
                        continue;
                    }
                }
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            object_record record;
            record.storage_class = storage_class;
            record.charged_bytes = object.charged_bytes;
            record.generation    = object.manifest.snapshot_generation;
            record.identity      = object.manifest.identity;
            record.last_access   = ++clock;
            add_record(counters, record);
            scanned.emplace(name, std::move(record));
        }
        if (error) {
            state_->ready = status::io_error;
            return state_->ready;
        }
    }
    state_->objects      = std::move(scanned);
    state_->counters     = counters;
    state_->access_clock = clock;

    if (state_->counters.live_committed_bytes > state_->cfg.live_quota_bytes ||
        state_->counters.live_objects > state_->cfg.max_live_objects) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return state_->ready;
    }
    while (state_->counters.prefix_committed_bytes > state_->cfg.prefix_quota_bytes ||
           committed_total(state_->counters) > state_->cfg.live_quota_bytes ||
           state_->counters.prefix_objects > state_->cfg.max_prefix_objects) {
        auto victim = oldest_disposable_prefix(*state_, "");
        if (victim == state_->objects.end() || erase_prefix_locked(*state_, victim, false) != status::ok) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            return state_->ready;
        }
    }
    state_->counters.reconciliation_needed = false;
    state_->ready                          = status::ok;
    return state_->ready;
}

stats store::get_stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->counters;
}

write_result store::write_generation(const object_key &                  key,
                                     object_class                        storage_class,
                                     const llama_snapshot_metadata &     metadata,
                                     const std::vector<uint8_t> &        payload,
                                     const llama_snapshot_cancel_check & cancelled,
                                     const faults &                      injected) {
    vector_source source(payload);
    return write_generation_streamed(key, storage_class, metadata, payload.size(), source, cancelled, injected);
}

write_result store::write_generation_streamed(const object_key &                  key,
                                              object_class                        storage_class,
                                              const llama_snapshot_metadata &     metadata,
                                              uint64_t                            total_payload_bytes,
                                              llama_snapshot_chunk_source_i &     source,
                                              const llama_snapshot_cancel_check & cancelled,
                                              const faults &                      injected,
                                              const llama_snapshot_commit_fence & commit_fence) {
    write_result result;
    result.generation = metadata.snapshot_generation;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->ready != status::ok) {
        result.store_status =
            state_->ready == status::invalid_config ? status::invalid_config : status::reconciliation_required;
        return result;
    }
    if (key_is_zero(key)) {
        return result;
    }

    const std::string key_text = llama_snapshot_digest_hex(key);
    auto              existing = state_->objects.find(key_text);
    if (existing != state_->objects.end()) {
        if (existing->second.storage_class != storage_class) {
            result.store_status = status::class_conflict;
            return result;
        }
        if (existing->second.leases->value.load(std::memory_order_acquire) != 0) {
            result.store_status = status::object_in_use;
            return result;
        }
        if (metadata.snapshot_generation <= existing->second.generation) {
            result.store_status    = metadata.snapshot_generation < existing->second.generation ?
                                         status::stale_generation :
                                         status::snapshot_error;
            result.snapshot_status = metadata.snapshot_generation < existing->second.generation ?
                                         llama_snapshot_status::stale_generation :
                                         llama_snapshot_status::generation_exists;
            return result;
        }
        if (metadata.identity != existing->second.identity) {
            result.store_status    = status::snapshot_error;
            result.snapshot_status = llama_snapshot_status::identity_mismatch;
            return result;
        }
    }
    const size_t class_objects =
        storage_class == object_class::live ? state_->counters.live_objects : state_->counters.prefix_objects;
    const size_t class_limit =
        storage_class == object_class::live ? state_->cfg.max_live_objects : state_->cfg.max_prefix_objects;
    if (existing == state_->objects.end() && class_objects >= class_limit) {
        result.store_status = status::object_limit;
        return result;
    }

    const llama_snapshot_store_config     snapshot_cfg = object_config(state_->cfg, storage_class, key);
    const llama_snapshot_storage_estimate estimate =
        llama_snapshot_estimate_storage(snapshot_cfg, metadata, total_payload_bytes);
    result.snapshot_status = estimate.status;
    if (estimate.status != llama_snapshot_status::ok) {
        result.store_status = estimate.status == llama_snapshot_status::invalid_argument ? status::invalid_argument :
                                                                                           status::snapshot_error;
        return result;
    }
    const uint64_t candidate = estimate.replacement_peak_bytes;
    if (storage_class == object_class::live &&
        (candidate > state_->cfg.live_quota_bytes ||
         state_->counters.live_committed_bytes > state_->cfg.live_quota_bytes - candidate ||
         state_->counters.live_reserved_bytes >
             state_->cfg.live_quota_bytes - candidate - state_->counters.live_committed_bytes)) {
        result.store_status = status::live_quota_exceeded;
        return result;
    }
    if (storage_class == object_class::prefix && candidate > state_->cfg.prefix_quota_bytes) {
        result.store_status = status::prefix_quota_exceeded;
        return result;
    }

    const auto needs_eviction = [&]() {
        const bool total_over = candidate > state_->cfg.live_quota_bytes ||
                                committed_total(state_->counters) + reserved_total(state_->counters) >
                                    state_->cfg.live_quota_bytes - candidate;
        const bool prefix_over = storage_class == object_class::prefix &&
                                 (state_->counters.prefix_committed_bytes + state_->counters.prefix_reserved_bytes >
                                  state_->cfg.prefix_quota_bytes - candidate);
        return total_over || prefix_over;
    };
    while (needs_eviction()) {
        auto victim = oldest_disposable_prefix(*state_, key_text);
        if (victim == state_->objects.end()) {
            result.store_status = has_leased_prefix(*state_, key_text) ?
                                      status::blocked_by_prefix_lease :
                                      (storage_class == object_class::prefix ? status::prefix_quota_exceeded :
                                                                               status::live_quota_exceeded);
            return result;
        }
        const object_key evicted_key = parse_key(victim->first);
        const status     erased = erase_prefix_locked(*state_, victim, injected.fail_prefix_delete, &result.os_error);
        if (erased != status::ok) {
            result.store_status = erased;
            return result;
        }
        result.evicted_prefixes.push_back(evicted_key);
    }

    reserve(state_->counters, storage_class, candidate);
    llama_snapshot_store              snapshot(snapshot_cfg);
    const llama_snapshot_write_result written = snapshot.write_generation_streamed(
        metadata, total_payload_bytes, source, cancelled, injected.snapshot, commit_fence);
    result.snapshot_status = written.status;
    result.os_error        = written.os_error;
    result.committed       = written.committed;

    if (!written.committed) {
        const llama_snapshot_cleanup_result cleanup = snapshot.cleanup_temporary_generations();
        release(state_->counters, storage_class, candidate);
        if (cleanup.status != llama_snapshot_status::ok) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
        } else {
            if (existing == state_->objects.end()) {
                std::error_code remove_error;
                fs::remove(fs::u8path(snapshot_cfg.root_path), remove_error);
                if (remove_error) {
                    state_->ready                          = status::reconciliation_required;
                    state_->counters.reconciliation_needed = true;
                    result.store_status                    = status::reconciliation_required;
                    return result;
                }
            }
            result.store_status = status::snapshot_error;
        }
        return result;
    }

    const scanned_object scanned = scan_object(state_->cfg, storage_class, key);
    if (scanned.store_status != status::ok) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = status::commit_uncertain;
        return result;
    }
    release(state_->counters, storage_class, candidate);
    if (existing != state_->objects.end()) {
        remove_record(state_->counters, existing->second);
        existing->second.charged_bytes = scanned.charged_bytes;
        existing->second.generation    = scanned.manifest.snapshot_generation;
        existing->second.identity      = scanned.manifest.identity;
        existing->second.last_access   = ++state_->access_clock;
        add_record(state_->counters, existing->second);
    } else {
        object_record record;
        record.storage_class = storage_class;
        record.charged_bytes = scanned.charged_bytes;
        record.generation    = scanned.manifest.snapshot_generation;
        record.identity      = scanned.manifest.identity;
        record.last_access   = ++state_->access_clock;
        add_record(state_->counters, record);
        state_->objects.emplace(key_text, std::move(record));
    }
    result.charged_bytes = scanned.charged_bytes;
    if (written.status == llama_snapshot_status::commit_uncertain) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = status::commit_uncertain;
        return result;
    }
    result.store_status = written.status == llama_snapshot_status::ok ? status::ok : status::snapshot_error;
    return result;
}

open_result store::acquire(const object_key &              key,
                           object_class                    storage_class,
                           const llama_snapshot_identity & expected_identity) {
    open_result                 result;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->ready != status::ok) {
        result.store_status = status::reconciliation_required;
        return result;
    }
    const std::string key_text = llama_snapshot_digest_hex(key);
    auto              object   = state_->objects.find(key_text);
    if (object == state_->objects.end()) {
        result.store_status = status::not_found;
        return result;
    }
    if (object->second.storage_class != storage_class) {
        result.store_status = status::class_conflict;
        return result;
    }
    const llama_snapshot_store_config snapshot_cfg = object_config(state_->cfg, storage_class, key);
    const llama_snapshot_open_result  opened       = llama_snapshot_store(snapshot_cfg).open_current(expected_identity);
    result.snapshot_status                         = opened.status;
    result.os_error                                = opened.os_error;
    if (opened.status != llama_snapshot_status::ok ||
        opened.manifest.snapshot_generation != object->second.generation) {
        result.store_status =
            opened.status == llama_snapshot_status::ok ? status::reconciliation_required : status::snapshot_error;
        return result;
    }
    result.lease.reset(new read_lease(object->second.leases, snapshot_cfg, opened.manifest));
    object->second.leases->value.fetch_add(1, std::memory_order_release);
    object->second.last_access = ++state_->access_clock;
    result.store_status        = status::ok;
    return result;
}

erase_result store::erase(const object_key & key, object_class storage_class, uint64_t expected_generation) {
    erase_result                result;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->ready != status::ok) {
        result.store_status = status::reconciliation_required;
        return result;
    }
    const std::string key_text = llama_snapshot_digest_hex(key);
    auto              object   = state_->objects.find(key_text);
    if (object == state_->objects.end()) {
        result.store_status    = status::not_found;
        result.snapshot_status = llama_snapshot_status::no_current_generation;
        return result;
    }
    if (object->second.storage_class != storage_class) {
        result.store_status = status::class_conflict;
        return result;
    }
    if (object->second.generation != expected_generation) {
        result.store_status    = status::stale_generation;
        result.snapshot_status = llama_snapshot_status::stale_generation;
        return result;
    }
    if (object->second.leases->value.load(std::memory_order_acquire) != 0) {
        result.store_status = status::object_in_use;
        return result;
    }
    const fs::path  path = object_path(state_->cfg, storage_class, key);
    std::error_code error;
    fs::remove_all(path, error);
    if (error) {
        result.store_status                    = status::io_error;
        result.os_error                        = error.value();
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return result;
    }
    int sync_error = 0;
    if (fsync_directory(path.parent_path(), &sync_error) != status::ok) {
        result.store_status                    = status::commit_uncertain;
        result.os_error                        = sync_error;
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return result;
    }
    result.released_bytes = object->second.charged_bytes;
    remove_record(state_->counters, object->second);
    state_->objects.erase(object);
    result.snapshot_status = llama_snapshot_status::ok;
    result.store_status    = status::ok;
    return result;
}

}  // namespace server_kv_store

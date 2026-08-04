#include "server-kv-store.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace server_kv_store {

struct lease_counter {
    std::atomic<size_t> value{ 0 };
};

namespace detail {

bool checked_add(uint64_t & target, uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - target) {
        return false;
    }
    target += value;
    return true;
}

}  // namespace detail

namespace {

struct object_record {
    object_class                   storage_class = object_class::live;
    uint64_t                       charged_bytes = 0;
    uint64_t                       generation    = 0;
    uint64_t                       last_access   = 0;
    std::shared_ptr<lease_counter> leases        = std::make_shared<lease_counter>();
    llama_snapshot_identity        identity;
};

struct delete_intent_record {
    object_class storage_class = object_class::live;
    object_key   key{};
    uint64_t     generation = 0;
};

constexpr const char * AUTHORITY_LOCK_NAME        = ".server-kv.lock";
constexpr const char * GENERATION_CLOCK_NAME      = ".server-kv-generation";
constexpr const char * GENERATION_CLOCK_NEXT_NAME = ".server-kv-generation.next";
constexpr const char * DELETE_INTENT_NAME         = ".server-kv-delete-intent";

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

bool parse_hex_u64(const std::string & value, uint64_t & parsed) {
    if (!is_lower_hex(value, 16)) {
        return false;
    }
    parsed = 0;
    for (char byte : value) {
        parsed = parsed * 16 + static_cast<uint64_t>(byte <= '9' ? byte - '0' : byte - 'a' + 10);
    }
    return true;
}

bool parse_generation_suffix(const std::string & value, const std::string & prefix, uint64_t & generation) {
    if (value.size() != prefix.size() + 16 || value.rfind(prefix, 0) != 0 ||
        !parse_hex_u64(value.substr(prefix.size()), generation)) {
        return false;
    }
    return generation != 0;
}

object_key parse_key(const std::string & text);

std::string encode_generation_clock(uint64_t generation) {
    char buffer[128];
    std::snprintf(buffer, sizeof(buffer), "server-kv-generation-v1\nvalue=%016llx\ninverse=%016llx\n",
                  static_cast<unsigned long long>(generation), static_cast<unsigned long long>(~generation));
    return buffer;
}

bool decode_generation_clock(const std::string & value, uint64_t & generation) {
    constexpr const char * prefix = "server-kv-generation-v1\nvalue=";
    constexpr const char * middle = "\ninverse=";
    if (value.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t value_offset = std::char_traits<char>::length(prefix);
    const size_t middle_at    = value_offset + 16;
    if (value.size() != middle_at + std::char_traits<char>::length(middle) + 16 + 1 ||
        value.compare(middle_at, std::char_traits<char>::length(middle), middle) != 0 || value.back() != '\n') {
        return false;
    }
    uint64_t inverse = 0;
    if (!parse_hex_u64(value.substr(value_offset, 16), generation) ||
        !parse_hex_u64(value.substr(middle_at + std::char_traits<char>::length(middle), 16), inverse) ||
        inverse != ~generation) {
        return false;
    }
    return value == encode_generation_clock(generation);
}

std::string encode_delete_intent(const delete_intent_record & intent) {
    char buffer[256];
    std::snprintf(
        buffer, sizeof(buffer), "server-kv-delete-v1\nclass=%s\nkey=%s\ngeneration=%016llx\ninverse=%016llx\n",
        intent.storage_class == object_class::live ? "live" : "prefix", llama_snapshot_digest_hex(intent.key).c_str(),
        static_cast<unsigned long long>(intent.generation), static_cast<unsigned long long>(~intent.generation));
    return buffer;
}

bool decode_delete_intent(const std::string & value, delete_intent_record & intent) {
    constexpr const char * prefix = "server-kv-delete-v1\nclass=";
    if (value.rfind(prefix, 0) != 0) {
        return false;
    }
    const size_t class_at = std::char_traits<char>::length(prefix);
    size_t       key_at   = 0;
    if (value.compare(class_at, 9, "live\nkey=") == 0) {
        intent.storage_class = object_class::live;
        key_at               = class_at + 9;
    } else if (value.compare(class_at, 11, "prefix\nkey=") == 0) {
        intent.storage_class = object_class::prefix;
        key_at               = class_at + 11;
    } else {
        return false;
    }
    if (key_at + 64 > value.size() || !is_lower_hex(value.substr(key_at, 64), 64)) {
        return false;
    }
    intent.key                            = parse_key(value.substr(key_at, 64));
    constexpr const char * generation_tag = "\ngeneration=";
    const size_t           generation_at  = key_at + 64;
    if (value.compare(generation_at, std::char_traits<char>::length(generation_tag), generation_tag) != 0) {
        return false;
    }
    const size_t           generation_value_at = generation_at + std::char_traits<char>::length(generation_tag);
    constexpr const char * inverse_tag         = "\ninverse=";
    const size_t           inverse_at          = generation_value_at + 16;
    if (value.compare(inverse_at, std::char_traits<char>::length(inverse_tag), inverse_tag) != 0) {
        return false;
    }
    const size_t inverse_value_at = inverse_at + std::char_traits<char>::length(inverse_tag);
    if (value.size() != inverse_value_at + 16 + 1 || value.back() != '\n') {
        return false;
    }
    uint64_t inverse = 0;
    if (!parse_hex_u64(value.substr(generation_value_at, 16), intent.generation) || intent.generation == 0 ||
        !parse_hex_u64(value.substr(inverse_value_at, 16), inverse) || inverse != ~intent.generation) {
        return false;
    }
    return value == encode_delete_intent(intent);
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

status read_small_file(const fs::path & path, std::string & contents, bool & exists, int * os_error = nullptr) {
    exists               = false;
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
        if (errno == ENOENT) {
            return status::ok;
        }
        if (os_error != nullptr) {
            *os_error = errno;
        }
        return status::io_error;
    }
    struct stat file_status{};
    int         error = 0;
    if (::fstat(descriptor, &file_status) != 0) {
        error = errno;
    } else if (!S_ISREG(file_status.st_mode) || file_status.st_size < 0 || file_status.st_size > 512) {
        error = EIO;
    }
    if (error == 0) {
        contents.assign(static_cast<size_t>(file_status.st_size), '\0');
        size_t offset = 0;
        while (offset < contents.size()) {
            const ssize_t amount = ::read(descriptor, contents.data() + offset, contents.size() - offset);
            if (amount < 0 && errno == EINTR) {
                continue;
            }
            if (amount <= 0) {
                error = amount < 0 ? errno : EIO;
                break;
            }
            offset += static_cast<size_t>(amount);
        }
    }
    if (::close(descriptor) != 0 && error == 0) {
        error = errno;
    }
    if (os_error != nullptr) {
        *os_error = error;
    }
    exists = error == 0;
    return error == 0 ? status::ok : status::io_error;
}

status write_new_file_synced(const fs::path & path, const std::string & contents, int * os_error = nullptr) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0) {
        if (os_error != nullptr) {
            *os_error = errno;
        }
        return status::io_error;
    }
    int    error  = 0;
    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t amount = ::write(descriptor, contents.data() + offset, contents.size() - offset);
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount <= 0) {
            error = amount < 0 ? errno : EIO;
            break;
        }
        offset += static_cast<size_t>(amount);
    }
    if (error == 0 && ::fsync(descriptor) != 0) {
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

status ensure_directory(const fs::path & path, bool fail_parent_fsync) {
    std::error_code       error;
    const fs::file_status existing = fs::symlink_status(path, error);
    if (!error && fs::exists(existing)) {
        if (!fs::is_directory(existing) || fs::is_symlink(existing)) {
            return status::io_error;
        }
        const status child_synced = fsync_directory(path);
        if (child_synced != status::ok) {
            return child_synced;
        }
        if (fail_parent_fsync) {
            return status::commit_uncertain;
        }
        const fs::path parent = path.parent_path();
        return parent.empty() ? status::io_error : fsync_directory(parent);
    }
    if (error && error != std::errc::no_such_file_or_directory) {
        return status::io_error;
    }

    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return status::io_error;
    }
    const status parent_ready = ensure_directory(parent, false);
    if (parent_ready != status::ok) {
        return parent_ready;
    }
    if (!fs::create_directory(path, error) || error) {
        return status::io_error;
    }
    const status child_synced = fsync_directory(path);
    if (child_synced != status::ok) {
        return child_synced;
    }
    if (fail_parent_fsync) {
        return status::commit_uncertain;
    }
    return fsync_directory(parent);
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs ? std::numeric_limits<uint64_t>::max() : lhs + rhs;
}

struct scanned_object {
    status                  store_status    = status::reconciliation_required;
    llama_snapshot_status   snapshot_status = llama_snapshot_status::invalid_argument;
    int                     os_error        = 0;
    uint64_t                charged_bytes   = 0;
    bool                    layout_valid    = false;
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
            !detail::checked_add(bytes, static_cast<uint64_t>(file_bytes))) {
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
                !detail::checked_add(result.charged_bytes, static_cast<uint64_t>(file_bytes))) {
                result.os_error = error.value();
                return result;
            }
        } else if (is_generation_name(name) && fs::is_directory(file_status)) {
            const status counted = count_generation_files(iterator->path(), result.charged_bytes, result.os_error);
            if (counted != status::ok) {
                return result;
            }
        } else {
            result.os_error = error.value();
            return result;
        }
    }
    if (error) {
        result.os_error = error.value();
        return result;
    }
    result.layout_valid = true;
    if (!has_current) {
        result.store_status    = status::not_found;
        result.snapshot_status = llama_snapshot_status::no_current_generation;
        return result;
    }
    const llama_snapshot_open_result opened = snapshot.inspect_current();
    if (opened.status != llama_snapshot_status::ok) {
        result.snapshot_status = opened.status;
        result.os_error        = opened.os_error;
        return result;
    }
    result.manifest                            = opened.manifest;
    int                         validate_error = 0;
    const llama_snapshot_status validated      = snapshot.validate(opened.manifest, &validate_error);
    if (validated != llama_snapshot_status::ok) {
        result.snapshot_status = validated;
        result.os_error        = validate_error;
        return result;
    }
    result.store_status    = status::ok;
    result.snapshot_status = llama_snapshot_status::ok;
    return result;
}

status prune_object(const fs::path & root, uint64_t keep_generation, int * os_error = nullptr) {
    std::error_code error;
    bool            changed = false;
    for (fs::directory_iterator iterator(root, error), end; !error && iterator != end; iterator.increment(error)) {
        const std::string name       = iterator->path().filename().string();
        bool              remove     = name == "current.manifest" && keep_generation == 0;
        uint64_t          generation = 0;
        if (parse_generation_suffix(name, "generation-", generation)) {
            remove = generation != keep_generation;
        }
        if (!remove) {
            continue;
        }
        fs::remove_all(iterator->path(), error);
        if (error) {
            break;
        }
        changed = true;
    }
    if (error) {
        if (os_error != nullptr) {
            *os_error = error.value();
        }
        return status::io_error;
    }
    if (!changed) {
        return status::ok;
    }
    const status synced = fsync_directory(root, os_error);
    return synced == status::ok ? status::ok : status::commit_uncertain;
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

    ~shared_state() {
        if (authority_fd >= 0) {
            ::close(authority_fd);
        }
    }

    config                               cfg;
    mutable std::mutex                   mutex;
    std::map<std::string, object_record> objects;
    stats                                counters;
    status                               ready              = status::reconciliation_required;
    uint64_t                             access_clock       = 0;
    uint64_t                             durable_generation = 0;
    int                                  authority_fd       = -1;
};

namespace {

uint64_t committed_total(const stats & counters) {
    return saturating_add(counters.live_committed_bytes, counters.prefix_committed_bytes);
}

uint64_t reserved_total(const stats & counters) {
    return saturating_add(counters.live_reserved_bytes, counters.prefix_reserved_bytes);
}

bool add_record(stats & counters, const object_record & record) {
    if (record.storage_class == object_class::live) {
        if (!detail::checked_add(counters.live_committed_bytes, record.charged_bytes) ||
            counters.live_objects == std::numeric_limits<size_t>::max()) {
            counters.live_committed_bytes = std::numeric_limits<uint64_t>::max();
            return false;
        }
        ++counters.live_objects;
    } else {
        if (!detail::checked_add(counters.prefix_committed_bytes, record.charged_bytes) ||
            counters.prefix_objects == std::numeric_limits<size_t>::max()) {
            counters.prefix_committed_bytes = std::numeric_limits<uint64_t>::max();
            return false;
        }
        ++counters.prefix_objects;
    }
    return true;
}

bool remove_record(stats & counters, const object_record & record) {
    if (record.storage_class == object_class::live) {
        if (record.charged_bytes > counters.live_committed_bytes || counters.live_objects == 0) {
            return false;
        }
        counters.live_committed_bytes -= record.charged_bytes;
        --counters.live_objects;
    } else {
        if (record.charged_bytes > counters.prefix_committed_bytes || counters.prefix_objects == 0) {
            return false;
        }
        counters.prefix_committed_bytes -= record.charged_bytes;
        --counters.prefix_objects;
    }
    return true;
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

status acquire_authority(shared_state & state) {
    const fs::path lock_path  = pool_path(state.cfg) / AUTHORITY_LOCK_NAME;
    const int      descriptor = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (descriptor < 0) {
        return status::io_error;
    }
    if (::flock(descriptor, LOCK_EX | LOCK_NB) != 0) {
        const int error = errno;
        ::close(descriptor);
        return error == EWOULDBLOCK || error == EAGAIN ? status::authority_unavailable : status::io_error;
    }
    struct stat file_status{};
    if (::fstat(descriptor, &file_status) != 0 || !S_ISREG(file_status.st_mode) || file_status.st_size != 0 ||
        ::fsync(descriptor) != 0 || fsync_directory(pool_path(state.cfg)) != status::ok) {
        ::close(descriptor);
        return status::io_error;
    }
    state.authority_fd = descriptor;
    return status::ok;
}

status persist_generation_clock(shared_state & state, uint64_t generation, int * os_error = nullptr) {
    const fs::path root    = pool_path(state.cfg);
    const fs::path next    = root / GENERATION_CLOCK_NEXT_NAME;
    const status   written = write_new_file_synced(next, encode_generation_clock(generation), os_error);
    if (written != status::ok) {
        return written;
    }
    if (::rename(next.c_str(), (root / GENERATION_CLOCK_NAME).c_str()) != 0) {
        if (os_error != nullptr) {
            *os_error = errno;
        }
        return status::io_error;
    }
    const status synced = fsync_directory(root, os_error);
    if (synced != status::ok) {
        return status::commit_uncertain;
    }
    state.durable_generation = generation;
    return status::ok;
}

status recover_generation_clock(shared_state & state) {
    const fs::path root = pool_path(state.cfg);
    std::string    current_contents;
    std::string    next_contents;
    bool           current_exists = false;
    bool           next_exists    = false;
    if (read_small_file(root / GENERATION_CLOCK_NAME, current_contents, current_exists) != status::ok ||
        read_small_file(root / GENERATION_CLOCK_NEXT_NAME, next_contents, next_exists) != status::ok) {
        return status::reconciliation_required;
    }
    uint64_t current = 0;
    uint64_t next    = 0;
    if ((current_exists && !decode_generation_clock(current_contents, current)) ||
        (next_exists && !decode_generation_clock(next_contents, next))) {
        return status::reconciliation_required;
    }
    if (!current_exists && !next_exists) {
        return persist_generation_clock(state, 0);
    }
    if (next_exists) {
        std::error_code error;
        if (!current_exists || next >= current) {
            fs::rename(root / GENERATION_CLOCK_NEXT_NAME, root / GENERATION_CLOCK_NAME, error);
            current = next;
        } else {
            fs::remove(root / GENERATION_CLOCK_NEXT_NAME, error);
        }
        if (error || fsync_directory(root) != status::ok) {
            return status::reconciliation_required;
        }
    }
    state.durable_generation = current;
    return status::ok;
}

status allocate_generation(shared_state & state, uint64_t & generation, int * os_error = nullptr) {
    if (state.durable_generation == std::numeric_limits<uint64_t>::max()) {
        return status::object_limit;
    }
    generation = state.durable_generation + 1;
    return persist_generation_clock(state, generation, os_error);
}

status publish_delete_intent(const shared_state &         state,
                             const delete_intent_record & intent,
                             int *                        os_error = nullptr) {
    const status written =
        write_new_file_synced(pool_path(state.cfg) / DELETE_INTENT_NAME, encode_delete_intent(intent), os_error);
    if (written != status::ok) {
        return written;
    }
    const status synced = fsync_directory(pool_path(state.cfg), os_error);
    return synced == status::ok ? status::ok : status::commit_uncertain;
}

status remove_object_directory(const shared_state &         state,
                               const delete_intent_record & intent,
                               int *                        os_error = nullptr) {
    const fs::path        path = object_path(state.cfg, intent.storage_class, intent.key);
    std::error_code       error;
    const fs::file_status existing = fs::symlink_status(path, error);
    if (error && error != std::errc::no_such_file_or_directory) {
        if (os_error != nullptr) {
            *os_error = error.value();
        }
        return status::io_error;
    }
    if (!error && fs::exists(existing)) {
        if (!fs::is_directory(existing) || fs::is_symlink(existing)) {
            return status::reconciliation_required;
        }
        fs::remove_all(path, error);
        if (error) {
            if (os_error != nullptr) {
                *os_error = error.value();
            }
            return status::io_error;
        }
    }
    const status synced = fsync_directory(class_path(state.cfg, intent.storage_class), os_error);
    return synced == status::ok ? status::ok : status::commit_uncertain;
}

status clear_delete_intent(const shared_state & state, int * os_error = nullptr) {
    std::error_code error;
    const bool      removed = fs::remove(pool_path(state.cfg) / DELETE_INTENT_NAME, error);
    if (error || !removed) {
        if (os_error != nullptr) {
            *os_error = error ? error.value() : ENOENT;
        }
        return status::io_error;
    }
    const status synced = fsync_directory(pool_path(state.cfg), os_error);
    return synced == status::ok ? status::ok : status::commit_uncertain;
}

status recover_delete_intent(shared_state & state) {
    std::string contents;
    bool        exists = false;
    if (read_small_file(pool_path(state.cfg) / DELETE_INTENT_NAME, contents, exists) != status::ok) {
        return status::reconciliation_required;
    }
    if (!exists) {
        return status::ok;
    }
    delete_intent_record intent;
    if (!decode_delete_intent(contents, intent) || intent.generation > state.durable_generation) {
        return status::reconciliation_required;
    }
    if (remove_object_directory(state, intent) != status::ok || clear_delete_intent(state) != status::ok) {
        return status::reconciliation_required;
    }
    return status::ok;
}

status delete_object_locked(shared_state &                                 state,
                            std::map<std::string, object_record>::iterator object,
                            const faults &                                 injected,
                            int *                                          os_error = nullptr) {
    const delete_intent_record intent{ object->second.storage_class, parse_key(object->first),
                                       object->second.generation };
    const status               published = publish_delete_intent(state, intent, os_error);
    if (published != status::ok) {
        return published;
    }
    if (injected.fail_after_delete_intent) {
        if (os_error != nullptr) {
            *os_error = EIO;
        }
        return status::commit_uncertain;
    }
    const status removed = remove_object_directory(state, intent, os_error);
    if (removed != status::ok) {
        return removed;
    }
    if (injected.fail_after_delete_unlink) {
        if (os_error != nullptr) {
            *os_error = EIO;
        }
        return status::commit_uncertain;
    }
    return clear_delete_intent(state, os_error);
}

status erase_prefix_locked(shared_state &                                 state,
                           std::map<std::string, object_record>::iterator victim,
                           const faults &                                 injected,
                           int *                                          os_error = nullptr) {
    if (injected.fail_prefix_delete) {
        if (os_error != nullptr) {
            *os_error = EIO;
        }
        return status::io_error;
    }
    const status removed = delete_object_locked(state, victim, injected, os_error);
    if (removed != status::ok) {
        state.ready                          = status::reconciliation_required;
        state.counters.reconciliation_needed = true;
        return removed;
    }
    if (!remove_record(state.counters, victim->second)) {
        state.ready                          = status::reconciliation_required;
        state.counters.reconciliation_needed = true;
        return status::reconciliation_required;
    }
    state.objects.erase(victim);
    ++state.counters.prefix_evictions;
    return status::ok;
}

std::map<std::string, object_record>::iterator oldest_disposable_prefix(
    shared_state &                state,
    const std::string &           excluded_key,
    const std::set<std::string> * excluded = nullptr) {
    auto best = state.objects.end();
    for (auto iterator = state.objects.begin(); iterator != state.objects.end(); ++iterator) {
        const object_record & candidate = iterator->second;
        if (iterator->first == excluded_key || (excluded != nullptr && excluded->count(iterator->first) != 0) ||
            candidate.storage_class != object_class::prefix ||
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

bool reserve(stats & counters, object_class storage_class, uint64_t bytes) {
    uint64_t & reserved =
        storage_class == object_class::live ? counters.live_reserved_bytes : counters.prefix_reserved_bytes;
    return detail::checked_add(reserved, bytes);
}

bool release(stats & counters, object_class storage_class, uint64_t bytes) {
    uint64_t & reserved =
        storage_class == object_class::live ? counters.live_reserved_bytes : counters.prefix_reserved_bytes;
    if (bytes > reserved) {
        return false;
    }
    reserved -= bytes;
    return true;
}

status remove_empty_object_directory(const config &     cfg,
                                     object_class       storage_class,
                                     const object_key & key,
                                     int *              os_error = nullptr) {
    const fs::path  path = object_path(cfg, storage_class, key);
    std::error_code error;
    const bool      removed = fs::remove(path, error);
    if (error || !removed) {
        if (os_error != nullptr) {
            *os_error = error ? error.value() : ENOTEMPTY;
        }
        return status::io_error;
    }
    const status synced = fsync_directory(class_path(cfg, storage_class), os_error);
    return synced == status::ok ? status::ok : status::commit_uncertain;
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
        case status::authority_unavailable:
            return "authority_unavailable";
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

read_lease::read_lease(std::shared_ptr<shared_state>  authority,
                       std::shared_ptr<lease_counter> counter,
                       llama_snapshot_store_config    snapshot,
                       llama_snapshot_manifest        manifest) :
    authority_(std::move(authority)),
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
    const status root_ready = ensure_directory(pool_path(state_->cfg), state_->cfg.test_faults.fail_pool_parent_fsync);
    if (root_ready != status::ok) {
        state_->ready = root_ready;
        return;
    }
    const status authority = acquire_authority(*state_);
    if (authority != status::ok) {
        state_->ready = authority;
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
    if (state_->authority_fd < 0) {
        state_->ready = status::authority_unavailable;
        return state_->ready;
    }
    for (const auto & item : state_->objects) {
        if (item.second.leases->value.load(std::memory_order_acquire) != 0) {
            return status::object_in_use;
        }
    }
    const status live_ready =
        ensure_directory(class_path(state_->cfg, object_class::live), state_->cfg.test_faults.fail_class_parent_fsync);
    const status prefix_ready = ensure_directory(class_path(state_->cfg, object_class::prefix),
                                                 state_->cfg.test_faults.fail_class_parent_fsync);
    if (live_ready != status::ok || prefix_ready != status::ok) {
        state_->ready = live_ready != status::ok ? live_ready : prefix_ready;
        return state_->ready;
    }
    if (recover_generation_clock(*state_) != status::ok || recover_delete_intent(*state_) != status::ok) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return state_->ready;
    }

    std::error_code error;
    for (fs::directory_iterator iterator(pool_path(state_->cfg), error), end; !error && iterator != end;
         iterator.increment(error)) {
        const std::string     name           = iterator->path().filename().string();
        const fs::file_status file_status    = iterator->symlink_status(error);
        const bool            authority_lock = name == AUTHORITY_LOCK_NAME && fs::is_regular_file(file_status) &&
                                    iterator->file_size(error) == 0 && !error;
        const bool generation_clock = name == GENERATION_CLOCK_NAME && fs::is_regular_file(file_status) && !error;
        if (error || fs::is_symlink(file_status) ||
            (!authority_lock && !generation_clock &&
             (!fs::is_directory(file_status) || (name != "live" && name != "prefix")))) {
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
            scanned_object object = scan_object(state_->cfg, storage_class, key);
            if (!object.layout_valid) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            uint64_t keep_generation = 0;
            if (object.store_status == status::ok) {
                keep_generation = object.manifest.snapshot_generation;
            } else if (object.store_status != status::not_found) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            const status pruned = prune_object(iterator->path(), keep_generation);
            if (pruned != status::ok) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                return state_->ready;
            }
            if (keep_generation == 0) {
                std::error_code remove_error;
                fs::remove(iterator->path(), remove_error);
                if (remove_error || fsync_directory(root) != status::ok) {
                    state_->ready                          = status::reconciliation_required;
                    state_->counters.reconciliation_needed = true;
                    return state_->ready;
                }
                continue;
            }
            object = scan_object(state_->cfg, storage_class, key);
            if (object.store_status != status::ok || object.manifest.snapshot_generation != keep_generation ||
                keep_generation > state_->durable_generation) {
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
            if (!add_record(counters, record)) {
                state_->counters                       = counters;
                state_->counters.reconciliation_needed = true;
                state_->ready                          = status::reconciliation_required;
                return state_->ready;
            }
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
        if (victim == state_->objects.end() || erase_prefix_locked(*state_, victim, {}) != status::ok) {
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
    write_result                result;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->ready != status::ok) {
        result.store_status =
            state_->ready == status::invalid_config || state_->ready == status::authority_unavailable ?
                state_->ready :
                status::reconciliation_required;
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
        if (metadata.identity != existing->second.identity) {
            result.store_status    = status::snapshot_error;
            result.snapshot_status = llama_snapshot_status::identity_mismatch;
            return result;
        }
    }

    if (state_->durable_generation == std::numeric_limits<uint64_t>::max()) {
        result.store_status = status::object_limit;
        return result;
    }
    llama_snapshot_metadata assigned_metadata = metadata;
    assigned_metadata.snapshot_generation     = state_->durable_generation + 1;

    const llama_snapshot_store_config     snapshot_cfg = object_config(state_->cfg, storage_class, key);
    const llama_snapshot_storage_estimate estimate =
        llama_snapshot_estimate_storage(snapshot_cfg, assigned_metadata, total_payload_bytes);
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
    if (existing == state_->objects.end() && storage_class == object_class::live &&
        state_->counters.live_objects >= state_->cfg.max_live_objects) {
        result.store_status = status::object_limit;
        return result;
    }

    const auto needs_eviction = [&](const stats & counters) {
        const uint64_t total_used = saturating_add(committed_total(counters), reserved_total(counters));
        const bool     total_over =
            candidate > state_->cfg.live_quota_bytes || total_used > state_->cfg.live_quota_bytes - candidate;
        const uint64_t prefix_used = saturating_add(counters.prefix_committed_bytes, counters.prefix_reserved_bytes);
        const bool     prefix_over =
            storage_class == object_class::prefix && prefix_used > state_->cfg.prefix_quota_bytes - candidate;
        return total_over || prefix_over;
    };
    const auto prefix_count_over = [&](const stats & counters) {
        return existing == state_->objects.end() && storage_class == object_class::prefix &&
               counters.prefix_objects >= state_->cfg.max_prefix_objects;
    };

    stats                    projected = state_->counters;
    std::set<std::string>    planned_set;
    std::vector<std::string> planned_evictions;
    while (prefix_count_over(projected) || needs_eviction(projected)) {
        auto victim = oldest_disposable_prefix(*state_, key_text, &planned_set);
        if (victim == state_->objects.end()) {
            result.store_status = has_leased_prefix(*state_, key_text) ?
                                      status::blocked_by_prefix_lease :
                                      (storage_class == object_class::prefix ? status::prefix_quota_exceeded :
                                                                               status::live_quota_exceeded);
            return result;
        }
        if (!remove_record(projected, victim->second)) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
            return result;
        }
        planned_set.insert(victim->first);
        planned_evictions.push_back(victim->first);
    }

    const bool   new_object   = existing == state_->objects.end();
    const status object_ready = ensure_directory(fs::u8path(snapshot_cfg.root_path), injected.fail_object_parent_fsync);
    if (object_ready != status::ok) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = object_ready;
        return result;
    }

    const status generation_allocated = allocate_generation(*state_, result.generation, &result.os_error);
    if (generation_allocated != status::ok) {
        if (new_object) {
            (void) remove_empty_object_directory(state_->cfg, storage_class, key, &result.os_error);
        }
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = generation_allocated;
        return result;
    }
    assigned_metadata.snapshot_generation = result.generation;

    for (const std::string & victim_key : planned_evictions) {
        auto victim = state_->objects.find(victim_key);
        if (victim == state_->objects.end()) {
            // The planning pass and execution pass are serialized by the pool
            // mutex, but an out-of-band filesystem mutation can still remove
            // a planned victim.  Never leave the candidate directory behind
            // when that happens: publication cannot proceed and reconciliation
            // must observe a bounded, explicit state.
            if (new_object &&
                remove_empty_object_directory(state_->cfg, storage_class, key, &result.os_error) != status::ok) {
                result.os_error = result.os_error == 0 ? EIO : result.os_error;
            }
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
            return result;
        }
        const object_key evicted_key = parse_key(victim->first);
        const status     erased      = erase_prefix_locked(*state_, victim, injected, &result.os_error);
        if (erased != status::ok) {
            if (new_object &&
                remove_empty_object_directory(state_->cfg, storage_class, key, &result.os_error) != status::ok) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                result.store_status                    = status::reconciliation_required;
                return result;
            }
            result.store_status = erased;
            return result;
        }
        result.evicted_prefixes.push_back(evicted_key);
    }

    if (!reserve(state_->counters, storage_class, candidate)) {
        if (new_object &&
            remove_empty_object_directory(state_->cfg, storage_class, key, &result.os_error) != status::ok) {
            result.os_error = result.os_error == 0 ? EIO : result.os_error;
        }
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = status::reconciliation_required;
        return result;
    }
    llama_snapshot_store              snapshot(snapshot_cfg);
    const llama_snapshot_write_result written = snapshot.write_generation_streamed(
        assigned_metadata, total_payload_bytes, source, cancelled, injected.snapshot, commit_fence);
    result.snapshot_status = written.status;
    result.os_error        = written.os_error;
    result.committed       = written.committed;

    if (!written.committed) {
        const llama_snapshot_cleanup_result cleanup = snapshot.cleanup_temporary_generations();
        if (cleanup.status != llama_snapshot_status::ok) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
            return result;
        }
        const scanned_object after_failure = scan_object(state_->cfg, storage_class, key);
        const bool expected_existing = existing != state_->objects.end() && after_failure.store_status == status::ok &&
                                       after_failure.charged_bytes == existing->second.charged_bytes &&
                                       after_failure.manifest.snapshot_generation == existing->second.generation;
        const bool expected_empty = existing == state_->objects.end() && after_failure.layout_valid &&
                                    after_failure.store_status == status::not_found && after_failure.charged_bytes == 0;
        if (!expected_existing && !expected_empty) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
            return result;
        }
        if (expected_empty) {
            if (remove_empty_object_directory(state_->cfg, storage_class, key, &result.os_error) != status::ok) {
                state_->ready                          = status::reconciliation_required;
                state_->counters.reconciliation_needed = true;
                result.store_status                    = status::reconciliation_required;
                return result;
            }
        }
        if (!release(state_->counters, storage_class, candidate)) {
            state_->ready                          = status::reconciliation_required;
            state_->counters.reconciliation_needed = true;
            result.store_status                    = status::reconciliation_required;
            return result;
        }
        result.store_status = status::snapshot_error;
        return result;
    }

    const scanned_object scanned = scan_object(state_->cfg, storage_class, key);
    if (scanned.store_status != status::ok || scanned.manifest.snapshot_generation != result.generation) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = status::commit_uncertain;
        return result;
    }
    object_record record;
    record.storage_class = storage_class;
    record.charged_bytes = scanned.charged_bytes;
    record.generation    = scanned.manifest.snapshot_generation;
    record.identity      = scanned.manifest.identity;
    record.last_access   = ++state_->access_clock;
    if (existing != state_->objects.end()) {
        record.leases = existing->second.leases;
    }
    stats      updated  = state_->counters;
    const bool released = release(updated, storage_class, candidate);
    const bool removed  = existing == state_->objects.end() || remove_record(updated, existing->second);
    if (!released || !removed || !add_record(updated, record) ||
        updated.live_committed_bytes > state_->cfg.live_quota_bytes ||
        updated.prefix_committed_bytes > state_->cfg.prefix_quota_bytes ||
        committed_total(updated) > state_->cfg.live_quota_bytes) {
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        result.store_status                    = status::commit_uncertain;
        return result;
    }
    state_->counters = updated;
    if (existing != state_->objects.end()) {
        existing->second = std::move(record);
    } else {
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
        result.store_status =
            state_->ready == status::authority_unavailable ? state_->ready : status::reconciliation_required;
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
    result.lease.reset(new read_lease(state_, object->second.leases, snapshot_cfg, opened.manifest));
    object->second.leases->value.fetch_add(1, std::memory_order_release);
    object->second.last_access = ++state_->access_clock;
    result.store_status        = status::ok;
    return result;
}

erase_result store::erase(const object_key & key,
                          object_class       storage_class,
                          uint64_t           expected_generation,
                          const faults &     injected) {
    erase_result                result;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->ready != status::ok) {
        result.store_status =
            state_->ready == status::authority_unavailable ? state_->ready : status::reconciliation_required;
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
    const status removed = delete_object_locked(*state_, object, injected, &result.os_error);
    if (removed != status::ok) {
        result.store_status                    = removed;
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return result;
    }
    result.released_bytes = object->second.charged_bytes;
    if (!remove_record(state_->counters, object->second)) {
        result.store_status                    = status::reconciliation_required;
        state_->ready                          = status::reconciliation_required;
        state_->counters.reconciliation_needed = true;
        return result;
    }
    state_->objects.erase(object);
    result.snapshot_status = llama_snapshot_status::ok;
    result.store_status    = status::ok;
    return result;
}

}  // namespace server_kv_store

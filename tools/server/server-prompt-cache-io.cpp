#include "server-prompt-cache-io.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <utility>

namespace {

constexpr uint64_t     PCACHE_IO_HASH_P1     = 0x9E3779B185EBCA87ull;
constexpr uint64_t     PCACHE_IO_HASH_P2     = 0xC2B2AE3D27D4EB4Full;
constexpr uint64_t     PCACHE_IO_HASH_P3     = 0x165667B19E3779F9ull;
constexpr uint64_t     PCACHE_IO_HASH_P4     = 0x85EBCA77C2B2AE63ull;
constexpr uint64_t     PCACHE_IO_HASH_P5     = 0x27D4EB2F165667C5ull;
constexpr size_t       PCACHE_IO_WRITE_CHUNK = 1u << 20;
constexpr const char * PCACHE_IO_TEMP_PREFIX = ".pcache-io-";
constexpr const char * PCACHE_IO_TEMP_SUFFIX = ".tmp";

uint64_t rotl64(uint64_t value, int bits) {
    return (value << bits) | (value >> (64 - bits));
}

uint64_t load_u64(const uint8_t * data) {
    uint64_t value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

uint64_t hash_round(uint64_t accumulator, uint64_t value) {
    return rotl64(accumulator + value * PCACHE_IO_HASH_P2, 31) * PCACHE_IO_HASH_P1;
}

uint64_t hash_merge(uint64_t accumulator, uint64_t value) {
    return (accumulator ^ (rotl64(value * PCACHE_IO_HASH_P2, 31) * PCACHE_IO_HASH_P1)) * PCACHE_IO_HASH_P1 +
           PCACHE_IO_HASH_P4;
}

bool add_overflow(uint64_t left, uint64_t right, uint64_t & result) {
    if (left > UINT64_MAX - right) {
        return true;
    }
    result = left + right;
    return false;
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

uint64_t milliseconds_to_nanoseconds(uint64_t milliseconds) {
    if (milliseconds > UINT64_MAX / 1000000ull) {
        return UINT64_MAX;
    }
    return milliseconds * 1000000ull;
}

bool is_private_temp_name(const std::string & path) {
    const size_t separator   = path.find_last_of('/');
    const size_t name_begin  = separator == std::string::npos ? 0 : separator + 1;
    const size_t name_size   = path.size() - name_begin;
    const size_t suffix_size = std::strlen(PCACHE_IO_TEMP_SUFFIX);
    const size_t prefix_size = std::strlen(PCACHE_IO_TEMP_PREFIX);
    return name_size > prefix_size + suffix_size && path.compare(name_begin, prefix_size, PCACHE_IO_TEMP_PREFIX) == 0 &&
           path.compare(path.size() - suffix_size, suffix_size, PCACHE_IO_TEMP_SUFFIX) == 0;
}

std::string parent_directory(const std::string & path) {
    const size_t separator = path.find_last_of('/');
    if (separator == std::string::npos) {
        return ".";
    }
    if (separator == 0) {
        return "/";
    }
    return path.substr(0, separator);
}

server_prompt_cache_io_status status_from_errno(int error) {
    return error == ENOSPC || error == EDQUOT ? server_prompt_cache_io_status::no_space :
                                                server_prompt_cache_io_status::io_error;
}

bool valid_faults(const server_prompt_cache_io_faults & faults) {
    return faults.max_write_size != 0 && (faults.fail_after_bytes == UINT64_MAX || faults.fail_write_errno != 0);
}

}  // namespace

const char * server_prompt_cache_io_lifecycle_name(server_prompt_cache_io_lifecycle state) {
    switch (state) {
        case server_prompt_cache_io_lifecycle::ram_ready:
            return "ram_ready";
        case server_prompt_cache_io_lifecycle::queued:
            return "queued";
        case server_prompt_cache_io_lifecycle::writing:
            return "writing";
        case server_prompt_cache_io_lifecycle::durable:
            return "durable";
        case server_prompt_cache_io_lifecycle::retired:
            return "retired";
        case server_prompt_cache_io_lifecycle::write_failed:
            return "write_failed";
        case server_prompt_cache_io_lifecycle::commit_uncertain:
            return "commit_uncertain";
    }
    return "unknown";
}

const char * server_prompt_cache_io_status_name(server_prompt_cache_io_status status) {
    switch (status) {
        case server_prompt_cache_io_status::ok:
            return "ok";
        case server_prompt_cache_io_status::invalid_argument:
            return "invalid_argument";
        case server_prompt_cache_io_status::allocation_failed:
            return "allocation_failed";
        case server_prompt_cache_io_status::not_found:
            return "not_found";
        case server_prompt_cache_io_status::sealed:
            return "sealed";
        case server_prompt_cache_io_status::job_limit:
            return "job_limit";
        case server_prompt_cache_io_status::byte_limit:
            return "byte_limit";
        case server_prompt_cache_io_status::disk_limit:
            return "disk_limit";
        case server_prompt_cache_io_status::completion_limit:
            return "completion_limit";
        case server_prompt_cache_io_status::generation_not_newer:
            return "generation_not_newer";
        case server_prompt_cache_io_status::path_conflict:
            return "path_conflict";
        case server_prompt_cache_io_status::cooldown:
            return "cooldown";
        case server_prompt_cache_io_status::cancel_requested:
            return "cancel_requested";
        case server_prompt_cache_io_status::too_late:
            return "too_late";
        case server_prompt_cache_io_status::stale_generation:
            return "stale_generation";
        case server_prompt_cache_io_status::checksum_mismatch:
            return "checksum_mismatch";
        case server_prompt_cache_io_status::io_error:
            return "io_error";
        case server_prompt_cache_io_status::no_space:
            return "no_space";
        case server_prompt_cache_io_status::cancelled:
            return "cancelled";
        case server_prompt_cache_io_status::commit_uncertain:
            return "commit_uncertain";
    }
    return "unknown";
}

uint64_t server_prompt_cache_io_checksum(const void * data, size_t size, uint64_t seed) {
    if (size != 0 && data == nullptr) {
        return 0;
    }

    static constexpr uint8_t empty  = 0;
    const auto *             cursor = size == 0 ? &empty : static_cast<const uint8_t *>(data);
    const auto *             end    = cursor + size;
    uint64_t                 hash;

    if (size >= 32) {
        uint64_t        lane1 = seed + PCACHE_IO_HASH_P1 + PCACHE_IO_HASH_P2;
        uint64_t        lane2 = seed + PCACHE_IO_HASH_P2;
        uint64_t        lane3 = seed;
        uint64_t        lane4 = seed - PCACHE_IO_HASH_P1;
        const uint8_t * limit = end - 32;
        do {
            lane1 = hash_round(lane1, load_u64(cursor));
            lane2 = hash_round(lane2, load_u64(cursor + 8));
            lane3 = hash_round(lane3, load_u64(cursor + 16));
            lane4 = hash_round(lane4, load_u64(cursor + 24));
            cursor += 32;
        } while (cursor <= limit);
        hash = rotl64(lane1, 1) + rotl64(lane2, 7) + rotl64(lane3, 12) + rotl64(lane4, 18);
        hash = hash_merge(hash, lane1);
        hash = hash_merge(hash, lane2);
        hash = hash_merge(hash, lane3);
        hash = hash_merge(hash, lane4);
    } else {
        hash = seed + PCACHE_IO_HASH_P5;
    }

    hash += size;
    while (cursor + 8 <= end) {
        hash = rotl64(hash ^ (rotl64(load_u64(cursor) * PCACHE_IO_HASH_P2, 31) * PCACHE_IO_HASH_P1), 27) *
                   PCACHE_IO_HASH_P1 +
               PCACHE_IO_HASH_P4;
        cursor += 8;
    }
    while (cursor < end) {
        hash = rotl64(hash ^ (*cursor * PCACHE_IO_HASH_P5), 11) * PCACHE_IO_HASH_P1;
        ++cursor;
    }
    hash ^= hash >> 33;
    hash *= PCACHE_IO_HASH_P2;
    hash ^= hash >> 29;
    hash *= PCACHE_IO_HASH_P3;
    hash ^= hash >> 32;
    return hash;
}

struct server_prompt_cache_io::impl {
    using key                  = std::pair<uint64_t, uint64_t>;
    using wake_callback_handle = std::shared_ptr<const std::function<void()>>;

    enum class disk_charge : uint8_t {
        none,
        reserved,
        committed,
        uncertain,
    };

    struct write_record {
        uint64_t                         entry_id   = 0;
        uint64_t                         generation = 0;
        uint64_t                         job_id     = 0;
        std::string                      final_path;
        std::string                      temp_path;
        std::string                      parent_path;
        std::string                      completion_path;
        server_prompt_cache_io_payload   payload;
        uint64_t                         bytes             = 0;
        bool                             verify_checksum   = false;
        uint64_t                         expected_checksum = 0;
        uint64_t                         checksum          = 0;
        server_prompt_cache_io_faults    faults;
        server_prompt_cache_io_lifecycle lifecycle            = server_prompt_cache_io_lifecycle::queued;
        server_prompt_cache_io_status    status               = server_prompt_cache_io_status::ok;
        int                              os_error             = 0;
        bool                             cancel_requested     = false;
        bool                             retire_requested     = false;
        bool                             publication_fenced   = false;
        bool                             rename_succeeded     = false;
        bool                             cleanup_pending      = false;
        bool                             completion_emitted   = false;
        disk_charge                      charge               = disk_charge::reserved;
        uint64_t                         accepted_ns          = 0;
        uint64_t                         started_ns           = 0;
        uint64_t                         publication_fence_ns = 0;
        uint64_t                         completed_ns         = 0;
        server_prompt_cache_io_test_pause pause_reached  = server_prompt_cache_io_test_pause::none;
        bool                              pause_released = false;
    };

    struct cleanup_record {
        uint64_t                      job_id     = 0;
        uint64_t                      entry_id   = 0;
        uint64_t                      generation = 0;
        std::string                   path;
        std::string                   parent_path;
        std::string                   completion_path;
        bool                          abandoned = false;
        server_prompt_cache_io_faults faults;
        uint64_t                      accepted_ns        = 0;
        uint64_t                      started_ns         = 0;
        bool                          completion_emitted = false;
    };

    struct work_item {
        server_prompt_cache_io_event_kind kind = server_prompt_cache_io_event_kind::write;
        std::shared_ptr<write_record>     write;
        cleanup_record                    cleanup;
    };

    struct path_owner {
        key         owner;
        uint64_t    bytes  = 0;
        disk_charge charge = disk_charge::committed;
    };

    explicit impl(server_prompt_cache_io_config config_arg) :
        config(std::move(config_arg)),
        initial_disk_bytes(config.initial_disk_bytes) {
        valid = valid_config();
        if (!valid) {
            accepting            = false;
            startup_cleanup_done = true;
            return;
        }
        accepting                   = true;
        const uint64_t address_bits = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
        temp_nonce                  = now_ns() ^ (address_bits << 17) ^ static_cast<uint64_t>(::getpid());
        try {
            completion_queue.reserve(config.max_completions);
            worker = std::thread(&impl::run, this);
        } catch (...) {
            accepting            = false;
            valid                = false;
            startup_cleanup_done = true;
        }
    }

    ~impl() { shutdown(config.destructor_shutdown); }

    bool valid_config() const {
        if (config.max_jobs == 0 || config.max_pending_bytes == 0 || config.max_completions < config.max_jobs) {
            return false;
        }
        if (config.max_disk_bytes != 0 && config.initial_disk_bytes > config.max_disk_bytes) {
            return false;
        }
        for (const std::string & directory : config.startup_cleanup_directories) {
            if (directory.empty()) {
                return false;
            }
        }
        return true;
    }

    bool current_locked(const write_record & record) const {
        const auto found = latest_generation.find(record.entry_id);
        return found != latest_generation.end() && found->second == record.generation;
    }

    bool current_locked(const key & owner) const {
        const auto found = latest_generation.find(owner.first);
        return found != latest_generation.end() && found->second == owner.second;
    }

    server_prompt_cache_io_enqueue_result submit(server_prompt_cache_io_write request) {
        server_prompt_cache_io_enqueue_result result;
        result.payload = request.payload;
        result.bytes   = request.payload ? static_cast<uint64_t>(request.payload->size()) : 0;

        if (request.entry_id == 0 || request.generation == 0 || request.final_path.empty() || !request.payload ||
            request.payload->empty() || !valid_faults(request.faults) || is_private_temp_name(request.final_path)) {
            result.status = server_prompt_cache_io_status::invalid_argument;
            return result;
        }

        std::shared_ptr<write_record> record;
        try {
            record                    = std::make_shared<write_record>();
            record->entry_id          = request.entry_id;
            record->generation        = request.generation;
            record->final_path        = std::move(request.final_path);
            record->parent_path       = parent_directory(record->final_path);
            record->completion_path   = record->final_path;
            record->payload           = request.payload;
            record->bytes             = static_cast<uint64_t>(request.payload->size());
            record->verify_checksum   = request.verify_checksum;
            record->expected_checksum = request.expected_checksum;
            record->faults            = request.faults;
        } catch (const std::bad_alloc &) {
            result.status = server_prompt_cache_io_status::allocation_failed;
            return result;
        } catch (...) {
            result.status = server_prompt_cache_io_status::invalid_argument;
            return result;
        }

        {
            std::lock_guard<std::mutex>         lock(mutex);
            const server_prompt_cache_io_status admission = write_admission_locked(*record);
            if (admission != server_prompt_cache_io_status::ok) {
                result.status = admission;
                if (admission == server_prompt_cache_io_status::cooldown) {
                    // A rejected owner retry has no completion of its own. Arm
                    // one bounded worker wake at the admission deadline so an
                    // otherwise-idle owner can try the immutable payload again.
                    cooldown_wake_deadline_ns = cooldown_until_ns;
                    cv.notify_all();
                }
                return result;
            }

            const key      record_key{ record->entry_id, record->generation };
            const auto     old_latest       = latest_generation.find(record->entry_id);
            const bool     had_latest       = old_latest != latest_generation.end();
            const uint64_t old_latest_value = had_latest ? old_latest->second : 0;
            bool           queue_added      = false;
            bool           record_added     = false;
            bool           latest_added     = false;
            try {
                record->job_id      = next_nonzero_job_id_locked();
                record->accepted_ns = now_ns();
                record->temp_path   = make_temp_path_locked(*record);
                work_item item;
                item.kind  = server_prompt_cache_io_event_kind::write;
                item.write = record;
                queue.push_back(std::move(item));
                queue_added = true;
                records.emplace(record_key, record);
                record_added = true;
                if (had_latest) {
                    old_latest->second = record->generation;
                } else {
                    latest_generation.emplace(record->entry_id, record->generation);
                    latest_added = true;
                }
            } catch (const std::bad_alloc &) {
                if (record_added) {
                    records.erase(record_key);
                }
                if (queue_added) {
                    queue.pop_back();
                }
                if (had_latest) {
                    old_latest->second = old_latest_value;
                } else if (latest_added) {
                    latest_generation.erase(record->entry_id);
                }
                result.status = server_prompt_cache_io_status::allocation_failed;
                return result;
            } catch (...) {
                if (record_added) {
                    records.erase(record_key);
                }
                if (queue_added) {
                    queue.pop_back();
                }
                if (had_latest) {
                    old_latest->second = old_latest_value;
                } else if (latest_added) {
                    latest_generation.erase(record->entry_id);
                }
                result.status = server_prompt_cache_io_status::invalid_argument;
                return result;
            }

            supersede_older_locked(*record);
            ++active_jobs;
            ++accepted_writes;
            ram_held_bytes += record->bytes;
            disk_reserved_bytes += record->bytes;
            result.status    = server_prompt_cache_io_status::ok;
            result.lifecycle = server_prompt_cache_io_lifecycle::queued;
            result.job_id    = record->job_id;
        }
        cv.notify_all();
        return result;
    }

    server_prompt_cache_io_status write_admission_locked(const write_record & record) const {
        if (!valid) {
            return server_prompt_cache_io_status::invalid_argument;
        }
        if (!accepting) {
            return server_prompt_cache_io_status::sealed;
        }
        if (cooldown_until_ns != 0 && now_ns() < cooldown_until_ns) {
            return server_prompt_cache_io_status::cooldown;
        }
        const auto latest = latest_generation.find(record.entry_id);
        if (latest != latest_generation.end() && record.generation <= latest->second) {
            return server_prompt_cache_io_status::generation_not_newer;
        }
        const server_prompt_cache_io_status capacity = job_capacity_locked();
        if (capacity != server_prompt_cache_io_status::ok) {
            return capacity;
        }

        uint64_t total = 0;
        if (add_overflow(ram_held_bytes, record.bytes, total) || total > config.max_pending_bytes) {
            return server_prompt_cache_io_status::byte_limit;
        }
        if (add_overflow(initial_disk_bytes, disk_reserved_bytes, total) ||
            add_overflow(total, disk_committed_bytes, total) || add_overflow(total, disk_uncertain_bytes, total) ||
            add_overflow(total, record.bytes, total) || (config.max_disk_bytes != 0 && total > config.max_disk_bytes)) {
            return server_prompt_cache_io_status::disk_limit;
        }

        const auto owner = path_owners.find(record.final_path);
        if (owner != path_owners.end() && owner->second.owner.first != record.entry_id) {
            return server_prompt_cache_io_status::path_conflict;
        }
        for (const auto & value : records) {
            const write_record & other = *value.second;
            if ((other.lifecycle == server_prompt_cache_io_lifecycle::queued ||
                 other.lifecycle == server_prompt_cache_io_lifecycle::writing) &&
                other.final_path == record.final_path && other.entry_id != record.entry_id) {
                return server_prompt_cache_io_status::path_conflict;
            }
        }
        return server_prompt_cache_io_status::ok;
    }

    server_prompt_cache_io_status job_capacity_locked() const {
        if (active_jobs >= config.max_jobs) {
            return server_prompt_cache_io_status::job_limit;
        }
        if (completion_queue.size() + active_jobs >= config.max_completions) {
            return server_prompt_cache_io_status::completion_limit;
        }
        return server_prompt_cache_io_status::ok;
    }

    server_prompt_cache_io_status adopt_existing(uint64_t            entry_id,
                                                 uint64_t            generation,
                                                 const std::string & exact_path,
                                                 uint64_t            bytes) {
        if (entry_id == 0 || generation == 0 || exact_path.empty() || bytes == 0 || is_private_temp_name(exact_path)) {
            return server_prompt_cache_io_status::invalid_argument;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!valid) {
            return server_prompt_cache_io_status::invalid_argument;
        }
        if (!accepting) {
            return server_prompt_cache_io_status::sealed;
        }
        if (active_jobs != 0) {
            return server_prompt_cache_io_status::job_limit;
        }
        const auto latest = latest_generation.find(entry_id);
        if (latest != latest_generation.end() && generation <= latest->second) {
            return server_prompt_cache_io_status::generation_not_newer;
        }
        if (path_owners.find(exact_path) != path_owners.end()) {
            return server_prompt_cache_io_status::path_conflict;
        }
        if (bytes > initial_disk_bytes) {
            return server_prompt_cache_io_status::disk_limit;
        }

        const bool     had_latest   = latest != latest_generation.end();
        const uint64_t old_latest   = had_latest ? latest->second : 0;
        bool           owner_added  = false;
        bool           latest_added = false;
        try {
            path_owner owner;
            owner.owner  = { entry_id, generation };
            owner.bytes  = bytes;
            owner.charge = disk_charge::committed;
            path_owners.emplace(exact_path, owner);
            owner_added = true;
            if (had_latest) {
                latest->second = generation;
            } else {
                latest_generation.emplace(entry_id, generation);
                latest_added = true;
            }
        } catch (const std::bad_alloc &) {
            if (owner_added) {
                path_owners.erase(exact_path);
            }
            if (had_latest) {
                latest->second = old_latest;
            } else if (latest_added) {
                latest_generation.erase(entry_id);
            }
            return server_prompt_cache_io_status::allocation_failed;
        }
        initial_disk_bytes -= bytes;
        disk_committed_bytes += bytes;
        return server_prompt_cache_io_status::ok;
    }

    std::string make_temp_path_locked(const write_record & record) const {
        const std::string directory = parent_directory(record.final_path);
        const std::string filename  = std::string(PCACHE_IO_TEMP_PREFIX) + std::to_string(::getpid()) + "-" +
                                     std::to_string(temp_nonce) + "-" + std::to_string(record.job_id) +
                                     PCACHE_IO_TEMP_SUFFIX;
        return directory == "/" ? directory + filename : directory + "/" + filename;
    }

    void supersede_older_locked(const write_record & newest) {
        for (auto & value : records) {
            write_record & older = *value.second;
            if (older.entry_id != newest.entry_id || older.generation >= newest.generation) {
                continue;
            }
            older.retire_requested = true;
            if ((older.lifecycle == server_prompt_cache_io_lifecycle::queued ||
                 older.lifecycle == server_prompt_cache_io_lifecycle::writing) &&
                !older.publication_fenced) {
                older.cancel_requested = true;
            }
        }
    }

    server_prompt_cache_io_status cancel(uint64_t entry_id, uint64_t generation) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto                  found = records.find({ entry_id, generation });
        if (found == records.end()) {
            return server_prompt_cache_io_status::not_found;
        }
        write_record & record = *found->second;
        if (record.lifecycle != server_prompt_cache_io_lifecycle::queued &&
            record.lifecycle != server_prompt_cache_io_lifecycle::writing) {
            return server_prompt_cache_io_status::too_late;
        }
        if (record.publication_fenced) {
            return server_prompt_cache_io_status::too_late;
        }
        record.cancel_requested = true;
        cv.notify_all();
        return server_prompt_cache_io_status::cancel_requested;
    }

    server_prompt_cache_io_status retire(uint64_t entry_id, uint64_t generation) {
        const key                   requested{ entry_id, generation };
        std::lock_guard<std::mutex> lock(mutex);
        const auto                  found = records.find(requested);
        if (found != records.end()) {
            write_record & record   = *found->second;
            record.retire_requested = true;
            if ((record.lifecycle == server_prompt_cache_io_lifecycle::queued ||
                 record.lifecycle == server_prompt_cache_io_lifecycle::writing) &&
                !record.publication_fenced) {
                record.cancel_requested = true;
                cv.notify_all();
                return server_prompt_cache_io_status::cancel_requested;
            }
            if (record.lifecycle == server_prompt_cache_io_lifecycle::writing && record.publication_fenced) {
                cv.notify_all();
                return server_prompt_cache_io_status::too_late;
            }
            if (record.cleanup_pending) {
                return server_prompt_cache_io_status::ok;
            }
        }

        const auto owned = find_owned_path_locked(requested);
        if (owned == path_owners.end()) {
            // No worker or exact path remains to produce another event. Keep
            // too_late reserved for the fenced-writing branch above, where a
            // write completion is guaranteed by the accepted job contract.
            return server_prompt_cache_io_status::not_found;
        }
        const server_prompt_cache_io_enqueue_result enqueued =
            enqueue_cleanup_locked(entry_id, generation, owned->first, {}, false);
        if (enqueued.status != server_prompt_cache_io_status::ok) {
            return enqueued.status;
        }
        if (found != records.end()) {
            found->second->cleanup_pending = true;
        }
        cv.notify_all();
        return server_prompt_cache_io_status::ok;
    }

    server_prompt_cache_io_enqueue_result cleanup_exact(uint64_t                              entry_id,
                                                        uint64_t                              generation,
                                                        const std::string &                   exact_path,
                                                        const server_prompt_cache_io_faults & faults,
                                                        bool                                  abandoned) {
        server_prompt_cache_io_enqueue_result result;
        if (exact_path.empty() || !valid_faults(faults) || (abandoned && !is_private_temp_name(exact_path)) ||
            (!abandoned && (entry_id == 0 || generation == 0))) {
            result.status = server_prompt_cache_io_status::invalid_argument;
            return result;
        }

        std::lock_guard<std::mutex> lock(mutex);
        if (!valid) {
            result.status = server_prompt_cache_io_status::invalid_argument;
            return result;
        }
        if (!accepting) {
            result.status = server_prompt_cache_io_status::sealed;
            return result;
        }
        if (!abandoned) {
            const auto owner = path_owners.find(exact_path);
            if (owner == path_owners.end()) {
                result.status = server_prompt_cache_io_status::not_found;
                return result;
            }
            if (owner->second.owner != key{ entry_id, generation }) {
                result.status = server_prompt_cache_io_status::stale_generation;
                return result;
            }
            const auto record = records.find({ entry_id, generation });
            if (record != records.end() && record->second->cleanup_pending) {
                result.status = server_prompt_cache_io_status::job_limit;
                return result;
            }
        }
        result = enqueue_cleanup_locked(entry_id, generation, exact_path, faults, abandoned);
        if (result.status != server_prompt_cache_io_status::ok) {
            return result;
        }
        if (!abandoned) {
            const auto record = records.find({ entry_id, generation });
            if (record != records.end()) {
                record->second->cleanup_pending = true;
            }
        }
        cv.notify_all();
        return result;
    }

    server_prompt_cache_io_enqueue_result enqueue_cleanup_locked(uint64_t                              entry_id,
                                                                 uint64_t                              generation,
                                                                 const std::string &                   exact_path,
                                                                 const server_prompt_cache_io_faults & faults,
                                                                 bool                                  abandoned) {
        server_prompt_cache_io_enqueue_result result;
        result.status = job_capacity_locked();
        if (result.status != server_prompt_cache_io_status::ok) {
            return result;
        }

        try {
            cleanup_record cleanup;
            cleanup.job_id          = next_nonzero_job_id_locked();
            cleanup.entry_id        = entry_id;
            cleanup.generation      = generation;
            cleanup.path            = exact_path;
            cleanup.parent_path     = parent_directory(cleanup.path);
            cleanup.completion_path = cleanup.path;
            cleanup.abandoned       = abandoned;
            cleanup.faults          = faults;
            cleanup.accepted_ns     = now_ns();
            result.job_id           = cleanup.job_id;
            work_item item;
            item.kind    = server_prompt_cache_io_event_kind::cleanup;
            item.cleanup = std::move(cleanup);
            queue.push_back(std::move(item));
        } catch (const std::bad_alloc &) {
            result.status = server_prompt_cache_io_status::allocation_failed;
            result.job_id = 0;
            return result;
        } catch (...) {
            result.status = server_prompt_cache_io_status::invalid_argument;
            result.job_id = 0;
            return result;
        }
        ++active_jobs;
        result.status    = server_prompt_cache_io_status::ok;
        result.lifecycle = server_prompt_cache_io_lifecycle::retired;
        return result;
    }

    server_prompt_cache_io_payload pending_payload(uint64_t entry_id, uint64_t generation) const {
        std::lock_guard<std::mutex> lock(mutex);
        const auto                  found = records.find({ entry_id, generation });
        return found == records.end() ? server_prompt_cache_io_payload{} : found->second->payload;
    }

    server_prompt_cache_io_snapshot inspect(uint64_t entry_id, uint64_t generation) const {
        server_prompt_cache_io_snapshot result;
        std::lock_guard<std::mutex>     lock(mutex);
        const auto                      found = records.find({ entry_id, generation });
        if (found == records.end()) {
            const auto owned = find_owned_path_locked({ entry_id, generation });
            if (owned == path_owners.end()) {
                return result;
            }
            result.status     = server_prompt_cache_io_status::ok;
            result.lifecycle  = owned->second.charge == disk_charge::uncertain ?
                                    server_prompt_cache_io_lifecycle::commit_uncertain :
                                    server_prompt_cache_io_lifecycle::durable;
            result.entry_id   = entry_id;
            result.generation = generation;
            result.bytes      = owned->second.bytes;
            result.final_path = owned->first;
            result.is_latest  = current_locked(owned->second.owner);
            return result;
        }
        const write_record & record = *found->second;
        result.status               = record.status;
        result.lifecycle            = record.lifecycle;
        result.entry_id             = record.entry_id;
        result.generation           = record.generation;
        result.job_id               = record.job_id;
        result.bytes                = record.bytes;
        result.checksum             = record.checksum;
        result.is_latest            = current_locked(record);
        result.cancel_requested     = record.cancel_requested;
        result.retire_requested     = record.retire_requested;
        result.publication_fenced   = record.publication_fenced;
        result.cleanup_pending      = record.cleanup_pending;
        result.final_path           = record.final_path;
        result.temp_path            = record.temp_path;
        result.accepted_ns          = record.accepted_ns;
        result.started_ns           = record.started_ns;
        result.publication_fence_ns = record.publication_fence_ns;
        result.completed_ns         = record.completed_ns;
        return result;
    }

    std::vector<server_prompt_cache_io_completion> poll_completions(size_t max_count) {
        std::vector<server_prompt_cache_io_completion> result;
        std::lock_guard<std::mutex>                    lock(mutex);
        const size_t                                   count = std::min(max_count, completion_queue.size());
        result.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            result.push_back(std::move(completion_queue[index]));
        }
        completion_queue.erase(completion_queue.begin(), completion_queue.begin() + static_cast<std::ptrdiff_t>(count));
        prune_records_locked();
        return result;
    }

    server_prompt_cache_io_stats stats() const {
        std::lock_guard<std::mutex>  lock(mutex);
        server_prompt_cache_io_stats result;
        result.accepting               = accepting;
        result.worker_running          = worker_running;
        result.startup_cleanup_done    = startup_cleanup_done;
        result.active_jobs             = active_jobs;
        result.queued_jobs             = static_cast<uint32_t>(queue.size());
        result.writing_jobs            = writing_jobs;
        result.completion_backlog      = static_cast<uint32_t>(completion_queue.size());
        result.ram_held_bytes          = ram_held_bytes;
        result.disk_reserved_bytes     = disk_reserved_bytes;
        result.disk_committed_bytes    = disk_committed_bytes;
        result.disk_uncertain_bytes    = disk_uncertain_bytes;
        result.initial_disk_bytes      = initial_disk_bytes;
        result.accepted_writes         = accepted_writes;
        result.durable_writes          = durable_writes;
        result.retired_writes          = retired_writes;
        result.failed_writes           = failed_writes;
        result.uncertain_writes        = uncertain_writes;
        result.completed_cleanups      = completed_cleanups;
        result.failed_cleanups         = failed_cleanups;
        result.abandoned_temps_removed = abandoned_temps_removed;
        result.callback_calls          = callback_calls;
        result.callback_failures       = callback_failures;
        result.last_error              = last_error;
        result.last_error_ns           = last_error_ns;
        result.cooldown_until_ns       = cooldown_until_ns;
        result.tracked_records         = records.size();
        result.tracked_generations     = latest_generation.size();
        result.tracked_paths           = path_owners.size();
        return result;
    }

    void set_wake_callback(std::function<void()> callback) {
        wake_callback_handle replacement;
        if (callback) {
            replacement = std::make_shared<const std::function<void()>>(std::move(callback));
        }

        wake_callback_handle         prior;
        std::unique_lock<std::mutex> lock(mutex);
        prior = std::move(wake_callback);
        cv.wait(lock, [&] { return callbacks_in_flight == 0; });
        wake_callback = std::move(replacement);
        lock.unlock();
        // A callback capture may run arbitrary destructors. Never release the
        // engine's last reference while holding its mutex.
        prior.reset();
    }

    void clear_error_cooldown() {
        std::lock_guard<std::mutex> lock(mutex);
        cooldown_until_ns = 0;
        if (cooldown_wake_deadline_ns != 0) {
            cooldown_wake_deadline_ns = now_ns();
            cv.notify_all();
        }
    }

    server_prompt_cache_io_status shutdown(server_prompt_cache_io_shutdown_mode mode) {
        {
            std::lock_guard<std::mutex> join_lock(join_mutex);
            {
                std::lock_guard<std::mutex> lock(mutex);
                accepting = false;
                if (mode == server_prompt_cache_io_shutdown_mode::fast) {
                    fast_stop = true;
                    for (auto & value : records) {
                        write_record & record = *value.second;
                        if ((record.lifecycle == server_prompt_cache_io_lifecycle::queued ||
                             record.lifecycle == server_prompt_cache_io_lifecycle::writing) &&
                            !record.publication_fenced) {
                            record.cancel_requested = true;
                            record.retire_requested = true;
                        }
                    }
                }
            }
            cv.notify_all();
            if (worker.joinable()) {
                worker.join();
            }
        }
        return valid ? server_prompt_cache_io_status::ok : server_prompt_cache_io_status::invalid_argument;
    }

    bool test_wait_for_pause(uint64_t job_id, server_prompt_cache_io_test_pause point, uint64_t timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            for (const auto & value : records) {
                if (value.second->job_id == job_id && value.second->pause_reached == point) {
                    return true;
                }
            }
            return false;
        });
    }

    void test_release_pause(uint64_t job_id) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & value : records) {
            if (value.second->job_id == job_id) {
                value.second->pause_released = true;
                break;
            }
        }
        cv.notify_all();
    }

    uint64_t next_nonzero_job_id_locked() {
        uint64_t result = next_job_id++;
        if (result == 0) {
            result = next_job_id++;
        }
        return result;
    }

    std::map<std::string, path_owner>::iterator find_owned_path_locked(const key & requested) {
        return std::find_if(path_owners.begin(), path_owners.end(),
                            [&](const auto & value) { return value.second.owner == requested; });
    }

    std::map<std::string, path_owner>::const_iterator find_owned_path_locked(const key & requested) const {
        return std::find_if(path_owners.cbegin(), path_owners.cend(),
                            [&](const auto & value) { return value.second.owner == requested; });
    }

    void maybe_erase_latest_locked(uint64_t entry_id) {
        const auto record = records.lower_bound({ entry_id, 0 });
        if (record != records.end() && record->first.first == entry_id) {
            return;
        }
        const bool has_path = std::any_of(path_owners.begin(), path_owners.end(), [&](const auto & value) {
            return value.second.owner.first == entry_id;
        });
        if (!has_path) {
            latest_generation.erase(entry_id);
        }
    }

    void prune_records_locked() {
        for (auto iterator = records.begin(); iterator != records.end();) {
            const write_record & record   = *iterator->second;
            const bool           terminal = record.lifecycle != server_prompt_cache_io_lifecycle::queued &&
                                  record.lifecycle != server_prompt_cache_io_lifecycle::writing;
            const bool has_owner = find_owned_path_locked(iterator->first) != path_owners.end();
            if (terminal && !record.payload && !record.cleanup_pending && !has_owner) {
                const uint64_t entry_id = record.entry_id;
                iterator = records.erase(iterator);
                maybe_erase_latest_locked(entry_id);
            } else {
                ++iterator;
            }
        }
    }

    struct io_result {
        server_prompt_cache_io_status status   = server_prompt_cache_io_status::ok;
        int                           os_error = 0;
        bool                          removed  = false;
    };

    void run() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            worker_running = true;
        }
        cv.notify_all();
        try {
            run_startup_cleanup();
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            startup_cleanup_done = true;
            record_error_locked(ENOMEM);
        }

        for (;;) {
            work_item           item;
            wake_callback_handle deadline_wake;
            bool                 deadline_expired = false;
            bool                 stop             = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                for (;;) {
                    if (!queue.empty() || !accepting) {
                        break;
                    }
                    if (cooldown_wake_deadline_ns == 0) {
                        cv.wait(lock);
                        continue;
                    }

                    const uint64_t now = now_ns();
                    if (now >= cooldown_wake_deadline_ns) {
                        cooldown_wake_deadline_ns = 0;
                        deadline_expired          = true;
                        deadline_wake             = take_wake_locked();
                        break;
                    }
                    const uint64_t remaining = cooldown_wake_deadline_ns - now;
                    const uint64_t bounded = std::min<uint64_t>(
                        remaining, static_cast<uint64_t>(std::chrono::nanoseconds::max().count()));
                    cv.wait_for(lock, std::chrono::nanoseconds(static_cast<int64_t>(bounded)));
                }
                if (!deadline_expired) {
                    if (queue.empty()) {
                        stop = !accepting;
                    } else {
                        item = std::move(queue.front());
                        queue.pop_front();
                    }
                }
            }

            if (stop) {
                break;
            }
            if (deadline_expired) {
                call_wake(std::move(deadline_wake));
                continue;
            }

            if (item.kind == server_prompt_cache_io_event_kind::write) {
                try {
                    process_write(item.write);
                } catch (const std::bad_alloc &) {
                    finalize_worker_exception(item.write, server_prompt_cache_io_status::allocation_failed, ENOMEM);
                } catch (...) {
                    finalize_worker_exception(item.write, server_prompt_cache_io_status::io_error, EIO);
                }
            } else {
                try {
                    process_cleanup(item.cleanup);
                } catch (const std::bad_alloc &) {
                    finalize_cleanup_exception(item.cleanup, server_prompt_cache_io_status::allocation_failed, ENOMEM);
                } catch (...) {
                    finalize_cleanup_exception(item.cleanup, server_prompt_cache_io_status::io_error, EIO);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            worker_running = false;
        }
        cv.notify_all();
    }

    void run_startup_cleanup() {
        for (const std::string & directory : config.startup_cleanup_directories) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (fast_stop) {
                    break;
                }
            }

            DIR * stream = ::opendir(directory.c_str());
            if (stream == nullptr) {
                continue;
            }
            bool removed_any = false;
            while (dirent * entry = ::readdir(stream)) {
                const std::string name(entry->d_name);
                if (name == "." || name == "..") {
                    continue;
                }
                const std::string exact =
                    directory.empty() || directory.back() == '/' ? directory + name : directory + "/" + name;
                if (!is_private_temp_name(exact)) {
                    continue;
                }
                struct stat file_stat{};
                if (::lstat(exact.c_str(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) {
                    continue;
                }
                if (::unlink(exact.c_str()) == 0) {
                    removed_any = true;
                    std::lock_guard<std::mutex> lock(mutex);
                    ++abandoned_temps_removed;
                }
            }
            (void) ::closedir(stream);
            if (removed_any) {
                (void) sync_directory(directory, 0);
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            startup_cleanup_done = true;
        }
        cv.notify_all();
    }

    void process_write(const std::shared_ptr<write_record> & record) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            record->lifecycle  = server_prompt_cache_io_lifecycle::writing;
            record->started_ns = now_ns();
            ++writing_jobs;
        }

        maybe_pause(record, server_prompt_cache_io_test_pause::before_write);
        server_prompt_cache_io_status abort = pre_fence_abort_status(record);
        if (abort != server_prompt_cache_io_status::ok) {
            finalize_precommit(record, abort, 0);
            return;
        }

        const uint64_t checksum = server_prompt_cache_io_checksum(record->payload->data(), record->payload->size());
        {
            std::lock_guard<std::mutex> lock(mutex);
            record->checksum = checksum;
        }
        if (record->verify_checksum && checksum != record->expected_checksum) {
            finalize_precommit(record, server_prompt_cache_io_status::checksum_mismatch, 0);
            return;
        }

        const int descriptor = ::open(record->temp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (descriptor < 0) {
            finalize_precommit(record, status_from_errno(errno), errno);
            return;
        }

        io_result written = write_all(descriptor, record);
        if (written.status == server_prompt_cache_io_status::ok) {
            struct stat file_stat{};
            if (::fstat(descriptor, &file_stat) != 0) {
                written.status   = status_from_errno(errno);
                written.os_error = errno;
            } else if (file_stat.st_size < 0 || static_cast<uint64_t>(file_stat.st_size) != record->bytes) {
                written.status   = server_prompt_cache_io_status::io_error;
                written.os_error = EIO;
            }
        }
        if (written.status == server_prompt_cache_io_status::ok) {
            const int injected = record->faults.fail_file_fsync_errno;
            if (injected != 0) {
                written.status   = status_from_errno(injected);
                written.os_error = injected;
            } else if (::fsync(descriptor) != 0) {
                written.status   = status_from_errno(errno);
                written.os_error = errno;
            }
        }
        if (::close(descriptor) != 0 && written.status == server_prompt_cache_io_status::ok) {
            written.status   = status_from_errno(errno);
            written.os_error = errno;
        }
        if (written.status != server_prompt_cache_io_status::ok) {
            finalize_precommit(record, written.status, written.os_error);
            return;
        }

        maybe_pause(record, server_prompt_cache_io_test_pause::before_publication_fence);
        abort = pre_fence_abort_status(record);
        if (abort != server_prompt_cache_io_status::ok) {
            finalize_precommit(record, abort, 0);
            return;
        }
        {
            std::lock_guard<std::mutex>         lock(mutex);
            const server_prompt_cache_io_status final_abort = pre_fence_abort_status_locked(*record);
            if (final_abort != server_prompt_cache_io_status::ok) {
                abort = final_abort;
            } else {
                record->publication_fenced   = true;
                record->publication_fence_ns = now_ns();
            }
        }
        if (abort != server_prompt_cache_io_status::ok) {
            finalize_precommit(record, abort, 0);
            return;
        }

        maybe_pause(record, server_prompt_cache_io_test_pause::after_publication_fence);
        if (record->faults.fail_worker_exception_before_rename) {
            throw std::bad_alloc();
        }
        int publish_error = 0;
        if (record->faults.fail_rename_errno != 0) {
            publish_error = record->faults.fail_rename_errno;
        } else if (::rename(record->temp_path.c_str(), record->final_path.c_str()) != 0) {
            publish_error = errno;
        }
        if (publish_error != 0) {
            finalize_precommit(record, status_from_errno(publish_error), publish_error);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            record->rename_succeeded = true;
        }

        maybe_pause(record, server_prompt_cache_io_test_pause::after_rename);
        const int directory_error = sync_directory(record->parent_path, record->faults.fail_directory_fsync_errno);

        bool should_retire = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            should_retire = !current_locked(*record) || record->retire_requested;
        }

        if (should_retire) {
            const io_result cleanup = remove_record_publication(*record);
            finalize_retired_publication(record, cleanup);
            return;
        }

        if (directory_error != 0) {
            finalize_uncertain(record, directory_error);
            return;
        }

        finalize_durable(record);
        cleanup_superseded_paths(record->entry_id, record->generation, record->final_path);
    }

    io_result write_all(int descriptor, const std::shared_ptr<write_record> & record) {
        io_result result;
        uint64_t  offset      = 0;
        uint32_t  eintrs_left = record->faults.write_eintrs;
        while (offset < record->bytes) {
            const server_prompt_cache_io_status abort = pre_fence_abort_status(record);
            if (abort != server_prompt_cache_io_status::ok) {
                result.status = abort;
                return result;
            }
            if (offset >= record->faults.fail_after_bytes) {
                result.status   = status_from_errno(record->faults.fail_write_errno);
                result.os_error = record->faults.fail_write_errno;
                return result;
            }
            size_t amount = static_cast<size_t>(std::min<uint64_t>(record->bytes - offset, PCACHE_IO_WRITE_CHUNK));
            amount        = static_cast<size_t>(std::min<uint64_t>(amount, record->faults.max_write_size));
            if (record->faults.fail_after_bytes != UINT64_MAX) {
                amount = static_cast<size_t>(std::min<uint64_t>(amount, record->faults.fail_after_bytes - offset));
            }
            if (amount == 0) {
                result.status   = server_prompt_cache_io_status::io_error;
                result.os_error = EIO;
                return result;
            }

            ssize_t count;
            if (eintrs_left != 0) {
                --eintrs_left;
                errno = EINTR;
                count = -1;
            } else {
                count = ::write(descriptor, record->payload->data() + offset, amount);
            }
            if (count < 0) {
                if (errno == EINTR) {
                    continue;
                }
                result.status   = status_from_errno(errno);
                result.os_error = errno;
                return result;
            }
            if (count == 0) {
                result.status   = server_prompt_cache_io_status::io_error;
                result.os_error = EIO;
                return result;
            }
            offset += static_cast<uint64_t>(count);
        }
        return result;
    }

    server_prompt_cache_io_status pre_fence_abort_status(const std::shared_ptr<write_record> & record) const {
        std::lock_guard<std::mutex> lock(mutex);
        return pre_fence_abort_status_locked(*record);
    }

    server_prompt_cache_io_status pre_fence_abort_status_locked(const write_record & record) const {
        if (!current_locked(record)) {
            return server_prompt_cache_io_status::stale_generation;
        }
        if (record.cancel_requested || record.retire_requested || fast_stop) {
            return server_prompt_cache_io_status::cancelled;
        }
        return server_prompt_cache_io_status::ok;
    }

    void maybe_pause(const std::shared_ptr<write_record> & record, server_prompt_cache_io_test_pause point) {
        if (record->faults.pause != point) {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex);
        record->pause_reached = point;
        cv.notify_all();
        cv.wait(lock, [&] {
            return record->pause_released || fast_stop ||
                   (!record->publication_fenced && (record->cancel_requested || !current_locked(*record)));
        });
    }

    void finalize_precommit(const std::shared_ptr<write_record> & record,
                            server_prompt_cache_io_status         status,
                            int                                   os_error) {
        if (!record->temp_path.empty()) {
            if (::unlink(record->temp_path.c_str()) == 0) {
                (void) sync_directory(record->parent_path, 0);
            }
        }

        wake_callback_handle wake;
        {
            std::lock_guard<std::mutex> lock(mutex);
            release_record_reservation_locked(*record);
            release_record_payload_locked(*record);
            record->status       = status;
            record->os_error     = os_error;
            record->completed_ns = now_ns();
            const bool retired   = status == server_prompt_cache_io_status::cancelled ||
                                 status == server_prompt_cache_io_status::stale_generation;
            record->lifecycle =
                retired ? server_prompt_cache_io_lifecycle::retired : server_prompt_cache_io_lifecycle::write_failed;
            if (retired) {
                ++retired_writes;
            } else {
                ++failed_writes;
                record_error_locked(os_error);
            }
            wake = complete_write_locked(*record);
        }
        cv.notify_all();
        call_wake(std::move(wake));
    }

    void finalize_durable(const std::shared_ptr<write_record> & record) {
        wake_callback_handle wake;
        bool                 retire_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            retire_now = !current_locked(*record) || record->retire_requested;
            if (retire_now) {
                // A newer admission crossed the small gap between the first
                // post-fsync check and this generation-fenced completion.
            } else {
                replace_path_owner_locked(*record, disk_charge::committed);
                release_record_payload_locked(*record);
                record->status       = server_prompt_cache_io_status::ok;
                record->lifecycle    = server_prompt_cache_io_lifecycle::durable;
                record->completed_ns = now_ns();
                ++durable_writes;
                wake = complete_write_locked(*record);
            }
        }
        if (retire_now) {
            const io_result cleanup = remove_record_publication(*record);
            finalize_retired_publication(record, cleanup);
            return;
        }
        cv.notify_all();
        call_wake(std::move(wake));
    }

    void finalize_uncertain(const std::shared_ptr<write_record> & record, int os_error) {
        wake_callback_handle wake;
        bool                 retire_now = false;
        {
            std::lock_guard<std::mutex> lock(mutex);
            retire_now = !current_locked(*record) || record->retire_requested;
            if (!retire_now) {
                replace_path_owner_locked(*record, disk_charge::uncertain);
                // The engine intentionally retains the immutable RAM payload:
                // the directory durability boundary was not proven.
                record->status       = server_prompt_cache_io_status::commit_uncertain;
                record->lifecycle    = server_prompt_cache_io_lifecycle::commit_uncertain;
                record->os_error     = os_error;
                record->completed_ns = now_ns();
                ++uncertain_writes;
                record_error_locked(os_error);
                wake = complete_write_locked(*record);
            }
        }
        if (retire_now) {
            const io_result cleanup = remove_record_publication(*record);
            finalize_retired_publication(record, cleanup);
            return;
        }
        cv.notify_all();
        call_wake(std::move(wake));
    }

    void finalize_retired_publication(const std::shared_ptr<write_record> & record, const io_result & cleanup) {
        wake_callback_handle wake;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (cleanup.status == server_prompt_cache_io_status::ok) {
                if (!path_owned_by_newer_locked(*record)) {
                    erase_path_owner_locked(record->final_path);
                }
                release_record_reservation_locked(*record);
            } else if (cleanup.status == server_prompt_cache_io_status::stale_generation ||
                       path_owned_by_newer_locked(*record)) {
                release_record_reservation_locked(*record);
            } else {
                replace_path_owner_locked(*record, disk_charge::uncertain);
                // This was an inline cleanup attempt by the writer, not an
                // accepted cleanup job. Leave the exact owner retryable;
                // retire() must enqueue a real job and promise its completion.
                record->cleanup_pending = false;
                record_error_locked(cleanup.os_error);
            }
            release_record_payload_locked(*record);
            record->status = cleanup.status;
            if (cleanup.status == server_prompt_cache_io_status::ok) {
                record->status = current_locked(*record) ? server_prompt_cache_io_status::cancelled :
                                                           server_prompt_cache_io_status::stale_generation;
            }
            record->lifecycle    = server_prompt_cache_io_lifecycle::retired;
            record->os_error     = cleanup.os_error;
            record->completed_ns = now_ns();
            ++retired_writes;
            wake = complete_write_locked(*record, cleanup.removed);
        }
        cv.notify_all();
        call_wake(std::move(wake));
    }

    void process_cleanup(cleanup_record & cleanup) {
        cleanup.started_ns         = now_ns();
        bool cancelled_by_shutdown = false;
        bool ownership_valid       = cleanup.abandoned;
        {
            std::lock_guard<std::mutex> lock(mutex);
            cancelled_by_shutdown = fast_stop;
            if (!cleanup.abandoned) {
                const auto owner = path_owners.find(cleanup.path);
                ownership_valid =
                    owner != path_owners.end() && owner->second.owner == key{ cleanup.entry_id, cleanup.generation };
            }
        }

        io_result removed;
        if (cancelled_by_shutdown) {
            removed.status = server_prompt_cache_io_status::cancelled;
        } else if (!ownership_valid) {
            removed.status = server_prompt_cache_io_status::stale_generation;
        } else {
            removed = remove_exact_path(cleanup.path, cleanup.parent_path, cleanup.faults.fail_unlink_errno,
                                        cleanup.faults.fail_directory_fsync_errno);
        }

        wake_callback_handle wake;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const key                   requested{ cleanup.entry_id, cleanup.generation };
            if (!cleanup.abandoned) {
                const auto record = records.find(requested);
                if (record != records.end()) {
                    record->second->cleanup_pending = false;
                }
            }

            if (removed.status == server_prompt_cache_io_status::ok) {
                if (!cleanup.abandoned) {
                    const auto owner = path_owners.find(cleanup.path);
                    if (owner != path_owners.end() && owner->second.owner == requested) {
                        erase_path_owner_locked(owner);
                    }
                    const auto record = records.find(requested);
                    if (record != records.end()) {
                        release_record_payload_locked(*record->second);
                        record->second->lifecycle = server_prompt_cache_io_lifecycle::retired;
                        record->second->status    = server_prompt_cache_io_status::ok;
                    }
                } else if (removed.removed) {
                    ++abandoned_temps_removed;
                }
                ++completed_cleanups;
            } else if (removed.status == server_prompt_cache_io_status::commit_uncertain) {
                if (!cleanup.abandoned) {
                    const auto owner = path_owners.find(cleanup.path);
                    if (owner != path_owners.end() && owner->second.owner == requested &&
                        owner->second.charge == disk_charge::committed) {
                        disk_committed_bytes -= owner->second.bytes;
                        disk_uncertain_bytes += owner->second.bytes;
                        owner->second.charge = disk_charge::uncertain;
                    }
                    const auto record = records.find(requested);
                    if (record != records.end()) {
                        release_record_payload_locked(*record->second);
                        record->second->lifecycle = server_prompt_cache_io_lifecycle::commit_uncertain;
                        record->second->status    = server_prompt_cache_io_status::commit_uncertain;
                    }
                }
                ++failed_cleanups;
                record_error_locked(removed.os_error);
            } else {
                ++failed_cleanups;
                record_error_locked(removed.os_error);
            }

            server_prompt_cache_io_completion completion;
            completion.event              = server_prompt_cache_io_event_kind::cleanup;
            completion.status             = removed.status;
            completion.lifecycle          = removed.status == server_prompt_cache_io_status::commit_uncertain ?
                                                server_prompt_cache_io_lifecycle::commit_uncertain :
                                                server_prompt_cache_io_lifecycle::retired;
            completion.job_id             = cleanup.job_id;
            completion.entry_id           = cleanup.entry_id;
            completion.generation         = cleanup.generation;
            completion.os_error           = removed.os_error;
            completion.stale_generation   = removed.status == server_prompt_cache_io_status::stale_generation;
            completion.exact_path_removed = removed.removed;
            completion.final_path         = std::move(cleanup.completion_path);
            completion.accepted_ns        = cleanup.accepted_ns;
            completion.started_ns         = cleanup.started_ns;
            completion.completed_ns       = now_ns();
            completion_queue.push_back(std::move(completion));
            cleanup.completion_emitted = true;
            if (active_jobs != 0) {
                --active_jobs;
            }
            wake = take_wake_locked();
        }
        cv.notify_all();
        call_wake(std::move(wake));
    }

    void finalize_worker_exception(const std::shared_ptr<write_record> & record,
                                   server_prompt_cache_io_status         status,
                                   int                                   os_error) noexcept {
        try {
            bool already_complete = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                already_complete = record->completion_emitted;
                if (already_complete) {
                    ++failed_cleanups;
                    record_error_locked(os_error);
                }
            }
            if (already_complete) {
                return;
            }

            (void) ::unlink(record->temp_path.c_str());
            bool renamed = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                renamed = record->rename_succeeded;
            }
            if (!renamed) {
                finalize_precommit(record, status, os_error);
                return;
            }

            const io_result      cleanup = remove_record_publication(*record);
            wake_callback_handle wake;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (cleanup.status == server_prompt_cache_io_status::ok) {
                    if (!path_owned_by_newer_locked(*record)) {
                        erase_path_owner_locked(record->final_path);
                    }
                    release_record_reservation_locked(*record);
                    release_record_payload_locked(*record);
                    record->status    = status;
                    record->lifecycle = server_prompt_cache_io_lifecycle::write_failed;
                    ++failed_writes;
                } else if (cleanup.status == server_prompt_cache_io_status::stale_generation ||
                           path_owned_by_newer_locked(*record)) {
                    release_record_reservation_locked(*record);
                    release_record_payload_locked(*record);
                    record->status    = server_prompt_cache_io_status::stale_generation;
                    record->lifecycle = server_prompt_cache_io_lifecycle::retired;
                    ++retired_writes;
                } else {
                    replace_path_owner_locked(*record, disk_charge::uncertain);
                    // No cleanup job exists for this inline recovery attempt.
                    // A later retire() must enqueue one before reporting that
                    // a future completion is armed.
                    record->cleanup_pending = false;
                    record->status          = server_prompt_cache_io_status::commit_uncertain;
                    record->lifecycle       = server_prompt_cache_io_lifecycle::commit_uncertain;
                    ++uncertain_writes;
                }
                record->os_error     = cleanup.os_error != 0 ? cleanup.os_error : os_error;
                record->completed_ns = now_ns();
                record_error_locked(record->os_error);
                wake = complete_write_locked(*record);
            }
            cv.notify_all();
            call_wake(std::move(wake));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            accepting = false;
            fast_stop = true;
            cv.notify_all();
        }
    }

    void finalize_cleanup_exception(cleanup_record &              cleanup,
                                    server_prompt_cache_io_status status,
                                    int                           os_error) noexcept {
        try {
            wake_callback_handle wake;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (cleanup.completion_emitted) {
                    ++failed_cleanups;
                    record_error_locked(os_error);
                    return;
                }
                const auto record = records.find({ cleanup.entry_id, cleanup.generation });
                if (record != records.end()) {
                    record->second->cleanup_pending = false;
                }
                ++failed_cleanups;
                record_error_locked(os_error);
                server_prompt_cache_io_completion completion;
                completion.event        = server_prompt_cache_io_event_kind::cleanup;
                completion.status       = status;
                completion.lifecycle    = server_prompt_cache_io_lifecycle::retired;
                completion.job_id       = cleanup.job_id;
                completion.entry_id     = cleanup.entry_id;
                completion.generation   = cleanup.generation;
                completion.os_error     = os_error;
                completion.final_path   = std::move(cleanup.completion_path);
                completion.accepted_ns  = cleanup.accepted_ns;
                completion.started_ns   = cleanup.started_ns;
                completion.completed_ns = now_ns();
                completion_queue.push_back(std::move(completion));
                cleanup.completion_emitted = true;
                if (active_jobs != 0) {
                    --active_jobs;
                }
                wake = take_wake_locked();
            }
            cv.notify_all();
            call_wake(std::move(wake));
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            accepting = false;
            fast_stop = true;
            cv.notify_all();
        }
    }

    bool path_owned_by_newer_locked(const write_record & record) const {
        const auto owner = path_owners.find(record.final_path);
        return owner != path_owners.end() &&
               (owner->second.owner.first != record.entry_id || owner->second.owner.second > record.generation);
    }

    io_result remove_record_publication(const write_record & record) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!record.rename_succeeded || path_owned_by_newer_locked(record)) {
                io_result result;
                result.status = server_prompt_cache_io_status::stale_generation;
                return result;
            }
        }
        return remove_exact_path(record.final_path, record.parent_path, record.faults.fail_unlink_errno,
                                 record.faults.fail_directory_fsync_errno);
    }

    io_result remove_exact_path(const std::string & path,
                                const std::string & directory,
                                int                 unlink_error,
                                int                 directory_sync_error) {
        io_result result;
        if (unlink_error != 0) {
            result.status   = status_from_errno(unlink_error);
            result.os_error = unlink_error;
            return result;
        }
        if (::unlink(path.c_str()) != 0) {
            if (errno == ENOENT) {
                const int sync_error = sync_directory(directory, directory_sync_error);
                if (sync_error != 0) {
                    result.status   = server_prompt_cache_io_status::commit_uncertain;
                    result.os_error = sync_error;
                }
                return result;
            }
            result.status   = status_from_errno(errno);
            result.os_error = errno;
            return result;
        }
        result.removed       = true;
        const int sync_error = sync_directory(directory, directory_sync_error);
        if (sync_error != 0) {
            result.status   = server_prompt_cache_io_status::commit_uncertain;
            result.os_error = sync_error;
        }
        return result;
    }

    int sync_directory(const std::string & directory, int injected_error) const {
        if (injected_error != 0) {
            return injected_error;
        }
        const int descriptor = ::open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (descriptor < 0) {
            return errno;
        }
        int result = 0;
        if (::fsync(descriptor) != 0) {
            result = errno;
        }
        if (::close(descriptor) != 0 && result == 0) {
            result = errno;
        }
        return result;
    }

    void cleanup_superseded_paths(uint64_t entry_id, uint64_t generation, const std::string & keep_path) {
        std::vector<std::pair<std::string, key>> candidates;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (const auto & value : path_owners) {
                if (value.first != keep_path && value.second.owner.first == entry_id &&
                    value.second.owner.second < generation) {
                    candidates.emplace_back(value.first, value.second.owner);
                }
            }
        }
        for (const auto & candidate : candidates) {
            bool still_owned = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                const auto                  owner = path_owners.find(candidate.first);
                still_owned = owner != path_owners.end() && owner->second.owner == candidate.second;
            }
            if (!still_owned) {
                continue;
            }
            const io_result removed = remove_exact_path(candidate.first, parent_directory(candidate.first), 0, 0);
            std::lock_guard<std::mutex> lock(mutex);
            const auto                  owner = path_owners.find(candidate.first);
            if (owner == path_owners.end() || owner->second.owner != candidate.second) {
                continue;
            }
            if (removed.status == server_prompt_cache_io_status::ok) {
                erase_path_owner_locked(owner);
                const auto record = records.find(candidate.second);
                if (record != records.end()) {
                    release_record_payload_locked(*record->second);
                    record->second->cleanup_pending = false;
                    record->second->lifecycle       = server_prompt_cache_io_lifecycle::retired;
                    record->second->status          = server_prompt_cache_io_status::ok;
                }
                ++completed_cleanups;
            } else {
                ++failed_cleanups;
                record_error_locked(removed.os_error);
            }
        }
    }

    void release_record_reservation_locked(write_record & record) {
        if (record.charge == disk_charge::reserved) {
            disk_reserved_bytes -= record.bytes;
            record.charge = disk_charge::none;
        }
    }

    void release_record_payload_locked(write_record & record) {
        if (record.payload) {
            ram_held_bytes -= record.bytes;
            record.payload.reset();
        }
    }

    void replace_path_owner_locked(write_record & record, disk_charge new_charge) {
        path_owner replacement;
        replacement.owner  = { record.entry_id, record.generation };
        replacement.bytes  = record.bytes;
        replacement.charge = new_charge;

        auto owner = path_owners.find(record.final_path);
        if (owner == path_owners.end()) {
            // Allocate the map node before touching any accounting. If this
            // throws, the reservation and prior state remain intact for the
            // worker exception fallback.
            owner = path_owners.emplace(record.final_path, replacement).first;
        } else {
            if (owner->second.charge == disk_charge::committed) {
                disk_committed_bytes -= owner->second.bytes;
            } else if (owner->second.charge == disk_charge::uncertain) {
                disk_uncertain_bytes -= owner->second.bytes;
            }
            const auto old_record = records.find(owner->second.owner);
            if (old_record != records.end()) {
                old_record->second->charge          = disk_charge::none;
                old_record->second->cleanup_pending = false;
                old_record->second->lifecycle       = server_prompt_cache_io_lifecycle::retired;
                release_record_payload_locked(*old_record->second);
            }
            owner->second = replacement;
        }
        release_record_reservation_locked(record);
        if (new_charge == disk_charge::committed) {
            disk_committed_bytes += record.bytes;
        } else {
            disk_uncertain_bytes += record.bytes;
        }
        record.charge = new_charge;
    }

    void erase_path_owner_locked(const std::string & path) {
        const auto owner = path_owners.find(path);
        if (owner != path_owners.end()) {
            erase_path_owner_locked(owner);
        }
    }

    void erase_path_owner_locked(std::map<std::string, path_owner>::iterator owner) {
        const uint64_t entry_id = owner->second.owner.first;
        if (owner->second.charge == disk_charge::committed) {
            disk_committed_bytes -= owner->second.bytes;
        } else if (owner->second.charge == disk_charge::uncertain) {
            disk_uncertain_bytes -= owner->second.bytes;
        }
        const auto record = records.find(owner->second.owner);
        if (record != records.end()) {
            record->second->charge = disk_charge::none;
        }
        path_owners.erase(owner);
        maybe_erase_latest_locked(entry_id);
    }

    void record_error_locked(int error) {
        if (error == 0) {
            return;
        }
        last_error              = error;
        last_error_ns           = now_ns();
        const uint64_t cooldown = milliseconds_to_nanoseconds(config.error_cooldown_ms);
        cooldown_until_ns       = cooldown > UINT64_MAX - last_error_ns ? UINT64_MAX : last_error_ns + cooldown;
    }

    wake_callback_handle complete_write_locked(write_record & record, bool removed = false) {
        server_prompt_cache_io_completion completion;
        completion.event                = server_prompt_cache_io_event_kind::write;
        completion.status               = record.status;
        completion.lifecycle            = record.lifecycle;
        completion.job_id               = record.job_id;
        completion.entry_id             = record.entry_id;
        completion.generation           = record.generation;
        completion.bytes                = record.bytes;
        completion.checksum             = record.checksum;
        completion.os_error             = record.os_error;
        completion.stale_generation     = !current_locked(record);
        completion.publication_fenced   = record.publication_fenced;
        completion.exact_path_removed   = removed;
        completion.final_path           = std::move(record.completion_path);
        completion.accepted_ns          = record.accepted_ns;
        completion.started_ns           = record.started_ns;
        completion.publication_fence_ns = record.publication_fence_ns;
        completion.completed_ns         = record.completed_ns;
        completion_queue.push_back(std::move(completion));
        record.completion_emitted = true;
        if (active_jobs != 0) {
            --active_jobs;
        }
        if (writing_jobs != 0) {
            --writing_jobs;
        }
        return take_wake_locked();
    }

    wake_callback_handle take_wake_locked() {
        if (!wake_callback) {
            return {};
        }
        wake_callback_handle callback = wake_callback;
        ++callbacks_in_flight;
        return callback;
    }

    void call_wake(wake_callback_handle callback) {
        if (!callback) {
            return;
        }
        bool failed = false;
        try {
            (*callback)();
        } catch (...) {
            failed = true;
        }
        std::lock_guard<std::mutex> lock(mutex);
        ++callback_calls;
        if (failed) {
            ++callback_failures;
        }
        --callbacks_in_flight;
        cv.notify_all();
    }

    server_prompt_cache_io_config config;
    bool                          valid = false;
    mutable std::mutex            mutex;
    std::mutex                    join_mutex;
    std::condition_variable       cv;
    std::thread                   worker;
    bool                          accepting            = false;
    bool                          fast_stop            = false;
    bool                          worker_running       = false;
    bool                          startup_cleanup_done = false;
    uint64_t                      next_job_id          = 1;
    uint64_t                      temp_nonce           = 0;

    std::deque<work_item>                          queue;
    std::vector<server_prompt_cache_io_completion> completion_queue;
    std::map<key, std::shared_ptr<write_record>>   records;
    std::map<uint64_t, uint64_t>                   latest_generation;
    std::map<std::string, path_owner>              path_owners;
    wake_callback_handle                           wake_callback;

    uint32_t active_jobs             = 0;
    uint32_t writing_jobs            = 0;
    uint64_t ram_held_bytes          = 0;
    uint64_t disk_reserved_bytes     = 0;
    uint64_t disk_committed_bytes    = 0;
    uint64_t disk_uncertain_bytes    = 0;
    uint64_t initial_disk_bytes      = 0;
    uint64_t accepted_writes         = 0;
    uint64_t durable_writes          = 0;
    uint64_t retired_writes          = 0;
    uint64_t failed_writes           = 0;
    uint64_t uncertain_writes        = 0;
    uint64_t completed_cleanups      = 0;
    uint64_t failed_cleanups         = 0;
    uint64_t abandoned_temps_removed = 0;
    uint64_t callback_calls          = 0;
    uint64_t callback_failures       = 0;
    uint32_t callbacks_in_flight     = 0;
    int      last_error              = 0;
    uint64_t last_error_ns           = 0;
    uint64_t cooldown_until_ns       = 0;
    uint64_t cooldown_wake_deadline_ns = 0;
};

server_prompt_cache_io::server_prompt_cache_io(server_prompt_cache_io_config config) :
    state(std::make_unique<impl>(std::move(config))) {}

server_prompt_cache_io::~server_prompt_cache_io() = default;

server_prompt_cache_io_enqueue_result server_prompt_cache_io::submit(server_prompt_cache_io_write write) {
    return state->submit(std::move(write));
}

server_prompt_cache_io_status server_prompt_cache_io::adopt_existing(uint64_t            entry_id,
                                                                     uint64_t            generation,
                                                                     const std::string & exact_path,
                                                                     uint64_t            bytes) {
    return state->adopt_existing(entry_id, generation, exact_path, bytes);
}

server_prompt_cache_io_status server_prompt_cache_io::cancel(uint64_t entry_id, uint64_t generation) {
    return state->cancel(entry_id, generation);
}

server_prompt_cache_io_status server_prompt_cache_io::retire(uint64_t entry_id, uint64_t generation) {
    return state->retire(entry_id, generation);
}

server_prompt_cache_io_enqueue_result server_prompt_cache_io::cleanup_exact(
    uint64_t                              entry_id,
    uint64_t                              generation,
    const std::string &                   exact_path,
    const server_prompt_cache_io_faults & faults) {
    return state->cleanup_exact(entry_id, generation, exact_path, faults, false);
}

server_prompt_cache_io_enqueue_result server_prompt_cache_io::cleanup_abandoned_temp(
    const std::string &                   exact_path,
    const server_prompt_cache_io_faults & faults) {
    return state->cleanup_exact(0, 0, exact_path, faults, true);
}

server_prompt_cache_io_payload server_prompt_cache_io::pending_payload(uint64_t entry_id, uint64_t generation) const {
    return state->pending_payload(entry_id, generation);
}

server_prompt_cache_io_snapshot server_prompt_cache_io::inspect(uint64_t entry_id, uint64_t generation) const {
    return state->inspect(entry_id, generation);
}

std::vector<server_prompt_cache_io_completion> server_prompt_cache_io::poll_completions(size_t max_count) {
    return state->poll_completions(max_count);
}

server_prompt_cache_io_stats server_prompt_cache_io::stats() const {
    return state->stats();
}

void server_prompt_cache_io::set_wake_callback(std::function<void()> callback) {
    state->set_wake_callback(std::move(callback));
}

void server_prompt_cache_io::clear_error_cooldown() {
    state->clear_error_cooldown();
}

server_prompt_cache_io_status server_prompt_cache_io::shutdown(server_prompt_cache_io_shutdown_mode mode) {
    return state->shutdown(mode);
}

bool server_prompt_cache_io::test_wait_for_pause(uint64_t                          job_id,
                                                 server_prompt_cache_io_test_pause point,
                                                 uint64_t                          timeout_ms) {
    return state->test_wait_for_pause(job_id, point, timeout_ms);
}

void server_prompt_cache_io::test_release_pause(uint64_t job_id) {
    state->test_release_pause(job_id);
}

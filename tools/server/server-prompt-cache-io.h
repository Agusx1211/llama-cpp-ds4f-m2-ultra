#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Host-only persistence for already-serialized prompt-cache files. The engine
// never owns a llama_context, slot, cache-list iterator, Metal object, or server
// callback. Callers hand it immutable bytes plus value metadata and reconcile
// generation-fenced completions on their owner thread.

enum class server_prompt_cache_io_lifecycle : uint8_t {
    ram_ready,
    queued,
    writing,
    durable,
    retired,
    write_failed,
    commit_uncertain,
};

const char * server_prompt_cache_io_lifecycle_name(server_prompt_cache_io_lifecycle state);

enum class server_prompt_cache_io_status : uint8_t {
    ok,
    invalid_argument,
    allocation_failed,
    not_found,
    sealed,
    job_limit,
    byte_limit,
    disk_limit,
    completion_limit,
    generation_not_newer,
    path_conflict,
    cooldown,
    cancel_requested,
    too_late,
    stale_generation,
    checksum_mismatch,
    io_error,
    no_space,
    cancelled,
    commit_uncertain,
};

const char * server_prompt_cache_io_status_name(server_prompt_cache_io_status status);

enum class server_prompt_cache_io_shutdown_mode : uint8_t {
    graceful,
    fast,
};

enum class server_prompt_cache_io_event_kind : uint8_t {
    write,
    cleanup,
};

// Deterministic value-only seams used by the filesystem lifecycle tests. They
// stay inert at their defaults and never carry a callback or server pointer to
// the worker. max_write_size exercises successful short writes; write_eintrs
// injects retryable EINTRs before real writes; fail_after_bytes caps the final
// successful write so the selected errno occurs at an exact byte boundary.
enum class server_prompt_cache_io_test_pause : uint8_t {
    none,
    before_write,
    before_publication_fence,
    after_publication_fence,
    after_rename,
};

struct server_prompt_cache_io_faults {
    uint64_t                          max_write_size                      = UINT64_MAX;
    uint32_t                          write_eintrs                        = 0;
    uint64_t                          fail_after_bytes                    = UINT64_MAX;
    int                               fail_write_errno                    = 0;
    int                               fail_file_fsync_errno               = 0;
    int                               fail_rename_errno                   = 0;
    int                               fail_directory_fsync_errno          = 0;
    int                               fail_unlink_errno                   = 0;
    bool                              fail_worker_exception_before_rename = false;
    server_prompt_cache_io_test_pause pause                               = server_prompt_cache_io_test_pause::none;
};

struct server_prompt_cache_io_config {
    // max_jobs bounds queued plus active writes and explicit cleanup jobs.
    // max_pending_bytes is authoritative for immutable payload RAM. A payload
    // in commit_uncertain remains charged until it is explicitly retired.
    uint32_t max_jobs          = 8;
    uint32_t max_completions   = 32;
    uint64_t max_pending_bytes = 1ull << 30;

    // 0 means unlimited. initial_disk_bytes lets the live cache owner include
    // files found by its existing startup rescan without exposing them here.
    uint64_t max_disk_bytes     = 0;
    uint64_t initial_disk_bytes = 0;

    uint64_t                             error_cooldown_ms   = 5000;
    server_prompt_cache_io_shutdown_mode destructor_shutdown = server_prompt_cache_io_shutdown_mode::graceful;

    // Startup cleanup is limited to regular files whose basenames are in this
    // engine's private `.pcache-io-*.tmp` namespace. Enumeration and deletion
    // happen on the worker before accepted jobs, never on the caller thread.
    std::vector<std::string> startup_cleanup_directories;
};

using server_prompt_cache_io_payload = std::shared_ptr<const std::vector<uint8_t>>;

struct server_prompt_cache_io_write {
    uint64_t                       entry_id   = 0;
    uint64_t                       generation = 0;
    std::string                    final_path;
    server_prompt_cache_io_payload payload;

    // When true, the worker hashes the immutable payload before opening a temp
    // file and rejects a mismatch without touching the destination namespace.
    bool     verify_checksum   = false;
    uint64_t expected_checksum = 0;

    server_prompt_cache_io_faults faults;
};

struct server_prompt_cache_io_enqueue_result {
    server_prompt_cache_io_status    status    = server_prompt_cache_io_status::invalid_argument;
    server_prompt_cache_io_lifecycle lifecycle = server_prompt_cache_io_lifecycle::ram_ready;
    uint64_t                         job_id    = 0;
    uint64_t                         bytes     = 0;
    // The same immutable object supplied by the caller. It remains useful on a
    // rejected admission and is held by the engine through queued/writing.
    server_prompt_cache_io_payload   payload;
};

struct server_prompt_cache_io_snapshot {
    server_prompt_cache_io_status    status             = server_prompt_cache_io_status::not_found;
    server_prompt_cache_io_lifecycle lifecycle          = server_prompt_cache_io_lifecycle::ram_ready;
    uint64_t                         entry_id           = 0;
    uint64_t                         generation         = 0;
    uint64_t                         job_id             = 0;
    uint64_t                         bytes              = 0;
    uint64_t                         checksum           = 0;
    bool                             is_latest          = false;
    bool                             cancel_requested   = false;
    bool                             retire_requested   = false;
    bool                             publication_fenced = false;
    bool                             cleanup_pending    = false;
    std::string                      final_path;
    std::string                      temp_path;
    uint64_t                         accepted_ns          = 0;
    uint64_t                         started_ns           = 0;
    uint64_t                         publication_fence_ns = 0;
    uint64_t                         completed_ns         = 0;
};

struct server_prompt_cache_io_completion {
    server_prompt_cache_io_event_kind event              = server_prompt_cache_io_event_kind::write;
    server_prompt_cache_io_status     status             = server_prompt_cache_io_status::invalid_argument;
    server_prompt_cache_io_lifecycle  lifecycle          = server_prompt_cache_io_lifecycle::ram_ready;
    uint64_t                          job_id             = 0;
    uint64_t                          entry_id           = 0;
    uint64_t                          generation         = 0;
    uint64_t                          bytes              = 0;
    uint64_t                          checksum           = 0;
    int                               os_error           = 0;
    bool                              stale_generation   = false;
    bool                              publication_fenced = false;
    bool                              exact_path_removed = false;
    std::string                       final_path;
    uint64_t                          accepted_ns          = 0;
    uint64_t                          started_ns           = 0;
    uint64_t                          publication_fence_ns = 0;
    uint64_t                          completed_ns         = 0;
};

struct server_prompt_cache_io_stats {
    bool     accepting               = false;
    bool     worker_running          = false;
    bool     startup_cleanup_done    = false;
    uint32_t active_jobs             = 0;
    uint32_t queued_jobs             = 0;
    uint32_t writing_jobs            = 0;
    uint32_t completion_backlog      = 0;
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
    int      last_error              = 0;
    uint64_t last_error_ns           = 0;
    uint64_t cooldown_until_ns       = 0;
};

// Whole-payload checksum used for optional caller/worker agreement. This is a
// fast corruption/truncation detector, not an authenticity primitive.
uint64_t server_prompt_cache_io_checksum(const void * data, size_t size, uint64_t seed = 0);

class server_prompt_cache_io {
  public:
    explicit server_prompt_cache_io(server_prompt_cache_io_config config);
    ~server_prompt_cache_io();

    server_prompt_cache_io(const server_prompt_cache_io &)             = delete;
    server_prompt_cache_io & operator=(const server_prompt_cache_io &) = delete;

    server_prompt_cache_io_enqueue_result submit(server_prompt_cache_io_write write);

    // Transfer a file already validated by the caller's startup rescan from
    // initial_disk_bytes into exact path ownership. No filesystem operation is
    // performed. Adoption is accepted only while no job is active. This is the
    // reconciliation seam that prevents a later same-path replacement from
    // double-counting startup-owned bytes.
    server_prompt_cache_io_status adopt_existing(uint64_t            entry_id,
                                                 uint64_t            generation,
                                                 const std::string & exact_path,
                                                 uint64_t            bytes);

    // Cancellation wins only before the publication fence. retire() also
    // handles post-fence/durable state and schedules an ownership-checked exact
    // path cleanup when necessary.
    server_prompt_cache_io_status cancel(uint64_t entry_id, uint64_t generation);
    server_prompt_cache_io_status retire(uint64_t entry_id, uint64_t generation);

    // Delete only the path currently owned by this exact generation. A stale
    // request is refused before unlink, so it cannot remove a newer file that
    // reused the same pathname.
    server_prompt_cache_io_enqueue_result cleanup_exact(uint64_t                              entry_id,
                                                        uint64_t                              generation,
                                                        const std::string &                   exact_path,
                                                        const server_prompt_cache_io_faults & faults = {});

    // Restricted to `.pcache-io-*.tmp` basenames and intended for a precise
    // path returned by startup discovery or telemetry.
    server_prompt_cache_io_enqueue_result cleanup_abandoned_temp(const std::string &                   exact_path,
                                                                 const server_prompt_cache_io_faults & faults = {});

    server_prompt_cache_io_payload                 pending_payload(uint64_t entry_id, uint64_t generation) const;
    server_prompt_cache_io_snapshot                inspect(uint64_t entry_id, uint64_t generation) const;
    std::vector<server_prompt_cache_io_completion> poll_completions(size_t max_count = SIZE_MAX);
    server_prompt_cache_io_stats                   stats() const;

    // The callback is a narrow wake signal. It runs after the engine lock is
    // released, is exception-contained, and may safely call polling/stats.
    // Replacing/clearing it waits out prior calls; destruction joins the
    // worker, so it cannot fire after either lifetime boundary. A callback
    // must not call set_wake_callback() or shutdown() recursively.
    void set_wake_callback(std::function<void()> callback);

    void                          clear_error_cooldown();
    server_prompt_cache_io_status shutdown(server_prompt_cache_io_shutdown_mode mode);

#ifdef SERVER_PROMPT_CACHE_IO_TESTING
    bool test_wait_for_pause(uint64_t job_id, server_prompt_cache_io_test_pause point, uint64_t timeout_ms);
    void test_release_pause(uint64_t job_id);
#endif

  private:
    struct impl;
    std::unique_ptr<impl> state;
};

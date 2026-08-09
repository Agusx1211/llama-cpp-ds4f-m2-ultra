#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace server_request_history {

constexpr uint32_t REQUEST_HISTORY_FORMAT_VERSION = 1;
constexpr uint32_t REQUEST_HISTORY_FRAME_VERSION  = 1;
constexpr uint64_t REQUEST_HISTORY_DEFAULT_RETENTION_NS = 3ULL * 24ULL * 60ULL * 60ULL * 1000ULL * 1000ULL * 1000ULL;

enum class status : uint8_t {
    ok = 0,
    disabled,
    invalid_config,
    stopped,
    timeout,
    owner_busy,
    path_security,
    not_found,
    malformed,
    truncated,
    trailing_data,
    checksum_mismatch,
    too_large,
    no_space,
    short_write,
    io_error,
    commit_uncertain,
};

const char * status_name(status value) noexcept;

enum class outcome : uint8_t {
    complete = 0,
    aborted,
    error,
    recovered_incomplete,
};

const char * outcome_name(outcome value) noexcept;

struct result {
    status value       = status::invalid_config;
    int    os_error    = 0;
    bool   durable     = false;
};

struct faults {
    // Test-only deterministic write seams. Production callers leave these at
    // their defaults. max_write_size exercises exact retry of short writes;
    // fail_after_bytes applies to the writer's lifetime byte count.
    uint64_t max_write_size   = UINT64_MAX;
    uint64_t fail_after_bytes = UINT64_MAX;
    bool     fail_no_space    = false;
    bool     inject_eintr_once = false;
    bool     fail_file_fsync   = false;
    bool     fail_directory_fsync = false;
    bool     fail_rename       = false;
    bool     crash_before_rename = false;
    bool     crash_after_rename  = false;
    bool     preserve_open_on_shutdown = false;

    // Called by the writer before each frame. Tests use it to model a slow
    // disk without sleeping in an HTTP/inference thread.
    std::function<void()> before_frame_write;

    // Namespace-race seams. Tests may exchange the named file with a
    // same-mode/same-size inode; production callers leave these empty.
    std::function<void(const std::string &)> before_publish_rename;
    std::function<void(const std::string &)> before_retention_unlink;
};

struct config {
    // Empty means disabled. Enabled roots must be canonical absolute paths;
    // every component is opened without following symlinks.
    std::string root_path;
    bool        require_private_root = true;

    // Retention uses wall-clock epoch time. Zero age disables only the age
    // bound; byte and file bounds remain mandatory and non-zero.
    uint64_t retention_age_ns  = REQUEST_HISTORY_DEFAULT_RETENTION_NS;
    uint64_t max_retained_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t max_retained_files = 100000;

    // Every body is split across bounded frames without changing its bytes.
    uint64_t max_frame_payload = 1024ULL * 1024ULL;

    // Queued plus writer-in-flight bulk request/non-stream response payloads
    // are globally bounded. Streaming uses a separate per-session frame
    // reservation so one large bulk request cannot prevent active streams
    // from draining. Payload memory is bounded by max_pending_bytes plus
    // active_streams * max_stream_pending_frames_per_session *
    // max_frame_payload (apart from small event/container overhead).
    uint64_t max_pending_bytes = 64ULL * 1024ULL * 1024ULL;
    uint32_t max_stream_pending_frames_per_session = 2;

    // finish() waits for file fsync + atomic rename + directory fsync. It
    // returns timeout, never a false durable success, after this bound.
    uint64_t terminal_ack_timeout_ms = 30000;

    // Deterministic clock seams used by host tests. Production uses
    // system_clock and steady_clock when these are empty.
    std::function<uint64_t()> wall_time_ns;
    std::function<uint64_t()> monotonic_time_ns;

    faults test_faults;
};

struct stats {
    uint64_t accepted_records    = 0;
    uint64_t durable_records     = 0;
    uint64_t incomplete_records  = 0;
    uint64_t failed_records      = 0;
    uint64_t ingress_failures    = 0;
    uint64_t append_failures     = 0;
    uint64_t terminal_failures   = 0;
    uint64_t recovered_records   = 0;
    uint64_t quarantined_records = 0;
    uint64_t namespace_scans     = 0;
    uint64_t retained_files      = 0;
    uint64_t retained_bytes      = 0;
    uint64_t pending_bytes       = 0;
    uint64_t peak_pending_bytes  = 0;
    uint64_t backpressure_waits  = 0;
    uint64_t terminal_timeouts   = 0;
    result   last_failure;
};

struct record_metadata {
    std::string file_name;
    uint64_t    record_id            = 0;
    uint64_t    started_wall_ns       = 0;
    uint64_t    started_monotonic_ns  = 0;
    uint64_t    ended_wall_ns         = 0;
    uint64_t    duration_ns           = 0;
    std::string method;
    std::string path;
    std::string requested_model;
    std::string requested_profile;
    std::string usage_json;
    int         http_status           = 0;
    outcome     terminal_outcome      = outcome::recovered_incomplete;
    bool        streaming             = false;
    bool        recovered             = false;
    bool        transport_complete_known = false;
    bool        transport_complete    = false;
    bool        terminal_present      = false;
    uint64_t    request_bytes         = 0;
    uint64_t    response_bytes        = 0;
    uint64_t    response_chunks       = 0;
    uint64_t    file_bytes            = 0;
};

struct record {
    record_metadata          metadata;
    std::string              request_body;
    std::string              response_body;
    std::vector<std::string> response_stream_chunks;
};

struct read_limits {
    uint64_t max_files            = 1024;
    // Bounds all directory entries examined, including unrelated names.
    uint64_t max_namespace_entries = 4096;
    uint64_t max_file_bytes       = 1024ULL * 1024ULL * 1024ULL;
    uint64_t max_total_body_bytes = 1024ULL * 1024ULL * 1024ULL;
};

class session;

class store {
  public:
    explicit store(config cfg);
    ~store();

    store(const store &)             = delete;
    store & operator=(const store &) = delete;

    // decoded_request_entity is copied before this returns. No headers or query
    // parameters are accepted by this API, preventing accidental secret
    // persistence at the storage boundary.
    std::shared_ptr<session> begin(
            const std::string & method,
            const std::string & path,
            const std::string & decoded_request_entity,
            const std::string & requested_model = {},
            const std::string & requested_profile = {});

    // Waits until all events accepted before this call have reached the
    // writer. Active records need not yet be terminal.
    result flush(uint64_t timeout_ms = 30000);
    // Rejects new records and cancels capacity waits without discarding events
    // already accepted. Used before stopping the HTTP worker pool; shutdown()
    // subsequently drains the preserved queue.
    void stop_accepting();
    result shutdown(bool drain = true);

    stats get_stats() const;

  private:
    friend class session;
    struct impl;
    std::shared_ptr<impl> data;
};

class session {
  public:
    ~session();

    session(const session &)             = delete;
    session & operator=(const session &) = delete;

    // Set stream_chunk only for one exact byte range handed successfully to
    // the HTTP sink. Large chunks are split internally and reconstructed by
    // the reader with their original boundary.
    bool append_response(const void * data, size_t size, bool stream_chunk);
    bool append_response(const std::string & data, bool stream_chunk) {
        return append_response(data.data(), data.size(), stream_chunk);
    }

    // JSON is stored verbatim as metadata. It is derived from response bodies,
    // never from request headers.
    bool set_usage_json(std::string usage_json);

    // Diagnostic snapshot for a preceding false append/set_usage result.
    result failure_status() const;

    // A successful result proves file fsync, atomic publication, and root
    // directory fsync. Timeout/failure is explicit and never reports durable.
    result finish(
            int http_status,
            outcome terminal_outcome,
            bool streaming,
            bool transport_complete = false,
            bool transport_complete_known = false);

    uint64_t record_id() const noexcept;

  private:
    friend class store;
    session(std::shared_ptr<store::impl> owner, uint64_t id);

    struct state;
    std::unique_ptr<state> data;
};

class reader {
  public:
    explicit reader(std::string root_path, bool require_private_root = true);
    ~reader();

    reader(const reader &)             = delete;
    reader & operator=(const reader &) = delete;

    // Lists oldest first. Quarantine and owner-lock files are never exposed.
    result inspect(const read_limits & limits, std::vector<record_metadata> & records) const;

    // file_name must be one exact name returned by inspect(); separators,
    // traversal components, symlinks and non-private files are rejected.
    result read(const std::string & file_name, const read_limits & limits, record & output) const;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

// Keep capture policy reviewable and independent of handler JSON transforms.
// path_prefix is stripped only when it is an exact leading path component.
bool should_capture(const std::string & method, const std::string & path, const std::string & path_prefix = {});

}  // namespace server_request_history

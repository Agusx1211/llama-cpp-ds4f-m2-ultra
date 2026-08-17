#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace server_capture {

// Sidecar request-stream log: one JSON line per completed request holding the
// prompt and committed token streams plus sampler/model identity. This is the
// deliberately simple JSONL prototype from the 2026-08-03 capture design (the
// cycle shard store stays the hardened binary path); training joins the two
// through the shared salted request tag.
//
// The producer copies token vectors under a short lock and never blocks on
// I/O: a bounded queue drops the newest record when the writer falls behind.
// Files are 0600 inside a 0700 root; rotation caps file size and retention
// deletes oldest-first by total bytes. Partial trailing lines from a crash are
// tolerated by consumers skipping unparseable lines.

struct request_record {
    uint64_t request_id   = 0;  // raw ID: used only to derive the salted tag, never written
    int32_t  slot_id      = -1;
    int64_t  wall_time_s  = 0;
    uint64_t monotonic_ns = 0;

    std::vector<int32_t> prompt_tokens;
    std::vector<int32_t> generated_tokens;
    int32_t              n_prompt_cached = 0;

    // final-state and speculative telemetry
    bool    truncated        = false;
    int32_t n_decoded        = 0;
    int32_t n_draft_total    = 0;
    int32_t n_draft_accepted = 0;
    double  prompt_ms        = 0.0;
    double  generation_ms    = 0.0;

    // sampler identity
    float    temperature = 0.0f;
    int32_t  top_k       = 0;
    float    top_p       = 0.0f;
    float    min_p       = 0.0f;
    uint32_t seed        = 0;
};

struct request_log_stats {
    uint64_t enqueued        = 0;
    uint64_t dropped         = 0;
    uint64_t written_records = 0;
    uint64_t written_bytes   = 0;
    uint64_t failed_writes   = 0;
    bool     worker_failed   = false;
};

struct request_log_config {
    std::string root_path;

    // Same salt as the cycle store so tags join across both logs.
    std::array<uint8_t, 16> identity_salt = {};

    size_t   queue_capacity  = 64;
    uint64_t max_file_bytes  = 64ull * 1024 * 1024;
    uint64_t max_total_bytes = 8ull * 1024 * 1024 * 1024;

    // Written once as the first line of every file so records stay
    // interpretable after model or runtime changes.
    std::string runtime_commit;
    std::string model_identity;
    std::string draft_identity;
    std::string speculative_config;
};

class request_log {
  public:
    // Throws std::runtime_error when the root cannot be created/secured.
    explicit request_log(request_log_config config);
    ~request_log();

    request_log(const request_log &)             = delete;
    request_log & operator=(const request_log &) = delete;

    bool try_enqueue(request_record && record) noexcept;

    request_log_stats stats() const noexcept;

    // Drains the queue and joins the worker. Safe to call repeatedly.
    void shutdown();

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

}  // namespace server_capture

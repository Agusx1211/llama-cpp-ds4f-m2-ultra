#pragma once

#include "llama-snapshot-store.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

constexpr uint32_t LLAMA_SNAPSHOT_MIN_STAGING_BUFFERS = 2;
constexpr uint32_t LLAMA_SNAPSHOT_MAX_STAGING_BUFFERS = 3;

enum class llama_snapshot_worker_status : uint8_t {
    ok,
    invalid_argument,
    busy,
    would_block,
    stale_job,
    job_complete,
    wrong_job_kind,
    source_failed,
    sink_failed,
    allocation_failed,
    cancelled,
    stopped,
};

const char * llama_snapshot_worker_status_name(llama_snapshot_worker_status status);

enum class llama_snapshot_worker_job_kind : uint8_t {
    none,
    write,
    read,
};

struct llama_snapshot_worker_config {
    llama_snapshot_store_config store;
    // The buffers and their full chunk-sized byte storage are allocated once
    // at construction. Two slots form the default producer/I/O pipeline; a
    // third is allowed only for target overlap measurements.
    uint32_t staging_buffers = LLAMA_SNAPSHOT_MIN_STAGING_BUFFERS;
};

struct llama_snapshot_worker_begin_result {
    llama_snapshot_worker_status   status              = llama_snapshot_worker_status::invalid_argument;
    llama_snapshot_worker_job_kind kind                = llama_snapshot_worker_job_kind::none;
    uint64_t                       job_id              = 0;
    uint32_t                       chunk_count         = 0;
    uint64_t                       total_payload_bytes = 0;
};

using llama_snapshot_stage_fill = std::function<llama_snapshot_status(uint8_t * destination, size_t size)>;
using llama_snapshot_stage_consume =
    std::function<llama_snapshot_status(const uint8_t * source, size_t size)>;

struct llama_snapshot_worker_transfer_result {
    llama_snapshot_worker_status status = llama_snapshot_worker_status::invalid_argument;
    uint32_t                     index  = 0;
    uint64_t                     bytes  = 0;
};

struct llama_snapshot_worker_completion {
    llama_snapshot_worker_status   status              = llama_snapshot_worker_status::invalid_argument;
    bool                           complete            = false;
    llama_snapshot_worker_job_kind kind                = llama_snapshot_worker_job_kind::none;
    llama_snapshot_status          operation_status    = llama_snapshot_status::invalid_argument;
    uint64_t                       generation          = 0;
    uint64_t                       total_payload_bytes = 0;
    int                            os_error            = 0;
    bool                           committed           = false;
};

struct llama_snapshot_worker_stats {
    uint32_t staging_buffers        = 0;
    uint64_t staging_buffer_bytes   = 0;
    uint32_t staging_in_use         = 0;
    uint32_t staging_high_watermark = 0;
    uint64_t write_staged_chunks    = 0;
    uint64_t write_released_chunks  = 0;
    uint64_t read_staged_chunks     = 0;
    uint64_t read_consumed_chunks   = 0;
    uint64_t dropped_chunks         = 0;
    uint64_t completed_jobs         = 0;
    uint64_t failed_jobs            = 0;
    uint64_t cancelled_jobs         = 0;
};

// One worker owns one active job and one bounded staging queue. Write producers
// and read consumers are non-blocking: would_block is the backpressure signal.
// wait() on a read completes only after every staged chunk is consumed or the
// job is cancelled/failed. The fill/consume callbacks are host seams for the
// next Metal-adapter step and run on the caller thread.
class llama_snapshot_worker {
  public:
    explicit llama_snapshot_worker(llama_snapshot_worker_config config);
    ~llama_snapshot_worker();

    llama_snapshot_worker(const llama_snapshot_worker &)             = delete;
    llama_snapshot_worker & operator=(const llama_snapshot_worker &) = delete;

    llama_snapshot_worker_begin_result begin_write(
            const llama_snapshot_metadata & metadata,
            uint64_t                        total_payload_bytes,
            const llama_snapshot_faults &   faults = {});
    llama_snapshot_worker_begin_result begin_read(const llama_snapshot_identity & expected_identity);

    llama_snapshot_worker_transfer_result try_stage_next(
            uint64_t job_id, const llama_snapshot_stage_fill & fill);
    llama_snapshot_worker_transfer_result try_consume_next(
            uint64_t job_id, const llama_snapshot_stage_consume & consume);

    llama_snapshot_worker_status     cancel(uint64_t job_id);
    llama_snapshot_worker_completion poll(uint64_t job_id) const;
    llama_snapshot_worker_completion wait(uint64_t job_id);
    llama_snapshot_worker_stats      stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> state;
};

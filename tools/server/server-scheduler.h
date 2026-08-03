#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace server_scheduler {

enum class lane : uint8_t {
    low = 0,
    normal,
    fast,
    count,
};

constexpr size_t lane_count = static_cast<size_t>(lane::count);

enum class feasibility : uint8_t {
    feasible_now,
    temporarily_blocked,
    impossible,
};

enum class service_disposition : uint8_t {
    complete,
    requeue,
    cancelled,
};

// These codes are stable simulator/output identifiers. Additions must not
// renumber existing values because replay artifacts use their integer form.
enum class reason_code : uint16_t {
    none = 0,

    admission_ready            = 100,
    admission_capacity_wait    = 101,
    reject_invalid_request     = 102,
    reject_duplicate_id        = 103,
    reject_queue_full          = 104,
    reject_context_limit       = 105,
    reject_capacity_impossible = 106,

    wait_empty               = 200,
    wait_no_feasible_request = 201,
    wait_service_in_flight   = 202,

    lane_initial_priority = 300,
    lane_hdrr_debt        = 301,
    lane_work_conserving  = 302,

    request_fifo                    = 400,
    request_virtual_runtime         = 401,
    request_cache_affinity          = 402,
    request_bypass_protected        = 403,
    request_aged                    = 404,
    request_feasible_behind_blocked = 405,

    service_complete         = 500,
    service_lease_requeue    = 501,
    service_cancelled        = 502,
    service_invalid_decision = 503,

    width_empty            = 600,
    width_lane_cap         = 601,
    width_profiled_shape   = 602,
    width_avoid_low_valley = 603,

    replay_arrival       = 700,
    replay_stalled       = 701,
    replay_limit_reached = 702,
    replay_restore_start = 703,
    replay_restore_ready = 704,
    replay_spill_start   = 705,
    replay_spill_done    = 706,

    kv_physical_pressure_retry  = 800,
    kv_physical_pressure_victim = 801,
};

const char * to_string(lane value);
const char * to_string(reason_code value);

struct lane_descriptor {
    lane         id;
    const char * name;
    uint32_t     weight;
    uint32_t     decode_cap;
    uint32_t     queue_cap;
};

const std::array<lane_descriptor, lane_count> & default_lane_descriptors();

struct resource_vector {
    // Phase 1 may map the dimensions to raw, CSA, HCA, indexer, rollback,
    // COW, or other physical pools. The policy never interprets the units.
    static constexpr size_t max_dimensions = 8;

    std::array<uint64_t, max_dimensions> units = {};

    bool operator==(const resource_vector & other) const;
};

struct feasibility_quote {
    feasibility     state = feasibility::feasible_now;
    resource_vector delta;
    uint32_t        limiting_dimension = 0;
};

struct request {
    uint64_t id                      = 0;
    lane     priority                = lane::low;
    uint64_t arrival_us              = 0;
    uint64_t virtual_runtime_us      = 0;
    uint64_t prompt_tokens           = 0;
    uint64_t cached_prompt_tokens    = 0;
    uint64_t requested_output_tokens = 0;  // zero means unlimited
    uint64_t decode_runway_tokens    = 1;
};

struct admission_result {
    bool        accepted           = false;
    bool        ready              = false;
    reason_code reason             = reason_code::none;
    uint32_t    limiting_dimension = 0;
};

struct candidate_evaluation {
    feasibility state              = feasibility::feasible_now;
    uint64_t    predicted_gpu_us   = 1;
    uint64_t    cached_prefix_us   = 0;
    uint64_t    restore_cost_us    = 0;
    uint32_t    limiting_dimension = 0;
};

using evaluation_provider = std::function<candidate_evaluation(const request &)>;

struct scheduler_config {
    std::array<lane_descriptor, lane_count> lanes               = default_lane_descriptors();
    uint64_t                                context_tokens      = 1048576;
    uint64_t                                fairness_quantum_us = 1000;
    size_t                                  cache_lookahead     = 8;
    uint32_t                                max_bypasses        = 3;
    uint64_t                                aging_interval_us   = 1000;
    uint64_t                                aging_credit_us     = 250;
    uint64_t                                max_aging_credit_us = 1000000;
};

struct service_decision {
    bool        selected            = false;
    uint64_t    decision_id         = 0;
    uint64_t    request_id          = 0;
    lane        selected_lane       = lane::low;
    reason_code reason              = reason_code::none;
    reason_code lane_reason         = reason_code::none;
    uint64_t    predicted_gpu_us    = 0;
    uint64_t    affinity_bonus_us   = 0;
    uint32_t    bypass_count_before = 0;
    uint32_t    blocked_before      = 0;
    uint32_t    limiting_dimension  = 0;
};

struct completion_result {
    bool        completed = false;
    reason_code reason    = reason_code::none;
};

struct decode_width_decision {
    uint32_t    width  = 0;
    reason_code reason = reason_code::width_empty;
};

struct lane_snapshot {
    size_t   queued            = 0;
    int64_t  hdrr_credit_units = 0;
    uint64_t actual_gpu_us     = 0;
    uint64_t dispatches        = 0;
    uint64_t oldest_arrival_us = 0;
};

class scheduler {
  public:
    explicit scheduler(scheduler_config config = {});

    admission_result  admit(const request & req, const feasibility_quote & quote);
    service_decision  select_next(uint64_t now_us, const evaluation_provider & evaluate);
    completion_result complete_service(uint64_t decision_id, uint64_t actual_gpu_us, service_disposition disposition);

    decode_width_decision choose_decode_width(lane   dominant_lane,
                                              size_t runnable_in_dominant_lane,
                                              size_t total_runnable,
                                              bool   cross_lane_fill_is_profiled_safe) const;

    bool                     contains(uint64_t request_id) const;
    size_t                   queued(lane priority) const;
    size_t                   queued_total() const;
    lane_snapshot            snapshot(lane priority) const;
    const scheduler_config & config() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;

  public:
    ~scheduler();
    scheduler(scheduler && other) noexcept;
    scheduler & operator=(scheduler && other) noexcept;
    scheduler(const scheduler &)             = delete;
    scheduler & operator=(const scheduler &) = delete;
};

enum class kv_pressure_action : uint8_t {
    retry,
    victim,
};

struct kv_pressure_event {
    kv_pressure_action action             = kv_pressure_action::retry;
    reason_code        reason             = reason_code::kv_physical_pressure_retry;
    uint32_t           attempt            = 0;
    uint32_t           next_batch_size    = 0;
    int32_t            victim_sequence    = -1;
    uint32_t           victim_span_tokens = 0;

    bool operator==(const kv_pressure_event & other) const;
};

struct kv_pressure_snapshot {
    uint64_t total   = 0;
    uint64_t retries = 0;
    uint64_t victims = 0;

    bool operator==(const kv_pressure_snapshot & other) const;
};

// Physical pressure is transient for a bounded number of attempts. Once the
// bound is exhausted, the sequence at the head of the failed batch is the
// deterministic victim; the caller skips its whole contiguous token span and
// retains every other sequence.
class kv_pressure_controller {
  public:
    explicit kv_pressure_controller(uint32_t max_retries = 2);

    kv_pressure_event on_pressure(
            uint32_t current_batch_size,
            int32_t victim_sequence,
            uint32_t victim_span_tokens);

    void reset_attempts();
    kv_pressure_snapshot snapshot() const;

  private:
    uint32_t max_retries;
    uint32_t attempts = 0;
    kv_pressure_snapshot counters;
};

enum class replay_event_kind : uint8_t {
    arrival     = 0,
    admission   = 1,
    dispatch    = 2,
    completion  = 3,
    stalled     = 4,
    limit       = 5,
    io_start    = 6,
    io_complete = 7,
};

struct trace_job {
    request         req;
    resource_vector page_demand;
    uint64_t        observed_output_tokens = 1;
    uint64_t        service_gpu_us         = 1000;
    uint64_t        cached_prefix_us       = 0;
    uint64_t        restore_cost_us        = 0;  // policy estimate used for cache-affinity ranking
    uint64_t        restore_io_us          = 0;  // observed asynchronous input duration
    uint64_t        spill_io_us            = 0;  // observed asynchronous output duration
};

struct replay_trace {
    scheduler_config       policy;
    resource_vector        capacity;
    resource_vector        safety_watermark;
    uint64_t               decode_lease_tokens = 1;
    uint64_t               max_dispatches      = 10000;
    std::vector<trace_job> jobs;
};

struct replay_event {
    replay_event_kind kind        = replay_event_kind::arrival;
    uint64_t          time_us     = 0;
    uint64_t          request_id  = 0;
    lane              priority    = lane::low;
    reason_code       reason      = reason_code::none;
    reason_code       lane_reason = reason_code::none;
    uint64_t          gpu_us      = 0;
    uint64_t          io_us       = 0;

    bool operator==(const replay_event & other) const;
};

struct replay_result {
    std::vector<replay_event> events;
    resource_vector           resident;
    uint64_t                  dispatches  = 0;
    uint64_t                  end_time_us = 0;

    bool operator==(const replay_result & other) const;
};

class simulator {
  public:
    // Replay models one non-preemptive GPU dispatch at a time. Restore and
    // spill durations are asynchronous: they hold the request's resource
    // vector but never advance or block unrelated GPU work. Decode cohort
    // widths remain independent until a batch-feasibility interface can
    // account for the whole cohort.
    replay_result replay(const replay_trace & trace) const;
};

}  // namespace server_scheduler

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace server_request_registry {

using request_id = uint64_t;
using slot_id    = uint32_t;

constexpr slot_id no_slot = UINT32_MAX;

// The lane is accepted only at registration and cannot be changed afterward.
// It is expected to come from trusted server-side classification, not request
// parameters supplied by a client.
enum class trusted_lane : uint8_t {
    low = 0,
    normal,
    fast,
    count,
};

enum class queue_state : uint8_t {
    none = 0,
    admission,
    ready,
    blocked,
    count,
};

enum class lifecycle : uint8_t {
    absent = 0,
    registered,
    queued,
    executing,
    completed,
    cancelled,
    timed_out,
    failed,
    count,
};

bool is_terminal(lifecycle value);

// Stable, low-cardinality reason values. They are suitable for aggregate
// counters; request IDs, slot IDs, and other request-derived data must never be
// promoted to metric label values.
enum class reason_code : uint16_t {
    none = 0,

    registered       = 100,
    admission_wait   = 200,
    admission_ready  = 201,
    capacity_blocked = 202,

    dispatched  = 300,
    slot_rebind = 301,
    yielded     = 302,

    progress_observed = 400,

    client_cancel = 500,
    server_cancel = 501,
    queue_timeout = 502,
    // Absolute run backstop exceeded (total bound runtime).
    run_timeout   = 503,
    // Progress-based run expiry: no slot advancement for the stall threshold.
    run_stall     = 504,

    completed      = 600,
    request_failed = 601,
};

enum class result_code : uint8_t {
    ok = 0,
    invalid_registration,
    duplicate_request,
    request_capacity_exhausted,
    unknown_request,
    stale_handle,
    terminal_request,
    invalid_reason,
    invalid_queue_state,
    invalid_transition,
    invalid_progress,
    request_not_runnable,
    slot_out_of_range,
    slot_occupied,
    binding_capacity_exhausted,
    stale_lease,
    active_bindings,
    not_terminal,
    generation_exhausted,
};

struct request_handle {
    request_id id    = 0;
    uint64_t   epoch = 0;

    bool operator==(const request_handle & other) const;
};

struct request_counts {
    uint64_t prompt_tokens           = 0;
    uint64_t cached_prompt_tokens    = 0;
    uint64_t requested_output_tokens = 0;  // zero means unlimited
    uint64_t observed_output_tokens  = 0;

    bool operator==(const request_counts & other) const;
};

struct request_estimates {
    uint64_t predicted_prefill_us    = 0;
    uint64_t predicted_cache_restore_us = 0;
    uint64_t predicted_decode_us     = 0;
    uint64_t predicted_gpu_us        = 0;
    uint64_t predicted_memory_bytes  = 0;
    uint64_t predicted_output_tokens = 0;

    bool operator==(const request_estimates & other) const;
};

struct request_registration {
    request_id        id                 = 0;
    trusted_lane      lane               = trusted_lane::low;
    uint64_t          arrival_us         = 0;
    uint64_t          virtual_runtime_us = 0;
    int64_t           debt_us            = 0;
    request_counts    counts;
    request_estimates estimates;
    reason_code       reason = reason_code::registered;
};

struct request_progress {
    uint64_t          virtual_runtime_us     = 0;
    int64_t           debt_us                = 0;
    uint64_t          observed_output_tokens = 0;
    request_estimates estimates;
};

struct binding_lease {
    request_handle request;
    slot_id        slot            = no_slot;
    uint64_t       slot_generation = 0;

    bool operator==(const binding_lease & other) const;
};

// Snapshots are detached value copies. Later registry mutations cannot change
// a previously returned snapshot.
struct request_snapshot {
    request_handle             handle;
    trusted_lane               lane               = trusted_lane::low;
    uint64_t                   arrival_us         = 0;
    uint64_t                   virtual_runtime_us = 0;
    int64_t                    debt_us            = 0;
    request_counts             counts;
    request_estimates          estimates;
    reason_code                last_reason      = reason_code::none;
    lifecycle                  state            = lifecycle::registered;
    queue_state                queue            = queue_state::none;
    bool                       cancel_requested = false;
    bool                       timeout_expired  = false;
    uint64_t                   revision         = 0;
    std::vector<binding_lease> bindings;

    bool operator==(const request_snapshot & other) const;
};

enum class event_kind : uint8_t {
    registered = 0,
    queue_changed,
    progress_updated,
    bound,
    unbound,
    rebound,
    cancel_requested,
    timeout_expired,
    terminal,
    removed,
};

struct registry_event {
    uint64_t       sequence = 0;
    uint64_t       at_us    = 0;
    event_kind     kind     = event_kind::registered;
    request_handle request;
    trusted_lane   lane             = trusted_lane::low;
    reason_code    reason           = reason_code::none;
    lifecycle      lifecycle_before = lifecycle::absent;
    lifecycle      lifecycle_after  = lifecycle::absent;
    queue_state    queue_before     = queue_state::none;
    queue_state    queue_after      = queue_state::none;
    slot_id        slot_from        = no_slot;
    slot_id        slot_to          = no_slot;
    uint64_t       slot_generation  = 0;

    bool operator==(const registry_event & other) const;
};

struct event_log_snapshot {
    std::vector<registry_event> events;
    uint64_t                    first_sequence = 1;
    uint64_t                    next_sequence  = 1;
    uint64_t                    total_events   = 0;
    uint64_t                    dropped_events = 0;

    bool operator==(const event_log_snapshot & other) const;
};

struct registry_summary {
    size_t   active_requests = 0;
    size_t   occupied_slots  = 0;
    size_t   retained_events = 0;
    size_t   event_capacity  = 0;
    uint64_t total_events    = 0;
    uint64_t dropped_events  = 0;

    bool operator==(const registry_summary & other) const;
};

struct registry_config {
    size_t max_requests             = 4096;
    size_t max_slots                = 256;
    size_t max_bindings_per_request = 8;
    size_t event_capacity           = 1024;
};

struct operation_result {
    result_code code    = result_code::ok;
    bool        changed = false;

    operator bool() const;
};

struct registration_result {
    result_code    code = result_code::ok;
    request_handle handle;

    operator bool() const;
};

struct binding_result {
    result_code   code = result_code::ok;
    binding_lease lease;

    operator bool() const;
};

class request_registry {
  public:
    explicit request_registry(registry_config config = {});
    ~request_registry();

    request_registry(request_registry && other) noexcept;
    request_registry & operator=(request_registry && other) noexcept;
    request_registry(const request_registry &)             = delete;
    request_registry & operator=(const request_registry &) = delete;

    registration_result register_request(const request_registration & request);

    operation_result set_queue_state(request_handle handle, queue_state state, reason_code reason, uint64_t at_us);
    operation_result update_progress(request_handle           handle,
                                     const request_progress & progress,
                                     reason_code              reason,
                                     uint64_t                 at_us);

    binding_result   bind(request_handle handle, slot_id slot, reason_code reason, uint64_t at_us);
    operation_result unbind(binding_lease lease, reason_code reason, uint64_t at_us);
    binding_result   rebind(binding_lease lease, slot_id new_slot, reason_code reason, uint64_t at_us);

    operation_result mark_cancel_requested(request_handle handle, reason_code reason, uint64_t at_us);
    operation_result mark_timeout_expired(request_handle handle, reason_code reason, uint64_t at_us);
    operation_result mark_terminal(request_handle handle, lifecycle terminal_state, reason_code reason, uint64_t at_us);
    operation_result remove_terminal(request_handle handle, reason_code reason, uint64_t at_us);

    std::optional<request_snapshot> get(request_handle handle) const;
    std::vector<request_snapshot>   snapshot() const;
    event_log_snapshot              events() const;
    registry_summary                summary() const;
    const registry_config &         config() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

}  // namespace server_request_registry

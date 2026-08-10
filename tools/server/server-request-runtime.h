#pragma once

#include "server-request-registry.h"
#include "server-scheduler.h"

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace server_request_runtime {

enum class result_code : uint8_t {
    ok = 0,
    invalid_request,
    duplicate_request,
    queue_full,
    context_limit,
    capacity_impossible,
    registry_capacity,
    unknown_request,
    invalid_transition,
    deadline_expired,
};

enum class deadline_kind : uint8_t {
    queue = 0,
    // Absolute run backstop: total time bound to a slot exceeded the run cap.
    run,
    // Progress-based run expiry: the bound request stopped advancing (no
    // decoded/prefilled tokens) for longer than its stall threshold.
    run_stall,
};

struct expiration {
    uint64_t      request_id  = 0;
    deadline_kind kind        = deadline_kind::queue;
    bool          was_running = false;
    // Only meaningful for deadline_kind::run_stall: the observed gap between
    // the request's last recorded progress and the expiry decision.
    uint64_t      stalled_for_us = 0;
};

struct publication_result {
    bool       publish = false;
    expiration expired;
};

struct release_result {
    bool       released = false;
    expiration expired;

    operator bool() const { return released; }
};

enum class failure_publication_code : uint8_t {
    first_terminal = 0,
    already_terminal,
    unknown_request,
    transition_failed,
};

struct failure_publication_result {
    failure_publication_code code = failure_publication_code::unknown_request;

    operator bool() const { return code == failure_publication_code::first_terminal; }
};

struct request_metadata {
    uint64_t                                   id                 = 0;
    server_scheduler::lane                     lane               = server_scheduler::lane::normal;
    uint64_t                                   arrival_us         = 0;
    uint64_t                                   virtual_runtime_us = 0;
    int64_t                                    debt_us            = 0;
    server_request_registry::request_counts    counts;
    server_request_registry::request_estimates estimates;
    uint64_t                                   parent_id        = 0;
    uint64_t                                   queue_timeout_us = 0;
    // Absolute run backstop (zero selects the bounded server default).
    uint64_t                                   run_timeout_us   = 0;
    // Stall threshold for progress-based run expiry (zero selects the bounded
    // server default).
    uint64_t                                   run_stall_timeout_us = 0;
};

struct dispatch_permit_snapshot {
    std::array<size_t, server_scheduler::lane_count> claimed = {};
    std::array<size_t, server_scheduler::lane_count> bound   = {};
    size_t                                           total   = 0;
};

struct fast_refill_status {
    size_t                 fast_members_remaining = 0;
    uint64_t               remaining_us           = 0;
    bool                   window_open            = false;
    bool                   one_member_eligible_now = false;
};

struct fast_refill_snapshot {
    bool                   enabled                = false;
    size_t                 max_members_per_cohort = 0;
    uint64_t               window_us              = 0;
    bool                   cohort_active          = false;
    bool                   selection_open         = false;
    server_scheduler::lane dominant               = server_scheduler::lane::count;
    size_t                 cohort_limit           = 0;
    size_t                 fast_members_used      = 0;
    uint64_t               deadline_us            = 0;

    fast_refill_status status_at(uint64_t now_us, size_t permit_width) const;
};

struct admission_result {
    result_code                   code   = result_code::ok;
    server_scheduler::reason_code reason = server_scheduler::reason_code::none;
    uint32_t                      retry_after_seconds = 0;

    operator bool() const { return code == result_code::ok; }
};

struct dispatch_result {
    bool                          selected       = false;
    uint64_t                      request_id     = 0;
    server_scheduler::lane        lane           = server_scheduler::lane::low;
    server_scheduler::reason_code request_reason = server_scheduler::reason_code::none;
    server_scheduler::reason_code lane_reason    = server_scheduler::reason_code::none;
};

struct runtime_config {
    // Queue waits scale with the number of elastic lanes: admissions are
    // serialized behind the prefill owner, so a cold burst on -np 4 can
    // legitimately hold a request for several prefills (30 s expired queued
    // requests with 408 during 4-lane validation). Stuck clients are still
    // bounded by the running-state deadlines below and by disconnect-driven
    // task cancellation.
    static constexpr uint64_t queue_timeout_default_us  = 10ULL * 60 * 1000 * 1000;
    // Running requests expire on STALL, not on total runtime. Measured
    // 2026-08-08 (four independent kills): hard-task agent turns at ~280K
    // context legitimately decode continuously for >30 min (thinking-heavy
    // model, context-degraded tg ~12-18 t/s solo); the previous total-runtime
    // deadline killed such turns at 10:00 (three times) and, after being
    // raised, at exactly 30:00 while the request was actively decoding under
    // light load. A total-runtime bound of ANY value either kills healthy
    // long turns or stops protecting against hangs, so the run deadline is
    // now (a) a stall threshold on the gap since the request's slot last
    // advanced (decoded token, accepted draft tokens, or a processed prefill
    // ubatch) plus (b) a high absolute backstop.
    //
    // 5 min without any progress = hung: a genuinely alive request produces
    // a token every few hundred ms (worst measured decode gap is one 2048-tok
    // prefill chunk of another slot, ~10 s), so 5 min is ~30x slack while
    // still releasing wedged slots long before the old 10-min budget.
    static constexpr uint64_t run_stall_timeout_default_us = 5ULL * 60 * 1000 * 1000;
    // Absolute backstop for runaway-but-progressing requests. This is NOT a
    // typical-turn budget any more (that role failed, see above): it exists
    // only so an endlessly progressing request cannot hold a slot forever.
    static constexpr uint64_t run_timeout_default_us    = 2ULL * 60 * 60 * 1000 * 1000;
    static constexpr size_t   fast_refill_min_members   = 3;
    static constexpr size_t   fast_refill_max_members   = 16;
    static constexpr uint64_t fast_refill_min_window_us = 1000;
    static constexpr uint64_t fast_refill_max_window_us = 30ULL * 1000 * 1000;

    server_scheduler::scheduler_config       scheduler;
    server_request_registry::registry_config registry;
    uint64_t                                 default_queue_timeout_us     = queue_timeout_default_us;
    uint64_t                                 default_run_timeout_us       = run_timeout_default_us;
    uint64_t                                 default_run_stall_timeout_us = run_stall_timeout_default_us;
    uint32_t                                 overload_retry_after_seconds = 1;
    // Same-fast arrivals refill available width by default. Setting both
    // fields opts into the legacy bounded member/time limiter for controlled
    // experiments; zero leaves elastic refill unbounded.
    size_t                                   fast_refill_max_members_per_epoch = 0;
    uint64_t                                 fast_refill_window_us              = 0;
};

bool is_overload(result_code code);

// This adapter is deliberately task-payload agnostic. The server queue owns
// movable work items; this object owns their durable scheduling and binding
// metadata. Callers serialize mutations (server_queue does so with its mutex).
class request_runtime {
  public:
    explicit request_runtime(runtime_config config = {});

    admission_result admit(const request_metadata & request, bool scheduled = true);
    admission_result validate_dispatch_family(
            server_scheduler::lane priority,
            const std::array<size_t, server_scheduler::lane_count> & demand,
            size_t physical_slot_capacity) const;
    dispatch_result  take_next(uint64_t now_us, size_t physical_slot_capacity = 64);
    bool             mark_deferred(
            uint64_t request_id,
            uint64_t at_us,
            server_request_registry::reason_code reason =
                    server_request_registry::reason_code::capacity_blocked);
    admission_result resume(uint64_t request_id, uint64_t at_us);
    expiration        expiration_due(uint64_t request_id, uint64_t at_us) const;
    std::vector<expiration> expirations_due(uint64_t at_us) const;
    expiration        expire_prepared(const expiration & planned, uint64_t at_us);
    std::vector<expiration> expire_due(uint64_t now_us);

    bool bind_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    // Progress heartbeat for stall-based run expiry. Records that the
    // request's slot advanced (decoded token, accepted draft tokens, or a
    // processed prefill ubatch) and pushes its stall deadline forward. A
    // deliberately cheap timestamp store: no registry event, no state
    // transition. Returns false for unknown, terminal, or not-yet-running
    // requests (queued/deferred requests stay governed by the queue
    // deadline alone).
    bool note_progress(uint64_t request_id, uint64_t at_us);
    publication_result gate_result_publication(uint64_t request_id,
                                               server_request_registry::slot_id slot,
                                               bool                             final,
                                               uint64_t                         at_us);
    release_result release_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    bool cancel(uint64_t request_id, uint64_t at_us);
    bool cancel(uint64_t request_id,
                server_request_registry::reason_code reason,
                uint64_t at_us);
    bool fail(uint64_t request_id, uint64_t at_us);
    failure_publication_result failure_publication_status(uint64_t request_id) const;
    failure_publication_result fail_for_publication(uint64_t request_id, uint64_t at_us);
    failure_publication_result fail_one_for_publication(uint64_t request_id, uint64_t at_us);

    bool                                                   contains(uint64_t request_id) const;
    bool                                                   is_family_member(uint64_t request_id,
                                                                            uint64_t family_id) const;
    bool                                                   family_has_bindings(uint64_t request_id) const;
    size_t                                                 queued_total() const;
    std::vector<server_request_registry::request_snapshot> snapshot() const;
    server_request_registry::event_log_snapshot            events() const;
    server_request_registry::registry_summary              summary() const;
    dispatch_permit_snapshot                               permits() const;
    fast_refill_snapshot                                    fast_refill() const;

  private:
    struct record {
        enum class permit_state : uint8_t {
            none = 0,
            selected_unbound,
            bound,
        };

        request_metadata                                    metadata;
        server_request_registry::request_handle             handle;
        std::vector<server_request_registry::binding_lease> bindings;
        bool                                                scheduler_queued = false;
        server_request_registry::lifecycle                  terminal = server_request_registry::lifecycle::completed;
        server_request_registry::reason_code                cancel_reason =
                server_request_registry::reason_code::none;
        uint64_t                                            queue_deadline_us = 0;
        uint64_t                                            run_timeout_us    = 0;
        // Absolute backstop deadline; non-zero also marks the run phase as
        // started (first slot bind).
        uint64_t                                            run_deadline_us   = 0;
        uint64_t                                            run_stall_timeout_us = 0;
        // Last observed forward progress of the bound slot; initialized at
        // first bind, advanced by note_progress(). The stall deadline is
        // last_progress_us + run_stall_timeout_us.
        uint64_t                                            last_progress_us  = 0;
        // Exact timeout reason (run_timeout vs run_stall) captured at expiry
        // so the durable terminal event preserves the cause after the lease
        // is released.
        server_request_registry::reason_code                timeout_reason =
                server_request_registry::reason_code::none;
        permit_state                                        permit            = permit_state::none;
    };

    // The dominant lane is the highest lane admitted while any resident permit
    // survives. New work in that lane elastically refills free physical width;
    // a higher lane upgrades dominance immediately and lower work remains
    // resident for compute-level trickle service. The epoch resets only after
    // every permit drains. An explicit bounded-fast configuration limits
    // same-fast refills without changing the shipped elastic default.
    struct cohort_epoch {
        bool                   active                  = false;
        bool                   open                    = false;
        server_scheduler::lane dominant                = server_scheduler::lane::low;
        size_t                 limit                   = 0;
        size_t                 fast_members            = 0;
        uint64_t               fast_refill_deadline_us = 0;
    };

    admission_result enqueue(record & request, uint64_t at_us);
    expiration       planned_expiration_of(const record & request, uint64_t at_us) const;
    expiration       expire(std::map<uint64_t, record>::iterator it, deadline_kind kind, uint64_t at_us);
    expiration       expire_if_due(std::map<uint64_t, record>::iterator it, uint64_t at_us);
    bool             release_bound_slot(std::map<uint64_t, record>::iterator it,
                                        server_request_registry::slot_id     slot,
                                        uint64_t                             at_us);
    bool             finish(std::map<uint64_t, record>::iterator it,
                            server_request_registry::lifecycle   terminal,
                            server_request_registry::reason_code reason,
                            uint64_t                             at_us);
    bool                  cancel_one(uint64_t request_id,
                                     server_request_registry::reason_code reason,
                                     uint64_t at_us);
    bool                  fail_one(uint64_t request_id, uint64_t at_us);
    std::vector<uint64_t> terminal_targets(uint64_t request_id) const;
    void                  release_permit(record & request);
    bool                  fast_refill_enabled() const;
    bool                  fast_claim_within_epoch(size_t members, uint64_t now_us) const;

    server_scheduler::scheduler               scheduler;
    server_request_registry::request_registry registry;
    std::map<uint64_t, record>                records;
    uint64_t                                  default_queue_timeout_us;
    uint64_t                                  default_run_timeout_us;
    uint64_t                                  default_run_stall_timeout_us;
    uint32_t                                  overload_retry_after_seconds;
    size_t                                    fast_refill_max_members_per_epoch = 0;
    uint64_t                                  fast_refill_window_us              = 0;
    dispatch_permit_snapshot                  permit_counts;
    cohort_epoch                              cohort;
};

}  // namespace server_request_runtime

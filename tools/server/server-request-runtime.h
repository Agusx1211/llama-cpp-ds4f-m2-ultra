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
    run,
};

struct expiration {
    uint64_t      request_id  = 0;
    deadline_kind kind        = deadline_kind::queue;
    bool          was_running = false;
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
    uint64_t                                   run_timeout_us   = 0;
};

struct dispatch_permit_snapshot {
    std::array<size_t, server_scheduler::lane_count> claimed = {};
    std::array<size_t, server_scheduler::lane_count> bound   = {};
    size_t                                           total   = 0;
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
    static constexpr uint64_t queue_timeout_default_us  = 30ULL * 1000 * 1000;
    static constexpr uint64_t run_timeout_default_us    = 10ULL * 60 * 1000 * 1000;
    static constexpr size_t   fast_refill_min_members   = 3;
    static constexpr size_t   fast_refill_max_members   = 16;
    static constexpr uint64_t fast_refill_min_window_us = 1000;
    static constexpr uint64_t fast_refill_max_window_us = 30ULL * 1000 * 1000;

    server_scheduler::scheduler_config       scheduler;
    server_request_registry::registry_config registry;
    uint64_t                                 default_queue_timeout_us     = queue_timeout_default_us;
    uint64_t                                 default_run_timeout_us       = run_timeout_default_us;
    uint32_t                                 overload_retry_after_seconds = 1;
    // Both fields must be in range to opt into bounded same-fast-lane refill.
    // The member limit counts every fast permit first claimed in one cohort,
    // while the window starts when that cohort first becomes fast-dominant.
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
    bool             mark_deferred(uint64_t request_id, uint64_t at_us);
    admission_result resume(uint64_t request_id, uint64_t at_us);
    expiration        expiration_due(uint64_t request_id, uint64_t at_us) const;
    std::vector<expiration> expirations_due(uint64_t at_us) const;
    expiration        expire_prepared(const expiration & planned, uint64_t at_us);
    std::vector<expiration> expire_due(uint64_t now_us);

    bool bind_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    publication_result gate_result_publication(uint64_t request_id,
                                               server_request_registry::slot_id slot,
                                               bool                             final,
                                               uint64_t                         at_us);
    release_result release_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    bool cancel(uint64_t request_id, uint64_t at_us);
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
        uint64_t                                            queue_deadline_us = 0;
        uint64_t                                            run_timeout_us    = 0;
        uint64_t                                            run_deadline_us   = 0;
        permit_state                                        permit            = permit_state::none;
    };

    // A cohort normally closes after its initial fill opportunity and stays
    // closed across staggered releases. An explicitly configured fast epoch
    // may refill only its fast member within the bounded member/time budget.
    // Every cohort resets after all permits drain, and each higher lane may
    // otherwise reopen it once as a bounded priority upgrade.
    struct cohort_epoch {
        bool                   active                  = false;
        bool                   open                    = false;
        server_scheduler::lane dominant                = server_scheduler::lane::low;
        size_t                 limit                   = 0;
        size_t                 fast_members            = 0;
        uint64_t               fast_refill_deadline_us = 0;
    };

    admission_result enqueue(record & request, uint64_t at_us);
    expiration       expire(std::map<uint64_t, record>::iterator it, deadline_kind kind, uint64_t at_us);
    expiration       expire_if_due(std::map<uint64_t, record>::iterator it, uint64_t at_us);
    bool             release_bound_slot(std::map<uint64_t, record>::iterator it,
                                        server_request_registry::slot_id     slot,
                                        uint64_t                             at_us);
    bool             finish(std::map<uint64_t, record>::iterator it,
                            server_request_registry::lifecycle   terminal,
                            server_request_registry::reason_code reason,
                            uint64_t                             at_us);
    bool                  cancel_one(uint64_t request_id, uint64_t at_us);
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
    uint32_t                                  overload_retry_after_seconds;
    size_t                                    fast_refill_max_members_per_epoch = 0;
    uint64_t                                  fast_refill_window_us              = 0;
    dispatch_permit_snapshot                  permit_counts;
    cohort_epoch                              cohort;
};

}  // namespace server_request_runtime

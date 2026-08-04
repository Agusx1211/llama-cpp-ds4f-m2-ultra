#pragma once

#include "server-request-registry.h"
#include "server-scheduler.h"

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

struct request_metadata {
    uint64_t                                   id                 = 0;
    server_scheduler::lane                     lane               = server_scheduler::lane::normal;
    uint64_t                                   arrival_us         = 0;
    uint64_t                                   virtual_runtime_us = 0;
    int64_t                                    debt_us            = 0;
    server_request_registry::request_counts    counts;
    server_request_registry::request_estimates estimates;
    uint64_t                                   queue_timeout_us = 0;
    uint64_t                                   run_timeout_us   = 0;
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
    static constexpr uint64_t queue_timeout_default_us = 30ULL * 1000 * 1000;
    static constexpr uint64_t run_timeout_default_us   = 10ULL * 60 * 1000 * 1000;

    server_scheduler::scheduler_config       scheduler;
    server_request_registry::registry_config registry;
    uint64_t                                 default_queue_timeout_us     = queue_timeout_default_us;
    uint64_t                                 default_run_timeout_us       = run_timeout_default_us;
    uint32_t                                 overload_retry_after_seconds = 1;
};

bool is_overload(result_code code);

// This adapter is deliberately task-payload agnostic. The server queue owns
// movable work items; this object owns their durable scheduling and binding
// metadata. Callers serialize mutations (server_queue does so with its mutex).
class request_runtime {
  public:
    explicit request_runtime(runtime_config config = {});

    admission_result admit(const request_metadata & request, bool scheduled = true);
    dispatch_result  take_next(uint64_t now_us);
    bool             mark_deferred(uint64_t request_id, uint64_t at_us);
    admission_result resume(uint64_t request_id, uint64_t at_us);
    std::vector<expiration> expire_due(uint64_t now_us);

    bool bind_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    bool release_slot(uint64_t request_id, server_request_registry::slot_id slot, uint64_t at_us);
    bool cancel(uint64_t request_id, uint64_t at_us);
    bool fail(uint64_t request_id, uint64_t at_us);

    bool                                                   contains(uint64_t request_id) const;
    size_t                                                 queued_total() const;
    std::vector<server_request_registry::request_snapshot> snapshot() const;
    server_request_registry::event_log_snapshot            events() const;
    server_request_registry::registry_summary              summary() const;

  private:
    struct record {
        request_metadata                                    metadata;
        server_request_registry::request_handle             handle;
        std::vector<server_request_registry::binding_lease> bindings;
        bool                                                scheduler_queued = false;
        server_request_registry::lifecycle                  terminal = server_request_registry::lifecycle::completed;
        uint64_t                                            queue_deadline_us = 0;
        uint64_t                                            run_timeout_us    = 0;
        uint64_t                                            run_deadline_us   = 0;
    };

    admission_result enqueue(record & request, uint64_t at_us);
    expiration       expire(std::map<uint64_t, record>::iterator it, deadline_kind kind, uint64_t at_us);
    bool             finish(std::map<uint64_t, record>::iterator it,
                            server_request_registry::lifecycle   terminal,
                            server_request_registry::reason_code reason,
                            uint64_t                             at_us);

    server_scheduler::scheduler               scheduler;
    server_request_registry::request_registry registry;
    std::map<uint64_t, record>                records;
    uint64_t                                  default_queue_timeout_us;
    uint64_t                                  default_run_timeout_us;
    uint32_t                                  overload_retry_after_seconds;
};

}  // namespace server_request_runtime

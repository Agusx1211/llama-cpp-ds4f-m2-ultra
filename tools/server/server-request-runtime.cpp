#include "server-request-runtime.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace server_request_runtime {

namespace {

using namespace server_request_registry;
using namespace server_scheduler;

trusted_lane to_registry_lane(lane value) {
    switch (value) {
        case lane::low:
            return trusted_lane::low;
        case lane::normal:
            return trusted_lane::normal;
        case lane::fast:
            return trusted_lane::fast;
        case lane::count:
            break;
    }
    return trusted_lane::count;
}

result_code scheduler_rejection(server_scheduler::reason_code reason) {
    switch (reason) {
        case server_scheduler::reason_code::reject_duplicate_id:
            return result_code::duplicate_request;
        case server_scheduler::reason_code::reject_queue_full:
            return result_code::queue_full;
        case server_scheduler::reason_code::reject_context_limit:
            return result_code::context_limit;
        case server_scheduler::reason_code::reject_capacity_impossible:
            return result_code::capacity_impossible;
        default:
            return result_code::invalid_request;
    }
}

result_code registry_rejection(server_request_registry::result_code code) {
    switch (code) {
        case server_request_registry::result_code::duplicate_request:
            return result_code::duplicate_request;
        case server_request_registry::result_code::request_capacity_exhausted:
            return result_code::registry_capacity;
        case server_request_registry::result_code::invalid_registration:
            return result_code::invalid_request;
        default:
            return result_code::invalid_transition;
    }
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs ? std::numeric_limits<uint64_t>::max() : lhs + rhs;
}

}  // namespace

request_runtime::request_runtime(runtime_config config) :
    scheduler(std::move(config.scheduler)),
    registry(std::move(config.registry)) {}

admission_result request_runtime::admit(const request_metadata & request, bool scheduled) {
    const bool duplicate = records.count(request.id) != 0;
    if (request.id == 0 || duplicate || to_registry_lane(request.lane) == trusted_lane::count) {
        return { duplicate ? result_code::duplicate_request : result_code::invalid_request,
                 duplicate ? server_scheduler::reason_code::reject_duplicate_id :
                             server_scheduler::reason_code::reject_invalid_request };
    }

    record next;
    next.metadata = request;

    request_registration registration;
    registration.id                 = request.id;
    registration.lane               = to_registry_lane(request.lane);
    registration.arrival_us         = request.arrival_us;
    registration.virtual_runtime_us = request.virtual_runtime_us;
    registration.debt_us            = request.debt_us;
    registration.counts             = request.counts;
    registration.estimates          = request.estimates;

    const auto registered = registry.register_request(registration);
    if (!registered) {
        return { registry_rejection(registered.code), server_scheduler::reason_code::reject_invalid_request };
    }
    next.handle = registered.handle;

    auto inserted = records.emplace(request.id, std::move(next)).first;
    if (!scheduled) {
        if (!registry.set_queue_state(inserted->second.handle, queue_state::ready,
                                      server_request_registry::reason_code::admission_ready, request.arrival_us)) {
            finish(inserted, lifecycle::failed, server_request_registry::reason_code::request_failed,
                   request.arrival_us);
            return { result_code::invalid_transition, server_scheduler::reason_code::reject_invalid_request };
        }
        return { result_code::ok, server_scheduler::reason_code::admission_ready };
    }

    const admission_result result = enqueue(inserted->second, request.arrival_us);
    if (!result) {
        finish(inserted, lifecycle::failed, server_request_registry::reason_code::request_failed, request.arrival_us);
    }
    return result;
}

admission_result request_runtime::enqueue(record & request, uint64_t at_us) {
    server_scheduler::request scheduling;
    scheduling.id                      = request.metadata.id;
    scheduling.priority                = request.metadata.lane;
    scheduling.arrival_us              = request.metadata.arrival_us;
    scheduling.virtual_runtime_us      = request.metadata.virtual_runtime_us;
    scheduling.prompt_tokens           = request.metadata.counts.prompt_tokens;
    scheduling.cached_prompt_tokens    = request.metadata.counts.cached_prompt_tokens;
    scheduling.requested_output_tokens = request.metadata.counts.requested_output_tokens;
    scheduling.decode_runway_tokens    = 1;

    feasibility_quote quote;
    const auto        admitted = scheduler.admit(scheduling, quote);
    if (!admitted.accepted) {
        return { scheduler_rejection(admitted.reason), admitted.reason };
    }

    const queue_state queue  = admitted.ready ? queue_state::ready : queue_state::admission;
    const auto        reason = admitted.ready ? server_request_registry::reason_code::admission_ready :
                                                server_request_registry::reason_code::admission_wait;
    if (!registry.set_queue_state(request.handle, queue, reason, at_us)) {
        scheduler.cancel(request.metadata.id);
        return { result_code::invalid_transition, server_scheduler::reason_code::reject_invalid_request };
    }
    request.scheduler_queued = true;
    return { result_code::ok, admitted.reason };
}

dispatch_result request_runtime::take_next(uint64_t now_us) {
    const auto evaluate = [this](const server_scheduler::request & request) {
        candidate_evaluation result;
        const auto           it = records.find(request.id);
        if (it != records.end()) {
            result.predicted_gpu_us = std::max<uint64_t>(1, it->second.metadata.estimates.predicted_gpu_us);
            if (it->second.metadata.counts.cached_prompt_tokens != 0) {
                result.cached_prefix_us = it->second.metadata.estimates.predicted_prefill_us;
            }
        }
        return result;
    };

    const service_decision decision = scheduler.select_next(now_us, evaluate);
    if (!decision.selected) {
        return {};
    }

    auto it = records.find(decision.request_id);
    if (it == records.end() || !it->second.scheduler_queued) {
        scheduler.complete_service(decision.decision_id, 0, service_disposition::cancelled);
        return {};
    }

    record & request          = it->second;
    request.scheduler_queued  = false;
    const uint64_t charged_us = std::max<uint64_t>(1, decision.predicted_gpu_us);
    if (!scheduler.complete_service(decision.decision_id, charged_us, service_disposition::complete).completed) {
        return {};
    }

    const auto before = registry.get(request.handle);
    if (!before) {
        return {};
    }
    request.metadata.virtual_runtime_us = saturating_add(request.metadata.virtual_runtime_us, charged_us);
    request_progress progress;
    progress.virtual_runtime_us     = request.metadata.virtual_runtime_us;
    progress.debt_us                = request.metadata.debt_us;
    progress.observed_output_tokens = before->counts.observed_output_tokens;
    progress.estimates              = request.metadata.estimates;
    registry.update_progress(request.handle, progress, server_request_registry::reason_code::progress_observed, now_us);
    registry.set_queue_state(request.handle, queue_state::none, server_request_registry::reason_code::dispatched,
                             now_us);

    return { true, decision.request_id, decision.selected_lane, decision.reason, decision.lane_reason };
}

bool request_runtime::mark_deferred(uint64_t request_id, uint64_t at_us) {
    const auto it = records.find(request_id);
    return it != records.end() && !it->second.scheduler_queued && it->second.bindings.empty() &&
           registry.set_queue_state(it->second.handle, queue_state::blocked,
                                    server_request_registry::reason_code::capacity_blocked, at_us);
}

admission_result request_runtime::resume(uint64_t request_id, uint64_t at_us) {
    const auto it = records.find(request_id);
    if (it == records.end()) {
        return { result_code::unknown_request, server_scheduler::reason_code::reject_invalid_request };
    }
    if (it->second.scheduler_queued || !it->second.bindings.empty()) {
        return { result_code::invalid_transition, server_scheduler::reason_code::reject_invalid_request };
    }
    return enqueue(it->second, at_us);
}

bool request_runtime::bind_slot(uint64_t request_id, slot_id slot, uint64_t at_us) {
    const auto it = records.find(request_id);
    if (it == records.end() || it->second.scheduler_queued) {
        return false;
    }
    const binding_result result =
        registry.bind(it->second.handle, slot, server_request_registry::reason_code::dispatched, at_us);
    if (!result) {
        return false;
    }
    it->second.bindings.push_back(result.lease);
    return true;
}

bool request_runtime::release_slot(uint64_t request_id, slot_id slot, uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    auto &     bindings = it->second.bindings;
    const auto lease    = std::find_if(bindings.begin(), bindings.end(),
                                       [slot](const binding_lease & value) { return value.slot == slot; });
    if (lease == bindings.end() || !registry.unbind(*lease, server_request_registry::reason_code::yielded, at_us)) {
        return false;
    }
    bindings.erase(lease);
    if (!bindings.empty()) {
        return true;
    }

    const auto snapshot = registry.get(it->second.handle);
    if (!snapshot) {
        return false;
    }
    if (snapshot->cancel_requested) {
        return finish(it, lifecycle::cancelled, server_request_registry::reason_code::client_cancel, at_us);
    }
    return finish(it, it->second.terminal,
                  it->second.terminal == lifecycle::failed ? server_request_registry::reason_code::request_failed :
                                                             server_request_registry::reason_code::completed,
                  at_us);
}

bool request_runtime::cancel(uint64_t request_id, uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    if (!registry.mark_cancel_requested(it->second.handle, server_request_registry::reason_code::client_cancel,
                                        at_us)) {
        return false;
    }
    if (it->second.scheduler_queued) {
        if (!scheduler.cancel(request_id).completed) {
            return false;
        }
        it->second.scheduler_queued = false;
    }
    if (it->second.bindings.empty()) {
        return finish(it, lifecycle::cancelled, server_request_registry::reason_code::client_cancel, at_us);
    }
    return true;
}

bool request_runtime::fail(uint64_t request_id, uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    it->second.terminal = lifecycle::failed;
    if (it->second.scheduler_queued) {
        if (!scheduler.cancel(request_id).completed) {
            return false;
        }
        it->second.scheduler_queued = false;
    }
    return !it->second.bindings.empty() ||
           finish(it, lifecycle::failed, server_request_registry::reason_code::request_failed, at_us);
}

bool request_runtime::finish(std::map<uint64_t, record>::iterator it,
                             lifecycle                            terminal,
                             server_request_registry::reason_code reason,
                             uint64_t                             at_us) {
    const auto handle = it->second.handle;
    if (!registry.mark_terminal(handle, terminal, reason, at_us) || !registry.remove_terminal(handle, reason, at_us)) {
        return false;
    }
    records.erase(it);
    return true;
}

bool request_runtime::contains(uint64_t request_id) const {
    return records.count(request_id) != 0;
}

size_t request_runtime::queued_total() const {
    return scheduler.queued_total();
}

std::vector<request_snapshot> request_runtime::snapshot() const {
    return registry.snapshot();
}

event_log_snapshot request_runtime::events() const {
    return registry.events();
}

registry_summary request_runtime::summary() const {
    return registry.summary();
}

}  // namespace server_request_runtime

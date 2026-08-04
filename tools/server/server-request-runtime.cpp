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

uint64_t deadline_from(uint64_t start_us, uint64_t timeout_us) {
    return timeout_us == 0 ? std::numeric_limits<uint64_t>::max() : saturating_add(start_us, timeout_us);
}

uint32_t bounded_retry_after(uint32_t seconds) {
    return std::min<uint32_t>(60, std::max<uint32_t>(1, seconds));
}

}  // namespace

bool is_overload(result_code code) {
    return code == result_code::queue_full || code == result_code::registry_capacity;
}

request_runtime::request_runtime(runtime_config config) :
    scheduler(std::move(config.scheduler)),
    registry(std::move(config.registry)),
    default_queue_timeout_us(config.default_queue_timeout_us == 0 ? runtime_config::queue_timeout_default_us :
                                                                    config.default_queue_timeout_us),
    default_run_timeout_us(config.default_run_timeout_us == 0 ? runtime_config::run_timeout_default_us :
                                                                config.default_run_timeout_us),
    overload_retry_after_seconds(bounded_retry_after(config.overload_retry_after_seconds)) {}

admission_result request_runtime::admit(const request_metadata & request, bool scheduled) {
    const bool duplicate = records.count(request.id) != 0;
    if (request.id == 0 || duplicate || to_registry_lane(request.lane) == trusted_lane::count) {
        return { duplicate ? result_code::duplicate_request : result_code::invalid_request,
                 duplicate ? server_scheduler::reason_code::reject_duplicate_id :
                             server_scheduler::reason_code::reject_invalid_request };
    }

    record next;
    next.metadata = request;
    next.queue_deadline_us = deadline_from(
        request.arrival_us, request.queue_timeout_us == 0 ? default_queue_timeout_us : request.queue_timeout_us);
    next.run_timeout_us = request.run_timeout_us == 0 ? default_run_timeout_us : request.run_timeout_us;

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
        const result_code code = registry_rejection(registered.code);
        return { code, server_scheduler::reason_code::reject_invalid_request,
                 is_overload(code) ? overload_retry_after_seconds : 0 };
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
        const result_code code = scheduler_rejection(admitted.reason);
        return { code, admitted.reason, is_overload(code) ? overload_retry_after_seconds : 0 };
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

expiration request_runtime::expire(std::map<uint64_t, record>::iterator it, deadline_kind kind, uint64_t at_us) {
    record &   request = it->second;
    expiration result  = { request.metadata.id, kind, !request.bindings.empty() };
    const auto reason  = kind == deadline_kind::queue ? server_request_registry::reason_code::queue_timeout :
                                                        server_request_registry::reason_code::run_timeout;
    if (!registry.mark_timeout_expired(request.handle, reason, at_us)) {
        return {};
    }
    request.terminal = lifecycle::timed_out;
    if (request.scheduler_queued) {
        if (!scheduler.cancel(request.metadata.id).completed) {
            return {};
        }
        request.scheduler_queued = false;
    }
    if (!request.bindings.empty()) {
        return result;
    }

    if (!finish(it, lifecycle::timed_out, reason, at_us)) {
        return {};
    }
    return result;
}

std::vector<expiration> request_runtime::expire_due(uint64_t now_us) {
    std::vector<expiration> result;
    auto                    it = records.begin();
    while (it != records.end()) {
        const record & request = it->second;
        if (request.terminal != lifecycle::completed) {
            ++it;
            continue;
        }

        const bool     run_started = request.run_deadline_us != 0;
        const uint64_t deadline    = run_started ? request.run_deadline_us : request.queue_deadline_us;
        if (now_us < deadline) {
            ++it;
            continue;
        }

        const auto current = it++;
        expiration expired = expire(current, run_started ? deadline_kind::run : deadline_kind::queue, now_us);
        if (expired.request_id != 0) {
            result.push_back(expired);
        }
    }
    return result;
}

bool request_runtime::mark_deferred(uint64_t request_id, uint64_t at_us) {
    const auto it = records.find(request_id);
    return it != records.end() && it->second.terminal == lifecycle::completed && at_us < it->second.queue_deadline_us &&
           !it->second.scheduler_queued && it->second.bindings.empty() &&
           registry.set_queue_state(it->second.handle, queue_state::blocked,
                                    server_request_registry::reason_code::capacity_blocked, at_us);
}

admission_result request_runtime::resume(uint64_t request_id, uint64_t at_us) {
    const auto it = records.find(request_id);
    if (it == records.end()) {
        return { result_code::unknown_request, server_scheduler::reason_code::reject_invalid_request };
    }
    if (it->second.terminal != lifecycle::completed || at_us >= it->second.queue_deadline_us) {
        return { result_code::deadline_expired, server_scheduler::reason_code::reject_invalid_request };
    }
    if (it->second.scheduler_queued || !it->second.bindings.empty()) {
        return { result_code::invalid_transition, server_scheduler::reason_code::reject_invalid_request };
    }
    return enqueue(it->second, at_us);
}

bool request_runtime::bind_slot(uint64_t request_id, slot_id slot, uint64_t at_us) {
    const auto it = records.find(request_id);
    if (it == records.end() || it->second.terminal != lifecycle::completed || it->second.scheduler_queued ||
        (it->second.run_deadline_us == 0 && at_us >= it->second.queue_deadline_us)) {
        return false;
    }
    const binding_result result =
        registry.bind(it->second.handle, slot, server_request_registry::reason_code::dispatched, at_us);
    if (!result) {
        return false;
    }
    it->second.bindings.push_back(result.lease);
    if (it->second.run_deadline_us == 0) {
        it->second.run_deadline_us = deadline_from(at_us, it->second.run_timeout_us);
    }
    return true;
}

bool request_runtime::can_publish(uint64_t request_id, slot_id slot) const {
    const auto it = records.find(request_id);
    if (it == records.end() || it->second.terminal != lifecycle::completed) {
        return false;
    }
    const auto & bindings = it->second.bindings;
    return std::any_of(bindings.begin(), bindings.end(),
                       [slot](const binding_lease & value) { return value.slot == slot; });
}

bool request_runtime::release_slot(uint64_t request_id, slot_id slot, uint64_t at_us) {
    expire_due(at_us);
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    auto &     request  = it->second;
    auto &     bindings = request.bindings;
    const auto lease    = std::find_if(bindings.begin(), bindings.end(),
                                       [slot](const binding_lease & value) { return value.slot == slot; });
    const auto unbind_reason =
        request.terminal == lifecycle::timed_out ? server_request_registry::reason_code::run_timeout :
        request.terminal == lifecycle::cancelled ? server_request_registry::reason_code::client_cancel :
                                                   server_request_registry::reason_code::yielded;
    if (lease == bindings.end() || !registry.unbind(*lease, unbind_reason, at_us)) {
        return false;
    }
    bindings.erase(lease);
    if (!bindings.empty()) {
        return true;
    }

    const auto terminal_reason =
        request.terminal == lifecycle::cancelled ? server_request_registry::reason_code::client_cancel :
        request.terminal == lifecycle::timed_out ? unbind_reason :
        request.terminal == lifecycle::failed    ? server_request_registry::reason_code::request_failed :
                                                   server_request_registry::reason_code::completed;
    return finish(it, request.terminal, terminal_reason, at_us);
}

bool request_runtime::cancel(uint64_t request_id, uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    if (it->second.terminal != lifecycle::completed) {
        return true;
    }
    if (!registry.mark_cancel_requested(it->second.handle, server_request_registry::reason_code::client_cancel,
                                        at_us)) {
        return false;
    }
    it->second.terminal = lifecycle::cancelled;
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
    if (it->second.terminal != lifecycle::completed) {
        return true;
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

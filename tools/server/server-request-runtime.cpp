#include "server-request-runtime.h"

#include "server-dashboard-bus.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace server_request_runtime {

namespace {

using namespace server_request_registry;
using namespace server_scheduler;

constexpr size_t lane_index(lane value) {
    return static_cast<size_t>(value);
}

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

uint64_t proportional_cached_prefix_us(const request_metadata & request) {
    const uint64_t prompt_tokens = request.counts.prompt_tokens;
    if (prompt_tokens == 0) {
        return 0;
    }
    const uint64_t cached_tokens = std::min(request.counts.cached_prompt_tokens, prompt_tokens);
    // The ratio is bounded by predicted_prefill_us because cached_tokens <= prompt_tokens.
    // Widening only the product preserves the exact floor without overflow.
    using wide_uint              = __uint128_t;
    return static_cast<uint64_t>(static_cast<wide_uint>(request.estimates.predicted_prefill_us) * cached_tokens /
                                 prompt_tokens);
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
    default_run_stall_timeout_us(config.default_run_stall_timeout_us == 0 ?
                                         runtime_config::run_stall_timeout_default_us :
                                         config.default_run_stall_timeout_us),
    overload_retry_after_seconds(bounded_retry_after(config.overload_retry_after_seconds)) {
    const bool valid_fast_refill =
        config.fast_refill_max_members_per_epoch >= runtime_config::fast_refill_min_members &&
        config.fast_refill_max_members_per_epoch <= runtime_config::fast_refill_max_members &&
        config.fast_refill_window_us >= runtime_config::fast_refill_min_window_us &&
        config.fast_refill_window_us <= runtime_config::fast_refill_max_window_us;
    if (valid_fast_refill) {
        fast_refill_max_members_per_epoch = config.fast_refill_max_members_per_epoch;
        fast_refill_window_us              = config.fast_refill_window_us;
    }
}

bool request_runtime::fast_refill_enabled() const {
    return fast_refill_max_members_per_epoch != 0;
}

bool request_runtime::fast_claim_within_epoch(size_t members, uint64_t now_us) const {
    if (!fast_refill_enabled() || members == 0) {
        return true;
    }
    const size_t prior_members = cohort.active ? cohort.fast_members : 0;
    return prior_members < fast_refill_max_members_per_epoch &&
           members <= fast_refill_max_members_per_epoch - prior_members &&
           (prior_members == 0 || now_us < cohort.fast_refill_deadline_us);
}

admission_result request_runtime::validate_dispatch_family(
        lane priority,
        const std::array<size_t, lane_count> & demand,
        size_t physical_slot_capacity) const {
    if (priority == lane::count) {
        return { result_code::invalid_request, server_scheduler::reason_code::reject_invalid_request };
    }

    size_t total = 0;
    for (size_t i = 0; i < lane_count; ++i) {
        if (fast_refill_enabled() && i != lane_index(priority) && demand[i] != 0) {
            return { result_code::invalid_request, server_scheduler::reason_code::reject_invalid_request };
        }
        const size_t cap = scheduler.config().lanes[i].decode_cap;
        if (demand[i] > cap) {
            return { result_code::capacity_impossible,
                     server_scheduler::reason_code::reject_capacity_impossible };
        }
        total += demand[i];
    }
    const size_t cohort_cap = scheduler.config().lanes[lane_index(priority)].decode_cap;
    if (total == 0 || total > cohort_cap || total > physical_slot_capacity) {
        return { result_code::capacity_impossible, server_scheduler::reason_code::reject_capacity_impossible };
    }
    if (priority == lane::fast && fast_refill_max_members_per_epoch != 0 &&
        total > fast_refill_max_members_per_epoch) {
        return { result_code::capacity_impossible, server_scheduler::reason_code::reject_capacity_impossible };
    }
    if (priority == lane::low && scheduler.choose_decode_width(lane::low, total, total, false).width < total) {
        return { result_code::capacity_impossible, server_scheduler::reason_code::reject_capacity_impossible };
    }
    return { result_code::ok, server_scheduler::reason_code::admission_ready };
}

admission_result request_runtime::admit(const request_metadata & request, bool scheduled) {
    const bool duplicate = records.count(request.id) != 0;
    const auto parent = request.parent_id == 0 ? records.end() : records.find(request.parent_id);
    const bool invalid_parent = fast_refill_enabled() && request.parent_id != 0 &&
        (parent == records.end() || parent->second.metadata.parent_id != 0 ||
         parent->second.metadata.lane != request.lane || parent->second.terminal != lifecycle::completed ||
         !parent->second.scheduler_queued || parent->second.permit != record::permit_state::none ||
         !parent->second.bindings.empty());
    if (request.id == 0 || duplicate || invalid_parent || to_registry_lane(request.lane) == trusted_lane::count) {
        return { duplicate ? result_code::duplicate_request : result_code::invalid_request,
                 duplicate ? server_scheduler::reason_code::reject_duplicate_id :
                             server_scheduler::reason_code::reject_invalid_request };
    }

    record next;
    next.metadata = request;
    next.queue_deadline_us = deadline_from(
        request.arrival_us, request.queue_timeout_us == 0 ? default_queue_timeout_us : request.queue_timeout_us);
    next.run_timeout_us = request.run_timeout_us == 0 ? default_run_timeout_us : request.run_timeout_us;
    next.run_stall_timeout_us =
        request.run_stall_timeout_us == 0 ? default_run_stall_timeout_us : request.run_stall_timeout_us;

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

dispatch_result request_runtime::take_next(uint64_t now_us, size_t physical_slot_capacity) {
    struct permit_demand {
        std::array<size_t, lane_count> lanes = {};
        size_t                         total = 0;
    };

    std::map<uint64_t, permit_demand> demand_by_parent;
    for (const auto & entry : records) {
        const record & current = entry.second;
        if (current.terminal != lifecycle::completed || current.permit != record::permit_state::none) {
            continue;
        }
        const uint64_t  owner  = current.metadata.parent_id == 0 ? current.metadata.id : current.metadata.parent_id;
        permit_demand & demand = demand_by_parent[owner];
        ++demand.lanes[lane_index(current.metadata.lane)];
        ++demand.total;
    }

    const auto demand_for = [&demand_by_parent](uint64_t request_id) {
        const auto found = demand_by_parent.find(request_id);
        return found == demand_by_parent.end() ? permit_demand{} : found->second;
    };

    permit_demand queued_demand;
    for (const auto & entry : records) {
        if (!entry.second.scheduler_queued) {
            continue;
        }
        const permit_demand current = demand_for(entry.first);
        for (size_t i = 0; i < lane_count; ++i) {
            queued_demand.lanes[i] += current.lanes[i];
        }
    }

    if (permit_counts.total == 0) {
        cohort = {};
    } else if (!cohort.active) {
        throw std::logic_error("live permits exist without a cohort epoch");
    }

    const auto cohort_limit_for = [this, physical_slot_capacity, &queued_demand](lane priority) {
        size_t limit = scheduler.config().lanes[lane_index(priority)].decode_cap;
        if (priority == lane::low) {
            const size_t runnable_low =
                std::min(permit_counts.claimed[lane_index(lane::low)] + queued_demand.lanes[lane_index(lane::low)],
                         physical_slot_capacity);
            limit = scheduler.choose_decode_width(lane::low, runnable_low, runnable_low, false).width;
        }
        return std::min(limit, physical_slot_capacity);
    };

    const bool has_cohort = cohort.active;
    lane       eligible_lane = cohort.dominant;
    size_t     eligible_limit = cohort.limit;
    bool       upgrading         = false;
    bool       allow_new_members = !has_cohort || cohort.open;
    if (has_cohort) {
        const bool permit_reentry = std::any_of(records.begin(), records.end(), [](const auto & entry) {
            return entry.second.scheduler_queued && entry.second.permit != record::permit_state::none;
        });
        lane higher = lane::count;
        if (cohort.dominant != lane::fast && scheduler.queued(lane::fast) != 0) {
            higher = lane::fast;
        } else if (cohort.dominant == lane::low && scheduler.queued(lane::normal) != 0) {
            higher = lane::normal;
        }

        if (higher != lane::count) {
            const size_t higher_limit = cohort_limit_for(higher);
            if (permit_counts.total >= higher_limit) {
                allow_new_members = false;
                cohort.open      = false;
                if (!permit_reentry) {
                    return {};
                }
            } else {
                eligible_lane     = higher;
                eligible_limit    = higher_limit;
                upgrading         = true;
                allow_new_members = true;
            }
        } else if (!allow_new_members && !permit_reentry) {
            const bool can_refill_fast = fast_refill_enabled() && cohort.dominant == lane::fast &&
                                         scheduler.queued(lane::fast) != 0 && permit_counts.total < cohort.limit &&
                                         fast_claim_within_epoch(1, now_us);
            // Elastic same-lane refill: a normal/low cohort that closed while
            // full reopens to its own lane as soon as a permit drains below
            // the cohort limit. Without this, a request that arrived while
            // all slots were busy waited for the ENTIRE cohort to drain
            // (permit_counts.total == 0 resets the epoch) instead of joining
            // when one slot freed - serializing turnover under sustained
            // multi-agent load. The fast lane keeps its bounded epoch refill
            // above, and cross-lane upgrades keep the stricter width rule in
            // the `higher` branch.
            const bool can_refill_elastic = cohort.dominant != lane::fast &&
                                            scheduler.queued(cohort.dominant) != 0 &&
                                            permit_counts.total < cohort.limit;
            if (!can_refill_fast && !can_refill_elastic) {
                return {};
            }
            allow_new_members = true;
        }
    }

    const auto evaluate = [this, now_us, physical_slot_capacity, has_cohort, eligible_lane, eligible_limit,
                           allow_new_members, &cohort_limit_for, &demand_for](const server_scheduler::request & request) {
        candidate_evaluation result;
        const auto           it = records.find(request.id);
        if (it != records.end()) {
            const bool owns_permit = it->second.permit != record::permit_state::none;
            if (has_cohort && !owns_permit && (!allow_new_members || request.priority != eligible_lane)) {
                result.state = feasibility::temporarily_blocked;
                return result;
            }
            result.predicted_gpu_us = std::max<uint64_t>(1, it->second.metadata.estimates.predicted_gpu_us);
            result.cached_prefix_us = proportional_cached_prefix_us(it->second.metadata);
            if (it->second.metadata.counts.cached_prompt_tokens != 0) {
                result.restore_cost_us  = it->second.metadata.estimates.predicted_cache_restore_us;
            }

            const permit_demand demand      = demand_for(request.id);
            const size_t        fast_demand = demand.lanes[lane_index(lane::fast)];
            bool mixed_lane_family = false;
            for (size_t i = 0; i < lane_count; ++i) {
                mixed_lane_family = mixed_lane_family ||
                    (i != lane_index(request.priority) && demand.lanes[i] != 0);
            }
            if (fast_refill_enabled() && mixed_lane_family) {
                result.state = feasibility::impossible;
                return result;
            }
            if (!owns_permit && !fast_claim_within_epoch(fast_demand, now_us)) {
                result.state = feasibility::temporarily_blocked;
                return result;
            }
            if (demand.total > physical_slot_capacity || permit_counts.total > physical_slot_capacity - demand.total) {
                result.state = feasibility::temporarily_blocked;
                return result;
            }
            const size_t cohort_limit = has_cohort ? eligible_limit : cohort_limit_for(request.priority);
            if (demand.total > cohort_limit || permit_counts.total > cohort_limit - demand.total) {
                result.state = feasibility::temporarily_blocked;
                return result;
            }
            for (size_t i = 0; i < lane_count; ++i) {
                const size_t cap = scheduler.config().lanes[i].decode_cap;
                if (demand.lanes[i] > cap || permit_counts.claimed[i] > cap - demand.lanes[i]) {
                    result.state = feasibility::temporarily_blocked;
                    return result;
                }
            }
        }
        return result;
    };

    const service_decision decision = scheduler.select_next(now_us, evaluate);
    if (!decision.selected) {
        if (has_cohort) {
            cohort.open = false;
        }
        return {};
    }

    auto it = records.find(decision.request_id);
    if (it == records.end() || !it->second.scheduler_queued) {
        if (!scheduler.complete_service(decision.decision_id, 0, service_disposition::cancelled).completed) {
            throw std::logic_error("failed to cancel inconsistent scheduler selection");
        }
        return {};
    }

    record &   request = it->second;
    const auto before  = registry.get(request.handle);
    if (!before) {
        if (!scheduler.complete_service(decision.decision_id, 0, service_disposition::cancelled).completed) {
            throw std::logic_error("failed to cancel selection with missing registry state");
        }
        return {};
    }
    const bool     claims_new_permit = request.permit == record::permit_state::none;
    const uint64_t charged_us        = std::max<uint64_t>(1, decision.predicted_gpu_us);
    if (!scheduler.complete_service(decision.decision_id, charged_us, service_disposition::complete).completed) {
        throw std::logic_error("failed to commit scheduler selection");
    }
    request.scheduler_queued = false;

    const permit_demand selected_demand = demand_for(decision.request_id);
    if (selected_demand.total > physical_slot_capacity ||
        permit_counts.total > physical_slot_capacity - selected_demand.total) {
        throw std::logic_error("scheduler selected work without physical permit capacity");
    }
    const size_t selected_cohort_limit = has_cohort ? eligible_limit : cohort_limit_for(decision.selected_lane);
    if (selected_demand.total > selected_cohort_limit ||
        permit_counts.total > selected_cohort_limit - selected_demand.total) {
        throw std::logic_error("scheduler selected work beyond dominant lane cohort limit");
    }
    if (!has_cohort) {
        cohort = { true, true, decision.selected_lane, selected_cohort_limit, 0, 0 };
    } else if (upgrading && claims_new_permit) {
        cohort.dominant = decision.selected_lane;
        cohort.limit    = selected_cohort_limit;
        cohort.open     = true;
    }
    const size_t selected_fast_members = selected_demand.lanes[lane_index(lane::fast)];
    if (fast_refill_enabled() && claims_new_permit && selected_fast_members != 0) {
        if (decision.selected_lane != lane::fast) {
            throw std::logic_error("scheduler selected a mixed-lane dispatch family");
        }
        if (!fast_claim_within_epoch(selected_fast_members, now_us)) {
            throw std::logic_error("scheduler selected fast refill beyond epoch member limit");
        }
        if (cohort.fast_members == 0) {
            cohort.fast_refill_deadline_us = saturating_add(now_us, fast_refill_window_us);
        }
        cohort.fast_members += selected_fast_members;
    }
    for (auto & entry : records) {
        record & current = entry.second;
        if ((current.metadata.id == decision.request_id || current.metadata.parent_id == decision.request_id) &&
            current.terminal == lifecycle::completed && current.permit == record::permit_state::none) {
            const size_t index = lane_index(current.metadata.lane);
            if (permit_counts.claimed[index] >= scheduler.config().lanes[index].decode_cap) {
                throw std::logic_error("scheduler selected work without lane permit capacity");
            }
            current.permit = record::permit_state::selected_unbound;
            ++permit_counts.claimed[index];
            ++permit_counts.total;
        }
    }
    if (permit_counts.total >= cohort.limit) {
        cohort.open = false;
    }
    request.metadata.virtual_runtime_us = saturating_add(request.metadata.virtual_runtime_us, charged_us);
    request_progress progress;
    progress.virtual_runtime_us     = request.metadata.virtual_runtime_us;
    progress.debt_us                = request.metadata.debt_us;
    progress.observed_output_tokens = before->counts.observed_output_tokens;
    progress.estimates              = request.metadata.estimates;
    if (!registry.update_progress(request.handle, progress, server_request_registry::reason_code::progress_observed,
                                  now_us) ||
        !registry.set_queue_state(request.handle, queue_state::none,
                                  server_request_registry::reason_code::dispatched, now_us)) {
        throw std::logic_error("failed to publish committed dispatch state");
    }

    return { true, decision.request_id, decision.selected_lane, decision.reason, decision.lane_reason };
}

expiration request_runtime::expire(std::map<uint64_t, record>::iterator it, deadline_kind kind, uint64_t at_us) {
    record &   request = it->second;
    expiration result  = { request.metadata.id, kind, !request.bindings.empty() };
    if (kind == deadline_kind::run_stall && at_us > request.last_progress_us) {
        result.stalled_for_us = at_us - request.last_progress_us;
    }
    const auto reason = kind == deadline_kind::queue     ? server_request_registry::reason_code::queue_timeout :
                        kind == deadline_kind::run_stall ? server_request_registry::reason_code::run_stall :
                                                           server_request_registry::reason_code::run_timeout;
    if (!registry.mark_timeout_expired(request.handle, reason, at_us)) {
        return {};
    }
    request.timeout_reason = reason;
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

// Single source of truth for which deadline governs a live request. Queued
// work keeps the arrival-anchored queue deadline. Running work (run phase
// started at first bind) expires on STALL - the gap since the slot last
// advanced - with the absolute run cap as the outer backstop. When both run
// bounds are due at once the cap wins: it is the harder guarantee and its
// message names the policy that ultimately bounded the request.
expiration request_runtime::planned_expiration_of(const record & request, uint64_t at_us) const {
    const bool run_started = request.run_deadline_us != 0;
    if (!run_started) {
        if (at_us < request.queue_deadline_us) {
            return {};
        }
        return { request.metadata.id, deadline_kind::queue, !request.bindings.empty() };
    }
    if (at_us >= request.run_deadline_us) {
        return { request.metadata.id, deadline_kind::run, !request.bindings.empty() };
    }
    const uint64_t stall_deadline_us = deadline_from(request.last_progress_us, request.run_stall_timeout_us);
    if (at_us < stall_deadline_us) {
        return {};
    }
    return { request.metadata.id, deadline_kind::run_stall, !request.bindings.empty(),
             at_us - request.last_progress_us };
}

expiration request_runtime::expiration_due(uint64_t request_id, uint64_t at_us) const {
    const auto it = records.find(request_id);
    if (it == records.end() || it->second.terminal != lifecycle::completed) {
        return {};
    }
    return planned_expiration_of(it->second, at_us);
}

std::vector<expiration> request_runtime::expirations_due(uint64_t at_us) const {
    std::vector<expiration> result;
    result.reserve(records.size());
    for (const auto & entry : records) {
        const expiration planned = expiration_due(entry.first, at_us);
        if (planned.request_id != 0) {
            result.push_back(planned);
        }
    }
    return result;
}

expiration request_runtime::expire_prepared(const expiration & planned, uint64_t at_us) {
    if (planned.request_id == 0) {
        return {};
    }
    const auto it = records.find(planned.request_id);
    const expiration current = expiration_due(planned.request_id, at_us);
    if (it == records.end() || current.request_id == 0 || current.kind != planned.kind ||
        current.was_running != planned.was_running) {
        return {};
    }
    return expire(it, planned.kind, at_us);
}

std::vector<expiration> request_runtime::expire_due(uint64_t now_us) {
    auto planned = expirations_due(now_us);
    size_t committed = 0;
    for (const expiration & current : planned) {
        const expiration expired = expire_prepared(current, now_us);
        if (expired.request_id != 0) {
            planned[committed++] = expired;
        }
    }
    planned.resize(committed);
    return planned;
}

expiration request_runtime::expire_if_due(std::map<uint64_t, record>::iterator it, uint64_t at_us) {
    const record & request = it->second;
    if (request.terminal != lifecycle::completed) {
        return {};
    }
    const expiration planned = planned_expiration_of(request, at_us);
    if (planned.request_id == 0) {
        return {};
    }
    return expire(it, planned.kind, at_us);
}

bool request_runtime::mark_deferred(
        uint64_t request_id,
        uint64_t at_us,
        server_request_registry::reason_code reason) {
    const auto it = records.find(request_id);
    const bool valid_reason = reason == server_request_registry::reason_code::capacity_blocked ||
                              reason == server_request_registry::reason_code::admission_wait;
    return valid_reason && it != records.end() && it->second.terminal == lifecycle::completed &&
           at_us < it->second.queue_deadline_us &&
           !it->second.scheduler_queued && it->second.bindings.empty() &&
           it->second.permit == record::permit_state::selected_unbound &&
           registry.set_queue_state(it->second.handle, queue_state::blocked, reason, at_us);
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
        it->second.permit != record::permit_state::selected_unbound ||
        (it->second.run_deadline_us == 0 && at_us >= it->second.queue_deadline_us)) {
        return false;
    }
    const binding_result result =
        registry.bind(it->second.handle, slot, server_request_registry::reason_code::dispatched, at_us);
    if (!result) {
        return false;
    }
    it->second.bindings.push_back(result.lease);
    it->second.permit = record::permit_state::bound;
    ++permit_counts.bound[lane_index(it->second.metadata.lane)];
    if (it->second.run_deadline_us == 0) {
        it->second.run_deadline_us  = deadline_from(at_us, it->second.run_timeout_us);
        it->second.last_progress_us = at_us;
    }
    return true;
}

bool request_runtime::note_progress(uint64_t request_id, uint64_t at_us) {
    const auto it = records.find(request_id);
    if (it == records.end() || it->second.terminal != lifecycle::completed ||
        it->second.run_deadline_us == 0) {
        return false;
    }
    if (at_us > it->second.last_progress_us) {
        it->second.last_progress_us = at_us;
    }
    return true;
}

publication_result request_runtime::gate_result_publication(uint64_t request_id,
                                                            slot_id  slot,
                                                            bool     final,
                                                            uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end() || it->second.terminal != lifecycle::completed) {
        return {};
    }

    const expiration expired = expire_if_due(it, at_us);
    if (expired.request_id != 0) {
        return { false, expired };
    }

    const record & request = it->second;
    if (!std::any_of(request.bindings.begin(), request.bindings.end(),
                     [slot](const binding_lease & value) { return value.slot == slot; })) {
        return {};
    }
    return { !final || release_bound_slot(it, slot, at_us), {} };
}

release_result request_runtime::release_slot(uint64_t request_id, slot_id slot, uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return {};
    }
    const expiration expired = expire_if_due(it, at_us);
    it = records.find(request_id);
    return { it != records.end() && release_bound_slot(it, slot, at_us), expired };
}

bool request_runtime::release_bound_slot(std::map<uint64_t, record>::iterator it, slot_id slot, uint64_t at_us) {
    auto &     request  = it->second;
    auto &     bindings = request.bindings;
    const auto lease    = std::find_if(bindings.begin(), bindings.end(),
                                       [slot](const binding_lease & value) { return value.slot == slot; });
    const auto unbind_reason =
        request.terminal == lifecycle::timed_out ?
                (request.timeout_reason != server_request_registry::reason_code::none ?
                         request.timeout_reason :
                         server_request_registry::reason_code::run_timeout) :
        request.terminal == lifecycle::cancelled ? request.cancel_reason :
                                                   server_request_registry::reason_code::yielded;
    if (lease == bindings.end() || !registry.unbind(*lease, unbind_reason, at_us)) {
        return false;
    }
    bindings.erase(lease);
    if (!bindings.empty()) {
        return true;
    }

    release_permit(request);

    const auto terminal_reason =
        request.terminal == lifecycle::cancelled ? request.cancel_reason :
        request.terminal == lifecycle::timed_out ? unbind_reason :
        request.terminal == lifecycle::failed    ? server_request_registry::reason_code::request_failed :
                                                   server_request_registry::reason_code::completed;
    return finish(it, request.terminal, terminal_reason, at_us);
}

bool request_runtime::cancel(uint64_t request_id, uint64_t at_us) {
    return cancel(request_id, server_request_registry::reason_code::client_cancel, at_us);
}

bool request_runtime::cancel(
        uint64_t request_id,
        server_request_registry::reason_code reason,
        uint64_t at_us) {
    if (reason != server_request_registry::reason_code::client_cancel &&
        reason != server_request_registry::reason_code::server_cancel) {
        return false;
    }
    const std::vector<uint64_t> targets = terminal_targets(request_id);
    if (targets.empty()) {
        return false;
    }
    bool cancelled = true;
    for (uint64_t target : targets) {
        cancelled = cancel_one(target, reason, at_us) && cancelled;
    }
    return cancelled;
}

bool request_runtime::cancel_one(
        uint64_t request_id,
        server_request_registry::reason_code reason,
        uint64_t at_us) {
    auto it = records.find(request_id);
    if (it == records.end()) {
        return false;
    }
    if (it->second.terminal != lifecycle::completed) {
        return true;
    }
    if (!registry.mark_cancel_requested(it->second.handle, reason, at_us)) {
        return false;
    }
    it->second.terminal = lifecycle::cancelled;
    it->second.cancel_reason = reason;
    if (it->second.scheduler_queued) {
        if (!scheduler.cancel(request_id).completed) {
            return false;
        }
        it->second.scheduler_queued = false;
    }
    if (it->second.bindings.empty()) {
        return finish(it, lifecycle::cancelled, reason, at_us);
    }
    return true;
}

bool request_runtime::fail(uint64_t request_id, uint64_t at_us) {
    const std::vector<uint64_t> targets = terminal_targets(request_id);
    if (targets.empty()) {
        return false;
    }
    bool failed = true;
    for (uint64_t target : targets) {
        failed = fail_one(target, at_us) && failed;
    }
    return failed;
}

failure_publication_result request_runtime::failure_publication_status(uint64_t request_id) const {
    const auto requested = records.find(request_id);
    if (requested == records.end()) {
        return { failure_publication_code::unknown_request };
    }
    if (requested->second.terminal != lifecycle::completed) {
        return { failure_publication_code::already_terminal };
    }
    return { failure_publication_code::first_terminal };
}

failure_publication_result request_runtime::fail_for_publication(uint64_t request_id, uint64_t at_us) {
    const auto status = failure_publication_status(request_id);
    if (!status) {
        return status;
    }
    return { fail(request_id, at_us) ? failure_publication_code::first_terminal :
                                       failure_publication_code::transition_failed };
}

failure_publication_result request_runtime::fail_one_for_publication(uint64_t request_id, uint64_t at_us) {
    const auto status = failure_publication_status(request_id);
    if (!status) {
        return status;
    }
    return { fail_one(request_id, at_us) ? failure_publication_code::first_terminal :
                                          failure_publication_code::transition_failed };
}

bool request_runtime::fail_one(uint64_t request_id, uint64_t at_us) {
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

std::vector<uint64_t> request_runtime::terminal_targets(uint64_t request_id) const {
    const auto requested = records.find(request_id);
    if (requested == records.end()) {
        return {};
    }

    // A child remains independently cancellable for per-result cancellation.
    // Parent cancellation or setup failure retires its passive children in
    // the same serialized operation.
    if (requested->second.metadata.parent_id != 0) {
        return { request_id };
    }

    std::vector<uint64_t> result;
    for (const auto & entry : records) {
        if (entry.first == request_id || entry.second.metadata.parent_id == request_id) {
            result.push_back(entry.first);
        }
    }
    return result;
}

bool request_runtime::finish(std::map<uint64_t, record>::iterator it,
                             lifecycle                            terminal,
                             server_request_registry::reason_code reason,
                             uint64_t                             at_us) {
    const auto handle = it->second.handle;
    if (!registry.mark_terminal(handle, terminal, reason, at_us) || !registry.remove_terminal(handle, reason, at_us)) {
        return false;
    }
    // m2-dashboard: finish() is the single terminal choke point (completion,
    // cancel, timeout/stall, failure all land here), so one emit covers every
    // terminal transition with the exact registry reason
    {
        server_dashboard::event ev;
        ev.kind       = server_dashboard::event_kind::terminal;
        ev.at_us      = at_us;
        ev.request_id = it->second.metadata.id;
        ev.lane       = static_cast<uint8_t>(it->second.metadata.lane);
        ev.code       = static_cast<uint16_t>(reason);
        ev.a          = static_cast<int64_t>(terminal);
        server_dashboard::instance().emit(ev);
    }
    // v3: close out any live /m2-dashboard/watch stream on this request, and
    // let its retained content become evictable. A cancelled or stalled request
    // never reaches send_final_response, so this is the only place that does
    // either for those paths.
    if (server_dashboard::watching()) {
        server_dashboard::watches().finish(it->second.metadata.id, 0);
    }
    if (server_dashboard::content().enabled()) {
        server_dashboard::content().mark_final(it->second.metadata.id);
    }
    release_permit(it->second);
    records.erase(it);
    return true;
}

void request_runtime::release_permit(record & request) {
    if (request.permit == record::permit_state::none) {
        return;
    }
    const size_t index = lane_index(request.metadata.lane);
    if (permit_counts.claimed[index] == 0 || permit_counts.total == 0) {
        throw std::logic_error("dispatch permit accounting underflow");
    }
    --permit_counts.claimed[index];
    --permit_counts.total;
    if (request.permit == record::permit_state::bound) {
        if (permit_counts.bound[index] == 0) {
            throw std::logic_error("bound dispatch permit accounting underflow");
        }
        --permit_counts.bound[index];
    }
    request.permit = record::permit_state::none;
    if (permit_counts.total == 0) {
        cohort = {};
    }
}

bool request_runtime::contains(uint64_t request_id) const {
    return records.count(request_id) != 0;
}

bool request_runtime::is_family_member(uint64_t request_id, uint64_t family_id) const {
    const auto it = records.find(request_id);
    return it != records.end() && (it->first == family_id || it->second.metadata.parent_id == family_id);
}

bool request_runtime::family_has_bindings(uint64_t request_id) const {
    return std::any_of(records.begin(), records.end(), [request_id](const auto & entry) {
        return (entry.first == request_id || entry.second.metadata.parent_id == request_id) &&
               !entry.second.bindings.empty();
    });
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

dispatch_permit_snapshot request_runtime::permits() const {
    return permit_counts;
}

fast_refill_status fast_refill_snapshot::status_at(uint64_t now_us, size_t permit_width) const {
    fast_refill_status result;
    result.fast_members_remaining = enabled && fast_members_used < max_members_per_cohort
        ? max_members_per_cohort - fast_members_used
        : 0;
    result.remaining_us = deadline_us != 0 && now_us < deadline_us ? deadline_us - now_us : 0;
    result.window_open = enabled && cohort_active && dominant == lane::fast && fast_members_used != 0 &&
                         result.fast_members_remaining != 0 && deadline_us != 0 && now_us < deadline_us;
    result.one_member_eligible_now = result.window_open && permit_width < cohort_limit;
    return result;
}

fast_refill_snapshot request_runtime::fast_refill() const {
    fast_refill_snapshot result;
    result.enabled                = fast_refill_enabled();
    result.max_members_per_cohort = fast_refill_max_members_per_epoch;
    result.window_us              = fast_refill_window_us;
    result.cohort_active          = cohort.active;
    result.selection_open         = cohort.active && cohort.open;
    result.dominant               = cohort.active ? cohort.dominant : lane::count;
    result.cohort_limit           = cohort.active ? cohort.limit : 0;
    result.fast_members_used      = cohort.active ? cohort.fast_members : 0;
    result.deadline_us = cohort.active && cohort.fast_members != 0 ? cohort.fast_refill_deadline_us : 0;
    return result;
}

}  // namespace server_request_runtime

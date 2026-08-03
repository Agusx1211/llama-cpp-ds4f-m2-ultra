#include "server-scheduler.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace server_scheduler {

namespace {

constexpr size_t lane_index(lane value) {
    return static_cast<size_t>(value);
}

bool valid_lane(lane value) {
    return lane_index(value) < lane_count;
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

uint64_t saturating_multiply(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

int64_t saturating_add_signed(int64_t lhs, int64_t rhs) {
    if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return std::numeric_limits<int64_t>::max();
    }
    if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) {
        return std::numeric_limits<int64_t>::min();
    }
    return lhs + rhs;
}

int64_t positive_product_as_signed(uint64_t lhs, uint64_t rhs) {
    const uint64_t product = saturating_multiply(lhs, rhs);
    return product > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? std::numeric_limits<int64_t>::max() :
                                                                                  static_cast<int64_t>(product);
}

bool vector_fits(const resource_vector & resident,
                 const resource_vector & demand,
                 const resource_vector & capacity,
                 const resource_vector & safety) {
    for (size_t i = 0; i < resource_vector::max_dimensions; ++i) {
        if (safety.units[i] > capacity.units[i]) {
            return false;
        }
        const uint64_t usable = capacity.units[i] - safety.units[i];
        if (resident.units[i] > usable || demand.units[i] > usable - resident.units[i]) {
            return false;
        }
    }
    return true;
}

bool vector_can_ever_fit(const resource_vector & demand,
                         const resource_vector & capacity,
                         const resource_vector & safety,
                         uint32_t &              limiting_dimension) {
    for (size_t i = 0; i < resource_vector::max_dimensions; ++i) {
        if (safety.units[i] > capacity.units[i] || demand.units[i] > capacity.units[i] - safety.units[i]) {
            limiting_dimension = static_cast<uint32_t>(i);
            return false;
        }
    }
    return true;
}

void vector_add(resource_vector & target, const resource_vector & value) {
    for (size_t i = 0; i < resource_vector::max_dimensions; ++i) {
        target.units[i] = saturating_add(target.units[i], value.units[i]);
    }
}

void vector_subtract(resource_vector & target, const resource_vector & value) {
    for (size_t i = 0; i < resource_vector::max_dimensions; ++i) {
        target.units[i] = value.units[i] > target.units[i] ? 0 : target.units[i] - value.units[i];
    }
}

}  // namespace

const std::array<lane_descriptor, lane_count> & default_lane_descriptors() {
    // Array order is deliberately low-to-fast while tie breaks use weight.
    // This prevents an enum ordinal from accidentally becoming a priority.
    static const std::array<lane_descriptor, lane_count> descriptors = {
        {
         { lane::low, "low", 1, 64, 256 },
         { lane::normal, "normal", 4, 8, 64 },
         { lane::fast, "fast", 16, 2, 16 },
         }
    };
    return descriptors;
}

const char * to_string(lane value) {
    switch (value) {
        case lane::low:
            return "low";
        case lane::normal:
            return "normal";
        case lane::fast:
            return "fast";
        case lane::count:
            return "count";
    }
    return "invalid";
}

const char * to_string(reason_code value) {
    switch (value) {
        case reason_code::none:
            return "none";
        case reason_code::admission_ready:
            return "admission_ready";
        case reason_code::admission_capacity_wait:
            return "admission_capacity_wait";
        case reason_code::reject_invalid_request:
            return "reject_invalid_request";
        case reason_code::reject_duplicate_id:
            return "reject_duplicate_id";
        case reason_code::reject_queue_full:
            return "reject_queue_full";
        case reason_code::reject_context_limit:
            return "reject_context_limit";
        case reason_code::reject_capacity_impossible:
            return "reject_capacity_impossible";
        case reason_code::wait_empty:
            return "wait_empty";
        case reason_code::wait_no_feasible_request:
            return "wait_no_feasible_request";
        case reason_code::wait_service_in_flight:
            return "wait_service_in_flight";
        case reason_code::lane_initial_priority:
            return "lane_initial_priority";
        case reason_code::lane_hdrr_debt:
            return "lane_hdrr_debt";
        case reason_code::lane_work_conserving:
            return "lane_work_conserving";
        case reason_code::request_fifo:
            return "request_fifo";
        case reason_code::request_virtual_runtime:
            return "request_virtual_runtime";
        case reason_code::request_cache_affinity:
            return "request_cache_affinity";
        case reason_code::request_bypass_protected:
            return "request_bypass_protected";
        case reason_code::request_aged:
            return "request_aged";
        case reason_code::request_feasible_behind_blocked:
            return "request_feasible_behind_blocked";
        case reason_code::service_complete:
            return "service_complete";
        case reason_code::service_lease_requeue:
            return "service_lease_requeue";
        case reason_code::service_cancelled:
            return "service_cancelled";
        case reason_code::service_invalid_decision:
            return "service_invalid_decision";
        case reason_code::width_empty:
            return "width_empty";
        case reason_code::width_lane_cap:
            return "width_lane_cap";
        case reason_code::width_profiled_shape:
            return "width_profiled_shape";
        case reason_code::width_avoid_low_valley:
            return "width_avoid_low_valley";
        case reason_code::replay_arrival:
            return "replay_arrival";
        case reason_code::replay_stalled:
            return "replay_stalled";
        case reason_code::replay_limit_reached:
            return "replay_limit_reached";
        case reason_code::replay_restore_start:
            return "replay_restore_start";
        case reason_code::replay_restore_ready:
            return "replay_restore_ready";
        case reason_code::replay_spill_start:
            return "replay_spill_start";
        case reason_code::replay_spill_done:
            return "replay_spill_done";
    }
    return "unknown";
}

bool resource_vector::operator==(const resource_vector & other) const {
    return units == other.units;
}

struct scheduler::impl {
    struct queued_request {
        request     req;
        feasibility initial_feasibility = feasibility::feasible_now;
        uint64_t    enqueue_sequence    = 0;
        uint32_t    bypass_count        = 0;
    };

    struct ranked_candidate {
        size_t               queue_index       = 0;
        uint64_t             aged_runtime_us   = 0;
        uint64_t             age_credit_us     = 0;
        uint64_t             affinity_bonus_us = 0;
        uint32_t             blocked_before    = 0;
        candidate_evaluation evaluation;
    };

    struct lane_candidate {
        bool                 available         = false;
        size_t               queue_index       = 0;
        uint64_t             aged_runtime_us   = 0;
        uint64_t             enqueue_sequence  = 0;
        uint64_t             affinity_bonus_us = 0;
        uint32_t             bypass_count      = 0;
        uint32_t             blocked_before    = 0;
        reason_code          reason            = reason_code::none;
        candidate_evaluation evaluation;
        std::vector<size_t>  bypassed_queue_indices;
    };

    struct pending_service {
        bool                         active        = false;
        uint64_t                     decision_id   = 0;
        lane                         selected_lane = lane::low;
        queued_request               selected;
        std::array<bool, lane_count> active_lanes = {};
    };

    explicit impl(scheduler_config config_in) : config(std::move(config_in)) {
        if (config.fairness_quantum_us == 0 || config.cache_lookahead == 0 || config.max_bypasses == 0 ||
            config.aging_interval_us == 0 || config.context_tokens == 0) {
            throw std::invalid_argument("scheduler configuration values must be non-zero");
        }
        for (size_t i = 0; i < lane_count; ++i) {
            const lane_descriptor & descriptor = config.lanes[i];
            if (descriptor.id != static_cast<lane>(i) || descriptor.weight == 0 || descriptor.decode_cap == 0 ||
                descriptor.queue_cap == 0) {
                throw std::invalid_argument("invalid lane descriptor");
            }
        }
    }

    uint64_t age_credit(const queued_request & queued, uint64_t now_us) const {
        const uint64_t waited  = now_us > queued.req.arrival_us ? now_us - queued.req.arrival_us : 0;
        const uint64_t periods = waited / config.aging_interval_us;
        return std::min(saturating_multiply(periods, config.aging_credit_us), config.max_aging_credit_us);
    }

    uint64_t aged_runtime(const queued_request & queued, uint64_t now_us) const {
        const uint64_t credit = age_credit(queued, now_us);
        return credit >= queued.req.virtual_runtime_us ? 0 : queued.req.virtual_runtime_us - credit;
    }

    lane_candidate candidate_for_lane(size_t lane_id, uint64_t now_us, const evaluation_provider & evaluate) const {
        lane_candidate                      result;
        const std::vector<queued_request> & lane_queue = queues[lane_id];
        if (lane_queue.empty()) {
            return result;
        }

        std::vector<size_t> order(lane_queue.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
            const queued_request & left          = lane_queue[lhs];
            const queued_request & right         = lane_queue[rhs];
            const uint64_t         left_runtime  = aged_runtime(left, now_us);
            const uint64_t         right_runtime = aged_runtime(right, now_us);
            if (left_runtime != right_runtime) {
                return left_runtime < right_runtime;
            }
            if (left.req.arrival_us != right.req.arrival_us) {
                return left.req.arrival_us < right.req.arrival_us;
            }
            if (left.enqueue_sequence != right.enqueue_sequence) {
                return left.enqueue_sequence < right.enqueue_sequence;
            }
            return left.req.id < right.req.id;
        });

        std::vector<ranked_candidate> feasible;
        uint32_t                      blocked_before = 0;
        for (size_t rank = 0; rank < order.size(); ++rank) {
            const size_t           queue_index = order[rank];
            const queued_request & queued      = lane_queue[queue_index];
            const uint64_t         runtime     = aged_runtime(queued, now_us);

            candidate_evaluation current;
            if (evaluate) {
                current = evaluate(queued.req);
            } else {
                current.state = queued.initial_feasibility;
            }
            if (current.state != feasibility::feasible_now) {
                blocked_before += current.state == feasibility::temporarily_blocked;
                continue;
            }
            if (!feasible.empty() &&
                runtime > saturating_add(feasible.front().aged_runtime_us, config.fairness_quantum_us)) {
                break;
            }

            const uint64_t gross_affinity = current.cached_prefix_us > current.restore_cost_us ?
                                                current.cached_prefix_us - current.restore_cost_us :
                                                0;
            const uint64_t bonus_cap      = config.fairness_quantum_us - 1;
            feasible.push_back({
                queue_index,
                runtime,
                age_credit(queued, now_us),
                std::min(gross_affinity, bonus_cap),
                blocked_before,
                current,
            });
            if (feasible.size() == config.cache_lookahead) {
                break;
            }
        }

        if (feasible.empty()) {
            return result;
        }

        size_t chosen            = 0;
        bool   protected_request = false;
        for (size_t i = 0; i < feasible.size(); ++i) {
            if (lane_queue[feasible[i].queue_index].bypass_count >= config.max_bypasses) {
                chosen            = i;
                protected_request = true;
                break;
            }
        }

        if (!protected_request) {
            for (size_t i = 1; i < feasible.size(); ++i) {
                const ranked_candidate & current       = feasible[i];
                const ranked_candidate & best          = feasible[chosen];
                const uint64_t           current_score = current.affinity_bonus_us >= current.aged_runtime_us ?
                                                             0 :
                                                             current.aged_runtime_us - current.affinity_bonus_us;
                const uint64_t           best_score =
                    best.affinity_bonus_us >= best.aged_runtime_us ? 0 : best.aged_runtime_us - best.affinity_bonus_us;
                if (current_score < best_score) {
                    chosen = i;
                }
            }
        }

        const ranked_candidate & selected         = feasible[chosen];
        const queued_request &   selected_request = lane_queue[selected.queue_index];
        result.available                          = true;
        result.queue_index                        = selected.queue_index;
        result.aged_runtime_us                    = selected.aged_runtime_us;
        result.enqueue_sequence                   = selected_request.enqueue_sequence;
        result.affinity_bonus_us                  = selected.affinity_bonus_us;
        result.bypass_count                       = selected_request.bypass_count;
        result.blocked_before                     = selected.blocked_before;
        result.evaluation                         = selected.evaluation;

        if (protected_request) {
            result.reason = reason_code::request_bypass_protected;
        } else if (chosen == 0 && selected.blocked_before > 0) {
            result.reason = reason_code::request_feasible_behind_blocked;
        } else if (selected.affinity_bonus_us > 0 && chosen > 0) {
            result.reason = reason_code::request_cache_affinity;
        } else if (selected.age_credit_us > 0 && selected_request.req.virtual_runtime_us != selected.aged_runtime_us) {
            result.reason = reason_code::request_aged;
        } else if (selected_request.enqueue_sequence !=
                   std::min_element(lane_queue.begin(), lane_queue.end(),
                                    [](const queued_request & lhs, const queued_request & rhs) {
                                        return lhs.enqueue_sequence < rhs.enqueue_sequence;
                                    })
                       ->enqueue_sequence) {
            result.reason = reason_code::request_virtual_runtime;
        } else {
            result.reason = reason_code::request_fifo;
        }

        for (size_t i = 0; i < chosen; ++i) {
            result.bypassed_queue_indices.push_back(feasible[i].queue_index);
        }
        return result;
    }

    scheduler_config                                    config;
    std::array<std::vector<queued_request>, lane_count> queues;
    std::array<int64_t, lane_count>                     hdrr_credit_units = {};
    std::array<uint64_t, lane_count>                    actual_gpu_us     = {};
    std::array<uint64_t, lane_count>                    dispatches        = {};
    std::set<uint64_t>                                  request_ids;
    uint64_t                                            next_enqueue_sequence = 1;
    uint64_t                                            next_decision_id      = 1;
    pending_service                                     pending;
};

scheduler::scheduler(scheduler_config config) : pimpl(std::make_unique<impl>(std::move(config))) {}

scheduler::~scheduler() = default;

scheduler::scheduler(scheduler && other) noexcept = default;

scheduler & scheduler::operator=(scheduler && other) noexcept = default;

admission_result scheduler::admit(const request & req, const feasibility_quote & quote) {
    admission_result result;
    result.limiting_dimension = quote.limiting_dimension;
    if (!valid_lane(req.priority) || req.id == 0 || req.cached_prompt_tokens > req.prompt_tokens ||
        req.decode_runway_tokens == 0) {
        result.reason = reason_code::reject_invalid_request;
        return result;
    }
    if (pimpl->request_ids.count(req.id) != 0) {
        result.reason = reason_code::reject_duplicate_id;
        return result;
    }
    if (req.prompt_tokens > pimpl->config.context_tokens) {
        result.reason = reason_code::reject_context_limit;
        return result;
    }
    const uint64_t remaining_context = pimpl->config.context_tokens - req.prompt_tokens;
    if (req.decode_runway_tokens > remaining_context ||
        (req.requested_output_tokens != 0 && req.requested_output_tokens > remaining_context)) {
        result.reason = reason_code::reject_context_limit;
        return result;
    }
    if (quote.state == feasibility::impossible) {
        result.reason = reason_code::reject_capacity_impossible;
        return result;
    }

    const size_t index = lane_index(req.priority);
    if (pimpl->queues[index].size() >= pimpl->config.lanes[index].queue_cap) {
        result.reason = reason_code::reject_queue_full;
        return result;
    }

    pimpl->queues[index].push_back({ req, quote.state, pimpl->next_enqueue_sequence++, 0 });
    pimpl->request_ids.insert(req.id);
    result.accepted = true;
    result.ready    = quote.state == feasibility::feasible_now;
    result.reason   = result.ready ? reason_code::admission_ready : reason_code::admission_capacity_wait;
    return result;
}

service_decision scheduler::select_next(uint64_t now_us, const evaluation_provider & evaluate) {
    service_decision result;
    if (pimpl->pending.active) {
        result.reason = reason_code::wait_service_in_flight;
        return result;
    }
    if (queued_total() == 0) {
        result.reason = reason_code::wait_empty;
        return result;
    }

    std::array<impl::lane_candidate, lane_count> candidates;
    size_t                                       available_lanes = 0;
    for (size_t i = 0; i < lane_count; ++i) {
        candidates[i] = pimpl->candidate_for_lane(i, now_us, evaluate);
        if (candidates[i].available) {
            ++available_lanes;
        }
    }
    if (available_lanes == 0) {
        result.reason = reason_code::wait_no_feasible_request;
        return result;
    }

    size_t selected_lane_index = lane_count;
    bool   tied_credit         = false;
    for (size_t i = 0; i < lane_count; ++i) {
        if (!candidates[i].available) {
            continue;
        }
        if (selected_lane_index == lane_count) {
            selected_lane_index = i;
            continue;
        }
        if (pimpl->hdrr_credit_units[i] > pimpl->hdrr_credit_units[selected_lane_index]) {
            selected_lane_index = i;
            tied_credit         = false;
            continue;
        }
        if (pimpl->hdrr_credit_units[i] < pimpl->hdrr_credit_units[selected_lane_index]) {
            continue;
        }
        tied_credit                      = true;
        const lane_descriptor & current  = pimpl->config.lanes[i];
        const lane_descriptor & selected = pimpl->config.lanes[selected_lane_index];
        if (current.weight > selected.weight ||
            (current.weight == selected.weight &&
             candidates[i].aged_runtime_us < candidates[selected_lane_index].aged_runtime_us) ||
            (current.weight == selected.weight &&
             candidates[i].aged_runtime_us == candidates[selected_lane_index].aged_runtime_us &&
             candidates[i].enqueue_sequence < candidates[selected_lane_index].enqueue_sequence)) {
            selected_lane_index = i;
        }
    }

    impl::lane_candidate &              selected_candidate = candidates[selected_lane_index];
    std::vector<impl::queued_request> & selected_queue     = pimpl->queues[selected_lane_index];
    for (size_t queue_index : selected_candidate.bypassed_queue_indices) {
        if (queue_index < selected_queue.size()) {
            selected_queue[queue_index].bypass_count =
                std::min(selected_queue[queue_index].bypass_count + 1, pimpl->config.max_bypasses);
        }
    }

    pimpl->pending.active        = true;
    pimpl->pending.decision_id   = pimpl->next_decision_id++;
    pimpl->pending.selected_lane = static_cast<lane>(selected_lane_index);
    pimpl->pending.selected      = selected_queue[selected_candidate.queue_index];
    for (size_t i = 0; i < lane_count; ++i) {
        pimpl->pending.active_lanes[i] = candidates[i].available;
    }
    selected_queue.erase(selected_queue.begin() + static_cast<std::ptrdiff_t>(selected_candidate.queue_index));

    result.selected            = true;
    result.decision_id         = pimpl->pending.decision_id;
    result.request_id          = pimpl->pending.selected.req.id;
    result.selected_lane       = pimpl->pending.selected_lane;
    result.reason              = selected_candidate.reason;
    result.predicted_gpu_us    = selected_candidate.evaluation.predicted_gpu_us;
    result.affinity_bonus_us   = selected_candidate.affinity_bonus_us;
    result.bypass_count_before = selected_candidate.bypass_count;
    result.blocked_before      = selected_candidate.blocked_before;
    result.limiting_dimension  = selected_candidate.evaluation.limiting_dimension;
    if (available_lanes == 1) {
        result.lane_reason = reason_code::lane_work_conserving;
    } else if (tied_credit) {
        result.lane_reason = reason_code::lane_initial_priority;
    } else {
        result.lane_reason = reason_code::lane_hdrr_debt;
    }
    return result;
}

completion_result scheduler::complete_service(uint64_t            decision_id,
                                              uint64_t            actual_gpu_us_value,
                                              service_disposition disposition) {
    completion_result result;
    if (!pimpl->pending.active || pimpl->pending.decision_id != decision_id) {
        result.reason = reason_code::service_invalid_decision;
        return result;
    }

    const size_t selected      = lane_index(pimpl->pending.selected_lane);
    uint64_t     active_weight = 0;
    for (size_t i = 0; i < lane_count; ++i) {
        if (pimpl->pending.active_lanes[i]) {
            active_weight += pimpl->config.lanes[i].weight;
            const int64_t earned = positive_product_as_signed(actual_gpu_us_value, pimpl->config.lanes[i].weight);
            pimpl->hdrr_credit_units[i] = saturating_add_signed(pimpl->hdrr_credit_units[i], earned);
        }
    }
    const int64_t charge               = positive_product_as_signed(actual_gpu_us_value, active_weight);
    // One completed GPU epoch is one HDRR accounting round: every lane that
    // had feasible work earns its weighted entitlement, while the served lane
    // pays the complete measured GPU cost in the same normalized units.
    pimpl->hdrr_credit_units[selected] = saturating_add_signed(pimpl->hdrr_credit_units[selected], -charge);
    pimpl->actual_gpu_us[selected]     = saturating_add(pimpl->actual_gpu_us[selected], actual_gpu_us_value);
    ++pimpl->dispatches[selected];

    impl::queued_request completed_request = pimpl->pending.selected;
    pimpl->pending                         = {};
    result.completed                       = true;
    switch (disposition) {
        case service_disposition::requeue:
            completed_request.req.virtual_runtime_us =
                saturating_add(completed_request.req.virtual_runtime_us, actual_gpu_us_value);
            completed_request.bypass_count = 0;
            pimpl->queues[selected].push_back(std::move(completed_request));
            result.reason = reason_code::service_lease_requeue;
            break;
        case service_disposition::complete:
            pimpl->request_ids.erase(completed_request.req.id);
            result.reason = reason_code::service_complete;
            break;
        case service_disposition::cancelled:
            pimpl->request_ids.erase(completed_request.req.id);
            result.reason = reason_code::service_cancelled;
            break;
    }

    if (pimpl->queues[selected].empty()) {
        pimpl->hdrr_credit_units[selected] = 0;
    }
    return result;
}

decode_width_decision scheduler::choose_decode_width(lane   dominant_lane,
                                                     size_t runnable_in_dominant_lane,
                                                     size_t total_runnable,
                                                     bool   cross_lane_fill_is_profiled_safe) const {
    decode_width_decision result;
    if (!valid_lane(dominant_lane)) {
        return result;
    }
    const size_t available = cross_lane_fill_is_profiled_safe ? total_runnable : runnable_in_dominant_lane;
    if (available == 0) {
        return result;
    }

    const uint32_t cap     = pimpl->config.lanes[lane_index(dominant_lane)].decode_cap;
    const uint32_t bounded = static_cast<uint32_t>(std::min<size_t>(available, cap));
    if (dominant_lane != lane::low) {
        result.width  = bounded;
        result.reason = available > cap ? reason_code::width_lane_cap : reason_code::width_profiled_shape;
        return result;
    }

    if (bounded >= 64) {
        result.width  = 64;
        result.reason = available > 64 ? reason_code::width_lane_cap : reason_code::width_profiled_shape;
    } else if (bounded >= 32) {
        result.width  = 32;
        result.reason = reason_code::width_profiled_shape;
    } else if (bounded >= 24) {
        result.width  = 24;
        result.reason = reason_code::width_profiled_shape;
    } else if (bounded >= 9) {
        result.width  = 8;
        result.reason = reason_code::width_avoid_low_valley;
    } else if (bounded >= 8) {
        result.width  = 8;
        result.reason = reason_code::width_profiled_shape;
    } else if (bounded >= 4) {
        result.width  = 4;
        result.reason = reason_code::width_profiled_shape;
    } else if (bounded >= 2) {
        result.width  = 2;
        result.reason = reason_code::width_profiled_shape;
    } else {
        result.width  = bounded;
        result.reason = reason_code::width_profiled_shape;
    }
    return result;
}

bool scheduler::contains(uint64_t request_id) const {
    return pimpl->request_ids.count(request_id) != 0;
}

size_t scheduler::queued(lane priority) const {
    return valid_lane(priority) ? pimpl->queues[lane_index(priority)].size() : 0;
}

size_t scheduler::queued_total() const {
    size_t result = 0;
    for (const auto & lane_queue : pimpl->queues) {
        result += lane_queue.size();
    }
    return result;
}

lane_snapshot scheduler::snapshot(lane priority) const {
    lane_snapshot result;
    if (!valid_lane(priority)) {
        return result;
    }
    const size_t index       = lane_index(priority);
    result.queued            = pimpl->queues[index].size();
    result.hdrr_credit_units = pimpl->hdrr_credit_units[index];
    result.actual_gpu_us     = pimpl->actual_gpu_us[index];
    result.dispatches        = pimpl->dispatches[index];
    if (!pimpl->queues[index].empty()) {
        result.oldest_arrival_us = pimpl->queues[index].front().req.arrival_us;
        for (const impl::queued_request & queued_request : pimpl->queues[index]) {
            result.oldest_arrival_us = std::min(result.oldest_arrival_us, queued_request.req.arrival_us);
        }
    }
    return result;
}

const scheduler_config & scheduler::config() const {
    return pimpl->config;
}

bool replay_event::operator==(const replay_event & other) const {
    return kind == other.kind && time_us == other.time_us && request_id == other.request_id &&
           priority == other.priority && reason == other.reason && lane_reason == other.lane_reason &&
           gpu_us == other.gpu_us && io_us == other.io_us;
}

bool replay_result::operator==(const replay_result & other) const {
    return events == other.events && resident == other.resident && dispatches == other.dispatches &&
           end_time_us == other.end_time_us;
}

replay_result simulator::replay(const replay_trace & trace) const {
    struct runtime_job {
        trace_job job;
        bool      admitted         = false;
        bool      resources_held   = false;
        bool      restore_started  = false;
        bool      restore_ready    = false;
        bool      spill_started    = false;
        bool      spill_done       = false;
        bool      finished         = false;
        uint64_t  restore_ready_us = 0;
        uint64_t  spill_done_us    = 0;
        uint64_t  completed_quanta = 0;
        uint64_t  required_quanta  = 0;
    };

    if (trace.decode_lease_tokens == 0) {
        throw std::invalid_argument("decode lease must be non-zero");
    }

    replay_result            result;
    scheduler                policy(trace.policy);
    std::vector<runtime_job> jobs;
    jobs.reserve(trace.jobs.size());
    for (const trace_job & job : trace.jobs) {
        runtime_job runtime;
        runtime.job = job;
        if (job.req.requested_output_tokens == 0 && job.observed_output_tokens == 0) {
            runtime.required_quanta = 0;
        } else {
            const uint64_t observed = std::max<uint64_t>(1, job.observed_output_tokens);
            runtime.required_quanta = 1 + (observed - 1) / trace.decode_lease_tokens;
        }
        jobs.push_back(std::move(runtime));
    }

    std::vector<size_t> arrival_order(jobs.size());
    for (size_t i = 0; i < arrival_order.size(); ++i) {
        arrival_order[i] = i;
    }
    std::stable_sort(arrival_order.begin(), arrival_order.end(), [&](size_t lhs, size_t rhs) {
        const request & left  = jobs[lhs].job.req;
        const request & right = jobs[rhs].job.req;
        if (left.arrival_us != right.arrival_us) {
            return left.arrival_us < right.arrival_us;
        }
        if (left.id != right.id) {
            return left.id < right.id;
        }
        return lhs < rhs;
    });

    std::map<uint64_t, size_t> job_by_id;
    for (size_t i = 0; i < jobs.size(); ++i) {
        job_by_id.emplace(jobs[i].job.req.id, i);
    }

    size_t   next_arrival = 0;
    uint64_t now_us       = arrival_order.empty() ? 0 : jobs[arrival_order.front()].job.req.arrival_us;

    const auto start_restore = [&](runtime_job & runtime, uint64_t event_us) {
        if (runtime.restore_started || runtime.resources_held ||
            !vector_fits(result.resident, runtime.job.page_demand, trace.capacity, trace.safety_watermark)) {
            return false;
        }

        vector_add(result.resident, runtime.job.page_demand);
        runtime.resources_held  = true;
        runtime.restore_started = true;
        if (runtime.job.restore_io_us == 0) {
            runtime.restore_ready = true;
            return true;
        }

        runtime.restore_ready_us = saturating_add(event_us, runtime.job.restore_io_us);
        result.events.push_back({
            replay_event_kind::io_start,
            event_us,
            runtime.job.req.id,
            runtime.job.req.priority,
            reason_code::replay_restore_start,
            reason_code::none,
            0,
            runtime.job.restore_io_us,
        });
        return true;
    };

    const auto start_waiting_restores = [&](uint64_t event_us) {
        for (size_t index : arrival_order) {
            runtime_job & runtime = jobs[index];
            // Zero-duration restores remain scheduler candidates and reserve
            // only when selected, preserving lane policy as capacity frees.
            // Timed restores start FIFO because this Phase 0 simulator does
            // not yet model the Phase 5 prioritized device queue.
            if (runtime.admitted && !runtime.finished && !runtime.restore_started && runtime.job.restore_io_us > 0) {
                start_restore(runtime, event_us);
            }
        }
    };

    const auto next_external_time = [&]() -> std::optional<uint64_t> {
        std::optional<uint64_t> next;
        const auto              consider = [&](uint64_t candidate) {
            if (!next || candidate < *next) {
                next = candidate;
            }
        };
        if (next_arrival < arrival_order.size()) {
            consider(jobs[arrival_order[next_arrival]].job.req.arrival_us);
        }
        for (const runtime_job & runtime : jobs) {
            if (runtime.restore_started && !runtime.restore_ready) {
                consider(runtime.restore_ready_us);
            }
            if (runtime.spill_started && !runtime.spill_done) {
                consider(runtime.spill_done_us);
            }
        }
        return next;
    };

    const auto process_external_at = [&](uint64_t event_us) {
        // Complete I/O before same-timestamp arrivals so newly freed capacity
        // is visible to their admission quote.
        for (size_t index : arrival_order) {
            runtime_job & runtime = jobs[index];
            if (runtime.restore_started && !runtime.restore_ready && runtime.restore_ready_us == event_us) {
                runtime.restore_ready = true;
                result.events.push_back({
                    replay_event_kind::io_complete,
                    event_us,
                    runtime.job.req.id,
                    runtime.job.req.priority,
                    reason_code::replay_restore_ready,
                    reason_code::none,
                    0,
                    runtime.job.restore_io_us,
                });
            }
            if (runtime.spill_started && !runtime.spill_done && runtime.spill_done_us == event_us) {
                runtime.spill_done = true;
                if (runtime.resources_held) {
                    vector_subtract(result.resident, runtime.job.page_demand);
                    runtime.resources_held = false;
                }
                result.events.push_back({
                    replay_event_kind::io_complete,
                    event_us,
                    runtime.job.req.id,
                    runtime.job.req.priority,
                    reason_code::replay_spill_done,
                    reason_code::none,
                    0,
                    runtime.job.spill_io_us,
                });
            }
        }

        while (next_arrival < arrival_order.size()) {
            runtime_job & runtime = jobs[arrival_order[next_arrival]];
            if (runtime.job.req.arrival_us != event_us) {
                break;
            }
            result.events.push_back({
                replay_event_kind::arrival,
                runtime.job.req.arrival_us,
                runtime.job.req.id,
                runtime.job.req.priority,
                reason_code::replay_arrival,
            });

            feasibility_quote quote;
            quote.delta = runtime.job.page_demand;
            if (!vector_can_ever_fit(runtime.job.page_demand, trace.capacity, trace.safety_watermark,
                                         quote.limiting_dimension)) {
                quote.state = feasibility::impossible;
            } else if (vector_fits(result.resident, runtime.job.page_demand, trace.capacity, trace.safety_watermark)) {
                quote.state = feasibility::feasible_now;
            } else {
                quote.state = feasibility::temporarily_blocked;
            }
            const admission_result admission = policy.admit(runtime.job.req, quote);
            runtime.admitted                 = admission.accepted;
            result.events.push_back({
                replay_event_kind::admission,
                event_us,
                runtime.job.req.id,
                runtime.job.req.priority,
                admission.reason,
            });
            if (admission.accepted && admission.ready) {
                start_restore(runtime, event_us);
            }
            ++next_arrival;
        }

        start_waiting_restores(event_us);
    };

    if (next_external_time()) {
        process_external_at(now_us);
    }

    while (true) {
        if (result.dispatches >= trace.max_dispatches) {
            result.events.push_back({
                replay_event_kind::limit,
                now_us,
                0,
                lane::low,
                reason_code::replay_limit_reached,
            });
            break;
        }

        if (policy.queued_total() == 0) {
            const auto next = next_external_time();
            if (!next) {
                break;
            }
            now_us = std::max(now_us, *next);
            process_external_at(now_us);
            continue;
        }

        const evaluation_provider evaluate = [&](const request & req) {
            candidate_evaluation value;
            const auto           found = job_by_id.find(req.id);
            if (found == job_by_id.end()) {
                value.state = feasibility::impossible;
                return value;
            }
            const runtime_job & runtime = jobs[found->second];
            value.state                 = (runtime.resources_held && runtime.restore_ready) ||
                                  (!runtime.restore_started && runtime.job.restore_io_us == 0 &&
                                   vector_fits(result.resident, runtime.job.page_demand, trace.capacity,
                                                               trace.safety_watermark)) ?
                                              feasibility::feasible_now :
                                              feasibility::temporarily_blocked;
            value.predicted_gpu_us = runtime.job.service_gpu_us;
            value.cached_prefix_us = runtime.job.cached_prefix_us;
            value.restore_cost_us  = runtime.job.restore_cost_us;
            return value;
        };

        const service_decision decision = policy.select_next(now_us, evaluate);
        if (!decision.selected) {
            const auto next = next_external_time();
            if (next) {
                now_us = std::max(now_us, *next);
                process_external_at(now_us);
                continue;
            }
            result.events.push_back({
                replay_event_kind::stalled,
                now_us,
                0,
                lane::low,
                reason_code::replay_stalled,
                decision.reason,
            });
            break;
        }

        runtime_job & runtime = jobs[job_by_id.at(decision.request_id)];
        if (!runtime.resources_held && (!start_restore(runtime, now_us) || !runtime.restore_ready)) {
            throw std::runtime_error("selected replay request could not reserve ready resources");
        }
        result.events.push_back({
            replay_event_kind::dispatch,
            now_us,
            decision.request_id,
            decision.selected_lane,
            decision.reason,
            decision.lane_reason,
            runtime.job.service_gpu_us,
        });

        const uint64_t completion_us = saturating_add(now_us, runtime.job.service_gpu_us);
        while (true) {
            const auto next = next_external_time();
            if (!next || *next > completion_us) {
                break;
            }
            now_us = std::max(now_us, *next);
            process_external_at(now_us);
        }

        ++runtime.completed_quanta;
        const bool completes = runtime.required_quanta != 0 && runtime.completed_quanta >= runtime.required_quanta;
        const service_disposition disposition =
            completes ? service_disposition::complete : service_disposition::requeue;
        const completion_result completion =
            policy.complete_service(decision.decision_id, runtime.job.service_gpu_us, disposition);
        now_us = completion_us;
        ++result.dispatches;
        result.events.push_back({
            replay_event_kind::completion,
            now_us,
            decision.request_id,
            decision.selected_lane,
            completion.reason,
            reason_code::none,
            runtime.job.service_gpu_us,
        });
        if (completes) {
            runtime.finished = true;
            if (runtime.job.spill_io_us == 0) {
                vector_subtract(result.resident, runtime.job.page_demand);
                runtime.resources_held = false;
            } else {
                runtime.spill_started = true;
                runtime.spill_done_us = saturating_add(now_us, runtime.job.spill_io_us);
                result.events.push_back({
                    replay_event_kind::io_start,
                    now_us,
                    runtime.job.req.id,
                    runtime.job.req.priority,
                    reason_code::replay_spill_start,
                    reason_code::none,
                    0,
                    runtime.job.spill_io_us,
                });
            }
            start_waiting_restores(now_us);
        }
    }

    result.end_time_us = now_us;
    return result;
}

}  // namespace server_scheduler

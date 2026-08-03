#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace server_scheduler;

namespace {

request make_request(uint64_t id,
                     lane     priority,
                     uint64_t arrival_us         = 0,
                     uint64_t virtual_runtime_us = 0,
                     uint64_t prompt_tokens      = 1,
                     uint64_t output_tokens      = 0) {
    request value;
    value.id                      = id;
    value.priority                = priority;
    value.arrival_us              = arrival_us;
    value.virtual_runtime_us      = virtual_runtime_us;
    value.prompt_tokens           = prompt_tokens;
    value.requested_output_tokens = output_tokens;
    value.decode_runway_tokens    = 1;
    return value;
}

feasibility_quote feasible_quote(uint64_t units = 0) {
    feasibility_quote quote;
    quote.delta.units[0] = units;
    return quote;
}

candidate_evaluation feasible_candidate(uint64_t gpu_us = 1000, uint64_t cache_us = 0, uint64_t restore_us = 0) {
    candidate_evaluation value;
    value.predicted_gpu_us = gpu_us;
    value.cached_prefix_us = cache_us;
    value.restore_cost_us  = restore_us;
    return value;
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_default_descriptors() {
    const auto & lanes = default_lane_descriptors();
    require(lanes[static_cast<size_t>(lane::low)].weight == 1, "low weight");
    require(lanes[static_cast<size_t>(lane::normal)].weight == 4, "normal weight");
    require(lanes[static_cast<size_t>(lane::fast)].weight == 16, "fast weight");
    require(lanes[static_cast<size_t>(lane::low)].decode_cap == 64, "low cap");
    require(lanes[static_cast<size_t>(lane::normal)].decode_cap == 8, "normal cap");
    require(lanes[static_cast<size_t>(lane::fast)].decode_cap == 2, "fast cap");
    require(lanes[static_cast<size_t>(lane::low)].queue_cap == 256, "low queue cap");
    require(lanes[static_cast<size_t>(lane::normal)].queue_cap == 64, "normal queue cap");
    require(lanes[static_cast<size_t>(lane::fast)].queue_cap == 16, "fast queue cap");
}

void test_fifo_virtual_runtime_and_ties() {
    scheduler policy;
    require(policy.admit(make_request(1, lane::normal, 0, 2000), feasible_quote()).accepted, "admit 1");
    require(policy.admit(make_request(2, lane::normal, 1, 1000), feasible_quote()).accepted, "admit 2");
    require(policy.admit(make_request(3, lane::normal, 1, 1000), feasible_quote()).accepted, "admit 3");

    const auto evaluate = [](const request &) {
        return feasible_candidate();
    };
    service_decision decision = policy.select_next(1, evaluate);
    require(decision.selected && decision.request_id == 2, "virtual runtime must precede FIFO");
    require(decision.reason == reason_code::request_virtual_runtime, "virtual-runtime reason");
    require(policy.complete_service(decision.decision_id, 1000, service_disposition::complete).completed,
            "complete virtual-runtime head");

    decision = policy.select_next(1, evaluate);
    require(decision.selected && decision.request_id == 3, "equal runtime must use stable FIFO tie break");
    policy.complete_service(decision.decision_id, 1000, service_disposition::complete);

    decision = policy.select_next(1, evaluate);
    require(decision.selected && decision.request_id == 1, "remaining request must run");
}

void test_cancel_queued_request() {
    scheduler_config config;
    config.lanes[static_cast<size_t>(lane::fast)].queue_cap = 1;
    scheduler policy(config);

    require(policy.admit(make_request(1, lane::fast), feasible_quote()).accepted, "admit cancellable request");
    require(!policy.admit(make_request(2, lane::fast), feasible_quote()).accepted, "queue cap reached");

    const completion_result cancelled = policy.cancel(1);
    require(cancelled.completed && cancelled.reason == reason_code::service_cancelled, "cancel queued request");
    require(!policy.contains(1) && policy.queued_total() == 0, "cancel removes scheduler identity");
    require(policy.admit(make_request(2, lane::fast), feasible_quote()).accepted, "cancel releases lane capacity");
    require(!policy.cancel(999).completed, "unknown cancellation is rejected");

    const service_decision selected = policy.select_next(1, {});
    require(selected.selected && selected.request_id == 2, "remaining request selected");
    require(!policy.cancel(2).completed, "in-flight decision cannot be cancelled as queued work");
    require(policy.complete_service(selected.decision_id, 1, service_disposition::cancelled).completed,
            "in-flight request uses decision completion");
}

void test_cache_lookahead_bonus_and_bypass_bound() {
    scheduler_config config;
    config.aging_credit_us     = 1;
    config.max_aging_credit_us = 1;
    scheduler policy(config);
    require(policy.admit(make_request(1, lane::normal, 0, 500), feasible_quote()).accepted, "admit miss");

    const auto evaluate = [](const request & req) {
        return feasible_candidate(1000, req.id == 1 ? 0 : 1000000, 0);
    };

    for (uint64_t id = 2; id <= 4; ++id) {
        require(policy.admit(make_request(id, lane::normal, 0, 500), feasible_quote()).accepted, "admit cache hit");
        const service_decision decision = policy.select_next(0, evaluate);
        require(decision.selected && decision.request_id == id, "cache hit should bypass old miss");
        require(decision.reason == reason_code::request_cache_affinity, "cache-affinity reason");
        require(decision.affinity_bonus_us == config.fairness_quantum_us - 1, "bonus capped below quantum");
        policy.complete_service(decision.decision_id, 1000, service_disposition::complete);
    }

    require(policy.admit(make_request(5, lane::normal, 0, 500), feasible_quote()).accepted, "admit fourth hit");
    service_decision decision = policy.select_next(0, evaluate);
    require(decision.selected && decision.request_id == 1, "three bypass maximum must protect oldest feasible");
    require(decision.reason == reason_code::request_bypass_protected, "bypass-protection reason");
    require(decision.bypass_count_before == 3, "protected request records three bypasses");

    scheduler bounded(config);
    for (uint64_t id = 1; id <= 9; ++id) {
        require(bounded.admit(make_request(id, lane::low, 0, 500), feasible_quote()).accepted, "bounded admit");
    }
    const auto ninth_hit = [](const request & req) {
        return feasible_candidate(1000, req.id == 9 ? 1000000 : 0, 0);
    };
    decision = bounded.select_next(0, ninth_hit);
    require(decision.selected && decision.request_id == 1, "ninth request must remain outside K=8 lookahead");
}

void test_infeasible_head_is_not_bypassed() {
    scheduler policy;
    require(policy.admit(make_request(1, lane::fast), feasible_quote()).accepted, "admit blocked head");
    require(policy.admit(make_request(2, lane::fast), feasible_quote()).accepted, "admit feasible tail");
    auto evaluate = [](const request & req) {
        candidate_evaluation value = feasible_candidate();
        value.state                = req.id == 1 ? feasibility::temporarily_blocked : feasibility::feasible_now;
        return value;
    };
    service_decision decision = policy.select_next(0, evaluate);
    require(decision.selected && decision.request_id == 2, "feasible tail should preserve work conservation");
    require(decision.reason == reason_code::request_feasible_behind_blocked,
            "feasible tail needs an explicit blocked-head reason");
    require(decision.blocked_before == 1, "decision records one earlier temporarily blocked request");
    policy.complete_service(decision.decision_id, 1000, service_disposition::complete);

    decision = policy.select_next(0, [](const request &) { return feasible_candidate(); });
    require(decision.selected && decision.request_id == 1, "unblocked head should run");
    require(decision.bypass_count_before == 0, "infeasible head must not consume bypass allowance");

    scheduler long_blocked_prefix;
    for (uint64_t id = 1; id <= 9; ++id) {
        long_blocked_prefix.admit(make_request(id, lane::low), feasible_quote());
    }
    decision = long_blocked_prefix.select_next(0, [](const request & req) {
        candidate_evaluation value = feasible_candidate();
        value.state                = req.id == 9 ? feasibility::feasible_now : feasibility::temporarily_blocked;
        return value;
    });
    require(decision.selected && decision.request_id == 9,
            "K=8 counts eligible candidates rather than blocked queue entries");
    require(decision.reason == reason_code::request_feasible_behind_blocked && decision.blocked_before == 8,
            "decision preserves the complete blocked-prefix reason context");
}

void test_aging() {
    scheduler_config config;
    config.aging_interval_us   = 1000;
    config.aging_credit_us     = 250;
    config.max_aging_credit_us = 10000;
    scheduler policy(config);
    require(policy.admit(make_request(1, lane::normal, 0, 4000), feasible_quote()).accepted, "admit old");
    require(policy.admit(make_request(2, lane::normal, 20000, 0), feasible_quote()).accepted, "admit new");

    const service_decision decision = policy.select_next(20000, [](const request &) { return feasible_candidate(); });
    require(decision.selected && decision.request_id == 1, "aging must make old request runnable first");
    require(decision.reason == reason_code::request_aged, "aging reason");
}

void test_hdrr_shares_and_low_progress() {
    scheduler_config config;
    config.aging_credit_us     = 1;
    config.max_aging_credit_us = 1;
    scheduler policy(config);
    require(policy.admit(make_request(1, lane::low), feasible_quote()).accepted, "admit low");
    require(policy.admit(make_request(2, lane::normal), feasible_quote()).accepted, "admit normal");
    require(policy.admit(make_request(3, lane::fast), feasible_quote()).accepted, "admit fast");

    std::array<uint64_t, lane_count> counts = {};
    for (size_t i = 0; i < 2100; ++i) {
        const service_decision decision =
            policy.select_next(i * 1000, [](const request &) { return feasible_candidate(); });
        require(decision.selected, "HDRR must always select a backlogged lane");
        ++counts[static_cast<size_t>(decision.selected_lane)];
        require(policy.complete_service(decision.decision_id, 1000, service_disposition::requeue).completed,
                "HDRR completion");
    }

    require(counts[static_cast<size_t>(lane::low)] >= 98, "low lane must make sustained progress");
    require(counts[static_cast<size_t>(lane::low)] <= 102, "low share must approximate 1/21");
    require(counts[static_cast<size_t>(lane::normal)] >= 398, "normal share lower bound");
    require(counts[static_cast<size_t>(lane::normal)] <= 402, "normal share upper bound");
    require(counts[static_cast<size_t>(lane::fast)] >= 1598, "fast share lower bound");
    require(counts[static_cast<size_t>(lane::fast)] <= 1602, "fast share upper bound");

    const lane_snapshot low    = policy.snapshot(lane::low);
    const lane_snapshot normal = policy.snapshot(lane::normal);
    const lane_snapshot fast   = policy.snapshot(lane::fast);
    require(low.actual_gpu_us == counts[0] * 1000, "low charged in actual GPU microseconds");
    require(normal.actual_gpu_us == counts[1] * 1000, "normal actual GPU charge");
    require(fast.actual_gpu_us == counts[2] * 1000, "fast actual GPU charge");

    scheduler actual_cost;
    actual_cost.admit(make_request(1, lane::low), feasible_quote());
    actual_cost.admit(make_request(2, lane::fast), feasible_quote());
    service_decision decision = actual_cost.select_next(0, [](const request &) { return feasible_candidate(); });
    require(decision.selected && decision.selected_lane == lane::fast, "fast wins initial tie");
    actual_cost.complete_service(decision.decision_id, 8000, service_disposition::requeue);
    decision = actual_cost.select_next(8000, [](const request &) { return feasible_candidate(); });
    require(decision.selected && decision.selected_lane == lane::low,
            "large measured fast epoch creates debt independent of token count");
}

void test_empty_lane_work_conservation() {
    scheduler policy;
    require(policy.admit(make_request(1, lane::low), feasible_quote()).accepted, "admit only low");
    const service_decision decision = policy.select_next(0, [](const request &) { return feasible_candidate(); });
    require(decision.selected && decision.selected_lane == lane::low, "empty higher lanes reserve no service");
    require(decision.lane_reason == reason_code::lane_work_conserving, "work-conserving reason");
}

void test_admission_and_queue_reason_codes() {
    scheduler policy;
    require(policy.admit(make_request(0, lane::low), feasible_quote()).reason == reason_code::reject_invalid_request,
            "invalid ID");

    request cached              = make_request(1, lane::low);
    cached.cached_prompt_tokens = 2;
    require(policy.admit(cached, feasible_quote()).reason == reason_code::reject_invalid_request, "invalid cache size");

    request too_long = make_request(2, lane::normal, 0, 0, 1048576, 1);
    require(policy.admit(too_long, feasible_quote()).reason == reason_code::reject_context_limit, "context limit");

    feasibility_quote impossible = feasible_quote();
    impossible.state             = feasibility::impossible;
    require(policy.admit(make_request(3, lane::normal), impossible).reason == reason_code::reject_capacity_impossible,
            "impossible capacity");

    feasibility_quote waiting       = feasible_quote();
    waiting.state                   = feasibility::temporarily_blocked;
    const admission_result deferred = policy.admit(make_request(4, lane::normal), waiting);
    require(deferred.accepted && !deferred.ready && deferred.reason == reason_code::admission_capacity_wait,
            "temporary capacity wait");
    require(policy.admit(make_request(4, lane::normal), feasible_quote()).reason == reason_code::reject_duplicate_id,
            "duplicate ID");

    scheduler_config cap_config;
    for (lane_descriptor & descriptor : cap_config.lanes) {
        descriptor.queue_cap = 1;
    }
    scheduler capped(cap_config);
    uint64_t  next_id = 100;
    for (lane priority : { lane::low, lane::normal, lane::fast }) {
        require(
            capped.admit(make_request(next_id++, priority), feasible_quote()).reason == reason_code::admission_ready,
            "fill per-lane queue");
        require(
            capped.admit(make_request(next_id++, priority), feasible_quote()).reason == reason_code::reject_queue_full,
            "per-lane queue cap");
    }

    scheduler blocked;
    blocked.admit(make_request(1, lane::low), waiting);
    const service_decision no_service = blocked.select_next(0, {});
    require(!no_service.selected && no_service.reason == reason_code::wait_no_feasible_request,
            "blocked queue wait reason");

    scheduler busy;
    busy.admit(make_request(1, lane::low), feasible_quote());
    const service_decision first = busy.select_next(0, [](const request &) { return feasible_candidate(); });
    require(first.selected, "busy first selection");
    const service_decision second = busy.select_next(0, [](const request &) { return feasible_candidate(); });
    require(!second.selected && second.reason == reason_code::wait_service_in_flight, "in-flight reason");
    require(busy.complete_service(first.decision_id + 1, 1000, service_disposition::complete).reason ==
                reason_code::service_invalid_decision,
            "invalid completion decision reason");
}

void test_decode_width_profiles() {
    scheduler policy;
    require(policy.choose_decode_width(lane::fast, 100, 100, false).width == 2, "fast width cap");
    require(policy.choose_decode_width(lane::normal, 100, 100, false).width == 8, "normal width cap");
    require(policy.choose_decode_width(lane::fast, 1, 20, false).width == 1, "unsafe lower-lane fill stays disabled");
    require(policy.choose_decode_width(lane::fast, 1, 20, true).width == 2,
            "profiled-safe cross-lane fill uses fast cap");
    require(policy.choose_decode_width(lane::low, 3, 3, false).width == 2, "low profiled width 2");
    require(policy.choose_decode_width(lane::low, 7, 7, false).width == 4, "low profiled width 4");

    for (size_t count = 9; count <= 20; ++count) {
        const decode_width_decision decision = policy.choose_decode_width(lane::low, count, count, false);
        require(decision.width == 8, "low width must avoid measured 9-20 valley");
        require(decision.reason == reason_code::width_avoid_low_valley, "valley reason");
    }
    require(policy.choose_decode_width(lane::low, 23, 23, false).width == 8, "low 23 shape");
    require(policy.choose_decode_width(lane::low, 24, 24, false).width == 24, "low 24 shape");
    require(policy.choose_decode_width(lane::low, 31, 31, false).width == 24, "low 31 shape");
    require(policy.choose_decode_width(lane::low, 32, 32, false).width == 32, "low 32 shape");
    require(policy.choose_decode_width(lane::low, 63, 63, false).width == 32, "low 63 shape");
    require(policy.choose_decode_width(lane::low, 64, 64, false).width == 64, "low 64 shape");
    require(policy.choose_decode_width(lane::low, 0, 0, false).reason == reason_code::width_empty, "empty width");
}

trace_job make_trace_job(uint64_t id,
                         lane     priority,
                         uint64_t arrival_us,
                         uint64_t prompt_tokens,
                         uint64_t demand,
                         uint64_t observed_output_tokens = 1,
                         uint64_t gpu_us                 = 1000) {
    trace_job job;
    job.req                    = make_request(id, priority, arrival_us, 0, prompt_tokens,
                           observed_output_tokens == 0 ? 0 : observed_output_tokens);
    job.page_demand.units[0]   = demand;
    job.observed_output_tokens = observed_output_tokens;
    job.service_gpu_us         = gpu_us;
    return job;
}

size_t count_events(const replay_result & result, replay_event_kind kind, reason_code reason) {
    return static_cast<size_t>(
        std::count_if(result.events.begin(), result.events.end(),
                      [&](const replay_event & event) { return event.kind == kind && event.reason == reason; }));
}

const replay_event & find_event(const replay_result & result,
                                replay_event_kind     kind,
                                reason_code           reason,
                                uint64_t              request_id) {
    const auto found = std::find_if(result.events.begin(), result.events.end(), [&](const replay_event & event) {
        return event.kind == kind && event.reason == reason && event.request_id == request_id;
    });
    if (found == result.events.end()) {
        throw std::runtime_error("expected replay event was not emitted");
    }
    return *found;
}

void require_monotonic_event_time(const replay_result & result) {
    for (size_t i = 1; i < result.events.size(); ++i) {
        require(result.events[i - 1].time_us <= result.events[i].time_us,
                "replay events must be emitted in timestamp order");
    }
}

void test_frozen_capacity_traces() {
    simulator replay;

    replay_trace eight_by_20k;
    eight_by_20k.capacity.units[0] = 1000000;
    eight_by_20k.max_dispatches    = 0;
    for (uint64_t id = 1; id <= 8; ++id) {
        eight_by_20k.jobs.push_back(make_trace_job(id, lane::normal, 0, 20000, 20000));
    }
    replay_result result = replay.replay(eight_by_20k);
    require(count_events(result, replay_event_kind::admission, reason_code::admission_ready) == 8,
            "8x20K must all be immediately admissible");
    require(result.resident.units[0] == 160000, "8x20K resident demand");

    replay_trace capacity_limited;
    capacity_limited.capacity.units[0] = 1000000;
    capacity_limited.max_dispatches    = 0;
    capacity_limited.jobs.push_back(make_trace_job(1, lane::fast, 0, 500000, 500000));
    for (uint64_t id = 2; id <= 7; ++id) {
        capacity_limited.jobs.push_back(make_trace_job(id, lane::normal, 0, 90000, 90000));
    }
    result = replay.replay(capacity_limited);
    require(count_events(result, replay_event_kind::admission, reason_code::admission_ready) == 6,
            "500K plus five 90K requests fit");
    require(count_events(result, replay_event_kind::admission, reason_code::admission_capacity_wait) == 1,
            "sixth 90K request waits for capacity");
    require(result.resident.units[0] == 950000, "capacity-limited resident demand");

    replay_trace serialized;
    serialized.capacity.units[0] = 1000000;
    serialized.max_dispatches    = 0;
    serialized.jobs.push_back(make_trace_job(1, lane::low, 0, 900000, 900000));
    serialized.jobs.push_back(make_trace_job(2, lane::fast, 0, 200000, 200000));
    result = replay.replay(serialized);
    require(count_events(result, replay_event_kind::admission, reason_code::admission_ready) == 1,
            "900K request fits alone");
    require(count_events(result, replay_event_kind::admission, reason_code::admission_capacity_wait) == 1,
            "200K request must serialize");
    require(result.resident.units[0] == 900000, "serialized resident demand");
}

replay_trace perpetual_low_trace() {
    replay_trace trace;
    trace.capacity.units[0]   = 1000;
    trace.capacity.units[1]   = 100;
    trace.max_dispatches      = 300;
    trace.decode_lease_tokens = 1;
    trace.jobs.push_back(make_trace_job(1, lane::low, 0, 1, 1, 0));
    trace.jobs.push_back(make_trace_job(2, lane::low, 0, 1, 1, 0));
    trace_job normal                   = make_trace_job(10, lane::normal, 5000, 200, 1, 20);
    normal.req.cached_prompt_tokens    = 128;
    normal.req.requested_output_tokens = 100;
    normal.page_demand.units[1]        = 2;
    normal.cached_prefix_us            = 2500;
    normal.restore_cost_us             = 500;
    trace.jobs.push_back(normal);
    trace.jobs.push_back(make_trace_job(20, lane::fast, 7000, 1, 1, 20));
    trace.jobs.push_back(make_trace_job(21, lane::fast, 17000, 1, 1, 20));
    return trace;
}

void test_replay_determinism_and_perpetual_low_progress() {
    simulator           replay;
    const replay_trace  trace  = perpetual_low_trace();
    const replay_result first  = replay.replay(trace);
    const replay_result second = replay.replay(trace);
    require(first == second, "same frozen trace must replay byte-for-byte deterministically");

    std::array<size_t, lane_count> dispatches = {};
    for (const replay_event & event : first.events) {
        require(event.reason != reason_code::none, "every replay event must carry a primary reason code");
        if (event.kind == replay_event_kind::dispatch) {
            ++dispatches[static_cast<size_t>(event.priority)];
        }
    }
    require(dispatches[static_cast<size_t>(lane::low)] > 0, "perpetual low jobs must progress");
    require(dispatches[static_cast<size_t>(lane::normal)] == 20,
            "observed generation ends normal request before requested maximum");
    require(dispatches[static_cast<size_t>(lane::fast)] == 40, "both finite fast arrivals must complete");
    require(first.dispatches == trace.max_dispatches, "infinite trace stops only at deterministic limit");
}

void test_replay_is_explicitly_single_dispatch() {
    replay_trace trace;
    trace.capacity.units[0] = 10;
    trace.jobs.push_back(make_trace_job(1, lane::low, 0, 1, 1, 1, 700));
    trace.jobs.push_back(make_trace_job(2, lane::fast, 0, 1, 1, 1, 300));

    const replay_result result      = simulator().replay(trace);
    bool                in_flight   = false;
    size_t              dispatches  = 0;
    size_t              completions = 0;
    for (const replay_event & event : result.events) {
        if (event.kind == replay_event_kind::dispatch) {
            require(!in_flight, "replay cannot start a cohort or overlapping dispatch");
            in_flight = true;
            ++dispatches;
        } else if (event.kind == replay_event_kind::completion) {
            require(in_flight, "completion must correspond to the sole active dispatch");
            in_flight = false;
            ++completions;
        }
    }
    require(!in_flight && dispatches == 2 && completions == 2, "both jobs execute as individual dispatches");
    require(result.end_time_us == 1000, "single-dispatch replay advances by GPU service time only");
}

void test_async_restore_does_not_block_ready_gpu_work() {
    replay_trace trace;
    trace.capacity.units[0] = 2;

    trace_job restoring     = make_trace_job(1, lane::low, 0, 1, 1, 1, 500);
    restoring.restore_io_us = 5000;
    trace.jobs.push_back(restoring);
    trace.jobs.push_back(make_trace_job(2, lane::fast, 0, 1, 1, 1, 1000));

    const replay_result first  = simulator().replay(trace);
    const replay_result second = simulator().replay(trace);
    require(first == second, "asynchronous restore replay must remain byte-for-byte deterministic");
    require_monotonic_event_time(first);

    const replay_event & restore_start =
        find_event(first, replay_event_kind::io_start, reason_code::replay_restore_start, 1);
    const replay_event & fast_done = find_event(first, replay_event_kind::completion, reason_code::service_complete, 2);
    const replay_event & restore_ready =
        find_event(first, replay_event_kind::io_complete, reason_code::replay_restore_ready, 1);
    require(restore_start.time_us == 0 && restore_start.io_us == 5000, "restore start duration");
    require(fast_done.time_us == 1000, "ready GPU work must complete while restore is outstanding");
    require(restore_ready.time_us == 5000 && restore_ready.io_us == 5000, "restore readiness timestamp");
}

void test_arrivals_are_recorded_during_non_preemptive_dispatch() {
    replay_trace trace;
    trace.capacity.units[0] = 2;
    trace.jobs.push_back(make_trace_job(1, lane::fast, 0, 1, 1, 1, 5000));
    trace.jobs.push_back(make_trace_job(2, lane::low, 1000, 1, 1, 1, 250));

    const replay_result result = simulator().replay(trace);
    require_monotonic_event_time(result);
    const replay_event & arrival = find_event(result, replay_event_kind::arrival, reason_code::replay_arrival, 2);
    const replay_event & first_done =
        find_event(result, replay_event_kind::completion, reason_code::service_complete, 1);
    const replay_event & second_dispatch =
        find_event(result, replay_event_kind::dispatch, reason_code::request_fifo, 2);
    require(arrival.time_us == 1000, "arrival retains its real timestamp while GPU is occupied");
    require(first_done.time_us == 5000, "non-preemptive dispatch completion timestamp");
    require(second_dispatch.time_us == 5000, "arrival queues but cannot preempt the active dispatch");
}

void test_async_spill_holds_capacity_without_blocking_gpu() {
    replay_trace trace;
    trace.capacity.units[0] = 1;

    trace_job spilling   = make_trace_job(1, lane::fast, 0, 1, 1, 1, 1000);
    spilling.spill_io_us = 5000;
    trace.jobs.push_back(spilling);
    trace.jobs.push_back(make_trace_job(2, lane::normal, 0, 1, 0, 1, 1000));
    trace.jobs.push_back(make_trace_job(3, lane::fast, 1500, 1, 1, 1, 250));

    const replay_result result = simulator().replay(trace);
    require_monotonic_event_time(result);
    const replay_event & spill_start =
        find_event(result, replay_event_kind::io_start, reason_code::replay_spill_start, 1);
    const replay_event & unrelated_done =
        find_event(result, replay_event_kind::completion, reason_code::service_complete, 2);
    const replay_event & blocked =
        find_event(result, replay_event_kind::admission, reason_code::admission_capacity_wait, 3);
    const replay_event & spill_done =
        find_event(result, replay_event_kind::io_complete, reason_code::replay_spill_done, 1);
    require(spill_start.time_us == 1000 && spill_start.io_us == 5000, "spill starts at service completion");
    require(unrelated_done.time_us == 2000, "zero-demand GPU work proceeds during spill I/O");
    require(blocked.time_us == 1500, "spill-held pages block a capacity-consuming arrival");
    require(spill_done.time_us == 6000 && spill_done.io_us == 5000, "spill releases pages at I/O completion");
    const replay_event & unblocked_dispatch =
        find_event(result, replay_event_kind::dispatch, reason_code::request_fifo, 3);
    require(unblocked_dispatch.time_us == 6000, "capacity waiter dispatches immediately after spill release");
}

void test_zero_duration_capacity_waiters_preserve_lane_policy() {
    replay_trace trace;
    trace.capacity.units[0] = 1;
    trace.jobs.push_back(make_trace_job(1, lane::normal, 0, 1, 1, 1, 1000));
    trace.jobs.push_back(make_trace_job(2, lane::low, 0, 1, 1, 1, 250));
    trace.jobs.push_back(make_trace_job(3, lane::fast, 0, 1, 1, 1, 250));

    const replay_result result = simulator().replay(trace);
    require(count_events(result, replay_event_kind::admission, reason_code::admission_capacity_wait) == 2,
            "both later arrivals initially wait on the holder");
    const replay_event & fast_dispatch = find_event(result, replay_event_kind::dispatch, reason_code::request_fifo, 3);
    const replay_event & low_dispatch  = find_event(result, replay_event_kind::dispatch, reason_code::request_fifo, 2);
    require(fast_dispatch.time_us == 1000, "freed capacity follows scheduler lane priority");
    require(low_dispatch.time_us == 1250, "lower lane runs after the higher-priority waiter");
}

void test_resource_vectors_and_safety_watermark() {
    simulator    replay;
    replay_trace trace;
    trace.capacity.units[0]         = 100;
    trace.capacity.units[1]         = 50;
    trace.safety_watermark.units[0] = 10;
    trace.safety_watermark.units[1] = 5;
    trace.max_dispatches            = 0;

    trace_job ready            = make_trace_job(1, lane::low, 0, 1, 80);
    ready.page_demand.units[1] = 45;
    trace.jobs.push_back(ready);
    trace_job impossible            = make_trace_job(2, lane::low, 0, 1, 1);
    impossible.page_demand.units[1] = 46;
    trace.jobs.push_back(impossible);

    const replay_result result = replay.replay(trace);
    require(count_events(result, replay_event_kind::admission, reason_code::admission_ready) == 1,
            "multi-pool quote within watermark");
    require(count_events(result, replay_event_kind::admission, reason_code::reject_capacity_impossible) == 1,
            "limiting pool beyond watermark rejects");
}

void test_kv_physical_pressure_policy() {
    kv_pressure_controller pressure(2);

    const auto retry_1 = pressure.on_pressure(8, 7, 3);
    require(retry_1.action == kv_pressure_action::retry, "first pressure must retry");
    require(retry_1.reason == reason_code::kv_physical_pressure_retry, "retry reason");
    require(retry_1.attempt == 1 && retry_1.next_batch_size == 4, "first bounded retry shape");
    require(retry_1.victim_sequence == -1 && retry_1.victim_span_tokens == 0, "retry has no victim");

    const auto retry_2 = pressure.on_pressure(4, 7, 3);
    require(retry_2.action == kv_pressure_action::retry, "second pressure must retry");
    require(retry_2.attempt == 2 && retry_2.next_batch_size == 2, "second bounded retry shape");

    const auto victim = pressure.on_pressure(2, 7, 3);
    require(victim.action == kv_pressure_action::victim, "retry bound must select a victim");
    require(victim.reason == reason_code::kv_physical_pressure_victim, "victim reason");
    require(victim.attempt == 3 && victim.victim_sequence == 7 && victim.victim_span_tokens == 3,
            "head sequence and full contiguous span are deterministic");
    require(std::string(to_string(retry_1.reason)) == "kv_physical_pressure_retry", "stable retry notification");
    require(std::string(to_string(victim.reason)) == "kv_physical_pressure_victim", "stable victim notification");

    std::vector<int32_t> cohort = { 7, 7, 7, 11, 12 };
    cohort.erase(cohort.begin(), cohort.begin() + victim.victim_span_tokens);
    require(cohort == std::vector<int32_t>({ 11, 12 }), "victim handling must retain the rest of the cohort");

    const kv_pressure_snapshot snapshot = pressure.snapshot();
    require(snapshot == kv_pressure_snapshot({ 3, 2, 1 }), "low-cardinality pressure counters");

    kv_pressure_controller replay(2);
    require(replay.on_pressure(8, 7, 3) == retry_1, "first retry event deterministic");
    require(replay.on_pressure(4, 7, 3) == retry_2, "second retry event deterministic");
    require(replay.on_pressure(2, 7, 3) == victim, "victim event deterministic");

    kv_pressure_controller reset(2);
    reset.on_pressure(1, 3, 1);
    reset.reset_attempts();
    require(reset.on_pressure(1, 3, 1).attempt == 1, "successful decode resets only the retry window");

    kv_pressure_controller no_retry(0);
    require(no_retry.on_pressure(1, 9, 1).action == kv_pressure_action::victim,
            "zero retry configuration is an immediate deterministic victim");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "default descriptors",              test_default_descriptors                                  },
        { "FIFO, virtual runtime, and ties",  test_fifo_virtual_runtime_and_ties                        },
        { "queued cancellation",              test_cancel_queued_request                               },
        { "cache lookahead and bypass bound", test_cache_lookahead_bonus_and_bypass_bound               },
        { "infeasible head",                  test_infeasible_head_is_not_bypassed                      },
        { "aging",                            test_aging                                                },
        { "HDRR shares",                      test_hdrr_shares_and_low_progress                         },
        { "work conservation",                test_empty_lane_work_conservation                         },
        { "admission reasons",                test_admission_and_queue_reason_codes                     },
        { "decode widths",                    test_decode_width_profiles                                },
        { "capacity traces",                  test_frozen_capacity_traces                               },
        { "deterministic mixed replay",       test_replay_determinism_and_perpetual_low_progress        },
        { "single-dispatch replay",           test_replay_is_explicitly_single_dispatch                 },
        { "asynchronous restore",             test_async_restore_does_not_block_ready_gpu_work          },
        { "arrival during dispatch",          test_arrivals_are_recorded_during_non_preemptive_dispatch },
        { "asynchronous spill",               test_async_spill_holds_capacity_without_blocking_gpu      },
        { "capacity-wait lane policy",        test_zero_duration_capacity_waiters_preserve_lane_policy  },
        { "resource vectors",                 test_resource_vectors_and_safety_watermark                },
        { "KV physical pressure",             test_kv_physical_pressure_policy                           },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-scheduler: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-scheduler: all %zu tests passed\n", tests.size());
    return 0;
}

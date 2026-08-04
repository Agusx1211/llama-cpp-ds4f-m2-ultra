#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-request-runtime.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace server_request_registry;
using namespace server_request_runtime;
using namespace server_scheduler;

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

request_metadata make_request(uint64_t id, lane priority, uint64_t arrival_us) {
    request_metadata request;
    request.id                                = id;
    request.lane                              = priority;
    request.arrival_us                        = arrival_us;
    request.virtual_runtime_us                = 10;
    request.debt_us                           = -3;
    request.counts.prompt_tokens              = 128;
    request.counts.cached_prompt_tokens       = 64;
    request.counts.requested_output_tokens    = 32;
    request.estimates.predicted_prefill_us    = 100;
    request.estimates.predicted_cache_restore_us = 25;
    request.estimates.predicted_decode_us     = 200;
    request.estimates.predicted_gpu_us        = 300;
    request.estimates.predicted_memory_bytes  = 4096;
    request.estimates.predicted_output_tokens = 16;
    return request;
}

request_snapshot find_request(const request_runtime & runtime, uint64_t id) {
    for (const auto & request : runtime.snapshot()) {
        if (request.handle.id == id) {
            return request;
        }
    }
    throw std::runtime_error("request snapshot not found");
}

void test_live_ingress_and_dispatch() {
    request_runtime runtime;
    require(runtime.admit(make_request(1, lane::low, 100)), "admit low");
    require(runtime.admit(make_request(2, lane::normal, 101)), "admit normal");
    require(runtime.admit(make_request(3, lane::fast, 102)), "admit fast");

    const request_snapshot fast = find_request(runtime, 3);
    require(fast.lane == trusted_lane::fast && fast.arrival_us == 102, "trusted ingress identity");
    require(fast.virtual_runtime_us == 10 && fast.debt_us == -3, "runtime and debt stamped");
    require(fast.counts.prompt_tokens == 128 && fast.counts.cached_prompt_tokens == 64, "token counts stamped");
    require(fast.estimates.predicted_gpu_us == 300 && fast.estimates.predicted_cache_restore_us == 25 &&
                fast.last_reason == server_request_registry::reason_code::admission_ready,
            "estimates and reason stamped");

    const dispatch_result first = runtime.take_next(110);
    require(first.selected && first.request_id == 3 && first.lane == lane::fast, "trusted fast lane wins initial tie");
    const request_snapshot dispatched = find_request(runtime, 3);
    require(dispatched.state == lifecycle::registered && dispatched.queue == queue_state::none,
            "dispatch detaches request from queue state");
    require(dispatched.virtual_runtime_us == 310, "dispatch charges deterministic predicted service");

    require(runtime.bind_slot(3, 7, 111), "bind execution slot");
    const request_snapshot executing = find_request(runtime, 3);
    require(executing.state == lifecycle::executing && executing.bindings.size() == 1,
            "durable request binds temporary slot");
    require(runtime.release_slot(3, 7, 112), "release completed slot");
    require(!runtime.contains(3), "completed request metadata retired");

    const auto events = runtime.events();
    require(events.total_events >= 8 && events.events.back().kind == event_kind::removed,
            "runtime publishes complete bounded reason history");
}

void test_queue_cap_and_queued_cancellation() {
    runtime_config config;
    config.scheduler.lanes[static_cast<size_t>(lane::fast)].queue_cap = 1;
    config.overload_retry_after_seconds                               = 77;
    request_runtime runtime(config);

    require(runtime.admit(make_request(1, lane::fast, 10)), "fill fast queue");
    const server_request_runtime::admission_result rejected = runtime.admit(make_request(2, lane::fast, 11));
    require(!rejected && rejected.code == server_request_runtime::result_code::queue_full &&
                rejected.reason == server_scheduler::reason_code::reject_queue_full &&
                rejected.retry_after_seconds == 60,
            "lane queue cap rejects deterministically");
    require(runtime.cancel(1, 12), "cancel queued request");
    require(runtime.queued_total() == 0 && runtime.summary().active_requests == 0,
            "queued cancellation releases policy and registry capacity");
    require(runtime.admit(make_request(2, lane::fast, 13)), "capacity reusable after cancellation");
}

void test_defer_resume_and_cancel() {
    request_runtime runtime;
    require(runtime.admit(make_request(11, lane::normal, 20)), "admit deferred request");
    require(runtime.take_next(21).request_id == 11, "dispatch deferred request");
    require(runtime.mark_deferred(11, 22), "mark capacity block");

    const request_snapshot blocked = find_request(runtime, 11);
    require(blocked.state == lifecycle::queued && blocked.queue == queue_state::blocked &&
                blocked.last_reason == server_request_registry::reason_code::capacity_blocked,
            "deferred request remains durable and blocked");
    require(runtime.resume(11, 30), "resume blocked request");
    const request_snapshot resumed = find_request(runtime, 11);
    require(resumed.queue == queue_state::ready && resumed.arrival_us == 20,
            "resume preserves original arrival and becomes ready");
    require(runtime.cancel(11, 31), "cancel resumed request");
    require(!runtime.contains(11), "cancelled ready request retired");
}

void test_cancel_while_bound() {
    request_runtime runtime;
    require(runtime.admit(make_request(21, lane::low, 40)), "admit bound cancellation");
    require(runtime.take_next(41).request_id == 21, "dispatch bound cancellation");
    require(runtime.bind_slot(21, 3, 42), "bind cancellable request");
    require(runtime.cancel(21, 43), "record cancellation while executing");

    const request_snapshot cancelled = find_request(runtime, 21);
    require(cancelled.cancel_requested && cancelled.state == lifecycle::executing,
            "cancellation survives until slot release");
    require(runtime.release_slot(21, 3, 44), "release cancelled slot");
    require(!runtime.contains(21) && runtime.summary().occupied_slots == 0,
            "bound cancellation retires only after lease release");
}

void test_passive_child_binding() {
    request_runtime runtime;
    require(runtime.admit(make_request(30, lane::normal, 50)), "register child owner");
    request_metadata child = make_request(31, lane::normal, 50);
    child.parent_id        = 30;
    require(runtime.admit(child, false), "register passive child");
    require(runtime.queued_total() == 1 && find_request(runtime, 31).queue == queue_state::ready,
            "child is durable without a second scheduler admission");
    require(!runtime.take_next(50, 1).selected && runtime.permits().total == 0,
            "parent and child never acquire a partial physical permit group");
    require(runtime.take_next(50, 2).request_id == 30 && runtime.permits().total == 2,
            "parent selection transactionally claims both family permits");
    require(runtime.bind_slot(31, 4, 51), "bind passive child");
    require(runtime.bind_slot(30, 3, 51), "bind child owner");
    require(runtime.release_slot(31, 4, 52), "complete passive child");
    require(runtime.release_slot(30, 3, 52), "complete child owner");
}

void test_dispatch_permits_cover_selection_defer_and_caps() {
    request_runtime fast;
    for (uint64_t id = 1; id <= 3; ++id) {
        require(fast.admit(make_request(id, lane::fast, id)), "admit fast permit request");
    }
    require(fast.take_next(10).request_id == 1, "select first fast permit");
    require(fast.take_next(11).request_id == 2, "select second fast permit");
    auto permits = fast.permits();
    require(permits.total == 2 && permits.claimed[static_cast<size_t>(lane::fast)] == 2 &&
                permits.bound[static_cast<size_t>(lane::fast)] == 0,
            "selected-unbound work consumes the exact fast ceiling");
    require(!fast.take_next(12).selected, "third fast request cannot pass selected-unbound permits");

    require(fast.bind_slot(1, 0, 13), "first fast permit converts on bind");
    require(fast.mark_deferred(2, 14) && fast.resume(2, 15), "second fast permit survives defer and resume");
    require(fast.take_next(16).request_id == 2 && fast.bind_slot(2, 1, 17),
            "resumed request reuses its original permit");
    permits = fast.permits();
    require(permits.total == 2 && permits.bound[static_cast<size_t>(lane::fast)] == 2,
            "bind converts permits without double accounting");
    require(fast.release_slot(1, 0, 18) && !fast.take_next(19).selected,
            "closed fast cohort cannot refill after one staggered release");
    require(fast.release_slot(2, 1, 20) && fast.take_next(21).request_id == 3,
            "fast refill waits for the prior cohort to drain completely");
    require(fast.fail(3, 22) && fast.permits().total == 0,
            "deferred and selected permit lifecycle retires without leaks");

    for (lane priority : { lane::normal, lane::low }) {
        request_runtime capped;
        const size_t    cap = priority == lane::normal ? 8 : 64;
        for (size_t i = 0; i <= cap; ++i) {
            require(capped.admit(make_request(i + 1, priority, i + 1)), "admit lane-cap request");
        }
        std::vector<std::pair<uint64_t, slot_id>> bound;
        for (size_t i = 0; i < cap; ++i) {
            const dispatch_result selected = capped.take_next(100 + i, 64);
            require(selected.selected && capped.bind_slot(selected.request_id, static_cast<slot_id>(i), 200 + i),
                    "claim and bind lane-cap permit");
            bound.emplace_back(selected.request_id, static_cast<slot_id>(i));
        }
        require(!capped.take_next(1000, 64).selected, "live lane ceiling blocks one excess request");
        permits = capped.permits();
        require(permits.claimed[static_cast<size_t>(priority)] == cap && permits.total == cap,
                "live lane ceiling is exact");
        require(capped.release_slot(bound[0].first, bound[0].second, 1001), "release first lane-cap request");
        require(!capped.take_next(1002, 64).selected, "closed cohort rejects staggered same-lane refill");
        for (size_t i = 1; i < bound.size(); ++i) {
            require(capped.release_slot(bound[i].first, bound[i].second, 1002 + i),
                    "drain remaining lane-cap request");
        }
        require(capped.take_next(2000, 64).selected, "same-lane refill resumes after complete cohort drain");
    }

    request_runtime physical;
    for (uint64_t id = 1; id <= 3; ++id) {
        require(physical.admit(make_request(id, lane::normal, id)), "admit physical-cap request");
    }
    require(physical.take_next(1, 2).selected && physical.take_next(2, 2).selected &&
                !physical.take_next(3, 2).selected && physical.permits().total == 2,
            "physical permits are claimed at selection, before binding");
}

void test_bounded_same_fast_cohort_refill() {
    runtime_config config;
    config.fast_refill_max_members_per_epoch = 4;
    config.fast_refill_window_us              = 1000;
    request_runtime runtime(config);

    require(runtime.admit(make_request(1, lane::low, 1)), "admit long low refill anchor");
    require(runtime.take_next(10, 2).request_id == 1 && runtime.bind_slot(1, 0, 11),
            "bind long low refill anchor");
    for (uint64_t id = 2; id <= 6; ++id) {
        require(runtime.admit(make_request(id, lane::fast, id)), "admit ordered fast refill member");
    }

    for (uint64_t id = 2; id <= 5; ++id) {
        const uint64_t        at_us    = 20 + id;
        const dispatch_result selected = runtime.take_next(at_us, 2);
        require(selected.selected && selected.request_id == id && selected.lane == lane::fast,
                "bounded refill preserves fast FIFO order");
        require(runtime.bind_slot(id, 1, at_us + 1) && runtime.permits().total == 2,
                "refilled fast work preserves the total width-two ceiling");
        require(runtime.release_slot(id, 1, at_us + 2) && runtime.permits().total == 1,
                "sequential fast completion returns only its reusable permit");
    }
    require(!runtime.take_next(100, 2).selected && runtime.contains(1) && runtime.contains(6),
            "fifth fast member cannot exceed the four-member epoch budget");
    require(runtime.cancel(6, 101) && runtime.release_slot(1, 0, 102) && runtime.permits().total == 0 &&
                runtime.summary().active_requests == 0,
            "exhausted refill epoch cancels and drains without permit leaks");

    request_runtime ordinary(config);
    for (uint64_t id = 11; id <= 13; ++id) {
        require(ordinary.admit(make_request(id, lane::normal, id)), "admit ordinary non-refill request");
    }
    require(ordinary.take_next(110, 2).request_id == 11 && ordinary.bind_slot(11, 0, 111) &&
                ordinary.take_next(112, 2).request_id == 12 && ordinary.bind_slot(12, 1, 113) &&
                ordinary.release_slot(11, 0, 114) && !ordinary.take_next(115, 2).selected,
            "fast refill opt-in does not reopen an ordinary closed cohort");
    require(ordinary.release_slot(12, 1, 116) && ordinary.cancel(13, 117) &&
                ordinary.permits().total == 0 && ordinary.summary().active_requests == 0,
            "ordinary closed cohort drains without leaks");
}

void test_fast_refill_window_and_terminal_boundaries() {
    runtime_config config;
    config.fast_refill_max_members_per_epoch = 6;
    config.fast_refill_window_us              = 1000;
    request_runtime runtime(config);

    require(runtime.admit(make_request(1, lane::low, 1)) && runtime.take_next(2, 2).request_id == 1 &&
                runtime.bind_slot(1, 0, 3),
            "bind refill-window low anchor");
    for (uint64_t id = 2; id <= 4; ++id) {
        require(runtime.admit(make_request(id, lane::fast, id)), "admit refill-window fast request");
    }
    require(runtime.take_next(10, 2).request_id == 2 && runtime.bind_slot(2, 1, 11) &&
                runtime.release_slot(2, 1, 12),
            "first fast member starts the refill window");
    require(runtime.take_next(1009, 2).request_id == 3 && runtime.bind_slot(3, 1, 1009) &&
                runtime.release_slot(3, 1, 1009),
            "same-fast refill remains eligible immediately before its wall-time limit");
    require(!runtime.take_next(1010, 2).selected,
            "same-fast refill closes at the exact wall-time limit");
    require(runtime.cancel(4, 1011) && runtime.release_slot(1, 0, 1012) && runtime.permits().total == 0,
            "wall-time exhaustion retires without permit leaks");

    runtime_config open_config;
    open_config.scheduler.lanes[static_cast<size_t>(lane::fast)].decode_cap = 8;
    open_config.fast_refill_max_members_per_epoch                           = 6;
    open_config.fast_refill_window_us                                      = 1000;
    request_runtime open(open_config);
    for (uint64_t id = 21; id <= 23; ++id) {
        require(open.admit(make_request(id, lane::fast, id)), "admit open-window fast member");
    }
    require(open.take_next(10, 8).request_id == 21 && open.take_next(1009, 8).request_id == 22 &&
                !open.take_next(1010, 8).selected && open.permits().total == 2,
            "open initial fast fill also closes at the exact refill-window boundary");
    for (uint64_t id = 21; id <= 23; ++id) {
        require(open.cancel(id, 1011), "clean open-window fast member");
    }
    require(open.permits().total == 0 && open.summary().active_requests == 0,
            "open-window boundary fixture leaves no permits");

    runtime_config terminal_config;
    terminal_config.default_queue_timeout_us          = 50;
    terminal_config.fast_refill_max_members_per_epoch = 4;
    terminal_config.fast_refill_window_us              = 1000;
    request_runtime terminal(terminal_config);
    require(terminal.admit(make_request(11, lane::low, 100)) && terminal.take_next(100, 2).request_id == 11 &&
                terminal.bind_slot(11, 0, 101),
            "bind terminal-boundary low anchor");
    require(terminal.admit(make_request(12, lane::fast, 101)) &&
                terminal.admit(make_request(13, lane::fast, 101)) &&
                terminal.admit(make_request(14, lane::fast, 104)) &&
                terminal.admit(make_request(15, lane::fast, 150)),
            "admit terminal-boundary fast requests");
    require(terminal.take_next(102, 2).request_id == 12 && terminal.bind_slot(12, 1, 103),
            "bind cancellable first fast member");
    require(terminal.cancel(12, 104) && terminal.permits().total == 2 &&
                terminal.release_slot(12, 1, 105) && terminal.permits().total == 1,
            "bound cancellation keeps then releases exactly one fast permit");
    require(terminal.take_next(106, 2).request_id == 13 && terminal.bind_slot(13, 1, 107) &&
                terminal.release_slot(13, 1, 108),
            "cancellation opens one bounded refill opportunity");
    const auto expired = terminal.expire_due(154);
    require(expired.size() == 1 && expired.front().request_id == 14 &&
                expired.front().kind == deadline_kind::queue,
            "queued refill candidate expires at its original-arrival deadline");
    require(terminal.take_next(155, 2).request_id == 15 && terminal.bind_slot(15, 1, 156),
            "timed-out candidate does not leak a permit or consume the epoch member budget");
    require(terminal.cancel(15, 157) && terminal.release_slot(15, 1, 158) &&
                terminal.release_slot(11, 0, 159) && terminal.permits().total == 0 &&
                terminal.summary().active_requests == 0,
            "timeout and cancellation boundaries leave no refill permits or requests");
}

void test_fast_refill_telemetry_is_authoritative_at_sample_time() {
    request_runtime disabled;
    const auto disabled_state  = disabled.fast_refill();
    const auto disabled_status = disabled_state.status_at(100, 0);
    require(!disabled_state.enabled && disabled_state.max_members_per_cohort == 0 &&
                disabled_state.window_us == 0 && !disabled_state.cohort_active &&
                disabled_state.dominant == lane::count && disabled_state.fast_members_used == 0 &&
                disabled_state.deadline_us == 0 && disabled_status.fast_members_remaining == 0 &&
                disabled_status.remaining_us == 0 && !disabled_status.window_open &&
                !disabled_status.one_member_eligible_now,
            "default-off refill telemetry is explicit and empty");

    runtime_config config;
    config.fast_refill_max_members_per_epoch = 4;
    config.fast_refill_window_us              = 1000;
    request_runtime active(config);
    require(active.admit(make_request(1, lane::low, 1)) && active.take_next(10, 2).request_id == 1 &&
                active.bind_slot(1, 0, 11) && active.admit(make_request(2, lane::fast, 12)) &&
                active.take_next(20, 2).request_id == 2 && active.bind_slot(2, 1, 21) &&
                active.release_slot(2, 1, 22),
            "build one reusable fast refill slot");
    const auto open        = active.fast_refill();
    const auto open_status = open.status_at(100, active.permits().total);
    require(open.enabled && open.max_members_per_cohort == 4 && open.window_us == 1000 &&
                open.cohort_active && !open.selection_open && open.dominant == lane::fast &&
                open.cohort_limit == 2 && open.fast_members_used == 1 && open.deadline_us == 1020 &&
                open_status.fast_members_remaining == 3 && open_status.remaining_us == 920 &&
                open_status.window_open && open_status.one_member_eligible_now,
            "sampled telemetry distinguishes a closed initial selection from an eligible refill window");
    const auto expired = open.status_at(1020, active.permits().total);
    require(open.deadline_us == 1020 && expired.remaining_us == 0 && !expired.window_open &&
                !expired.one_member_eligible_now,
            "exact refill deadline cannot report an open or eligible window");
    require(active.release_slot(1, 0, 1021), "drain active telemetry fixture");

    runtime_config quota_config;
    quota_config.scheduler.lanes[static_cast<size_t>(lane::fast)].decode_cap = 8;
    quota_config.fast_refill_max_members_per_epoch                           = 3;
    quota_config.fast_refill_window_us                                      = 1000;
    request_runtime quota(quota_config);
    for (uint64_t id = 11; id <= 14; ++id) {
        require(quota.admit(make_request(id, lane::fast, id)), "admit quota telemetry member");
    }
    require(quota.take_next(30, 8).request_id == 11 && quota.take_next(31, 8).request_id == 12 &&
                quota.take_next(32, 8).request_id == 13,
            "consume complete telemetry member quota");
    const auto exhausted        = quota.fast_refill();
    const auto exhausted_status = exhausted.status_at(33, quota.permits().total);
    require(exhausted.selection_open && exhausted.fast_members_used == 3 &&
                exhausted_status.fast_members_remaining == 0 && exhausted_status.remaining_us == 997 &&
                !exhausted_status.window_open && !exhausted_status.one_member_eligible_now,
            "raw initial selection state cannot overstate an exhausted refill member budget");
    for (uint64_t id = 11; id <= 14; ++id) {
        require(quota.cancel(id, 34), "clean quota telemetry member");
    }
}

void test_fast_member_budget_covers_initial_fill_and_upgrade() {
    runtime_config config;
    config.scheduler.lanes[static_cast<size_t>(lane::fast)].decode_cap = 8;
    config.fast_refill_max_members_per_epoch                           = 3;
    config.fast_refill_window_us                                      = 1000;

    request_runtime initial(config);
    for (uint64_t id = 1; id <= 4; ++id) {
        require(initial.admit(make_request(id, lane::fast, id)), "admit wide initial fast member");
    }
    require(initial.take_next(10, 8).request_id == 1 && initial.take_next(11, 8).request_id == 2 &&
                initial.take_next(12, 8).request_id == 3 && !initial.take_next(13, 8).selected &&
                initial.permits().total == 3,
            "configured member budget bounds the open initial fast fill below decode cap");
    for (uint64_t id = 1; id <= 4; ++id) {
        require(initial.cancel(id, 14), "clean bounded initial fast member");
    }
    require(initial.permits().total == 0 && initial.summary().active_requests == 0,
            "initial-fill budget fixture leaves no permits");

    request_runtime upgraded(config);
    require(upgraded.admit(make_request(11, lane::low, 20)) && upgraded.take_next(20, 8).request_id == 11 &&
                upgraded.bind_slot(11, 0, 21),
            "bind low member before wide fast upgrade");
    for (uint64_t id = 12; id <= 15; ++id) {
        require(upgraded.admit(make_request(id, lane::fast, id)), "admit wide fast upgrade member");
    }
    require(upgraded.take_next(30, 8).request_id == 12 && upgraded.take_next(31, 8).request_id == 13 &&
                upgraded.take_next(32, 8).request_id == 14 && !upgraded.take_next(33, 8).selected &&
                upgraded.permits().claimed[static_cast<size_t>(lane::fast)] == 3,
            "configured member budget bounds a low-to-fast upgrade below decode cap");
    for (uint64_t id = 12; id <= 15; ++id) {
        require(upgraded.cancel(id, 34), "clean bounded upgrade fast member");
    }
    require(upgraded.release_slot(11, 0, 35) && upgraded.permits().total == 0 &&
                upgraded.summary().active_requests == 0,
            "upgrade budget fixture leaves no permits");

    request_runtime families(config);
    std::array<size_t, lane_count> mixed = {};
    mixed[static_cast<size_t>(lane::normal)] = 1;
    mixed[static_cast<size_t>(lane::fast)]   = 1;
    const auto mixed_validation = families.validate_dispatch_family(lane::normal, mixed, 8);
    require(!mixed_validation && mixed_validation.code == server_request_runtime::result_code::invalid_request,
            "mixed-lane family is rejected before it can start a fast window");
    std::array<size_t, lane_count> oversized_fast = {};
    oversized_fast[static_cast<size_t>(lane::fast)] = 4;
    require(families.validate_dispatch_family(lane::fast, oversized_fast, 8).code ==
                server_request_runtime::result_code::capacity_impossible,
            "fast family wider than the total member budget is rejected");
    require(families.admit(make_request(21, lane::normal, 40)), "admit direct-runtime family parent");
    request_metadata mixed_child = make_request(22, lane::fast, 40);
    mixed_child.parent_id         = 21;
    require(families.admit(mixed_child, false).code == server_request_runtime::result_code::invalid_request &&
                families.cancel(21, 41) && families.summary().active_requests == 0,
            "direct-runtime mixed-lane child is rejected without durable residue");

    request_runtime sealed(config);
    require(sealed.admit(make_request(31, lane::fast, 50)) && sealed.take_next(51, 8).request_id == 31 &&
                sealed.mark_deferred(31, 52),
            "select and defer fast parent before late family admission");
    request_metadata late_child = make_request(32, lane::fast, 50);
    late_child.parent_id         = 31;
    require(sealed.admit(late_child, false).code == server_request_runtime::result_code::invalid_request &&
                !sealed.contains(32) && sealed.summary().active_requests == 1,
            "late same-fast child is rejected after parent family membership seals");
    require(sealed.resume(31, 53) && sealed.cancel(31, 54) && sealed.permits().total == 0 &&
                sealed.summary().active_requests == 0,
            "sealed late-child fixture resumes and cancels without residue");

    request_runtime disabled;
    const auto disabled_mixed = disabled.validate_dispatch_family(lane::normal, mixed, 2);
    require(disabled_mixed, "default-off runtime preserves prior mixed-family validation");
    require(disabled.admit(make_request(41, lane::normal, 60)), "admit default-off mixed parent");
    request_metadata default_mixed_child = make_request(42, lane::fast, 60);
    default_mixed_child.parent_id         = 41;
    require(disabled.admit(default_mixed_child, false) && disabled.take_next(61, 2).request_id == 41 &&
                disabled.permits().total == 2 && disabled.bind_slot(41, 0, 62) &&
                disabled.bind_slot(42, 1, 62) && disabled.release_slot(42, 1, 63) &&
                disabled.release_slot(41, 0, 63) && disabled.summary().active_requests == 0,
            "default-off runtime preserves dispatch and cleanup of an accepted mixed family");

    request_runtime default_late;
    require(default_late.admit(make_request(51, lane::fast, 70)) &&
                default_late.take_next(71, 2).request_id == 51 && default_late.mark_deferred(51, 72),
            "defer default-off parent before prior-compatible late child admission");
    request_metadata default_late_child = make_request(52, lane::fast, 70);
    default_late_child.parent_id         = 51;
    require(default_late.admit(default_late_child, false) && default_late.resume(51, 73) &&
                default_late.take_next(74, 2).request_id == 51 && default_late.bind_slot(51, 0, 75) &&
                default_late.bind_slot(52, 1, 75) && default_late.release_slot(52, 1, 76) &&
                default_late.release_slot(51, 0, 76) && default_late.summary().active_requests == 0,
            "default-off runtime preserves late same-lane family membership behavior");

    request_runtime epochs(config);
    require(epochs.admit(make_request(61, lane::low, 100)) && epochs.take_next(100, 8).request_id == 61 &&
                epochs.bind_slot(61, 0, 101),
            "bind family-budget low anchor");
    require(epochs.admit(make_request(62, lane::fast, 102)), "admit homogeneous fast family parent");
    request_metadata homogeneous_child = make_request(63, lane::fast, 102);
    homogeneous_child.parent_id         = 62;
    require(epochs.admit(homogeneous_child, false) && epochs.admit(make_request(64, lane::fast, 103)) &&
                epochs.admit(make_request(65, lane::fast, 104)),
            "admit homogeneous family and two independent fast members");
    require(epochs.take_next(110, 8).request_id == 62 && epochs.permits().total == 3 &&
                epochs.bind_slot(62, 1, 111) && epochs.bind_slot(63, 2, 111) &&
                epochs.release_slot(63, 2, 112) && epochs.release_slot(62, 1, 112),
            "homogeneous two-member family consumes two cumulative fast members");
    require(epochs.take_next(113, 8).request_id == 64 && epochs.fail(64, 114) &&
                !epochs.take_next(115, 8).selected,
            "only one independent fast member remains in the max-three epoch budget");
    require(epochs.cancel(65, 116) && epochs.release_slot(61, 0, 117) && epochs.permits().total == 0,
            "final anchored permit drain closes the family-budget epoch");

    for (uint64_t id = 71; id <= 74; ++id) {
        require(epochs.admit(make_request(id, lane::fast, 5000)), "admit fresh-epoch fast member");
    }
    require(epochs.take_next(5000, 8).request_id == 71 && epochs.take_next(5001, 8).request_id == 72 &&
                epochs.take_next(5002, 8).request_id == 73 && !epochs.take_next(5003, 8).selected &&
                epochs.permits().total == 3,
            "full permit drain resets the cumulative budget and starts a fresh refill window");
    for (uint64_t id = 71; id <= 74; ++id) {
        require(epochs.cancel(id, 5004), "clean fresh-epoch fast member");
    }
    require(epochs.permits().total == 0 && epochs.summary().active_requests == 0,
            "fresh epoch fixture drains without residue");
}

void test_fast_refill_reentry_and_unbound_terminal_budget() {
    runtime_config config;
    config.fast_refill_max_members_per_epoch = 3;
    config.fast_refill_window_us              = 1000;

    request_runtime reentry(config);
    require(reentry.admit(make_request(1, lane::low, 1)) && reentry.take_next(2, 2).request_id == 1 &&
                reentry.bind_slot(1, 0, 3),
            "bind reentry low anchor");
    require(reentry.admit(make_request(2, lane::fast, 2)) && reentry.admit(make_request(3, lane::fast, 3)) &&
                reentry.take_next(10, 2).request_id == 2 && reentry.mark_deferred(2, 11),
            "select and defer first fast epoch member");
    require(reentry.resume(2, 1010) && reentry.take_next(1010, 2).request_id == 2 &&
                reentry.bind_slot(2, 1, 1011) && reentry.release_slot(2, 1, 1012),
            "permit reentry survives the exact refill-window boundary without a new claim");
    require(!reentry.take_next(1013, 2).selected,
            "genuinely new fast work remains blocked after the reentry window expires");
    require(reentry.cancel(3, 1014) && reentry.release_slot(1, 0, 1015) && reentry.permits().total == 0 &&
                reentry.summary().active_requests == 0,
            "expired-window reentry fixture drains without leaks");

    request_runtime terminal(config);
    require(terminal.admit(make_request(11, lane::low, 100)) && terminal.take_next(100, 2).request_id == 11 &&
                terminal.bind_slot(11, 0, 100),
            "bind unbound-terminal low anchor");
    request_metadata expiring = make_request(12, lane::fast, 100);
    expiring.queue_timeout_us  = 5;
    require(terminal.admit(expiring), "admit expiring selected-unbound fast member");
    for (uint64_t id = 13; id <= 15; ++id) {
        request_metadata fast = make_request(id, lane::fast, 101);
        fast.queue_timeout_us  = std::numeric_limits<uint64_t>::max();
        require(terminal.admit(fast), "admit persistent unbound-terminal fast member");
    }
    require(terminal.take_next(101, 2).request_id == 12 && terminal.permits().total == 2,
            "claim expiring selected-unbound fast permit");
    const auto expired = terminal.expire_due(105);
    require(expired.size() == 1 && expired.front().request_id == 12 && !expired.front().was_running &&
                terminal.permits().total == 1,
            "selected-unbound expiry releases its live permit");
    require(terminal.take_next(106, 2).request_id == 13 && terminal.fail(13, 107) &&
                terminal.permits().total == 1 && terminal.take_next(108, 2).request_id == 14 &&
                terminal.fail(14, 109) && terminal.permits().total == 1,
            "selected-unbound failures release their live permits");
    require(!terminal.take_next(110, 2).selected,
            "expiry and failure do not refund the cumulative fast member budget");
    require(terminal.cancel(15, 111) && terminal.release_slot(11, 0, 112) && terminal.permits().total == 0 &&
                terminal.summary().active_requests == 0,
            "unbound-terminal budget fixture drains without leaks");
}

void test_profiled_exclusive_low_widths() {
    for (const auto & fixture : std::vector<std::pair<size_t, size_t>>({
             { 20, 8  },
             { 24, 24 }
    })) {
        request_runtime runtime;
        for (size_t i = 0; i < fixture.first; ++i) {
            require(runtime.admit(make_request(i + 1, lane::low, i + 1)), "admit low-width request");
        }
        for (size_t i = 0; i < fixture.second; ++i) {
            const dispatch_result selected = runtime.take_next(100 + i, 64);
            require(selected.selected && runtime.bind_slot(selected.request_id, static_cast<slot_id>(i), 200 + i),
                    "bind profiled low-width request");
        }
        require(!runtime.take_next(1000, 64).selected && runtime.permits().total == fixture.second,
                "exclusive low permits stop at the profiled shape");
    }
}

bool has_terminal_event(const request_runtime &              runtime,
                        uint64_t                             id,
                        lifecycle                            terminal,
                        server_request_registry::reason_code reason) {
    for (const auto & event : runtime.events().events) {
        if (event.request.id == id && event.kind == event_kind::terminal && event.lifecycle_after == terminal &&
            event.reason == reason) {
            return true;
        }
    }
    return false;
}

void test_queue_deadlines_and_exact_capacity_reuse() {
    runtime_config config;
    config.default_queue_timeout_us = 10;
    config.default_run_timeout_us   = 20;
    config.registry.max_requests    = 1;
    request_runtime runtime(config);

    const request_metadata ready = make_request(41, lane::normal, 100);
    require(ready.queue_timeout_us == 0 && runtime.admit(ready),
            "zero request timeout selects bounded server queue default");
    require(runtime.expire_due(109).empty(), "ready request lives before exact deadline");
    const auto ready_expired = runtime.expire_due(110);
    require(ready_expired.size() == 1 && ready_expired[0].request_id == 41 &&
                ready_expired[0].kind == deadline_kind::queue && !ready_expired[0].was_running,
            "ready request expires at exact queue deadline");
    require(!runtime.contains(41) && has_terminal_event(runtime, 41, lifecycle::timed_out,
                                                        server_request_registry::reason_code::queue_timeout),
            "ready timeout retires durable metadata with queue reason");

    require(runtime.admit(make_request(42, lane::normal, 110)), "reuse capacity immediately after ready timeout");
    require(runtime.take_next(111).request_id == 42, "dispatch request for blocked timeout");
    require(runtime.mark_deferred(42, 112), "mark request blocked before timeout");
    require(runtime.expire_due(119).empty(), "blocked request lives before deadline");
    require(runtime.expire_due(120).size() == 1 && !runtime.contains(42),
            "blocked request expires and releases capacity");

    require(runtime.admit(make_request(43, lane::normal, 120), false), "register passive child deadline");
    require(runtime.expire_due(130).size() == 1 && !runtime.contains(43),
            "passive child expires from durable ready state");

    require(runtime.admit(make_request(44, lane::normal, 130)), "admit setup-phase deadline request");
    require(runtime.take_next(131).request_id == 44, "dispatch setup-phase deadline request");
    require(runtime.expire_due(140).size() == 1 && !runtime.contains(44),
            "dispatched unbound request remains governed by queue deadline");
}

void test_run_deadline_and_first_terminal_wins() {
    runtime_config config;
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 20;
    request_runtime runtime(config);

    const request_metadata running = make_request(51, lane::fast, 200);
    require(running.run_timeout_us == 0 && runtime.admit(running),
            "zero request timeout selects bounded server run default");
    require(runtime.take_next(201).request_id == 51, "dispatch run deadline request");
    require(runtime.bind_slot(51, 0, 205), "bind starts run deadline");
    require(runtime.expire_due(224).empty(), "bound request lives before run deadline");
    const auto expired = runtime.expire_due(225);
    require(expired.size() == 1 && expired[0].request_id == 51 && expired[0].kind == deadline_kind::run &&
                expired[0].was_running,
            "bound request expires at exact run deadline");
    const request_snapshot timed_out = find_request(runtime, 51);
    require(timed_out.state == lifecycle::executing && timed_out.timeout_expired &&
                timed_out.last_reason == server_request_registry::reason_code::run_timeout,
            "run timeout remains durable until its lease is released");
    require(runtime.cancel(51, 226), "late cancellation is an idempotent no-op");
    require(runtime.release_slot(51, 0, 227), "release timed-out lease");
    require(!runtime.contains(51) && has_terminal_event(runtime, 51, lifecycle::timed_out,
                                                        server_request_registry::reason_code::run_timeout),
            "timeout wins race against later cancellation");

    require(runtime.admit(make_request(52, lane::low, 300)), "admit cancellation winner");
    require(runtime.take_next(301).request_id == 52 && runtime.bind_slot(52, 1, 302), "bind cancellation winner");
    require(runtime.cancel(52, 303), "cancel before run deadline");
    require(runtime.expire_due(400).empty(), "cancelled request cannot be overwritten by timeout");
    require(runtime.release_slot(52, 1, 401), "release cancelled lease");
    require(has_terminal_event(runtime, 52, lifecycle::cancelled, server_request_registry::reason_code::client_cancel),
            "cancellation remains terminal winner");

    require(runtime.admit(make_request(53, lane::normal, 500)), "admit completed winner");
    require(runtime.take_next(501).request_id == 53 && runtime.bind_slot(53, 2, 502), "bind completed winner");
    require(runtime.release_slot(53, 2, 503), "complete before run deadline");
    require(!runtime.cancel(53, 504) && runtime.expire_due(600).empty(),
            "completed request cannot be cancelled or expired after retirement");
}

void test_zero_configuration_keeps_bounded_defaults() {
    runtime_config config;
    config.default_queue_timeout_us = 0;
    config.default_run_timeout_us   = 0;
    request_runtime runtime(config);

    request_metadata queued = make_request(61, lane::normal, 100);
    require(runtime.admit(queued), "admit zero-config queue default request");
    require(runtime.expire_due(100 + runtime_config::queue_timeout_default_us - 1).empty(),
            "zero configuration keeps request before bounded queue default");
    require(runtime.expire_due(100 + runtime_config::queue_timeout_default_us).size() == 1,
            "zero configuration cannot disable bounded queue default");

    request_metadata running = make_request(62, lane::normal, 1000);
    require(runtime.admit(running) && runtime.take_next(1001).request_id == 62 && runtime.bind_slot(62, 0, 1002),
            "bind zero-config run default request");
    require(runtime.expire_due(1002 + runtime_config::run_timeout_default_us - 1).empty(),
            "zero configuration keeps request before bounded run default");
    require(runtime.expire_due(1002 + runtime_config::run_timeout_default_us).size() == 1 &&
                runtime.release_slot(62, 0, 1002 + runtime_config::run_timeout_default_us),
            "zero configuration cannot disable bounded run default");
}

void test_cache_affinity_scales_partial_hits_without_overflow() {
    runtime_config config;
    config.scheduler.context_tokens       = std::numeric_limits<uint64_t>::max();
    config.scheduler.aging_credit_us      = 1;
    config.scheduler.max_aging_credit_us  = 1;

    {
        request_runtime runtime(config);
        request_metadata empty = make_request(71, lane::normal, 1);
        empty.virtual_runtime_us                   = 500;
        empty.counts.prompt_tokens                 = 0;
        empty.counts.cached_prompt_tokens          = 0;
        empty.counts.requested_output_tokens       = 0;
        empty.estimates.predicted_prefill_us       = std::numeric_limits<uint64_t>::max();
        empty.estimates.predicted_cache_restore_us = 0;

        request_metadata miss = make_request(72, lane::normal, 1);
        miss.virtual_runtime_us             = 500;
        miss.counts.prompt_tokens           = 1;
        miss.counts.cached_prompt_tokens    = 0;
        miss.counts.requested_output_tokens = 0;
        miss.estimates.predicted_prefill_us = 0;
        require(runtime.admit(empty) && runtime.admit(miss), "admit zero-prompt cache-scaling fixture");
        const dispatch_result selected = runtime.take_next(1, 1);
        require(selected.selected && selected.request_id == 71,
                "zero-prompt work has zero cache affinity and preserves FIFO");
        require(runtime.cancel(71, 2) && runtime.cancel(72, 2), "zero-prompt fixture cleanup");
    }

    {
        request_runtime runtime(config);
        request_metadata miss = make_request(81, lane::normal, 1);
        miss.virtual_runtime_us             = 500;
        miss.counts.prompt_tokens           = 1;
        miss.counts.cached_prompt_tokens    = 0;
        miss.counts.requested_output_tokens = 0;
        miss.estimates.predicted_prefill_us = 0;

        request_metadata partial = make_request(82, lane::normal, 1);
        partial.virtual_runtime_us                   = 501;
        partial.counts.prompt_tokens                 = std::numeric_limits<uint64_t>::max() - 1;
        partial.counts.cached_prompt_tokens          = 2;
        partial.counts.requested_output_tokens       = 0;
        partial.estimates.predicted_prefill_us       = std::numeric_limits<uint64_t>::max();
        partial.estimates.predicted_cache_restore_us = 0;

        require(runtime.admit(miss) && runtime.admit(partial), "admit overflow-safe cache-scaling fixture");
        const dispatch_result selected = runtime.take_next(1, 1);
        require(selected.selected && selected.request_id == 82 &&
                    selected.request_reason == server_scheduler::reason_code::request_cache_affinity,
                "overflow-safe proportional estimate preserves the exact two-microsecond cache benefit");
        require(runtime.cancel(82, 2) && runtime.cancel(81, 2), "overflow-safe fixture cleanup");
    }
}

event_log_snapshot replay() {
    request_runtime runtime;
    runtime.admit(make_request(1, lane::low, 1));
    runtime.admit(make_request(2, lane::fast, 2));
    const auto first = runtime.take_next(3);
    runtime.bind_slot(first.request_id, 0, 4);
    runtime.release_slot(first.request_id, 0, 5);
    const auto second = runtime.take_next(6);
    runtime.mark_deferred(second.request_id, 7);
    runtime.cancel(second.request_id, 8);
    return runtime.events();
}

void test_deterministic_replay() {
    require(replay() == replay(), "identical live operations produce identical event history");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "live ingress and dispatch",  test_live_ingress_and_dispatch         },
        { "queue cap and cancellation", test_queue_cap_and_queued_cancellation },
        { "defer, resume, and cancel",  test_defer_resume_and_cancel           },
        { "cancel while bound",         test_cancel_while_bound                },
        { "passive child binding",      test_passive_child_binding             },
        { "dispatch permit lifecycle",  test_dispatch_permits_cover_selection_defer_and_caps },
        { "bounded fast refill",        test_bounded_same_fast_cohort_refill                  },
        { "fast refill boundaries",     test_fast_refill_window_and_terminal_boundaries       },
        { "fast refill telemetry",      test_fast_refill_telemetry_is_authoritative_at_sample_time },
        { "fast refill total budget",   test_fast_member_budget_covers_initial_fill_and_upgrade },
        { "fast refill reentry",        test_fast_refill_reentry_and_unbound_terminal_budget    },
        { "profiled low widths",        test_profiled_exclusive_low_widths                   },
        { "queue deadlines",            test_queue_deadlines_and_exact_capacity_reuse  },
        { "run deadline races",         test_run_deadline_and_first_terminal_wins      },
        { "zero timeout defaults",      test_zero_configuration_keeps_bounded_defaults },
        { "proportional cache affinity", test_cache_affinity_scales_partial_hits_without_overflow },
        { "deterministic replay",       test_deterministic_replay              },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-request-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-request-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

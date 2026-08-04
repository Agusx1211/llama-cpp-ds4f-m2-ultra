#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-request-runtime.h"

#include <cstdio>
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
    require(fast.estimates.predicted_gpu_us == 300 &&
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
        { "profiled low widths",        test_profiled_exclusive_low_widths                   },
        { "queue deadlines",            test_queue_deadlines_and_exact_capacity_reuse  },
        { "run deadline races",         test_run_deadline_and_first_terminal_wins      },
        { "zero timeout defaults",      test_zero_configuration_keeps_bounded_defaults },
        { "deterministic replay",       test_deterministic_replay              },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-request-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-request-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

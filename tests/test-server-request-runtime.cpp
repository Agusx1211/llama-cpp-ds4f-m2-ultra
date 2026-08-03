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
    request_runtime runtime(config);

    require(runtime.admit(make_request(1, lane::fast, 10)), "fill fast queue");
    const server_request_runtime::admission_result rejected = runtime.admit(make_request(2, lane::fast, 11));
    require(!rejected && rejected.code == server_request_runtime::result_code::queue_full &&
                rejected.reason == server_scheduler::reason_code::reject_queue_full,
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
    require(runtime.admit(make_request(31, lane::normal, 50), false), "register passive child");
    require(runtime.queued_total() == 0 && find_request(runtime, 31).queue == queue_state::ready,
            "child is durable without a second scheduler admission");
    require(runtime.bind_slot(31, 4, 51), "bind passive child");
    require(runtime.release_slot(31, 4, 52), "complete passive child");
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
        { "deterministic replay",       test_deterministic_replay              },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-request-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-request-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

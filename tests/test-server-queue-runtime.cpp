#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-queue.h"

#include <cstdio>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

server_task make_user(server_queue & queue, server_task::trusted_lane lane, uint64_t arrival_us) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id                          = queue.get_new_id();
    task.scheduling.lane             = lane;
    task.scheduling.arrival_us       = arrival_us;
    task.scheduling.predicted_gpu_us = 1;
    return task;
}

void test_internal_and_three_lane_dispatch() {
    server_queue     queue;
    std::vector<int> order;

    server_task low       = make_user(queue, server_task::trusted_lane::low, 1);
    server_task normal    = make_user(queue, server_task::trusted_lane::normal, 2);
    server_task fast      = make_user(queue, server_task::trusted_lane::fast, 3);
    const int   low_id    = low.id;
    const int   normal_id = normal.id;
    const int   fast_id   = fast.id;

    server_task control(SERVER_TASK_TYPE_METRICS);
    control.id           = queue.get_new_id();
    const int control_id = control.id;

    require(queue.post(std::move(low)) == low_id, "post low");
    require(queue.post(std::move(normal)) == normal_id, "post normal");
    require(queue.post(std::move(fast)) == fast_id, "post fast");
    require(queue.post(std::move(control)) == control_id, "post internal control");

    queue.on_new_task([&](server_task && task) {
        order.push_back(task.id);
        if (task.type == SERVER_TASK_TYPE_COMPLETION) {
            require(queue.bind_slot(task.id, static_cast<int>(order.size())), "bind selected user task");
            require(queue.release_slot(task.id, static_cast<int>(order.size())), "release selected user task");
        }
        if (order.size() == 4) {
            queue.terminate();
        }
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(order == std::vector<int>({ control_id, fast_id, normal_id, low_id }),
            "internal control is isolated and lane dispatch is deterministic");
    require(queue.request_summary().active_requests == 0, "completed user metadata retired");
    const auto events = queue.request_events();
    require(events.total_events >= 24 && events.events.back().kind == server_request_registry::event_kind::removed,
            "live queue publishes registry events");
}

void test_deferred_request_reenters_policy() {
    server_queue queue;
    server_task  task = make_user(queue, server_task::trusted_lane::normal, 10);
    const int    id   = task.id;
    require(queue.post(std::move(task)) == id, "post deferrable task");

    int dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        ++dispatches;
        if (dispatches == 1) {
            queue.defer(std::move(selected));
            return;
        }
        require(queue.bind_slot(selected.id, 0), "bind resumed task");
        require(queue.release_slot(selected.id, 0), "complete resumed task");
        queue.terminate();
    });
    queue.on_update_slots([&] {
        if (dispatches == 1) {
            queue.pop_deferred_task(0);
        }
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(dispatches == 2, "deferred request dispatched exactly twice");
    const auto events      = queue.request_events();
    bool       saw_blocked = false;
    for (const auto & event : events.events) {
        saw_blocked |= event.queue_after == server_request_registry::queue_state::blocked;
    }
    require(saw_blocked, "defer records blocked durable state");
}

void test_cancel_before_dispatch() {
    server_queue queue;
    server_task  task = make_user(queue, server_task::trusted_lane::low, 20);
    const int    id   = task.id;
    require(queue.post(std::move(task)) == id, "post cancellable task");

    std::vector<server_task> cancellations;
    server_task              cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id_target = id;
    cancellations.push_back(std::move(cancel));
    require(queue.post(std::move(cancellations), true) == 0, "post cancellation");

    int user_dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        user_dispatches += selected.type == SERVER_TASK_TYPE_COMPLETION;
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(user_dispatches == 0, "queued cancellation prevents user dispatch");
    require(queue.request_summary().active_requests == 0, "queued cancellation retires durable state");
}

void test_live_fast_queue_cap() {
    server_queue     queue;
    std::vector<int> accepted;
    for (int i = 0; i < 16; ++i) {
        server_task task = make_user(queue, server_task::trusted_lane::fast, static_cast<uint64_t>(100 + i));
        const int   id   = task.id;
        require(queue.post(std::move(task)) == id, "fill configured fast queue");
        accepted.push_back(id);
    }

    server_response        responses;
    server_response_reader reader(queue, responses, 1);
    server_task            rejected = make_user(queue, server_task::trusted_lane::fast, 200);
    reader.post_task(std::move(rejected));
    auto         backpressure = reader.next([] { return false; });
    const auto * error        = dynamic_cast<server_task_result_error *>(backpressure.get());
    require(error != nullptr && error->err_type == ERROR_TYPE_UNAVAILABLE && error->err_msg == "request queue is full",
            "seventeenth fast request receives explicit backpressure");
    require(queue.request_summary().active_requests == accepted.size(), "rejected request leaves no live metadata");

    std::vector<server_task> cancellations;
    for (int id : accepted) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true) == 0, "cancel capped queue");
    require(queue.request_summary().active_requests == 0, "capacity is reusable after queued cancellation");
}

void test_internal_defer_path_is_preserved() {
    server_queue queue;
    server_task  task(SERVER_TASK_TYPE_SLOT_SAVE);
    task.id      = queue.get_new_id();
    const int id = task.id;
    queue.defer(std::move(task));
    queue.pop_deferred_task(0);

    int dispatched = -1;
    queue.on_new_task([&](server_task && selected) {
        dispatched = selected.id;
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(dispatched == id, "legacy deferred control task returns to internal queue");
}

void test_parallel_children_share_parent_admission() {
    server_queue queue;
    server_task  parent    = make_user(queue, server_task::trusted_lane::normal, 300);
    const int    parent_id = parent.id;
    const int    child_id  = queue.get_new_id();
    parent.add_child(parent_id, child_id);

    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)) == 0, "post parent and passive child");
    require(queue.request_summary().active_requests == 2, "parent and child have separate durable identities");

    queue.on_new_task([&](server_task && selected) {
        require(selected.id == parent_id && selected.child_tasks.size() == 1, "only parent consumes lane admission");
        require(queue.bind_slot(child_id, 1), "bind passive child slot");
        require(queue.bind_slot(parent_id, 0), "bind parent slot");
        require(queue.release_slot(child_id, 1), "release passive child slot");
        require(queue.release_slot(parent_id, 0), "release parent slot");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(queue.request_summary().active_requests == 0, "parent group retires independently of slots");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "internal and lane dispatch",   test_internal_and_three_lane_dispatch         },
        { "deferred policy reentry",      test_deferred_request_reenters_policy         },
        { "cancel before dispatch",       test_cancel_before_dispatch                   },
        { "live fast queue cap",          test_live_fast_queue_cap                      },
        { "internal defer compatibility", test_internal_defer_path_is_preserved         },
        { "parallel child registration",  test_parallel_children_share_parent_admission },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-queue-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-queue-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-queue.h"

#include <cstdio>
#include <limits>
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

void require_posted(const server_queue_post_result & result, int task_id, const char * message) {
    require(result && result.task_id == task_id, message);
}

server_request_runtime::runtime_config test_runtime_config() {
    server_request_runtime::runtime_config config;
    config.scheduler.context_tokens = std::numeric_limits<uint64_t>::max();
    return config;
}

struct test_partial_result : server_task_result {
    bool is_stop() override { return false; }

    json to_json() override {
        return {
            { "partial", true }
        };
    }
};

server_task make_user(server_queue & queue, server_task::trusted_lane lane, uint64_t arrival_us) {
    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.id                          = queue.get_new_id();
    task.scheduling.lane             = lane;
    task.scheduling.arrival_us       = arrival_us;
    task.scheduling.predicted_gpu_us = 1;
    // Most legacy scheduling tests use tiny synthetic arrival timestamps. Keep
    // them independent of wall-clock deadlines; deadline tests override these.
    task.scheduling.queue_timeout_us = std::numeric_limits<uint64_t>::max();
    task.scheduling.run_timeout_us   = std::numeric_limits<uint64_t>::max();
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

    require_posted(queue.post(std::move(low)), low_id, "post low");
    require_posted(queue.post(std::move(normal)), normal_id, "post normal");
    require_posted(queue.post(std::move(fast)), fast_id, "post fast");
    require_posted(queue.post(std::move(control)), control_id, "post internal control");

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
    require_posted(queue.post(std::move(task)), id, "post deferrable task");

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
    require_posted(queue.post(std::move(task)), id, "post cancellable task");

    std::vector<server_task> cancellations;
    server_task              cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id_target = id;
    cancellations.push_back(std::move(cancel));
    require(queue.post(std::move(cancellations), true), "post cancellation");

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

void test_cancel_deferred_payload() {
    server_queue     queue;
    server_task  task = make_user(queue, server_task::trusted_lane::normal, 30);
    const int    id   = task.id;
    require_posted(queue.post(std::move(task)), id, "post deferred cancellation request");

    bool deferred                = false;
    bool cancel_posted           = false;
    int  cancellation_dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_COMPLETION) {
            deferred = true;
            queue.defer(std::move(selected));
            return;
        }
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++cancellation_dispatches;
            queue.terminate();
        }
    });
    queue.on_update_slots([&] {
        if (deferred && !cancel_posted) {
            cancel_posted = true;
            std::vector<server_task> cancellations;
            server_task              cancel(SERVER_TASK_TYPE_CANCEL);
            cancel.id_target = id;
            cancellations.push_back(std::move(cancel));
            require(queue.post(std::move(cancellations), true), "cancel blocked deferred request");
        }
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(cancellation_dispatches == 1 && queue.queue_tasks_deferred_size() == 0 &&
                queue.request_summary().active_requests == 0,
            "deferred cancellation removes payload and durable state exactly once");
}

void test_live_fast_queue_cap() {
    auto config                         = test_runtime_config();
    config.overload_retry_after_seconds = 999;
    uint64_t         now_us             = 100;
    server_queue     queue(config, [&] { return now_us; });
    std::vector<int> accepted;
    for (int i = 0; i < 16; ++i) {
        server_task task = make_user(queue, server_task::trusted_lane::fast, static_cast<uint64_t>(100 + i));
        const int   id   = task.id;
        require_posted(queue.post(std::move(task)), id, "fill configured fast queue");
        accepted.push_back(id);
    }

    server_response        responses;
    server_response_reader reader(queue, responses, 1);
    server_task            rejected = make_user(queue, server_task::trusted_lane::fast, 200);
    reader.post_task(std::move(rejected));
    auto         backpressure = reader.next([] { return false; });
    auto * error        = dynamic_cast<server_task_result_error *>(backpressure.get());
    require(error != nullptr && error->err_type == ERROR_TYPE_OVERLOADED && error->err_msg == "request queue is full",
            "seventeenth fast request receives overload backpressure");
    const json overload = error->to_json();
    require(json_value(overload, "code", 0) == 429 && json_value(overload, "retry_after", 0) == 60 &&
                retry_after_header_value(overload) == "60",
            "overload response carries HTTP 429 and bounded Retry-After");
    require(bounded_retry_after_seconds(0) == 1 && bounded_retry_after_seconds(999) == 60,
            "Retry-After public bounds are deterministic");
    require(queue.request_summary().active_requests == accepted.size(), "rejected request leaves no live metadata");

    std::vector<server_task> cancellations;
    for (int id : accepted) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true), "cancel capped queue");
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
    require(queue.post(std::move(tasks)), "post parent and passive child");
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

void test_non_overload_rejection_stays_unavailable() {
    auto config                     = test_runtime_config();
    config.scheduler.context_tokens = 1;
    uint64_t               now_us   = 100;
    server_queue           queue(config, [&] { return now_us; });
    server_response        responses;
    server_response_reader reader(queue, responses, 1);

    server_task rejected      = make_user(queue, server_task::trusted_lane::normal, 0);
    rejected.params.n_predict = 2;
    reader.post_task(std::move(rejected));
    auto   result = reader.next([] { return false; });
    auto * error  = dynamic_cast<server_task_result_error *>(result.get());
    require(error != nullptr && error->err_type == ERROR_TYPE_UNAVAILABLE,
            "non-overload admission rejection remains unavailable");
    const json unavailable = error->to_json();
    require(json_value(unavailable, "code", 0) == 503 && retry_after_header_value(unavailable).empty(),
            "HTTP 503 does not acquire an overload Retry-After header");
}

void test_transactional_multi_post_rollback() {
    auto config                                                                         = test_runtime_config();
    config.scheduler.lanes[static_cast<size_t>(server_scheduler::lane::fast)].queue_cap = 1;
    uint64_t     now_us                                                                 = 200;
    server_queue queue(config, [&] { return now_us; });

    server_task              first  = make_user(queue, server_task::trusted_lane::fast, 0);
    server_task              second = make_user(queue, server_task::trusted_lane::fast, 0);
    std::vector<server_task> batch;
    batch.push_back(std::move(first));
    batch.push_back(std::move(second));
    const auto rejected = queue.post(std::move(batch));
    require(!rejected && rejected.code == server_queue_post_code::overloaded && rejected.retry_after_seconds == 1,
            "multi-post reports the admission failure as overload");
    require(queue.request_summary().active_requests == 0, "failed multi-post rolls back every durable registration");

    server_task replacement    = make_user(queue, server_task::trusted_lane::fast, 0);
    const int   replacement_id = replacement.id;
    require_posted(queue.post(std::move(replacement)), replacement_id,
                   "rolled-back lane capacity is immediately reusable");
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = queue.get_new_id();
    cancel.id_target = replacement_id;
    require(queue.post(std::move(cancel), true), "clean replacement after rollback test");
}

void test_queue_and_deferred_deadlines() {
    auto config                                 = test_runtime_config();
    config.default_queue_timeout_us             = 10;
    config.registry.max_requests                = 1;
    uint64_t                             now_us = 300;
    server_queue                         queue(config, [&] { return now_us; });
    std::vector<server_queue_expiration> expirations;
    queue.on_request_expired([&](server_queue_expiration event) { expirations.push_back(event); });

    server_task first                 = make_user(queue, server_task::trusted_lane::normal, 0);
    first.scheduling.queue_timeout_us = 0;
    const int first_id                = first.id;
    require_posted(queue.post(std::move(first)), first_id, "post exact-deadline request");
    now_us                                  = 310;
    server_task replacement                 = make_user(queue, server_task::trusted_lane::normal, 0);
    replacement.scheduling.queue_timeout_us = 0;
    const int replacement_id                = replacement.id;
    require_posted(queue.post(std::move(replacement)), replacement_id,
                   "post boundary sweep reuses exact registry and lane capacity");
    require(expirations.size() == 1 && expirations[0].task_id == first_id &&
                expirations[0].kind == server_request_runtime::deadline_kind::queue,
            "queued request expires at exact injected-clock boundary");

    queue.on_new_task([&](server_task && selected) { queue.defer(std::move(selected)); });
    queue.on_update_slots([&] { now_us = 320; });
    queue.on_sleeping_state([](bool) {});
    queue.on_request_expired([&](server_queue_expiration event) {
        expirations.push_back(event);
        require(queue.request_summary().active_requests == 0, "expiry callback can reenter queue after mutex release");
        queue.terminate();
    });
    queue.start_loop();
    require(expirations.size() == 2 && expirations.back().task_id == replacement_id &&
                queue.queue_tasks_deferred_size() == 0 && queue.request_summary().active_requests == 0,
            "blocked deferred request expires, removes payload, and retires metadata");
}

void test_selected_request_expires_before_bind() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 10;
    uint64_t               now_us   = 350;
    server_queue           queue(config, [&] { return now_us; });
    int                    expiration_callbacks = 0;
    server_queue_bind_code bind_code            = server_queue_bind_code::rejected;

    queue.on_request_expired([&](server_queue_expiration event) {
        ++expiration_callbacks;
        require(event.kind == server_request_runtime::deadline_kind::queue,
                "selected unbound request keeps queue deadline");
        require(queue.request_summary().active_requests == 0, "pre-bind expiry callback runs outside queue mutex");
    });
    queue.on_new_task([&](server_task && selected) {
        now_us    = 360;
        bind_code = queue.bind_slot(selected.id, 0).code;
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});

    server_task task                 = make_user(queue, server_task::trusted_lane::normal, 0);
    task.scheduling.queue_timeout_us = 0;
    require(queue.post(std::move(task)), "post pre-bind deadline request");
    queue.start_loop();
    require(bind_code == server_queue_bind_code::expired && expiration_callbacks == 1 &&
                queue.request_summary().active_requests == 0,
            "selected payload crossing its deadline cannot bind or remain durable");
}

void test_passive_child_deadline_order() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 10;
    uint64_t         now_us         = 400;
    server_queue     queue(config, [&] { return now_us; });
    std::vector<int> expired_ids;
    queue.on_request_expired([&](server_queue_expiration event) { expired_ids.push_back(event.task_id); });

    server_task parent                 = make_user(queue, server_task::trusted_lane::normal, 0);
    parent.scheduling.queue_timeout_us = 0;
    const int parent_id                = parent.id;
    const int child_id                 = queue.get_new_id();
    parent.add_child(parent_id, child_id);
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post parent and passive deadline child");
    now_us = 410;
    require(queue.expire_requests() == 2 && expired_ids == std::vector<int>({ parent_id, child_id }),
            "parent and passive child expire in deterministic durable-ID order");
    require(queue.request_summary().active_requests == 0, "passive child deadline releases all capacity");
}

void test_passive_child_cancellation() {
    server_queue queue;
    server_task  parent    = make_user(queue, server_task::trusted_lane::normal, 450);
    const int    parent_id = parent.id;
    const int    child_id  = queue.get_new_id();
    parent.add_child(parent_id, child_id);
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post cancellable passive child");

    std::vector<server_task> cancellations;
    server_task              cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id_target = child_id;
    cancellations.push_back(std::move(cancel));
    require(queue.post(std::move(cancellations), true), "cancel passive child before parent dispatch");
    require(queue.request_summary().active_requests == 1,
            "passive-child cancellation preserves parent durable request");

    int cancellation_dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++cancellation_dispatches;
            return;
        }
        require(selected.id == parent_id && selected.child_tasks.empty(),
                "cancelled passive child payload cannot reach slot setup");
        require(queue.bind_slot(parent_id, 0) && queue.release_slot(parent_id, 0), "unaffected parent still completes");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(cancellation_dispatches == 1 && queue.request_summary().active_requests == 0,
            "passive-child cancellation retires independently");
}

void test_bound_stream_timeout_and_cancel_race() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 20;
    uint64_t               now_us   = 500;
    server_queue           queue(config, [&] { return now_us; });
    server_response        responses;
    server_response_reader reader(queue, responses, 1);
    int                    cancellation_dispatches = 0;
    int                    released_leases         = 0;
    int                    expiration_callbacks    = 0;

    queue.on_request_expired([&](server_queue_expiration event) {
        ++expiration_callbacks;
        auto error      = std::make_unique<server_task_result_error>();
        error->id       = event.task_id;
        error->err_type = ERROR_TYPE_TIMEOUT;
        error->err_msg  = event.kind == server_request_runtime::deadline_kind::run ? "request run deadline exceeded" :
                                                                                     "request queue deadline exceeded";
        responses.send(std::move(error));
    });
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_COMPLETION) {
            require(queue.bind_slot(selected.id, 0), "bind streaming request");
            auto partial = std::make_unique<test_partial_result>();
            partial->id  = selected.id;
            responses.send(std::move(partial));
            queue.terminate();
            return;
        }
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++cancellation_dispatches;
            released_leases += queue.release_slot(selected.id_target, 0);
            if (cancellation_dispatches == 2) {
                queue.terminate();
            }
        }
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});

    server_task streaming                 = make_user(queue, server_task::trusted_lane::fast, 0);
    streaming.scheduling.queue_timeout_us = 0;
    streaming.scheduling.run_timeout_us   = 0;
    streaming.params.stream               = true;
    const int streaming_id                = streaming.id;
    reader.post_task(std::move(streaming));
    queue.start_loop();
    auto partial = reader.next([] { return false; });
    require(partial != nullptr && !partial->is_error() && !partial->is_stop(),
            "stream produces a partial response before its run deadline");

    now_us = 520;
    require(queue.release_slot(streaming_id, 0),
            "release crossing exact run deadline expires and retires the bound lease");
    auto   timeout = reader.next([] { return false; });
    auto * error   = dynamic_cast<server_task_result_error *>(timeout.get());
    require(error != nullptr && error->err_type == ERROR_TYPE_TIMEOUT && json_value(error->to_json(), "code", 0) == 408,
            "stream receives deterministic timeout envelope after headers are already sent");
    // next() stops an errored reader and explicitly posts a client CANCEL in
    // addition to the run-expiry CANCEL already queued by release_slot().
    require(reader.cancelled, "timeout reader posts duplicate client cancellation control");

    queue.start_loop();
    require(expiration_callbacks == 1 && cancellation_dispatches == 2 && released_leases == 0 &&
                queue.request_summary().active_requests == 0,
            "deadline emits once and duplicate cancellation controls cannot retire the lease a second time");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "internal and lane dispatch",   test_internal_and_three_lane_dispatch         },
        { "deferred policy reentry",      test_deferred_request_reenters_policy         },
        { "cancel before dispatch",       test_cancel_before_dispatch                   },
        { "cancel deferred payload",      test_cancel_deferred_payload                  },
        { "live fast queue cap",          test_live_fast_queue_cap                      },
        { "internal defer compatibility", test_internal_defer_path_is_preserved         },
        { "parallel child registration",  test_parallel_children_share_parent_admission },
        { "non-overload unavailable",     test_non_overload_rejection_stays_unavailable },
        { "transactional rollback",       test_transactional_multi_post_rollback        },
        { "queue and deferred deadlines", test_queue_and_deferred_deadlines             },
        { "selected pre-bind deadline",   test_selected_request_expires_before_bind     },
        { "passive child deadlines",      test_passive_child_deadline_order             },
        { "passive child cancellation",   test_passive_child_cancellation               },
        { "bound stream timeout",         test_bound_stream_timeout_and_cancel_race     },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-queue-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-queue-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

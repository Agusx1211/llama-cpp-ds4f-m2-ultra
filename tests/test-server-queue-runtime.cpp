#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-queue.h"

#include <array>

#include <condition_variable>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>
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

server_task make_user(server_queue &              queue,
                      server_task::trusted_lane lane,
                      uint64_t                   arrival_us,
                      server_task_type           type = SERVER_TASK_TYPE_COMPLETION) {
    server_task task(type);
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

server_task make_family(server_queue &              queue,
                        server_task::trusted_lane lane,
                        uint64_t                   arrival_us,
                        size_t                     children = 2) {
    server_task parent = make_user(queue, lane, arrival_us);
    for (size_t i = 0; i < children; ++i) {
        parent.add_child(parent.id, queue.get_new_id());
    }
    return parent;
}

struct bound_family_slot {
    server_queue *                     queue = nullptr;
    int                                slot  = -1;
    std::unique_ptr<const server_task> task;

    bound_family_slot(server_queue & queue, int slot, server_task && task) :
        queue(&queue), slot(slot), task(std::make_unique<const server_task>(std::move(task))) {}

    void release() {
        const int task_id = task->id;
        require(queue->release_slot(task_id, slot), "family release helper drains bound durable slot");
        task.reset();
    }
};

void bind_only(server_queue & queue, int task_id, int slot_id) {
    queue.on_new_task([&](server_task && selected) {
        require(selected.id == task_id, "dispatch expected publication task");
        require(queue.bind_slot(selected.id, slot_id), "bind publication task");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
}

server_task_result_ptr make_final_result(server_task_type type, int task_id) {
    server_task_result_ptr result;
    switch (type) {
        case SERVER_TASK_TYPE_COMPLETION:
            result = std::make_unique<server_task_result_cmpl_final>();
            break;
        case SERVER_TASK_TYPE_EMBEDDING:
            result = std::make_unique<server_task_result_embd>();
            break;
        case SERVER_TASK_TYPE_RERANK:
            result = std::make_unique<server_task_result_rerank>();
            break;
        default:
            throw std::runtime_error("unsupported publication fixture type");
    }
    result->id = task_id;
    return result;
}

void send_timeout_result(server_response & responses, server_queue_expiration event) {
    auto error      = std::make_unique<server_task_result_error>();
    error->id       = event.task_id;
    error->err_type = ERROR_TYPE_TIMEOUT;
    error->err_msg  = event.kind == server_request_runtime::deadline_kind::run ? "request run deadline exceeded" :
                                                                                 "request queue deadline exceeded";
    responses.send(std::move(error));
}

void require_timeout_only(server_response & responses, int task_id, const char * message) {
    auto result = responses.recv_with_timeout({ task_id }, 0);
    auto error  = dynamic_cast<server_task_result_error *>(result.get());
    require(error != nullptr && error->err_type == ERROR_TYPE_TIMEOUT, message);
    require(responses.recv_with_timeout({ task_id }, 0) == nullptr,
            "terminal timeout leaves no queued success result");
}

void retire_bound_and_queued(server_queue &           queue,
                             const std::vector<int> & all_ids,
                             const std::vector<int> & bound_ids,
                             const char *             message) {
    for (size_t i = 0; i < bound_ids.size(); ++i) {
        require(queue.release_slot(bound_ids[i], static_cast<int>(i)), "release fixture bound permit");
    }

    const std::unordered_set<int> bound_set(bound_ids.begin(), bound_ids.end());
    std::vector<server_task>      cancellations;
    for (int id : all_ids) {
        if (bound_set.count(id) == 0) {
            server_task cancel(SERVER_TASK_TYPE_CANCEL);
            cancel.id_target = id;
            cancellations.push_back(std::move(cancel));
        }
    }
    require(queue.post(std::move(cancellations), true) && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
            message);
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
                queue.queue_tasks_deferred_size() == 0 && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
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
                queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
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

void test_parent_cancellation_retires_family() {
    server_queue queue;
    server_task  parent    = make_family(queue, server_task::trusted_lane::normal, 460);
    const int    parent_id = parent.id;
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post family for parent cancellation");

    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = queue.get_new_id();
    cancel.id_target = parent_id;
    require(queue.post(std::move(cancel), true) && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
            "parent cancellation retires every passive durable child");

    int user_dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        user_dispatches += selected.type == SERVER_TASK_TYPE_COMPLETION;
    });
    queue.on_update_slots([&] { queue.terminate(); });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(user_dispatches == 0, "cancelled parent payload and children cannot reach slot launch");
}

void test_parent_failure_retires_queued_family() {
    server_queue queue;
    server_task  parent    = make_family(queue, server_task::trusted_lane::normal, 470);
    const int    parent_id = parent.id;
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post family for parent failure");
    require(queue.fail_task(parent_id) && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
            "parent setup failure retires the queued family");

    int user_dispatches = 0;
    queue.on_new_task([&](server_task && selected) {
        user_dispatches += selected.type == SERVER_TASK_TYPE_COMPLETION;
    });
    queue.on_update_slots([&] { queue.terminate(); });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(user_dispatches == 0, "failed parent payload and children cannot reach slot launch");
}

void test_partial_family_launch_failure_retires_permits() {
    server_queue queue;
    server_task  parent    = make_family(queue, server_task::trusted_lane::normal, 480);
    const int    parent_id = parent.id;
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post family for partial launch failure");

    queue.on_new_task([&](server_task && selected) {
        require(selected.id == parent_id && selected.child_tasks.size() == 2,
                "family dispatch preserves both child payloads");
        const int launched_child = selected.child_tasks[0].id;
        const int unbound_child  = selected.child_tasks[1].id;
        require(queue.bind_slot(launched_child, 0), "simulate one launched child");
        require(queue.fail_task(parent_id), "partial launch failure marks the complete family");
        require(!queue.bind_slot(unbound_child, 1), "unlaunched child cannot bind after parent failure");
        require(queue.release_slot(launched_child, 0), "release the one failed launched child");
        require(queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
                "partial family launch failure retires all records and permits");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
}

void run_bound_parent_terminal_releases_family(bool fail_parent) {
    server_queue queue;
    server_task  parent    = make_family(queue, server_task::trusted_lane::normal, fail_parent ? 490 : 485);
    const int    parent_id = parent.id;
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post family for bound parent terminal path");

    std::vector<bound_family_slot> slots;
    int                            cancel_controls = 0;
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++cancel_controls;
            require(selected.id_target == parent_id && release_server_task_family_slots(slots, parent_id) == 3,
                    "server family control releases parent and every id_parent child");
            require(queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
                    "server family control drains every bound durable permit");
            queue.terminate();
            return;
        }

        require(selected.id == parent_id && selected.child_tasks.size() == 2,
                "bound terminal fixture dispatches complete family");
        int slot = 0;
        for (server_task & child : selected.child_tasks) {
            const int child_id = child.id;
            require(queue.bind_slot(child_id, slot), "bind child before parent terminal event");
            slots.emplace_back(queue, slot++, std::move(child));
        }
        require(queue.bind_slot(parent_id, slot), "bind parent before terminal event");
        slots.emplace_back(queue, slot, std::move(selected));

        if (fail_parent) {
            require(queue.fail_task(parent_id), "bound parent failure schedules family release control");
        } else {
            server_task cancel(SERVER_TASK_TYPE_CANCEL);
            cancel.id        = queue.get_new_id();
            cancel.id_target = parent_id;
            require(queue.post(std::move(cancel), true), "bound parent cancellation posts family release control");
        }
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(cancel_controls == 1, "bound parent terminal event executes one family release control");
}

void test_bound_parent_cancellation_releases_family_slots() {
    run_bound_parent_terminal_releases_family(false);
}

void test_bound_parent_failure_releases_family_slots() {
    run_bound_parent_terminal_releases_family(true);
}

server_queue_post_result post_family(server_queue & queue, server_task::trusted_lane lane, size_t children) {
    server_task parent = make_family(queue, lane, 500, children);
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    return queue.post(std::move(tasks));
}

void run_accepted_family_width(server_task::trusted_lane lane, size_t children, size_t expected_members) {
    server_queue queue;
    require(post_family(queue, lane, children), "accept family at exact cohort shape");
    queue.on_new_task([&](server_task && selected) {
        require(selected.child_tasks.size() + 1 == expected_members,
                "accepted family retains every passive child payload");
        int slot = 0;
        for (const server_task & child : selected.child_tasks) {
            require(queue.bind_slot(child.id, slot), "bind accepted boundary child");
            ++slot;
        }
        require(queue.bind_slot(selected.id, slot), "bind accepted boundary parent");
        slot = 0;
        for (const server_task & child : selected.child_tasks) {
            require(queue.release_slot(child.id, slot), "release accepted boundary child");
            ++slot;
        }
        require(queue.release_slot(selected.id, slot), "release accepted boundary parent");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
            "accepted family boundary drains without leaks");
}

void test_family_width_admission_is_atomic() {
    for (const auto & rejected : std::vector<std::pair<server_task::trusted_lane, size_t>>({
             { server_task::trusted_lane::normal, 8  },
             { server_task::trusted_lane::fast,   2  },
             { server_task::trusted_lane::low,    19 },
         })) {
        server_queue queue;
        const auto result = post_family(queue, rejected.first, rejected.second);
        require(!result && result.code == server_queue_post_code::unavailable &&
                    queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
                "impossible family shape is rejected before durable insertion");
    }

    server_queue physical;
    physical.set_physical_slot_capacity(2);
    const auto physical_rejection = post_family(physical, server_task::trusted_lane::normal, 2);
    require(!physical_rejection && physical_rejection.code == server_queue_post_code::unavailable &&
                physical.request_summary().active_requests == 0,
            "family wider than authoritative physical slots is rejected atomically");

    server_queue atomic;
    server_task valid   = make_user(atomic, server_task::trusted_lane::normal, 510);
    server_task invalid = make_family(atomic, server_task::trusted_lane::fast, 511, 2);
    std::vector<server_task> mixed;
    mixed.push_back(std::move(valid));
    mixed.push_back(std::move(invalid));
    require(!atomic.post(std::move(mixed)) && atomic.request_summary().active_requests == 0,
            "multi-post preflight rejects a later impossible family without admitting earlier work");

    run_accepted_family_width(server_task::trusted_lane::normal, 7, 8);
    run_accepted_family_width(server_task::trusted_lane::fast, 1, 2);
    run_accepted_family_width(server_task::trusted_lane::low, 23, 24);
}

void test_final_result_publication_deadline_gate() {
    const std::vector<server_task_type> result_types = {
        SERVER_TASK_TYPE_COMPLETION,
        SERVER_TASK_TYPE_EMBEDDING,
        SERVER_TASK_TYPE_RERANK,
    };

    auto run_case = [](server_task_type type, uint64_t epoch_us, bool deadline_wins) {
        auto config                     = test_runtime_config();
        config.default_queue_timeout_us = 100;
        config.default_run_timeout_us   = 10;
        uint64_t        now_us = epoch_us;
        server_queue    queue(config, [&] { return now_us; });
        server_response responses;
        int expiration_callbacks = 0;
        queue.on_request_expired([&](server_queue_expiration event) {
            ++expiration_callbacks;
            send_timeout_result(responses, event);
        });

        server_task task                 = make_user(queue, server_task::trusted_lane::normal, 0, type);
        task.scheduling.queue_timeout_us = 0;
        task.scheduling.run_timeout_us   = 0;
        const int task_id                = task.id;
        responses.add_waiting_task_id(task_id);
        require_posted(queue.post(std::move(task)), task_id, "post gated final publication");
        bind_only(queue, task_id, 0);

        now_us = epoch_us + (deadline_wins ? 10 : 9);
        const bool published = queue.publish_slot_result(responses, task_id, 0,
                                                         server_queue_result_kind::final,
                                                         make_final_result(type, task_id));
        require(published != deadline_wins,
                "completion, embedding, or rerank publication obeys the exact run boundary");
        if (deadline_wins) {
            require_timeout_only(responses, task_id, "exact-boundary final publication returns timeout");
            require(expiration_callbacks == 1 && queue.release_slot(task_id, 0) &&
                        queue.request_summary().active_requests == 0,
                    "exact-boundary final timeout retires its slot exactly once");
        } else {
            auto success = responses.recv_with_timeout({ task_id }, 0);
            require(success != nullptr && !success->is_error() && success->is_stop(),
                    "pre-boundary final publication emits exactly one success");
            now_us = epoch_us + 10;
            require(queue.expire_requests() == 0 && expiration_callbacks == 0 &&
                        queue.request_summary().active_requests == 0 &&
                        responses.recv_with_timeout({ task_id }, 0) == nullptr,
                    "committed final success cannot be overwritten by a later timeout");
        }
    };

    uint64_t epoch_us = 1000;
    for (server_task_type type : result_types) {
        run_case(type, epoch_us, true);
        epoch_us += 100;
    }
    for (server_task_type type : result_types) {
        run_case(type, epoch_us, false);
        epoch_us += 100;
    }
}

void test_stream_result_publication_deadline_gate() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 10;
    uint64_t        now_us          = 3000;
    server_queue    queue(config, [&] { return now_us; });
    server_response responses;
    int             expiration_callbacks = 0;
    queue.on_request_expired([&](server_queue_expiration event) {
        ++expiration_callbacks;
        send_timeout_result(responses, event);
    });

    server_task streaming                 = make_user(queue, server_task::trusted_lane::fast, 0);
    streaming.scheduling.queue_timeout_us = 0;
    streaming.scheduling.run_timeout_us   = 0;
    streaming.params.stream               = true;
    const int streaming_id                = streaming.id;
    responses.add_waiting_task_id(streaming_id);
    require_posted(queue.post(std::move(streaming)), streaming_id, "post streaming publication task");
    bind_only(queue, streaming_id, 0);

    now_us = 3009;
    auto partial = std::make_unique<test_partial_result>();
    partial->id  = streaming_id;
    require(queue.publish_slot_result(responses, streaming_id, 0, server_queue_result_kind::partial,
                                      std::move(partial)),
            "streaming partial publishes before run deadline");
    auto published_partial = responses.recv_with_timeout({ streaming_id }, 0);
    require(published_partial != nullptr && !published_partial->is_error() && !published_partial->is_stop(),
            "pre-boundary streaming partial is observable");

    now_us = 3010;
    require(!queue.publish_slot_result(responses, streaming_id, 0, server_queue_result_kind::final,
                                       make_final_result(SERVER_TASK_TYPE_COMPLETION, streaming_id)),
            "streaming final is denied at exact run deadline");
    require_timeout_only(responses, streaming_id, "streaming final exact boundary returns timeout");
    require(expiration_callbacks == 1 && queue.release_slot(streaming_id, 0),
            "streaming final timeout retires once after an earlier partial");
    queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_CANCEL && selected.id_target == streaming_id,
                "streaming expiry queues its internal cancellation control");
        queue.terminate();
    });
    queue.start_loop();

    now_us = 3100;
    server_task exact_partial                 = make_user(queue, server_task::trusted_lane::fast, 0);
    exact_partial.scheduling.queue_timeout_us = 0;
    exact_partial.scheduling.run_timeout_us   = 0;
    exact_partial.params.stream               = true;
    const int exact_partial_id                = exact_partial.id;
    responses.add_waiting_task_id(exact_partial_id);
    require_posted(queue.post(std::move(exact_partial)), exact_partial_id,
                   "post exact-boundary streaming partial task");
    bind_only(queue, exact_partial_id, 1);

    now_us = 3110;
    auto denied_partial = std::make_unique<test_partial_result>();
    denied_partial->id  = exact_partial_id;
    require(!queue.publish_slot_result(responses, exact_partial_id, 1, server_queue_result_kind::partial,
                                       std::move(denied_partial)),
            "streaming partial is denied at exact run deadline");
    require_timeout_only(responses, exact_partial_id, "streaming partial exact boundary returns timeout");
    require(expiration_callbacks == 2 && queue.release_slot(exact_partial_id, 1) &&
                queue.request_summary().active_requests == 0,
            "partial timeout emits once and leaks no success after terminal expiry");
}

void test_concurrent_update_publication_after_expiry() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 10;
    uint64_t        now_us          = 4000;
    server_queue    queue(config, [&] { return now_us; });
    server_response responses;
    std::mutex              rendezvous_mutex;
    std::condition_variable rendezvous;
    bool                    update_entered = false;
    bool                    expiry_done    = false;
    bool                    published      = true;
    int                     task_id        = -1;
    int                     expiration_callbacks = 0;
    int                     cancellation_dispatches = 0;
    int                     released_leases = 0;

    queue.on_request_expired([&](server_queue_expiration event) {
        ++expiration_callbacks;
        send_timeout_result(responses, event);
    });
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_COMPLETION) {
            require(queue.bind_slot(selected.id, 0), "bind concurrent update publication task");
            return;
        }
        require(selected.type == SERVER_TASK_TYPE_CANCEL && selected.id_target == task_id,
                "run expiry queues cancellation for concurrent update task");
        ++cancellation_dispatches;
        released_leases += queue.release_slot(selected.id_target, 0);
        queue.terminate();
    });
    queue.on_update_slots([&] {
        {
            std::lock_guard<std::mutex> lock(rendezvous_mutex);
            update_entered = true;
        }
        rendezvous.notify_one();
        {
            std::unique_lock<std::mutex> lock(rendezvous_mutex);
            rendezvous.wait(lock, [&] { return expiry_done; });
        }
        auto partial = std::make_unique<test_partial_result>();
        partial->id  = task_id;
        published = queue.publish_slot_result(responses, task_id, 0, server_queue_result_kind::partial,
                                              std::move(partial));
    });
    queue.on_sleeping_state([](bool) {});

    server_task task                 = make_user(queue, server_task::trusted_lane::normal, 0);
    task.scheduling.queue_timeout_us = 0;
    task.scheduling.run_timeout_us   = 0;
    task.params.stream               = true;
    task_id                          = task.id;
    responses.add_waiting_task_id(task_id);
    require_posted(queue.post(std::move(task)), task_id, "post concurrent update publication task");

    std::thread expirer([&] {
        {
            std::unique_lock<std::mutex> lock(rendezvous_mutex);
            rendezvous.wait(lock, [&] { return update_entered; });
        }
        now_us = 4010;
        require(queue.expire_requests() == 1, "concurrent sweep expires task at exact run deadline");
        {
            std::lock_guard<std::mutex> lock(rendezvous_mutex);
            expiry_done = true;
        }
        rendezvous.notify_one();
    });

    queue.start_loop();
    expirer.join();
    require(!published, "callback_update_slots cannot publish after concurrent terminal expiry");
    require_timeout_only(responses, task_id, "concurrent update race returns only timeout");
    require(expiration_callbacks == 1 && cancellation_dispatches == 1 && released_leases == 1 &&
                queue.request_summary().active_requests == 0,
            "concurrent expiry performs exactly one timeout and one slot cleanup");
}

void test_publication_gate_scales_without_global_sweep() {
    auto config = test_runtime_config();
    config.scheduler.lanes[static_cast<size_t>(server_scheduler::lane::low)].queue_cap = 128;
    config.registry.max_requests                                                        = 128;
    config.default_queue_timeout_us                                                     = 1000;
    config.default_run_timeout_us                                                       = 1000;
    uint64_t        now_us = 5000;
    server_queue    queue(config, [&] { return now_us; });
    server_response responses;
    std::vector<int> bound_ids;
    std::vector<int> queued_ids;
    std::vector<int> expired_ids;

    int target_id        = -1;
    int unrelated_due_id = -1;
    int release_probe_id = -1;
    for (int i = 0; i < 64; ++i) {
        server_task task = make_user(queue, server_task::trusted_lane::low, 0);
        if (i == 0) {
            target_id                      = task.id;
            task.scheduling.run_timeout_us = 100;
        } else if (i == 1) {
            unrelated_due_id               = task.id;
            task.scheduling.run_timeout_us = 10;
        } else if (i == 2) {
            release_probe_id = task.id;
        }
        const int task_id = task.id;
        require_posted(queue.post(std::move(task)), task_id, "post scaled bound publication task");
    }

    queue.on_request_expired([&](server_queue_expiration event) { expired_ids.push_back(event.task_id); });
    queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_COMPLETION, "scaled setup dispatches only user tasks");
        require(queue.bind_slot(selected.id, selected.id), "bind scaled publication task");
        bound_ids.push_back(selected.id);
        if (bound_ids.size() == 64) {
            queue.terminate();
        }
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    for (int i = 0; i < 32; ++i) {
        server_task task  = make_user(queue, server_task::trusted_lane::low, 0);
        const int task_id = task.id;
        require_posted(queue.post(std::move(task)), task_id, "post scaled queued publication task");
        queued_ids.push_back(task_id);
    }
    require(queue.request_summary().active_requests == 96, "scale fixture holds 64 bound and 32 queued requests");

    responses.add_waiting_task_id(target_id);
    now_us = 5010;
    auto partial = std::make_unique<test_partial_result>();
    partial->id  = target_id;
    require(queue.publish_slot_result(responses, target_id, target_id, server_queue_result_kind::partial,
                                      std::move(partial)),
            "target partial publishes while an unrelated request is due");
    auto published = responses.recv_with_timeout({ target_id }, 0);
    require(published != nullptr && !published->is_stop() && expired_ids.empty(),
            "current-request gate does not sweep unrelated deadlines");
    require(queue.release_slot(release_probe_id, release_probe_id) && expired_ids.empty(),
            "current-request slot release does not sweep unrelated deadlines");

    require(queue.expire_requests() == 1 && expired_ids == std::vector<int>({ unrelated_due_id }),
            "normal queue boundary later sweeps the unrelated due request");
    require(queue.release_slot(unrelated_due_id, unrelated_due_id), "release unrelated timed-out scale lease");

    require(queue.publish_slot_result(responses, target_id, target_id, server_queue_result_kind::final,
                                      make_final_result(SERVER_TASK_TYPE_COMPLETION, target_id)),
            "scaled target final commits before its own deadline");
    require(responses.recv_with_timeout({ target_id }, 0) != nullptr, "scaled target final is observable");
    for (int id : bound_ids) {
        if (id != target_id && id != unrelated_due_id && id != release_probe_id) {
            require(queue.release_slot(id, id), "release remaining scaled bound lease");
        }
    }

    std::vector<server_task> cancellations;
    cancellations.reserve(queued_ids.size());
    for (int id : queued_ids) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true) && queue.request_summary().active_requests == 0,
            "scale fixture cleanup retires all remaining durable requests");
}

void run_mixed_lane_isolation(bool include_fast,
                              const std::array<size_t, server_scheduler::lane_count> & expected) {
    server_queue queue;
    queue.set_physical_slot_capacity(64);
    std::vector<int> all_ids;
    std::vector<std::pair<server_task::trusted_lane, size_t>> fixtures = {
        { server_task::trusted_lane::low,    65 },
        { server_task::trusted_lane::normal, 9  },
    };
    if (include_fast) {
        fixtures.push_back({ server_task::trusted_lane::fast, 3 });
    }
    for (const auto & fixture : fixtures) {
        for (size_t i = 0; i < fixture.second; ++i) {
            server_task task = make_user(queue, fixture.first, i + 1);
            all_ids.push_back(task.id);
            require_posted(queue.post(std::move(task)), all_ids.back(), "post mixed permit task");
        }
    }

    std::array<size_t, server_scheduler::lane_count> dispatched = {};
    std::vector<int>                                 bound_ids;
    queue.on_new_task([&](server_task && selected) {
        const size_t lane_index = static_cast<size_t>(selected.scheduling.lane);
        ++dispatched[lane_index];
        const int slot = static_cast<int>(bound_ids.size());
        require(queue.bind_slot(selected.id, slot), "mixed dispatch binds claimed permit");
        bound_ids.push_back(selected.id);
    });
    queue.on_update_slots([&] {
        const auto permits = queue.dispatch_permits();
        require(dispatched == expected && permits.claimed == expected && permits.bound == expected &&
                    permits.total == (include_fast ? 2 : 8),
                "dominant live lane bounds the entire mixed cohort");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    retire_bound_and_queued(queue, all_ids, bound_ids, "mixed permit fixture retires without leaks");
}

void test_start_loop_enforces_live_lane_and_physical_permits() {
    run_mixed_lane_isolation(true, { 0, 0, 2 });
    run_mixed_lane_isolation(false, { 0, 8, 0 });
}

void run_exclusive_low_width(size_t ready, size_t expected) {
    server_queue queue;
    queue.set_physical_slot_capacity(64);
    std::vector<int> all_ids;
    std::vector<int> bound_ids;
    for (size_t i = 0; i < ready; ++i) {
        server_task task = make_user(queue, server_task::trusted_lane::low, i + 1);
        all_ids.push_back(task.id);
        require_posted(queue.post(std::move(task)), all_ids.back(), "post exclusive low task");
    }
    queue.on_new_task([&](server_task && selected) {
        const int slot = static_cast<int>(bound_ids.size());
        require(queue.bind_slot(selected.id, slot), "exclusive low selection binds permit");
        bound_ids.push_back(selected.id);
    });
    queue.on_update_slots([&] {
        require(bound_ids.size() == expected && queue.dispatch_permits().total == expected,
                "start_loop stops at the profiled exclusive-low shape");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    retire_bound_and_queued(queue, all_ids, bound_ids, "exclusive low fixture cleanup");
}

void test_start_loop_uses_profiled_exclusive_low_shapes() {
    run_exclusive_low_width(20, 8);
    run_exclusive_low_width(24, 24);
}

void test_fast_arrival_bypasses_ready_low_backlog() {
    server_queue queue;
    queue.set_physical_slot_capacity(4);
    std::vector<int> all_ids;
    for (size_t i = 0; i < 20; ++i) {
        server_task low = make_user(queue, server_task::trusted_lane::low, i + 1);
        all_ids.push_back(low.id);
        require_posted(queue.post(std::move(low)), all_ids.back(), "post fast-arrival low backlog");
    }

    std::vector<int>                       bound_ids;
    std::vector<server_task::trusted_lane> order;
    queue.on_new_task([&](server_task && selected) {
        order.push_back(selected.scheduling.lane);
        const int slot = static_cast<int>(bound_ids.size());
        require(queue.bind_slot(selected.id, slot), "bind dynamic-arrival selection");
        bound_ids.push_back(selected.id);
        if (order.size() == 1) {
            server_task fast = make_user(queue, server_task::trusted_lane::fast, 100);
            all_ids.push_back(fast.id);
            require_posted(queue.post(std::move(fast)), all_ids.back(), "post fast request during low dispatch");
        }
    });
    queue.on_update_slots([&] {
        require(order == std::vector<server_task::trusted_lane>(
                             { server_task::trusted_lane::low, server_task::trusted_lane::fast }) &&
                    queue.dispatch_permits().total == 2 && queue.request_summary().active_requests == 21,
                "fast arrival stops widening and leaves the remaining low backlog untouched");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(order == std::vector<server_task::trusted_lane>(
                         { server_task::trusted_lane::low, server_task::trusted_lane::fast }),
            "new fast work is selected before and stops the remaining ready low backlog");
    retire_bound_and_queued(queue, all_ids, bound_ids, "dynamic-arrival fixture cleanup");
}

void test_live_hdrr_under_sustained_mixed_queues() {
    server_queue queue;
    queue.set_physical_slot_capacity(1);
    std::array<size_t, server_scheduler::lane_count> dispatches = {};
    std::unordered_set<int>                          live_ids;
    uint64_t                                         arrival = 1;

    const auto post_lane = [&](server_task::trusted_lane lane) {
        server_task task                 = make_user(queue, lane, arrival++);
        task.scheduling.predicted_gpu_us = 1000;
        const int task_id                = task.id;
        live_ids.insert(task_id);
        require_posted(queue.post(std::move(task)), task_id, "post sustained HDRR task");
    };
    for (int i = 0; i < 2; ++i) {
        post_lane(server_task::trusted_lane::low);
        post_lane(server_task::trusted_lane::normal);
        post_lane(server_task::trusted_lane::fast);
    }

    size_t total = 0;
    queue.on_new_task([&](server_task && selected) {
        const auto lane = selected.scheduling.lane;
        live_ids.erase(selected.id);
        ++dispatches[static_cast<size_t>(lane)];
        ++total;
        require(queue.bind_slot(selected.id, 0) && queue.release_slot(selected.id, 0),
                "sustained HDRR dispatch converts and releases one permit");
        if (total == 2100) {
            queue.terminate();
        } else {
            post_lane(lane);
        }
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(dispatches == std::array<size_t, server_scheduler::lane_count>({ 100, 400, 1600 }),
            "live start_loop preserves the exact 1:4:16 HDRR share under sustained load");

    std::vector<server_task> cancellations;
    for (int id : live_ids) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true) && queue.request_summary().active_requests == 0,
            "sustained HDRR fixture cleanup");
}

void test_live_hdrr_starts_sustained_wide_cohorts() {
    server_queue queue;
    queue.set_physical_slot_capacity(2);
    std::array<size_t, server_scheduler::lane_count> cohort_starts = {};
    std::unordered_set<int>                          live_ids;
    std::vector<std::pair<int, server_task::trusted_lane>> cohort;
    uint64_t                                         arrival = 1;

    const auto post_lane = [&](server_task::trusted_lane lane) {
        server_task task                 = make_user(queue, lane, arrival++);
        task.scheduling.predicted_gpu_us = 1000;
        const int task_id                = task.id;
        live_ids.insert(task_id);
        require_posted(queue.post(std::move(task)), task_id, "post sustained wide-cohort task");
    };
    for (int i = 0; i < 3; ++i) {
        post_lane(server_task::trusted_lane::low);
        post_lane(server_task::trusted_lane::normal);
        post_lane(server_task::trusted_lane::fast);
    }

    size_t cohorts = 0;
    queue.on_new_task([&](server_task && selected) {
        const auto lane = selected.scheduling.lane;
        live_ids.erase(selected.id);
        const int slot = static_cast<int>(cohort.size());
        require(slot < 2 && queue.bind_slot(selected.id, slot), "bind sustained wide-cohort request");
        cohort.emplace_back(selected.id, lane);
    });
    queue.on_update_slots([&] {
        require(cohort.size() == 2 && queue.dispatch_permits().total == 2,
                "each sustained wide cohort fills the two physical slots");
        const auto first  = static_cast<size_t>(cohort[0].second);
        const auto second = static_cast<size_t>(cohort[1].second);
        require(second >= first, "a live cohort stays on its starting lane or upgrades to a higher lane");
        ++cohort_starts[first];
        for (size_t slot = 0; slot < cohort.size(); ++slot) {
            require(queue.release_slot(cohort[slot].first, static_cast<int>(slot)),
                    "release sustained wide-cohort permit");
        }

        ++cohorts;
        if (cohorts == 210) {
            queue.terminate();
        } else {
            post_lane(cohort[0].second);
            post_lane(cohort[1].second);
        }
        cohort.clear();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(cohort_starts[static_cast<size_t>(server_task::trusted_lane::low)] != 0 &&
                cohort_starts[static_cast<size_t>(server_task::trusted_lane::normal)] != 0 &&
                cohort_starts[static_cast<size_t>(server_task::trusted_lane::fast)] != 0,
            "HDRR lets every lane begin a cohort under sustained two-slot load");

    std::vector<server_task> cancellations;
    for (int id : live_ids) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true) && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
            "sustained wide-cohort fixture cleanup");
}

void test_staggered_cohorts_drain_before_same_lane_refill() {
    struct active_slot {
        int id;
        int slot;
        server_task::trusted_lane lane;
    };

    server_queue queue;
    queue.set_physical_slot_capacity(2);
    std::array<size_t, server_scheduler::lane_count> cohort_starts = {};
    std::array<bool, 2>                              slot_busy = {};
    std::vector<active_slot>                         active;
    std::unordered_set<int>                          live_ids;
    uint64_t                                         arrival = 1;
    size_t                                           epochs = 0;
    bool                                             stopping = false;

    const auto post_lane = [&](server_task::trusted_lane lane) {
        server_task task                 = make_user(queue, lane, arrival++);
        task.scheduling.predicted_gpu_us = 1000;
        const int task_id                = task.id;
        live_ids.insert(task_id);
        require_posted(queue.post(std::move(task)), task_id, "post staggered cohort task");
    };
    for (int i = 0; i < 3; ++i) {
        post_lane(server_task::trusted_lane::low);
        post_lane(server_task::trusted_lane::normal);
        post_lane(server_task::trusted_lane::fast);
    }

    queue.on_new_task([&](server_task && selected) {
        if (active.empty()) {
            ++cohort_starts[static_cast<size_t>(selected.scheduling.lane)];
            ++epochs;
            if (epochs >= 210 && std::all_of(cohort_starts.begin(), cohort_starts.end(),
                                              [](size_t starts) { return starts >= 5; })) {
                stopping = true;
            }
            require(epochs <= 420, "staggered cohort fairness converges within a deterministic bound");
        }
        const auto free_slot = std::find(slot_busy.begin(), slot_busy.end(), false);
        require(free_slot != slot_busy.end(), "staggered cohort never exceeds two physical slots");
        const int slot = static_cast<int>(free_slot - slot_busy.begin());
        slot_busy[slot] = true;
        live_ids.erase(selected.id);
        require(queue.bind_slot(selected.id, slot), "bind staggered cohort member");
        active.push_back({ selected.id, slot, selected.scheduling.lane });
    });
    queue.on_update_slots([&] {
        require(!active.empty() && active.size() <= 2, "staggered update owns a bounded live cohort");
        const active_slot completed = active.front();
        active.erase(active.begin());
        require(queue.release_slot(completed.id, completed.slot), "release one staggered cohort member");
        slot_busy[completed.slot] = false;

        if (stopping) {
            if (active.empty()) {
                queue.terminate();
            }
        } else {
            post_lane(completed.lane);
        }
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(std::all_of(cohort_starts.begin(), cohort_starts.end(), [](size_t starts) { return starts >= 5; }),
            "every HDRR lane repeatedly begins cohorts under staggered continuous completion");

    std::vector<server_task> cancellations;
    for (int id : live_ids) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id_target = id;
        cancellations.push_back(std::move(cancel));
    }
    require(queue.post(std::move(cancellations), true) && queue.request_summary().active_requests == 0 &&
                queue.dispatch_permits().total == 0,
            "staggered cohort fixture drains and cancels without permit leaks");
}

void test_live_aging() {
    uint64_t     now_us = 20001;
    server_queue queue(test_runtime_config(), [&] { return now_us; });
    queue.set_physical_slot_capacity(1);
    server_task old                   = make_user(queue, server_task::trusted_lane::normal, 1);
    old.scheduling.virtual_runtime_us = 4000;
    const int old_id                  = old.id;
    require_posted(queue.post(std::move(old)), old_id, "post aged request");
    server_task fresh    = make_user(queue, server_task::trusted_lane::normal, 20000);
    const int   fresh_id = fresh.id;
    require_posted(queue.post(std::move(fresh)), fresh_id, "post fresh request");
    int selected_id = -1;
    queue.on_new_task([&](server_task && selected) {
        selected_id = selected.id;
        require(queue.bind_slot(selected.id, 0) && queue.release_slot(selected.id, 0),
                "aged request completes its permit");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    require(selected_id == old_id, "bounded aging promotes the old request ahead of fresh work");
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = queue.get_new_id();
    cancel.id_target = fresh_id;
    require(queue.post(std::move(cancel), true) && queue.request_summary().active_requests == 0,
            "aging fixture cleanup");
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
            require(queue.publish_slot_result(responses, selected.id, 0, server_queue_result_kind::partial,
                                              std::move(partial)),
                    "publish streaming response through durable terminal gate");
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
        { "parent family cancellation",   test_parent_cancellation_retires_family       },
        { "parent family failure",        test_parent_failure_retires_queued_family     },
        { "partial family launch failure", test_partial_family_launch_failure_retires_permits },
        { "bound parent family cancellation", test_bound_parent_cancellation_releases_family_slots },
        { "bound parent family failure",      test_bound_parent_failure_releases_family_slots      },
        { "atomic family width",               test_family_width_admission_is_atomic                 },
        { "final publication gate",       test_final_result_publication_deadline_gate   },
        { "stream publication gate",      test_stream_result_publication_deadline_gate  },
        { "concurrent publication expiry", test_concurrent_update_publication_after_expiry },
        { "publication scale isolation",  test_publication_gate_scales_without_global_sweep },
        { "live lane and physical permits", test_start_loop_enforces_live_lane_and_physical_permits },
        { "profiled live low widths",       test_start_loop_uses_profiled_exclusive_low_shapes      },
        { "fast arrival bypass",            test_fast_arrival_bypasses_ready_low_backlog            },
        { "sustained live HDRR",            test_live_hdrr_under_sustained_mixed_queues             },
        { "sustained wide HDRR cohorts",    test_live_hdrr_starts_sustained_wide_cohorts            },
        { "staggered draining HDRR cohorts", test_staggered_cohorts_drain_before_same_lane_refill   },
        { "live aging",                     test_live_aging                                         },
        { "bound stream timeout",         test_bound_stream_timeout_and_cancel_race     },
    };

    for (const auto & test : tests) {
        std::fprintf(stderr, "test-server-queue-runtime: %s\n", test.first);
        test.second();
    }
    std::fprintf(stderr, "test-server-queue-runtime: all %zu tests passed\n", tests.size());
    return 0;
}

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-queue.h"
#include "server-prefill.h"

#include <array>

#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
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

struct scoped_environment {
    explicit scoped_environment(const char * name) : name(name) {
        const char * value = std::getenv(name);
        if (value != nullptr) {
            had_value = true;
            prior     = value;
        }
    }

    ~scoped_environment() {
        const int result = had_value ? ::setenv(name.c_str(), prior.c_str(), 1) : ::unsetenv(name.c_str());
        (void) result;
    }

    void assign(const char * value) {
        const int result = value == nullptr ? ::unsetenv(name.c_str()) : ::setenv(name.c_str(), value, 1);
        require(result == 0, "set test environment");
    }

    std::string name;
    std::string prior;
    bool        had_value = false;
};

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

void set_cache_affinity(server_task & task, uint64_t saved_prefill_us, uint64_t restore_us = 0) {
    if (task.tokens.empty()) {
        task.tokens.push_back(1);
    }
    task.scheduling.cached_prompt_tokens       = 1;
    task.scheduling.predicted_prefill_us       = saved_prefill_us;
    task.scheduling.predicted_cache_restore_us = restore_us;
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

server_task_result_ptr make_error_result(int task_id, const char * message) {
    auto result      = std::make_unique<server_task_result_error>();
    result->id       = task_id;
    result->err_type = ERROR_TYPE_SERVER;
    result->err_msg  = message;
    return result;
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
            queue.defer(std::move(selected), server_queue_defer_reason::physical_admission);
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
    bool       saw_admission_wait = false;
    for (const auto & event : events.events) {
        saw_blocked |= event.queue_after == server_request_registry::queue_state::blocked;
        saw_admission_wait |= event.queue_after == server_request_registry::queue_state::blocked &&
                              event.reason == server_request_registry::reason_code::admission_wait;
    }
    require(saw_blocked, "defer records blocked durable state");
    require(saw_admission_wait, "physical admission defer publishes its exact durable reason");
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

void test_admin_cancel_requires_current_handle_and_is_idempotent_while_live() {
    {
        server_queue queue;
        server_response responses;
        server_task task = make_user(queue, server_task::trusted_lane::normal, 21);
        const int id = task.id;
        require_posted(queue.post(std::move(task)), id, "post queued admin-cancel task");
        const auto handle = queue.request_snapshot().front().handle;
        responses.add_waiting_task_id(id);

        require(queue.admin_cancel(responses, { handle.id + 1, handle.epoch }) ==
                        server_queue_admin_cancel_code::unknown_request,
                "unknown handle fails closed");
        require(queue.admin_cancel(responses, { handle.id, handle.epoch + 1 }) ==
                        server_queue_admin_cancel_code::stale_handle,
                "stale generation fails closed");
        require(queue.request_summary().active_requests == 1,
                "failed handle checks do not mutate the queued request");
        require(responses.recv_with_timeout({ id }, 0) == nullptr,
                "failed handle checks publish no terminal result");
        require(queue.admin_cancel(responses, handle) == server_queue_admin_cancel_code::accepted,
                "exact queued handle is cancelled");
        require(queue.request_summary().active_requests == 0,
                "queued admin cancellation retires durable state immediately");
        auto terminal = responses.recv_with_timeout({ id }, 0);
        const auto * error = dynamic_cast<const server_task_result_error *>(terminal.get());
        require(error != nullptr && error->err_type == ERROR_TYPE_UNAVAILABLE &&
                        error->err_msg == "request cancelled by dashboard operator",
                "queued admin cancellation publishes one operator terminal to its waiter");
        require(queue.admin_cancel(responses, handle) == server_queue_admin_cancel_code::unknown_request,
                "retired queued request fails closed on replay");
        require(responses.recv_with_timeout({ id }, 0) == nullptr,
                "retired replay publishes no duplicate terminal result");
    }

    {
        server_queue queue;
        server_response responses;
        server_task task = make_user(queue, server_task::trusted_lane::fast, 22);
        const int id = task.id;
        require_posted(queue.post(std::move(task)), id, "post bound admin-cancel task");
        const auto handle = queue.request_snapshot().front().handle;
        bind_only(queue, id, 0);
        responses.add_waiting_task_id(id);

        queue.set_prepare_terminal_control_hook_for_tests([](size_t attempt) {
            if (attempt == 1) {
                throw std::runtime_error("injected admin cancel node preparation failure");
            }
        });
        bool preparation_failed = false;
        try {
            (void) queue.admin_cancel(responses, handle);
        } catch (const std::runtime_error &) {
            preparation_failed = true;
        }
        require(preparation_failed && !queue.request_snapshot().front().cancel_requested,
                "admin cancel preparation failure leaves durable state untouched");
        queue.set_prepare_terminal_control_hook_for_tests({});

        responses.set_prepare_send_hook_for_tests([](size_t attempt) {
            if (attempt == 1) {
                throw std::runtime_error("injected admin cancel result preparation failure");
            }
        });
        preparation_failed = false;
        try {
            (void) queue.admin_cancel(responses, handle);
        } catch (const std::runtime_error &) {
            preparation_failed = true;
        }
        require(preparation_failed && !queue.request_snapshot().front().cancel_requested &&
                        responses.recv_with_timeout({ id }, 0) == nullptr,
                "admin result preparation failure leaves durable state and waiter untouched");
        responses.set_prepare_send_hook_for_tests({});

        require(queue.admin_cancel(responses, handle) == server_queue_admin_cancel_code::accepted,
                "exact bound handle requests cancellation");
        const auto cancelled = queue.request_snapshot();
        require(cancelled.size() == 1 && cancelled.front().cancel_requested &&
                        cancelled.front().last_reason == server_request_registry::reason_code::server_cancel,
                "bound operator cancellation is durable and attributed before control dispatch");
        auto terminal = responses.recv_with_timeout({ id }, 0);
        const auto * error = dynamic_cast<const server_task_result_error *>(terminal.get());
        require(error != nullptr && error->err_msg == "request cancelled by dashboard operator",
                "bound admin cancellation releases the original waiter");
        require(queue.admin_cancel(responses, handle) == server_queue_admin_cancel_code::already_requested,
                "duplicate bound cancellation is idempotent");
        require(responses.recv_with_timeout({ id }, 0) == nullptr,
                "duplicate bound cancellation publishes no second terminal");
        require(queue.release_slot(id, 0), "release admin-cancelled bound task");
        bool saw_operator_terminal = false;
        bool saw_operator_removal  = false;
        for (const auto & event : queue.request_events().events) {
            if (!(event.request == handle)) {
                continue;
            }
            saw_operator_terminal |=
                    event.kind == server_request_registry::event_kind::terminal &&
                    event.lifecycle_after == server_request_registry::lifecycle::cancelled &&
                    event.reason == server_request_registry::reason_code::server_cancel;
            saw_operator_removal |=
                    event.kind == server_request_registry::event_kind::removed &&
                    event.lifecycle_after == server_request_registry::lifecycle::absent &&
                    event.reason == server_request_registry::reason_code::server_cancel;
        }
        require(saw_operator_terminal && saw_operator_removal,
                "operator provenance labels the terminal and removal events themselves");
    }

    {
        server_queue queue;
        server_response responses;
        server_task task = make_user(queue, server_task::trusted_lane::low, 23);
        const int id = task.id;
        require_posted(queue.post(std::move(task)), id, "post terminal admin-cancel task");
        const auto handle = queue.request_snapshot().front().handle;
        bind_only(queue, id, 0);
        require(queue.fail_task(id), "terminalize bound request before admin cancellation");
        require(queue.admin_cancel(responses, handle) == server_queue_admin_cancel_code::terminal_request,
                "terminal request fails closed without another mutation");
        require(queue.release_slot(id, 0), "release terminal admin-cancel fixture");
    }
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

void test_external_cancel_preparation_is_fault_atomic() {
    {
        server_queue queue;
        server_task  task = make_user(queue, server_task::trusted_lane::normal, 35);
        const int    task_id = task.id;
        require_posted(queue.post(std::move(task)), task_id, "post single cancel fault fixture");

        std::vector<bound_family_slot> bound;
        bool cancellation_attempted = false;
        bool preparation_failed     = false;
        int  cancel_controls        = 0;
        queue.on_new_task([&](server_task && selected) {
            if (selected.type == SERVER_TASK_TYPE_CANCEL) {
                ++cancel_controls;
                require(selected.id_target == task_id && bound.size() == 1,
                        "retried single cancellation dispatches its original target once");
                bound.front().release();
                queue.terminate();
                return;
            }
            require(selected.id == task_id && queue.bind_slot(task_id, 0),
                    "bind single cancel fault fixture");
            bound.emplace_back(queue, 0, std::move(selected));
        });
        queue.on_update_slots([&] {
            if (cancellation_attempted) {
                return;
            }
            cancellation_attempted = true;
            queue.set_prepare_terminal_control_hook_for_tests([](size_t attempt) {
                if (attempt == 1) {
                    throw std::runtime_error("injected single cancel node preparation failure");
                }
            });
            try {
                server_task cancel(SERVER_TASK_TYPE_CANCEL);
                cancel.id        = queue.get_new_id();
                cancel.id_target = task_id;
                queue.post(std::move(cancel), true);
            } catch (const std::runtime_error &) {
                preparation_failed = true;
            }
            require(preparation_failed && cancel_controls == 0 &&
                        queue.request_summary().active_requests == 1 &&
                        queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] == 1 &&
                        bound.size() == 1 && bound.front().task,
                    "single cancel allocation fault preserves durable binding, permit, and live payload");

            queue.set_prepare_terminal_control_hook_for_tests({});
            server_task retry(SERVER_TASK_TYPE_CANCEL);
            retry.id        = queue.get_new_id();
            retry.id_target = task_id;
            require(queue.post(std::move(retry), true), "retry prepared single cancellation");
        });
        queue.on_sleeping_state([](bool) {});
        queue.start_loop();
        require(preparation_failed && cancel_controls == 1 &&
                    queue.request_summary().active_requests == 0 &&
                    queue.dispatch_permits().total == 0,
                "single cancel retry retires exactly once without a permit leak");
    }

    {
        server_queue queue;
        queue.set_physical_slot_capacity(1);
        std::array<int, 2> task_ids;
        for (size_t i = 0; i < task_ids.size(); ++i) {
            server_task task = make_user(queue, server_task::trusted_lane::normal, 40 + i);
            task_ids[i] = task.id;
            require_posted(queue.post(std::move(task)), task_ids[i], "post vector cancel fault fixture");
        }

        std::vector<bound_family_slot> bound;
        bool cancellation_attempted = false;
        bool preparation_failed     = false;
        int  user_dispatches        = 0;
        int  cancel_controls        = 0;
        queue.on_new_task([&](server_task && selected) {
            if (selected.type == SERVER_TASK_TYPE_CANCEL) {
                ++cancel_controls;
                if (selected.id_target == task_ids[0]) {
                    require(bound.size() == 1, "vector retry retains the original bound payload");
                    bound.front().release();
                } else {
                    require(selected.id_target == task_ids[1], "vector retry preserves cancellation order");
                }
                if (cancel_controls == 2) {
                    queue.terminate();
                }
                return;
            }

            ++user_dispatches;
            require(selected.id == task_ids[0] && queue.bind_slot(selected.id, 0),
                    "physical cap binds only the first vector cancel fixture");
            bound.emplace_back(queue, 0, std::move(selected));
        });
        queue.on_update_slots([&] {
            if (cancellation_attempted) {
                return;
            }
            cancellation_attempted = true;
            queue.set_prepare_terminal_control_hook_for_tests([](size_t attempt) {
                if (attempt == 2) {
                    throw std::runtime_error("injected Nth vector cancel node preparation failure");
                }
            });

            auto make_cancellations = [&]() {
                std::vector<server_task> cancellations;
                for (int id_target : task_ids) {
                    server_task cancel(SERVER_TASK_TYPE_CANCEL);
                    cancel.id_target = id_target;
                    cancellations.push_back(std::move(cancel));
                }
                return cancellations;
            };
            try {
                queue.post(make_cancellations(), true);
            } catch (const std::runtime_error &) {
                preparation_failed = true;
            }
            const auto snapshot = queue.request_snapshot();
            const bool queued_payload_preserved = std::any_of(snapshot.begin(), snapshot.end(), [&](const auto & request) {
                return request.handle.id == static_cast<uint64_t>(task_ids[1]) + 1 &&
                       request.queue != server_request_registry::queue_state::none;
            });
            require(preparation_failed && cancel_controls == 0 && user_dispatches == 1 &&
                        queue.request_summary().active_requests == task_ids.size() &&
                        queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] == 1 &&
                        queued_payload_preserved,
                    "Nth vector cancel fault preserves bound and queued durable payloads");

            queue.set_prepare_terminal_control_hook_for_tests({});
            require(queue.post(make_cancellations(), true), "retry fully prepared cancellation vector");
        });
        queue.on_sleeping_state([](bool) {});
        queue.start_loop();
        require(preparation_failed && user_dispatches == 1 && cancel_controls == 2 &&
                    queue.request_summary().active_requests == 0 &&
                    queue.dispatch_permits().total == 0,
                "vector cancel retry suppresses queued dispatch and retires each target once");
    }
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
        queue.set_expiration_response(responses);
        queue.on_request_expired([&](server_queue_expiration) { ++expiration_callbacks; });

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

void test_staged_family_publication_is_exception_safe_and_at_most_once() {
    server_queue    queue;
    server_response responses;
    server_task     parent = make_family(queue, server_task::trusted_lane::normal, 2500);
    const std::array<int, 3> family_ids = {
        parent.id,
        parent.child_tasks[0].id,
        parent.child_tasks[1].id,
    };
    responses.add_waiting_task_ids(std::unordered_set<int>(family_ids.begin(), family_ids.end()));
    std::vector<server_task> family_tasks;
    family_tasks.push_back(std::move(parent));
    require_posted(queue.post(std::move(family_tasks)), family_ids[0], "post publication-failure family");

    queue.on_new_task([&](server_task && selected) {
        require(selected.id == family_ids[0] && selected.child_tasks.size() == 2,
                "publication-failure fixture dispatches the complete family");
        for (size_t i = 0; i < selected.child_tasks.size(); ++i) {
            require(queue.bind_slot(selected.child_tasks[i].id, static_cast<int>(i + 1)),
                    "bind publication-failure child");
        }
        require(queue.bind_slot(selected.id, 0), "bind publication-failure parent");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    server_task unrelated = make_user(queue, server_task::trusted_lane::normal, 2501);
    const int unrelated_id = unrelated.id;
    responses.add_waiting_task_id(unrelated_id);
    require_posted(queue.post(std::move(unrelated)), unrelated_id, "post unrelated publication probe");
    bind_only(queue, unrelated_id, 3);

    std::array<server_prefill::staged_member_disposition, 3> disposition;
    for (auto & member : disposition) {
        member.defer_prompt_metric();
    }
    int prompt_metrics     = 0;
    int prediction_metrics = 0;

    responses.set_prepare_send_hook_for_tests([](size_t attempt) {
        if (attempt == 2) {
            throw std::runtime_error("injected pre-gate response reservation failure");
        }
    });

    require(queue.publish_slot_result(
                responses, family_ids[0], 0, server_queue_result_kind::final,
                make_final_result(SERVER_TASK_TYPE_COMPLETION, family_ids[0])),
            "first family final reaches the real response queue");
    disposition[0].record_terminal_handoff();
    prompt_metrics += disposition[0].take_prompt_metric();
    prediction_metrics += disposition[0].take_prediction_metric();

    bool second_send_failed = false;
    try {
        queue.publish_slot_result(
            responses, family_ids[1], 1, server_queue_result_kind::final,
            make_final_result(SERVER_TASK_TYPE_COMPLETION, family_ids[1]));
    } catch (const std::runtime_error &) {
        second_send_failed = true;
    }
    require(second_send_failed && disposition[1].may_publish_failure_terminal(),
            "Nth send fails before the durable terminal gate is consumed");

    int error_terminals = 0;
    for (size_t i = 0; i < family_ids.size(); ++i) {
        if (!disposition[i].may_publish_failure_terminal()) {
            continue;
        }
        require(queue.fail_task_with_result(
                    responses, family_ids[i], make_error_result(family_ids[i], "staged family failed")),
                "undisposed family member accepts one failure terminal");
        disposition[i].record_terminal_handoff();
        ++error_terminals;
    }
    require(error_terminals == 2 && prompt_metrics == 1 && prediction_metrics == 1,
            "earlier accepted final keeps exactly-once metrics while failed siblings get one error");

    require(queue.publish_slot_result(
                responses, unrelated_id, 3, server_queue_result_kind::final,
                make_final_result(SERVER_TASK_TYPE_COMPLETION, unrelated_id)),
            "unrelated bound request remains publishable after family handoff failure");

    for (size_t i = 0; i < family_ids.size(); ++i) {
        auto terminal = responses.recv_with_timeout({ family_ids[i] }, 0);
        require(terminal != nullptr && terminal->is_stop() &&
                    terminal->is_error() == (i != 0) &&
                    responses.recv_with_timeout({ family_ids[i] }, 0) == nullptr,
                "each family task receives at most one terminal result");
    }
    auto unrelated_terminal = responses.recv_with_timeout({ unrelated_id }, 0);
    require(unrelated_terminal != nullptr && !unrelated_terminal->is_error() && unrelated_terminal->is_stop() &&
                responses.recv_with_timeout({ unrelated_id }, 0) == nullptr,
            "unrelated request receives one unaffected final");

    int cancellation_controls = 0;
    queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_CANCEL,
                "failed handoff cleanup dispatches only family cancellation controls");
        const auto found = std::find(family_ids.begin(), family_ids.end(), selected.id_target);
        require(found != family_ids.end(), "family cancellation targets a known child");
        const int slot = static_cast<int>(std::distance(family_ids.begin(), found));
        require(queue.release_slot(selected.id_target, slot), "family cancellation releases failed bound member");
        if (++cancellation_controls == 2) {
            queue.terminate();
        }
    });
    queue.start_loop();
    require(queue.request_summary().active_requests == 0 && queue.dispatch_permits().total == 0,
            "exception-safe family publication retires all durable state");

    server_queue    release_queue;
    server_response release_responses;
    server_task     release_parent = make_family(release_queue, server_task::trusted_lane::normal, 2600);
    const std::array<int, 3> release_ids = {
        release_parent.id,
        release_parent.child_tasks[0].id,
        release_parent.child_tasks[1].id,
    };
    release_responses.add_waiting_task_ids(std::unordered_set<int>(release_ids.begin(), release_ids.end()));
    std::vector<server_task> release_tasks;
    release_tasks.push_back(std::move(release_parent));
    require_posted(release_queue.post(std::move(release_tasks)), release_ids[0],
                   "post release-failure family");
    release_queue.on_new_task([&](server_task && selected) {
        for (size_t i = 0; i < selected.child_tasks.size(); ++i) {
            require(release_queue.bind_slot(selected.child_tasks[i].id, static_cast<int>(i + 1)),
                    "bind release-failure child");
        }
        require(release_queue.bind_slot(selected.id, 0), "bind release-failure parent");
        release_queue.terminate();
    });
    release_queue.on_update_slots([] {});
    release_queue.on_sleeping_state([](bool) {});
    release_queue.start_loop();

    std::array<server_prefill::staged_member_disposition, 3> release_disposition;
    int release_prediction_metrics = 0;
    for (size_t i = 0; i < release_ids.size(); ++i) {
        require(release_queue.publish_slot_result(
                    release_responses, release_ids[i], static_cast<int>(i), server_queue_result_kind::final,
                    make_final_result(SERVER_TASK_TYPE_COMPLETION, release_ids[i])),
                "all finals enqueue before injected release callback failure");
        release_disposition[i].record_terminal_handoff();
        release_prediction_metrics += release_disposition[i].take_prediction_metric();
    }

    bool release_callback_failed = false;
    try {
        throw std::runtime_error("injected post-final slot release callback failure");
    } catch (const std::runtime_error &) {
        release_callback_failed = true;
    }
    int duplicate_error_attempts = 0;
    for (const auto & member : release_disposition) {
        duplicate_error_attempts += member.may_publish_failure_terminal();
    }
    require(release_callback_failed && duplicate_error_attempts == 0 && release_prediction_metrics == 3,
            "post-final release failure preserves metrics and suppresses every duplicate error terminal");
    for (int id : release_ids) {
        auto terminal = release_responses.recv_with_timeout({ id }, 0);
        require(terminal != nullptr && !terminal->is_error() && terminal->is_stop() &&
                    release_responses.recv_with_timeout({ id }, 0) == nullptr,
                "release callback failure leaves exactly one final per family member");
    }
    require(release_queue.request_summary().active_requests == 0 &&
                release_queue.dispatch_permits().total == 0,
            "accepted family finals retire durable records before slot release callback");
}

void test_terminal_expiry_and_failure_preparation_are_atomic() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 10;

    uint64_t        now_us = 6000;
    server_queue    expiry_queue(config, [&] { return now_us; });
    server_response expiry_responses;
    expiry_queue.set_expiration_response(expiry_responses);
    int expiry_callbacks = 0;
    expiry_queue.on_request_expired([&](server_queue_expiration) {
        ++expiry_callbacks;
        throw std::runtime_error("injected post-terminal expiration observer failure");
    });

    server_task expiring                 = make_user(expiry_queue, server_task::trusted_lane::normal, 0);
    expiring.scheduling.queue_timeout_us = 0;
    expiring.scheduling.run_timeout_us   = 0;
    const int expiring_id                = expiring.id;
    expiry_responses.add_waiting_task_id(expiring_id);
    require_posted(expiry_queue.post(std::move(expiring)), expiring_id,
                   "post atomic expiry preparation task");
    bind_only(expiry_queue, expiring_id, 0);
    now_us = 6010;

    expiry_responses.set_prepare_send_hook_for_tests([](size_t attempt) {
        if (attempt == 1) {
            throw std::runtime_error("injected timeout response reservation failure");
        }
    });
    bool timeout_prepare_failed = false;
    try {
        expiry_queue.publish_slot_result(
            expiry_responses, expiring_id, 0, server_queue_result_kind::final,
            make_final_result(SERVER_TASK_TYPE_COMPLETION, expiring_id));
    } catch (const std::runtime_error &) {
        timeout_prepare_failed = true;
    }
    require(timeout_prepare_failed && expiry_callbacks == 0 &&
                expiry_queue.request_summary().active_requests == 1 &&
                expiry_queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] == 1 &&
                expiry_responses.recv_with_timeout({ expiring_id }, 0) == nullptr,
            "timeout result reservation failure leaves durable request and slot untouched");

    require(!expiry_queue.fail_task_with_result(
                expiry_responses, expiring_id, make_error_result(expiring_id, "must lose to timeout")),
            "exact-deadline failure result yields to prepared timeout terminal");
    require_timeout_only(expiry_responses, expiring_id,
                         "expired failure path emits one timeout and no SERVER terminal");
    require(!expiry_queue.fail_task_with_result(
                expiry_responses, expiring_id, make_error_result(expiring_id, "must stay timed out")) &&
                expiry_responses.recv_with_timeout({ expiring_id }, 0) == nullptr,
            "failure after a completed timeout sweep cannot publish a SERVER terminal");
    require(expiry_callbacks == 1 && expiry_queue.release_slot(expiring_id, 0),
            "prepared timeout retries and releases its bound slot once");
    expiry_queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_CANCEL && selected.id_target == expiring_id,
                "prepared timeout queues one allocation-free cancellation control");
        expiry_queue.terminate();
    });
    expiry_queue.on_update_slots([] {});
    expiry_queue.on_sleeping_state([](bool) {});
    expiry_queue.start_loop();
    require(expiry_queue.request_summary().active_requests == 0,
            "timeout/failure race retires without duplicate durable state");

    now_us = 7000;
    server_queue    failure_queue(config, [&] { return now_us; });
    server_response failure_responses;
    server_task     failing                 = make_user(failure_queue, server_task::trusted_lane::normal, 0);
    failing.scheduling.queue_timeout_us     = 0;
    failing.scheduling.run_timeout_us       = 100;
    const int failing_id                    = failing.id;
    failure_responses.add_waiting_task_id(failing_id);
    require_posted(failure_queue.post(std::move(failing)), failing_id,
                   "post failure-result retry task");
    bind_only(failure_queue, failing_id, 0);
    failure_responses.set_prepare_send_hook_for_tests([](size_t attempt) {
        if (attempt == 1) {
            throw std::runtime_error("injected failure-result reservation failure");
        }
    });

    bool failure_prepare_failed = false;
    try {
        failure_queue.fail_task_with_result(
            failure_responses, failing_id, make_error_result(failing_id, "first failure attempt"));
    } catch (const std::runtime_error &) {
        failure_prepare_failed = true;
    }
    require(failure_prepare_failed && failure_queue.request_summary().active_requests == 1 &&
                failure_queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] == 1 &&
                failure_responses.recv_with_timeout({ failing_id }, 0) == nullptr,
            "failure-result reserve fault cannot release or falsely complete the active record");

    require(failure_queue.fail_task_with_result(
                failure_responses, failing_id, make_error_result(failing_id, "retried failure terminal")),
            "retained failure result retries through the same pre-terminal preparation path");
    require(!failure_queue.fail_task_with_result(
                failure_responses, failing_id, make_error_result(failing_id, "duplicate failure terminal")),
            "a live but already-failed binding rejects a second terminal handoff");
    auto error = failure_responses.recv_with_timeout({ failing_id }, 0);
    require(error != nullptr && error->is_error() && error->is_stop() &&
                failure_responses.recv_with_timeout({ failing_id }, 0) == nullptr,
            "failure retry emits exactly one SERVER terminal");
    failure_queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_CANCEL && selected.id_target == failing_id,
                "failure retry queues its preallocated cancellation control");
        require(failure_queue.release_slot(failing_id, 0),
                "failure retry cancellation releases failed binding");
        failure_queue.terminate();
    });
    failure_queue.on_update_slots([] {});
    failure_queue.on_sleeping_state([](bool) {});
    failure_queue.start_loop();
    require(failure_queue.request_summary().active_requests == 0 &&
                failure_queue.dispatch_permits().total == 0,
            "failure-result retry retires without false completion or permit leak");
    failure_responses.set_prepare_send_hook_for_tests([](size_t) {
        throw std::runtime_error("stale failure must not prepare response storage");
    });
    require(!failure_queue.fail_task_with_result(
                failure_responses, failing_id, make_error_result(failing_id, "stale failure terminal")) &&
                failure_responses.recv_with_timeout({ failing_id }, 0) == nullptr,
            "a stale waiting ID cannot resurrect a removed durable request");
    const int missing_id = failure_queue.get_new_id();
    failure_responses.add_waiting_task_id(missing_id);
    require(!failure_queue.fail_task_with_result(
                failure_responses, missing_id, make_error_result(missing_id, "missing failure terminal")) &&
                failure_responses.recv_with_timeout({ missing_id }, 0) == nullptr,
            "a missing durable request cannot publish through a waiting response ID");

    now_us = 8000;
    server_queue    batch_queue(config, [&] { return now_us; });
    server_response batch_responses;
    batch_queue.set_expiration_response(batch_responses);
    int batch_expiry_callbacks = 0;
    batch_queue.on_request_expired([&](server_queue_expiration) { ++batch_expiry_callbacks; });

    std::array<int, 2> batch_ids;
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        server_task task               = make_user(batch_queue, server_task::trusted_lane::normal, 0);
        task.scheduling.queue_timeout_us = 0;
        task.scheduling.run_timeout_us   = 10;
        batch_ids[i] = task.id;
        batch_responses.add_waiting_task_id(task.id);
        require_posted(batch_queue.post(std::move(task)), batch_ids[i],
                       "post batched timeout preparation task");
    }
    size_t batch_bound = 0;
    batch_queue.on_new_task([&](server_task && selected) {
        require(batch_bound < batch_ids.size() && selected.id == batch_ids[batch_bound],
                "batch timeout fixture dispatches in request order");
        require(batch_queue.bind_slot(selected.id, static_cast<int>(batch_bound)),
                "bind batched timeout fixture");
        if (++batch_bound == batch_ids.size()) {
            batch_queue.terminate();
        }
    });
    batch_queue.on_update_slots([] {});
    batch_queue.on_sleeping_state([](bool) {});
    batch_queue.start_loop();

    now_us = 8010;
    batch_responses.set_prepare_send_hook_for_tests([](size_t attempt) {
        if (attempt == 2) {
            throw std::runtime_error("injected second batched timeout preparation failure");
        }
    });
    bool batch_prepare_failed = false;
    try {
        batch_queue.expire_requests();
    } catch (const std::runtime_error &) {
        batch_prepare_failed = true;
    }
    require(batch_prepare_failed && batch_expiry_callbacks == 0 &&
                batch_queue.request_summary().active_requests == batch_ids.size() &&
                batch_queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] ==
                    batch_ids.size(),
            "Nth batched timeout preparation failure leaves every durable request untouched");
    for (int id_task : batch_ids) {
        require(batch_responses.recv_with_timeout({ id_task }, 0) == nullptr,
                "failed timeout batch publishes no partial prefix");
    }

    require(batch_queue.expire_requests() == batch_ids.size() &&
                batch_expiry_callbacks == static_cast<int>(batch_ids.size()),
            "fully prepared timeout batch commits every expiration together");
    require(!batch_queue.fail_task_with_result(
                batch_responses, batch_ids[0], make_error_result(batch_ids[0], "failure after timeout sweep")),
            "failure after a timeout sweep cannot claim the still-bound terminal record");
    for (size_t i = 0; i < batch_ids.size(); ++i) {
        require_timeout_only(batch_responses, batch_ids[i],
                             "retried timeout batch emits one timeout per request");
        require(batch_queue.release_slot(batch_ids[i], static_cast<int>(i)),
                "release each expired batched binding");
    }
    require(batch_queue.request_summary().active_requests == 0 &&
                batch_queue.dispatch_permits().total == 0,
            "batched timeout retry retires without partial state or permit leaks");
}

void test_family_failure_prepares_all_results_before_live_cancel() {
    server_queue    queue;
    server_response responses;
    server_task     parent = make_family(queue, server_task::trusted_lane::normal, 5900);
    const int       parent_id = parent.id;
    std::array<int, 3> family_ids = {
        parent_id,
        parent.child_tasks[0].id,
        parent.child_tasks[1].id,
    };
    for (int id_task : family_ids) {
        responses.add_waiting_task_id(id_task);
    }
    std::vector<server_task> tasks;
    tasks.push_back(std::move(parent));
    require(queue.post(std::move(tasks)), "post family for atomic failure preparation");
    server_request_registry::request_handle protected_handle;
    for (const auto & request : queue.request_snapshot()) {
        if (request.handle.id == static_cast<uint64_t>(family_ids[1]) + 1) {
            protected_handle = request.handle;
        }
    }
    require(protected_handle.id != 0, "capture protected child generation for admin cancellation");

    std::vector<bound_family_slot> bound;
    int update_turns   = 0;
    int cancel_controls = 0;
    int error_terminals = 0;

    auto make_family_errors = [&]() {
        std::vector<server_task_result_ptr> results;
        results.reserve(family_ids.size());
        for (int id_task : family_ids) {
            results.push_back(make_error_result(id_task, "atomic family failure"));
        }
        return results;
    };

    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++cancel_controls;
            require(update_turns == 2 && selected.id_target == parent_id,
                    "family release control cannot dispatch before the full response retry");
            require(release_server_task_family_slots(bound, parent_id) == family_ids.size(),
                    "one family control releases every failed member");
            require(queue.request_summary().active_requests == 0 &&
                        queue.dispatch_permits().total == 0,
                    "family failure control retires every durable binding and permit");
            queue.terminate();
            return;
        }

        require(selected.id == parent_id && selected.child_tasks.size() == 2,
                "atomic failure fixture dispatches the complete three-member family");
        int id_slot = 0;
        for (server_task & child : selected.child_tasks) {
            require(queue.bind_slot(child.id, id_slot), "bind atomic failure child");
            bound.emplace_back(queue, id_slot++, std::move(child));
        }
        require(queue.bind_slot(selected.id, id_slot), "bind atomic failure parent");
        bound.emplace_back(queue, id_slot, std::move(selected));
    });
    queue.on_update_slots([&] {
        ++update_turns;
        if (update_turns == 1) {
            responses.set_prepare_send_hook_for_tests([](size_t attempt) {
                if (attempt == 2) {
                    throw std::runtime_error("injected Nth family failure preparation fault");
                }
            });
            bool preparation_failed = false;
            try {
                queue.fail_family_with_results(responses, parent_id, make_family_errors());
            } catch (const std::runtime_error &) {
                preparation_failed = true;
            }
            require(preparation_failed && cancel_controls == 0 &&
                        queue.request_summary().active_requests == family_ids.size() &&
                        queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] ==
                            family_ids.size(),
                    "Nth family result fault leaves all members bound and no control visible");
            for (int id_task : family_ids) {
                require(responses.recv_with_timeout({ id_task }, 0) == nullptr,
                        "failed family preparation publishes no terminal prefix");
            }

            server_task external_cancel(SERVER_TASK_TYPE_CANCEL);
            external_cancel.id        = queue.get_new_id();
            external_cancel.id_target = family_ids[1];
            require(queue.post(std::move(external_cancel), true) && cancel_controls == 0 &&
                        queue.request_summary().active_requests == family_ids.size() &&
                        queue.dispatch_permits().bound[static_cast<size_t>(server_scheduler::lane::normal)] ==
                            family_ids.size(),
                    "external child CANCEL canonicalizes behind the protected family terminal retry");
            server_task duplicate_parent_cancel(SERVER_TASK_TYPE_CANCEL);
            duplicate_parent_cancel.id        = queue.get_new_id();
            duplicate_parent_cancel.id_target = parent_id;
            require(queue.post(std::move(duplicate_parent_cancel), true),
                    "protected parent CANCEL deduplicates against the canonical child control");
            const auto admin_cancel = queue.admin_cancel(responses, protected_handle);
            const auto after_admin_cancel = queue.request_snapshot();
            const auto protected_request = std::find_if(
                    after_admin_cancel.begin(), after_admin_cancel.end(), [&](const auto & request) {
                        return request.handle == protected_handle;
                    });
            require(admin_cancel == server_queue_admin_cancel_code::terminal_request &&
                        protected_request != after_admin_cancel.end() &&
                        !protected_request->cancel_requested &&
                        responses.recv_with_timeout({ family_ids[1] }, 0) == nullptr,
                    "admin cancel fails closed while a protected family terminal owns publication");
            responses.set_prepare_send_hook_for_tests({});
            queue.request_update();
            return;
        }

        require(update_turns == 2 &&
                    queue.fail_family_with_results(responses, parent_id, make_family_errors()) == family_ids.size(),
                "next live update commits the fully prepared family failure");
        for (int id_task : family_ids) {
            auto terminal = responses.recv_with_timeout({ id_task }, 0);
            require(terminal != nullptr && terminal->is_error() && terminal->is_stop() &&
                        responses.recv_with_timeout({ id_task }, 0) == nullptr,
                    "family retry publishes exactly one terminal per member");
            ++error_terminals;
        }
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(update_turns == 2 && cancel_controls == 1 && error_terminals == 3,
            "live family retry orders all terminals before one release control");
}

void test_stream_result_publication_deadline_gate() {
    auto config                     = test_runtime_config();
    config.default_queue_timeout_us = 100;
    config.default_run_timeout_us   = 10;
    uint64_t        now_us          = 3000;
    server_queue    queue(config, [&] { return now_us; });
    server_response responses;
    int             expiration_callbacks = 0;
    queue.set_expiration_response(responses);
    queue.on_request_expired([&](server_queue_expiration) { ++expiration_callbacks; });

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

    queue.set_expiration_response(responses);
    queue.on_request_expired([&](server_queue_expiration) { ++expiration_callbacks; });
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

void test_live_cache_affinity_uses_trusted_costs() {
    auto config = test_runtime_config();
    config.scheduler.aging_credit_us     = 1;
    config.scheduler.max_aging_credit_us = 1;
    server_queue queue(config, [] { return 1; });
    queue.set_physical_slot_capacity(1);

    std::vector<int> all_ids;
    server_task miss                   = make_user(queue, server_task::trusted_lane::normal, 1);
    miss.scheduling.virtual_runtime_us = 500;
    all_ids.push_back(miss.id);
    require_posted(queue.post(std::move(miss)), all_ids.back(), "post cache miss");

    server_task costly_restore                   = make_user(queue, server_task::trusted_lane::normal, 1);
    costly_restore.scheduling.virtual_runtime_us = 500;
    set_cache_affinity(costly_restore, 1000000, 1000000);
    all_ids.push_back(costly_restore.id);
    require_posted(queue.post(std::move(costly_restore)), all_ids.back(), "post costly cache restore");

    server_task hit                   = make_user(queue, server_task::trusted_lane::normal, 1);
    hit.scheduling.virtual_runtime_us = 500;
    set_cache_affinity(hit, 1000000, 1);
    const int hit_id = hit.id;
    all_ids.push_back(hit_id);
    require_posted(queue.post(std::move(hit)), hit_id, "post useful cache hit");

    const auto snapshot = queue.request_snapshot();
    require(std::any_of(snapshot.begin(), snapshot.end(), [](const auto & request) {
                return request.counts.cached_prompt_tokens == 1 &&
                       request.estimates.predicted_cache_restore_us == 1000000;
            }),
            "trusted cache count and restore estimate reach durable metadata");

    int selected_id = -1;
    queue.on_new_task([&](server_task && selected) {
        selected_id = selected.id;
        require(queue.bind_slot(selected.id, 0) && queue.release_slot(selected.id, 0),
                "cache-affinity selection completes its permit");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    require(selected_id == hit_id, "net cache time saved reorders feasible work within one lane");
    all_ids.erase(std::find(all_ids.begin(), all_ids.end(), selected_id));
    retire_bound_and_queued(queue, all_ids, {}, "trusted cache cost fixture cleanup");
}

void test_live_cache_affinity_is_lane_and_cohort_bounded() {
    server_queue queue;
    queue.set_physical_slot_capacity(4);
    std::vector<int> all_ids;
    std::vector<int> fast_ids;
    for (int i = 0; i < 2; ++i) {
        server_task fast = make_user(queue, server_task::trusted_lane::fast, i + 1);
        fast_ids.push_back(fast.id);
        all_ids.push_back(fast.id);
        require_posted(queue.post(std::move(fast)), all_ids.back(), "post fast cohort miss");
    }
    server_task normal_hit = make_user(queue, server_task::trusted_lane::normal, 3);
    set_cache_affinity(normal_hit, 1000000);
    all_ids.push_back(normal_hit.id);
    require_posted(queue.post(std::move(normal_hit)), all_ids.back(), "post normal cache hit");

    std::vector<int> bound_ids;
    queue.on_new_task([&](server_task && selected) {
        require(selected.scheduling.lane == server_task::trusted_lane::fast,
                "cache affinity cannot cross the HDRR-selected lane");
        const int slot = static_cast<int>(bound_ids.size());
        require(queue.bind_slot(selected.id, slot), "bind cache-isolation cohort member");
        bound_ids.push_back(selected.id);
    });
    queue.on_update_slots([&] {
        require(bound_ids == fast_ids && queue.dispatch_permits().total == 2,
                "cache affinity preserves the fast-lane durable cohort cap");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
    retire_bound_and_queued(queue, all_ids, bound_ids, "cache lane-isolation fixture cleanup");
}

void test_live_cache_lookahead_and_bypass_bound() {
    auto config = test_runtime_config();
    config.scheduler.aging_credit_us     = 1;
    config.scheduler.max_aging_credit_us = 1;

    {
        server_queue queue(config, [] { return 1; });
        queue.set_physical_slot_capacity(1);
        std::vector<int> all_ids;
        for (int i = 0; i < 9; ++i) {
            server_task task                   = make_user(queue, server_task::trusted_lane::low, 1);
            task.scheduling.virtual_runtime_us = 500;
            if (i == 8) {
                set_cache_affinity(task, 1000000);
            }
            all_ids.push_back(task.id);
            require_posted(queue.post(std::move(task)), all_ids.back(), "post cache-lookahead task");
        }
        int selected_id = -1;
        queue.on_new_task([&](server_task && selected) {
            selected_id = selected.id;
            require(queue.bind_slot(selected.id, 0) && queue.release_slot(selected.id, 0),
                    "lookahead selection completes its permit");
            queue.terminate();
        });
        queue.on_update_slots([] {});
        queue.on_sleeping_state([](bool) {});
        queue.start_loop();
        require(selected_id == all_ids.front(), "cache affinity cannot inspect beyond K=8 eligible requests");
        all_ids.erase(std::find(all_ids.begin(), all_ids.end(), selected_id));
        retire_bound_and_queued(queue, all_ids, {}, "cache lookahead fixture cleanup");
    }

    {
        server_queue queue(config, [] { return 1; });
        queue.set_physical_slot_capacity(1);
        std::vector<int> all_ids;
        server_task miss                   = make_user(queue, server_task::trusted_lane::normal, 1);
        miss.scheduling.virtual_runtime_us = 500;
        const int miss_id                  = miss.id;
        all_ids.push_back(miss_id);
        require_posted(queue.post(std::move(miss)), miss_id, "post bypass-protected miss");

        const auto post_hit = [&] {
            server_task hit                   = make_user(queue, server_task::trusted_lane::normal, 1);
            hit.scheduling.virtual_runtime_us = 500;
            set_cache_affinity(hit, 1000000);
            all_ids.push_back(hit.id);
            require_posted(queue.post(std::move(hit)), all_ids.back(), "post continuous cache hit");
        };
        post_hit();

        std::vector<int> order;
        queue.on_new_task([&](server_task && selected) {
            order.push_back(selected.id);
            require(queue.bind_slot(selected.id, 0) && queue.release_slot(selected.id, 0),
                    "bypass-bound selection completes its permit");
            if (selected.id == miss_id) {
                queue.terminate();
            } else {
                post_hit();
            }
        });
        queue.on_update_slots([] {});
        queue.on_sleeping_state([](bool) {});
        queue.start_loop();
        require(order.size() == 4 && order.back() == miss_id,
                "three cache bypasses protect an old feasible miss from starvation");
        const std::unordered_set<int> completed(order.begin(), order.end());
        all_ids.erase(std::remove_if(all_ids.begin(), all_ids.end(), [&](int id) { return completed.count(id) != 0; }),
                      all_ids.end());
        retire_bound_and_queued(queue, all_ids, {}, "cache bypass fixture cleanup");
    }
}

void test_cache_affinity_preserves_atomic_families() {
    auto config = test_runtime_config();
    config.scheduler.aging_credit_us     = 1;
    config.scheduler.max_aging_credit_us = 1;
    server_queue queue(config, [] { return 1; });
    queue.set_physical_slot_capacity(3);

    server_task miss                   = make_user(queue, server_task::trusted_lane::normal, 1);
    miss.scheduling.virtual_runtime_us = 500;
    const int miss_id                  = miss.id;
    miss.add_child(miss_id, queue.get_new_id());
    miss.add_child(miss_id, queue.get_new_id());

    server_task hit                   = make_user(queue, server_task::trusted_lane::normal, 1);
    hit.scheduling.virtual_runtime_us = 500;
    set_cache_affinity(hit, 1000000);
    const int hit_id = hit.id;
    hit.add_child(hit_id, queue.get_new_id());
    hit.add_child(hit_id, queue.get_new_id());

    std::vector<server_task> tasks;
    tasks.push_back(std::move(miss));
    tasks.push_back(std::move(hit));
    require(queue.post(std::move(tasks)), "post cache-affinity families atomically");

    queue.on_new_task([&](server_task && selected) {
        require(selected.id == hit_id && selected.child_tasks.size() == 2,
                "cache affinity selects the complete cached family");
        require(std::all_of(selected.child_tasks.begin(), selected.child_tasks.end(), [](const server_task & child) {
                    return child.scheduling.cached_prompt_tokens == 1 &&
                           child.scheduling.predicted_prefill_us == 1000000;
                }),
                "trusted cache facts are copied to every passive child");
        require(queue.dispatch_permits().total == 3, "family selection claims its complete atomic permit width");
        require(queue.fail_task(selected.id) && queue.dispatch_permits().total == 0,
                "family setup failure retires every cache-selected permit");
        queue.terminate();
    });
    queue.on_update_slots([] {});
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();

    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = queue.get_new_id();
    cancel.id_target = miss_id;
    require(queue.post(std::move(cancel), true) && queue.request_summary().active_requests == 0,
            "atomic cache-family fixture cleanup");
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
        if (lane == server_task::trusted_lane::low) {
            set_cache_affinity(task, 1000000);
        }
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

void test_bounded_fast_refill_reuses_one_live_slot() {
    auto config = test_runtime_config();
    config.fast_refill_max_members_per_epoch = 4;
    config.fast_refill_window_us              = 1000;
    uint64_t     now_us = 100;
    server_queue queue(config, [&] { return now_us; });
    queue.set_physical_slot_capacity(2);

    server_task low    = make_user(queue, server_task::trusted_lane::low, now_us);
    const int   low_id = low.id;
    require_posted(queue.post(std::move(low)), low_id, "post refill low anchor");

    std::vector<int> fast_ids;
    std::vector<int> fast_order;
    std::unordered_set<int> queued_fast;
    queue.on_new_task([&](server_task && selected) {
        require(selected.type == SERVER_TASK_TYPE_COMPLETION, "refill fixture dispatches only user work");
        if (selected.id == low_id) {
            require(queue.bind_slot(selected.id, 0), "bind refill low anchor");
            for (int i = 0; i < 5; ++i) {
                server_task fast = make_user(queue, server_task::trusted_lane::fast, now_us + i + 1);
                fast_ids.push_back(fast.id);
                queued_fast.insert(fast.id);
                require_posted(queue.post(std::move(fast)), fast_ids.back(), "post sequential fast refill work");
            }
            return;
        }

        require(queued_fast.erase(selected.id) == 1 && queue.bind_slot(selected.id, 1),
                "bind the unique reusable fast slot");
        fast_order.push_back(selected.id);
        require(queue.dispatch_permits().total == 2 && queue.release_slot(selected.id, 1),
                "fast refill never widens beyond low plus one fast permit");
        now_us += 10;
    });
    queue.on_update_slots([&] {
        const auto state  = queue.request_state();
        const auto refill = state.fast_refill.status_at(now_us, state.permits.total);
        require(fast_order == std::vector<int>(fast_ids.begin(), fast_ids.begin() + 4) &&
                    queued_fast == std::unordered_set<int>({ fast_ids.back() }) &&
                    queue.dispatch_permits().total == 1 && queue.request_summary().active_requests == 2,
                "four-member refill is FIFO and closes at exact epoch exhaustion");
        require(state.fast_refill.enabled && state.fast_refill.cohort_active &&
                    state.fast_refill.dominant == server_scheduler::lane::fast &&
                    state.fast_refill.fast_members_used == 4 && state.fast_refill.deadline_us == 1100 &&
                    refill.fast_members_remaining == 0 && !refill.window_open &&
                    !refill.one_member_eligible_now,
                "queue-locked request state propagates authoritative exhausted refill telemetry");
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id        = queue.get_new_id();
        cancel.id_target = fast_ids.back();
        require(queue.post(std::move(cancel), true) && queue.release_slot(low_id, 0) &&
                    queue.dispatch_permits().total == 0 && queue.request_summary().active_requests == 0,
                "exhausted live refill fixture cancels and releases without leaks");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
}

void test_fast_refill_timeout_and_cancel_boundaries() {
    auto config = test_runtime_config();
    config.fast_refill_max_members_per_epoch = 4;
    config.fast_refill_window_us              = 1000;
    uint64_t     now_us = 200;
    server_queue queue(config, [&] { return now_us; });
    queue.set_physical_slot_capacity(2);

    server_task low    = make_user(queue, server_task::trusted_lane::low, now_us);
    const int   low_id = low.id;
    require_posted(queue.post(std::move(low)), low_id, "post terminal-boundary low anchor");

    std::vector<int> fast_ids;
    std::vector<int> fast_order;
    std::vector<int> expired_ids;
    int              terminal_controls = 0;
    queue.on_request_expired([&](server_queue_expiration event) {
        require(event.kind == server_request_runtime::deadline_kind::queue,
                "refill boundary emits a queue timeout");
        expired_ids.push_back(event.task_id);
    });
    queue.on_new_task([&](server_task && selected) {
        if (selected.type == SERVER_TASK_TYPE_CANCEL) {
            ++terminal_controls;
            return;
        }
        if (selected.id == low_id) {
            require(queue.bind_slot(selected.id, 0), "bind terminal-boundary low anchor");
            for (int i = 0; i < 3; ++i) {
                server_task fast = make_user(queue, server_task::trusted_lane::fast, now_us);
                if (i == 1) {
                    fast.scheduling.queue_timeout_us = 5;
                }
                fast_ids.push_back(fast.id);
                require_posted(queue.post(std::move(fast)), fast_ids.back(), "post terminal-boundary fast work");
            }
            return;
        }

        fast_order.push_back(selected.id);
        require(queue.bind_slot(selected.id, 1), "bind terminal-boundary fast work");
        if (selected.id == fast_ids.front()) {
            server_task cancel(SERVER_TASK_TYPE_CANCEL);
            cancel.id        = queue.get_new_id();
            cancel.id_target = selected.id;
            require(queue.post(std::move(cancel), true), "cancel first fast member while bound");
            now_us = 205;
        }
        require(queue.release_slot(selected.id, 1), "release terminal-boundary fast permit");
    });
    queue.on_update_slots([&] {
        require(fast_order == std::vector<int>({ fast_ids[0], fast_ids[2] }) &&
                    expired_ids == std::vector<int>({ fast_ids[1] }) && terminal_controls == 1 &&
                    queue.dispatch_permits().total == 1 && queue.request_summary().active_requests == 1,
                "exact timeout and bound cancellation preserve the next FIFO-eligible refill");
        require(queue.release_slot(low_id, 0) && queue.dispatch_permits().total == 0 &&
                    queue.request_summary().active_requests == 0,
                "terminal-boundary refill fixture drains without permit leaks");
        queue.terminate();
    });
    queue.on_sleeping_state([](bool) {});
    queue.start_loop();
}

void test_live_fast_refill_environment_is_atomic_and_bounded() {
    constexpr const char * members_name = "LLAMA_SERVER_TRUSTED_FAST_REFILL_MAX_MEMBERS";
    constexpr const char * window_name  = "LLAMA_SERVER_TRUSTED_FAST_REFILL_WINDOW_MS";
    scoped_environment members_environment(members_name);
    scoped_environment window_environment(window_name);

    const auto run_fixture = [&](const char * members, const char * window, size_t expected_fast) {
        members_environment.assign(members);
        window_environment.assign(window);
        server_queue queue;
        queue.set_physical_slot_capacity(2);

        server_task low    = make_user(queue, server_task::trusted_lane::low, 1);
        const int   low_id = low.id;
        require_posted(queue.post(std::move(low)), low_id, "post environment refill low anchor");

        std::vector<int>              fast_ids;
        std::vector<int>              fast_order;
        std::unordered_set<int>       queued_fast;
        queue.on_new_task([&](server_task && selected) {
            if (selected.id == low_id) {
                require(queue.bind_slot(selected.id, 0), "bind environment refill low anchor");
                for (int i = 0; i < 2; ++i) {
                    server_task fast = make_user(queue, server_task::trusted_lane::fast, i + 2);
                    fast_ids.push_back(fast.id);
                    queued_fast.insert(fast.id);
                    require_posted(queue.post(std::move(fast)), fast_ids.back(), "post environment fast work");
                }
                return;
            }
            require(queued_fast.erase(selected.id) == 1 && queue.bind_slot(selected.id, 1) &&
                        queue.release_slot(selected.id, 1),
                    "environment fast member reuses only the free slot");
            fast_order.push_back(selected.id);
        });
        queue.on_update_slots([&] {
            require(fast_order.size() == expected_fast &&
                        std::equal(fast_order.begin(), fast_order.end(), fast_ids.begin()),
                    "environment configuration preserves FIFO and expected refill state");
            std::vector<server_task> cancellations;
            for (int id : queued_fast) {
                server_task cancel(SERVER_TASK_TYPE_CANCEL);
                cancel.id_target = id;
                cancellations.push_back(std::move(cancel));
            }
            require((cancellations.empty() || queue.post(std::move(cancellations), true)) &&
                        queue.release_slot(low_id, 0) && queue.dispatch_permits().total == 0 &&
                        queue.request_summary().active_requests == 0,
                    "environment refill fixture cleanup");
            queue.terminate();
        });
        queue.on_sleeping_state([](bool) {});
        queue.start_loop();
    };

    run_fixture(nullptr, nullptr, 1);
    run_fixture("3", nullptr, 1);
    run_fixture("0", "1", 1);
    run_fixture("17", "30001", 1);
    run_fixture("three", "1", 1);
    run_fixture("3", "30000", 2);
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
    set_cache_affinity(fresh, 1000000);
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

    queue.set_expiration_response(responses);
    queue.on_request_expired([&](server_queue_expiration) { ++expiration_callbacks; });
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
        { "generation-safe admin cancel", test_admin_cancel_requires_current_handle_and_is_idempotent_while_live },
        { "cancel deferred payload",      test_cancel_deferred_payload                  },
        { "cancel preparation atomicity", test_external_cancel_preparation_is_fault_atomic },
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
        { "staged family publication",    test_staged_family_publication_is_exception_safe_and_at_most_once },
        { "atomic terminal preparation",  test_terminal_expiry_and_failure_preparation_are_atomic          },
        { "atomic family failure",        test_family_failure_prepares_all_results_before_live_cancel       },
        { "stream publication gate",      test_stream_result_publication_deadline_gate  },
        { "concurrent publication expiry", test_concurrent_update_publication_after_expiry },
        { "publication scale isolation",  test_publication_gate_scales_without_global_sweep },
        { "live lane and physical permits", test_start_loop_enforces_live_lane_and_physical_permits },
        { "profiled live low widths",       test_start_loop_uses_profiled_exclusive_low_shapes      },
        { "fast arrival bypass",            test_fast_arrival_bypasses_ready_low_backlog            },
        { "trusted live cache costs",       test_live_cache_affinity_uses_trusted_costs              },
        { "cache lane and cohort bounds",   test_live_cache_affinity_is_lane_and_cohort_bounded      },
        { "cache lookahead and bypass",     test_live_cache_lookahead_and_bypass_bound               },
        { "cache atomic families",          test_cache_affinity_preserves_atomic_families            },
        { "sustained live HDRR",            test_live_hdrr_under_sustained_mixed_queues             },
        { "sustained wide HDRR cohorts",    test_live_hdrr_starts_sustained_wide_cohorts            },
        { "staggered draining HDRR cohorts", test_staggered_cohorts_drain_before_same_lane_refill   },
        { "bounded fast cohort refill",      test_bounded_fast_refill_reuses_one_live_slot           },
        { "fast refill terminal boundaries", test_fast_refill_timeout_and_cancel_boundaries         },
        { "fast refill environment",         test_live_fast_refill_environment_is_atomic_and_bounded },
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

#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

#define QUE_INF(fmt, ...) LOG_INF("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_WRN(fmt, ...) LOG_WRN("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_ERR(fmt, ...) LOG_ERR("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define QUE_DBG(fmt, ...) LOG_DBG("que  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

#define RES_INF(fmt, ...) LOG_INF("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_WRN(fmt, ...) LOG_WRN("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_ERR(fmt, ...) LOG_ERR("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define RES_DBG(fmt, ...) LOG_DBG("res  %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// server_queue
//

namespace {

server_request_runtime::runtime_config live_runtime_config() {
    server_request_runtime::runtime_config config;
    // Live admission must preserve the server's existing context validation
    // until the allocator supplies a full prompt-plus-runway quote.
    config.scheduler.context_tokens = std::numeric_limits<uint64_t>::max();
    return config;
}

server_queue_post_result rejected_post(const server_request_runtime::admission_result & admission, int task_id) {
    const bool overload = server_request_runtime::is_overload(admission.code);
    return { overload ? server_queue_post_code::overloaded : server_queue_post_code::unavailable, task_id,
             overload ? bounded_retry_after_seconds(admission.retry_after_seconds) : 0 };
}

}  // namespace

server_queue::server_queue() : server_queue(live_runtime_config(), {}) {}

server_queue::server_queue(server_request_runtime::runtime_config config, std::function<uint64_t()> clock) :
    request_runtime(std::move(config)),
    monotonic_us(clock ? std::move(clock) :
                         std::function<uint64_t()>([] { return static_cast<uint64_t>(ggml_time_us()); })) {}

bool server_queue::is_user_task(const server_task & task) {
    switch (task.type) {
        case SERVER_TASK_TYPE_COMPLETION:
        case SERVER_TASK_TYPE_INFILL:
        case SERVER_TASK_TYPE_EMBEDDING:
        case SERVER_TASK_TYPE_RERANK:
            return true;
        default:
            return false;
    }
}

uint64_t server_queue::runtime_id(int id_task) {
    GGML_ASSERT(id_task >= 0);
    return static_cast<uint64_t>(id_task) + 1;
}

uint64_t server_queue::now_us() const {
    return monotonic_us();
}

server_request_runtime::request_metadata server_queue::make_request_metadata(server_task & task, uint64_t at_us) {
    using scheduler_lane = server_scheduler::lane;

    server_request_runtime::request_metadata result;
    result.id                 = runtime_id(task.id);
    result.arrival_us           = task.scheduling.arrival_us == 0 ? at_us : task.scheduling.arrival_us;
    result.virtual_runtime_us = task.scheduling.virtual_runtime_us;
    result.debt_us            = task.scheduling.debt_us;
    result.counts.prompt_tokens = static_cast<uint64_t>(std::max<int32_t>(0, task.n_tokens()));
    result.counts.requested_output_tokens = task.params.n_predict > 0 ?
                                                static_cast<uint64_t>(task.params.n_predict) : 0;
    result.estimates.predicted_prefill_us    = task.scheduling.predicted_prefill_us;
    result.estimates.predicted_decode_us     = task.scheduling.predicted_decode_us;
    result.estimates.predicted_gpu_us        = task.scheduling.predicted_gpu_us;
    result.estimates.predicted_memory_bytes  = task.scheduling.predicted_memory_bytes;
    result.estimates.predicted_output_tokens = task.scheduling.predicted_output_tokens;
    result.queue_timeout_us                  = task.scheduling.queue_timeout_us;
    result.run_timeout_us                    = task.scheduling.run_timeout_us;
    switch (task.scheduling.lane) {
        case server_task::trusted_lane::low:
            result.lane = scheduler_lane::low;
            break;
        case server_task::trusted_lane::normal:
            result.lane = scheduler_lane::normal;
            break;
        case server_task::trusted_lane::fast:
            result.lane = scheduler_lane::fast;
            break;
    }
    task.scheduling.arrival_us = result.arrival_us;
    return result;
}

server_request_runtime::admission_result server_queue::enqueue_user_task(server_task && task, uint64_t at_us) {
    const int task_id = task.id;
    const auto admitted = request_runtime.admit(make_request_metadata(task, at_us));
    if (!admitted) {
        QUE_WRN("reject task, id = %d, reason = %s\n", task_id, server_scheduler::to_string(admitted.reason));
        return admitted;
    }
    queue_user_tasks.emplace(task_id, std::move(task));
    return admitted;
}

server_queue_post_result server_queue::post(server_task && task, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(task.id != -1);
    const uint64_t at_us   = now_us();
    auto           expired = expire_requests_locked(at_us);
    // if this is cancel task make sure to clean up pending tasks
    if (task.type == SERVER_TASK_TYPE_CANCEL) {
        cleanup_pending_task(task.id_target, at_us);
    }
    const int task_id = task.id;
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (is_user_task(task)) {
        const auto admitted = enqueue_user_task(std::move(task), at_us);
        if (!admitted) {
            lock.unlock();
            notify_expired(expired);
            return rejected_post(admitted, task_id);
        }
    } else if (front) {
        queue_tasks.push_front(std::move(task));
    } else {
        queue_tasks.push_back(std::move(task));
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    lock.unlock();
    notify_expired(expired);
    return { server_queue_post_code::accepted, task_id, 0 };
}

server_queue_post_result server_queue::post(std::vector<server_task> && tasks, bool front) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us   = now_us();
    auto                         expired = expire_requests_locked(at_us);
    std::vector<uint64_t> admitted_ids;
    admitted_ids.reserve(tasks.size());
    for (auto & task : tasks) {
        if (task.id == -1) {
            task.id = id++;
        }
        // if this is cancel task make sure to clean up pending tasks
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            cleanup_pending_task(task.id_target, at_us);
        }
        QUE_DBG("new task, id = %d/%d, front = %d\n", task.id, (int) tasks.size(), front);
        if (is_user_task(task)) {
            const int parent_id = task.id;
            const auto parent_admission = enqueue_user_task(std::move(task), at_us);
            if (!parent_admission) {
                for (uint64_t admitted_id : admitted_ids) {
                    request_runtime.cancel(admitted_id, at_us);
                    queue_user_tasks.erase(static_cast<int>(admitted_id - 1));
                }
                lock.unlock();
                notify_expired(expired);
                return rejected_post(parent_admission, parent_id);
            }
            admitted_ids.push_back(runtime_id(parent_id));
            auto & parent = queue_user_tasks.at(parent_id);
            for (auto & child : parent.child_tasks) {
                if (child.scheduling.arrival_us == 0) {
                    child.scheduling.arrival_us = parent.scheduling.arrival_us;
                }
                const auto admitted = request_runtime.admit(make_request_metadata(child, at_us), false);
                if (!admitted) {
                    QUE_WRN("reject child task, id = %d, reason = %s\n", child.id,
                            server_scheduler::to_string(admitted.reason));
                    for (uint64_t admitted_id : admitted_ids) {
                        request_runtime.cancel(admitted_id, at_us);
                        queue_user_tasks.erase(static_cast<int>(admitted_id - 1));
                    }
                    lock.unlock();
                    notify_expired(expired);
                    return rejected_post(admitted, child.id);
                }
                admitted_ids.push_back(runtime_id(child.id));
            }
        } else if (front) {
            queue_tasks.push_front(std::move(task));
        } else {
            queue_tasks.push_back(std::move(task));
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    lock.unlock();
    notify_expired(expired);
    return { server_queue_post_code::accepted, 0, 0 };
}

server_queue_expiration server_queue::record_expiration_locked(const server_request_runtime::expiration & event) {
    const int task_id = static_cast<int>(event.request_id - 1);
    erase_pending_payload(task_id);
    return { task_id, event.kind };
}

void server_queue::enqueue_expiration_cancel_locked(const server_request_runtime::expiration & event) {
    if (event.was_running) {
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id        = id++;
        cancel.id_target = static_cast<int>(event.request_id - 1);
        queue_tasks.emplace_front(std::move(cancel));
    }
}

std::vector<server_queue_expiration> server_queue::expire_requests_locked(uint64_t at_us) {
    const auto                           runtime_expired = request_runtime.expire_due(at_us);
    std::vector<server_queue_expiration> expired;
    expired.reserve(runtime_expired.size());

    for (const auto & event : runtime_expired) {
        expired.push_back(record_expiration_locked(event));
    }

    // Running requests need their temporary slots released. Insert the
    // cancellation controls in reverse so ascending durable request order is
    // preserved at the front of the internal queue.
    for (auto it = runtime_expired.rbegin(); it != runtime_expired.rend(); ++it) {
        enqueue_expiration_cancel_locked(*it);
    }
    return expired;
}

void server_queue::notify_expired(const server_queue_expiration & expired) {
    if (callback_request_expired) {
        callback_request_expired(expired);
    }
}

void server_queue::notify_expired(const std::vector<server_queue_expiration> & expired) {
    for (const auto & event : expired) {
        notify_expired(event);
    }
}

size_t server_queue::expire_requests() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    auto                         expired = expire_requests_locked(now_us());
    if (!expired.empty()) {
        time_last_task = ggml_time_ms();
        condition_tasks.notify_one();
    }
    lock.unlock();
    notify_expired(expired);
    return expired.size();
}

void server_queue::defer(server_task && task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us   = now_us();
    auto                         expired = expire_requests_locked(at_us);
    QUE_DBG("defer task, id = %d\n", task.id);
    if (is_user_task(task) && !request_runtime.mark_deferred(runtime_id(task.id), at_us)) {
        const bool timed_out =
            std::any_of(expired.begin(), expired.end(),
                        [&task](const server_queue_expiration & event) { return event.task_id == task.id; });
        if (!timed_out) {
        QUE_ERR("failed to mark deferred task, id = %d\n", task.id);
            request_runtime.fail(runtime_id(task.id), at_us);
        }
        lock.unlock();
        notify_expired(expired);
        return;
    }
    queue_tasks_deferred.push_back(std::move(task));
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    lock.unlock();
    notify_expired(expired);
}

int server_queue::get_new_id() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    int new_id = id++;
    return new_id;
}

void server_queue::pop_deferred_task(int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us   = now_us();
    auto                         expired = expire_requests_locked(at_us);
    if (!queue_tasks_deferred.empty()) {
        // try to find a task that uses the specified slot
        bool found = false;
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id_slot == id_slot) {
                QUE_DBG("pop deferred task (use slot %d), id_task = %d\n", id_slot, it->id);
                if (is_user_task(*it) && !request_runtime.resume(runtime_id(it->id), at_us)) {
                    found = true;
                    break;
                }
                server_task task = std::move(*it);
                queue_tasks_deferred.erase(it);
                if (is_user_task(task)) {
                    queue_user_tasks.emplace(task.id, std::move(task));
                } else {
                    queue_tasks.emplace_front(std::move(task));
                }
                found = true;
                break;
            }
        }
        // if not tasks found using the slot, just pop the first deferred task (default behavior)
        if (!found) {
            QUE_DBG("pop deferred task, id_task = %d\n", queue_tasks_deferred.front().id);
            if (is_user_task(queue_tasks_deferred.front()) &&
                !request_runtime.resume(runtime_id(queue_tasks_deferred.front().id), at_us)) {
                condition_tasks.notify_one();
                lock.unlock();
                notify_expired(expired);
                return;
            }
            server_task task = std::move(queue_tasks_deferred.front());
            queue_tasks_deferred.pop_front();
            if (is_user_task(task)) {
                queue_user_tasks.emplace(task.id, std::move(task));
            } else {
                queue_tasks.emplace_front(std::move(task));
            }
        }
    }
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
    lock.unlock();
    notify_expired(expired);
}

void server_queue::wait_until_no_sleep() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    if (!sleeping) {
        return;
    } else {
        if (!req_stop_sleeping) {
            QUE_DBG("%s", "requesting to stop sleeping\n");
            req_stop_sleeping = true;
            condition_tasks.notify_one(); // only main thread is waiting on this
        }
        QUE_DBG("%s", "waiting until no sleep\n");
        condition_tasks.wait(lock, [&]{
            return !sleeping;
        });
    }
}

void server_queue::terminate() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    running = false;
    condition_tasks.notify_all();
}

void server_queue::start_loop(int64_t idle_sleep_ms) {
    running = true;
    time_last_task = ggml_time_ms();

    constexpr auto max_wait_time = std::chrono::seconds(1);
    auto should_sleep = [&]() -> bool {
        // caller must hold mutex_tasks
        if (idle_sleep_ms < 0) {
            return false;
        }
        int64_t now = ggml_time_ms();
        return (now - time_last_task) >= idle_sleep_ms;
    };

    while (true) {
        QUE_DBG("%s", "processing new tasks\n");

        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            if (!running) {
                QUE_DBG("%s", "terminate\n");
                return;
            }
            auto expired = expire_requests_locked(now_us());
            if (!expired.empty()) {
                lock.unlock();
                notify_expired(expired);
                continue;
            }
            if (queue_tasks.empty() && request_runtime.queued_total() == 0) {
                lock.unlock();
                break;
            }
            server_task task;
            if (!queue_tasks.empty()) {
                task = std::move(queue_tasks.front());
                queue_tasks.pop_front();
            } else {
                const auto decision = request_runtime.take_next(now_us());
                if (!decision.selected) {
                    lock.unlock();
                    break;
                }
                const int task_id = static_cast<int>(decision.request_id - 1);
                auto task_it = queue_user_tasks.find(task_id);
                GGML_ASSERT(task_it != queue_user_tasks.end());
                task = std::move(task_it->second);
                queue_user_tasks.erase(task_it);
            }
            lock.unlock();

            QUE_DBG("processing task, id = %d\n", task.id);
            callback_new_task(std::move(task));
        }
        // all tasks in the current loop is processed, slots data is now ready
        QUE_DBG("%s", "update slots\n");

        // A request may cross its queue deadline while task setup runs. Process
        // deadline cancellation controls before allowing another decode step.
        if (expire_requests() != 0) {
            continue;
        }

        // this will run the main inference process for all slots
        callback_update_slots();
        {
            // update_slots() may take a while to finish, we need to make sure it's not counted as idle
            std::unique_lock<std::mutex> lock(mutex_tasks);
            time_last_task = ggml_time_ms();
        }

        QUE_DBG("%s", "waiting for new tasks\n");
        while (true) {
            std::unique_lock<std::mutex> lock(mutex_tasks);
            auto                         expired = expire_requests_locked(now_us());
            if (!expired.empty()) {
                lock.unlock();
                notify_expired(expired);
                break;
            }
            if (!running || !queue_tasks.empty() || request_runtime.queued_total() != 0) {
                break; // go back to process new tasks or terminate
            }

            // no tasks, check for sleeping state
            if (should_sleep()) {
                QUE_INF("%s", "entering sleeping state\n");
                sleeping = true;
                callback_sleeping_state(true);
                req_stop_sleeping = false;
                // wait until we are requested to exit sleeping state
                condition_tasks.wait(lock, [&]{
                    return (!running || req_stop_sleeping);
                });
                if (!running) { // may changed during sleep
                    break; // terminate
                }
                QUE_INF("%s", "exiting sleeping state\n");
                req_stop_sleeping = false;
                callback_sleeping_state(false);
                sleeping = false;
                time_last_task = ggml_time_ms();
                condition_tasks.notify_all(); // notify wait_until_no_sleep()
                break; // process new tasks
            } else {
                // wait for new tasks or timeout for checking sleeping condition
                bool res = condition_tasks.wait_for(lock, max_wait_time, [&]{
                    return (!queue_tasks.empty() || request_runtime.queued_total() != 0 || !running);
                });
                if (res) {
                    break; // new task arrived or terminate
                }
                // otherwise, loop again to check sleeping condition
            }
        }
    }
}

void server_queue::erase_pending_payload(int id_target) {
    auto remove_target = [id_target](server_task & task) {
        task.child_tasks.erase(std::remove_if(task.child_tasks.begin(), task.child_tasks.end(),
                                              [id_target](const server_task & child) { return child.id == id_target; }),
                               task.child_tasks.end());
        return task.id == id_target;
    };
    queue_user_tasks.erase(id_target);
    for (auto & entry : queue_user_tasks) {
        remove_target(entry.second);
    }
    queue_tasks.erase(std::remove_if(queue_tasks.begin(), queue_tasks.end(), remove_target), queue_tasks.end());
    queue_tasks_deferred.erase(std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), remove_target),
        queue_tasks_deferred.end());
}

void server_queue::cleanup_pending_task(int id_target, uint64_t at_us) {
    // no need lock because this is called exclusively by post()
    if (id_target >= 0) {
        request_runtime.cancel(runtime_id(id_target), at_us);
        erase_pending_payload(id_target);
    }
}

server_queue_bind_result server_queue::bind_slot(int id_task, int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us   = now_us();
    auto                         expired = expire_requests_locked(at_us);
    const bool                   request_expired =
        std::any_of(expired.begin(), expired.end(),
                    [id_task](const server_queue_expiration & event) { return event.task_id == id_task; });
    const bool bound =
        !request_expired && id_task >= 0 && id_slot >= 0 &&
        request_runtime.bind_slot(runtime_id(id_task), static_cast<server_request_registry::slot_id>(id_slot), at_us);
    lock.unlock();
    notify_expired(expired);
    return { request_expired ? server_queue_bind_code::expired :
             bound           ? server_queue_bind_code::bound :
                               server_queue_bind_code::rejected };
}

bool server_queue::publish_slot_result(server_response &         response,
                                       int                       id_task,
                                       int                       id_slot,
                                       server_queue_result_kind  kind,
                                       server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(result && result->id == id_task);
    server_request_runtime::publication_result publication;
    if (id_task >= 0 && id_slot >= 0) {
        publication = request_runtime.gate_result_publication(
            runtime_id(id_task), static_cast<server_request_registry::slot_id>(id_slot),
            kind == server_queue_result_kind::final, now_us());
    }

    server_queue_expiration expired;
    if (publication.expired.request_id != 0) {
        expired = record_expiration_locked(publication.expired);
        enqueue_expiration_cancel_locked(publication.expired);
    }
    if (publication.publish) {
        // Keep the durable deadline/terminal decision and response insertion in
        // one serialized transaction. A concurrent cancel or expiry therefore
        // orders wholly before or wholly after this publication.
        response.send(std::move(result));
    }
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return publication.publish;
}

bool server_queue::release_slot(int id_task, int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    server_request_runtime::release_result release;
    if (id_task >= 0 && id_slot >= 0) {
        release = request_runtime.release_slot(runtime_id(id_task),
                                               static_cast<server_request_registry::slot_id>(id_slot), now_us());
    }
    server_queue_expiration expired;
    if (release.expired.request_id != 0) {
        expired = record_expiration_locked(release.expired);
        enqueue_expiration_cancel_locked(release.expired);
    }
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return release.released;
}

bool server_queue::fail_task(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us   = now_us();
    auto                         expired = expire_requests_locked(at_us);
    const bool                   failed  = id_task >= 0 && request_runtime.fail(runtime_id(id_task), at_us);
    lock.unlock();
    notify_expired(expired);
    return failed;
}

std::vector<server_request_registry::request_snapshot> server_queue::request_snapshot() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    return request_runtime.snapshot();
}

server_request_registry::event_log_snapshot server_queue::request_events() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    return request_runtime.events();
}

server_request_registry::registry_summary server_queue::request_summary() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    return request_runtime.summary();
}

//
// server_response
//

void server_response::add_waiting_task_id(int id_task) {
    RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.insert(id_task);
}

void server_response::add_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("add task %d to waiting list. current waiting = %d (before add)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.insert(id_task);
    }
}

void server_response::remove_waiting_task_id(int id_task) {
    RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());

    std::unique_lock<std::mutex> lock(mutex_results);
    waiting_task_ids.erase(id_task);
    // make sure to clean up all pending results
    queue_results.erase(
        std::remove_if(queue_results.begin(), queue_results.end(), [id_task](const server_task_result_ptr & res) {
            return res->id == id_task;
        }),
        queue_results.end());
}

void server_response::remove_waiting_task_ids(const std::unordered_set<int> & id_tasks) {
    std::unique_lock<std::mutex> lock(mutex_results);

    for (const auto & id_task : id_tasks) {
        RES_DBG("remove task %d from waiting list. current waiting = %d (before remove)\n", id_task, (int) waiting_task_ids.size());
        waiting_task_ids.erase(id_task);
    }
}

server_task_result_ptr server_response::recv(const std::unordered_set<int> & id_tasks) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);
        condition_results.wait(lock, [&]{
            if (!running) {
                RES_DBG("%s : queue result stop\n", "recv");
                std::terminate(); // we cannot return here since the caller is HTTP code
            }
            return !queue_results.empty();
        });

        for (size_t i = 0; i < queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout) {
    while (true) {
        std::unique_lock<std::mutex> lock(mutex_results);

        for (int i = 0; i < (int) queue_results.size(); i++) {
            if (id_tasks.find(queue_results[i]->id) != id_tasks.end()) {
                server_task_result_ptr res = std::move(queue_results[i]);
                queue_results.erase(queue_results.begin() + i);
                return res;
            }
        }

        std::cv_status cr_res = condition_results.wait_for(lock, std::chrono::seconds(timeout));
        if (!running) {
            RES_DBG("%s : queue result stop\n", __func__);
            std::terminate(); // we cannot return here since the caller is HTTP code
        }
        if (cr_res == std::cv_status::timeout) {
            return nullptr;
        }
    }

    // should never reach here
}

server_task_result_ptr server_response::recv(int id_task) {
    std::unordered_set<int> id_tasks = {id_task};
    return recv(id_tasks);
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);

    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        if (result->id == id_task) {
            RES_DBG("task id = %d pushed to result queue\n", result->id);

            queue_results.emplace_back(std::move(result));
            condition_results.notify_all();
            return;
        }
    }
}

void server_response::broadcast(server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_results);
    for (const auto & id_task : waiting_task_ids) {
        RES_DBG("task id = %d pushed to result queue\n", id_task);
        server_task_result_ptr res_copy(result->clone());
        res_copy->id = id_task; // override id with target task id
        queue_results.emplace_back(std::move(res_copy));
    }
    condition_results.notify_all();
}

void server_response::terminate() {
    running = false;
    condition_results.notify_all();
}

//
// server_response_reader
//

void server_response_reader::post_task(server_task && task, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_task() can only be called once per reader");
    GGML_ASSERT(!task.is_parent() && "not supported, use post_tasks() instead");
    task.index = 0;
    id_tasks.insert(task.id);
    states.push_back(task.create_state());
    queue_results.add_waiting_task_id(task.id);
    const auto posted = queue_tasks.post(std::move(task), front);
    if (!posted) {
        auto error = std::make_unique<server_task_result_error>();
        error->id       = *id_tasks.begin();
        if (posted.code == server_queue_post_code::overloaded) {
            error->err_type            = ERROR_TYPE_OVERLOADED;
            error->err_msg             = "request queue is full";
            error->retry_after_seconds = posted.retry_after_seconds;
        } else {
            error->err_type = ERROR_TYPE_UNAVAILABLE;
            error->err_msg  = "request is temporarily unavailable";
        }
        queue_results.send(std::move(error));
    }
}

void server_response_reader::post_tasks(std::vector<server_task> && tasks, bool front) {
    GGML_ASSERT(id_tasks.empty() && "post_tasks() can only be called once per reader");
    id_tasks = server_task::get_list_id(tasks);
    states.reserve(tasks.size());
    size_t index = 0;
    for (auto & task : tasks) {
        task.index = index++;
        states.push_back(task.create_state());
        // for child tasks
        for (auto & child_task : task.child_tasks) {
            child_task.index = index++;
            states.push_back(child_task.create_state());
        }
    }
    GGML_ASSERT(states.size() == id_tasks.size());
    queue_results.add_waiting_task_ids(id_tasks);
    const auto posted = queue_tasks.post(std::move(tasks), front);
    if (!posted) {
        auto error = std::make_unique<server_task_result_error>();
        error->id       = *std::min_element(id_tasks.begin(), id_tasks.end());
        if (posted.code == server_queue_post_code::overloaded) {
            error->err_type            = ERROR_TYPE_OVERLOADED;
            error->err_msg             = "request queue is full";
            error->retry_after_seconds = posted.retry_after_seconds;
        } else {
            error->err_type = ERROR_TYPE_UNAVAILABLE;
            error->err_msg  = "request is temporarily unavailable";
        }
        queue_results.send(std::move(error));
    }
}

bool server_response_reader::has_next() const {
    return !cancelled && received_count < id_tasks.size();
}

// return nullptr if should_stop() is true before receiving a result
// note: if one error is received, it will stop further processing and return error result
server_task_result_ptr server_response_reader::next(const std::function<bool()> & should_stop) {
    while (true) {
        server_task_result_ptr result = queue_results.recv_with_timeout(id_tasks, polling_interval_seconds);
        if (result == nullptr) {
            // timeout, check stop condition
            if (should_stop()) {
                return nullptr;
            }
        } else {
            if (result->is_error()) {
                stop(); // cancel remaining tasks
                SRV_DBG("%s", "received error result, stopping further processing\n");
                return result;
            }
            if (!states.empty()) {
                // update the generation state if needed
                const size_t idx = result->index;
                GGML_ASSERT(idx < states.size());
                result->update(states[idx]);
            }
            if (result->is_stop()) {
                received_count++;
            }
            return result;
        }
    }

    // should not reach here
}

server_response_reader::batch_response server_response_reader::wait_for_all(const std::function<bool()> & should_stop) {
    batch_response batch_res;
    batch_res.results.clear();
    batch_res.results.resize(id_tasks.size());
    while (has_next()) {
        auto res = next(should_stop);
        if (res == nullptr) {
            batch_res.is_terminated = true;
            return batch_res;
        }
        if (res->is_error()) {
            batch_res.error = std::move(res);
            return batch_res;
        }
        const size_t idx = res->index;
        GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
        GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
        batch_res.results[idx] = std::move(res);
    }
    return batch_res;
}

void server_response_reader::stop() {
    queue_results.remove_waiting_task_ids(id_tasks);
    if (has_next() && !cancelled) {
        // if tasks is not finished yet, cancel them
        cancelled = true;
        std::vector<server_task> cancel_tasks;
        cancel_tasks.reserve(id_tasks.size());
        for (const auto & id_task : id_tasks) {
            SRV_WRN("cancel task, id_task = %d\n", id_task);
            server_task task(SERVER_TASK_TYPE_CANCEL);
            task.id_target = id_task;
            queue_results.remove_waiting_task_id(id_task);
            cancel_tasks.push_back(std::move(task));
        }
        // push to beginning of the queue, so it has highest priority
        queue_tasks.post(std::move(cancel_tasks), true);
    } else {
        SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
    }
}

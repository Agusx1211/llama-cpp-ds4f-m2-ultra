#include "server-task.h"
#include "server-queue.h"

#include "log.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <type_traits>
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

constexpr const char * fast_refill_members_environment =
    "LLAMA_SERVER_TRUSTED_FAST_REFILL_MAX_MEMBERS";
constexpr const char * fast_refill_window_environment =
    "LLAMA_SERVER_TRUSTED_FAST_REFILL_WINDOW_MS";

bool parse_bounded_decimal(const char * value, uint64_t minimum, uint64_t maximum, uint64_t & parsed) {
    if (value == nullptr) {
        return false;
    }
    const std::string_view text(value);
    if (text.empty()) {
        return false;
    }
    uint64_t next = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), next);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || next < minimum || next > maximum) {
        return false;
    }
    parsed = next;
    return true;
}

server_request_runtime::runtime_config live_runtime_config() {
    server_request_runtime::runtime_config config;
    // Live admission must preserve the server's existing context validation
    // until the allocator supplies a full prompt-plus-runway quote.
    config.scheduler.context_tokens = std::numeric_limits<uint64_t>::max();

    const char * members_value = std::getenv(fast_refill_members_environment);
    const char * window_value  = std::getenv(fast_refill_window_environment);
    if (members_value == nullptr && window_value == nullptr) {
        return config;
    }
    if (members_value == nullptr || window_value == nullptr) {
        QUE_WRN("trusted fast refill disabled: set both %s and %s\n",
                fast_refill_members_environment, fast_refill_window_environment);
        return config;
    }

    uint64_t members   = 0;
    uint64_t window_ms = 0;
    const bool valid_members = parse_bounded_decimal(
        members_value, server_request_runtime::runtime_config::fast_refill_min_members,
        server_request_runtime::runtime_config::fast_refill_max_members, members);
    const bool valid_window = parse_bounded_decimal(
        window_value, server_request_runtime::runtime_config::fast_refill_min_window_us / 1000,
        server_request_runtime::runtime_config::fast_refill_max_window_us / 1000, window_ms);
    if (!valid_members || !valid_window) {
        QUE_WRN("trusted fast refill disabled: %s must be [%zu, %zu] and %s must be [%llu, %llu] ms\n",
                fast_refill_members_environment,
                server_request_runtime::runtime_config::fast_refill_min_members,
                server_request_runtime::runtime_config::fast_refill_max_members,
                fast_refill_window_environment,
                static_cast<unsigned long long>(
                    server_request_runtime::runtime_config::fast_refill_min_window_us / 1000),
                static_cast<unsigned long long>(
                    server_request_runtime::runtime_config::fast_refill_max_window_us / 1000));
        return config;
    }

    config.fast_refill_max_members_per_epoch = static_cast<size_t>(members);
    config.fast_refill_window_us              = window_ms * 1000;
    QUE_INF("trusted fast refill enabled: at most %zu fast members within %llu ms per cohort\n",
            config.fast_refill_max_members_per_epoch, static_cast<unsigned long long>(window_ms));
    return config;
}

server_queue_post_result rejected_post(const server_request_runtime::admission_result & admission, int task_id) {
    const bool overload = server_request_runtime::is_overload(admission.code);
    return { overload ? server_queue_post_code::overloaded : server_queue_post_code::unavailable, task_id,
             overload ? bounded_retry_after_seconds(admission.retry_after_seconds) : 0 };
}

server_task_result_ptr make_timeout_result(const server_request_runtime::expiration & expiration) {
    auto result      = std::make_unique<server_task_result_error>();
    result->id       = static_cast<int>(expiration.request_id - 1);
    result->err_type = ERROR_TYPE_TIMEOUT;
    result->err_msg  = expiration.kind == server_request_runtime::deadline_kind::queue
        ? "request queue deadline expired"
        : "request run deadline expired";
    return result;
}

server_scheduler::lane to_scheduler_lane(server_task::trusted_lane lane) {
    switch (lane) {
        case server_task::trusted_lane::low:
            return server_scheduler::lane::low;
        case server_task::trusted_lane::normal:
            return server_scheduler::lane::normal;
        case server_task::trusted_lane::fast:
            return server_scheduler::lane::fast;
    }
    return server_scheduler::lane::count;
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
    server_request_runtime::request_metadata result;
    result.id                 = runtime_id(task.id);
    result.arrival_us           = task.scheduling.arrival_us == 0 ? at_us : task.scheduling.arrival_us;
    result.virtual_runtime_us = task.scheduling.virtual_runtime_us;
    result.debt_us            = task.scheduling.debt_us;
    result.counts.prompt_tokens = static_cast<uint64_t>(std::max<int32_t>(0, task.n_tokens()));
    result.counts.cached_prompt_tokens = task.scheduling.cached_prompt_tokens;
    result.counts.requested_output_tokens = task.params.n_predict > 0 ?
                                                static_cast<uint64_t>(task.params.n_predict) : 0;
    result.estimates.predicted_prefill_us    = task.scheduling.predicted_prefill_us;
    result.estimates.predicted_cache_restore_us = task.scheduling.predicted_cache_restore_us;
    result.estimates.predicted_decode_us     = task.scheduling.predicted_decode_us;
    result.estimates.predicted_gpu_us        = task.scheduling.predicted_gpu_us;
    result.estimates.predicted_memory_bytes  = task.scheduling.predicted_memory_bytes;
    result.estimates.predicted_output_tokens = task.scheduling.predicted_output_tokens;
    result.parent_id                         = task.id_parent >= 0 ? runtime_id(task.id_parent) : 0;
    result.queue_timeout_us                  = task.scheduling.queue_timeout_us;
    result.run_timeout_us                    = task.scheduling.run_timeout_us;
    result.lane = to_scheduler_lane(task.scheduling.lane);
    task.scheduling.arrival_us = result.arrival_us;
    return result;
}

server_request_runtime::admission_result server_queue::validate_user_family(const server_task & task) const {
    std::array<size_t, server_scheduler::lane_count> demand = {};
    const auto priority = to_scheduler_lane(task.scheduling.lane);
    if (priority == server_scheduler::lane::count) {
        return { server_request_runtime::result_code::invalid_request,
                 server_scheduler::reason_code::reject_invalid_request };
    }
    ++demand[static_cast<size_t>(priority)];
    for (const server_task & child : task.child_tasks) {
        const auto child_lane = to_scheduler_lane(child.scheduling.lane);
        if (child_lane == server_scheduler::lane::count) {
            return { server_request_runtime::result_code::invalid_request,
                     server_scheduler::reason_code::reject_invalid_request };
        }
        ++demand[static_cast<size_t>(child_lane)];
    }
    return request_runtime.validate_dispatch_family(priority, demand, physical_slot_capacity);
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
    const int      task_id = task.id;
    const bool     is_cancel = task.type == SERVER_TASK_TYPE_CANCEL;
    auto prepared_cancel = is_cancel ? prepare_posted_cancel(std::move(task)) : std::list<server_task>{};
    if (is_cancel) {
        // The queue node already exists. Durable cancellation, payload erase,
        // and list splice therefore cannot strand a bound slot if allocation
        // fails while posting the control.
        const int id_target = canonical_cancel_target(prepared_cancel.front().id_target);
        prepared_cancel.front().id_target = id_target;
        if (protected_terminal_family != id_target || !has_terminal_control(id_target)) {
            cleanup_pending_task(id_target, at_us);
            commit_terminal_controls(prepared_cancel);
        }
        QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
        lock.unlock();
        notify_expired(expired);
        return { server_queue_post_code::accepted, task_id, 0 };
    }
    QUE_DBG("new task, id = %d, front = %d\n", task_id, front);
    if (is_user_task(task)) {
        const auto family = validate_user_family(task);
        if (!family) {
            lock.unlock();
            notify_expired(expired);
            return rejected_post(family, task_id);
        }
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
    }
    for (const auto & task : tasks) {
        if (!is_user_task(task)) {
            continue;
        }
        const auto family = validate_user_family(task);
        if (!family) {
            lock.unlock();
            notify_expired(expired);
            return rejected_post(family, task.id);
        }
    }

    std::list<server_task> prepared_cancels;
    for (auto & task : tasks) {
        if (task.type != SERVER_TASK_TYPE_CANCEL) {
            continue;
        }
        task.id_target = canonical_cancel_target(task.id_target);
        before_terminal_control_prepare();
        prepared_cancels.emplace_back(std::move(task));
    }
    while (!prepared_cancels.empty()) {
        const auto prepared_it = std::prev(prepared_cancels.end());
        const int  id_target   = prepared_it->id_target;
        std::list<server_task> prepared_cancel;
        prepared_cancel.splice(prepared_cancel.begin(), prepared_cancels, prepared_it);
        if (protected_terminal_family == id_target && has_terminal_control(id_target)) {
            continue;
        }
        cleanup_pending_task(id_target, at_us);
        commit_terminal_controls(prepared_cancel);
    }

    for (auto & task : tasks) {
        if (task.type == SERVER_TASK_TYPE_CANCEL) {
            continue;
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

std::list<server_task> server_queue::prepare_expiration_controls(
        const std::vector<server_request_runtime::expiration> & expirations) {
    std::list<server_task> result;
    for (const auto & event : expirations) {
        if (!event.was_running) {
            continue;
        }
        server_task cancel(SERVER_TASK_TYPE_CANCEL);
        cancel.id        = id++;
        cancel.id_target = static_cast<int>(event.request_id - 1);
        before_terminal_control_prepare();
        result.emplace_back(std::move(cancel));
    }
    return result;
}

std::list<server_task> server_queue::prepare_expiration_control(
        const server_request_runtime::expiration & expiration) {
    std::list<server_task> result;
    if (!expiration.was_running) {
        return result;
    }
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = id++;
    cancel.id_target = static_cast<int>(expiration.request_id - 1);
    before_terminal_control_prepare();
    result.emplace_back(std::move(cancel));
    return result;
}

std::list<server_task> server_queue::prepare_failure_control(int id_task) {
    std::list<server_task> result;
    if (id_task < 0 || !request_runtime.family_has_bindings(runtime_id(id_task))) {
        return result;
    }
    if (has_terminal_control(id_task)) {
        return result;
    }
    server_task cancel(SERVER_TASK_TYPE_CANCEL);
    cancel.id        = id++;
    cancel.id_target = id_task;
    before_terminal_control_prepare();
    result.emplace_back(std::move(cancel));
    return result;
}

std::list<server_task> server_queue::prepare_posted_cancel(server_task && task) {
    std::list<server_task> result;
    before_terminal_control_prepare();
    result.emplace_back(std::move(task));
    return result;
}

void server_queue::before_terminal_control_prepare() {
    const size_t attempt = ++prepare_terminal_control_count;
    if (prepare_terminal_control_hook_for_tests) {
        prepare_terminal_control_hook_for_tests(attempt);
    }
}

void server_queue::commit_terminal_controls(std::list<server_task> & prepared) noexcept {
    if (prepared.empty()) {
        return;
    }
    queue_terminal_controls.splice(queue_terminal_controls.begin(), prepared);
    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

std::vector<server_queue_expiration> server_queue::expire_requests_locked(uint64_t at_us) {
    const auto planned = request_runtime.expirations_due(at_us);
    std::vector<server_queue_expiration> expired;
    expired.reserve(planned.size());
    auto prepared_controls = prepare_expiration_controls(planned);

    std::vector<int>                    expiration_ids;
    std::vector<server_task_result_ptr> expiration_results;
    std::optional<server_response::prepared_batch> prepared_results;
    if (expiration_response != nullptr) {
        expiration_ids.reserve(planned.size());
        expiration_results.reserve(planned.size());
        for (const auto & event : planned) {
            expiration_ids.push_back(static_cast<int>(event.request_id - 1));
            expiration_results.push_back(make_timeout_result(event));
        }
        prepared_results.emplace(expiration_response->prepare_batch(expiration_ids));
    }

    for (size_t i = 0; i < planned.size(); ++i) {
        const auto & event = planned[i];
        const auto committed = request_runtime.expire_prepared(event, at_us);
        if (committed.request_id != 0) {
            expired.push_back(record_expiration_locked(committed));
            if (prepared_results) {
                prepared_results->commit(std::move(expiration_results[i]));
            }
        }
    }
    commit_terminal_controls(prepared_controls);
    return expired;
}

server_queue_expiration server_queue::expire_one_locked(
        const server_request_runtime::expiration & planned, uint64_t at_us) {
    if (planned.request_id == 0) {
        return {};
    }
    auto prepared_controls = prepare_expiration_control(planned);
    if (expiration_response != nullptr) {
        auto result   = make_timeout_result(planned);
        auto prepared = expiration_response->prepare_send(static_cast<int>(planned.request_id - 1));
        const auto committed = request_runtime.expire_prepared(planned, at_us);
        if (committed.request_id == 0) {
            return {};
        }
        const auto expired = record_expiration_locked(committed);
        prepared.commit(std::move(result));
        commit_terminal_controls(prepared_controls);
        return expired;
    }

    const auto committed = request_runtime.expire_prepared(planned, at_us);
    if (committed.request_id == 0) {
        return {};
    }
    const auto expired = record_expiration_locked(committed);
    commit_terminal_controls(prepared_controls);
    return expired;
}

void server_queue::notify_expired(const server_queue_expiration & expired) {
    if (callback_request_expired) {
        try {
            callback_request_expired(expired);
        } catch (const std::exception & e) {
            QUE_ERR("expiration observer failed, id_task = %d: %s\n", expired.task_id, e.what());
        } catch (...) {
            QUE_ERR("expiration observer failed, id_task = %d\n", expired.task_id);
        }
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

void server_queue::set_physical_slot_capacity(size_t capacity) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(!running && capacity > 0);
    physical_slot_capacity = capacity;
}

void server_queue::request_update() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    update_requested = true;
    if (sleeping) {
        req_stop_sleeping = true;
    }
    condition_tasks.notify_one();
}

void server_queue::begin_family_terminal_handoff(int id_family) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(id_family >= 0 &&
                (protected_terminal_family < 0 || protected_terminal_family == id_family));
    protected_terminal_family = id_family;
}

void server_queue::set_prepare_terminal_control_hook_for_tests(std::function<void(size_t)> hook) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    prepare_terminal_control_hook_for_tests = std::move(hook);
    prepare_terminal_control_count = 0;
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
            if (update_requested) {
                update_requested = false;
                lock.unlock();
                break;
            }
            if (queue_terminal_controls.empty() && queue_tasks.empty() && request_runtime.queued_total() == 0) {
                lock.unlock();
                break;
            }
            server_task task;
            if (!queue_terminal_controls.empty()) {
                task = std::move(queue_terminal_controls.front());
                queue_terminal_controls.pop_front();
            } else if (!queue_tasks.empty()) {
                task = std::move(queue_tasks.front());
                queue_tasks.pop_front();
            } else {
                const auto decision = request_runtime.take_next(now_us(), physical_slot_capacity);
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
            if (!running || update_requested || !queue_terminal_controls.empty() || !queue_tasks.empty() ||
                request_runtime.queued_total() != 0) {
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
                    return (update_requested || !queue_terminal_controls.empty() || !queue_tasks.empty() ||
                            request_runtime.queued_total() != 0 || !running);
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
    queue_terminal_controls.remove_if(remove_target);
    queue_tasks_deferred.erase(std::remove_if(queue_tasks_deferred.begin(), queue_tasks_deferred.end(), remove_target),
        queue_tasks_deferred.end());
}

int server_queue::canonical_cancel_target(int id_target) const {
    if (id_target >= 0 && protected_terminal_family >= 0 &&
        request_runtime.is_family_member(runtime_id(id_target), runtime_id(protected_terminal_family))) {
        return protected_terminal_family;
    }
    return id_target;
}

bool server_queue::has_terminal_control(int id_target) const {
    return std::any_of(queue_terminal_controls.begin(), queue_terminal_controls.end(),
                       [id_target](const server_task & task) {
                           return task.type == SERVER_TASK_TYPE_CANCEL && task.id_target == id_target;
                       });
}

void server_queue::cleanup_pending_task(int id_target, uint64_t at_us) {
    // no need lock because this is called exclusively by post()
    if (id_target >= 0) {
        if (protected_terminal_family >= 0 &&
            request_runtime.is_family_member(runtime_id(id_target), runtime_id(protected_terminal_family))) {
            // A staged-family error batch owns the first-terminal handoff but
            // has not reserved every response slot yet. Keep the durable
            // records/payloads intact; the already-prepared CANCEL node will
            // run only after the forced retry clears this guard.
            return;
        }
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
    const uint64_t at_us = now_us();
    const auto planned_expiration = id_task >= 0
        ? request_runtime.expiration_due(runtime_id(id_task), at_us)
        : server_request_runtime::expiration{};
    server_request_runtime::publication_result publication;
    server_queue_expiration expired;
    if (planned_expiration.request_id != 0) {
        expired = expire_one_locked(planned_expiration, at_us);
    } else {
        auto prepared = response.prepare_send(id_task);
        if (id_task >= 0 && id_slot >= 0) {
            publication = request_runtime.gate_result_publication(
                runtime_id(id_task), static_cast<server_request_registry::slot_id>(id_slot),
                kind == server_queue_result_kind::final, at_us);
        }

        GGML_ASSERT(publication.expired.request_id == 0);
        if (publication.publish) {
            // Response storage was reserved before the durable gate. The terminal
            // decision and no-throw insertion therefore remain one transaction.
            prepared.commit(std::move(result));
        }
    }
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return publication.publish;
}

bool server_queue::fail_task_with_result(server_response &         response,
                                         int                       id_task,
                                         server_task_result_ptr && result) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    GGML_ASSERT(result && result->id == id_task);
    const uint64_t at_us = now_us();
    const auto planned_expiration = id_task >= 0
        ? request_runtime.expiration_due(runtime_id(id_task), at_us)
        : server_request_runtime::expiration{};
    server_queue_expiration expired;
    bool                    failed = false;
    {
        if (planned_expiration.request_id != 0) {
            expired = expire_one_locked(planned_expiration, at_us);
        } else {
            const auto status = id_task >= 0
                ? request_runtime.failure_publication_status(runtime_id(id_task))
                : server_request_runtime::failure_publication_result{};
            if (status) {
                auto prepared_failure_control = prepare_failure_control(id_task);
                auto prepared = response.prepare_send(id_task);
                const auto outcome = request_runtime.fail_for_publication(runtime_id(id_task), at_us);
                GGML_ASSERT(outcome.code != server_request_runtime::failure_publication_code::transition_failed);
                failed = static_cast<bool>(outcome);
                GGML_ASSERT(failed);
                erase_pending_payload(id_task);
                commit_terminal_controls(prepared_failure_control);
                prepared.commit(std::move(result));
            }
        }
    }
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return failed;
}

size_t server_queue::fail_family_with_results(
        server_response &                    response,
        int                                  id_family,
        std::vector<server_task_result_ptr> && results) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t               at_us = now_us();
    GGML_ASSERT(id_family >= 0 &&
                (protected_terminal_family < 0 || protected_terminal_family == id_family));
    protected_terminal_family = id_family;

    std::vector<int> ids;
    std::vector<int> prepared_ids;
    std::vector<server_request_runtime::expiration> planned_expirations;
    std::vector<server_task_result_ptr> timeout_results;
    ids.reserve(results.size());
    prepared_ids.reserve(results.size());
    planned_expirations.reserve(results.size());
    timeout_results.reserve(results.size());
    for (const auto & result : results) {
        GGML_ASSERT(result && result->id >= 0);
        ids.push_back(result->id);
        const auto planned = request_runtime.expiration_due(runtime_id(result->id), at_us);
        planned_expirations.push_back(planned);
        timeout_results.push_back(planned.request_id != 0 ? make_timeout_result(planned) : nullptr);
        if (planned.request_id != 0 || request_runtime.failure_publication_status(runtime_id(result->id))) {
            prepared_ids.push_back(result->id);
        }
    }

    if (prepared_ids.empty()) {
        protected_terminal_family = -1;
        return 0;
    }

    auto prepared_controls = prepare_failure_control(id_family);
    std::vector<server_queue_expiration> expired;
    expired.reserve(results.size());
    size_t terminalized = 0;

    {
        auto prepared_results = response.prepare_batch(prepared_ids);
        for (size_t i = 0; i < results.size(); ++i) {
            const auto & planned = planned_expirations[i];
            if (planned.request_id != 0) {
                const auto committed = request_runtime.expire_prepared(planned, at_us);
                GGML_ASSERT(committed.request_id == planned.request_id);
                expired.push_back(record_expiration_locked(committed));
                prepared_results.commit(std::move(timeout_results[i]));
                ++terminalized;
                continue;
            }

            const auto outcome = request_runtime.fail_one_for_publication(runtime_id(ids[i]), at_us);
            GGML_ASSERT(outcome.code != server_request_runtime::failure_publication_code::transition_failed);
            if (!outcome) {
                continue;
            }
            erase_pending_payload(ids[i]);
            prepared_results.commit(std::move(results[i]));
            ++terminalized;
        }
    }
    if (terminalized != 0) {
        commit_terminal_controls(prepared_controls);
    }
    protected_terminal_family = -1;
    lock.unlock();
    notify_expired(expired);
    return terminalized;
}

bool server_queue::release_slot(int id_task, int id_slot) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t at_us = now_us();
    const auto planned_expiration = id_task >= 0
        ? request_runtime.expiration_due(runtime_id(id_task), at_us)
        : server_request_runtime::expiration{};
    server_request_runtime::release_result release;
    server_queue_expiration expired;
    if (planned_expiration.request_id != 0) {
        expired = expire_one_locked(planned_expiration, at_us);
    }
    if (id_task >= 0 && id_slot >= 0) {
        release = request_runtime.release_slot(runtime_id(id_task),
                                               static_cast<server_request_registry::slot_id>(id_slot), at_us);
    }
    GGML_ASSERT(release.expired.request_id == 0);
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return release.released;
}

bool server_queue::fail_task(int id_task) {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    const uint64_t at_us = now_us();
    const auto planned_expiration = id_task >= 0
        ? request_runtime.expiration_due(runtime_id(id_task), at_us)
        : server_request_runtime::expiration{};
    auto prepared_failure_control    = prepare_failure_control(id_task);
    server_queue_expiration expired;
    bool                    failed = false;
    if (planned_expiration.request_id != 0) {
        expired = expire_one_locked(planned_expiration, at_us);
    } else {
        failed = fail_task_locked(id_task, at_us, prepared_failure_control);
    }
    lock.unlock();
    if (expired.task_id >= 0) {
        notify_expired(expired);
    }
    return failed;
}

bool server_queue::fail_task_locked(int id_task,
                                    uint64_t at_us,
                                    std::list<server_task> & prepared_controls) {
    const bool failed = id_task >= 0 && request_runtime.fail(runtime_id(id_task), at_us);
    if (failed) {
        erase_pending_payload(id_task);
        commit_terminal_controls(prepared_controls);
    }
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

server_request_runtime::dispatch_permit_snapshot server_queue::dispatch_permits() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    return request_runtime.permits();
}

server_queue_request_state server_queue::request_state() {
    std::unique_lock<std::mutex> lock(mutex_tasks);
    return {
        request_runtime.snapshot(),
        request_runtime.events(),
        request_runtime.summary(),
        request_runtime.permits(),
    };
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

server_response::prepared_send::prepared_send(server_response & response, int id_task) :
    response(&response), lock(response.mutex_results) {
    const size_t attempt = ++response.prepare_send_count;
    if (response.prepare_send_hook_for_tests) {
        response.prepare_send_hook_for_tests(attempt);
    }
    waiting = response.waiting_task_ids.find(id_task) != response.waiting_task_ids.end();
    if (waiting) {
        response.queue_results.reserve(response.queue_results.size() + 1);
    }
}

void server_response::prepared_send::commit(server_task_result_ptr && result) noexcept {
    if (committed) {
        return;
    }
    committed = true;
    if (!waiting) {
        return;
    }

    GGML_ASSERT(response != nullptr && result && response->queue_results.size() < response->queue_results.capacity());
    static_assert(std::is_nothrow_move_constructible<server_task_result_ptr>::value, "result handoff must not throw");
    response->queue_results.emplace_back(std::move(result));
    response->condition_results.notify_all();
}

server_response::prepared_batch::prepared_batch(
        server_response & response, const std::vector<int> & id_tasks) :
    response(&response), lock(response.mutex_results) {
    for (int id_task : id_tasks) {
        const size_t attempt = ++response.prepare_send_count;
        if (response.prepare_send_hook_for_tests) {
            response.prepare_send_hook_for_tests(attempt);
        }
        capacity_remaining += response.waiting_task_ids.find(id_task) != response.waiting_task_ids.end();
    }
    response.queue_results.reserve(response.queue_results.size() + capacity_remaining);
}

void server_response::prepared_batch::commit(server_task_result_ptr && result) noexcept {
    GGML_ASSERT(response != nullptr && result);
    if (response->waiting_task_ids.find(result->id) == response->waiting_task_ids.end()) {
        return;
    }
    GGML_ASSERT(capacity_remaining > 0 &&
                response->queue_results.size() < response->queue_results.capacity());
    static_assert(std::is_nothrow_move_constructible<server_task_result_ptr>::value,
                  "result handoff must not throw");
    response->queue_results.emplace_back(std::move(result));
    --capacity_remaining;
    response->condition_results.notify_all();
}

server_response::prepared_send server_response::prepare_send(int id_task) {
    return prepared_send(*this, id_task);
}

server_response::prepared_batch server_response::prepare_batch(const std::vector<int> & id_tasks) {
    return prepared_batch(*this, id_tasks);
}

void server_response::set_prepare_send_hook_for_tests(std::function<void(size_t)> hook) {
    std::unique_lock<std::mutex> lock(mutex_results);
    prepare_send_hook_for_tests = std::move(hook);
    prepare_send_count = 0;
}

void server_response::send(server_task_result_ptr && result) {
    RES_DBG("sending result for task id = %d\n", result->id);
    auto prepared = prepare_send(result->id);
    prepared.commit(std::move(result));
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

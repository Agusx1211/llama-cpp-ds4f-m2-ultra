#pragma once

#include "server-request-runtime.h"
#include "server-task.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <vector>
#include <unordered_set>

enum class server_queue_post_code : uint8_t {
    accepted = 0,
    overloaded,
    unavailable,
};

struct server_queue_post_result {
    server_queue_post_code code                = server_queue_post_code::accepted;
    int                    task_id             = -1;
    uint32_t               retry_after_seconds = 0;

    operator bool() const { return code == server_queue_post_code::accepted; }
};

struct server_queue_expiration {
    int                                   task_id = -1;
    server_request_runtime::deadline_kind kind    = server_request_runtime::deadline_kind::queue;
};

enum class server_queue_bind_code : uint8_t {
    bound = 0,
    expired,
    rejected,
};

struct server_queue_bind_result {
    server_queue_bind_code code = server_queue_bind_code::rejected;

    operator bool() const { return code == server_queue_bind_code::bound; }
};

enum class server_queue_result_kind : uint8_t {
    partial = 0,
    final,
};

struct server_response;

// struct for managing server tasks
// in most cases, use server_response_reader to post new tasks and retrieve results
struct server_queue {
private:
    int id = 0;
    bool running  = false;
    bool sleeping = false;
    bool req_stop_sleeping = false;
    int64_t time_last_task = 0;

    // queues
    std::deque<server_task> queue_tasks;
    std::deque<server_task> queue_tasks_deferred;
    std::map<int, server_task> queue_user_tasks;

    server_request_runtime::request_runtime request_runtime;
    std::function<uint64_t()>               monotonic_us;

    std::mutex mutex_tasks;
    std::condition_variable condition_tasks;

    // callback functions
    std::function<void(server_task &&)> callback_new_task;
    std::function<void(void)>           callback_update_slots;
    std::function<void(bool)>           callback_sleeping_state;
    std::function<void(server_queue_expiration)> callback_request_expired;

public:
    server_queue();
    server_queue(server_request_runtime::runtime_config config, std::function<uint64_t()> monotonic_us);

    // Add a new task to the end of the queue
    server_queue_post_result post(server_task && task, bool front = false);

    // multi-task version of post()
    server_queue_post_result post(std::vector<server_task> && tasks, bool front = false);

    // Add a new task, but defer until one slot is available
    void defer(server_task && task);

    // Get the next id for creating a new task
    int get_new_id();

    // Call when the state of one slot is changed, it will move one task from deferred to main queue
    // prioritize tasks that use the specified slot (otherwise, pop the first deferred task)
    void pop_deferred_task(int id_slot);

    // if sleeping, request exiting sleep state and wait until it is done
    // returns immediately if not sleeping
    void wait_until_no_sleep();

    bool is_sleeping() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return sleeping;
    }

    // end the start_loop routine
    void terminate();

    /**
     * Main loop consists of these steps:
     * - Wait until a new task arrives
     * - Process the task (i.e. maybe copy data into slot)
     * - Check if multitask is finished
     * - Update all slots
     *
     * Sleeping procedure (disabled if idle_sleep_ms < 0):
     * - If there is no task after idle_sleep_ms, enter sleeping state
     * - Call callback_sleeping_state(true)
     * - Wait until req_stop_sleeping is set to true
     * - Call callback_sleeping_state(false)
     * - Exit sleeping state
     */
    void start_loop(int64_t idle_sleep_ms = -1);

    // for metrics
    size_t queue_tasks_deferred_size() {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        return queue_tasks_deferred.size();
    }

    server_queue_bind_result bind_slot(int id_task, int id_slot);
    bool publish_slot_result(server_response &          response,
                             int                        id_task,
                             int                        id_slot,
                             server_queue_result_kind   kind,
                             server_task_result_ptr &&  result);
    bool release_slot(int id_task, int id_slot);
    bool fail_task(int id_task);
    size_t expire_requests();

    std::vector<server_request_registry::request_snapshot> request_snapshot();
    server_request_registry::event_log_snapshot request_events();
    server_request_registry::registry_summary request_summary();

    //
    // Functions below are not thread-safe, must only be used before start_loop() is called
    //

    // Register function to process a new task
    void on_new_task(std::function<void(server_task &&)> callback) {
        callback_new_task = std::move(callback);
    }

    // Register the function to be called when all slots data is ready to be processed
    void on_update_slots(std::function<void(void)> callback) {
        callback_update_slots = std::move(callback);
    }

    // Register callback for sleeping state change; multiple callbacks are allowed
    // note: when entering sleeping state, the callback is called AFTER sleeping is set to true
    //       when leaving sleeping state, the callback is called BEFORE sleeping is set to false
    void on_sleeping_state(std::function<void(bool)> callback) {
        if (callback_sleeping_state) {
            auto prev_callback = std::move(callback_sleeping_state);
            callback_sleeping_state = [prev_callback, callback](bool sleeping) {
                prev_callback(sleeping);
                callback(sleeping);
            };
        } else {
            callback_sleeping_state = std::move(callback);
        }
    }

    void on_request_expired(std::function<void(server_queue_expiration)> callback) {
        callback_request_expired = std::move(callback);
    }

private:
    static bool is_user_task(const server_task & task);
    static uint64_t runtime_id(int id_task);
    uint64_t now_us() const;
    server_request_runtime::request_metadata make_request_metadata(server_task & task, uint64_t at_us);
    server_request_runtime::admission_result enqueue_user_task(server_task && task, uint64_t at_us);
    std::vector<server_queue_expiration> expire_requests_locked(uint64_t at_us);
    void notify_expired(const std::vector<server_queue_expiration> & expired);
    void erase_pending_payload(int id_target);
    void cleanup_pending_task(int id_target, uint64_t at_us);
};

// struct for managing server responses
// in most cases, use server_response_reader to retrieve results
struct server_response {
private:
    bool running = true;

    // for keeping track of all tasks waiting for the result
    std::unordered_set<int> waiting_task_ids;

    // the main result queue (using ptr for polymorphism)
    std::vector<server_task_result_ptr> queue_results;

    std::mutex mutex_results;
    std::condition_variable condition_results;

public:
    // add the id_task to the list of tasks waiting for response
    void add_waiting_task_id(int id_task);

    void add_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // when the request is finished, we can remove task associated with it
    void remove_waiting_task_id(int id_task);

    // remove multiple tasks from waiting list
    void remove_waiting_task_ids(const std::unordered_set<int> & id_tasks);

    // This function blocks the thread until there is a response for one of the id_tasks
    server_task_result_ptr recv(const std::unordered_set<int> & id_tasks);

    // same as recv(), but have timeout in seconds
    // if timeout is reached, nullptr is returned
    server_task_result_ptr recv_with_timeout(const std::unordered_set<int> & id_tasks, int timeout);

    // single-task version of recv()
    server_task_result_ptr recv(int id_task);

    // Send a new result to a waiting id_task
    void send(server_task_result_ptr && result);

    // broadcast a new result to all waiting tasks
    // (used by router mode)
    void broadcast(server_task_result_ptr && result);

    // terminate the waiting loop
    void terminate();
};

// RAII wrapper to make working with server_queue and server_response easier
// it provides a generator-like API for server responses
// support pooling connection state and aggregating multiple results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_queue & queue_tasks;
    server_response & queue_results;
    size_t received_count = 0;
    bool cancelled = false;
    int polling_interval_seconds;

    // tracking generation state and partial tool calls
    // only used by streaming completions
    std::vector<task_result_state> states;

    // should_stop function will be called each polling_interval_seconds
    server_response_reader(server_queue & queue_tasks, server_response & queue_results, int polling_interval_seconds)
        : queue_tasks(queue_tasks), queue_results(queue_results), polling_interval_seconds(polling_interval_seconds) {}
    ~server_response_reader() {
        stop();
    }

    int get_new_id() {
        return queue_tasks.get_new_id();
    }

    // if front = true, the task will be posted to the front of the queue (high priority)
    void post_task(server_task && task, bool front = false);
    void post_tasks(std::vector<server_task> && tasks, bool front = false);
    bool has_next() const;

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop);

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };
    // aggregate multiple results
    batch_response wait_for_all(const std::function<bool()> & should_stop);

    void stop();
};

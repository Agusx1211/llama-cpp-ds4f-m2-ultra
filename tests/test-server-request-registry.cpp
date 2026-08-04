#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-request-registry.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace server_request_registry;

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

request_registration make_request(request_id id, trusted_lane lane = trusted_lane::normal, uint64_t arrival_us = 0) {
    request_registration request;
    request.id                                = id;
    request.lane                              = lane;
    request.arrival_us                        = arrival_us;
    request.virtual_runtime_us                = 10;
    request.debt_us                           = -5;
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

request_handle add_request(request_registry & registry,
                           request_id         id,
                           trusted_lane       lane       = trusted_lane::normal,
                           uint64_t           arrival_us = 0) {
    const registration_result result = registry.register_request(make_request(id, lane, arrival_us));
    require(result.code == result_code::ok, "register request");
    require(result.handle.id == id && result.handle.epoch != 0, "valid request handle");
    return result.handle;
}

request_progress advance(const request_snapshot & before, uint64_t observed, int64_t debt) {
    request_progress progress;
    progress.virtual_runtime_us     = before.virtual_runtime_us + 25;
    progress.debt_us                = debt;
    progress.observed_output_tokens = observed;
    progress.estimates              = before.estimates;
    progress.estimates.predicted_decode_us += 7;
    progress.estimates.predicted_gpu_us += 11;
    return progress;
}

request_snapshot get_required(const request_registry & registry, request_handle handle) {
    const auto value = registry.get(handle);
    require(value.has_value(), "request snapshot exists");
    return *value;
}

void test_durable_state_outlives_bindings() {
    request_registry     registry;
    const request_handle handle = add_request(registry, 11, trusted_lane::fast, 1234);

    require(registry.set_queue_state(handle, queue_state::ready, reason_code::admission_ready, 1240), "queue request");
    const binding_result binding = registry.bind(handle, 3, reason_code::dispatched, 1250);
    require(binding, "bind request");

    request_snapshot active = get_required(registry, handle);
    require(active.state == lifecycle::executing && active.queue == queue_state::none, "executing state");
    require(active.lane == trusted_lane::fast && active.arrival_us == 1234, "trusted immutable identity fields");
    require(active.counts.prompt_tokens == 128 && active.counts.cached_prompt_tokens == 64, "prompt and cache counts");
    require(active.counts.requested_output_tokens == 32, "requested output count");

    require(registry.update_progress(handle, advance(active, 7, 19), reason_code::progress_observed, 1260),
            "update durable progress");
    require(registry.unbind(binding.lease, reason_code::yielded, 1270), "unbind request");

    const request_snapshot detached = get_required(registry, handle);
    require(detached.state == lifecycle::registered && detached.bindings.empty(), "identity survives slot release");
    require(detached.virtual_runtime_us == active.virtual_runtime_us + 25 && detached.debt_us == 19,
            "runtime and debt survive slot release");
    require(detached.counts.observed_output_tokens == 7, "observed count survives slot release");
    require(detached.estimates.predicted_gpu_us == active.estimates.predicted_gpu_us + 11,
            "estimates survive slot release");
    require(detached.estimates.predicted_cache_restore_us == 25, "cache restore estimate survives slot release");
    require(detached.last_reason == reason_code::yielded, "last reason survives slot release");
}

void test_slot_reuse_and_stale_leases() {
    request_registry     registry;
    const request_handle first  = add_request(registry, 1);
    const request_handle second = add_request(registry, 2);

    const binding_result first_binding = registry.bind(first, 0, reason_code::dispatched, 10);
    require(first_binding, "first bind");
    require(registry.unbind(first_binding.lease, reason_code::yielded, 11), "first unbind");

    const binding_result second_binding = registry.bind(second, 0, reason_code::dispatched, 12);
    require(second_binding, "reuse slot");
    require(second_binding.lease.slot_generation > first_binding.lease.slot_generation,
            "slot generation advances on reuse");

    const operation_result stale = registry.unbind(first_binding.lease, reason_code::yielded, 13);
    require(stale.code == result_code::stale_lease && !stale.changed, "old lease rejected after slot reuse");
    const request_snapshot second_state = get_required(registry, second);
    require(second_state.bindings.size() == 1 && second_state.bindings[0] == second_binding.lease,
            "stale lease cannot detach the new occupant");
}

void test_one_many_one_and_rebind() {
    registry_config config;
    config.max_slots                = 8;
    config.max_bindings_per_request = 4;
    request_registry     registry(config);
    const request_handle handle = add_request(registry, 7);

    const binding_result first  = registry.bind(handle, 1, reason_code::dispatched, 1);
    const binding_result second = registry.bind(handle, 2, reason_code::dispatched, 2);
    require(first && second, "one request binds to two slots");
    require(get_required(registry, handle).bindings.size() == 2, "one-to-many binding state");

    const binding_result moved = registry.rebind(first.lease, 4, reason_code::slot_rebind, 3);
    require(moved && moved.lease.slot == 4, "atomic explicit rebind");
    require(registry.rebind(first.lease, 5, reason_code::slot_rebind, 4).code == result_code::stale_lease,
            "pre-rebind lease is stale");

    require(registry.unbind(second.lease, reason_code::yielded, 5), "remove one of two bindings");
    const request_snapshot one = get_required(registry, handle);
    require(one.state == lifecycle::executing && one.bindings.size() == 1 && one.bindings[0] == moved.lease,
            "many-to-one binding state");
    require(registry.unbind(moved.lease, reason_code::yielded, 6), "remove final binding");
    require(get_required(registry, handle).state == lifecycle::registered, "final unbind preserves request");
}

void test_stale_request_epoch_and_terminal_removal() {
    request_registry     registry;
    const request_handle old_handle = add_request(registry, 42);
    require(registry.remove_terminal(old_handle, reason_code::completed, 1).code == result_code::not_terminal,
            "active request cannot be removed");

    const binding_result lease = registry.bind(old_handle, 1, reason_code::dispatched, 2);
    require(lease, "bind before terminal transition");
    require(registry.mark_terminal(old_handle, lifecycle::completed, reason_code::completed, 3).code ==
                result_code::active_bindings,
            "terminal transition cannot orphan a slot binding");
    require(registry.unbind(lease.lease, reason_code::yielded, 4), "release terminal candidate");
    require(registry.mark_terminal(old_handle, lifecycle::completed, reason_code::completed, 5), "mark terminal");
    require(registry.remove_terminal(old_handle, reason_code::completed, 6), "remove terminal request");
    require(!registry.get(old_handle).has_value(), "removed request absent");

    const request_handle new_handle = add_request(registry, 42);
    require(new_handle.epoch != old_handle.epoch, "request ID reuse advances epoch");
    require(registry.set_queue_state(old_handle, queue_state::ready, reason_code::admission_ready, 7).code ==
                result_code::stale_handle,
            "old request epoch cannot mutate reused ID");
    require(get_required(registry, new_handle).state == lifecycle::registered, "new incarnation unchanged");
}

void test_cancel_and_timeout_in_every_queue_state() {
    const std::vector<queue_state> queues = {
        queue_state::admission,
        queue_state::ready,
        queue_state::blocked,
    };

    request_registry registry;
    request_id       next_id = 100;
    for (queue_state queue : queues) {
        const request_handle handle = add_request(registry, next_id++);
        require(registry.set_queue_state(handle, queue, reason_code::admission_wait, next_id), "set cancel queue");
        require(registry.mark_cancel_requested(handle, reason_code::client_cancel, next_id + 1), "request cancel");
        const request_snapshot flagged = get_required(registry, handle);
        require(flagged.cancel_requested && !flagged.timeout_expired, "cancel flag stored");
        require(flagged.state == lifecycle::queued && flagged.queue == queue, "cancel preserves queue observation");
        require(
            registry.bind(handle, 0, reason_code::dispatched, next_id + 2).code == result_code::request_not_runnable,
            "cancelled queued request cannot bind");
        require(registry.mark_terminal(handle, lifecycle::cancelled, reason_code::client_cancel, next_id + 3),
                "finalize cancellation");
        require(registry.remove_terminal(handle, reason_code::client_cancel, next_id + 4), "remove cancellation");
    }

    for (queue_state queue : queues) {
        const request_handle handle = add_request(registry, next_id++);
        const reason_code    timeout_reason =
            queue == queue_state::blocked ? reason_code::queue_timeout : reason_code::run_timeout;
        require(registry.set_queue_state(handle, queue, reason_code::admission_wait, next_id), "set timeout queue");
        require(registry.mark_timeout_expired(handle, timeout_reason, next_id + 1), "request timeout");
        const request_snapshot flagged = get_required(registry, handle);
        require(!flagged.cancel_requested && flagged.timeout_expired, "timeout flag stored");
        require(flagged.state == lifecycle::queued && flagged.queue == queue, "timeout preserves queue observation");
        require(
            registry.bind(handle, 0, reason_code::dispatched, next_id + 2).code == result_code::request_not_runnable,
            "timed-out queued request cannot bind");
        require(registry.mark_terminal(handle, lifecycle::timed_out, timeout_reason, next_id + 3), "finalize timeout");
        require(registry.remove_terminal(handle, timeout_reason, next_id + 4), "remove timeout");
    }
}

void test_snapshot_immutability() {
    request_registry                    registry;
    const request_handle                handle        = add_request(registry, 9);
    const request_snapshot              before        = get_required(registry, handle);
    const std::vector<request_snapshot> all_before    = registry.snapshot();
    const event_log_snapshot            events_before = registry.events();

    require(registry.set_queue_state(handle, queue_state::ready, reason_code::admission_ready, 20), "mutate queue");
    const binding_result binding = registry.bind(handle, 2, reason_code::dispatched, 21);
    require(binding, "mutate binding");
    require(registry.update_progress(handle, advance(get_required(registry, handle), 3, 8),
                                     reason_code::progress_observed, 22),
            "mutate progress");

    require(
        before.state == lifecycle::registered && before.bindings.empty() && before.counts.observed_output_tokens == 0,
        "detached request snapshot remains unchanged");
    require(all_before.size() == 1 && all_before[0] == before, "detached registry snapshot remains unchanged");
    require(events_before.events.size() == 1 && events_before.total_events == 1,
            "detached event snapshot remains unchanged");
    require(get_required(registry, handle).revision > before.revision, "current state advances independently");
    require(registry.events().total_events == 4, "current event log advances independently");
}

void test_bounded_metadata_stress() {
    registry_config config;
    config.max_requests             = 4;
    config.max_slots                = 3;
    config.max_bindings_per_request = 2;
    config.event_capacity           = 7;
    request_registry registry(config);

    std::vector<request_handle> full;
    for (request_id id = 1; id <= config.max_requests; ++id) {
        full.push_back(add_request(registry, id));
    }
    require(registry.register_request(make_request(99)).code == result_code::request_capacity_exhausted,
            "request metadata capacity is bounded");
    for (const request_handle handle : full) {
        require(registry.mark_terminal(handle, lifecycle::completed, reason_code::completed, handle.id + 10),
                "finish capacity request");
        require(registry.remove_terminal(handle, reason_code::completed, handle.id + 20), "remove capacity request");
    }

    for (request_id id = 100; id < 1100; ++id) {
        const request_handle handle = add_request(registry, id);
        require(registry.set_queue_state(handle, queue_state::blocked, reason_code::capacity_blocked, id),
                "stress queue transition");
        require(registry.mark_timeout_expired(handle, reason_code::queue_timeout, id + 1), "stress timeout");
        require(registry.mark_terminal(handle, lifecycle::timed_out, reason_code::queue_timeout, id + 2),
                "stress terminal");
        require(registry.remove_terminal(handle, reason_code::queue_timeout, id + 3), "stress removal");
    }

    const registry_summary   summary = registry.summary();
    const event_log_snapshot events  = registry.events();
    require(summary.active_requests == 0 && summary.occupied_slots == 0, "stress leaves no live metadata");
    require(summary.retained_events == config.event_capacity && events.events.size() == config.event_capacity,
            "event metadata remains at configured capacity");
    require(events.dropped_events == events.total_events - events.events.size(), "exact event drop accounting");
    require(events.first_sequence + events.events.size() == events.next_sequence, "retained sequence window");
    for (size_t i = 1; i < events.events.size(); ++i) {
        require(events.events[i].sequence == events.events[i - 1].sequence + 1,
                "contiguous deterministic sequence IDs");
    }

    registry_config no_events_config = config;
    no_events_config.event_capacity  = 0;
    request_registry no_events(no_events_config);
    add_request(no_events, 1);
    const event_log_snapshot dropped = no_events.events();
    require(dropped.events.empty() && dropped.total_events == 1 && dropped.dropped_events == 1,
            "zero-capacity event ring has exact drop accounting");
}

struct replay_state {
    std::vector<request_snapshot> requests;
    event_log_snapshot            events;
    registry_summary              summary;

    bool operator==(const replay_state & other) const {
        return requests == other.requests && events == other.events && summary == other.summary;
    }
};

replay_state run_replay() {
    registry_config config;
    config.max_slots                = 6;
    config.max_bindings_per_request = 3;
    config.event_capacity           = 64;
    request_registry registry(config);

    const request_handle a = add_request(registry, 50, trusted_lane::fast, 100);
    const request_handle b = add_request(registry, 10, trusted_lane::low, 101);
    require(registry.set_queue_state(a, queue_state::ready, reason_code::admission_ready, 102), "replay queue a");
    require(registry.set_queue_state(b, queue_state::blocked, reason_code::capacity_blocked, 103), "replay queue b");
    const binding_result a0 = registry.bind(a, 0, reason_code::dispatched, 104);
    const binding_result a1 = registry.bind(a, 1, reason_code::dispatched, 105);
    require(a0 && a1, "replay binds");
    const binding_result moved = registry.rebind(a0.lease, 4, reason_code::slot_rebind, 106);
    require(moved, "replay rebind");
    require(registry.update_progress(a, advance(get_required(registry, a), 5, 33), reason_code::progress_observed, 107),
            "replay progress");
    require(registry.unbind(a1.lease, reason_code::yielded, 108), "replay unbind one");
    require(registry.unbind(moved.lease, reason_code::yielded, 109), "replay unbind two");
    require(registry.mark_terminal(a, lifecycle::completed, reason_code::completed, 110), "replay terminal a");
    require(registry.mark_cancel_requested(b, reason_code::server_cancel, 111), "replay cancel b");
    require(registry.mark_terminal(b, lifecycle::cancelled, reason_code::server_cancel, 112), "replay terminal b");

    return { registry.snapshot(), registry.events(), registry.summary() };
}

void test_deterministic_replay() {
    const replay_state first  = run_replay();
    const replay_state second = run_replay();
    require(first == second, "identical operation replay produces identical snapshots and events");
    require(first.requests.size() == 2 && first.requests[0].handle.id == 10 && first.requests[1].handle.id == 50,
            "registry snapshots use deterministic request-ID order");
    for (size_t i = 0; i < first.events.events.size(); ++i) {
        require(first.events.events[i].sequence == i + 1, "replay sequences begin at one and advance exactly once");
    }
}

void test_validation_and_binding_bounds() {
    registry_config config;
    config.max_requests             = 2;
    config.max_slots                = 3;
    config.max_bindings_per_request = 1;
    request_registry registry(config);

    request_registration invalid        = make_request(1);
    invalid.counts.cached_prompt_tokens = invalid.counts.prompt_tokens + 1;
    require(registry.register_request(invalid).code == result_code::invalid_registration,
            "cached prompt cannot exceed prompt");

    const request_handle handle = add_request(registry, 1);
    require(registry.set_queue_state(handle, static_cast<queue_state>(255), reason_code::admission_ready, 1).code ==
                result_code::invalid_queue_state,
            "invalid queue enum rejected");
    require(registry.set_queue_state(handle, queue_state::ready, static_cast<reason_code>(65535), 1).code ==
                result_code::invalid_reason,
            "reason codes remain bounded to the fixed low-cardinality set");
    require(registry.bind(handle, 3, reason_code::dispatched, 2).code == result_code::slot_out_of_range,
            "slot range checked");
    const binding_result first = registry.bind(handle, 0, reason_code::dispatched, 3);
    require(first, "bounded first binding");
    require(registry.bind(handle, 1, reason_code::dispatched, 4).code == result_code::binding_capacity_exhausted,
            "per-request binding metadata bounded");

    request_progress backward   = advance(get_required(registry, handle), 1, 0);
    backward.virtual_runtime_us = 0;
    require(registry.update_progress(handle, backward, reason_code::progress_observed, 5).code ==
                result_code::invalid_progress,
            "virtual runtime cannot move backward");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "durable state outlives bindings",         test_durable_state_outlives_bindings          },
        { "slot reuse and stale leases",             test_slot_reuse_and_stale_leases              },
        { "one-many-one and rebind",                 test_one_many_one_and_rebind                  },
        { "stale epoch and terminal removal",        test_stale_request_epoch_and_terminal_removal },
        { "cancel and timeout in every queue state", test_cancel_and_timeout_in_every_queue_state  },
        { "snapshot immutability",                   test_snapshot_immutability                    },
        { "bounded metadata stress",                 test_bounded_metadata_stress                  },
        { "deterministic replay",                    test_deterministic_replay                     },
        { "validation and binding bounds",           test_validation_and_binding_bounds            },
    };

    try {
        for (const auto & test : tests) {
            test.second();
            std::printf("PASS: %s\n", test.first);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }

    std::printf("PASS: %zu request registry tests\n", tests.size());
    return 0;
}

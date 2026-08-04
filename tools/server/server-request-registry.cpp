#include "server-request-registry.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace server_request_registry {

namespace {

bool valid_lane(trusted_lane value) {
    return value >= trusted_lane::low && value < trusted_lane::count;
}

bool valid_queue(queue_state value) {
    return value >= queue_state::none && value < queue_state::count;
}

bool valid_reason(reason_code value) {
    switch (value) {
        case reason_code::none:
        case reason_code::registered:
        case reason_code::admission_wait:
        case reason_code::admission_ready:
        case reason_code::capacity_blocked:
        case reason_code::dispatched:
        case reason_code::slot_rebind:
        case reason_code::yielded:
        case reason_code::progress_observed:
        case reason_code::client_cancel:
        case reason_code::server_cancel:
        case reason_code::queue_timeout:
        case reason_code::run_timeout:
        case reason_code::completed:
        case reason_code::request_failed:
            return true;
    }
    return false;
}

bool valid_terminal(lifecycle value) {
    return value == lifecycle::completed || value == lifecycle::cancelled || value == lifecycle::timed_out ||
           value == lifecycle::failed;
}

}  // namespace

bool is_terminal(lifecycle value) {
    return valid_terminal(value);
}

bool request_handle::operator==(const request_handle & other) const {
    return id == other.id && epoch == other.epoch;
}

bool request_counts::operator==(const request_counts & other) const {
    return prompt_tokens == other.prompt_tokens && cached_prompt_tokens == other.cached_prompt_tokens &&
           requested_output_tokens == other.requested_output_tokens &&
           observed_output_tokens == other.observed_output_tokens;
}

bool request_estimates::operator==(const request_estimates & other) const {
    return predicted_prefill_us == other.predicted_prefill_us &&
           predicted_cache_restore_us == other.predicted_cache_restore_us &&
           predicted_decode_us == other.predicted_decode_us &&
           predicted_gpu_us == other.predicted_gpu_us && predicted_memory_bytes == other.predicted_memory_bytes &&
           predicted_output_tokens == other.predicted_output_tokens;
}

bool binding_lease::operator==(const binding_lease & other) const {
    return request == other.request && slot == other.slot && slot_generation == other.slot_generation;
}

bool request_snapshot::operator==(const request_snapshot & other) const {
    return handle == other.handle && lane == other.lane && arrival_us == other.arrival_us &&
           virtual_runtime_us == other.virtual_runtime_us && debt_us == other.debt_us && counts == other.counts &&
           estimates == other.estimates && last_reason == other.last_reason && state == other.state &&
           queue == other.queue && cancel_requested == other.cancel_requested &&
           timeout_expired == other.timeout_expired && revision == other.revision && bindings == other.bindings;
}

bool registry_event::operator==(const registry_event & other) const {
    return sequence == other.sequence && at_us == other.at_us && kind == other.kind && request == other.request &&
           lane == other.lane && reason == other.reason && lifecycle_before == other.lifecycle_before &&
           lifecycle_after == other.lifecycle_after && queue_before == other.queue_before &&
           queue_after == other.queue_after && slot_from == other.slot_from && slot_to == other.slot_to &&
           slot_generation == other.slot_generation;
}

bool event_log_snapshot::operator==(const event_log_snapshot & other) const {
    return events == other.events && first_sequence == other.first_sequence && next_sequence == other.next_sequence &&
           total_events == other.total_events && dropped_events == other.dropped_events;
}

bool registry_summary::operator==(const registry_summary & other) const {
    return active_requests == other.active_requests && occupied_slots == other.occupied_slots &&
           retained_events == other.retained_events && event_capacity == other.event_capacity &&
           total_events == other.total_events && dropped_events == other.dropped_events;
}

operation_result::operator bool() const {
    return code == result_code::ok;
}

registration_result::operator bool() const {
    return code == result_code::ok;
}

binding_result::operator bool() const {
    return code == result_code::ok;
}

struct request_registry::impl {
    struct slot_entry {
        uint64_t                     generation = 0;
        std::optional<binding_lease> binding;
    };

    explicit impl(registry_config value) : config(value), slots(value.max_slots) {
        if (config.max_requests == 0 || config.max_slots == 0 || config.max_slots >= no_slot ||
            config.max_bindings_per_request == 0 || config.max_bindings_per_request > config.max_slots) {
            throw std::invalid_argument("invalid request registry capacity");
        }
    }

    using request_map = std::map<request_id, request_snapshot>;

    result_code find(request_handle handle, request_map::iterator & it) {
        if (handle.id == 0 || handle.epoch == 0) {
            return result_code::stale_handle;
        }
        it = requests.find(handle.id);
        if (it == requests.end()) {
            return result_code::unknown_request;
        }
        if (!(it->second.handle == handle)) {
            return result_code::stale_handle;
        }
        return result_code::ok;
    }

    result_code find(request_handle handle, request_map::const_iterator & it) const {
        if (handle.id == 0 || handle.epoch == 0) {
            return result_code::stale_handle;
        }
        it = requests.find(handle.id);
        if (it == requests.end()) {
            return result_code::unknown_request;
        }
        if (!(it->second.handle == handle)) {
            return result_code::stale_handle;
        }
        return result_code::ok;
    }

    result_code validate_lease(const binding_lease & lease) const {
        if (lease.slot >= slots.size() || lease.slot_generation == 0 || lease.request.id == 0 ||
            lease.request.epoch == 0) {
            return result_code::stale_lease;
        }
        const auto & current = slots[lease.slot].binding;
        return current && *current == lease ? result_code::ok : result_code::stale_lease;
    }

    void append_event(registry_event event) noexcept {
        event.sequence = next_event_sequence++;
        ++total_events;
        if (config.event_capacity == 0) {
            ++dropped_events;
            return;
        }

        try {
            // Append before dropping the oldest event so allocation failure
            // cannot damage the durable request transition or the retained
            // log. The event ring is diagnostic and may drop this event under
            // memory pressure; request state remains authoritative.
            event_ring.push_back(event);
        } catch (...) {
            ++dropped_events;
            return;
        }
        if (event_ring.size() > config.event_capacity) {
            event_ring.pop_front();
            ++dropped_events;
        }
    }

    static registry_event make_event(const request_snapshot & request,
                                     event_kind               kind,
                                     reason_code              reason,
                                     uint64_t                 at_us) {
        registry_event event;
        event.at_us            = at_us;
        event.kind             = kind;
        event.request          = request.handle;
        event.lane             = request.lane;
        event.reason           = reason;
        event.lifecycle_before = request.state;
        event.lifecycle_after  = request.state;
        event.queue_before     = request.queue;
        event.queue_after      = request.queue;
        return event;
    }

    static request_snapshot detached_snapshot(request_snapshot snapshot) {
        std::sort(snapshot.bindings.begin(), snapshot.bindings.end(),
                  [](const binding_lease & lhs, const binding_lease & rhs) { return lhs.slot < rhs.slot; });
        return snapshot;
    }

    registry_config            config;
    mutable std::mutex         mutex;
    request_map                requests;
    std::vector<slot_entry>    slots;
    std::deque<registry_event> event_ring;
    uint64_t                   next_request_epoch  = 1;
    uint64_t                   next_event_sequence = 1;
    uint64_t                   total_events        = 0;
    uint64_t                   dropped_events      = 0;
    size_t                     occupied_slots      = 0;
};

request_registry::request_registry(registry_config config) : pimpl(new impl(config)) {}

request_registry::~request_registry() = default;

request_registry::request_registry(request_registry && other) noexcept = default;

request_registry & request_registry::operator=(request_registry && other) noexcept = default;

registration_result request_registry::register_request(const request_registration & request) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    if (request.id == 0 || !valid_lane(request.lane) || !valid_reason(request.reason) ||
        request.counts.cached_prompt_tokens > request.counts.prompt_tokens ||
        (request.counts.requested_output_tokens != 0 &&
         request.counts.observed_output_tokens > request.counts.requested_output_tokens)) {
        return { result_code::invalid_registration, {} };
    }
    if (pimpl->requests.find(request.id) != pimpl->requests.end()) {
        return { result_code::duplicate_request, {} };
    }
    if (pimpl->requests.size() >= pimpl->config.max_requests) {
        return { result_code::request_capacity_exhausted, {} };
    }
    if (pimpl->next_request_epoch == 0) {
        return { result_code::generation_exhausted, {} };
    }

    request_snapshot state;
    state.handle             = { request.id, pimpl->next_request_epoch++ };
    state.lane               = request.lane;
    state.arrival_us         = request.arrival_us;
    state.virtual_runtime_us = request.virtual_runtime_us;
    state.debt_us            = request.debt_us;
    state.counts             = request.counts;
    state.estimates          = request.estimates;
    state.last_reason        = request.reason;
    state.state              = lifecycle::registered;
    state.queue              = queue_state::none;
    state.revision           = 1;

    pimpl->requests.emplace(request.id, state);

    registry_event event   = impl::make_event(state, event_kind::registered, request.reason, request.arrival_us);
    event.lifecycle_before = lifecycle::absent;
    event.queue_before     = queue_state::none;
    pimpl->append_event(event);
    return { result_code::ok, state.handle };
}

operation_result request_registry::set_queue_state(request_handle handle,
                                                   queue_state    state,
                                                   reason_code    reason,
                                                   uint64_t       at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, false };
    }
    if (!valid_queue(state)) {
        return { result_code::invalid_queue_state, false };
    }
    if (!request.bindings.empty()) {
        return { result_code::active_bindings, false };
    }
    if (state != queue_state::none && (request.cancel_requested || request.timeout_expired)) {
        return { result_code::request_not_runnable, false };
    }
    const lifecycle next_lifecycle = state == queue_state::none ? lifecycle::registered : lifecycle::queued;
    if (request.queue == state && request.state == next_lifecycle) {
        return { result_code::ok, false };
    }

    registry_event event = impl::make_event(request, event_kind::queue_changed, reason, at_us);
    request.queue        = state;
    request.state        = next_lifecycle;
    request.last_reason  = reason;
    ++request.revision;
    event.lifecycle_after = request.state;
    event.queue_after     = request.queue;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

operation_result request_registry::update_progress(request_handle           handle,
                                                   const request_progress & progress,
                                                   reason_code              reason,
                                                   uint64_t                 at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, false };
    }
    if (progress.virtual_runtime_us < request.virtual_runtime_us ||
        progress.observed_output_tokens < request.counts.observed_output_tokens ||
        (request.counts.requested_output_tokens != 0 &&
         progress.observed_output_tokens > request.counts.requested_output_tokens)) {
        return { result_code::invalid_progress, false };
    }

    registry_event event                  = impl::make_event(request, event_kind::progress_updated, reason, at_us);
    request.virtual_runtime_us            = progress.virtual_runtime_us;
    request.debt_us                       = progress.debt_us;
    request.counts.observed_output_tokens = progress.observed_output_tokens;
    request.estimates                     = progress.estimates;
    request.last_reason                   = reason;
    ++request.revision;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

binding_result request_registry::bind(request_handle handle, slot_id slot, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, {} };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, {} };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, {} };
    }
    if (request.cancel_requested || request.timeout_expired) {
        return { result_code::request_not_runnable, {} };
    }
    if (slot >= pimpl->slots.size()) {
        return { result_code::slot_out_of_range, {} };
    }
    if (pimpl->slots[slot].binding) {
        return { result_code::slot_occupied, {} };
    }
    if (request.bindings.size() >= pimpl->config.max_bindings_per_request) {
        return { result_code::binding_capacity_exhausted, {} };
    }
    if (pimpl->slots[slot].generation == std::numeric_limits<uint64_t>::max()) {
        return { result_code::generation_exhausted, {} };
    }

    binding_lease  lease = { request.handle, slot, ++pimpl->slots[slot].generation };
    registry_event event = impl::make_event(request, event_kind::bound, reason, at_us);
    request.bindings.push_back(lease);
    request.queue       = queue_state::none;
    request.state       = lifecycle::executing;
    request.last_reason = reason;
    ++request.revision;
    pimpl->slots[slot].binding = lease;
    ++pimpl->occupied_slots;
    event.lifecycle_after = request.state;
    event.queue_after     = request.queue;
    event.slot_to         = slot;
    event.slot_generation = lease.slot_generation;
    pimpl->append_event(event);
    return { result_code::ok, lease };
}

operation_result request_registry::unbind(binding_lease lease, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    if (pimpl->validate_lease(lease) != result_code::ok) {
        return { result_code::stale_lease, false };
    }

    auto it = pimpl->requests.find(lease.request.id);
    if (it == pimpl->requests.end() || !(it->second.handle == lease.request)) {
        return { result_code::stale_lease, false };
    }
    request_snapshot & request    = it->second;
    const auto         binding_it = std::find(request.bindings.begin(), request.bindings.end(), lease);
    if (binding_it == request.bindings.end()) {
        return { result_code::stale_lease, false };
    }

    registry_event event = impl::make_event(request, event_kind::unbound, reason, at_us);
    request.bindings.erase(binding_it);
    request.state       = request.bindings.empty() ? lifecycle::registered : lifecycle::executing;
    request.last_reason = reason;
    ++request.revision;
    pimpl->slots[lease.slot].binding.reset();
    --pimpl->occupied_slots;
    event.lifecycle_after = request.state;
    event.slot_from       = lease.slot;
    event.slot_generation = lease.slot_generation;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

binding_result request_registry::rebind(binding_lease lease, slot_id new_slot, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, {} };
    }
    if (pimpl->validate_lease(lease) != result_code::ok) {
        return { result_code::stale_lease, {} };
    }
    if (new_slot >= pimpl->slots.size()) {
        return { result_code::slot_out_of_range, {} };
    }
    if (new_slot == lease.slot || pimpl->slots[new_slot].binding) {
        return { result_code::slot_occupied, {} };
    }
    if (pimpl->slots[new_slot].generation == std::numeric_limits<uint64_t>::max()) {
        return { result_code::generation_exhausted, {} };
    }

    auto it = pimpl->requests.find(lease.request.id);
    if (it == pimpl->requests.end() || !(it->second.handle == lease.request)) {
        return { result_code::stale_lease, {} };
    }
    request_snapshot & request    = it->second;
    const auto         binding_it = std::find(request.bindings.begin(), request.bindings.end(), lease);
    if (binding_it == request.bindings.end()) {
        return { result_code::stale_lease, {} };
    }
    if (request.cancel_requested || request.timeout_expired) {
        return { result_code::request_not_runnable, {} };
    }

    const binding_lease next  = { request.handle, new_slot, ++pimpl->slots[new_slot].generation };
    registry_event      event = impl::make_event(request, event_kind::rebound, reason, at_us);
    *binding_it               = next;
    request.last_reason       = reason;
    ++request.revision;
    pimpl->slots[lease.slot].binding.reset();
    pimpl->slots[new_slot].binding = next;
    event.slot_from                = lease.slot;
    event.slot_to                  = new_slot;
    event.slot_generation          = next.slot_generation;
    pimpl->append_event(event);
    return { result_code::ok, next };
}

operation_result request_registry::mark_cancel_requested(request_handle handle, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, false };
    }
    if (request.cancel_requested) {
        return { result_code::ok, false };
    }
    registry_event event     = impl::make_event(request, event_kind::cancel_requested, reason, at_us);
    request.cancel_requested = true;
    request.last_reason      = reason;
    ++request.revision;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

operation_result request_registry::mark_timeout_expired(request_handle handle, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, false };
    }
    if (request.timeout_expired) {
        return { result_code::ok, false };
    }
    registry_event event    = impl::make_event(request, event_kind::timeout_expired, reason, at_us);
    request.timeout_expired = true;
    request.last_reason     = reason;
    ++request.revision;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

operation_result request_registry::mark_terminal(request_handle handle,
                                                 lifecycle      terminal_state,
                                                 reason_code    reason,
                                                 uint64_t       at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    request_snapshot & request = it->second;
    if (is_terminal(request.state)) {
        return { result_code::terminal_request, false };
    }
    if (!valid_terminal(terminal_state) || (terminal_state == lifecycle::cancelled && !request.cancel_requested) ||
        (terminal_state == lifecycle::timed_out && !request.timeout_expired)) {
        return { result_code::invalid_transition, false };
    }
    if (!request.bindings.empty()) {
        return { result_code::active_bindings, false };
    }

    registry_event event = impl::make_event(request, event_kind::terminal, reason, at_us);
    request.state        = terminal_state;
    request.queue        = queue_state::none;
    request.last_reason  = reason;
    ++request.revision;
    event.lifecycle_after = request.state;
    event.queue_after     = request.queue;
    pimpl->append_event(event);
    return { result_code::ok, true };
}

operation_result request_registry::remove_terminal(request_handle handle, reason_code reason, uint64_t at_us) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (!valid_reason(reason)) {
        return { result_code::invalid_reason, false };
    }
    impl::request_map::iterator it;
    const result_code           found = pimpl->find(handle, it);
    if (found != result_code::ok) {
        return { found, false };
    }
    const request_snapshot & request = it->second;
    if (!is_terminal(request.state)) {
        return { result_code::not_terminal, false };
    }
    if (!request.bindings.empty()) {
        return { result_code::active_bindings, false };
    }

    // Build the scalar event before erasing the request, without copying its
    // bindings vector. Removal follows a durable terminal transition and must
    // not introduce another allocation/failure point.
    registry_event event  = impl::make_event(request, event_kind::removed, reason, at_us);
    event.lifecycle_after = lifecycle::absent;
    event.queue_after     = queue_state::none;
    pimpl->requests.erase(it);
    pimpl->append_event(event);
    return { result_code::ok, true };
}

std::optional<request_snapshot> request_registry::get(request_handle handle) const {
    std::lock_guard<std::mutex>       lock(pimpl->mutex);
    impl::request_map::const_iterator it;
    if (pimpl->find(handle, it) != result_code::ok) {
        return std::nullopt;
    }
    return impl::detached_snapshot(it->second);
}

std::vector<request_snapshot> request_registry::snapshot() const {
    std::lock_guard<std::mutex>   lock(pimpl->mutex);
    std::vector<request_snapshot> result;
    result.reserve(pimpl->requests.size());
    for (const auto & entry : pimpl->requests) {
        result.push_back(impl::detached_snapshot(entry.second));
    }
    return result;
}

event_log_snapshot request_registry::events() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    event_log_snapshot          result;
    result.events.assign(pimpl->event_ring.begin(), pimpl->event_ring.end());
    result.next_sequence  = pimpl->next_event_sequence;
    result.total_events   = pimpl->total_events;
    result.dropped_events = pimpl->dropped_events;
    result.first_sequence = result.events.empty() ? result.next_sequence : result.events.front().sequence;
    return result;
}

registry_summary request_registry::summary() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return {
        pimpl->requests.size(),       pimpl->occupied_slots, pimpl->event_ring.size(),
        pimpl->config.event_capacity, pimpl->total_events,   pimpl->dropped_events,
    };
}

const registry_config & request_registry::config() const {
    return pimpl->config;
}

}  // namespace server_request_registry

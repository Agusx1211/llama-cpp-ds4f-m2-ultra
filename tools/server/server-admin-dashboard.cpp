#include "server-admin-dashboard.h"

#include "http.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <optional>

namespace server_admin_dashboard {

namespace {

using namespace server_request_registry;

std::atomic<size_t> live_streams{0};

std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::optional<std::string> find_header(
        const std::map<std::string, std::string> & headers,
        const char *                               name) {
    const std::string expected = ascii_lower(name);
    for (const auto & [key, value] : headers) {
        if (ascii_lower(key) == expected) {
            return value;
        }
    }
    return std::nullopt;
}

bool has_header(const std::map<std::string, std::string> & headers, const char * name) {
    return find_header(headers, name).has_value();
}

bool unsafe_browser_origin(const std::map<std::string, std::string> & headers) {
    const auto origin = find_header(headers, "Origin");
    if (origin) {
        try {
            const common_http_url parsed = common_http_parse_url(*origin);
            const std::string host = ascii_lower(parsed.host);
            if (!parsed.user.empty() || !parsed.password.empty() || parsed.path != "/" ||
                (host != "localhost" && !server_trusted_scheduling::is_loopback_address(host))) {
                return true;
            }
        } catch (const std::exception &) {
            return true;
        }
    }
    const auto site = find_header(headers, "Sec-Fetch-Site");
    return site && ascii_lower(*site) == "cross-site";
}

const char * lane_name(trusted_lane lane) {
    switch (lane) {
        case trusted_lane::low:    return "low";
        case trusted_lane::normal: return "normal";
        case trusted_lane::fast:   return "fast";
        case trusted_lane::count:  break;
    }
    return "normal";
}

const char * lane_name(server_scheduler::lane lane) {
    switch (lane) {
        case server_scheduler::lane::low:    return "low";
        case server_scheduler::lane::normal: return "normal";
        case server_scheduler::lane::fast:   return "fast";
        case server_scheduler::lane::count:  break;
    }
    return "normal";
}

const char * lifecycle_name(lifecycle value) {
    switch (value) {
        case lifecycle::absent:     return "absent";
        case lifecycle::registered: return "registered";
        case lifecycle::queued:     return "queued";
        case lifecycle::executing:  return "executing";
        case lifecycle::completed:  return "complete";
        case lifecycle::cancelled:  return "cancelled";
        case lifecycle::timed_out:  return "failed";
        case lifecycle::failed:     return "failed";
        case lifecycle::count:      break;
    }
    return "failed";
}

const char * dashboard_state(const request_snapshot & request) {
    switch (request.state) {
        case lifecycle::executing:
            return request.counts.observed_output_tokens == 0 ? "prefill" : "decode";
        case lifecycle::completed:
            return "complete";
        case lifecycle::cancelled:
            return "cancelled";
        case lifecycle::timed_out:
        case lifecycle::failed:
            return "failed";
        case lifecycle::absent:
        case lifecycle::registered:
        case lifecycle::queued:
        case lifecycle::count:
            return "queued";
    }
    return "failed";
}

const char * reason_name(reason_code value) {
    switch (value) {
        case reason_code::none:              return "none";
        case reason_code::registered:        return "registered";
        case reason_code::admission_wait:    return "admission_wait";
        case reason_code::admission_ready:   return "admission_ready";
        case reason_code::capacity_blocked:  return "capacity_blocked";
        case reason_code::dispatched:        return "dispatched";
        case reason_code::slot_rebind:       return "slot_rebind";
        case reason_code::yielded:           return "yielded";
        case reason_code::progress_observed: return "progress_observed";
        case reason_code::client_cancel:     return "client_cancel";
        case reason_code::server_cancel:     return "server_cancel";
        case reason_code::queue_timeout:     return "queue_timeout";
        case reason_code::run_timeout:       return "run_timeout";
        case reason_code::completed:         return "completed";
        case reason_code::request_failed:    return "request_failed";
    }
    return "unknown";
}

const char * event_name(event_kind value) {
    switch (value) {
        case event_kind::registered:       return "registered";
        case event_kind::queue_changed:    return "queue_changed";
        case event_kind::progress_updated: return "progress_updated";
        case event_kind::bound:            return "bound";
        case event_kind::unbound:          return "unbound";
        case event_kind::rebound:          return "rebound";
        case event_kind::cancel_requested: return "cancel_requested";
        case event_kind::timeout_expired:  return "timeout_expired";
        case event_kind::terminal:         return "terminal";
        case event_kind::removed:          return "removed";
    }
    return "unknown";
}

std::string request_id(request_handle handle) {
    return std::to_string(handle.id) + ":" + std::to_string(handle.epoch);
}

std::string monotonic_label(uint64_t at_us) {
    return "monotonic:" + std::to_string(at_us) + "us";
}

size_t lane_index(trusted_lane lane) {
    return static_cast<size_t>(lane);
}

json request_json(const request_snapshot & request, uint64_t now_us) {
    const uint64_t age_us = now_us >= request.arrival_us ? now_us - request.arrival_us : 0;
    return {
        { "id", request_id(request.handle) },
        { "lane", lane_name(request.lane) },
        { "state", dashboard_state(request) },
        { "arrival_at", monotonic_label(request.arrival_us) },
        { "age_ms", age_us / 1000.0 },
        { "prompt_tokens", request.counts.prompt_tokens },
        { "cache_hit_tokens", request.counts.cached_prompt_tokens },
        { "output_tokens", request.counts.observed_output_tokens },
        { "requested_output_tokens", request.counts.requested_output_tokens == 0
              ? json(nullptr)
              : json(request.counts.requested_output_tokens) },
        { "blocker", reason_name(request.last_reason) },
        { "predicted_start_ms", json::array({0, 0}) },
        { "ttft_ms", nullptr },
        { "tbt_ms", nullptr },
        { "kv", {
            { "logical_bytes", 0 },
            { "unique_bytes", 0 },
            { "lineage", "unavailable" },
        } },
        { "scheduler_reasons", json::array({reason_name(request.last_reason)}) },
        { "preemptions", 0 },
        { "dspark_cycles", 0 },
        { "content", {
            { "prompt", "" },
            { "output", "" },
            { "retained", false },
        } },
    };
}

json lane_json(
        trusted_lane                                      lane,
        const std::vector<request_snapshot> &             requests,
        const server_request_runtime::dispatch_permit_snapshot & permits,
        uint64_t                                          now_us) {
    size_t queued = 0;
    size_t active = 0;
    uint64_t oldest_arrival = now_us;
    bool has_queued = false;
    for (const auto & request : requests) {
        if (request.lane != lane) {
            continue;
        }
        if (request.state == lifecycle::executing) {
            ++active;
        } else if (!is_terminal(request.state)) {
            ++queued;
            has_queued = true;
            oldest_arrival = std::min(oldest_arrival, request.arrival_us);
        }
    }
    const uint64_t oldest_wait_us = has_queued && now_us >= oldest_arrival ? now_us - oldest_arrival : 0;
    const size_t index = lane_index(lane);
    return {
        { "id", lane_name(lane) },
        { "queued", queued },
        { "active", active },
        { "oldest_wait_ms", oldest_wait_us / 1000.0 },
        { "service_deficit", 0 },
        { "bypass_count", 0 },
        { "predicted_start_ms", json::array({0, 0}) },
        { "claimed_permits", permits.claimed[index] },
        { "bound_permits", permits.bound[index] },
    };
}

uint64_t latest_sequence(const event_log_snapshot & events) {
    return events.next_sequence == 0 ? 0 : events.next_sequence - 1;
}

json availability_json() {
    return {
        { "server_metrics", false },
        { "scheduler_predictions", false },
        { "request_latency", false },
        { "request_kv", false },
        { "request_preemption", false },
        { "content", false },
        { "allocator", false },
        { "cache", false },
        { "disks", false },
        { "dspark", false },
        { "capture", false },
        { "fast_refill", true },
    };
}

json lanes_json(const server_queue_request_state & source, uint64_t now_us) {
    json lanes = json::array();
    lanes.push_back(lane_json(trusted_lane::low, source.requests, source.permits, now_us));
    lanes.push_back(lane_json(trusted_lane::normal, source.requests, source.permits, now_us));
    lanes.push_back(lane_json(trusted_lane::fast, source.requests, source.permits, now_us));
    return lanes;
}

json registry_json(const server_queue_request_state & source) {
    return {
        { "active_requests", source.summary.active_requests },
        { "occupied_slots", source.summary.occupied_slots },
        { "retained_events", source.summary.retained_events },
        { "event_capacity", source.summary.event_capacity },
        { "total_events", source.summary.total_events },
        { "dropped_events", source.summary.dropped_events },
        { "claimed_permits", source.permits.claimed },
        { "bound_permits", source.permits.bound },
        { "total_permits", source.permits.total },
    };
}

json fast_refill_json(const server_queue_request_state & source, uint64_t now_us) {
    const auto & refill = source.fast_refill;
    const auto status = refill.status_at(now_us, source.permits.total);
    return {
        { "configuration", {
            { "enabled", refill.enabled },
            { "max_members", refill.max_members_per_cohort },
            { "window_ms", refill.window_us / 1000.0 },
        } },
        { "cohort", {
            { "active", refill.cohort_active },
            { "selection_open", refill.selection_open },
            { "dominant_lane", refill.cohort_active ? json(lane_name(refill.dominant)) : json(nullptr) },
            { "limit", refill.cohort_limit },
        } },
        { "refill", {
            { "fast_members_used", refill.fast_members_used },
            { "fast_members_remaining", status.fast_members_remaining },
            { "deadline_at", refill.deadline_us == 0 ? json(nullptr) : json(monotonic_label(refill.deadline_us)) },
            { "remaining_ms", status.remaining_us / 1000.0 },
            { "deadline_expired", refill.deadline_us != 0 && now_us >= refill.deadline_us },
            { "window_open", status.window_open },
            { "one_member_eligible_now", status.one_member_eligible_now },
        } },
    };
}

json event_envelope(const registry_event & event) {
    const bool has_slot = event.slot_to != no_slot || event.slot_from != no_slot;
    const slot_id slot = event.slot_to != no_slot ? event.slot_to : event.slot_from;
    return {
        { "schema_version", schema_version },
        { "id", std::to_string(event.sequence) },
        { "sequence", event.sequence },
        { "wall_time", monotonic_label(event.at_us) },
        { "monotonic_ms", event.at_us / 1000.0 },
        { "type", "request.remove" },
        { "request_id", request_id(event.request) },
        { "slot_id", has_slot ? json(slot) : json(nullptr) },
        { "sequence_id", nullptr },
        { "lane", lane_name(event.lane) },
        { "scheduler_epoch", nullptr },
        { "decision_id", nullptr },
        { "before", lifecycle_name(event.lifecycle_before) },
        { "after", lifecycle_name(event.lifecycle_after) },
        { "reason_code", reason_name(event.reason) },
        { "payload", { { "request_id", request_id(event.request) } } },
    };
}

std::string sse_frame(const json & event) {
    const std::string payload = event.dump();
    return "id: " + event.at("id").get<std::string>() + "\n"
           "event: admin\n"
           "data: " + payload + "\n\n";
}

} // namespace

authorization_result authorize(
        const server_trusted_scheduling::control & operator_control,
        bool                                       api_authentication_enabled,
        const std::string &                        remote_address,
        const std::map<std::string, std::string> & headers,
        bool                                       ambiguous_operator_header,
        bool                                       ambiguous_last_event_id,
        const std::string &                        query_string) {
    if (!operator_control.enabled()) {
        return { authorization_status::disabled, "read-only dashboard is disabled" };
    }
    if (!api_authentication_enabled) {
        return { authorization_status::api_authentication_disabled,
                 "read-only dashboard requires configured API-key authentication" };
    }
    if (!server_trusted_scheduling::is_loopback_address(remote_address)) {
        return { authorization_status::non_loopback, "read-only dashboard requires loopback ingress" };
    }
    if (ambiguous_operator_header || ambiguous_last_event_id) {
        return { authorization_status::ambiguous_header, "ambiguous read-only dashboard header" };
    }
    if (has_header(headers, server_trusted_scheduling::lane_header) ||
        has_header(headers, server_trusted_scheduling::tag_header)) {
        return { authorization_status::classification_header,
                 "read-only dashboard requests cannot carry scheduling classification headers" };
    }
    if (!query_string.empty()) {
        return { authorization_status::query_not_allowed,
                 "read-only dashboard credentials and cursors must use headers, not query parameters" };
    }
    if (unsafe_browser_origin(headers)) {
        return { authorization_status::cross_site_browser,
                 "dashboard browser requests require a loopback origin and same-site ingress" };
    }
    if (!operator_control.authorize_operator(remote_address, headers)) {
        return { authorization_status::missing_or_invalid_credential,
                 "read-only dashboard requires the loopback operator credential" };
    }
    return { authorization_status::allowed, "allowed" };
}

bool parse_last_event_id(const std::map<std::string, std::string> & headers, uint64_t & result) {
    result = 0;
    const auto value = find_header(headers, "Last-Event-ID");
    if (!value || value->empty()) {
        return true;
    }
    const char * begin = value->data();
    const char * end   = begin + value->size();
    const auto parsed = std::from_chars(begin, end, result, 10);
    return parsed.ec == std::errc() && parsed.ptr == end;
}

json make_snapshot(const server_queue_request_state & source, uint64_t now_us) {
    json requests = json::array();
    for (const auto & request : source.requests) {
        requests.push_back(request_json(request, now_us));
    }

    json timeline = json::array();
    const size_t begin = source.events.events.size() > maximum_timeline_events
            ? source.events.events.size() - maximum_timeline_events
            : 0;
    for (size_t index = begin; index < source.events.events.size(); ++index) {
        const auto & event = source.events.events[index];
        timeline.push_back({
            { "id", "registry-" + std::to_string(event.sequence) },
            { "at", monotonic_label(event.at_us) },
            { "type", event_name(event.kind) },
            { "label", reason_name(event.reason) },
            { "request_id", request_id(event.request) },
            { "lane", lane_name(event.lane) },
        });
    }

    return {
        { "schema_version", schema_version },
        { "sequence", latest_sequence(source.events) },
        { "generated_at", monotonic_label(now_us) },
        { "availability", availability_json() },
        { "registry", registry_json(source) },
        { "fast_refill", fast_refill_json(source, now_us) },
        { "server", {
            { "health", "unavailable" },
            { "build", "unavailable" },
            { "model", "unavailable" },
            { "uptime_seconds", 0 },
            { "rss_bytes", 0 },
            { "footprint_bytes", 0 },
            { "swap_bytes", 0 },
            { "memory_pressure_percent", 0 },
            { "decode_width", source.permits.total },
            { "aggregate_tokens_per_second", 0 },
        } },
        { "lanes", lanes_json(source, now_us) },
        { "requests", std::move(requests) },
        { "allocator", { { "pools", json::array() } } },
        { "cache", { { "objects", json::array() } } },
        { "disks", json::array() },
        { "dspark", {
            { "mode", "unavailable" },
            { "scheduled_decode_width", 0 },
            { "proposals", 0 },
            { "accepted", 0 },
            { "acceptance_by_position", json::array() },
        } },
        { "capture", {
            { "mode", "unavailable" },
            { "healthy", false },
            { "queued_records", 0 },
            { "written_records", 0 },
            { "dropped_records", 0 },
            { "bytes_written", 0 },
            { "last_error", nullptr },
        } },
        { "timeline", std::move(timeline) },
    };
}

resume_batch make_resume_batch(
        const server_queue_request_state & source,
        uint64_t                           after_sequence,
        uint64_t                           now_us) {
    resume_batch result;
    result.latest_sequence = latest_sequence(source.events);
    result.oldest_available_sequence = source.events.first_sequence;
    if (after_sequence > result.latest_sequence) {
        result.status = resume_status::future_cursor;
        return result;
    }
    if (after_sequence == result.latest_sequence) {
        return result;
    }

    auto event_it = std::find_if(source.events.events.begin(), source.events.events.end(),
            [after_sequence](const registry_event & event) { return event.sequence > after_sequence; });
    if (event_it == source.events.events.end() || event_it->sequence != after_sequence + 1) {
        result.status = resume_status::gap;
        return result;
    }

    const auto request_for = [&source](request_handle handle) -> const request_snapshot * {
        const auto found = std::find_if(source.requests.begin(), source.requests.end(),
                [handle](const request_snapshot & request) { return request.handle == handle; });
        return found == source.requests.end() ? nullptr : &*found;
    };

    const json lanes = lanes_json(source, now_us);
    const json registry = registry_json(source);
    const json fast_refill = fast_refill_json(source, now_us);

    size_t bytes = 0;
    for (; event_it != source.events.events.end() && result.frames.size() < maximum_events_per_chunk; ++event_it) {
        json event = event_envelope(*event_it);
        if (event_it->kind != event_kind::removed) {
            if (const request_snapshot * request = request_for(event_it->request)) {
                event["type"] = "request.upsert";
                event["payload"] = {
                    { "request", request_json(*request, now_us) },
                    { "lanes", lanes },
                    { "registry", registry },
                    { "fast_refill", fast_refill },
                };
            }
        }
        if (event["type"] == "request.remove") {
            event["payload"]["lanes"] = lanes;
            event["payload"]["registry"] = registry;
            event["payload"]["fast_refill"] = fast_refill;
        }
        std::string frame = sse_frame(event);
        if (frame.size() > maximum_sse_event_bytes || bytes + frame.size() > maximum_sse_chunk_bytes) {
            break;
        }
        bytes += frame.size();
        result.frames.push_back(std::move(frame));
    }
    result.status = result.frames.empty() ? resume_status::gap : resume_status::events;
    return result;
}

std::string make_overflow_frame(
        uint64_t event_sequence,
        uint64_t oldest_available_sequence,
        uint64_t now_us) {
    const json event = {
        { "schema_version", schema_version },
        { "id", std::to_string(event_sequence) },
        { "sequence", event_sequence },
        { "wall_time", monotonic_label(now_us) },
        { "monotonic_ms", now_us / 1000.0 },
        { "type", "stream.overflow" },
        { "request_id", nullptr },
        { "slot_id", nullptr },
        { "sequence_id", nullptr },
        { "lane", nullptr },
        { "scheduler_epoch", nullptr },
        { "decision_id", nullptr },
        { "before", nullptr },
        { "after", nullptr },
        { "reason_code", "registry_event_gap" },
        { "payload", { { "oldest_available_sequence", oldest_available_sequence } } },
    };
    return sse_frame(event);
}

std::shared_ptr<void> try_acquire_stream_lease() {
    size_t current = live_streams.load(std::memory_order_relaxed);
    while (current < maximum_live_streams) {
        if (live_streams.compare_exchange_weak(
                    current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return std::shared_ptr<void>(reinterpret_cast<void *>(1), [](void *) {
                live_streams.fetch_sub(1, std::memory_order_release);
            });
        }
    }
    return {};
}

size_t active_streams() {
    return live_streams.load(std::memory_order_acquire);
}

} // namespace server_admin_dashboard

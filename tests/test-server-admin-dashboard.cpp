#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-admin-dashboard.h"

#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

using namespace server_admin_dashboard;
using namespace server_request_registry;

namespace {

constexpr const char * secret = "0123456789abcdef0123456789abcdef";

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

request_snapshot request(
        uint64_t id,
        trusted_lane lane,
        lifecycle state,
        uint64_t arrival_us,
        uint64_t output_tokens = 0) {
    request_snapshot result;
    result.handle = { id, id + 10 };
    result.lane = lane;
    result.arrival_us = arrival_us;
    result.counts.prompt_tokens = id * 128;
    result.counts.cached_prompt_tokens = id * 32;
    result.counts.requested_output_tokens = 16;
    result.counts.observed_output_tokens = output_tokens;
    result.last_reason = state == lifecycle::executing ? reason_code::dispatched : reason_code::admission_wait;
    result.state = state;
    result.queue = state == lifecycle::queued ? queue_state::ready : queue_state::none;
    result.revision = 1;
    return result;
}

registry_event event(uint64_t sequence, const request_snapshot & request, event_kind kind) {
    registry_event result;
    result.sequence = sequence;
    result.at_us = 1000 + sequence;
    result.kind = kind;
    result.request = request.handle;
    result.lane = request.lane;
    result.reason = request.last_reason;
    result.lifecycle_before = lifecycle::registered;
    result.lifecycle_after = request.state;
    result.queue_before = queue_state::none;
    result.queue_after = request.queue;
    if (kind == event_kind::bound) {
        result.slot_to = 1;
        result.slot_generation = 3;
    }
    return result;
}

server_queue_request_state source_state() {
    server_queue_request_state source;
    source.requests = {
        request(1, trusted_lane::low, lifecycle::queued, 1000),
        request(2, trusted_lane::fast, lifecycle::executing, 2000, 4),
    };
    source.events.events = {
        event(1, source.requests[0], event_kind::registered),
        event(2, source.requests[0], event_kind::queue_changed),
        event(3, source.requests[1], event_kind::bound),
    };
    source.events.first_sequence = 1;
    source.events.next_sequence = 4;
    source.events.total_events = 3;
    source.summary = { 2, 1, 3, 1024, 3, 0 };
    source.permits.claimed[2] = 1;
    source.permits.bound[2] = 1;
    source.permits.total = 1;
    return source;
}

json frame_payload(const std::string & frame) {
    const size_t begin = frame.find("data: ");
    require(begin != std::string::npos, "SSE frame contains data");
    const size_t end = frame.find("\n\n", begin);
    require(end != std::string::npos, "SSE frame terminates");
    return json::parse(frame.substr(begin + 6, end - begin - 6));
}

void test_operator_authorization_is_loopback_read_only_and_trace_independent() {
    server_trusted_scheduling::control enabled({ secret, 0 });
    const std::map<std::string, std::string> headers = {
        { server_trusted_scheduling::token_header, secret },
    };
    require(static_cast<bool>(authorize(enabled, true, "127.0.0.1", headers, false, false, "")),
            "operator works with benchmark trace disabled");
    require(!authorize(enabled, true, "192.0.2.1", headers, false, false, ""),
            "remote operator request rejected");
    require(!authorize(enabled, true, "127.0.0.1", {}, false, false, ""),
            "missing operator credential rejected");
    require(!authorize(enabled, true, "127.0.0.1", headers, true, false, ""),
            "duplicate operator header rejected before map collapse");
    require(!authorize(enabled, true, "127.0.0.1", headers, false, true, ""),
            "duplicate resume header rejected before map collapse");
    require(!authorize(enabled, false, "127.0.0.1", headers, false, false, ""),
            "operator credential alone cannot replace API-key authentication");

    auto classified = headers;
    classified[server_trusted_scheduling::lane_header] = "fast";
    require(!authorize(enabled, true, "127.0.0.1", classified, false, false, ""),
            "read-only route rejects lane authority");
    auto cross_site = headers;
    cross_site["Sec-Fetch-Site"] = "cross-site";
    require(!authorize(enabled, true, "127.0.0.1", cross_site, false, false, ""),
            "cross-site browser request rejected");
    auto remote_origin = headers;
    remote_origin["Origin"] = "https://example.com";
    require(!authorize(enabled, true, "127.0.0.1", remote_origin, false, false, ""),
            "non-loopback browser origin rejected");
    auto loopback_origin = headers;
    loopback_origin["Origin"] = "http://127.0.0.1:8081";
    loopback_origin["Sec-Fetch-Site"] = "same-site";
    require(static_cast<bool>(authorize(enabled, true, "127.0.0.1", loopback_origin, false, false, "")),
            "loopback same-site browser origin accepted");
    require(!authorize(enabled, true, "127.0.0.1", headers, false, false, "token=secret"),
            "query parameters rejected");
}

void test_resume_header_is_strict_and_bounded() {
    uint64_t cursor = 99;
    require(parse_last_event_id({}, cursor) && cursor == 0, "missing cursor starts at zero");
    require(parse_last_event_id({ { "last-event-id", "42" } }, cursor) && cursor == 42,
            "case-insensitive decimal cursor accepted");
    require(!parse_last_event_id({ { "Last-Event-ID", "-1" } }, cursor), "negative cursor rejected");
    require(!parse_last_event_id({ { "Last-Event-ID", " 1" } }, cursor), "whitespace cursor rejected");
    require(!parse_last_event_id({ { "Last-Event-ID", "18446744073709551616" } }, cursor),
            "overflowing cursor rejected");
}

void test_snapshot_is_redacted_bounded_and_marks_unavailable_sources() {
    const json snapshot = make_snapshot(source_state(), 5000);
    const std::string encoded = snapshot.dump();
    require(encoded.size() < maximum_snapshot_bytes, "snapshot stays within response budget");
    require(snapshot.at("sequence") == 3, "snapshot sequence acknowledges registry state");
    require(snapshot.at("registry").at("active_requests") == 2, "registry summary exposed");
    require(snapshot.at("registry").at("total_permits") == 1, "permit total exposed");
    require(snapshot.at("availability").at("fast_refill") == true,
            "authoritative refill source marked available");
    require(snapshot.at("fast_refill").at("configuration").at("enabled") == false &&
                snapshot.at("fast_refill").at("configuration").at("max_members") == 0 &&
                snapshot.at("fast_refill").at("refill").at("deadline_at").is_null(),
            "default-off refill configuration is explicit");
    require(snapshot.at("availability").at("allocator") == false, "allocator marked unavailable");
    require(snapshot.at("availability").at("dspark") == false, "DSpark marked unavailable");
    require(snapshot.at("allocator").at("pools").empty(), "allocator facts remain empty");
    require(snapshot.at("disks").empty(), "disk facts remain empty");
    require(snapshot.at("requests").size() == 2, "requests serialized");
    require(snapshot.at("requests")[0].at("content").at("prompt") == "", "prompt is redacted");
    require(snapshot.at("requests")[1].at("content").at("output") == "", "output is redacted");
    require(snapshot.at("requests")[0].at("id") == "1:11", "request epoch prevents ID aliasing");
    require(snapshot.at("lanes")[0].at("queued") == 1, "low queue aggregate derived");
    require(snapshot.at("lanes")[2].at("active") == 1, "fast active aggregate derived");
    require(snapshot.at("lanes")[2].at("bound_permits") == 1, "fast bound permit exposed");
}

void test_refill_status_propagates_and_closes_at_exact_deadline() {
    auto source = source_state();
    source.fast_refill.enabled                = true;
    source.fast_refill.max_members_per_cohort = 4;
    source.fast_refill.window_us              = 1000;
    source.fast_refill.cohort_active          = true;
    source.fast_refill.selection_open         = false;
    source.fast_refill.dominant               = server_scheduler::lane::fast;
    source.fast_refill.cohort_limit           = 2;
    source.fast_refill.fast_members_used      = 1;
    source.fast_refill.deadline_us            = 7000;

    const json open = make_snapshot(source, 6000).at("fast_refill");
    require(open.at("cohort").at("active") == true &&
                open.at("cohort").at("selection_open") == false &&
                open.at("cohort").at("dominant_lane") == "fast" && open.at("cohort").at("limit") == 2,
            "snapshot preserves raw cohort selection state");
    require(open.at("refill").at("fast_members_used") == 1 &&
                open.at("refill").at("fast_members_remaining") == 3 &&
                open.at("refill").at("deadline_at") == "monotonic:7000us" &&
                open.at("refill").at("remaining_ms") == 1.0 &&
                open.at("refill").at("deadline_expired") == false &&
                open.at("refill").at("window_open") == true &&
                open.at("refill").at("one_member_eligible_now") == true,
            "snapshot derives an open refill window at dashboard sample time");

    const json expired = make_snapshot(source, 7000).at("fast_refill").at("refill");
    require(expired.at("remaining_ms") == 0.0 && expired.at("deadline_expired") == true &&
                expired.at("window_open") == false &&
                expired.at("one_member_eligible_now") == false,
            "exact deadline serializes zero remaining time and closed eligibility");

    const resume_batch batch = make_resume_batch(source, 2, 6000);
    require(batch.status == resume_status::events && batch.frames.size() == 1,
            "refill propagation event is resumable");
    const json event_refill = frame_payload(batch.frames[0]).at("payload").at("fast_refill");
    require(event_refill == open, "SSE request event carries the same authoritative refill status");

    source.fast_refill.selection_open    = true;
    source.fast_refill.fast_members_used = 4;
    const json exhausted = make_snapshot(source, 6500).at("fast_refill");
    require(exhausted.at("cohort").at("selection_open") == true &&
                exhausted.at("refill").at("fast_members_remaining") == 0 &&
                exhausted.at("refill").at("remaining_ms") == 0.5 &&
                exhausted.at("refill").at("window_open") == false &&
                exhausted.at("refill").at("one_member_eligible_now") == false,
            "exhausted quota stays closed despite raw selection-open state");
}

void test_resumption_is_contiguous_and_uses_current_request_state() {
    const auto source = source_state();
    const resume_batch batch = make_resume_batch(source, 1, 6000);
    require(batch.status == resume_status::events && batch.frames.size() == 2,
            "contiguous retained events resume");
    const json first = frame_payload(batch.frames[0]);
    const json second = frame_payload(batch.frames[1]);
    require(first.at("sequence") == 2 && second.at("sequence") == 3, "registry sequence preserved");
    require(first.at("type") == "request.upsert", "current request emits an upsert");
    require(first.at("payload").at("request").at("state") == "queued", "queued state serialized");
    require(second.at("payload").at("request").at("state") == "decode", "decode state serialized");
    require(second.at("payload").at("registry").at("total_events") == 3,
            "live event refreshes registry counters");
    require(second.at("slot_id") == 1, "authoritative bound slot exposed");
    require(!second.contains("scheduler_epoch") || second.at("scheduler_epoch").is_null(),
            "prefill cohort is not relabeled as scheduler epoch");
}

void test_gaps_future_cursors_and_chunk_limits_fail_closed() {
    auto source = source_state();
    require(make_resume_batch(source, 3, 6000).status == resume_status::waiting,
            "latest cursor waits without work");
    require(make_resume_batch(source, 4, 6000).status == resume_status::future_cursor,
            "future cursor requires resnapshot");

    source.events.events.erase(source.events.events.begin());
    source.events.events.erase(source.events.events.begin());
    source.events.first_sequence = 3;
    require(make_resume_batch(source, 1, 6000).status == resume_status::gap,
            "missing retained event requires resnapshot");

    source = source_state();
    source.events.events.clear();
    for (uint64_t sequence = 1; sequence <= 40; ++sequence) {
        source.events.events.push_back(event(sequence, source.requests[0], event_kind::progress_updated));
    }
    source.events.next_sequence = 41;
    source.events.total_events = 40;
    const auto bounded = make_resume_batch(source, 0, 6000);
    require(bounded.status == resume_status::events, "bounded batch emits retained events");
    require(bounded.frames.size() == maximum_events_per_chunk, "event count is bounded per chunk");
    size_t bytes = 0;
    for (const auto & frame : bounded.frames) {
        require(frame.size() <= maximum_sse_event_bytes, "individual SSE event bounded");
        bytes += frame.size();
    }
    require(bytes <= maximum_sse_chunk_bytes, "SSE chunk byte size bounded");

    const json overflow = frame_payload(make_overflow_frame(40, 9, 6000));
    require(overflow.at("type") == "stream.overflow", "gap signal has overflow type");
    require(overflow.at("payload").at("oldest_available_sequence") == 9,
            "gap signal advertises oldest retained sequence");
}

void test_live_stream_leases_bound_slow_clients() {
    std::vector<std::shared_ptr<void>> leases;
    for (size_t index = 0; index < maximum_live_streams; ++index) {
        auto lease = try_acquire_stream_lease();
        require(static_cast<bool>(lease), "in-capacity stream acquires a lease");
        leases.push_back(std::move(lease));
    }
    require(active_streams() == maximum_live_streams, "all bounded stream leases counted");
    require(!try_acquire_stream_lease(), "stream over capacity is rejected");
    leases.clear();
    require(active_streams() == 0, "closing streams releases every lease");
}

} // namespace

int main() {
    const struct {
        const char * name;
        void (*run)();
    } tests[] = {
        { "operator authorization is loopback read-only and trace independent",
          test_operator_authorization_is_loopback_read_only_and_trace_independent },
        { "resume header is strict and bounded", test_resume_header_is_strict_and_bounded },
        { "snapshot is redacted bounded and marks unavailable sources",
          test_snapshot_is_redacted_bounded_and_marks_unavailable_sources },
        { "refill status propagation and deadline closure",
          test_refill_status_propagates_and_closes_at_exact_deadline },
        { "resumption is contiguous and uses current request state",
          test_resumption_is_contiguous_and_uses_current_request_state },
        { "gaps future cursors and chunk limits fail closed",
          test_gaps_future_cursors_and_chunk_limits_fail_closed },
        { "live stream leases bound slow clients", test_live_stream_leases_bound_slow_clients },
    };

    for (const auto & test : tests) {
        try {
            test.run();
            std::printf("PASS: %s\n", test.name);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "FAIL: %s: %s\n", test.name, error.what());
            return 1;
        }
    }
    return 0;
}

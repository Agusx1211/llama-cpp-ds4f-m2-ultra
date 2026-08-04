#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-trusted-ingress.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

using namespace server_trusted_scheduling;

namespace {

constexpr const char * secret = "0123456789abcdef0123456789abcdef";

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::map<std::string, std::string> trusted_headers(
        const std::string & priority,
        const std::string & tag = "request-1") {
    return {
        { token_header, secret },
        { lane_header, priority },
        { tag_header, tag },
    };
}

void test_default_is_normal_and_lane_headers_fail_closed() {
    control disabled;
    const classification ordinary = disabled.classify("203.0.113.1", {});
    require(ordinary.status == classification_status::ordinary, "missing trusted header is ordinary");
    require(ordinary.priority == lane::normal, "ordinary requests preserve normal priority");

    const classification forged = disabled.classify("127.0.0.1", {
        { lane_header, "fast" },
    });
    require(forged.status == classification_status::rejected, "disabled trusted lane header is rejected");
    require(forged.priority == lane::normal, "rejected lane cannot elevate priority");
}

void test_only_authenticated_loopback_can_select_a_lane() {
    control enabled({ secret, 8 });

    for (const auto & value : { "low", "normal", "fast" }) {
        const classification accepted = enabled.classify("127.0.0.1", trusted_headers(value));
        require(accepted.status == classification_status::trusted, "valid loopback credential is trusted");
        require(std::string(to_string(accepted.priority)) == value, "trusted lane parses exactly");
        require(accepted.tag == "request-1", "trusted bounded tag is retained");
    }

    auto wrong = trusted_headers("fast");
    wrong[token_header] = "fedcba9876543210fedcba9876543210";
    require(enabled.classify("127.0.0.1", wrong).status == classification_status::rejected,
            "wrong credential is rejected");
    require(enabled.classify("192.0.2.4", trusted_headers("fast")).status == classification_status::rejected,
            "non-loopback credential is rejected");
    require(enabled.classify("127.0.0.1", trusted_headers("turbo")).status == classification_status::rejected,
            "unknown lane is rejected");
}

void test_header_names_are_case_insensitive_and_duplicates_fail() {
    control enabled({ secret, 8 });
    const classification accepted = enabled.classify("::1", {
        { "x-llama-trusted-scheduling-token", secret },
        { "x-llama-trusted-lane", "fast" },
        { "x-llama-benchmark-tag", "mixed.fast_1" },
    });
    require(accepted.status == classification_status::trusted, "lower-case HTTP headers are accepted");
    require(accepted.priority == lane::fast, "lower-case fast lane parses");

    auto duplicate = trusted_headers("fast");
    duplicate["x-llama-trusted-lane"] = "low";
    require(enabled.classify("127.0.0.1", duplicate).status == classification_status::rejected,
            "case-variant duplicate lane headers fail closed");
}

void test_http_route_adapter_excludes_json_and_applies_only_trusted_headers() {
    control enabled({ secret, 16 });
    const std::function<bool()> should_stop = [] { return false; };

    server_http_req forged_json {
        {}, {}, "/completion", "",
        R"({"prompt":[1,2,3],"lane":"fast","priority":"fast","scheduling":{"lane":"fast"}})",
        {}, should_stop, "127.0.0.1",
    };
    const classification ordinary = classify_http_request(enabled, forged_json);
    require(ordinary.status == classification_status::ordinary,
            "client JSON cannot request a scheduling lane");
    server_task ordinary_task(SERVER_TASK_TYPE_COMPLETION);
    require(apply_to_task(ordinary, ordinary_task), "ordinary route classification is applicable");
    require(ordinary_task.scheduling.lane == server_task::trusted_lane::normal,
            "forged JSON leaves the task in the normal lane");

    server_http_req trusted_fast {
        {}, trusted_headers("fast", "fast-smoke"), "/completion", "", "{}",
        {}, should_stop, "::1",
    };
    const classification fast = classify_http_request(enabled, trusted_fast);
    server_task fast_task(SERVER_TASK_TYPE_COMPLETION);
    require(fast.status == classification_status::trusted && apply_to_task(fast, fast_task),
            "authenticated loopback route classification is applicable");
    require(fast_task.scheduling.lane == server_task::trusted_lane::fast,
            "trusted fast header reaches actual task scheduling metadata");

    auto duplicate_headers = trusted_headers("fast");
    duplicate_headers["x-llama-trusted-lane"] = "low";
    server_http_req duplicate {
        {}, duplicate_headers, "/completion", "", "{}", {}, should_stop, "127.0.0.1", false,
    };
    server_task duplicate_task(SERVER_TASK_TYPE_COMPLETION);
    require(classify_http_request(enabled, duplicate).status == classification_status::rejected,
            "ambiguous route headers are rejected");
    require(!apply_to_task(classify_http_request(enabled, duplicate), duplicate_task),
            "rejected route classification cannot be applied");
    require(duplicate_task.scheduling.lane == server_task::trusted_lane::normal,
            "rejected route classification leaves the task normal");

    server_http_req exact_duplicate {
        {}, trusted_headers("fast"), "/completion", "", "{}", {}, should_stop, "127.0.0.1", true,
    };
    require(classify_http_request(enabled, exact_duplicate).status == classification_status::rejected,
            "same-case duplicate reserved headers detected before map collapse are rejected");

    server_http_req remote {
        {}, trusted_headers("fast"), "/completion", "", "{}", {}, should_stop, "192.0.2.1",
    };
    require(classify_http_request(enabled, remote).status == classification_status::rejected,
            "non-loopback route ingress is rejected");

    require(has_reserved_headers(trusted_fast.headers),
            "router guard detects reserved trusted scheduling headers");
    require(!has_reserved_headers(forged_json.headers),
            "ordinary requests remain eligible for router proxying");
}

void test_tags_are_bounded_and_operator_authorization_is_separate() {
    control enabled({ secret, 8 });
    require(enabled.classify("::ffff:127.0.0.1", trusted_headers("low", "low-900k")).status ==
                    classification_status::trusted,
            "IPv4-mapped loopback is accepted");
    require(enabled.classify("127.0.0.1", trusted_headers("low", "has space")).status ==
                    classification_status::rejected,
            "unsafe tag characters are rejected");
    require(enabled.classify("127.0.0.1", trusted_headers("low", std::string(65, 'a'))).status ==
                    classification_status::rejected,
            "oversized tag is rejected");

    const std::map<std::string, std::string> operator_headers = {
        { token_header, secret },
    };
    require(enabled.authorize_snapshot("[::1]", operator_headers), "operator can read trace on loopback");
    require(!enabled.authorize_snapshot("198.51.100.3", operator_headers),
            "operator credential cannot read trace remotely");
    require(!enabled.authorize_snapshot("127.0.0.1", {}), "trace rejects a missing credential");
}

void test_trace_is_lossless_until_capacity_and_never_overwrites() {
    control enabled({ secret, 2 });
    enabled.record({
        0, 10, trace_kind::request_registered, 1, 0, 0, 0, 0, 900000,
        lane::low, lane::low, false, false, false, release_reason::none, "low-900k",
    });
    enabled.record({
        0, 20, trace_kind::prefill_owner_selected, 1, 1, 0, 0, 0, 900000,
        lane::low, lane::low, false, false, false, release_reason::none, "",
    });
    const trace_snapshot before_overflow = enabled.snapshot();
    require(before_overflow.capacity == 2, "trace exposes its fixed capacity");
    require(before_overflow.total_events == 2 && before_overflow.overflow_events == 0,
            "in-capacity trace is lossless");
    require(before_overflow.events.size() == 2, "both in-capacity events are retained");
    require(before_overflow.events[0].sequence == 1 && before_overflow.events[1].sequence == 2,
            "trace sequence is contiguous");

    enabled.record({
        0, 30, trace_kind::prefill_chunk_staged, 1, 1, 7, 0, 128, 900000,
        lane::low, lane::low, false, true, false, release_reason::none, "",
    });
    const trace_snapshot overflow = enabled.snapshot();
    require(overflow.events.size() == 2, "overflow never overwrites retained events");
    require(overflow.total_events == 3 && overflow.overflow_events == 1,
            "overflow is explicit and countable");
    require(before_overflow.events.size() == 2, "prior snapshot remains immutable");
}

void test_trace_tag_joins_exact_request_and_prefill_events() {
    control enabled({ secret, 8 });
    enabled.record({
        0, 10, trace_kind::request_registered, 44, 0, 0, 0, 0, 8192,
        lane::low, lane::low, false, false, false, release_reason::none, "low-8k",
    });
    enabled.record({
        0, 20, trace_kind::prefill_owner_selected, 44, 9, 0, 0, 0, 0,
        lane::low, lane::low, false, false, false, release_reason::none, "",
    });
    enabled.record({
        0, 30, trace_kind::prefill_chunk_staged, 44, 9, 3, 0, 2048, 8192,
        lane::low, lane::fast, true, true, false, release_reason::none, "",
    });
    enabled.record({
        0, 40, trace_kind::prefill_chunk_committed, 44, 9, 3, 0, 2048, 0,
        lane::low, lane::fast, true, true, false, release_reason::none, "",
    });

    const trace_snapshot snapshot = enabled.snapshot();
    require(snapshot.overflow_events == 0 && snapshot.events.size() == 4,
            "tag join trace is retained without loss");
    require(snapshot.events.front().benchmark_tag == "low-8k" &&
            snapshot.events.front().request_id == 44,
            "registration joins the external benchmark tag to a numeric request ID");
    for (const auto & event : snapshot.events) {
        require(event.request_id == snapshot.events.front().request_id,
                "all scheduler events join back to the registered request ID");
    }
}

void test_disabled_trace_and_invalid_configuration() {
    control disabled;
    disabled.record({});
    const auto snapshot = disabled.snapshot();
    require(snapshot.events.empty() && snapshot.total_events == 0 && snapshot.overflow_events == 0,
            "disabled trace allocates and records nothing");

    bool short_secret_rejected = false;
    try {
        control invalid({ "too-short", 1 });
    } catch (const std::invalid_argument &) {
        short_secret_rejected = true;
    }
    require(short_secret_rejected, "short operator secret is rejected at startup");

    bool oversized_trace_rejected = false;
    try {
        control invalid({ secret, maximum_trace_capacity + 1 });
    } catch (const std::invalid_argument &) {
        oversized_trace_rejected = true;
    }
    require(oversized_trace_rejected, "unbounded trace capacity is rejected at startup");
}

} // namespace

int main() {
    const struct {
        const char * name;
        void (*run)();
    } tests[] = {
        { "default is normal and lane headers fail closed", test_default_is_normal_and_lane_headers_fail_closed },
        { "only authenticated loopback can select a lane", test_only_authenticated_loopback_can_select_a_lane },
        { "header names are case insensitive and duplicates fail", test_header_names_are_case_insensitive_and_duplicates_fail },
        { "HTTP route adapter excludes JSON and applies only trusted headers", test_http_route_adapter_excludes_json_and_applies_only_trusted_headers },
        { "tags are bounded and operator authorization is separate", test_tags_are_bounded_and_operator_authorization_is_separate },
        { "trace is lossless until capacity and never overwrites", test_trace_is_lossless_until_capacity_and_never_overwrites },
        { "trace tag joins exact request and prefill events", test_trace_tag_joins_exact_request_and_prefill_events },
        { "disabled trace and invalid configuration", test_disabled_trace_and_invalid_configuration },
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

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "common.h"
#include "server-chat.h"
#include "server-context.h"
#include "server-http.h"
#include "server-trusted-ingress.h"

#include <cpp-httplib/httplib.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
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

void test_model_profiles_are_an_exact_allowlist() {
    const auto & profiles = model_profiles();
    require(profiles.size() == 3, "exactly three model profiles are advertised");

    const std::array<const char *, 3> expected_models{
        "deepseek-v4-flash-slow",
        "deepseek-v4-flash",
        "deepseek-v4-flash-fast",
    };
    const std::array<const char *, 3> expected_profiles{ "slow", "normal", "fast" };
    const std::array<lane, 3>         expected_lanes{ lane::low, lane::normal, lane::fast };
    for (size_t index = 0; index < profiles.size(); ++index) {
        require(std::string(profiles[index].public_model) == expected_models[index],
                "public model profile order and spelling are stable");
        require(std::string(profiles[index].public_profile) == expected_profiles[index],
                "public profile annotation is stable");
        require(profiles[index].priority == expected_lanes[index], "public profile maps to its exact internal lane");

        const auto selected = classify_model_profile(true, std::string(expected_models[index]));
        require(selected.status == classification_status::trusted, "allowlisted model profile is trusted");
        require(selected.priority == expected_lanes[index], "allowlisted model profile selects the expected lane");
        require(selected.public_model == expected_models[index],
                "allowlisted model profile preserves its public response ID");
    }

    const auto omitted = classify_model_profile(true, std::nullopt);
    require(omitted.status == classification_status::trusted && omitted.priority == lane::normal,
            "omitted model selects normal in trusted-LAN profile mode");
    require(omitted.public_model == "deepseek-v4-flash", "omitted model receives the normal public response ID");

    const auto inactive = classify_model_profile(false, std::string("deepseek-v4-flash-fast"));
    require(inactive.status == classification_status::ordinary && inactive.public_model.empty(),
            "model profiles are inert outside trusted-LAN mode");

    for (const char * forged : {
             "deepseek-v4-flash-fastest",
             "prefix-deepseek-v4-flash-fast",
             "DEEPSEEK-V4-FLASH-FAST",
             "fast",
             "low",
             "underlying.gguf",
         }) {
        const auto rejected = classify_model_profile(true, forged);
        require(rejected.status == classification_status::rejected, "model-like substrings and aliases are rejected");
        require(rejected.priority == lane::normal && rejected.public_model.empty(),
                "rejected model IDs cannot carry scheduling or response authority");
    }
}

void test_model_listing_expands_only_in_profile_mode() {
    const json ollama_base = {
        { "name",            "underlying-model.gguf"                          },
        { "model",           "underlying-model.gguf"                          },
        { "description",     ""                                               },
        { "tags",            { "base" }                                       },
        { "details",         { { "format", "gguf" }, { "parent_model", "" } } },
        { "shared_metadata", "ollama-sentinel"                                },
    };
    const json openai_base = {
        { "id",              "underlying-model.gguf"                    },
        { "aliases",         { "legacy-alias" }                         },
        { "meta",            { { "n_ctx", 4096 }, { "ftype", "test" } } },
        { "shared_metadata", "openai-sentinel"                          },
    };

    const json ordinary = server_build_model_listing(ollama_base, openai_base, "underlying-model.gguf", false);
    require(ordinary.at("object") == "list", "ordinary listing keeps the list object type");
    require(ordinary.at("models").size() == 1 && ordinary.at("models").at(0) == ollama_base,
            "ordinary Ollama listing remains a single unchanged model");
    require(ordinary.at("data").size() == 1 && ordinary.at("data").at(0) == openai_base,
            "ordinary OpenAI listing remains a single unchanged model");

    const json profiled = server_build_model_listing(ollama_base, openai_base, "underlying-model.gguf", true);
    require(profiled.at("models").size() == 3, "trusted-LAN Ollama listing advertises exactly three profiles");
    require(profiled.at("data").size() == 3, "trusted-LAN OpenAI listing advertises exactly three profiles");

    const std::array<const char *, 3> ids{
        "deepseek-v4-flash-slow",
        "deepseek-v4-flash",
        "deepseek-v4-flash-fast",
    };
    const std::array<const char *, 3> annotations{ "slow", "normal", "fast" };
    const std::array<const char *, 3> lanes{ "low", "normal", "fast" };
    for (size_t index = 0; index < ids.size(); ++index) {
        const json & ollama = profiled.at("models").at(index);
        const json & openai = profiled.at("data").at(index);
        require(ollama.at("name") == ids[index] && ollama.at("model") == ids[index],
                "Ollama profile listing uses the stable public model ID order");
        require(openai.at("id") == ids[index], "OpenAI profile listing uses the stable public model ID order");
        for (const json * entry : { &ollama, &openai }) {
            require(entry->at("profile").at("name") == annotations[index],
                    "profile listing carries the public profile annotation");
            require(entry->at("profile").at("scheduler_lane") == lanes[index],
                    "profile listing carries the internal scheduler lane annotation");
            require(entry->at("profile").at("underlying_model") == "underlying-model.gguf",
                    "profile listing identifies the shared underlying model");
        }
        require(ollama.at("shared_metadata") == "ollama-sentinel" &&
                    openai.at("shared_metadata") == "openai-sentinel" && openai.at("meta").at("n_ctx") == 4096,
                "all variants preserve shared underlying model metadata");
        require(ollama.at("details").at("parent_model") == "underlying-model.gguf" &&
                    openai.at("meta").at("underlying_model") == "underlying-model.gguf" && openai.at("aliases").empty(),
                "variants identify but do not advertise the underlying alias");
    }
}

void test_trusted_lan_model_profile_is_the_only_task_control() {
    control                     ingress({ "", 0, true });
    const std::function<bool()> should_stop = [] {
        return false;
    };
    const std::string raw_body =
        R"({"model":"deepseek-v4-flash-fast","lane":"low","priority":"low","scheduling":{"lane":"low"}})";
    server_http_req request{
        {}, {}, "/v1/chat/completions", "", raw_body, {}, should_stop, "192.168.1.20",
    };

    const json forged_body        = json::parse(raw_body);
    const json forged_body_before = forged_body;
    const auto selected           = classify_http_model_profile(ingress, request, forged_body);
    require(selected.status == classification_status::trusted && selected.priority == lane::fast,
            "only the exact model ID selects the task lane");
    require(request.body == raw_body && forged_body == forged_body_before,
            "profile validation never rewrites raw or parsed request JSON used by capture");

    server_task task(SERVER_TASK_TYPE_COMPLETION);
    task.params.oaicompat_model = "underlying-model.gguf";
    require(apply_to_task(selected, task), "selected profile applies to a task");
    require(task.scheduling.lane == server_task::trusted_lane::fast, "selected profile reaches scheduling metadata");
    require(task.params.oaicompat_model == "deepseek-v4-flash-fast",
            "selected public model ID reaches every response serializer through task metadata");

    server_task_result_cmpl_final response{};
    response.oaicompat_model       = task.params.oaicompat_model;
    response.oaicompat_cmpl_id     = "profile-response";
    response.stop                  = STOP_TYPE_LIMIT;
    response.n_decoded             = 0;
    response.n_prompt_tokens       = 0;
    response.n_prompt_tokens_cache = 0;
    response.include_usage         = true;
    const json nonstream           = response.to_json_oaicompat_chat();
    require(nonstream.at("model") == "deepseek-v4-flash-fast",
            "non-stream response echoes the selected public model ID");
    const json stream = response.to_json_oaicompat_chat_stream();
    require(!stream.empty(), "stream serializer emits at least one event");
    for (const auto & event : stream) {
        require(event.at("model") == "deepseek-v4-flash-fast",
                "every streaming response event echoes the selected public model ID");
    }

    const auto omitted = classify_http_model_profile(ingress, request, json::object());
    require(omitted.status == classification_status::trusted && omitted.priority == lane::normal &&
                omitted.public_model == "deepseek-v4-flash",
            "omitted JSON model defaults to the normal public profile");

    const auto wrong_type = classify_http_model_profile(ingress, request,
                                                        {
                                                            { "model", 7 }
    });
    require(wrong_type.status == classification_status::rejected && wrong_type.error == "model must be a string",
            "non-string model fails clearly");

    request.headers[lane_header] = "fast";
    const auto conflicting       = classify_http_model_profile(ingress, request,
                                                               {
                                                             { "model", "deepseek-v4-flash-slow" }
    });
    require(conflicting.status == classification_status::rejected,
            "legacy lane headers cannot conflict with model profile scheduling");
    require(ingress.classify(request.remote_addr, request.headers).status == classification_status::rejected,
            "trusted-LAN control cannot be called directly to bypass model-only scheduling");
    request.headers.clear();

    control    ordinary;
    const auto inactive = classify_http_model_profile(ordinary, request,
                                                      {
                                                          { "model", "deepseek-v4-flash-fast" }
    });
    require(inactive.status == classification_status::ordinary,
            "HTTP model profiles remain inert outside trusted-LAN mode");
}

void test_profile_survives_all_compatible_request_conversions() {
    control                     ingress({ "", 0, true });
    const std::function<bool()> should_stop = [] {
        return false;
    };
    const server_http_req request{
        {}, {}, "/v1/messages", "", "", {}, should_stop, "10.0.0.2",
    };

    const json direct = {
        { "model",  "deepseek-v4-flash-slow" },
        { "prompt", "hello"                  },
    };
    const json responses = server_chat_convert_responses_to_chatcmpl({
        { "model", "deepseek-v4-flash-slow" },
        { "input", "hello"                  },
    });
    const json anthropic = server_chat_convert_anthropic_to_oai({
        { "model",      "deepseek-v4-flash-slow"                                        },
        { "messages",   json::array({ { { "role", "user" }, { "content", "hello" } } }) },
        { "max_tokens", 1                                                               },
    });

    for (const json * body : { &direct, &responses, &anthropic }) {
        const auto selected = classify_http_model_profile(ingress, request, *body);
        require(selected.status == classification_status::trusted && selected.priority == lane::low &&
                    selected.public_model == "deepseek-v4-flash-slow",
                "completion, Responses, and Anthropic conversions share one exact profile seam");
    }
}

struct running_http_server {
    server_http_context context;

    ~running_http_server() {
        context.stop();
        if (context.thread.joinable()) {
            context.thread.join();
        }
    }
};

void test_standard_routes_replace_all_lane_prefixes_and_keep_api_keys() {
    common_params params;
    params.hostname       = "127.0.0.1";
    params.port           = 0;
    params.ui             = false;
    params.n_threads_http = 1;
    params.api_keys       = { "profile-test-api-key" };

    running_http_server server;
    require(server.context.init(params), "focused HTTP server initializes");
    // This reproduces the old registration condition without depending on a
    // process-global environment variable. Old code would add every prefix.
    server.context.trust_lan = true;

    const auto handler = [](const server_http_req &) {
        auto response  = std::make_unique<server_http_res>();
        response->data = R"({"ok":true})";
        return response;
    };
    const std::array<const char *, 4> routes{
        "/completion",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/messages",
    };
    for (const char * route : routes) {
        server.context.post(route, handler);
    }
    server.context.is_ready = true;
    require(server.context.start(), "focused HTTP server starts");

    httplib::Client client("127.0.0.1", server.context.port);
    const auto      unauthorized = client.Post(routes[0], "{}", "application/json");
    require(unauthorized && unauthorized->status == 401, "standard inference routes preserve API-key enforcement");

    const httplib::Headers authorized_headers{
        { "Authorization", "Bearer profile-test-api-key" },
    };
    for (const char * route : routes) {
        const auto standard = client.Post(route, authorized_headers, "{}", "application/json");
        require(standard && standard->status == 200, "standard inference endpoint remains registered");
        for (const char * prefix : { "low", "normal", "fast" }) {
            const std::string legacy  = "/" + std::string(prefix) + route;
            const auto        removed = client.Post(legacy, authorized_headers, "{}", "application/json");
            require(removed && removed->status == 404, "legacy lane-prefixed inference endpoint is absent");
        }
    }
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
        { "default is normal and lane headers fail closed",                    test_default_is_normal_and_lane_headers_fail_closed      },
        { "model profiles are an exact allowlist",                             test_model_profiles_are_an_exact_allowlist               },
        { "model listing expands only in profile mode",                        test_model_listing_expands_only_in_profile_mode          },
        { "trusted-LAN model profile is the only task control",
         test_trusted_lan_model_profile_is_the_only_task_control                                                                        },
        { "profile survives compatible request conversions",                   test_profile_survives_all_compatible_request_conversions },
        { "standard routes replace lane prefixes and keep API keys",
         test_standard_routes_replace_all_lane_prefixes_and_keep_api_keys                                                               },
        { "only authenticated loopback can select a lane",                     test_only_authenticated_loopback_can_select_a_lane       },
        { "header names are case insensitive and duplicates fail",
         test_header_names_are_case_insensitive_and_duplicates_fail                                                                     },
        { "HTTP route adapter excludes JSON and applies only trusted headers",
         test_http_route_adapter_excludes_json_and_applies_only_trusted_headers                                                         },
        { "tags are bounded and operator authorization is separate",
         test_tags_are_bounded_and_operator_authorization_is_separate                                                                   },
        { "trace is lossless until capacity and never overwrites",
         test_trace_is_lossless_until_capacity_and_never_overwrites                                                                     },
        { "trace tag joins exact request and prefill events",                  test_trace_tag_joins_exact_request_and_prefill_events    },
        { "disabled trace and invalid configuration",                          test_disabled_trace_and_invalid_configuration            },
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

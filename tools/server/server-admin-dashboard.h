#pragma once

#include "server-queue.h"
#include "server-trusted-scheduling.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace server_admin_dashboard {

using json = nlohmann::ordered_json;

constexpr size_t schema_version             = 2;
constexpr size_t maximum_snapshot_bytes     = 4 * 1024 * 1024;
constexpr size_t maximum_sse_event_bytes    = 64 * 1024;
constexpr size_t maximum_sse_chunk_bytes    = 256 * 1024;
constexpr size_t maximum_events_per_chunk   = 32;
constexpr size_t maximum_timeline_events    = 128;
constexpr size_t maximum_live_streams       = 8;
constexpr size_t maximum_post_body_bytes    = 4 * 1024;
constexpr size_t maximum_detail_bytes       = 64 * 1024;
constexpr int    poll_interval_milliseconds = 100;
constexpr int    heartbeat_milliseconds     = 5000;

constexpr const char * snapshot_path = "/internal/admin/dashboard/snapshot";
constexpr const char * events_path   = "/internal/admin/dashboard/events";
constexpr const char * detail_path   = "/internal/admin/dashboard/request-detail";
constexpr const char * control_path  = "/internal/admin/dashboard/request-control";
constexpr const char * csrf_header   = "X-Llama-Dashboard-CSRF";
constexpr const char * csrf_value    = "1";

enum class authorization_status : uint8_t {
    allowed = 0,
    disabled,
    non_loopback,
    missing_or_invalid_credential,
    api_authentication_disabled,
    ambiguous_header,
    classification_header,
    cross_site_browser,
    query_not_allowed,
    missing_origin,
    invalid_content_type,
    invalid_csrf,
    body_too_large,
};

struct authorization_result {
    authorization_status status = authorization_status::disabled;
    const char * message         = "dashboard admin routes are disabled";

    explicit operator bool() const { return status == authorization_status::allowed; }
};

authorization_result authorize(
        const server_trusted_scheduling::control & operator_control,
        bool                                       api_authentication_enabled,
        const std::string &                        remote_address,
        const std::map<std::string, std::string> & headers,
        bool                                       ambiguous_operator_header,
        bool                                       ambiguous_last_event_id,
        const std::string &                        query_string);

authorization_result authorize_post(
        const server_trusted_scheduling::control & operator_control,
        bool                                       api_authentication_enabled,
        const std::string &                        remote_address,
        const std::map<std::string, std::string> & headers,
        bool                                       ambiguous_operator_header,
        bool                                       ambiguous_last_event_id,
        const std::string &                        query_string,
        size_t                                     body_size);

enum class request_command_status : uint8_t {
    ok = 0,
    malformed_json,
    invalid_shape,
    invalid_request_handle,
    unsupported_action,
};

struct request_command {
    request_command_status                   status = request_command_status::malformed_json;
    server_request_registry::request_handle request;
    const char *                             message = "malformed dashboard request body";

    explicit operator bool() const { return status == request_command_status::ok; }
};

request_command parse_detail_command(const std::string & body);
request_command parse_control_command(const std::string & body);

enum class request_lookup_status : uint8_t {
    found = 0,
    unknown,
    stale,
};

struct request_detail {
    request_lookup_status status = request_lookup_status::unknown;
    json                  body;

    explicit operator bool() const { return status == request_lookup_status::found; }
};

request_detail make_request_detail(
        const server_queue_request_state &      source,
        server_request_registry::request_handle handle,
        uint64_t                                now_us);

bool parse_last_event_id(const std::map<std::string, std::string> & headers, uint64_t & result);

json make_snapshot(const server_queue_request_state & source, uint64_t now_us);

enum class resume_status : uint8_t {
    waiting = 0,
    events,
    gap,
    future_cursor,
};

struct resume_batch {
    resume_status           status = resume_status::waiting;
    uint64_t                latest_sequence = 0;
    uint64_t                oldest_available_sequence = 0;
    std::vector<std::string> frames;
};

resume_batch make_resume_batch(
        const server_queue_request_state & source,
        uint64_t                           after_sequence,
        uint64_t                           now_us);

std::string make_overflow_frame(uint64_t event_sequence, uint64_t oldest_available_sequence, uint64_t now_us);

std::shared_ptr<void> try_acquire_stream_lease();
size_t active_streams();

} // namespace server_admin_dashboard

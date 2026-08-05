#include "server-trusted-scheduling.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace server_trusted_scheduling {

namespace {

std::string ascii_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

struct header_lookup {
    bool        present   = false;
    bool        ambiguous = false;
    std::string value;
};

header_lookup find_header(
        const std::map<std::string, std::string> & headers,
        const char * name) {
    const std::string expected = ascii_lower(name);
    header_lookup result;
    for (const auto & entry : headers) {
        if (ascii_lower(entry.first) != expected) {
            continue;
        }
        if (result.present) {
            result.ambiguous = true;
            continue;
        }
        result.present = true;
        result.value   = entry.second;
    }
    return result;
}

bool constant_time_equal(const std::string & supplied, const std::string & expected) {
    if (supplied.size() != expected.size()) {
        return false;
    }
    unsigned char difference = 0;
    for (size_t index = 0; index < expected.size(); ++index) {
        difference |= static_cast<unsigned char>(supplied[index]) ^
                      static_cast<unsigned char>(expected[index]);
    }
    return difference == 0;
}

bool valid_tag(const std::string & tag) {
    if (tag.empty() || tag.size() > maximum_tag_bytes) {
        return false;
    }
    return std::all_of(tag.begin(), tag.end(), [](unsigned char character) {
        return std::isalnum(character) || character == '-' || character == '_' || character == '.';
    });
}

bool parse_lane(const std::string & text, lane & result) {
    if (text == "low") {
        result = lane::low;
        return true;
    }
    if (text == "normal") {
        result = lane::normal;
        return true;
    }
    if (text == "fast") {
        result = lane::fast;
        return true;
    }
    return false;
}

size_t parse_trace_capacity(const char * text) {
    if (text == nullptr || text[0] == '\0') {
        return default_trace_capacity;
    }
    std::string value(text);
    size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument("invalid trusted scheduling trace capacity");
    }
    if (consumed != value.size() || parsed > maximum_trace_capacity) {
        throw std::invalid_argument("trusted scheduling trace capacity is out of range");
    }
    return static_cast<size_t>(parsed);
}

bool authorized(
        const config & cfg,
        const std::string & remote_address,
        const std::map<std::string, std::string> & headers) {
    if (cfg.token.empty() || !is_loopback_address(remote_address)) {
        return false;
    }
    const auto supplied = find_header(headers, token_header);
    return supplied.present && !supplied.ambiguous && constant_time_equal(supplied.value, cfg.token);
}

} // namespace

const char * to_string(lane value) {
    switch (value) {
        case lane::low:    return "low";
        case lane::normal: return "normal";
        case lane::fast:   return "fast";
    }
    return "invalid";
}

const char * to_string(trace_kind value) {
    switch (value) {
        case trace_kind::request_registered:       return "request_registered";
        case trace_kind::prefill_owner_selected:   return "prefill_owner_selected";
        case trace_kind::prefill_chunk_staged:     return "prefill_chunk_staged";
        case trace_kind::prefill_chunk_committed:  return "prefill_chunk_committed";
        case trace_kind::prefill_chunk_aborted:    return "prefill_chunk_aborted";
        case trace_kind::prefill_owner_released:   return "prefill_owner_released";
    }
    return "invalid";
}

const char * to_string(release_reason value) {
    switch (value) {
        case release_reason::none:        return "none";
        case release_reason::yielded:     return "yielded";
        case release_reason::completed:   return "completed";
        case release_reason::cancelled:   return "cancelled";
        case release_reason::disappeared: return "disappeared";
    }
    return "invalid";
}

config config_from_environment() {
    config result;
    const char * trust_lan = std::getenv(trust_lan_environment);
    result.trust_lan = trust_lan != nullptr && trust_lan[0] != '\0' && std::string(trust_lan) != "0";
    const char * token = std::getenv(token_environment);
    result.token = token == nullptr ? "" : token;
    if (result.token.empty()) {
        result.trace_capacity = 0;
        return result;
    }
    result.trace_capacity = parse_trace_capacity(std::getenv(trace_capacity_environment));
    return result;
}

bool is_loopback_address(const std::string & address) {
    return address == "::1" || address == "[::1]" ||
           address.rfind("127.", 0) == 0 || address.rfind("::ffff:127.", 0) == 0;
}

bool has_reserved_headers(const std::map<std::string, std::string> & headers) {
    return find_header(headers, token_header).present ||
           find_header(headers, lane_header).present ||
           find_header(headers, tag_header).present;
}

control::control(config config) : cfg(std::move(config)) {
    if (!cfg.token.empty() && (cfg.token.size() < 32 || cfg.token.size() > 256)) {
        throw std::invalid_argument("trusted scheduling token must contain 32 to 256 bytes");
    }
    if (cfg.trace_capacity > maximum_trace_capacity) {
        throw std::invalid_argument("trusted scheduling trace capacity is out of range");
    }
    if (trace_enabled()) {
        events.reserve(cfg.trace_capacity);
    }
}

classification control::classify(
        const std::string & remote_address,
        const std::map<std::string, std::string> & headers) const {
    const auto requested_lane = find_header(headers, lane_header);
    if (!requested_lane.present) {
        return {};
    }
    if (requested_lane.ambiguous) {
        return { classification_status::rejected, lane::normal, {}, "ambiguous trusted scheduling lane header" };
    }
    // Trusted-LAN bypass: a single-operator LAN/Tailscale deployment that sets
    // LLAMA_SERVER_TRUST_LAN does not require loopback ingress or the operator
    // secret.  Lane selection still flows through the request lane header
    // (injected by the path-prefix routes in server-http when trust_lan is on).
    if (cfg.trust_lan) {
        lane priority;
        if (!parse_lane(requested_lane.value, priority)) {
            return { classification_status::rejected, lane::normal, {}, "invalid trusted scheduling lane" };
        }
        const auto tag = find_header(headers, tag_header);
        if (tag.ambiguous || (tag.present && !valid_tag(tag.value))) {
            return { classification_status::rejected, lane::normal, {}, "invalid trusted scheduling benchmark tag" };
        }
        return { classification_status::trusted, priority, tag.present ? tag.value : std::string(), {} };
    }
    if (!enabled()) {
        return { classification_status::rejected, lane::normal, {}, "trusted scheduling lanes are disabled" };
    }
    if (!is_loopback_address(remote_address)) {
        return { classification_status::rejected, lane::normal, {}, "trusted scheduling requires loopback ingress" };
    }
    if (!authorized(cfg, remote_address, headers)) {
        return { classification_status::rejected, lane::normal, {}, "invalid trusted scheduling credential" };
    }

    lane priority;
    if (!parse_lane(requested_lane.value, priority)) {
        return { classification_status::rejected, lane::normal, {}, "invalid trusted scheduling lane" };
    }

    const auto tag = find_header(headers, tag_header);
    if (tag.ambiguous || (tag.present && !valid_tag(tag.value))) {
        return { classification_status::rejected, lane::normal, {}, "invalid trusted scheduling benchmark tag" };
    }
    return { classification_status::trusted, priority, tag.present ? tag.value : std::string(), {} };
}

bool control::authorize_snapshot(
        const std::string & remote_address,
        const std::map<std::string, std::string> & headers) const {
    return trace_enabled() && authorize_operator(remote_address, headers);
}

bool control::authorize_operator(
        const std::string & remote_address,
        const std::map<std::string, std::string> & headers) const {
    return authorized(cfg, remote_address, headers);
}

void control::record(trace_event event) {
    if (!trace_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    ++total_events;
    if (events.size() == cfg.trace_capacity) {
        ++overflow_events;
        return;
    }
    event.sequence = total_events;
    events.push_back(std::move(event));
}

trace_snapshot control::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex);
    return { events, cfg.trace_capacity, total_events, overflow_events };
}

} // namespace server_trusted_scheduling

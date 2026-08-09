#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace server_trusted_scheduling {

constexpr size_t default_trace_capacity = 65536;
constexpr size_t maximum_trace_capacity = 262144;
constexpr size_t maximum_tag_bytes      = 64;

constexpr const char * token_environment = "LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN";
constexpr const char * trace_capacity_environment = "LLAMA_SERVER_BENCH_TRACE_CAPACITY";
constexpr const char * trust_lan_environment       = "LLAMA_SERVER_TRUST_LAN";
constexpr const char * token_header = "X-Llama-Trusted-Scheduling-Token";
constexpr const char * lane_header  = "X-Llama-Trusted-Lane";
constexpr const char * tag_header   = "X-Llama-Benchmark-Tag";

enum class lane : uint8_t {
    low = 0,
    normal,
    fast,
};

const char * to_string(lane value);

// Public model IDs are the only trusted-LAN scheduling selector. Keep the
// spelling independent from the internal scheduler lane: "slow" is public,
// while the scheduler continues to call that lane "low".
struct model_profile {
    const char * public_model;
    const char * public_profile;
    lane         priority;
};

constexpr size_t model_profile_count = 3;

const std::array<model_profile, model_profile_count> & model_profiles();

struct config {
    // Empty disables trusted classification and tracing.  A configured secret
    // must contain at least 32 and at most 256 bytes.
    std::string token;
    size_t      trace_capacity = default_trace_capacity;
    // Opt-in: trust the local network.  When set, lane selection and the
    // dashboard accept non-loopback clients without the operator secret.
    // Intended for a single-operator LAN/Tailscale deployment behind no
    // gateway; the default (unset) keeps the loopback + token boundary.
    bool        trust_lan = false;
};

config config_from_environment();

enum class classification_status : uint8_t {
    ordinary = 0,
    trusted,
    rejected,
};

struct classification {
    classification_status status   = classification_status::ordinary;
    lane                  priority = lane::normal;
    std::string           tag;
    std::string           error;
    // Non-empty only for an exact trusted-LAN public model profile. Response
    // serializers echo this value rather than the underlying model artifact.
    std::string           public_model;

    classification() = default;

    classification(classification_status status,
                   lane                  priority,
                   std::string           tag,
                   std::string           error,
                   std::string           public_model = {}) :
        status(status),
        priority(priority),
        tag(std::move(tag)),
        error(std::move(error)),
        public_model(std::move(public_model)) {}

    explicit operator bool() const { return status != classification_status::rejected; }
};

// Pure exact classifier. When enabled, omission selects the normal public
// profile. When disabled, it is inert and preserves ordinary server behavior.
classification classify_model_profile(bool enabled, const std::optional<std::string> & requested_model);

bool is_loopback_address(const std::string & address);
bool has_reserved_headers(const std::map<std::string, std::string> & headers);
bool has_lane_header(const std::map<std::string, std::string> & headers);

enum class trace_kind : uint8_t {
    request_registered = 0,
    prefill_owner_selected,
    prefill_chunk_staged,
    prefill_chunk_committed,
    prefill_chunk_aborted,
    prefill_owner_released,
};

const char * to_string(trace_kind value);

enum class release_reason : uint8_t {
    none = 0,
    yielded,
    completed,
    cancelled,
    disappeared,
};

const char * to_string(release_reason value);

struct trace_event {
    uint64_t sequence      = 0;
    uint64_t at_us         = 0;
    trace_kind kind        = trace_kind::request_registered;
    uint64_t request_id    = 0;
    uint64_t cohort_id     = 0;
    uint64_t generation    = 0;
    uint64_t begin_token   = 0;
    uint64_t end_token     = 0;
    uint64_t prompt_tokens = 0;
    lane     priority      = lane::normal;
    lane     active_decode_lane = lane::low;
    bool     active_decode    = false;
    bool     yield_boundary   = false;
    bool     completes_prompt = false;
    release_reason reason     = release_reason::none;
    std::string benchmark_tag;
};

struct trace_snapshot {
    std::vector<trace_event> events;
    size_t   capacity        = 0;
    uint64_t total_events    = 0;
    uint64_t overflow_events = 0;
};

// Disabled by default.  When enabled, only a loopback request presenting the
// operator secret may assign a lane.  Client JSON is outside this API.
//
// Trace storage is append-only and bounded.  It never overwrites an event: a
// full buffer increments overflow_events so a benchmark can fail closed.
class control {
  public:
    explicit control(config cfg = {});

    bool enabled() const { return !cfg.token.empty(); }
    bool trace_enabled() const { return enabled() && cfg.trace_capacity != 0; }
    bool trust_lan() const { return cfg.trust_lan; }

    classification classify(
            const std::string & remote_address,
            const std::map<std::string, std::string> & headers) const;

    bool authorize_snapshot(
            const std::string & remote_address,
            const std::map<std::string, std::string> & headers) const;

    // Operator authorization is independent of benchmark trace collection.
    bool authorize_operator(
            const std::string & remote_address,
            const std::map<std::string, std::string> & headers) const;

    void record(trace_event event);
    trace_snapshot snapshot() const;

  private:
    config cfg;

    mutable std::mutex mutex;
    std::vector<trace_event> events;
    uint64_t total_events    = 0;
    uint64_t overflow_events = 0;
};

} // namespace server_trusted_scheduling

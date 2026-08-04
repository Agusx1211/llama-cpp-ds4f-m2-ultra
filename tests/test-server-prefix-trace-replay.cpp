#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-prefix-cache.h"
#include "server-scheduler.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace server_prefix_cache;
using namespace server_scheduler;

namespace {

constexpr uint64_t MIB                  = 1024ULL * 1024;
constexpr uint64_t CACHE_CAPACITY_BYTES = 8 * MIB;

enum class prefix_kind : uint8_t {
    system,
    message_boundary,
    complete_turn,
};

enum class cache_kind : uint8_t {
    none,
    policy,
    fifo,
    lru,
};

struct fixture_job {
    request                  req;
    prefix_kind              kind;
    uint64_t                 prefix_id;
    server_prefix_policy_key key;
    uint64_t                 unique_bytes;
    uint64_t                 prompt_us;
    uint64_t                 decode_us;
};

struct dispatch_event {
    uint64_t    request_id;
    lane        priority;
    reason_code request_reason;
    reason_code lane_reason;
    uint64_t    start_us;
    uint64_t    completion_us;
    uint64_t    latency_us;
    bool        cache_hit;
    bool        higher_lanes_queued;
};

struct replay_metrics {
    cache_kind                       kind;
    uint64_t                         saved_prefill_us    = 0;
    uint64_t                         resident_bytes      = 0;
    uint64_t                         peak_resident_bytes = 0;
    uint64_t                         end_time_us         = 0;
    std::array<uint64_t, lane_count> dispatches          = {};
    std::array<uint64_t, lane_count> max_latency_us      = {};
    std::array<uint64_t, lane_count> first_completion_us = {
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(),
        std::numeric_limits<uint64_t>::max(),
    };
    size_t                      cache_affinity_dispatches    = 0;
    size_t                      low_contended_dispatches     = 0;
    size_t                      policy_hysteresis_rejections = 0;
    std::vector<dispatch_event> events;
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

size_t lane_index(lane value) {
    return static_cast<size_t>(value);
}

lane parse_lane(const std::string & value) {
    if (value == "low") {
        return lane::low;
    }
    if (value == "normal") {
        return lane::normal;
    }
    if (value == "fast") {
        return lane::fast;
    }
    fail("unknown lane: " + value);
}

prefix_kind parse_kind(const std::string & value) {
    if (value == "system") {
        return prefix_kind::system;
    }
    if (value == "message_boundary") {
        return prefix_kind::message_boundary;
    }
    if (value == "complete_turn") {
        return prefix_kind::complete_turn;
    }
    fail("unknown prefix kind: " + value);
}

uint64_t parse_u64(const std::string & value, const char * field, size_t line_number) {
    size_t                   parsed = 0;
    const unsigned long long result = std::stoull(value, &parsed, 10);
    if (parsed != value.size()) {
        fail("invalid " + std::string(field) + " at fixture line " + std::to_string(line_number));
    }
    return static_cast<uint64_t>(result);
}

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::istringstream       input(line);
    std::string              field;
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

server_prefix_policy_key make_key(uint64_t prefix_id, uint32_t aligned_tokens) {
    server_prefix_policy_key result;
    for (size_t offset = 0; offset < result.digest.size(); offset += sizeof(prefix_id)) {
        const uint64_t mixed = prefix_id ^ (0x9e3779b97f4a7c15ULL * (offset / sizeof(prefix_id) + 1));
        for (size_t byte = 0; byte < sizeof(mixed); ++byte) {
            result.digest[offset + byte] = static_cast<uint8_t>(mixed >> (byte * 8));
        }
    }
    result.aligned_tokens = aligned_tokens;
    return result;
}

std::vector<fixture_job> load_fixture(const std::string & path) {
    std::ifstream input(path);
    require(input.good(), "cannot open fixture: " + path);

    std::vector<fixture_job> jobs;
    std::string              line;
    size_t                   line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto fields = split_csv(line);
        require(fields.size() == 9, "fixture line must have nine fields: " + std::to_string(line_number));

        fixture_job job;
        job.req.id                    = parse_u64(fields[0], "request_id", line_number);
        job.req.arrival_us            = parse_u64(fields[1], "arrival_us", line_number);
        job.req.priority              = parse_lane(fields[2]);
        job.kind                      = parse_kind(fields[3]);
        job.prefix_id                 = parse_u64(fields[4], "prefix_id", line_number);
        const uint64_t aligned_tokens = parse_u64(fields[5], "aligned_tokens", line_number);
        job.unique_bytes              = parse_u64(fields[6], "unique_bytes", line_number);
        job.prompt_us                 = parse_u64(fields[7], "prompt_us", line_number);
        job.decode_us                 = parse_u64(fields[8], "decode_us", line_number);
        require(aligned_tokens <= std::numeric_limits<uint32_t>::max(), "aligned token count exceeds uint32_t");
        job.key                         = make_key(job.prefix_id, static_cast<uint32_t>(aligned_tokens));
        job.req.prompt_tokens           = aligned_tokens;
        // The frozen v1 scenario cycles a small trusted runtime offset so the
        // real scheduler must exercise bounded within-lane cache affinity.
        job.req.virtual_runtime_us      = (job.req.id % 8) * 250;
        job.req.requested_output_tokens = 64;
        job.req.decode_runway_tokens    = 64;
        jobs.push_back(job);
    }
    require(input.eof(), "error while reading fixture: " + path);
    return jobs;
}

void validate_fixture(const std::vector<fixture_job> & jobs) {
    require(jobs.size() == 66, "frozen v1 fixture must contain 66 requests");

    std::array<size_t, lane_count> lane_jobs             = {};
    size_t                         system_jobs           = 0;
    size_t                         message_jobs          = 0;
    size_t                         pressure_message_jobs = 0;
    size_t                         complete_turn_jobs    = 0;
    for (size_t i = 0; i < jobs.size(); ++i) {
        const fixture_job & job = jobs[i];
        require(job.req.id == i + 1, "fixture request IDs must be contiguous and ordered");
        require(job.req.arrival_us == i * 2000, "fixture arrivals must remain at exact 2 ms intervals");
        require(job.key.aligned_tokens != 0 && job.key.aligned_tokens % SERVER_PREFIX_BLOCK_TOKENS == 0,
                "fixture anchors must be non-empty and 128-token aligned");
        require(job.unique_bytes > 0 && job.unique_bytes <= CACHE_CAPACITY_BYTES,
                "fixture resident object must fit the equal-byte cache");
        require(job.prompt_us > 0 && job.decode_us > 0, "fixture service values must be positive");
        ++lane_jobs[lane_index(job.req.priority)];

        if (job.kind == prefix_kind::system) {
            ++system_jobs;
            require(job.prefix_id == 100, "system rows must share the frozen system prefix");
        } else if (job.kind == prefix_kind::message_boundary) {
            ++message_jobs;
            require(job.prefix_id == 200 || job.prefix_id == 300, "unknown frozen message-boundary prefix");
            pressure_message_jobs += job.prefix_id == 300;
        } else {
            ++complete_turn_jobs;
            require(job.prefix_id == 1000 + complete_turn_jobs, "complete turns must be unique and ordered");
        }

        for (size_t prior = 0; prior < i; ++prior) {
            require(job.req.id != jobs[prior].req.id, "duplicate fixture request ID");
            if (job.prefix_id == jobs[prior].prefix_id) {
                require(job.key == jobs[prior].key && job.unique_bytes == jobs[prior].unique_bytes &&
                            job.prompt_us == jobs[prior].prompt_us,
                        "repeated prefix metadata must be identical");
                require(job.kind != prefix_kind::complete_turn, "complete-turn prefix must appear only once");
            }
        }
    }

    require(system_jobs == 14 && message_jobs == 16 && pressure_message_jobs == 2 && complete_turn_jobs == 36,
            "frozen semantic mix changed");
    for (size_t count : lane_jobs) {
        require(count >= 16, "every scheduler lane needs sustained work");
    }
}

server_prefix_policy_config trace_policy_config() {
    server_prefix_policy_config config;
    config.max_resident_entries = 8;
    config.max_resident_bytes   = CACHE_CAPACITY_BYTES;
    config.max_protected_bytes  = CACHE_CAPACITY_BYTES;
    config.max_pinned_entries   = 0;
    config.max_pinned_bytes     = 0;
    config.ghost_buckets        = 128;
    config.ghost_ways           = 4;
    config.sketch_width         = 1024;
    config.decay_interval       = 1ULL << 20;
    config.hysteresis_fraction  = 0.10;
    return config;
}

struct baseline_entry {
    server_prefix_policy_key key;
    uint64_t                 unique_bytes;
};

class replay_cache {
  public:
    explicit replay_cache(cache_kind kind) : kind(kind) {
        if (kind == cache_kind::policy) {
            policy = std::make_unique<server_prefix_policy>(trace_policy_config());
        }
    }

    bool contains(const fixture_job & job) const {
        if (kind == cache_kind::none) {
            return false;
        }
        if (kind == cache_kind::policy) {
            return policy->resident(job.key).has_value();
        }
        return find(job.key) != entries.end();
    }

    void touch(const fixture_job & job, bool hit) {
        if (!hit || kind != cache_kind::lru) {
            return;
        }
        const auto found = find(job.key);
        require(found != entries.end(), "LRU hit vanished before touch");
        const baseline_entry entry = *found;
        entries.erase(found);
        entries.push_back(entry);
    }

    void complete(const fixture_job & job, bool hit) {
        if (kind == cache_kind::none) {
            return;
        }
        if (kind == cache_kind::policy) {
            const server_prefix_policy_candidate candidate = {
                job.key, job.unique_bytes, static_cast<double>(job.prompt_us) / 1000.0, 0.0, false,
            };
            const auto result = policy->observe(candidate);
            require(result.status != server_prefix_policy_status::rejected_invalid &&
                        result.status != server_prefix_policy_status::rejected_capacity &&
                        result.status != server_prefix_policy_status::rejected_pin_quota,
                    "policy rejected a valid trace observation");
            policy_hysteresis_rejections_value += result.status == server_prefix_policy_status::rejected_hysteresis;
        } else if (!hit) {
            while (!entries.empty() && resident_bytes_value + job.unique_bytes > CACHE_CAPACITY_BYTES) {
                resident_bytes_value -= entries.front().unique_bytes;
                entries.erase(entries.begin());
            }
            entries.push_back({ job.key, job.unique_bytes });
            resident_bytes_value += job.unique_bytes;
        }
        peak_resident_bytes_value = std::max(peak_resident_bytes_value, resident_bytes());
    }

    uint64_t resident_bytes() const {
        return kind == cache_kind::policy ? policy->stats().resident_bytes : resident_bytes_value;
    }

    uint64_t peak_resident_bytes() const { return peak_resident_bytes_value; }

    size_t policy_hysteresis_rejections() const { return policy_hysteresis_rejections_value; }

  private:
    using iterator       = std::vector<baseline_entry>::iterator;
    using const_iterator = std::vector<baseline_entry>::const_iterator;

    iterator find(const server_prefix_policy_key & key) {
        return std::find_if(entries.begin(), entries.end(),
                            [&](const baseline_entry & entry) { return entry.key == key; });
    }

    const_iterator find(const server_prefix_policy_key & key) const {
        return std::find_if(entries.begin(), entries.end(),
                            [&](const baseline_entry & entry) { return entry.key == key; });
    }

    cache_kind                            kind;
    std::unique_ptr<server_prefix_policy> policy;
    std::vector<baseline_entry>           entries;
    uint64_t                              resident_bytes_value               = 0;
    uint64_t                              peak_resident_bytes_value          = 0;
    size_t                                policy_hysteresis_rejections_value = 0;
};

scheduler_config trace_scheduler_config() {
    scheduler_config config;
    for (lane_descriptor & descriptor : config.lanes) {
        descriptor.queue_cap = 128;
    }
    config.context_tokens      = 1048576;
    config.fairness_quantum_us = 5000;
    config.cache_lookahead     = 8;
    config.max_bypasses        = 3;
    config.aging_interval_us   = 10000;
    config.aging_credit_us     = 1;
    config.max_aging_credit_us = 1000000;
    return config;
}

const fixture_job & find_job(const std::vector<fixture_job> & jobs, uint64_t request_id) {
    require(request_id > 0 && request_id <= jobs.size(), "scheduler selected unknown fixture request");
    return jobs[static_cast<size_t>(request_id - 1)];
}

void admit_arrivals(scheduler &                      scheduling_policy,
                    const std::vector<fixture_job> & jobs,
                    size_t &                         next_arrival,
                    uint64_t                         now_us) {
    while (next_arrival < jobs.size() && jobs[next_arrival].req.arrival_us <= now_us) {
        feasibility_quote quote;
        quote.state          = feasibility::feasible_now;
        const auto admission = scheduling_policy.admit(jobs[next_arrival].req, quote);
        require(admission.accepted && admission.ready && admission.reason == reason_code::admission_ready,
                "fixture request admission failed");
        ++next_arrival;
    }
}

replay_metrics replay(const std::vector<fixture_job> & jobs, cache_kind kind) {
    scheduler      scheduling_policy(trace_scheduler_config());
    replay_cache   cache(kind);
    replay_metrics metrics;
    metrics.kind = kind;

    size_t   next_arrival = 0;
    uint64_t now_us       = 0;
    while (next_arrival < jobs.size() || scheduling_policy.queued_total() != 0) {
        if (scheduling_policy.queued_total() == 0) {
            require(next_arrival < jobs.size(), "replay lost work");
            now_us = std::max(now_us, jobs[next_arrival].req.arrival_us);
        }
        admit_arrivals(scheduling_policy, jobs, next_arrival, now_us);

        const evaluation_provider evaluate = [&](const request & req) {
            const fixture_job &  job = find_job(jobs, req.id);
            const bool           hit = cache.contains(job);
            candidate_evaluation result;
            result.state            = feasibility::feasible_now;
            result.predicted_gpu_us = job.decode_us + (hit ? 0 : job.prompt_us);
            result.cached_prefix_us = hit ? job.prompt_us : 0;
            result.restore_cost_us  = 0;
            return result;
        };

        const bool higher_lanes_queued =
            scheduling_policy.queued(lane::normal) != 0 || scheduling_policy.queued(lane::fast) != 0;
        const auto decision = scheduling_policy.select_next(now_us, evaluate);
        require(decision.selected, "real scheduler stalled on an all-feasible trace");
        const fixture_job & job        = find_job(jobs, decision.request_id);
        const bool          hit        = cache.contains(job);
        const uint64_t      service_us = job.decode_us + (hit ? 0 : job.prompt_us);
        require(decision.predicted_gpu_us == service_us, "scheduler evaluation and replay service disagree");
        cache.touch(job, hit);

        const uint64_t completion_us = now_us + service_us;
        admit_arrivals(scheduling_policy, jobs, next_arrival, completion_us);
        const auto completion =
            scheduling_policy.complete_service(decision.decision_id, service_us, service_disposition::complete);
        require(completion.completed && completion.reason == reason_code::service_complete,
                "real scheduler rejected replay completion");
        cache.complete(job, hit);

        const size_t selected_lane = lane_index(decision.selected_lane);
        ++metrics.dispatches[selected_lane];
        metrics.max_latency_us[selected_lane] =
            std::max(metrics.max_latency_us[selected_lane], completion_us - job.req.arrival_us);
        metrics.first_completion_us[selected_lane] =
            std::min(metrics.first_completion_us[selected_lane], completion_us);
        metrics.cache_affinity_dispatches += decision.reason == reason_code::request_cache_affinity;
        if (decision.selected_lane == lane::low && higher_lanes_queued) {
            ++metrics.low_contended_dispatches;
        }
        if (hit) {
            metrics.saved_prefill_us += job.prompt_us;
        }
        metrics.events.push_back({
            decision.request_id,
            decision.selected_lane,
            decision.reason,
            decision.lane_reason,
            now_us,
            completion_us,
            completion_us - job.req.arrival_us,
            hit,
            higher_lanes_queued,
        });
        now_us = completion_us;
    }

    require(metrics.events.size() == jobs.size(), "replay did not complete every fixture request");
    metrics.resident_bytes               = cache.resident_bytes();
    metrics.peak_resident_bytes          = cache.peak_resident_bytes();
    metrics.policy_hysteresis_rejections = cache.policy_hysteresis_rejections();
    metrics.end_time_us                  = now_us;
    return metrics;
}

bool same_events(const replay_metrics & lhs, const replay_metrics & rhs) {
    if (lhs.events.size() != rhs.events.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.events.size(); ++i) {
        const dispatch_event & left  = lhs.events[i];
        const dispatch_event & right = rhs.events[i];
        if (left.request_id != right.request_id || left.priority != right.priority ||
            left.request_reason != right.request_reason || left.lane_reason != right.lane_reason ||
            left.start_us != right.start_us || left.completion_us != right.completion_us ||
            left.latency_us != right.latency_us || left.cache_hit != right.cache_hit ||
            left.higher_lanes_queued != right.higher_lanes_queued) {
            return false;
        }
    }
    return true;
}

long double efficiency(const replay_metrics & metrics) {
    require(metrics.peak_resident_bytes > 0, "resident efficiency needs a nonzero byte denominator");
    constexpr long double bytes_per_gib = 1024.0L * 1024.0L * 1024.0L;
    const long double     saved_ms      = static_cast<long double>(metrics.saved_prefill_us) / 1000.0L;
    return saved_ms * bytes_per_gib / static_cast<long double>(metrics.peak_resident_bytes);
}

const char * cache_name(cache_kind kind) {
    switch (kind) {
        case cache_kind::none:
            return "none";
        case cache_kind::policy:
            return "policy";
        case cache_kind::fifo:
            return "fifo";
        case cache_kind::lru:
            return "lru";
    }
    return "invalid";
}

void print_metrics(const replay_metrics & metrics) {
    std::cout << cache_name(metrics.kind) << " saved_prefill_us=" << metrics.saved_prefill_us
              << " resident_bytes=" << metrics.resident_bytes << " peak_resident_bytes=" << metrics.peak_resident_bytes;
    if (metrics.kind != cache_kind::none) {
        std::cout << " saved_prefill_ms_per_resident_GiB=" << std::fixed << std::setprecision(3) << efficiency(metrics)
                  << std::defaultfloat;
    }
    std::cout << " end_us=" << metrics.end_time_us << " low_dispatches=" << metrics.dispatches[lane_index(lane::low)]
              << " low_contended=" << metrics.low_contended_dispatches
              << " max_latency_us=" << metrics.max_latency_us[lane_index(lane::low)] << '/'
              << metrics.max_latency_us[lane_index(lane::normal)] << '/'
              << metrics.max_latency_us[lane_index(lane::fast)]
              << " cache_affinity=" << metrics.cache_affinity_dispatches << '\n';
}

void validate_comparison(const replay_metrics & policy,
                         const replay_metrics & fifo,
                         const replay_metrics & lru,
                         const replay_metrics & oracle) {
    require(policy.peak_resident_bytes <= CACHE_CAPACITY_BYTES && fifo.peak_resident_bytes <= CACHE_CAPACITY_BYTES &&
                lru.peak_resident_bytes <= CACHE_CAPACITY_BYTES,
            "a replay cache exceeded the equal-byte capacity");
    require(policy.resident_bytes == 5 * MIB && policy.peak_resident_bytes == 5 * MIB,
            "policy should retain exactly the repeated system and message-boundary prefixes");
    require(policy.policy_hysteresis_rejections > 0,
            "the repeated low-value boundary must exercise policy admission pressure");
    require(fifo.peak_resident_bytes == CACHE_CAPACITY_BYTES && lru.peak_resident_bytes == CACHE_CAPACITY_BYTES,
            "both baselines must exercise the full equal-byte capacity");
    require(efficiency(policy) > efficiency(fifo), "policy efficiency must beat equal-byte FIFO");
    require(efficiency(policy) > efficiency(lru), "policy efficiency must beat equal-byte LRU");
    require(policy.saved_prefill_us > fifo.saved_prefill_us && policy.saved_prefill_us > lru.saved_prefill_us,
            "policy must save more absolute prefill time than both baselines");

    for (size_t i = 0; i < lane_count; ++i) {
        require(policy.dispatches[i] == oracle.dispatches[i], "policy changed scheduler completion counts");
        require(policy.max_latency_us[i] <= oracle.max_latency_us[i],
                "cache-aware replay regressed a lane's maximum latency");
    }
    require(policy.low_contended_dispatches > 0 && oracle.low_contended_dispatches > 0,
            "low-lane work must progress while normal/fast work remains queued");
    require(policy.first_completion_us[lane_index(lane::low)] <= oracle.first_completion_us[lane_index(lane::low)],
            "cache-aware replay delayed first low-lane progress");
    require(policy.cache_affinity_dispatches > 0,
            "trace must exercise the real scheduler's bounded cache-affinity ordering");
}

void validate_frozen_metrics(const replay_metrics & policy,
                             const replay_metrics & fifo,
                             const replay_metrics & lru,
                             const replay_metrics & oracle) {
    require(oracle.end_time_us == 5254000 &&
                oracle.max_latency_us == std::array<uint64_t, lane_count>({ 5162000, 4308000, 2638000 }) &&
                oracle.low_contended_dispatches == 6,
            "no-cache oracle metrics changed");
    require(policy.saved_prefill_us == 3600000 && policy.end_time_us == 1654000 &&
                policy.max_latency_us == std::array<uint64_t, lane_count>({ 1594000, 1345000, 1095000 }) &&
                policy.cache_affinity_dispatches == 21 && policy.low_contended_dispatches == 10,
            "policy replay metrics changed");
    require(fifo.saved_prefill_us == 3240000 && fifo.resident_bytes == CACHE_CAPACITY_BYTES &&
                fifo.end_time_us == 2014000 &&
                fifo.max_latency_us == std::array<uint64_t, lane_count>({ 1922000, 1572000, 868000 }) &&
                fifo.cache_affinity_dispatches == 17,
            "FIFO replay metrics changed");
    require(lru.saved_prefill_us == 3360000 && lru.resident_bytes == 7 * MIB && lru.end_time_us == 1894000 &&
                lru.max_latency_us == std::array<uint64_t, lane_count>({ 1834000, 1572000, 868000 }) &&
                lru.cache_affinity_dispatches == 18,
            "LRU replay metrics changed");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: test-server-prefix-trace-replay <fixture.csv>");
        const auto jobs = load_fixture(argv[1]);
        validate_fixture(jobs);

        const replay_metrics oracle        = replay(jobs, cache_kind::none);
        const replay_metrics oracle_repeat = replay(jobs, cache_kind::none);
        const replay_metrics policy        = replay(jobs, cache_kind::policy);
        const replay_metrics policy_repeat = replay(jobs, cache_kind::policy);
        const replay_metrics fifo          = replay(jobs, cache_kind::fifo);
        const replay_metrics lru           = replay(jobs, cache_kind::lru);

        require(same_events(oracle, oracle_repeat), "no-cache scheduler oracle replay is nondeterministic");
        require(same_events(policy, policy_repeat), "policy scheduler replay is nondeterministic");
        require(policy.saved_prefill_us == policy_repeat.saved_prefill_us &&
                    policy.peak_resident_bytes == policy_repeat.peak_resident_bytes,
                "policy cache metrics are nondeterministic");
        validate_comparison(policy, fifo, lru, oracle);
        validate_frozen_metrics(policy, fifo, lru, oracle);

        print_metrics(oracle);
        print_metrics(policy);
        print_metrics(fifo);
        print_metrics(lru);
        std::cout << "synthetic fixture: 66 requests, 36 unique complete turns, equal-byte capacity=8388608\n";
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "trace replay failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

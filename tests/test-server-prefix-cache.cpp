#include "server-prefix-cache.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace server_prefix_cache;

#undef assert
#define assert(expr)                                                                              \
    do {                                                                                          \
        if (!(expr)) {                                                                            \
            std::cerr << "check failed at " << __FILE__ << ':' << __LINE__ << ": " #expr << '\n'; \
            std::abort();                                                                         \
        }                                                                                         \
    } while (false)

static server_prefix_digest digest(uint8_t seed) {
    server_prefix_digest result;
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<uint8_t>(seed + 17 * i);
    }
    return result;
}

static server_prefix_identity identity(uint8_t seed = 1) {
    server_prefix_identity result;
    result.architecture       = "deepseek-v4-flash";
    result.model              = digest(seed + 0);
    result.tokenizer          = digest(seed + 1);
    result.chat_template      = digest(seed + 2);
    result.runtime_build      = digest(seed + 3);
    result.target_kv_layout   = digest(seed + 4);
    result.draft_kv_layout    = digest(seed + 5);
    result.rope               = digest(seed + 6);
    result.lora               = digest(seed + 7);
    result.context_size       = 1048576;
    result.dsv4_state_version = 2;
    return result;
}

static std::vector<int32_t> tokens(size_t count, int32_t seed = 0) {
    std::vector<int32_t> result(count);
    for (size_t i = 0; i < count; ++i) {
        const int64_t value = static_cast<int64_t>(seed) + 7919 * static_cast<int64_t>(i) + static_cast<int64_t>(i / 7);
        result[i]           = static_cast<int32_t>(value % 129280);
    }
    return result;
}

static server_prefix_entry entry(uint64_t handle, uint64_t generation = 1) {
    return { handle, generation, 64 * 1024, 12.5 };
}

static server_prefix_policy_key policy_key(uint64_t seed) {
    server_prefix_policy_key result;
    for (size_t offset = 0; offset < result.digest.size(); offset += sizeof(seed)) {
        const uint64_t value = seed ^ (0x9e3779b97f4a7c15ULL * (offset / sizeof(seed) + 1));
        for (size_t byte = 0; byte < sizeof(value); ++byte) {
            result.digest[offset + byte] = static_cast<uint8_t>(value >> (byte * 8));
        }
    }
    result.aligned_tokens = 128;
    return result;
}

static server_prefix_policy_candidate policy_candidate(uint64_t seed,
                                                       uint64_t unique_bytes = 100,
                                                       double   prefill_ms   = 10.0,
                                                       bool     pin          = false) {
    return { policy_key(seed), unique_bytes, prefill_ms, 0.0, pin };
}

static server_prefix_policy_config policy_config() {
    server_prefix_policy_config result;
    result.max_resident_entries = 8;
    result.max_resident_bytes   = 800;
    result.max_protected_bytes  = 400;
    result.max_pinned_entries   = 2;
    result.max_pinned_bytes     = 200;
    result.ghost_buckets        = 32;
    result.ghost_ways           = 4;
    result.sketch_width         = 128;
    result.decay_interval       = 1024;
    result.hysteresis_fraction  = 0.10;
    return result;
}

static server_prefix_session_key session_key(uint8_t identity_seed, const std::string & session_id) {
    return { identity(identity_seed), session_id };
}

static server_prefix_session_anchor session_anchor(uint64_t turn,
                                                   uint64_t generation,
                                                   uint64_t content_seed,
                                                   uint32_t message_end_tokens = 128,
                                                   uint64_t handle_id          = 100) {
    auto prefix           = policy_key(content_seed);
    prefix.aligned_tokens = message_end_tokens;
    return {
        prefix, { handle_id, generation },
         turn, 64 * 1024, 10.0, message_end_tokens
    };
}

static server_prefix_session_version session_version(const server_prefix_session_anchor & anchor) {
    return { anchor.owner, anchor.turn_id };
}

static server_prefix_session_config session_config() {
    server_prefix_session_config result;
    result.max_sessions           = 8;
    result.max_session_id_bytes   = 64;
    result.max_architecture_bytes = 64;
    result.ttl_ticks              = 100;
    return result;
}

static void test_longest_exact_prefix_and_shared_reuse() {
    server_prefix_radix radix;
    const auto          id         = identity();
    const auto          prefix_128 = tokens(128, 4);
    const auto          prefix_256 = tokens(256, 4);
    const auto          query      = tokens(391, 4);

    assert(radix.insert(id, prefix_128, entry(10)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id, prefix_256, entry(20)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id, prefix_256, entry(20)) == server_prefix_insert_status::already_present);
    for (size_t request = 0; request < 8; ++request) {
        const auto match = radix.lookup(id, query);
        assert(match.has_value());
        assert(match->matched_tokens == 256);
        assert(match->entry.handle_id == 20);
    }
    const auto short_query = radix.lookup(id, tokens(200, 4));
    assert(short_query.has_value() && short_query->matched_tokens == 128);
    assert(radix.stats().identities == 1);
    assert(radix.stats().nodes == 2);
    assert(radix.stats().entries == 2);
}

static void test_hash_collisions_never_authorize_reuse() {
    server_prefix_radix radix({ 64, 16, true });
    const auto          id_a     = identity(10);
    const auto          id_b     = identity(20);
    const auto          prefix_a = tokens(256, 11);
    const auto          prefix_b = tokens(256, 22);

    assert(radix.insert(id_a, prefix_a, entry(100)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id_a, prefix_b, entry(101)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id_b, prefix_a, entry(200)) == server_prefix_insert_status::inserted);
    assert(radix.lookup(id_a, prefix_a)->entry.handle_id == 100);
    assert(radix.lookup(id_a, prefix_b)->entry.handle_id == 101);
    assert(radix.lookup(id_b, prefix_a)->entry.handle_id == 200);
    assert(!radix.lookup(id_b, prefix_b).has_value());

    auto near_collision = prefix_a;
    near_collision[127] ^= 1;
    assert(!radix.lookup(id_a, near_collision).has_value());
}

static void assert_identity_miss(const server_prefix_identity & changed) {
    server_prefix_radix radix({ 8, 4, true });
    const auto          original = identity();
    const auto          prefix   = tokens(128, 3);
    assert(radix.insert(original, prefix, entry(1)) == server_prefix_insert_status::inserted);
    assert(!radix.lookup(changed, prefix).has_value());
}

static void test_every_compatibility_dimension_misses() {
    auto changed = identity();
    changed.architecture += "-other";
    assert_identity_miss(changed);
    changed = identity();
    changed.model[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.tokenizer[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.chat_template[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.runtime_build[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.target_kv_layout[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.draft_kv_layout[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.rope[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    changed.lora[0] ^= 1;
    assert_identity_miss(changed);
    changed = identity();
    ++changed.context_size;
    assert_identity_miss(changed);
    changed = identity();
    ++changed.dsv4_state_version;
    assert_identity_miss(changed);
}

static void test_capacity_preflight_is_atomic_and_erase_prunes() {
    server_prefix_radix radix({ 2, 2, false });
    const auto          id            = identity();
    const auto          prefix_128    = tokens(128, 1);
    const auto          prefix_256    = tokens(256, 1);
    const auto          unrelated_256 = tokens(256, 2);

    assert(radix.insert(id, prefix_128, entry(1)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id, unrelated_256, entry(2)) == server_prefix_insert_status::capacity);
    assert(radix.stats().nodes == 1 && radix.stats().entries == 1);
    assert(radix.insert(id, prefix_256, entry(2)) == server_prefix_insert_status::inserted);
    assert(radix.insert(id, prefix_256, entry(3)) == server_prefix_insert_status::conflict);
    assert(!radix.erase(id, prefix_256, 999, 1));
    assert(!radix.erase(id, prefix_256, 2, 999));
    assert(radix.erase(id, prefix_256, 2, 1));
    assert(radix.lookup(id, prefix_256)->entry.handle_id == 1);
    assert(radix.stats().nodes == 1 && radix.stats().entries == 1);
    assert(radix.erase(id, prefix_128, 1, 1));
    assert(radix.stats().identities == 0 && radix.stats().nodes == 0 && radix.stats().entries == 0);
}

static void test_invalid_inputs_do_not_allocate() {
    server_prefix_radix radix;
    const auto          id = identity();
    assert(radix.insert(id, tokens(127), entry(1)) == server_prefix_insert_status::invalid_argument);
    assert(radix.insert(id, {}, entry(1)) == server_prefix_insert_status::invalid_argument);
    assert(radix.insert(id, tokens(128), entry(0)) == server_prefix_insert_status::invalid_argument);
    auto nan_entry             = entry(1);
    nan_entry.saved_prefill_ms = std::numeric_limits<double>::quiet_NaN();
    assert(radix.insert(id, tokens(128), nan_entry) == server_prefix_insert_status::invalid_argument);
    auto invalid_identity         = id;
    invalid_identity.context_size = 0;
    assert(radix.insert(invalid_identity, tokens(128), entry(1)) == server_prefix_insert_status::invalid_argument);
    invalid_identity       = id;
    invalid_identity.model = {};
    assert(radix.insert(invalid_identity, tokens(128), entry(1)) == server_prefix_insert_status::invalid_argument);
    invalid_identity                  = id;
    invalid_identity.target_kv_layout = {};
    assert(radix.insert(invalid_identity, tokens(128), entry(1)) == server_prefix_insert_status::invalid_argument);
    assert(radix.stats().identities == 0 && radix.stats().nodes == 0);
}

static void test_one_million_token_anchor_is_bounded() {
    constexpr size_t    context_tokens = 1024 * 1024;
    server_prefix_radix radix({ context_tokens / SERVER_PREFIX_BLOCK_TOKENS, 1, false });
    const auto          id     = identity();
    auto                prefix = tokens(context_tokens, 29);
    assert(radix.insert(id, prefix, entry(77)) == server_prefix_insert_status::inserted);
    const auto match = radix.lookup(id, prefix);
    assert(match.has_value() && match->matched_tokens == context_tokens);
    assert(match->entry.handle_id == 77);
    assert(radix.stats().nodes == context_tokens / SERVER_PREFIX_BLOCK_TOKENS);
    assert(radix.stats().entries == 1);

    prefix.back() ^= 1;
    assert(!radix.lookup(id, prefix).has_value());
    prefix.back() ^= 1;
    assert(radix.erase(id, prefix, 77, 1));
    assert(radix.stats().nodes == 0 && radix.stats().identities == 0);
}

static void test_policy_second_hit_and_decay_window() {
    auto config           = policy_config();
    config.decay_interval = 4;
    server_prefix_policy policy(config);
    const auto           hot = policy_candidate(1);

    assert(policy.observe(hot).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(hot).status == server_prefix_policy_status::admitted_probation);
    assert(policy.resident(hot.key)->segment == server_prefix_policy_segment::probation);
    assert(policy.observe(hot).status == server_prefix_policy_status::resident_hit);
    assert(policy.resident(hot.key)->segment == server_prefix_policy_segment::protected_segment);

    const auto stale = policy_candidate(2);
    assert(policy.observe(stale).status == server_prefix_policy_status::first_sighting);
    for (uint64_t seed = 100; seed < 105; ++seed) {
        assert(policy.observe(policy_candidate(seed)).status == server_prefix_policy_status::first_sighting);
    }
    assert(policy.observe(stale).status == server_prefix_policy_status::first_sighting);
    assert(!policy.resident(stale.key).has_value());
    assert(policy.stats().decays >= 1);
}

static void test_policy_collisions_keep_exact_authority() {
    auto config                  = policy_config();
    config.force_hash_collisions = true;
    config.ghost_buckets         = 1;
    config.decay_interval        = 1000000;
    server_prefix_policy policy(config);
    const auto           a = policy_candidate(10);
    const auto           b = policy_candidate(20);

    assert(policy.observe(a).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(b).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(a).status == server_prefix_policy_status::admitted_probation);
    assert(policy.resident(a.key).has_value());
    assert(!policy.resident(b.key).has_value());
    assert(!policy.erase(policy_key(999)));
    assert(policy.resident(a.key).has_value());
    assert(policy.observe(b).status == server_prefix_policy_status::admitted_probation);
    assert(policy.resident(b.key).has_value());

    server_prefix_policy_result saturated;
    for (uint64_t seed = 1000; seed < 71000; ++seed) {
        saturated = policy.observe(policy_candidate(seed));
        assert(saturated.status == server_prefix_policy_status::first_sighting);
    }
    assert(saturated.estimated_frequency == std::numeric_limits<uint16_t>::max());
    assert(policy.stats().resident_entries == 2);
}

static void test_policy_probation_protection_and_pin_quota() {
    auto config                 = policy_config();
    config.max_resident_entries = 3;
    config.max_resident_bytes   = 300;
    config.max_protected_bytes  = 100;
    config.max_pinned_entries   = 1;
    config.max_pinned_bytes     = 100;
    server_prefix_policy policy(config);
    const auto           hot    = policy_candidate(1, 100, 5.0);
    const auto           cold   = policy_candidate(2, 100, 50.0);
    const auto           pinned = policy_candidate(3, 100, 1.0, true);

    assert(policy.observe(hot).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(hot).status == server_prefix_policy_status::admitted_probation);
    assert(policy.observe(hot).status == server_prefix_policy_status::resident_hit);
    assert(policy.resident(hot.key)->segment == server_prefix_policy_segment::protected_segment);
    assert(policy.observe(cold).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(cold).status == server_prefix_policy_status::admitted_probation);
    assert(policy.observe(pinned).status == server_prefix_policy_status::admitted_pinned);

    const auto pin_over_quota = policy.observe(policy_candidate(4, 100, 100.0, true));
    assert(pin_over_quota.status == server_prefix_policy_status::rejected_pin_quota);
    assert(!policy.resident(policy_key(4)).has_value());

    const auto challenger = policy_candidate(5, 100, 100.0);
    assert(policy.observe(challenger).status == server_prefix_policy_status::first_sighting);
    const auto admission = policy.observe(challenger);
    assert(admission.status == server_prefix_policy_status::admitted_probation);
    assert(admission.evicted.size() == 1 && admission.evicted.front() == cold.key);
    assert(policy.resident(hot.key)->segment == server_prefix_policy_segment::protected_segment);
    assert(policy.resident(pinned.key)->segment == server_prefix_policy_segment::pinned);
    assert(!policy.resident(cold.key).has_value());

    const auto stats = policy.stats();
    assert(stats.resident_entries == 3 && stats.resident_bytes == 300);
    assert(stats.protected_entries == 1 && stats.protected_bytes == 100);
    assert(stats.probation_entries == 1 && stats.probation_bytes == 100);
    assert(stats.pinned_entries == 1 && stats.pinned_bytes == 100);
}

static void test_policy_cost_size_hysteresis() {
    auto config                 = policy_config();
    config.max_resident_entries = 1;
    config.max_resident_bytes   = 100;
    config.max_protected_bytes  = 0;
    config.max_pinned_entries   = 0;
    config.max_pinned_bytes     = 0;
    server_prefix_policy policy(config);
    const auto           incumbent = policy_candidate(1, 100, 10.0);
    assert(policy.observe(incumbent).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(incumbent).status == server_prefix_policy_status::admitted_probation);

    const auto marginal = policy_candidate(2, 100, 10.5);
    assert(policy.observe(marginal).status == server_prefix_policy_status::first_sighting);
    assert(policy.observe(marginal).status == server_prefix_policy_status::rejected_hysteresis);
    assert(policy.resident(incumbent.key).has_value());

    const auto efficient = policy_candidate(3, 50, 8.0);
    assert(policy.observe(efficient).status == server_prefix_policy_status::first_sighting);
    const auto admission = policy.observe(efficient);
    assert(admission.status == server_prefix_policy_status::admitted_probation);
    assert(admission.evicted.size() == 1 && admission.evicted.front() == incumbent.key);
    assert(policy.resident(efficient.key).has_value());
}

static void test_one_million_one_hit_ghosts_are_bounded() {
    auto config                 = policy_config();
    config.max_resident_entries = 64;
    config.max_resident_bytes   = 64 * 1024;
    config.max_protected_bytes  = 32 * 1024;
    config.max_pinned_entries   = 0;
    config.max_pinned_bytes     = 0;
    config.ghost_buckets        = 1024;
    config.ghost_ways           = 4;
    config.sketch_width         = 2048;
    config.decay_interval       = 65536;
    server_prefix_policy policy(config);

    constexpr uint64_t candidate_count = 1000000;
    for (uint64_t seed = 1; seed <= candidate_count; ++seed) {
        assert(policy.observe(policy_candidate(seed, 1024, 1.0)).status == server_prefix_policy_status::first_sighting);
    }
    const auto stats = policy.stats();
    assert(stats.observations == candidate_count);
    assert(stats.resident_entries == 0 && stats.resident_bytes == 0);
    assert(stats.ghost_entries <= stats.ghost_capacity);
    assert(stats.ghost_capacity == config.ghost_buckets * config.ghost_ways);
    assert(stats.sketch_cells == config.sketch_width * 4);
    assert(stats.fixed_metadata_bytes ==
           stats.ghost_bytes + stats.sketch_bytes + config.ghost_buckets * sizeof(size_t));
    assert(stats.decays == 15);
    std::cout << "million one-hit candidates: ghosts=" << stats.ghost_entries << '/' << stats.ghost_capacity
              << " ghost_bytes=" << stats.ghost_bytes << " sketch_bytes=" << stats.sketch_bytes
              << " fixed_metadata_bytes=" << stats.fixed_metadata_bytes << " decays=" << stats.decays << '\n';
}

struct trace_cache {
    explicit trace_cache(bool lru) : lru(lru) {}

    bool access(uint64_t key) {
        const auto found = std::find(order.begin(), order.end(), key);
        if (found != order.end()) {
            if (lru) {
                order.erase(found);
                order.push_back(key);
            }
            return true;
        }
        if (order.size() == capacity) {
            order.erase(order.begin());
        }
        order.push_back(key);
        return false;
    }

    static constexpr size_t capacity = 4;
    bool                    lru;
    std::vector<uint64_t>   order;
};

static void test_policy_trace_beats_fifo_and_lru_efficiency() {
    constexpr uint64_t object_bytes = 1024 * 1024;
    auto               config       = policy_config();
    config.max_resident_entries     = 4;
    config.max_resident_bytes       = 4 * object_bytes;
    config.max_protected_bytes      = 2 * object_bytes;
    config.max_pinned_entries       = 0;
    config.max_pinned_bytes         = 0;
    config.ghost_buckets            = 4096;
    config.sketch_width             = 4096;
    config.decay_interval           = 1ULL << 20;
    server_prefix_policy policy(config);
    trace_cache          fifo(false);
    trace_cache          lru(true);
    double               policy_saved = 0.0;
    double               fifo_saved   = 0.0;
    double               lru_saved    = 0.0;

    const auto access = [&](uint64_t key, double saved_ms) {
        const auto candidate = policy_candidate(key, object_bytes, saved_ms);
        if (policy.resident(candidate.key).has_value()) {
            policy_saved += saved_ms;
        }
        policy.observe(candidate);
        if (fifo.access(key)) {
            fifo_saved += saved_ms;
        }
        if (lru.access(key)) {
            lru_saved += saved_ms;
        }
    };

    access(1, 100.0);
    access(1, 100.0);
    access(1, 100.0);
    uint64_t cold_key = 2;
    for (size_t round = 0; round < 200; ++round) {
        for (size_t cold = 0; cold < 8; ++cold) {
            access(cold_key++, 2.0);
        }
        access(1, 100.0);
    }

    const auto   stats             = policy.stats();
    const double gib               = 1024.0 * 1024.0 * 1024.0;
    const double policy_efficiency = policy_saved / (static_cast<double>(stats.resident_bytes) / gib);
    const double fifo_efficiency   = fifo_saved / (static_cast<double>(fifo.order.size() * object_bytes) / gib);
    const double lru_efficiency    = lru_saved / (static_cast<double>(lru.order.size() * object_bytes) / gib);
    assert(policy_saved > fifo_saved && policy_saved > lru_saved);
    assert(policy_efficiency > fifo_efficiency && policy_efficiency > lru_efficiency);
    assert(stats.resident_entries == 1 && policy.resident(policy_key(1)).has_value());
    std::cout << "synthetic trace saved_ms/GiB: policy=" << policy_efficiency << " fifo=" << fifo_efficiency
              << " lru=" << lru_efficiency << " saved_ms=" << policy_saved << '/' << fifo_saved << '/' << lru_saved
              << '\n';
}

static void test_policy_invalid_input_and_clear() {
    auto bad_config          = policy_config();
    bad_config.ghost_buckets = 0;
    bool threw               = false;
    try {
        server_prefix_policy ignored(bad_config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    server_prefix_policy policy(policy_config());
    auto                 invalid = policy_candidate(1);
    invalid.key.digest           = {};
    assert(policy.observe(invalid).status == server_prefix_policy_status::rejected_invalid);
    invalid              = policy_candidate(1);
    invalid.unique_bytes = 0;
    assert(policy.observe(invalid).status == server_prefix_policy_status::rejected_invalid);
    assert(policy.stats().observations == 0);
    assert(policy.observe(policy_candidate(2, 100, 1.0, true)).status == server_prefix_policy_status::admitted_pinned);
    policy.clear();
    const auto stats = policy.stats();
    assert(stats.observations == 0 && stats.resident_entries == 0 && stats.ghost_entries == 0);
    assert(stats.sketch_cells != 0 && stats.ghost_capacity != 0);
}

static void test_latest_session_transactional_replacement_and_generation() {
    server_prefix_session_cache cache(session_config());
    const auto                  key    = session_key(1, "agent-session");
    const auto                  first  = session_anchor(1, 1, 10);
    const auto                  insert = cache.replace({ key, first, std::nullopt, 10 });
    assert(insert.status == server_prefix_session_status::inserted && insert.expires_at_tick == 110);
    const auto original = cache.lookup(key, 10);
    assert(original.has_value() && original->anchor == first);

    const auto missing_expected = cache.replace({ key, session_anchor(2, 2, 20), std::nullopt, 20 });
    assert(missing_expected.status == server_prefix_session_status::conflict);
    assert(cache.lookup(key, 20) == original);

    auto stale_expected             = session_version(first);
    stale_expected.owner.generation = 99;
    const auto stale                = cache.replace({ key, session_anchor(2, 2, 20), stale_expected, 20 });
    assert(stale.status == server_prefix_session_status::stale_generation);
    assert(cache.lookup(key, 20) == original);

    const auto stale_turn = cache.replace({ key, session_anchor(1, 2, 20), session_version(first), 20 });
    assert(stale_turn.status == server_prefix_session_status::stale_turn);
    assert(cache.lookup(key, 20) == original);

    const auto reused_generation = cache.replace({ key, session_anchor(2, 1, 20), session_version(first), 20 });
    assert(reused_generation.status == server_prefix_session_status::stale_generation);
    assert(cache.lookup(key, 20) == original);

    auto invalid_anchor               = session_anchor(2, 2, 20);
    invalid_anchor.message_end_tokens = 256;
    const auto invalid                = cache.replace({ key, invalid_anchor, session_version(first), 20 });
    assert(invalid.status == server_prefix_session_status::invalid_argument);
    assert(cache.lookup(key, 20) == original);

    const auto second   = session_anchor(2, 2, 20);
    const auto replaced = cache.replace({ key, second, session_version(first), 20 });
    assert(replaced.status == server_prefix_session_status::replaced);
    assert(replaced.displaced == original && replaced.expires_at_tick == 120);
    assert(cache.lookup(key, 20)->anchor == second);
    assert(cache.stats(20).entries == 1 && cache.stats(20).replacements == 1);

    assert(!cache.erase(key, session_version(first)).has_value());
    assert(cache.lookup(key, 20)->anchor == second);
    const auto erased = cache.erase(key, session_version(second));
    assert(erased.has_value() && erased->anchor == second);
    assert(cache.stats(20).entries == 0 && cache.stats(20).erases == 1);
}

static void test_latest_session_exact_identity_and_hash_collisions() {
    auto config                  = session_config();
    config.force_hash_collisions = true;
    server_prefix_session_cache cache(config);
    const auto                  key_a = session_key(1, "same-session");
    const auto                  key_b = session_key(2, "same-session");
    const auto                  key_c = session_key(1, "other-session");
    const auto                  a     = session_anchor(1, 1, 10, 128, 10);
    const auto                  b     = session_anchor(1, 1, 20, 128, 20);
    const auto                  c     = session_anchor(1, 1, 30, 128, 30);

    assert(cache.replace({ key_a, a, std::nullopt, 0 }).status == server_prefix_session_status::inserted);
    assert(cache.replace({ key_b, b, std::nullopt, 0 }).status == server_prefix_session_status::inserted);
    assert(cache.replace({ key_c, c, std::nullopt, 0 }).status == server_prefix_session_status::inserted);
    assert(cache.lookup(key_a, 0)->anchor == a);
    assert(cache.lookup(key_b, 0)->anchor == b);
    assert(cache.lookup(key_c, 0)->anchor == c);

    const auto a2 = session_anchor(2, 2, 11, 256, 10);
    assert(cache.replace({ key_a, a2, session_version(a), 1 }).status == server_prefix_session_status::replaced);
    assert(cache.lookup(key_a, 1)->anchor == a2);
    assert(cache.lookup(key_b, 1)->anchor == b);
    assert(cache.lookup(key_c, 1)->anchor == c);
    assert(!cache.erase(session_key(3, "same-session"), session_version(a2)).has_value());
    assert(cache.stats(1).entries == 3);
}

static void test_latest_session_ttl_boundaries_and_deterministic_expiry() {
    auto config      = session_config();
    config.ttl_ticks = 10;
    server_prefix_session_cache cache(config);
    const auto                  key_a = session_key(1, "a");
    const auto                  key_b = session_key(1, "b");
    const auto                  first = session_anchor(1, 1, 10);

    assert(cache.replace({ key_b, first, std::nullopt, 100 }).expires_at_tick == 110);
    assert(cache.replace({ key_a, first, std::nullopt, 100 }).expires_at_tick == 110);
    assert(cache.lookup(key_a, 109).has_value());
    assert(!cache.lookup(key_a, 110).has_value());
    assert(cache.stats(109).active == 2 && cache.stats(109).expired == 0);
    assert(cache.stats(110).active == 0 && cache.stats(110).expired == 2);
    assert(cache.expire(109).empty());

    const auto expired = cache.expire(110);
    assert(expired.size() == 2);
    assert(expired[0].key == key_a && expired[1].key == key_b);
    assert(expired[0].record.anchor == first && expired[1].record.anchor == first);
    const auto stats = cache.stats(110);
    assert(stats.entries == 0 && stats.expirations == 2);

    assert(cache.replace({ key_a, first, std::nullopt, 120 }).status == server_prefix_session_status::inserted);
    const auto expired_record = cache.lookup(key_a, 129);
    assert(expired_record.has_value());
    const auto expected_after_expiry = cache.replace({ key_a, session_anchor(2, 2, 20), session_version(first), 130 });
    assert(expected_after_expiry.status == server_prefix_session_status::conflict);
    assert(!cache.lookup(key_a, 130).has_value());
    const auto renewed = cache.replace({ key_a, session_anchor(2, 2, 20), std::nullopt, 130 });
    assert(renewed.status == server_prefix_session_status::replaced && renewed.displaced == expired_record);
    assert(cache.lookup(key_a, 139).has_value() && !cache.lookup(key_a, 140).has_value());
}

static void test_latest_session_capacity_invalid_and_atomic_failures() {
    auto config                 = session_config();
    config.max_sessions         = 1;
    config.ttl_ticks            = 5;
    config.max_session_id_bytes = 8;
    server_prefix_session_cache cache(config);
    const auto                  key_a = session_key(1, "a");
    const auto                  key_b = session_key(1, "b");
    const auto                  first = session_anchor(1, 1, 10);
    assert(cache.replace({ key_a, first, std::nullopt, 0 }).status == server_prefix_session_status::inserted);

    assert(cache.replace({ key_b, first, std::nullopt, 1 }).status == server_prefix_session_status::capacity);
    assert(cache.lookup(key_a, 1)->anchor == first);

    auto invalid_key       = key_b;
    invalid_key.session_id = "too-long-id";
    assert(cache.replace({ invalid_key, first, std::nullopt, 1 }).status ==
           server_prefix_session_status::invalid_argument);
    auto invalid_identity          = key_b;
    invalid_identity.identity.rope = {};
    assert(cache.replace({ invalid_identity, first, std::nullopt, 1 }).status ==
           server_prefix_session_status::invalid_argument);
    auto useless             = first;
    useless.saved_prefill_ms = 0.0;
    assert(cache.replace({ key_a, useless, session_version(first), 1 }).status ==
           server_prefix_session_status::invalid_argument);
    assert(
        cache.replace({ key_a, session_anchor(2, 2, 20), session_version(first), std::numeric_limits<uint64_t>::max() })
            .status == server_prefix_session_status::invalid_argument);
    assert(cache.lookup(key_a, 1)->anchor == first);
    assert(cache.stats(1).insertions == 1 && cache.stats(1).replacements == 0);

    assert(!cache.lookup(key_a, 5).has_value());
    assert(cache.replace({ key_b, first, std::nullopt, 5 }).status == server_prefix_session_status::capacity);
    const auto expired = cache.expire(5);
    assert(expired.size() == 1 && expired.front().key == key_a);
    assert(cache.replace({ key_b, first, std::nullopt, 5 }).status == server_prefix_session_status::inserted);

    auto bad_config         = session_config();
    bad_config.max_sessions = 0;
    bool threw              = false;
    try {
        server_prefix_session_cache ignored(bad_config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

static void test_one_million_latest_session_replacements_are_bounded() {
    auto config         = session_config();
    config.max_sessions = 1;
    config.ttl_ticks    = 2000000;
    server_prefix_session_cache cache(config);
    const auto                  key = session_key(1, "million-replacements");

    server_prefix_session_update update;
    update.key      = key;
    update.anchor   = session_anchor(1, 1, 1);
    update.now_tick = 1;
    assert(cache.replace(update).status == server_prefix_session_status::inserted);

    constexpr uint64_t replacement_count = 1000000;
    for (uint64_t replacement = 1; replacement <= replacement_count; ++replacement) {
        update.expected_current = session_version(update.anchor);
        const uint64_t turn     = replacement + 1;
        const uint32_t boundary = static_cast<uint32_t>((replacement % 8192 + 1) * SERVER_PREFIX_BLOCK_TOKENS);
        update.anchor           = session_anchor(turn, turn, turn, boundary);
        update.now_tick         = turn;
        const auto result       = cache.replace(update);
        assert(result.status == server_prefix_session_status::replaced);
        assert(result.displaced.has_value() && result.displaced->anchor.turn_id == turn - 1);
    }

    const auto stats = cache.stats(update.now_tick);
    assert(stats.entries == 1 && stats.active == 1 && stats.expired == 0);
    assert(stats.peak_entries == 1 && stats.insertions == 1 && stats.replacements == replacement_count);
    const auto latest = cache.lookup(key, update.now_tick);
    assert(latest.has_value() && latest->anchor == update.anchor);
    std::cout << "million session replacements: entries=" << stats.entries << " peak=" << stats.peak_entries
              << " insertions=" << stats.insertions << " replacements=" << stats.replacements << '\n';
}

int main() {
    test_longest_exact_prefix_and_shared_reuse();
    test_hash_collisions_never_authorize_reuse();
    test_every_compatibility_dimension_misses();
    test_capacity_preflight_is_atomic_and_erase_prunes();
    test_invalid_inputs_do_not_allocate();
    test_one_million_token_anchor_is_bounded();
    test_policy_second_hit_and_decay_window();
    test_policy_collisions_keep_exact_authority();
    test_policy_probation_protection_and_pin_quota();
    test_policy_cost_size_hysteresis();
    test_one_million_one_hit_ghosts_are_bounded();
    test_policy_trace_beats_fifo_and_lru_efficiency();
    test_policy_invalid_input_and_clear();
    test_latest_session_transactional_replacement_and_generation();
    test_latest_session_exact_identity_and_hash_collisions();
    test_latest_session_ttl_boundaries_and_deterministic_expiry();
    test_latest_session_capacity_invalid_and_atomic_failures();
    test_one_million_latest_session_replacements_are_bounded();
    std::cout << "server prefix cache tests passed\n";
    return 0;
}

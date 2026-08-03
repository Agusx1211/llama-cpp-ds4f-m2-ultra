#include "server-prefix-cache.h"

#include <cstdlib>
#include <iostream>
#include <limits>
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

int main() {
    test_longest_exact_prefix_and_shared_reuse();
    test_hash_collisions_never_authorize_reuse();
    test_every_compatibility_dimension_misses();
    test_capacity_preflight_is_atomic_and_erase_prunes();
    test_invalid_inputs_do_not_allocate();
    test_one_million_token_anchor_is_bounded();
    std::cout << "server prefix radix tests passed\n";
    return 0;
}

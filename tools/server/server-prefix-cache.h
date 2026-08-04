#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace server_prefix_cache {

constexpr size_t SERVER_PREFIX_BLOCK_TOKENS = 128;

using server_prefix_digest = std::array<uint8_t, 32>;

struct server_prefix_identity {
    std::string          architecture;
    server_prefix_digest model              = {};
    server_prefix_digest tokenizer          = {};
    server_prefix_digest chat_template      = {};
    server_prefix_digest runtime_build      = {};
    server_prefix_digest target_kv_layout   = {};
    server_prefix_digest draft_kv_layout    = {};
    server_prefix_digest rope               = {};
    server_prefix_digest lora               = {};
    uint32_t             context_size       = 0;
    uint32_t             dsv4_state_version = 0;

    bool operator==(const server_prefix_identity & other) const;
};

struct server_prefix_entry {
    uint64_t handle_id        = 0;
    uint64_t generation       = 0;
    uint64_t unique_bytes     = 0;
    double   saved_prefill_ms = 0.0;

    bool operator==(const server_prefix_entry & other) const;
};

struct server_prefix_match {
    server_prefix_entry entry;
    size_t              matched_tokens = 0;
};

enum class server_prefix_insert_status : uint8_t {
    inserted,
    already_present,
    invalid_argument,
    capacity,
    conflict,
};

struct server_prefix_radix_config {
    size_t max_nodes             = 65536;
    size_t max_entries           = 8192;
    // Deterministic test seam proving that equality, not hashes, authorizes
    // reuse. It is not exposed by server configuration.
    bool   force_hash_collisions = false;
};

struct server_prefix_radix_stats {
    size_t identities = 0;
    size_t nodes      = 0;
    size_t entries    = 0;
};

// A compact, content-derived identity for one aligned prefix candidate. The
// digest must cover the full compatibility identity and exact token blocks.
// Hashes inside the policy are only table selectors; this complete key remains
// authoritative for resident and ghost equality.
struct server_prefix_policy_key {
    server_prefix_digest digest         = {};
    uint32_t             aligned_tokens = 0;

    bool operator==(const server_prefix_policy_key & other) const;
};

struct server_prefix_policy_candidate {
    server_prefix_policy_key key;
    uint64_t                 unique_bytes       = 0;
    double                   prefill_ms_avoided = 0.0;
    double                   restore_ms         = 0.0;
    bool                     pin                = false;
};

enum class server_prefix_policy_segment : uint8_t {
    probation,
    protected_segment,
    pinned,
};

enum class server_prefix_policy_status : uint8_t {
    first_sighting,
    admitted_probation,
    admitted_pinned,
    resident_hit,
    rejected_invalid,
    rejected_capacity,
    rejected_pin_quota,
    rejected_hysteresis,
};

struct server_prefix_policy_result {
    server_prefix_policy_status           status              = server_prefix_policy_status::rejected_invalid;
    uint16_t                              estimated_frequency = 0;
    double                                priority            = 0.0;
    std::vector<server_prefix_policy_key> evicted;
};

struct server_prefix_policy_config {
    size_t   max_resident_entries = 8192;
    uint64_t max_resident_bytes   = 8ULL * 1024 * 1024 * 1024;
    uint64_t max_protected_bytes  = 6ULL * 1024 * 1024 * 1024;
    size_t   max_pinned_entries   = 64;
    uint64_t max_pinned_bytes     = 512ULL * 1024 * 1024;
    size_t   ghost_buckets        = 4096;
    size_t   ghost_ways           = 4;
    size_t   sketch_width         = 16384;
    uint64_t decay_interval       = 65536;
    double   hysteresis_fraction  = 0.10;

    // Deterministic test seam. Equality still uses the complete policy key.
    bool force_hash_collisions = false;
};

struct server_prefix_policy_stats {
    size_t   resident_entries     = 0;
    uint64_t resident_bytes       = 0;
    size_t   probation_entries    = 0;
    uint64_t probation_bytes      = 0;
    size_t   protected_entries    = 0;
    uint64_t protected_bytes      = 0;
    size_t   pinned_entries       = 0;
    uint64_t pinned_bytes         = 0;
    size_t   ghost_entries        = 0;
    size_t   ghost_capacity       = 0;
    size_t   ghost_bytes          = 0;
    size_t   sketch_cells         = 0;
    size_t   sketch_bytes         = 0;
    size_t   fixed_metadata_bytes = 0;
    uint64_t observations         = 0;
    uint64_t decays               = 0;
};

struct server_prefix_policy_resident {
    server_prefix_policy_segment segment;
    uint64_t                     unique_bytes       = 0;
    double                       prefill_ms_avoided = 0.0;
    double                       restore_ms         = 0.0;
    uint16_t                     frequency          = 0;
    double                       priority           = 0.0;
};

class server_prefix_policy {
  public:
    explicit server_prefix_policy(server_prefix_policy_config config = {});
    ~server_prefix_policy();

    server_prefix_policy(const server_prefix_policy &)             = delete;
    server_prefix_policy & operator=(const server_prefix_policy &) = delete;

    server_prefix_policy_result observe(const server_prefix_policy_candidate & candidate);

    std::optional<server_prefix_policy_resident> resident(const server_prefix_policy_key & key) const;
    bool                                         erase(const server_prefix_policy_key & key);
    void                                         clear();
    server_prefix_policy_stats                   stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

// Session IDs are compared together with every compatibility field. Internal
// hashes may narrow lookup candidates but never authorize cross-session or
// cross-model reuse.
struct server_prefix_session_key {
    server_prefix_identity identity;
    std::string            session_id;

    bool operator==(const server_prefix_session_key & other) const;
};

struct server_prefix_session_owner {
    uint64_t handle_id  = 0;
    uint64_t generation = 0;

    bool operator==(const server_prefix_session_owner & other) const;
};

struct server_prefix_session_version {
    server_prefix_session_owner owner;
    uint64_t                    turn_id = 0;

    bool operator==(const server_prefix_session_version & other) const;
};

struct server_prefix_session_anchor {
    server_prefix_policy_key    prefix;
    server_prefix_session_owner owner;
    uint64_t                    turn_id            = 0;
    uint64_t                    unique_bytes       = 0;
    double                      saved_prefill_ms   = 0.0;
    uint32_t                    message_end_tokens = 0;

    bool operator==(const server_prefix_session_anchor & other) const;
};

struct server_prefix_session_record {
    server_prefix_session_anchor anchor;
    uint64_t                     expires_at_tick = 0;

    bool operator==(const server_prefix_session_record & other) const;
};

struct server_prefix_session_update {
    server_prefix_session_key                    key;
    server_prefix_session_anchor                 anchor;
    std::optional<server_prefix_session_version> expected_current;
    uint64_t                                     now_tick = 0;
};

enum class server_prefix_session_status : uint8_t {
    inserted,
    replaced,
    invalid_argument,
    capacity,
    conflict,
    stale_generation,
    stale_turn,
};

struct server_prefix_session_result {
    server_prefix_session_status                status = server_prefix_session_status::invalid_argument;
    std::optional<server_prefix_session_record> displaced;
    uint64_t                                    expires_at_tick = 0;
};

struct server_prefix_session_config {
    size_t   max_sessions           = 8192;
    size_t   max_session_id_bytes   = 256;
    size_t   max_architecture_bytes = 64;
    // The serialized owner chooses a monotonic tick unit and supplies every
    // now_tick. Entries are live on [publication, expires_at_tick).
    uint64_t ttl_ticks              = 3600000;

    // Deterministic test seam. Exact key equality remains authoritative.
    bool force_hash_collisions = false;
};

struct server_prefix_session_stats {
    size_t   entries      = 0;
    size_t   active       = 0;
    size_t   expired      = 0;
    size_t   peak_entries = 0;
    uint64_t insertions   = 0;
    uint64_t replacements = 0;
    uint64_t expirations  = 0;
    uint64_t erases       = 0;
};

struct server_prefix_session_expired {
    server_prefix_session_key    key;
    server_prefix_session_record record;
};

class server_prefix_session_cache {
  public:
    explicit server_prefix_session_cache(server_prefix_session_config config = {});
    ~server_prefix_session_cache();

    server_prefix_session_cache(const server_prefix_session_cache &)             = delete;
    server_prefix_session_cache & operator=(const server_prefix_session_cache &) = delete;

    server_prefix_session_result replace(const server_prefix_session_update & update);

    std::optional<server_prefix_session_record> lookup(const server_prefix_session_key & key, uint64_t now_tick) const;

    std::optional<server_prefix_session_record> erase(const server_prefix_session_key &     key,
                                                      const server_prefix_session_version & expected);

    // Expired records deliberately consume bounded capacity until this call
    // returns their exact owners for physical cleanup. A successful replace
    // similarly returns its one displaced record; failures mutate nothing.
    std::vector<server_prefix_session_expired> expire(uint64_t now_tick);

    void                        clear();
    server_prefix_session_stats stats(uint64_t now_tick) const;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

class server_prefix_radix {
  public:
    explicit server_prefix_radix(server_prefix_radix_config config = {});
    ~server_prefix_radix();

    server_prefix_radix(const server_prefix_radix &)             = delete;
    server_prefix_radix & operator=(const server_prefix_radix &) = delete;

    server_prefix_insert_status insert(const server_prefix_identity & identity,
                                       const std::vector<int32_t> &   tokens,
                                       const server_prefix_entry &    entry);

    std::optional<server_prefix_match> lookup(const server_prefix_identity & identity,
                                              const std::vector<int32_t> &   tokens) const;

    bool erase(const server_prefix_identity & identity,
               const std::vector<int32_t> &   tokens,
               uint64_t                       expected_handle_id,
               uint64_t                       expected_generation);

    void                      clear();
    server_prefix_radix_stats stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

}  // namespace server_prefix_cache

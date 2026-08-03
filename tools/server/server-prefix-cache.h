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

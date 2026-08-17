#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace server_capture {

// Schema v2: widened from 5 to 16 proposal slots so one record covers the
// deployed ngram-mod (n_max 16) + DSpark configuration. Drafts deeper than 16
// truncate and set flag_truncated_proposals.
constexpr size_t  MAX_PROPOSAL_TOKENS = 16;
constexpr uint8_t NO_REJECTION        = UINT8_MAX;

enum class mode : uint8_t {
    off = 0,
    metrics_only,
    compact,
    sampled_rich,
};

// Which speculator produced the proposals of a cycle. Values are part of the
// persisted schema; append only.
enum class proposer : uint8_t {
    none = 0,
    draft_simple,
    eagle3,
    mtp,
    dflash,
    dspark,
    ngram_simple,
    ngram_map_k,
    ngram_map_k4v,
    ngram_mod,
    ngram_cache,
    other,
};

enum class bypass_reason : uint8_t {
    none = 0,
    disabled,
    unaligned,
    stale,
    pressure,
    unsupported,
};

// Bits of cycle_observation::flags. Part of the persisted schema; append only.
constexpr uint8_t flag_truncated_proposals = 1u << 0;  // draft was deeper than MAX_PROPOSAL_TOKENS
constexpr uint8_t flag_checkpoint_restore  = 1u << 1;  // partial acceptance forced a checkpoint restore;
                                                       // the accepted prefix will be replayed next cycle
constexpr uint8_t flag_replay              = 1u << 2;  // replay pass of a previously recorded restore cycle

// The inference thread fills this value in place. Its fixed 256-byte shape
// covers the maximum sixteen-token cycle without allocating or retaining
// request-owned memory. Stable enum values are serialized only by a later
// versioned background writer, not by copying this process ABI directly.
struct alignas(64) cycle_observation {
    uint64_t request_id         = 0;
    uint64_t committed_position = 0;
    uint64_t scheduler_epoch    = 0;
    uint64_t monotonic_ns       = 0;

    uint32_t schema_version = 2;
    uint32_t cycle_sequence = 0;

    std::array<int32_t, MAX_PROPOSAL_TOKENS> proposal_token_ids     = {};
    std::array<float, MAX_PROPOSAL_TOKENS>   selected_probabilities = {};
    std::array<float, MAX_PROPOSAL_TOKENS>   raw_confidences        = {};

    int32_t  target_correction_or_bonus_id = -1;
    uint32_t draft_time_us                 = 0;
    uint32_t verify_time_us                = 0;
    uint32_t scheduler_time_us             = 0;

    uint8_t       scheduled_decode_width = 0;
    uint8_t       verifier_geometry      = 0;
    uint8_t       proposal_count         = 0;
    uint8_t       accepted_prefix_length = 0;
    uint8_t       first_rejection        = NO_REJECTION;
    proposer      proposed_by            = proposer::none;
    bypass_reason bypass                 = bypass_reason::none;
    uint8_t       flags                  = 0;
};

static_assert(sizeof(cycle_observation) == 256, "capture observation must remain compact");
static_assert(std::is_trivially_copyable<cycle_observation>::value, "capture observation must copy without ownership");
static_assert(std::is_standard_layout<cycle_observation>::value,
              "capture observation must have stable in-process layout");

struct ring_stats {
    size_t   capacity    = 0;
    size_t   size_approx = 0;
    uint64_t pushed      = 0;
    uint64_t popped      = 0;
    uint64_t dropped     = 0;
};

// Single-producer/single-consumer by contract: the inference loop owns push,
// and one background writer owns pop. Operations are wait-free, non-blocking,
// noexcept, and allocation-free after construction. A full ring drops the new
// record so capture can never impose storage backpressure on inference.
class spsc_ring {
  public:
    explicit spsc_ring(size_t capacity);
    ~spsc_ring();

    spsc_ring(const spsc_ring &)             = delete;
    spsc_ring & operator=(const spsc_ring &) = delete;

    bool try_push(const cycle_observation & observation) noexcept;
    bool try_pop(cycle_observation & observation) noexcept;

    size_t     capacity() const noexcept;
    ring_stats stats() const noexcept;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

}  // namespace server_capture

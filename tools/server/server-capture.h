#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace server_capture {

constexpr size_t  MAX_PROPOSAL_TOKENS = 5;
constexpr uint8_t NO_REJECTION        = UINT8_MAX;

enum class mode : uint8_t {
    off = 0,
    metrics_only,
    compact,
    sampled_rich,
};

enum class dspark_mode : uint8_t {
    target_only = 0,
    single_only,
    adaptive_depth_one,
};

enum class bypass_reason : uint8_t {
    none = 0,
    disabled,
    unaligned,
    stale,
    pressure,
    unsupported,
};

// The inference thread fills this value in place. Its fixed 128-byte shape
// covers the maximum five-token DSpark cycle without allocating or retaining
// request-owned memory. Stable enum values are serialized only by a later
// versioned background writer, not by copying this process ABI directly.
struct alignas(64) cycle_observation {
    uint64_t request_id         = 0;
    uint64_t committed_position = 0;
    uint64_t scheduler_epoch    = 0;
    uint64_t monotonic_ns       = 0;

    uint32_t schema_version = 1;
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
    dspark_mode   active_mode            = dspark_mode::target_only;
    bypass_reason bypass                 = bypass_reason::none;
    uint8_t       flags                  = 0;

    std::array<uint8_t, 4> reserved = {};
};

static_assert(sizeof(cycle_observation) == 128, "capture observation must remain compact");
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

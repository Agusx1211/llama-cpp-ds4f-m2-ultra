#pragma once

#include "server-scheduler.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace server_prefill {

struct coordinator_config {
    // Keep prefill handoffs on stable DeepSeek V4 token anchors.
    uint32_t alignment_tokens = 128;

    // These independent deterministic safety caps are not measured or
    // predicted time budgets. The live loop further clamps every value to the
    // available llama batch space.
    uint32_t idle_chunk_tokens          = 2048;
    uint32_t active_decode_chunk_tokens = 256;
    uint32_t active_fast_chunk_tokens   = 128;

    // [TAG_PREFILL_PRIORITY]
    // On the M2 Ultra DSV4 vertical a decode ubatch inserted into a prefill
    // iteration costs a fixed deep-context weight sweep no matter how large
    // the prefill chunk it rides with is. Shrinking the chunk under active
    // decode therefore multiplies that fixed cost instead of amortizing it.
    // priority_chunk_tokens = idle_chunk_tokens retains large-M efficiency;
    // decode cadence, rather than lane-specific chunk shrinkage, bounds
    // generator service.
    //
    // prefill_priority = false restores the pre-priority behavior exactly,
    // including active_decode_chunk_tokens and active_fast_chunk_tokens.
    // priority_chunk_tokens = 0 means "no extra cap", i.e. the full
    // idle_chunk_tokens budget; a non-zero value caps the priority chunk and
    // exists so the chunk size can be swept without rebuilding.
    bool     prefill_priority      = true;
    uint32_t priority_chunk_tokens = 0;

    // A cohort normally keeps a multi-chunk lease. At an aligned committed
    // boundary this limit gives another waiting cohort a deterministic turn.
    uint32_t max_lease_chunks = 4;

    // Coordinator-local chunk service mirrors the server scheduler's 16:4:1
    // lane weights. Empty lanes earn no credit, so service is work-conserving.
    std::array<uint32_t, server_scheduler::lane_count> lane_weights = { 1, 4, 16 };
};

struct media_chunk_plan {
    bool decode_allowed           = false;
    bool clear_backend_on_failure = false;
};

// Media decode does not produce a server_batch row. A final media chunk would
// therefore reach prompt completion without logits and is rejected before the
// backend call. Any allowed media decode owns an explicit backend-cleanup
// obligation if processing or prompt bookkeeping fails.
media_chunk_plan plan_media_chunk(uint64_t begin_token, uint64_t end_token, uint64_t prompt_tokens);

struct candidate {
    uint64_t               request_id = 0;
    uint64_t               cohort_id  = 0;
    server_scheduler::lane priority   = server_scheduler::lane::normal;
    uint64_t               arrival_us = 0;
};

struct decode_activity {
    bool                   active       = false;
    server_scheduler::lane highest_lane = server_scheduler::lane::low;

    void include(server_scheduler::lane priority) {
        if (!active || static_cast<uint8_t>(priority) > static_cast<uint8_t>(highest_lane)) {
            highest_lane = priority;
        }
        active = true;
    }
};

// Prefill-progress clock for generation service. Wall time is circular here:
// chunk duration is an output of the policy, while committed chunks are an
// input the scheduler controls. Entering a window services decode immediately;
// after that, at most chunks_per_decode successful prefill commits may occur
// before decode is due again. Leaving the window restores unrestricted decode.
class decode_cadence {
  public:
    explicit decode_cadence(uint32_t chunks_per_decode = 1);

    bool begin_iteration(bool prefill_pending);
    void note_chunk_committed();
    void note_decode_served();
    void force_decode_next();

    uint32_t chunks_per_decode() const { return chunks_per_decode_; }
    uint32_t chunks_since_decode() const { return chunks_since_decode_; }

  private:
    uint32_t chunks_per_decode_   = 1;
    uint32_t chunks_since_decode_ = 0;
    bool     window_open_         = false;
    bool     force_next_          = false;
};

// Keeps resident lower-priority generators alive without letting them dilute
// the highest runnable lane. A preemption window starts with dominant-only
// iterations; after every interval successful dominant selections, one mixed
// iteration is due. If no lower row can join that due iteration, service stays
// due rather than silently restarting the interval.
class priority_decode_cadence {
  public:
    explicit priority_decode_cadence(uint32_t dominant_iterations_per_lower = 64);

    bool begin_iteration(bool mixed_priority_decode);
    void note_iteration(bool mixed_priority_decode, bool dominant_selected, bool lower_selected);

    uint32_t dominant_iterations_per_lower() const { return dominant_iterations_per_lower_; }

    uint32_t dominant_iterations_since_lower() const { return dominant_iterations_since_lower_; }

  private:
    uint32_t dominant_iterations_per_lower_   = 64;
    uint32_t dominant_iterations_since_lower_ = 0;
    bool     window_open_                     = false;
};

struct owner_selection {
    bool                   selected   = false;
    uint64_t               request_id = 0;
    uint64_t               cohort_id  = 0;
    server_scheduler::lane priority   = server_scheduler::lane::low;

    explicit operator bool() const { return selected; }
};

struct chunk_limit {
    bool     allowed   = false;
    uint64_t end_token = 0;  // exclusive

    explicit operator bool() const { return allowed; }
};

struct chunk_lease {
    uint64_t request_id       = 0;
    uint64_t cohort_id        = 0;
    uint64_t generation       = 0;
    uint64_t begin_token      = 0;
    uint64_t end_token        = 0;  // exclusive
    bool     yield_boundary   = false;
    bool     completes_prompt = false;

    explicit operator bool() const { return generation != 0; }

    bool operator==(const chunk_lease & other) const;
};

struct staged_batch_abort_plan {
    bool abort_coordinator  = false;
    bool clear_prompt_state = false;
};

// Durable per-slot disposition for the post-commit half of a staged family.
// A terminal handoff is consumed only after the queue's exception-safe gate
// returns. Metric take methods are one-shot so a later sibling/release failure
// cannot either duplicate or erase accounting already made externally visible.
struct staged_member_disposition {
    void defer_prompt_metric() { prompt_metric_pending = true; }
    bool take_prompt_metric();
    bool take_prediction_metric();

    void record_terminal_handoff() { terminal_handed_off = true; }
    bool may_publish_failure_terminal() const { return !terminal_handed_off; }

  private:
    bool terminal_handed_off       = false;
    bool prompt_metric_pending     = false;
    bool prompt_metric_applied     = false;
    bool prediction_metric_applied = false;
};

// Tracks the physical decode transaction for one exact coordinator lease.
// Successful decode and post-decode sub-batches advance contiguous prefixes.
// Every incomplete abort conservatively clears target/draft/logical prompt
// state because logical staging precedes backend mutation. Parent/child fan-out
// is a one-shot action exposed only after the complete range has committed.
class staged_batch_lifecycle {
  public:
    bool begin(const chunk_lease & lease, int32_t batch_offset, int32_t batch_tokens);

    bool record_decoded_view(int32_t view_offset, int32_t view_tokens);
    bool record_post_decoded_view(int32_t view_offset, int32_t view_tokens);

    bool view_overlaps(int32_t view_offset, int32_t view_tokens) const;
    bool decoded_complete() const;
    bool ready_for_family_preparation() const;
    bool ready_to_commit() const;
    bool record_commit();
    bool take_parent_activation();
    bool owns_family_task(uint64_t request_id, uint64_t parent_request_id = 0) const;

    staged_batch_abort_plan abort_plan() const;

    void reset();

    explicit operator bool() const { return static_cast<bool>(lease_); }

    const chunk_lease & lease() const { return lease_; }
    int32_t decoded_tokens() const { return decoded_tokens_; }

  private:
    bool view_overlap(int32_t view_offset, int32_t view_tokens, int32_t & begin, int32_t & end) const;

    chunk_lease lease_;
    int32_t     batch_offset_       = 0;
    int32_t     batch_tokens_       = 0;
    int32_t     decoded_tokens_     = 0;
    int32_t     postdecoded_tokens_ = 0;
    bool        committed_          = false;
    bool        activation_taken_   = false;
};

struct staged_view_result {
    bool valid     = false;
    bool committed = false;
};

// Production and host tests share this ordering seam. A decoded view is not
// eligible to commit until its real post-decode work returns successfully.
// Commit precedes the one-shot finalization callback. The lifecycle remains in
// flight through activation and buffered publication, so pre-commit failures
// can abort the exact lease and post-commit failures can still clear/release
// the whole family with fail-closed semantics.
template <typename PostDecode, typename Commit, typename Finalize>
staged_view_result finish_decoded_view(staged_batch_lifecycle & lifecycle,
                                       int32_t                  view_offset,
                                       int32_t                  view_tokens,
                                       PostDecode &&            post_decode,
                                       Commit &&                commit,
                                       Finalize &&              finalize) {
    if (lifecycle.view_overlaps(view_offset, view_tokens)) {
        post_decode();
    }
    if (!lifecycle.record_post_decoded_view(view_offset, view_tokens)) {
        return {};
    }
    if (!lifecycle.ready_to_commit()) {
        return { true, false };
    }

    const chunk_lease lease = lifecycle.lease();
    if (!commit(lease) || !lifecycle.record_commit()) {
        return {};
    }

    const bool activate_parent = lifecycle.take_parent_activation();
    finalize(lease, activate_parent);
    lifecycle.reset();
    return { true, true };
}

struct coordinator_snapshot {
    uint64_t request_id             = 0;
    uint64_t cohort_id              = 0;
    uint32_t committed_lease_chunks = 0;
    bool     at_yield_boundary      = true;
    bool     chunk_in_flight        = false;
};

// Serialized control for prompt ownership. The server loop supplies exactly
// one candidate per parent/cohort. Tokens remain owned by one cohort until an
// exact staged range commits; a higher lane or fairness turn may take over only
// at an aligned committed anchor (the final tail is also a safe boundary).
class coordinator {
  public:
    explicit coordinator(coordinator_config config = {});

    owner_selection select_owner(const std::vector<candidate> & candidates);

    chunk_limit limit_chunk(uint64_t                request_id,
                            uint64_t                committed_tokens,
                            uint64_t                prompt_tokens,
                            uint32_t                available_batch_tokens,
                            const decode_activity & activity) const;

    chunk_lease stage_chunk(uint64_t request_id, uint64_t begin_token, uint64_t end_token, uint64_t prompt_tokens);

    bool commit_chunk(const chunk_lease & lease);
    bool abort_chunk(const chunk_lease & lease);

    // Cancellation is serialized between decode iterations. It therefore
    // releases ownership at the current committed boundary and never tears
    // down an in-flight chunk.
    bool cancel_request(uint64_t request_id);
    bool cancel_cohort(uint64_t cohort_id);

    void                 reset();
    coordinator_snapshot snapshot() const;

    const coordinator_config & config() const { return cfg; }

  private:
    const candidate * next_candidate_in_lane(const std::vector<candidate> & candidates,
                                             server_scheduler::lane         priority,
                                             uint64_t                       excluded_cohort = 0) const;
    const candidate * next_weighted_candidate(const std::vector<candidate> & candidates,
                                              uint64_t                       excluded_cohort = 0) const;
    void              account_lane_service(server_scheduler::lane served_lane);
    void              remember_selection(const candidate & selected);
    void              clear_owner();

    coordinator_config cfg;
    candidate          owner;
    chunk_lease        in_flight;
    uint64_t           next_generation        = 1;
    uint32_t           committed_lease_chunks = 0;
    bool               at_yield_boundary      = true;

    struct fairness_cursor {
        bool     valid      = false;
        uint64_t arrival_us = 0;
        uint64_t request_id = 0;
    };

    std::array<fairness_cursor, server_scheduler::lane_count> fairness_cursors = {};
    std::array<int64_t, server_scheduler::lane_count>         lane_credits     = {};
    std::array<bool, server_scheduler::lane_count>            candidate_lanes  = {};

    struct committed_cursor {
        uint64_t cohort_id = 0;
        uint64_t token     = 0;
    };

    std::map<uint64_t, committed_cursor> committed_cursors;
};

}  // namespace server_prefill

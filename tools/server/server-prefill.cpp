#include "server-prefill.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace server_prefill {

namespace {

bool valid_lane(server_scheduler::lane value) {
    return static_cast<uint8_t>(value) < static_cast<uint8_t>(server_scheduler::lane::count);
}

int64_t saturating_add_signed(int64_t lhs, int64_t rhs) {
    if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return std::numeric_limits<int64_t>::max();
    }
    if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) {
        return std::numeric_limits<int64_t>::min();
    }
    return lhs + rhs;
}

bool candidate_better(const candidate & lhs, const candidate & rhs) {
    if (lhs.priority != rhs.priority) {
        return static_cast<uint8_t>(lhs.priority) > static_cast<uint8_t>(rhs.priority);
    }
    if (lhs.arrival_us != rhs.arrival_us) {
        return lhs.arrival_us < rhs.arrival_us;
    }
    return lhs.request_id < rhs.request_id;
}

const candidate * best_candidate(const std::vector<candidate> & candidates, uint64_t excluded_cohort = 0) {
    const candidate * best = nullptr;
    for (const candidate & current : candidates) {
        if (current.cohort_id == excluded_cohort) {
            continue;
        }
        if (best == nullptr || candidate_better(current, *best)) {
            best = &current;
        }
    }
    return best;
}

}  // namespace

decode_cadence::decode_cadence(uint32_t chunks_per_decode) : chunks_per_decode_(chunks_per_decode) {
}

bool decode_cadence::begin_iteration(bool prefill_pending) {
    if (!prefill_pending) {
        window_open_         = false;
        force_next_          = false;
        chunks_since_decode_ = chunks_per_decode_;
        return true;
    }
    if (!window_open_) {
        window_open_         = true;
        chunks_since_decode_ = chunks_per_decode_;
    }

    const bool due =
        force_next_ || (chunks_per_decode_ > 0 && chunks_since_decode_ >= chunks_per_decode_);
    force_next_ = false;
    return due;
}

void decode_cadence::note_chunk_committed() {
    if (chunks_since_decode_ != std::numeric_limits<uint32_t>::max()) {
        ++chunks_since_decode_;
    }
}

void decode_cadence::note_decode_served() {
    chunks_since_decode_ = 0;
}

void decode_cadence::force_decode_next() {
    force_next_ = true;
}

priority_decode_cadence::priority_decode_cadence(uint32_t dominant_iterations_per_lower) :
    dominant_iterations_per_lower_(dominant_iterations_per_lower) {
    if (dominant_iterations_per_lower_ == 0) {
        throw std::invalid_argument("invalid priority decode cadence");
    }
}

bool priority_decode_cadence::begin_iteration(bool mixed_priority_decode) {
    if (!mixed_priority_decode) {
        window_open_                     = false;
        dominant_iterations_since_lower_ = 0;
        return true;
    }
    if (!window_open_) {
        window_open_                     = true;
        dominant_iterations_since_lower_ = 0;
    }
    return dominant_iterations_since_lower_ >= dominant_iterations_per_lower_;
}

void priority_decode_cadence::note_iteration(bool mixed_priority_decode, bool dominant_selected, bool lower_selected) {
    if (!mixed_priority_decode) {
        window_open_                     = false;
        dominant_iterations_since_lower_ = 0;
        return;
    }
    if (lower_selected) {
        dominant_iterations_since_lower_ = 0;
    } else if (dominant_selected && dominant_iterations_since_lower_ != std::numeric_limits<uint32_t>::max()) {
        ++dominant_iterations_since_lower_;
    }
}

bool chunk_lease::operator==(const chunk_lease & other) const {
    return request_id == other.request_id && cohort_id == other.cohort_id && generation == other.generation &&
           begin_token == other.begin_token && end_token == other.end_token && yield_boundary == other.yield_boundary &&
           completes_prompt == other.completes_prompt;
}

bool staged_member_disposition::take_prompt_metric() {
    if (!prompt_metric_pending || prompt_metric_applied) {
        return false;
    }
    prompt_metric_applied = true;
    return true;
}

bool staged_member_disposition::take_prediction_metric() {
    if (!terminal_handed_off || prediction_metric_applied) {
        return false;
    }
    prediction_metric_applied = true;
    return true;
}

bool staged_batch_lifecycle::begin(const chunk_lease & lease, int32_t batch_offset, int32_t batch_tokens) {
    if (lease_ || !lease || lease.begin_token >= lease.end_token || batch_offset < 0 || batch_tokens <= 0 ||
        batch_offset > std::numeric_limits<int32_t>::max() - batch_tokens ||
        lease.end_token - lease.begin_token != static_cast<uint64_t>(batch_tokens)) {
        return false;
    }
    lease_        = lease;
    batch_offset_ = batch_offset;
    batch_tokens_ = batch_tokens;
    return true;
}

bool staged_batch_lifecycle::view_overlap(
        int32_t view_offset, int32_t view_tokens, int32_t & begin, int32_t & end) const {
    if (!lease_ || view_offset < 0 || view_tokens <= 0 ||
        view_offset > std::numeric_limits<int32_t>::max() - view_tokens) {
        return false;
    }
    begin = std::max(view_offset, batch_offset_);
    end   = std::min(view_offset + view_tokens, batch_offset_ + batch_tokens_);
    return true;
}

bool staged_batch_lifecycle::record_decoded_view(int32_t view_offset, int32_t view_tokens) {
    int32_t begin = 0;
    int32_t end   = 0;
    if (!view_overlap(view_offset, view_tokens, begin, end) || committed_) {
        return false;
    }
    if (end <= begin) {
        return true;
    }
    if (begin != batch_offset_ + decoded_tokens_) {
        return false;
    }
    decoded_tokens_ += end - begin;
    return decoded_tokens_ <= batch_tokens_;
}

bool staged_batch_lifecycle::record_post_decoded_view(int32_t view_offset, int32_t view_tokens) {
    int32_t begin = 0;
    int32_t end   = 0;
    if (!view_overlap(view_offset, view_tokens, begin, end) || committed_) {
        return false;
    }
    if (end <= begin) {
        return true;
    }
    if (begin != batch_offset_ + postdecoded_tokens_ || end > batch_offset_ + decoded_tokens_) {
        return false;
    }
    postdecoded_tokens_ += end - begin;
    return postdecoded_tokens_ <= batch_tokens_;
}

bool staged_batch_lifecycle::view_overlaps(int32_t view_offset, int32_t view_tokens) const {
    int32_t begin = 0;
    int32_t end   = 0;
    return view_overlap(view_offset, view_tokens, begin, end) && end > begin;
}

bool staged_batch_lifecycle::decoded_complete() const {
    return lease_ && !committed_ && decoded_tokens_ == batch_tokens_;
}

bool staged_batch_lifecycle::ready_for_family_preparation() const {
    return decoded_complete() && lease_.completes_prompt;
}

bool staged_batch_lifecycle::ready_to_commit() const {
    return decoded_complete() && postdecoded_tokens_ == batch_tokens_;
}

bool staged_batch_lifecycle::record_commit() {
    if (!ready_to_commit()) {
        return false;
    }
    committed_ = true;
    return true;
}

bool staged_batch_lifecycle::take_parent_activation() {
    if (!committed_ || !lease_.completes_prompt || activation_taken_) {
        return false;
    }
    activation_taken_ = true;
    return true;
}

bool staged_batch_lifecycle::owns_family_task(uint64_t request_id, uint64_t parent_request_id) const {
    return lease_ && (request_id == lease_.request_id || parent_request_id == lease_.request_id);
}

staged_batch_abort_plan staged_batch_lifecycle::abort_plan() const {
    if (!lease_) {
        return {};
    }
    // Staging has already advanced the logical prompt. Clear target, draft,
    // and logical state on every incomplete abort, including failures between
    // successful target decode and speculative draft processing.
    return { !committed_, true };
}

void staged_batch_lifecycle::reset() {
    *this = {};
}

media_chunk_plan plan_media_chunk(uint64_t begin_token, uint64_t end_token, uint64_t prompt_tokens) {
    if (begin_token >= end_token || end_token > prompt_tokens) {
        throw std::invalid_argument("invalid server prefill media range");
    }
    if (end_token == prompt_tokens) {
        return {};
    }
    return { true, true };
}

coordinator::coordinator(coordinator_config config) : cfg(config) {
    if (cfg.alignment_tokens == 0 || cfg.idle_chunk_tokens == 0 || cfg.active_decode_chunk_tokens == 0 ||
        cfg.active_fast_chunk_tokens == 0 || cfg.max_lease_chunks == 0 ||
        cfg.active_decode_chunk_tokens > cfg.idle_chunk_tokens ||
        cfg.priority_chunk_tokens > cfg.idle_chunk_tokens ||
        cfg.active_fast_chunk_tokens > cfg.active_decode_chunk_tokens ||
        std::any_of(cfg.lane_weights.begin(), cfg.lane_weights.end(), [](uint32_t weight) { return weight == 0; })) {
        throw std::invalid_argument("invalid server prefill coordinator configuration");
    }
}

owner_selection coordinator::select_owner(const std::vector<candidate> & candidates) {
    for (size_t index = 0; index < candidates.size(); ++index) {
        const candidate & current = candidates[index];
        if (current.request_id == 0 || current.cohort_id == 0 || !valid_lane(current.priority)) {
            return {};
        }
        for (size_t prior = 0; prior < index; ++prior) {
            if (candidates[prior].request_id == current.request_id ||
                candidates[prior].cohort_id == current.cohort_id) {
                return {};
            }
        }
    }

    candidate_lanes = {};
    for (const candidate & current : candidates) {
        candidate_lanes[static_cast<size_t>(current.priority)] = true;
    }

    for (auto cursor = committed_cursors.begin(); cursor != committed_cursors.end();) {
        const bool present = std::any_of(candidates.begin(), candidates.end(), [&](const candidate & current) {
            return current.request_id == cursor->first;
        });
        if (!present && (!in_flight || cursor->first != in_flight.request_id)) {
            cursor = committed_cursors.erase(cursor);
        } else {
            ++cursor;
        }
    }

    const candidate * owner_candidate = nullptr;
    if (owner.request_id != 0) {
        const auto found = std::find_if(candidates.begin(), candidates.end(), [&](const candidate & current) {
            return current.request_id == owner.request_id && current.cohort_id == owner.cohort_id;
        });
        if (found != candidates.end()) {
            owner_candidate = &*found;
            owner           = *found;
        } else if (!in_flight) {
            clear_owner();
        }
    }

    if (in_flight) {
        candidate_lanes[static_cast<size_t>(owner.priority)] = true;
        return { true, owner.request_id, owner.cohort_id, owner.priority };
    }

    const candidate * selected = nullptr;
    if (owner_candidate == nullptr) {
        selected = next_weighted_candidate(candidates);
    } else if (at_yield_boundary) {
        const candidate * best = best_candidate(candidates);
        if (best != nullptr && static_cast<uint8_t>(best->priority) > static_cast<uint8_t>(owner.priority)) {
            selected = next_candidate_in_lane(candidates, best->priority);
        } else if (committed_lease_chunks >= cfg.max_lease_chunks) {
            selected = next_weighted_candidate(candidates, owner.cohort_id);
        }
    }

    if (selected != nullptr) {
        owner                  = *selected;
        committed_lease_chunks = 0;
        at_yield_boundary      = true;
        remember_selection(*selected);
    }

    if (owner.request_id == 0) {
        return {};
    }
    return { true, owner.request_id, owner.cohort_id, owner.priority };
}

chunk_limit coordinator::limit_chunk(uint64_t                request_id,
                                     uint64_t                committed_tokens,
                                     uint64_t                prompt_tokens,
                                     uint32_t                available_batch_tokens,
                                     const decode_activity & activity) const {
    if (request_id == 0 || request_id != owner.request_id || in_flight || committed_tokens >= prompt_tokens ||
        available_batch_tokens == 0) {
        return {};
    }

    uint64_t budget = std::min<uint64_t>(available_batch_tokens, cfg.idle_chunk_tokens);
    if (activity.active) {
        // [TAG_PREFILL_PRIORITY] Priority keeps large-M prompt work on the
        // idle chunk budget for every generating lane. The server's
        // prefill-progress cadence bounds decode service independently.
        if (!cfg.prefill_priority) {
            budget = std::min<uint64_t>(budget, cfg.active_decode_chunk_tokens);
            if (activity.highest_lane == server_scheduler::lane::fast) {
                budget = std::min<uint64_t>(budget, cfg.active_fast_chunk_tokens);
            }
        } else if (cfg.priority_chunk_tokens != 0) {
            budget = std::min<uint64_t>(budget, cfg.priority_chunk_tokens);
        }
    }

    const uint64_t remaining = prompt_tokens - committed_tokens;
    uint64_t       end       = committed_tokens + std::min(remaining, budget);
    if (end != prompt_tokens) {
        // Prefer the furthest absolute anchor that fits. If even the next
        // anchor is unreachable, the original bounded endpoint remains so a
        // small llama batch cannot deadlock prefill.
        const uint64_t aligned_end = end - end % cfg.alignment_tokens;
        if (aligned_end > committed_tokens) {
            end = aligned_end;
        }
    }
    return { true, end };
}

chunk_lease coordinator::stage_chunk(uint64_t request_id,
                                     uint64_t begin_token,
                                     uint64_t end_token,
                                     uint64_t prompt_tokens) {
    if (request_id == 0 || request_id != owner.request_id || in_flight || begin_token >= end_token ||
        end_token > prompt_tokens || next_generation == 0) {
        return {};
    }

    const auto cursor = committed_cursors.find(request_id);
    if (cursor != committed_cursors.end() &&
        (cursor->second.cohort_id != owner.cohort_id || cursor->second.token != begin_token)) {
        return {};
    }
    if (cursor == committed_cursors.end()) {
        committed_cursors.emplace(request_id, committed_cursor{ owner.cohort_id, begin_token });
    }

    in_flight = {
        request_id,
        owner.cohort_id,
        next_generation++,
        begin_token,
        end_token,
        end_token == prompt_tokens || end_token % cfg.alignment_tokens == 0,
        end_token == prompt_tokens,
    };
    return in_flight;
}

bool coordinator::commit_chunk(const chunk_lease & lease) {
    if (!in_flight || !(in_flight == lease)) {
        return false;
    }

    const auto cursor = committed_cursors.find(lease.request_id);
    if (cursor == committed_cursors.end() || cursor->second.cohort_id != lease.cohort_id ||
        cursor->second.token != lease.begin_token || !candidate_lanes[static_cast<size_t>(owner.priority)]) {
        return false;
    }

    account_lane_service(owner.priority);
    in_flight            = {};
    cursor->second.token = lease.end_token;
    if (lease.completes_prompt) {
        committed_cursors.erase(cursor);
        clear_owner();
        return true;
    }

    at_yield_boundary = lease.yield_boundary;
    if (lease.yield_boundary && committed_lease_chunks != std::numeric_limits<uint32_t>::max()) {
        ++committed_lease_chunks;
    }
    return true;
}

bool coordinator::abort_chunk(const chunk_lease & lease) {
    if (!in_flight || !(in_flight == lease)) {
        return false;
    }
    in_flight = {};
    return true;
}

bool coordinator::cancel_request(uint64_t request_id) {
    if (request_id == 0 || (in_flight && request_id == in_flight.request_id)) {
        return false;
    }
    bool changed = committed_cursors.erase(request_id) != 0;
    if (request_id == owner.request_id) {
        clear_owner();
        changed = true;
    }
    return changed;
}

bool coordinator::cancel_cohort(uint64_t cohort_id) {
    if (cohort_id == 0 || (in_flight && cohort_id == in_flight.cohort_id)) {
        return false;
    }
    bool changed = false;
    for (auto cursor = committed_cursors.begin(); cursor != committed_cursors.end();) {
        if (cursor->second.cohort_id == cohort_id) {
            cursor  = committed_cursors.erase(cursor);
            changed = true;
        } else {
            ++cursor;
        }
    }
    if (cohort_id == owner.cohort_id) {
        clear_owner();
        changed = true;
    }
    return changed;
}

void coordinator::reset() {
    clear_owner();
    next_generation  = 1;
    fairness_cursors = {};
    lane_credits     = {};
    candidate_lanes  = {};
    committed_cursors.clear();
}

coordinator_snapshot coordinator::snapshot() const {
    return { owner.request_id, owner.cohort_id, committed_lease_chunks, at_yield_boundary,
             static_cast<bool>(in_flight) };
}

const candidate * coordinator::next_candidate_in_lane(const std::vector<candidate> & candidates,
                                                      server_scheduler::lane         priority,
                                                      uint64_t                       excluded_cohort) const {
    const size_t            lane_index = static_cast<size_t>(priority);
    const fairness_cursor & cursor     = fairness_cursors[lane_index];
    const candidate *       oldest     = nullptr;
    const candidate *       after      = nullptr;
    for (const candidate & current : candidates) {
        if (current.cohort_id == excluded_cohort || current.priority != priority) {
            continue;
        }
        if (oldest == nullptr || candidate_better(current, *oldest)) {
            oldest = &current;
        }

        const bool follows_cursor = current.arrival_us > cursor.arrival_us ||
                                    (current.arrival_us == cursor.arrival_us && current.request_id > cursor.request_id);
        if (cursor.valid && follows_cursor && (after == nullptr || candidate_better(current, *after))) {
            after = &current;
        }
    }
    return after != nullptr ? after : oldest;
}

const candidate * coordinator::next_weighted_candidate(const std::vector<candidate> & candidates,
                                                       uint64_t                       excluded_cohort) const {
    server_scheduler::lane selected_lane = server_scheduler::lane::count;
    for (const candidate & current : candidates) {
        if (current.cohort_id == excluded_cohort) {
            continue;
        }
        if (selected_lane == server_scheduler::lane::count) {
            selected_lane = current.priority;
            continue;
        }

        const size_t current_index  = static_cast<size_t>(current.priority);
        const size_t selected_index = static_cast<size_t>(selected_lane);
        if (lane_credits[current_index] > lane_credits[selected_index] ||
            (lane_credits[current_index] == lane_credits[selected_index] &&
             (cfg.lane_weights[current_index] > cfg.lane_weights[selected_index] ||
              (cfg.lane_weights[current_index] == cfg.lane_weights[selected_index] &&
               current_index > selected_index)))) {
            selected_lane = current.priority;
        }
    }
    if (selected_lane == server_scheduler::lane::count) {
        return nullptr;
    }
    return next_candidate_in_lane(candidates, selected_lane, excluded_cohort);
}

void coordinator::account_lane_service(server_scheduler::lane served_lane) {
    int64_t active_weight = 0;
    for (size_t lane_index = 0; lane_index < server_scheduler::lane_count; ++lane_index) {
        if (!candidate_lanes[lane_index]) {
            continue;
        }
        const int64_t weight     = static_cast<int64_t>(cfg.lane_weights[lane_index]);
        active_weight            = saturating_add_signed(active_weight, weight);
        lane_credits[lane_index] = saturating_add_signed(lane_credits[lane_index], weight);
    }

    const size_t served_index  = static_cast<size_t>(served_lane);
    lane_credits[served_index] = saturating_add_signed(lane_credits[served_index], -active_weight);
}

void coordinator::remember_selection(const candidate & selected) {
    fairness_cursors[static_cast<size_t>(selected.priority)] = {
        true,
        selected.arrival_us,
        selected.request_id,
    };
}

void coordinator::clear_owner() {
    owner                  = {};
    in_flight              = {};
    committed_lease_chunks = 0;
    at_yield_boundary      = true;
}

}  // namespace server_prefill

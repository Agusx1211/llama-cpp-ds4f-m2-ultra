#include "server-prefix-cache.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace server_prefix_cache {

namespace {

using token_block = std::array<int32_t, SERVER_PREFIX_BLOCK_TOKENS>;

size_t hash_mix(size_t hash, uint64_t value) {
    // Stable enough for in-process buckets. Exact equality below remains the
    // compatibility authority and is intentionally exercised under collisions.
    hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

size_t hash_bytes(size_t hash, const uint8_t * bytes, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hash = hash_mix(hash, bytes[i]);
    }
    return hash;
}

token_block make_block(const std::vector<int32_t> & tokens, size_t offset) {
    token_block result;
    std::copy_n(tokens.data() + offset, result.size(), result.data());
    return result;
}

bool valid_entry(const server_prefix_entry & entry) {
    return entry.handle_id != 0 && entry.generation != 0 && std::isfinite(entry.saved_prefill_ms) &&
           entry.saved_prefill_ms >= 0.0;
}

bool digest_present(const server_prefix_digest & digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte != 0; });
}

bool valid_identity(const server_prefix_identity & identity) {
    // An all-zero template, draft layout, or LoRA digest explicitly represents
    // an absent optional artifact. The remaining inputs are mandatory.
    return !identity.architecture.empty() && digest_present(identity.model) && digest_present(identity.tokenizer) &&
           digest_present(identity.runtime_build) && digest_present(identity.target_kv_layout) &&
           digest_present(identity.rope) && identity.context_size != 0 && identity.dsv4_state_version != 0;
}

}  // namespace

bool server_prefix_identity::operator==(const server_prefix_identity & other) const {
    return architecture == other.architecture && model == other.model && tokenizer == other.tokenizer &&
           chat_template == other.chat_template && runtime_build == other.runtime_build &&
           target_kv_layout == other.target_kv_layout && draft_kv_layout == other.draft_kv_layout &&
           rope == other.rope && lora == other.lora && context_size == other.context_size &&
           dsv4_state_version == other.dsv4_state_version;
}

bool server_prefix_entry::operator==(const server_prefix_entry & other) const {
    return handle_id == other.handle_id && generation == other.generation && unique_bytes == other.unique_bytes &&
           saved_prefill_ms == other.saved_prefill_ms;
}

bool server_prefix_policy_key::operator==(const server_prefix_policy_key & other) const {
    return digest == other.digest && aligned_tokens == other.aligned_tokens;
}

bool server_prefix_session_key::operator==(const server_prefix_session_key & other) const {
    return identity == other.identity && session_id == other.session_id;
}

bool server_prefix_session_owner::operator==(const server_prefix_session_owner & other) const {
    return handle_id == other.handle_id && generation == other.generation;
}

bool server_prefix_session_version::operator==(const server_prefix_session_version & other) const {
    return owner == other.owner && turn_id == other.turn_id;
}

bool server_prefix_session_anchor::operator==(const server_prefix_session_anchor & other) const {
    return prefix == other.prefix && owner == other.owner && turn_id == other.turn_id &&
           unique_bytes == other.unique_bytes && saved_prefill_ms == other.saved_prefill_ms &&
           message_end_tokens == other.message_end_tokens;
}

bool server_prefix_session_record::operator==(const server_prefix_session_record & other) const {
    return anchor == other.anchor && expires_at_tick == other.expires_at_tick;
}

struct identity_hash {
    bool constant = false;

    size_t operator()(const server_prefix_identity & identity) const {
        if (constant) {
            return 0;
        }
        size_t hash = std::hash<std::string>{}(identity.architecture);
        hash        = hash_bytes(hash, identity.model.data(), identity.model.size());
        hash        = hash_bytes(hash, identity.tokenizer.data(), identity.tokenizer.size());
        hash        = hash_bytes(hash, identity.chat_template.data(), identity.chat_template.size());
        hash        = hash_bytes(hash, identity.runtime_build.data(), identity.runtime_build.size());
        hash        = hash_bytes(hash, identity.target_kv_layout.data(), identity.target_kv_layout.size());
        hash        = hash_bytes(hash, identity.draft_kv_layout.data(), identity.draft_kv_layout.size());
        hash        = hash_bytes(hash, identity.rope.data(), identity.rope.size());
        hash        = hash_bytes(hash, identity.lora.data(), identity.lora.size());
        hash        = hash_mix(hash, identity.context_size);
        return hash_mix(hash, identity.dsv4_state_version);
    }
};

struct token_block_hash {
    bool constant = false;

    size_t operator()(const token_block & block) const {
        if (constant) {
            return 0;
        }
        size_t hash = 0;
        for (int32_t token : block) {
            hash = hash_mix(hash, static_cast<uint32_t>(token));
        }
        return hash;
    }
};

constexpr size_t POLICY_SKETCH_DEPTH = 4;

static size_t policy_key_hash_value(const server_prefix_policy_key & key, uint64_t salt, bool constant) {
    if (constant) {
        return 0;
    }
    size_t hash = hash_mix(static_cast<size_t>(salt), key.aligned_tokens);
    return hash_bytes(hash, key.digest.data(), key.digest.size());
}

struct policy_key_hash {
    bool constant = false;

    size_t operator()(const server_prefix_policy_key & key) const {
        return policy_key_hash_value(key, 0xd6e8feb86659fd93ULL, constant);
    }
};

static bool policy_key_less(const server_prefix_policy_key & lhs, const server_prefix_policy_key & rhs) {
    if (lhs.digest != rhs.digest) {
        return lhs.digest < rhs.digest;
    }
    return lhs.aligned_tokens < rhs.aligned_tokens;
}

static bool valid_policy_key(const server_prefix_policy_key & key) {
    return digest_present(key.digest) && key.aligned_tokens != 0 &&
           key.aligned_tokens % SERVER_PREFIX_BLOCK_TOKENS == 0;
}

static bool valid_policy_candidate(const server_prefix_policy_candidate & candidate) {
    return valid_policy_key(candidate.key) && candidate.unique_bytes != 0 &&
           std::isfinite(candidate.prefill_ms_avoided) && candidate.prefill_ms_avoided >= 0.0 &&
           std::isfinite(candidate.restore_ms) && candidate.restore_ms >= 0.0;
}

static double policy_value(uint16_t frequency, double prefill_ms_avoided, double restore_ms, uint64_t unique_bytes) {
    const double benefit = std::max(0.0, prefill_ms_avoided - restore_ms);
    return static_cast<double>(frequency) * benefit / static_cast<double>(unique_bytes);
}

struct policy_ghost_slot {
    server_prefix_policy_key key;
    uint64_t                 seen_at  = 0;
    bool                     occupied = false;
};

struct policy_resident_record {
    server_prefix_policy_segment segment;
    uint64_t                     unique_bytes;
    double                       prefill_ms_avoided;
    double                       restore_ms;
    uint16_t                     frequency;
    double                       priority;
    uint64_t                     recency;
};

struct policy_victim {
    server_prefix_policy_key     key;
    server_prefix_policy_segment segment;
    uint64_t                     unique_bytes;
    double                       priority;
    uint64_t                     recency;
};

static bool policy_victim_less(const policy_victim & lhs, const policy_victim & rhs) {
    if (lhs.segment != rhs.segment) {
        return lhs.segment == server_prefix_policy_segment::probation;
    }
    if (lhs.priority != rhs.priority) {
        return lhs.priority < rhs.priority;
    }
    if (lhs.recency != rhs.recency) {
        return lhs.recency < rhs.recency;
    }
    return policy_key_less(lhs.key, rhs.key);
}

static bool valid_policy_config(const server_prefix_policy_config & config) {
    if (config.max_resident_entries == 0 || config.max_resident_bytes == 0 ||
        config.max_protected_bytes > config.max_resident_bytes ||
        config.max_pinned_entries > config.max_resident_entries ||
        config.max_pinned_bytes > config.max_resident_bytes || config.ghost_buckets == 0 || config.ghost_ways == 0 ||
        config.sketch_width == 0 || config.decay_interval == 0 || !std::isfinite(config.hysteresis_fraction) ||
        config.hysteresis_fraction < 0.0) {
        return false;
    }
    if (config.ghost_buckets > std::numeric_limits<size_t>::max() / config.ghost_ways) {
        return false;
    }
    const size_t ghost_cells = config.ghost_buckets * config.ghost_ways;
    return ghost_cells <= std::vector<policy_ghost_slot>().max_size() &&
           config.sketch_width <= std::vector<uint16_t>().max_size() / POLICY_SKETCH_DEPTH;
}

struct server_prefix_policy::impl {
    explicit impl(server_prefix_policy_config config) :
        cfg(config),
        residents(0, policy_key_hash{ config.force_hash_collisions }),
        sketch(config.sketch_width * POLICY_SKETCH_DEPTH),
        ghosts(config.ghost_buckets * config.ghost_ways),
        ghost_hands(config.ghost_buckets) {}

    void decay_if_due() {
        if (observations == 0 || observations % cfg.decay_interval != 0) {
            return;
        }
        for (uint16_t & counter : sketch) {
            counter = static_cast<uint16_t>(counter >> 1);
        }
        ++decays;
    }

    uint16_t record_frequency(const server_prefix_policy_key & key) {
        static constexpr uint64_t salts[POLICY_SKETCH_DEPTH] = {
            0x9e3779b97f4a7c15ULL,
            0xbf58476d1ce4e5b9ULL,
            0x94d049bb133111ebULL,
            0xd6e8feb86659fd93ULL,
        };
        uint16_t estimate = std::numeric_limits<uint16_t>::max();
        for (size_t depth = 0; depth < POLICY_SKETCH_DEPTH; ++depth) {
            const size_t column =
                policy_key_hash_value(key, salts[depth], cfg.force_hash_collisions) % cfg.sketch_width;
            uint16_t & counter = sketch[depth * cfg.sketch_width + column];
            if (counter != std::numeric_limits<uint16_t>::max()) {
                ++counter;
            }
            estimate = std::min(estimate, counter);
        }
        return estimate;
    }

    size_t ghost_bucket(const server_prefix_policy_key & key) const {
        return policy_key_hash_value(key, 0xa0761d6478bd642fULL, cfg.force_hash_collisions) % cfg.ghost_buckets;
    }

    bool take_fresh_ghost(const server_prefix_policy_key & key) {
        const size_t bucket = ghost_bucket(key);
        const size_t begin  = bucket * cfg.ghost_ways;
        for (size_t way = 0; way < cfg.ghost_ways; ++way) {
            policy_ghost_slot & slot = ghosts[begin + way];
            if (!slot.occupied || !(slot.key == key)) {
                continue;
            }
            const bool fresh = observations - slot.seen_at <= cfg.decay_interval;
            if (fresh) {
                slot.occupied = false;
                --ghost_entries;
                return true;
            }
            slot.seen_at = observations;
            return false;
        }
        return false;
    }

    void remember_ghost(const server_prefix_policy_key & key) {
        const size_t bucket = ghost_bucket(key);
        const size_t begin  = bucket * cfg.ghost_ways;
        for (size_t way = 0; way < cfg.ghost_ways; ++way) {
            policy_ghost_slot & slot = ghosts[begin + way];
            if (slot.occupied && slot.key == key) {
                slot.seen_at = observations;
                return;
            }
            if (!slot.occupied) {
                slot = { key, observations, true };
                ++ghost_entries;
                return;
            }
        }
        const size_t replacement    = ghost_hands[bucket];
        ghosts[begin + replacement] = { key, observations, true };
        ghost_hands[bucket]         = (replacement + 1) % cfg.ghost_ways;
    }

    void remove_resident(const server_prefix_policy_key & key) {
        const auto found = residents.find(key);
        if (found == residents.end()) {
            return;
        }
        const policy_resident_record & record = found->second;
        resident_bytes -= record.unique_bytes;
        if (record.segment == server_prefix_policy_segment::protected_segment) {
            protected_bytes -= record.unique_bytes;
        } else if (record.segment == server_prefix_policy_segment::pinned) {
            pinned_bytes -= record.unique_bytes;
            --pinned_entries;
        }
        residents.erase(found);
    }

    void enforce_protected_budget() {
        while (protected_bytes > cfg.max_protected_bytes) {
            auto oldest = residents.end();
            for (auto it = residents.begin(); it != residents.end(); ++it) {
                if (it->second.segment != server_prefix_policy_segment::protected_segment) {
                    continue;
                }
                if (oldest == residents.end() || it->second.recency < oldest->second.recency ||
                    (it->second.recency == oldest->second.recency && policy_key_less(it->first, oldest->first))) {
                    oldest = it;
                }
            }
            if (oldest == residents.end()) {
                break;
            }
            oldest->second.segment = server_prefix_policy_segment::probation;
            protected_bytes -= oldest->second.unique_bytes;
        }
    }

    server_prefix_policy_config                                                           cfg;
    std::unordered_map<server_prefix_policy_key, policy_resident_record, policy_key_hash> residents;
    std::vector<uint16_t>                                                                 sketch;
    std::vector<policy_ghost_slot>                                                        ghosts;
    std::vector<size_t>                                                                   ghost_hands;
    uint64_t                                                                              resident_bytes  = 0;
    uint64_t                                                                              protected_bytes = 0;
    uint64_t                                                                              pinned_bytes    = 0;
    size_t                                                                                pinned_entries  = 0;
    size_t                                                                                ghost_entries   = 0;
    uint64_t                                                                              observations    = 0;
    uint64_t                                                                              decays          = 0;
    double                                                                                aging           = 0.0;
};

server_prefix_policy::server_prefix_policy(server_prefix_policy_config config) {
    if (!valid_policy_config(config)) {
        throw std::invalid_argument("invalid server prefix policy configuration");
    }
    data = std::make_unique<impl>(config);
}

server_prefix_policy::~server_prefix_policy() = default;

server_prefix_policy_result server_prefix_policy::observe(const server_prefix_policy_candidate & candidate) {
    server_prefix_policy_result result;
    if (!valid_policy_candidate(candidate)) {
        return result;
    }

    auto resident = data->residents.find(candidate.key);
    if (resident != data->residents.end() && resident->second.unique_bytes != candidate.unique_bytes) {
        return result;
    }

    data->decay_if_due();
    ++data->observations;
    const uint16_t estimate    = data->record_frequency(candidate.key);
    result.estimated_frequency = estimate;

    if (resident != data->residents.end()) {
        policy_resident_record & record = resident->second;
        if (candidate.pin && record.segment != server_prefix_policy_segment::pinned) {
            if (data->pinned_entries >= data->cfg.max_pinned_entries ||
                record.unique_bytes > data->cfg.max_pinned_bytes - data->pinned_bytes) {
                result.status   = server_prefix_policy_status::rejected_pin_quota;
                result.priority = record.priority;
                return result;
            }
            if (record.segment == server_prefix_policy_segment::protected_segment) {
                data->protected_bytes -= record.unique_bytes;
            }
            record.segment = server_prefix_policy_segment::pinned;
            data->pinned_bytes += record.unique_bytes;
            ++data->pinned_entries;
        } else if (record.segment == server_prefix_policy_segment::probation) {
            record.segment = server_prefix_policy_segment::protected_segment;
            data->protected_bytes += record.unique_bytes;
        }
        record.frequency          = estimate;
        record.prefill_ms_avoided = candidate.prefill_ms_avoided;
        record.restore_ms         = candidate.restore_ms;
        record.priority =
            data->aging + policy_value(estimate, record.prefill_ms_avoided, record.restore_ms, record.unique_bytes);
        record.recency = data->observations;
        data->enforce_protected_budget();
        result.status   = server_prefix_policy_status::resident_hit;
        result.priority = record.priority;
        return result;
    }

    if (candidate.pin) {
        if (data->pinned_entries >= data->cfg.max_pinned_entries ||
            candidate.unique_bytes > data->cfg.max_pinned_bytes - data->pinned_bytes) {
            result.status = server_prefix_policy_status::rejected_pin_quota;
            return result;
        }
    } else if (!data->take_fresh_ghost(candidate.key)) {
        data->remember_ghost(candidate.key);
        result.status = server_prefix_policy_status::first_sighting;
        return result;
    }

    if (candidate.unique_bytes > data->cfg.max_resident_bytes) {
        if (!candidate.pin) {
            data->remember_ghost(candidate.key);
        }
        result.status = server_prefix_policy_status::rejected_capacity;
        return result;
    }

    const uint16_t admission_frequency = candidate.pin ? estimate : std::max<uint16_t>(2, estimate);
    const double   candidate_priority  = data->aging + policy_value(admission_frequency, candidate.prefill_ms_avoided,
                                                                    candidate.restore_ms, candidate.unique_bytes);
    result.priority                    = candidate_priority;

    std::vector<policy_victim> victims;
    size_t                     victim_count = 0;
    if (data->residents.size() >= data->cfg.max_resident_entries ||
        candidate.unique_bytes > data->cfg.max_resident_bytes - data->resident_bytes) {
        victims.reserve(data->residents.size());
        for (const auto & item : data->residents) {
            if (item.second.segment == server_prefix_policy_segment::pinned) {
                continue;
            }
            victims.push_back({ item.first, item.second.segment, item.second.unique_bytes, item.second.priority,
                                item.second.recency });
        }
        // SLRU retention makes probation the first victim class. GreedyDual
        // cost/size priority and deterministic recency/key ties order each
        // class. Explicit pins may cross hysteresis, but never their quotas.
        std::sort(victims.begin(), victims.end(), policy_victim_less);

        size_t   retained_entries = data->residents.size();
        uint64_t retained_bytes   = data->resident_bytes;
        while (retained_entries >= data->cfg.max_resident_entries ||
               candidate.unique_bytes > data->cfg.max_resident_bytes - retained_bytes) {
            if (victim_count == victims.size()) {
                if (!candidate.pin) {
                    data->remember_ghost(candidate.key);
                }
                result.status = server_prefix_policy_status::rejected_capacity;
                return result;
            }
            const policy_victim & victim = victims[victim_count];
            if (!candidate.pin && candidate_priority <= victim.priority * (1.0 + data->cfg.hysteresis_fraction)) {
                data->remember_ghost(candidate.key);
                result.status = server_prefix_policy_status::rejected_hysteresis;
                return result;
            }
            --retained_entries;
            retained_bytes -= victim.unique_bytes;
            ++victim_count;
        }
    }

    for (size_t i = 0; i < victim_count; ++i) {
        data->aging = std::max(data->aging, victims[i].priority);
        result.evicted.push_back(victims[i].key);
        data->remove_resident(victims[i].key);
    }

    const server_prefix_policy_segment segment =
        candidate.pin ? server_prefix_policy_segment::pinned : server_prefix_policy_segment::probation;
    const double final_priority = data->aging + policy_value(admission_frequency, candidate.prefill_ms_avoided,
                                                             candidate.restore_ms, candidate.unique_bytes);
    data->residents.emplace(
        candidate.key,
        policy_resident_record{ segment, candidate.unique_bytes, candidate.prefill_ms_avoided, candidate.restore_ms,
                                admission_frequency, final_priority, data->observations });
    data->resident_bytes += candidate.unique_bytes;
    if (candidate.pin) {
        data->pinned_bytes += candidate.unique_bytes;
        ++data->pinned_entries;
        result.status = server_prefix_policy_status::admitted_pinned;
    } else {
        result.status = server_prefix_policy_status::admitted_probation;
    }
    result.priority = final_priority;
    return result;
}

std::optional<server_prefix_policy_resident> server_prefix_policy::resident(
    const server_prefix_policy_key & key) const {
    const auto found = data->residents.find(key);
    if (found == data->residents.end()) {
        return std::nullopt;
    }
    const policy_resident_record & record = found->second;
    return server_prefix_policy_resident{ record.segment,    record.unique_bytes, record.prefill_ms_avoided,
                                          record.restore_ms, record.frequency,    record.priority };
}

bool server_prefix_policy::erase(const server_prefix_policy_key & key) {
    if (data->residents.find(key) == data->residents.end()) {
        return false;
    }
    data->remove_resident(key);
    return true;
}

void server_prefix_policy::clear() {
    data->residents.clear();
    std::fill(data->sketch.begin(), data->sketch.end(), 0);
    std::fill(data->ghosts.begin(), data->ghosts.end(), policy_ghost_slot{});
    std::fill(data->ghost_hands.begin(), data->ghost_hands.end(), 0);
    data->resident_bytes  = 0;
    data->protected_bytes = 0;
    data->pinned_bytes    = 0;
    data->pinned_entries  = 0;
    data->ghost_entries   = 0;
    data->observations    = 0;
    data->decays          = 0;
    data->aging           = 0.0;
}

server_prefix_policy_stats server_prefix_policy::stats() const {
    server_prefix_policy_stats result;
    result.resident_entries     = data->residents.size();
    result.resident_bytes       = data->resident_bytes;
    result.pinned_entries       = data->pinned_entries;
    result.pinned_bytes         = data->pinned_bytes;
    result.ghost_entries        = data->ghost_entries;
    result.ghost_capacity       = data->ghosts.size();
    result.ghost_bytes          = data->ghosts.size() * sizeof(policy_ghost_slot);
    result.sketch_cells         = data->sketch.size();
    result.sketch_bytes         = data->sketch.size() * sizeof(uint16_t);
    result.fixed_metadata_bytes = result.ghost_bytes + result.sketch_bytes + data->ghost_hands.size() * sizeof(size_t);
    result.observations         = data->observations;
    result.decays               = data->decays;
    for (const auto & item : data->residents) {
        if (item.second.segment == server_prefix_policy_segment::probation) {
            ++result.probation_entries;
            result.probation_bytes += item.second.unique_bytes;
        } else if (item.second.segment == server_prefix_policy_segment::protected_segment) {
            ++result.protected_entries;
            result.protected_bytes += item.second.unique_bytes;
        }
    }
    return result;
}

struct session_key_hash {
    bool constant = false;

    size_t operator()(const server_prefix_session_key & key) const {
        if (constant) {
            return 0;
        }
        size_t hash = identity_hash{}(key.identity);
        return hash_mix(hash, std::hash<std::string>{}(key.session_id));
    }
};

static bool session_key_less(const server_prefix_session_key & lhs, const server_prefix_session_key & rhs) {
    return std::tie(lhs.identity.architecture, lhs.identity.model, lhs.identity.tokenizer, lhs.identity.chat_template,
                    lhs.identity.runtime_build, lhs.identity.target_kv_layout, lhs.identity.draft_kv_layout,
                    lhs.identity.rope, lhs.identity.lora, lhs.identity.context_size, lhs.identity.dsv4_state_version,
                    lhs.session_id) < std::tie(rhs.identity.architecture, rhs.identity.model, rhs.identity.tokenizer,
                                               rhs.identity.chat_template, rhs.identity.runtime_build,
                                               rhs.identity.target_kv_layout, rhs.identity.draft_kv_layout,
                                               rhs.identity.rope, rhs.identity.lora, rhs.identity.context_size,
                                               rhs.identity.dsv4_state_version, rhs.session_id);
}

static bool valid_session_config(const server_prefix_session_config & config) {
    return config.max_sessions != 0 && config.max_session_id_bytes != 0 && config.max_architecture_bytes != 0 &&
           config.ttl_ticks != 0;
}

static bool valid_session_update(const server_prefix_session_update & update,
                                 const server_prefix_session_config & config) {
    const auto & key    = update.key;
    const auto & anchor = update.anchor;
    return valid_identity(key.identity) && !key.session_id.empty() &&
           key.session_id.size() <= config.max_session_id_bytes &&
           key.identity.architecture.size() <= config.max_architecture_bytes && valid_policy_key(anchor.prefix) &&
           anchor.prefix.aligned_tokens == anchor.message_end_tokens &&
           anchor.message_end_tokens <= key.identity.context_size && anchor.owner.handle_id != 0 &&
           anchor.owner.generation != 0 && anchor.turn_id != 0 && anchor.unique_bytes != 0 &&
           std::isfinite(anchor.saved_prefill_ms) && anchor.saved_prefill_ms > 0.0 &&
           update.now_tick <= std::numeric_limits<uint64_t>::max() - config.ttl_ticks;
}

static server_prefix_session_version session_version(const server_prefix_session_anchor & anchor) {
    return { anchor.owner, anchor.turn_id };
}

struct server_prefix_session_cache::impl {
    explicit impl(server_prefix_session_config config) :
        cfg(config),
        sessions(0, session_key_hash{ config.force_hash_collisions }) {}

    server_prefix_session_config                                                                  cfg;
    std::unordered_map<server_prefix_session_key, server_prefix_session_record, session_key_hash> sessions;
    size_t                                                                                        peak_entries = 0;
    uint64_t                                                                                      insertions   = 0;
    uint64_t                                                                                      replacements = 0;
    uint64_t                                                                                      expirations  = 0;
    uint64_t                                                                                      erases       = 0;
};

server_prefix_session_cache::server_prefix_session_cache(server_prefix_session_config config) {
    if (!valid_session_config(config)) {
        throw std::invalid_argument("invalid server prefix session configuration");
    }
    data = std::make_unique<impl>(config);
}

server_prefix_session_cache::~server_prefix_session_cache() = default;

server_prefix_session_result server_prefix_session_cache::replace(const server_prefix_session_update & update) {
    server_prefix_session_result result;
    if (!valid_session_update(update, data->cfg)) {
        return result;
    }

    const uint64_t expires_at = update.now_tick + data->cfg.ttl_ticks;
    auto           current    = data->sessions.find(update.key);
    if (current == data->sessions.end()) {
        if (update.expected_current.has_value()) {
            result.status = server_prefix_session_status::conflict;
            return result;
        }
        if (data->sessions.size() >= data->cfg.max_sessions) {
            result.status = server_prefix_session_status::capacity;
            return result;
        }
        data->sessions.emplace(update.key, server_prefix_session_record{ update.anchor, expires_at });
        ++data->insertions;
        data->peak_entries     = std::max(data->peak_entries, data->sessions.size());
        result.status          = server_prefix_session_status::inserted;
        result.expires_at_tick = expires_at;
        return result;
    }

    const server_prefix_session_record & old        = current->second;
    const bool                           old_active = update.now_tick < old.expires_at_tick;
    if (old_active) {
        if (!update.expected_current.has_value()) {
            result.status = server_prefix_session_status::conflict;
            return result;
        }
        const auto & expected = update.expected_current.value();
        if (expected.owner.handle_id == old.anchor.owner.handle_id &&
            expected.owner.generation != old.anchor.owner.generation) {
            result.status = server_prefix_session_status::stale_generation;
            return result;
        }
        if (!(expected == session_version(old.anchor))) {
            result.status = server_prefix_session_status::conflict;
            return result;
        }
    } else if (update.expected_current.has_value()) {
        result.status = server_prefix_session_status::conflict;
        return result;
    }

    if (update.anchor.turn_id <= old.anchor.turn_id) {
        result.status = server_prefix_session_status::stale_turn;
        return result;
    }
    if (update.anchor.owner.handle_id == old.anchor.owner.handle_id &&
        update.anchor.owner.generation <= old.anchor.owner.generation) {
        result.status = server_prefix_session_status::stale_generation;
        return result;
    }

    result.displaced       = old;
    current->second        = { update.anchor, expires_at };
    result.status          = server_prefix_session_status::replaced;
    result.expires_at_tick = expires_at;
    ++data->replacements;
    return result;
}

std::optional<server_prefix_session_record> server_prefix_session_cache::lookup(const server_prefix_session_key & key,
                                                                                uint64_t now_tick) const {
    const auto found = data->sessions.find(key);
    if (found == data->sessions.end() || now_tick >= found->second.expires_at_tick) {
        return std::nullopt;
    }
    return found->second;
}

std::optional<server_prefix_session_record> server_prefix_session_cache::erase(
    const server_prefix_session_key &     key,
    const server_prefix_session_version & expected) {
    const auto found = data->sessions.find(key);
    if (found == data->sessions.end() || !(expected == session_version(found->second.anchor))) {
        return std::nullopt;
    }
    const server_prefix_session_record removed = found->second;
    data->sessions.erase(found);
    ++data->erases;
    return removed;
}

std::vector<server_prefix_session_expired> server_prefix_session_cache::expire(uint64_t now_tick) {
    std::vector<server_prefix_session_expired> result;
    for (auto it = data->sessions.begin(); it != data->sessions.end();) {
        if (now_tick < it->second.expires_at_tick) {
            ++it;
            continue;
        }
        result.push_back({ it->first, it->second });
        it = data->sessions.erase(it);
    }
    std::sort(result.begin(), result.end(),
              [](const auto & lhs, const auto & rhs) { return session_key_less(lhs.key, rhs.key); });
    data->expirations += result.size();
    return result;
}

void server_prefix_session_cache::clear() {
    data->sessions.clear();
    data->peak_entries = 0;
    data->insertions   = 0;
    data->replacements = 0;
    data->expirations  = 0;
    data->erases       = 0;
}

server_prefix_session_stats server_prefix_session_cache::stats(uint64_t now_tick) const {
    server_prefix_session_stats result;
    result.entries      = data->sessions.size();
    result.peak_entries = data->peak_entries;
    result.insertions   = data->insertions;
    result.replacements = data->replacements;
    result.expirations  = data->expirations;
    result.erases       = data->erases;
    for (const auto & item : data->sessions) {
        if (now_tick < item.second.expires_at_tick) {
            ++result.active;
        } else {
            ++result.expired;
        }
    }
    return result;
}

struct radix_node {
    explicit radix_node(bool force_collisions) : children(0, token_block_hash{ force_collisions }) {}

    std::unordered_map<token_block, std::unique_ptr<radix_node>, token_block_hash> children;
    std::optional<server_prefix_entry>                                             entry;
};

struct server_prefix_radix::impl {
    explicit impl(server_prefix_radix_config config) :
        cfg(config),
        roots(0, identity_hash{ config.force_hash_collisions }) {}

    server_prefix_radix_config                                                             cfg;
    std::unordered_map<server_prefix_identity, std::unique_ptr<radix_node>, identity_hash> roots;
    size_t                                                                                 node_count  = 0;
    size_t                                                                                 entry_count = 0;
};

server_prefix_radix::server_prefix_radix(server_prefix_radix_config config) : data(std::make_unique<impl>(config)) {}

server_prefix_radix::~server_prefix_radix() = default;

server_prefix_insert_status server_prefix_radix::insert(const server_prefix_identity & identity,
                                                        const std::vector<int32_t> &   tokens,
                                                        const server_prefix_entry &    entry) {
    auto & cfg         = data->cfg;
    auto & roots       = data->roots;
    auto & node_count  = data->node_count;
    auto & entry_count = data->entry_count;
    if (!valid_identity(identity) || !valid_entry(entry) || tokens.empty() ||
        tokens.size() % SERVER_PREFIX_BLOCK_TOKENS != 0 || tokens.size() > identity.context_size) {
        return server_prefix_insert_status::invalid_argument;
    }

    const auto         root_found    = roots.find(identity);
    const radix_node * cursor        = root_found == roots.end() ? nullptr : root_found->second.get();
    size_t             missing_nodes = 0;
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        if (cursor == nullptr) {
            ++missing_nodes;
            continue;
        }
        const auto child = cursor->children.find(make_block(tokens, offset));
        if (child == cursor->children.end()) {
            cursor = nullptr;
            ++missing_nodes;
        } else {
            cursor = child->second.get();
        }
    }

    if (cursor != nullptr && cursor->entry.has_value()) {
        return cursor->entry.value() == entry ? server_prefix_insert_status::already_present :
                                                server_prefix_insert_status::conflict;
    }
    if (missing_nodes > cfg.max_nodes - std::min(cfg.max_nodes, node_count) || entry_count >= cfg.max_entries) {
        return server_prefix_insert_status::capacity;
    }

    auto         root           = roots.try_emplace(identity, std::make_unique<radix_node>(cfg.force_hash_collisions));
    radix_node * mutable_cursor = root.first->second.get();
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const token_block block = make_block(tokens, offset);
        auto              child = mutable_cursor->children.find(block);
        if (child == mutable_cursor->children.end()) {
            auto inserted =
                mutable_cursor->children.emplace(block, std::make_unique<radix_node>(cfg.force_hash_collisions));
            child = inserted.first;
            ++node_count;
        }
        mutable_cursor = child->second.get();
    }
    mutable_cursor->entry = entry;
    ++entry_count;
    return server_prefix_insert_status::inserted;
}

std::optional<server_prefix_match> server_prefix_radix::lookup(const server_prefix_identity & identity,
                                                               const std::vector<int32_t> &   tokens) const {
    const auto & roots = data->roots;
    const auto   root  = roots.find(identity);
    if (root == roots.end()) {
        return std::nullopt;
    }

    const radix_node *                 cursor = root->second.get();
    std::optional<server_prefix_match> result;
    const size_t                       complete_tokens = tokens.size() - tokens.size() % SERVER_PREFIX_BLOCK_TOKENS;
    for (size_t offset = 0; offset < complete_tokens; offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const auto child = cursor->children.find(make_block(tokens, offset));
        if (child == cursor->children.end()) {
            break;
        }
        cursor = child->second.get();
        if (cursor->entry.has_value()) {
            result = server_prefix_match{ cursor->entry.value(), offset + SERVER_PREFIX_BLOCK_TOKENS };
        }
    }
    return result;
}

bool server_prefix_radix::erase(const server_prefix_identity & identity,
                                const std::vector<int32_t> &   tokens,
                                uint64_t                       expected_handle_id,
                                uint64_t                       expected_generation) {
    auto & roots       = data->roots;
    auto & node_count  = data->node_count;
    auto & entry_count = data->entry_count;
    if (tokens.empty() || tokens.size() % SERVER_PREFIX_BLOCK_TOKENS != 0) {
        return false;
    }
    const auto root = roots.find(identity);
    if (root == roots.end()) {
        return false;
    }

    std::vector<std::pair<radix_node *, token_block>> path;
    radix_node *                                      cursor = root->second.get();
    for (size_t offset = 0; offset < tokens.size(); offset += SERVER_PREFIX_BLOCK_TOKENS) {
        const token_block block = make_block(tokens, offset);
        const auto        child = cursor->children.find(block);
        if (child == cursor->children.end()) {
            return false;
        }
        path.emplace_back(cursor, block);
        cursor = child->second.get();
    }
    if (!cursor->entry.has_value() || cursor->entry->handle_id != expected_handle_id ||
        cursor->entry->generation != expected_generation) {
        return false;
    }
    cursor->entry.reset();
    --entry_count;

    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        radix_node * parent = it->first;
        const auto   child  = parent->children.find(it->second);
        if (child == parent->children.end() || child->second->entry.has_value() || !child->second->children.empty()) {
            break;
        }
        parent->children.erase(child);
        --node_count;
    }
    if (root->second->children.empty() && !root->second->entry.has_value()) {
        roots.erase(root);
    }
    return true;
}

void server_prefix_radix::clear() {
    data->roots.clear();
    data->node_count  = 0;
    data->entry_count = 0;
}

server_prefix_radix_stats server_prefix_radix::stats() const {
    return { data->roots.size(), data->node_count, data->entry_count };
}

}  // namespace server_prefix_cache

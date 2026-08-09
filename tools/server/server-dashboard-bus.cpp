#include "server-dashboard-bus.h"

#include "ggml.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

// m2-dashboard event bus implementation. Serialization lives here so the SSE
// handler, the cache-state handler, and the host tests share one source of
// field names for schema v1.

namespace server_dashboard {

// bus ----------------------------------------------------------------------

bus::bus(size_t capacity) {
    if (capacity == 0) {
        capacity = 1;
    }
    ring.resize(capacity);
}

void bus::emit(event e) noexcept {
    if (e.at_us == 0) {
        e.at_us = (uint64_t) ggml_time_us();
    }
    std::lock_guard<std::mutex> lock(mtx);
    e.seq = next++;
    ring[(e.seq - 1) % ring.size()] = e;
    if (count < ring.size()) {
        ++count;
    } else {
        ++dropped;
    }
}

ring_snapshot bus::snapshot_after(uint64_t after_seq, size_t max_events) const {
    ring_snapshot out;
    std::lock_guard<std::mutex> lock(mtx);
    out.next_seq  = next;
    out.dropped   = dropped;
    out.first_seq = next - count;
    if (after_seq >= next) {
        // future cursor (e.g. from a previous server run): restart from the
        // oldest retained event instead of overflowing begin below
        after_seq = out.first_seq - 1;
    }
    uint64_t begin = after_seq + 1;
    if (begin < out.first_seq) {
        begin = out.first_seq;
    }
    if (begin >= next) {
        return out;
    }
    uint64_t n = next - begin;
    if (n > max_events) {
        n = max_events; // max_events == 0 returns the window stats only
    }
    out.events.reserve(n);
    for (uint64_t seq = begin; seq < begin + n; ++seq) {
        out.events.push_back(ring[(seq - 1) % ring.size()]);
    }
    return out;
}

uint64_t bus::next_seq() const {
    std::lock_guard<std::mutex> lock(mtx);
    return next;
}

void bus::publish_cache_state(cache_state && state) {
    if (state.at_us == 0) {
        state.at_us = (uint64_t) ggml_time_us();
    }
    std::lock_guard<std::mutex> lock(cache_mtx);
    cache_snapshot = std::move(state);
}

cache_state bus::cache() const {
    std::lock_guard<std::mutex> lock(cache_mtx);
    return cache_snapshot;
}

bus & instance() {
    static bus the_bus;
    return the_bus;
}

// wall-clock mapping --------------------------------------------------------

uint64_t wall_offset_ms() {
    // one stable offset per process: wall_ms(at_us) = at_us/1000 + offset
    static const uint64_t offset = [] {
        const uint64_t wall_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        const uint64_t mono_ms = (uint64_t) (ggml_time_us() / 1000);
        return wall_ms - mono_ms;
    }();
    return offset;
}

// serialization -------------------------------------------------------------

const char * kind_name(event_kind kind) {
    switch (kind) {
        case event_kind::queued:           return "queued";
        case event_kind::deferred:         return "deferred";
        case event_kind::dispatched:       return "dispatched";
        case event_kind::slot_selected:    return "slot_selected";
        case event_kind::admission:        return "admission";
        case event_kind::cache_restore:    return "cache_restore";
        case event_kind::prompt_start:     return "prompt_start";
        case event_kind::prefill_progress: return "prefill_progress";
        case event_kind::decode_progress:  return "decode_progress";
        case event_kind::finished:         return "finished";
        case event_kind::terminal:         return "terminal";
        case event_kind::cache_op:         return "cache_op";
    }
    return "unknown";
}

static const char * defer_name(uint16_t code) {
    switch ((defer_code) code) {
        case defer_code::capacity:           return "capacity";
        case defer_code::physical_admission: return "physical_admission";
    }
    return "unknown";
}

static const char * select_name(uint16_t code) {
    switch ((select_code) code) {
        case select_code::by_id:        return "by_id";
        case select_code::lcp_affinity: return "lcp_affinity";
        case select_code::lru:          return "lru";
    }
    return "unknown";
}

static const char * admission_name(uint16_t code) {
    switch ((admission_code) code) {
        case admission_code::ready_full_clear: return "ready_full_clear";
        case admission_code::ready_reuse:      return "ready_reuse";
        case admission_code::ready_spanless:   return "ready_spanless";
        case admission_code::deferred:         return "deferred";
        case admission_code::rejected:         return "rejected";
    }
    return "unknown";
}

static const char * restore_name(uint16_t code) {
    switch ((restore_code) code) {
        case restore_code::resident: return "resident";
        case restore_code::ram:      return "ram";
        case restore_code::disk:     return "disk";
        case restore_code::donor:    return "donor";
        case restore_code::miss:     return "miss";
    }
    return "unknown";
}

static const char * span_name(uint16_t code) {
    switch ((span_code) code) {
        case span_code::src_resident:             return "resident";
        case span_code::src_cache_ram:            return "cache_ram";
        case span_code::src_cache_disk:           return "cache_disk";
        case span_code::src_donor:                return "donor";
        case span_code::recompute_no_cache:       return "no_cache";
        case span_code::recompute_checkpoint_gap: return "checkpoint_gap";
        case span_code::recompute_reset:          return "reset";
        case span_code::recompute_divergent_tail: return "divergent_tail";
        case span_code::recompute_last_token:     return "last_token";
    }
    return "unknown";
}

static const char * clear_name(int64_t code) {
    switch ((clear_code) code) {
        case clear_code::empty_slot:             return "empty_slot";
        case clear_code::family:                 return "family";
        case clear_code::no_lcp:                 return "no_lcp";
        case clear_code::lcp_covers_task:        return "lcp_covers_task";
        case clear_code::mtmd:                   return "mtmd";
        case clear_code::cache_reuse_opt:        return "cache_reuse_opt";
        case clear_code::no_cache_prompt:        return "no_cache_prompt";
        case clear_code::no_pos_min:             return "no_pos_min";
        case clear_code::no_covering_checkpoint: return "no_covering_checkpoint";
        case clear_code::rs_window:              return "rs_window";
    }
    return "unknown";
}

static const char * stop_name(uint16_t code) {
    switch ((stop_code) code) {
        case stop_code::eos:     return "eos";
        case stop_code::limit:   return "limit";
        case stop_code::word:    return "word";
        case stop_code::error:   return "error";
        case stop_code::aborted: return "aborted";
    }
    return "unknown";
}

static const char * cache_op_name(uint16_t code) {
    switch ((cache_op_code) code) {
        case cache_op_code::save:      return "save";
        case cache_op_code::spill:     return "spill";
        case cache_op_code::disk_load: return "disk_load";
        case cache_op_code::drop:      return "drop";
    }
    return "unknown";
}

// registry reason/lifecycle names are duplicated here (numeric values are the
// contract, see server-request-registry.h reason_code); keep in sync
static const char * registry_reason_name(uint16_t code) {
    switch (code) {
        case 0:   return "none";
        case 100: return "registered";
        case 200: return "admission_wait";
        case 201: return "admission_ready";
        case 202: return "capacity_blocked";
        case 300: return "dispatched";
        case 301: return "slot_rebind";
        case 302: return "yielded";
        case 400: return "progress_observed";
        case 500: return "client_cancel";
        case 501: return "server_cancel";
        case 502: return "queue_timeout";
        case 503: return "run_timeout";
        case 504: return "run_stall";
        case 600: return "completed";
        case 601: return "request_failed";
    }
    return "unknown";
}

static const char * lifecycle_name(int64_t value) {
    switch (value) {
        case 0: return "absent";
        case 1: return "registered";
        case 2: return "queued";
        case 3: return "executing";
        case 4: return "completed";
        case 5: return "cancelled";
        case 6: return "timed_out";
        case 7: return "failed";
    }
    return "unknown";
}

static const char * lane_name(uint8_t lane) {
    switch (lane) {
        case 0: return "low";
        case 1: return "normal";
        case 2: return "fast";
    }
    return "unknown";
}

json event_to_json(const event & e, uint64_t wall_offset) {
    json out = {
        { "v", schema_version },
        { "seq", e.seq },
        { "t_ms", e.at_us / 1000 + wall_offset },
        { "k", kind_name(e.kind) },
    };
    if (e.request_id != 0) {
        out["req"] = e.request_id;
    }
    if (e.slot >= 0) {
        out["slot"] = e.slot;
    }

    switch (e.kind) {
        case event_kind::queued:
            out["lane"]      = lane_name(e.lane);
            out["n_prompt"]  = e.a;
            out["max_out"]   = e.b; // 0 = unlimited
            if (e.c != 0) {
                out["parent"] = e.c;
            }
            break;
        case event_kind::deferred:
            out["why"] = defer_name(e.code);
            break;
        case event_kind::dispatched:
            out["lane"]    = lane_name(e.lane);
            out["wait_ms"] = e.a / 1000;
            break;
        case event_kind::slot_selected:
            out["sel"]     = select_name(e.code);
            out["lcp"]     = e.a;  // common-prefix tokens with the selected slot residue
            out["n_task"]  = e.b;  // task prompt tokens
            if (e.f > 0.0) {
                out["sim"] = e.f;  // lcp / n_task similarity when affinity fired
            }
            break;
        case event_kind::admission:
            out["outcome"] = admission_name(e.code);
            if (e.code == (uint16_t) admission_code::ready_reuse ||
                e.code == (uint16_t) admission_code::ready_spanless) {
                out["lcp"]      = e.a;
                out["resident"] = e.b;
                out["frontier"] = e.c;
                out["span_end"] = e.d;
            } else if (e.code == (uint16_t) admission_code::ready_full_clear) {
                // why the resident KV was cleared instead of reused, plus how
                // much resident state the clear destroyed
                out["why"]      = clear_name(e.a);
                out["resident"] = e.b;
                out["lcp"]      = e.c;
                out["span_end"] = e.d;
            }
            break;
        case event_kind::cache_restore:
            out["src"]      = restore_name(e.code);
            out["tokens"]   = e.a;  // reusable prefix tokens after the consult
            out["n_task"]   = e.c;
            out["bytes"]    = e.b;
            if (e.d >= 0) {
                out["donor_slot"] = e.d;
            }
            out["ms"] = e.f;
            break;
        case event_kind::prompt_start:
            out["n_prompt"] = e.a;
            out["n_past"]   = e.b;  // final landing: tokens NOT re-processed
            out["lcp"]      = e.c;  // reusable prefix before checkpoint landing
            if (e.b < e.c) {
                out["gap_why"] = span_name(e.code); // why n_past landed below lcp
            }
            break;
        case event_kind::prefill_progress:
            out["n_done"]   = e.a;  // prompt tokens in KV after this ubatch
            out["n_prompt"] = e.b;
            out["n_batch"]  = e.c;  // tokens processed in this ubatch
            out["ms"]       = e.d / 1000.0; // since prompt processing started
            break;
        case event_kind::decode_progress:
            out["n_dec"]   = e.a;   // decoded tokens so far
            out["draft_n"] = e.b;   // drafted tokens so far
            out["draft_a"] = e.c;   // accepted draft tokens so far
            out["ms"]      = e.d / 1000.0; // since generation started
            out["tps"]     = e.f;   // rate since the previous decode_progress
            break;
        case event_kind::finished:
            out["stop"]     = stop_name(e.code);
            out["n_prompt"] = e.a;
            out["n_cached"] = e.b;
            out["n_dec"]    = e.c;
            out["draft_n"]  = e.d;
            out["draft_a"]  = e.e;
            out["pp_ms"]    = e.f;
            out["tg_ms"]    = e.g;
            break;
        case event_kind::terminal:
            out["state"]  = lifecycle_name(e.a);
            out["reason"] = registry_reason_name(e.code);
            out["code"]   = e.code;
            break;
        case event_kind::cache_op:
            out["op"]     = cache_op_name(e.code);
            out["tokens"] = e.a;
            out["bytes"]  = e.b;
            out["ms"]     = e.f;
            break;
    }
    return out;
}

std::string make_sse_frame(const event & e, uint64_t wall_offset) {
    return "id: " + std::to_string(e.seq) + "\n" +
           "data: " + event_to_json(e, wall_offset).dump() + "\n\n";
}

std::string make_hello_frame(const ring_snapshot & window, uint64_t now_us, uint64_t wall_ms) {
    const json hello = {
        { "v", schema_version },
        { "k", "hello" },
        { "first_seq", window.first_seq },
        { "next_seq", window.next_seq },
        { "dropped", window.dropped },
        { "now_ms", now_us / 1000 + wall_offset_ms() },
        { "wall_ms", wall_ms },
    };
    return "event: hello\ndata: " + hello.dump() + "\n\n";
}

json cache_state_to_json(const cache_state & state, uint64_t now_us) {
    json entries = json::array();
    for (const auto & entry : state.entries) {
        json row = {
            { "id", entry.id },
            { "tokens", entry.tokens },
            { "tier", entry.on_disk ? "disk" : "ram" },
            { "bytes", entry.on_disk ? entry.bytes_disk : entry.bytes_ram },
            { "age_s", entry.created_us != 0 && now_us > entry.created_us
                    ? (now_us - entry.created_us) / 1000000.0 : 0.0 },
            { "hits", entry.hits },
        };
        if (entry.last_hit_us != 0 && now_us >= entry.last_hit_us) {
            row["last_hit_s"] = (now_us - entry.last_hit_us) / 1000000.0;
        } else {
            row["last_hit_s"] = nullptr;
        }
        if (!entry.file.empty()) {
            row["file"] = entry.file;
        }
        row["persistence"] = entry.persistence;
        row["generation"]  = entry.generation;
        // v3, additive: does a text preview exist for this entry, and which
        // request's save created it. The preview TEXT is never in this payload
        // — it is fetched one entry at a time from /m2-dashboard/cache-preview.
        row["preview"] = !entry.head_ids.empty();
        if (entry.request_id != 0) {
            row["req"] = entry.request_id;
        }
        entries.push_back(std::move(row));
    }

    return {
        { "v", schema_version },
        { "enabled", state.enabled },
        { "t_ms", state.at_us / 1000 + wall_offset_ms() },
        { "age_ms", now_us > state.at_us ? (now_us - state.at_us) / 1000 : 0 },
        { "ram", {
            { "used_bytes", state.used_ram_bytes },
            { "limit_bytes", state.limit_ram_bytes },
            { "tokens", state.tokens_total },
            { "limit_tokens", state.limit_tokens },
        } },
        { "disk", {
            { "used_bytes", state.used_disk_bytes },
            { "limit_bytes", state.limit_disk_bytes },
            { "reserved_bytes", state.io_disk_reserved_bytes },
            { "committed_bytes", state.io_disk_committed_bytes },
            { "uncertain_bytes", state.io_disk_uncertain_bytes },
        } },
        { "io", {
            { "accepting", state.io_accepting },
            { "active_jobs", state.io_active_jobs },
            { "queued_jobs", state.io_queued_jobs },
            { "writing_jobs", state.io_writing_jobs },
            { "completion_backlog", state.io_completion_backlog },
            { "pending_ram_bytes", state.io_pending_ram_bytes },
            { "cooldown_until_ns", state.io_cooldown_until_ns },
            { "last_error", state.io_last_error },
            { "retirement_tombstones", state.io_retirement_tombstones },
        } },
        { "counters", {
            { "lookups", state.counters.lookups },
            { "hits_entry", state.counters.hits_entry },
            { "hits_resident", state.counters.hits_resident },
            { "misses", state.counters.misses },
            { "saves", state.counters.saves },
            { "spills", state.counters.spills },
            { "disk_loads", state.counters.disk_loads },
            { "drops", state.counters.drops },
            { "bytes_saved", state.counters.bytes_saved },
            { "bytes_spilled", state.counters.bytes_spilled },
            { "bytes_disk_load", state.counters.bytes_disk_load },
            { "io_rejections", state.counters.io_rejections },
            { "io_uncertain", state.counters.io_uncertain },
            { "io_stale", state.counters.io_stale },
        } },
        { "entries", std::move(entries) },
    };
}

// stream leases -------------------------------------------------------------

static std::atomic<size_t> g_active_streams{0};

std::shared_ptr<void> try_acquire_stream_lease() {
    size_t current = g_active_streams.load();
    while (true) {
        if (current >= maximum_live_streams) {
            return nullptr;
        }
        if (g_active_streams.compare_exchange_weak(current, current + 1)) {
            break;
        }
    }
    // non-null token so callers can test the lease with operator bool
    return std::shared_ptr<void>((void *) &g_active_streams, [](void *) {
        g_active_streams.fetch_sub(1);
    });
}

size_t active_streams() {
    return g_active_streams.load();
}

// UTF-8 safe truncation -----------------------------------------------------

// Longest prefix of `s` that is at most `max_bytes` long and does not end in
// the middle of a multi-byte sequence.
size_t utf8_prefix_len(const std::string & s, size_t max_bytes) {
    if (s.size() <= max_bytes) {
        return s.size();
    }
    size_t n = max_bytes;
    // back off while the next byte is a continuation byte (10xxxxxx)
    while (n > 0 && ((unsigned char) s[n] & 0xC0) == 0x80) {
        --n;
    }
    return n;
}

// Start offset of the longest suffix of `s` that is at most `max_bytes` long
// and begins on a code-point boundary.
size_t utf8_suffix_start(const std::string & s, size_t max_bytes) {
    if (s.size() <= max_bytes) {
        return 0;
    }
    size_t start = s.size() - max_bytes;
    while (start < s.size() && ((unsigned char) s[start] & 0xC0) == 0x80) {
        ++start;
    }
    return start;
}

// content store -------------------------------------------------------------

void content_store::set_enabled(bool enable) {
    std::lock_guard<std::mutex> lock(mtx);
    on = enable;
    if (!enable) {
        entries.clear();
        order.clear();
        tombstones.clear();
        bytes_used = 0;
    }
}

bool content_store::enabled() const {
    std::lock_guard<std::mutex> lock(mtx);
    return on;
}

void content_store::evict_locked() {
    // A running request is the one thing that must survive FIFO pressure: a
    // 45-minute agent turn is the OLDEST record on the box, so plain FIFO
    // would evict the prompt of the request you are currently watching. The
    // exemption is itself bounded — a hard ceiling at twice the nominal cap
    // catches records that never reach a terminal state.
    const bool hard = entries.size() > content_max_requests * 2;
    size_t     skipped = 0;
    while (order.size() > skipped &&
           (entries.size() > content_max_requests || bytes_used > content_max_bytes)) {
        const uint64_t oldest = order.front();
        order.pop_front();
        const auto it = entries.find(oldest);
        if (it == entries.end()) {
            continue;
        }
        if (!hard && !it->second.complete) {
            order.push_back(oldest); // still generating: keep it, try the next
            ++skipped;
            continue;
        }
        bytes_used -= std::min(bytes_used, it->second.bytes());
        entries.erase(it);
        ++evicted;
        // a small tombstone list lets the UI say "no longer retained" instead
        // of "never seen" for a request the store actually held
        tombstones.push_back(oldest);
        if (tombstones.size() > content_max_requests * 4) {
            tombstones.pop_front();
        }
    }
}

void content_store::capture_input(uint64_t req, int32_t slot, uint64_t n_tokens,
                                  std::vector<int32_t> && head, std::vector<int32_t> && tail) {
    if (req == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (!on) {
        return;
    }
    auto it = entries.find(req);
    if (it == entries.end()) {
        it = entries.emplace(req, request_content{}).first;
        it->second.request_id = req;
        order.push_back(req);
    } else {
        bytes_used -= std::min(bytes_used, it->second.bytes());
    }
    request_content & c = it->second;
    c.slot      = slot;
    c.at_us     = (uint64_t) ggml_time_us();
    c.has_input = true;
    c.in_tokens = n_tokens;
    c.in_head   = std::move(head);
    c.in_tail   = std::move(tail);
    bytes_used += c.bytes();
    evict_locked();
}

void content_store::capture_output(uint64_t req, int32_t slot, const std::string & text,
                                   uint64_t n_tokens) {
    if (req == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (!on) {
        return;
    }
    auto it = entries.find(req);
    if (it == entries.end()) {
        it = entries.emplace(req, request_content{}).first;
        it->second.request_id = req;
        order.push_back(req);
    } else {
        bytes_used -= std::min(bytes_used, it->second.bytes());
    }
    request_content & c = it->second;
    c.slot       = slot;
    c.at_us      = (uint64_t) ggml_time_us();
    c.complete   = true;
    c.has_output = true;
    c.out_tokens = n_tokens;
    c.out_bytes  = text.size();
    c.out_head.assign(text, 0, utf8_prefix_len(text, content_out_head_bytes));
    if (text.size() > c.out_head.size()) {
        const size_t remaining = text.size() - c.out_head.size();
        const size_t want      = std::min(remaining, content_out_tail_bytes);
        size_t       start     = text.size() - want;
        while (start < text.size() && ((unsigned char) text[start] & 0xC0) == 0x80) {
            ++start;
        }
        if (start < c.out_head.size()) {
            start = c.out_head.size();
        }
        c.out_tail.assign(text, start, text.size() - start);
    } else {
        c.out_tail.clear();
    }
    bytes_used += c.bytes();
    evict_locked();
}

void content_store::mark_final(uint64_t req) {
    std::lock_guard<std::mutex> lock(mtx);
    const auto it = entries.find(req);
    if (it == entries.end()) {
        return;
    }
    // the request reached a terminal state (including cancel/stall, which never
    // reach send_final_response): it stops being eviction-exempt
    it->second.complete = true;
    evict_locked();
}

bool content_store::lookup(uint64_t req, request_content & out) const {
    std::lock_guard<std::mutex> lock(mtx);
    if (!on) {
        return false;
    }
    const auto it = entries.find(req);
    if (it == entries.end()) {
        return false;
    }
    out = it->second;
    return true;
}

content_stats content_store::usage() const {
    std::lock_guard<std::mutex> lock(mtx);
    content_stats st;
    st.requests = entries.size();
    st.bytes    = bytes_used;
    st.evicted  = evicted;
    return st;
}

bool content_store::was_evicted(uint64_t req) const {
    std::lock_guard<std::mutex> lock(mtx);
    return std::find(tombstones.begin(), tombstones.end(), req) != tombstones.end();
}

void content_store::clear() {
    std::lock_guard<std::mutex> lock(mtx);
    entries.clear();
    order.clear();
    tombstones.clear();
    bytes_used = 0;
    evicted    = 0;
}

content_store & content() {
    static content_store store;
    return store;
}

// live token mirror ---------------------------------------------------------

namespace detail {
std::atomic<uint32_t> watch_active{0};
}

int watch_registry::attach(uint64_t req) {
    if (req == 0) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(mtx);
    for (size_t i = 0; i < maximum_watch_streams; ++i) {
        if (slots[i].in_use) {
            continue;
        }
        slots[i]          = slot_state{};
        slots[i].in_use   = true;
        slots[i].req      = req;
        slots[i].replace  = true; // the first drain is authoritative, not a delta
        detail::watch_active.fetch_add(1, std::memory_order_relaxed);
        return (int) i;
    }
    return -1;
}

void watch_registry::detach(int lease) {
    if (lease < 0 || (size_t) lease >= maximum_watch_streams) {
        return;
    }
    std::lock_guard<std::mutex> lock(mtx);
    if (!slots[lease].in_use) {
        return;
    }
    slots[lease] = slot_state{};
    detail::watch_active.fetch_sub(1, std::memory_order_relaxed);
}

void watch_registry::sync(uint64_t req, int32_t slot_id, const std::string & generated,
                          int32_t n_dec) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto & s : slots) {
        if (!s.in_use || s.req != req) {
            continue;
        }
        s.slot  = slot_id;
        s.n_dec = n_dec;
        if (generated.size() < s.mirrored) {
            // the slot trimmed a stop word off its own tail: everything the
            // reader has is now wrong, so hand it the whole current text
            s.replace  = true;
            s.pending  = generated;
            s.mirrored = generated.size();
            s.produced = generated.size();
            s.start    = 0;
        } else if (generated.size() > s.mirrored) {
            if (s.pending.empty()) {
                s.start = s.produced;
            }
            s.pending.append(generated, s.mirrored, generated.size() - s.mirrored);
            s.produced += generated.size() - s.mirrored;
            s.mirrored  = generated.size();
        }
        if (s.pending.size() > watch_mirror_bytes) {
            // the reader is not keeping up. Keep the newest bytes and account
            // for what was dropped rather than growing without bound. After a
            // trim the reader cannot splice, so it is resynced with a replace.
            const size_t start = utf8_suffix_start(s.pending, watch_mirror_bytes);
            s.pending.erase(0, start);
            s.dropped += start;
            s.start   += start;
            s.replace  = true;
        }
    }
}

void watch_registry::finish(uint64_t req, int32_t n_dec) {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto & s : slots) {
        if (s.in_use && s.req == req) {
            s.ended = true;
            if (n_dec > 0) {
                s.n_dec = n_dec;
            }
        }
    }
}

bool watch_registry::drain(int lease, watch_view & out) {
    if (lease < 0 || (size_t) lease >= maximum_watch_streams) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx);
    slot_state & s = slots[lease];
    if (!s.in_use) {
        return false;
    }
    out.req     = s.req;
    out.slot    = s.slot;
    out.n_dec   = s.n_dec;
    out.cursor  = s.produced;
    out.offset  = s.start;
    out.dropped = s.dropped;
    out.replace = s.replace;
    out.ended   = s.ended;
    out.text    = std::move(s.pending);
    s.pending.clear();
    s.start   = s.produced;
    s.replace = false;
    return true;
}

size_t watch_registry::active() const {
    return detail::watch_active.load(std::memory_order_relaxed);
}

void watch_registry::reset() {
    std::lock_guard<std::mutex> lock(mtx);
    for (auto & s : slots) {
        s = slot_state{};
    }
    detail::watch_active.store(0, std::memory_order_relaxed);
}

watch_registry & watches() {
    static watch_registry registry;
    return registry;
}

// content serialization -----------------------------------------------------

static json content_caps_json() {
    return {
        { "in_head_tokens", content_in_head_tokens },
        { "in_tail_tokens", content_in_tail_tokens },
        { "out_head_bytes", content_out_head_bytes },
        { "out_tail_bytes", content_out_tail_bytes },
        { "max_requests", content_max_requests },
        { "max_bytes", content_max_bytes },
    };
}

static json content_stats_json(const content_stats & stats) {
    return {
        { "requests", stats.requests },
        { "bytes", stats.bytes },
        { "evicted", stats.evicted },
    };
}

static json input_json(const request_content & c, const detokenizer & detok) {
    const uint64_t shown = (uint64_t) (c.in_head.size() + c.in_tail.size());
    return {
        { "n_tokens", c.in_tokens },
        { "head", detok(c.in_head) },
        { "tail", detok(c.in_tail) },
        { "head_tokens", c.in_head.size() },
        { "tail_tokens", c.in_tail.size() },
        { "elided_tokens", c.in_tokens > shown ? c.in_tokens - shown : 0 },
        { "truncated", c.in_tokens > shown },
    };
}

json content_to_json(uint64_t req, const request_content * found, const char * reason,
                     const content_stats & stats, const detokenizer & detok) {
    if (found == nullptr) {
        return {
            { "v", schema_version },
            { "req", req },
            { "found", false },
            { "reason", reason ? reason : "absent" },
            { "caps", content_caps_json() },
            { "store", content_stats_json(stats) },
        };
    }

    const request_content & c = *found;
    json out = {
        { "v", schema_version },
        { "req", req },
        { "found", true },
        { "state", c.complete ? "final" : "live" },
        { "slot", c.slot },
        { "t_ms", c.at_us / 1000 + wall_offset_ms() },
        { "caps", content_caps_json() },
        { "store", content_stats_json(stats) },
    };
    out["input"] = c.has_input ? input_json(c, detok) : json(nullptr);
    if (c.has_output) {
        const uint64_t shown = (uint64_t) (c.out_head.size() + c.out_tail.size());
        out["output"] = {
            { "n_tokens", c.out_tokens },
            { "n_bytes", c.out_bytes },
            { "head", c.out_head },
            { "tail", c.out_tail },
            { "elided_bytes", c.out_bytes > shown ? c.out_bytes - shown : 0 },
            { "truncated", c.out_bytes > shown },
        };
    } else {
        out["output"] = nullptr;
    }
    return out;
}

json cache_preview_to_json(uint64_t id, const cache_entry_state * entry, uint64_t now_us,
                           uint64_t snapshot_at_us, const detokenizer & detok) {
    // "found" means "there is text to show": an entry with no retained tokens
    // (the same thing cache-state reports as preview:false) is not previewable
    if (entry == nullptr || entry->head_ids.empty()) {
        return {
            { "v", schema_version },
            { "id", id },
            { "found", false },
        };
    }
    const uint64_t shown = (uint64_t) (entry->head_ids.size() + entry->tail_ids.size());
    json out = {
        { "v", schema_version },
        { "id", id },
        { "found", true },
        { "tier", entry->on_disk ? "disk" : "ram" },
        { "tokens", entry->tokens },
        { "head", detok(entry->head_ids) },
        { "tail", detok(entry->tail_ids) },
        { "head_tokens", entry->head_ids.size() },
        { "tail_tokens", entry->tail_ids.size() },
        { "elided_tokens", entry->tokens > shown ? entry->tokens - shown : 0 },
        { "truncated", entry->tokens > shown },
        { "age_ms", now_us > snapshot_at_us ? (now_us - snapshot_at_us) / 1000 : 0 },
    };
    if (entry->request_id != 0) {
        out["req"] = entry->request_id;
    }
    return out;
}

std::string make_watch_hello_frame(uint64_t req, const watch_view & view, const char * state,
                                   const json & input) {
    const json hello = {
        { "v", schema_version },
        { "k", "watch-hello" },
        { "req", req },
        { "slot", view.slot },
        { "state", state },
        { "n_dec", view.n_dec },
        { "cursor", view.cursor },
        { "text", view.text },
        { "dropped", view.dropped },
        { "input", input },
        { "caps", { { "mirror_bytes", watch_mirror_bytes }, { "max_streams", maximum_watch_streams } } },
    };
    return "event: watch-hello\ndata: " + hello.dump() + "\n\n";
}

std::string make_watch_frame(const watch_view & view) {
    // cursor: END offset for a delta, START offset for a rewind (see
    // watch_view). `offset` carries the start offset unconditionally so a
    // client never has to branch on the kind to know where `text` sits.
    const json frame = {
        { "v", schema_version },
        { "k", view.replace ? "rewind" : "delta" },
        { "req", view.req },
        { "cursor", view.replace ? view.offset : view.cursor },
        { "offset", view.offset },
        { "text", view.text },
        { "n_dec", view.n_dec },
        { "dropped", view.dropped },
    };
    return "data: " + frame.dump() + "\n\n";
}

std::string make_watch_end_frame(uint64_t req, const char * reason, int32_t n_dec) {
    const json frame = {
        { "v", schema_version },
        { "k", "end" },
        { "req", req },
        { "reason", reason },
        { "n_dec", n_dec },
    };
    return "event: watch-end\ndata: " + frame.dump() + "\n\n";
}

}  // namespace server_dashboard

#pragma once

// m2-dashboard event bus: the server-side half of the LAN dashboard's realtime
// feed (dashboard v2). A single process-wide bounded ring receives compact,
// fixed-size lifecycle/phase events from the scheduler, the slot-selection and
// admission logic, the prompt-cache tiers, and the prefill/decode loop. The
// SSE handler (/m2-dashboard/events) replays the retained window and then
// tails the ring; /m2-dashboard/cache-state serves the latest prompt-cache
// occupancy snapshot published by the main loop.
//
// Design rules (see notes/2026-08-08-dashboard-v2.md):
// - the emit path is allocation-free: one leaf mutex, one 96-byte struct copy
//   into a preallocated ring slot. Emit sites never format JSON.
// - no per-token events: producers aggregate per ubatch / phase transition /
//   request transition. JSON serialization happens on the HTTP thread, only
//   for connected consumers.
// - the bus is a leaf: it never calls back into the queue, registry, or
//   llama context, so it can be invoked under the queue mutex or from the
//   main loop without lock-order concerns.
// - events carry the registry's runtime request id convention
//   (uint64_t(task.id) + 1) so consumers can correlate with the admin
//   dashboard and scheduler traces.

#include "ggml.h" // GGML_ASSERT (JSON_ASSERT) and ggml_time_us

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace server_dashboard {

using json = nlohmann::ordered_json;

// bump when the serialized event field set changes incompatibly; every SSE
// frame and snapshot carries this as "v"
constexpr uint32_t schema_version = 1;

constexpr const char * events_path        = "/m2-dashboard/events";
constexpr const char * cache_state_path   = "/m2-dashboard/cache-state";
// dashboard v3 content routes (see notes/2026-08-09-dashboard-v3.md)
constexpr const char * content_path       = "/m2-dashboard/content";
constexpr const char * watch_path         = "/m2-dashboard/watch";
constexpr const char * cache_preview_path = "/m2-dashboard/cache-preview";

constexpr size_t ring_capacity_default   = 8192;
constexpr size_t maximum_live_streams    = 8;
constexpr size_t maximum_events_per_read = 256;
constexpr int    poll_interval_ms        = 100;
constexpr int    heartbeat_ms            = 5000;

enum class event_kind : uint8_t {
    // request lifecycle (queue side, mirrors the registry transitions)
    queued = 0,       // request admitted to the scheduler queue
    deferred,         // admission deferred (code = defer_code)
    dispatched,       // scheduler granted a dispatch permit
    // inference side (main loop)
    slot_selected,    // slot chosen for the task (code = select_code)
    admission,        // DSV4 physical admission outcome (code = admission_code)
    cache_restore,    // prompt-cache consult restored/kept a prefix (code = restore_code)
    prompt_start,     // prefill landing decided (n_past final)
    prefill_progress, // one processed prefill ubatch
    decode_progress,  // aggregated decode advance (>=32 tokens or >=1s)
    finished,         // final response sent, with timings (code = stop_code)
    terminal,         // registry terminal state (code = registry reason_code)
    // prompt cache tier traffic (slot = -1)
    cache_op,         // code = cache_op_code
};

// event.code values per kind ----------------------------------------------

enum class defer_code : uint16_t {
    capacity = 0,           // no free scheduler capacity / cohort closed
    physical_admission = 1, // DSV4 physical KV admission deferred the family
};

enum class select_code : uint16_t {
    by_id = 0,       // client pinned id_slot
    lcp_affinity,    // prefix-affinity (longest common prefix similarity)
    lru,             // least-recently-used fallback
};

enum class admission_code : uint16_t {
    ready_full_clear = 0, // admitted, slot cleared, full span reserved
    ready_reuse,          // admitted, resident prefix kept, suffix span reserved
    ready_spanless,       // admitted, task ends below resident frontier, no span
    deferred,             // family deferred on physical capacity
    rejected,             // rejected with an error response
};

enum class restore_code : uint16_t {
    resident = 0, // resident slot state already held the best prefix
    ram,          // restored a RAM-tier prompt-cache entry
    disk,         // restored an SSD-tier entry (disk load included in ms)
    donor,        // zero-copy seq_cp share from a live donor slot
    miss,         // consult found nothing reusable
};

// why a prompt token range had to be recomputed. Carried by prompt_start as
// gap_why; the client derives the full [start,end) span decomposition from
// prompt_start + cache_restore (see tools/m2-dashboard/lib/events.mjs
// tokenSpans) rather than the server emitting one event per range.
enum class span_code : uint16_t {
    // sources for restored ranges
    src_resident = 0,
    src_cache_ram,
    src_cache_disk,
    src_donor,
    // reasons for recomputed ranges
    recompute_no_cache = 16,      // no entry / no resident prefix covered this range
    recompute_checkpoint_gap,     // landing fell back to a checkpoint below the LCP
    recompute_reset,              // no usable checkpoint: full re-process from 0
    recompute_divergent_tail,     // suffix after the common prefix diverged
    recompute_last_token,         // full-prefix match still re-evaluates one token [TAG_PROMPT_LOGITS]
};

// why an elastic admission cleared the slot's resident KV instead of reusing
// it. Mirrors the conditions of the reuse_resident predicate in
// prepare_dsv4_admission (server-context.cpp ~2277) in priority order, so the
// dashboard can attribute every full clear. Lane admission-reuse-fix adds an
// "elastic admission clears resident KV: <reason>" log over the same
// predicate; keep the two reason sets aligned and append new values at the end
// (the numeric values are the wire contract).
enum class clear_code : uint16_t {
    empty_slot = 0,        // slot held no resident tokens: a cold start, not a discard
    family,                // parent/child family admission: reuse is single-slot only
    no_lcp,                // resident prefix shares nothing with the task
    lcp_covers_task,       // task ends at or below the common prefix
    mtmd,                  // multimodal residue cannot be partially reused
    cache_reuse_opt,       // request set n_cache_reuse != 0
    no_cache_prompt,       // request disabled prompt caching
    no_pos_min,            // sequence reported no minimum position
    no_covering_checkpoint,// checkpoint pass would run but nothing covers the landing
    rs_window,             // tail trim exceeds the bounded rs rollback window
};

enum class stop_code : uint16_t {
    eos = 0,     // model emitted end-of-sequence
    limit,       // n_predict budget exhausted
    word,        // stop word
    error,       // request errored
    aborted,     // released without a normal stop (cancel/timeout path)
};

enum class cache_op_code : uint16_t {
    save = 0,    // slot state serialized into the RAM tier
    spill,       // RAM-tier entry written to the SSD tier
    disk_load,   // SSD-tier entry read back into RAM
    drop,        // entry erased (limit pressure or invalidation)
};

// fixed-size event record; a..e/f/g are kind-specific and documented by the
// serializer in server-dashboard-bus.cpp (single source of field names)
struct event {
    uint64_t   seq        = 0;  // assigned by the bus
    uint64_t   at_us      = 0;  // ggml_time_us(); assigned by the bus when 0
    event_kind kind       = event_kind::queued;
    uint64_t   request_id = 0;  // registry runtime id (task.id + 1); 0 = server-scoped
    int32_t    slot       = -1;
    uint8_t    lane       = 0;  // server_scheduler::lane / trusted_lane numeric value
    uint16_t   code       = 0;  // kind-specific enum above (or registry reason_code for terminal)
    int64_t    a          = 0;
    int64_t    b          = 0;
    int64_t    c          = 0;
    int64_t    d          = 0;
    int64_t    e          = 0;
    double     f          = 0.0;
    double     g          = 0.0;
};

struct ring_snapshot {
    std::vector<event> events;
    uint64_t first_seq = 1; // oldest sequence still retained
    uint64_t next_seq  = 1; // sequence the next emit will take
    uint64_t dropped   = 0; // events overwritten since start
};

// prompt-cache introspection ------------------------------------------------

struct cache_entry_state {
    uint64_t    id           = 0;  // stable per-entry id (allocation counter)
    uint64_t    tokens       = 0;
    uint64_t    bytes_ram    = 0;  // in-RAM blob + checkpoint bytes (0 when on disk)
    uint64_t    bytes_disk   = 0;  // backing file size (0 when RAM-resident)
    bool        on_disk      = false;
    uint64_t    created_us   = 0;
    uint64_t    last_hit_us  = 0;  // 0 = never hit
    uint32_t    hits         = 0;
    std::string file;              // basename of the SSD-tier file, empty in RAM
    std::string persistence;       // ram_ready/queued/writing/durable/failed/uncertain
    uint64_t    generation = 0;

    // dashboard v3 preview material. Both tiers keep prompt.tokens resident
    // (the SSD tier only spills the state blobs), so a bounded head/tail slice
    // of TOKEN IDS rides along in the snapshot and is detokenized on the HTTP
    // thread only when GET /m2-dashboard/cache-preview asks for that entry.
    // No decoded text is ever retained here.
    std::vector<int32_t> head_ids;
    std::vector<int32_t> tail_ids;
    uint64_t             request_id = 0; // request whose save created it (0 = unknown)
};

struct cache_counters {
    uint64_t lookups          = 0; // prompt-cache consults
    uint64_t hits_entry       = 0; // consults that restored a cache entry
    uint64_t hits_resident    = 0; // consults where the resident state already won
    uint64_t misses           = 0; // consults with nothing reusable
    uint64_t saves            = 0;
    uint64_t spills           = 0;
    uint64_t disk_loads       = 0;
    uint64_t drops            = 0;
    uint64_t bytes_saved      = 0;
    uint64_t bytes_spilled    = 0;
    uint64_t bytes_disk_load  = 0;
    uint64_t io_rejections    = 0;
    uint64_t io_uncertain     = 0;
    uint64_t io_stale         = 0;
};

struct cache_state {
    bool     enabled          = false;
    uint64_t at_us            = 0;
    uint64_t limit_ram_bytes  = 0; // 0 = unlimited
    uint64_t limit_disk_bytes = 0; // 0 = unlimited
    uint64_t limit_tokens     = 0; // 0 = unlimited
    uint64_t used_ram_bytes   = 0;
    uint64_t used_disk_bytes  = 0;
    uint64_t tokens_total     = 0;
    bool     io_accepting             = false;
    uint32_t io_active_jobs           = 0;
    uint32_t io_queued_jobs           = 0;
    uint32_t io_writing_jobs          = 0;
    uint32_t io_completion_backlog    = 0;
    uint64_t io_pending_ram_bytes     = 0;
    uint64_t io_disk_reserved_bytes   = 0;
    uint64_t io_disk_committed_bytes  = 0;
    uint64_t io_disk_uncertain_bytes  = 0;
    uint64_t io_cooldown_until_ns     = 0;
    int      io_last_error            = 0;
    uint64_t io_retirement_tombstones = 0;
    std::vector<cache_entry_state> entries;
    cache_counters counters;
};

// bus ----------------------------------------------------------------------

class bus {
  public:
    explicit bus(size_t capacity = ring_capacity_default);

    // assigns seq (and at_us when zero) and appends; overwrites the oldest
    // event when full. noexcept + allocation-free by design.
    void emit(event e) noexcept;

    // events with seq > after_seq, oldest first, capped at max_events
    ring_snapshot snapshot_after(uint64_t after_seq, size_t max_events = maximum_events_per_read) const;

    uint64_t next_seq() const;

    void        publish_cache_state(cache_state && state);
    cache_state cache() const;

  private:
    mutable std::mutex mtx;
    std::vector<event> ring;   // preallocated, size == capacity
    size_t             count   = 0; // valid events in ring
    uint64_t           next    = 1; // next sequence to assign
    uint64_t           dropped = 0;

    mutable std::mutex cache_mtx;
    cache_state        cache_snapshot;
};

// process-wide instance used by all emit sites
bus & instance();

// content capture (dashboard v3) ---------------------------------------------
//
// Request content (the prompt as sent, the text generated) is deliberately NOT
// on the bus: the bus event is a fixed 96-byte POD and the replay ring is
// bounded history that any late consumer receives, so a ring of prompts would
// be a standing liability. Content lives in its own store with its own caps,
// its own routes, and its own kill switch:
//
//   - retention is bounded in entries AND bytes and evicts oldest-first;
//   - the complete target prompt is retained as TOKEN IDS and detokenized only
//     on the HTTP thread, on demand, for one request at a time (the same thing
//     POST /detokenize already does; the vocab is immutable, so this is safe
//     off the main loop);
//   - the complete target generation is retained as UTF-8 (it is already text
//     on the producer side, so no detokenize happens on the main loop);
//   - nothing here is ever logged, and none of it appears in /m2-dashboard/
//     events, /m2-dashboard/cache-state or the registry snapshot.

constexpr size_t   content_max_requests   = 64;          // retained requests
constexpr size_t   content_max_bytes      = 256u << 20;  // 256 MiB across the FIFO

// Production runs a shared 512K context and requests at most 32K output
// tokens. Keep those target-sized values whole. The legacy head/tail wire
// shape remains for compatibility and still reports honest truncation for a
// request outside this fork's target envelope.
constexpr uint32_t content_in_head_tokens = 512u << 10;
constexpr uint32_t content_in_tail_tokens = 0;
constexpr size_t   content_out_head_bytes = 8u << 20;
constexpr size_t   content_out_tail_bytes = 0;

constexpr size_t cache_preview_head_tokens = 64;
constexpr size_t cache_preview_tail_tokens = 32;

constexpr size_t maximum_watch_streams   = 2;
constexpr size_t watch_mirror_bytes      = 64u << 10;    // per watcher
constexpr int    watch_poll_interval_ms  = 60;
constexpr int    watch_idle_timeout_ms   = 180000;       // silent watched request

struct request_content {
    uint64_t request_id = 0;
    int32_t  slot       = -1;
    uint64_t at_us      = 0;
    bool     complete   = false; // the generation finished (output is final)

    bool                 has_input = false;
    uint64_t             in_tokens = 0;  // full prompt length, not the slice length
    std::vector<int32_t> in_head;
    std::vector<int32_t> in_tail;

    bool        has_output  = false;
    uint64_t    out_tokens  = 0;
    uint64_t    out_bytes   = 0;  // full generation length in bytes
    std::string out_head;
    std::string out_tail;

    size_t bytes() const {
        return (in_head.size() + in_tail.size()) * sizeof(int32_t) +
               out_head.size() + out_tail.size();
    }
};

struct content_stats {
    size_t   requests = 0;
    size_t   bytes    = 0;
    uint64_t evicted  = 0;
};

class content_store {
  public:
    void set_enabled(bool on);
    bool enabled() const;

    void capture_input(uint64_t req, int32_t slot, uint64_t n_tokens,
                       std::vector<int32_t> && head, std::vector<int32_t> && tail);
    void capture_output(uint64_t req, int32_t slot, const std::string & text, uint64_t n_tokens);
    // terminal transition without an output capture (cancel, stall, timeout):
    // drops the record's eviction exemption
    void mark_final(uint64_t req);

    // true when the request is retained; fills `out` with a copy
    bool          lookup(uint64_t req, request_content & out) const;
    content_stats usage() const;
    bool          was_evicted(uint64_t req) const;
    void          clear();

  private:
    void evict_locked();

    mutable std::mutex                             mtx;
    std::unordered_map<uint64_t, request_content>  entries;
    std::deque<uint64_t>                           order;      // FIFO eviction order
    std::deque<uint64_t>                           tombstones; // recently evicted ids
    size_t                                         bytes_used = 0;
    uint64_t                                       evicted    = 0;
    bool                                           on         = true;
};

content_store & content();

// Slice a bounded head/tail out of any indexable token container and hand it to
// the store. Returns immediately (no slicing, no allocation) when capture is
// disabled. Negative ids (media placeholders) are kept as-is; the detokenizer
// renders them as a placeholder rather than passing them to the vocab.
template <class Tokens>
void capture_prompt_into(content_store & store, uint64_t req, int32_t slot, const Tokens & toks) {
    if (!store.enabled() || req == 0) {
        return;
    }
    const size_t n  = toks.size();
    const size_t nh = std::min<size_t>(n, content_in_head_tokens);
    std::vector<int32_t> head;
    std::vector<int32_t> tail;
    head.reserve(nh);
    for (size_t i = 0; i < nh; ++i) {
        head.push_back((int32_t) toks[i]);
    }
    if (n > nh) {
        const size_t nt = std::min<size_t>(n - nh, content_in_tail_tokens);
        tail.reserve(nt);
        for (size_t i = n - nt; i < n; ++i) {
            tail.push_back((int32_t) toks[i]);
        }
    }
    store.capture_input(req, slot, (uint64_t) n, std::move(head), std::move(tail));
}

template <class Tokens>
void capture_prompt(uint64_t req, int32_t slot, const Tokens & toks) {
    capture_prompt_into(content(), req, slot, toks);
}

// live token mirror (dashboard v3) -------------------------------------------
//
// GET /m2-dashboard/watch mirrors ONE in-flight request's generated text to one
// HTTP consumer. The decode loop's cost when nobody is watching is a single
// relaxed atomic load (see watching() below): no event, no lock, no copy. When
// a watcher IS attached, the producer appends only the bytes generated since
// its last call into a bounded per-watcher buffer.

namespace detail {
extern std::atomic<uint32_t> watch_active;
}

// hot-path guard. Call this before touching the mirror at all.
inline bool watching() noexcept {
    return detail::watch_active.load(std::memory_order_relaxed) != 0;
}

// One drained frame. `cursor` is what the wire carries, and its meaning is
// deliberately kind-dependent so a bounded mirror can still be described
// exactly (see notes/2026-08-09-dashboard-v3.md, "watch frames"):
//   delta   -> total bytes produced so far, i.e. the END offset of `text`
//   rewind  -> the absolute offset at which `text` BEGINS (0 = replace all)
// A rewind therefore says "throw away what you have; here is the generation
// from byte `cursor` onward", which is representable even when the trim point
// is older than the mirror.
struct watch_view {
    uint64_t    req      = 0;
    int32_t     slot     = -1;
    int32_t     n_dec    = 0;
    uint64_t    cursor   = 0; // total bytes ever produced for this watcher
    uint64_t    offset   = 0; // absolute offset at which `text` begins
    uint64_t    dropped  = 0; // bytes discarded because the reader lagged
    bool        replace  = false; // `text` supersedes everything the reader has
    bool        ended    = false;
    std::string text;
};

class watch_registry {
  public:
    // -1 when every watch slot is taken
    int  attach(uint64_t req);
    void detach(int lease);

    // producer side (main loop). `generated` is the slot's whole generated text
    // so far; only the new suffix is copied. A shrink (stop-word trim) is
    // reported to the consumer as a replace.
    void sync(uint64_t req, int32_t slot, const std::string & generated, int32_t n_dec);
    void finish(uint64_t req, int32_t n_dec);

    // consumer side (HTTP thread). Returns false when the lease is not attached.
    // `out.text` is empty when nothing new arrived.
    bool drain(int lease, watch_view & out);

    size_t active() const;
    void   reset(); // tests

  private:
    struct slot_state {
        bool        in_use   = false;
        uint64_t    req      = 0;
        int32_t     slot     = -1;
        int32_t     n_dec    = 0;
        uint64_t    produced = 0; // total bytes appended for this watcher
        uint64_t    start    = 0; // absolute offset of pending[0]
        uint64_t    dropped  = 0;
        size_t      mirrored = 0; // bytes of generated_text already mirrored
        bool        replace  = true;
        bool        ended    = false;
        std::string pending;
    };

    mutable std::mutex mtx;
    slot_state         slots[maximum_watch_streams];
};

watch_registry & watches();

// serialization (shared by the SSE handler, the snapshot handler, and tests)
const char * kind_name(event_kind kind);
json         event_to_json(const event & e, uint64_t wall_offset_ms = 0);
std::string  make_sse_frame(const event & e, uint64_t wall_offset_ms = 0);
std::string  make_hello_frame(const ring_snapshot & window, uint64_t now_us, uint64_t wall_ms);
json         cache_state_to_json(const cache_state & state, uint64_t now_us);

// The content routes never link against llama: the caller supplies a
// detokenizer so the bus stays a leaf and the tests can inject a fake one.
using detokenizer = std::function<std::string(const std::vector<int32_t> &)>;

// `found == nullptr` renders the not-retained body; `reason` is "absent" or
// "evicted" in that case.
json content_to_json(uint64_t req, const request_content * found, const char * reason,
                     const content_stats & stats, const detokenizer & detok);
json cache_preview_to_json(uint64_t id, const cache_entry_state * entry, uint64_t now_us,
                           uint64_t snapshot_at_us, const detokenizer & detok);

std::string make_watch_hello_frame(uint64_t req, const watch_view & view, const char * state,
                                   const json & input);
std::string make_watch_frame(const watch_view & view);
std::string make_watch_end_frame(uint64_t req, const char * reason, int32_t n_dec);

// milliseconds to add to (at_us / 1000) to obtain wall-clock ms for this
// process (computed once from the current wall clock and ggml_time_us)
uint64_t wall_offset_ms();

// bounded number of concurrent SSE consumers
std::shared_ptr<void> try_acquire_stream_lease();
size_t                active_streams();

// UTF-8 safe truncation helpers used by the content store and its tests: never
// cut a multi-byte sequence in half, or the JSON encoder produces mojibake.
size_t utf8_prefix_len(const std::string & s, size_t max_bytes);
size_t utf8_suffix_start(const std::string & s, size_t max_bytes);

}  // namespace server_dashboard

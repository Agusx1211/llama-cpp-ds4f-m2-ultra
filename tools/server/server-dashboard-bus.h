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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace server_dashboard {

using json = nlohmann::ordered_json;

// bump when the serialized event field set changes incompatibly; every SSE
// frame and snapshot carries this as "v"
constexpr uint32_t schema_version = 1;

constexpr const char * events_path      = "/m2-dashboard/events";
constexpr const char * cache_state_path = "/m2-dashboard/cache-state";

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

// serialization (shared by the SSE handler, the snapshot handler, and tests)
const char * kind_name(event_kind kind);
json         event_to_json(const event & e, uint64_t wall_offset_ms = 0);
std::string  make_sse_frame(const event & e, uint64_t wall_offset_ms = 0);
std::string  make_hello_frame(const ring_snapshot & window, uint64_t now_us, uint64_t wall_ms);
json         cache_state_to_json(const cache_state & state, uint64_t now_us);

// milliseconds to add to (at_us / 1000) to obtain wall-clock ms for this
// process (computed once from the current wall clock and ggml_time_us)
uint64_t wall_offset_ms();

// bounded number of concurrent SSE consumers
std::shared_ptr<void> try_acquire_stream_lease();
size_t                active_streams();

}  // namespace server_dashboard

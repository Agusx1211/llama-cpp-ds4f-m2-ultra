// Host-side tests for the m2-dashboard event bus (tools/server/server-dashboard-bus.*):
// ring retention/cursor semantics, schema-v1 serialization field names, the
// cache-state snapshot JSON, and the bounded stream leases. No model, no
// server process.

#include "server-dashboard-bus.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace server_dashboard;

static int failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            ++failures;                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

static event make_event(event_kind kind, uint64_t request_id, uint64_t at_us = 1) {
    event ev;
    ev.kind       = kind;
    ev.request_id = request_id;
    ev.at_us      = at_us;
    return ev;
}

static void test_ring_basics() {
    bus b(4);

    CHECK(b.next_seq() == 1);
    CHECK(b.snapshot_after(0).events.empty());

    for (uint64_t i = 1; i <= 3; ++i) {
        b.emit(make_event(event_kind::queued, i));
    }

    const auto all = b.snapshot_after(0);
    CHECK(all.events.size() == 3);
    CHECK(all.first_seq == 1);
    CHECK(all.next_seq == 4);
    CHECK(all.dropped == 0);
    CHECK(all.events[0].seq == 1 && all.events[0].request_id == 1);
    CHECK(all.events[2].seq == 3 && all.events[2].request_id == 3);

    // cursor resume: strictly after
    const auto tail = b.snapshot_after(2);
    CHECK(tail.events.size() == 1);
    CHECK(tail.events[0].seq == 3);

    // overwrite: capacity 4, emit 3 more -> oldest 2 dropped
    for (uint64_t i = 4; i <= 6; ++i) {
        b.emit(make_event(event_kind::queued, i));
    }
    const auto window = b.snapshot_after(0);
    CHECK(window.first_seq == 3);
    CHECK(window.next_seq == 7);
    CHECK(window.dropped == 2);
    CHECK(window.events.size() == 4);
    CHECK(window.events.front().seq == 3);
    CHECK(window.events.back().seq == 6);

    // a cursor below the retained window replays from the oldest retained
    const auto gap = b.snapshot_after(1);
    CHECK(gap.events.size() == 4 && gap.events.front().seq == 3);

    // a future cursor (stale client from a previous run) restarts from the
    // oldest retained event instead of returning nothing forever
    const auto future = b.snapshot_after(1000000);
    CHECK(future.events.size() == 4 && future.events.front().seq == 3);

    // max_events == 0: stats only
    const auto stats = b.snapshot_after(0, 0);
    CHECK(stats.events.empty());
    CHECK(stats.next_seq == 7 && stats.first_seq == 3);

    // max_events cap
    const auto capped = b.snapshot_after(0, 2);
    CHECK(capped.events.size() == 2 && capped.events[0].seq == 3 && capped.events[1].seq == 4);
}

static void test_event_serialization() {
    // queued
    {
        event ev  = make_event(event_kind::queued, 7, 1000);
        ev.seq    = 42;
        ev.lane   = 2; // fast
        ev.a      = 1234;
        ev.b      = 0;
        const json out = event_to_json(ev, 5);
        CHECK(out.at("v") == schema_version);
        CHECK(out.at("seq") == 42);
        CHECK(out.at("t_ms") == 1000 / 1000 + 5);
        CHECK(out.at("k") == "queued");
        CHECK(out.at("req") == 7);
        CHECK(out.at("lane") == "fast");
        CHECK(out.at("n_prompt") == 1234);
        CHECK(out.at("max_out") == 0);
        CHECK(!out.contains("slot"));   // slot -1 omitted
        CHECK(!out.contains("parent")); // parent 0 omitted
    }

    // deferred reason names
    {
        event ev = make_event(event_kind::deferred, 7);
        ev.code  = (uint16_t) defer_code::physical_admission;
        CHECK(event_to_json(ev).at("why") == "physical_admission");
    }

    // slot_selected
    {
        event ev = make_event(event_kind::slot_selected, 9);
        ev.slot  = 3;
        ev.code  = (uint16_t) select_code::lcp_affinity;
        ev.a     = 4096;
        ev.b     = 5000;
        ev.f     = 0.8192;
        const json out = event_to_json(ev);
        CHECK(out.at("slot") == 3);
        CHECK(out.at("sel") == "lcp_affinity");
        CHECK(out.at("lcp") == 4096);
        CHECK(out.at("n_task") == 5000);
        CHECK(out.at("sim").get<double>() > 0.8);
    }

    // cache_restore source names
    {
        event ev = make_event(event_kind::cache_restore, 9);
        ev.code  = (uint16_t) restore_code::disk;
        ev.a     = 30000;
        ev.b     = 123456789;
        ev.c     = 31000;
        ev.d     = -1;
        ev.f     = 8123.5;
        const json out = event_to_json(ev);
        CHECK(out.at("src") == "disk");
        CHECK(out.at("tokens") == 30000);
        CHECK(out.at("bytes") == 123456789);
        CHECK(!out.contains("donor_slot")); // d < 0 omitted
        CHECK(out.at("ms").get<double>() > 8000.0);
    }
    {
        event ev = make_event(event_kind::cache_restore, 9);
        ev.code  = (uint16_t) restore_code::donor;
        ev.d     = 2;
        CHECK(event_to_json(ev).at("src") == "donor");
        CHECK(event_to_json(ev).at("donor_slot") == 2);
    }

    // prompt_start: gap_why present only when n_past < lcp
    {
        event ev = make_event(event_kind::prompt_start, 11);
        ev.a     = 20000; // n_prompt
        ev.b     = 14000; // n_past landing
        ev.c     = 15000; // lcp
        ev.code  = (uint16_t) span_code::recompute_checkpoint_gap;
        const json out = event_to_json(ev);
        CHECK(out.at("n_prompt") == 20000);
        CHECK(out.at("n_past") == 14000);
        CHECK(out.at("lcp") == 15000);
        CHECK(out.at("gap_why") == "checkpoint_gap");
    }
    {
        event ev = make_event(event_kind::prompt_start, 11);
        ev.a     = 20000;
        ev.b     = 15000;
        ev.c     = 15000;
        CHECK(!event_to_json(ev).contains("gap_why"));
    }

    // admission
    {
        event ev = make_event(event_kind::admission, 12);
        ev.code  = (uint16_t) admission_code::ready_spanless;
        ev.a     = 100;
        ev.b     = 200;
        ev.c     = 210;
        ev.d     = 150;
        const json out = event_to_json(ev);
        CHECK(out.at("outcome") == "ready_spanless");
        CHECK(out.at("lcp") == 100);
        CHECK(out.at("resident") == 200);
        CHECK(out.at("frontier") == 210);
        CHECK(out.at("span_end") == 150);
    }
    {
        // full clear carries its attribution: why, and how much resident state
        // the clear destroyed (a nonzero lcp here means restore work was thrown
        // away, which the dashboard surfaces as wasted work)
        event ev = make_event(event_kind::admission, 12);
        ev.code  = (uint16_t) admission_code::ready_full_clear;
        ev.a     = (int64_t) clear_code::no_covering_checkpoint;
        ev.b     = 31000;
        ev.c     = 12000;
        ev.d     = 40000;
        const json out = event_to_json(ev);
        CHECK(out.at("outcome") == "ready_full_clear");
        CHECK(out.at("why") == "no_covering_checkpoint");
        CHECK(out.at("resident") == 31000);
        CHECK(out.at("lcp") == 12000);
        CHECK(out.at("span_end") == 40000);
        CHECK(!out.contains("frontier"));
    }
    {
        // every clear_code has a name (no "unknown" leaking into the feed)
        const std::vector<clear_code> all = {
            clear_code::empty_slot, clear_code::family, clear_code::no_lcp,
            clear_code::lcp_covers_task, clear_code::mtmd, clear_code::cache_reuse_opt,
            clear_code::no_cache_prompt, clear_code::no_pos_min,
            clear_code::no_covering_checkpoint, clear_code::rs_window,
        };
        for (const auto code : all) {
            event ev = make_event(event_kind::admission, 12);
            ev.code  = (uint16_t) admission_code::ready_full_clear;
            ev.a     = (int64_t) code;
            CHECK(event_to_json(ev).at("why") != "unknown");
        }
    }

    // decode_progress
    {
        event ev = make_event(event_kind::decode_progress, 13);
        ev.a     = 320;
        ev.b     = 400;
        ev.c     = 280;
        ev.d     = 16000000; // 16 s
        ev.f     = 21.5;
        const json out = event_to_json(ev);
        CHECK(out.at("n_dec") == 320);
        CHECK(out.at("draft_n") == 400);
        CHECK(out.at("draft_a") == 280);
        CHECK(out.at("ms").get<double>() == 16000.0);
        CHECK(out.at("tps").get<double>() == 21.5);
    }

    // finished
    {
        event ev = make_event(event_kind::finished, 14);
        ev.code  = (uint16_t) stop_code::eos;
        ev.a = 12000; ev.b = 11000; ev.c = 300; ev.d = 350; ev.e = 260;
        ev.f = 4200.0; ev.g = 15000.0;
        const json out = event_to_json(ev);
        CHECK(out.at("stop") == "eos");
        CHECK(out.at("n_prompt") == 12000);
        CHECK(out.at("n_cached") == 11000);
        CHECK(out.at("n_dec") == 300);
        CHECK(out.at("draft_n") == 350);
        CHECK(out.at("draft_a") == 260);
        CHECK(out.at("pp_ms").get<double>() == 4200.0);
        CHECK(out.at("tg_ms").get<double>() == 15000.0);
    }

    // terminal: registry reason codes stay numeric AND named (incl. run_stall)
    {
        event ev = make_event(event_kind::terminal, 15);
        ev.code  = 504;
        ev.a     = 6; // lifecycle::timed_out
        const json out = event_to_json(ev);
        CHECK(out.at("state") == "timed_out");
        CHECK(out.at("reason") == "run_stall");
        CHECK(out.at("code") == 504);
    }

    // cache_op
    {
        event ev = make_event(event_kind::cache_op, 0);
        ev.code  = (uint16_t) cache_op_code::spill;
        ev.a     = 50000;
        ev.b     = 2147483648;
        ev.f     = 1900.0;
        const json out = event_to_json(ev);
        CHECK(out.at("op") == "spill");
        CHECK(out.at("tokens") == 50000);
        CHECK(out.at("bytes") == 2147483648);
        CHECK(!out.contains("req")); // request 0 omitted
    }
}

static void test_sse_frames() {
    event ev = make_event(event_kind::queued, 3, 2000);
    bus b(8);
    b.emit(ev);
    const auto snap = b.snapshot_after(0);
    CHECK(snap.events.size() == 1);

    const std::string frame = make_sse_frame(snap.events[0], 0);
    CHECK(frame.rfind("id: 1\n", 0) == 0);
    CHECK(frame.find("\ndata: {") != std::string::npos);
    CHECK(frame.size() >= 2 && frame.substr(frame.size() - 2) == "\n\n");

    const std::string hello = make_hello_frame(snap, 5000, 1754600000000ull);
    CHECK(hello.rfind("event: hello\n", 0) == 0);
    const auto body = hello.substr(hello.find("data: ") + 6);
    const json parsed = json::parse(body);
    CHECK(parsed.at("k") == "hello");
    CHECK(parsed.at("v") == schema_version);
    CHECK(parsed.at("first_seq") == 1);
    CHECK(parsed.at("next_seq") == 2);
    CHECK(parsed.at("wall_ms") == 1754600000000ull);
}

static void test_cache_state_json() {
    cache_state state;
    state.enabled          = true;
    state.at_us            = 10 * 1000 * 1000;
    state.limit_ram_bytes  = 8ull << 30;
    state.limit_disk_bytes = 400ull << 30;
    state.used_ram_bytes   = 4ull << 30;
    state.used_disk_bytes  = 100ull << 30;
    state.tokens_total     = 123456;
    state.counters.lookups       = 10;
    state.counters.hits_entry    = 4;
    state.counters.hits_resident = 3;
    state.counters.misses        = 3;
    state.counters.spills        = 2;
    state.counters.disk_loads    = 1;

    cache_entry_state ram_entry;
    ram_entry.id         = 1;
    ram_entry.tokens     = 1000;
    ram_entry.bytes_ram  = 1ull << 30;
    ram_entry.created_us = 4 * 1000 * 1000;
    ram_entry.hits       = 0;
    state.entries.push_back(ram_entry);

    cache_entry_state disk_entry;
    disk_entry.id          = 2;
    disk_entry.tokens      = 2000;
    disk_entry.bytes_disk  = 2ull << 30;
    disk_entry.on_disk     = true;
    disk_entry.created_us  = 2 * 1000 * 1000;
    disk_entry.last_hit_us = 9 * 1000 * 1000;
    disk_entry.hits        = 5;
    disk_entry.file        = "pc-abc-1.lcpc";
    state.entries.push_back(disk_entry);

    const uint64_t now_us = 12 * 1000 * 1000;
    const json out = cache_state_to_json(state, now_us);

    CHECK(out.at("v") == schema_version);
    CHECK(out.at("enabled") == true);
    CHECK(out.at("age_ms") == 2000);
    CHECK(out.at("ram").at("used_bytes") == (4ull << 30));
    CHECK(out.at("ram").at("limit_bytes") == (8ull << 30));
    CHECK(out.at("disk").at("used_bytes") == (100ull << 30));
    CHECK(out.at("disk").at("limit_bytes") == (400ull << 30));
    CHECK(out.at("counters").at("lookups") == 10);
    CHECK(out.at("counters").at("hits_entry") == 4);
    CHECK(out.at("entries").size() == 2);

    const auto & ram_row = out.at("entries").at(0);
    CHECK(ram_row.at("tier") == "ram");
    CHECK(ram_row.at("bytes") == (1ull << 30));
    CHECK(ram_row.at("age_s").get<double>() == 8.0);
    CHECK(ram_row.at("last_hit_s").is_null());
    CHECK(!ram_row.contains("file"));

    const auto & disk_row = out.at("entries").at(1);
    CHECK(disk_row.at("tier") == "disk");
    CHECK(disk_row.at("bytes") == (2ull << 30));
    CHECK(disk_row.at("last_hit_s").get<double>() == 3.0);
    CHECK(disk_row.at("hits") == 5);
    CHECK(disk_row.at("file") == "pc-abc-1.lcpc");

    // default state serializes as disabled
    const json off = cache_state_to_json(cache_state{}, now_us);
    CHECK(off.at("enabled") == false);
    CHECK(off.at("entries").empty());
}

static void test_cache_state_publish() {
    bus b(8);
    CHECK(b.cache().enabled == false);

    cache_state state;
    state.enabled      = true;
    state.at_us        = 77;
    state.tokens_total = 5;
    b.publish_cache_state(std::move(state));

    const auto out = b.cache();
    CHECK(out.enabled == true);
    CHECK(out.at_us == 77);
    CHECK(out.tokens_total == 5);
}

static void test_stream_leases() {
    std::vector<std::shared_ptr<void>> leases;
    for (size_t i = 0; i < maximum_live_streams; ++i) {
        auto lease = try_acquire_stream_lease();
        CHECK(lease != nullptr);
        leases.push_back(std::move(lease));
    }
    CHECK(active_streams() == maximum_live_streams);
    CHECK(try_acquire_stream_lease() == nullptr);
    leases.pop_back();
    CHECK(active_streams() == maximum_live_streams - 1);
    auto again = try_acquire_stream_lease();
    CHECK(again != nullptr);
    leases.clear();
    again.reset();
    CHECK(active_streams() == 0);
}

// The emit path runs on the main loop and under the queue mutex, so its cost
// is a correctness-adjacent property, not a nice-to-have. Measure it rather
// than asserting it in prose. The bound is deliberately loose (it only has to
// catch a change of ALGORITHM - an allocation, a format, a syscall - not
// machine-to-machine variation).
static void test_emit_overhead() {
    bus local(ring_capacity_default);
    const int n = 200000;

    event ev;
    ev.kind       = event_kind::decode_progress;
    ev.request_id = 42;
    ev.slot       = 1;
    ev.at_us      = 1; // skip the clock read: measure the ring append itself

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < n; ++i) {
        local.emit(ev);
    }
    const int64_t dt = ggml_time_us() - t0;

    const double ns_per_emit = (double) dt * 1000.0 / n;
    std::printf("  emit: %.1f ns/event (%d events in %.1f ms)\n", ns_per_emit, n, dt / 1000.0);

    // an allocation or a JSON dump on this path would be orders of magnitude
    // above this; a lock + 96-byte copy is tens of ns
    CHECK(ns_per_emit < 2000.0);

    // the ring never grows: capacity events retained, the rest counted dropped
    const auto snap = local.snapshot_after(0, ring_capacity_default * 2);
    CHECK(snap.events.size() == ring_capacity_default);
    CHECK(snap.dropped == (uint64_t) n - ring_capacity_default);
}

int main() {
    test_ring_basics();
    test_event_serialization();
    test_sse_frames();
    test_cache_state_json();
    test_cache_state_publish();
    test_stream_leases();
    test_emit_overhead();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all server-dashboard-bus tests passed\n");
    return 0;
}

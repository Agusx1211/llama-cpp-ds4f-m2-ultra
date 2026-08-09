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

// ---------------------------------------------------------------------------
// dashboard v3: content store, live token mirror, cache preview
// ---------------------------------------------------------------------------

static std::vector<int32_t> ids(size_t n, int32_t base = 100) {
    std::vector<int32_t> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(base + (int32_t) i);
    }
    return out;
}

// the tests never link llama, so the detokenizer is a stand-in that makes the
// token ids visible in the output
static std::string fake_detok(const std::vector<int32_t> & tokens) {
    std::string out;
    for (const int32_t t : tokens) {
        if (!out.empty()) {
            out += ' ';
        }
        out += (t < 0 ? std::string("[media]") : std::to_string(t));
    }
    return out;
}

static void test_utf8_truncation() {
    // "aé€𝄞" - 1, 2, 3 and 4 byte sequences back to back
    const std::string s = "a\xc3\xa9\xe2\x82\xac\xf0\x9d\x84\x9e";
    CHECK(s.size() == 10);

    // never cut inside a sequence
    CHECK(utf8_prefix_len(s, 10) == 10);
    CHECK(utf8_prefix_len(s, 9) == 6);  // would land inside the 4-byte char
    CHECK(utf8_prefix_len(s, 5) == 3);  // would land inside the 3-byte char
    CHECK(utf8_prefix_len(s, 2) == 1);  // would land inside the 2-byte char
    CHECK(utf8_prefix_len(s, 0) == 0);

    // the suffix start moves FORWARD past a partial sequence: a tail never
    // begins mid-character, and it is shorter than the budget when it has to be
    CHECK(utf8_suffix_start(s, 10) == 0);
    CHECK(utf8_suffix_start(s, 4) == 6);  // exactly the 4-byte char
    CHECK(utf8_suffix_start(s, 3) == 10); // cannot fit it: empty tail, not garbage
    CHECK(utf8_suffix_start(s, 1) == 10);
}

static void test_content_capture_and_caps() {
    content_store store;
    store.set_enabled(true);

    // The complete production-sized context is retained. The old head/tail
    // response shape remains, but the tail is empty because the head now holds
    // the whole target prompt.
    const auto full_prompt = ids(content_in_head_tokens);
    const int64_t capture_start = ggml_time_us();
    capture_prompt_into(store, 1, 3, full_prompt);
    const int64_t capture_us = ggml_time_us() - capture_start;

    request_content got;
    CHECK(store.lookup(1, got));
    CHECK(got.has_input);
    CHECK(got.in_tokens == content_in_head_tokens);
    CHECK(got.in_head.size() == content_in_head_tokens);
    CHECK(got.in_tail.empty());
    CHECK(got.in_head.front() == 100);
    CHECK(got.in_head.back() == 100 + (int32_t) content_in_head_tokens - 1);
    CHECK(got.complete == false); // no output yet: the request is still live
    std::printf("  full dashboard prompt capture: %.3f ms (%u tokens)\n",
                capture_us / 1000.0, content_in_head_tokens);

    // A generously target-sized generation is also retained whole.
    const std::string target_out(1u << 20, 'x');
    store.capture_output(1, 3, target_out, 32768);
    CHECK(store.lookup(1, got));
    CHECK(got.complete);
    CHECK(got.out_tokens == 32768);
    CHECK(got.out_bytes == target_out.size());
    CHECK(got.out_head == target_out);
    CHECK(got.out_tail.empty());
    const json target_body = content_to_json(1, &got, nullptr, store.usage(), fake_detok);
    CHECK(target_body["input"]["head_tokens"] == content_in_head_tokens);
    CHECK(target_body["input"]["tail_tokens"] == 0);
    CHECK(target_body["input"]["elided_tokens"] == 0);
    CHECK(target_body["input"]["truncated"] == false);
    CHECK(target_body["output"]["elided_bytes"] == 0);
    CHECK(target_body["output"]["truncated"] == false);

    // Inputs outside the target envelope still fail honestly: they are
    // bounded, not silently described as complete.
    capture_prompt_into(store, 3, 1, ids(content_in_head_tokens + 500));
    CHECK(store.lookup(3, got));
    CHECK(got.in_head.size() == content_in_head_tokens);
    CHECK(got.in_tail.empty());
    CHECK(got.in_tokens == content_in_head_tokens + 500);
    const json oversized_body = content_to_json(3, &got, nullptr, store.usage(), fake_detok);
    CHECK(oversized_body["input"]["elided_tokens"] == 500);
    CHECK(oversized_body["input"]["truncated"] == true);

    const std::string oversized_out(content_out_head_bytes + 4096, 'q');
    store.capture_output(4, 1, oversized_out, 1000000);
    CHECK(store.lookup(4, got));
    CHECK(got.out_head.size() == content_out_head_bytes);
    CHECK(got.out_tail.empty());
    const json oversized_output_body = content_to_json(4, &got, nullptr, store.usage(), fake_detok);
    CHECK(oversized_output_body["output"]["elided_bytes"] == 4096);
    CHECK(oversized_output_body["output"]["truncated"] == true);

    // a short generation is kept whole with no tail
    store.capture_output(2, 0, "short answer", 3);
    CHECK(store.lookup(2, got));
    CHECK(got.out_head == "short answer");
    CHECK(got.out_tail.empty());

    // request id 0 is never a runtime id and must not create an entry
    store.capture_output(0, 0, "nope", 1);
    CHECK(!store.lookup(0, got));
}

static void test_content_eviction_bounds() {
    content_store store;
    store.set_enabled(true);

    // one live request, then far more finished ones than the cap
    capture_prompt_into(store, 1, 0, ids(64));
    for (uint64_t r = 2; r < content_max_requests * 2; ++r) {
        capture_prompt_into(store, r, 1, ids(64));
        store.capture_output(r, 1, std::string(2048, 'y'), 100);
    }

    const auto usage = store.usage();
    CHECK(usage.requests <= content_max_requests);
    CHECK(usage.bytes <= content_max_bytes);
    CHECK(usage.evicted > 0);

    // the live request survives FIFO pressure: it is the OLDEST record, and it
    // is the one an operator is most likely to be watching right now
    request_content got;
    CHECK(store.lookup(1, got));
    CHECK(!got.complete);

    // an evicted id is reported as evicted, an unseen one as merely absent
    CHECK(store.was_evicted(2));
    CHECK(!store.was_evicted(999999));

    // once the live request reaches a terminal state it becomes evictable
    store.mark_final(1);
    for (uint64_t r = 1000; r < 1000 + content_max_requests; ++r) {
        capture_prompt_into(store, r, 1, ids(64));
        store.capture_output(r, 1, std::string(2048, 'z'), 100);
    }
    CHECK(!store.lookup(1, got));
    CHECK(store.usage().requests <= content_max_requests);

    // the hard ceiling still applies when nothing ever completes
    content_store live_only;
    live_only.set_enabled(true);
    for (uint64_t r = 1; r <= content_max_requests * 3; ++r) {
        capture_prompt_into(live_only, r, 1, ids(64));
    }
    CHECK(live_only.usage().requests <= content_max_requests * 2 + 1);
}

static void test_content_disable_clears_everything() {
    content_store store;
    store.set_enabled(true);
    capture_prompt_into(store, 7, 0, ids(8));
    store.capture_output(7, 0, "text", 1);
    CHECK(store.usage().requests == 1);

    store.set_enabled(false);
    request_content got;
    CHECK(!store.enabled());
    CHECK(!store.lookup(7, got));
    CHECK(store.usage().requests == 0);
    CHECK(store.usage().bytes == 0);

    // and nothing new is retained while it stays off
    capture_prompt_into(store, 8, 0, ids(8));
    store.capture_output(8, 0, "text", 1);
    CHECK(store.usage().requests == 0);
}

static void test_content_json() {
    content_store store;
    store.set_enabled(true);
    capture_prompt_into(store, 11, 2, ids(4096));
    store.capture_output(11, 2, "generated", 4);

    request_content got;
    CHECK(store.lookup(11, got));
    const json body = content_to_json(11, &got, nullptr, store.usage(), fake_detok);

    CHECK(body["v"] == schema_version);
    CHECK(body["req"] == 11);
    CHECK(body["found"] == true);
    CHECK(body["state"] == "final");
    CHECK(body["slot"] == 2);
    CHECK(body["caps"]["in_head_tokens"] == content_in_head_tokens);
    CHECK(body["caps"]["max_requests"] == content_max_requests);
    CHECK(body["caps"]["max_bytes"] == content_max_bytes);
    CHECK(body["store"]["requests"] == 1);
    CHECK(body["input"]["n_tokens"] == 4096);
    CHECK(body["input"]["head_tokens"] == 4096);
    CHECK(body["input"]["tail_tokens"] == 0);
    CHECK(body["input"]["elided_tokens"] == 0);
    CHECK(body["input"]["truncated"] == false);
    CHECK(body["input"]["head"].get<std::string>().rfind("100 101", 0) == 0);
    CHECK(body["output"]["head"] == "generated");
    CHECK(body["output"]["truncated"] == false);

    // the not-retained body carries a reason and NO content keys at all
    const json missing = content_to_json(12, nullptr, "evicted", store.usage(), fake_detok);
    CHECK(missing["found"] == false);
    CHECK(missing["reason"] == "evicted");
    CHECK(!missing.contains("input"));
    CHECK(!missing.contains("output"));

    // a media placeholder is named, never handed to the vocab
    request_content media;
    media.request_id = 13;
    media.has_input  = true;
    media.in_tokens  = 3;
    media.in_head    = { 5, -1, 6 };
    const json with_media = content_to_json(13, &media, nullptr, store.usage(), fake_detok);
    CHECK(with_media["input"]["head"] == "5 [media] 6");
}

static void test_watch_mirror() {
    watch_registry reg;
    reg.reset();
    CHECK(!watching());

    const int a = reg.attach(77);
    CHECK(a >= 0);
    CHECK(watching());
    CHECK(reg.active() == 1);

    // the cap is deliberately tighter than the event feed's: this is the only
    // per-token hook in the server
    const int b = reg.attach(78);
    CHECK(b >= 0);
    CHECK(reg.attach(79) < 0);
    reg.detach(b);
    CHECK(reg.active() == 1);

    // the producer hands over the whole generated text; only the new suffix moves
    watch_view view;
    reg.sync(77, 4, "Hello", 1);
    CHECK(reg.drain(a, view));
    CHECK(view.text == "Hello");
    CHECK(view.cursor == 5);
    CHECK(view.offset == 0);
    CHECK(view.replace); // the first drain after attach is authoritative
    CHECK(view.slot == 4);

    reg.sync(77, 4, "Hello world", 2);
    CHECK(reg.drain(a, view));
    CHECK(view.text == " world");
    CHECK(!view.replace);
    CHECK(view.offset == 5);
    CHECK(view.cursor == 11);

    // nothing new: an empty, non-terminal frame
    CHECK(reg.drain(a, view));
    CHECK(view.text.empty());
    CHECK(!view.ended);

    // a stop-word trim shortens the slot's own text: the reader is resynced
    reg.sync(77, 4, "Hello wor", 2);
    CHECK(reg.drain(a, view));
    CHECK(view.replace);
    CHECK(view.text == "Hello wor");
    CHECK(view.offset == 0);

    // a different request's tokens never reach this watcher
    reg.sync(999, 5, "not for you", 9);
    CHECK(reg.drain(a, view));
    CHECK(view.text.empty());

    reg.finish(77, 512);
    CHECK(reg.drain(a, view));
    CHECK(view.ended);
    CHECK(view.n_dec == 512);

    reg.detach(a);
    CHECK(!watching());
    CHECK(!reg.drain(a, view));
}

static void test_watch_mirror_is_bounded() {
    watch_registry reg;
    reg.reset();
    const int lease = reg.attach(5);
    CHECK(lease >= 0);

    // a reader that never drains must not make the server grow: the mirror
    // keeps the newest window and says how much it discarded
    std::string generated;
    while (generated.size() < watch_mirror_bytes * 3) {
        generated += "0123456789";
        reg.sync(5, 0, generated, (int32_t) (generated.size() / 4));
    }
    watch_view view;
    CHECK(reg.drain(lease, view));
    CHECK(view.text.size() <= watch_mirror_bytes);
    CHECK(view.dropped > 0);
    CHECK(view.replace); // a trimmed reader cannot splice; it must resync
    CHECK(view.offset == view.dropped);
    CHECK(view.cursor == generated.size());
    reg.detach(lease);
}

// The mirror hook sits in process_token, i.e. on the per-token path. When
// nobody is watching it must cost a single relaxed load - measure it rather
// than assert it, exactly like test_emit_overhead does for the bus.
static void test_watch_guard_overhead() {
    watches().reset();
    const int n = 5000000;
    volatile int sink = 0;

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < n; ++i) {
        if (watching()) {
            sink += 1;
        }
    }
    const int64_t dt = ggml_time_us() - t0;
    const double ns = (double) dt * 1000.0 / n;
    std::printf("  watch guard (unwatched): %.2f ns/token (%d iterations in %.1f ms)\n", ns, n, dt / 1000.0);
    CHECK(sink == 0);
    // a relaxed atomic load is sub-nanosecond; anything that takes a lock or
    // allocates here would be orders of magnitude above this bound
    CHECK(ns < 50.0);
}

static void test_watch_frames() {
    watch_view view;
    view.req    = 9;
    view.slot   = 2;
    view.n_dec  = 12;
    view.cursor = 40;
    view.offset = 30;
    view.text   = " more";

    const std::string delta = make_watch_frame(view);
    CHECK(delta.rfind("data: ", 0) == 0);
    const json d = json::parse(delta.substr(6));
    CHECK(d["k"] == "delta");
    CHECK(d["cursor"] == 40);   // delta: END offset
    CHECK(d["offset"] == 30);
    CHECK(d["text"] == " more");

    view.replace = true;
    const json r = json::parse(make_watch_frame(view).substr(6));
    CHECK(r["k"] == "rewind");
    CHECK(r["cursor"] == 30);   // rewind: START offset
    CHECK(r["offset"] == 30);

    const std::string hello = make_watch_hello_frame(9, view, "live", json(nullptr));
    CHECK(hello.rfind("event: watch-hello\ndata: ", 0) == 0);
    const json h = json::parse(hello.substr(std::string("event: watch-hello\ndata: ").size()));
    CHECK(h["k"] == "watch-hello");
    CHECK(h["state"] == "live");
    CHECK(h["input"].is_null());
    CHECK(h["caps"]["max_streams"] == maximum_watch_streams);
    CHECK(h["caps"]["mirror_bytes"] == watch_mirror_bytes);

    const std::string end = make_watch_end_frame(9, "finished", 512);
    CHECK(end.rfind("event: watch-end\ndata: ", 0) == 0);
    const json e = json::parse(end.substr(std::string("event: watch-end\ndata: ").size()));
    CHECK(e["k"] == "end");
    CHECK(e["reason"] == "finished");
    CHECK(e["n_dec"] == 512);
}

static void test_cache_preview() {
    cache_state state;
    state.enabled = true;
    state.at_us   = 1000;

    cache_entry_state ram;
    ram.id         = 4;
    ram.tokens     = 100000;
    ram.bytes_ram  = 1 << 20;
    ram.request_id = 88;
    ram.head_ids   = ids(cache_preview_head_tokens, 10);
    ram.tail_ids   = ids(cache_preview_tail_tokens, 900);
    state.entries.push_back(ram);

    cache_entry_state bare; // an entry with no retained tokens
    bare.id      = 5;
    bare.tokens  = 12;
    bare.on_disk = true;
    state.entries.push_back(bare);

    // the bulk snapshot advertises previewability and provenance but carries
    // NO text at all
    const json snap = cache_state_to_json(state, 2000);
    CHECK(snap["entries"][0]["preview"] == true);
    CHECK(snap["entries"][0]["req"] == 88);
    CHECK(snap["entries"][1]["preview"] == false);
    CHECK(!snap["entries"][1].contains("req"));
    const std::string dumped = snap.dump();
    CHECK(dumped.find("\"head\"") == std::string::npos);
    CHECK(dumped.find("\"tail\"") == std::string::npos);

    const json preview = cache_preview_to_json(4, &state.entries[0], 2000, state.at_us, fake_detok);
    CHECK(preview["found"] == true);
    CHECK(preview["tier"] == "ram");
    CHECK(preview["tokens"] == 100000);
    CHECK(preview["head_tokens"] == cache_preview_head_tokens);
    CHECK(preview["tail_tokens"] == cache_preview_tail_tokens);
    CHECK(preview["elided_tokens"] == 100000 - cache_preview_head_tokens - cache_preview_tail_tokens);
    CHECK(preview["truncated"] == true);
    CHECK(preview["req"] == 88);
    CHECK(preview["age_ms"] == 1);
    CHECK(preview["head"].get<std::string>().rfind("10 11", 0) == 0);

    // no tokens retained, or no such entry: not previewable, and no empty
    // strings pretending to be content
    const json bare_json = cache_preview_to_json(5, &state.entries[1], 2000, state.at_us, fake_detok);
    CHECK(bare_json["found"] == false);
    CHECK(!bare_json.contains("head"));
    const json missing = cache_preview_to_json(99, nullptr, 2000, state.at_us, fake_detok);
    CHECK(missing["found"] == false);
}

int main() {
    test_ring_basics();
    test_event_serialization();
    test_sse_frames();
    test_cache_state_json();
    test_cache_state_publish();
    test_stream_leases();
    test_emit_overhead();
    // v3
    test_utf8_truncation();
    test_content_capture_and_caps();
    test_content_eviction_bounds();
    test_content_disable_clears_everything();
    test_content_json();
    test_watch_mirror();
    test_watch_mirror_is_bounded();
    test_watch_guard_overhead();
    test_watch_frames();
    test_cache_preview();

    if (failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all server-dashboard-bus tests passed\n");
    return 0;
}

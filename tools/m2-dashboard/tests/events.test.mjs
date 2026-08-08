// Tests for lib/events.mjs: folding the /m2-dashboard/events stream into
// request models, the restored-vs-recomputed span derivation, waterfall time
// segments, cache-state shaping, and the reconnecting stream wrapper.

import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import {
    EventStore,
    LiveEventStream,
    MAX_REQUESTS,
    STREAM_FALLBACK_AFTER,
    cacheView,
    hitRateSeries,
    phaseBreakdown,
    timeSegments,
    tokenSpans,
    wastedRestore,
} from "../lib/events.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(readFileSync(join(here, "../fixtures/events-v2.json"), "utf8"));
const captured = JSON.parse(readFileSync(join(here, "../fixtures/captured-v2.json"), "utf8"));

function feedAll(store, events) {
    for (const evt of events) assert.equal(store.feed(evt), true, "event accepted: " + JSON.stringify(evt).slice(0, 80));
}

test("folds a full request lifecycle from the fixture stream", () => {
    const store = new EventStore();
    feedAll(store, fixture.events);

    // the fixture's fast chat turn: resident reuse, finished eos
    const req = store.requests.get(102);
    assert.ok(req);
    assert.equal(req.lane, "fast");
    assert.equal(req.slot, 1);
    assert.equal(req.promptTokens, 14350);
    assert.ok(req.queuedAt < req.dispatchedAt);
    assert.equal(req.select.sel, "lcp_affinity");
    assert.equal(req.restore.src, "resident");
    assert.equal(req.admission.outcome, "ready_spanless");
    assert.equal(req.promptStart.nPast, 14190);
    assert.equal(req.finished.stop, "eos");
    assert.equal(req.terminal.state, "completed");

    // the run_stall kill carries the registry reason
    const stalled = store.requests.get(106);
    assert.equal(stalled.terminal.reason, "run_stall");
    assert.equal(stalled.terminal.code, 504);
    assert.equal(stalled.deferred[0].why, "physical_admission");

    // the live agent turn has no terminal and rows() puts live first
    const agent = store.requests.get(101);
    assert.equal(agent.terminal, null);
    const rows = store.rows();
    assert.ok(rows.findIndex((r) => r.id === 101) < rows.findIndex((r) => r.id === 102));

    // cache ops aggregate server-side traffic
    assert.equal(store.cacheOps.spill, 1);
    assert.equal(store.cacheOps.disk_load, 2); // req 103 restore + req 108 wasted restore
    assert.ok(store.cacheOps.bytes_spilled > 0);
    assert.ok(store.cacheOps.bytes_disk_load > 0);
});

test("rejects duplicates, stale sequences, and unknown schema versions", () => {
    const store = new EventStore();
    const evt = { v: 1, seq: 5, t_ms: 1000, k: "queued", req: 1, lane: "normal", n_prompt: 10, max_out: 0 };
    assert.equal(store.feed(evt), true);
    assert.equal(store.feed(evt), false); // duplicate seq
    assert.equal(store.feed({ ...evt, seq: 4 }), false); // stale
    assert.equal(store.feed({ ...evt, seq: 6, v: 99 }), false); // future schema
    assert.equal(store.feed({ ...evt, seq: 7, k: "brand_new_kind" }), false); // unknown kind skipped
    assert.equal(store.lastSeq, 7); // but the cursor still advances past it
});

test("hello frames set clock skew and detect gaps", () => {
    const store = new EventStore();
    assert.equal(store.feed({ k: "hello", v: 1, first_seq: 1, next_seq: 1, dropped: 0, now_ms: Date.now() + 5000 }), true);
    assert.ok(Math.abs(store.clockSkewMs - 5000) < 1000);
    store.feed({ v: 1, seq: 10, t_ms: 1, k: "queued", req: 1, lane: "low", n_prompt: 1, max_out: 0 });
    assert.equal(store.gapped, false);
    store.feed({ k: "hello", v: 1, first_seq: 400, next_seq: 500, dropped: 390 });
    assert.equal(store.gapped, true); // reconnect landed beyond our cursor
    assert.equal(store.dropped, 390);
});

test("prunes terminal requests first once the model cap is hit", () => {
    const store = new EventStore();
    let seq = 0;
    for (let i = 1; i <= MAX_REQUESTS + 20; i++) {
        store.feed({ v: 1, seq: ++seq, t_ms: i, k: "queued", req: i, lane: "low", n_prompt: 5, max_out: 0 });
        if (i !== 3) { // request 3 stays live forever
            store.feed({ v: 1, seq: ++seq, t_ms: i, k: "terminal", req: i, state: "completed", reason: "completed", code: 600 });
        }
    }
    assert.ok(store.requests.size <= MAX_REQUESTS);
    assert.ok(store.requests.has(3), "live request survives pruning");
    assert.ok(!store.requests.has(1), "oldest terminal request pruned");
});

// ---------------------------------------------------------------------------
// tokenSpans — the restore-vs-recompute derivation
// ---------------------------------------------------------------------------

function reqWith(promptStart, restore = null) {
    return {
        promptStart, restore,
        deferred: [], prefill: [], decode: [], finished: null, terminal: null,
    };
}

test("spans: RAM restore with checkpoint landing gap and divergent tail", () => {
    const spans = tokenSpans(reqWith(
        { t: 0, nPrompt: 281_400, nPast: 210_250, lcp: 212_300, gapWhy: "checkpoint_gap" },
        { src: "ram", tokens: 212_300, bytes: 1, ms: 8000 },
    ));
    assert.deepEqual(spans.map((s) => [s.start, s.end, s.kind, s.src ?? s.why]), [
        [0, 210_250, "restored", "ram"],
        [210_250, 212_300, "recomputed", "checkpoint_gap"],
        [212_300, 281_400, "recomputed", "divergent_tail"],
    ]);
    // spans tile [0, nPrompt) exactly
    assert.equal(spans.reduce((acc, s) => acc + s.tokens, 0), 281_400);
});

test("spans: cold request is one no_cache span", () => {
    const spans = tokenSpans(reqWith({ t: 0, nPrompt: 52_400, nPast: 0, lcp: 0 }, { src: "miss", tokens: 0, bytes: 0, ms: 1 }));
    assert.deepEqual(spans, [{ start: 0, end: 52_400, tokens: 52_400, kind: "recomputed", why: "no_cache" }]);
});

test("spans: full-prefix match re-evaluates exactly one token", () => {
    const spans = tokenSpans(reqWith(
        { t: 0, nPrompt: 1000, nPast: 999, lcp: 1000, gapWhy: "last_token" },
        { src: "resident", tokens: 1000, bytes: 0, ms: 0.5 },
    ));
    assert.deepEqual(spans.map((s) => [s.tokens, s.kind, s.src ?? s.why]), [
        [999, "restored", "resident"],
        [1, "recomputed", "last_token"],
    ]);
});

test("spans: donor share and SSD restore carry their sources", () => {
    const donor = tokenSpans(reqWith(
        { t: 0, nPrompt: 97_200, nPast: 96_100, lcp: 96_100 },
        { src: "donor", tokens: 96_100, bytes: 0, ms: 14, donorSlot: 0 },
    ));
    assert.equal(donor[0].src, "donor");
    assert.equal(donor[1].why, "divergent_tail");

    const disk = tokenSpans(reqWith(
        { t: 0, nPrompt: 31_900, nPast: 31_000, lcp: 31_000 },
        { src: "disk", tokens: 31_000, bytes: 9, ms: 872 },
    ));
    assert.equal(disk[0].src, "disk");
});

test("spans: reset landing recomputes everything below the old prefix", () => {
    const spans = tokenSpans(reqWith(
        { t: 0, nPrompt: 5000, nPast: 0, lcp: 4200, gapWhy: "reset" },
        { src: "resident", tokens: 4200, bytes: 0, ms: 1 },
    ));
    assert.deepEqual(spans.map((s) => [s.tokens, s.kind, s.why ?? s.src]), [
        [4200, "recomputed", "reset"],
        [800, "recomputed", "divergent_tail"],
    ]);
});

test("spans: no prompt_start yet means no spans", () => {
    assert.deepEqual(tokenSpans(reqWith(null)), []);
});

// ---------------------------------------------------------------------------
// timeSegments
// ---------------------------------------------------------------------------

test("segments: finished request splits at the exact decode start", () => {
    const req = {
        queuedAt: 1000, dispatchedAt: 3000,
        promptStart: { t: 3500, nPrompt: 100, nPast: 0, lcp: 0 },
        restore: { src: "ram", tokens: 50, bytes: 1, ms: 400 },
        prefill: [], decode: [],
        finished: { t: 10_000, tgMs: 4000, ppMs: 2500, stop: "eos", nDecoded: 80, nCached: 0, nPrompt: 100, draftN: 0, draftA: 0 },
        terminal: { t: 10_001, state: "completed", reason: "completed", code: 600 },
        deferred: [], lastAt: 10_001,
    };
    const seg = timeSegments(req, 99_999);
    assert.equal(seg.live, false);
    assert.deepEqual(seg.segments.map((s) => [s.kind, s.t0, s.t1]), [
        ["queued", 1000, 3000],
        ["restore", 3000, 3500],
        ["prefill", 3500, 6000],   // finished.t - tgMs = 6000
        ["decode", 6000, 10_001],
    ]);
    assert.equal(seg.segments[1].src, "ram");
});

test("segments: live prefilling request extends to now", () => {
    const req = {
        queuedAt: 1000, dispatchedAt: 1200,
        promptStart: { t: 1300, nPrompt: 100, nPast: 0, lcp: 0 },
        restore: null, prefill: [{ t: 2000, done: 50, total: 100, batch: 50, ms: 700 }],
        decode: [], finished: null, terminal: null, deferred: [], lastAt: 2000,
    };
    const seg = timeSegments(req, 5000);
    assert.equal(seg.live, true);
    assert.equal(seg.segments[seg.segments.length - 1].kind, "prefill");
    assert.equal(seg.segments[seg.segments.length - 1].t1, 5000);
});

test("segments: queued-only request is a single waiting bar", () => {
    const req = { queuedAt: 1000, dispatchedAt: null, promptStart: null, restore: null, prefill: [], decode: [], finished: null, terminal: null, deferred: [{ t: 1500, why: "physical_admission" }], lastAt: 1500 };
    const seg = timeSegments(req, 9000);
    assert.equal(seg.segments.length, 1);
    assert.deepEqual([seg.segments[0].kind, seg.segments[0].t0, seg.segments[0].t1], ["queued", 1000, 9000]);
});

// ---------------------------------------------------------------------------
// cache view + hit rate
// ---------------------------------------------------------------------------

test("cacheView shapes the snapshot and computes the hit rate", () => {
    const view = cacheView(fixture.cache_state);
    assert.equal(view.enabled, true);
    assert.equal(view.ram.limit, 8_589_934_592);
    assert.ok(Math.abs(view.ram.fraction - 0.75) < 0.01);
    // hit rate counts entry hits + resident wins over all consults
    assert.ok(Math.abs(view.hitRate - 6 / 7) < 1e-9);
    assert.equal(view.entries.length, 7);
    const disk = view.entries.find((e) => e.id === 3);
    assert.equal(disk.tier, "disk");
    assert.equal(disk.file, "pc-4fca21-3.lcpc");
    assert.equal(cacheView(null), null);
    assert.equal(cacheView({ v: 99 }), null);
});

test("hitRateSeries is a rolling window over consults", () => {
    const lookups = [
        { t: 1, hit: 1 }, { t: 2, hit: 0 }, { t: 3, hit: 1 }, { t: 4, hit: 1 },
    ];
    const points = hitRateSeries(lookups, { window: 2 });
    assert.deepEqual(points.map((p) => p.v), [1, 0.5, 0.5, 1]);
});

// ---------------------------------------------------------------------------
// LiveEventStream reconnect behavior (no sockets: fake transport via fetch)
// ---------------------------------------------------------------------------

function fakeFetchScript(script) {
    // each entry: {status} to reject, or {frames: [..]} to stream then end
    let call = 0;
    return async () => {
        const step = script[Math.min(call++, script.length - 1)];
        if (step.status) {
            return { ok: false, status: step.status, statusText: "x", body: null };
        }
        const chunks = step.frames.map((f) => new TextEncoder().encode(f));
        let i = 0;
        return {
            ok: true, status: 200, statusText: "OK",
            body: {
                getReader: () => ({
                    read: async () => (i < chunks.length ? { value: chunks[i++], done: false } : { done: true }),
                }),
            },
        };
    };
}

test("stream parses frames, tracks Last-Event-ID, reconnects with backoff", async () => {
    const seen = [];
    const statuses = [];
    const timers = [];
    const stream = new LiveEventStream({
        apiKey: "k",
        fetchImpl: fakeFetchScript([
            { frames: ["id: 7\ndata: {\"v\":1,\"seq\":7,\"t_ms\":1,\"k\":\"queued\",\"req\":1,\"lane\":\"low\",\"n_prompt\":2,\"max_out\":0}\n\n"] },
            { frames: [": keepalive\n\n"] },
        ]),
        onEvent: (e) => seen.push(e),
        onStatus: (s) => statuses.push(s),
        setTimeoutImpl: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
        clearTimeoutImpl: () => {},
    });
    stream.start();
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(seen.length, 1);
    assert.equal(seen[0].k, "queued");
    assert.equal(stream.lastEventId, "7");
    // first disconnect scheduled a 1s retry
    assert.equal(timers.length, 1);
    assert.equal(timers[0].ms, 1000);
    assert.ok(statuses.includes("open"));
    assert.ok(statuses.includes("retrying"));
    // run the retry: stream ends again -> backoff doubles
    timers[0].fn();
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(timers[1].ms, 2000);
    stream.stop();
});

test("stream marks degraded after repeated failures and stops on auth errors", async () => {
    const statuses = [];
    const timers = [];
    const stream = new LiveEventStream({
        fetchImpl: fakeFetchScript([{ frames: [] }]),
        onEvent: () => {},
        onStatus: (s) => statuses.push(s),
        setTimeoutImpl: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
        clearTimeoutImpl: () => {},
    });
    stream.start();
    for (let i = 0; i < STREAM_FALLBACK_AFTER; i++) {
        await new Promise((r) => setTimeout(r, 10));
        if (timers[i]) timers[i].fn();
    }
    assert.ok(statuses.includes("degraded"));
    stream.stop();

    const authStatuses = [];
    const authTimers = [];
    const auth = new LiveEventStream({
        fetchImpl: fakeFetchScript([{ status: 401 }]),
        onEvent: () => {},
        onStatus: (s) => authStatuses.push(s),
        setTimeoutImpl: (fn, ms) => { authTimers.push({ fn, ms }); return 1; },
        clearTimeoutImpl: () => {},
    });
    auth.start();
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(authTimers.length, 0, "auth failure does not schedule retries");
    assert.ok(authStatuses.includes("degraded"));
});

test("real captured server payloads fold without loss", () => {
    // fixtures/captured-v2.json is a byte-exact capture from a real
    // llama-server build of this branch (see the fixture's `captured` note):
    // the wire schema the UI parses is the wire schema the server emits
    const store = new EventStore();
    for (const evt of captured.events) {
        assert.equal(store.feed(evt), true, "captured event accepted: " + JSON.stringify(evt).slice(0, 100));
    }
    assert.ok(store.requests.size >= 3);
    for (const req of store.requests.values()) {
        assert.ok(req.terminal, "captured requests all completed");
        assert.equal(req.terminal.reason, "completed");
        if (req.promptStart) {
            const spans = tokenSpans(req);
            assert.equal(spans.reduce((acc, s) => acc + s.tokens, 0), req.promptStart.nPrompt);
        }
    }
    const view = cacheView(captured.cache_state);
    assert.equal(view.enabled, true);
    assert.ok(view.counters.lookups >= 3);
});

test("fixture events replay cleanly through the store (round trip)", () => {
    const store = new EventStore();
    let accepted = 0;
    for (const evt of fixture.events) {
        if (store.feed(evt)) accepted += 1;
    }
    assert.equal(accepted, fixture.events.length);
    // every request in the fixture yields drawable segments and, once the
    // prompt landed, spans that exactly tile the prompt
    for (const req of store.rows()) {
        const seg = timeSegments(req, fixture.events[fixture.events.length - 1].t_ms + 1000);
        assert.ok(seg !== null);
        assert.ok(seg.segments.length >= 1);
        for (const s of seg.segments) assert.ok(s.t1 >= s.t0, "segment has non-negative width");
        if (req.promptStart) {
            const spans = tokenSpans(req);
            assert.equal(spans.reduce((acc, s) => acc + s.tokens, 0), req.promptStart.nPrompt);
        }
    }
});

// ---------------------------------------------------------------------------
// Per-request normalized scale ("fit")
// ---------------------------------------------------------------------------

test("phaseBreakdown normalizes to the request's own duration", () => {
    const store = new EventStore();
    for (const evt of fixture.events) store.feed(evt);
    const now = fixture.events[fixture.events.length - 1].t_ms + 1000;

    for (const req of store.rows()) {
        const bd = phaseBreakdown(req, now);
        if (bd === null) continue;
        const sum = bd.phases.reduce((acc, p) => acc + p.frac, 0);
        assert.ok(Math.abs(sum - 1) < 1e-9, "fractions tile the bar exactly");
        assert.ok(bd.total > 0);
        for (const p of bd.phases) {
            assert.ok(p.ms > 0, "no zero-width phases are emitted");
            assert.ok(p.frac > 0 && p.frac <= 1);
        }
    }
});

test("a 2 s request and a 40 min request both produce full-width breakdowns", () => {
    const store = new EventStore();
    for (const evt of fixture.events) store.feed(evt);
    const now = fixture.events[fixture.events.length - 1].t_ms + 1000;
    const byId = new Map(store.rows().map((r) => [r.id, r]));

    const short = phaseBreakdown(byId.get(102), now); // ~2 s chat turn
    const long = phaseBreakdown(byId.get(101), now);  // ~45 min agent turn
    assert.ok(short && long);
    assert.ok(long.total > short.total * 100, "the fixture really does span orders of magnitude");
    // ...yet both tile the same [0,1] track, which is the point of the scale
    for (const bd of [short, long]) {
        assert.ok(Math.abs(bd.phases.reduce((a, p) => a + p.frac, 0) - 1) < 1e-9);
    }
    // the short request's phase structure survives: it is not one flat block
    assert.ok(short.phases.length >= 2, "short requests keep their internal structure");
});

test("phaseBreakdown returns null when a request has no time extent", () => {
    assert.equal(phaseBreakdown({ queuedAt: null, dispatchedAt: null, promptStart: null, lastAt: null,
        terminal: null, finished: null, decode: [], deferred: [] }, 1000), null);
});

// ---------------------------------------------------------------------------
// Wasted restore (full-clear admission discarding a cache restore)
// ---------------------------------------------------------------------------

test("wastedRestore flags a restore thrown away by a full-clear admission", () => {
    const store = new EventStore();
    for (const evt of fixture.events) store.feed(evt);
    const byId = new Map(store.rows().map((r) => [r.id, r]));

    const wasted = wastedRestore(byId.get(108));
    assert.ok(wasted !== null, "req 108 is the wasted-restore scenario");
    assert.equal(wasted.tokens, 118_200);
    assert.equal(wasted.why, "no_covering_checkpoint");
    assert.ok(wasted.ms > 4000);
    assert.ok(wasted.bytes > 0);
});

test("wastedRestore stays silent for reuse, cold starts and surviving restores", () => {
    const store = new EventStore();
    for (const evt of fixture.events) store.feed(evt);
    const byId = new Map(store.rows().map((r) => [r.id, r]));

    for (const id of [101, 102, 103, 104, 106]) {
        assert.equal(wastedRestore(byId.get(id)), null, "reuse admissions waste nothing: " + id);
    }
    // req 105 full-clears an EMPTY slot after a cache miss: nothing was restored
    assert.equal(wastedRestore(byId.get(105)), null);
});

test("wastedRestore requires the landing to confirm the loss", () => {
    const base = {
        admission: { outcome: "ready_full_clear", why: "rs_window" },
        restore: { src: "ram", tokens: 5000, bytes: 100, ms: 30 },
        promptStart: { nPast: 5000, nPrompt: 6000, lcp: 5000, gapWhy: null },
    };
    // the prefix survived the clear (should not happen, but never cry wolf)
    assert.equal(wastedRestore(base), null);
    // the prefix did not survive
    assert.ok(wastedRestore({ ...base, promptStart: { ...base.promptStart, nPast: 0 } }) !== null);
});

// ---------------------------------------------------------------------------
// Schema-additive fields
// ---------------------------------------------------------------------------

test("full-clear admission attribution is folded into the request model", () => {
    const store = new EventStore();
    store.feed({ v: 1, seq: 1, t_ms: 1000, k: "queued", req: 7, lane: "normal", n_prompt: 10 });
    store.feed({ v: 1, seq: 2, t_ms: 1001, k: "admission", req: 7, slot: 0,
        outcome: "ready_full_clear", why: "no_covering_checkpoint", resident: 900, lcp: 800, span_end: 1200 });
    const req = store.rows()[0];
    assert.equal(req.admission.outcome, "ready_full_clear");
    assert.equal(req.admission.why, "no_covering_checkpoint");
    assert.equal(req.admission.resident, 900);
});

// Regenerates events-v2.json: a synthetic /m2-dashboard/events stream (schema
// v1, field-exact per tools/server/server-dashboard-bus.cpp) plus a matching
// /m2-dashboard/cache-state snapshot. Times are absolute ms from 0; the
// dashboard loader shifts them so the newest event lands at "now".
//
//   node fixtures/make-events.mjs
//
// Scenarios cover every span source and recompute reason the server emits:
// RAM/SSD/donor/resident restores, checkpoint landing gap, divergent tail,
// full recompute, a still-running 40-minute agent turn, a run_stall kill,
// and an admission-deferred queued request.

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const V = 1;
const events = [];
let seq = 0;

function emit(t, k, fields = {}) {
    events.push({ v: V, seq: ++seq, t_ms: Math.round(t), k, ...fields });
}

// --- request lifecycle helpers --------------------------------------------

function queued(t, req, lane, nPrompt, maxOut = 0) {
    emit(t, "queued", { req, lane, n_prompt: nPrompt, max_out: maxOut });
}

// Event order mirrors the server: get_available_slot picks the slot and runs
// the prompt-cache consult (slot_selected, cache_restore), and only then does
// prepare_dsv4_admission decide reuse-vs-clear (admission). A full clear after
// a successful restore therefore DISCARDS that restore — see wastedRestore().
function started(t, req, slot, { sel = "lru", lcp = 0, sim = null, restore = null, admission = null } = {}) {
    emit(t, "dispatched", { req, lane: eventLane(req), wait_ms: Math.round(t - queuedAt[req]) });
    const selected = { req, slot, sel, lcp, n_task: promptOf[req] };
    if (sim !== null) selected.sim = sim;
    emit(t + 2, "slot_selected", selected);
    if (restore) emit(t + 3 + (restore.ms ?? 0), "cache_restore", { req, slot, ...restore, n_task: promptOf[req] });
    if (admission) emit(t + 4 + (restore?.ms ?? 0), "admission", { req, slot, ...admission });
}

const queuedAt = {};
const promptOf = {};
const laneOf = {};
function eventLane(req) { return laneOf[req] ?? "normal"; }

function request(t, req, { lane = "normal", nPrompt, maxOut = 0 }) {
    queuedAt[req] = t;
    promptOf[req] = nPrompt;
    laneOf[req] = lane;
    queued(t, req, lane, nPrompt, maxOut);
}

function promptStart(t, req, slot, { nPast, lcp, gapWhy = null }) {
    const f = { req, slot, n_prompt: promptOf[req], n_past: nPast, lcp };
    if (gapWhy && nPast < lcp) f.gap_why = gapWhy;
    emit(t, "prompt_start", f);
    return t;
}

function prefill(t0, req, slot, nPast, nPrompt, tokPerSec, chunk = 2048) {
    let done = nPast;
    let t = t0;
    while (done < nPrompt) {
        const batch = Math.min(chunk, nPrompt - done);
        done += batch;
        t += (batch / tokPerSec) * 1000;
        emit(t, "prefill_progress", { req, slot, n_done: done, n_prompt: nPrompt, n_batch: batch, ms: t - t0 });
    }
    return t;
}

function decode(t0, req, slot, nTokens, tps, { draftRate = 0, acc = 0.75, until = null } = {}) {
    let n = 0;
    let draftN = 0;
    let draftA = 0;
    let t = t0;
    while (n < nTokens) {
        const step = Math.min(64, nTokens - n);
        n += step;
        draftN += Math.round(step * draftRate);
        draftA += Math.round(step * draftRate * acc);
        t += (step / tps) * 1000;
        if (until !== null && t > until) { t = until; n = Math.min(n, nTokens); }
        emit(t, "decode_progress", {
            req, slot, n_dec: n, draft_n: draftN, draft_a: draftA, ms: t - t0,
            tps: +(tps * (0.9 + 0.2 * Math.sin(n / 40))).toFixed(2),
        });
        if (until !== null && t >= until) break;
    }
    return { t, n, draftN, draftA };
}

function finish(t, req, slot, { stop = "eos", nPast, nDec, draftN = 0, draftA = 0, ppMs, tgMs }) {
    emit(t, "finished", {
        req, slot, stop, n_prompt: promptOf[req], n_cached: nPast, n_dec: nDec,
        draft_n: draftN, draft_a: draftA, pp_ms: +ppMs.toFixed(1), tg_ms: +tgMs.toFixed(1),
    });
    emit(t + 1, "terminal", { req, state: "completed", reason: "completed", code: 600 });
}

// --- scenario timeline (t = 0 … ~46 min) -----------------------------------

const MIN = 60 * 1000;

// req 101: the 40-minute agent turn — 281K prompt, 210K restored from the RAM
// tier, 2K checkpoint landing gap, 69K recomputed prefill, long decode, LIVE
request(0.4 * MIN, 101, { lane: "normal", nPrompt: 281_400 });
started(0.4 * MIN + 350, 101, 0, {
    sel: "lcp_affinity", lcp: 190_000, sim: 0.675,
    restore: { src: "ram", tokens: 212_300, bytes: 2_047_483_000, ms: 8123.5 },
    admission: { outcome: "ready_reuse", lcp: 212_300, resident: 212_300, frontier: 212_300, span_end: 281_500 },
});
{
    const t0 = 0.4 * MIN + 350 + 8200;
    promptStart(t0, 101, 0, { nPast: 210_250, lcp: 212_300, gapWhy: "checkpoint_gap" });
    const tPp = prefill(t0, 101, 0, 210_250, 281_400, 780);
    decode(tPp, 101, 0, 24_000, 13.5, { draftRate: 1.3, acc: 0.78, until: 46 * MIN });
}

// req 102: 2.3-second chat turn — resident reuse, spanless admission
request(9.1 * MIN, 102, { lane: "fast", nPrompt: 14_350, maxOut: 512 });
started(9.1 * MIN + 40, 102, 1, {
    sel: "lcp_affinity", lcp: 14_190, sim: 0.989,
    restore: { src: "resident", tokens: 14_190, bytes: 0, ms: 0.9 },
    admission: { outcome: "ready_spanless", lcp: 14_190, resident: 14_800, frontier: 14_820, span_end: 14_700 },
});
{
    const t0 = 9.1 * MIN + 60;
    promptStart(t0, 102, 1, { nPast: 14_190, lcp: 14_190 });
    const tPp = prefill(t0, 102, 1, 14_190, 14_350, 900);
    const d = decode(tPp, 102, 1, 180, 92, { draftRate: 1.4, acc: 0.82 });
    finish(d.t, 102, 1, { stop: "eos", nPast: 14_190, nDec: 180, draftN: d.draftN, draftA: d.draftA, ppMs: tPp - t0, tgMs: d.t - tPp });
}

// req 103: SSD-tier restore with a divergent tail
request(14 * MIN, 103, { lane: "normal", nPrompt: 31_900 });
started(14 * MIN + 60, 103, 2, {
    sel: "lru", lcp: 0,
    restore: { src: "disk", tokens: 31_000, bytes: 2_254_857_830, ms: 872.4 },
    admission: { outcome: "ready_reuse", lcp: 31_000, resident: 31_000, frontier: 31_000, span_end: 32_000 },
});
emit(14 * MIN + 900, "cache_op", { op: "disk_load", tokens: 31_000, bytes: 2_254_857_830, ms: 851.2 });
{
    const t0 = 14 * MIN + 950;
    promptStart(t0, 103, 2, { nPast: 31_000, lcp: 31_000 });
    const tPp = prefill(t0, 103, 2, 31_000, 31_900, 820);
    const d = decode(tPp, 103, 2, 800, 24, { draftRate: 1.3, acc: 0.71 });
    finish(d.t, 103, 2, { stop: "eos", nPast: 31_000, nDec: 800, draftN: d.draftN, draftA: d.draftA, ppMs: tPp - t0, tgMs: d.t - tPp });
}

// req 104: zero-copy donor share from the busy agent slot
request(18.5 * MIN, 104, { lane: "normal", nPrompt: 97_200 });
started(18.5 * MIN + 45, 104, 3, {
    sel: "lru", lcp: 0,
    restore: { src: "donor", tokens: 96_100, bytes: 0, ms: 14.2, donor_slot: 0 },
    admission: { outcome: "ready_reuse", lcp: 96_100, resident: 96_100, frontier: 96_100, span_end: 97_300 },
});
{
    const t0 = 18.5 * MIN + 80;
    promptStart(t0, 104, 3, { nPast: 96_100, lcp: 96_100 });
    const tPp = prefill(t0, 104, 3, 96_100, 97_200, 800);
    const d = decode(tPp, 104, 3, 2_400, 19, { draftRate: 1.3, acc: 0.74 });
    finish(d.t, 104, 3, { stop: "eos", nPast: 96_100, nDec: 2_400, draftN: d.draftN, draftA: d.draftA, ppMs: tPp - t0, tgMs: d.t - tPp });
}

// req 105: cold start — nothing cached, full 52K prefill, save + spill after
request(27 * MIN, 105, { lane: "low", nPrompt: 52_400, maxOut: 4096 });
started(27 * MIN + 800, 105, 1, {
    sel: "lru", lcp: 0,
    restore: { src: "miss", tokens: 0, bytes: 0, ms: 1.8 },
    admission: { outcome: "ready_full_clear", why: "empty_slot", resident: 0, lcp: 0, span_end: 56_500 },
});
{
    const t0 = 27 * MIN + 850;
    promptStart(t0, 105, 1, { nPast: 0, lcp: 0 });
    const tPp = prefill(t0, 105, 1, 0, 52_400, 750);
    const d = decode(tPp, 105, 1, 4_096, 21, { draftRate: 1.35, acc: 0.69 });
    finish(d.t, 105, 1, { stop: "limit", nPast: 0, nDec: 4_096, draftN: d.draftN, draftA: d.draftA, ppMs: tPp - t0, tgMs: d.t - tPp });
    emit(d.t + 1200, "cache_op", { op: "save", tokens: 56_496, bytes: 3_812_456_120 });
    emit(d.t + 9500, "cache_op", { op: "spill", tokens: 56_496, bytes: 3_812_456_120, ms: 3721.0 });
    emit(d.t + 9600, "cache_op", { op: "drop", tokens: 8_400, bytes: 590_558_003 });
}

// req 106: run_stall kill — progress stops, the 5-minute stall deadline fires
request(31 * MIN, 106, { lane: "normal", nPrompt: 44_100 });
emit(31 * MIN + 15, "deferred", { req: 106, why: "physical_admission" });
started(31 * MIN + 5200, 106, 2, {
    sel: "lru", lcp: 0,
    restore: { src: "ram", tokens: 41_800, bytes: 2_841_230_887, ms: 3182.0 },
    admission: { outcome: "ready_reuse", lcp: 41_800, resident: 41_800, frontier: 41_800, span_end: 44_200 },
});
{
    const t0 = 31 * MIN + 8400;
    promptStart(t0, 106, 2, { nPast: 41_800, lcp: 41_800 });
    const tPp = prefill(t0, 106, 2, 41_800, 44_100, 800);
    decode(tPp, 106, 2, 900, 18, { draftRate: 1.3, acc: 0.7 });
    // ...then nothing for five minutes: stall expiry
    emit(31 * MIN + 8400 + 3000 + 52_000 + 5 * MIN, "terminal", { req: 106, state: "timed_out", reason: "run_stall", code: 504 });
}

// req 108: WASTED RESTORE — the consult pulls a 118K prefix off the SSD tier
// (4.1 s), then the elastic admission finds no checkpoint covering the landing
// and clears the slot anyway, so every one of those tokens is recomputed. The
// dashboard flags this; it is the expensive failure mode the views exist for.
request(37 * MIN, 108, { lane: "normal", nPrompt: 119_400 });
started(37 * MIN + 90, 108, 3, {
    sel: "lcp_affinity", lcp: 118_200, sim: 0.990,
    restore: { src: "disk", tokens: 118_200, bytes: 7_902_113_400, ms: 4118.7 },
    admission: {
        outcome: "ready_full_clear", why: "no_covering_checkpoint",
        resident: 118_200, lcp: 118_200, span_end: 123_500,
    },
});
emit(37 * MIN + 4200, "cache_op", { op: "disk_load", tokens: 118_200, bytes: 7_902_113_400, ms: 4090.3 });
{
    const t0 = 37 * MIN + 4400;
    promptStart(t0, 108, 3, { nPast: 0, lcp: 0 });
    const tPp = prefill(t0, 108, 3, 0, 119_400, 770);
    const d = decode(tPp, 108, 3, 1_500, 17.5, { draftRate: 1.3, acc: 0.72 });
    finish(d.t, 108, 3, { stop: "eos", nPast: 0, nDec: 1_500, draftN: d.draftN, draftA: d.draftA, ppMs: tPp - t0, tgMs: d.t - tPp });
}

// req 107: queued behind the admission gate right now (live, never dispatched)
request(44.4 * MIN, 107, { lane: "normal", nPrompt: 128_700 });
emit(44.4 * MIN + 20, "deferred", { req: 107, why: "physical_admission" });
emit(44.9 * MIN, "deferred", { req: 107, why: "physical_admission" });

// keep the ordering the bus guarantees (monotone seq already; sort by time for realism)
events.sort((a, b) => a.t_ms - b.t_ms || a.seq - b.seq);
events.forEach((e, i) => { e.seq = i + 1; });

// --- cache-state snapshot ---------------------------------------------------

const lastT = events[events.length - 1].t_ms;
const cacheState = {
    v: V,
    enabled: true,
    t_ms: lastT,
    age_ms: 400,
    ram: { used_bytes: 6_442_450_944, limit_bytes: 8_589_934_592, tokens: 352_100, limit_tokens: 0 },
    disk: { used_bytes: 229_780_316_160, limit_bytes: 429_496_729_600 },
    counters: {
        lookups: 7, hits_entry: 5, hits_resident: 1, misses: 1,
        saves: 4, spills: 2, disk_loads: 2, drops: 2,
        bytes_saved: 11_824_222_310, bytes_spilled: 6_100_000_000, bytes_disk_load: 10_156_971_230,
    },
    entries: [
        { id: 9, tokens: 212_300, tier: "ram", bytes: 2_047_483_000, age_s: 2710.0, hits: 1, last_hit_s: 2650.4 },
        { id: 12, tokens: 96_100, tier: "ram", bytes: 1_580_312_400, age_s: 1620.2, hits: 0, last_hit_s: null },
        { id: 14, tokens: 43_700, tier: "ram", bytes: 2_814_655_192, age_s: 812.7, hits: 0, last_hit_s: null },
        { id: 3, tokens: 31_000, tier: "disk", bytes: 2_254_857_830, age_s: 71_003.9, hits: 4, last_hit_s: 1890.1, file: "pc-4fca21-3.lcpc" },
        { id: 15, tokens: 56_496, tier: "disk", bytes: 3_812_456_120, age_s: 1090.0, hits: 0, last_hit_s: null, file: "pc-4fca21-15.lcpc" },
        { id: 5, tokens: 148_800, tier: "disk", bytes: 9_931_003_192, age_s: 154_800.2, hits: 2, last_hit_s: 84_100.0, file: "pc-4fca21-5.lcpc" },
        { id: 6, tokens: 202_150, tier: "disk", bytes: 13_401_882_112, age_s: 121_412.8, hits: 1, last_hit_s: 99_412.0, file: "pc-4fca21-6.lcpc" },
    ],
};

const out = { v: V, generated_by: "make-events.mjs", events, cache_state: cacheState };
const path = join(dirname(fileURLToPath(import.meta.url)), "events-v2.json");
writeFileSync(path, JSON.stringify(out, null, 1) + "\n");
console.log("wrote", path, "-", events.length, "events");

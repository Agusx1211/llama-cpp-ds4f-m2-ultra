#!/usr/bin/env node
// Regenerates fixtures/demo.json: a synthetic-but-realistically-shaped scene
// for developing the dashboard offline (`?fixture=demo`).
//
// The payload shapes mirror real prod captures from 2026-08-07 (see
// replay.json, which is an untouched recording of /slots + the registry
// snapshot). Numbers are invented but plausible for DeepSeek V4 Flash on the
// M2 Ultra: ~20-25 tok/s decode per stream, ~700-900 tok/s prefill, heavy
// prefix-cache reuse on agent traffic.
//
// Usage: node make-demo.mjs   (writes demo.json next to this file)

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const props = JSON.parse(fs.readFileSync(path.join(here, "props.json"), "utf8"));

const N = 80;            // ticks
const STEP_S = 2;        // seconds per tick

const mkSlot = (id, o) => ({
    id, n_ctx: 524288, speculative: true, is_processing: o.proc ?? false,
    ...(o.task !== undefined ? {
        id_task: o.task,
        n_prompt_tokens: o.prompt,
        n_prompt_tokens_processed: o.processed,
        n_prompt_tokens_cache: o.cached,
        params: { n_predict: o.max ?? 32768, stream: true },
        next_token: [{
            has_next_token: o.proc ?? false, has_new_line: false,
            n_remain: (o.max ?? 32768) - o.decoded, n_decoded: o.decoded,
        }],
    } : {}),
});

const wobble = (base, amp, i, f = 1) =>
    base + amp * Math.sin(i / (5.1 * f)) + amp * 0.4 * Math.sin(i / (1.7 * f));

const ticks = [];

// cumulative counters so every rate is derivable (monotone within a task)
let s0dec = 400;
let s1pp = 0, s1dec = 0;
let s2task = 900, s2dec = 0, s2cycleLen = 7, s2target = 260, s2phase = 0;
let s3dec = 0;

for (let i = 0; i < N; i++) {
    // slot 0: steady long decode ~24 tok/s, 21k prompt, 79% cached
    s0dec += Math.round(wobble(24, 5, i) * STEP_S);
    const s0 = mkSlot(0, { proc: true, task: 4848, prompt: 18428 + s0dec, processed: 1575, cached: 16853, decoded: s0dec, max: 32768 });

    // slot 1: 121k prompt (58k cached) prefilling ~780 tok/s for ~40 ticks, then decode ~19 tok/s
    let s1;
    const s1new = 63_000;
    if (s1pp < s1new) {
        s1pp = Math.min(s1new, s1pp + Math.round(wobble(780, 140, i, 2) * STEP_S));
        s1 = mkSlot(1, { proc: true, task: 5011, prompt: 121_000, processed: s1pp, cached: 58_000, decoded: 0, max: 16384 });
    } else {
        s1dec += Math.round(wobble(19, 3, i) * STEP_S);
        s1 = mkSlot(1, { proc: true, task: 5011, prompt: 121_000 + s1dec, processed: s1new, cached: 58_000, decoded: s1dec, max: 16384 });
    }

    // slot 2: short cache-hot agent probes cycling; varying lengths.
    // Each cycle: 1 prefill tick (processed 0 -> 210), N decode ticks, 2 idle.
    if (s2phase >= s2cycleLen + 3) {
        s2task += 17;
        s2dec = 0;
        s2phase = 0;
        s2cycleLen = 4 + (i % 4);
        s2target = 180 + (i % 5) * 60;
    }
    let s2;
    if (s2phase === 0) {
        s2 = mkSlot(2, { proc: true, task: s2task, prompt: 9_400, processed: 0, cached: 9_190, decoded: 0, max: 4096 });
    } else if (s2phase <= s2cycleLen) {
        s2dec = Math.min(s2target, s2dec + Math.round(wobble(22, 4, i) * STEP_S));
        s2 = mkSlot(2, { proc: true, task: s2task, prompt: 9_400 + s2dec, processed: 210, cached: 9_190, decoded: s2dec, max: 4096 });
    } else {
        s2 = mkSlot(2, { proc: false, task: s2task, prompt: 9_400 + s2dec, processed: 210, cached: 9_190, decoded: s2dec, max: 4096 });
    }
    s2phase += 1;

    // slot 3: idle with a resident 44k sequence until tick 60, then decode
    let s3;
    if (i < 60) {
        s3 = mkSlot(3, { proc: false, task: 4630, prompt: 44_120, processed: 380, cached: 43_400, decoded: 340, max: 32768 });
    } else {
        s3dec += Math.round(wobble(22, 4, i) * STEP_S);
        s3 = mkSlot(3, { proc: true, task: 5300, prompt: 44_500 + s3dec, processed: 610, cached: 43_800, decoded: s3dec, max: 32768 });
    }

    const slots = [s0, s1, s2, s3];
    const queued = i >= 30 && i < 55 ? [{
        id: "6001:44", lane: "fast", state: "queued", age_ms: (i - 30) * STEP_S * 1000 + 400,
        prompt_tokens: 15_400, cache_hit_tokens: 0, output_tokens: 0, requested_output_tokens: 32768,
    }] : [];

    const reqOf = (slot, id, lane, startedTick) => ({
        id, lane,
        state: (slot.next_token?.[0]?.n_decoded ?? 0) > 0 ? "decode" : "prefill",
        age_ms: (i - startedTick) * STEP_S * 1000,
        prompt_tokens: slot.n_prompt_tokens ?? 0,
        cache_hit_tokens: 0, output_tokens: 0, requested_output_tokens: 32768,
    });
    const activeReqs = [];
    const bindings = {};
    const bind = (req, slotId) => { activeReqs.push(req); bindings[req.id] = slotId; };
    if (s0.is_processing) bind(reqOf(s0, "5501:31", "fast", -140), 0);
    if (s1.is_processing) bind(reqOf(s1, "5502:31", "normal", -6), 1);
    if (s2.is_processing) bind(reqOf(s2, "5" + (600 + s2task % 100) + ":31", "fast", i - s2phase + 1), 2);
    if (s3.is_processing) bind(reqOf(s3, "5701:31", "low", 60), 3);

    const laneRow = (id, queuedN, activeN, oldestMs) => ({
        id, queued: queuedN, active: activeN, oldest_wait_ms: oldestMs,
        service_deficit: 0, bypass_count: 0, predicted_start_ms: [0, 0],
        claimed_permits: activeN, bound_permits: activeN,
    });
    const registry = {
        schema_version: 2, sequence: 1000 + i,
        availability: { fast_refill: true },
        registry: {
            active_requests: activeReqs.length + queued.length,
            occupied_slots: slots.filter((s) => s.is_processing).length,
            retained_events: 500, event_capacity: 1024, total_events: 1500 + i,
            dropped_events: 0, claimed_permits: [0, 1, 2], bound_permits: [0, 1, 2], total_permits: 4,
        },
        lanes: [
            laneRow("low", 0, s3.is_processing ? 1 : 0, 0),
            laneRow("normal", 0, s1.is_processing ? 1 : 0, 0),
            laneRow("fast", queued.length, (s0.is_processing ? 1 : 0) + (s2.is_processing ? 1 : 0), queued[0]?.age_ms ?? 0),
        ],
        requests: [...activeReqs, ...queued],
        timeline: [],
    };

    ticks.push({ slots, registry, bindings });
}

fs.writeFileSync(path.join(here, "demo.json"),
    JSON.stringify({ interval_ms: STEP_S * 1000, props, ticks }));
console.log("wrote demo.json:", ticks.length, "ticks");

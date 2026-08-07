import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
    PHASE,
    SlotTracker,
    decodeProgress,
    formatDuration,
    formatPercent,
    formatRate,
    formatTokens,
    normalizeSlot,
    parseMetrics,
    prefillProgress,
    registryView,
    shortModelName,
    slotContextTokens,
    slotPhase,
    sparkPath,
} from "../lib/core.mjs";

const replay = JSON.parse(readFileSync(new URL("../fixtures/replay.json", import.meta.url), "utf8"));

// ---------------------------------------------------------------------------
// formatters
// ---------------------------------------------------------------------------

test("formatTokens compacts with units", () => {
    assert.equal(formatTokens(53), "53");
    assert.equal(formatTokens(1575), "1.6k");
    assert.equal(formatTokens(16853), "16.9k");
    assert.equal(formatTokens(524288), "524k");
    assert.equal(formatTokens(1_500_000), "1.5M");
    assert.equal(formatTokens(undefined), "—");
});

test("formatRate adapts precision", () => {
    assert.equal(formatRate(null), "—");
    assert.equal(formatRate(0), "0");
    assert.equal(formatRate(7.256), "7.26");
    assert.equal(formatRate(24.31), "24.3");
    assert.equal(formatRate(812), "812");
});

test("formatDuration picks sensible units", () => {
    assert.equal(formatDuration(450), "450ms");
    assert.equal(formatDuration(4200), "4.2s");
    assert.equal(formatDuration(147_150), "2m27s");
    assert.equal(formatDuration(3_720_000), "1h02m");
});

test("formatPercent floors small non-zero values", () => {
    assert.equal(formatPercent(0.004), "<1%");
    assert.equal(formatPercent(0.77), "77%");
    assert.equal(formatPercent(0), "0%");
});

test("shortModelName strips path and extension", () => {
    assert.equal(shortModelName("/Users/x/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf"), "dsv4-flash-0731-full.m2");
});

// ---------------------------------------------------------------------------
// slot normalization against the real captured payload
// ---------------------------------------------------------------------------

test("normalizeSlot reads the real /slots shape", () => {
    const raw = replay.ticks[0].slots.find((s) => s.id === 2);
    const slot = normalizeSlot(raw);
    assert.equal(slot.processing, true);
    assert.equal(slot.cached, 16853);
    assert.equal(slot.processed, 1575);
    assert.equal(slot.decoded > 0, true);
    assert.equal(slot.maxTokens, 32768);
    assert.equal(slotPhase(slot), PHASE.GENERATING);
});

test("idle slot without a task normalizes safely", () => {
    const slot = normalizeSlot({ id: 0, n_ctx: 524288, speculative: true, is_processing: false });
    assert.equal(slot.taskId, null);
    assert.equal(slotPhase(slot), PHASE.IDLE);
    assert.equal(slotContextTokens(slot), 0);
});

test("slotContextTokens covers both observed accountings", () => {
    // decode-time: n_prompt_tokens already includes generated tokens
    const decode = normalizeSlot({
        id: 2, n_ctx: 524288, is_processing: true, id_task: 1,
        n_prompt_tokens: 21412, n_prompt_tokens_processed: 1575, n_prompt_tokens_cache: 16853,
        next_token: [{ n_decoded: 2985, n_remain: 100 }],
    });
    assert.equal(slotContextTokens(decode), 16853 + 1575 + 2985); // 21413 > 21412
    // prefill-time: n_prompt_tokens is the target prompt
    const prefill = normalizeSlot({
        id: 1, n_ctx: 524288, is_processing: true, id_task: 2,
        n_prompt_tokens: 121_000, n_prompt_tokens_processed: 9000, n_prompt_tokens_cache: 58_000,
        next_token: [{ n_decoded: 0, n_remain: 4096 }],
    });
    assert.equal(slotContextTokens(prefill), 121_000);
    assert.equal(slotPhase(prefill), PHASE.PREFILL);
    assert.ok(Math.abs(prefillProgress(prefill) - 67_000 / 121_000) < 1e-9);
    assert.equal(decodeProgress(decode), 2985 / 3085);
});

// ---------------------------------------------------------------------------
// SlotTracker: rates, history, tail
// ---------------------------------------------------------------------------

function mkSlot(id, { proc = true, task = 7, prompt = 1000, processed = 100, cached = 800, decoded = 0, remain = 4096 } = {}) {
    return {
        id, n_ctx: 524288, speculative: true, is_processing: proc,
        ...(task === null ? {} : {
            id_task: task,
            n_prompt_tokens: prompt, n_prompt_tokens_processed: processed, n_prompt_tokens_cache: cached,
            params: { n_predict: 4096, stream: true },
            next_token: [{ n_decoded: decoded, n_remain: remain }],
        }),
    };
}

test("tracker derives generation rate from n_decoded deltas", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { decoded: 100 })]);
    const view = tr.feed(2000, [mkSlot(0, { decoded: 150 })]);
    assert.equal(view.slots[0].genRate, 25); // 50 tokens / 2 s
    assert.equal(view.genRate, 25);
    assert.equal(view.processingCount, 1);
});

test("tracker derives prefill rate and ignores rate across task change", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { task: 1, processed: 1000, decoded: 0 })]);
    let view = tr.feed(2000, [mkSlot(0, { task: 1, processed: 2500, decoded: 0 })]);
    assert.equal(view.slots[0].ppRate, 750);
    assert.equal(view.slots[0].phase, PHASE.PREFILL);
    // new task: no rate on the first sample
    view = tr.feed(4000, [mkSlot(0, { task: 2, processed: 50, decoded: 10 })]);
    assert.equal(view.slots[0].genRate, null);
});

test("tracker aggregates history and respects the cap", () => {
    const tr = new SlotTracker({ historyLimit: 5 });
    for (let i = 0; i < 10; i++) {
        tr.feed(i * 2000, [mkSlot(0, { decoded: i * 40 })]);
    }
    assert.equal(tr.history.length, 5);
    assert.equal(tr.history.at(-1).gen, 20);
});

test("finished task lands in the tail with cache and rate data", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { task: 9, processed: 0, cached: 800, prompt: 1000, decoded: 0 })]);
    tr.feed(2000, [mkSlot(0, { task: 9, processed: 200, cached: 800, prompt: 1000, decoded: 0 })]);
    tr.feed(4000, [mkSlot(0, { task: 9, processed: 200, cached: 800, prompt: 1240, decoded: 240 })]);
    tr.feed(6000, [mkSlot(0, { task: 9, proc: false, processed: 200, cached: 800, prompt: 1290, decoded: 290 })]);
    assert.equal(tr.tail.length, 1);
    const rec = tr.tail[0];
    assert.equal(rec.promptTokens, 1000);
    assert.equal(rec.cachedTokens, 800);
    assert.equal(rec.generated, 290);
    assert.equal(rec.ppRate, 100);   // 200 tokens over 2 s
    assert.equal(rec.genRate, 72.5); // 290 tokens over 4 s of decode-observed time
    assert.equal(rec.partial, false);
    assert.equal(rec.durationMs, 6000);
});

test("task replaced by a new task is also finalized", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { task: 1, decoded: 0 })]);
    tr.feed(2000, [mkSlot(0, { task: 1, decoded: 50 })]);
    tr.feed(4000, [mkSlot(0, { task: 2, decoded: 0 })]);
    assert.equal(tr.tail.length, 1);
    assert.equal(tr.tail[0].taskId, 1);
    assert.equal(tr.tail[0].generated, 50);
});

test("pre-existing finished tasks never enter the tail", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { proc: false, task: 5, decoded: 42 })]);
    tr.feed(2000, [mkSlot(0, { proc: false, task: 5, decoded: 42 })]);
    tr.feed(4000, [mkSlot(0, { task: 6, decoded: 0 })]);
    assert.equal(tr.tail.length, 0);
});

test("cacheReuse combines current slots and the tail", () => {
    const tr = new SlotTracker();
    tr.feed(0, [mkSlot(0, { task: 1, prompt: 1000, cached: 900, processed: 100, decoded: 0 })]);
    const view = tr.feed(2000, [mkSlot(0, { task: 1, prompt: 1000, cached: 900, processed: 100, decoded: 10 })]);
    const reuse = tr.cacheReuse(view.slots);
    assert.equal(reuse.cachedTokens, 900);
    assert.equal(reuse.promptTokens, 1000);
});

// ---------------------------------------------------------------------------
// registry view against the real captured snapshot
// ---------------------------------------------------------------------------

test("registryView reduces the real snapshot", () => {
    const snap = replay.ticks.find((t) => t.registry)?.registry;
    const view = registryView(snap);
    assert.equal(view.active.length, 1);
    assert.equal(view.active[0].lane, "fast");
    assert.match(view.active[0].id, /^\d+:\d+$/);
    assert.equal(view.queued.length, 0);
    assert.equal(view.lanes.fast.active, 1);
    assert.equal(view.lanes.low.queued, 0);
});

test("registryView handles null and malformed input", () => {
    assert.equal(registryView(null), null);
    assert.equal(registryView({}), null);
});

// ---------------------------------------------------------------------------
// metrics parsing
// ---------------------------------------------------------------------------

test("parseMetrics extracts llamacpp gauges and counters", () => {
    const text = [
        "# HELP llamacpp:prompt_tokens_total x",
        "# TYPE llamacpp:prompt_tokens_total counter",
        "llamacpp:prompt_tokens_total 1234",
        "llamacpp:kv_physical_pressure_total 3",
        "other:metric 9",
    ].join("\n");
    const parsed = parseMetrics(text);
    assert.equal(parsed.prompt_tokens_total, 1234);
    assert.equal(parsed.kv_physical_pressure_total, 3);
    assert.equal(parsed["other:metric"], undefined);
    assert.equal(parseMetrics(""), null);
});

// ---------------------------------------------------------------------------
// sparkline geometry
// ---------------------------------------------------------------------------

test("sparkPath maps history into bounded coordinates", () => {
    const history = [];
    for (let i = 0; i < 30; i++) history.push({ t: i * 2000, gen: 20 + (i % 5), pp: 0 });
    const geo = sparkPath(history, "gen", { width: 200, height: 34, windowMs: 60_000, now: 60_000 });
    assert.ok(geo.path.startsWith("M"));
    assert.ok(geo.points.length > 2);
    for (const p of geo.points) {
        assert.ok(p.x >= 0 && p.x <= 200);
        assert.ok(p.y >= 0 && p.y <= 34);
    }
    assert.equal(geo.max, 24);
});

test("sparkPath yields empty output for insufficient data", () => {
    const geo = sparkPath([{ t: 0, gen: 5 }], "gen", { width: 200, height: 34, windowMs: 60_000, now: 60_000 });
    assert.equal(geo.path, "");
});

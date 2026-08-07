// Pure data logic for the M2 operations dashboard.
//
// Everything here is DOM-free and Node-testable. The browser entry point
// (dashboard.js) feeds raw payloads from the real endpoints:
//
//   GET /slots                                per-slot live state (the live truth)
//   GET /internal/admin/dashboard/snapshot    request registry: queue, lanes, ids
//   GET /props                                model identity (once)
//   GET /metrics                              optional Prometheus counters
//
// and renders the derived views computed here. Rates are derived client-side
// by differencing counters between polls because the server exposes none of
// them per slot.

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

export function formatTokens(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return "—";
    if (n < 1000) return String(n);
    if (n < 100_000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
    if (n < 1_000_000) return Math.round(n / 1000) + "k";
    return (n / 1_000_000).toFixed(2).replace(/\.?0+$/, "") + "M";
}

export function formatRate(tps) {
    if (tps === null || tps === undefined || !Number.isFinite(tps)) return "—";
    if (tps === 0) return "0";
    if (tps >= 100) return String(Math.round(tps));
    if (tps >= 10) return tps.toFixed(1);
    return tps.toFixed(2);
}

export function formatDuration(ms) {
    if (ms === null || ms === undefined || !Number.isFinite(ms)) return "—";
    if (ms < 1000) return Math.round(ms) + "ms";
    const s = ms / 1000;
    if (s < 60) return (s < 10 ? s.toFixed(1) : String(Math.round(s))) + "s";
    const m = Math.floor(s / 60);
    if (m < 60) return m + "m" + String(Math.round(s % 60)).padStart(2, "0") + "s";
    const h = Math.floor(m / 60);
    return h + "h" + String(m % 60).padStart(2, "0") + "m";
}

export function formatClock(epochMs) {
    const d = new Date(epochMs);
    return String(d.getHours()).padStart(2, "0") + ":" +
           String(d.getMinutes()).padStart(2, "0") + ":" +
           String(d.getSeconds()).padStart(2, "0");
}

export function formatPercent(fraction) {
    if (!Number.isFinite(fraction)) return "—";
    const pct = fraction * 100;
    if (pct > 0 && pct < 1) return "<1%";
    return Math.round(pct) + "%";
}

export function shortModelName(pathOrAlias) {
    if (typeof pathOrAlias !== "string" || pathOrAlias.length === 0) return "unknown model";
    const base = pathOrAlias.split("/").pop();
    return base.replace(/\.gguf$/i, "");
}

// ---------------------------------------------------------------------------
// Slot normalization and phase
// ---------------------------------------------------------------------------

export const PHASE = Object.freeze({
    IDLE: "idle",
    PREFILL: "prefill",
    GENERATING: "generating",
});

function num(value, fallback = 0) {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

// Normalize one raw /slots entry to the fields the dashboard uses.
export function normalizeSlot(raw) {
    const nt = Array.isArray(raw.next_token) ? raw.next_token[0] ?? {} : raw.next_token ?? {};
    const params = raw.params ?? {};
    const hasTask = raw.id_task !== undefined && raw.id_task !== null;
    return {
        id: num(raw.id, -1),
        nCtx: num(raw.n_ctx),
        processing: raw.is_processing === true,
        taskId: hasTask ? raw.id_task : null,
        promptTokens: num(raw.n_prompt_tokens),
        processed: num(raw.n_prompt_tokens_processed),
        cached: num(raw.n_prompt_tokens_cache),
        decoded: num(nt.n_decoded),
        remain: typeof nt.n_remain === "number" ? nt.n_remain : null,
        maxTokens: typeof params.n_predict === "number" ? params.n_predict : null,
        stream: params.stream === true,
    };
}

// Phase from a single normalized sample.
export function slotPhase(slot) {
    if (!slot.processing) return PHASE.IDLE;
    return slot.decoded > 0 ? PHASE.GENERATING : PHASE.PREFILL;
}

// Context tokens resident for this slot's sequence. The server reports
// n_prompt_tokens as either the target prompt length (during prefill) or the
// full resident sequence (during decode, prompt + generated); the max of the
// two accountings is correct in both observed layouts.
export function slotContextTokens(slot) {
    if (slot.taskId === null) return 0;
    return Math.max(slot.promptTokens, slot.cached + slot.processed + slot.decoded);
}

// Prefill progress fraction (0..1) or null when it does not apply.
export function prefillProgress(slot) {
    if (slotPhase(slot) !== PHASE.PREFILL || slot.promptTokens <= 0) return null;
    return Math.min(1, (slot.cached + slot.processed) / slot.promptTokens);
}

// Decode progress fraction, only meaningful when a token budget exists.
export function decodeProgress(slot) {
    if (slotPhase(slot) !== PHASE.GENERATING) return null;
    if (slot.remain === null || slot.remain < 0) return null;
    const total = slot.decoded + slot.remain;
    return total > 0 ? slot.decoded / total : null;
}

// ---------------------------------------------------------------------------
// SlotTracker: rates, history, finished-request tail
// ---------------------------------------------------------------------------

const EWMA_ALPHA = 0.5;

function blend(prev, next) {
    return prev === null ? next : EWMA_ALPHA * next + (1 - EWMA_ALPHA) * prev;
}

export class SlotTracker {
    constructor({ historyLimit = 240, tailLimit = 40 } = {}) {
        this.historyLimit = historyLimit;
        this.tailLimit = tailLimit;
        this.slots = new Map();     // slot id -> per-slot tracking state
        this.history = [];          // [{t, gen, pp}] aggregate rates, newest last
        this.tail = [];             // finished request records, newest first
        this.lastFeedT = null;
        this.firstFeed = true;
    }

    // Feed one /slots payload sampled at time t (ms). Returns the render view.
    feed(t, rawSlots) {
        const views = [];
        let aggGen = 0;
        let aggPp = 0;
        for (const raw of rawSlots) {
            const slot = normalizeSlot(raw);
            const view = this.#feedSlot(t, slot);
            aggGen += view.genRate ?? 0;
            aggPp += view.ppRate ?? 0;
            views.push(view);
        }
        // drop tracking for slots that disappeared (server restart with fewer lanes)
        const present = new Set(views.map((v) => v.slot.id));
        for (const id of [...this.slots.keys()]) {
            if (!present.has(id)) this.slots.delete(id);
        }
        if (!this.firstFeed) {
            this.history.push({ t, gen: aggGen, pp: aggPp });
            if (this.history.length > this.historyLimit) {
                this.history.splice(0, this.history.length - this.historyLimit);
            }
        }
        this.lastFeedT = t;
        this.firstFeed = false;
        views.sort((a, b) => a.slot.id - b.slot.id);
        return {
            slots: views,
            genRate: aggGen,
            ppRate: aggPp,
            contextTokens: views.reduce((sum, v) => sum + slotContextTokens(v.slot), 0),
            processingCount: views.filter((v) => v.slot.processing).length,
        };
    }

    #feedSlot(t, slot) {
        let st = this.slots.get(slot.id);
        if (st === undefined) {
            st = { prev: null, prevT: null, task: null };
            this.slots.set(slot.id, st);
        }

        const prev = st.prev;
        const sameTask = prev !== null && slot.taskId !== null && prev.taskId === slot.taskId &&
            // decode counter moving backwards means the task id was reused
            slot.decoded >= prev.decoded && slot.processed >= prev.processed;

        // a new task replaced the previous one: the old task is finished
        if (st.task !== null && slot.taskId !== st.task.taskId) {
            this.#finalize(st.task, slot.id);
            st.task = null;
        }

        let genRate = null;
        let ppRate = null;
        if (sameTask && slot.processing && st.prevT !== null && t > st.prevT) {
            const dt = (t - st.prevT) / 1000;
            const dGen = slot.decoded - prev.decoded;
            const dPp = slot.processed - prev.processed;
            if (dGen > 0) genRate = dGen / dt;
            if (dPp > 0) ppRate = dPp / dt;
        }

        // per-task accumulation for the tail
        if (slot.taskId !== null) {
            if (st.task === null || st.task.taskId !== slot.taskId) {
                st.task = {
                    taskId: slot.taskId,
                    firstT: t,
                    sawStart: slot.processing && slot.decoded === 0,
                    preexistingIdle: !slot.processing && this.firstFeed,
                    lastT: t,
                    last: slot,
                    genTokens: 0,
                    genSeconds: 0,
                    ppTokens: 0,
                    ppSeconds: 0,
                    wasProcessing: slot.processing,
                };
            } else {
                const task = st.task;
                if (sameTask && st.prevT !== null && t > st.prevT) {
                    const dt = (t - st.prevT) / 1000;
                    const dGen = slot.decoded - prev.decoded;
                    const dPp = slot.processed - prev.processed;
                    if (dGen > 0) { task.genTokens += dGen; task.genSeconds += dt; }
                    if (dPp > 0) { task.ppTokens += dPp; task.ppSeconds += dt; }
                }
                task.lastT = t;
                task.last = slot;
                task.wasProcessing = task.wasProcessing || slot.processing;
                // processing -> idle on the same task: request finished
                if (prev !== null && prev.processing && !slot.processing) {
                    this.#finalize(task, slot.id);
                    task.finalized = true;
                }
            }
        }

        // smooth displayed rates per slot
        if (sameTask && slot.processing) {
            st.genRateSmooth = genRate !== null ? blend(st.genRateSmooth ?? null, genRate) : (slotPhase(slot) === PHASE.GENERATING ? st.genRateSmooth ?? null : null);
            st.ppRateSmooth = ppRate !== null ? blend(st.ppRateSmooth ?? null, ppRate) : (slotPhase(slot) === PHASE.PREFILL ? st.ppRateSmooth ?? null : null);
        } else {
            st.genRateSmooth = null;
            st.ppRateSmooth = null;
        }

        st.prev = slot;
        st.prevT = t;

        return {
            slot,
            phase: slotPhase(slot),
            genRate: slot.processing ? st.genRateSmooth : null,
            ppRate: slot.processing ? st.ppRateSmooth : null,
            contextTokens: slotContextTokens(slot),
            prefillFraction: prefillProgress(slot),
            decodeFraction: decodeProgress(slot),
        };
    }

    #finalize(task, slotId) {
        if (task.finalized || task.preexistingIdle || !task.wasProcessing) return;
        const last = task.last;
        const promptTokens = last.cached + last.processed > 0 ? last.cached + last.processed : last.promptTokens;
        const nowMono = globalThis.performance?.now?.() ?? task.lastT;
        this.tail.unshift({
            endedAt: Date.now() - Math.max(0, nowMono - task.lastT),
            slot: slotId,
            taskId: task.taskId,
            promptTokens,
            cachedTokens: last.cached,
            generated: last.decoded,
            genRate: task.genSeconds > 0 ? task.genTokens / task.genSeconds : null,
            ppRate: task.ppSeconds > 0 ? task.ppTokens / task.ppSeconds : null,
            durationMs: task.lastT - task.firstT,
            partial: !task.sawStart,
        });
        if (this.tail.length > this.tailLimit) this.tail.length = this.tailLimit;
    }

    // Cache reuse across the current tasks and recent tail (fraction 0..1, or null).
    cacheReuse(currentViews) {
        let cached = 0;
        let prompt = 0;
        for (const v of currentViews) {
            if (v.slot.taskId !== null && v.slot.processing) {
                cached += v.slot.cached;
                prompt += Math.max(v.slot.promptTokens, v.slot.cached + v.slot.processed);
            }
        }
        for (const rec of this.tail.slice(0, 10)) {
            cached += rec.cachedTokens;
            prompt += rec.promptTokens;
        }
        return prompt > 0 ? { fraction: cached / prompt, cachedTokens: cached, promptTokens: prompt } : null;
    }
}

// ---------------------------------------------------------------------------
// Request registry (queue, lanes, ids for cancel)
// ---------------------------------------------------------------------------

export const LANE_ORDER = ["fast", "normal", "low"];

// Reduce a registry snapshot to what the dashboard shows. The registry's
// per-request state/output counters lag live decode badly (observed on prod:
// "prefill", output 0, while the slot had 3k tokens decoded), so /slots is
// the live truth; the registry contributes the queue, lane attribution and
// the id:epoch handles needed for cancel.
export function registryView(snapshot) {
    if (!snapshot || !Array.isArray(snapshot.requests)) return null;
    const lanes = {};
    for (const lane of snapshot.lanes ?? []) {
        lanes[lane.id] = {
            queued: num(lane.queued),
            active: num(lane.active),
            oldestWaitMs: num(lane.oldest_wait_ms),
        };
    }
    const active = [];
    const queued = [];
    for (const req of snapshot.requests) {
        const item = {
            id: String(req.id ?? ""),
            lane: String(req.lane ?? "normal"),
            state: String(req.state ?? ""),
            ageMs: num(req.age_ms),
            promptTokens: num(req.prompt_tokens),
            requestedOutput: typeof req.requested_output_tokens === "number" ? req.requested_output_tokens : null,
        };
        (item.state === "queued" ? queued : active).push(item);
    }
    queued.sort((a, b) => b.ageMs - a.ageMs);
    return {
        lanes,
        active,
        queued,
        activeRequests: num(snapshot.registry?.active_requests),
        occupiedSlots: num(snapshot.registry?.occupied_slots),
    };
}

// ---------------------------------------------------------------------------
// Optional /metrics (Prometheus text) parsing
// ---------------------------------------------------------------------------

export function parseMetrics(text) {
    if (typeof text !== "string") return null;
    const out = {};
    for (const line of text.split("\n")) {
        if (line.startsWith("#") || line.length === 0) continue;
        const space = line.lastIndexOf(" ");
        if (space <= 0) continue;
        const name = line.slice(0, space);
        const value = Number(line.slice(space + 1));
        if (name.startsWith("llamacpp:") && Number.isFinite(value)) {
            out[name.slice("llamacpp:".length)] = value;
        }
    }
    return Object.keys(out).length > 0 ? out : null;
}

// ---------------------------------------------------------------------------
// Sparkline geometry (pure; the renderer draws the result)
// ---------------------------------------------------------------------------

// Map history entries to an SVG polyline path over the last windowMs.
export function sparkPath(history, key, { width, height, windowMs, now, pad = 2 }) {
    const from = now - windowMs;
    const points = history.filter((h) => h.t >= from);
    if (points.length < 2) return { path: "", area: "", max: 0, points: [] };
    let max = 0;
    for (const p of points) max = Math.max(max, p[key]);
    const scaleMax = max > 0 ? max * 1.15 : 1;
    const xy = points.map((p) => [
        pad + ((p.t - from) / windowMs) * (width - 2 * pad),
        height - pad - (p[key] / scaleMax) * (height - 2 * pad),
    ]);
    const path = xy.map(([x, y], i) => (i === 0 ? "M" : "L") + x.toFixed(1) + " " + y.toFixed(1)).join("");
    const area = path + "L" + xy[xy.length - 1][0].toFixed(1) + " " + (height - pad) +
                 "L" + xy[0][0].toFixed(1) + " " + (height - pad) + "Z";
    return {
        path,
        area,
        max,
        last: points[points.length - 1][key],
        points: xy.map(([x, y], i) => ({ x, y, t: points[i].t, v: points[i][key] })),
    };
}

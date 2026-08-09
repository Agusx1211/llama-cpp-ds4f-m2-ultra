// Dashboard v2 event-model logic (DOM-free, node-testable).
//
// The server's /m2-dashboard/events SSE stream (schema v1, see
// tools/server/server-dashboard-bus.h) is folded into per-request timeline
// models here. Everything a view needs — the time waterfall segments, the
// restored-vs-recomputed token spans, cache traffic — is derived by pure
// functions in this module so fixture replay and unit tests exercise the
// exact code the live page runs.

import { createFetchSseTransport } from "./sse.mjs";

export const EVENTS_SCHEMA_V = 1;
export const EVENTS_PATH = "/m2-dashboard/events";
export const CACHE_STATE_PATH = "/m2-dashboard/cache-state";

// bounded model sizes (a shared box: keep memory flat over week-long tabs)
export const MAX_REQUESTS = 160;
export const MAX_PREFILL_POINTS = 512;
export const MAX_DECODE_POINTS = 512;
export const MAX_LOOKUP_POINTS = 512;

// ---------------------------------------------------------------------------
// Event store: fold bus events into request models
// ---------------------------------------------------------------------------

export class EventStore {
    constructor() {
        this.requests = new Map(); // req id -> model
        this.lastSeq = 0;
        this.dropped = 0;          // events lost to ring overwrite (from hello)
        this.gapped = false;       // true when a hello revealed missed events
        this.cacheOps = { save: 0, spill: 0, disk_load: 0, drop: 0, bytes_spilled: 0, bytes_disk_load: 0 };
        this.lookups = [];         // {t, hit} per cache_restore consult (sparkline source)
        this.clockSkewMs = 0;      // server wall-clock minus client, from hello
    }

    // fold one parsed event object (the JSON payload of an SSE frame)
    feed(evt) {
        if (!evt || typeof evt !== "object") return false;
        if (evt.k === "hello") {
            if (this.lastSeq > 0 && evt.first_seq > this.lastSeq + 1) this.gapped = true;
            this.dropped = evt.dropped ?? 0;
            if (typeof evt.now_ms === "number") {
                this.clockSkewMs = evt.now_ms - Date.now();
            }
            return true;
        }
        if (evt.v !== EVENTS_SCHEMA_V || typeof evt.seq !== "number") return false;
        if (evt.seq <= this.lastSeq) return false; // replayed duplicate
        this.lastSeq = evt.seq;

        const t = evt.t_ms;
        switch (evt.k) {
            case "queued": {
                const r = this.#request(evt.req);
                r.lane = evt.lane ?? r.lane;
                r.promptTokens = evt.n_prompt ?? r.promptTokens;
                r.maxOut = evt.max_out ?? r.maxOut;
                r.parent = evt.parent ?? r.parent;
                r.queuedAt = t;
                r.lastAt = t;
                break;
            }
            case "deferred": {
                const r = this.#request(evt.req);
                r.deferred.push({ t, why: evt.why });
                if (r.deferred.length > 16) r.deferred.shift();
                r.lastAt = t;
                break;
            }
            case "dispatched": {
                const r = this.#request(evt.req);
                r.dispatchedAt = t;
                r.waitMs = evt.wait_ms ?? null;
                r.lane = evt.lane ?? r.lane;
                r.lastAt = t;
                break;
            }
            case "slot_selected": {
                const r = this.#request(evt.req);
                r.slot = evt.slot ?? r.slot;
                r.select = { sel: evt.sel, lcp: evt.lcp ?? 0, sim: evt.sim ?? null };
                r.lastAt = t;
                break;
            }
            case "admission": {
                const r = this.#request(evt.req);
                r.admission = {
                    outcome: evt.outcome,
                    why: evt.why ?? null,          // full-clear attribution
                    lcp: evt.lcp ?? null,
                    resident: evt.resident ?? null,
                    frontier: evt.frontier ?? null,
                    spanEnd: evt.span_end ?? null,
                };
                if (evt.slot !== undefined) r.slot = evt.slot;
                r.lastAt = t;
                break;
            }
            case "cache_restore": {
                const r = this.#request(evt.req);
                r.restore = {
                    t,
                    src: evt.src,
                    tokens: evt.tokens ?? 0,
                    bytes: evt.bytes ?? 0,
                    ms: evt.ms ?? 0,
                    donorSlot: evt.donor_slot ?? null,
                };
                if (evt.slot !== undefined) r.slot = evt.slot;
                r.lastAt = t;
                this.lookups.push({ t, hit: evt.src !== "miss" ? 1 : 0 });
                if (this.lookups.length > MAX_LOOKUP_POINTS) this.lookups.shift();
                break;
            }
            case "prompt_start": {
                const r = this.#request(evt.req);
                r.promptStart = {
                    t,
                    nPrompt: evt.n_prompt ?? 0,
                    nPast: evt.n_past ?? 0,
                    lcp: evt.lcp ?? 0,
                    gapWhy: evt.gap_why ?? null,
                };
                r.promptTokens = evt.n_prompt ?? r.promptTokens;
                if (evt.slot !== undefined) r.slot = evt.slot;
                r.lastAt = t;
                break;
            }
            case "prefill_progress": {
                const r = this.#request(evt.req);
                r.prefill.push({ t, done: evt.n_done ?? 0, total: evt.n_prompt ?? 0, batch: evt.n_batch ?? 0, ms: evt.ms ?? 0 });
                if (r.prefill.length > MAX_PREFILL_POINTS) {
                    // decimate rather than truncate so the progress curve keeps
                    // its full extent for very long prefills
                    r.prefill = r.prefill.filter((_, i) => i % 2 === 0);
                }
                r.lastAt = t;
                break;
            }
            case "decode_progress": {
                const r = this.#request(evt.req);
                r.decode.push({ t, n: evt.n_dec ?? 0, draftN: evt.draft_n ?? 0, draftA: evt.draft_a ?? 0, ms: evt.ms ?? 0, tps: evt.tps ?? null });
                if (r.decode.length > MAX_DECODE_POINTS) {
                    r.decode = r.decode.filter((_, i) => i % 2 === 0);
                }
                r.lastAt = t;
                break;
            }
            case "finished": {
                const r = this.#request(evt.req);
                r.finished = {
                    t,
                    stop: evt.stop,
                    nPrompt: evt.n_prompt ?? 0,
                    nCached: evt.n_cached ?? 0,
                    nDecoded: evt.n_dec ?? 0,
                    draftN: evt.draft_n ?? 0,
                    draftA: evt.draft_a ?? 0,
                    ppMs: evt.pp_ms ?? 0,
                    tgMs: evt.tg_ms ?? 0,
                };
                r.lastAt = t;
                break;
            }
            case "terminal": {
                const r = this.#request(evt.req);
                r.terminal = { t, state: evt.state, reason: evt.reason, code: evt.code };
                r.lastAt = t;
                break;
            }
            case "cache_op": {
                const op = evt.op;
                if (op in this.cacheOps) this.cacheOps[op] += 1;
                if (op === "spill") this.cacheOps.bytes_spilled += evt.bytes ?? 0;
                if (op === "disk_load") this.cacheOps.bytes_disk_load += evt.bytes ?? 0;
                return true; // server-scoped, no request model
            }
            default:
                return false; // unknown kind: forward-compatible skip
        }
        this.#prune();
        return true;
    }

    #request(id) {
        let r = this.requests.get(id);
        if (r === undefined) {
            r = {
                id,
                lane: null,
                parent: null,
                slot: null,
                promptTokens: null,
                maxOut: null,
                queuedAt: null,
                dispatchedAt: null,
                waitMs: null,
                deferred: [],
                select: null,
                admission: null,
                restore: null,
                promptStart: null,
                prefill: [],
                decode: [],
                finished: null,
                terminal: null,
                lastAt: null,
            };
            this.requests.set(id, r);
        }
        return r;
    }

    #prune() {
        if (this.requests.size <= MAX_REQUESTS) return;
        // drop the oldest terminal requests first, then the oldest of any state
        const all = [...this.requests.values()].sort((a, b) => (a.lastAt ?? 0) - (b.lastAt ?? 0));
        for (const r of all) {
            if (this.requests.size <= MAX_REQUESTS) break;
            if (r.terminal !== null) this.requests.delete(r.id);
        }
        for (const r of all) {
            if (this.requests.size <= MAX_REQUESTS) break;
            this.requests.delete(r.id);
        }
    }

    // most-recent-first request list for the timeline (live requests first)
    rows() {
        const live = [];
        const done = [];
        for (const r of this.requests.values()) {
            (r.terminal === null ? live : done).push(r);
        }
        live.sort((a, b) => (b.queuedAt ?? b.lastAt ?? 0) - (a.queuedAt ?? a.lastAt ?? 0));
        done.sort((a, b) => (b.lastAt ?? 0) - (a.lastAt ?? 0));
        return live.concat(done);
    }
}

// ---------------------------------------------------------------------------
// Token spans: restored vs recomputed, with source and reason (the heart)
// ---------------------------------------------------------------------------

// Returns [{start, end, tokens, kind: "restored"|"recomputed", src?, why?}]
// covering [0, n_prompt). Sources: resident | ram | disk | donor. Reasons:
// checkpoint_gap | reset | last_token | divergent_tail | no_cache.
export function tokenSpans(req) {
    const ps = req.promptStart;
    if (!ps || !(ps.nPrompt > 0)) return [];
    const spans = [];
    const nPast = Math.max(0, Math.min(ps.nPast, ps.nPrompt));
    const lcp = Math.max(nPast, Math.min(ps.lcp, ps.nPrompt));

    if (nPast > 0) {
        // where the restored prefix physically came from
        let src = req.restore && req.restore.src !== "miss" ? req.restore.src : "resident";
        if (src === "donor") src = "donor";
        spans.push({ start: 0, end: nPast, tokens: nPast, kind: "restored", src });
    }
    if (lcp > nPast) {
        // tokens that WERE available as a matching prefix but had to be
        // re-decoded anyway: the landing fell below the reusable prefix
        spans.push({ start: nPast, end: lcp, tokens: lcp - nPast, kind: "recomputed", why: ps.gapWhy ?? "checkpoint_gap" });
    }
    if (ps.nPrompt > lcp) {
        // tokens beyond the common prefix: never available anywhere
        const why = lcp > 0 || nPast > 0 ? "divergent_tail" : "no_cache";
        spans.push({ start: lcp, end: ps.nPrompt, tokens: ps.nPrompt - lcp, kind: "recomputed", why });
    }
    return spans;
}

// ---------------------------------------------------------------------------
// Time segments for the waterfall
// ---------------------------------------------------------------------------

// Returns { start, end, live, segments: [{kind, t0, t1, src?}] } in server
// wall-clock ms. Segment kinds: queued | restore | prefill | decode.
export function timeSegments(req, nowMs) {
    const t0 = req.queuedAt ?? req.dispatchedAt ?? req.promptStart?.t ?? req.lastAt;
    if (t0 === null || t0 === undefined) return null;
    const live = req.terminal === null && req.finished === null;
    const end = req.terminal?.t ?? req.finished?.t ?? nowMs;

    const segments = [];
    const tDispatch = req.dispatchedAt ?? t0;
    if (tDispatch > t0) {
        segments.push({ kind: "queued", t0, t1: tDispatch });
    }

    const tPrompt = req.promptStart?.t ?? null;
    if (tPrompt !== null) {
        if (tPrompt > tDispatch) {
            // between dispatch and the prefill landing sits slot selection +
            // cache consult/restore; color it by the restore source
            segments.push({
                kind: "restore",
                t0: tDispatch,
                t1: tPrompt,
                src: req.restore && req.restore.src !== "miss" ? req.restore.src : null,
            });
        }

        // decode start: prefer the exact server timing (finished.tgMs), fall
        // back to the first decode_progress anchor, else still prefilling
        let tDecode = null;
        if (req.finished !== null) {
            tDecode = req.finished.t - req.finished.tgMs;
        } else if (req.decode.length > 0) {
            const d = req.decode[0];
            tDecode = d.t - d.ms;
        }
        if (tDecode !== null && tDecode > tPrompt) {
            segments.push({ kind: "prefill", t0: tPrompt, t1: Math.min(tDecode, end) });
            segments.push({ kind: "decode", t0: Math.min(tDecode, end), t1: end });
        } else if (tDecode !== null) {
            segments.push({ kind: "decode", t0: tPrompt, t1: end });
        } else {
            segments.push({ kind: "prefill", t0: tPrompt, t1: end });
        }
    } else if (end > tDispatch) {
        // dispatched but no prefill landing yet (admission-deferred retries)
        segments.push({ kind: "queued", t0: tDispatch, t1: end });
    }

    return { start: t0, end, live, segments };
}

// Work the server did and then threw away.
//
// The prompt-cache consult runs in get_available_slot, but the DSV4 elastic
// admission that follows (prepare_dsv4_admission) may clear the slot's KV
// outright — which discards the prefix that was just restored, including the
// SSD read that paid for it. That is invisible in any single event and is
// exactly the kind of waste an operator wants flagged.
//
// Returns null when nothing was wasted, else {tokens, ms, bytes, why}.
export function wastedRestore(req) {
    if (!req.admission || req.admission.outcome !== "ready_full_clear") return null;
    const r = req.restore;
    if (!r || r.src === "miss" || !(r.tokens > 0)) return null;
    // the landing confirms the restored prefix did not survive
    if (req.promptStart && req.promptStart.nPast > 0) return null;
    return { tokens: r.tokens, ms: r.ms ?? 0, bytes: r.bytes ?? 0, why: req.admission.why ?? null };
}

// Per-request normalized waterfall ("fit" scale).
//
// The shared wall-clock axis answers "what ran at the same time", but it cannot
// answer "where did this request's time go" when the rows on screen span three
// orders of magnitude: on a 45-minute axis a 2-second request is 0.07 % wide.
// This projects one request's segments onto its OWN duration, so a 2 s request
// and a 40 min agent turn are both fully legible; the absolute duration is
// always shown as a number beside the bar, so normalizing never hides scale.
//
// Returns { total, phases: [{kind, src?, ms, frac}] } with sum(frac) == 1,
// or null when the request has no usable time extent yet.
export function phaseBreakdown(req, nowMs) {
    const seg = timeSegments(req, nowMs);
    if (!seg) return null;
    const total = seg.end - seg.start;
    if (!(total > 0)) return null;
    const phases = seg.segments
        .map((s) => ({ kind: s.kind, src: s.src ?? null, ms: Math.max(0, s.t1 - s.t0) }))
        .filter((p) => p.ms > 0);
    if (phases.length === 0) return null;
    const sum = phases.reduce((acc, p) => acc + p.ms, 0) || 1;
    for (const p of phases) p.frac = p.ms / sum;
    return { total, phases, live: seg.live };
}

// ---------------------------------------------------------------------------
// Cache view shaping
// ---------------------------------------------------------------------------

export function cacheView(stateJson) {
    if (!stateJson || stateJson.v !== EVENTS_SCHEMA_V) return null;
    const c = stateJson.counters ?? {};
    const hits = (c.hits_entry ?? 0) + (c.hits_resident ?? 0);
    const lookups = c.lookups ?? 0;
    return {
        enabled: stateJson.enabled === true,
        ageMs: stateJson.age_ms ?? 0,
        ram: {
            used: stateJson.ram?.used_bytes ?? 0,
            limit: stateJson.ram?.limit_bytes ?? 0,
            tokens: stateJson.ram?.tokens ?? 0,
            fraction: fraction(stateJson.ram?.used_bytes, stateJson.ram?.limit_bytes),
        },
        disk: {
            used: stateJson.disk?.used_bytes ?? 0,
            limit: stateJson.disk?.limit_bytes ?? 0,
            fraction: fraction(stateJson.disk?.used_bytes, stateJson.disk?.limit_bytes),
        },
        counters: c,
        hitRate: lookups > 0 ? hits / lookups : null,
        entries: (stateJson.entries ?? []).map((e) => ({
            id: e.id,
            tier: e.tier,
            tokens: e.tokens ?? 0,
            bytes: e.bytes ?? 0,
            ageS: e.age_s ?? 0,
            lastHitS: e.last_hit_s ?? null,
            hits: e.hits ?? 0,
            file: e.file ?? null,
            // v3, additive: which request's save created the entry, and whether
            // a text preview can be fetched for it. The preview TEXT is never
            // in this payload — it comes from /m2-dashboard/cache-preview, one
            // entry at a time, on request.
            req: e.req ?? null,
            preview: e.preview === true,
        })),
    };
}

function fraction(used, limit) {
    if (!(limit > 0)) return null;
    return Math.min(1, (used ?? 0) / limit);
}

// rolling hit-rate points for a sparkline: fraction of hits over a trailing
// window of consults, sampled at each consult
export function hitRateSeries(lookups, { window = 20 } = {}) {
    const points = [];
    let sum = 0;
    const q = [];
    for (const item of lookups) {
        q.push(item.hit);
        sum += item.hit;
        if (q.length > window) sum -= q.shift();
        points.push({ t: item.t, v: sum / q.length });
    }
    return points;
}

// ---------------------------------------------------------------------------
// Reconnecting SSE stream (fetch-based so the API key rides an Authorization
// header; EventSource cannot set headers)
// ---------------------------------------------------------------------------

export const STREAM_BACKOFF_BASE_MS = 1000;
export const STREAM_BACKOFF_MAX_MS = 30000;
export const STREAM_FALLBACK_AFTER = 4; // consecutive failures before "degraded"

export class LiveEventStream {
    // onEvent(parsedJson), onStatus("connecting"|"open"|"retrying"|"degraded"|"stopped")
    constructor({ url = EVENTS_PATH, apiKey = "", onEvent, onStatus = () => {}, fetchImpl = globalThis.fetch?.bind(globalThis), setTimeoutImpl = setTimeout, clearTimeoutImpl = clearTimeout }) {
        this.url = url;
        this.onEvent = onEvent;
        this.onStatus = onStatus;
        this.failures = 0;
        this.lastEventId = "";
        this.stopped = false;
        this.cancel = null;
        this.timer = null;
        this.setTimeoutImpl = setTimeoutImpl;
        this.clearTimeoutImpl = clearTimeoutImpl;
        const authedFetch = (input, init = {}) => fetchImpl(input, {
            ...init,
            headers: { ...(init.headers ?? {}), ...(apiKey ? { Authorization: "Bearer " + apiKey } : {}) },
        });
        this.transport = createFetchSseTransport(url, authedFetch);
    }

    start() {
        if (this.stopped) return;
        this.onStatus(this.failures > 0 ? "retrying" : "connecting");
        this.cancel = this.transport({
            lastEventId: this.lastEventId,
            onOpen: () => {
                this.onStatus("open");
            },
            onEvent: (msg) => {
                if (msg.lastEventId) this.lastEventId = msg.lastEventId;
                let parsed = null;
                try {
                    parsed = JSON.parse(msg.data);
                } catch {
                    return; // malformed frame: skip, the stream continues
                }
                if (parsed?.k === "hello") {
                    // the server's greeting proves a working stream; a merely
                    // established-then-dropped connection does NOT reset the
                    // failure count, so flapping still escalates to degraded
                    this.failures = 0;
                }
                this.onEvent(parsed);
            },
            onDisconnect: (error) => {
                if (this.stopped) return;
                this.failures += 1;
                const auth = typeof error?.message === "string" && /\b(401|403)\b/.test(error.message);
                if (auth) {
                    this.onStatus("degraded");
                    return; // credentials problem: do not hammer the server
                }
                this.onStatus(this.failures >= STREAM_FALLBACK_AFTER ? "degraded" : "retrying");
                const wait = Math.min(STREAM_BACKOFF_MAX_MS, STREAM_BACKOFF_BASE_MS * 2 ** (this.failures - 1));
                this.timer = this.setTimeoutImpl(() => this.start(), wait);
            },
        });
    }

    stop() {
        this.stopped = true;
        this.clearTimeoutImpl(this.timer);
        if (this.cancel) this.cancel();
        this.onStatus("stopped");
    }
}

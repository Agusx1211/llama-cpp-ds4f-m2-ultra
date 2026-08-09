// Dashboard v3 content model (DOM-free, node-testable).
//
// v2 showed only metadata. v3 can additionally show, on explicit request:
//   - what a request was actually sent (INPUT) and what it produced (OUTPUT),
//     from GET /m2-dashboard/content;
//   - a live token-by-token mirror of one in-flight request, from the
//     GET /m2-dashboard/watch SSE stream;
//   - the decoded text sitting in one prompt-cache entry, from
//     GET /m2-dashboard/cache-preview.
//
// Design rules this module encodes (see notes/2026-08-09-dashboard-v3.md):
//   - content is NEVER on the event stream and never in the replay ring, so
//     nothing here reads from EventStore;
//   - every server payload is bounded (head + tail with an explicit elision
//     count) and this module keeps its own client-side bound on top, so a page
//     left open on a 45-minute agent turn cannot grow without limit;
//   - reveal is per request / per entry and lives only in memory: nothing is
//     written to localStorage, the URL, or any log.

import { createFetchSseTransport } from "./sse.mjs";

export const CONTENT_SCHEMA_V = 1;
export const CONTENT_PATH = "/m2-dashboard/content";
export const WATCH_PATH = "/m2-dashboard/watch";
export const CACHE_PREVIEW_PATH = "/m2-dashboard/cache-preview";

// Client-side ceiling on a watched generation. The server mirror is bounded
// per-poll, not per-request: a 45-minute turn emits megabytes over its life, so
// the page keeps only the newest slice and says how much it dropped.
export const WATCH_TEXT_MAX = 256 * 1024;
export const WATCH_TEXT_TRIM_TO = 192 * 1024;

// ---------------------------------------------------------------------------
// GET /m2-dashboard/content
// ---------------------------------------------------------------------------

function sideView(raw, kind) {
    if (!raw || typeof raw !== "object") return null;
    const head = typeof raw.head === "string" ? raw.head : "";
    const tail = typeof raw.tail === "string" ? raw.tail : "";
    return {
        kind,                                  // "input" | "output"
        nTokens: raw.n_tokens ?? null,
        nBytes: raw.n_bytes ?? null,
        head,
        tail,
        headTokens: raw.head_tokens ?? null,
        tailTokens: raw.tail_tokens ?? null,
        elidedTokens: raw.elided_tokens ?? 0,
        elidedBytes: raw.elided_bytes ?? 0,
        truncated: raw.truncated === true,
        empty: head.length === 0 && tail.length === 0,
    };
}

// Normalize a /m2-dashboard/content body. Returns null for anything that is not
// a schema-v1 payload, so an unexpected response can never be rendered as text.
export function parseContent(json) {
    if (!json || typeof json !== "object" || json.v !== CONTENT_SCHEMA_V) return null;
    const caps = json.caps ?? {};
    const store = json.store ?? {};
    const view = {
        req: json.req ?? null,
        found: json.found === true,
        reason: json.reason ?? null,     // "absent" | "evicted" when !found
        state: json.state ?? null,       // "live" | "final"
        slot: json.slot ?? null,
        tMs: json.t_ms ?? null,
        caps: {
            inHeadTokens: caps.in_head_tokens ?? null,
            inTailTokens: caps.in_tail_tokens ?? null,
            outHeadBytes: caps.out_head_bytes ?? null,
            outTailBytes: caps.out_tail_bytes ?? null,
            maxRequests: caps.max_requests ?? null,
            maxBytes: caps.max_bytes ?? null,
        },
        store: {
            requests: store.requests ?? 0,
            bytes: store.bytes ?? 0,
            evicted: store.evicted ?? 0,
        },
        input: sideView(json.input, "input"),
        output: sideView(json.output, "output"),
    };
    return view;
}

// One-line provenance for a rendered block: what fraction of the real thing is
// on screen. This is the anti-"looks complete" guard — a truncated prompt must
// never read as the whole prompt.
export function extentLabel(side) {
    if (!side) return "";
    if (side.kind === "input") {
        const total = side.nTokens ?? 0;
        if (!side.truncated) return formatCount(total) + " tokens · complete";
        return formatCount(total) + " tokens · showing first " + formatCount(side.headTokens ?? 0) +
            " and last " + formatCount(side.tailTokens ?? 0) + " · " +
            formatCount(side.elidedTokens ?? 0) + " elided";
    }
    const tokens = side.nTokens ?? 0;
    const bytes = side.nBytes ?? 0;
    if (!side.truncated) return formatCount(tokens) + " tokens · " + formatCount(bytes) + " bytes · complete";
    return formatCount(tokens) + " tokens · " + formatCount(bytes) + " bytes · " +
        formatCount(side.elidedBytes ?? 0) + " bytes elided from the middle";
}

function formatCount(n) {
    if (!Number.isFinite(n)) return "—";
    return n.toLocaleString("en-US");
}

// ---------------------------------------------------------------------------
// GET /m2-dashboard/cache-preview
// ---------------------------------------------------------------------------

export function parseCachePreview(json) {
    if (!json || typeof json !== "object" || json.v !== CONTENT_SCHEMA_V) return null;
    if (json.found !== true) {
        return { id: json.id ?? null, found: false };
    }
    const head = typeof json.head === "string" ? json.head : "";
    const tail = typeof json.tail === "string" ? json.tail : "";
    return {
        id: json.id ?? null,
        found: true,
        tier: json.tier ?? null,
        tokens: json.tokens ?? 0,
        head,
        tail,
        headTokens: json.head_tokens ?? 0,
        tailTokens: json.tail_tokens ?? 0,
        elidedTokens: json.elided_tokens ?? 0,
        truncated: json.truncated === true,
        req: json.req ?? null,
        ageMs: json.age_ms ?? 0,
        empty: head.length === 0 && tail.length === 0,
    };
}

export function cachePreviewLabel(preview) {
    if (!preview || !preview.found) return "";
    if (!preview.truncated) return formatCount(preview.tokens) + " tokens · complete";
    return formatCount(preview.tokens) + " cached tokens · showing first " +
        formatCount(preview.headTokens) + " and last " + formatCount(preview.tailTokens) +
        " · " + formatCount(preview.elidedTokens) + " elided";
}

// ---------------------------------------------------------------------------
// GET /m2-dashboard/watch — live token mirror
// ---------------------------------------------------------------------------

export function emptyWatchState(req) {
    return {
        req,
        slot: null,
        phase: null,       // server's view: "live" | "final" | "absent"
        text: "",
        cursor: 0,
        nDec: 0,
        dropped: 0,        // bytes the SERVER dropped (reader lagged)
        clientDropped: 0,  // bytes this page dropped to stay bounded
        input: null,
        ended: false,
        reason: null,
        frames: 0,
    };
}

// Fold one parsed watch frame into the accumulated state. Pure: returns a new
// object, never mutates, so it is directly unit-testable.
//
// `delta` appends. `rewind` supersedes everything the reader holds — either the
// slot trimmed a stop word off its own generated text (so what follows the trim
// is wrong) or the reader fell behind the server's bounded mirror. On a rewind
// the frame's `offset` (falling back to `cursor`, which for a rewind carries the
// same value) is the absolute byte offset at which the replacement text starts:
// non-zero means earlier output can no longer be recovered.
export function applyWatchFrame(state, frame) {
    if (!frame || typeof frame !== "object" || frame.v !== CONTENT_SCHEMA_V) return state;
    const next = { ...state, frames: state.frames + 1 };
    switch (frame.k) {
        case "watch-hello":
            next.slot = frame.slot ?? null;
            next.phase = frame.state ?? null;
            next.nDec = frame.n_dec ?? 0;
            next.cursor = frame.cursor ?? 0;
            next.dropped = frame.dropped ?? 0;
            next.input = frame.input ?? null;
            next.text = typeof frame.text === "string" ? frame.text : "";
            break;
        case "rewind": {
            const startsAt = frame.offset ?? frame.cursor ?? 0;
            next.text = typeof frame.text === "string" ? frame.text : "";
            next.cursor = startsAt + next.text.length;
            next.nDec = frame.n_dec ?? next.nDec;
            next.dropped = Math.max(frame.dropped ?? 0, startsAt);
            break;
        }
        case "delta":
            next.text = state.text + (typeof frame.text === "string" ? frame.text : "");
            next.cursor = frame.cursor ?? next.cursor;
            next.nDec = frame.n_dec ?? next.nDec;
            next.dropped = frame.dropped ?? next.dropped;
            break;
        case "end":
            next.ended = true;
            next.reason = frame.reason ?? "finished";
            next.nDec = frame.n_dec ?? next.nDec;
            break;
        default:
            return state; // unknown kind: forward-compatible skip
    }
    if (next.text.length > WATCH_TEXT_MAX) {
        const cut = next.text.length - WATCH_TEXT_TRIM_TO;
        next.text = next.text.slice(cut);
        next.clientDropped = next.clientDropped + cut;
    }
    return next;
}

// A single-shot SSE consumer for one request's mirror.
//
// Unlike LiveEventStream this deliberately does NOT reconnect: the stream arms
// a per-token hook on the server for as long as it is open, so a dead watch
// must stay dead until a human asks again rather than silently re-arming.
export class WatchStream {
    // onUpdate(state), onClose(state)
    constructor({
        req,
        apiKey = "",
        onUpdate = () => {},
        onClose = () => {},
        url = WATCH_PATH,
        fetchImpl = globalThis.fetch?.bind(globalThis),
    }) {
        this.req = req;
        this.state = emptyWatchState(req);
        this.onUpdate = onUpdate;
        this.onClose = onClose;
        this.cancel = null;
        this.stopped = false;
        const authedFetch = (input, init = {}) => fetchImpl(input, {
            ...init,
            headers: { ...(init.headers ?? {}), ...(apiKey ? { Authorization: "Bearer " + apiKey } : {}) },
        });
        this.transport = createFetchSseTransport(url + "?req=" + encodeURIComponent(String(req)), authedFetch);
    }

    start() {
        if (this.stopped) return;
        this.cancel = this.transport({
            onEvent: (msg) => {
                let parsed = null;
                try {
                    parsed = JSON.parse(msg.data);
                } catch {
                    return;
                }
                this.state = applyWatchFrame(this.state, parsed);
                this.onUpdate(this.state);
                if (this.state.ended) this.stop();
            },
            onDisconnect: (error) => {
                if (this.stopped) return;
                this.state = { ...this.state, ended: true, reason: this.state.reason ?? errorReason(error) };
                this.stopped = true;
                this.onUpdate(this.state);
                this.onClose(this.state);
            },
        });
    }

    stop() {
        if (this.stopped) {
            if (this.cancel) { this.cancel(); this.cancel = null; }
            return;
        }
        this.stopped = true;
        if (this.cancel) { this.cancel(); this.cancel = null; }
        this.onClose(this.state);
    }
}

function errorReason(error) {
    const message = typeof error?.message === "string" ? error.message : "";
    if (/\b429\b/.test(message)) return "busy";
    if (/\b(401|403)\b/.test(message)) return "denied";
    if (/\b501\b/.test(message)) return "disabled";
    return "disconnected";
}

export const WATCH_REASON_LABEL = {
    finished: "generation finished",
    gone: "request is no longer running",
    idle: "no output for 3 minutes — stream closed",
    busy: "both watch slots are in use — try again in a moment",
    denied: "not authorized",
    disabled: "content disclosure is disabled on this server",
    disconnected: "stream disconnected",
};

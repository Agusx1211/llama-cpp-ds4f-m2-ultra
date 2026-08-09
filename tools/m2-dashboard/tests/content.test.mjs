// Dashboard v3 content model tests (lib/content.mjs).
//
// The invariants under test are the ones that keep a bounded preview honest and
// a page bounded:
//   - a truncated payload must never describe itself as complete;
//   - a non-v1 or malformed body must never be rendered as content;
//   - watch frames fold deterministically, including the rewind case that a
//     stop-word trim (or a lagging reader) produces;
//   - the accumulated live text has a hard client-side ceiling.

import assert from "node:assert/strict";
import test from "node:test";

import {
    CONTENT_SCHEMA_V,
    WATCH_REASON_LABEL,
    WATCH_RENDER_CHARS,
    WATCH_TEXT_MAX,
    WATCH_TEXT_TRIM_TO,
    WatchStream,
    applyWatchFrame,
    cachePreviewLabel,
    emptyWatchState,
    extentLabel,
    parseCachePreview,
    parseContent,
    watchRenderView,
} from "../lib/content.mjs";

const CAPS = {
    in_head_tokens: 512, in_tail_tokens: 256,
    out_head_bytes: 4096, out_tail_bytes: 2048,
    max_requests: 64, max_bytes: 1048576,
};
const STORE = { requests: 37, bytes: 412344, evicted: 12 };

function contentBody(overrides = {}) {
    return {
        v: 1, req: 1234, found: true, state: "final", slot: 2, t_ms: 1700000000000,
        caps: CAPS, store: STORE,
        input: {
            n_tokens: 118432, head: "SYSTEM…", tail: "…user: go",
            head_tokens: 512, tail_tokens: 256, elided_tokens: 117664, truncated: true,
        },
        output: {
            n_tokens: 512, n_bytes: 2044, head: "answer", tail: "",
            elided_bytes: 0, truncated: false,
        },
        ...overrides,
    };
}

// ---------------------------------------------------------------------------
// /m2-dashboard/content
// ---------------------------------------------------------------------------

test("parseContent normalizes a v1 body", () => {
    const view = parseContent(contentBody());
    assert.equal(view.req, 1234);
    assert.equal(view.found, true);
    assert.equal(view.state, "final");
    assert.equal(view.input.nTokens, 118432);
    assert.equal(view.input.truncated, true);
    assert.equal(view.input.elidedTokens, 117664);
    assert.equal(view.output.truncated, false);
    assert.equal(view.caps.maxRequests, 64);
    assert.equal(view.store.evicted, 12);
});

test("parseContent refuses anything that is not schema v1", () => {
    assert.equal(parseContent(null), null);
    assert.equal(parseContent("a string"), null);
    assert.equal(parseContent({ found: true, input: { head: "leak" } }), null);
    assert.equal(parseContent({ ...contentBody(), v: CONTENT_SCHEMA_V + 1 }), null);
});

test("parseContent carries the not-retained reason instead of inventing content", () => {
    const view = parseContent({ v: 1, req: 9, found: false, reason: "evicted", caps: CAPS, store: STORE });
    assert.equal(view.found, false);
    assert.equal(view.reason, "evicted");
    assert.equal(view.input, null);
    assert.equal(view.output, null);
});

test("parseContent keeps output null while the request is still generating", () => {
    const view = parseContent(contentBody({ state: "live", output: null }));
    assert.equal(view.state, "live");
    assert.equal(view.output, null);
});

test("a target-sized full payload is rendered as complete", () => {
    const view = parseContent(contentBody({
        caps: {
            in_head_tokens: 524288, in_tail_tokens: 0,
            out_head_bytes: 8 * 1024 * 1024, out_tail_bytes: 0,
            max_requests: 64, max_bytes: 256 * 1024 * 1024,
        },
        input: {
            n_tokens: 118432, head: "complete prompt", tail: "",
            head_tokens: 118432, tail_tokens: 0, elided_tokens: 0, truncated: false,
        },
    }));
    assert.equal(view.input.truncated, false);
    assert.match(extentLabel(view.input), /118,432 tokens · complete/);
    assert.equal(view.caps.maxBytes, 256 * 1024 * 1024);
});

test("a side with no text is flagged empty rather than rendered as blank content", () => {
    const view = parseContent(contentBody({
        input: { n_tokens: 0, head: "", tail: "", head_tokens: 0, tail_tokens: 0, elided_tokens: 0, truncated: false },
    }));
    assert.equal(view.input.empty, true);
});

test("extentLabel never says complete for a truncated side", () => {
    const view = parseContent(contentBody());
    const input = extentLabel(view.input);
    assert.match(input, /118,432 tokens/);
    assert.match(input, /showing first 512 and last 256/);
    assert.match(input, /117,664 elided/);
    assert.ok(!input.includes("complete"));

    const output = extentLabel(view.output);
    assert.match(output, /complete/);
    assert.ok(!output.includes("elided"));
});

test("extentLabel reports output truncation in bytes, not tokens", () => {
    const view = parseContent(contentBody({
        output: { n_tokens: 4096, n_bytes: 40000, head: "a", tail: "z", elided_bytes: 33856, truncated: true },
    }));
    const label = extentLabel(view.output);
    assert.match(label, /33,856 bytes elided from the middle/);
    assert.ok(!label.includes("complete"));
});

// ---------------------------------------------------------------------------
// /m2-dashboard/cache-preview
// ---------------------------------------------------------------------------

test("parseCachePreview normalizes and labels a truncated entry", () => {
    const preview = parseCachePreview({
        v: 1, id: 42, found: true, tier: "disk", tokens: 118432,
        head: "You are an agent", tail: "user: go", head_tokens: 64, tail_tokens: 32,
        elided_tokens: 118336, truncated: true, req: 1188, age_ms: 3120,
    });
    assert.equal(preview.found, true);
    assert.equal(preview.tier, "disk");
    assert.equal(preview.req, 1188);
    const label = cachePreviewLabel(preview);
    assert.match(label, /118,432 cached tokens/);
    assert.match(label, /first 64 and last 32/);
    assert.match(label, /118,336 elided/);
});

test("parseCachePreview passes through a not-found entry and rejects a bad schema", () => {
    assert.deepEqual(parseCachePreview({ v: 1, id: 7, found: false }), { id: 7, found: false });
    assert.equal(parseCachePreview({ id: 7, found: true, head: "leak" }), null);
    assert.equal(parseCachePreview(undefined), null);
});

// ---------------------------------------------------------------------------
// /m2-dashboard/watch frame folding
// ---------------------------------------------------------------------------

const hello = (over = {}) => ({
    v: 1, k: "watch-hello", req: 5, slot: 1, state: "live", n_dec: 0,
    cursor: 0, text: "", dropped: 0, input: null, ...over,
});
const delta = (text, over = {}) => ({ v: 1, k: "delta", req: 5, text, cursor: 0, n_dec: 0, ...over });

test("watch frames fold hello then deltas into one growing text", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello({ slot: 3, input: { n_tokens: 12 } }));
    assert.equal(s.slot, 3);
    assert.equal(s.phase, "live");
    assert.equal(s.input.n_tokens, 12);

    s = applyWatchFrame(s, delta("Hello", { cursor: 5, n_dec: 2 }));
    s = applyWatchFrame(s, delta(" world", { cursor: 11, n_dec: 4 }));
    assert.equal(s.text, "Hello world");
    assert.equal(s.cursor, 11);
    assert.equal(s.nDec, 4);
    assert.equal(s.ended, false);
    assert.equal(s.frames, 3);
});

test("a rewind replaces everything the reader holds", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    s = applyWatchFrame(s, delta("Hello world STOPWORD", { cursor: 20, n_dec: 6 }));
    s = applyWatchFrame(s, { v: 1, k: "rewind", req: 5, cursor: 0, offset: 0, text: "Hello world", n_dec: 6 });
    assert.equal(s.text, "Hello world");
    assert.equal(s.cursor, 11);
});

test("a rewind that starts mid-stream records the unrecoverable prefix", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    s = applyWatchFrame(s, delta("abc", { cursor: 3 }));
    // the reader lagged: the server kept only the newest bytes
    s = applyWatchFrame(s, { v: 1, k: "rewind", req: 5, cursor: 900, offset: 900, text: "tail", n_dec: 40 });
    assert.equal(s.text, "tail");
    assert.equal(s.cursor, 904);
    assert.equal(s.dropped, 900);
});

test("a rewind falls back to cursor when offset is absent (older server)", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    s = applyWatchFrame(s, { v: 1, k: "rewind", req: 5, cursor: 64, text: "tail", n_dec: 9 });
    assert.equal(s.dropped, 64);
    assert.equal(s.cursor, 68);
});

test("end closes the stream and keeps the reason", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    s = applyWatchFrame(s, delta("out", { cursor: 3, n_dec: 1 }));
    s = applyWatchFrame(s, { v: 1, k: "end", req: 5, reason: "finished", n_dec: 512 });
    assert.equal(s.ended, true);
    assert.equal(s.reason, "finished");
    assert.equal(s.nDec, 512);
    assert.ok(WATCH_REASON_LABEL[s.reason]);
});

test("unknown kinds and wrong schema versions are ignored, not rendered", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    const before = s;
    s = applyWatchFrame(s, { v: 1, k: "something-new", req: 5, text: "ignore me" });
    assert.equal(s, before, "unknown kind must not change state at all");
    s = applyWatchFrame(s, { v: 99, k: "delta", req: 5, text: "wrong version" });
    assert.equal(s.text, "");
    s = applyWatchFrame(s, null);
    assert.equal(s.text, "");
});

test("accumulated live text is bounded and reports what the page dropped", () => {
    let s = emptyWatchState(5);
    s = applyWatchFrame(s, hello());
    const chunk = "x".repeat(WATCH_TEXT_MAX + 64 * 1024);
    s = applyWatchFrame(s, delta(chunk, { cursor: chunk.length }));
    assert.ok(s.text.length <= WATCH_TEXT_MAX, "text must stay under the client ceiling");
    assert.equal(s.text.length, WATCH_TEXT_TRIM_TO);
    assert.ok(s.clientDropped > 0, "the page must account for what it dropped");
    assert.equal(s.text.length + s.clientDropped, chunk.length);
});

test("live rendering can reveal the complete accumulated target output", () => {
    const text = "x".repeat(WATCH_RENDER_CHARS + 4096);
    const state = { ...emptyWatchState(5), text };
    const tail = watchRenderView(state, false);
    assert.equal(tail.text.length, WATCH_RENDER_CHARS);
    assert.equal(tail.clipped, true);
    assert.equal(tail.hidden, 4096);
    const full = watchRenderView(state, true);
    assert.equal(full.text, text);
    assert.equal(full.clipped, true);
    assert.equal(full.hidden, 0);
    assert.equal(full.dropped, 0);
});

// ---------------------------------------------------------------------------
// WatchStream transport behaviour
// ---------------------------------------------------------------------------

function sseResponse(chunks) {
    const encoder = new TextEncoder();
    let i = 0;
    return {
        ok: true,
        status: 200,
        statusText: "OK",
        body: {
            getReader: () => ({
                read: async () => (i < chunks.length
                    ? { value: encoder.encode(chunks[i++]), done: false }
                    : { value: undefined, done: true }),
            }),
        },
    };
}

test("WatchStream sends the key, folds frames, and stops on end", async () => {
    const seen = [];
    let calledWith = null;
    const fetchImpl = async (url, init) => {
        calledWith = { url, init };
        return sseResponse([
            "event: watch-hello\ndata: " + JSON.stringify(hello({ n_dec: 3, text: "so far" })) + "\n\n",
            "data: " + JSON.stringify(delta(" more", { cursor: 11, n_dec: 5 })) + "\n\n",
            "event: watch-end\ndata: " + JSON.stringify({ v: 1, k: "end", req: 5, reason: "finished", n_dec: 5 }) + "\n\n",
        ]);
    };
    const closed = [];
    const stream = new WatchStream({
        req: 5, apiKey: "secret", fetchImpl,
        onUpdate: (s) => seen.push(s.text),
        onClose: (s) => closed.push(s),
    });
    stream.start();
    await new Promise((r) => setTimeout(r, 20));

    assert.match(calledWith.url, /\/m2-dashboard\/watch\?req=5$/);
    assert.equal(calledWith.init.headers.Authorization, "Bearer secret");
    assert.deepEqual(seen, ["so far", "so far more", "so far more"]);
    assert.equal(closed.length, 1);
    assert.equal(closed[0].ended, true);
    assert.equal(closed[0].reason, "finished");
});

test("WatchStream does not reconnect — the mirror must not silently re-arm", async () => {
    let calls = 0;
    const fetchImpl = async () => {
        calls += 1;
        return { ok: false, status: 429, statusText: "Too Many Requests", body: null };
    };
    const closed = [];
    const stream = new WatchStream({ req: 5, fetchImpl, onClose: (s) => closed.push(s) });
    stream.start();
    await new Promise((r) => setTimeout(r, 60));
    assert.equal(calls, 1, "a refused watch must not be retried automatically");
    assert.equal(closed.length, 1);
    assert.equal(closed[0].reason, "busy");
    assert.equal(WATCH_REASON_LABEL.busy, "both watch slots are in use — try again in a moment");
});

test("WatchStream maps auth and disabled failures to their own reasons", async () => {
    for (const [status, reason] of [[401, "denied"], [403, "denied"], [501, "disabled"], [500, "disconnected"]]) {
        const closed = [];
        const stream = new WatchStream({
            req: 5,
            fetchImpl: async () => ({ ok: false, status, statusText: "x", body: null }),
            onClose: (s) => closed.push(s),
        });
        stream.start();
        await new Promise((r) => setTimeout(r, 20));
        assert.equal(closed[0]?.reason, reason, "status " + status);
    }
});

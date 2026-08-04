import assert from "node:assert/strict";
import test from "node:test";

import { AdminStateClient } from "../lib/client.mjs";
import { createFetchSseTransport, SseParser } from "../lib/sse.mjs";
import {
    clone,
    loadEvents,
    loadSnapshot,
    snapshotAt,
    streamSignal,
} from "./helpers.mjs";

function fakeTimers() {
    const pending = [];
    return {
        pending,
        setTimer(callback, delay) {
            const timer = { callback, delay, cancelled: false };
            pending.push(timer);
            return timer;
        },
        clearTimer(timer) {
            timer.cancelled = true;
        },
        runNext() {
            const index = pending.findIndex((timer) => !timer.cancelled);
            assert.notEqual(index, -1, "expected a pending timer");
            const [timer] = pending.splice(index, 1);
            timer.callback();
            return timer;
        },
    };
}

function harness(snapshots, options = {}) {
    const openings = [];
    const observed = [];
    let snapshotIndex = 0;
    const client = new AdminStateClient({
        getSnapshot: async () => snapshots[Math.min(snapshotIndex++, snapshots.length - 1)],
        openEvents: (connection) => {
            const opening = { ...connection, closed: false };
            openings.push(opening);
            return () => {
                opening.closed = true;
            };
        },
        onState: (state) => observed.push(state),
        reconnectDelayMs: 0,
        ...options,
    });
    return { client, openings, observed, snapshotCalls: () => snapshotIndex };
}

test("client resumes reconnects from its last accepted event ID", async () => {
    const snapshot = await loadSnapshot();
    const [event] = await loadEvents();
    const timers = fakeTimers();
    const { client, openings } = harness([snapshot], {
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    assert.equal(openings[0].lastEventId, "100");
    openings[0].onEvent({ data: JSON.stringify(event), lastEventId: "101" });
    await client.idle();
    assert.equal(client.state.lastEventId, "101");

    openings[0].onDisconnect(new Error("fixture disconnect"));
    assert.equal(client.state.connection, "retry_wait");
    timers.runNext();
    assert.equal(openings[0].closed, true);
    assert.equal(openings[1].lastEventId, "101");
    assert.equal(client.state.connection, "live");
    client.stop();
});

test("client sequence gaps force snapshot replacement before reopening", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const replacement = snapshotAt(snapshot, 200);
    replacement.server.health = "healthy";
    const { client, openings, observed, snapshotCalls } = harness([snapshot, replacement]);

    await client.start();
    openings[0].onEvent(events[2]);
    await client.idle();

    assert.ok(observed.some((state) => state.synchronization.reason === "sequence_gap"));
    assert.equal(snapshotCalls(), 2);
    assert.equal(client.state.lastEventId, "200");
    assert.equal(client.state.snapshot.server.health, "healthy");
    assert.equal(openings.at(-1).lastEventId, "200");
    client.stop();
});

test("server overflow forces snapshot replacement", async () => {
    const snapshot = await loadSnapshot();
    const [template] = await loadEvents();
    const replacement = snapshotAt(snapshot, 150);
    const overflow = streamSignal(template, 140, "stream.overflow", {
        oldest_available_sequence: 140,
    });
    const { client, openings, observed } = harness([snapshot, replacement]);

    await client.start();
    openings[0].onEvent(overflow);
    await client.idle();

    assert.ok(observed.some((state) => state.synchronization.reason === "server_overflow"));
    assert.equal(client.state.lastEventId, "150");
    client.stop();
});

test("HTTP resume rejection forces snapshot replacement instead of blind stream retry", async () => {
    const snapshot = await loadSnapshot();
    const replacement = snapshotAt(snapshot, 151);
    const { client, openings, observed, snapshotCalls } = harness([snapshot, replacement]);

    await client.start();
    const error = new Error("SSE request failed: 409 Conflict");
    error.resnapshotRequired = true;
    openings[0].onDisconnect(error);
    await client.idle();

    assert.ok(observed.some((state) => state.synchronization.reason === "server_resume_rejected"));
    assert.equal(snapshotCalls(), 2);
    assert.equal(client.state.lastEventId, "151");
    assert.equal(openings.at(-1).lastEventId, "151");
    client.stop();
});

test("automatic resnapshot exposes rejection then recovers on a paced retry", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const replacement = snapshotAt(snapshot, 220);
    const timers = fakeTimers();
    let snapshotCalls = 0;
    const { client, openings, observed } = harness([snapshot], {
        getSnapshot: async () => {
            snapshotCalls += 1;
            if (snapshotCalls === 1) {
                return snapshot;
            }
            if (snapshotCalls === 2) {
                throw new Error("fixture snapshot unavailable");
            }
            return replacement;
        },
        snapshotRetryDelaysMs: [25, 100],
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    openings[0].onEvent(events[2]);
    await client.idle();

    assert.equal(client.state.connection, "retry_wait");
    assert.equal(client.state.recovery.status, "retry_wait");
    assert.equal(client.state.recovery.reason, "sequence_gap");
    assert.equal(client.state.recovery.lastError, "fixture snapshot unavailable");
    assert.equal(client.state.recovery.retryDelayMs, 25);
    assert.ok(observed.some((state) => state.recovery.lastError === "fixture snapshot unavailable"));

    timers.runNext();
    await client.idle();
    assert.equal(snapshotCalls, 3);
    assert.equal(client.state.connection, "live");
    assert.equal(client.state.recovery.status, "idle");
    assert.equal(client.state.lastEventId, "220");
    assert.equal(openings.at(-1).lastEventId, "220");
    client.stop();
});

test("stopping during snapshot retry prevents stale retry work and stream reopen", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const timers = fakeTimers();
    let snapshotCalls = 0;
    const { client, openings } = harness([snapshot], {
        getSnapshot: async () => {
            snapshotCalls += 1;
            if (snapshotCalls === 1) {
                return snapshot;
            }
            throw new Error("still unavailable");
        },
        snapshotRetryDelaysMs: [10],
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    openings[0].onEvent(events[2]);
    await client.idle();
    const retry = timers.pending.find((timer) => !timer.cancelled);
    assert.ok(retry);

    client.stop();
    retry.callback();
    await Promise.resolve();

    assert.equal(snapshotCalls, 2);
    assert.equal(openings.length, 1);
    assert.equal(client.state.connection, "stopped");
});

test("snapshot retries are bounded and end in an observable error state", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const timers = fakeTimers();
    let snapshotCalls = 0;
    const { client, openings } = harness([snapshot], {
        getSnapshot: async () => {
            snapshotCalls += 1;
            if (snapshotCalls === 1) {
                return snapshot;
            }
            throw new Error(`snapshot failure ${snapshotCalls}`);
        },
        snapshotRetryDelaysMs: [15],
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    openings[0].onEvent(events[2]);
    await client.idle();
    timers.runNext();
    await client.idle();

    assert.equal(snapshotCalls, 3);
    assert.equal(client.state.connection, "error");
    assert.equal(client.state.recovery.status, "error");
    assert.equal(client.state.recovery.attempt, 2);
    assert.equal(client.state.recovery.lastError, "snapshot failure 3");
    assert.equal(timers.pending.some((timer) => !timer.cancelled), false);
    client.stop();
});

test("synchronous stream-open failure is caught and retried", async () => {
    const snapshot = await loadSnapshot();
    const timers = fakeTimers();
    const observed = [];
    let openCalls = 0;
    const client = new AdminStateClient({
        getSnapshot: async () => snapshot,
        openEvents: () => {
            openCalls += 1;
            if (openCalls === 1) {
                throw new Error("fixture open failed synchronously");
            }
            return () => {};
        },
        onState: (state) => observed.push(state),
        streamRetryDelaysMs: [40],
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    assert.equal(client.state.connection, "retry_wait");
    assert.equal(client.state.recovery.reason, "open_events_failed");
    assert.equal(client.state.recovery.lastError, "fixture open failed synchronously");
    assert.ok(observed.some((state) => state.recovery.retryDelayMs === 40));

    timers.runNext();
    assert.equal(openCalls, 2);
    assert.equal(client.state.connection, "live");
    assert.equal(client.state.recovery.status, "idle");
    client.stop();
});

test("stream-open retries stop at their configured bound", async () => {
    const snapshot = await loadSnapshot();
    const timers = fakeTimers();
    let openCalls = 0;
    const client = new AdminStateClient({
        getSnapshot: async () => snapshot,
        openEvents: () => {
            openCalls += 1;
            throw new Error(`open failure ${openCalls}`);
        },
        streamRetryDelaysMs: [20],
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });

    await client.start();
    timers.runNext();

    assert.equal(openCalls, 2);
    assert.equal(client.state.connection, "error");
    assert.equal(client.state.recovery.reason, "open_events_failed");
    assert.equal(client.state.recovery.lastError, "open failure 2");
    assert.equal(timers.pending.some((timer) => !timer.cancelled), false);
    client.stop();
});

test("bounded pending queue treats a local slow consumer as resnapshot pressure", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const replacement = snapshotAt(snapshot, 175);
    const { client, observed, snapshotCalls } = harness([snapshot, replacement], {
        pendingLimit: 2,
        autoDrain: false,
    });

    await client.start();
    client.enqueue(events[0]);
    client.enqueue(events[1]);
    client.enqueue(events[2]);
    await client.idle();

    assert.ok(observed.some((state) => state.synchronization.reason === "local_slow_consumer"));
    assert.equal(snapshotCalls(), 2);
    assert.equal(client.state.lastEventId, "175");
    assert.equal(client.pending.length, 0);
    client.stop();
});

test("SSE parser handles chunking, comments, IDs, and multiline data", () => {
    const messages = [];
    const parser = new SseParser((message) => messages.push(message));
    parser.push(": heartbeat\r\nid: 101\r\nevent: admin\r\ndata: {\"one\":\r\n");
    parser.push("data: 1}\r\n\r\n");
    parser.finish();

    assert.deepEqual(messages, [{
        data: "{\"one\":\n1}",
        lastEventId: "101",
        type: "admin",
    }]);
});

test("SSE parser preserves CRLF boundaries split across transport chunks", () => {
    const messages = [];
    const parser = new SseParser((message) => messages.push(message));
    parser.push("id: 101\r");
    parser.push("\ndata: payload\r");
    parser.push("\n\r");
    parser.push("\n");
    parser.finish();

    assert.deepEqual(messages, [{
        data: "payload",
        lastEventId: "101",
        type: "message",
    }]);
});

test("SSE parser bounds incomplete line buffers and complete event bytes", () => {
    const bufferParser = new SseParser(() => {}, {
        maxBufferBytes: 8,
        maxEventBytes: 64,
    });
    assert.throws(() => bufferParser.push("123456789"), /line buffer exceeds 8 bytes/);

    const eventParser = new SseParser(() => {}, {
        maxBufferBytes: 64,
        maxEventBytes: 16,
    });
    assert.throws(() => eventParser.push("data: 1234567890\n\n"), /event exceeds 16 bytes/);
});

test("one oversized queued SSE message forces resnapshot despite message-count capacity", async () => {
    const snapshot = await loadSnapshot();
    const replacement = snapshotAt(snapshot, 240);
    const { client, openings, observed } = harness([snapshot, replacement], {
        maxEventBytes: 64,
        pendingLimit: 8,
    });

    await client.start();
    openings[0].onEvent({
        data: JSON.stringify({ id: "101", padding: "x".repeat(128) }),
        lastEventId: "101",
    });
    await client.idle();

    assert.ok(observed.some((state) => state.synchronization.reason === "invalid_sse_message"));
    assert.equal(client.state.lastEventId, "240");
    client.stop();
});

test("fetch SSE transport sends the tracked Last-Event-ID header using GET", async () => {
    const calls = [];
    const encoded = new TextEncoder().encode("id: 101\ndata: {\"id\":\"101\"}\n\n");
    const fetchImpl = async (url, options) => {
        calls.push({ url, options });
        return {
            ok: true,
            status: 200,
            statusText: "OK",
            body: new ReadableStream({
                start(controller) {
                    controller.enqueue(encoded);
                    controller.close();
                },
            }),
        };
    };
    const messages = [];
    let disconnected = false;
    const open = createFetchSseTransport("/admin/v1/events", fetchImpl);
    const close = open({
        lastEventId: "100",
        onEvent: (message) => messages.push(message),
        onDisconnect: () => {
            disconnected = true;
        },
    });
    await new Promise((resolve) => setImmediate(resolve));

    assert.equal(calls.length, 1);
    assert.equal(calls[0].options.method, "GET");
    assert.equal(calls[0].options.headers["Last-Event-ID"], "100");
    assert.equal(messages[0].lastEventId, "101");
    assert.equal(disconnected, true);
    close();
});

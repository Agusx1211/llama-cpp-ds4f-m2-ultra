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
    const { client, openings } = harness([snapshot]);

    await client.start();
    assert.equal(openings[0].lastEventId, "100");
    openings[0].onEvent({ data: JSON.stringify(event), lastEventId: "101" });
    await client.idle();
    assert.equal(client.state.lastEventId, "101");

    openings[0].onDisconnect(new Error("fixture disconnect"));
    await new Promise((resolve) => setTimeout(resolve, 0));
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
    const overflow = streamSignal(template, 101, "stream.overflow", {
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

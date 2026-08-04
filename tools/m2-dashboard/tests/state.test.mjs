import assert from "node:assert/strict";
import test from "node:test";

import {
    createDashboardState,
    reduceDashboardEvent,
} from "../lib/state.mjs";
import {
    clone,
    loadEvents,
    loadSnapshot,
    streamSignal,
} from "./helpers.mjs";

test("sequential events immutably advance snapshot state and Last-Event-ID", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    let state = createDashboardState(snapshot, { historyLimit: 3 });
    const original = state;

    for (const event of events.slice(0, 4)) {
        state = reduceDashboardEvent(state, event);
    }

    assert.equal(original.lastEventId, "100");
    assert.equal(original.snapshot.server.health, "degraded");
    assert.equal(state.lastSequence, 104);
    assert.equal(state.lastEventId, "104");
    assert.equal(state.snapshot.server.health, "healthy");
    assert.equal(state.snapshot.requests.find((request) => request.id === "req-fast-a").output_tokens, 35);
    assert.equal(state.history.length, 3);
    assert.deepEqual(state.history.map((event) => event.sequence), [102, 103, 104]);
    assert.ok(Object.isFrozen(state));
    assert.ok(Object.isFrozen(state.snapshot.requests));
});

test("duplicate reconnect delivery is ignored without altering snapshot", async () => {
    const snapshot = await loadSnapshot();
    const [event] = await loadEvents();
    const once = reduceDashboardEvent(createDashboardState(snapshot), event);
    const duplicate = reduceDashboardEvent(once, clone(event));

    assert.equal(duplicate.lastSequence, 101);
    assert.equal(duplicate.history.length, 1);
    assert.equal(duplicate.synchronization.duplicates, 1);
    assert.equal(duplicate.snapshot, once.snapshot);
});

test("request events refresh authoritative registry counters", async () => {
    const snapshot = await loadSnapshot();
    const [fixture] = await loadEvents();
    const event = clone(fixture);
    event.payload.registry = {
        ...snapshot.registry,
        retained_events: snapshot.registry.retained_events + 1,
        total_events: snapshot.registry.total_events + 1,
    };

    const state = reduceDashboardEvent(createDashboardState(snapshot), event);
    assert.equal(state.snapshot.registry.total_events, snapshot.registry.total_events + 1);
    assert.equal(state.snapshot.registry.retained_events, snapshot.registry.retained_events + 1);
    assert.equal(snapshot.fast_refill.refill.fast_members_used, 2);
    assert.equal(state.snapshot.fast_refill.refill.fast_members_used, 3);
    assert.equal(state.snapshot.fast_refill.refill.fast_members_remaining, 1);
});

test("a sequence gap forces a full resnapshot before applying later deltas", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    const state = reduceDashboardEvent(createDashboardState(snapshot), events[2]);

    assert.equal(state.synchronization.status, "resnapshot_required");
    assert.equal(state.synchronization.reason, "sequence_gap");
    assert.deepEqual(state.synchronization.details, {
        expected: 101,
        received: 103,
        lastEventId: "100",
    });
    assert.equal(state.snapshot.sequence, 100);
    assert.equal(state.history.length, 0);
});

test("explicit server overflow and slow-client signals force resnapshot", async () => {
    const snapshot = await loadSnapshot();
    const [template] = await loadEvents();

    const overflow = streamSignal(template, 140, "stream.overflow", {
        oldest_available_sequence: 120,
    });
    const overflowState = reduceDashboardEvent(createDashboardState(snapshot), overflow);
    assert.equal(overflowState.synchronization.reason, "server_overflow");
    assert.equal(overflowState.snapshot.sequence, 100);

    const slow = streamSignal(template, 115, "stream.slow_client", {
        dropped_events: 9,
    });
    const slowState = reduceDashboardEvent(createDashboardState(snapshot), slow);
    assert.equal(slowState.synchronization.reason, "server_slow_client");
    assert.equal(slowState.synchronization.details.droppedEvents, 9);
});

test("stale and replayed stream signals are duplicates, not fresh recovery causes", async () => {
    const snapshot = await loadSnapshot();
    const [event] = await loadEvents();
    const accepted = reduceDashboardEvent(createDashboardState(snapshot), event);
    const replayedOverflow = streamSignal(event, 101, "stream.overflow", {
        oldest_available_sequence: 140,
    });
    const staleSlowClient = streamSignal(event, 99, "stream.slow_client", {
        dropped_events: 2,
    });

    const afterReplay = reduceDashboardEvent(accepted, replayedOverflow);
    const afterStale = reduceDashboardEvent(afterReplay, staleSlowClient);

    assert.equal(afterStale.synchronization.status, "live");
    assert.equal(afterStale.synchronization.reason, null);
    assert.equal(afterStale.synchronization.duplicates, 2);
    assert.equal(afterStale.lastSequence, 101);
    assert.equal(afterStale.snapshot, accepted.snapshot);
});

test("malformed typed events become resnapshot state rather than partial updates", async () => {
    const snapshot = await loadSnapshot();
    const [fixture] = await loadEvents();
    const malformed = clone(fixture);
    delete malformed.payload.request.content;

    const state = reduceDashboardEvent(createDashboardState(snapshot), malformed);
    assert.equal(state.synchronization.status, "resnapshot_required");
    assert.equal(state.synchronization.reason, "invalid_event");
    assert.equal(state.lastEventId, "100");
});

test("aggregate budget rejects oversized object-form events before cloning", async () => {
    const snapshot = await loadSnapshot();
    const [fixture] = await loadEvents();
    const oversized = clone(fixture);
    oversized.unknown_debug_tree = Array.from({ length: 8 }, () => "x".repeat(600000));

    const initial = createDashboardState(snapshot);
    const state = reduceDashboardEvent(initial, oversized);
    assert.equal(state.synchronization.reason, "invalid_event");
    assert.match(state.synchronization.details.message, /aggregate bytes/);
    assert.equal(state.snapshot, initial.snapshot);
    assert.equal(state.lastEventId, "100");
});

test("timeline history is independently bounded", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();
    let state = createDashboardState(snapshot, { timelineLimit: 3 });
    state = reduceDashboardEvent(state, events[0]);
    state = reduceDashboardEvent(state, events[1]);

    assert.equal(state.snapshot.timeline.length, 3);
    assert.equal(state.snapshot.timeline.at(-1).id, "timeline-102");
    assert.equal(state.snapshot.timeline[0].id, "timeline-98");
});

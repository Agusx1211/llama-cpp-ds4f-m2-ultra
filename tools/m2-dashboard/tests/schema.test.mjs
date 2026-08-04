import assert from "node:assert/strict";
import test from "node:test";

import {
    MAX_SCHEMA_ARRAY_LENGTH,
    MAX_SCHEMA_DEPTH,
    MAX_SCHEMA_NODES,
    MAX_SCHEMA_STRING_LENGTH,
    MAX_SCHEMA_TOTAL_BYTES,
    SCHEMA_VERSION,
    SchemaError,
    parseEvent,
    parseSnapshot,
    validateEvent,
    validateSnapshot,
} from "../lib/schema.mjs";
import { clone, loadEvents, loadSnapshot } from "./helpers.mjs";

test("deterministic snapshot and event fixtures satisfy schema version 2", async () => {
    const snapshot = await loadSnapshot();
    const events = await loadEvents();

    assert.equal(validateSnapshot(snapshot), snapshot);
    for (const event of events) {
        assert.equal(validateEvent(event), event);
    }
    assert.equal(snapshot.schema_version, SCHEMA_VERSION);
    assert.equal(events[0].sequence, snapshot.sequence + 1);
});

test("parsed fixture values are cloned and deeply immutable", async () => {
    const source = await loadSnapshot();
    const parsed = parseSnapshot(source);

    source.server.health = "unavailable";
    assert.equal(parsed.server.health, "degraded");
    assert.ok(Object.isFrozen(parsed));
    assert.ok(Object.isFrozen(parsed.server));
    assert.throws(() => {
        parsed.server.health = "healthy";
    }, TypeError);

    const [eventSource] = await loadEvents();
    const event = parseEvent(eventSource);
    assert.ok(Object.isFrozen(event.payload.request.content));
});

test("snapshot validation rejects schema drift and incomplete lane sets", async () => {
    const snapshot = await loadSnapshot();
    const badVersion = clone(snapshot);
    badVersion.schema_version = 3;
    assert.throws(() => validateSnapshot(badVersion), (error) => {
        assert.ok(error instanceof SchemaError);
        assert.match(error.message, /schema_version/);
        return true;
    });

    const missingLane = clone(snapshot);
    missingLane.lanes.pop();
    assert.throws(() => validateSnapshot(missingLane), /missing fast/);

    const duplicateLane = clone(snapshot);
    duplicateLane.lanes[2].id = "normal";
    assert.throws(() => validateSnapshot(duplicateLane), /duplicate lane normal/);

    const missingAvailability = clone(snapshot);
    delete missingAvailability.availability.dspark;
    assert.throws(() => validateSnapshot(missingAvailability), /availability.dspark must be boolean/);

    const malformedRegistry = clone(snapshot);
    malformedRegistry.registry.claimed_permits.pop();
    assert.throws(() => validateSnapshot(malformedRegistry), /one count per lane/);
});

test("fast-refill schema rejects contradictory disabled, quota, and expiry states", async () => {
    const snapshot = await loadSnapshot();

    const disabledWithLimits = clone(snapshot);
    disabledWithLimits.fast_refill.configuration.enabled = false;
    assert.throws(() => validateSnapshot(disabledWithLimits), /disabled state must use zero limits/);

    const exhaustedButOpen = clone(snapshot);
    exhaustedButOpen.fast_refill.refill.fast_members_used = 4;
    exhaustedButOpen.fast_refill.refill.fast_members_remaining = 0;
    assert.throws(() => validateSnapshot(exhaustedButOpen), /window_open requires a live bounded fast epoch/);

    const openAtExpiry = clone(snapshot);
    openAtExpiry.fast_refill.refill.remaining_ms = 0;
    assert.throws(() => validateSnapshot(openAtExpiry), /window_open requires a live bounded fast epoch/);

    const deadlineFreeOpen = clone(snapshot);
    deadlineFreeOpen.fast_refill.refill.deadline_at = null;
    deadlineFreeOpen.fast_refill.refill.remaining_ms = 0;
    assert.throws(() => validateSnapshot(deadlineFreeOpen), /without a deadline must be unexpired, closed, and ineligible/);
});

test("event validation rejects mismatched SSE IDs and malformed typed payloads", async () => {
    const snapshot = await loadSnapshot();
    const [fixture] = await loadEvents();
    const badId = clone(fixture);
    badId.id = "999";
    assert.throws(() => validateEvent(badId), /decimal event.sequence/);

    const badPayload = clone(fixture);
    delete badPayload.payload.request.kv;
    assert.throws(() => validateEvent(badPayload), /payload.request.kv/);

    const badRequestIdentity = clone(fixture);
    badRequestIdentity.request_id = "req-envelope-mismatch";
    assert.throws(() => validateEvent(badRequestIdentity), /request_id must equal.*request.id/);

    const badLaneIdentity = clone(fixture);
    badLaneIdentity.lane = "normal";
    assert.throws(() => validateEvent(badLaneIdentity), /event.lane must equal.*request.lane/);

    const incompleteLanes = clone(fixture);
    incompleteLanes.payload.lanes = snapshot.lanes.slice(0, 2);
    assert.throws(() => validateEvent(incompleteLanes), /payload.lanes is missing fast/);
});

test("event validation ties analogous envelope identities to typed payloads", async () => {
    const snapshot = await loadSnapshot();
    const [requestEvent, timelineEvent] = await loadEvents();

    const badTimeline = clone(timelineEvent);
    badTimeline.payload.item.request_id = "req-other";
    assert.throws(() => validateEvent(badTimeline), /request_id must equal.*item.request_id/);

    const badRemoval = clone(requestEvent);
    badRemoval.type = "request.remove";
    badRemoval.payload = { request_id: "req-other" };
    assert.throws(() => validateEvent(badRemoval), /request_id must equal.*payload.request_id/);

    const badLane = clone(requestEvent);
    badLane.type = "lane.replace";
    badLane.payload = { lane: clone(snapshot.lanes[0]) };
    assert.throws(() => validateEvent(badLane), /event.lane must equal.*payload.lane.id/);
});

test("schema validation bounds arrays and strings before cloning", async () => {
    const snapshot = await loadSnapshot();
    const tooManyTimelineItems = clone(snapshot);
    tooManyTimelineItems.timeline = Array.from(
        { length: MAX_SCHEMA_ARRAY_LENGTH + 1 },
        (_, index) => ({
            id: `bounded-${index}`,
            at: snapshot.generated_at,
            type: "fixture",
            label: "bounded",
            request_id: null,
            lane: null,
        }),
    );
    assert.throws(() => validateSnapshot(tooManyTimelineItems), /at most 4096 items/);

    const [event] = await loadEvents();
    const oversizedString = clone(event);
    oversizedString.payload.request.content.prompt = "x".repeat(MAX_SCHEMA_STRING_LENGTH + 1);
    assert.throws(() => validateEvent(oversizedString), /at most 1048576 characters/);
});

test("pre-clone schema budgets bound aggregate bytes, nodes, and depth", async () => {
    const snapshot = await loadSnapshot();
    const aggregateSnapshot = clone(snapshot);
    aggregateSnapshot.timeline = Array.from({ length: 8 }, (_, index) => ({
        id: `aggregate-${index}`,
        at: snapshot.generated_at,
        type: "fixture",
        label: "x".repeat(Math.floor(MAX_SCHEMA_TOTAL_BYTES / 8)),
        request_id: null,
        lane: null,
    }));
    assert.throws(() => parseSnapshot(aggregateSnapshot), /aggregate bytes/);

    const [fixture] = await loadEvents();

    const aggregate = clone(fixture);
    aggregate.payload.request.scheduler_reasons = Array.from(
        { length: 8 },
        () => "x".repeat(Math.floor(MAX_SCHEMA_TOTAL_BYTES / 8)),
    );
    assert.throws(() => parseEvent(aggregate), /aggregate bytes/);

    const tooManyNodes = clone(fixture);
    tooManyNodes.unexpected = Array.from({ length: MAX_SCHEMA_NODES }, () => null);
    assert.throws(() => parseEvent(tooManyNodes), /JSON nodes/);

    const tooDeep = clone(fixture);
    let cursor = tooDeep;
    for (let depth = 0; depth <= MAX_SCHEMA_DEPTH; depth += 1) {
        cursor.unexpected = {};
        cursor = cursor.unexpected;
    }
    assert.throws(() => parseEvent(tooDeep), /JSON depth/);
});

test("unknown snapshot and event fields are rejected instead of retained", async () => {
    const snapshot = await loadSnapshot();
    const unknownSnapshot = clone(snapshot);
    unknownSnapshot.server.debug_dump = "must not be retained";
    assert.throws(() => parseSnapshot(unknownSnapshot), /server.debug_dump is not allowed/);

    const [fixture] = await loadEvents();
    const unknownEvent = clone(fixture);
    unknownEvent.payload.request.content.internal = "must not be retained";
    assert.throws(() => parseEvent(unknownEvent), /content.internal is not allowed/);
});

test("snapshot validation enforces lane aggregates and cache byte accounting", async () => {
    const snapshot = await loadSnapshot();
    for (const lane of snapshot.lanes) {
        const requests = snapshot.requests.filter((request) => request.lane === lane.id);
        const queued = requests.filter((request) => request.state === "queued").length;
        const active = requests.filter((request) => ![
            "queued",
            "complete",
            "cancelled",
            "failed",
        ].includes(request.state)).length;
        assert.equal(lane.queued, queued);
        assert.equal(lane.active, active);
    }
    for (const object of snapshot.cache.objects) {
        assert.equal(object.logical_bytes, object.unique_bytes + object.shared_bytes);
    }

    const badAggregate = clone(snapshot);
    badAggregate.lanes[0].queued += 1;
    assert.throws(() => validateSnapshot(badAggregate), /queued must equal request aggregate/);

    const badCacheBytes = clone(snapshot);
    badCacheBytes.cache.objects[0].shared_bytes += 1;
    assert.throws(() => validateSnapshot(badCacheBytes), /logical_bytes must equal/);
});

test("malicious-looking prompt strings remain schema-valid data", async () => {
    const snapshot = await loadSnapshot();
    assert.match(snapshot.requests[0].content.prompt, /<script>/);
    assert.doesNotThrow(() => validateSnapshot(snapshot));
});

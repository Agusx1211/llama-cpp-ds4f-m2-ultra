import assert from "node:assert/strict";
import test from "node:test";

import {
    SCHEMA_VERSION,
    SchemaError,
    parseEvent,
    parseSnapshot,
    validateEvent,
    validateSnapshot,
} from "../lib/schema.mjs";
import { clone, loadEvents, loadSnapshot } from "./helpers.mjs";

test("deterministic snapshot and event fixtures satisfy schema version 1", async () => {
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
    badVersion.schema_version = 2;
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
});

test("event validation rejects mismatched SSE IDs and malformed typed payloads", async () => {
    const [fixture] = await loadEvents();
    const badId = clone(fixture);
    badId.id = "999";
    assert.throws(() => validateEvent(badId), /decimal event.sequence/);

    const badPayload = clone(fixture);
    delete badPayload.payload.request.kv;
    assert.throws(() => validateEvent(badPayload), /payload.request.kv/);
});

test("malicious-looking prompt strings remain schema-valid data", async () => {
    const snapshot = await loadSnapshot();
    assert.match(snapshot.requests[0].content.prompt, /<script>/);
    assert.doesNotThrow(() => validateSnapshot(snapshot));
});

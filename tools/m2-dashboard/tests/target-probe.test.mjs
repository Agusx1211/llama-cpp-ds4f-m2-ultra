import assert from "node:assert/strict";
import test from "node:test";

import {
    refillStateLabels,
    verifyMonotonicEvents,
    verifyOptionalRefill,
    verifyRedaction,
} from "../target/live-probe.mjs";

function refill(overrides = {}) {
    return {
        configuration: { enabled: true, max_members: 4, window_ms: 30000 },
        cohort: { active: true, selection_open: false, dominant_lane: "fast", limit: 2 },
        refill: {
            fast_members_used: 1,
            fast_members_remaining: 3,
            deadline_at: "monotonic:1000000us",
            remaining_ms: 900,
            deadline_expired: false,
            window_open: true,
            one_member_eligible_now: true,
        },
        ...overrides,
    };
}

test("target probe accepts a fully redacted snapshot and rejects retained content", () => {
    const snapshot = {
        availability: { content: false },
        requests: [{
            id: "request-1",
            content: { prompt: "", output: "", retained: false },
        }],
    };
    assert.doesNotThrow(() => verifyRedaction(snapshot, ["credential-value"]));
    assert.throws(
        () => verifyRedaction({
            ...snapshot,
            requests: [{
                id: "request-1",
                content: { prompt: "secret prompt", output: "", retained: true },
            }],
        }, ["credential-value"]),
        /exposed retained prompt or output content/,
    );
    assert.throws(
        () => verifyRedaction({ ...snapshot, reflected: "credential-value" }, ["credential-value"]),
        /reflected a credential value/,
    );
});

test("target probe rejects non-contiguous SSE sequences", () => {
    assert.equal(verifyMonotonicEvents([{ sequence: 11 }, { sequence: 12 }], 10), 12);
    assert.throws(
        () => verifyMonotonicEvents([{ sequence: 11 }, { sequence: 13 }], 10),
        /expected 12, received 13/,
    );
});

test("target probe derives authoritative refill state labels", () => {
    assert.deepEqual(
        refillStateLabels(refill()),
        ["window_open", "one_member_eligible"],
    );
    const widthBlocked = refill();
    widthBlocked.refill.one_member_eligible_now = false;
    assert.deepEqual(refillStateLabels(widthBlocked), ["window_open", "full_width"]);
    assert.deepEqual(
        refillStateLabels(refill({
            configuration: { enabled: false, max_members: 0, window_ms: 0 },
            cohort: { active: false, selection_open: false, dominant_lane: null, limit: 0 },
            refill: {
                fast_members_used: 0,
                fast_members_remaining: 0,
                deadline_at: null,
                remaining_ms: 0,
                deadline_expired: false,
                window_open: false,
                one_member_eligible_now: false,
            },
        })),
        ["disabled", "inactive", "quota_exhausted"],
    );
});

test("optional refill checks require shared-parser snapshot and event objects", () => {
    assert.deepEqual(
        verifyOptionalRefill({}, [], { required: false }),
        {
            required: false,
            available: false,
            snapshot_exposed: false,
            event_objects: 0,
            observed_states: [],
        },
    );
    assert.throws(
        () => verifyOptionalRefill({ fast_refill: refill() }, [], {
            required: true,
            expected: ["window_open"],
        }),
        /snapshot\/event objects were not observed/,
    );
    const result = verifyOptionalRefill(
        { fast_refill: refill() },
        [{ payload: { fast_refill: refill() } }],
        { required: true, expected: ["window_open", "one_member_eligible"] },
    );
    assert.equal(result.available, true);
    assert.deepEqual(result.observed_states, ["window_open", "one_member_eligible"]);
    assert.throws(
        () => verifyOptionalRefill(
            { fast_refill: refill() },
            [{ payload: { fast_refill: refill() } }],
            { required: true, expected: ["deadline_expired"] },
        ),
        /required refill states were not observed/,
    );
});

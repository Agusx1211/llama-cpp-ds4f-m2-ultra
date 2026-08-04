import assert from "node:assert/strict";
import test from "node:test";

import { cancelConfirmation, confirmAndCancel } from "../lib/admin.mjs";

test("live cancel requires an explicit request-specific confirmation", async () => {
    const calls = [];
    const result = await confirmAndCancel({
        requestId: "7:19",
        confirmImpl(message) {
            calls.push(["confirm", message]);
            return true;
        },
        async cancelRequest(requestId) {
            calls.push(["cancel", requestId]);
            return { status: "accepted" };
        },
    });

    assert.equal(result.confirmed, true);
    assert.equal(result.response.status, "accepted");
    assert.deepEqual(calls, [
        ["confirm", "Cancel live request 7:19? This stops generation and cannot be undone."],
        ["cancel", "7:19"],
    ]);
});

test("declined live cancel sends no mutation", async () => {
    let cancelCalls = 0;
    const result = await confirmAndCancel({
        requestId: "7:19",
        confirmImpl: () => false,
        cancelRequest: async () => {
            cancelCalls += 1;
        },
    });
    assert.deepEqual(result, { confirmed: false, response: null });
    assert.equal(cancelCalls, 0);
});

test("confirmation rejects non-canonical handles before prompting", () => {
    assert.throws(() => cancelConfirmation("7"), /canonical id:epoch/);
    assert.throws(() => cancelConfirmation("07:19"), /canonical id:epoch/);
});

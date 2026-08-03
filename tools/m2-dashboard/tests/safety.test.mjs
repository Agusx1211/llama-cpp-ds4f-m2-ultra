import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

import { ControlIntentBuffer, createControlIntent } from "../lib/controls.mjs";
import { contentForDisplay, setSafeText } from "../lib/safe-dom.mjs";
import { loadSnapshot } from "./helpers.mjs";

test("content is redacted unless reveal is explicitly allowed and selected", () => {
    const malicious = "<script>alert('x')</script>";
    assert.equal(contentForDisplay(malicious), "[redacted]");
    assert.equal(contentForDisplay(malicious, { allowReveal: true }), "[redacted]");
    assert.equal(contentForDisplay(malicious, { revealed: true }), "[redacted]");
    assert.equal(contentForDisplay(malicious, { allowReveal: true, revealed: true }), malicious);
});

test("browser-safe helper writes malicious HTML as text and never requests HTML parsing", () => {
    const writes = [];
    const target = {
        set textContent(value) {
            writes.push(["textContent", value]);
        },
        set innerHTML(_value) {
            throw new Error("innerHTML must not be used");
        },
    };
    const malicious = "<img src=x onerror=globalThis.pwned=true><script>pwn()</script>";
    setSafeText(target, malicious);
    assert.deepEqual(writes, [["textContent", malicious]]);
});

test("fixture script content remains inert data through redaction and text rendering", async () => {
    const snapshot = await loadSnapshot();
    const prompt = snapshot.requests[0].content.prompt;
    assert.match(prompt, /<script>/);
    assert.equal(contentForDisplay(prompt), "[redacted]");

    let rendered;
    setSafeText({ set textContent(value) { rendered = value; } }, prompt);
    assert.equal(rendered, prompt);
});

test("dashboard sources avoid HTML injection and browser persistence primitives", async () => {
    const sourceRoot = new URL("../", import.meta.url);
    const libraryRoot = new URL("../lib/", import.meta.url);
    const libraryFiles = (await readdir(libraryRoot)).filter((name) => name.endsWith(".mjs"));
    const sources = await Promise.all([
        new URL("dashboard.js", sourceRoot),
        ...libraryFiles.map((name) => new URL(name, libraryRoot)),
    ].map((url) => readFile(url, "utf8")));
    const combined = sources.join("\n");
    for (const forbidden of [
        ".innerHTML",
        "insertAdjacentHTML",
        "document.write",
        "localStorage",
        "sessionStorage",
    ]) {
        assert.equal(combined.includes(forbidden), false, `forbidden source primitive: ${forbidden}`);
    }

    const networkMethods = [...combined.matchAll(/method:\s*"([A-Z]+)"/g)].map((match) => match[1]);
    assert.ok(networkMethods.length > 0, "expected explicit read-only network methods");
    assert.ok(networkMethods.every((method) => method === "GET"), `state-changing network method found: ${networkMethods}`);
});

test("control intents are immutable, bounded, and explicitly transport-free", () => {
    const buffer = new ControlIntentBuffer(2);
    const first = createControlIntent("request.pause", "req-1", {}, "2026-08-03T22:00:00Z");
    const sourceParameters = { lane: "fast", policy: { reasons: ["latency"] } };
    const second = createControlIntent("request.reprioritize", "req-2", sourceParameters, "2026-08-03T22:00:01Z");
    const third = createControlIntent("cache.pin", "prefix-3", {}, "2026-08-03T22:00:02Z");
    sourceParameters.policy.reasons[0] = "mutated";
    buffer.add(first);
    buffer.add(second);
    const snapshot = buffer.add(third);

    assert.equal(first.transport, "local-preview-only");
    assert.ok(Object.isFrozen(first));
    assert.ok(Object.isFrozen(second.parameters));
    assert.ok(Object.isFrozen(second.parameters.policy));
    assert.ok(Object.isFrozen(second.parameters.policy.reasons));
    assert.equal(second.parameters.policy.reasons[0], "latency");
    assert.deepEqual(snapshot.map((intent) => intent.targetId), ["req-2", "prefix-3"]);
    assert.throws(() => {
        second.parameters.policy.reasons.push("unsafe");
    }, TypeError);
    assert.throws(() => createControlIntent("request.delete", "req-1"), /unsupported/);
});

test("control buffer clones externally supplied local intents before retaining them", () => {
    const buffer = new ControlIntentBuffer(1);
    const external = {
        action: "request.pause",
        targetId: "req-1",
        parameters: { nested: { value: 1 } },
        createdAt: "2026-08-03T22:00:00Z",
        transport: "local-preview-only",
    };
    buffer.add(external);
    external.parameters.nested.value = 2;

    const [stored] = buffer.snapshot();
    assert.equal(stored.parameters.nested.value, 1);
    assert.ok(Object.isFrozen(stored.parameters.nested));
});

test("control parameters reject cyclic, deep, wide, and aggregate-oversized graphs", () => {
    const cyclic = {};
    cyclic.self = cyclic;
    assert.throws(
        () => createControlIntent("request.pause", "req-1", cyclic),
        /must not contain cycles/,
    );

    const deep = {};
    let cursor = deep;
    for (let depth = 0; depth < 14; depth += 1) {
        cursor.next = {};
        cursor = cursor.next;
    }
    assert.throws(
        () => createControlIntent("request.pause", "req-1", deep),
        /JSON depth/,
    );

    assert.throws(
        () => createControlIntent("request.pause", "req-1", {
            values: Array.from({ length: 256 }, () => 0),
        }),
        /JSON nodes/,
    );
    assert.throws(
        () => createControlIntent("request.pause", "req-1", {
            value: "x".repeat(64 * 1024),
        }),
        /aggregate bytes/,
    );

    const specialKey = createControlIntent(
        "request.pause",
        "req-1",
        JSON.parse('{"__proto__":{"polluted":true}}'),
    );
    assert.equal(Object.getPrototypeOf(specialKey.parameters), Object.prototype);
    assert.equal(Object.hasOwn(specialKey.parameters, "__proto__"), true);
    assert.equal(Object.prototype.polluted, undefined);
});

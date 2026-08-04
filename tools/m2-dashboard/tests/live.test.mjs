import assert from "node:assert/strict";
import test from "node:test";

import {
    createLiveDashboardTransport,
    CONTROL_PATH,
    CSRF_HEADER,
    DETAIL_PATH,
    EVENTS_PATH,
    normalizeLoopbackBaseUrl,
    OPERATOR_TOKEN_HEADER,
    SNAPSHOT_PATH,
} from "../lib/live.mjs";
import { loadSnapshot } from "./helpers.mjs";

const apiKey = "dashboard-api-key";
const operatorToken = "0123456789abcdef0123456789abcdef";

test("live transport accepts explicit loopback bases and rejects secret-bearing or remote URLs", () => {
    assert.equal(normalizeLoopbackBaseUrl("http://127.0.0.1:18130").origin, "http://127.0.0.1:18130");
    assert.equal(normalizeLoopbackBaseUrl("https://localhost:8443/api/").pathname, "/api");
    assert.throws(() => normalizeLoopbackBaseUrl("https://example.com"), /loopback/);
    assert.throws(() => normalizeLoopbackBaseUrl("http://user:secret@127.0.0.1:8080"), /credentials/);
    assert.throws(() => normalizeLoopbackBaseUrl("http://127.0.0.1:8080?token=secret"), /query/);
    assert.throws(() => normalizeLoopbackBaseUrl("file:///tmp/socket"), /HTTP or HTTPS/);
});

test("live snapshot sends both credentials in headers with no ambient browser credentials", async () => {
    const calls = [];
    const fetchImpl = async (url, options) => {
        calls.push({ url, options });
        return new Response(JSON.stringify({ schema_version: 1 }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
        });
    };
    const transport = createLiveDashboardTransport({
        baseUrl: "http://127.0.0.1:18130/prefix",
        apiKey,
        operatorToken,
        fetchImpl,
    });

    await transport.getSnapshot();
    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, `http://127.0.0.1:18130/prefix${SNAPSHOT_PATH}`);
    assert.equal(calls[0].options.method, "GET");
    assert.equal(calls[0].options.credentials, "omit");
    assert.equal(calls[0].options.redirect, "error");
    assert.equal(calls[0].options.referrerPolicy, "no-referrer");
    assert.equal(calls[0].options.headers.get("Authorization"), `Bearer ${apiKey}`);
    assert.equal(calls[0].options.headers.get(OPERATOR_TOKEN_HEADER), operatorToken);
    assert.equal(calls[0].options.headers.has("X-Llama-Trusted-Lane"), false);
    assert.equal(calls[0].options.headers.has("X-Llama-Benchmark-Tag"), false);
    assert.equal(calls[0].url.includes(apiKey), false);
    assert.equal(calls[0].url.includes(operatorToken), false);

    transport.clear();
    await assert.rejects(() => transport.getSnapshot(), /credentials were cleared/);
});

test("live snapshot rejects a response above its transport byte budget", async () => {
    const transport = createLiveDashboardTransport({
        baseUrl: "http://127.0.0.1:18130",
        apiKey,
        operatorToken,
        fetchImpl: async () => new Response("{}", {
            status: 200,
            headers: { "Content-Length": String(4 * 1024 * 1024 + 1) },
        }),
    });

    await assert.rejects(() => transport.getSnapshot(), /exceeds 4194304 bytes/);
    transport.clear();
});

test("live detail and cancel use bounded JSON POSTs with CSRF proof and exact identity", async () => {
    const snapshot = structuredClone(await loadSnapshot());
    const request = structuredClone(snapshot.requests[0]);
    request.id = "7:19";
    request.content = { prompt: "", output: "", retained: false };
    const calls = [];
    const fetchImpl = async (url, options) => {
        calls.push({ url, options });
        const payload = url.endsWith(DETAIL_PATH)
            ? {
                schema_version: 2,
                request,
                registry: {
                    revision: 3,
                    cancel_requested: false,
                    timeout_expired: false,
                    binding_count: 0,
                    bindings: [],
                },
                content_reveal: false,
            }
            : { schema_version: 2, request_id: request.id, action: "cancel", status: "accepted" };
        return new Response(JSON.stringify(payload), {
            status: url.endsWith(CONTROL_PATH) ? 202 : 200,
            headers: { "Content-Type": "application/json" },
        });
    };
    const transport = createLiveDashboardTransport({
        baseUrl: "http://127.0.0.1:18130",
        apiKey,
        operatorToken,
        fetchImpl,
    });

    const detail = await transport.getRequestDetail(request.id);
    const control = await transport.cancelRequest(request.id);
    assert.equal(detail.request.id, request.id);
    assert.equal(detail.content_reveal, false);
    assert.equal(control.status, "accepted");
    assert.deepEqual(calls.map((call) => new URL(call.url).pathname), [DETAIL_PATH, CONTROL_PATH]);
    for (const call of calls) {
        assert.equal(call.options.method, "POST");
        assert.equal(call.options.credentials, "omit");
        assert.equal(call.options.headers.get("Content-Type"), "application/json");
        assert.equal(call.options.headers.get(CSRF_HEADER), "1");
        assert.equal(call.options.headers.get(OPERATOR_TOKEN_HEADER), operatorToken);
        assert.equal(call.options.headers.has("X-Llama-Trusted-Lane"), false);
    }
    assert.deepEqual(JSON.parse(calls[0].options.body), { request_id: request.id });
    assert.deepEqual(JSON.parse(calls[1].options.body), { action: "cancel", request_id: request.id });
    transport.clear();
});

test("live control rejects non-canonical identity before fetch", async () => {
    let fetches = 0;
    const transport = createLiveDashboardTransport({
        baseUrl: "http://127.0.0.1:18130",
        apiKey,
        operatorToken,
        fetchImpl: async () => {
            fetches += 1;
        },
    });
    await assert.rejects(() => transport.cancelRequest("9"), /canonical id:epoch/);
    await assert.rejects(() => transport.getRequestDetail("09:2"), /canonical id:epoch/);
    assert.equal(fetches, 0);
    transport.clear();
});

test("live SSE sends Last-Event-ID beside credentials and reports a server cursor rejection", async () => {
    const calls = [];
    const fetchImpl = async (url, options) => {
        calls.push({ url, options });
        return {
            ok: false,
            status: 409,
            statusText: "Conflict",
            body: null,
        };
    };
    const transport = createLiveDashboardTransport({
        baseUrl: "http://localhost:18130",
        apiKey,
        operatorToken,
        fetchImpl,
    });
    const disconnected = new Promise((resolve) => {
        transport.openEvents({
            lastEventId: "77",
            onEvent: () => assert.fail("409 stream cannot emit an event"),
            onDisconnect: resolve,
        });
    });
    const error = await disconnected;

    assert.equal(calls[0].url, `http://localhost:18130${EVENTS_PATH}`);
    assert.equal(calls[0].options.headers.get("Last-Event-ID"), "77");
    assert.equal(calls[0].options.headers.get(OPERATOR_TOKEN_HEADER), operatorToken);
    assert.equal(error.resnapshotRequired, true);
    transport.clear();
});

test("live credential validation bounds and rejects header injection", () => {
    const base = {
        baseUrl: "http://127.0.0.1:8080",
        apiKey,
        operatorToken,
        fetchImpl: async () => {},
    };
    assert.throws(() => createLiveDashboardTransport({ ...base, apiKey: "bad\nkey" }), /forbidden/);
    assert.throws(() => createLiveDashboardTransport({ ...base, operatorToken: "short" }), /32 to 256/);
    assert.throws(() => createLiveDashboardTransport({ ...base, operatorToken: "x".repeat(257) }), /32 to 256/);
});

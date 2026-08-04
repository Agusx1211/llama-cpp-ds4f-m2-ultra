import { createFetchSseTransport } from "./sse.mjs";
import { parseControlResponse, parseRequestDetail } from "./schema.mjs";

export const OPERATOR_TOKEN_HEADER = "X-Llama-Trusted-Scheduling-Token";
export const SNAPSHOT_PATH = "/internal/admin/dashboard/snapshot";
export const EVENTS_PATH = "/internal/admin/dashboard/events";
export const DETAIL_PATH = "/internal/admin/dashboard/request-detail";
export const CONTROL_PATH = "/internal/admin/dashboard/request-control";
export const CSRF_HEADER = "X-Llama-Dashboard-CSRF";

const MAX_API_KEY_BYTES = 1024;
const MIN_OPERATOR_TOKEN_BYTES = 32;
const MAX_OPERATOR_TOKEN_BYTES = 256;
const MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024;
const MAX_DETAIL_BYTES = 64 * 1024;
const MAX_SSE_EVENT_BYTES = 64 * 1024;
const textEncoder = new TextEncoder();

function credential(value, name, minimum, maximum) {
    const bytes = typeof value === "string" ? textEncoder.encode(value).byteLength : 0;
    if (typeof value !== "string" || bytes < minimum || bytes > maximum) {
        throw new RangeError(`${name} must contain ${minimum} to ${maximum} bytes`);
    }
    if ([...value].some((character) => [0, 10, 13].includes(character.charCodeAt(0)))) {
        throw new RangeError(`${name} contains a forbidden header character`);
    }
    return value;
}

function loopbackHostname(hostname) {
    const lower = hostname.toLowerCase();
    if (lower === "localhost" || lower === "::1") {
        return true;
    }
    const parts = lower.split(".");
    return parts.length === 4
        && parts[0] === "127"
        && parts.every((part) => /^\d{1,3}$/u.test(part) && Number(part) <= 255);
}

export function normalizeLoopbackBaseUrl(value) {
    let url;
    try {
        url = new URL(value);
    } catch {
        throw new TypeError("server URL must be an absolute URL");
    }
    if (!["http:", "https:"].includes(url.protocol)) {
        throw new RangeError("server URL must use HTTP or HTTPS");
    }
    if (!loopbackHostname(url.hostname)) {
        throw new RangeError("server URL must resolve explicitly to loopback");
    }
    if (url.username || url.password || url.search || url.hash) {
        throw new RangeError("server URL cannot contain credentials, query parameters, or a fragment");
    }
    url.pathname = url.pathname.replace(/\/+$/u, "");
    return url;
}

export function normalizeGatewayOrigin(value) {
    let url;
    try {
        url = new URL(value);
    } catch {
        throw new TypeError("gateway origin must be an absolute URL");
    }
    if (url.protocol !== "https:") {
        throw new RangeError("gateway origin must use HTTPS");
    }
    if (url.username || url.password || url.search || url.hash || url.pathname !== "/") {
        throw new RangeError("gateway origin cannot contain credentials, a path, query parameters, or a fragment");
    }
    return url;
}

function endpointUrl(base, path) {
    const result = new URL(base.href);
    const prefix = base.pathname === "/" ? "" : base.pathname;
    result.pathname = `${prefix}${path}`;
    return result.href;
}

async function boundedJson(response, maximumBytes = MAX_SNAPSHOT_BYTES, label = "snapshot") {
    const contentLength = Number(response.headers.get("Content-Length"));
    if (Number.isFinite(contentLength) && contentLength > maximumBytes) {
        throw new RangeError(`${label} response exceeds ${maximumBytes} bytes`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error(`${label} response body is not a readable stream`);
    }
    const decoder = new TextDecoder();
    let total = 0;
    let text = "";
    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) {
                break;
            }
            total += value.byteLength;
            if (total > maximumBytes) {
                await reader.cancel();
                throw new RangeError(`${label} response exceeds ${maximumBytes} bytes`);
            }
            text += decoder.decode(value, { stream: true });
        }
        text += decoder.decode();
    } finally {
        reader.releaseLock();
    }
    return JSON.parse(text);
}

function canonicalRequestId(value) {
    if (typeof value !== "string" || value.length > 41 || !/^[1-9]\d*:[1-9]\d*$/u.test(value)) {
        throw new TypeError("request ID must be canonical id:epoch");
    }
    return value;
}

function createDashboardTransport({ base, apiKey, operatorToken, fetchImpl, parserOptions }) {
    if (typeof fetchImpl !== "function") {
        throw new Error("fetch is unavailable");
    }
    const secrets = {
        apiKey: credential(apiKey, "API key", 1, MAX_API_KEY_BYTES),
        operatorToken,
    };

    const credentialedFetch = (url, options = {}) => {
        if (secrets.apiKey === "") {
            throw new Error("dashboard credentials were cleared");
        }
        const headers = new Headers(options.headers ?? {});
        headers.set("Authorization", `Bearer ${secrets.apiKey}`);
        if (secrets.operatorToken === null) {
            headers.delete(OPERATOR_TOKEN_HEADER);
        } else {
            headers.set(OPERATOR_TOKEN_HEADER, secrets.operatorToken);
        }
        headers.delete("X-Llama-Trusted-Lane");
        headers.delete("X-Llama-Benchmark-Tag");
        return fetchImpl(url, {
            ...options,
            method: options.method ?? "GET",
            headers,
            credentials: "omit",
            cache: "no-store",
            redirect: "error",
            referrerPolicy: "no-referrer",
        });
    };

    const snapshotUrl = endpointUrl(base, SNAPSHOT_PATH);
    const eventsUrl = endpointUrl(base, EVENTS_PATH);
    const detailUrl = endpointUrl(base, DETAIL_PATH);
    const controlUrl = endpointUrl(base, CONTROL_PATH);
    const boundedParserOptions = {
        ...parserOptions,
        maxBufferBytes: Math.min(parserOptions.maxBufferBytes ?? MAX_SSE_EVENT_BYTES, MAX_SSE_EVENT_BYTES),
        maxEventBytes: Math.min(parserOptions.maxEventBytes ?? MAX_SSE_EVENT_BYTES, MAX_SSE_EVENT_BYTES),
    };
    return {
        async getSnapshot() {
            const response = await credentialedFetch(snapshotUrl, {
                headers: { Accept: "application/json" },
            });
            if (!response.ok) {
                throw new Error(`snapshot request failed: ${response.status} ${response.statusText}`);
            }
            return boundedJson(response);
        },
        async getRequestDetail(requestId) {
            const expected = canonicalRequestId(requestId);
            const response = await credentialedFetch(detailUrl, {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                    [CSRF_HEADER]: "1",
                },
                body: JSON.stringify({ request_id: expected }),
            });
            if (!response.ok) {
                throw new Error(`request detail failed: ${response.status} ${response.statusText}`);
            }
            const detail = parseRequestDetail(await boundedJson(response, MAX_DETAIL_BYTES, "request detail"));
            if (detail.request.id !== expected) {
                throw new Error("request detail identity mismatch");
            }
            return detail;
        },
        async cancelRequest(requestId) {
            const expected = canonicalRequestId(requestId);
            const response = await credentialedFetch(controlUrl, {
                method: "POST",
                headers: {
                    Accept: "application/json",
                    "Content-Type": "application/json",
                    [CSRF_HEADER]: "1",
                },
                body: JSON.stringify({ action: "cancel", request_id: expected }),
            });
            if (!response.ok) {
                throw new Error(`cancel request failed: ${response.status} ${response.statusText}`);
            }
            const result = parseControlResponse(await boundedJson(response, MAX_DETAIL_BYTES, "control"));
            if (result.request_id !== expected || result.action !== "cancel") {
                throw new Error("cancel response identity mismatch");
            }
            return result;
        },
        openEvents: createFetchSseTransport(eventsUrl, credentialedFetch, boundedParserOptions),
        clear() {
            secrets.apiKey = "";
            secrets.operatorToken = null;
        },
        endpoints: Object.freeze({ snapshotUrl, eventsUrl, detailUrl, controlUrl }),
    };
}

export function createLiveDashboardTransport({
    baseUrl,
    apiKey,
    operatorToken,
    fetchImpl = globalThis.fetch,
    parserOptions = {},
}) {
    return createDashboardTransport({
        base: normalizeLoopbackBaseUrl(baseUrl),
        apiKey,
        operatorToken: credential(
            operatorToken,
            "operator token",
            MIN_OPERATOR_TOKEN_BYTES,
            MAX_OPERATOR_TOKEN_BYTES,
        ),
        fetchImpl,
        parserOptions,
    });
}

export function createGatewayDashboardTransport({
    pageOrigin,
    apiKey,
    fetchImpl = globalThis.fetch,
    parserOptions = {},
}) {
    return createDashboardTransport({
        base: normalizeGatewayOrigin(pageOrigin),
        apiKey,
        operatorToken: null,
        fetchImpl,
        parserOptions,
    });
}

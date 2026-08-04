import { createFetchSseTransport } from "./sse.mjs";

export const OPERATOR_TOKEN_HEADER = "X-Llama-Trusted-Scheduling-Token";
export const SNAPSHOT_PATH = "/internal/admin/dashboard/snapshot";
export const EVENTS_PATH = "/internal/admin/dashboard/events";

const MAX_API_KEY_BYTES = 1024;
const MIN_OPERATOR_TOKEN_BYTES = 32;
const MAX_OPERATOR_TOKEN_BYTES = 256;
const MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024;
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

function endpointUrl(base, path) {
    const result = new URL(base.href);
    const prefix = base.pathname === "/" ? "" : base.pathname;
    result.pathname = `${prefix}${path}`;
    return result.href;
}

async function boundedJson(response) {
    const contentLength = Number(response.headers.get("Content-Length"));
    if (Number.isFinite(contentLength) && contentLength > MAX_SNAPSHOT_BYTES) {
        throw new RangeError(`snapshot response exceeds ${MAX_SNAPSHOT_BYTES} bytes`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
        throw new Error("snapshot response body is not a readable stream");
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
            if (total > MAX_SNAPSHOT_BYTES) {
                await reader.cancel();
                throw new RangeError(`snapshot response exceeds ${MAX_SNAPSHOT_BYTES} bytes`);
            }
            text += decoder.decode(value, { stream: true });
        }
        text += decoder.decode();
    } finally {
        reader.releaseLock();
    }
    return JSON.parse(text);
}

export function createLiveDashboardTransport({
    baseUrl,
    apiKey,
    operatorToken,
    fetchImpl = globalThis.fetch,
    parserOptions = {},
}) {
    if (typeof fetchImpl !== "function") {
        throw new Error("fetch is unavailable");
    }
    const base = normalizeLoopbackBaseUrl(baseUrl);
    const secrets = {
        apiKey: credential(apiKey, "API key", 1, MAX_API_KEY_BYTES),
        operatorToken: credential(
            operatorToken,
            "operator token",
            MIN_OPERATOR_TOKEN_BYTES,
            MAX_OPERATOR_TOKEN_BYTES,
        ),
    };

    const credentialedFetch = (url, options = {}) => {
        if (secrets.apiKey === "" || secrets.operatorToken === "") {
            throw new Error("live dashboard credentials were cleared");
        }
        const headers = new Headers(options.headers ?? {});
        headers.set("Authorization", `Bearer ${secrets.apiKey}`);
        headers.set(OPERATOR_TOKEN_HEADER, secrets.operatorToken);
        headers.delete("X-Llama-Trusted-Lane");
        headers.delete("X-Llama-Benchmark-Tag");
        return fetchImpl(url, {
            ...options,
            method: "GET",
            headers,
            credentials: "omit",
            cache: "no-store",
            redirect: "error",
            referrerPolicy: "no-referrer",
        });
    };

    const snapshotUrl = endpointUrl(base, SNAPSHOT_PATH);
    const eventsUrl = endpointUrl(base, EVENTS_PATH);
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
        openEvents: createFetchSseTransport(eventsUrl, credentialedFetch, boundedParserOptions),
        clear() {
            secrets.apiKey = "";
            secrets.operatorToken = "";
        },
        endpoints: Object.freeze({ snapshotUrl, eventsUrl }),
    };
}

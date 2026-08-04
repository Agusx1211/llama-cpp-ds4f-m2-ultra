#!/usr/bin/env node

import { createInterface } from "node:readline";
import { pathToFileURL } from "node:url";

import {
    createLiveDashboardTransport,
    OPERATOR_TOKEN_HEADER,
} from "../lib/live.mjs";
import { parseEvent, parseSnapshot, SCHEMA_VERSION } from "../lib/schema.mjs";

const API_KEY_ENV = "LLAMA_API_KEY";
const OPERATOR_TOKEN_ENV = "LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN";
const DEFAULT_TIMEOUT_MS = 120_000;
const MAX_TIMEOUT_MS = 15 * 60_000;
const MAX_CAPTURED_EVENTS = 4096;

function requiredEnvironment(name) {
    const value = process.env[name];
    if (typeof value !== "string" || value.length === 0) {
        throw new Error(`${name} is required in the environment`);
    }
    return value;
}

function boundedInteger(name, fallback, minimum, maximum) {
    const text = process.env[name];
    if (text === undefined || text === "") {
        return fallback;
    }
    const value = Number(text);
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
        throw new RangeError(`${name} must be an integer from ${minimum} through ${maximum}`);
    }
    return value;
}

function truthyEnvironment(name) {
    return ["1", "true", "yes"].includes((process.env[name] ?? "").toLowerCase());
}

function credentialHeaders(apiKey = null, operatorToken = null, extra = {}) {
    const headers = new Headers(extra);
    if (apiKey !== null) {
        headers.set("Authorization", `Bearer ${apiKey}`);
    }
    if (operatorToken !== null) {
        headers.set(OPERATOR_TOKEN_HEADER, operatorToken);
    }
    return headers;
}

async function fetchStatus(url, headers, timeoutMs = 30_000) {
    const response = await fetch(url, {
        method: "GET",
        headers,
        credentials: "omit",
        cache: "no-store",
        redirect: "error",
        referrerPolicy: "no-referrer",
        signal: AbortSignal.timeout(timeoutMs),
    });
    await response.body?.cancel();
    return response.status;
}

function requireDenied(status, label) {
    if (status < 400 || status > 499) {
        throw new Error(`${label} returned HTTP ${status}, expected a 4xx denial`);
    }
    return status;
}

async function within(promise, timeoutMs, label) {
    let timer;
    try {
        return await Promise.race([
            promise,
            new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error(`${label} exceeded ${timeoutMs} ms`)), timeoutMs);
            }),
        ]);
    } finally {
        clearTimeout(timer);
    }
}

export function verifyRedaction(snapshot, secrets) {
    if (snapshot.availability?.content !== false) {
        throw new Error("live snapshot must mark request content unavailable");
    }
    for (const request of snapshot.requests) {
        if (request.content.retained !== false || request.content.prompt !== "" || request.content.output !== "") {
            throw new Error(`request ${request.id} exposed retained prompt or output content`);
        }
    }
    const encoded = JSON.stringify(snapshot);
    for (const secret of secrets) {
        if (encoded.includes(secret)) {
            throw new Error("dashboard snapshot reflected a credential value");
        }
    }
}

function createStopWaiter(timeoutMs) {
    const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
    let timer;
    let settled = false;
    const promise = new Promise((resolve, reject) => {
        timer = setTimeout(() => {
            settled = true;
            input.close();
            reject(new Error(`probe exceeded its ${timeoutMs} ms connected hold bound`));
        }, timeoutMs);
        input.once("line", (line) => {
            clearTimeout(timer);
            settled = true;
            input.close();
            if (line !== "STOP") {
                reject(new Error("probe control input must be STOP"));
                return;
            }
            resolve();
        });
    });
    return {
        promise,
        cancel() {
            if (!settled) {
                settled = true;
                clearTimeout(timer);
                input.close();
            }
        },
    };
}

function openBoundedStream(transport, lastEventId, events) {
    let close = () => {};
    let closing = false;
    let settled = false;
    let rejectFailure = () => {};
    const failure = new Promise((_, reject) => {
        rejectFailure = reject;
    });
    failure.catch(() => {});
    const opened = new Promise((resolve, reject) => {
        close = transport.openEvents({
            lastEventId,
            onOpen: () => {
                settled = true;
                resolve();
            },
            onEvent: (message) => {
                if (events.length >= MAX_CAPTURED_EVENTS) {
                    rejectFailure(new Error(`probe captured more than ${MAX_CAPTURED_EVENTS} SSE events`));
                    close();
                    return;
                }
                let event;
                try {
                    event = parseEvent(JSON.parse(message.data));
                } catch (error) {
                    rejectFailure(error);
                    close();
                    return;
                }
                if (message.lastEventId !== event.id) {
                    rejectFailure(new Error("SSE transport and JSON event IDs differ"));
                    close();
                    return;
                }
                events.push(event);
            },
            onDisconnect: (error) => {
                if (!closing) {
                    reject(error);
                    rejectFailure(error);
                }
            },
        });
    });
    return {
        opened,
        failure,
        close() {
            closing = true;
            close();
        },
        established() {
            return settled;
        },
    };
}

export function verifyMonotonicEvents(events, startingSequence) {
    let sequence = startingSequence;
    for (const event of events) {
        if (event.sequence !== sequence + 1) {
            throw new Error(`non-contiguous SSE sequence: expected ${sequence + 1}, received ${event.sequence}`);
        }
        sequence = event.sequence;
    }
    return sequence;
}

export function refillStateLabels(value) {
    if (value === null || typeof value !== "object") {
        return [];
    }
    const configuration = value.configuration ?? {};
    const cohort = value.cohort ?? {};
    const refill = value.refill ?? {};
    const states = [];
    if (configuration.enabled === false) {
        states.push("disabled");
    }
    if (cohort.active === false) {
        states.push("inactive");
    }
    if (refill.window_open === true) {
        states.push("window_open");
    }
    if (refill.one_member_eligible_now === true) {
        states.push("one_member_eligible");
    }
    if (refill.fast_members_remaining === 0) {
        states.push("quota_exhausted");
    }
    if (refill.deadline_expired === true) {
        states.push("deadline_expired");
    }
    if (refill.window_open === true && refill.one_member_eligible_now === false) {
        states.push("full_width");
    }
    return states;
}

export function verifyOptionalRefill(snapshot, events, { required = false, expected = [] } = {}) {
    const snapshotRefill = snapshot.fast_refill;
    const eventRefills = events
        .map((event) => event.payload?.fast_refill)
        .filter((value) => value !== null && typeof value === "object");
    const observed = [snapshotRefill, ...eventRefills].flatMap(refillStateLabels);
    const missing = expected.filter((value) => !observed.includes(value));
    if (required && (snapshotRefill === undefined || eventRefills.length === 0)) {
        throw new Error("required authoritative fast_refill snapshot/event objects were not observed");
    }
    if (required && missing.length > 0) {
        throw new Error(`required refill states were not observed: ${missing.join(", ")}`);
    }
    return {
        required,
        available: snapshotRefill !== undefined && eventRefills.length > 0,
        snapshot_exposed: snapshotRefill !== undefined,
        event_objects: eventRefills.length,
        observed_states: [...new Set(observed)],
    };
}

async function main() {
    const baseUrl = requiredEnvironment("M2_DASHBOARD_BASE_URL");
    const apiKey = requiredEnvironment(API_KEY_ENV);
    const operatorToken = requiredEnvironment(OPERATOR_TOKEN_ENV);
    if (apiKey === operatorToken) {
        throw new Error("API and operator credentials must be distinct");
    }
    const timeoutMs = boundedInteger(
        "M2_DASHBOARD_PROBE_TIMEOUT_MS",
        DEFAULT_TIMEOUT_MS,
        1000,
        MAX_TIMEOUT_MS,
    );
    const minimumEvents = boundedInteger("M2_DASHBOARD_MIN_EVENTS", 1, 0, MAX_CAPTURED_EVENTS);
    const establishTimeoutMs = boundedInteger(
        "M2_DASHBOARD_ESTABLISH_TIMEOUT_MS",
        30_000,
        1000,
        120_000,
    );
    const transport = createLiveDashboardTransport({
        baseUrl,
        apiKey,
        operatorToken,
        fetchImpl(url, options) {
            return fetch(url, {
                ...options,
                signal: AbortSignal.timeout(
                    String(url).includes("/events") ? timeoutMs : establishTimeoutMs,
                ),
            });
        },
    });
    try {
        const { snapshotUrl, eventsUrl } = transport.endpoints;
        const both = credentialHeaders(apiKey, operatorToken);
        const negativeAuth = {};
        for (const [route, url] of Object.entries({ snapshot: snapshotUrl, events: eventsUrl })) {
            negativeAuth[`${route}_api_only`] = requireDenied(
                await fetchStatus(url, credentialHeaders(apiKey)),
                `API-only ${route}`,
            );
            negativeAuth[`${route}_operator_only`] = requireDenied(
                await fetchStatus(url, credentialHeaders(null, operatorToken)),
                `operator-only ${route}`,
            );
            negativeAuth[`${route}_query_rejected`] = requireDenied(
                await fetchStatus(`${url}?probe=1`, both),
                `query-bearing ${route}`,
            );
            negativeAuth[`${route}_classification_rejected`] = requireDenied(
                await fetchStatus(url, credentialHeaders(apiKey, operatorToken, {
                    "X-Llama-Trusted-Lane": "fast",
                })),
                `classification-bearing ${route}`,
            );
        }

        const firstSnapshot = parseSnapshot(await transport.getSnapshot());
        verifyRedaction(firstSnapshot, [apiKey, operatorToken]);

        const futureCursorStatus = await fetchStatus(eventsUrl, credentialHeaders(apiKey, operatorToken, {
            "Last-Event-ID": String(Number.MAX_SAFE_INTEGER),
        }));
        if (futureCursorStatus !== 409) {
            throw new Error(`future SSE cursor returned HTTP ${futureCursorStatus}, expected 409`);
        }
        const recoveredSnapshot = parseSnapshot(await transport.getSnapshot());
        verifyRedaction(recoveredSnapshot, [apiKey, operatorToken]);

        const events = [];
        const stream = openBoundedStream(transport, String(recoveredSnapshot.sequence), events);
        try {
            await within(stream.opened, establishTimeoutMs, "initial SSE establishment");
        } catch (error) {
            stream.close();
            throw error;
        }
        process.stdout.write(`${JSON.stringify({ status: "READY", sequence: recoveredSnapshot.sequence })}\n`);
        const stopWaiter = createStopWaiter(timeoutMs);
        try {
            await Promise.race([stopWaiter.promise, stream.failure]);
        } finally {
            stopWaiter.cancel();
            stream.close();
        }

        if (events.length < minimumEvents) {
            throw new Error(`captured ${events.length} SSE events, required at least ${minimumEvents}`);
        }
        const finalSequence = verifyMonotonicEvents(events, recoveredSnapshot.sequence);
        const resumed = openBoundedStream(transport, String(finalSequence), []);
        try {
            await within(resumed.opened, establishTimeoutMs, "resumed SSE establishment");
        } finally {
            resumed.close();
        }

        const result = {
            report_schema_version: 1,
            status: "PASS",
            negative_auth_statuses: negativeAuth,
            snapshot: {
                schema_version: recoveredSnapshot.schema_version,
                client_schema_version: SCHEMA_VERSION,
                sequence: recoveredSnapshot.sequence,
                request_count: recoveredSnapshot.requests.length,
                redacted: true,
            },
            sse: {
                established: stream.established(),
                captured_events: events.length,
                first_sequence: events[0]?.sequence ?? null,
                final_sequence: finalSequence,
                contiguous: true,
                resume_established: resumed.established(),
                future_cursor_resnapshot_status: futureCursorStatus,
                maximum_captured_events: MAX_CAPTURED_EVENTS,
            },
            refill: verifyOptionalRefill(recoveredSnapshot, events, {
                required: truthyEnvironment("M2_DASHBOARD_REQUIRE_REFILL_STATES"),
                expected: (process.env.M2_DASHBOARD_REFILL_STATES ?? "")
                    .split(",")
                    .map((value) => value.trim())
                    .filter(Boolean),
            }),
            credentials: {
                source: "environment",
                emitted: false,
                stored: false,
            },
        };
        const encoded = JSON.stringify(result);
        if (encoded.includes(apiKey) || encoded.includes(operatorToken)) {
            throw new Error("probe result contains a credential value");
        }
        process.stdout.write(`${encoded}\n`);
    } finally {
        transport.clear();
    }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
    main().catch((error) => {
        process.stderr.write(`dashboard live probe failed: ${error.message ?? error}\n`);
        process.exitCode = 1;
    });
}

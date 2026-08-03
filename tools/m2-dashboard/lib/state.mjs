import {
    SchemaError,
    cloneJson,
    deepFreeze,
    parseEvent,
    parseSnapshot,
} from "./schema.mjs";

export const DEFAULT_EVENT_HISTORY_LIMIT = 128;
export const DEFAULT_TIMELINE_LIMIT = 256;

function positiveLimit(value, name) {
    if (!Number.isSafeInteger(value) || value < 1) {
        throw new RangeError(`${name} must be a positive integer`);
    }
    return value;
}

function freezeState(value) {
    return deepFreeze(value);
}

export function createDashboardState(snapshotValue, options = {}) {
    const historyLimit = positiveLimit(
        options.historyLimit ?? DEFAULT_EVENT_HISTORY_LIMIT,
        "historyLimit",
    );
    const timelineLimit = positiveLimit(
        options.timelineLimit ?? DEFAULT_TIMELINE_LIMIT,
        "timelineLimit",
    );
    const parsedSnapshot = parseSnapshot(snapshotValue);
    const snapshot = parsedSnapshot.timeline.length > timelineLimit
        ? deepFreeze({
            ...parsedSnapshot,
            timeline: parsedSnapshot.timeline.slice(-timelineLimit),
        })
        : parsedSnapshot;

    return freezeState({
        snapshot,
        lastSequence: snapshot.sequence,
        lastEventId: String(snapshot.sequence),
        history: [],
        historyLimit,
        timelineLimit,
        synchronization: {
            status: "live",
            reason: null,
            details: null,
            duplicates: 0,
        },
        recovery: {
            status: "idle",
            reason: null,
            attempt: 0,
            lastError: null,
            retryDelayMs: null,
        },
        connection: "disconnected",
    });
}

export function requireResnapshot(state, reason, details = null) {
    if (state.synchronization.status === "resnapshot_required") {
        return state;
    }
    return freezeState({
        ...state,
        synchronization: {
            ...state.synchronization,
            status: "resnapshot_required",
            reason,
            details: cloneJson(details),
        },
    });
}

export function setConnection(state, connection) {
    if (!["disconnected", "connecting", "live", "retry_wait", "error", "stopped"].includes(connection)) {
        throw new RangeError(`unknown connection state: ${connection}`);
    }
    if (state.connection === connection) {
        return state;
    }
    return freezeState({ ...state, connection });
}

export function setRecovery(state, recovery) {
    return freezeState({
        ...state,
        recovery: {
            ...state.recovery,
            ...cloneJson(recovery),
        },
    });
}

function replaceById(values, replacement) {
    const index = values.findIndex((value) => value.id === replacement.id);
    if (index < 0) {
        return [...values, replacement];
    }
    return values.map((value, current) => current === index ? replacement : value);
}

function applyEvent(snapshot, event, timelineLimit) {
    const next = {
        ...snapshot,
        sequence: event.sequence,
        generated_at: event.wall_time,
    };

    switch (event.type) {
        case "request.upsert":
            next.requests = replaceById(snapshot.requests, event.payload.request);
            break;
        case "request.remove":
            next.requests = snapshot.requests.filter((request) => request.id !== event.payload.request_id);
            break;
        case "lane.replace":
            next.lanes = replaceById(snapshot.lanes, event.payload.lane);
            break;
        case "allocator.replace":
            next.allocator = event.payload.allocator;
            break;
        case "cache.replace":
            next.cache = event.payload.cache;
            break;
        case "disk.replace":
            next.disks = event.payload.disks;
            break;
        case "dspark.replace":
            next.dspark = event.payload.dspark;
            break;
        case "capture.replace":
            next.capture = event.payload.capture;
            break;
        case "server.replace":
            next.server = event.payload.server;
            break;
        case "timeline.append":
            next.timeline = [...snapshot.timeline, event.payload.item].slice(-timelineLimit);
            break;
        default:
            break;
    }
    return deepFreeze(next);
}

function parseEventOrResnapshot(state, eventValue) {
    try {
        return { event: parseEvent(eventValue), state };
    } catch (error) {
        if (!(error instanceof SchemaError)) {
            throw error;
        }
        return {
            event: null,
            state: requireResnapshot(state, "invalid_event", {
                message: error.message,
            }),
        };
    }
}

export function reduceDashboardEvent(state, eventValue) {
    if (state.synchronization.status === "resnapshot_required") {
        return state;
    }

    const parsed = parseEventOrResnapshot(state, eventValue);
    if (parsed.event === null) {
        return parsed.state;
    }
    const event = parsed.event;

    if (event.sequence <= state.lastSequence) {
        return freezeState({
            ...state,
            synchronization: {
                ...state.synchronization,
                duplicates: state.synchronization.duplicates + 1,
            },
        });
    }

    if (event.type === "stream.overflow") {
        return requireResnapshot(state, "server_overflow", {
            eventId: event.id,
            oldestAvailableSequence: event.payload.oldest_available_sequence,
        });
    }
    if (event.type === "stream.slow_client") {
        return requireResnapshot(state, "server_slow_client", {
            eventId: event.id,
            droppedEvents: event.payload.dropped_events,
        });
    }

    const expected = state.lastSequence + 1;
    if (event.sequence !== expected) {
        return requireResnapshot(state, "sequence_gap", {
            expected,
            received: event.sequence,
            lastEventId: state.lastEventId,
        });
    }

    const snapshot = applyEvent(state.snapshot, event, state.timelineLimit);
    const history = [...state.history, event].slice(-state.historyLimit);
    return freezeState({
        ...state,
        snapshot,
        lastSequence: event.sequence,
        lastEventId: event.id,
        history,
    });
}

export function replaceDashboardSnapshot(state, snapshotValue) {
    const replacement = createDashboardState(snapshotValue, {
        historyLimit: state.historyLimit,
        timelineLimit: state.timelineLimit,
    });
    return freezeState({
        ...replacement,
        connection: state.connection,
    });
}

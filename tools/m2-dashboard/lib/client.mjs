import { DEFAULT_MAX_SSE_EVENT_BYTES } from "./sse.mjs";
import {
    createDashboardState,
    reduceDashboardEvent,
    replaceDashboardSnapshot,
    requireResnapshot,
    setConnection,
    setRecovery,
} from "./state.mjs";

const DEFAULT_SNAPSHOT_RETRY_DELAYS_MS = Object.freeze([250, 1000, 3000]);
const MAX_RETRY_STEPS = 16;
const MAX_ERROR_MESSAGE_LENGTH = 2048;
const textEncoder = new TextEncoder();

function defaultScheduler(callback) {
    queueMicrotask(callback);
}

function validateRetryDelays(value, name) {
    if (!Array.isArray(value) || value.length > MAX_RETRY_STEPS) {
        throw new RangeError(`${name} must be an array with at most ${MAX_RETRY_STEPS} entries`);
    }
    for (const delay of value) {
        if (!Number.isSafeInteger(delay) || delay < 0) {
            throw new RangeError(`${name} entries must be non-negative integers`);
        }
    }
    return Object.freeze([...value]);
}

function parseJsonMessage(value, maxEventBytes) {
    if (value.length > maxEventBytes || textEncoder.encode(value).byteLength > maxEventBytes) {
        throw new RangeError(`SSE event exceeds ${maxEventBytes} bytes`);
    }
    return JSON.parse(value);
}

function normalizeMessage(message, maxEventBytes) {
    if (message !== null && typeof message === "object" && typeof message.data === "string") {
        const value = parseJsonMessage(message.data, maxEventBytes);
        if (message.lastEventId && value.id !== message.lastEventId) {
            throw new Error(`SSE id mismatch: envelope=${value.id} transport=${message.lastEventId}`);
        }
        return value;
    }
    if (typeof message === "string") {
        return parseJsonMessage(message, maxEventBytes);
    }
    return message;
}

function errorMessage(error) {
    return String(error?.message ?? error).slice(0, MAX_ERROR_MESSAGE_LENGTH);
}

export class AdminStateClient {
    constructor({
        getSnapshot,
        openEvents,
        onState = () => {},
        historyLimit = 128,
        timelineLimit = 256,
        pendingLimit = 64,
        maxEventBytes = DEFAULT_MAX_SSE_EVENT_BYTES,
        schedule = defaultScheduler,
        autoDrain = true,
        reconnectDelayMs = 750,
        snapshotRetryDelaysMs = DEFAULT_SNAPSHOT_RETRY_DELAYS_MS,
        streamRetryDelaysMs,
        setTimer = globalThis.setTimeout,
        clearTimer = globalThis.clearTimeout,
    }) {
        if (typeof getSnapshot !== "function") {
            throw new TypeError("getSnapshot must be a function");
        }
        if (typeof openEvents !== "function") {
            throw new TypeError("openEvents must be a function");
        }
        if (!Number.isSafeInteger(pendingLimit) || pendingLimit < 1) {
            throw new RangeError("pendingLimit must be a positive integer");
        }
        if (!Number.isSafeInteger(maxEventBytes) || maxEventBytes < 1) {
            throw new RangeError("maxEventBytes must be a positive integer");
        }
        if (!Number.isSafeInteger(reconnectDelayMs) || reconnectDelayMs < 0) {
            throw new RangeError("reconnectDelayMs must be a non-negative integer");
        }
        if (typeof setTimer !== "function" || typeof clearTimer !== "function") {
            throw new TypeError("setTimer and clearTimer must be functions");
        }

        const defaultStreamDelays = [reconnectDelayMs, reconnectDelayMs * 2, reconnectDelayMs * 4];
        this.getSnapshot = getSnapshot;
        this.openEvents = openEvents;
        this.onState = onState;
        this.historyLimit = historyLimit;
        this.timelineLimit = timelineLimit;
        this.pendingLimit = pendingLimit;
        this.maxEventBytes = maxEventBytes;
        this.schedule = schedule;
        this.autoDrain = autoDrain;
        this.snapshotRetryDelaysMs = validateRetryDelays(snapshotRetryDelaysMs, "snapshotRetryDelaysMs");
        this.streamRetryDelaysMs = validateRetryDelays(
            streamRetryDelaysMs ?? defaultStreamDelays,
            "streamRetryDelaysMs",
        );
        this.setTimer = setTimer;
        this.clearTimer = clearTimer;

        this.state = null;
        this.pending = [];
        this.closeEvents = null;
        this.generation = 0;
        this.stopped = false;
        this.drainScheduled = false;
        this.recoveryPromise = null;
        this.snapshotRetryTimer = null;
        this.streamRetryTimer = null;
    }

    publish() {
        if (this.state !== null) {
            this.onState(this.state);
        }
    }

    closeStream() {
        if (this.closeEvents !== null) {
            const close = this.closeEvents;
            this.closeEvents = null;
            close();
        }
    }

    clearRetryTimer(field) {
        if (this[field] !== null) {
            this.clearTimer(this[field]);
            this[field] = null;
        }
    }

    clearRetries() {
        this.clearRetryTimer("snapshotRetryTimer");
        this.clearRetryTimer("streamRetryTimer");
    }

    installSnapshot(snapshotValue) {
        if (this.state === null) {
            this.state = createDashboardState(snapshotValue, {
                historyLimit: this.historyLimit,
                timelineLimit: this.timelineLimit,
            });
        } else {
            this.state = replaceDashboardSnapshot(this.state, snapshotValue);
        }
        this.state = setRecovery(this.state, {
            status: "idle",
            reason: null,
            attempt: 0,
            lastError: null,
            retryDelayMs: null,
        });
        this.state = setConnection(this.state, "connecting");
        this.publish();
    }

    async start() {
        this.stopped = false;
        const generation = ++this.generation;
        this.pending.length = 0;
        this.clearRetries();
        this.closeStream();
        if (this.state !== null) {
            this.state = setRecovery(this.state, {
                status: "fetching",
                reason: "initial",
                attempt: 1,
                lastError: null,
                retryDelayMs: null,
            });
            this.state = setConnection(this.state, "connecting");
            this.publish();
        }

        try {
            const snapshotValue = await Promise.resolve().then(() => this.getSnapshot({
                reason: "initial",
                attempt: 1,
            }));
            if (this.stopped || generation !== this.generation) {
                return this.state;
            }
            this.installSnapshot(snapshotValue);
            this.connect(generation);
            return this.state;
        } catch (error) {
            if (!this.stopped && generation === this.generation && this.state !== null) {
                this.state = setRecovery(this.state, {
                    status: "error",
                    reason: "initial",
                    attempt: 1,
                    lastError: errorMessage(error),
                    retryDelayMs: null,
                });
                this.state = setConnection(this.state, "error");
                this.publish();
            }
            throw error;
        }
    }

    stop() {
        this.stopped = true;
        this.generation += 1;
        this.pending.length = 0;
        this.clearRetries();
        this.closeStream();
        if (this.state !== null) {
            this.state = setConnection(this.state, "stopped");
            this.publish();
        }
    }

    resnapshot(reason) {
        if (this.stopped) {
            return Promise.resolve(this.state);
        }
        if (this.recoveryPromise !== null) {
            return this.recoveryPromise;
        }
        if (this.snapshotRetryTimer !== null) {
            return Promise.resolve(this.state);
        }

        const generation = ++this.generation;
        this.pending.length = 0;
        this.clearRetries();
        this.closeStream();
        return this.runSnapshotAttempt(reason, generation, 1);
    }

    runSnapshotAttempt(reason, generation, attempt) {
        if (this.stopped || this.state === null || generation !== this.generation) {
            return Promise.resolve(this.state);
        }

        this.state = setRecovery(this.state, {
            status: "fetching",
            reason,
            attempt,
            lastError: null,
            retryDelayMs: null,
        });
        this.state = setConnection(this.state, "connecting");
        this.publish();

        const attemptPromise = Promise.resolve()
            .then(() => this.getSnapshot({ reason, attempt }))
            .then((snapshotValue) => {
                if (this.stopped || generation !== this.generation) {
                    return this.state;
                }
                this.installSnapshot(snapshotValue);
                this.connect(generation);
                return this.state;
            })
            .catch((error) => {
                if (this.stopped || generation !== this.generation) {
                    return this.state;
                }
                this.scheduleSnapshotRetry(reason, generation, attempt, error);
                return this.state;
            });

        const trackedPromise = attemptPromise.finally(() => {
            if (this.recoveryPromise === trackedPromise) {
                this.recoveryPromise = null;
            }
        });
        this.recoveryPromise = trackedPromise;
        return trackedPromise;
    }

    scheduleSnapshotRetry(reason, generation, attempt, error) {
        const retryIndex = attempt - 1;
        if (retryIndex >= this.snapshotRetryDelaysMs.length) {
            this.state = setRecovery(this.state, {
                status: "error",
                reason,
                attempt,
                lastError: errorMessage(error),
                retryDelayMs: null,
            });
            this.state = setConnection(this.state, "error");
            this.publish();
            return;
        }

        const delay = this.snapshotRetryDelaysMs[retryIndex];
        this.state = setRecovery(this.state, {
            status: "retry_wait",
            reason,
            attempt,
            lastError: errorMessage(error),
            retryDelayMs: delay,
        });
        this.state = setConnection(this.state, "retry_wait");
        this.publish();
        const timer = this.setTimer(() => {
            if (this.snapshotRetryTimer === timer) {
                this.snapshotRetryTimer = null;
            }
            if (!this.stopped && generation === this.generation) {
                void this.runSnapshotAttempt(reason, generation, attempt + 1);
            }
        }, delay);
        this.snapshotRetryTimer = timer;
    }

    connect(generation = this.generation, retryIndex = 0) {
        if (this.stopped || this.state === null || generation !== this.generation) {
            return;
        }
        this.clearRetryTimer("streamRetryTimer");
        this.closeStream();
        const streamGeneration = generation;
        let disconnected = false;
        let streamEstablished = false;
        let close;
        try {
            close = this.openEvents({
                lastEventId: this.state.lastEventId,
                onOpen: () => {
                    if (!this.stopped && !disconnected && streamGeneration === this.generation) {
                        streamEstablished = true;
                    }
                },
                onEvent: (message) => {
                    if (!this.stopped && !disconnected && streamGeneration === this.generation) {
                        this.enqueue(message);
                    }
                },
                onDisconnect: (error) => {
                    if (!this.stopped && !disconnected && streamGeneration === this.generation) {
                        disconnected = true;
                        this.scheduleStreamRetry(
                            "event_stream_disconnected",
                            error,
                            streamGeneration,
                            streamEstablished ? 0 : retryIndex,
                        );
                    }
                },
            });
        } catch (error) {
            this.scheduleStreamRetry("open_events_failed", error, streamGeneration, retryIndex);
            return;
        }

        if (disconnected || this.stopped || streamGeneration !== this.generation) {
            if (typeof close === "function") {
                close();
            }
            return;
        }
        const closeTransport = typeof close === "function" ? close : () => {};
        this.closeEvents = () => {
            disconnected = true;
            closeTransport();
        };
        this.state = setRecovery(this.state, {
            status: "idle",
            reason: null,
            attempt: 0,
            lastError: null,
            retryDelayMs: null,
        });
        this.state = setConnection(this.state, "live");
        this.publish();
    }

    scheduleStreamRetry(reason, error, generation, retryIndex) {
        if (this.stopped || this.state === null || generation !== this.generation) {
            return;
        }
        this.closeStream();
        if (error?.resnapshotRequired === true) {
            this.state = requireResnapshot(this.state, "server_resume_rejected", {
                message: errorMessage(error),
            });
            this.publish();
            void this.resnapshot("server_resume_rejected");
            return;
        }
        if (retryIndex >= this.streamRetryDelaysMs.length) {
            this.state = setRecovery(this.state, {
                status: "error",
                reason,
                attempt: retryIndex + 1,
                lastError: errorMessage(error),
                retryDelayMs: null,
            });
            this.state = setConnection(this.state, "error");
            this.publish();
            return;
        }

        const delay = this.streamRetryDelaysMs[retryIndex];
        this.state = setRecovery(this.state, {
            status: "retry_wait",
            reason,
            attempt: retryIndex + 1,
            lastError: errorMessage(error),
            retryDelayMs: delay,
        });
        this.state = setConnection(this.state, "retry_wait");
        this.publish();
        const timer = this.setTimer(() => {
            if (this.streamRetryTimer === timer) {
                this.streamRetryTimer = null;
            }
            if (!this.stopped && generation === this.generation) {
                this.connect(generation, retryIndex + 1);
            }
        }, delay);
        this.streamRetryTimer = timer;
    }

    enqueue(message) {
        if (this.stopped || this.state === null) {
            return;
        }
        if (this.pending.length >= this.pendingLimit) {
            this.pending.length = 0;
            this.state = requireResnapshot(this.state, "local_slow_consumer", {
                pendingLimit: this.pendingLimit,
            });
            this.publish();
            void this.resnapshot("local_slow_consumer");
            return;
        }

        this.pending.push(message);
        if (this.autoDrain && !this.drainScheduled) {
            this.drainScheduled = true;
            this.schedule(() => {
                this.drainScheduled = false;
                this.drain();
            });
        }
    }

    drain() {
        if (this.stopped || this.state === null) {
            this.pending.length = 0;
            return;
        }

        while (this.pending.length > 0 && this.state.synchronization.status === "live") {
            let event;
            try {
                event = normalizeMessage(this.pending.shift(), this.maxEventBytes);
            } catch (error) {
                this.pending.length = 0;
                this.state = requireResnapshot(this.state, "invalid_sse_message", {
                    message: errorMessage(error),
                });
                break;
            }
            this.state = reduceDashboardEvent(this.state, event);
        }
        this.publish();

        if (this.state.synchronization.status === "resnapshot_required") {
            const reason = this.state.synchronization.reason;
            this.pending.length = 0;
            void this.resnapshot(reason);
        }
    }

    async idle() {
        await Promise.resolve();
        if (this.recoveryPromise !== null) {
            await this.recoveryPromise;
        }
        await Promise.resolve();
        return this.state;
    }
}

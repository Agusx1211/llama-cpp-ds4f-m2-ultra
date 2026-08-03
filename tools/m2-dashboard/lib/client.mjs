import {
    createDashboardState,
    reduceDashboardEvent,
    replaceDashboardSnapshot,
    requireResnapshot,
    setConnection,
} from "./state.mjs";

function defaultScheduler(callback) {
    queueMicrotask(callback);
}

function normalizeMessage(message) {
    if (message !== null && typeof message === "object" && typeof message.data === "string") {
        const value = JSON.parse(message.data);
        if (message.lastEventId && value.id !== message.lastEventId) {
            throw new Error(`SSE id mismatch: envelope=${value.id} transport=${message.lastEventId}`);
        }
        return value;
    }
    if (typeof message === "string") {
        return JSON.parse(message);
    }
    return message;
}

export class AdminStateClient {
    constructor({
        getSnapshot,
        openEvents,
        onState = () => {},
        historyLimit = 128,
        timelineLimit = 256,
        pendingLimit = 64,
        schedule = defaultScheduler,
        autoDrain = true,
        reconnectDelayMs = 750,
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
        if (!Number.isSafeInteger(reconnectDelayMs) || reconnectDelayMs < 0) {
            throw new RangeError("reconnectDelayMs must be a non-negative integer");
        }

        this.getSnapshot = getSnapshot;
        this.openEvents = openEvents;
        this.onState = onState;
        this.historyLimit = historyLimit;
        this.timelineLimit = timelineLimit;
        this.pendingLimit = pendingLimit;
        this.schedule = schedule;
        this.autoDrain = autoDrain;
        this.reconnectDelayMs = reconnectDelayMs;

        this.state = null;
        this.pending = [];
        this.closeEvents = null;
        this.generation = 0;
        this.stopped = false;
        this.drainScheduled = false;
        this.recoveryPromise = null;
        this.reconnectTimer = null;
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

    clearReconnect() {
        if (this.reconnectTimer !== null) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    async start() {
        this.stopped = false;
        await this.resnapshot("initial");
        return this.state;
    }

    stop() {
        this.stopped = true;
        this.generation += 1;
        this.pending.length = 0;
        this.clearReconnect();
        this.closeStream();
        if (this.state !== null) {
            this.state = setConnection(this.state, "stopped");
            this.publish();
        }
    }

    async resnapshot(reason) {
        if (this.stopped) {
            return this.state;
        }
        if (this.recoveryPromise !== null) {
            return this.recoveryPromise;
        }

        const generation = ++this.generation;
        this.pending.length = 0;
        this.clearReconnect();
        this.closeStream();
        if (this.state !== null) {
            this.state = setConnection(this.state, "connecting");
            this.publish();
        }

        this.recoveryPromise = (async () => {
            const snapshotValue = await this.getSnapshot({ reason });
            if (this.stopped || generation !== this.generation) {
                return this.state;
            }

            if (this.state === null) {
                this.state = createDashboardState(snapshotValue, {
                    historyLimit: this.historyLimit,
                    timelineLimit: this.timelineLimit,
                });
            } else {
                this.state = replaceDashboardSnapshot(this.state, snapshotValue);
            }
            this.state = setConnection(this.state, "connecting");
            this.publish();
            this.connect(generation);
            return this.state;
        })();

        try {
            return await this.recoveryPromise;
        } finally {
            this.recoveryPromise = null;
        }
    }

    connect(generation = this.generation) {
        if (this.stopped || this.state === null || generation !== this.generation) {
            return;
        }
        this.closeStream();
        const streamGeneration = generation;
        const close = this.openEvents({
            lastEventId: this.state.lastEventId,
            onEvent: (message) => {
                if (!this.stopped && streamGeneration === this.generation) {
                    this.enqueue(message);
                }
            },
            onDisconnect: () => {
                if (!this.stopped && streamGeneration === this.generation) {
                    this.scheduleReconnect();
                }
            },
        });
        this.closeEvents = typeof close === "function" ? close : () => {};
        this.state = setConnection(this.state, "live");
        this.publish();
    }

    reconnect() {
        if (this.stopped || this.state === null) {
            return;
        }
        this.closeStream();
        this.state = setConnection(this.state, "reconnecting");
        this.publish();
        this.connect(this.generation);
    }

    scheduleReconnect() {
        if (this.stopped || this.state === null || this.reconnectTimer !== null) {
            return;
        }
        this.closeStream();
        this.state = setConnection(this.state, "reconnecting");
        this.publish();
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.reconnect();
        }, this.reconnectDelayMs);
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
                event = normalizeMessage(this.pending.shift());
            } catch (error) {
                this.pending.length = 0;
                this.state = requireResnapshot(this.state, "invalid_sse_message", {
                    message: String(error.message ?? error),
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

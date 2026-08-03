export const CONTROL_ACTIONS = Object.freeze([
    "request.cancel",
    "request.pause",
    "request.resume",
    "request.reprioritize",
    "cache.pin",
    "cache.unpin",
    "cache.evict",
    "capture.exclude",
    "server.drain",
]);

export function createControlIntent(action, targetId, parameters = {}, createdAt = new Date().toISOString()) {
    if (!CONTROL_ACTIONS.includes(action)) {
        throw new RangeError(`unsupported control action: ${action}`);
    }
    if (typeof targetId !== "string" || targetId.length === 0) {
        throw new TypeError("targetId must be a non-empty string");
    }
    if (parameters === null || typeof parameters !== "object" || Array.isArray(parameters)) {
        throw new TypeError("parameters must be an object");
    }
    return Object.freeze({
        action,
        targetId,
        parameters: Object.freeze({ ...parameters }),
        createdAt,
        transport: "local-preview-only",
    });
}

export class ControlIntentBuffer {
    constructor(limit = 20) {
        if (!Number.isSafeInteger(limit) || limit < 1) {
            throw new RangeError("intent buffer limit must be a positive integer");
        }
        this.limit = limit;
        this.intents = [];
    }

    add(intent) {
        if (intent?.transport !== "local-preview-only") {
            throw new TypeError("only local preview intents are accepted");
        }
        this.intents = [...this.intents, intent].slice(-this.limit);
        return this.snapshot();
    }

    snapshot() {
        return Object.freeze([...this.intents]);
    }
}

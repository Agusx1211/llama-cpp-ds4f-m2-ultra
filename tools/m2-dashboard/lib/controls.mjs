import { deepFreeze } from "./schema.mjs";

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

const MAX_CONTROL_PARAMETER_DEPTH = 12;
const MAX_CONTROL_PARAMETER_NODES = 256;
const MAX_CONTROL_PARAMETER_BYTES = 64 * 1024;
const MAX_CONTROL_IDENTIFIER_BYTES = 4096;
const MAX_CONTROL_TIMESTAMP_BYTES = 256;
const textEncoder = new TextEncoder();

function addParameterBytes(budget, value) {
    if (value.length > MAX_CONTROL_PARAMETER_BYTES - budget.bytes) {
        throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_BYTES} aggregate bytes`);
    }
    budget.bytes += textEncoder.encode(value).byteLength;
    if (budget.bytes > MAX_CONTROL_PARAMETER_BYTES) {
        throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_BYTES} aggregate bytes`);
    }
}

function cloneParameterValue(value, path, depth, budget, ancestors) {
    budget.nodes += 1;
    budget.bytes += 1;
    if (budget.nodes > MAX_CONTROL_PARAMETER_NODES) {
        throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_NODES} JSON nodes`);
    }
    if (budget.bytes > MAX_CONTROL_PARAMETER_BYTES) {
        throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_BYTES} aggregate bytes`);
    }
    if (depth > MAX_CONTROL_PARAMETER_DEPTH) {
        throw new RangeError(`parameters exceed JSON depth ${MAX_CONTROL_PARAMETER_DEPTH}`);
    }

    if (value === null || typeof value === "string" || typeof value === "boolean") {
        if (typeof value === "string") {
            addParameterBytes(budget, value);
        }
        return value;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
        addParameterBytes(budget, String(value));
        return value;
    }
    if (Array.isArray(value)) {
        if (value.length > MAX_CONTROL_PARAMETER_NODES - budget.nodes) {
            throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_NODES} JSON nodes`);
        }
        if (ancestors.has(value)) {
            throw new TypeError(`${path} must not contain cycles`);
        }
        ancestors.add(value);
        try {
            return value.map((item, index) => cloneParameterValue(
                item,
                `${path}[${index}]`,
                depth + 1,
                budget,
                ancestors,
            ));
        } finally {
            ancestors.delete(value);
        }
    }
    if (value !== null && typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
        if (ancestors.has(value)) {
            throw new TypeError(`${path} must not contain cycles`);
        }
        const keys = Reflect.ownKeys(value);
        if (keys.some((key) => typeof key !== "string")) {
            throw new TypeError(`${path} must not contain symbol properties`);
        }
        if (keys.length > MAX_CONTROL_PARAMETER_NODES - budget.nodes) {
            throw new RangeError(`parameters exceed ${MAX_CONTROL_PARAMETER_NODES} JSON nodes`);
        }
        ancestors.add(value);
        try {
            const clone = {};
            for (const key of keys) {
                const descriptor = Object.getOwnPropertyDescriptor(value, key);
                if (!descriptor.enumerable || descriptor.get || descriptor.set) {
                    throw new TypeError(`${path}.${key} must be an enumerable data property`);
                }
                addParameterBytes(budget, key);
                const clonedValue = cloneParameterValue(
                    descriptor.value,
                    `${path}.${key}`,
                    depth + 1,
                    budget,
                    ancestors,
                );
                Object.defineProperty(clone, key, {
                    value: clonedValue,
                    enumerable: true,
                    writable: true,
                    configurable: true,
                });
            }
            return clone;
        } finally {
            ancestors.delete(value);
        }
    }
    throw new TypeError(`${path} must contain only JSON values`);
}

function cloneParameters(parameters) {
    return cloneParameterValue(
        parameters,
        "parameters",
        0,
        { bytes: 0, nodes: 0 },
        new WeakSet(),
    );
}

export function createControlIntent(action, targetId, parameters = {}, createdAt = new Date().toISOString()) {
    if (!CONTROL_ACTIONS.includes(action)) {
        throw new RangeError(`unsupported control action: ${action}`);
    }
    if (typeof targetId !== "string" || targetId.length === 0) {
        throw new TypeError("targetId must be a non-empty string");
    }
    if (
        targetId.length > MAX_CONTROL_IDENTIFIER_BYTES
        || textEncoder.encode(targetId).byteLength > MAX_CONTROL_IDENTIFIER_BYTES
    ) {
        throw new RangeError(`targetId must not exceed ${MAX_CONTROL_IDENTIFIER_BYTES} bytes`);
    }
    if (parameters === null || typeof parameters !== "object" || Array.isArray(parameters)) {
        throw new TypeError("parameters must be an object");
    }
    if (typeof createdAt !== "string" || createdAt.length === 0) {
        throw new TypeError("createdAt must be a non-empty string");
    }
    if (
        createdAt.length > MAX_CONTROL_TIMESTAMP_BYTES
        || textEncoder.encode(createdAt).byteLength > MAX_CONTROL_TIMESTAMP_BYTES
    ) {
        throw new RangeError(`createdAt must not exceed ${MAX_CONTROL_TIMESTAMP_BYTES} bytes`);
    }
    return deepFreeze({
        action,
        targetId,
        parameters: cloneParameters(parameters),
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
        const storedIntent = createControlIntent(
            intent.action,
            intent.targetId,
            intent.parameters,
            intent.createdAt,
        );
        this.intents = [...this.intents, storedIntent].slice(-this.limit);
        return this.snapshot();
    }

    snapshot() {
        return Object.freeze([...this.intents]);
    }
}

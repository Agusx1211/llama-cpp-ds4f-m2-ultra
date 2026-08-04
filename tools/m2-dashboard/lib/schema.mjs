export const SCHEMA_VERSION = 1;
export const MAX_SCHEMA_ARRAY_LENGTH = 4096;
export const MAX_SCHEMA_STRING_LENGTH = 1024 * 1024;
export const MAX_SCHEMA_TOTAL_BYTES = 4 * 1024 * 1024;
export const MAX_SCHEMA_NODES = 65536;
export const MAX_SCHEMA_DEPTH = 32;

export const LANES = Object.freeze(["low", "normal", "fast"]);

export const EVENT_TYPES = Object.freeze([
    "request.upsert",
    "request.remove",
    "lane.replace",
    "allocator.replace",
    "cache.replace",
    "disk.replace",
    "dspark.replace",
    "capture.replace",
    "server.replace",
    "timeline.append",
    "stream.overflow",
    "stream.slow_client",
]);

const REQUEST_STATES = Object.freeze([
    "queued",
    "prefill",
    "decode",
    "parked",
    "spilling",
    "restoring",
    "complete",
    "cancelled",
    "failed",
]);

export class SchemaError extends Error {
    constructor(kind, issues) {
        super(`${kind} schema validation failed: ${issues.join("; ")}`);
        this.name = "SchemaError";
        this.kind = kind;
        this.issues = Object.freeze([...issues]);
    }
}

const textEncoder = new TextEncoder();

function jsonByteLength(value) {
    return textEncoder.encode(value).byteLength;
}

function validateJsonBudget(value, kind) {
    const seen = new WeakSet();
    const stack = [{ value, depth: 0 }];
    let bytes = 0;
    let nodes = 0;
    const addStringBytes = (text, structuralBytes = 0) => {
        if (text.length + structuralBytes > MAX_SCHEMA_TOTAL_BYTES - bytes) {
            throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_TOTAL_BYTES} aggregate bytes`]);
        }
        bytes += jsonByteLength(text) + structuralBytes;
        if (bytes > MAX_SCHEMA_TOTAL_BYTES) {
            throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_TOTAL_BYTES} aggregate bytes`]);
        }
    };

    while (stack.length > 0) {
        const current = stack.pop();
        nodes += 1;
        bytes += 1;
        if (nodes > MAX_SCHEMA_NODES) {
            throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_NODES} JSON nodes`]);
        }
        if (current.depth > MAX_SCHEMA_DEPTH) {
            throw new SchemaError(kind, [`payload exceeds JSON depth ${MAX_SCHEMA_DEPTH}`]);
        }

        const currentValue = current.value;
        if (typeof currentValue === "string") {
            addStringBytes(currentValue);
        } else if (typeof currentValue === "number") {
            if (!Number.isFinite(currentValue)) {
                throw new SchemaError(kind, ["payload must contain only finite JSON numbers"]);
            }
            bytes += String(currentValue).length;
        } else if (typeof currentValue === "boolean" || currentValue === null) {
            bytes += currentValue === null ? 4 : 5;
        } else if (typeof currentValue === "object") {
            if (seen.has(currentValue)) {
                throw new SchemaError(kind, ["payload must not contain cyclic or repeated object references"]);
            }
            seen.add(currentValue);
            if (Array.isArray(currentValue)) {
                if (currentValue.length > MAX_SCHEMA_NODES - nodes) {
                    throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_NODES} JSON nodes`]);
                }
                for (let index = currentValue.length - 1; index >= 0; index -= 1) {
                    stack.push({ value: currentValue[index], depth: current.depth + 1 });
                }
            } else {
                const prototype = Object.getPrototypeOf(currentValue);
                if (prototype !== Object.prototype && prototype !== null) {
                    throw new SchemaError(kind, ["payload must contain only plain JSON objects"]);
                }
                const keys = Reflect.ownKeys(currentValue);
                if (keys.some((key) => typeof key !== "string")) {
                    throw new SchemaError(kind, ["payload must not contain symbol properties"]);
                }
                if (keys.length > MAX_SCHEMA_NODES - nodes) {
                    throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_NODES} JSON nodes`]);
                }
                for (let index = keys.length - 1; index >= 0; index -= 1) {
                    const key = keys[index];
                    const descriptor = Object.getOwnPropertyDescriptor(currentValue, key);
                    if (!descriptor.enumerable || descriptor.get || descriptor.set) {
                        throw new SchemaError(kind, ["payload must contain only enumerable JSON data properties"]);
                    }
                    addStringBytes(key, 1);
                    stack.push({ value: descriptor.value, depth: current.depth + 1 });
                }
            }
        } else {
            throw new SchemaError(kind, ["payload must contain only JSON values"]);
        }

        if (bytes > MAX_SCHEMA_TOTAL_BYTES) {
            throw new SchemaError(kind, [`payload exceeds ${MAX_SCHEMA_TOTAL_BYTES} aggregate bytes`]);
        }
    }
}

function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
}

function record(value, path, issues) {
    if (!isRecord(value)) {
        issues.push(`${path} must be an object`);
        return {};
    }
    return value;
}

function knownFields(value, allowed, path, issues) {
    if (!isRecord(value)) {
        return;
    }
    for (const key of Object.keys(value)) {
        if (!allowed.includes(key)) {
            issues.push(`${path}.${key} is not allowed`);
        }
    }
}

function array(value, path, issues, { maxLength = MAX_SCHEMA_ARRAY_LENGTH } = {}) {
    if (!Array.isArray(value)) {
        issues.push(`${path} must be an array`);
        return [];
    }
    if (value.length > maxLength) {
        issues.push(`${path} must contain at most ${maxLength} items`);
        return value.slice(0, maxLength);
    }
    return value;
}

function string(value, path, issues, {
    allowEmpty = false,
    maxLength = MAX_SCHEMA_STRING_LENGTH,
} = {}) {
    if (typeof value !== "string" || (!allowEmpty && value.length === 0)) {
        issues.push(`${path} must be ${allowEmpty ? "a string" : "a non-empty string"}`);
    } else if (value.length > maxLength) {
        issues.push(`${path} must contain at most ${maxLength} characters`);
    }
}

function nullableString(value, path, issues) {
    if (value !== null) {
        string(value, path, issues);
    }
}

function finiteNumber(value, path, issues, { min = -Infinity } = {}) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < min) {
        issues.push(`${path} must be a finite number >= ${min}`);
    }
}

function nullableNumber(value, path, issues, options) {
    if (value !== null) {
        finiteNumber(value, path, issues, options);
    }
}

function integer(value, path, issues, { min = Number.MIN_SAFE_INTEGER } = {}) {
    if (!Number.isSafeInteger(value) || value < min) {
        issues.push(`${path} must be an integer >= ${min}`);
    }
}

function nullableInteger(value, path, issues, options) {
    if (value !== null) {
        integer(value, path, issues, options);
    }
}

function boolean(value, path, issues) {
    if (typeof value !== "boolean") {
        issues.push(`${path} must be boolean`);
    }
}

function enumeration(value, choices, path, issues) {
    if (!choices.includes(value)) {
        issues.push(`${path} must be one of ${choices.join(", ")}`);
    }
}

function range(value, path, issues) {
    const values = array(value, path, issues);
    if (values.length !== 2) {
        issues.push(`${path} must contain [minimum, maximum]`);
        return;
    }
    finiteNumber(values[0], `${path}[0]`, issues, { min: 0 });
    finiteNumber(values[1], `${path}[1]`, issues, { min: 0 });
    if (Number.isFinite(values[0]) && Number.isFinite(values[1]) && values[0] > values[1]) {
        issues.push(`${path}[0] must not exceed ${path}[1]`);
    }
}

function validateServer(value, path, issues) {
    const server = record(value, path, issues);
    knownFields(server, [
        "health", "build", "model", "uptime_seconds", "rss_bytes", "footprint_bytes",
        "swap_bytes", "memory_pressure_percent", "decode_width", "aggregate_tokens_per_second",
    ], path, issues);
    enumeration(server.health, ["healthy", "degraded", "unavailable"], `${path}.health`, issues);
    string(server.build, `${path}.build`, issues);
    string(server.model, `${path}.model`, issues);
    finiteNumber(server.uptime_seconds, `${path}.uptime_seconds`, issues, { min: 0 });
    finiteNumber(server.rss_bytes, `${path}.rss_bytes`, issues, { min: 0 });
    finiteNumber(server.footprint_bytes, `${path}.footprint_bytes`, issues, { min: 0 });
    finiteNumber(server.swap_bytes, `${path}.swap_bytes`, issues, { min: 0 });
    finiteNumber(server.memory_pressure_percent, `${path}.memory_pressure_percent`, issues, { min: 0 });
    integer(server.decode_width, `${path}.decode_width`, issues, { min: 0 });
    finiteNumber(server.aggregate_tokens_per_second, `${path}.aggregate_tokens_per_second`, issues, { min: 0 });
}

function validateLane(value, path, issues) {
    const lane = record(value, path, issues);
    knownFields(lane, [
        "id", "queued", "active", "oldest_wait_ms", "service_deficit", "bypass_count",
        "predicted_start_ms",
        "claimed_permits", "bound_permits",
    ], path, issues);
    enumeration(lane.id, LANES, `${path}.id`, issues);
    integer(lane.queued, `${path}.queued`, issues, { min: 0 });
    integer(lane.active, `${path}.active`, issues, { min: 0 });
    finiteNumber(lane.oldest_wait_ms, `${path}.oldest_wait_ms`, issues, { min: 0 });
    finiteNumber(lane.service_deficit, `${path}.service_deficit`, issues, { min: 0 });
    integer(lane.bypass_count, `${path}.bypass_count`, issues, { min: 0 });
    range(lane.predicted_start_ms, `${path}.predicted_start_ms`, issues);
    if (Object.hasOwn(lane, "claimed_permits")) {
        integer(lane.claimed_permits, `${path}.claimed_permits`, issues, { min: 0 });
    }
    if (Object.hasOwn(lane, "bound_permits")) {
        integer(lane.bound_permits, `${path}.bound_permits`, issues, { min: 0 });
    }
}

function validateLanes(value, path, issues) {
    const lanes = array(value, path, issues, { maxLength: LANES.length });
    const laneIds = new Set();
    for (const [index, lane] of lanes.entries()) {
        validateLane(lane, `${path}[${index}]`, issues);
        if (isRecord(lane) && typeof lane.id === "string") {
            if (laneIds.has(lane.id)) {
                issues.push(`${path} contains duplicate lane ${lane.id}`);
            }
            laneIds.add(lane.id);
        }
    }
    for (const lane of LANES) {
        if (!laneIds.has(lane)) {
            issues.push(`${path} is missing ${lane}`);
        }
    }
    return lanes;
}

const AVAILABILITY_FIELDS = Object.freeze([
    "server_metrics",
    "scheduler_predictions",
    "request_latency",
    "request_kv",
    "request_preemption",
    "content",
    "allocator",
    "cache",
    "disks",
    "dspark",
    "capture",
]);

function validateAvailability(value, path, issues) {
    const availability = record(value, path, issues);
    knownFields(availability, AVAILABILITY_FIELDS, path, issues);
    for (const field of AVAILABILITY_FIELDS) {
        boolean(availability[field], `${path}.${field}`, issues);
    }
}

function validateLanePermitCounts(value, path, issues) {
    const counts = array(value, path, issues, { maxLength: LANES.length });
    if (counts.length !== LANES.length) {
        issues.push(`${path} must contain one count per lane`);
    }
    for (const [index, count] of counts.entries()) {
        integer(count, `${path}[${index}]`, issues, { min: 0 });
    }
}

function validateRegistry(value, path, issues) {
    const registry = record(value, path, issues);
    knownFields(registry, [
        "active_requests", "occupied_slots", "retained_events", "event_capacity",
        "total_events", "dropped_events", "claimed_permits", "bound_permits", "total_permits",
    ], path, issues);
    for (const field of [
        "active_requests", "occupied_slots", "retained_events", "event_capacity",
        "total_events", "dropped_events", "total_permits",
    ]) {
        integer(registry[field], `${path}.${field}`, issues, { min: 0 });
    }
    validateLanePermitCounts(registry.claimed_permits, `${path}.claimed_permits`, issues);
    validateLanePermitCounts(registry.bound_permits, `${path}.bound_permits`, issues);
}

function validateRequest(value, path, issues) {
    const request = record(value, path, issues);
    knownFields(request, [
        "id", "lane", "state", "arrival_at", "age_ms", "prompt_tokens", "cache_hit_tokens",
        "output_tokens", "requested_output_tokens", "blocker", "predicted_start_ms", "ttft_ms",
        "tbt_ms", "kv", "scheduler_reasons", "preemptions", "dspark_cycles", "content",
    ], path, issues);
    string(request.id, `${path}.id`, issues);
    enumeration(request.lane, LANES, `${path}.lane`, issues);
    enumeration(request.state, REQUEST_STATES, `${path}.state`, issues);
    string(request.arrival_at, `${path}.arrival_at`, issues);
    finiteNumber(request.age_ms, `${path}.age_ms`, issues, { min: 0 });
    integer(request.prompt_tokens, `${path}.prompt_tokens`, issues, { min: 0 });
    integer(request.cache_hit_tokens, `${path}.cache_hit_tokens`, issues, { min: 0 });
    integer(request.output_tokens, `${path}.output_tokens`, issues, { min: 0 });
    nullableInteger(request.requested_output_tokens, `${path}.requested_output_tokens`, issues, { min: 0 });
    string(request.blocker, `${path}.blocker`, issues, { allowEmpty: true });
    range(request.predicted_start_ms, `${path}.predicted_start_ms`, issues);
    nullableNumber(request.ttft_ms, `${path}.ttft_ms`, issues, { min: 0 });
    nullableNumber(request.tbt_ms, `${path}.tbt_ms`, issues, { min: 0 });

    const kv = record(request.kv, `${path}.kv`, issues);
    knownFields(kv, ["logical_bytes", "unique_bytes", "lineage"], `${path}.kv`, issues);
    finiteNumber(kv.logical_bytes, `${path}.kv.logical_bytes`, issues, { min: 0 });
    finiteNumber(kv.unique_bytes, `${path}.kv.unique_bytes`, issues, { min: 0 });
    string(kv.lineage, `${path}.kv.lineage`, issues);

    for (const [index, reason] of array(request.scheduler_reasons, `${path}.scheduler_reasons`, issues).entries()) {
        string(reason, `${path}.scheduler_reasons[${index}]`, issues);
    }
    integer(request.preemptions, `${path}.preemptions`, issues, { min: 0 });
    integer(request.dspark_cycles, `${path}.dspark_cycles`, issues, { min: 0 });

    const content = record(request.content, `${path}.content`, issues);
    knownFields(content, ["prompt", "output", "retained"], `${path}.content`, issues);
    string(content.prompt, `${path}.content.prompt`, issues, { allowEmpty: true });
    string(content.output, `${path}.content.output`, issues, { allowEmpty: true });
    boolean(content.retained, `${path}.content.retained`, issues);
}

function validatePool(value, path, issues) {
    const pool = record(value, path, issues);
    knownFields(pool, [
        "id", "capacity_pages", "free_pages", "reserved_pages", "mapped_pages", "shared_pages",
        "cow_pages", "high_water_pages",
    ], path, issues);
    string(pool.id, `${path}.id`, issues);
    for (const field of [
        "capacity_pages",
        "free_pages",
        "reserved_pages",
        "mapped_pages",
        "shared_pages",
        "cow_pages",
        "high_water_pages",
    ]) {
        integer(pool[field], `${path}.${field}`, issues, { min: 0 });
    }
}

function validateAllocator(value, path, issues) {
    const allocator = record(value, path, issues);
    knownFields(allocator, ["pools"], path, issues);
    for (const [index, pool] of array(allocator.pools, `${path}.pools`, issues).entries()) {
        validatePool(pool, `${path}.pools[${index}]`, issues);
    }
}

function validateCacheObject(value, path, issues) {
    const object = record(value, path, issues);
    knownFields(object, [
        "id", "tier", "kind", "logical_bytes", "unique_bytes", "shared_bytes", "hits", "score",
        "pinned",
    ], path, issues);
    string(object.id, `${path}.id`, issues);
    enumeration(object.tier, ["resident", "disk"], `${path}.tier`, issues);
    enumeration(object.kind, ["live", "session", "prefix"], `${path}.kind`, issues);
    for (const field of ["logical_bytes", "unique_bytes", "shared_bytes", "hits", "score"]) {
        finiteNumber(object[field], `${path}.${field}`, issues, { min: 0 });
    }
    if (
        Number.isFinite(object.logical_bytes)
        && Number.isFinite(object.unique_bytes)
        && Number.isFinite(object.shared_bytes)
        && object.logical_bytes !== object.unique_bytes + object.shared_bytes
    ) {
        issues.push(`${path}.logical_bytes must equal unique_bytes + shared_bytes`);
    }
    boolean(object.pinned, `${path}.pinned`, issues);
}

function validateCache(value, path, issues) {
    const cache = record(value, path, issues);
    knownFields(cache, ["objects"], path, issues);
    for (const [index, object] of array(cache.objects, `${path}.objects`, issues).entries()) {
        validateCacheObject(object, `${path}.objects[${index}]`, issues);
    }
}

function validateDisk(value, path, issues) {
    const disk = record(value, path, issues);
    knownFields(disk, [
        "id", "path", "capacity_bytes", "free_bytes", "queue_depth", "read_bps", "write_bps",
        "snapshots", "errors", "healthy",
    ], path, issues);
    string(disk.id, `${path}.id`, issues);
    string(disk.path, `${path}.path`, issues);
    for (const field of ["capacity_bytes", "free_bytes", "queue_depth", "read_bps", "write_bps", "snapshots", "errors"]) {
        finiteNumber(disk[field], `${path}.${field}`, issues, { min: 0 });
    }
    boolean(disk.healthy, `${path}.healthy`, issues);
}

function validateDspark(value, path, issues) {
    const dspark = record(value, path, issues);
    knownFields(dspark, [
        "mode", "scheduled_decode_width", "proposals", "accepted", "acceptance_by_position",
    ], path, issues);
    string(dspark.mode, `${path}.mode`, issues);
    integer(dspark.scheduled_decode_width, `${path}.scheduled_decode_width`, issues, { min: 0 });
    integer(dspark.proposals, `${path}.proposals`, issues, { min: 0 });
    integer(dspark.accepted, `${path}.accepted`, issues, { min: 0 });
    for (const [index, rate] of array(dspark.acceptance_by_position, `${path}.acceptance_by_position`, issues).entries()) {
        finiteNumber(rate, `${path}.acceptance_by_position[${index}]`, issues, { min: 0 });
    }
}

function validateCapture(value, path, issues) {
    const capture = record(value, path, issues);
    knownFields(capture, [
        "mode", "healthy", "queued_records", "written_records", "dropped_records", "bytes_written",
        "last_error",
    ], path, issues);
    string(capture.mode, `${path}.mode`, issues);
    boolean(capture.healthy, `${path}.healthy`, issues);
    for (const field of ["queued_records", "written_records", "dropped_records", "bytes_written"]) {
        finiteNumber(capture[field], `${path}.${field}`, issues, { min: 0 });
    }
    nullableString(capture.last_error, `${path}.last_error`, issues);
}

function validateTimelineItem(value, path, issues) {
    const item = record(value, path, issues);
    knownFields(item, ["id", "at", "type", "label", "request_id", "lane"], path, issues);
    string(item.id, `${path}.id`, issues);
    string(item.at, `${path}.at`, issues);
    string(item.type, `${path}.type`, issues);
    string(item.label, `${path}.label`, issues);
    nullableString(item.request_id, `${path}.request_id`, issues);
    if (item.lane !== null) {
        enumeration(item.lane, LANES, `${path}.lane`, issues);
    }
}

function validateSnapshotInto(snapshotValue, path, issues) {
    const snapshot = record(snapshotValue, path, issues);
    knownFields(snapshot, [
        "schema_version", "sequence", "generated_at", "availability", "registry", "server", "lanes",
        "requests", "allocator", "cache", "disks", "dspark", "capture", "timeline",
    ], path, issues);
    integer(snapshot.schema_version, `${path}.schema_version`, issues, { min: 1 });
    if (snapshot.schema_version !== SCHEMA_VERSION) {
        issues.push(`${path}.schema_version must equal ${SCHEMA_VERSION}`);
    }
    integer(snapshot.sequence, `${path}.sequence`, issues, { min: 0 });
    string(snapshot.generated_at, `${path}.generated_at`, issues);
    validateAvailability(snapshot.availability, `${path}.availability`, issues);
    validateRegistry(snapshot.registry, `${path}.registry`, issues);
    validateServer(snapshot.server, `${path}.server`, issues);

    const lanes = validateLanes(snapshot.lanes, `${path}.lanes`, issues);

    const requests = array(snapshot.requests, `${path}.requests`, issues);
    const requestIds = new Set();
    for (const [index, request] of requests.entries()) {
        validateRequest(request, `${path}.requests[${index}]`, issues);
        if (isRecord(request) && typeof request.id === "string") {
            if (requestIds.has(request.id)) {
                issues.push(`${path}.requests contains duplicate request ${request.id}`);
            }
            requestIds.add(request.id);
        }
    }

    for (const laneId of LANES) {
        const lane = lanes.find((candidate) => isRecord(candidate) && candidate.id === laneId);
        if (!lane) {
            continue;
        }
        const laneRequests = requests.filter((request) => isRecord(request) && request.lane === laneId);
        const queued = laneRequests.filter((request) => request.state === "queued").length;
        const active = laneRequests.filter((request) => ![
            "queued",
            "complete",
            "cancelled",
            "failed",
        ].includes(request.state)).length;
        if (Number.isSafeInteger(lane.queued) && lane.queued !== queued) {
            issues.push(`${path}.lanes.${laneId}.queued must equal request aggregate ${queued}`);
        }
        if (Number.isSafeInteger(lane.active) && lane.active !== active) {
            issues.push(`${path}.lanes.${laneId}.active must equal request aggregate ${active}`);
        }
    }

    validateAllocator(snapshot.allocator, `${path}.allocator`, issues);
    validateCache(snapshot.cache, `${path}.cache`, issues);
    for (const [index, disk] of array(snapshot.disks, `${path}.disks`, issues).entries()) {
        validateDisk(disk, `${path}.disks[${index}]`, issues);
    }
    validateDspark(snapshot.dspark, `${path}.dspark`, issues);
    validateCapture(snapshot.capture, `${path}.capture`, issues);
    for (const [index, item] of array(snapshot.timeline, `${path}.timeline`, issues).entries()) {
        validateTimelineItem(item, `${path}.timeline[${index}]`, issues);
    }
}

export function validateSnapshot(value) {
    validateJsonBudget(value, "snapshot");
    const issues = [];
    validateSnapshotInto(value, "snapshot", issues);
    if (issues.length > 0) {
        throw new SchemaError("snapshot", issues);
    }
    return value;
}

function validateEventPayload(event, issues) {
    const payload = record(event.payload, "event.payload", issues);
    switch (event.type) {
        case "request.upsert":
            knownFields(payload, ["request", "lanes", "registry"], "event.payload", issues);
            validateRequest(payload.request, "event.payload.request", issues);
            if (Object.hasOwn(payload, "lanes")) {
                validateLanes(payload.lanes, "event.payload.lanes", issues);
            }
            if (Object.hasOwn(payload, "registry")) {
                validateRegistry(payload.registry, "event.payload.registry", issues);
            }
            if (isRecord(payload.request)) {
                if (event.request_id !== payload.request.id) {
                    issues.push("event.request_id must equal event.payload.request.id");
                }
                if (event.lane !== payload.request.lane) {
                    issues.push("event.lane must equal event.payload.request.lane");
                }
            }
            break;
        case "request.remove":
            knownFields(payload, ["request_id", "lanes", "registry"], "event.payload", issues);
            string(payload.request_id, "event.payload.request_id", issues);
            if (Object.hasOwn(payload, "lanes")) {
                validateLanes(payload.lanes, "event.payload.lanes", issues);
            }
            if (Object.hasOwn(payload, "registry")) {
                validateRegistry(payload.registry, "event.payload.registry", issues);
            }
            if (event.request_id !== payload.request_id) {
                issues.push("event.request_id must equal event.payload.request_id");
            }
            break;
        case "lane.replace":
            knownFields(payload, ["lane"], "event.payload", issues);
            validateLane(payload.lane, "event.payload.lane", issues);
            if (isRecord(payload.lane) && event.lane !== payload.lane.id) {
                issues.push("event.lane must equal event.payload.lane.id");
            }
            break;
        case "allocator.replace":
            knownFields(payload, ["allocator"], "event.payload", issues);
            validateAllocator(payload.allocator, "event.payload.allocator", issues);
            break;
        case "cache.replace":
            knownFields(payload, ["cache"], "event.payload", issues);
            validateCache(payload.cache, "event.payload.cache", issues);
            break;
        case "disk.replace":
            knownFields(payload, ["disks"], "event.payload", issues);
            for (const [index, disk] of array(payload.disks, "event.payload.disks", issues).entries()) {
                validateDisk(disk, `event.payload.disks[${index}]`, issues);
            }
            break;
        case "dspark.replace":
            knownFields(payload, ["dspark"], "event.payload", issues);
            validateDspark(payload.dspark, "event.payload.dspark", issues);
            break;
        case "capture.replace":
            knownFields(payload, ["capture"], "event.payload", issues);
            validateCapture(payload.capture, "event.payload.capture", issues);
            break;
        case "server.replace":
            knownFields(payload, ["server"], "event.payload", issues);
            validateServer(payload.server, "event.payload.server", issues);
            break;
        case "timeline.append":
            knownFields(payload, ["item"], "event.payload", issues);
            validateTimelineItem(payload.item, "event.payload.item", issues);
            if (isRecord(payload.item)) {
                if (event.request_id !== payload.item.request_id) {
                    issues.push("event.request_id must equal event.payload.item.request_id");
                }
                if (event.lane !== payload.item.lane) {
                    issues.push("event.lane must equal event.payload.item.lane");
                }
            }
            break;
        case "stream.overflow":
            knownFields(payload, ["oldest_available_sequence"], "event.payload", issues);
            integer(payload.oldest_available_sequence, "event.payload.oldest_available_sequence", issues, { min: 0 });
            break;
        case "stream.slow_client":
            knownFields(payload, ["dropped_events"], "event.payload", issues);
            integer(payload.dropped_events, "event.payload.dropped_events", issues, { min: 1 });
            break;
        default:
            break;
    }
}

export function validateEvent(value) {
    validateJsonBudget(value, "event");
    const issues = [];
    const event = record(value, "event", issues);
    knownFields(event, [
        "schema_version", "id", "sequence", "wall_time", "monotonic_ms", "type", "request_id",
        "slot_id", "sequence_id", "lane", "scheduler_epoch", "decision_id", "before", "after",
        "reason_code", "payload",
    ], "event", issues);
    integer(event.schema_version, "event.schema_version", issues, { min: 1 });
    if (event.schema_version !== SCHEMA_VERSION) {
        issues.push(`event.schema_version must equal ${SCHEMA_VERSION}`);
    }
    string(event.id, "event.id", issues);
    integer(event.sequence, "event.sequence", issues, { min: 1 });
    if (typeof event.id === "string" && Number.isSafeInteger(event.sequence) && event.id !== String(event.sequence)) {
        issues.push("event.id must be the decimal event.sequence");
    }
    string(event.wall_time, "event.wall_time", issues);
    finiteNumber(event.monotonic_ms, "event.monotonic_ms", issues, { min: 0 });
    enumeration(event.type, EVENT_TYPES, "event.type", issues);
    nullableString(event.request_id, "event.request_id", issues);
    nullableInteger(event.slot_id, "event.slot_id", issues, { min: 0 });
    nullableInteger(event.sequence_id, "event.sequence_id", issues, { min: 0 });
    if (event.lane !== null) {
        enumeration(event.lane, LANES, "event.lane", issues);
    }
    nullableInteger(event.scheduler_epoch, "event.scheduler_epoch", issues, { min: 0 });
    nullableString(event.decision_id, "event.decision_id", issues);
    nullableString(event.before, "event.before", issues);
    nullableString(event.after, "event.after", issues);
    string(event.reason_code, "event.reason_code", issues);
    validateEventPayload(event, issues);

    if (issues.length > 0) {
        throw new SchemaError("event", issues);
    }
    return value;
}

export function cloneJson(value) {
    if (typeof structuredClone === "function") {
        return structuredClone(value);
    }
    return JSON.parse(JSON.stringify(value));
}

export function deepFreeze(value) {
    if (value === null || typeof value !== "object" || Object.isFrozen(value)) {
        return value;
    }
    for (const child of Object.values(value)) {
        deepFreeze(child);
    }
    return Object.freeze(value);
}

export function parseSnapshot(value) {
    validateSnapshot(value);
    const snapshot = cloneJson(value);
    return deepFreeze(snapshot);
}

export function parseEvent(value) {
    validateEvent(value);
    const event = cloneJson(value);
    return deepFreeze(event);
}

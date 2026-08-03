export const SCHEMA_VERSION = 1;

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

function array(value, path, issues) {
    if (!Array.isArray(value)) {
        issues.push(`${path} must be an array`);
        return [];
    }
    return value;
}

function string(value, path, issues, { allowEmpty = false } = {}) {
    if (typeof value !== "string" || (!allowEmpty && value.length === 0)) {
        issues.push(`${path} must be ${allowEmpty ? "a string" : "a non-empty string"}`);
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
    enumeration(lane.id, LANES, `${path}.id`, issues);
    integer(lane.queued, `${path}.queued`, issues, { min: 0 });
    integer(lane.active, `${path}.active`, issues, { min: 0 });
    finiteNumber(lane.oldest_wait_ms, `${path}.oldest_wait_ms`, issues, { min: 0 });
    finiteNumber(lane.service_deficit, `${path}.service_deficit`, issues, { min: 0 });
    integer(lane.bypass_count, `${path}.bypass_count`, issues, { min: 0 });
    range(lane.predicted_start_ms, `${path}.predicted_start_ms`, issues);
}

function validateRequest(value, path, issues) {
    const request = record(value, path, issues);
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
    finiteNumber(kv.logical_bytes, `${path}.kv.logical_bytes`, issues, { min: 0 });
    finiteNumber(kv.unique_bytes, `${path}.kv.unique_bytes`, issues, { min: 0 });
    string(kv.lineage, `${path}.kv.lineage`, issues);

    for (const [index, reason] of array(request.scheduler_reasons, `${path}.scheduler_reasons`, issues).entries()) {
        string(reason, `${path}.scheduler_reasons[${index}]`, issues);
    }
    integer(request.preemptions, `${path}.preemptions`, issues, { min: 0 });
    integer(request.dspark_cycles, `${path}.dspark_cycles`, issues, { min: 0 });

    const content = record(request.content, `${path}.content`, issues);
    string(content.prompt, `${path}.content.prompt`, issues, { allowEmpty: true });
    string(content.output, `${path}.content.output`, issues, { allowEmpty: true });
    boolean(content.retained, `${path}.content.retained`, issues);
}

function validatePool(value, path, issues) {
    const pool = record(value, path, issues);
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
    for (const [index, pool] of array(allocator.pools, `${path}.pools`, issues).entries()) {
        validatePool(pool, `${path}.pools[${index}]`, issues);
    }
}

function validateCacheObject(value, path, issues) {
    const object = record(value, path, issues);
    string(object.id, `${path}.id`, issues);
    enumeration(object.tier, ["resident", "disk"], `${path}.tier`, issues);
    enumeration(object.kind, ["live", "session", "prefix"], `${path}.kind`, issues);
    for (const field of ["logical_bytes", "unique_bytes", "shared_bytes", "hits", "score"]) {
        finiteNumber(object[field], `${path}.${field}`, issues, { min: 0 });
    }
    boolean(object.pinned, `${path}.pinned`, issues);
}

function validateCache(value, path, issues) {
    const cache = record(value, path, issues);
    for (const [index, object] of array(cache.objects, `${path}.objects`, issues).entries()) {
        validateCacheObject(object, `${path}.objects[${index}]`, issues);
    }
}

function validateDisk(value, path, issues) {
    const disk = record(value, path, issues);
    string(disk.id, `${path}.id`, issues);
    string(disk.path, `${path}.path`, issues);
    for (const field of ["capacity_bytes", "free_bytes", "queue_depth", "read_bps", "write_bps", "snapshots", "errors"]) {
        finiteNumber(disk[field], `${path}.${field}`, issues, { min: 0 });
    }
    boolean(disk.healthy, `${path}.healthy`, issues);
}

function validateDspark(value, path, issues) {
    const dspark = record(value, path, issues);
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
    string(capture.mode, `${path}.mode`, issues);
    boolean(capture.healthy, `${path}.healthy`, issues);
    for (const field of ["queued_records", "written_records", "dropped_records", "bytes_written"]) {
        finiteNumber(capture[field], `${path}.${field}`, issues, { min: 0 });
    }
    nullableString(capture.last_error, `${path}.last_error`, issues);
}

function validateTimelineItem(value, path, issues) {
    const item = record(value, path, issues);
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
    integer(snapshot.schema_version, `${path}.schema_version`, issues, { min: 1 });
    if (snapshot.schema_version !== SCHEMA_VERSION) {
        issues.push(`${path}.schema_version must equal ${SCHEMA_VERSION}`);
    }
    integer(snapshot.sequence, `${path}.sequence`, issues, { min: 0 });
    string(snapshot.generated_at, `${path}.generated_at`, issues);
    validateServer(snapshot.server, `${path}.server`, issues);

    const lanes = array(snapshot.lanes, `${path}.lanes`, issues);
    const laneIds = new Set();
    for (const [index, lane] of lanes.entries()) {
        validateLane(lane, `${path}.lanes[${index}]`, issues);
        if (isRecord(lane) && typeof lane.id === "string") {
            if (laneIds.has(lane.id)) {
                issues.push(`${path}.lanes contains duplicate lane ${lane.id}`);
            }
            laneIds.add(lane.id);
        }
    }
    for (const lane of LANES) {
        if (!laneIds.has(lane)) {
            issues.push(`${path}.lanes is missing ${lane}`);
        }
    }

    const requestIds = new Set();
    for (const [index, request] of array(snapshot.requests, `${path}.requests`, issues).entries()) {
        validateRequest(request, `${path}.requests[${index}]`, issues);
        if (isRecord(request) && typeof request.id === "string") {
            if (requestIds.has(request.id)) {
                issues.push(`${path}.requests contains duplicate request ${request.id}`);
            }
            requestIds.add(request.id);
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
            validateRequest(payload.request, "event.payload.request", issues);
            break;
        case "request.remove":
            string(payload.request_id, "event.payload.request_id", issues);
            break;
        case "lane.replace":
            validateLane(payload.lane, "event.payload.lane", issues);
            break;
        case "allocator.replace":
            validateAllocator(payload.allocator, "event.payload.allocator", issues);
            break;
        case "cache.replace":
            validateCache(payload.cache, "event.payload.cache", issues);
            break;
        case "disk.replace":
            for (const [index, disk] of array(payload.disks, "event.payload.disks", issues).entries()) {
                validateDisk(disk, `event.payload.disks[${index}]`, issues);
            }
            break;
        case "dspark.replace":
            validateDspark(payload.dspark, "event.payload.dspark", issues);
            break;
        case "capture.replace":
            validateCapture(payload.capture, "event.payload.capture", issues);
            break;
        case "server.replace":
            validateServer(payload.server, "event.payload.server", issues);
            break;
        case "timeline.append":
            validateTimelineItem(payload.item, "event.payload.item", issues);
            break;
        case "stream.overflow":
            integer(payload.oldest_available_sequence, "event.payload.oldest_available_sequence", issues, { min: 0 });
            break;
        case "stream.slow_client":
            integer(payload.dropped_events, "event.payload.dropped_events", issues, { min: 1 });
            break;
        default:
            break;
    }
}

export function validateEvent(value) {
    const issues = [];
    const event = record(value, "event", issues);
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
    const snapshot = cloneJson(value);
    validateSnapshot(snapshot);
    return deepFreeze(snapshot);
}

export function parseEvent(value) {
    const event = cloneJson(value);
    validateEvent(event);
    return deepFreeze(event);
}

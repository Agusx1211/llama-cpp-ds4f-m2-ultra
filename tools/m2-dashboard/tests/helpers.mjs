import { readFile } from "node:fs/promises";

const fixtureRoot = new URL("../fixtures/", import.meta.url);

export async function loadSnapshot() {
    return JSON.parse(await readFile(new URL("state.json", fixtureRoot), "utf8"));
}

export async function loadEvents() {
    return JSON.parse(await readFile(new URL("events.json", fixtureRoot), "utf8"));
}

export function clone(value) {
    return structuredClone(value);
}

export function snapshotAt(snapshot, sequence) {
    const result = clone(snapshot);
    result.sequence = sequence;
    result.generated_at = `2026-08-03T22:00:${String(sequence % 60).padStart(2, "0")}.000Z`;
    return result;
}

export function streamSignal(template, sequence, type, payload) {
    const result = clone(template);
    result.id = String(sequence);
    result.sequence = sequence;
    result.type = type;
    result.request_id = null;
    result.slot_id = null;
    result.sequence_id = null;
    result.lane = null;
    result.before = null;
    result.after = null;
    result.reason_code = type.replace(".", "_");
    result.payload = payload;
    return result;
}

#!/usr/bin/env python3

"""Dependency-free target probe for the authenticated live-dashboard wire contract."""

from __future__ import annotations

import json
import math
import os
import select
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_KEY_ENV = "LLAMA_API_KEY"
OPERATOR_TOKEN_ENV = "LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN"
OPERATOR_TOKEN_HEADER = "X-Llama-Trusted-Scheduling-Token"
SNAPSHOT_PATH = "/internal/admin/dashboard/snapshot"
EVENTS_PATH = "/internal/admin/dashboard/events"
SCHEMA_VERSION = 2
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024
MAX_SSE_EVENT_BYTES = 64 * 1024
MAX_CAPTURED_EVENTS = 4096
MAX_CONTROL_BYTES = 16
SUPPORTED_REFILL_STATES = {
    "disabled", "inactive", "window_open", "one_member_eligible",
    "quota_exhausted", "deadline_expired", "full_width",
}


class ProbeError(RuntimeError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        del request, file_pointer, code, message, headers, new_url
        return None


OPENER = urllib.request.build_opener(NoRedirect)


def required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ProbeError(f"{name} is required in the environment")
    return value


def credential(value: str, name: str, minimum: int, maximum: int) -> str:
    size = len(value.encode())
    if size < minimum or size > maximum or any(character in value for character in "\0\r\n"):
        raise ProbeError(f"{name} must contain {minimum}..{maximum} safe header bytes")
    return value


def bounded_environment_integer(name: str, fallback: int, minimum: int, maximum: int) -> int:
    text = os.environ.get(name, "")
    if not text:
        return fallback
    try:
        value = int(text)
    except ValueError as error:
        raise ProbeError(f"{name} must be an integer from {minimum} through {maximum}") from error
    if str(value) != text.strip() or value < minimum or value > maximum:
        raise ProbeError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def normalize_loopback_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise ProbeError("server URL must be HTTP(S) without embedded credentials")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ProbeError("server URL must use an explicit loopback hostname")
    if parsed.query or parsed.fragment:
        raise ProbeError("server URL cannot contain a query or fragment")
    return value.rstrip("/")


def credential_headers(api_key: str | None, operator_token: str | None, **extra: str) -> dict[str, str]:
    headers = {"Accept": "application/json", **extra}
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"
    if operator_token is not None:
        headers[OPERATOR_TOKEN_HEADER] = operator_token
    return headers


def open_request(url: str, headers: dict[str, str], timeout_seconds: float):
    request = urllib.request.Request(url, headers=headers, method="GET")
    return OPENER.open(request, timeout=timeout_seconds)


def fetch_status(url: str, headers: dict[str, str], timeout_seconds: float) -> int:
    try:
        response = open_request(url, headers, timeout_seconds)
    except urllib.error.HTTPError as error:
        error.close()
        return error.code
    try:
        return response.status
    finally:
        response.close()


def require_denied(status: int, label: str) -> int:
    if status < 400 or status > 499:
        raise ProbeError(f"{label} returned HTTP {status}, expected a 4xx denial")
    return status


def parse_json_bytes(data: bytes, kind: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ProbeError(f"{kind} contains non-finite JSON number {value}")

    try:
        return json.loads(data, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(f"{kind} is not valid UTF-8 JSON") from error


def fetch_snapshot(url: str, headers: dict[str, str], timeout_seconds: float) -> dict[str, Any]:
    response = open_request(url, headers, timeout_seconds)
    try:
        if response.status != 200:
            raise ProbeError(f"snapshot request returned HTTP {response.status}")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_SNAPSHOT_BYTES:
            raise ProbeError(f"snapshot response exceeds {MAX_SNAPSHOT_BYTES} bytes")
        data = response.read(MAX_SNAPSHOT_BYTES + 1)
        if len(data) > MAX_SNAPSHOT_BYTES:
            raise ProbeError(f"snapshot response exceeds {MAX_SNAPSHOT_BYTES} bytes")
        return require_record(parse_json_bytes(data, "snapshot"), "snapshot")
    finally:
        response.close()


def require_record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeError(f"{path} must be an object")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProbeError(
            f"{path} fields differ: missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}")


def require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProbeError(f"{path} must be boolean")
    return value


def require_integer(value: Any, path: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > MAX_SAFE_INTEGER:
        raise ProbeError(f"{path} must be a safe integer >= {minimum}")
    return value


def require_number(value: Any, path: str, minimum: float = 0) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < minimum):
        raise ProbeError(f"{path} must be a finite number >= {minimum}")
    return float(value)


def require_string(value: Any, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value) or len(value) > 1024 * 1024:
        raise ProbeError(f"{path} must be a bounded{' non-empty' if not allow_empty else ''} string")
    return value


def validate_fast_refill(value: Any, path: str = "fast_refill") -> dict[str, Any]:
    fast_refill = require_record(value, path)
    require_exact_keys(fast_refill, {"configuration", "cohort", "refill"}, path)

    configuration = require_record(fast_refill["configuration"], f"{path}.configuration")
    require_exact_keys(configuration, {"enabled", "max_members", "window_ms"}, f"{path}.configuration")
    enabled = require_bool(configuration["enabled"], f"{path}.configuration.enabled")
    max_members = require_integer(configuration["max_members"], f"{path}.configuration.max_members")
    window_ms = require_number(configuration["window_ms"], f"{path}.configuration.window_ms")
    if (not enabled and (max_members != 0 or window_ms != 0)) or (
            enabled and (max_members < 1 or window_ms <= 0)):
        raise ProbeError(f"{path}.configuration limits conflict with enabled state")

    cohort = require_record(fast_refill["cohort"], f"{path}.cohort")
    require_exact_keys(cohort, {"active", "selection_open", "dominant_lane", "limit"}, f"{path}.cohort")
    active = require_bool(cohort["active"], f"{path}.cohort.active")
    selection_open = require_bool(cohort["selection_open"], f"{path}.cohort.selection_open")
    dominant_lane = cohort["dominant_lane"]
    if dominant_lane is not None and dominant_lane not in {"low", "normal", "fast"}:
        raise ProbeError(f"{path}.cohort.dominant_lane is invalid")
    limit = require_integer(cohort["limit"], f"{path}.cohort.limit")
    if not active and (selection_open or dominant_lane is not None or limit != 0):
        raise ProbeError(f"{path}.cohort inactive state must be closed, lane-less, and zero-width")
    if active and (dominant_lane is None or limit < 1):
        raise ProbeError(f"{path}.cohort active state requires a lane and positive limit")

    refill = require_record(fast_refill["refill"], f"{path}.refill")
    require_exact_keys(refill, {
        "fast_members_used", "fast_members_remaining", "deadline_at", "remaining_ms",
        "deadline_expired", "window_open", "one_member_eligible_now",
    }, f"{path}.refill")
    used = require_integer(refill["fast_members_used"], f"{path}.refill.fast_members_used")
    remaining = require_integer(refill["fast_members_remaining"], f"{path}.refill.fast_members_remaining")
    deadline_at = refill["deadline_at"]
    if deadline_at is not None:
        require_string(deadline_at, f"{path}.refill.deadline_at")
    remaining_ms = require_number(refill["remaining_ms"], f"{path}.refill.remaining_ms")
    expired = require_bool(refill["deadline_expired"], f"{path}.refill.deadline_expired")
    window_open = require_bool(refill["window_open"], f"{path}.refill.window_open")
    eligible = require_bool(refill["one_member_eligible_now"], f"{path}.refill.one_member_eligible_now")
    if used + remaining != max_members:
        raise ProbeError(f"{path}.refill member counts must equal configured max_members")
    if not active and (used != 0 or deadline_at is not None):
        raise ProbeError(f"{path}.refill inactive cohort cannot retain members or a deadline")
    if deadline_at is None and (remaining_ms != 0 or expired or window_open or eligible):
        raise ProbeError(f"{path}.refill deadline-free state must be unexpired, closed, and ineligible")
    if expired and (remaining_ms != 0 or window_open or eligible):
        raise ProbeError(f"{path}.refill expired state must be closed with zero remaining time")
    if window_open and (
            not enabled or not active or dominant_lane != "fast" or used < 1 or remaining < 1 or
            deadline_at is None or remaining_ms <= 0 or expired):
        raise ProbeError(f"{path}.refill window_open requires a live bounded fast epoch")
    if eligible and not window_open:
        raise ProbeError(f"{path}.refill eligibility requires window_open")
    return fast_refill


SNAPSHOT_KEYS = {
    "schema_version", "sequence", "generated_at", "availability", "registry", "server", "lanes",
    "requests", "fast_refill", "allocator", "cache", "disks", "dspark", "capture", "timeline",
}
AVAILABILITY_KEYS = {
    "server_metrics", "scheduler_predictions", "request_latency", "request_kv", "request_preemption",
    "content", "allocator", "cache", "disks", "dspark", "capture", "fast_refill",
}


def validate_snapshot(snapshot: dict[str, Any], secrets: list[str]) -> dict[str, Any]:
    require_exact_keys(snapshot, SNAPSHOT_KEYS, "snapshot")
    if require_integer(snapshot["schema_version"], "snapshot.schema_version", 1) != SCHEMA_VERSION:
        raise ProbeError(f"snapshot.schema_version must equal {SCHEMA_VERSION}")
    require_integer(snapshot["sequence"], "snapshot.sequence")
    require_string(snapshot["generated_at"], "snapshot.generated_at")
    availability = require_record(snapshot["availability"], "snapshot.availability")
    require_exact_keys(availability, AVAILABILITY_KEYS, "snapshot.availability")
    for field in AVAILABILITY_KEYS:
        require_bool(availability[field], f"snapshot.availability.{field}")
    if availability["content"] is not False or availability["fast_refill"] is not True:
        raise ProbeError("snapshot must mark content unavailable and fast_refill authoritative")
    for field in ("registry", "server", "allocator", "cache", "dspark", "capture"):
        require_record(snapshot[field], f"snapshot.{field}")
    for field in ("lanes", "requests", "disks", "timeline"):
        if not isinstance(snapshot[field], list) or len(snapshot[field]) > MAX_CAPTURED_EVENTS:
            raise ProbeError(f"snapshot.{field} must be a bounded array")
    for index, request in enumerate(snapshot["requests"]):
        request_record = require_record(request, f"snapshot.requests[{index}]")
        content = require_record(request_record.get("content"), f"snapshot.requests[{index}].content")
        require_exact_keys(content, {"prompt", "output", "retained"}, f"snapshot.requests[{index}].content")
        if content["retained"] is not False or content["prompt"] != "" or content["output"] != "":
            raise ProbeError(f"snapshot.requests[{index}] exposed retained prompt or output content")
    validate_fast_refill(snapshot["fast_refill"], "snapshot.fast_refill")
    encoded = json.dumps(snapshot, separators=(",", ":"))
    if any(secret and secret in encoded for secret in secrets):
        raise ProbeError("snapshot reflected a credential value")
    return snapshot


EVENT_KEYS = {
    "schema_version", "id", "sequence", "wall_time", "monotonic_ms", "type", "request_id",
    "slot_id", "sequence_id", "lane", "scheduler_epoch", "decision_id", "before", "after",
    "reason_code", "payload",
}
EVENT_TYPES = {
    "request.upsert", "request.remove", "lane.replace", "allocator.replace", "cache.replace",
    "disk.replace", "dspark.replace", "capture.replace", "server.replace", "timeline.append",
    "stream.overflow", "stream.slow_client",
}


def validate_event(value: Any) -> dict[str, Any]:
    event = require_record(value, "event")
    require_exact_keys(event, EVENT_KEYS, "event")
    if require_integer(event["schema_version"], "event.schema_version", 1) != SCHEMA_VERSION:
        raise ProbeError(f"event.schema_version must equal {SCHEMA_VERSION}")
    sequence = require_integer(event["sequence"], "event.sequence", 1)
    if require_string(event["id"], "event.id") != str(sequence):
        raise ProbeError("event.id must equal decimal event.sequence")
    require_string(event["wall_time"], "event.wall_time")
    require_number(event["monotonic_ms"], "event.monotonic_ms")
    if event["type"] not in EVENT_TYPES:
        raise ProbeError("event.type is invalid")
    require_string(event["reason_code"], "event.reason_code")
    payload = require_record(event["payload"], "event.payload")
    if "fast_refill" in payload:
        validate_fast_refill(payload["fast_refill"], "event.payload.fast_refill")
    if event["type"] in {"request.upsert", "request.remove"} and "fast_refill" not in payload:
        raise ProbeError("request event must carry authoritative fast_refill state")
    return event


class SseCapture:
    def __init__(self, response):
        self.response = response
        self.events: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        data_lines: list[str] = []
        event_id = ""
        frame_bytes = 0
        try:
            while not self.closing.is_set():
                line = self.response.readline(MAX_SSE_EVENT_BYTES + 1)
                if not line:
                    if not self.closing.is_set():
                        raise ProbeError("SSE stream closed before probe teardown")
                    return
                frame_bytes += len(line)
                if len(line) > MAX_SSE_EVENT_BYTES or frame_bytes > MAX_SSE_EVENT_BYTES:
                    raise ProbeError(f"SSE event exceeds {MAX_SSE_EVENT_BYTES} bytes")
                text = line.decode("utf-8").rstrip("\r\n")
                if text == "":
                    if data_lines:
                        if len(self.events) >= MAX_CAPTURED_EVENTS:
                            raise ProbeError(f"probe captured more than {MAX_CAPTURED_EVENTS} SSE events")
                        event = validate_event(parse_json_bytes("\n".join(data_lines).encode(), "SSE event"))
                        if event_id != event["id"]:
                            raise ProbeError("SSE transport and JSON event IDs differ")
                        self.events.append(event)
                    data_lines = []
                    event_id = ""
                    frame_bytes = 0
                elif text.startswith(":"):
                    continue
                elif text.startswith("id:"):
                    event_id = text[3:].lstrip()
                elif text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
        except Exception as error:
            if not self.closing.is_set():
                self.error = error

    def close(self, timeout_seconds: float) -> None:
        self.closing.set()
        self.response.close()
        self.thread.join(timeout_seconds)
        if self.thread.is_alive():
            raise ProbeError("SSE reader did not terminate within its bound")
        if self.error is not None:
            raise self.error


def open_event_stream(url: str, headers: dict[str, str], last_event_id: int, timeout_seconds: float):
    event_headers = dict(headers)
    event_headers.update({"Accept": "text/event-stream", "Last-Event-ID": str(last_event_id)})
    response = open_request(url, event_headers, timeout_seconds)
    if response.status != 200:
        response.close()
        raise ProbeError(f"SSE request returned HTTP {response.status}")
    content_type = response.headers.get_content_type()
    if content_type != "text/event-stream":
        response.close()
        raise ProbeError(f"SSE response content type is {content_type}, expected text/event-stream")
    return response


def wait_for_stop(timeout_seconds: float) -> None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if not ready:
        raise ProbeError(f"probe exceeded its {timeout_seconds:g} second connected hold bound")
    line = sys.stdin.readline(MAX_CONTROL_BYTES)
    if line.rstrip("\r\n") != "STOP":
        raise ProbeError("probe control input must be STOP")


def refill_state_labels(value: Any) -> list[str]:
    refill_value = validate_fast_refill(value)
    configuration = refill_value["configuration"]
    cohort = refill_value["cohort"]
    refill = refill_value["refill"]
    states = []
    if configuration["enabled"] is False:
        states.append("disabled")
    if cohort["active"] is False:
        states.append("inactive")
    if refill["window_open"] is True:
        states.append("window_open")
    if refill["one_member_eligible_now"] is True:
        states.append("one_member_eligible")
    if refill["fast_members_remaining"] == 0:
        states.append("quota_exhausted")
    if refill["deadline_expired"] is True:
        states.append("deadline_expired")
    if refill["window_open"] is True and refill["one_member_eligible_now"] is False:
        states.append("full_width")
    return states


def verify_refill(snapshot: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [state.strip() for state in os.environ.get("M2_DASHBOARD_REFILL_STATES", "").split(",") if state.strip()]
    unknown = set(expected) - SUPPORTED_REFILL_STATES
    if unknown:
        raise ProbeError(f"expected refill states contain unknown values: {sorted(unknown)}")
    event_refills = [
        event["payload"]["fast_refill"] for event in events
        if isinstance(event.get("payload"), dict) and "fast_refill" in event["payload"]
    ]
    observed: list[str] = []
    for value in [snapshot["fast_refill"], *event_refills]:
        observed.extend(refill_state_labels(value))
    required = truthy_environment("M2_DASHBOARD_REQUIRE_REFILL_STATES")
    missing = [state for state in expected if state not in observed]
    if required and not event_refills:
        raise ProbeError("required authoritative fast_refill event objects were not observed")
    if required and missing:
        raise ProbeError(f"required refill states were not observed: {', '.join(missing)}")
    return {
        "required": required,
        "available": bool(event_refills),
        "snapshot_exposed": True,
        "event_objects": len(event_refills),
        "observed_states": sorted(set(observed)),
    }


def main() -> int:
    base_url = normalize_loopback_url(required_environment("M2_DASHBOARD_BASE_URL"))
    api_key = credential(required_environment(API_KEY_ENV), API_KEY_ENV, 1, 1024)
    operator_token = credential(required_environment(OPERATOR_TOKEN_ENV), OPERATOR_TOKEN_ENV, 32, 256)
    if api_key == operator_token:
        raise ProbeError("API and operator credentials must be distinct")
    timeout_ms = bounded_environment_integer("M2_DASHBOARD_PROBE_TIMEOUT_MS", 120_000, 1000, 900_000)
    establish_ms = bounded_environment_integer("M2_DASHBOARD_ESTABLISH_TIMEOUT_MS", 30_000, 1000, 120_000)
    minimum_events = bounded_environment_integer("M2_DASHBOARD_MIN_EVENTS", 1, 0, MAX_CAPTURED_EVENTS)
    timeout_seconds = timeout_ms / 1000
    establish_seconds = establish_ms / 1000
    snapshot_url = base_url + SNAPSHOT_PATH
    events_url = base_url + EVENTS_PATH
    both = credential_headers(api_key, operator_token)

    negative_auth: dict[str, int] = {}
    for route, url in (("snapshot", snapshot_url), ("events", events_url)):
        negative_auth[f"{route}_api_only"] = require_denied(
            fetch_status(url, credential_headers(api_key, None), establish_seconds), f"API-only {route}")
        negative_auth[f"{route}_operator_only"] = require_denied(
            fetch_status(url, credential_headers(None, operator_token), establish_seconds), f"operator-only {route}")
        negative_auth[f"{route}_query_rejected"] = require_denied(
            fetch_status(url + "?probe=1", both, establish_seconds), f"query-bearing {route}")
        negative_auth[f"{route}_classification_rejected"] = require_denied(
            fetch_status(url, credential_headers(api_key, operator_token, **{"X-Llama-Trusted-Lane": "fast"}),
                         establish_seconds), f"classification-bearing {route}")

    validate_snapshot(fetch_snapshot(snapshot_url, both, establish_seconds), [api_key, operator_token])
    future_status = fetch_status(
        events_url, credential_headers(api_key, operator_token, **{"Last-Event-ID": str(MAX_SAFE_INTEGER)}),
        establish_seconds)
    if future_status != 409:
        raise ProbeError(f"future SSE cursor returned HTTP {future_status}, expected 409")
    snapshot = validate_snapshot(
        fetch_snapshot(snapshot_url, both, establish_seconds), [api_key, operator_token])

    response = open_event_stream(events_url, both, snapshot["sequence"], establish_seconds)
    capture = SseCapture(response)
    capture.start()
    print(json.dumps({"status": "READY", "sequence": snapshot["sequence"]}, sort_keys=True), flush=True)
    try:
        wait_for_stop(timeout_seconds)
    finally:
        capture.close(establish_seconds)

    events = capture.events
    if len(events) < minimum_events:
        raise ProbeError(f"captured {len(events)} SSE events, required at least {minimum_events}")
    sequence = snapshot["sequence"]
    for event in events:
        if event["sequence"] != sequence + 1:
            raise ProbeError(
                f"non-contiguous SSE sequence: expected {sequence + 1}, received {event['sequence']}")
        sequence = event["sequence"]

    resumed = open_event_stream(events_url, both, sequence, establish_seconds)
    resumed.close()
    result = {
        "report_schema_version": 1,
        "status": "PASS",
        "negative_auth_statuses": negative_auth,
        "snapshot": {
            "schema_version": snapshot["schema_version"],
            "validator_schema_version": SCHEMA_VERSION,
            "sequence": snapshot["sequence"],
            "request_count": len(snapshot["requests"]),
            "redacted": True,
        },
        "sse": {
            "established": True,
            "captured_events": len(events),
            "first_sequence": events[0]["sequence"] if events else None,
            "final_sequence": sequence,
            "contiguous": True,
            "resume_established": True,
            "future_cursor_resnapshot_status": future_status,
            "maximum_captured_events": MAX_CAPTURED_EVENTS,
        },
        "refill": verify_refill(snapshot, events),
        "credentials": {"source": "environment", "emitted": False, "stored": False},
    }
    encoded = json.dumps(result, sort_keys=True)
    if api_key in encoded or operator_token in encoded:
        raise ProbeError("probe result contains a credential value")
    print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    secrets = [os.environ.get(API_KEY_ENV, ""), os.environ.get(OPERATOR_TOKEN_ENV, "")]
    try:
        raise SystemExit(main())
    except Exception as error:
        message = str(error)
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
                if len(secret) >= 4:
                    message = message.replace(secret[-4:], "[REDACTED]")
        print(f"dashboard live probe failed: {message}", file=sys.stderr, flush=True)
        raise SystemExit(1)

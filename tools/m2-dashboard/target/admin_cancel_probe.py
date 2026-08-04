"""Exercise the first playable dashboard mutation against a live target server."""

from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import live_probe as live

DETAIL_PATH = "/internal/admin/dashboard/request-detail"
CONTROL_PATH = "/internal/admin/dashboard/request-control"
COMPLETION_PATH = "/completion"
TRACE_PATH = "/internal/benchmark/scheduler-trace"
CSRF_HEADER = "X-Llama-Dashboard-CSRF"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_ADMIN_RESPONSE_BYTES = 64 * 1024
REQUEST_ID = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


def new_benchmark_tag() -> str:
    return "phase8-cancel-" + secrets.token_hex(16)


def distinct_credential(value: str) -> str:
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


class HttpResult:
    def __init__(self, status: int, body: Any, headers: list[tuple[str, str]] | None = None):
        self.status = status
        self.body = body
        self.headers = [] if headers is None else headers


def request_json(
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any | None,
        timeout_seconds: float,
        maximum_bytes: int = MAX_RESPONSE_BYTES) -> HttpResult:
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        response = live.OPENER.open(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        response = error
    try:
        data = response.read(maximum_bytes + 1)
        if len(data) > maximum_bytes:
            raise live.ProbeError(f"{url} response exceeds {maximum_bytes} bytes")
        parsed = live.parse_json_bytes(data, url) if data else None
        return HttpResult(response.status, parsed, list(response.headers.items()))
    finally:
        response.close()


def request_json_header_pairs(
        method: str,
        url: str,
        headers: list[tuple[str, str]],
        body: Any | None,
        timeout_seconds: float,
        maximum_bytes: int = MAX_RESPONSE_BYTES) -> HttpResult:
    """Issue a direct request while preserving duplicate security headers."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise live.ProbeError(f"unsupported raw-request URL: {url}")
    connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout_seconds)
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    path = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    try:
        connection.putrequest(method, path)
        for name, value in headers:
            connection.putheader(name, value)
        if encoded is not None:
            connection.putheader("Content-Length", str(len(encoded)))
        connection.endheaders(encoded)
        response = connection.getresponse()
        data = response.read(maximum_bytes + 1)
        if len(data) > maximum_bytes:
            raise live.ProbeError(f"{url} response exceeds {maximum_bytes} bytes")
        parsed_body = live.parse_json_bytes(data, url) if data else None
        return HttpResult(response.status, parsed_body, response.getheaders())
    finally:
        connection.close()


def post_headers(
        api_key: str | None,
        operator_token: str | None,
        origin: str,
        *,
        content_type: str = "application/json",
        csrf: str | None = "1") -> dict[str, str]:
    headers = live.credential_headers(
        api_key,
        operator_token,
        Origin=origin,
        **{"Content-Type": content_type, "Sec-Fetch-Site": "same-site"},
    )
    if csrf is not None:
        headers[CSRF_HEADER] = csrf
    return headers


def require_status(result: HttpResult, expected: int, label: str) -> HttpResult:
    if result.status != expected:
        raise live.ProbeError(f"{label} returned HTTP {result.status}, expected {expected}: {result.body!r}")
    return result


def require_error(
        result: HttpResult,
        status: int,
        error_type: str,
        label: str,
        secrets_to_hide: list[str] | None = None) -> int:
    if secrets_to_hide is not None:
        assert_response_no_secrets(result, secrets_to_hide, label)
    if result.status != status:
        raise live.ProbeError(f"{label} returned HTTP {result.status}, expected {status}: {result.body!r}")
    body = result.body
    if not isinstance(body, dict) or set(body) != {"error"} or not isinstance(body["error"], dict):
        raise live.ProbeError(f"{label} returned a malformed error envelope: {body!r}")
    error = body["error"]
    if set(error) != {"code", "message", "type"} or error.get("code") != status or error.get("type") != error_type:
        raise live.ProbeError(f"{label} returned the wrong error semantics: {body!r}")
    return result.status


def assert_no_secrets(value: Any, secrets_to_hide: list[str], label: str) -> None:
    def contains_secret(candidate: Any, secret: str) -> bool:
        if isinstance(candidate, str):
            return secret in candidate
        if isinstance(candidate, dict):
            return any(
                contains_secret(key, secret) or contains_secret(item, secret)
                for key, item in candidate.items()
            )
        if isinstance(candidate, (list, tuple)):
            return any(contains_secret(item, secret) for item in candidate)
        return False

    if any(secret and contains_secret(value, secret) for secret in secrets_to_hide):
        raise live.ProbeError(f"{label} reflected a credential value")


def assert_response_no_secrets(result: HttpResult, secrets_to_hide: list[str], label: str) -> None:
    assert_no_secrets({"body": result.body, "headers": result.headers}, secrets_to_hide, label)


def response_header_values(result: HttpResult, name: str) -> list[str]:
    expected = name.lower()
    return [value for key, value in result.headers if key.lower() == expected]


def require_single_header(result: HttpResult, name: str, expected: str, label: str) -> None:
    values = response_header_values(result, name)
    if values != [expected]:
        raise live.ProbeError(f"{label} returned invalid {name}: {values!r}")


def validate_response_protection_headers(result: HttpResult, label: str) -> None:
    require_single_header(result, "Cache-Control", "no-store, no-cache, must-revalidate", label)
    require_single_header(result, "Pragma", "no-cache", label)
    require_single_header(result, "X-Content-Type-Options", "nosniff", label)
    require_single_header(result, "Referrer-Policy", "no-referrer", label)
    if response_header_values(result, "Set-Cookie"):
        raise live.ProbeError(f"{label} attempted to persist a browser cookie")


def validate_dashboard_response_headers(result: HttpResult, origin: str, label: str) -> None:
    require_single_header(result, "Access-Control-Allow-Origin", origin, label)
    validate_response_protection_headers(result, label)


def require_protected_error(
        result: HttpResult,
        status: int,
        error_type: str,
        label: str,
        secrets_to_hide: list[str] | None = None) -> int:
    actual = require_error(result, status, error_type, label, secrets_to_hide)
    validate_response_protection_headers(result, label)
    return actual


PREFLIGHT_HEADERS = {
    "authorization",
    "content-type",
    "last-event-id",
    "x-llama-dashboard-csrf",
    "x-llama-trusted-scheduling-token",
}


def comma_separated_header(result: HttpResult, name: str, label: str) -> set[str]:
    values = response_header_values(result, name)
    if len(values) != 1:
        raise live.ProbeError(f"{label} returned invalid {name}: {values!r}")
    return {item.strip().lower() for item in values[0].split(",") if item.strip()}


def validate_preflight(result: HttpResult, origin: str, label: str) -> None:
    require_status(result, 200, label)
    if result.body is not None:
        raise live.ProbeError(f"{label} returned a non-empty response body")
    require_single_header(result, "Access-Control-Allow-Origin", origin, label)
    require_single_header(result, "Access-Control-Allow-Credentials", "false", label)
    allowed_methods = comma_separated_header(result, "Access-Control-Allow-Methods", label)
    if allowed_methods != {"get", "post"}:
        raise live.ProbeError(f"{label} methods differ from exact GET, POST: {sorted(allowed_methods)}")
    allowed_headers = comma_separated_header(result, "Access-Control-Allow-Headers", label)
    if allowed_headers != PREFLIGHT_HEADERS:
        raise live.ProbeError(
            f"{label} headers differ: missing={sorted(PREFLIGHT_HEADERS - allowed_headers)}, "
            f"unexpected={sorted(allowed_headers - PREFLIGHT_HEADERS)}")
    if response_header_values(result, "Set-Cookie"):
        raise live.ProbeError(f"{label} attempted to persist a browser cookie")


def validated_trace_events(result: HttpResult, secrets_to_hide: list[str], label: str) -> list[dict[str, Any]]:
    require_status(result, 200, label)
    assert_response_no_secrets(result, secrets_to_hide, label)
    trace = live.require_record(result.body, label)
    live.require_exact_keys(trace, {"schema", "capacity", "total_events", "overflow_events", "events"}, label)
    if live.require_integer(trace["schema"], f"{label}.schema", 1) != 1:
        raise live.ProbeError(f"{label} uses an unsupported schema")
    if live.require_integer(trace["capacity"], f"{label}.capacity") < 1:
        raise live.ProbeError("scheduler trace is disabled; set LLAMA_SERVER_BENCH_TRACE_CAPACITY before launch")
    total_events = live.require_integer(trace["total_events"], f"{label}.total_events")
    if live.require_integer(trace["overflow_events"], f"{label}.overflow_events") != 0:
        raise live.ProbeError("scheduler trace overflowed before request ownership was authenticated")
    events = trace["events"]
    if not isinstance(events, list) or len(events) > trace["capacity"]:
        raise live.ProbeError(f"{label}.events is not a capacity-bounded array")
    if not all(isinstance(event, dict) for event in events):
        raise live.ProbeError(f"{label}.events contains a non-object event")
    if total_events != len(events):
        raise live.ProbeError(
            f"{label}.total_events differs from its zero-overflow event array")
    return events


TRACE_EVENT_KEYS = {
    "schema", "sequence", "at_us", "event", "request_id", "cohort_id", "generation",
    "begin_token", "end_token", "prompt_tokens", "lane", "active_decode", "active_decode_lane",
    "yield_boundary", "completes_prompt", "reason", "benchmark_tag",
}


def validate_registration_event(value: Any, expected_tag: str) -> int:
    event = live.require_record(value, "scheduler trace registration")
    live.require_exact_keys(event, TRACE_EVENT_KEYS, "scheduler trace registration")
    if live.require_integer(event["schema"], "scheduler trace registration.schema", 1) != 1:
        raise live.ProbeError("scheduler trace registration uses an unsupported schema")
    live.require_integer(event["sequence"], "scheduler trace registration.sequence", 1)
    live.require_integer(event["at_us"], "scheduler trace registration.at_us")
    if live.require_string(event["event"], "scheduler trace registration.event") != "request_registered":
        raise live.ProbeError("scheduler trace match is not a request registration")
    request_id = live.require_integer(event["request_id"], "scheduler trace registration.request_id", 1)
    for field in ("cohort_id", "generation", "begin_token", "end_token", "prompt_tokens"):
        live.require_integer(event[field], f"scheduler trace registration.{field}")
    for field in ("active_decode", "yield_boundary", "completes_prompt"):
        if not isinstance(event[field], bool):
            raise live.ProbeError(f"scheduler trace registration.{field} must be boolean")
    if event["lane"] not in {"low", "normal", "fast"}:
        raise live.ProbeError("scheduler trace registration.lane is invalid")
    if event["active_decode_lane"] not in {"low", "normal", "fast"}:
        raise live.ProbeError("scheduler trace registration.active_decode_lane is invalid")
    live.require_string(event["reason"], "scheduler trace registration.reason")
    if live.require_string(event["benchmark_tag"], "scheduler trace registration.benchmark_tag") != expected_tag:
        raise live.ProbeError("scheduler trace registration carries a different benchmark tag")
    return request_id


REQUEST_KEYS = {
    "id", "lane", "state", "arrival_at", "age_ms", "prompt_tokens", "cache_hit_tokens",
    "output_tokens", "requested_output_tokens", "blocker", "predicted_start_ms", "ttft_ms",
    "tbt_ms", "kv", "scheduler_reasons", "preemptions", "dspark_cycles", "content",
}


def validate_detail(value: Any, expected_handle: str) -> dict[str, Any]:
    detail_value = live.require_record(value, "detail")
    live.require_exact_keys(detail_value, {"schema_version", "request", "registry", "content_reveal"}, "detail")
    if detail_value["schema_version"] != live.SCHEMA_VERSION or detail_value["content_reveal"] is not False:
        raise live.ProbeError("detail schema version or content-reveal policy is invalid")
    request = live.require_record(detail_value["request"], "detail.request")
    live.require_exact_keys(request, REQUEST_KEYS, "detail.request")
    if live.require_string(request.get("id"), "detail.request.id") != expected_handle:
        raise live.ProbeError("detail returned a different request generation")
    if request.get("lane") not in {"low", "normal", "fast"}:
        raise live.ProbeError("detail.request.lane is invalid")
    if request.get("state") not in {"queued", "prefill", "decode", "complete", "cancelled", "failed"}:
        raise live.ProbeError("detail.request.state is invalid")
    for field in ("arrival_at", "blocker"):
        live.require_string(request.get(field), f"detail.request.{field}")
    for field in ("age_ms",):
        live.require_number(request.get(field), f"detail.request.{field}")
    for field in ("prompt_tokens", "cache_hit_tokens", "output_tokens", "preemptions", "dspark_cycles"):
        live.require_integer(request.get(field), f"detail.request.{field}")
    requested = request.get("requested_output_tokens")
    if requested is not None:
        live.require_integer(requested, "detail.request.requested_output_tokens")
    for field in ("ttft_ms", "tbt_ms"):
        if request.get(field) is not None:
            live.require_number(request[field], f"detail.request.{field}")
    predicted = request.get("predicted_start_ms")
    if not isinstance(predicted, list) or len(predicted) != 2:
        raise live.ProbeError("detail.request.predicted_start_ms must contain exactly two bounds")
    for index, value_bound in enumerate(predicted):
        live.require_number(value_bound, f"detail.request.predicted_start_ms[{index}]")
    kv = live.require_record(request.get("kv"), "detail.request.kv")
    live.require_exact_keys(kv, {"logical_bytes", "unique_bytes", "lineage"}, "detail.request.kv")
    live.require_number(kv["logical_bytes"], "detail.request.kv.logical_bytes")
    live.require_number(kv["unique_bytes"], "detail.request.kv.unique_bytes")
    live.require_string(kv["lineage"], "detail.request.kv.lineage")
    reasons = request.get("scheduler_reasons")
    if not isinstance(reasons, list) or len(reasons) > live.MAX_CAPTURED_EVENTS:
        raise live.ProbeError("detail.request.scheduler_reasons must be a bounded array")
    for index, reason in enumerate(reasons):
        live.require_string(reason, f"detail.request.scheduler_reasons[{index}]")
    content = live.require_record(request["content"], "detail.request.content")
    live.require_exact_keys(content, {"prompt", "output", "retained"}, "detail.request.content")
    if content != {"prompt": "", "output": "", "retained": False}:
        raise live.ProbeError("detail exposed retained prompt or output content")
    registry = live.require_record(detail_value["registry"], "detail.registry")
    live.require_exact_keys(
        registry,
        {"revision", "cancel_requested", "timeout_expired", "binding_count", "bindings"},
        "detail.registry",
    )
    bindings = registry["bindings"]
    live.require_integer(registry["revision"], "detail.registry.revision", 1)
    if not isinstance(registry["cancel_requested"], bool) or not isinstance(registry["timeout_expired"], bool):
        raise live.ProbeError("detail registry cancellation and timeout fields must be boolean")
    binding_count = live.require_integer(registry["binding_count"], "detail.registry.binding_count")
    if not isinstance(bindings, list) or binding_count != len(bindings) or len(bindings) > 4096:
        raise live.ProbeError("detail binding count does not match its bounded binding array")
    for index, binding_value in enumerate(bindings):
        binding = live.require_record(binding_value, f"detail.registry.bindings[{index}]")
        live.require_exact_keys(binding, {"slot_id", "slot_generation"}, f"detail.registry.bindings[{index}]")
        live.require_integer(binding["slot_id"], f"detail.registry.bindings[{index}].slot_id")
        live.require_integer(binding["slot_generation"], f"detail.registry.bindings[{index}].slot_generation", 1)
    return detail_value


def validate_control(value: Any, expected_handle: str, expected_status: str) -> dict[str, Any]:
    control = live.require_record(value, "control")
    live.require_exact_keys(control, {"schema_version", "request_id", "action", "status"}, "control")
    expected = {
        "schema_version": live.SCHEMA_VERSION,
        "request_id": expected_handle,
        "action": "cancel",
        "status": expected_status,
    }
    if control != expected:
        raise live.ProbeError(f"control response differs from the exact contract: {control!r}")
    return control


def stale_handle(handle: str) -> str:
    identifier, epoch = handle.split(":", 1)
    return f"{identifier}:{int(epoch) + 1}"


class CompletionCall:
    def __init__(self, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float):
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout_seconds = timeout_seconds
        self.result: HttpResult | None = None
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            self.result = request_json("POST", self.url, self.headers, self.body, self.timeout_seconds)
        except Exception as error:  # noqa: BLE001 - cross-thread propagation must preserve every failure
            self.error = error

    def start(self) -> None:
        self.thread.start()

    def wait(self, timeout_seconds: float) -> HttpResult:
        self.thread.join(timeout_seconds)
        if self.thread.is_alive():
            raise live.ProbeError("cancelled completion did not terminate within its bound")
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise live.ProbeError("completion thread terminated without a result")
        return self.result


def detail(
        base_url: str,
        headers: dict[str, str],
        handle: str,
        timeout_seconds: float) -> HttpResult:
    return request_json(
        "POST",
        base_url + DETAIL_PATH,
        headers,
        {"request_id": handle},
        timeout_seconds,
        MAX_ADMIN_RESPONSE_BYTES,
    )


def wait_for_owned_runtime_id(
        trace_url: str,
        headers: dict[str, str],
        benchmark_tag: str,
        timeout_seconds: float,
        secrets_to_hide: list[str]) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        trace = request_json("GET", trace_url, headers, None, min(timeout_seconds, 30))
        events = validated_trace_events(trace, secrets_to_hide, "scheduler trace")
        matching = [
            event for event in events
            if isinstance(event, dict) and event.get("event") == "request_registered" and
            event.get("benchmark_tag") == benchmark_tag
        ]
        if len(matching) > 1:
            raise live.ProbeError("scheduler trace contains a duplicate probe benchmark tag")
        if matching:
            return validate_registration_event(matching[0], benchmark_tag)
        time.sleep(0.05)
    raise live.ProbeError("probe-owned request did not appear in the authenticated scheduler trace")


def wait_for_bound_request(
        snapshot_url: str,
        detail_url: str,
        get_headers: dict[str, str],
        post: dict[str, str],
        expected_runtime_id: int,
        timeout_seconds: float,
        secrets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = live.validate_snapshot(
            live.fetch_snapshot(snapshot_url, get_headers, min(timeout_seconds, 30)), secrets)
        for request in snapshot["requests"]:
            handle = request.get("id")
            if (not isinstance(handle, str) or not REQUEST_ID.fullmatch(handle) or
                    handle.split(":", 1)[0] != str(expected_runtime_id)):
                continue
            outcome = request_json(
                "POST",
                detail_url,
                post,
                {"request_id": handle},
                min(timeout_seconds, 30),
                MAX_ADMIN_RESPONSE_BYTES,
            )
            if outcome.status != 200 or not isinstance(outcome.body, dict):
                continue
            assert_response_no_secrets(outcome, secrets, "bound request detail")
            validated = validate_detail(outcome.body, handle)
            registry = validated["registry"]
            if (request.get("state") in {"prefill", "decode"} and isinstance(registry, dict) and
                    registry.get("binding_count", 0) >= 1):
                return snapshot, validated
        time.sleep(0.05)
    raise live.ProbeError("no new bound request became visible before the establish timeout")


def wait_for_removed(
        snapshot_url: str,
        headers: dict[str, str],
        handle: str,
        timeout_seconds: float,
        secrets: list[str]) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = live.validate_snapshot(
            live.fetch_snapshot(snapshot_url, headers, min(timeout_seconds, 30)), secrets)
        if all(request.get("id") != handle for request in snapshot["requests"]):
            return snapshot
        time.sleep(0.05)
    raise live.ProbeError("cancelled request remained in the live registry past the timeout")


def main() -> int:
    base_url = live.normalize_loopback_url(live.required_environment("M2_DASHBOARD_BASE_URL"))
    api_key = live.credential(live.required_environment(live.API_KEY_ENV), live.API_KEY_ENV, 1, 1024)
    operator_token = live.credential(
        live.required_environment(live.OPERATOR_TOKEN_ENV), live.OPERATOR_TOKEN_ENV, 32, 256)
    if api_key == operator_token:
        raise live.ProbeError("API and operator credentials must be distinct")
    origin = live.normalize_loopback_url(
        os.environ.get("M2_DASHBOARD_ORIGIN", "http://127.0.0.1:8081"))
    establish_ms = live.bounded_environment_integer(
        "M2_DASHBOARD_ESTABLISH_TIMEOUT_MS", 120_000, 1000, 900_000)
    request_ms = live.bounded_environment_integer(
        "M2_DASHBOARD_REQUEST_TIMEOUT_MS", 300_000, 1000, 900_000)
    prompt_repetitions = live.bounded_environment_integer(
        "M2_DASHBOARD_CANCEL_PROMPT_REPETITIONS", 4096, 512, 8192)
    establish_seconds = establish_ms / 1000
    request_seconds = request_ms / 1000
    secrets = [api_key, operator_token]

    snapshot_url = base_url + live.SNAPSHOT_PATH
    events_url = base_url + live.EVENTS_PATH
    trace_url = base_url + TRACE_PATH
    get_headers = live.credential_headers(api_key, operator_token, Origin=origin)
    posts = post_headers(api_key, operator_token, origin)
    unknown_handle = "9007199254740991:9007199254740991"
    negative: dict[str, int] = {}
    preflight_headers = [
        ("Origin", origin),
        ("Access-Control-Request-Method", "POST"),
        ("Access-Control-Request-Headers", ", ".join(sorted(PREFLIGHT_HEADERS))),
    ]
    preflight = request_json_header_pairs(
        "OPTIONS",
        base_url + CONTROL_PATH,
        preflight_headers,
        None,
        establish_seconds,
        MAX_ADMIN_RESPONSE_BYTES,
    )
    validate_preflight(preflight, origin, "loopback dashboard preflight")

    remote_preflight = request_json_header_pairs(
        "OPTIONS",
        base_url + CONTROL_PATH,
        [
            ("Origin", "https://example.invalid"),
            ("Access-Control-Request-Method", "POST"),
            ("Access-Control-Request-Headers", ", ".join(sorted(PREFLIGHT_HEADERS))),
        ],
        None,
        establish_seconds,
        MAX_ADMIN_RESPONSE_BYTES,
    )
    require_status(remote_preflight, 200, "remote dashboard preflight")
    if response_header_values(remote_preflight, "Access-Control-Allow-Origin"):
        raise live.ProbeError("remote dashboard preflight received an allow-origin header")
    if remote_preflight.body is not None or response_header_values(remote_preflight, "Set-Cookie"):
        raise live.ProbeError("remote dashboard preflight returned body or persistence state")

    authenticated_detail = detail(base_url, posts, unknown_handle, establish_seconds)
    negative["detail_authenticated_unknown"] = require_protected_error(
        authenticated_detail,
        404,
        "not_found_error",
        "authenticated unknown detail",
        secrets,
    )
    validate_dashboard_response_headers(
        authenticated_detail, origin, "authenticated unknown detail")
    authenticated_control = request_json(
        "POST",
        base_url + CONTROL_PATH,
        posts,
        {"action": "cancel", "request_id": unknown_handle},
        establish_seconds,
        MAX_ADMIN_RESPONSE_BYTES,
    )
    negative["control_authenticated_unknown"] = require_protected_error(
        authenticated_control,
        404,
        "not_found_error",
        "authenticated unknown control",
        secrets,
    )
    validate_dashboard_response_headers(
        authenticated_control, origin, "authenticated unknown control")

    security_cases = {
        "missing_csrf": post_headers(api_key, operator_token, origin, csrf=None),
        "cross_origin": post_headers(api_key, operator_token, "https://example.invalid"),
        "wrong_content_type": post_headers(api_key, operator_token, origin, content_type="text/plain"),
    }
    for case, headers in security_cases.items():
        detail_denial = detail(base_url, headers, unknown_handle, establish_seconds)
        negative[f"detail_{case}"] = require_protected_error(
            detail_denial,
            403,
            "permission_error",
            f"detail {case}",
            secrets,
        )
        control_denial = request_json(
            "POST",
            base_url + CONTROL_PATH,
            headers,
            {"action": "cancel", "request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        )
        negative[f"control_{case}"] = require_protected_error(
            control_denial,
            403,
            "permission_error",
            f"control {case}",
            secrets,
        )

    wrong_api_key = distinct_credential(api_key)
    wrong_operator_token = distinct_credential(operator_token)
    classification_headers = dict(posts)
    classification_headers.update({
        "X-Llama-Trusted-Lane": "normal",
        "X-Llama-Benchmark-Tag": "dashboard-auth-negative",
    })
    ambiguous_api_headers = dict(posts)
    ambiguous_api_headers["X-Api-Key"] = api_key
    authorization_cases = {
        "api_only": (post_headers(api_key, None, origin), 403, "permission_error", secrets),
        "operator_only": (post_headers(None, operator_token, origin), 401, "authentication_error", secrets),
        "wrong_api": (
            post_headers(wrong_api_key, operator_token, origin),
            401,
            "authentication_error",
            [*secrets, wrong_api_key],
        ),
        "wrong_operator": (
            post_headers(api_key, wrong_operator_token, origin),
            403,
            "permission_error",
            [*secrets, wrong_operator_token],
        ),
        "classification_headers": (classification_headers, 403, "permission_error", secrets),
        "ambiguous_api_credentials": (ambiguous_api_headers, 403, "permission_error", secrets),
    }
    for case, (headers, status, error_type, case_secrets) in authorization_cases.items():
        detail_denial = detail(base_url, headers, unknown_handle, establish_seconds)
        negative[f"detail_{case}"] = require_protected_error(
            detail_denial,
            status,
            error_type,
            f"detail {case}",
            case_secrets,
        )
        control_denial = request_json(
            "POST",
            base_url + CONTROL_PATH,
            headers,
            {"action": "cancel", "request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        )
        negative[f"control_{case}"] = require_protected_error(
            control_denial,
            status,
            error_type,
            f"control {case}",
            case_secrets,
        )

    query_suffix = "?unexpected=1"
    negative["detail_query"] = require_protected_error(
        request_json(
            "POST",
            base_url + DETAIL_PATH + query_suffix,
            posts,
            {"request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        ),
        403,
        "permission_error",
        "detail query",
        secrets,
    )
    negative["control_query"] = require_protected_error(
        request_json(
            "POST",
            base_url + CONTROL_PATH + query_suffix,
            posts,
            {"action": "cancel", "request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        ),
        403,
        "permission_error",
        "control query",
        secrets,
    )

    duplicate_lane_headers = list(posts.items()) + [
        ("X-Llama-Trusted-Lane", "normal"),
        ("X-Llama-Trusted-Lane", "fast"),
    ]
    negative["detail_ambiguous_classification"] = require_protected_error(
        request_json_header_pairs(
            "POST",
            base_url + DETAIL_PATH,
            duplicate_lane_headers,
            {"request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        ),
        403,
        "permission_error",
        "detail ambiguous classification headers",
        secrets,
    )
    negative["control_ambiguous_classification"] = require_protected_error(
        request_json_header_pairs(
            "POST",
            base_url + CONTROL_PATH,
            duplicate_lane_headers,
            {"action": "cancel", "request_id": unknown_handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        ),
        403,
        "permission_error",
        "control ambiguous classification headers",
        secrets,
    )

    authenticated_snapshot = require_status(
        request_json("GET", snapshot_url, get_headers, None, establish_seconds),
        200,
        "authenticated dashboard snapshot",
    )
    live.validate_snapshot(authenticated_snapshot.body, secrets)
    assert_response_no_secrets(authenticated_snapshot, secrets, "authenticated dashboard snapshot")
    validate_dashboard_response_headers(
        authenticated_snapshot, origin, "authenticated dashboard snapshot")
    benchmark_tag = new_benchmark_tag()
    initial_trace = request_json("GET", trace_url, get_headers, None, establish_seconds)
    initial_trace_events = validated_trace_events(initial_trace, secrets, "initial scheduler trace")
    if any(
            isinstance(event, dict) and event.get("benchmark_tag") == benchmark_tag
            for event in initial_trace_events):
        raise live.ProbeError("fresh probe benchmark tag already exists in scheduler trace")

    completion_headers = live.credential_headers(
        api_key,
        operator_token,
        **{
            "Content-Type": "application/json",
            "X-Llama-Trusted-Lane": "normal",
            "X-Llama-Benchmark-Tag": benchmark_tag,
        },
    )
    plain_completion_headers = live.credential_headers(api_key, None, **{"Content-Type": "application/json"})
    completion = CompletionCall(
        base_url + COMPLETION_PATH,
        completion_headers,
        {
            "prompt": "vertical cancellation probe " * prompt_repetitions,
            "n_predict": 512,
            "temperature": 0,
            "stream": False,
            "cache_prompt": False,
        },
        request_seconds,
    )
    capture: live.SseCapture | None = None
    capture_closed = False
    handle = ""
    accepted_cancel = False
    try:
        completion.start()
        owned_runtime_id = wait_for_owned_runtime_id(
            trace_url, get_headers, benchmark_tag, establish_seconds, secrets)
        snapshot, current_detail = wait_for_bound_request(
            snapshot_url,
            base_url + DETAIL_PATH,
            get_headers,
            posts,
            owned_runtime_id,
            establish_seconds,
            secrets,
        )
        handle = current_detail["request"]["id"]
        registry = current_detail["registry"]
        assert_no_secrets(current_detail, secrets, "request detail")
        if registry["cancel_requested"] is not False:
            raise live.ProbeError("new bound request was already cancelling")

        response = live.open_event_stream(events_url, get_headers, snapshot["sequence"], establish_seconds)
        event_stream_headers = HttpResult(response.status, None, list(response.headers.items()))
        assert_response_no_secrets(event_stream_headers, secrets, "dashboard event stream")
        validate_dashboard_response_headers(event_stream_headers, origin, "dashboard event stream")
        capture = live.SseCapture(response)
        capture.start()

        stale = stale_handle(handle)
        negative["stale_detail"] = require_protected_error(
            detail(base_url, posts, stale, establish_seconds),
            409,
            "stale_request_handle",
            "stale-handle detail",
            secrets,
        )
        negative["stale_cancel"] = require_protected_error(
            request_json(
                "POST",
                base_url + CONTROL_PATH,
                posts,
                {"action": "cancel", "request_id": stale},
                establish_seconds,
                MAX_ADMIN_RESPONSE_BYTES,
            ),
            409,
            "stale_request_handle",
            "stale-handle cancel",
            secrets,
        )

        unchanged = require_status(detail(base_url, posts, handle, establish_seconds), 200, "post-denial detail")
        unchanged_detail = validate_detail(unchanged.body, handle)
        assert_response_no_secrets(unchanged, secrets, "post-denial detail")
        validate_dashboard_response_headers(unchanged, origin, "post-denial detail")
        if unchanged_detail["registry"]["cancel_requested"] is not False:
            raise live.ProbeError("a rejected dashboard mutation changed cancellation state")

        accepted = require_status(
            request_json(
                "POST",
                base_url + CONTROL_PATH,
                posts,
                {"action": "cancel", "request_id": handle},
                establish_seconds,
                MAX_ADMIN_RESPONSE_BYTES,
            ),
            202,
            "valid cancel",
        )
        validate_control(accepted.body, handle, "accepted")
        assert_response_no_secrets(accepted, secrets, "accepted cancel")
        validate_dashboard_response_headers(accepted, origin, "accepted cancel")
        accepted_cancel = True

        replay = request_json(
            "POST",
            base_url + CONTROL_PATH,
            posts,
            {"action": "cancel", "request_id": handle},
            establish_seconds,
            MAX_ADMIN_RESPONSE_BYTES,
        )
        if replay.status == 200:
            validate_control(replay.body, handle, "already_requested")
            replay_state = "already_requested"
        elif replay.status == 404:
            require_protected_error(replay, 404, "not_found_error", "retired cancel replay", secrets)
            replay_state = "retired"
        else:
            raise live.ProbeError(f"cancel replay returned unexpected HTTP {replay.status}: {replay.body!r}")
        assert_response_no_secrets(replay, secrets, "cancel replay")
        validate_dashboard_response_headers(replay, origin, "cancel replay")

        cancelled_completion = completion.wait(request_seconds)
        require_error(
            cancelled_completion,
            503,
            "unavailable_error",
            "operator-cancelled completion",
            secrets,
        )
        if cancelled_completion.body["error"]["message"] != "request cancelled by dashboard operator":
            raise live.ProbeError("cancelled completion did not receive the operator terminal reason")
        final_snapshot = wait_for_removed(
            snapshot_url, get_headers, handle, establish_seconds, secrets)

        deadline = time.monotonic() + establish_seconds
        target_events: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            target_events = [event for event in capture.events if event.get("request_id") == handle]
            terminal_events = [event for event in target_events if event.get("after") == "cancelled"]
            removal_events = [event for event in target_events if event.get("after") == "absent"]
            if terminal_events and removal_events:
                break
            time.sleep(0.05)
        capture_closed = True
        capture.close(establish_seconds)
        target_events = [event for event in capture.events if event.get("request_id") == handle]
        for event in capture.events:
            assert_no_secrets(event, secrets, "SSE event")
        terminal_events = [event for event in target_events if event.get("after") == "cancelled"]
        removal_events = [event for event in target_events if event.get("after") == "absent"]
        if not terminal_events or any(
                event.get("type") != "request.remove" or event.get("reason_code") != "server_cancel"
                for event in terminal_events):
            raise live.ProbeError("SSE terminal event did not carry server_cancel provenance")
        if not removal_events or any(
                event.get("type") != "request.remove" or event.get("reason_code") != "server_cancel"
                for event in removal_events):
            raise live.ProbeError("SSE removal event did not carry server_cancel provenance")
        sequence = snapshot["sequence"]
        for event in capture.events:
            if event["sequence"] != sequence + 1:
                raise live.ProbeError(
                    f"non-contiguous SSE sequence: expected {sequence + 1}, received {event['sequence']}")
            sequence = event["sequence"]

        terminal_detail = detail(base_url, posts, handle, establish_seconds)
        require_protected_error(terminal_detail, 404, "not_found_error", "removed request detail", secrets)
        validate_dashboard_response_headers(terminal_detail, origin, "terminal detail")

        fresh = require_status(
            request_json(
                "POST",
                base_url + COMPLETION_PATH,
                plain_completion_headers,
                {
                    "prompt": "Reply briefly: the server is healthy after cancellation.",
                    "n_predict": 8,
                    "temperature": 0,
                    "stream": False,
                    "cache_prompt": False,
                },
                request_seconds,
            ),
            200,
            "fresh post-cancel completion",
        )
        if not isinstance(fresh.body, dict) or "content" not in fresh.body:
            raise live.ProbeError("fresh completion lacked a normal completion response")
        assert_response_no_secrets(fresh, secrets, "fresh completion")

        result = {
            "report_schema_version": 1,
            "status": "PASS",
            "request_id": handle,
            "correlation": {
                "source": "authenticated_scheduler_trace",
                "runtime_id": owned_runtime_id,
                "benchmark_tag": benchmark_tag,
            },
            "detail": {
                "redacted": True,
                "binding_count": registry["binding_count"],
                "state": current_detail["request"]["state"],
            },
            "negative_statuses": negative,
            "cors": {
                "loopback_preflight": preflight.status,
                "remote_origin_allowed": False,
                "required_headers": sorted(PREFLIGHT_HEADERS),
                "audited_routes": ["snapshot", "events", "detail", "control"],
                "admin_response_headers": "no-store-nosniff-no-referrer-no-cookie",
            },
            "cancel": {
                "http_status": accepted.status,
                "completion_http_status": cancelled_completion.status,
                "completion_error_type": cancelled_completion.body["error"]["type"],
                "replay": replay_state,
                "removed_from_registry": all(
                    request.get("id") != handle for request in final_snapshot["requests"]),
                "terminal_detail_status": terminal_detail.status,
            },
            "sse": {
                "captured_events": len(capture.events),
                "target_events": len(target_events),
                "first_sequence": capture.events[0]["sequence"] if capture.events else None,
                "final_sequence": sequence,
                "contiguous": True,
                "target_reasons": sorted({event["reason_code"] for event in target_events}),
                "target_after": sorted({event["after"] for event in target_events}),
            },
            "fresh_completion": {"http_status": fresh.status, "healthy": True},
            "credentials": {"source": "environment", "reflected": False},
        }
        assert_no_secrets(result, secrets, "probe report")
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    finally:
        try:
            if capture is not None and not capture_closed:
                capture_closed = True
                capture.close(establish_seconds)
        finally:
            if completion.thread.is_alive():
                try:
                    if handle and not accepted_cancel:
                        request_json(
                            "POST",
                            base_url + CONTROL_PATH,
                            posts,
                            {"action": "cancel", "request_id": handle},
                            establish_seconds,
                            MAX_ADMIN_RESPONSE_BYTES,
                        )
                finally:
                    completion.thread.join(establish_seconds)
                    if completion.thread.is_alive():
                        raise live.ProbeError("probe could not deterministically clean up its completion")


if __name__ == "__main__":
    redaction_secrets = [os.environ.get(live.API_KEY_ENV, ""), os.environ.get(live.OPERATOR_TOKEN_ENV, "")]
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - redact credentials from every top-level failure
        message = str(error)
        for secret in redaction_secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(f"dashboard cancel probe failed: {message}", file=os.sys.stderr, flush=True)
        raise SystemExit(1)

#!/usr/bin/env python3

"""Exercise the first playable dashboard mutation against a live target server."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import live_probe as live


DETAIL_PATH = "/internal/admin/dashboard/request-detail"
CONTROL_PATH = "/internal/admin/dashboard/request-control"
COMPLETION_PATH = "/completion"
CSRF_HEADER = "X-Llama-Dashboard-CSRF"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_ID = re.compile(r"^[1-9][0-9]*:[1-9][0-9]*$")


class HttpResult:
    def __init__(self, status: int, body: Any):
        self.status = status
        self.body = body


def request_json(
        method: str,
        url: str,
        headers: dict[str, str],
        body: Any | None,
        timeout_seconds: float) -> HttpResult:
    encoded = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        response = live.OPENER.open(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as error:
        response = error
    try:
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise live.ProbeError(f"{url} response exceeds {MAX_RESPONSE_BYTES} bytes")
        parsed = live.parse_json_bytes(data, url) if data else None
        return HttpResult(response.status, parsed)
    finally:
        response.close()


def post_headers(
        api_key: str,
        operator_token: str,
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


def require_denied(result: HttpResult, label: str) -> int:
    if result.status < 400 or result.status > 499:
        raise live.ProbeError(f"{label} returned HTTP {result.status}, expected a 4xx denial")
    return result.status


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
        except Exception as error:
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
        "POST", base_url + DETAIL_PATH, headers, {"request_id": handle}, timeout_seconds)


def wait_for_bound_request(
        snapshot_url: str,
        detail_url: str,
        get_headers: dict[str, str],
        post: dict[str, str],
        prior: set[str],
        timeout_seconds: float,
        secrets: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        snapshot = live.validate_snapshot(
            live.fetch_snapshot(snapshot_url, get_headers, min(timeout_seconds, 30)), secrets)
        for request in snapshot["requests"]:
            handle = request.get("id")
            if not isinstance(handle, str) or handle in prior or not REQUEST_ID.fullmatch(handle):
                continue
            outcome = request_json("POST", detail_url, post, {"request_id": handle}, min(timeout_seconds, 30))
            if outcome.status != 200 or not isinstance(outcome.body, dict):
                continue
            registry = outcome.body.get("registry")
            if (request.get("state") in {"prefill", "decode"} and isinstance(registry, dict) and
                    registry.get("binding_count", 0) >= 1):
                return snapshot, outcome.body
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
    get_headers = live.credential_headers(api_key, operator_token)
    posts = post_headers(api_key, operator_token, origin)

    initial = live.validate_snapshot(
        live.fetch_snapshot(snapshot_url, get_headers, establish_seconds), secrets)
    prior = {request["id"] for request in initial["requests"]}
    completion_headers = live.credential_headers(api_key, None, **{"Content-Type": "application/json"})
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
    completion.start()

    snapshot, current_detail = wait_for_bound_request(
        snapshot_url,
        base_url + DETAIL_PATH,
        get_headers,
        posts,
        prior,
        establish_seconds,
        secrets,
    )
    handle = current_detail["request"]["id"]
    content = current_detail["request"].get("content")
    registry = current_detail.get("registry")
    if (current_detail.get("content_reveal") is not False or not isinstance(content, dict) or
            content.get("prompt") != "" or content.get("output") != "" or
            content.get("retained") is not False):
        raise live.ProbeError("live request detail exposed retained prompt or output content")
    if not isinstance(registry, dict) or registry.get("cancel_requested") is not False:
        raise live.ProbeError("new bound request was already cancelling")

    response = live.open_event_stream(events_url, get_headers, snapshot["sequence"], establish_seconds)
    capture = live.SseCapture(response)
    capture.start()

    negative: dict[str, int] = {}
    negative["missing_csrf"] = require_denied(
        detail(base_url, post_headers(api_key, operator_token, origin, csrf=None), handle, establish_seconds),
        "missing-CSRF detail")
    negative["cross_origin"] = require_denied(
        detail(base_url, post_headers(api_key, operator_token, "https://example.invalid"), handle, establish_seconds),
        "cross-origin detail")
    negative["wrong_content_type"] = require_denied(
        detail(
            base_url,
            post_headers(api_key, operator_token, origin, content_type="text/plain"),
            handle,
            establish_seconds,
        ),
        "wrong-content-type detail")
    stale = stale_handle(handle)
    negative["stale_detail"] = require_status(
        detail(base_url, posts, stale, establish_seconds), 409, "stale-handle detail").status
    negative["stale_cancel"] = require_status(
        request_json(
            "POST",
            base_url + CONTROL_PATH,
            posts,
            {"action": "cancel", "request_id": stale},
            establish_seconds,
        ),
        409,
        "stale-handle cancel",
    ).status

    unchanged = require_status(detail(base_url, posts, handle, establish_seconds), 200, "post-denial detail")
    if unchanged.body["registry"]["cancel_requested"] is not False:
        raise live.ProbeError("a rejected dashboard mutation changed cancellation state")

    accepted = require_status(
        request_json(
            "POST",
            base_url + CONTROL_PATH,
            posts,
            {"action": "cancel", "request_id": handle},
            establish_seconds,
        ),
        202,
        "valid cancel",
    )
    if accepted.body != {
            "schema_version": live.SCHEMA_VERSION,
            "request_id": handle,
            "action": "cancel",
            "status": "accepted"}:
        raise live.ProbeError(f"valid cancel response had an unexpected shape: {accepted.body!r}")

    cancelled_completion = completion.wait(request_seconds)
    final_snapshot = wait_for_removed(
        snapshot_url, get_headers, handle, establish_seconds, secrets)

    deadline = time.monotonic() + establish_seconds
    target_events: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        target_events = [event for event in capture.events if event.get("request_id") == handle]
        if any(event.get("reason_code") == "server_cancel" for event in target_events) and any(
                event.get("after") in {"cancelled", "absent"} for event in target_events):
            break
        time.sleep(0.05)
    capture.close(establish_seconds)
    if capture.error is not None:
        raise capture.error
    target_events = [event for event in capture.events if event.get("request_id") == handle]
    if not any(event.get("reason_code") == "server_cancel" for event in target_events):
        raise live.ProbeError("SSE did not expose the accepted server_cancel lifecycle event")
    if not any(event.get("after") in {"cancelled", "absent"} for event in target_events):
        raise live.ProbeError("SSE did not expose request cancellation or removal")
    sequence = snapshot["sequence"]
    for event in capture.events:
        if event["sequence"] != sequence + 1:
            raise live.ProbeError(
                f"non-contiguous SSE sequence: expected {sequence + 1}, received {event['sequence']}")
        sequence = event["sequence"]

    terminal_detail = detail(base_url, posts, handle, establish_seconds)
    if terminal_detail.status not in {404, 409}:
        raise live.ProbeError(
            f"removed request detail returned HTTP {terminal_detail.status}, expected 404 or terminal 409")

    fresh = require_status(
        request_json(
            "POST",
            base_url + COMPLETION_PATH,
            completion_headers,
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

    result = {
        "report_schema_version": 1,
        "status": "PASS",
        "request_id": handle,
        "detail": {
            "redacted": True,
            "binding_count": registry["binding_count"],
            "state": current_detail["request"]["state"],
        },
        "negative_statuses": negative,
        "cancel": {
            "http_status": accepted.status,
            "completion_http_status": cancelled_completion.status,
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
        "credentials": {"emitted": False, "stored": False},
    }
    encoded = json.dumps(result, sort_keys=True)
    if api_key in encoded or operator_token in encoded:
        raise live.ProbeError("probe result contains a credential value")
    print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    secrets = [os.environ.get(live.API_KEY_ENV, ""), os.environ.get(live.OPERATOR_TOKEN_ENV, "")]
    try:
        raise SystemExit(main())
    except Exception as error:
        message = str(error)
        for secret in secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        print(f"dashboard cancel probe failed: {message}", file=os.sys.stderr, flush=True)
        raise SystemExit(1)

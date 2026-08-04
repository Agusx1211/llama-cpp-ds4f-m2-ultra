#!/usr/bin/env python3

"""Serve the dashboard with a bounded mock API for a real-browser storage audit."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

API_KEY_ENV = "M2_DASHBOARD_BROWSER_API_KEY"
OPERATOR_TOKEN_ENV = "M2_DASHBOARD_BROWSER_OPERATOR_TOKEN"
OPERATOR_HEADER = "X-Llama-Trusted-Scheduling-Token"
CSRF_HEADER = "X-Llama-Dashboard-CSRF"
SNAPSHOT_PATH = "/internal/admin/dashboard/snapshot"
EVENTS_PATH = "/internal/admin/dashboard/events"
DETAIL_PATH = "/internal/admin/dashboard/request-detail"
CONTROL_PATH = "/internal/admin/dashboard/request-control"
AUDIT_PATH = "/__audit/requests"
STORAGE_PATH = "/__audit/storage.html"
HTML_INJECTION_PATH = "/__audit/html-injection-sentinel"
MAX_BODY_BYTES = 4096


def live_snapshot(root: Path) -> dict[str, Any]:
    snapshot = json.loads((root / "fixtures" / "state.json").read_text())
    identity_map: dict[str, str] = {}
    for index, request in enumerate(snapshot["requests"], 1):
        previous = request["id"]
        request["id"] = f"{index}:1"
        identity_map[previous] = request["id"]
    for item in snapshot["timeline"]:
        if item.get("request_id") in identity_map:
            item["request_id"] = identity_map[item["request_id"]]
    return snapshot


def storage_audit_page() -> bytes:
    return b"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>dashboard browser security audit</title></head>
<body><pre id=\"result\">running</pre><script>
(async () => {
  const databaseNames = typeof indexedDB.databases === "function"
    ? (await indexedDB.databases()).map((entry) => entry.name)
    : ["unavailable"];
  const cacheNames = globalThis.caches ? await caches.keys() : ["unavailable"];
  const registrations = navigator.serviceWorker
    ? await navigator.serviceWorker.getRegistrations()
    : ["unavailable"];
  const requestAudit = await (await fetch("/__audit/requests", {
    cache: "no-store", credentials: "omit", redirect: "error",
  })).json();
  const report = {
    local_storage: Object.keys(localStorage),
    session_storage: Object.keys(sessionStorage),
    cookie: document.cookie,
    indexed_db: databaseNames,
    cache_storage: cacheNames,
    service_workers: registrations.map((entry) => entry === "unavailable" ? entry : entry.scope),
    request_audit: requestAudit,
  };
  document.querySelector("#result").textContent = JSON.stringify(report);
})().catch((error) => {
  document.querySelector("#result").textContent = JSON.stringify({error: String(error)});
});
</script></body></html>"""


class AuditState:
    def __init__(self, api_key: str, operator_token: str):
        self.api_key = api_key
        self.operator_token = operator_token
        self.lock = threading.Lock()
        self.api_requests = 0
        self.credentials_correct = True
        self.secret_in_url = False
        self.html_injection_requested = False

    def record_url(self, path: str) -> None:
        decoded_targets = (path, urllib.parse.unquote(path), urllib.parse.unquote_plus(path))
        with self.lock:
            self.secret_in_url |= any(
                self.api_key in target or self.operator_token in target
                for target in decoded_targets
            )
            self.html_injection_requested |= any(
                urllib.parse.urlsplit(target).path == HTML_INJECTION_PATH
                for target in decoded_targets
            )

    def record(self, path: str, authorization: str, operator: str) -> None:
        with self.lock:
            self.api_requests += 1
            self.credentials_correct &= (
                authorization == f"Bearer {self.api_key}" and operator == self.operator_token
            )
            self.secret_in_url |= self.api_key in path or self.operator_token in path

    def report(self) -> dict[str, Any]:
        with self.lock:
            return {
                "api_requests": self.api_requests,
                "credentials_correct": self.credentials_correct,
                "secret_in_url": self.secret_in_url,
                "html_injection_requested": self.html_injection_requested,
            }


def make_handler(root: Path, snapshot: dict[str, Any], audit: AuditState):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def send_json(self, status: HTTPStatus, value: Any) -> None:
            data = json.dumps(value, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def record_api_request(self) -> bool:
            audit.record(
                self.path,
                self.headers.get("Authorization", ""),
                self.headers.get(OPERATOR_HEADER, ""),
            )
            if not audit.report()["credentials_correct"]:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": {"code": 401, "message": "invalid audit credential", "type": "authentication_error"}},
                )
                return False
            return True

        def do_GET(self) -> None:
            audit.record_url(self.path)
            if self.path == SNAPSHOT_PATH:
                if self.record_api_request():
                    self.send_json(HTTPStatus.OK, snapshot)
                return
            if self.path == EVENTS_PATH:
                if not self.record_api_request():
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    for _ in range(60):
                        self.wfile.write(b": audit keepalive\n\n")
                        self.wfile.flush()
                        time.sleep(0.5)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path == AUDIT_PATH:
                self.send_json(HTTPStatus.OK, audit.report())
                return
            if self.path == STORAGE_PATH:
                data = storage_audit_page()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            super().do_GET()

        def do_HEAD(self) -> None:
            audit.record_url(self.path)
            super().do_HEAD()

        def do_OPTIONS(self) -> None:
            audit.record_url(self.path)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def do_POST(self) -> None:
            audit.record_url(self.path)
            if self.path not in {DETAIL_PATH, CONTROL_PATH}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not self.record_api_request():
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY_BYTES or self.headers.get(CSRF_HEADER) != "1":
                self.send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": {"code": 403, "message": "invalid audit POST", "type": "permission_error"}},
                )
                return
            body = json.loads(self.rfile.read(length))
            request_id = body.get("request_id")
            request = next((item for item in snapshot["requests"] if item["id"] == request_id), None)
            if request is None:
                self.send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"code": 404, "message": "unknown audit request", "type": "not_found_error"}},
                )
                return
            if self.path == DETAIL_PATH:
                detail_request = {**request, "content": {"prompt": "", "output": "", "retained": False}}
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "schema_version": 2,
                        "request": detail_request,
                        "registry": {
                            "revision": 1,
                            "cancel_requested": False,
                            "timeout_expired": False,
                            "binding_count": 0,
                            "bindings": [],
                        },
                        "content_reveal": False,
                    },
                )
                return
            self.send_json(
                HTTPStatus.ACCEPTED,
                {"schema_version": 2, "request_id": request_id, "action": "cancel", "status": "accepted"},
            )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    api_key = os.environ.get(API_KEY_ENV, "")
    operator_token = os.environ.get(OPERATOR_TOKEN_ENV, "")
    if not api_key or len(operator_token.encode()) < 32:
        parser.error(f"{API_KEY_ENV} and a 32-byte {OPERATOR_TOKEN_ENV} are required")
    root = args.root.resolve()
    audit = AuditState(api_key, operator_token)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, live_snapshot(root), audit))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

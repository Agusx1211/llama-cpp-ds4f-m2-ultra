#!/usr/bin/env python3
"""Real-Caddy adversarial integration test for trusted lane classification."""

from __future__ import annotations

import contextlib
import http.client
import http.server
import json
import os
import pathlib
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest

import validate_gateway

HERE = pathlib.Path(__file__).resolve().parent
TOKEN = "gateway-integration-token-00000000000000000000000000000000"
HOSTS = {
    "low.test": "low",
    "normal.test": "normal",
    "fast.test": "fast",
    "admin.test": None,
}
API_KEYS = {
    host: f"{host.split('.', 1)[0]}-api-key-000000000000000000000000" for host in HOSTS
}


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


class RecordingBackend(http.server.ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int]):
        super().__init__(address, BackendHandler)
        self.requests: list[dict[str, object]] = []


class BackendHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.respond()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.respond()

    def respond(self) -> None:
        headers = {name.lower(): self.headers.get_all(name) for name in self.headers}
        record = {
            "client": self.client_address[0],
            "method": self.command,
            "path": self.path,
            "headers": headers,
        }
        self.server.requests.append(record)  # type: ignore[attr-defined]
        valid_authorizations = {f"Bearer {value}" for value in API_KEYS.values()}
        if self.headers.get("Authorization") not in valid_authorizations:
            body = b'{"error":"unauthorized"}'
            status = 401
        else:
            body = json.dumps(record, sort_keys=True).encode("utf-8")
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        pass


class ResolvedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self) -> None:
        plain = socket.create_connection(("127.0.0.1", self.port), self.timeout)
        self.sock = self._context.wrap_socket(plain, server_hostname=self.host)


def https_request(
    host: str,
    port: int,
    path: str = "/v1/models",
    headers: dict[str, str] | None = None,
    method: str = "GET",
    body: bytes | None = None,
):
    connection = ResolvedHTTPSConnection(
        host, port, context=ssl._create_unverified_context(), timeout=5
    )
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read(), dict(response.headers)
    finally:
        connection.close()


def raw_https_request(host: str, port: int, header_lines: list[str]) -> bytes:
    context = ssl._create_unverified_context()
    with (
        socket.create_connection(("127.0.0.1", port), timeout=5) as connection,
        context.wrap_socket(connection, server_hostname=host) as tls,
    ):
        request = [
            "GET /v1/models HTTP/1.1",
            f"Host: {host}",
            *header_lines,
            "Connection: close",
            "",
            "",
        ]
        tls.sendall("\r\n".join(request).encode("ascii"))
        chunks = []
        while True:
            chunk = tls.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)


class GatewayIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.caddy_source = os.environ.get("CADDY", "caddy")
        cls.backend_port = free_port()
        cls.gateway_port = free_port()
        cls.backend = RecordingBackend(("127.0.0.1", cls.backend_port))
        cls.backend_thread = threading.Thread(
            target=cls.backend.serve_forever, daemon=True
        )
        cls.backend_thread.start()
        cls.temp = tempfile.TemporaryDirectory(prefix="m2-gateway-")
        cls.temp_path = pathlib.Path(cls.temp.name)
        cls.caddy_safe_path = cls.temp_path / "safe-caddy"
        cls.caddy_safe_path.mkdir()
        cls.caddy_safe_path.chmod(0o755)
        cls.caddy = str(cls.caddy_safe_path / "caddy")
        shutil.copyfile(shutil.which(cls.caddy_source) or cls.caddy_source, cls.caddy)
        pathlib.Path(cls.caddy).chmod(0o755)
        cls.dashboard_safe_path = cls.temp_path / "safe-dashboard"
        cls.dashboard_safe_path.mkdir()
        cls.dashboard_safe_path.chmod(0o755)
        cls.dashboard_link = cls.temp_path / "dashboard-root"
        cls.dashboard_link.symlink_to(cls.dashboard_safe_path, target_is_directory=True)
        index = cls.dashboard_safe_path / "index.html"
        index.write_text("gateway dashboard", encoding="utf-8")
        index.chmod(0o644)
        cls.caddyfile_safe_path = cls.temp_path / "safe-caddyfile"
        cls.caddyfile_safe_path.mkdir()
        cls.caddyfile_safe_path.chmod(0o755)
        cls.caddyfile_target = cls.caddyfile_safe_path / "Caddyfile"
        cls.caddyfile_target.write_text(
            (HERE / "Caddyfile").read_text(encoding="utf-8"), encoding="utf-8"
        )
        cls.caddyfile_target.chmod(0o444)
        cls.caddyfile_link = cls.temp_path / "gateway.Caddyfile"
        cls.caddyfile_link.symlink_to(cls.caddyfile_target)

        cls.environment = os.environ.copy()
        for host, lane in HOSTS.items():
            key = "ADMIN" if lane is None else lane.upper()
            cls.environment[f"M2_GATEWAY_{key}_HOST"] = (
                f"https://{host}:{cls.gateway_port}"
            )
        cls.environment.update(
            {
                "M2_GATEWAY_BACKEND": f"http://127.0.0.1:{cls.backend_port}",
                "M2_GATEWAY_TRUSTED_TOKEN": TOKEN,
                "M2_GATEWAY_TLS_MODE": "internal",
                "M2_GATEWAY_DASHBOARD_ROOT": str(cls.dashboard_link),
                "M2_GATEWAY_ACME_EMAIL": "gateway-test@example.invalid",
            }
        )
        for name, host in zip(validate_gateway.API_KEY_ENV_NAMES, HOSTS):
            cls.environment[name] = API_KEYS[host]

        cls.process = subprocess.Popen(
            [
                "python3",
                str(HERE / "validate_gateway.py"),
                "--test-mode",
                "--skip-dns",
                "--caddy",
                cls.caddy,
                "--caddyfile",
                str(cls.caddyfile_link),
                "--run",
            ],
            env=cls.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                _, stderr = cls.process.communicate(timeout=1)
                raise RuntimeError(f"caddy exited during startup: {stderr}")
            try:
                status, _, _ = https_request("low.test", cls.gateway_port)
                if status == 401:
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise RuntimeError("caddy did not become ready")

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "process"):
            cls.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                cls.process.wait(timeout=5)
            if cls.process.poll() is None:
                cls.process.kill()
                cls.process.wait(timeout=5)
        if hasattr(cls, "backend"):
            cls.backend.shutdown()
            cls.backend.server_close()
        if hasattr(cls, "temp"):
            cls.temp.cleanup()

    def assert_backend_classification(
        self, host: str, expected_lane: str | None, extra=None
    ) -> None:
        before = len(self.backend.requests)
        status, body, _ = https_request(
            host,
            self.gateway_port,
            headers={
                "Authorization": f"Bearer {API_KEYS[host]}",
                **(extra or {}),
            },
        )
        self.assertEqual(status, 200, body)
        self.assertEqual(len(self.backend.requests), before + 1)
        request = self.backend.requests[-1]
        self.assertEqual(request["client"], "127.0.0.1")
        headers = request["headers"]
        self.assertEqual(headers["x-llama-trusted-scheduling-token"], [TOKEN])
        self.assertNotIn("x-llama-benchmark-tag", headers)
        if expected_lane is None:
            self.assertNotIn("x-llama-trusted-lane", headers)
        else:
            self.assertEqual(headers["x-llama-trusted-lane"], [expected_lane])

    def test_exact_host_lane_and_admin_classification(self) -> None:
        for host, lane in HOSTS.items():
            path = (
                "/internal/admin/dashboard/snapshot" if lane is None else "/v1/models"
            )
            before = len(self.backend.requests)
            status, _, _ = https_request(
                host,
                self.gateway_port,
                path,
                {
                    "Authorization": f"Bearer {API_KEYS[host]}",
                    "Origin": f"https://{host}:{self.gateway_port}",
                    "Sec-Fetch-Site": "same-origin",
                    **(
                        {
                            "Content-Type": "application/json",
                            "X-Llama-Dashboard-CSRF": "1",
                        }
                        if lane is None
                        else {}
                    ),
                },
                method="POST" if lane is None else "GET",
                body=b"{}" if lane is None else None,
            )
            self.assertEqual(status, 200)
            self.assertEqual(len(self.backend.requests), before + 1)
            headers = self.backend.requests[-1]["headers"]
            self.assertEqual(headers["x-llama-trusted-scheduling-token"], [TOKEN])
            if lane is None:
                self.assertEqual(self.backend.requests[-1]["method"], "POST")
                self.assertNotIn("x-llama-trusted-lane", headers)
                self.assertEqual(
                    headers["origin"], [f"http://127.0.0.1:{self.backend_port}"]
                )
                self.assertEqual(headers["sec-fetch-site"], ["same-origin"])
                self.assertEqual(headers["x-llama-dashboard-csrf"], ["1"])
            else:
                self.assertEqual(headers["x-llama-trusted-lane"], [lane])

    def test_forged_fast_and_credentials_are_replaced(self) -> None:
        for host, expected_lane in (
            ("low.test", "low"),
            ("normal.test", "normal"),
            ("fast.test", "fast"),
        ):
            self.assert_backend_classification(
                host,
                expected_lane,
                {
                    "X-Llama-Trusted-Lane": "fast",
                    "X-Llama-Trusted-Scheduling-Token": "attacker-token-that-is-long-enough-but-wrong",
                    "X-Llama-Benchmark-Tag": "attacker",
                },
            )

    def test_duplicate_mixed_case_forgery_cannot_survive(self) -> None:
        before = len(self.backend.requests)
        response = raw_https_request(
            "low.test",
            self.gateway_port,
            [
                f"Authorization: Bearer {API_KEYS['low.test']}",
                "X-Llama-Trusted-Lane: fast",
                "x-llama-trusted-lane: fast",
                "X-Llama-Trusted-Scheduling-Token: attacker-one",
                "x-llama-trusted-scheduling-token: attacker-two",
                "X-Llama-Benchmark-Tag: forged",
            ],
        )
        self.assertIn(b" 200 ", response.split(b"\r\n", 1)[0])
        self.assertEqual(len(self.backend.requests), before + 1)
        headers = self.backend.requests[-1]["headers"]
        self.assertEqual(headers["x-llama-trusted-lane"], ["low"])
        self.assertEqual(headers["x-llama-trusted-scheduling-token"], [TOKEN])
        self.assertNotIn("x-llama-benchmark-tag", headers)

    def test_gateway_auth_blocks_before_backend(self) -> None:
        before = len(self.backend.requests)
        status, _, _ = https_request("fast.test", self.gateway_port)
        self.assertEqual(status, 401)
        self.assertEqual(len(self.backend.requests), before)

        status, _, _ = https_request(
            "fast.test",
            self.gateway_port,
            headers={
                "Authorization": "Bearer attacker-key-0000000000000000000000000",
            },
        )
        self.assertEqual(status, 401)
        self.assertEqual(len(self.backend.requests), before)

        status, _, _ = https_request(
            "fast.test",
            self.gateway_port,
            headers={
                "Authorization": f"Bearer {API_KEYS['low.test']}",
            },
        )
        self.assertEqual(status, 401)
        self.assertEqual(len(self.backend.requests), before)

    def test_runtime_dashboard_root_retarget_is_pinned(self) -> None:
        self.dashboard_link.unlink()
        self.dashboard_link.symlink_to("/", target_is_directory=True)
        try:
            before = len(self.backend.requests)
            status, body, _ = https_request(
                "admin.test", self.gateway_port, "/etc/passwd"
            )
            self.assertEqual(status, 200)
            self.assertEqual(body, b"gateway dashboard")
            self.assertNotIn(b"root:", body)
            self.assertEqual(len(self.backend.requests), before)
        finally:
            self.dashboard_link.unlink()
            self.dashboard_link.symlink_to(
                self.dashboard_safe_path, target_is_directory=True
            )

    def test_runtime_caddyfile_alias_retarget_is_pinned(self) -> None:
        self.caddyfile_link.unlink()
        self.caddyfile_link.symlink_to("/etc/passwd")
        try:
            self.assert_backend_classification("low.test", "low")
        finally:
            self.caddyfile_link.unlink()
            self.caddyfile_link.symlink_to(self.caddyfile_target)

    def test_direct_caddy_launch_without_guard_fails_closed(self) -> None:
        hostile = self.temp_path / "hostile-direct-cwd"
        hostile.mkdir()
        hostile.chmod(0o755)
        (hostile / "*").symlink_to("/", target_is_directory=True)
        environment = self.environment.copy()
        environment.pop(validate_gateway.PINNED_DASHBOARD_ENV, None)
        environment.pop(validate_gateway.LAUNCH_GUARD_ENV, None)
        for command in ("validate", "run"):
            completed = subprocess.run(
                [
                    self.caddy,
                    command,
                    "--config",
                    str(self.caddyfile_target),
                    "--adapter",
                    "caddyfile",
                ],
                cwd=hostile,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("shutdown_delay", completed.stderr)

    def test_admin_does_not_expose_inference_routes(self) -> None:
        before = len(self.backend.requests)
        status, body, headers = https_request(
            "admin.test",
            self.gateway_port,
            "/v1/models",
            {
                "Authorization": f"Bearer {API_KEYS['admin.test']}",
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(body, b"gateway dashboard")
        self.assertEqual(len(self.backend.requests), before)
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertEqual(headers.get("X-Content-Type-Options"), "nosniff")


class ValidatorTest(unittest.TestCase):
    def test_backend_must_be_literal_loopback(self) -> None:
        for value in (
            "http://localhost:8080",
            "http://0.0.0.0:8080",
            "https://127.0.0.1:8080",
            "http://10.0.0.2:8080",
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(validate_gateway.ValidationError),
            ):
                validate_gateway.validate_backend(value)

    def test_production_host_rejects_ports_and_test_domains(self) -> None:
        for value in (
            "https://fast.test",
            "https://fast.example.com:8443",
            "http://fast.example.com",
        ):
            with (
                self.subTest(value=value),
                self.assertRaises(validate_gateway.ValidationError),
            ):
                validate_gateway.validate_host("FAST", value, False)

    def test_client_and_operator_secrets_are_separate(self) -> None:
        keys = [f"key-{index}-0000000000000000000000000000" for index in range(4)]
        validate_gateway.validate_secrets(TOKEN, keys)
        with self.assertRaises(validate_gateway.ValidationError):
            validate_gateway.validate_secrets(
                TOKEN, [keys[0], keys[0], keys[2], keys[3]]
            )
        with self.assertRaises(validate_gateway.ValidationError):
            validate_gateway.validate_secrets(TOKEN, [TOKEN, keys[1], keys[2], keys[3]])
        with self.assertRaises(validate_gateway.ValidationError):
            validate_gateway.validate_secrets(
                "secret with spaces that is deliberately invalid", keys
            )

    def test_internal_tls_is_test_only(self) -> None:
        validate_gateway.validate_tls("operator@example.com", "internal", True)
        validate_gateway.validate_tls(
            "operator@example.com", "operator@example.com", False
        )
        with self.assertRaises(validate_gateway.ValidationError):
            validate_gateway.validate_tls("operator@example.com", "internal", False)

    def test_dashboard_rejects_writable_and_symlinked_assets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2-dashboard-validator-") as temporary:
            root = pathlib.Path(temporary)
            index = root / "index.html"
            index.write_text("safe", encoding="utf-8")
            index.chmod(0o644)
            validate_gateway.validate_dashboard(root)

            index.chmod(0o666)
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_dashboard(root)
            index.chmod(0o644)

            (root / "outside-link").symlink_to("/etc/passwd")
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_dashboard(root)

    def test_production_dashboard_rejects_untrusted_canonical_owner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2-dashboard-owner-") as temporary:
            root = pathlib.Path(temporary)
            index = root / "index.html"
            index.write_text("safe", encoding="utf-8")
            index.chmod(0o644)
            untrusted_uid = root.stat().st_uid + 1
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_dashboard(
                    root, trusted_owner_uids={untrusted_uid}
                )

    def test_sticky_writable_root_assets_and_runtime_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m2-dashboard-sticky-") as temporary:
            root = pathlib.Path(temporary)
            index = root / "index.html"
            index.write_text("safe", encoding="utf-8")
            index.chmod(0o644)

            root.chmod(0o1777)
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_dashboard(root)
            root.chmod(0o700)

            index.chmod(0o1666)
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_dashboard(root)
            with self.assertRaises(validate_gateway.ValidationError):
                validate_gateway.validate_runtime_file(
                    index, "sticky test file", {os.geteuid()}
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)

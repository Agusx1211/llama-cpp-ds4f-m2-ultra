"""Host-only regressions for the bounded real-browser audit server."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import browser_security_audit_server as audit_server


class BrowserSecurityAuditServerTests(unittest.TestCase):
    def test_live_snapshot_canonicalizes_request_and_timeline_identities(self) -> None:
        fixture = b"""{
          "requests": [
            {"id": "fixture-a", "content": {"prompt": "<script>localStorage.pwned=1</script>"}},
            {"id": "fixture-b", "content": {"prompt": "safe"}}
          ],
          "timeline": [
            {"request_id": "fixture-a"},
            {"request_id": "fixture-b"},
            {"request_id": null}
          ]
        }"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixtures").mkdir()
            (root / "fixtures" / "state.json").write_bytes(fixture)
            snapshot = audit_server.live_snapshot(root)

        self.assertEqual([item["id"] for item in snapshot["requests"]], ["1:1", "2:1"])
        self.assertEqual(
            [item["request_id"] for item in snapshot["timeline"]],
            ["1:1", "2:1", None],
        )
        self.assertIn("localStorage.pwned", snapshot["requests"][0]["content"]["prompt"])

    def test_storage_audit_covers_every_disallowed_persistence_surface(self) -> None:
        page = audit_server.storage_audit_page().decode()
        for required in (
                "localStorage",
                "sessionStorage",
                "document.cookie",
                "indexedDB.databases",
                "caches.keys",
                "serviceWorker.getRegistrations",
                "/__audit/requests"):
            self.assertIn(required, page)

    def test_request_audit_retains_only_bounded_security_outcomes(self) -> None:
        state = audit_server.AuditState("api secret/\u00fc", "operator")
        state.record("/internal/admin/dashboard/snapshot", "Bearer api secret/\u00fc", "operator")
        self.assertEqual(
            state.report(),
            {
                "api_requests": 1,
                "credentials_correct": True,
                "secret_in_url": False,
                "html_injection_requested": False,
            },
        )
        state.record_url("/unmatched-static-path?token=api+secret%2F%C3%BC")
        state.record("/snapshot", "Bearer wrong", "operator")
        self.assertEqual(
            state.report(),
            {
                "api_requests": 2,
                "credentials_correct": False,
                "secret_in_url": True,
                "html_injection_requested": False,
            },
        )
        state.record_url(audit_server.HTML_INJECTION_PATH)
        self.assertTrue(state.report()["html_injection_requested"])


if __name__ == "__main__":
    unittest.main()

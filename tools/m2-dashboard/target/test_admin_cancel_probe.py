"""Host-only regressions for the live cancellation target probe."""

from __future__ import annotations

import unittest
from unittest import mock

import admin_cancel_probe as probe


class AdminCancelProbeTests(unittest.TestCase):
    def test_bound_lookup_uses_authenticated_runtime_id(self) -> None:
        snapshot = {
            "requests": [
                {"id": "2:1", "state": "decode"},
                {"id": "3:7", "state": "prefill"},
            ],
        }
        posted_handles: list[str] = []

        def fake_request_json(method, url, headers, body, timeout, maximum_bytes):
            del method, url, headers, timeout, maximum_bytes
            posted_handles.append(body["request_id"])
            return probe.HttpResult(
                200,
                {
                    "request": {"id": body["request_id"]},
                    "registry": {"binding_count": 1},
                },
            )

        with (
                mock.patch.object(probe.live, "fetch_snapshot", return_value=snapshot),
                mock.patch.object(probe.live, "validate_snapshot", side_effect=lambda value, _: value),
                mock.patch.object(probe, "request_json", side_effect=fake_request_json),
                mock.patch.object(probe, "validate_detail", side_effect=lambda value, _: value)):
            selected_snapshot, detail = probe.wait_for_bound_request(
                "http://127.0.0.1/snapshot",
                "http://127.0.0.1/detail",
                {},
                {},
                3,
                0.1,
                [],
            )

        self.assertIs(selected_snapshot, snapshot)
        self.assertEqual(detail["request"]["id"], "3:7")
        self.assertEqual(posted_handles, ["3:7"])

    def test_error_contract_rejects_unrelated_4xx(self) -> None:
        with self.assertRaisesRegex(probe.live.ProbeError, "wrong error semantics"):
            probe.require_error(
                probe.HttpResult(
                    403,
                    {"error": {"code": 403, "message": "denied", "type": "authentication_error"}},
                ),
                403,
                "permission_error",
                "security control",
            )

    def test_control_contract_is_exact(self) -> None:
        expected = {
            "schema_version": probe.live.SCHEMA_VERSION,
            "request_id": "3:7",
            "action": "cancel",
            "status": "accepted",
        }
        self.assertEqual(probe.validate_control(expected, "3:7", "accepted"), expected)
        with self.assertRaisesRegex(probe.live.ProbeError, "fields differ"):
            probe.validate_control({**expected, "extra": True}, "3:7", "accepted")

    def test_secret_scan_uses_raw_scalars_not_json_escaping(self) -> None:
        credential = 'operator"token\\suffix'
        with self.assertRaisesRegex(probe.live.ProbeError, "reflected a credential"):
            probe.assert_no_secrets(
                {"error": {"message": f"rejected {credential}"}},
                [credential],
                "quoted credential response",
            )
        with self.assertRaisesRegex(probe.live.ProbeError, "reflected a credential"):
            probe.assert_no_secrets(
                {f"credential-{credential}": "hidden"},
                [credential],
                "credential object key",
            )

    def test_trace_envelope_requires_exact_zero_overflow_count(self) -> None:
        result = probe.HttpResult(
            200,
            {
                "schema": 1,
                "capacity": 8,
                "total_events": 999,
                "overflow_events": 0,
                "events": [],
            },
        )
        with self.assertRaisesRegex(probe.live.ProbeError, "total_events differs"):
            probe.validated_trace_events(result, [], "trace")

    def test_trace_registration_requires_exact_event_schema(self) -> None:
        with self.assertRaisesRegex(probe.live.ProbeError, "fields differ"):
            probe.validate_registration_event(
                {
                    "event": "request_registered",
                    "benchmark_tag": "owned",
                    "request_id": 7,
                },
                "owned",
            )

        event = {
            "schema": 1,
            "sequence": 3,
            "at_us": 100,
            "event": "request_registered",
            "request_id": 7,
            "cohort_id": 0,
            "generation": 0,
            "begin_token": 0,
            "end_token": 0,
            "prompt_tokens": 1024,
            "lane": "normal",
            "active_decode": False,
            "active_decode_lane": "low",
            "yield_boundary": False,
            "completes_prompt": False,
            "reason": "none",
            "benchmark_tag": "owned",
        }
        self.assertEqual(probe.validate_registration_event(event, "owned"), 7)


if __name__ == "__main__":
    unittest.main()

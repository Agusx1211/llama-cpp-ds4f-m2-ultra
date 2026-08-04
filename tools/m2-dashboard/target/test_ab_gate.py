import json
import pathlib
import tempfile
import unittest

import ab_gate


def measurements(disconnected, connected, tag="gate"):
    values = []
    for phase, samples in (("disconnected", disconnected), ("connected", connected)):
        for value in samples:
            values.append({
                "phase": phase,
                "tag": tag,
                "ttft_ms": value,
                "elapsed_ms": value * 2,
            })
    return values


class TargetGateTests(unittest.TestCase):
    def test_redactor_removes_full_secrets_and_suffixes(self):
        redactor = ab_gate.Redactor(["api-01234567", "operator-89abcdef"])
        value = redactor.text("api-01234567 4567 operator-89abcdef cdef")
        self.assertNotIn("api-01234567", value)
        self.assertNotIn("4567", value)
        self.assertNotIn("operator-89abcdef", value)
        self.assertNotIn("cdef", value)
        self.assertFalse(redactor.contains(value))

    def test_secret_boundary_rejects_credentials_and_api_arguments(self):
        redactor = ab_gate.Redactor(["api-01234567", "operator-89abcdef"])
        with self.assertRaisesRegex(ab_gate.GateError, "credential value or fragment"):
            ab_gate.validate_secret_boundary(
                {"value": "api-01234567"}, ["server"], redactor)
        with self.assertRaisesRegex(ab_gate.GateError, "never command arguments"):
            ab_gate.validate_secret_boundary(
                {}, ["server", "--api-key", "not-a-real-key"], redactor)

    def test_probe_command_is_explicit_and_bounded(self):
        self.assertEqual(
            ab_gate.validate_probe_command(["/usr/bin/python3", "target/live_probe.py"]),
            ["/usr/bin/python3", "target/live_probe.py"],
        )
        for command in (None, [], [""], ["python3", "bad\nargument"], ["x"] * 17):
            with self.assertRaises(ab_gate.GateError):
                ab_gate.validate_probe_command(command)

    def test_loopback_url_boundary(self):
        self.assertEqual(ab_gate.require_loopback_url("http://127.0.0.1:18130/"),
                         "http://127.0.0.1:18130")
        for value in ("http://example.com:18130", "http://user:pass@127.0.0.1:18130",
                      "http://127.0.0.1:18130?token=value"):
            with self.assertRaises(ab_gate.GateError):
                ab_gate.require_loopback_url(value)

    def test_counterbalanced_phase_plan_and_sample_floor(self):
        order, repetitions, minimum, available = ab_gate.validate_phase_plan({
            "phase_order": ["disconnected", "connected", "connected", "disconnected"],
            "repetitions_per_block": 2,
            "minimum_samples_per_phase": 4,
        })
        self.assertEqual(order, ["disconnected", "connected", "connected", "disconnected"])
        self.assertEqual(repetitions, 2)
        self.assertEqual(minimum, 4)
        self.assertEqual(available, {"disconnected": 4, "connected": 4})
        with self.assertRaisesRegex(ab_gate.GateError, "insufficient declared sample size"):
            ab_gate.validate_phase_plan({
                "phase_order": ["disconnected", "connected", "connected", "disconnected"],
                "repetitions_per_block": 1,
                "minimum_samples_per_phase": 3,
            })
        with self.assertRaisesRegex(ab_gate.GateError, "counterbalanced"):
            ab_gate.validate_phase_plan({
                "phase_order": ["disconnected", "disconnected", "connected", "connected"],
                "repetitions_per_block": 2,
                "minimum_samples_per_phase": 4,
            })

    def test_comparison_accepts_bounded_overhead_and_rejects_sample_shortfall(self):
        result = ab_gate.compare_phases(
            measurements([100, 101, 99, 100], [102, 101, 103, 102]),
            minimum_samples=4,
            impact_threshold_percent=5,
            maximum_cv_percent=10,
        )
        self.assertTrue(result["passed"])
        result = ab_gate.compare_phases(
            measurements([100, 101], [101, 102]),
            minimum_samples=3,
            impact_threshold_percent=5,
            maximum_cv_percent=10,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("requires 3" in failure for failure in result["failures"]))

    def test_comparison_rejects_impact_and_variance(self):
        impact = ab_gate.compare_phases(
            measurements([100, 100, 100], [110, 110, 110]), 3, 5, 20)
        self.assertFalse(impact["passed"])
        self.assertTrue(any("impact" in failure for failure in impact["failures"]))
        variance = ab_gate.compare_phases(
            measurements([10, 100, 190], [10, 100, 190]), 3, 1000, 10)
        self.assertFalse(variance["passed"])
        self.assertTrue(any("CV" in failure for failure in variance["failures"]))

    def test_exact_output_gate_and_request_artifact(self):
        result = ab_gate.RequestResult(
            tag="gate",
            requested_tokens=2,
            status=200,
            start_ns=1_000_000,
            first_token_ns=2_000_000,
            end_ns=4_000_000,
            reported_tokens=2,
            tokens=[7, 8],
            content="answer",
        )
        result.validate()
        artifact = result.artifact("connected", 1, 0, 2)
        self.assertEqual(artifact["token_ids"], [7, 8])
        self.assertEqual(artifact["ttft_ms"], 1.0)
        ab_gate.verify_exact_outputs([artifact, dict(artifact)])
        changed = dict(artifact)
        changed["output_sha256"] = "different"
        with self.assertRaisesRegex(ab_gate.GateError, "exact output differs"):
            ab_gate.verify_exact_outputs([artifact, changed])

    def test_swap_parser_and_artifact_scanner(self):
        samples = [{
            "phase": "connected",
            "block": 1,
            "boundary": "before",
            "output": "vm.swapusage: total = 2048.00M  used = 1.50G  free = 512.00M",
        }]
        self.assertEqual(
            ab_gate.parse_swap_used_bytes(samples, "connected", 1, "before"),
            int(1.5 * 1024 ** 3),
        )
        redactor = ab_gate.Redactor(["api-01234567"])
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "safe.json").write_text(json.dumps({"status": "PASS"}))
            ab_gate.scan_artifacts(root, redactor)
            (root / "leak.txt").write_text("suffix 4567")
            with self.assertRaisesRegex(ab_gate.GateError, "credential value or fragment leaked"):
                ab_gate.scan_artifacts(root, redactor)


if __name__ == "__main__":
    unittest.main()

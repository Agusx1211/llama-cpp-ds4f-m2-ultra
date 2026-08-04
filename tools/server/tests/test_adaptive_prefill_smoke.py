#!/usr/bin/env python3
# Signed-by: GPT-5
# Date: 2026-08-04 UTC

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import unittest


SCRIPT = pathlib.Path(__file__).with_name("adaptive-prefill-smoke.py")
SPEC = importlib.util.spec_from_file_location("adaptive_prefill_smoke", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


def arguments(**overrides: object) -> argparse.Namespace:
    values = {
        "api_key": "api-key",
        "scheduling_token": "0123456789abcdef0123456789abcdef",
        "trace_endpoint": "/internal/benchmark/scheduler-trace",
        "timeout_seconds": 900,
        "low_prompt_tokens": 8192,
        "low_n_predict": 8,
        "fast_prompt_tokens": 128,
        "fast_n_predict": 512,
        "reference_n_predict": 32,
        "sequential_bursts": False,
        "burst_count": 0,
        "burst_interval_seconds": 2,
        "burst_n_predict": 16,
        "alignment_tokens": 128,
        "active_fast_chunk_limit_tokens": 128,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def result(tag: str, tokens: list[int], content: str) -> object:
    value = SMOKE.Result(tag, "fast", 128, len(tokens))
    value.status = 200
    value.reported_tokens_predicted = len(tokens)
    value.tokens = tokens
    value.content_parts = [content]
    return value


class AdaptivePrefillSmokeTests(unittest.TestCase):
    def test_default_vertical_request_set_is_unchanged(self) -> None:
        args = arguments()
        SMOKE.validate_args(args)
        config = SMOKE.config_from_args(args)
        self.assertNotIn("mode", config)
        self.assertEqual(
            SMOKE.expected_trusted_requests(config),
            {
                "reference-before": ("fast", 128),
                "reference-after": ("fast", 128),
                "low-8k": ("low", 8192),
                "isolated-fast": ("fast", 128),
                "sustained-fast": ("fast", 128),
            })

    def test_sequential_mode_has_only_short_controls_and_bursts(self) -> None:
        args = arguments(sequential_bursts=True, burst_count=4)
        SMOKE.validate_args(args)
        config = SMOKE.config_from_args(args)
        expected = SMOKE.expected_trusted_requests(config)
        self.assertEqual(config["mode"], SMOKE.SEQUENTIAL_BURST_MODE)
        self.assertNotIn("isolated-fast", expected)
        self.assertNotIn("sustained-fast", expected)
        self.assertEqual(
            set(expected),
            {
                "reference-before", "reference-after", "isolated-low", "isolated-burst", "low-8k",
                "burst-00", "burst-01", "burst-02", "burst-03",
            })
        self.assertEqual(SMOKE.required_trace_capacity(config), 69230)

    def test_sequential_mode_requires_exact_count_and_positive_interval(self) -> None:
        SMOKE.validate_args(arguments(sequential_bursts=True, burst_count=4))
        for count in (0, 1, 3, 5):
            with self.assertRaises(ValueError):
                SMOKE.validate_args(arguments(sequential_bursts=True, burst_count=count))
        with self.assertRaises(ValueError):
            SMOKE.validate_args(arguments(sequential_bursts=True, burst_count=1, burst_interval_seconds=0))
        with self.assertRaises(ValueError):
            SMOKE.validate_args(arguments(burst_count=1))

    def test_trace_preflight_fails_before_work_for_capacity_or_history(self) -> None:
        config = SMOKE.config_from_args(arguments(sequential_bursts=True, burst_count=4))
        required = SMOKE.required_trace_capacity(config)
        accepted = {"schema": 1, "capacity": required, "total_events": 0, "overflow_events": 0, "events": []}
        self.assertEqual(
            SMOKE.validate_trace_preflight(accepted, config),
            {"capacity": required, "required_capacity": required})
        with self.assertRaises(RuntimeError):
            SMOKE.validate_trace_preflight({**accepted, "capacity": required - 1}, config)
        with self.assertRaises(RuntimeError):
            SMOKE.validate_trace_preflight(
                {**accepted, "total_events": 1, "events": [{"sequence": 1}]}, config)

    def test_burst_validation_checks_oracles_overlap_and_timings(self) -> None:
        results = {
            "isolated-low": result("isolated-low", [3], "low"),
            "low-8k": result("low-8k", [3], "low"),
            "isolated-burst": result("isolated-burst", [7, 8], "burst"),
            "burst-00": result("burst-00", [7, 8], "burst"),
            "burst-01": result("burst-01", [7, 8], "burst"),
        }
        low = results["low-8k"]
        low.end_ns = 1000
        for ordinal in range(2):
            burst = results[f"burst-{ordinal:02d}"]
            burst.intended_launch_ns = 100 + ordinal * 100
            burst.start_ns = 110 + ordinal * 100
            burst.first_token_ns = 130 + ordinal * 100
            burst.end_ns = 150 + ordinal * 100
        low.progress = [
            {"processed": 128, "client_monotonic_ns": 175},
            {"processed": 256, "client_monotonic_ns": 275},
        ]

        timings = SMOKE.validate_sequential_burst_results(results, 2)
        self.assertEqual([timing["launch_lag_ms"] for timing in timings], [0.00001, 0.00001])
        self.assertEqual([timing["ttft_ms"] for timing in timings], [0.00002, 0.00002])
        self.assertEqual([timing["low_progress_after_ns"] for timing in timings], [175, 275])

        results["burst-01"].tokens = [9, 9]
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 2)
        results["burst-01"].tokens = [7, 8]
        results["burst-01"].first_token_ns = low.end_ns
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 2)
        results["burst-01"].first_token_ns = 230
        results["burst-01"].end_ns = low.end_ns
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 2)
        results["burst-01"].end_ns = 250
        low.progress = [{"processed": 256, "client_monotonic_ns": 275}]
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# Signed-by: GPT-5
# Date: 2026-08-04 UTC

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import threading
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
        "priority_chunk_limit_tokens": 2048,
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


def sequential_results(count: int = 4) -> dict[str, object]:
    results = {
        "isolated-low": result("isolated-low", [3], "low"),
        "low-8k": result("low-8k", [3], "low"),
        "isolated-burst": result("isolated-burst", [7, 8], "burst"),
    }
    low = results["low-8k"]
    low.end_ns = 1000
    progress = []
    for ordinal in range(count):
        tag = f"burst-{ordinal:02d}"
        burst = result(tag, [7, 8], "burst")
        burst.intended_launch_ns = 90 + ordinal * 100
        burst.start_ns = 100 + ordinal * 100
        burst.first_token_ns = 120 + ordinal * 100
        burst.end_ns = 150 + ordinal * 100
        results[tag] = burst
        progress.append({
            "processed": 128 * (ordinal + 1),
            "client_monotonic_ns": 225 + ordinal * 100 if ordinal + 1 < count else burst.end_ns + 25,
        })
    low.progress = progress
    return results


class AdaptivePrefillSmokeTests(unittest.TestCase):
    def test_default_vertical_request_set_is_unchanged(self) -> None:
        args = arguments()
        SMOKE.validate_args(args)
        config = SMOKE.config_from_args(args)
        self.assertNotIn("mode", config)
        self.assertEqual(config["acceptance"]["priority_chunk_limit_tokens"], 2048)
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
        args = arguments(
            sequential_bursts=True, burst_count=4, burst_interval_seconds=0, burst_n_predict=4)
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

    def test_sequential_mode_requires_exact_count_and_zero_interval(self) -> None:
        SMOKE.validate_args(arguments(sequential_bursts=True, burst_count=4, burst_interval_seconds=0))
        for count in (0, 1, 3, 5):
            with self.assertRaises(ValueError):
                SMOKE.validate_args(
                    arguments(sequential_bursts=True, burst_count=count, burst_interval_seconds=0))
        for interval in (-1, 1, 2):
            with self.assertRaises(ValueError):
                SMOKE.validate_args(arguments(
                    sequential_bursts=True, burst_count=4, burst_interval_seconds=interval))
        with self.assertRaises(ValueError):
            SMOKE.validate_args(arguments(burst_count=1))

    def test_trace_preflight_fails_before_work_for_capacity_or_history(self) -> None:
        config = SMOKE.config_from_args(arguments(
            sequential_bursts=True, burst_count=4, burst_interval_seconds=0))
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

    def test_valid_four_burst_timeline_records_post_handoff_progress(self) -> None:
        results = sequential_results()
        timings = SMOKE.validate_sequential_burst_results(results, 4)
        self.assertEqual([timing["launch_lag_ms"] for timing in timings], [0.00001] * 4)
        self.assertEqual([timing["ttft_ms"] for timing in timings], [0.00002] * 4)
        self.assertEqual(
            [timing.get("low_progress_during_next_burst_ns") for timing in timings],
            [225, 325, 425, None])

    def test_sticky_or_handoff_only_progress_is_rejected(self) -> None:
        for timestamp in (149, 150, 199, 200):
            with self.subTest(timestamp=timestamp):
                results = sequential_results()
                results["low-8k"].progress[0]["client_monotonic_ns"] = timestamp
                with self.assertRaises(RuntimeError):
                    SMOKE.validate_sequential_burst_results(results, 4)

    def test_progress_at_or_after_next_completion_is_rejected(self) -> None:
        for timestamp in (250, 251):
            with self.subTest(timestamp=timestamp):
                results = sequential_results()
                results["low-8k"].progress[0]["client_monotonic_ns"] = timestamp
                with self.assertRaises(RuntimeError):
                    SMOKE.validate_sequential_burst_results(results, 4)

    def test_final_burst_needs_no_post_teardown_prompt_progress(self) -> None:
        results = sequential_results()
        results["low-8k"].progress.pop()
        timings = SMOKE.validate_sequential_burst_results(results, 4)
        self.assertNotIn("low_progress_during_next_burst_ns", timings[-1])

    def test_missing_transition_progress_is_rejected(self) -> None:
        for ordinal in range(3):
            with self.subTest(ordinal=ordinal):
                results = sequential_results()
                results["low-8k"].progress.pop(ordinal)
                with self.assertRaises(RuntimeError):
                    SMOKE.validate_sequential_burst_results(results, 4)

    def test_every_burst_must_teardown_before_low_completion(self) -> None:
        for end_ns in (1000, 1001):
            with self.subTest(end_ns=end_ns):
                results = sequential_results()
                results["burst-03"].end_ns = end_ns
                with self.assertRaises(RuntimeError):
                    SMOKE.validate_sequential_burst_results(results, 4)

    def test_default_off_serial_timeline_is_rejected(self) -> None:
        results = sequential_results()
        results["burst-01"].start_ns = 1001
        results["burst-01"].first_token_ns = 1020
        results["burst-01"].end_ns = 1050
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 4)

    def test_oracle_and_result_cardinality_gates_are_unchanged(self) -> None:
        results = sequential_results()
        results["burst-01"].tokens = [9, 9]
        with self.assertRaises(RuntimeError):
            SMOKE.validate_sequential_burst_results(results, 4)

        for mutation in ("reported", "tokens", "content"):
            with self.subTest(mutation=mutation):
                burst = result("burst", [7, 8], "burst")
                if mutation == "reported":
                    burst.reported_tokens_predicted = 1
                elif mutation == "tokens":
                    burst.tokens = [7]
                else:
                    burst.content_parts = []
                with self.assertRaises(RuntimeError):
                    SMOKE.require_result(burst)

    def test_pipelined_loop_has_no_wait_between_teardown_and_next_launch(self) -> None:
        smoke = SMOKE.Smoke.__new__(SMOKE.Smoke)
        smoke.low_done = threading.Event()
        order = []

        def run_request(**kwargs: object) -> object:
            tag = str(kwargs["tag"])
            order.append(f"start:{tag}")
            burst = result(tag, [7, 8, 9, 10], "burst")
            burst.intended_launch_ns = int(kwargs["intended_launch_ns"])
            burst.start_ns = burst.intended_launch_ns
            burst.first_token_ns = burst.start_ns + 1
            burst.end_ns = burst.first_token_ns + 1
            order.append(f"teardown:{tag}")
            return burst

        def unexpected_wait(*_args: object, **_kwargs: object) -> None:
            self.fail("pipelined burst loop invoked a progress wait")

        smoke.run_request = run_request
        smoke.wait_before_low_done = unexpected_wait
        bursts = {"count": 4, "prompt_tokens": 128, "n_predict": 4, "salt": 5001, "seed": 42}
        last = smoke.run_pipelined_bursts(bursts)
        self.assertEqual(last.tag, "burst-03")
        self.assertEqual(order, [
            "start:burst-00", "teardown:burst-00",
            "start:burst-01", "teardown:burst-01",
            "start:burst-02", "teardown:burst-02",
            "start:burst-03", "teardown:burst-03",
        ])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

"""Analyze balanced Metal fast-sync throughput, CPU, and correctness data."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_delta(
    wait: list[float], poll: list[float], seed: int, iterations: int = 50_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iterations):
        wait_mean = statistics.fmean(rng.choice(wait) for _ in wait)
        poll_mean = statistics.fmean(rng.choice(poll) for _ in poll)
        deltas.append(100.0 * (poll_mean / wait_mean - 1.0))
    return percentile(deltas, 0.025), percentile(deltas, 0.975)


def describe(label: str, values: list[float], unit: str) -> None:
    mean = statistics.fmean(values)
    stdev = statistics.pstdev(values)
    print(
        label,
        f"n={len(values)}",
        f"mean={mean:.6f}{unit}",
        f"stdev={stdev:.6f}{unit}",
        f"cv_pct={100.0 * stdev / mean:.4f}",
        f"min={min(values):.6f}{unit}",
        f"max={max(values):.6f}{unit}",
    )


def values_by(
    records: list[dict[str, object]],
    kind: str,
    getter: Callable[[dict[str, object]], float],
) -> dict[tuple[str, str], list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        if record["kind"] == kind:
            grouped[(str(record["mode"]), str(record["arm"]))].append(getter(record))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    args = parser.parse_args()

    with args.samples.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    failures: list[str] = []
    for record in records:
        if record.get("predicted_n") != record.get("requested_n"):
            failures.append(
                f"{record.get('run_id')}/{record.get('kind')}/{record.get('ordinal')}: "
                f"predicted_n={record.get('predicted_n')} requested_n={record.get('requested_n')}"
            )

    for mode in ("target", "spec"):
        for kind in ("sequential", "stress"):
            selected = [
                record
                for record in records
                if record["mode"] == mode and record["kind"] == kind
            ]
            hashes = {str(record["sha256"]) for record in selected}
            if len(hashes) != 1:
                failures.append(f"{mode}/{kind}: {len(hashes)} distinct output hashes")
            print(
                "CORRECTNESS",
                mode,
                kind,
                f"n={len(selected)}",
                f"hashes={len(hashes)}",
                f"sha256={next(iter(hashes)) if len(hashes) == 1 else 'MISMATCH'}",
            )

    sequential_tps = values_by(records, "sequential", lambda record: float(record["tps"]))
    sequential_cpu = values_by(
        records, "sequential", lambda record: float(record["batch_cpu_cores"])
    )

    print("\nSEQUENTIAL")
    for mode in ("target", "spec"):
        for arm in ("wait", "poll"):
            describe(f"{mode}/{arm}/tps", sequential_tps[(mode, arm)], " tok/s")
            describe(f"{mode}/{arm}/cpu", sequential_cpu[(mode, arm)], " cores")

        wait = sequential_tps[(mode, "wait")]
        poll = sequential_tps[(mode, "poll")]
        low, high = bootstrap_delta(wait, poll, seed=101 if mode == "target" else 202)
        delta = 100.0 * (statistics.fmean(poll) / statistics.fmean(wait) - 1.0)

        instance_values: dict[str, list[float]] = defaultdict(list)
        for record in records:
            if record["mode"] == mode and record["kind"] == "sequential":
                instance_values[str(record["run_id"])].append(float(record["tps"]))
        wait_instances = [
            statistics.fmean(values)
            for run_id, values in instance_values.items()
            if run_id.endswith("-wait")
        ]
        poll_instances = [
            statistics.fmean(values)
            for run_id, values in instance_values.items()
            if run_id.endswith("-poll")
        ]
        cluster_low, cluster_high = bootstrap_delta(
            wait_instances, poll_instances, seed=303 if mode == "target" else 404
        )
        print(
            "DELTA",
            mode,
            f"poll_vs_wait_pct={delta:.4f}",
            f"raw_bootstrap_95=[{low:.4f},{high:.4f}]",
            f"instance_bootstrap_95=[{cluster_low:.4f},{cluster_high:.4f}]",
            f"instances={len(wait_instances)}+{len(poll_instances)}",
        )

    batches: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["kind"] == "stress":
            batches[str(record["batch_id"])].append(record)

    stress_tps: dict[tuple[str, str], list[float]] = defaultdict(list)
    stress_cpu: dict[tuple[str, str], list[float]] = defaultdict(list)
    for batch in batches.values():
        first = batch[0]
        key = (str(first["mode"]), str(first["arm"]))
        wall_s = float(first["batch_wall_s"])
        stress_tps[key].append(sum(int(record["predicted_n"]) for record in batch) / wall_s)
        stress_cpu[key].append(float(first["batch_cpu_cores"]))

    print("\nFOUR_LANE_STRESS")
    for mode in ("target", "spec"):
        for arm in ("wait", "poll"):
            describe(f"{mode}/{arm}/aggregate_tps", stress_tps[(mode, arm)], " tok/s")
            describe(f"{mode}/{arm}/cpu", stress_cpu[(mode, arm)], " cores")
        delta = 100.0 * (
            statistics.fmean(stress_tps[(mode, "poll")])
            / statistics.fmean(stress_tps[(mode, "wait")])
            - 1.0
        )
        low, high = bootstrap_delta(
            stress_tps[(mode, "wait")],
            stress_tps[(mode, "poll")],
            seed=505 if mode == "target" else 606,
        )
        print(
            "STRESS_DELTA",
            mode,
            f"poll_vs_wait_pct={delta:.4f}",
            f"batch_bootstrap_95=[{low:.4f},{high:.4f}]",
        )

    print("\nVALIDATION", "PASS" if not failures else "FAIL")
    for failure in failures:
        print("FAILURE", failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

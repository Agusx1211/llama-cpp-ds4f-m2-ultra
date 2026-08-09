#!/usr/bin/env python3

"""Summarize HOSTPROF cbw Metal completion-to-host-wake records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[int], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []

    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.startswith("HOSTPROF "):
                continue
            try:
                value = json.loads(line[len("HOSTPROF ") :])
            except json.JSONDecodeError:
                continue

            if value.get("op") == "cbw":
                value["line"] = line_number
                pending.append(value)
            elif value.get("op") == "sync" and pending:
                context = str(value.get("c", "unknown"))
                for record in pending:
                    record["context"] = context
                    record["sync_wait_us"] = value.get("wait")
                    records.append(record)
                pending.clear()

    for record in pending:
        record["context"] = "unpaired"
        records.append(record)
    return records


def summarize(name: str, records: list[dict[str, object]]) -> None:
    wakes = [int(record["wake"]) for record in records]
    waits = [int(record["wait"]) for record in records]
    print(
        name,
        f"n={len(records)}",
        f"wake_mean_us={statistics.fmean(wakes):.3f}",
        f"wake_stdev_us={statistics.pstdev(wakes):.3f}",
        f"wake_min_us={min(wakes)}",
        f"wake_p05_us={percentile(wakes, 0.05):.3f}",
        f"wake_p50_us={percentile(wakes, 0.50):.3f}",
        f"wake_p95_us={percentile(wakes, 0.95):.3f}",
        f"wake_p99_us={percentile(wakes, 0.99):.3f}",
        f"wake_max_us={max(wakes)}",
        f"wait_p50_us={percentile(waits, 0.50):.3f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--min-wait-us", type=int, default=100)
    parser.add_argument("--max-wait-us", type=int, default=200_000)
    parser.add_argument("--skip-per-context", type=int, default=8)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    records = [
        record
        for record in parse(args.log)
        if int(record.get("status", -1)) == 4
        and int(record.get("wake", -1)) >= 0
        and args.min_wait_us <= int(record.get("wait", -1)) <= args.max_wait_us
    ]

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["context"])].append(record)

    retained: list[dict[str, object]] = []
    for context, values in sorted(grouped.items()):
        selected = values[args.skip_per_context :]
        if selected:
            summarize(f"context={context}", selected)
            retained.extend(selected)

    if retained:
        summarize("all", retained)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ["line", "context", "t", "wait", "ge", "wake", "status", "sync_wait_us"]
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(retained)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the opt-in exact DeepSeek V4 resident gate in the foreground.

The queue invokes this wrapper as one bounded job.  It keeps the model process
attached to the caller, captures stdout/stderr in a retained artifact directory,
and terminates the whole process group on timeout.  The C++ gate remains the
authority for model identity, Metal device, correctness, and accounting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_TIMEOUT_SECONDS = 7200


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, help="test-dsv4-resident-model executable")
    parser.add_argument("--model", required=True, help="canonical first GGUF shard")
    parser.add_argument("--manifest", required=True, help="tracked pinned model manifest")
    parser.add_argument("--artifacts-dir", required=True, help="retained stdout/stderr/status directory")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def terminate_group(process: subprocess.Popen[bytes], grace_seconds: float = 10.0) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    artifacts = Path(args.artifacts_dir).expanduser()
    artifacts.mkdir(parents=True, exist_ok=False)
    stdout_path = artifacts / "stdout.log"
    stderr_path = artifacts / "stderr.log"
    status_path = artifacts / "status.json"
    command = [args.executable, "--model", args.model, "--manifest", args.manifest]
    environment = os.environ.copy()
    environment["LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE"] = "1"
    environment["LLAMA_DSV4_AGGREGATE_POOL_FORCE"] = "1"
    environment.pop("LLAMA_DSV4_AMX_COEXEC", None)
    started = dt.datetime.now(dt.timezone.utc)
    start_monotonic = time.monotonic()
    record: dict[str, object] = {
        "status": "started",
        "started_utc": started.isoformat(),
        "command": command,
        "timeout_seconds": args.timeout_seconds,
        "model": str(Path(args.model).expanduser()),
        "manifest": str(Path(args.manifest).expanduser()),
    }
    (artifacts / "command.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    timed_out = False
    return_code: int
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process)
                return_code = 124
    except OSError as error:
        stderr_path.write_text(f"runner failed to start child: {error}\n", encoding="utf-8")
        return_code = 127

    elapsed = time.monotonic() - start_monotonic
    record.update(
        {
            "status": "timeout" if timed_out else ("pass" if return_code == 0 else "fail"),
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    status_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"dsv4-resident-model-gate status={record['status']} return_code={return_code} "
        f"elapsed_seconds={elapsed:.3f} artifacts={artifacts}",
        file=sys.stderr,
        flush=True,
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

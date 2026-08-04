#!/usr/bin/env python3
"""Run the opt-in exact DeepSeek V4 resident gate in the foreground.

The queue invokes this wrapper as one bounded job.  It keeps the model process
attached to the caller, captures stdout/stderr in a retained artifact directory,
forwards SIGINT/SIGTERM to the child process group, and terminates the whole
process group on timeout.  The C++ gate remains the authority for model
identity, Metal device, correctness, and accounting.
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

_ACTIVE_PROCESS: subprocess.Popen[bytes] | None = None
_FORWARDED_SIGNAL: int | None = None


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


def process_group_exists(process: subprocess.Popen[bytes]) -> bool:
    """Return whether the child session still has any process in its group."""
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The group exists but is not probeable by policy.  Cleanup must stay
        # conservative and escalate rather than assume it disappeared.
        return True
    return True


def terminate_group(
    process: subprocess.Popen[bytes], signal_number: int = signal.SIGTERM, grace_seconds: float = 10.0
) -> None:
    """Signal and reap the entire child session with bounded escalation.

    ``Popen.wait()`` only observes the leader.  A leader can exit while a
    descendant keeps the process group alive, so probe the group explicitly
    before returning and escalate that group to SIGKILL after the grace
    period.  The short post-kill wait is bounded as well; this avoids hanging
    the foreground queue job on a descendant that ignores SIGTERM.
    """
    grace_seconds = max(0.0, grace_seconds)

    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass

    deadline = time.monotonic() + grace_seconds
    while process_group_exists(process) and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        elif remaining > 0:
            time.sleep(min(0.05, remaining))

    if process_group_exists(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    # Reap the leader without an unbounded wait.  SIGKILL should make this
    # immediate; a timeout is an actionable runner failure rather than an
    # opportunity to hang indefinitely.
    if process.poll() is None:
        try:
            process.wait(timeout=max(1.0, grace_seconds))
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("child process did not exit after process-group SIGKILL") from error

    # The leader may have exited before the initial signal.  Probe once more
    # and issue a final SIGKILL to descendants that appeared during teardown.
    if process_group_exists(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def forward_signal(signum: int, _frame: object) -> None:
    """Forward parent SIGINT/SIGTERM to the child process group and reap it."""
    global _FORWARDED_SIGNAL
    if _FORWARDED_SIGNAL is not None:
        return
    _FORWARDED_SIGNAL = signum
    process = _ACTIVE_PROCESS
    if process is not None:
        terminate_group(process, signal_number=signum)


def main(argv: list[str]) -> int:
    global _ACTIVE_PROCESS, _FORWARDED_SIGNAL
    _ACTIVE_PROCESS = None
    _FORWARDED_SIGNAL = None
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
    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, signal.Handlers] = {}
    try:
        # Install handlers before Popen so a signal delivered during process
        # creation cannot leave an untracked child running.  The post-Popen
        # forwarded-signal check closes the remaining assignment window.
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            _ACTIVE_PROCESS = process
            try:
                if _FORWARDED_SIGNAL is not None:
                    terminate_group(process, signal_number=_FORWARDED_SIGNAL)
                    return_code = 128 + _FORWARDED_SIGNAL
                else:
                    return_code = process.wait(timeout=args.timeout_seconds)
                    if process_group_exists(process):
                        # A successful leader exit does not prove the child
                        # session is empty.  Reap descendants left behind by
                        # a fork/daemon-style child before returning pass.
                        terminate_group(process)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_group(process)
                return_code = 124
            finally:
                if _FORWARDED_SIGNAL is not None:
                    terminate_group(process, signal_number=_FORWARDED_SIGNAL)
                    return_code = 128 + _FORWARDED_SIGNAL
                _ACTIVE_PROCESS = None
    except OSError as error:
        stderr_path.write_text(f"runner failed to start child: {error}\n", encoding="utf-8")
        return_code = 127
    finally:
        _ACTIVE_PROCESS = None
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    # A signal can arrive in the narrow restoration window after child-group
    # cleanup.  Preserve the interruption result even though the active
    # handler no longer has a process to terminate.
    if _FORWARDED_SIGNAL is not None and process is not None:
        return_code = 128 + _FORWARDED_SIGNAL

    elapsed = time.monotonic() - start_monotonic
    record.update(
        {
            "status": "timeout" if timed_out else ("pass" if return_code == 0 else "fail"),
            "return_code": return_code,
            "elapsed_seconds": elapsed,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    if _FORWARDED_SIGNAL is not None:
        record["forwarded_signal"] = _FORWARDED_SIGNAL
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

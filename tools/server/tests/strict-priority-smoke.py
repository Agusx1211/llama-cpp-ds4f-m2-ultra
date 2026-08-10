#!/usr/bin/env python3
# Signed-by: openai-codex/gpt-5.6-sol
# Date: 2026-08-10 UTC

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import threading
import time
import traceback
import urllib.request
from typing import Any

MODEL_BY_LANE = {
    "low": "deepseek-v4-flash-slow",
    "normal": "deepseek-v4-flash",
    "fast": "deepseek-v4-flash-fast",
}


def now_ns() -> int:
    return time.monotonic_ns()


def numeric_prompt(count: int, salt: int) -> list[int]:
    return [1000 + ((131 * index + salt) % 10000) for index in range(count)]


@dataclasses.dataclass
class Result:
    tag: str
    lane: str
    prompt_tokens: int
    requested_tokens: int
    start_ns: int = 0
    first_token_ns: int = 0
    end_ns: int = 0
    status: int = 0
    reported_tokens_predicted: int = -1
    token_times_ns: list[int] = dataclasses.field(default_factory=list)
    progress_times_ns: list[int] = dataclasses.field(default_factory=list)
    error: str = ""
    first_token: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)
    done: threading.Event = dataclasses.field(default_factory=threading.Event, repr=False)

    def summary(self) -> dict[str, Any]:
        gaps = [
            (right - left) / 1e6
            for left, right in zip(self.token_times_ns, self.token_times_ns[1:])
        ]
        return {
            "tag": self.tag,
            "lane": self.lane,
            "prompt_tokens": self.prompt_tokens,
            "requested_tokens": self.requested_tokens,
            "generated_tokens": len(self.token_times_ns),
            "reported_tokens_predicted": self.reported_tokens_predicted,
            "status": self.status,
            "start_ns": self.start_ns,
            "first_token_ns": self.first_token_ns,
            "end_ns": self.end_ns,
            "ttft_ms": (self.first_token_ns - self.start_ns) / 1e6 if self.first_token_ns else None,
            "elapsed_seconds": (self.end_ns - self.start_ns) / 1e9,
            "progress_events": len(self.progress_times_ns),
            "first_progress_ns": self.progress_times_ns[0] if self.progress_times_ns else 0,
            "last_progress_ns": self.progress_times_ns[-1] if self.progress_times_ns else 0,
            "maximum_token_gap_ms": max(gaps, default=0.0),
            "error": self.error,
        }


def run_request(base_url: str, api_key: str, result: Result, salt: int, timeout: int) -> None:
    body = {
        "model": MODEL_BY_LANE[result.lane],
        "prompt": numeric_prompt(result.prompt_tokens, salt),
        "n_predict": result.requested_tokens,
        "stream": True,
        "return_tokens": True,
        "return_progress": True,
        "cache": False,
        "cache_prompt": False,
        "temperature": 0.0,
        "seed": 42,
        "ignore_eos": True,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/completion",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    result.start_ns = now_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result.status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                decoded = json.loads(payload)
                for item in decoded if isinstance(decoded, list) else [decoded]:
                    timestamp = now_ns()
                    if isinstance(item.get("prompt_progress"), dict):
                        result.progress_times_ns.append(timestamp)
                        continue
                    tokens = item.get("tokens") or []
                    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                        raise RuntimeError(f"{result.tag}: response contained invalid token IDs")
                    if tokens:
                        result.token_times_ns.extend([timestamp] * len(tokens))
                        if not result.first_token_ns:
                            result.first_token_ns = timestamp
                            result.first_token.set()
                    if item.get("tokens_predicted") is not None:
                        result.reported_tokens_predicted = int(item["tokens_predicted"])
    except Exception as error:
        result.error = f"{type(error).__name__}: {error}"
    finally:
        result.end_ns = now_ns()
        result.done.set()


def wait_for_first_token(result: Result, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result.first_token.wait(0.05):
            return
        if result.done.is_set():
            raise RuntimeError(f"{result.tag} ended before its first token: {result.error}")
    raise RuntimeError(f"timed out waiting for {result.tag} first token")


def require_complete(result: Result) -> None:
    if result.error or result.status != 200:
        raise RuntimeError(f"{result.tag} failed: HTTP {result.status}; {result.error}")
    if len(result.token_times_ns) != result.requested_tokens:
        raise RuntimeError(
            f"{result.tag} generated {len(result.token_times_ns)} tokens, expected {result.requested_tokens}"
        )
    if result.reported_tokens_predicted != result.requested_tokens:
        raise RuntimeError(
            f"{result.tag} reported {result.reported_tokens_predicted} tokens, expected {result.requested_tokens}"
        )


def run_pair(
    *, base_url: str, api_key: str, lower_lane: str, higher_lane: str,
    prompt_tokens: int, higher_tokens: int, lower_tokens: int, timeout: int,
) -> tuple[dict[str, Any], list[str]]:
    name = f"{lower_lane}-{higher_lane}"
    lower = Result(f"{name}-incumbent", lower_lane, 128, lower_tokens)
    higher = Result(f"{name}-arrival", higher_lane, prompt_tokens, higher_tokens)

    lower_thread = threading.Thread(
        target=run_request, args=(base_url, api_key, lower, 1001, timeout), daemon=False)
    lower_thread.start()
    wait_for_first_token(lower, timeout)

    higher_thread = threading.Thread(
        target=run_request, args=(base_url, api_key, higher, 2001, timeout), daemon=False)
    higher_thread.start()
    higher_thread.join()
    lower_thread.join()
    require_complete(lower)
    require_complete(higher)

    checks: list[str] = []
    violations: list[str] = []
    if not higher.progress_times_ns:
        violations.append("arrival did not stream prompt progress")
        prefill_start_ns = higher.start_ns
    else:
        prefill_start_ns = higher.progress_times_ns[0]
        checks.append("arrival streamed prompt progress")

    incumbent_during_prefill = sum(
        prefill_start_ns <= timestamp < higher.first_token_ns
        for timestamp in lower.token_times_ns
    )
    if incumbent_during_prefill == 0:
        checks.append("incumbent decode stalled from first prompt progress through arrival TTFT")
    else:
        violations.append(f"incumbent emitted {incumbent_during_prefill} tokens during arrival prefill")

    overlap_end_ns = min(lower.end_ns, higher.end_ns)
    lower_overlap_tokens = sum(higher.first_token_ns <= timestamp < overlap_end_ns for timestamp in lower.token_times_ns)
    higher_overlap_tokens = sum(higher.first_token_ns <= timestamp < overlap_end_ns for timestamp in higher.token_times_ns)
    if lower_overlap_tokens > 0 and higher_overlap_tokens > 0:
        checks.append("both requests decoded after arrival prefill")
    else:
        violations.append(
            f"missing concurrent decode evidence: incumbent={lower_overlap_tokens}, arrival={higher_overlap_tokens}"
        )

    if lower_lane != higher_lane:
        lower_during_higher = sum(
            higher.first_token_ns <= timestamp < higher.end_ns
            for timestamp in lower.token_times_ns
        )
        ratio = lower_during_higher / max(1, len(higher.token_times_ns))
        if lower_during_higher > 0:
            checks.append("lower lane received keepalive decode service")
        else:
            violations.append("lower lane received no keepalive service while fast decode was active")
        if ratio <= 0.10:
            checks.append("lower lane service stayed below ten percent of dominant output")
        else:
            violations.append(f"lower lane service ratio {ratio:.4f} exceeded 0.10")
    else:
        lower_during_higher = lower_overlap_tokens
        ratio = lower_overlap_tokens / max(1, higher_overlap_tokens)

    return {
        "name": name,
        "checks": checks,
        "violations": violations,
        "incumbent_tokens_during_prefill": incumbent_during_prefill,
        "incumbent_tokens_during_arrival_decode": lower_during_higher,
        "incumbent_to_arrival_output_ratio": ratio,
        "prefill_stall_ms": (higher.first_token_ns - prefill_start_ns) / 1e6,
        "incumbent": lower.summary(),
        "arrival": higher.summary(),
    }, violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise strict lane preemption against a live llama-server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key", default="llamacpp")
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=4096)
    parser.add_argument("--higher-tokens", type=int, default=512)
    parser.add_argument("--lower-tokens", type=int, default=128)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--observe-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.prompt_tokens, args.higher_tokens, args.lower_tokens, args.timeout_seconds) <= 0:
        raise ValueError("token counts and timeout must be positive")
    args.artifact.mkdir(parents=True, exist_ok=False)

    scenarios = []
    violations = []
    for lower, higher in (("fast", "fast"), ("low", "fast"), ("normal", "fast")):
        scenario, scenario_violations = run_pair(
            base_url=args.base_url,
            api_key=args.api_key,
            lower_lane=lower,
            higher_lane=higher,
            prompt_tokens=args.prompt_tokens,
            higher_tokens=args.higher_tokens,
            lower_tokens=args.lower_tokens,
            timeout=args.timeout_seconds,
        )
        scenarios.append(scenario)
        violations.extend(f"{scenario['name']}: {item}" for item in scenario_violations)

    summary = {
        "schema": 1,
        "status": "OBSERVED" if args.observe_only else ("PASS" if not violations else "FAIL"),
        "observe_only": args.observe_only,
        "violations": violations,
        "scenarios": scenarios,
    }
    output = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (args.artifact / "summary.json").write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if args.observe_only or not violations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

#!/usr/bin/env python3
# Signed-by: GPT-5
# Date: 2026-08-04 UTC

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any

TOKEN_HEADER = "X-Llama-Trusted-Scheduling-Token"
LANE_HEADER = "X-Llama-Trusted-Lane"
TAG_HEADER = "X-Llama-Benchmark-Tag"

SEQUENTIAL_BURST_MODE = "sequential-bursts"
ORDINARY_PROMPT_TOKENS = 17
SEQUENTIAL_BURST_COUNT = 4
TRACE_EVENTS_PER_PROMPT_TOKEN = 4
TRACE_EVENT_HEADROOM = 32


def now_ns() -> int:
    return time.monotonic_ns()


def numeric_prompt(count: int, salt: int) -> list[int]:
    return [1000 + ((131 * index + salt) % 10000) for index in range(count)]


def token_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest()


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
    tokens: list[int] = dataclasses.field(default_factory=list)
    content_parts: list[str] = dataclasses.field(default_factory=list)
    progress: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    error: str = ""
    intended_launch_ns: int = 0

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    def summary(self, retain_ids: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tag": self.tag,
            "lane": self.lane,
            "prompt_tokens": self.prompt_tokens,
            "requested_tokens": self.requested_tokens,
            "reported_tokens_predicted": self.reported_tokens_predicted,
            "status": self.status,
            "start_ns": self.start_ns,
            "first_token_ns": self.first_token_ns,
            "end_ns": self.end_ns,
            "elapsed_seconds": (self.end_ns - self.start_ns) / 1e9,
            "generated_tokens": len(self.tokens),
            "token_sha256": token_hash(self.tokens),
            "content_sha256": hashlib.sha256(self.content.encode("utf-8")).hexdigest(),
            "progress_events": len(self.progress),
            "error": self.error,
        }
        if retain_ids:
            result["token_ids"] = self.tokens
        if self.intended_launch_ns:
            result.update({
                "intended_launch_ns": self.intended_launch_ns,
                "actual_launch_ns": self.start_ns,
                "launch_lag_ms": (self.start_ns - self.intended_launch_ns) / 1e6,
                "ttft_ms": (self.first_token_ns - self.start_ns) / 1e6 if self.first_token_ns else None,
            })
        return result


def sequential_burst_mode(config: dict[str, Any]) -> bool:
    return config.get("mode") == SEQUENTIAL_BURST_MODE


def expected_trusted_requests(config: dict[str, Any]) -> dict[str, tuple[str, int]]:
    requests = config["requests"]
    reference = requests["reference"]
    low = requests["trusted_low"]
    expected = {
        "reference-before": ("fast", int(reference["prompt_tokens"])),
        "reference-after": ("fast", int(reference["prompt_tokens"])),
        "low-8k": ("low", int(low["prompt_tokens"])),
    }
    if sequential_burst_mode(config):
        bursts = requests["fast_bursts"]
        expected.update({
            "isolated-low": ("low", int(low["prompt_tokens"])),
            "isolated-burst": ("fast", int(bursts["prompt_tokens"])),
            **{
                f"burst-{ordinal:02d}": ("fast", int(bursts["prompt_tokens"]))
                for ordinal in range(int(bursts["count"]))
            },
        })
    else:
        fast = requests["isolated_and_sustained_fast"]
        expected.update({
            "isolated-fast": ("fast", int(fast["prompt_tokens"])),
            "sustained-fast": ("fast", int(fast["prompt_tokens"])),
        })
    return expected


def required_trace_capacity(config: dict[str, Any]) -> int:
    # A maximally fragmented prompt can select/release an owner and stage/commit
    # once per token. The fixed margin covers authenticated snapshots and any
    # owner handoff that occurs without advancing a prompt cursor.
    expected = expected_trusted_requests(config)
    prompt_tokens = ORDINARY_PROMPT_TOKENS + sum(prompt for _, prompt in expected.values())
    registrations = 1 + len(expected)
    return TRACE_EVENT_HEADROOM + registrations + TRACE_EVENTS_PER_PROMPT_TOKEN * prompt_tokens


def validate_trace_preflight(snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, int]:
    if snapshot.get("schema") != 1 or snapshot.get("overflow_events") != 0:
        raise RuntimeError("trace preflight schema/overflow gate failed")
    events = snapshot.get("events", [])
    if snapshot.get("total_events") != 0 or events:
        raise RuntimeError("sequential burst smoke requires an empty scheduler trace")
    required = required_trace_capacity(config)
    capacity = int(snapshot.get("capacity", 0))
    if capacity < required:
        raise RuntimeError(
            f"scheduler trace capacity={capacity} is below required worst-case capacity={required}")
    return {"capacity": capacity, "required_capacity": required}


def require_result(result: Result) -> None:
    if result.error or result.status != 200:
        raise RuntimeError(f"request status/error gate failed: {result.summary(False)}")
    if result.reported_tokens_predicted != result.requested_tokens:
        raise RuntimeError(
            f"{result.tag}: tokens_predicted={result.reported_tokens_predicted}, expected={result.requested_tokens}")
    if len(result.tokens) != result.requested_tokens:
        raise RuntimeError(
            f"{result.tag}: token ID count={len(result.tokens)}, expected={result.requested_tokens}")
    if not result.content:
        raise RuntimeError(f"{result.tag}: empty output content")


def require_exact_output(results: dict[str, Result], left: str, right: str) -> None:
    if (results[left].tokens, results[left].content) != (results[right].tokens, results[right].content):
        raise RuntimeError(f"exact output mismatch: {left} vs {right}")


def validate_sequential_burst_results(results: dict[str, Result], count: int) -> list[dict[str, Any]]:
    require_exact_output(results, "isolated-low", "low-8k")
    low = results["low-8k"]
    if not low.progress:
        raise RuntimeError("mixed low request returned no prompt progress")

    timings = []
    progress_times = [int(event["client_monotonic_ns"]) for event in low.progress]
    for ordinal in range(count):
        tag = f"burst-{ordinal:02d}"
        burst = results[tag]
        require_exact_output(results, "isolated-burst", tag)
        if not burst.intended_launch_ns or burst.start_ns < burst.intended_launch_ns:
            raise RuntimeError(f"{tag}: invalid intended/actual launch timestamps")
        if not burst.first_token_ns or burst.first_token_ns < burst.start_ns or burst.first_token_ns >= low.end_ns:
            raise RuntimeError(f"{tag}: first output did not precede low completion")
        if not burst.end_ns or burst.end_ns < burst.first_token_ns or burst.end_ns >= low.end_ns:
            raise RuntimeError(f"{tag}: HTTP request did not complete before low completion")
        progress_after = None
        if ordinal + 1 < count:
            next_burst = results[f"burst-{ordinal + 1:02d}"]
            if next_burst.start_ns < burst.end_ns:
                raise RuntimeError(f"{tag}: next burst launched before HTTP teardown")
            progress_cutoff_ns = max(burst.end_ns, next_burst.start_ns)
            progress_deadline_ns = next_burst.end_ns
            progress_after = next(
                (timestamp for timestamp in progress_times
                 if progress_cutoff_ns < timestamp < progress_deadline_ns), None)
            if progress_after is None:
                raise RuntimeError(f"{tag}: no fresh low prompt progress during next burst")
        timing = {
            "tag": tag,
            "intended_launch_ns": burst.intended_launch_ns,
            "actual_launch_ns": burst.start_ns,
            "first_token_ns": burst.first_token_ns,
            "launch_lag_ms": (burst.start_ns - burst.intended_launch_ns) / 1e6,
            "ttft_ms": (burst.first_token_ns - burst.start_ns) / 1e6,
            "overlap_lead_ms": (low.end_ns - burst.first_token_ns) / 1e6,
            "completion_lead_ms": (low.end_ns - burst.end_ns) / 1e6,
        }
        if progress_after is not None:
            timing["low_progress_during_next_burst_ns"] = progress_after
        timings.append(timing)
    return timings


class Smoke:
    def __init__(self, config: dict[str, Any], base_url: str, artifact: pathlib.Path):
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.artifact = artifact
        self.api_key = os.environ["API_BEARER_KEY"]
        self.scheduling_token = os.environ["LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN"]
        if self.api_key == self.scheduling_token:
            raise RuntimeError("API bearer and scheduling token are not distinct")
        self.common = config["requests"]["common"]
        self.timeout = int(config["requests"]["timeout_seconds"])
        self.results: dict[str, Result] = {}
        self.results_lock = threading.Lock()
        self.low_progress = threading.Event()
        self.low_done = threading.Event()
        self.sustained_first_token = threading.Event()
        self.ordinary_request_id = 0
        self.trace_preflight: dict[str, int] | None = None
        self.events = (artifact / "client-events.jsonl").open("w", encoding="utf-8", buffering=1)
        self.events_lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, "client_monotonic_ns": now_ns(), **fields}
        with self.events_lock:
            self.events.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def api_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def trusted_headers(self, lane: str, tag: str) -> dict[str, str]:
        return {
            **self.api_headers(),
            TOKEN_HEADER: self.scheduling_token,
            LANE_HEADER: lane,
            TAG_HEADER: tag,
        }

    def fetch_trace(self, operator_token: bool = True) -> dict[str, Any]:
        headers = self.api_headers()
        if operator_token:
            headers[TOKEN_HEADER] = self.scheduling_token
        request = urllib.request.Request(
            self.base_url + self.config["trusted_contract"]["trace_endpoint"], headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def expect_http_error(self, request: urllib.request.Request, label: str, expected_status: int) -> None:
        try:
            urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            (self.artifact / f"{label}-response.json").write_bytes(error.read())
            if error.code != expected_status:
                raise RuntimeError(f"{label}: HTTP {error.code}, expected {expected_status}") from error
            self.emit("expected_rejection", label=label, status=expected_status)
            return
        raise RuntimeError(f"{label} unexpectedly succeeded")

    def expect_403(self, request: urllib.request.Request, label: str) -> None:
        self.expect_http_error(request, label, 403)

    def credential_probes(self) -> None:
        trace_request = urllib.request.Request(
            self.base_url + self.config["trusted_contract"]["trace_endpoint"],
            headers=self.api_headers())
        self.expect_403(trace_request, "api-only-trace")
        before = self.fetch_trace()
        body = {**self.common, "prompt": numeric_prompt(1, 811), "n_predict": 1}
        encoded = json.dumps(body, separators=(",", ":")).encode()
        scheduling_only = urllib.request.Request(
            self.base_url + "/completion",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                TOKEN_HEADER: self.scheduling_token,
                LANE_HEADER: "fast",
                TAG_HEADER: "scheduling-only-reject",
            },
            method="POST")
        self.expect_http_error(scheduling_only, "scheduling-only-lane", 401)
        request = urllib.request.Request(
            self.base_url + "/completion",
            data=encoded,
            headers={
                "Content-Type": "application/json",
                **self.api_headers(),
                LANE_HEADER: "fast",
                TAG_HEADER: "api-only-reject",
            },
            method="POST")
        self.expect_403(request, "api-only-lane")
        if self.fetch_trace()["total_events"] != before["total_events"]:
            raise RuntimeError("single-credential rejection registered scheduler work")

    def run_request(
        self, *, tag: str, lane: str, prompt_count: int, n_predict: int,
        salt: int, seed: int = 42, trusted: bool = True, forge_body: bool = False,
        intended_launch_ns: int = 0,
    ) -> Result:
        result = Result(tag, lane, prompt_count, n_predict, intended_launch_ns=intended_launch_ns)
        body: dict[str, Any] = {
            **self.common,
            "prompt": numeric_prompt(prompt_count, salt),
            "n_predict": n_predict,
            "seed": seed,
        }
        if forge_body:
            body.update({"lane": "fast", "priority": "fast", "scheduling": {"lane": "fast"}})
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        (self.artifact / f"{tag}-request.json").write_bytes(encoded + b"\n")
        headers = {"Content-Type": "application/json"}
        headers.update(self.trusted_headers(lane, tag) if trusted else self.api_headers())
        request = urllib.request.Request(
            self.base_url + "/completion", data=encoded, headers=headers, method="POST")
        result.start_ns = now_ns()
        start_fields = {
            "tag": tag, "lane": lane, "prompt_tokens": prompt_count,
            "n_predict": n_predict, "trusted": trusted,
        }
        if intended_launch_ns:
            start_fields.update({
                "intended_launch_ns": intended_launch_ns,
                "actual_launch_ns": result.start_ns,
                "launch_lag_ms": (result.start_ns - intended_launch_ns) / 1e6,
            })
        self.emit("request_start", **start_fields)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result.status = response.status
                with (self.artifact / f"{tag}-response.sse").open("wb") as output:
                    for raw_line in response:
                        output.write(raw_line)
                        line = raw_line.decode("utf-8", errors="strict").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        decoded = json.loads(payload)
                        for item in decoded if isinstance(decoded, list) else [decoded]:
                            timestamp = now_ns()
                            progress = item.get("prompt_progress")
                            if isinstance(progress, dict):
                                if item.get("tokens") not in (None, []):
                                    raise RuntimeError(f"{tag}: prompt-progress frame returned generated tokens")
                                if item.get("content") not in (None, ""):
                                    raise RuntimeError(f"{tag}: prompt-progress frame returned generated content")
                                saved = {**progress, "client_monotonic_ns": timestamp}
                                result.progress.append(saved)
                                self.emit("prompt_progress", tag=tag, **progress)
                                if tag == "low-8k":
                                    self.low_progress.set()
                                continue
                            tokens = item.get("tokens") or []
                            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                                raise RuntimeError(f"{tag}: invalid token array")
                            if tokens:
                                if not result.first_token_ns:
                                    result.first_token_ns = timestamp
                                    if tag == "sustained-fast":
                                        self.sustained_first_token.set()
                                result.tokens.extend(tokens)
                            reported = item.get("tokens_predicted")
                            if reported is not None:
                                result.reported_tokens_predicted = int(reported)
                            content = item.get("content")
                            if content is not None:
                                if not isinstance(content, str):
                                    raise RuntimeError(f"{tag}: non-string content")
                                result.content_parts.append(content)
        except Exception as error:
            result.error = f"{type(error).__name__}: {error}"
        finally:
            result.end_ns = now_ns()
            if tag == "low-8k":
                self.low_done.set()
            with self.results_lock:
                self.results[tag] = result
            self.emit("request_end", **result.summary(retain_ids=False))
        return result

    def start(self, **kwargs: Any) -> threading.Thread:
        thread = threading.Thread(target=self.run_request, kwargs=kwargs, daemon=False)
        thread.start()
        return thread

    def wait_before_low_done(self, event: threading.Event, label: str) -> None:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if event.wait(0.05):
                return
            if self.low_done.is_set():
                raise RuntimeError(f"low completed before {label}")
        raise RuntimeError(f"timeout waiting for {label}")

    def run_pipelined_bursts(self, bursts: dict[str, Any]) -> Result:
        last: Result | None = None
        for ordinal in range(int(bursts["count"])):
            if self.low_done.is_set():
                raise RuntimeError(f"low completed before burst-{ordinal:02d} launch")
            intended = now_ns()
            burst = self.run_request(
                tag=f"burst-{ordinal:02d}", lane="fast", prompt_count=bursts["prompt_tokens"],
                n_predict=bursts["n_predict"], salt=bursts["salt"], seed=bursts["seed"],
                intended_launch_ns=intended)
            require_result(burst)
            if ordinal + 1 < int(bursts["count"]) and self.low_done.is_set():
                raise RuntimeError(f"low completed before burst-{ordinal + 1:02d} launch")
            last = burst
        if last is None:
            raise RuntimeError("sequential burst mode requires at least one burst")
        return last

    def ordinary_probe(self) -> None:
        before = self.fetch_trace()
        result = self.run_request(
            tag="ordinary-forgery", lane="normal", prompt_count=ORDINARY_PROMPT_TOKENS, n_predict=1,
            salt=901, trusted=False, forge_body=True)
        self.require_result(result)
        after = self.fetch_trace()
        registrations = [
            event for event in after["events"]
            if int(event["sequence"]) > int(before["total_events"])
            and event["event"] == "request_registered"
        ]
        if len(registrations) != 1:
            raise RuntimeError("ordinary forgery did not create exactly one registration")
        registration = registrations[0]
        if (registration["lane"], registration["benchmark_tag"], int(registration["prompt_tokens"])) != (
                "normal", "", ORDINARY_PROMPT_TOKENS):
            raise RuntimeError(f"ordinary JSON forged scheduler metadata: {registration}")
        self.ordinary_request_id = int(registration["request_id"])
        (self.artifact / "ordinary-forgery-trace-proof.json").write_text(
            json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def execute(self) -> None:
        if sequential_burst_mode(self.config):
            self.trace_preflight = validate_trace_preflight(self.fetch_trace(), self.config)
        self.credential_probes()
        self.ordinary_probe()
        ref = self.config["requests"]["reference"]
        fast = self.config["requests"]["isolated_and_sustained_fast"]
        low = self.config["requests"]["trusted_low"]
        bursts = self.config["requests"]["fast_bursts"]
        self.run_request(tag="reference-before", lane="fast", prompt_count=ref["prompt_tokens"],
                         n_predict=ref["n_predict"], salt=ref["salt"], seed=ref["seed"])
        if sequential_burst_mode(self.config):
            self.run_request(tag="isolated-low", lane="low", prompt_count=low["prompt_tokens"],
                             n_predict=low["n_predict"], salt=low["salt"], seed=low["seed"])
            self.run_request(tag="isolated-burst", lane="fast", prompt_count=bursts["prompt_tokens"],
                             n_predict=bursts["n_predict"], salt=bursts["salt"], seed=bursts["seed"])
            low_thread = self.start(tag="low-8k", lane="low", prompt_count=low["prompt_tokens"],
                                    n_predict=low["n_predict"], salt=low["salt"], seed=low["seed"])
            try:
                self.wait_before_low_done(self.low_progress, "low prompt progress")
                self.run_pipelined_bursts(bursts)
            finally:
                low_thread.join()
            self.run_request(tag="reference-after", lane="fast", prompt_count=ref["prompt_tokens"],
                             n_predict=ref["n_predict"], salt=ref["salt"], seed=ref["seed"])
            self.validate()
            return

        self.run_request(tag="isolated-fast", lane="fast", prompt_count=fast["prompt_tokens"],
                         n_predict=fast["n_predict"], salt=fast["salt"], seed=fast["seed"])
        low_thread = self.start(tag="low-8k", lane="low", prompt_count=low["prompt_tokens"],
                                n_predict=low["n_predict"], salt=low["salt"], seed=low["seed"])
        self.wait_before_low_done(self.low_progress, "low prompt progress")
        fast_thread = self.start(tag="sustained-fast", lane="fast", prompt_count=fast["prompt_tokens"],
                                 n_predict=fast["n_predict"], salt=fast["salt"], seed=fast["seed"])
        self.wait_before_low_done(self.sustained_first_token, "sustained-fast first output")
        fast_thread.join()
        low_thread.join()
        self.run_request(tag="reference-after", lane="fast", prompt_count=ref["prompt_tokens"],
                         n_predict=ref["n_predict"], salt=ref["salt"], seed=ref["seed"])
        self.validate()

    @staticmethod
    def require_result(result: Result) -> None:
        require_result(result)

    def validate_trace(self, snapshot: dict[str, Any]) -> dict[str, int]:
        events = snapshot.get("events", [])
        if snapshot.get("schema") != 1 or snapshot.get("overflow_events") != 0:
            raise RuntimeError("trace schema/overflow gate failed")
        if snapshot.get("total_events") != len(events):
            raise RuntimeError("trace is not lossless")
        if [event["sequence"] for event in events] != list(range(1, len(events) + 1)):
            raise RuntimeError("trace sequence is not contiguous")
        registrations = [event for event in events if event["event"] == "request_registered"]
        expected = expected_trusted_requests(self.config)
        request_ids: dict[str, int] = {}
        for tag, (lane, prompt_tokens) in expected.items():
            matches = [event for event in registrations if event["benchmark_tag"] == tag]
            if len(matches) != 1 or matches[0]["lane"] != lane or matches[0]["prompt_tokens"] != prompt_tokens:
                raise RuntimeError(f"wrong registration for {tag}: {matches}")
            request_ids[tag] = int(matches[0]["request_id"])
        ordinary = [event for event in registrations if int(event["request_id"]) == self.ordinary_request_id]
        if len(ordinary) != 1 or ordinary[0]["lane"] != "normal" or ordinary[0]["benchmark_tag"]:
            raise RuntimeError("ordinary forgery registration is not normal")
        if len(registrations) != len(expected) + 1:
            raise RuntimeError("unexpected registration cardinality")

        owner: tuple[int, int] | None = None
        staged: dict[int, dict[str, Any]] = {}
        generations: set[int] = set()
        cursors: dict[int, int] = {}
        low_active_fast = 0
        low_large_active_fast = 0
        low_aligned_yields = 0
        alignment = self.config["acceptance"]["prefill_alignment_tokens"]
        priority_limit = self.config["acceptance"]["priority_chunk_limit_tokens"]
        low_id = request_ids["low-8k"]
        for event in events:
            kind = event["event"]
            request_id, cohort_id = int(event["request_id"]), int(event["cohort_id"])
            if kind == "prefill_owner_selected":
                if owner is not None:
                    raise RuntimeError("more than one prefill owner")
                owner = (request_id, cohort_id)
            elif kind == "prefill_chunk_staged":
                generation = int(event["generation"])
                if owner != (request_id, cohort_id) or staged or generation in generations:
                    raise RuntimeError(f"invalid concurrent staged lease: {event}")
                if int(event["begin_token"]) >= int(event["end_token"]):
                    raise RuntimeError(f"empty/reversed staged range: {event}")
                generations.add(generation)
                staged[generation] = event
            elif kind == "prefill_chunk_committed":
                generation = int(event["generation"])
                original = staged.pop(generation, None)
                if original is None or owner != (request_id, cohort_id):
                    raise RuntimeError(f"commit without exact owner/stage: {event}")
                for key in ("request_id", "cohort_id", "generation", "begin_token", "end_token", "lane",
                            "active_decode", "active_decode_lane", "yield_boundary", "completes_prompt"):
                    if original[key] != event[key]:
                        raise RuntimeError(f"stage/commit mismatch generation={generation} key={key}")
                begin, end = int(event["begin_token"]), int(event["end_token"])
                if begin != cursors.get(request_id, 0):
                    raise RuntimeError(f"non-contiguous committed range: {event}")
                cursors[request_id] = end
                if (event["yield_boundary"] and not event["completes_prompt"]
                        and end % alignment):
                    raise RuntimeError(f"non-final yield ends off alignment: {event}")
                if event["active_decode"] and end - begin > priority_limit:
                    raise RuntimeError(f"mixed prefill chunk over priority limit: {event}")
                if event["active_decode"] and event["active_decode_lane"] == "fast" and request_id == low_id:
                    low_active_fast += 1
                    if end - begin > alignment:
                        low_large_active_fast += 1
            elif kind == "prefill_chunk_aborted":
                raise RuntimeError(f"prefill abort: {event}")
            elif kind == "prefill_owner_released":
                if owner != (request_id, cohort_id) or staged:
                    raise RuntimeError(f"owner release mismatch: {event}")
                if event["reason"] not in {"yielded", "completed"}:
                    raise RuntimeError(f"unexpected release reason: {event}")
                if request_id == low_id and event["reason"] == "yielded":
                    end = int(event["end_token"])
                    low_prompt_tokens = self.config["requests"]["trusted_low"]["prompt_tokens"]
                    if not event["yield_boundary"] or not 0 < end < low_prompt_tokens or end % alignment:
                        raise RuntimeError(f"low yielded off alignment: {event}")
                    low_aligned_yields += 1
                owner = None
        if owner is not None or staged:
            raise RuntimeError("trace ends with live owner/stage")
        if low_active_fast < 1 or low_large_active_fast < 1 or low_aligned_yields < 1:
            raise RuntimeError(
                f"missing vertical overlap proof: low_active_fast={low_active_fast}, "
                f"low_large_active_fast={low_large_active_fast}, low_yields={low_aligned_yields}")
        prompt_by_id = {int(event["request_id"]): int(event["prompt_tokens"]) for event in registrations}
        if cursors != prompt_by_id:
            raise RuntimeError(f"committed prompt ranges differ from registrations: {cursors} != {prompt_by_id}")
        return {
            "events": len(events), "registrations": len(registrations),
            "unique_generations": len(generations), "maximum_prefill_owners": 1,
            "low_active_fast_commits": low_active_fast,
            "low_aligned_yielded_releases": low_aligned_yields,
            "low_large_active_fast_commits": low_large_active_fast,
        }

    def exact_pair(self, left: str, right: str) -> None:
        require_exact_output(self.results, left, right)

    def validate(self) -> None:
        count = int(self.config["requests"]["fast_bursts"]["count"])
        expected = {"ordinary-forgery", *expected_trusted_requests(self.config)}
        if set(self.results) != expected:
            raise RuntimeError(f"request cardinality mismatch: {sorted(self.results)}")
        for result in self.results.values():
            self.require_result(result)
        self.exact_pair("reference-before", "reference-after")
        if sequential_burst_mode(self.config):
            burst_timings = validate_sequential_burst_results(self.results, count)
        else:
            low, fast = self.results["low-8k"], self.results["sustained-fast"]
            if not fast.first_token_ns or fast.first_token_ns >= low.end_ns:
                raise RuntimeError("sustained-fast first output did not precede low completion")
            self.exact_pair("isolated-fast", "sustained-fast")
        trace = self.fetch_trace()
        (self.artifact / "scheduler-trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        trace_summary = self.validate_trace(trace)
        if sequential_burst_mode(self.config):
            low = self.results["low-8k"]
            summary = {
                "schema": 1,
                "status": "PASS",
                "mode": SEQUENTIAL_BURST_MODE,
                "checks": {
                    "credential_composition": True,
                    "ordinary_json_forgery_normal": True,
                    "every_burst_completed_before_low_completion": True,
                    "low_progress_across_three_burst_handoffs": True,
                    "exact_tokens_predicted_and_id_cardinality": True,
                    "exact_reference_burst_and_low_outputs": True,
                    "exact_stage_commit_pairs": True,
                    "maximum_one_owner": True,
                    "low_commit_during_fast_decode": True,
                    "low_aligned_yield": True,
                    "trace_capacity_preflight": True,
                },
                "minimum_burst_overlap_lead_ms": min(
                    (low.end_ns - self.results[f"burst-{ordinal:02d}"].first_token_ns) / 1e6
                    for ordinal in range(count)),
                "burst_timings": burst_timings,
                "trace_capacity": self.trace_preflight,
                "requests": {tag: result.summary() for tag, result in sorted(self.results.items())},
                "trace": trace_summary,
            }
        else:
            low, fast = self.results["low-8k"], self.results["sustained-fast"]
            summary = {
                "schema": 1,
                "status": "PASS",
                "checks": {
                    "credential_composition": True,
                    "ordinary_json_forgery_normal": True,
                    "fast_output_before_low_completion": True,
                    "exact_tokens_predicted_and_id_cardinality": True,
                    "exact_reference_fast_and_configured_burst_outputs": True,
                    "exact_stage_commit_pairs": True,
                    "maximum_one_owner": True,
                    "low_commit_during_fast_decode": True,
                    "low_aligned_yield": True,
                },
                "overlap_lead_ms": (low.end_ns - fast.first_token_ns) / 1e6,
                "requests": {tag: result.summary() for tag, result in sorted(self.results.items())},
                "trace": trace_summary,
            }
        (self.artifact / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))

    def close(self) -> None:
        self.events.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DeepSeek V4 Phase 3 split-KV prefill/decode vertical smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:18130")
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY", ""))
    parser.add_argument(
        "--scheduling-token",
        default=os.environ.get("LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN", ""))
    parser.add_argument("--trace-endpoint", default="/internal/benchmark/scheduler-trace")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--low-prompt-tokens", type=int, default=8192)
    parser.add_argument("--low-n-predict", type=int, default=8)
    parser.add_argument("--fast-prompt-tokens", type=int, default=128)
    parser.add_argument("--fast-n-predict", type=int, default=512)
    parser.add_argument("--reference-n-predict", type=int, default=32)
    parser.add_argument(
        "--sequential-bursts", action="store_true",
        help="run the two-slot low plus sequential fast-burst workload instead of the vertical 512-token workload")
    parser.add_argument("--burst-count", type=int, default=0)
    parser.add_argument("--burst-interval-seconds", type=int, default=2)
    parser.add_argument("--burst-n-predict", type=int, default=16)
    parser.add_argument("--alignment-tokens", type=int, default=128)
    parser.add_argument("--priority-chunk-limit-tokens", type=int, default=2048)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "trusted_contract": {"trace_endpoint": args.trace_endpoint},
        "requests": {
            "common": {
                "stream": True,
                "return_tokens": True,
                "return_progress": True,
                "cache": False,
                "cache_prompt": False,
                "temperature": 0.0,
                "seed": 42,
                "ignore_eos": True,
            },
            "reference": {
                "prompt_tokens": args.fast_prompt_tokens,
                "n_predict": args.reference_n_predict,
                "salt": 1042,
                "seed": 42,
            },
            "isolated_and_sustained_fast": {
                "prompt_tokens": args.fast_prompt_tokens,
                "n_predict": args.fast_n_predict,
                "salt": 2001,
                "seed": 42,
            },
            "trusted_low": {
                "prompt_tokens": args.low_prompt_tokens,
                "n_predict": args.low_n_predict,
                "salt": 3001,
                "seed": 42,
            },
            "fast_bursts": {
                "count": args.burst_count,
                "interval_seconds": args.burst_interval_seconds,
                "prompt_tokens": args.fast_prompt_tokens,
                "n_predict": args.burst_n_predict,
                "salt": 5001,
                "seed": 42,
            },
            "timeout_seconds": args.timeout_seconds,
        },
        "acceptance": {
            "prefill_alignment_tokens": args.alignment_tokens,
            "priority_chunk_limit_tokens": args.priority_chunk_limit_tokens,
        },
    }
    if args.sequential_bursts:
        config["mode"] = SEQUENTIAL_BURST_MODE
    return config


def validate_args(args: argparse.Namespace) -> None:
    if not args.api_key or not 32 <= len(args.scheduling_token) <= 256:
        raise ValueError("--api-key and a 32..256-byte --scheduling-token are required")
    if args.api_key == args.scheduling_token:
        raise ValueError("API key and trusted scheduling token must be distinct")
    if args.low_prompt_tokens <= 0 or args.fast_prompt_tokens <= 0:
        raise ValueError("prompt lengths must be positive")
    if min(args.low_n_predict, args.fast_n_predict, args.reference_n_predict,
           args.burst_n_predict, args.alignment_tokens,
           args.priority_chunk_limit_tokens) <= 0:
        raise ValueError("prediction, alignment, and priority chunk limits must be positive")
    if args.sequential_bursts:
        if args.burst_count != SEQUENTIAL_BURST_COUNT:
            raise ValueError(f"sequential burst mode requires --burst-count {SEQUENTIAL_BURST_COUNT}")
        if args.burst_interval_seconds != 0:
            raise ValueError("sequential burst mode requires --burst-interval-seconds 0")
    elif args.burst_count != 0:
        raise ValueError("--burst-count requires --sequential-bursts")


def main() -> int:
    args = parse_args()
    validate_args(args)

    config = config_from_args(args)
    args.artifact.mkdir(parents=True, exist_ok=False)
    (args.artifact / "effective-config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ["API_BEARER_KEY"] = args.api_key
    os.environ["LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN"] = args.scheduling_token
    smoke = Smoke(config, args.base_url, args.artifact)
    try:
        smoke.execute()
        return 0
    finally:
        smoke.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


API_KEY_ENV = "LLAMA_API_KEY"
OPERATOR_TOKEN_ENV = "LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN"
OPERATOR_TOKEN_HEADER = "X-Llama-Trusted-Scheduling-Token"
LANE_HEADER = "X-Llama-Trusted-Lane"
TAG_HEADER = "X-Llama-Benchmark-Tag"
HARNESS_SCHEMA_VERSION = 1
MAX_CONFIG_BYTES = 1024 * 1024
MAX_SERVER_LOG_BYTES = 16 * 1024 * 1024
MAX_PROBE_OUTPUT_BYTES = 1024 * 1024
MAX_SYSTEM_OUTPUT_BYTES = 64 * 1024
MAX_WORKLOAD_REQUESTS = 64
MAX_TOTAL_PROMPT_TOKENS = 1024 * 1024
MAX_TOTAL_PREDICT_TOKENS = 64 * 1024
MAX_PREDICT_TOKENS = 4096
MAX_RESPONSE_LINE_BYTES = 1024 * 1024
DEFAULT_ERROR_PATTERNS = (
    "ggml_metal_graph_compute: command buffer",
    "failed to allocate",
    "critical memory pressure",
    "request queue deadline expired",
    "request run deadline expired",
    "segmentation fault",
    "assertion failed",
)


class GateError(RuntimeError):
    pass


def reject_json_constant(value: str) -> None:
    raise GateError(f"configuration contains non-finite number {value}")


@dataclasses.dataclass
class RequestResult:
    tag: str
    requested_tokens: int
    status: int = 0
    start_ns: int = 0
    first_token_ns: int = 0
    end_ns: int = 0
    reported_tokens: int = -1
    tokens: list[int] = dataclasses.field(default_factory=list)
    content: str = ""
    error: str = ""

    def validate(self) -> None:
        if self.error or self.status != 200:
            raise GateError(f"{self.tag}: request failed with HTTP {self.status}: {self.error}")
        if self.reported_tokens != self.requested_tokens:
            raise GateError(
                f"{self.tag}: tokens_predicted={self.reported_tokens}, expected={self.requested_tokens}")
        if len(self.tokens) != self.requested_tokens:
            raise GateError(
                f"{self.tag}: returned {len(self.tokens)} token IDs, expected={self.requested_tokens}")
        if not self.content:
            raise GateError(f"{self.tag}: output content is empty")
        if not self.first_token_ns or self.first_token_ns < self.start_ns or self.end_ns < self.first_token_ns:
            raise GateError(f"{self.tag}: invalid request timing boundaries")

    def output_fingerprint(self) -> str:
        encoded = json.dumps(
            {"tokens": self.tokens, "content": self.content},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def artifact(self, phase: str, block: int, repetition: int, phase_sample: int) -> dict[str, Any]:
        return {
            "phase": phase,
            "block": block,
            "block_repetition": repetition,
            "phase_sample": phase_sample,
            "tag": self.tag,
            "status": self.status,
            "requested_tokens": self.requested_tokens,
            "reported_tokens": self.reported_tokens,
            "returned_token_ids": len(self.tokens),
            "token_ids": self.tokens,
            "content_sha256": hashlib.sha256(self.content.encode()).hexdigest(),
            "output_sha256": self.output_fingerprint(),
            "ttft_ms": (self.first_token_ns - self.start_ns) / 1e6,
            "elapsed_ms": (self.end_ns - self.start_ns) / 1e6,
            "error": self.error,
        }


class Redactor:
    def __init__(self, secrets: list[str]):
        values = [value for value in secrets if value]
        fragments = [value[-4:] for value in values if len(value) >= 4]
        self.forbidden = tuple(dict.fromkeys(values + fragments))

    def text(self, value: str) -> str:
        result = value
        for secret in self.forbidden:
            result = result.replace(secret, "[REDACTED]")
        return result

    def contains(self, value: str | bytes) -> bool:
        if isinstance(value, bytes):
            return any(secret.encode() in value for secret in self.forbidden)
        return any(secret in value for secret in self.forbidden)


class BoundedProcessLog:
    def __init__(self, stream: Any, maximum: int):
        self.stream = stream
        self.maximum = maximum
        self.data = bytearray()
        self.truncated = False
        self.thread = threading.Thread(target=self._read, daemon=True)

    def _read(self) -> None:
        while True:
            chunk = self.stream.read(65536)
            if not chunk:
                return
            remaining = self.maximum - len(self.data)
            if remaining > 0:
                self.data.extend(chunk[:remaining])
            if len(chunk) > remaining:
                self.truncated = True

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise GateError("server log reader did not terminate")


def load_config(path: pathlib.Path) -> dict[str, Any]:
    encoded = path.read_bytes()
    if len(encoded) > MAX_CONFIG_BYTES:
        raise GateError(f"configuration exceeds {MAX_CONFIG_BYTES} bytes")
    value = json.loads(encoded, parse_constant=reject_json_constant)
    if not isinstance(value, dict) or value.get("schema") != HARNESS_SCHEMA_VERSION:
        raise GateError(f"configuration schema must equal {HARNESS_SCHEMA_VERSION}")
    return value


def require_loopback_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password:
        raise GateError("server.base_url must be an HTTP(S) URL without embedded credentials")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise GateError("server.base_url must use an explicit loopback hostname")
    if parsed.query or parsed.fragment:
        raise GateError("server.base_url cannot contain a query or fragment")
    return value.rstrip("/")


def json_write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_identity(command: list[str], repo: pathlib.Path) -> dict[str, Any]:
    executable = pathlib.Path(command[0])
    if not executable.is_absolute():
        executable = repo / executable
    executable = executable.resolve()
    if not executable.is_file():
        raise GateError(f"server executable does not exist: {executable}")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    source_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True).stdout.splitlines()
    return {
        "source_commit": source_commit,
        "source_dirty_paths": source_status,
        "server_executable": str(executable),
        "server_binary_sha256": sha256_file(executable),
        "server_binary_bytes": executable.stat().st_size,
    }


def expand_command(command: Any, model: str) -> list[str]:
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
        raise GateError("server.command must be a non-empty string array")
    return [value.replace("{model}", model) for value in command]


def validate_secret_boundary(config: dict[str, Any], command: list[str], redactor: Redactor) -> None:
    encoded = json.dumps(config, sort_keys=True)
    if redactor.contains(encoded) or redactor.contains("\0".join(command)):
        raise GateError("configuration or server command contains a credential value or fragment")
    lowered = [value.lower() for value in command]
    if any(value in {"--api-key", "--api-key-file"} or
           value.startswith("--api-key=") or value.startswith("--api-key-file=")
           for value in lowered):
        raise GateError("server API credentials must come from the environment, never command arguments")


def credential(value: str, name: str, minimum: int, maximum: int) -> str:
    size = len(value.encode())
    if size < minimum or size > maximum or any(character in value for character in ("\0", "\n", "\r")):
        raise GateError(f"{name} must contain {minimum}..{maximum} safe header bytes")
    return value


def bounded_number(name: str, value: Any, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise GateError(f"{name} must be from {minimum} through {maximum}")
    return result


def bounded_integer(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or value > maximum:
        raise GateError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def validate_workload(value: Any) -> list[dict[str, Any]]:
    if (not isinstance(value, list) or not value or len(value) > MAX_WORKLOAD_REQUESTS or
            not all(isinstance(item, dict) for item in value)):
        raise GateError(f"workload.requests must contain 1..{MAX_WORKLOAD_REQUESTS} objects")
    tags = [str(item.get("tag", "")) for item in value]
    if any(not tag or len(tag) > 128 for tag in tags) or len(set(tags)) != len(tags):
        raise GateError("workload request tags must be unique strings of 1..128 characters")
    prompt_counts = [int(item.get("prompt_tokens", 0)) for item in value]
    predict_counts = [int(item.get("n_predict", 0)) for item in value]
    if any(count < 1 or count > 1_048_576 for count in prompt_counts):
        raise GateError("each workload prompt_tokens value must be from 1 through 1048576")
    if sum(prompt_counts) > MAX_TOTAL_PROMPT_TOKENS:
        raise GateError(f"total prompt tokens must not exceed {MAX_TOTAL_PROMPT_TOKENS}")
    if any(count < 1 or count > MAX_PREDICT_TOKENS for count in predict_counts):
        raise GateError(f"each workload n_predict value must be from 1 through {MAX_PREDICT_TOKENS}")
    if sum(predict_counts) > MAX_TOTAL_PREDICT_TOKENS:
        raise GateError(f"total predicted tokens must not exceed {MAX_TOTAL_PREDICT_TOKENS}")
    for item in value:
        bounded_number(f"{item['tag']}.offset_ms", item.get("offset_ms", 0), 0, 900_000)
        if item.get("lane") not in {None, "low", "normal", "fast"}:
            raise GateError(f"{item['tag']}: lane must be low, normal, or fast")
    return value


def wait_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    request = urllib.request.Request(base_url + "/health", headers={"Accept": "application/json"})
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise GateError(f"server did not become ready within {timeout_seconds:.0f} seconds")


def numeric_prompt(count: int, salt: int) -> list[int]:
    if count < 1 or count > 1_048_576:
        raise GateError("workload prompt_tokens must be from 1 through 1048576")
    return [1000 + ((131 * index + salt) % 10000) for index in range(count)]


def run_request(
        base_url: str,
        specification: dict[str, Any],
        api_key: str,
        operator_token: str,
        timeout_seconds: float) -> RequestResult:
    tag = str(specification["tag"])
    requested_tokens = int(specification["n_predict"])
    if requested_tokens < 1 or requested_tokens > MAX_PREDICT_TOKENS:
        raise GateError(f"{tag}: n_predict must be from 1 through {MAX_PREDICT_TOKENS}")
    result = RequestResult(tag=tag, requested_tokens=requested_tokens, start_ns=time.monotonic_ns())
    body = {
        "prompt": numeric_prompt(int(specification["prompt_tokens"]), int(specification.get("salt", 1001))),
        "n_predict": requested_tokens,
        "seed": int(specification.get("seed", 42)),
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "return_tokens": True,
        "cache": False,
        "cache_prompt": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    lane = specification.get("lane")
    if lane is not None:
        if lane not in {"low", "normal", "fast"}:
            raise GateError(f"{tag}: lane must be low, normal, or fast")
        headers.update({OPERATOR_TOKEN_HEADER: operator_token, LANE_HEADER: lane, TAG_HEADER: tag})
    request = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers=headers,
        method="POST",
    )
    content_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result.status = response.status
            while raw_line := response.readline(MAX_RESPONSE_LINE_BYTES + 1):
                if len(raw_line) > MAX_RESPONSE_LINE_BYTES:
                    raise GateError(
                        f"{tag}: response line exceeds {MAX_RESPONSE_LINE_BYTES} bytes")
                line = raw_line.decode("utf-8", errors="strict").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                decoded = json.loads(payload)
                for item in decoded if isinstance(decoded, list) else [decoded]:
                    tokens = item.get("tokens") or []
                    if tokens:
                        if not result.first_token_ns:
                            result.first_token_ns = time.monotonic_ns()
                        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                            raise GateError(f"{tag}: response returned an invalid token array")
                        result.tokens.extend(tokens)
                    if item.get("tokens_predicted") is not None:
                        result.reported_tokens = int(item["tokens_predicted"])
                    if item.get("content") is not None:
                        if not isinstance(item["content"], str):
                            raise GateError(f"{tag}: response returned non-string content")
                        content_parts.append(item["content"])
    except Exception as error:  # recorded and normalized by validate()
        result.error = f"{type(error).__name__}: {error}"
    finally:
        result.end_ns = time.monotonic_ns()
        result.content = "".join(content_parts)
    result.validate()
    return result


def run_iteration(
        base_url: str,
        workload: list[dict[str, Any]],
        api_key: str,
        operator_token: str,
        timeout_seconds: float) -> list[RequestResult]:
    origin = time.monotonic()

    def scheduled(specification: dict[str, Any]) -> RequestResult:
        target = origin + float(specification.get("offset_ms", 0)) / 1000.0
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.01, remaining))
        return run_request(base_url, specification, api_key, operator_token, timeout_seconds)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workload)) as executor:
        futures = [executor.submit(scheduled, specification) for specification in workload]
        return [future.result() for future in futures]


def process_sample(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    completed = subprocess.run(
        ["ps", "-o", "rss=", "-o", "%cpu=", "-p", str(process.pid)],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    match = re.search(r"([0-9]+)\s+([0-9.]+)", completed.stdout)
    return {
        "monotonic_ns": time.monotonic_ns(),
        "rss_kib": int(match.group(1)) if match else None,
        "cpu_percent": float(match.group(2)) if match else None,
        "available": match is not None,
    }


def run_system_commands(
        entries: list[dict[str, Any]],
        redactor: Redactor,
        phase: str,
        block: int,
        boundary: str) -> list[dict[str, Any]]:
    results = []
    for entry in entries:
        command = entry.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
            raise GateError("system_samples entries require a non-empty command string array")
        completed = subprocess.run(command, check=False, capture_output=True, timeout=30)
        output = (completed.stdout + completed.stderr)[:MAX_SYSTEM_OUTPUT_BYTES].decode("utf-8", errors="replace")
        output = redactor.text(output)
        if entry.get("required", True) and completed.returncode != 0:
            raise GateError(f"system sample {entry.get('name', command[0])} exited {completed.returncode}")
        results.append({
            "phase": phase,
            "block": block,
            "boundary": boundary,
            "name": str(entry.get("name", command[0])),
            "command": command,
            "exit_code": completed.returncode,
            "output": output,
            "truncated": len(completed.stdout) + len(completed.stderr) > MAX_SYSTEM_OUTPUT_BYTES,
        })
    return results


def metric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise GateError("cannot summarize an empty metric sample")
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values)
    return {
        "samples": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "stdev": deviation,
        "cv_percent": 0.0 if mean == 0 else deviation / mean * 100.0,
    }


def compare_phases(
        artifacts: list[dict[str, Any]],
        minimum_samples: int,
        impact_threshold_percent: float,
        maximum_cv_percent: float) -> dict[str, Any]:
    tags = sorted({item["tag"] for item in artifacts})
    comparison: dict[str, Any] = {}
    failures = []
    for tag in tags:
        tag_result: dict[str, Any] = {}
        for metric in ("ttft_ms", "elapsed_ms"):
            phases = {}
            for phase in ("disconnected", "connected"):
                values = [float(item[metric]) for item in artifacts if item["tag"] == tag and item["phase"] == phase]
                if len(values) < minimum_samples:
                    failures.append(
                        f"{tag}/{metric}/{phase} has {len(values)} samples, requires {minimum_samples}")
                    continue
                phases[phase] = metric_summary(values)
                if phases[phase]["cv_percent"] > maximum_cv_percent:
                    failures.append(
                        f"{tag}/{metric}/{phase} CV {phases[phase]['cv_percent']:.2f}% exceeds "
                        f"{maximum_cv_percent:.2f}%")
            if len(phases) == 2:
                baseline = float(phases["disconnected"]["median"])
                connected = float(phases["connected"]["median"])
                impact = 0.0 if baseline == 0 else (connected - baseline) / baseline * 100.0
                phases["connected_vs_disconnected_median_percent"] = impact
                phases["no_meaningful_impact_threshold_percent"] = impact_threshold_percent
                if impact > impact_threshold_percent:
                    failures.append(
                        f"{tag}/{metric} connected median impact {impact:.2f}% exceeds "
                        f"{impact_threshold_percent:.2f}%")
            tag_result[metric] = phases
        comparison[tag] = tag_result
    return {"requests": comparison, "failures": failures, "passed": not failures}


def verify_exact_outputs(results: list[dict[str, Any]]) -> dict[str, str]:
    expected: dict[str, str] = {}
    for result in results:
        prior = expected.setdefault(result["tag"], result["output_sha256"])
        if result["output_sha256"] != prior:
            raise GateError(f"{result['tag']}: exact output differs across A/B repetitions")
    return expected


def read_probe_ready(process: subprocess.Popen[str], timeout_seconds: float) -> dict[str, Any]:
    messages: queue.Queue[str] = queue.Queue(maxsize=1)

    def read_line() -> None:
        messages.put(process.stdout.readline() if process.stdout is not None else "")

    thread = threading.Thread(target=read_line, daemon=True)
    thread.start()
    try:
        line = messages.get(timeout=timeout_seconds)
    except queue.Empty as error:
        process.kill()
        raise GateError("dashboard probe did not establish before its readiness timeout") from error
    if not line:
        raise GateError("dashboard probe exited before reporting readiness")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise GateError(f"dashboard probe emitted invalid readiness output: {line.strip()}") from error
    if value.get("status") != "READY":
        raise GateError(f"dashboard probe did not report READY: {value}")
    return value


def finish_probe(process: subprocess.Popen[str], timeout_seconds: float, redactor: Redactor) -> dict[str, Any]:
    if process.stdin is None:
        raise GateError("dashboard probe control pipe is unavailable")
    process.stdin.write("STOP\n")
    process.stdin.flush()
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise GateError("dashboard probe did not tear down within its timeout") from error
    output = redactor.text(stdout)
    if len(output.encode()) > MAX_PROBE_OUTPUT_BYTES:
        raise GateError("dashboard probe output exceeds its bounded result budget")
    if process.returncode != 0:
        raise GateError(f"dashboard probe exited {process.returncode}: {output.strip()}")
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise GateError(f"dashboard probe emitted {len(lines)} result lines after readiness, expected one")
    value = json.loads(lines[0])
    if value.get("status") != "PASS":
        raise GateError(f"dashboard probe did not pass: {value}")
    return value


def stop_server(process: subprocess.Popen[bytes], timeout_seconds: float) -> dict[str, Any]:
    if process.poll() is not None:
        return {"clean": False, "forced": False, "exit_code": process.returncode, "reason": "early_exit"}
    process.send_signal(signal.SIGTERM)
    try:
        exit_code = process.wait(timeout=timeout_seconds)
        return {"clean": True, "forced": False, "exit_code": exit_code, "reason": "sigterm"}
    except subprocess.TimeoutExpired:
        process.kill()
        exit_code = process.wait(timeout=10)
        return {"clean": False, "forced": True, "exit_code": exit_code, "reason": "sigkill"}


def scan_artifacts(artifact: pathlib.Path, redactor: Redactor) -> None:
    for path in artifact.rglob("*"):
        if path.is_file() and redactor.contains(path.read_bytes()):
            raise GateError(f"credential value or fragment leaked into artifact {path.name}")


def parse_swap_used_bytes(
        samples: list[dict[str, Any]], phase: str, block: int, boundary: str) -> int | None:
    units = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    for sample in samples:
        if sample["phase"] != phase or sample["block"] != block or sample["boundary"] != boundary:
            continue
        match = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", sample["output"], re.IGNORECASE)
        if match:
            return int(float(match.group(1)) * units[match.group(2).upper()])
    return None


def validate_phase_plan(config: dict[str, Any]) -> tuple[list[str], int, int, dict[str, int]]:
    phase_order = config.get("phase_order")
    if (not isinstance(phase_order, list) or
            len(phase_order) > 8 or
            any(phase not in {"disconnected", "connected"} for phase in phase_order) or
            phase_order.count("disconnected") < 2 or phase_order.count("connected") < 2):
        raise GateError("phase_order must contain at least two disconnected and two connected blocks")
    if phase_order != list(reversed(phase_order)):
        raise GateError("phase_order must be counterbalanced (for example disconnected/connected/connected/disconnected)")
    repetitions = bounded_integer(
        "repetitions_per_block", config.get("repetitions_per_block", 0), 1, 10)
    minimum_samples = bounded_integer(
        "minimum_samples_per_phase", config.get("minimum_samples_per_phase", 3), 3, 40)
    available_per_phase = {
        phase: repetitions * phase_order.count(phase)
        for phase in ("disconnected", "connected")
    }
    if minimum_samples < 3 or any(count < minimum_samples for count in available_per_phase.values()):
        raise GateError(
            f"insufficient declared sample size: per_block={repetitions}, order={phase_order}, "
            f"available={available_per_phase}, minimum={minimum_samples}; each mode needs at least three samples")
    return phase_order, repetitions, minimum_samples, available_per_phase


def validate_system_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or not all(isinstance(entry, dict) for entry in value):
        raise GateError("system_samples must be a non-empty object array")
    names = {str(entry.get("name", "")) for entry in value}
    if not {"swap", "pressure"}.issubset(names):
        raise GateError("system_samples must include required entries named swap and pressure")
    for entry in value:
        command = entry.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
            raise GateError("system_samples entries require a non-empty command string array")
    return value


def run_gate(config: dict[str, Any], artifact: pathlib.Path, repo: pathlib.Path) -> dict[str, Any]:
    api_key = credential(os.environ.get(API_KEY_ENV, ""), API_KEY_ENV, 1, 1024)
    operator_token = credential(
        os.environ.get(OPERATOR_TOKEN_ENV, ""), OPERATOR_TOKEN_ENV, 32, 256)
    if api_key == operator_token:
        raise GateError("API and operator credentials must be distinct")
    redactor = Redactor([api_key, operator_token])

    server = config.get("server", {})
    base_url = require_loopback_url(str(server.get("base_url", "")))
    model = os.environ.get("M2_DASHBOARD_MODEL", "")
    if not model:
        raise GateError("M2_DASHBOARD_MODEL is required in the environment")
    command = expand_command(server.get("command"), model)
    validate_secret_boundary(config, command, redactor)
    workload = validate_workload(config.get("workload", {}).get("requests"))

    phase_order, repetitions, minimum_samples, available_per_phase = validate_phase_plan(config)
    warmups = bounded_integer("warmups_per_block", config.get("warmups_per_block", 1), 0, 10)
    impact_threshold = bounded_number(
        "no_meaningful_impact_percent", config.get("no_meaningful_impact_percent", 5.0), 0, 100)
    maximum_cv = bounded_number(
        "maximum_cv_percent", config.get("maximum_cv_percent", 15.0), 0.001, 1000)
    timeout_seconds = bounded_number(
        "request_timeout_seconds", config.get("request_timeout_seconds", 900), 1, 3600)
    server_ready_timeout = bounded_number(
        "server.ready_timeout_seconds", server.get("ready_timeout_seconds", 900), 1, 1800)
    server_teardown_timeout = bounded_number(
        "server.teardown_timeout_seconds", server.get("teardown_timeout_seconds", 30), 1, 120)
    system_entries = validate_system_entries(config.get("system_samples"))
    probe = config.get("dashboard_probe", {})
    if not isinstance(probe, dict):
        raise GateError("dashboard_probe must be an object")
    probe_timeout_ms = bounded_integer(
        "dashboard_probe.timeout_ms", probe.get("timeout_ms", 120000), 1000, 900000)
    probe_establish_timeout_ms = bounded_integer(
        "dashboard_probe.establish_timeout_ms", probe.get("establish_timeout_ms", 30000), 1000, 120000)
    probe_minimum_events = bounded_integer(
        "dashboard_probe.minimum_events", probe.get("minimum_events", 1), 0, 4096)
    probe_ready_timeout = bounded_number(
        "dashboard_probe.ready_timeout_seconds", probe.get("ready_timeout_seconds", 30), 1, 120)
    probe_teardown_timeout = bounded_number(
        "dashboard_probe.teardown_timeout_seconds", probe.get("teardown_timeout_seconds", 30), 1, 120)
    refill_states = probe.get("refill_states", [])
    if not isinstance(refill_states, list) or not all(isinstance(state, str) for state in refill_states):
        raise GateError("dashboard_probe.refill_states must be a string array")
    supported_refill_states = {
        "disabled", "inactive", "window_open", "one_member_eligible",
        "quota_exhausted", "deadline_expired", "full_width",
    }
    unknown_refill_states = set(refill_states) - supported_refill_states
    if unknown_refill_states:
        raise GateError(f"dashboard_probe.refill_states contains unknown values: {sorted(unknown_refill_states)}")

    artifact.mkdir(parents=True, exist_ok=False)
    identity = command_identity(command, repo)
    if config.get("require_clean_source", True) and identity["source_dirty_paths"]:
        raise GateError("source tree is dirty; commit or remove changes before a reproducible target run")
    json_write(artifact / "effective-config.json", config)
    json_write(artifact / "identity.json", {**identity, "server_command": command})

    environment = dict(os.environ)
    process = subprocess.Popen(
        command,
        cwd=repo,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        raise GateError("server output pipe is unavailable")
    server_log = BoundedProcessLog(process.stdout, MAX_SERVER_LOG_BYTES)
    server_log.start()
    probe_process: subprocess.Popen[str] | None = None
    request_artifacts: list[dict[str, Any]] = []
    process_samples: list[dict[str, Any]] = []
    system_samples: list[dict[str, Any]] = []
    probe_results: list[dict[str, Any]] = []
    teardown: dict[str, Any] = {"clean": False, "reason": "not_started"}
    try:
        wait_ready(base_url, server_ready_timeout)
        phase_sample_counts = {"disconnected": 0, "connected": 0}
        for block, phase in enumerate(phase_order):
            if phase == "connected":
                probe_script = pathlib.Path(str(probe.get(
                    "script", "tools/m2-dashboard/target/live-probe.mjs")))
                if not probe_script.is_absolute():
                    probe_script = repo / probe_script
                probe_environment = dict(environment)
                probe_environment.update({
                    "M2_DASHBOARD_BASE_URL": base_url,
                    "M2_DASHBOARD_PROBE_TIMEOUT_MS": str(probe_timeout_ms),
                    "M2_DASHBOARD_ESTABLISH_TIMEOUT_MS": str(probe_establish_timeout_ms),
                    "M2_DASHBOARD_MIN_EVENTS": str(probe_minimum_events),
                })
                if probe.get("require_refill_states", False):
                    probe_environment["M2_DASHBOARD_REQUIRE_REFILL_STATES"] = "1"
                if refill_states:
                    probe_environment["M2_DASHBOARD_REFILL_STATES"] = ",".join(refill_states)
                probe_process = subprocess.Popen(
                    [str(probe.get("node", "node")), str(probe_script)],
                    cwd=repo,
                    env=probe_environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                read_probe_ready(probe_process, probe_ready_timeout)

            system_samples.extend(run_system_commands(system_entries, redactor, phase, block, "before"))
            process_samples.append({
                "phase": phase, "block": block, "boundary": "before", **process_sample(process)})
            for _ in range(warmups):
                run_iteration(base_url, workload, api_key, operator_token, timeout_seconds)
            for repetition in range(repetitions):
                results = run_iteration(base_url, workload, api_key, operator_token, timeout_seconds)
                for result in results:
                    request_artifacts.append(result.artifact(
                        phase, block, repetition, phase_sample_counts[phase]))
                phase_sample_counts[phase] += 1
                process_samples.append({
                    "phase": phase,
                    "block": block,
                    "boundary": f"repetition-{repetition}",
                    **process_sample(process),
                })
            process_samples.append({
                "phase": phase, "block": block, "boundary": "after", **process_sample(process)})
            system_samples.extend(run_system_commands(system_entries, redactor, phase, block, "after"))
            if phase == "connected" and probe_process is not None:
                probe_result = finish_probe(
                    probe_process,
                    probe_teardown_timeout,
                    redactor,
                )
                probe_results.append({"block": block, **probe_result})
                probe_process = None
    finally:
        try:
            if probe_process is not None:
                probe_process.kill()
                probe_process.communicate(timeout=10)
        finally:
            teardown = stop_server(process, server_teardown_timeout)
        server_log.join()
        log_text = redactor.text(server_log.data.decode("utf-8", errors="replace"))
        (artifact / "server.log").write_text(log_text, encoding="utf-8")
        json_write(artifact / "teardown.json", teardown)
        json_write(artifact / "requests.json", request_artifacts)
        json_write(artifact / "process-samples.json", process_samples)
        json_write(artifact / "system-samples.json", system_samples)
        if probe_results:
            json_write(artifact / "dashboard-probes.json", probe_results)

    fingerprints = verify_exact_outputs(request_artifacts)
    comparison = compare_phases(
        request_artifacts,
        minimum_samples,
        impact_threshold,
        maximum_cv,
    )
    failures = list(comparison["failures"])
    if not teardown.get("clean") or teardown.get("forced"):
        failures.append(f"server teardown was not clean: {teardown}")
    log_lower = (artifact / "server.log").read_text(encoding="utf-8").lower()
    for pattern in config.get("server_error_patterns", DEFAULT_ERROR_PATTERNS):
        if str(pattern).lower() in log_lower:
            failures.append(f"server log contains error pattern: {pattern}")
    for block, phase in enumerate(phase_order):
        before = parse_swap_used_bytes(system_samples, phase, block, "before")
        after = parse_swap_used_bytes(system_samples, phase, block, "after")
        if before is None or after is None:
            failures.append(f"{phase} block {block} did not expose parseable swap usage")
            continue
        maximum_swap_delta = int(config.get("maximum_swap_delta_bytes", 0))
        if after - before > maximum_swap_delta:
            failures.append(
                f"{phase} block {block} swap grew by {after - before} bytes, limit={maximum_swap_delta}")
    if any("critical" in sample["output"].lower() for sample in system_samples):
        failures.append("system sample reported critical pressure")
    if server_log.truncated:
        failures.append(f"server log exceeded its {MAX_SERVER_LOG_BYTES}-byte artifact bound")
    if len(probe_results) != phase_order.count("connected"):
        failures.append("every connected block must produce a dashboard probe result")
    if any(sample["truncated"] for sample in system_samples):
        failures.append(f"system sample exceeded its {MAX_SYSTEM_OUTPUT_BYTES}-byte output bound")

    scan_artifacts(artifact, redactor)
    summary = {
        "report_schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "source_commit": identity["source_commit"],
        "server_binary_sha256": identity["server_binary_sha256"],
        "block_order": phase_order,
        "repetitions_per_block": repetitions,
        "samples_per_phase": available_per_phase,
        "minimum_samples_per_phase": minimum_samples,
        "no_meaningful_impact_percent": impact_threshold,
        "maximum_cv_percent": maximum_cv,
        "exact_output_sha256_by_tag": fingerprints,
        "comparison": comparison,
        "dashboard_probes": probe_results,
        "teardown": teardown,
        "failures": failures,
        "limitations": [
            "This host-created harness does not itself establish the M2 exit gate.",
            "Refill assertions remain unavailable unless the integrated shared schema exposes authoritative fast_refill state.",
            "Process CPU samples are point observations; request TTFT and elapsed measurements are the A/B decision metrics.",
        ],
    }
    json_write(artifact / "summary.json", summary)
    scan_artifacts(artifact, redactor)
    if failures:
        raise GateError("; ".join(failures))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the credential-safe disconnected/connected dashboard target A/B gate")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = pathlib.Path(__file__).resolve().parents[3]
    config: dict[str, Any] = {}
    try:
        config = load_config(args.config)
        summary = run_gate(config, args.artifact, repo)
        print(json.dumps({"status": summary["status"], "artifact": str(args.artifact)}, sort_keys=True))
        return 0
    except Exception as error:
        message = str(error)
        secrets = [os.environ.get(API_KEY_ENV, ""), os.environ.get(OPERATOR_TOKEN_ENV, "")]
        redactor = Redactor(secrets)
        message = redactor.text(message)
        if args.artifact.exists():
            failure_path = args.artifact / "failure.json"
            if not failure_path.exists():
                json_write(failure_path, {
                    "report_schema_version": 1, "status": "FAIL", "error": message})
            try:
                scan_artifacts(args.artifact, redactor)
            except Exception as scan_error:
                message = f"{message}; {redactor.text(str(scan_error))}"
        print(f"dashboard target gate failed: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

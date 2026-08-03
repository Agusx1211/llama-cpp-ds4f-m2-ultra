#!/usr/bin/env python3
"""Full-model control/candidate telemetry gate for production DSV4 F32 AMX.

This gate runs one exact binary in two fresh processes. The control explicitly
disables AMX; the candidate enables production coexecution with validation and
timing disabled after the retained strict-oracle gate has passed. Both capture
complete vocabulary logits for a pp2048 natural-language prompt and 64 steps
of teacher-forced and independent greedy continuation. Control/candidate
logit and token divergence is recorded but never rejects this quality-first
F32 experiment. Structural validity, finite values, exact per-process token
chains, current-commit F32-oracle provenance, and zero swap remain hard gates.
No timing value is an acceptance criterion.
"""

from __future__ import annotations

import argparse
import array
from collections import Counter
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "dsv4-amx-production-gate.py"
BASE_SPEC = importlib.util.spec_from_file_location("dsv4_amx_production_gate_base", BASE_SCRIPT)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"could not load base gate: {BASE_SCRIPT}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)

GateFailure = BASE.GateFailure

PROMPT_TOKENS = 2048
CONTINUATION_TOKENS = 64
LOGICAL_TOKENS = PROMPT_TOKENS + CONTINUATION_TOKENS
CONTEXT_ALIGNMENT = 256
RUNTIME_N_CTX = ((LOGICAL_TOKENS + CONTEXT_ALIGNMENT - 1) // CONTEXT_ALIGNMENT) * CONTEXT_ALIGNMENT
CONTEXT_PADDING = RUNTIME_N_CTX - LOGICAL_TOKENS
PATHS = ("teacher", "greedy")
VECTORS_PER_PATH = CONTINUATION_TOKENS + 1
NO_INPUT_TOKEN = -(1 << 31)
FORMAT_VERSION = 1
PROMPT_VERSION = 1
ENDIAN_MARKER = 0x01020304
FILE_MAGIC = b"DSV4LG01"
PATH_MARKERS = {"teacher": b"TCHLOG01", "greedy": b"GRYLOG01"}
FOOTER_MARKER = b"DSV4DONE"
HEADER = struct.Struct("<8s9I40s")
RECORD_HEADER = struct.Struct("<iii")
ORACLE_MODEL_SET_SHA256 = "86830f37f33eb8ecaf909ffbfc005ad8068bf9d3d852608d53c169eb0a41c4da"

# Retain the previous full-logit limits only as historical telemetry markers.
# The accepted F32 quality policy explicitly forbids using Metal divergence as
# a correctness rejection. Quality is decided later by frozen behavioral gates.
LEGACY_LOGIT_NMSE_LIMIT = 1.0e-5
LEGACY_LOGIT_MAX_ABS_LIMIT = 2.5e-1
LEGACY_LOGIT_KL_LIMIT = 1.0e-6

MODE_ENVIRONMENTS = {
    "control": {
        "LLAMA_DSV4_AMX_COEXEC": "1",
        "LLAMA_DSV4_AMX_COEXEC_DISABLE": "1",
    },
    "candidate": {
        "LLAMA_DSV4_AMX_COEXEC": "1",
    },
}

DRIVER_RE = re.compile(r"dsv4_amx_full_gate_driver\s+event=(\S+)\s+outcome=(\S+)(?:\s+(.*))?$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def mode_child_environment(environment: Mapping[str, str], mode: str) -> dict[str, str]:
    _require(mode in MODE_ENVIRONMENTS, f"unknown gate mode: {mode}")
    child = {key: value for key, value in environment.items() if not key.startswith(BASE.RUNTIME_ENV_PREFIXES)}
    child.update(MODE_ENVIRONMENTS[mode])
    return child


def verify_oracle_artifacts(path: Path, expected_commit: str) -> dict[str, object]:
    path = path.expanduser().resolve()
    required = {
        "PASS",
        "artifact-sha256.txt",
        "environment.json",
        "model-set.json",
        "run.log",
        "verification.json",
    }
    _require(path.is_dir(), f"oracle artifact directory is missing: {path}")
    _require(not (path / "FAIL").exists(), "oracle artifact directory contains FAIL")
    _require((path / "PASS").read_text(encoding="utf-8") == "production AMX F32 oracle gate passed\n",
             "oracle PASS marker mismatch")
    manifest_path = path / "artifact-sha256.txt"
    manifest = manifest_path.read_text(encoding="utf-8").splitlines()
    observed: dict[str, str] = {}
    for line in manifest:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        _require(match is not None, f"malformed oracle artifact manifest line: {line}")
        digest, name = match.groups()
        _require(name not in observed, f"duplicate oracle artifact manifest entry: {name}")
        artifact = path / name
        _require(artifact.is_file() and not artifact.is_symlink(), f"missing oracle artifact: {artifact}")
        _require(_sha256(artifact) == digest, f"oracle artifact digest mismatch: {artifact}")
        observed[name] = digest
    _require(required <= set(observed) | {"artifact-sha256.txt"}, "oracle artifact manifest is incomplete")

    verification = _json(path / "verification.json")
    model = _json(path / "model-set.json")
    environment = _json(path / "environment.json")
    _require(verification.get("outcome") == "pass", "retained oracle verification did not pass")
    _require(verification.get("expected_commit") == expected_commit, "retained oracle commit mismatch")
    _require(environment.get("expected_commit") == expected_commit, "retained oracle environment mismatch")
    _require(verification.get("oracle_passes") == 172 and verification.get("metal_telemetry_records") == 43,
             "retained oracle/Metal-telemetry cardinality mismatch")
    _require(verification.get("completed_layers") == 43 and verification.get("callback_mask") == "0x7ffffffffff",
             "retained oracle callback completion mismatch")
    _require(verification.get("system_swap_before_bytes") == 0 and
             verification.get("system_swap_after_bytes") == 0,
             "retained oracle gate used system swap")
    _require(verification.get("resource_metrics") == {"swaps": 0, "thermal_nominal": 1},
             "retained oracle resource-health mismatch")
    _require(verification.get("thermal_nominal") is True, "retained oracle thermal state was not nominal")
    _require(verification.get("benchmark") is False, "retained oracle artifact was a benchmark")
    _require(model.get("set_sha256") == ORACLE_MODEL_SET_SHA256, "retained oracle model set mismatch")
    _require(verification.get("model", {}).get("set_sha256") == ORACLE_MODEL_SET_SHA256,
             "retained oracle verification model mismatch")
    return {
        "path": str(path),
        "oracle_commit": expected_commit,
        "model_set_sha256": ORACLE_MODEL_SET_SHA256,
        "verification_sha256": _sha256(path / "verification.json"),
        "artifact_manifest_sha256": _sha256(manifest_path),
    }


def token_hash(tokens: Sequence[int]) -> int:
    result = 14695981039346656037
    for token in tokens:
        value = token & 0xFFFFFFFF
        for shift in range(0, 32, 8):
            result ^= (value >> shift) & 0xFF
            result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return result


class CaptureReader:
    def __init__(self, path: Path, expected_mode: int, expected_commit: str):
        self.path = path
        self.source = path.open("rb")
        try:
            fields = HEADER.unpack(self._read(HEADER.size))
            (magic, version, endian, mode, n_vocab, prompt_count, continuation_count, path_count,
             vectors_per_path, prompt_version, commit_bytes) = fields
            _require(magic == FILE_MAGIC and version == FORMAT_VERSION and endian == ENDIAN_MARKER,
                     f"capture header format mismatch: {path}")
            _require(mode == expected_mode, f"capture mode mismatch: {path}")
            _require(commit_bytes.decode("ascii") == expected_commit, f"capture commit mismatch: {path}")
            _require(prompt_count == PROMPT_TOKENS and continuation_count == CONTINUATION_TOKENS,
                     f"capture token shape mismatch: {path}")
            _require(path_count == len(PATHS) and vectors_per_path == VECTORS_PER_PATH,
                     f"capture path/vector shape mismatch: {path}")
            _require(prompt_version == PROMPT_VERSION and n_vocab > 0,
                     f"capture prompt/vocab contract mismatch: {path}")
            self.mode = mode
            self.n_vocab = n_vocab
            self.prompt = self._read_ints(PROMPT_TOKENS)
            self.teacher = self._read_ints(CONTINUATION_TOKENS)
            self.expected_size = (HEADER.size + 4 * (PROMPT_TOKENS + CONTINUATION_TOKENS) +
                                  len(PATHS) * 8 + len(PATHS) * VECTORS_PER_PATH *
                                  (RECORD_HEADER.size + 4 * n_vocab) + len(FOOTER_MARKER))
            _require(path.stat().st_size == self.expected_size, f"capture file size mismatch: {path}")
        except Exception:
            self.source.close()
            raise

    def _read(self, count: int) -> bytes:
        data = self.source.read(count)
        _require(len(data) == count, f"truncated capture: {self.path}")
        return data

    def _read_ints(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"<{count}i", self._read(4 * count))

    def begin_path(self, name: str) -> None:
        _require(self._read(8) == PATH_MARKERS[name], f"capture path marker mismatch: {self.path} {name}")

    def read_record(self) -> tuple[int, int, int, array.array[float]]:
        step, input_token, top1 = RECORD_HEADER.unpack(self._read(RECORD_HEADER.size))
        logits = array.array("f")
        _require(logits.itemsize == 4, "host float array is not 32-bit")
        logits.frombytes(self._read(4 * self.n_vocab))
        if sys.byteorder != "little":
            logits.byteswap()
        return step, input_token, top1, logits

    def finish(self) -> None:
        _require(self._read(8) == FOOTER_MARKER, f"capture footer mismatch: {self.path}")
        _require(self.source.read(1) == b"", f"capture contains trailing bytes: {self.path}")
        self.source.close()

    def close(self) -> None:
        self.source.close()


def _logit_metrics(control: array.array[float], candidate: array.array[float]) -> dict[str, object]:
    _require(len(control) == len(candidate) and len(control) > 0, "logit vector shape mismatch")
    squared_error = 0.0
    reference_l2 = 0.0
    max_abs = 0.0
    control_top1 = 0
    candidate_top1 = 0
    for index, (reference, actual) in enumerate(zip(control, candidate)):
        _require(math.isfinite(reference) and math.isfinite(actual), "nonfinite full-model logit")
        if index == 0 or reference > control[control_top1]:
            control_top1 = index
        if index == 0 or actual > candidate[candidate_top1]:
            candidate_top1 = index
        difference = float(actual) - float(reference)
        squared_error += difference * difference
        reference_l2 += float(reference) * float(reference)
        max_abs = max(max_abs, abs(difference))
    _require(reference_l2 > 0.0 and math.isfinite(reference_l2), "invalid control logit L2")
    nmse = squared_error / reference_l2

    control_max = float(control[control_top1])
    candidate_max = float(candidate[candidate_top1])
    control_exp_sum = sum(math.exp(float(value) - control_max) for value in control)
    candidate_exp_sum = sum(math.exp(float(value) - candidate_max) for value in candidate)
    _require(control_exp_sum > 0.0 and candidate_exp_sum > 0.0, "invalid softmax normalization")
    control_log_z = control_max + math.log(control_exp_sum)
    candidate_log_z = candidate_max + math.log(candidate_exp_sum)
    kl = 0.0
    for reference, actual in zip(control, candidate):
        probability = math.exp(float(reference) - control_log_z)
        kl += probability * ((float(reference) - control_log_z) - (float(actual) - candidate_log_z))
    _require(kl >= -1.0e-9 and math.isfinite(kl), f"invalid KL divergence: {kl}")
    return {
        "nmse": nmse,
        "max_abs": max_abs,
        "kl": max(0.0, kl),
        "control_top1": control_top1,
        "candidate_top1": candidate_top1,
        "reference_l2": reference_l2,
    }


def compare_captures(control_path: Path, candidate_path: Path, expected_commit: str) -> dict[str, object]:
    control = CaptureReader(control_path, 0, expected_commit)
    try:
        candidate = CaptureReader(candidate_path, 1, expected_commit)
    except Exception:
        control.close()
        raise
    try:
        _require(control.n_vocab == candidate.n_vocab, "control/candidate vocabulary shape mismatch")
        _require(control.prompt == candidate.prompt, "control/candidate prompt tokens differ")
        _require(control.teacher == candidate.teacher, "control/candidate teacher inputs differ")
        _require(all(0 <= token < control.n_vocab for token in control.prompt + control.teacher),
                 "capture contains an invalid input token")
        metrics: list[dict[str, object]] = []
        control_greedy_tokens: list[int] = []
        candidate_greedy_tokens: list[int] = []
        top1_mismatches: list[dict[str, object]] = []
        max_nmse = 0.0
        max_abs = 0.0
        max_kl = 0.0
        outside_legacy_nmse = 0
        outside_legacy_max_abs = 0
        outside_legacy_kl = 0
        for path_name in PATHS:
            control.begin_path(path_name)
            candidate.begin_path(path_name)
            prior_control_top1 = NO_INPUT_TOKEN
            prior_candidate_top1 = NO_INPUT_TOKEN
            for expected_step in range(VECTORS_PER_PATH):
                control_step, control_input, control_record_top1, control_logits = control.read_record()
                candidate_step, candidate_input, candidate_record_top1, candidate_logits = candidate.read_record()
                _require(control_step == expected_step and candidate_step == expected_step,
                         f"{path_name} logit step sequence mismatch")
                if expected_step == 0:
                    expected_control_input = NO_INPUT_TOKEN
                    expected_candidate_input = NO_INPUT_TOKEN
                elif path_name == "teacher":
                    expected_control_input = control.teacher[expected_step - 1]
                    expected_candidate_input = candidate.teacher[expected_step - 1]
                else:
                    expected_control_input = prior_control_top1
                    expected_candidate_input = prior_candidate_top1
                _require(control_input == expected_control_input and candidate_input == expected_candidate_input,
                         f"{path_name} input-token chain mismatch at step {expected_step}")

                values = _logit_metrics(control_logits, candidate_logits)
                _require(control_record_top1 == values["control_top1"],
                         f"control recorded top-1 mismatch at {path_name}/{expected_step}")
                _require(candidate_record_top1 == values["candidate_top1"],
                         f"candidate recorded top-1 mismatch at {path_name}/{expected_step}")
                prior_control_top1 = int(values["control_top1"])
                prior_candidate_top1 = int(values["candidate_top1"])
                if prior_control_top1 != prior_candidate_top1:
                    top1_mismatches.append({
                        "path": path_name,
                        "step": expected_step,
                        "control": prior_control_top1,
                        "candidate": prior_candidate_top1,
                    })
                if path_name == "greedy" and expected_step < CONTINUATION_TOKENS:
                    control_greedy_tokens.append(prior_control_top1)
                    candidate_greedy_tokens.append(prior_candidate_top1)
                max_nmse = max(max_nmse, float(values["nmse"]))
                max_abs = max(max_abs, float(values["max_abs"]))
                max_kl = max(max_kl, float(values["kl"]))
                outside_legacy_nmse += float(values["nmse"]) > LEGACY_LOGIT_NMSE_LIMIT
                outside_legacy_max_abs += float(values["max_abs"]) > LEGACY_LOGIT_MAX_ABS_LIMIT
                outside_legacy_kl += float(values["kl"]) > LEGACY_LOGIT_KL_LIMIT
                metrics.append({"path": path_name, "step": expected_step, **values})
        control.finish()
        candidate.finish()
        _require(len(control_greedy_tokens) == CONTINUATION_TOKENS and
                 len(candidate_greedy_tokens) == CONTINUATION_TOKENS,
                 "greedy token sequence shape mismatch")
        return {
            "outcome": "telemetry",
            "policy": "f32_quality_first",
            "metal_divergence_rejects": False,
            "n_vocab": control.n_vocab,
            "prompt_tokens": PROMPT_TOKENS,
            "continuation_tokens": CONTINUATION_TOKENS,
            "logical_tokens": LOGICAL_TOKENS,
            "runtime_n_ctx": RUNTIME_N_CTX,
            "context_padding": CONTEXT_PADDING,
            "paths": list(PATHS),
            "vectors_per_path": VECTORS_PER_PATH,
            "total_vectors_per_process": len(PATHS) * VECTORS_PER_PATH,
            "prompt_token_fnv1a64": f"0x{token_hash(control.prompt):016x}",
            "teacher_token_fnv1a64": f"0x{token_hash(control.teacher):016x}",
            "prompt_token_sha256": hashlib.sha256(struct.pack(f"<{PROMPT_TOKENS}i", *control.prompt)).hexdigest(),
            "teacher_token_sha256": hashlib.sha256(
                struct.pack(f"<{CONTINUATION_TOKENS}i", *control.teacher)).hexdigest(),
            "control_greedy_tokens": control_greedy_tokens,
            "candidate_greedy_tokens": candidate_greedy_tokens,
            "top1_identical": not top1_mismatches,
            "top1_matches": len(PATHS) * VECTORS_PER_PATH - len(top1_mismatches),
            "top1_mismatches": top1_mismatches,
            "max_nmse": max_nmse,
            "max_abs": max_abs,
            "max_kl": max_kl,
            "legacy_limits": {
                "nmse": LEGACY_LOGIT_NMSE_LIMIT,
                "max_abs": LEGACY_LOGIT_MAX_ABS_LIMIT,
                "kl": LEGACY_LOGIT_KL_LIMIT,
                "enforced": False,
            },
            "outside_legacy_limits": {
                "nmse_vectors": outside_legacy_nmse,
                "max_abs_vectors": outside_legacy_max_abs,
                "kl_vectors": outside_legacy_kl,
            },
            "per_step": metrics,
            "control_bytes": control.expected_size,
            "candidate_bytes": candidate.expected_size,
        }
    finally:
        control.close()
        candidate.close()


def _driver_records(text: str) -> list[object]:
    return BASE._records(text, DRIVER_RE, "dsv4_amx_full_gate_driver")


def _check_driver(text: str, mode: str, expected_commit: str, capture: dict[str, object], capture_bytes: int) -> None:
    records = _driver_records(text)
    expected = [
        ("start", "begin"),
        ("model_load", "pass"),
        ("prompt", "pass"),
        ("path", "begin"),
        ("context_contract", "pass"),
        ("path", "pass"),
        ("context_teardown", "pass"),
        ("path", "begin"),
        ("context_contract", "pass"),
        ("path", "pass"),
        ("context_teardown", "pass"),
        ("output", "pass"),
        ("model_teardown", "pass"),
        ("backend_teardown", "pass"),
        ("complete", "pass"),
    ]
    _require([(record.event, record.outcome) for record in records] == expected,
             f"{mode} driver lifecycle mismatch")
    _require(all(record.fields.get("mode") == mode for record in records), f"{mode} driver mode mismatch")
    start = records[0]
    _require(start.fields.get("source_commit") == expected_commit, f"{mode} source commit mismatch")
    build_commit = start.fields.get("build_commit", "")
    _require(len(build_commit) >= 7 and expected_commit.startswith(build_commit), f"{mode} build commit mismatch")
    for field, expected_value in (("n_ctx", RUNTIME_N_CTX), ("logical_tokens", LOGICAL_TOKENS),
                                  ("context_alignment", CONTEXT_ALIGNMENT), ("context_padding", CONTEXT_PADDING),
                                  ("n_batch", PROMPT_TOKENS),
                                  ("n_ubatch", PROMPT_TOKENS), ("prompt_tokens", PROMPT_TOKENS),
                                  ("n_seq_max", 1), ("n_rs_seq", 0), ("flash_attn", 1),
                                  ("n_gpu_layers", 999), ("main_gpu", 0), ("load_mtp", 0),
                                  ("n_outputs_max", 1), ("no_perf", 1),
                                  ("steps", CONTINUATION_TOKENS), ("paths", 2),
                                  ("validation", 0), ("timing", 0), ("benchmark", 0)):
        _require(BASE._integer(start, field) == expected_value, f"{mode} start field mismatch: {field}")
    _require(start.fields.get("split_mode") == "layer", f"{mode} split mode mismatch")
    _require(BASE._integer(records[1], "n_vocab") == capture["n_vocab"], f"{mode} vocab mismatch")
    prompt = records[2]
    _require(BASE._integer(prompt, "prompt_version") == PROMPT_VERSION and
             BASE._integer(prompt, "tokens") == PROMPT_TOKENS and
             BASE._integer(prompt, "teacher_tokens") == CONTINUATION_TOKENS and
             BASE._integer(prompt, "padded") == 1,
             f"{mode} prompt shape mismatch")
    _require(prompt.fields.get("prompt_hash") == capture["prompt_token_fnv1a64"] and
             prompt.fields.get("teacher_hash") == capture["teacher_token_fnv1a64"],
             f"{mode} prompt hash mismatch")
    _require(BASE._integer(prompt, "seed_tokens") > 0 and BASE._integer(prompt, "filler_tokens") > 0,
             f"{mode} natural-language prompt components were empty")
    path_records = [record for record in records if record.event == "path"]
    _require([record.fields.get("path") for record in path_records] == ["teacher", "teacher", "greedy", "greedy"],
             f"{mode} path lifecycle mismatch")
    for record in path_records:
        _require(BASE._integer(record, "prompt_tokens") == PROMPT_TOKENS and
                 BASE._integer(record, "steps") == CONTINUATION_TOKENS,
                 f"{mode} path shape mismatch")
        if record.outcome == "pass":
            _require(BASE._integer(record, "vectors") == VECTORS_PER_PATH, f"{mode} path vector mismatch")
    teardowns = [record for record in records if record.event == "context_teardown"]
    _require([record.fields.get("path") for record in teardowns] == ["teacher", "greedy"],
             f"{mode} context teardown path mismatch")
    context_contracts = [record for record in records if record.event == "context_contract"]
    _require([record.fields.get("path") for record in context_contracts] == ["teacher", "greedy"],
             f"{mode} context contract path mismatch")
    for record in context_contracts:
        for field, expected_value in (("n_ctx", RUNTIME_N_CTX), ("n_batch", PROMPT_TOKENS),
                                      ("n_ubatch", PROMPT_TOKENS), ("n_seq_max", 1),
                                      ("logical_tokens", LOGICAL_TOKENS), ("context_padding", CONTEXT_PADDING)):
            _require(BASE._integer(record, field) == expected_value,
                     f"{mode} context contract mismatch: {field}")
    output = [record for record in records if record.event == "output"][0]
    _require(BASE._integer(output, "bytes") == capture_bytes and BASE._integer(output, "paths") == 2 and
             BASE._integer(output, "vectors") == 130,
             f"{mode} output shape mismatch")
    complete = records[-1]
    _require(BASE._integer(complete, "prompt_tokens") == PROMPT_TOKENS and
             BASE._integer(complete, "steps") == CONTINUATION_TOKENS and
             BASE._integer(complete, "paths") == 2 and BASE._integer(complete, "vectors") == 130 and
             BASE._integer(complete, "validation") == 0 and BASE._integer(complete, "benchmark") == 0,
             f"{mode} completion mismatch")


def _check_binding(record: object) -> None:
    _require(BASE._integer(record, "layers") == 43, "candidate binding layer mismatch")
    _require(BASE._integer(record, "callback_events") == 86 and BASE._integer(record, "tensor_metadata") == 43,
             "candidate binding cardinality mismatch")
    _require(BASE._integer(record, "external_output_bytes") == BASE.OUTPUT_BYTES and
             BASE._integer(record, "gallocr_output_bytes") == 0,
             "candidate binding output allocation mismatch")
    for field in ("unique_buffers", "unique_data", "producerless", "visible_before_consumer",
                  "routed_join_exact"):
        _require(BASE._integer(record, field) == 1, f"candidate binding violated {field}")
    _require(record.fields.get("callback_edges") == "norm_completion", "candidate binding edge mismatch")
    _require(BASE._integer(record, "routed_joins") == 43 and
             BASE._integer(record, "telemetry_consumers") == 0 and
             BASE._integer(record, "direct_consumer_edges") == 43,
             "candidate routed-join topology mismatch")


def _check_candidate_audits(audits: Sequence[object]) -> None:
    allowed = {
        ("eligibility", "eligible"),
        ("context", "eligible"),
        ("context", "allocated"),
        ("context_output", "allocated"),
        ("graph_mode", "eligible"),
        ("graph_mode", "fallback"),
        ("bindings", "complete"),
        ("bindings", "fallback"),
        ("pack", "begin"),
        ("pack", "complete"),
        ("pack_tensor", "pass"),
        ("pack_contract", "complete"),
        ("graph_reuse", "bound"),
        ("graph_reuse", "submit"),
        ("callback", "ask"),
        ("callback", "after"),
        ("output_visibility", "ready"),
        ("callbacks", "complete"),
        ("graph", "complete"),
    }
    for record in audits:
        _require((record.event, record.outcome) in allowed,
                 f"unexpected candidate AMX audit: {record.event}/{record.outcome}")
    eligibility = BASE._require_exact(audits, "eligibility", "eligible", 1)[0]
    _require(eligibility.fields.get("scope") == "model" and BASE._integer(eligibility, "layers") == 43,
             "candidate eligibility mismatch")
    contexts = BASE._require_exact(audits, "context", "eligible", 2)
    for record in contexts:
        _require(BASE._integer(record, "validation") == 0 and BASE._integer(record, "timing") == 0,
                 "candidate context enabled validation/timing")
    allocations = BASE._require_exact(audits, "context", "allocated", 2)
    for record in allocations:
        _require(BASE._integer(record, "worker_scratch_bytes") == 67_108_864 and
                 BASE._integer(record, "oracle") == 0,
                 "candidate worker/oracle allocation mismatch")
    outputs = BASE._require_exact(audits, "context_output", "allocated", 2)
    for record in outputs:
        _require(BASE._integer(record, "bytes") == BASE.OUTPUT_BYTES and
                 record.fields.get("base") not in (None, "0x0", "(nil)"),
                 "candidate output allocation mismatch")

    eligible_modes = BASE._require_exact(audits, "graph_mode", "eligible", 6)
    for record in eligible_modes:
        _require(BASE._integer(record, "mode") == 1 and BASE._integer(record, "tokens") == PROMPT_TOKENS and
                 BASE._integer(record, "lora") == 0 and BASE._integer(record, "pack_ready") == 0,
                 "candidate eligible graph-mode mismatch")
    expected_fallbacks = tuple(BASE.EXPECTED_GRAPH_FALLBACK_TOKENS) + (1,) * CONTINUATION_TOKENS
    expected_fallbacks = expected_fallbacks + expected_fallbacks
    fallbacks = BASE._require_exact(audits, "graph_mode", "fallback", len(expected_fallbacks))
    _require(tuple(BASE._integer(record, "tokens") for record in fallbacks) == expected_fallbacks,
             "candidate fallback token sequence mismatch")
    for record in fallbacks:
        _require(BASE._integer(record, "graph_type") == 0 and BASE._integer(record, "lora") == 0,
                 "candidate fallback graph-type/LoRA mismatch")

    bindings = BASE._require_exact(audits, "bindings", "complete", 6)
    _require([BASE._integer(record, "executable") for record in bindings] == [0, 0, 1, 0, 0, 1],
             "candidate binding lifecycle mismatch")
    for record in bindings:
        _check_binding(record)
    # Frozen before the rerun. llama_context rounded the failed a082 request to
    # 2304 before graph reservation, so requesting 2304 explicitly does not
    # change reserve topology. Each path builds the first one-token graph, then
    # rebuilds it at position 2051 when CSA/LID n_kv grows from 512 to 768.
    disabled_bindings = BASE._require_exact(audits, "bindings", "fallback", 4)
    for record in disabled_bindings:
        _require(record.fields.get("mode") == "disabled" and BASE._integer(record, "executable") == 1,
                 "candidate fallback binding mismatch")

    BASE._check_pack_provenance(audits)

    bounds = BASE._require_exact(audits, "graph_reuse", "bound", 2)
    submits = BASE._require_exact(audits, "graph_reuse", "submit", 2)
    for record in bounds:
        _require(BASE._integer(record, "mode") == 1 and BASE._integer(record, "prior_submissions") == 0,
                 "candidate graph binding mismatch")
    for record in submits:
        _require(BASE._integer(record, "mode") == 1 and BASE._integer(record, "submission") == 1 and
                 BASE._integer(record, "reused") == 0,
                 "candidate graph submission mismatch")

    callbacks = [record for record in audits if record.event == "callback"]
    expected_callbacks = [(outcome, layer, edge) for _ in PATHS for layer in range(43)
                          for edge in ("start", "end") for outcome in ("ask", "after")]
    actual_callbacks = [(record.outcome, BASE._integer(record, "layer"), record.fields.get("edge"))
                        for record in callbacks]
    _require(actual_callbacks == expected_callbacks, "candidate callback stream mismatch")
    for record in callbacks:
        _require(BASE._integer(record, "submission") == 1 and record.fields.get("error") == "ok",
                 "candidate callback state mismatch")
    visibility = BASE._require_exact(audits, "output_visibility", "ready", 86)
    _require([BASE._integer(record, "layer") for record in visibility] == list(range(43)) * 2,
             "candidate output visibility sequence mismatch")
    for record in visibility:
        _require(BASE._integer(record, "submission") == 1 and BASE._integer(record, "shared_buffer") == 1 and
                 BASE._integer(record, "upload") == 0 and BASE._integer(record, "worker_join") == 1 and
                 BASE._integer(record, "release_acquire") == 1,
                 "candidate output visibility contract mismatch")
    completions = BASE._require_exact(audits, "callbacks", "complete", 2)
    for record in completions:
        _require(BASE._integer(record, "submission") == 1 and BASE._integer(record, "expected_layer") == 43 and
                 record.fields.get("expected_edge") == "start" and
                 BASE._integer(record, "starts") == BASE.CALLBACK_MASK and
                 BASE._integer(record, "ends") == BASE.CALLBACK_MASK and
                 BASE._integer(record, "completed_layers") == 43,
                 "candidate 43-layer completion mismatch")
    graphs = BASE._require_exact(audits, "graph", "complete", 2)
    for record in graphs:
        _require(BASE._integer(record, "submission") == 1 and BASE._integer(record, "mode") == 1 and
                 BASE._integer(record, "layers") == 43,
                 "candidate graph completion mismatch")


def verify_process_log(text: str, mode: str, expected_commit: str, capture: dict[str, object], capture_bytes: int,
                       before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    _check_driver(text, mode, expected_commit, capture, capture_bytes)
    audits = BASE._records(text, BASE.AUDIT_RE, "dsv4_amx_audit")
    _require(bool(audits), f"{mode} log contained no AMX audit")
    if mode == "control":
        record = BASE._require_exact(audits, "eligibility", "fallback", 1)[0]
        _require(len(audits) == 1 and record.fields.get("scope") == "model" and
                 record.fields.get("reason") == "explicit_disable",
                 "control did not take the exact explicit-disable path")
    else:
        _check_candidate_audits(audits)
    resource_metrics = BASE._check_resource_health(text, before, after)
    return {
        "mode": mode,
        "audit_counts": {
            f"{event}/{outcome}": count
            for (event, outcome), count in sorted(Counter((record.event, record.outcome) for record in audits).items())
        },
        "resource_metrics": resource_metrics,
        "thermal_nominal": True,
        "system_swap_before_bytes": int(before["swap_used_bytes"]),
        "system_swap_after_bytes": int(after["swap_used_bytes"]),
    }


def verify_outputs(artifact_dir: Path, expected_commit: str) -> dict[str, object]:
    control_capture = artifact_dir / "control-logits.bin"
    candidate_capture = artifact_dir / "candidate-logits.bin"
    comparison = compare_captures(control_capture, candidate_capture, expected_commit)
    process_reports = {}
    for mode in ("control", "candidate"):
        before = _json(artifact_dir / f"resource-{mode}-before.json")
        after = _json(artifact_dir / f"resource-{mode}-after.json")
        log = (artifact_dir / f"{mode}.log").read_text(encoding="utf-8", errors="replace")
        process_reports[mode] = verify_process_log(
            log, mode, expected_commit, comparison, (artifact_dir / f"{mode}-logits.bin").stat().st_size,
            before, after)
    return {
        "outcome": "pass",
        "expected_commit": expected_commit,
        "benchmark": False,
        "timing_acceptance": False,
        "comparison": comparison,
        "processes": process_reports,
        "captures": {
            "control": {"path": str(control_capture), "sha256": _sha256(control_capture)},
            "candidate": {"path": str(candidate_capture), "sha256": _sha256(candidate_capture)},
        },
    }


def _run_logged(command: Sequence[str], cwd: Path, destination: Path, environment: dict[str, str],
                timeout_seconds: float) -> int:
    return BASE._run_logged(
        command, cwd, destination, env=environment, required=False, timeout_seconds=timeout_seconds)


def run_gate(args: argparse.Namespace) -> int:
    expected_commit = args.expected_commit
    now = dt.datetime.now(dt.timezone.utc)
    artifact_dir = (Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else
                    ROOT / "notes" /
                    f"{now:%Y-%m-%d}-dsv4-amx-full-model-gate-{now:%Y%m%dT%H%M%SZ}-{expected_commit[:9]}")
    _require(not artifact_dir.exists(), f"artifact directory already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)
    try:
        _require(bool(BASE.GIT_COMMIT_RE.fullmatch(expected_commit)), "expected commit must be a full lowercase SHA")
        _require(args.jobs > 0, "--jobs must be positive")
        _require(args.timeout_seconds > 0, "--timeout-seconds must be positive")
        _require(BASE._output(["git", "rev-parse", "HEAD"], ROOT) == expected_commit, "source commit mismatch")
        _require(BASE._output(["git", "status", "--porcelain"], ROOT) == "", "source worktree is dirty")
        BASE.require_clean_runtime_environment(os.environ)
        _require(platform.system() == "Darwin" and platform.machine() == "arm64", "gate requires macOS arm64")
        _require(BASE._output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], ROOT) == "Apple M2 Ultra",
                 "gate requires exact Apple M2 Ultra")

        oracle = verify_oracle_artifacts(Path(args.oracle_artifacts), expected_commit)
        (artifact_dir / "oracle-provenance.json").write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n",
                                                               encoding="utf-8")
        shards = BASE.canonical_model_shards(Path(args.model))
        model = shards[0]
        model_set = BASE.hash_model_set(shards, artifact_dir)
        _require(model_set["set_sha256"] == ORACLE_MODEL_SET_SHA256, "current model differs from oracle model")

        build_dir = Path(args.build_dir).expanduser()
        if not build_dir.is_absolute():
            build_dir = (ROOT / build_dir).resolve()
        environment = BASE.capture_environment(ROOT, expected_commit, build_dir, model)
        environment["artifact_dir"] = str(artifact_dir)
        environment["oracle_artifacts"] = oracle
        environment["gate_environments"] = MODE_ENVIRONMENTS
        environment.pop("gate_environment", None)
        environment["comparison_policy"] = {
            "name": "f32_quality_first",
            "metal_divergence_rejects": False,
            "legacy_nmse": LEGACY_LOGIT_NMSE_LIMIT,
            "legacy_max_abs": LEGACY_LOGIT_MAX_ABS_LIMIT,
            "legacy_kl": LEGACY_LOGIT_KL_LIMIT,
            "behavioral_quality_gate_required_after_telemetry": True,
        }
        environment["model_timeout_seconds"] = args.timeout_seconds
        environment["context_contract"] = {
            "logical_tokens": LOGICAL_TOKENS,
            "runtime_n_ctx": RUNTIME_N_CTX,
            "alignment": CONTEXT_ALIGNMENT,
            "padding_tokens": CONTEXT_PADDING,
            "binding_counts_frozen_before_rerun": True,
        }
        (artifact_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")

        cmake = "/opt/homebrew/bin/cmake"
        configure = [
            cmake, "-S", str(ROOT), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_METAL=ON",
            "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_ACCELERATE=ON",
            "-DGGML_NATIVE=ON",
            "-DGGML_OPENMP=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_SERVER=OFF",
        ]
        BASE._run_logged(configure, ROOT, artifact_dir / "configure.log")
        BASE._run_logged([cmake, "--build", str(build_dir), "--target", "llama-dsv4-amx-full-model-gate", "-j",
                          str(args.jobs)], ROOT, artifact_dir / "build.log")
        cache = build_dir / "CMakeCache.txt"
        _require(cache.is_file(), "configured build omitted CMakeCache.txt")
        shutil.copyfile(cache, artifact_dir / "cmake-cache.txt")
        binary = build_dir / "bin" / "llama-dsv4-amx-full-model-gate"
        _require(binary.is_file() and os.access(binary, os.X_OK), "full-model gate binary is missing")
        binary_sha256 = _sha256(binary)
        (artifact_dir / "binary-sha256.txt").write_text(f"{binary_sha256}  {binary}\n", encoding="utf-8")

        exit_codes: dict[str, int] = {}
        commands: dict[str, list[str]] = {}
        for mode in ("control", "candidate"):
            before = BASE.capture_resource(ROOT)
            (artifact_dir / f"resource-{mode}-before.json").write_text(
                json.dumps(before, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            capture = artifact_dir / f"{mode}-logits.bin"
            command = [
                "/usr/bin/time", "-l", str(binary), "--mode", mode, "--model", str(model),
                "--expected-commit", expected_commit, "--output", str(capture),
            ]
            commands[mode] = command
            exit_codes[mode] = _run_logged(command, ROOT, artifact_dir / f"{mode}.log",
                                           mode_child_environment(os.environ, mode), args.timeout_seconds)
            after = BASE.capture_resource(ROOT)
            (artifact_dir / f"resource-{mode}-after.json").write_text(
                json.dumps(after, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            _require(exit_codes[mode] == 0, f"{mode} process exited with status {exit_codes[mode]}")
            BASE._check_resource_health(
                (artifact_dir / f"{mode}.log").read_text(encoding="utf-8", errors="replace"), before, after)
            _require(_sha256(binary) == binary_sha256, "gate binary changed between fresh processes")

        _require(verify_oracle_artifacts(Path(args.oracle_artifacts), expected_commit) == oracle,
                 "retained oracle provenance changed during the full-model gate")
        report = verify_outputs(artifact_dir, expected_commit)
        report["artifact_dir"] = str(artifact_dir)
        report["binary"] = {"path": str(binary), "sha256": binary_sha256}
        report["commands"] = commands
        report["model"] = model_set
        report["oracle"] = oracle
        (artifact_dir / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
        (artifact_dir / "PASS").write_text("production AMX full-model telemetry gate passed\n", encoding="utf-8")
        BASE._artifact_manifest(artifact_dir)
        print(f"dsv4_amx_full_model_gate outcome=pass artifacts={artifact_dir}")
        return 0
    except Exception as exc:
        (artifact_dir / "FAIL").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        BASE._artifact_manifest(artifact_dir)
        print(f"dsv4_amx_full_model_gate outcome=fail artifacts={artifact_dir} reason={exc}", file=sys.stderr)
        return 1


def verify_command(args: argparse.Namespace) -> int:
    try:
        artifact_dir = Path(args.artifacts_dir).expanduser().resolve()
        environment = _json(artifact_dir / "environment.json")
        expected_commit = str(environment["expected_commit"])
        report = verify_outputs(artifact_dir, expected_commit)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="build and run fresh control/candidate processes")
    run.add_argument("--model", required=True, help="canonical first UD-Q8_K_XL shard")
    run.add_argument("--oracle-artifacts", required=True, help="retained passed strict-oracle artifact directory")
    run.add_argument("--expected-commit", required=True, help="full clean source commit")
    run.add_argument("--build-dir", default="build-m2-amx-full-model-gate")
    run.add_argument("--artifacts-dir", help="new artifact directory; defaults under notes/")
    run.add_argument("--jobs", type=int, default=8)
    run.add_argument("--timeout-seconds", type=int, default=BASE.DEFAULT_MODEL_TIMEOUT_SECONDS)
    run.set_defaults(function=run_gate)
    verify = subparsers.add_parser("verify", help="reverify retained captures and logs")
    verify.add_argument("--artifacts-dir", required=True)
    verify.set_defaults(function=verify_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

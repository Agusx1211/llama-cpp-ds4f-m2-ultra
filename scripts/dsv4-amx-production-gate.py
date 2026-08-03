#!/usr/bin/env python3
"""Fail-closed F32-oracle gate for the production DeepSeek V4 AMX path.

The ``run`` command is intended to execute inside ``scripts/m2-ultra-queue.sh``.
It configures and builds an oracle/telemetry driver, hashes the canonical model
shards, performs one 2048-token decode, and retains a self-contained artifact
directory under ``notes/``. The independent Accelerate F32 oracle and runtime
health contracts are hard gates. Differences from the retained Metal shared
FFN are telemetry only. The driver deliberately has no warmup, repetition,
timing sample, throughput calculation, or performance acceptance criterion.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import signal
import shutil
import stat
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


TOKENS = 2048
LAYERS = 43
PACKED_BYTES = 4_328_521_728
PACKED_VALUES = PACKED_BYTES // 4
TENSOR_ELEMENTS = 8_388_608
SOURCE_BYTES = TENSOR_ELEMENTS * 2
SOURCE_BYTES_TOTAL = 3 * LAYERS * SOURCE_BYTES
EXPECTED_CLAMP_BITS = 0x41200000
OUTPUT_BYTES = 33_554_432
DEFAULT_MODEL_TIMEOUT_SECONDS = 7200
CALLBACK_MASK = (1 << LAYERS) - 1
ORACLE_NMSE_LIMIT = 1.0e-12
ORACLE_MAX_ABS_LIMIT = 5.0e-4
CANONICAL_MODEL_PREFIX = "DeepSeek-V4-Flash-0731-UD-Q8_K_XL"
CANONICAL_MODEL_SHARDS = 5
CANONICAL_MODEL_SHA256 = (
    "d13ce8f90855547bdaebe7312f531a1f2c4f822178d3103951f27fe884395cfa",
    "3da2f2443063f83635986f9b67fa7e8e3d03c53b81a9a08d2007936612423610",
    "7d622a7760d359ec9257b3493ad531e3bf0bfbe6f6533267e16e6dde8153ddce",
    "6ed2bce452214f156b85e7c5f7d4fc242a3052f409d1b90a61422f60669c2de3",
    "ea4727af4888fdca0fff796ec81ac2f3ebb43c310b2feb4798f41d82744b42ea",
)
RUNTIME_ENV_PREFIXES = ("GGML_", "LLAMA_", "METAL_", "MTL_", "DYLD_", "ACCELERATE_", "OMP_")
GATE_ENVIRONMENT = {
    "LLAMA_DSV4_AMX_COEXEC": "1",
    "LLAMA_DSV4_AMX_COEXEC_VALIDATE": "1",
}
# Context setup makes nine fused-operation probes, then one token-generation
# reserve between the two exact pp2048 reserves.
EXPECTED_GRAPH_FALLBACK_TOKENS = (1, 16, 1, 1, 1, 1, 512, 1, 1, 1)

AUDIT_RE = re.compile(r"dsv4_amx_audit\s+event=(\S+)\s+outcome=(\S+)(?:\s+(.*))?$")
DRIVER_RE = re.compile(r"dsv4_amx_gate_driver\s+event=(\S+)\s+outcome=(\S+)(?:\s+(.*))?$")
TIME_SWAP_RE = re.compile(r"^\s*(\d+)\s+(swaps|swapins|swapouts)\s*$", re.MULTILINE)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
THERMAL_NOMINAL_MESSAGES = (
    "No thermal warning level has been recorded",
    "No performance warning level has been recorded",
    "No CPU power status has been recorded",
)


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Record:
    event: str
    outcome: str
    fields: dict[str, str]
    line: str


def _fields(detail: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in (detail or "").split():
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", token)
        _require(match is not None, f"malformed audit field: {token}")
        key, value = match.groups()
        _require(key not in result, f"duplicate audit field: {key}")
        result[key] = value
    return result


def _records(text: str, pattern: re.Pattern[str], prefix: str) -> list[Record]:
    result: list[Record] = []
    for line in text.splitlines():
        if prefix not in line:
            continue
        _require(line.count(prefix) == 1, f"ambiguous {prefix} record: {line}")
        match = pattern.search(line)
        _require(match is not None and match.start() == line.index(prefix), f"malformed {prefix} record: {line}")
        result.append(Record(match.group(1), match.group(2), _fields(match.group(3)), line))
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def _require_exact(records: Sequence[Record], event: str, outcome: str, count: int) -> list[Record]:
    selected = [record for record in records if record.event == event and record.outcome == outcome]
    _require(len(selected) == count, f"expected {count} {event}/{outcome} records, found {len(selected)}")
    return selected


def _integer(record: Record, field: str) -> int:
    _require(field in record.fields, f"{record.event}/{record.outcome} omitted {field}")
    try:
        return int(record.fields[field], 0)
    except ValueError as exc:
        raise GateFailure(f"invalid integer {field}={record.fields[field]}") from exc


def _floating(record: Record, field: str) -> float:
    _require(field in record.fields, f"{record.event}/{record.outcome} omitted {field}")
    try:
        value = float(record.fields[field])
    except ValueError as exc:
        raise GateFailure(f"invalid float {field}={record.fields[field]}") from exc
    _require(value == value and abs(value) != float("inf"), f"nonfinite {field}={record.fields[field]}")
    return value


def _check_allowed_audits(records: Sequence[Record]) -> None:
    allowed = {
        ("eligibility", "eligible"),
        ("context", "eligible"),
        ("context", "allocated"),
        ("context_output", "allocated"),
        ("graph_mode", "eligible"),
        ("graph_mode", "fallback"),
        ("bindings", "complete"),
        ("pack", "begin"),
        ("pack", "complete"),
        ("pack_tensor", "pass"),
        ("pack_contract", "complete"),
        ("oracle", "begin"),
        ("oracle", "pass"),
        ("oracle", "complete"),
        ("graph_reuse", "bound"),
        ("graph_reuse", "submit"),
        ("callback", "ask"),
        ("callback", "after"),
        ("output_visibility", "ready"),
        ("metal_telemetry", "recorded"),
        ("callbacks", "complete"),
        ("graph", "complete"),
    }
    for record in records:
        _require((record.event, record.outcome) in allowed,
                 f"unexpected AMX audit outcome: {record.event}/{record.outcome}")


def _check_driver(records: Sequence[Record], expected_commit: str) -> None:
    expected_order = [
        ("start", "begin"),
        ("decode", "begin"),
        ("decode", "pass"),
        ("context_teardown", "pass"),
        ("model_teardown", "pass"),
        ("backend_teardown", "pass"),
        ("complete", "pass"),
    ]
    actual_order = [(record.event, record.outcome) for record in records]
    _require(actual_order == expected_order, f"driver lifecycle mismatch: {actual_order}")
    start = records[0]
    _require(start.fields.get("source_commit") == expected_commit, "driver source commit mismatch")
    build_commit = start.fields.get("build_commit", "")
    _require(len(build_commit) >= 7 and expected_commit.startswith(build_commit), "driver build commit mismatch")
    for record in (records[0], records[1], records[2], records[-1]):
        _require(_integer(record, "tokens") == TOKENS, "driver token contract mismatch")
        _require(_integer(record, "submissions") == 1, "driver submission contract mismatch")
    _require(_integer(records[0], "benchmark") == 0 and _integer(records[-1], "benchmark") == 0,
             "driver unexpectedly enabled benchmarking")


def _check_pack_provenance(records: Sequence[Record]) -> None:
    pack_lifecycle = [(record.event, record.outcome) for record in records
                      if record.event in ("pack", "pack_tensor", "pack_contract")]
    expected_lifecycle = ([("pack", "begin")] + [("pack_tensor", "pass")] * (3 * LAYERS) +
                          [("pack_contract", "complete"), ("pack", "complete")])
    _require(pack_lifecycle == expected_lifecycle, "packed tensor audit lifecycle mismatch")
    pack_begin = _require_exact(records, "pack", "begin", 1)[0]
    pack_complete = _require_exact(records, "pack", "complete", 1)[0]
    _require(_integer(pack_begin, "expected_bytes") == PACKED_BYTES, "pack begin byte contract mismatch")
    _require(_integer(pack_complete, "bytes") == PACKED_BYTES, "packed byte contract mismatch")
    _require(_integer(pack_complete, "layers") == LAYERS, "packed layer count mismatch")
    pack_tensors = _require_exact(records, "pack_tensor", "pass", 3 * LAYERS)
    expected_tensors = [(layer, role) for layer in range(LAYERS) for role in ("gate", "up", "down")]
    _require([(_integer(record, "layer"), record.fields.get("role")) for record in pack_tensors] == expected_tensors,
             "packed tensor layer/role sequence mismatch")
    tensor_addresses: set[int] = set()
    source_ranges: list[tuple[int, int]] = []
    for record in pack_tensors:
        layer = _integer(record, "layer")
        role = record.fields.get("role", "")
        _require(record.fields.get("name") == f"blk.{layer}.ffn_{role}_shexp.weight",
                 "packed tensor name identity mismatch")
        _require(_integer(record, "elements") == TENSOR_ELEMENTS and
                 _integer(record, "exact_bf16_bits") == 1 and
                 record.fields.get("mapping") == "output_input_to_panel_k_col",
                 "packed tensor expansion/mapping contract mismatch")
        address = _integer(record, "tensor")
        _require(address > 0 and address not in tensor_addresses,
                 "packed tensor address identity mismatch")
        tensor_addresses.add(address)
        source = _integer(record, "source")
        source_size = _integer(record, "source_bytes")
        _require(source > 0 and source_size == SOURCE_BYTES, "packed source storage range mismatch")
        source_ranges.append((source, source + source_size))
        _require(_integer(record, "clamp_bits") == EXPECTED_CLAMP_BITS,
                 "packed split-SwiGLU clamp metadata mismatch")
    source_ranges.sort()
    _require(all(first[1] <= second[0] for first, second in zip(source_ranges, source_ranges[1:])),
             "packed source storage ranges overlap or alias")
    pack_contract = _require_exact(records, "pack_contract", "complete", 1)[0]
    _require(_integer(pack_contract, "layers") == LAYERS and
             _integer(pack_contract, "matrices") == 3 * LAYERS and
             _integer(pack_contract, "unique_tensors") == 3 * LAYERS and
             _integer(pack_contract, "unique_sources") == 3 * LAYERS and
             _integer(pack_contract, "source_bytes") == SOURCE_BYTES_TOTAL and
             _integer(pack_contract, "values") == PACKED_VALUES,
             "packed layer/tensor/value identity contract mismatch")
    _require(_integer(pack_contract, "exact_bf16_expand") == 1 and
             _integer(pack_contract, "clamp_metadata") == 1 and
             _integer(pack_contract, "clamp_layers") == LAYERS and
             pack_contract.fields.get("orientation") == "output_input_to_panel_k_col",
             "packed conversion/orientation/clamp contract mismatch")


def _check_pack_and_storage(records: Sequence[Record]) -> None:
    eligibility = _require_exact(records, "eligibility", "eligible", 1)[0]
    _require(eligibility.fields.get("scope") == "model" and _integer(eligibility, "layers") == LAYERS,
             "model eligibility contract mismatch")
    context = _require_exact(records, "context", "eligible", 1)[0]
    _require(_integer(context, "validation") == 1 and _integer(context, "timing") == 0,
             "context was not correctness-only validation mode")
    context_allocation = _require_exact(records, "context", "allocated", 1)[0]
    _require(_integer(context_allocation, "worker_scratch_bytes") == 67_108_864 and
             _integer(context_allocation, "oracle") == 1,
             "worker allocation/oracle contract mismatch")
    allocation = _require_exact(records, "context_output", "allocated", 1)[0]
    _require(_integer(allocation, "bytes") == OUTPUT_BYTES, "shared output allocation was not exactly 32 MiB")
    _require(allocation.fields.get("base") not in (None, "0x0", "(nil)"), "shared output allocation had a null base")

    graph_modes = _require_exact(records, "graph_mode", "eligible", 3)
    for record in graph_modes:
        _require(_integer(record, "mode") == 2 and _integer(record, "tokens") == TOKENS and
                 _integer(record, "lora") == 0,
                 "eligible graph-mode contract mismatch")
    fallbacks = _require_exact(records, "graph_mode", "fallback", len(EXPECTED_GRAPH_FALLBACK_TOKENS))
    _require([_integer(record, "tokens") for record in fallbacks] == list(EXPECTED_GRAPH_FALLBACK_TOKENS),
             "graph-mode fallback probe/reserve sequence mismatch")
    for record in fallbacks:
        _require(_integer(record, "graph_type") == 0 and _integer(record, "lora") == 0,
                 "graph-mode fallback contract mismatch")

    _check_pack_provenance(records)

    bindings = [record for record in records if record.event == "bindings" and record.outcome == "complete"]
    _require(len(bindings) == 3, "expected two reserve bindings and one executable binding")
    _require(sum(_integer(record, "executable") == 1 for record in bindings) == 1,
             "expected exactly one executable graph binding")
    _require(sum(_integer(record, "executable") == 0 for record in bindings) == 2,
             "expected exactly two reserve graph bindings")
    for record in bindings:
        _require(_integer(record, "layers") == LAYERS, "binding layer count mismatch")
        _require(_integer(record, "callback_events") == 2 * LAYERS, "binding callback cardinality mismatch")
        _require(_integer(record, "tensor_metadata") == LAYERS, "binding metadata cardinality mismatch")
        _require(_integer(record, "external_output_bytes") == OUTPUT_BYTES, "binding output byte mismatch")
        _require(_integer(record, "gallocr_output_bytes") == 0, "gallocr retained AMX output bytes")
        for field in ("unique_buffers", "unique_data", "producerless", "visible_before_consumer",
                      "routed_join_exact"):
            _require(_integer(record, field) == 1, f"binding violated {field}")
        _require(record.fields.get("callback_edges") == "norm_completion", "binding callback-edge contract mismatch")
        _require(_integer(record, "routed_joins") == LAYERS and
                 _integer(record, "telemetry_consumers") == LAYERS and
                 _integer(record, "direct_consumer_edges") == 2 * LAYERS,
                 "binding routed-join/telemetry topology mismatch")


def _check_oracle(records: Sequence[Record]) -> None:
    begin = _require_exact(records, "oracle", "begin", 1)[0]
    complete = _require_exact(records, "oracle", "complete", 1)[0]
    passes = _require_exact(records, "oracle", "pass", LAYERS * 4)
    _require(_integer(begin, "layers") == LAYERS and _integer(begin, "rows") == 4,
             "oracle begin geometry mismatch")
    _require(_floating(begin, "nmse_limit") == ORACLE_NMSE_LIMIT, "oracle NMSE limit mismatch")
    _require(_floating(begin, "max_abs_limit") == ORACLE_MAX_ABS_LIMIT, "oracle max-abs limit mismatch")
    _require(_integer(complete, "layers") == LAYERS and _integer(complete, "stage_records") == LAYERS * 4,
             "oracle completion cardinality mismatch")

    expected = [(layer, stage) for layer in range(LAYERS) for stage in ("gate", "up", "post_swiglu", "down")]
    actual = [(_integer(record, "layer"), record.fields.get("stage", "")) for record in passes]
    _require(actual == expected, "oracle layer/stage sequence mismatch")
    for record in passes:
        stage = record.fields["stage"]
        cols = 4096 if stage == "down" else 2048
        _require(_integer(record, "full_actual_elements") == TOKENS * cols,
                 "oracle full-output geometry mismatch")
        _require(_integer(record, "compared") == 4 * cols, "oracle sampled-reference geometry mismatch")
        for flag in ("finite", "nonzero", "reference_l2_valid", "metrics_finite"):
            _require(_integer(record, flag) == 1, f"oracle {flag} gate failed")
        _require(_floating(record, "nmse_limit") == ORACLE_NMSE_LIMIT, "oracle record NMSE limit mismatch")
        _require(_floating(record, "max_abs_limit") == ORACLE_MAX_ABS_LIMIT,
                 "oracle record max-abs limit mismatch")
        _require(_floating(record, "nmse") <= ORACLE_NMSE_LIMIT, "oracle NMSE exceeded limit")
        _require(_floating(record, "max_abs") <= ORACLE_MAX_ABS_LIMIT, "oracle max-abs exceeded limit")
        _require(_floating(record, "reference_l2") > 0.0, "oracle reference L2 was not positive")


def _check_callbacks_and_telemetry(records: Sequence[Record]) -> dict[str, float | int]:
    callback_records = [record for record in records if record.event == "callback"]
    expected_callbacks = [
        (outcome, layer, edge)
        for layer in range(LAYERS)
        for edge in ("start", "end")
        for outcome in ("ask", "after")
    ]
    actual_callbacks = [
        (record.outcome, _integer(record, "layer"), record.fields.get("edge", "")) for record in callback_records
    ]
    _require(actual_callbacks == expected_callbacks, "callback ask/after stream mismatch")
    for record in callback_records:
        _require(record.fields.get("error") == "ok", "callback reported a non-ok state")
        _require(_integer(record, "submission") == 1, "callback submission mismatch")

    telemetry = _require_exact(records, "metal_telemetry", "recorded", LAYERS)
    _require([_integer(record, "layer") for record in telemetry] == list(range(LAYERS)),
             "retained Metal telemetry layer sequence mismatch")
    for record in telemetry:
        _require(record.fields.get("stage") == "down", "retained Metal telemetry stage mismatch")
        for field in ("direct_finite", "direct_nonzero", "reference_finite", "reference_l2_valid",
                      "metrics_finite"):
            _require(_integer(record, field) == 1, f"retained Metal telemetry health gate failed: {field}")
        _require(_floating(record, "nmse") >= 0.0, "retained Metal telemetry NMSE was negative")
        _require(_floating(record, "max_abs") >= 0.0, "retained Metal telemetry max-abs was negative")
        _require(_floating(record, "reference_l2") > 0.0, "retained Metal reference L2 was not positive")
        _require(_integer(record, "divergence_rejects") == 0,
                 "retained Metal telemetry unexpectedly enabled divergence rejection")

    visibility = _require_exact(records, "output_visibility", "ready", LAYERS)
    _require([_integer(record, "layer") for record in visibility] == list(range(LAYERS)),
             "output visibility layer sequence mismatch")
    for record in visibility:
        _require(_integer(record, "submission") == 1, "output visibility submission mismatch")
        for field in ("shared_buffer", "worker_join", "release_acquire"):
            _require(_integer(record, field) == 1, f"output visibility omitted {field}")
        _require(_integer(record, "upload") == 0, "output visibility unexpectedly uploaded data")

    callbacks = _require_exact(records, "callbacks", "complete", 1)[0]
    _require(_integer(callbacks, "submission") == 1, "completion submission mismatch")
    _require(_integer(callbacks, "expected_layer") == LAYERS, "completion expected-layer mismatch")
    _require(callbacks.fields.get("expected_edge") == "start", "completion expected-edge mismatch")
    _require(_integer(callbacks, "starts") == CALLBACK_MASK, "start callback mask mismatch")
    _require(_integer(callbacks, "ends") == CALLBACK_MASK, "end callback mask mismatch")
    _require(_integer(callbacks, "completed_layers") == LAYERS, "completed layer count mismatch")

    bound = _require_exact(records, "graph_reuse", "bound", 1)[0]
    _require(_integer(bound, "mode") == 2 and _integer(bound, "prior_submissions") == 0,
             "graph binding mode/submission contract mismatch")
    _require(bound.fields.get("graph") not in (None, "0x0", "(nil)"), "graph binding had a null graph")
    submit = _require_exact(records, "graph_reuse", "submit", 1)[0]
    _require(_integer(submit, "mode") == 2 and _integer(submit, "submission") == 1 and
             _integer(submit, "reused") == 0,
             "graph submission/reuse contract mismatch")
    graph = _require_exact(records, "graph", "complete", 1)[0]
    _require(_integer(graph, "submission") == 1 and _integer(graph, "mode") == 2 and
             _integer(graph, "layers") == LAYERS,
             "graph completion contract mismatch")
    return {
        "max_nmse": max(_floating(record, "nmse") for record in telemetry),
        "max_abs": max(_floating(record, "max_abs") for record in telemetry),
        "outside_legacy_nmse_layers": sum(_integer(record, "within_legacy_nmse") == 0
                                            for record in telemetry),
    }


def _check_resource_health(run_log: str, before: dict[str, object], after: dict[str, object]) -> dict[str, int]:
    samples = [(name, int(value)) for value, name in TIME_SWAP_RE.findall(run_log)]
    _require(bool(samples), "/usr/bin/time -l emitted no recognized swap metric")
    _require(all(value == 0 for _, value in samples), f"process swap metric was nonzero: {samples}")
    metrics = {name: value for name, value in samples}
    before_used = int(before.get("swap_used_bytes", -1))
    after_used = int(after.get("swap_used_bytes", -1))
    _require(before_used >= 0 and after_used >= 0, "system swap usage was not captured")
    _require(after_used <= before_used, f"system swap usage increased from {before_used} to {after_used}")
    expected_thermal = {f"Note: {message}" for message in THERMAL_NOMINAL_MESSAGES}
    for phase, resource in (("before", before), ("after", after)):
        thermal = {line.strip() for line in str(resource.get("thermal_state", "")).splitlines() if line.strip()}
        _require(thermal == expected_thermal, f"{phase} thermal state was not nominal: {sorted(thermal)}")
    metrics["thermal_nominal"] = 1
    return metrics


def verify_gate(run_log: str, expected_commit: str, before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    _require(bool(GIT_COMMIT_RE.fullmatch(expected_commit)), "expected commit must be a full lowercase Git SHA")
    audits = _records(run_log, AUDIT_RE, "dsv4_amx_audit")
    drivers = _records(run_log, DRIVER_RE, "dsv4_amx_gate_driver")
    _require(bool(audits), "run log contained no AMX audit records")
    _require(bool(drivers), "run log contained no validation-driver records")
    _check_allowed_audits(audits)
    _check_driver(drivers, expected_commit)
    _check_pack_and_storage(audits)
    _check_oracle(audits)
    metal_telemetry = _check_callbacks_and_telemetry(audits)
    resource_metrics = _check_resource_health(run_log, before, after)
    counts = Counter((record.event, record.outcome) for record in audits)
    return {
        "outcome": "pass",
        "expected_commit": expected_commit,
        "tokens": TOKENS,
        "submissions": 1,
        "benchmark": False,
        "packed_bytes": PACKED_BYTES,
        "output_bytes": OUTPUT_BYTES,
        "oracle_passes": counts[("oracle", "pass")],
        "metal_telemetry_records": counts[("metal_telemetry", "recorded")],
        "metal_telemetry": metal_telemetry,
        "completed_layers": LAYERS,
        "callback_mask": f"0x{CALLBACK_MASK:x}",
        "resource_metrics": resource_metrics,
        "thermal_nominal": True,
        "system_swap_before_bytes": int(before["swap_used_bytes"]),
        "system_swap_after_bytes": int(after["swap_used_bytes"]),
        "audit_counts": {f"{event}/{outcome}": count for (event, outcome), count in sorted(counts.items())},
    }


def canonical_model_shards(first_shard: Path) -> list[Path]:
    first_shard = first_shard.expanduser()
    expected_name = f"{CANONICAL_MODEL_PREFIX}-00001-of-00005.gguf"
    _require(first_shard.name == expected_name, f"model must be canonical first shard {expected_name}")
    shards = [
        first_shard.with_name(f"{CANONICAL_MODEL_PREFIX}-{index:05d}-of-{CANONICAL_MODEL_SHARDS:05d}.gguf")
        for index in range(1, CANONICAL_MODEL_SHARDS + 1)
    ]
    for shard in shards:
        _require(shard.is_file() and not shard.is_symlink(), f"missing or non-regular canonical model shard: {shard}")
    return [shard.resolve() for shard in shards]


def validate_canonical_model_digests(digests: Sequence[str]) -> None:
    _require(len(digests) == CANONICAL_MODEL_SHARDS,
             f"expected {CANONICAL_MODEL_SHARDS} canonical model digests, found {len(digests)}")
    for index, (actual, expected) in enumerate(zip(digests, CANONICAL_MODEL_SHA256), start=1):
        _require(actual == expected,
                 f"canonical model shard {index:05d} SHA-256 mismatch: expected {expected}, found {actual}")


def hash_model_set(shards: Sequence[Path], artifact_dir: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    manifest_lines: list[str] = []
    for shard in shards:
        print(f"hashing model shard: {shard}", flush=True)
        digest = hashlib.sha256()
        with shard.open("rb") as source:
            before = os.fstat(source.fileno())
            _require(stat.S_ISREG(before.st_mode), f"model shard is not a regular file: {shard}")
            for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(source.fileno())
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        _require(identity_before == identity_after, f"model shard changed while hashing: {shard}")
        value = digest.hexdigest()
        manifest_lines.append(f"{value}  {shard.name}\n")
        entries.append({
            "path": str(shard),
            "name": shard.name,
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": value,
        })
    manifest = "".join(manifest_lines)
    (artifact_dir / "model-sha256.txt").write_text(manifest, encoding="utf-8")
    validate_canonical_model_digests([str(entry["sha256"]) for entry in entries])
    result = {
        "id": CANONICAL_MODEL_PREFIX,
        "first_shard": str(shards[0]),
        "shard_count": len(shards),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "set_sha256": hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
        "expected_sha256": list(CANONICAL_MODEL_SHA256),
        "shards": entries,
    }
    (artifact_dir / "model-set.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _output(command: Sequence[str], cwd: Path, *, required: bool = True) -> str:
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if required and result.returncode != 0:
        raise GateFailure(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}")
    return result.stdout.strip()


def capture_resource(root: Path) -> dict[str, object]:
    raw = _output(["/usr/sbin/sysctl", "vm.swapusage"], root)
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", raw)
    _require(match is not None, f"could not parse vm.swapusage: {raw}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[match.group(2)]
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "swapusage_raw": raw,
        "swap_used_bytes": int(float(match.group(1)) * scale),
        "memory_pressure": _output(["/usr/bin/memory_pressure", "-Q"], root, required=False),
        "vm_stat": _output(["/usr/bin/vm_stat"], root, required=False),
        "thermal_state": _output(["/usr/bin/pmset", "-g", "therm"], root),
    }


def capture_environment(root: Path, expected_commit: str, build_dir: Path, model: Path) -> dict[str, object]:
    relevant_env: dict[str, str] = {}
    for key, value in sorted(os.environ.items()):
        if key.startswith(RUNTIME_ENV_PREFIXES):
            relevant_env[key] = "<redacted>" if key.endswith(("TOKEN", "KEY", "SECRET")) else value
    return {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_commit": expected_commit,
        "git_head": _output(["git", "rev-parse", "HEAD"], root),
        "git_status_porcelain": _output(["git", "status", "--porcelain"], root),
        "git_log": _output(["git", "log", "-1", "--format=fuller"], root),
        "model": str(model),
        "build_dir": str(build_dir),
        "uname": _output(["/usr/bin/uname", "-a"], root),
        "machine": platform.machine(),
        "macos": _output(["/usr/bin/sw_vers"], root),
        "cpu_brand": _output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], root),
        "hw_memsize": _output(["/usr/sbin/sysctl", "-n", "hw.memsize"], root),
        "hw_ncpu": _output(["/usr/sbin/sysctl", "-n", "hw.ncpu"], root),
        "sdk_path": _output(["/usr/bin/xcrun", "--show-sdk-path"], root),
        "sdk_version": _output(["/usr/bin/xcrun", "--show-sdk-version"], root),
        "clang": _output(["/usr/bin/clang++", "--version"], root),
        "cmake": _output(["/opt/homebrew/bin/cmake", "--version"], root),
        "relevant_parent_environment": relevant_env,
        "gate_environment": GATE_ENVIRONMENT,
    }


def forbidden_runtime_environment(environment: Mapping[str, str]) -> list[str]:
    return sorted(key for key in environment if key.startswith(RUNTIME_ENV_PREFIXES))


def require_clean_runtime_environment(environment: Mapping[str, str]) -> None:
    forbidden = forbidden_runtime_environment(environment)
    _require(not forbidden, "forbidden ambient runtime environment: " + ", ".join(forbidden))


def gate_child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    child = {key: value for key, value in environment.items() if not key.startswith(RUNTIME_ENV_PREFIXES)}
    child.update(GATE_ENVIRONMENT)
    return child


def _run_logged(command: Sequence[str], cwd: Path, destination: Path, *, env: dict[str, str] | None = None,
                required: bool = True, timeout_seconds: float | None = None) -> int:
    with destination.open("w", encoding="utf-8") as output:
        output.write("command=" + json.dumps(list(command)) + "\n")
        output.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, text=True, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=timeout_seconds is not None)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            output.write(f"timeout_seconds={timeout_seconds}\n")
            output.flush()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            output.write("outcome=timeout\n")
            raise GateFailure(f"command timed out after {timeout_seconds} seconds; see {destination}") from exc
        output.write(f"exit_code={returncode}\n")
    if required and returncode != 0:
        raise GateFailure(f"command failed ({returncode}); see {destination}")
    return returncode


def _artifact_manifest(artifact_dir: Path) -> None:
    lines: list[str] = []
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file() or path.name == "artifact-sha256.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}\n")
    (artifact_dir / "artifact-sha256.txt").write_text("".join(lines), encoding="utf-8")


def run_gate(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[1]
    expected_commit = args.expected_commit
    now = dt.datetime.now(dt.timezone.utc)
    artifact_dir = (Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else
                    root / "notes" /
                    f"{now:%Y-%m-%d}-dsv4-production-amx-gate-{now:%Y%m%dT%H%M%SZ}-{expected_commit[:9]}")
    _require(not artifact_dir.exists(), f"artifact directory already exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True)

    try:
        _require(bool(GIT_COMMIT_RE.fullmatch(expected_commit)), "--expected-commit must be a full lowercase Git SHA")
        _require(args.jobs > 0, "--jobs must be positive")
        _require(args.timeout_seconds > 0, "--timeout-seconds must be positive")
        head = _output(["git", "rev-parse", "HEAD"], root)
        _require(head == expected_commit, f"source commit mismatch: expected {expected_commit}, found {head}")
        _require(_output(["git", "status", "--porcelain"], root) == "", "source worktree is dirty")
        require_clean_runtime_environment(os.environ)
        _require(platform.system() == "Darwin" and platform.machine() == "arm64", "gate requires macOS arm64")
        cpu_brand = _output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], root)
        _require(cpu_brand == "Apple M2 Ultra", f"gate requires exact Apple M2 Ultra, found {cpu_brand}")

        shards = canonical_model_shards(Path(args.model))
        model = shards[0]
        build_dir = Path(args.build_dir).expanduser()
        if not build_dir.is_absolute():
            build_dir = (root / build_dir).resolve()

        environment = capture_environment(root, expected_commit, build_dir, model)
        environment["model_timeout_seconds"] = args.timeout_seconds
        environment["artifact_dir"] = str(artifact_dir)
        (artifact_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")
        model_set = hash_model_set(shards, artifact_dir)

        cmake = "/opt/homebrew/bin/cmake"
        configure = [
            cmake, "-S", str(root), "-B", str(build_dir),
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
        _run_logged(configure, root, artifact_dir / "configure.log")
        _run_logged([cmake, "--build", str(build_dir), "--target", "llama-dsv4-amx-production-gate", "-j",
                     str(args.jobs)], root, artifact_dir / "build.log")
        cache = build_dir / "CMakeCache.txt"
        _require(cache.is_file(), "configured build omitted CMakeCache.txt")
        shutil.copyfile(cache, artifact_dir / "cmake-cache.txt")

        binary = build_dir / "bin" / "llama-dsv4-amx-production-gate"
        _require(binary.is_file() and os.access(binary, os.X_OK), f"gate binary missing or not executable: {binary}")
        before = capture_resource(root)
        (artifact_dir / "resource-before.json").write_text(json.dumps(before, indent=2, sort_keys=True) + "\n",
                                                            encoding="utf-8")

        child_env = gate_child_environment(os.environ)
        command = [
            "/usr/bin/time", "-l", str(binary), "--model", str(model), "--expected-commit", expected_commit,
        ]
        run_exit_code = _run_logged(
            command, root, artifact_dir / "run.log", env=child_env, required=False,
            timeout_seconds=args.timeout_seconds)

        after = capture_resource(root)
        (artifact_dir / "resource-after.json").write_text(json.dumps(after, indent=2, sort_keys=True) + "\n",
                                                           encoding="utf-8")
        run_log = (artifact_dir / "run.log").read_text(encoding="utf-8", errors="replace")
        report = verify_gate(run_log, expected_commit, before, after)
        _require(run_exit_code == 0, f"validation driver exited with status {run_exit_code}")
        report["artifact_dir"] = str(artifact_dir)
        report["model"] = model_set
        report["binary"] = str(binary)
        report["run_command"] = command
        (artifact_dir / "verification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
        (artifact_dir / "PASS").write_text("production AMX F32 oracle gate passed\n", encoding="utf-8")
        _artifact_manifest(artifact_dir)
        print(f"dsv4_amx_production_gate outcome=pass artifacts={artifact_dir}")
        return 0
    except Exception as exc:
        (artifact_dir / "FAIL").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        _artifact_manifest(artifact_dir)
        print(f"dsv4_amx_production_gate outcome=fail artifacts={artifact_dir} reason={exc}", file=sys.stderr)
        return 1


def verify_command(args: argparse.Namespace) -> int:
    try:
        before = json.loads(Path(args.resource_before).read_text(encoding="utf-8"))
        after = json.loads(Path(args.resource_after).read_text(encoding="utf-8"))
        report = verify_gate(Path(args.log).read_text(encoding="utf-8", errors="replace"),
                             args.expected_commit, before, after)
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            Path(args.output).write_text(output, encoding="utf-8")
        else:
            print(output, end="")
        return 0
    except Exception as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="build and run the one-submission production gate")
    run.add_argument("--model", required=True, help="canonical first UD-Q8_K_XL shard")
    run.add_argument("--expected-commit", required=True, help="full clean source commit expected in binary and tree")
    run.add_argument("--build-dir", default="build-m2-amx-gate")
    run.add_argument("--artifacts-dir", help="new artifact directory; defaults under notes/")
    run.add_argument("--jobs", type=int, default=8)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_MODEL_TIMEOUT_SECONDS)
    run.set_defaults(function=run_gate)

    verify = subparsers.add_parser("verify", help="verify retained gate artifacts without running the model")
    verify.add_argument("--log", required=True)
    verify.add_argument("--resource-before", required=True)
    verify.add_argument("--resource-after", required=True)
    verify.add_argument("--expected-commit", required=True)
    verify.add_argument("--output")
    verify.set_defaults(function=verify_command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Analyze DSpark drafter GPU cost at two live-context depths.

The companion harness makes two unprofiled deep-context references and two
fresh KPROF processes at each depth.  Every deep process consumes its own
byte-identical LCPC clone, while an immutable template remains outside all
``--cache-disk`` directories.

The E1 response-level sensitivity statistic is computed for every combination
of a near process mean, deep process mean, and unprofiled deep response:

    100 * (deep steady-width cost - near steady-width cost) * C
    ----------------------------------------------------------------
                   unprofiled deep predicted_ms

``C`` is the exact number of steady-width SPECTRACE draft calls in the
identical 16-token deep response.  The eight combinations form an observed
sensitivity envelope, not a confidence interval and not eight independent
observations.
The result is rejected unless replay, response, trace, graph, timestamp,
scratch-buffer, and immutable-template gates all pass.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InvalidRun(RuntimeError):
    """The measurement cannot support a performance decision."""


@dataclass(frozen=True)
class DraftCompute:
    uid: int
    seq: int
    width: int
    fa_ns: int
    indexer_ns: int
    compressor_ns: int
    total_graph_ns: int

    @property
    def measured_ns(self) -> int:
        return self.fa_ns + self.indexer_ns + self.compressor_ns


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidRun(message)


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return path.open("rt", errors="replace")


def resolve_log(root: Path, value: str) -> Path:
    path = root / value
    if path.exists():
        return path
    gz = Path(str(path) + ".gz")
    require(gz.exists(), f"missing server log: {path} (or {gz})")
    return gz


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidRun(f"cannot read {path}: {exc}") from exc


def node_src_names(node: dict[str, Any]) -> Iterable[str]:
    for src in node.get("src", []):
        yield str(src.get("name", ""))


def is_dspark_graph(nodes: dict[int, dict[str, Any]]) -> bool:
    """Recognize the production DSpark decode graph, not a target graph."""
    src_names = [name for node in nodes.values() for name in node_src_names(node)]
    has_markov = any(name.startswith("markov_w1.weight") for name in src_names)
    has_confidence = any(name.startswith("conf_proj.weight") for name in src_names)
    fas = [node for node in nodes.values() if node.get("op") == "FLASH_ATTN_EXT"]
    return has_markov and has_confidence and len(fas) == 3


def draft_width(nodes: dict[int, dict[str, Any]]) -> int:
    widths: list[int] = []
    for node in nodes.values():
        if node.get("op") != "FLASH_ATTN_EXT":
            continue
        src = node.get("src", [])
        require(src and len(src[0].get("ne", [])) >= 2, "malformed DSpark FA graph")
        widths.append(int(src[0]["ne"][1]))
    require(len(widths) == 3 and len(set(widths)) == 1,
            f"DSpark graph has inconsistent FA widths: {widths}")
    return widths[0]


def is_indexer_node(node: dict[str, Any]) -> bool:
    op = str(node.get("op", ""))
    if "LIGHTNING_INDEXER" in op:
        return True
    if op == "TOP_K":
        return any(name.startswith("lid_score") for name in node_src_names(node))
    return False


def is_compressor_node(node: dict[str, Any]) -> bool:
    if node.get("op") == "DSV4_COMPRESS":
        return True
    return "_state_compress" in str(node.get("name", ""))


def parse_kprof(log_path: Path, begin_line: int, end_line: int) -> list[DraftCompute]:
    """Return all signed DSpark graphs for one request, in execution order."""
    graphs: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    starts: list[tuple[int, dict[str, Any]]] = []
    timings: dict[int, list[dict[str, Any]]] = defaultdict(list)
    errors: dict[int, list[dict[str, Any]]] = defaultdict(list)
    scratch: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with open_text(log_path) as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                if line.startswith("KPROFG "):
                    record = json.loads(line[7:])
                    graphs[int(record["uid"])][int(record["i"])] = record
                elif line.startswith("KPROFS "):
                    starts.append((line_no, json.loads(line[7:])))
                elif line.startswith("KPROFSB "):
                    record = json.loads(line[8:])
                    scratch[int(record["seq"])].append(record)
                elif line.startswith("KPROF "):
                    record = json.loads(line[6:])
                    if "error" in record:
                        errors[int(record["seq"])].append(record)
                    else:
                        timings[int(record["seq"])].append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise InvalidRun(f"malformed KPROF record at {log_path}:{line_no}: {exc}") from exc

    seqs = sorted(set(timings) | set(errors) | set(scratch))
    require(starts, f"no KPROFS records in {log_path}")
    require(len(starts) == len(seqs),
            f"KPROF start/result mismatch in {log_path}: {len(starts)} starts, {len(seqs)} sequences")

    selected: list[DraftCompute] = []
    accepted_sequences = 0
    for (line_no, start), seq in zip(starts, seqs):
        if not (begin_line <= line_no <= end_line):
            continue
        accepted_sequences += 1
        uid = int(start["uid"])
        require(not errors.get(seq),
                f"KPROF error in accepted request: uid={uid} seq={seq} errors={errors[seq]}")
        require(len(scratch.get(seq, [])) == 1,
                f"uid={uid} seq={seq} has {len(scratch.get(seq, []))} KPROFSB records")
        require(int(scratch[seq][0].get("failed", -1)) == 0,
                f"uid={uid} seq={seq} KPROFSB failed={scratch[seq][0].get('failed')}")

        nodes = graphs.get(uid, {})
        if not nodes or not is_dspark_graph(nodes):
            continue

        expected_nodes = set(range(int(start["n_nodes"])))
        require(set(nodes) == expected_nodes,
                f"signed DSpark uid={uid} graph definition is incomplete: "
                f"missing={sorted(expected_nodes - set(nodes))} "
                f"extra={sorted(set(nodes) - expected_nodes)}")
        unknown_timed = sorted({int(record["node"]) for record in timings.get(seq, [])
                                if int(record["node"]) >= 0 and int(record["node"]) not in nodes})
        require(not unknown_timed,
                f"signed DSpark uid={uid} seq={seq} timed unknown nodes {unknown_timed}")

        indexer_nodes = {i for i, node in nodes.items() if is_indexer_node(node)}
        compressor_nodes = {i for i, node in nodes.items() if is_compressor_node(node)}
        require(not indexer_nodes and not compressor_nodes,
                f"pinned DSpark uid={uid} unexpectedly declares indexer={sorted(indexer_nodes)} "
                f"compressor={sorted(compressor_nodes)}")

        records = timings.get(seq, [])
        require(records, f"signed DSpark uid={uid} seq={seq} has no timing samples")
        fa_nodes = {i for i, node in nodes.items() if node.get("op") == "FLASH_ATTN_EXT"}
        require(len(fa_nodes) == 3, f"signed DSpark uid={uid} has {len(fa_nodes)} FA nodes")
        fa_counts = Counter(int(record["node"]) for record in records
                            if int(record["node"]) in fa_nodes)
        require(set(fa_counts) == fa_nodes and sorted(fa_counts.values()) == [2, 2, 2],
                f"uid={uid} seq={seq} FA timestamp multiplicities are "
                f"{[fa_counts.get(node, 0) for node in sorted(fa_nodes)]}, expected [2, 2, 2]")

        fa_ns = indexer_ns = compressor_ns = graph_ns = 0
        for record in records:
            node_index = int(record["node"])
            elapsed = int(record["ns"])
            require(elapsed >= 0, f"negative KPROF duration for uid={uid} node={node_index}")
            graph_ns += elapsed
            node = nodes.get(node_index)
            if node is None:
                continue
            if node.get("op") == "FLASH_ATTN_EXT":
                fa_ns += elapsed
            if is_indexer_node(node):
                indexer_ns += elapsed
            if is_compressor_node(node):
                compressor_ns += elapsed
        require(indexer_ns == 0 and compressor_ns == 0,
                f"pinned DSpark uid={uid} measured nonzero indexer/compressor cost")
        selected.append(DraftCompute(
            uid=uid,
            seq=seq,
            width=draft_width(nodes),
            fa_ns=fa_ns,
            indexer_ns=indexer_ns,
            compressor_ns=compressor_ns,
            total_graph_ns=graph_ns,
        ))

    require(accepted_sequences > 0,
            f"no KPROF sequences begin in request range {begin_line}:{end_line} of {log_path}")
    require(selected, f"no signed DSpark graphs in request range {begin_line}:{end_line} of {log_path}")
    return selected


def request_trace(log_path: Path, begin_line: int, end_line: int) -> list[str]:
    trace: list[str] = []
    with open_text(log_path) as handle:
        for line_no, line in enumerate(handle, 1):
            if begin_line <= line_no <= end_line and line.startswith("SPECTRACE "):
                trace.append(re.sub(r"\b(slot|seq)=\d+", r"\1=*", line.rstrip()))
    return trace


def trace_draft_widths(trace: list[str], label: str) -> list[int]:
    require(len(trace) >= 2, f"{label} has fewer than two SPECTRACE events")
    require(trace[0].startswith("SPECTRACE tgt ") and
            re.search(r"\bn_dec=0\b", trace[0]) is not None,
            f"{label} first SPECTRACE event is not tgt n_dec=0: {trace[0]}")
    require(trace[1].startswith("SPECTRACE draft "),
            f"{label} second SPECTRACE event is not draft: {trace[1]}")
    widths: list[int] = []
    for event in trace:
        if not event.startswith("SPECTRACE draft "):
            continue
        match = re.search(r"\bn_blk=(\d+)\b", event)
        require(match is not None, f"{label} draft event lacks n_blk: {event}")
        widths.append(int(match.group(1)))
    require(widths, f"{label} has no SPECTRACE draft events")
    return widths


def cache_restore_evidence(log_path: Path, begin_line: int, end_line: int) -> dict[str, Any]:
    loaded: list[int] = []
    prefixes: list[tuple[int, int]] = []
    with open_text(log_path) as handle:
        for line_no, line in enumerate(handle, 1):
            if not (begin_line <= line_no <= end_line):
                continue
            match = re.search(r"loaded prompt \((\d+) tokens,", line)
            if match:
                loaded.append(int(match.group(1)))
            match = re.search(r"prompt cache: (\d+)/(\d+) prefix tokens available", line)
            if match:
                prefixes.append((int(match.group(1)), int(match.group(2))))
    return {"loaded_tokens": loaded, "prefixes": prefixes}


def response_signature(response: dict[str, Any]) -> dict[str, Any]:
    tokens = response.get("tokens")
    require(isinstance(tokens, list) and tokens, "response lacks return_tokens output")
    content = response.get("content")
    require(isinstance(content, str), "response lacks content")
    timings = response.get("timings")
    require(isinstance(timings, dict), "response lacks timings")
    return {
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "tokens_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
        "predicted_n": timings.get("predicted_n"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
    }


def sample_summary(values: list[float]) -> dict[str, float | int]:
    require(values, "cannot summarize an empty sample")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def read_run(root: Path, label: str) -> dict[str, Any]:
    meta = load_json(root / f"request-{label}.meta.json")
    response = load_json(root / f"request-{label}.json")
    log_path = resolve_log(root, str(meta["log"]))
    begin = int(meta["begin_line"])
    end = int(meta["end_line"])
    require(begin > 0 and end >= begin, f"invalid request range for {label}: {begin}:{end}")
    trace = request_trace(log_path, begin, end)
    signature = response_signature(response)
    require(signature["predicted_n"] == 16,
            f"{label} predicted_n={signature['predicted_n']}, expected 16")
    return {
        "label": label,
        "meta": meta,
        "response": response,
        "signature": signature,
        "trace": trace,
        "draft_widths": trace_draft_widths(trace, label),
        "restore": cache_restore_evidence(log_path, begin, end),
        "log_path": log_path,
    }


def assert_exact(reference: dict[str, Any], candidate: dict[str, Any], family: str) -> None:
    require(reference["signature"] == candidate["signature"],
            f"{family} response mismatch: {reference['label']} != {candidate['label']}")
    require(reference["trace"] == candidate["trace"],
            f"{family} SPECTRACE mismatch: {reference['label']} != {candidate['label']}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidRun(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def validate_replay_provenance(root: Path, labels: list[str]) -> dict[str, Any]:
    template_meta = load_json(root / "deep-template.json")
    template_path = root / "deep-template" / "deep-state.lcpc"
    require(template_path.is_file(), "immutable deep template is missing after all replays")
    template_real = template_path.resolve()
    expected_size = int(template_meta.get("size", -1))
    expected_sha = str(template_meta.get("sha256", ""))
    require(template_meta.get("role") == "immutable-template" and
            template_meta.get("cmp_equal") is True,
            "immutable template provenance is incomplete")
    require(Path(str(template_meta.get("destination", ""))) == template_real,
            "immutable template provenance points to a different destination")
    require(template_path.stat().st_size == expected_size,
            "immutable template size changed during replay")
    require(file_sha256(template_path) == expected_sha,
            "immutable template hash changed during replay")

    destinations: set[Path] = set()
    seeds: dict[str, Any] = {}
    for label in labels:
        value = load_json(root / f"cache-seed-{label}.json")
        source = Path(str(value.get("source", "")))
        destination = Path(str(value.get("destination", "")))
        require(value.get("role") == f"{label}-seed" and value.get("cmp_equal") is True,
                f"{label} seed provenance is incomplete")
        require(source == template_real, f"{label} was not seeded from the immutable template")
        require(int(value.get("size", -1)) == expected_size and
                value.get("sha256") == expected_sha,
                f"{label} seed size/hash differs from the immutable template")
        require(destination not in destinations, f"reused destructive cache destination: {destination}")
        destinations.add(destination)
        require(destination.parent == (root / f"cache-{label}").resolve(),
                f"{label} seed is outside its unique cache directory")
        require(template_real.parent not in destination.parents,
                f"{label} destructive seed overlaps the immutable template")
        seeds[label] = value
    return {"template": template_meta, "seeds": seeds}


def classify_envelope(shares: list[float], threshold: float) -> dict[str, Any]:
    require(len(shares) == 8, f"expected 2x2x2=8 sensitivity values, got {len(shares)}")
    low = min(shares)
    high = max(shares)
    width = high - low
    if high < threshold:
        edge_distance = threshold - high
        decision = "reject-windowing" if edge_distance >= width else "inconclusive"
    elif low >= threshold:
        edge_distance = low - threshold
        decision = "extend-review" if edge_distance >= width else "inconclusive"
    else:
        edge_distance = 0.0
        decision = "inconclusive"
    return {
        "kind": "observed-sensitivity-envelope-not-ci",
        "n_combinations": len(shares),
        "min_pct": low,
        "max_pct": high,
        "width_pct": width,
        "closest_edge_distance_pct": edge_distance,
        "margin_sufficient": edge_distance >= width,
        "values_pct": shares,
        "decision": decision,
    }


def steady_call_count(widths: list[int], steady_width: int) -> int:
    count = sum(width == steady_width for width in widths)
    require(count > 0, "exact deep trace contains no steady-width draft calls")
    return count


def analyze(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    require(manifest.get("schema") == 2, "unexpected manifest schema")
    require(manifest.get("decision_threshold_pct") == 5.0,
            "manifest does not predeclare the E1 5% decision threshold")
    require(manifest.get("kprof_stride") == 1,
            "FA timestamp multiplicity invariant requires KPROF stride=1")
    require(manifest.get("deep_replay_cache_mode") ==
            "immutable-template+unique-per-run-clone",
            "manifest does not require immutable per-run deep replay clones")

    replay_tokens = load_json(root / "deep-replay-tokens.json")
    require(isinstance(replay_tokens, list) and len(replay_tokens) >= 90_000,
            "deep replay is not near 100k live tokens")
    cache_header = load_json(root / "deep-cache-header.json")
    cache_tokens = int(cache_header.get("n_tokens", -1))
    cache_tail_gap = len(replay_tokens) - cache_tokens
    allowed_tail_gaps = [int(value) for value in manifest.get("allowed_cache_tail_gap_tokens", [])]
    require(cache_tail_gap in allowed_tail_gaps,
            f"disk cache/replay tail gap is {cache_tail_gap}, expected {allowed_tail_gaps}")
    require(int(cache_header.get("size_drft", 0)) > 0,
            "disk cache entry is target-only; drafter state was not preserved")

    deep_labels = [f"deep-ref-{index}" for index in (1, 2)] + [
        f"deep-kprof-{index}" for index in (1, 2)]
    replay_provenance = validate_replay_provenance(root, deep_labels)

    near_ref = read_run(root, "near-ref")
    deep_refs = [read_run(root, f"deep-ref-{index}") for index in (1, 2)]
    near_profiles = [read_run(root, f"near-kprof-{index}") for index in (1, 2)]
    deep_profiles = [read_run(root, f"deep-kprof-{index}") for index in (1, 2)]
    all_runs = [near_ref] + deep_refs + near_profiles + deep_profiles

    for candidate in near_profiles:
        assert_exact(near_ref, candidate, "near")
    assert_exact(deep_refs[0], deep_refs[1], "deep reference")
    for candidate in deep_profiles:
        assert_exact(deep_refs[0], candidate, "deep")

    steady_width = int(manifest.get("spec_draft_n_max", 5))
    call_count = steady_call_count(deep_refs[0]["draft_widths"], steady_width)
    for run in deep_refs + deep_profiles:
        require(steady_call_count(run["draft_widths"], steady_width) == call_count,
                f"{run['label']} steady draft-call count differs from exact deep reference")
        timing = run["response"]["timings"]
        prompt_n = int(timing.get("prompt_n", 1 << 30))
        require(prompt_n <= 2, f"{run['label']} reprocessed {prompt_n} prompt tokens")
        loaded = run["restore"]["loaded_tokens"]
        prefixes = run["restore"]["prefixes"]
        require(cache_tokens in loaded,
                f"{run['label']} lacks a disk-load record for {cache_tokens} tokens")
        require(any(have == cache_tokens and total == len(replay_tokens)
                    for have, total in prefixes),
                f"{run['label']} lacks an exact cache-prefix record: {prefixes}")

    profile_computes: dict[str, list[DraftCompute]] = {}
    for run in near_profiles + deep_profiles:
        meta = run["meta"]
        computes = parse_kprof(run["log_path"], int(meta["begin_line"]), int(meta["end_line"]))
        graph_widths = [compute.width for compute in computes]
        require(graph_widths == run["draft_widths"],
                f"{run['label']} ordered DSpark widths {graph_widths} != "
                f"SPECTRACE draft widths {run['draft_widths']}")
        profile_computes[run["label"]] = computes

    all_widths = [compute.width for values in profile_computes.values() for compute in values]
    width_counts = Counter(all_widths)
    require(steady_width in width_counts,
            f"no full-width DSpark graph at declared n_max={steady_width}")

    process_means: dict[str, float] = {}
    process_samples: dict[str, list[float]] = {}
    for run in near_profiles + deep_profiles:
        values = [compute.measured_ns / 1e6 for compute in profile_computes[run["label"]]
                  if compute.width == steady_width]
        require(values, f"{run['label']} has no steady-width DSpark samples")
        process_samples[run["label"]] = values
        process_means[run["label"]] = statistics.fmean(values)

    near_means = [process_means[run["label"]] for run in near_profiles]
    deep_means = [process_means[run["label"]] for run in deep_profiles]
    predicted_ms: list[float] = []
    for run in deep_refs:
        value = float(run["response"]["timings"].get("predicted_ms", 0.0))
        require(value > 0, f"invalid unprofiled predicted_ms for {run['label']}")
        predicted_ms.append(value)

    sensitivity_rows: list[dict[str, Any]] = []
    for near_run, near_cost in zip(near_profiles, near_means):
        for deep_run, deep_cost in zip(deep_profiles, deep_means):
            delta = deep_cost - near_cost
            for reference, response_ms in zip(deep_refs, predicted_ms):
                share = 100.0 * delta * call_count / response_ms
                sensitivity_rows.append({
                    "near_process": near_run["label"],
                    "deep_process": deep_run["label"],
                    "unprofiled_response": reference["label"],
                    "near_mean_ms": near_cost,
                    "deep_mean_ms": deep_cost,
                    "raw_delta_ms_per_steady_call": delta,
                    "steady_draft_calls_per_16_token_response": call_count,
                    "unprofiled_predicted_ms": response_ms,
                    "share_pct": share,
                })
    threshold = float(manifest["decision_threshold_pct"])
    envelope = classify_envelope([row["share_pct"] for row in sensitivity_rows], threshold)

    def compute_rows(values: list[DraftCompute]) -> list[dict[str, Any]]:
        return [{
            "uid": value.uid,
            "seq": value.seq,
            "width": value.width,
            "fa_ms": value.fa_ns / 1e6,
            "indexer_ms": value.indexer_ns / 1e6,
            "compressor_ms": value.compressor_ns / 1e6,
            "measured_ms": value.measured_ns / 1e6,
            "graph_ms": value.total_graph_ns / 1e6,
        } for value in values]

    near_all = [value for run in near_profiles for value in process_samples[run["label"]]]
    deep_all = [value for run in deep_profiles for value in process_samples[run["label"]]]
    result = {
        "schema": 2,
        "decision": envelope["decision"],
        "decision_scope": ">=5% authorizes review only; analyzer never authorizes a 500k run",
        "threshold_pct": threshold,
        "deep_tokens": len(replay_tokens),
        "cache_tokens": cache_tokens,
        "cache_tail_gap_tokens": cache_tail_gap,
        "steady_draft_width": steady_width,
        "steady_draft_calls_per_16_token_response": call_count,
        "drafter_graph_signature": "markov_w1+conf_proj+3xFLASH_ATTN_EXT; zero indexer/compressor",
        "fa_timestamp_multiplicity": "exactly 2 records for each of 3 FA nodes at stride=1",
        "near_cost_ms": sample_summary(near_all),
        "deep_cost_ms": sample_summary(deep_all),
        "process_mean_ms": process_means,
        "process_samples_ms": process_samples,
        "unprofiled_deep_predicted_ms": sample_summary(predicted_ms),
        "observed_sensitivity_envelope": envelope,
        "sensitivity_combinations": sensitivity_rows,
        "indexer_ns": {"declared_nodes": 0, "measured": 0},
        "compressor_ns": {"declared_nodes": 0, "measured": 0},
        "response_signatures": {run["label"]: run["signature"] for run in all_runs},
        "draft_width_sequences": {run["label"]: run["draft_widths"] for run in all_runs},
        "width_counts": dict(sorted(width_counts.items())),
        "raw_computes": {
            label: compute_rows(values) for label, values in profile_computes.items()
        },
        "replay_provenance": replay_provenance,
    }
    return result


def render(result: dict[str, Any]) -> str:
    near = result["near_cost_ms"]
    deep = result["deep_cost_ms"]
    response = result["unprofiled_deep_predicted_ms"]
    envelope = result["observed_sensitivity_envelope"]
    lines = [
        f"decision={result['decision']}",
        f"decision_scope={result['decision_scope']}",
        f"deep_tokens={result['deep_tokens']}",
        f"steady_draft_width={result['steady_draft_width']}",
        ("steady_draft_calls_per_16_token_response="
         f"{result['steady_draft_calls_per_16_token_response']}"),
        ("near_process_steady_cost_ms "
         f"n={near['n']} mean={near['mean']:.6f} median={near['median']:.6f} "
         f"range=[{near['min']:.6f},{near['max']:.6f}] stdev={near['stdev']:.6f}"),
        ("deep_process_steady_cost_ms "
         f"n={deep['n']} mean={deep['mean']:.6f} median={deep['median']:.6f} "
         f"range=[{deep['min']:.6f},{deep['max']:.6f}] stdev={deep['stdev']:.6f}"),
        ("unprofiled_deep_16_token_predicted_ms "
         f"n={response['n']} mean={response['mean']:.6f} "
         f"range=[{response['min']:.6f},{response['max']:.6f}]"),
        ("observed_sensitivity_envelope_pct "
         f"combinations={envelope['n_combinations']} min={envelope['min_pct']:.6f} "
         f"max={envelope['max_pct']:.6f} width={envelope['width_pct']:.6f} "
         f"closest_edge_distance={envelope['closest_edge_distance_pct']:.6f} "
         f"margin_sufficient={str(envelope['margin_sufficient']).lower()}"),
        f"decision_threshold_pct={result['threshold_pct']:.6f}",
        "uncertainty=observed process-level sensitivity envelope; not a CI and not independent n=8",
        "drafter_indexer_compressor_presence declared_indexer=0 declared_compressor=0 measured_both_ns=0",
        "kprof_integrity=PASS (ordered graph/trace 1:1; no errors; KPROFSB failed=0; FA counts [2,2,2])",
        "correctness=PASS (predicted_n=16; tokens, content, draft counts, and SPECTRACE exact by depth)",
        ("cache_replay=PASS (immutable template plus unique destructive clones; exact prefix; "
         f"non-empty draft state; tail_gap={result['cache_tail_gap_tokens']}; <=2 tokens reprocessed)"),
    ]
    return "\n".join(lines) + "\n"


def expect_invalid(action, label: str) -> None:
    try:
        action()
    except InvalidRun:
        return
    raise InvalidRun(f"fail-closed self-test did not reject {label}")


def synthetic_nodes(width: int = 5) -> dict[int, dict[str, Any]]:
    nodes: dict[int, dict[str, Any]] = {}
    for index in range(3):
        nodes[index] = {
            "uid": 7,
            "i": index,
            "op": "FLASH_ATTN_EXT",
            "name": f"fa-{index}",
            "src": [{"name": f"q-{index}", "ne": [512, width, 64, 1]}],
        }
    nodes[3] = {
        "uid": 7, "i": 3, "op": "MUL_MAT", "name": "markov",
        "src": [{"name": "markov_w1.weight", "ne": [256, 129280, 1, 1]}],
    }
    nodes[4] = {
        "uid": 7, "i": 4, "op": "MUL_MAT", "name": "confidence",
        "src": [{"name": "conf_proj.weight", "ne": [4352, 1, 1, 1]}],
    }
    nodes[5] = {
        "uid": 7, "i": 5, "op": "ADD", "name": "harmless", "src": [],
    }
    return nodes


def synthetic_kprof_lines(nodes: dict[int, dict[str, Any]]) -> list[str]:
    lines = ["KPROFG " + json.dumps(node, separators=(",", ":")) for node in nodes.values()]
    lines.append("KPROFS " + json.dumps({
        "uid": 7, "n_nodes": len(nodes), "n_cb": 2,
    }, separators=(",", ":")))
    segment = 0
    for batch in (0, 1):
        for node_index in range(3):
            lines.append("KPROF " + json.dumps({
                "seq": 0, "b": batch, "seg": segment, "node": node_index,
                "t0": segment * 100, "ns": 100,
            }, separators=(",", ":")))
            segment += 1
    lines.append('KPROFSB {"seq":0,"batches":2,"created":2,"reused":0,"failed":0,"free":2}')
    return lines


def self_test() -> None:
    nodes = synthetic_nodes()
    require(is_dspark_graph(nodes), "synthetic DSpark signature rejected")
    require(draft_width(nodes) == 5, "synthetic DSpark width rejected")
    trace = [
        "SPECTRACE tgt slot=* n_dec=0 n_blk=1",
        "SPECTRACE draft seq=* n_blk=5",
        "SPECTRACE ver slot=* n_dec=5",
        "SPECTRACE draft seq=* n_blk=2",
    ]
    require(trace_draft_widths(trace, "synthetic") == [5, 2],
            "synthetic SPECTRACE width parse failed")
    require(steady_call_count([5, 5, 2], 5) == 2,
            "terminal partial-width call was counted as steady work")
    expect_invalid(lambda: trace_draft_widths(trace[1:], "missing-boundary"),
                   "missing request boundary")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kprof.log"
        lines = synthetic_kprof_lines(nodes)
        path.write_text("\n".join(lines) + "\n")
        values = parse_kprof(path, 1, len(lines))
        require(len(values) == 1 and values[0].fa_ns == 600,
                "synthetic two-segment FA sum failed")

        def reject_mutation(name: str, mutate) -> None:
            broken_lines = mutate(lines.copy())
            broken = Path(tmp) / f"{name}.log"
            broken.write_text("\n".join(broken_lines) + "\n")
            expect_invalid(lambda: parse_kprof(broken, 1, len(broken_lines)), name)

        reject_mutation("missing-fa-segment", lambda value: [
            line for line in value
            if not (line.startswith("KPROF ") and '"b":1' in line and '"node":2' in line)
        ])
        reject_mutation("explicit-error", lambda value: value[:-1] + [
            'KPROF {"seq":0,"b":1,"error":"resolve failed","sb":1,"n_seg":3}',
            value[-1],
        ])
        reject_mutation("scratch-failure", lambda value: [
            line.replace('"failed":0', '"failed":1') if line.startswith("KPROFSB ") else line
            for line in value
        ])
        reject_mutation("incomplete-graph-definition", lambda value: [
            line for line in value
            if not (line.startswith("KPROFG ") and '"i":5' in line)
        ])
        reject_mutation("timed-unknown-node", lambda value: value[:-1] + [
            'KPROF {"seq":0,"b":0,"seg":99,"node":999,"t0":999,"ns":1}',
            value[-1],
        ])
        indexer_nodes = synthetic_nodes()
        indexer_nodes[5] = {
            "uid": 7, "i": 5, "op": "LIGHTNING_INDEXER", "name": "indexer", "src": []}
        reject_mutation("declared-indexer", lambda _value: synthetic_kprof_lines(indexer_nodes))

    below = classify_envelope([-1.0, 0.0, 0.2, 0.5, 0.6, 0.7, 0.8, 1.0], 5.0)
    require(below["decision"] == "reject-windowing", "below-threshold envelope was not rejected")
    near_edge = classify_envelope([4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.8], 5.0)
    require(near_edge["decision"] == "inconclusive", "near-edge envelope ignored uncertainty")
    above = classify_envelope([7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.0], 5.0)
    require(above["decision"] == "extend-review", "above-threshold envelope was not reviewable")
    crossing = classify_envelope([4.0, 4.5, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5], 5.0)
    require(crossing["decision"] == "inconclusive", "crossing envelope was not inconclusive")
    print("self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", nargs="?", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    try:
        if args.self_test:
            self_test()
            return 0
        require(args.result_dir is not None, "result_dir is required")
        result = analyze(args.result_dir)
        if args.json_out:
            args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        sys.stdout.write(render(result))
        return 0
    except InvalidRun as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

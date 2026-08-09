#!/usr/bin/env python3
"""Analyze DSpark drafter GPU cost at two live-context depths.

The companion ``bench-dspark-draft-depth.sh`` produces fresh-process
near-zero and deep-context replays.  This analyzer deliberately recognizes
the DSpark graph from model tensors (the Markov/confidence head plus exactly
three dense-attention layers); it never treats the target model's compressed
attention as drafter work.

The E1 decision statistic is the depth-dependent part of the steady-state,
full-width DSpark proposal graph:

    (deep FA/indexer/compressor - same-width near cost) * calls/verify
    -----------------------------------------------------------------
                unprofiled deep speculative-step wall time

GGML_METAL_KPROF=1 adds a fixed cost to every split pass.  Comparing the same
three FA nodes at the same proposal width cancels that fixed term.  Results are
rejected unless exact replay, cache restoration, KPROF association, and graph
shape checks all pass.
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
    name = str(node.get("name", ""))
    return "_state_compress" in name


def parse_kprof(log_path: Path, begin_line: int, end_line: int) -> list[DraftCompute]:
    graphs: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    starts: list[tuple[int, dict[str, Any]]] = []
    flushes: dict[int, list[dict[str, Any]]] = defaultdict(list)

    with open_text(log_path) as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                if line.startswith("KPROFG "):
                    record = json.loads(line[7:])
                    graphs[int(record["uid"])][int(record["i"])] = record
                elif line.startswith("KPROFS "):
                    starts.append((line_no, json.loads(line[7:])))
                elif line.startswith("KPROF "):
                    record = json.loads(line[6:])
                    if "error" not in record:
                        flushes[int(record["seq"])].append(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise InvalidRun(f"malformed KPROF record at {log_path}:{line_no}: {exc}") from exc

    seqs = sorted(flushes)
    require(starts, f"no KPROFS records in {log_path}")
    require(len(starts) == len(seqs),
            f"KPROF start/flush mismatch in {log_path}: {len(starts)} starts, {len(seqs)} sequences")

    selected: list[DraftCompute] = []
    for (line_no, start), seq in zip(starts, seqs):
        if not (begin_line <= line_no <= end_line):
            continue
        uid = int(start["uid"])
        nodes = graphs.get(uid, {})
        if not nodes:
            # The graph dump is intentionally bounded. Missing unrelated graph
            # definitions are harmless, but a missing DSpark definition would
            # make the drafter disappear and is caught by the count gates below.
            continue
        if not is_dspark_graph(nodes):
            continue

        records = flushes[seq]
        require(records, f"DSpark uid={uid} seq={seq} has no samples")
        seen_nodes: set[int] = set()
        fa_ns = indexer_ns = compressor_ns = graph_ns = 0
        for record in records:
            node_index = int(record["node"])
            elapsed = int(record["ns"])
            require(elapsed >= 0, f"negative KPROF duration for uid={uid} node={node_index}")
            graph_ns += elapsed
            node = nodes.get(node_index)
            if node is None:
                continue
            seen_nodes.add(node_index)
            if node.get("op") == "FLASH_ATTN_EXT":
                fa_ns += elapsed
            if is_indexer_node(node):
                indexer_ns += elapsed
            if is_compressor_node(node):
                compressor_ns += elapsed

        fa_nodes = {i for i, node in nodes.items() if node.get("op") == "FLASH_ATTN_EXT"}
        require(fa_nodes <= seen_nodes,
                f"incomplete DSpark FA samples for uid={uid}: missing {sorted(fa_nodes - seen_nodes)}")
        selected.append(DraftCompute(
            uid=uid,
            seq=seq,
            width=draft_width(nodes),
            fa_ns=fa_ns,
            indexer_ns=indexer_ns,
            compressor_ns=compressor_ns,
            total_graph_ns=graph_ns,
        ))

    require(selected, f"no signed DSpark graphs in request range {begin_line}:{end_line} of {log_path}")
    return selected


def request_trace(log_path: Path, begin_line: int, end_line: int) -> list[str]:
    trace: list[str] = []
    with open_text(log_path) as handle:
        for line_no, line in enumerate(handle, 1):
            if begin_line <= line_no <= end_line and line.startswith("SPECTRACE "):
                value = re.sub(r"\b(slot|seq)=\d+", r"\1=*", line.rstrip())
                trace.append(value)
    return trace


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


def mean(values: list[float], label: str) -> float:
    require(values, f"no values for {label}")
    return statistics.fmean(values)


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
    return {
        "label": label,
        "meta": meta,
        "response": response,
        "signature": response_signature(response),
        "trace": trace,
        "verify_steps": sum(line.startswith("SPECTRACE ver ") for line in trace),
        "restore": cache_restore_evidence(log_path, begin, end),
        "log_path": log_path,
    }


def assert_exact(reference: dict[str, Any], candidate: dict[str, Any], family: str) -> None:
    require(reference["signature"] == candidate["signature"],
            f"{family} response mismatch: {reference['label']} != {candidate['label']}")
    require(reference["trace"], f"{reference['label']} has no SPECTRACE evidence")
    require(candidate["trace"], f"{candidate['label']} has no SPECTRACE evidence")
    require(reference["trace"] == candidate["trace"],
            f"{family} SPECTRACE mismatch: {reference['label']} != {candidate['label']}")


def analyze(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    require(manifest.get("decision_threshold_pct") == 5.0,
            "manifest does not predeclare the E1 5% decision threshold")
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

    near_ref = read_run(root, "near-ref")
    deep_refs = [read_run(root, f"deep-ref-{index}") for index in (1, 2)]
    near_profiles = [read_run(root, f"near-kprof-{index}") for index in (1, 2)]
    deep_profiles = [read_run(root, f"deep-kprof-{index}") for index in (1, 2)]

    for candidate in near_profiles:
        assert_exact(near_ref, candidate, "near")
    assert_exact(deep_refs[0], deep_refs[1], "deep reference")
    for candidate in deep_profiles:
        assert_exact(deep_refs[0], candidate, "deep")

    for run in deep_refs + deep_profiles:
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
        profile_computes[run["label"]] = parse_kprof(
            run["log_path"], int(meta["begin_line"]), int(meta["end_line"]))

    # Steady-state DSpark is one batched proposal graph, normally width 5.
    # Terminal partial-budget calls are reported but excluded from the paired
    # depth delta so proposal width cannot masquerade as a context effect.
    all_widths = [compute.width for values in profile_computes.values() for compute in values]
    width_counts = Counter(all_widths)
    steady_width = max(width_counts, key=lambda value: (width_counts[value], value))
    require(steady_width == int(manifest.get("spec_draft_n_max", 5)),
            f"modal DSpark width is {steady_width}, expected spec n_max")

    near_steady = [
        compute for run in near_profiles for compute in profile_computes[run["label"]]
        if compute.width == steady_width
    ]
    deep_steady = [
        compute for run in deep_profiles for compute in profile_computes[run["label"]]
        if compute.width == steady_width
    ]
    require(len(near_steady) >= 4 and len(deep_steady) >= 4,
            f"too few steady DSpark samples: near={len(near_steady)} deep={len(deep_steady)}")

    # Architecture reality check. This DSpark artifact contains the first
    # three dense-attention layers and no lightning-indexer/compressor nodes.
    # Reporting zero is important: target-model CSA must not be charged here.
    near_indexer = [compute.indexer_ns for compute in near_steady]
    deep_indexer = [compute.indexer_ns for compute in deep_steady]
    near_compressor = [compute.compressor_ns for compute in near_steady]
    deep_compressor = [compute.compressor_ns for compute in deep_steady]

    near_cost_ms = [compute.measured_ns / 1e6 for compute in near_steady]
    deep_cost_ms = [compute.measured_ns / 1e6 for compute in deep_steady]
    delta_ms = mean(deep_cost_ms, "deep cost") - mean(near_cost_ms, "near cost")

    verify_counts = [run["verify_steps"] for run in deep_profiles]
    graph_counts = [len(profile_computes[run["label"]]) for run in deep_profiles]
    require(all(value > 0 for value in verify_counts), f"missing deep verification traces: {verify_counts}")
    calls_per_verify_samples = [graphs / verifies for graphs, verifies in zip(graph_counts, verify_counts)]
    calls_per_verify = mean(calls_per_verify_samples, "calls per verify")
    require(0.8 <= calls_per_verify <= 1.2,
            f"DSpark graph/verify association is not one-to-one: {calls_per_verify_samples}")

    step_ms_samples: list[float] = []
    for run in deep_refs:
        predicted_ms = float(run["response"]["timings"].get("predicted_ms", 0.0))
        require(predicted_ms > 0 and run["verify_steps"] > 0,
                f"invalid unprofiled timing for {run['label']}")
        step_ms_samples.append(predicted_ms / run["verify_steps"])
    step_ms = mean(step_ms_samples, "unprofiled speculative step")

    share_pct = 100.0 * delta_ms * calls_per_verify / step_ms
    threshold = float(manifest["decision_threshold_pct"])
    decision = "reject-windowing" if share_pct < threshold else "extend-to-500k-after-review"

    def compute_rows(values: list[DraftCompute]) -> list[dict[str, Any]]:
        return [
            {
                "uid": value.uid,
                "seq": value.seq,
                "width": value.width,
                "fa_ms": value.fa_ns / 1e6,
                "indexer_ms": value.indexer_ns / 1e6,
                "compressor_ms": value.compressor_ns / 1e6,
                "measured_ms": value.measured_ns / 1e6,
                "graph_ms": value.total_graph_ns / 1e6,
            }
            for value in values
        ]

    result = {
        "schema": 1,
        "decision": decision,
        "threshold_pct": threshold,
        "share_pct": share_pct,
        "deep_tokens": len(replay_tokens),
        "cache_tokens": cache_tokens,
        "cache_tail_gap_tokens": cache_tail_gap,
        "steady_draft_width": steady_width,
        "drafter_graph_signature": "markov_w1+conf_proj+3xFLASH_ATTN_EXT",
        "near_cost_ms": sample_summary(near_cost_ms),
        "deep_cost_ms": sample_summary(deep_cost_ms),
        "context_delta_ms_per_draft_call": delta_ms,
        "calls_per_verify": sample_summary(calls_per_verify_samples),
        "unprofiled_step_ms": sample_summary(step_ms_samples),
        "indexer_ns": {"near": sample_summary(near_indexer), "deep": sample_summary(deep_indexer)},
        "compressor_ns": {
            "near": sample_summary(near_compressor),
            "deep": sample_summary(deep_compressor),
        },
        "response_signatures": {
            run["label"]: run["signature"]
            for run in [near_ref] + deep_refs + near_profiles + deep_profiles
        },
        "verify_steps": {run["label"]: run["verify_steps"] for run in deep_refs + deep_profiles},
        "width_counts": dict(sorted(width_counts.items())),
        "raw_computes": {
            label: compute_rows(values) for label, values in profile_computes.items()
        },
    }
    return result


def render(result: dict[str, Any]) -> str:
    near = result["near_cost_ms"]
    deep = result["deep_cost_ms"]
    step = result["unprofiled_step_ms"]
    calls = result["calls_per_verify"]
    lines = [
        f"decision={result['decision']}",
        f"deep_tokens={result['deep_tokens']}",
        f"steady_draft_width={result['steady_draft_width']}",
        ("near_drafter_fa_indexer_compressor_ms "
         f"n={near['n']} mean={near['mean']:.6f} median={near['median']:.6f} "
         f"range=[{near['min']:.6f},{near['max']:.6f}] stdev={near['stdev']:.6f}"),
        ("deep_drafter_fa_indexer_compressor_ms "
         f"n={deep['n']} mean={deep['mean']:.6f} median={deep['median']:.6f} "
         f"range=[{deep['min']:.6f},{deep['max']:.6f}] stdev={deep['stdev']:.6f}"),
        f"context_delta_ms_per_draft_call={result['context_delta_ms_per_draft_call']:.6f}",
        ("draft_calls_per_verify "
         f"samples={calls['n']} mean={calls['mean']:.6f} "
         f"range=[{calls['min']:.6f},{calls['max']:.6f}]"),
        ("unprofiled_spec_step_ms "
         f"samples={step['n']} mean={step['mean']:.6f} "
         f"range=[{step['min']:.6f},{step['max']:.6f}]"),
        f"context_proportional_share_pct={result['share_pct']:.6f}",
        f"decision_threshold_pct={result['threshold_pct']:.6f}",
        ("drafter_indexer_compressor_presence "
         f"indexer_deep_mean_ns={result['indexer_ns']['deep']['mean']:.3f} "
         f"compressor_deep_mean_ns={result['compressor_ns']['deep']['mean']:.3f}"),
        "correctness=PASS (tokens, content, draft counts, and SPECTRACE exact within each depth)",
        ("cache_replay=PASS (exact numeric prefix, non-empty draft state, "
         f"tail_gap={result['cache_tail_gap_tokens']}, <=2 prompt tokens reprocessed)"),
    ]
    return "\n".join(lines) + "\n"


def self_test() -> None:
    nodes: dict[int, dict[str, Any]] = {}
    for index in range(3):
        nodes[index] = {
            "uid": 7,
            "i": index,
            "op": "FLASH_ATTN_EXT",
            "name": f"fa-{index}",
            "src": [{"name": f"q-{index}", "ne": [512, 5, 64, 1]}],
        }
    nodes[3] = {
        "uid": 7,
        "i": 3,
        "op": "MUL_MAT",
        "name": "markov",
        "src": [{"name": "markov_w1.weight", "ne": [256, 129280, 1, 1]}],
    }
    nodes[4] = {
        "uid": 7,
        "i": 4,
        "op": "MUL_MAT",
        "name": "confidence",
        "src": [{"name": "conf_proj.weight", "ne": [4352, 1, 1, 1]}],
    }
    require(is_dspark_graph(nodes), "synthetic DSpark signature rejected")
    require(draft_width(nodes) == 5, "synthetic DSpark width rejected")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kprof.log"
        lines: list[str] = []
        for node in nodes.values():
            lines.append("KPROFG " + json.dumps(node, separators=(",", ":")))
        lines.append('KPROFS {"uid":7,"n_nodes":5,"n_cb":1}')
        for node_index in range(5):
            lines.append("KPROF " + json.dumps({
                "seq": 0, "b": 0, "seg": node_index, "node": node_index,
                "t0": node_index * 100, "ns": 100,
            }, separators=(",", ":")))
        path.write_text("\n".join(lines) + "\n")
        values = parse_kprof(path, 1, len(lines))
        require(len(values) == 1 and values[0].fa_ns == 300, "synthetic KPROF sum failed")

        broken = Path(tmp) / "broken.log"
        broken.write_text(path.read_text().replace('KPROFS {"uid":7,"n_nodes":5,"n_cb":1}\n', ""))
        try:
            parse_kprof(broken, 1, len(lines))
        except InvalidRun:
            pass
        else:
            raise InvalidRun("fail-closed KPROF association test did not fail")

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

#!/usr/bin/env python3
"""Summarize fixed-width DSV4 speculative-verify GGML_METAL_KPROF logs.

The companion dsv4-verify-q-census.sh starts one server per query width with
DSpark p_min=0, so the draft always contains Q-1 tokens.  This analyzer locates
the target graph by its FLASH_ATTN_EXT query shape, maps later graph instances
to the first dumped graph with the same node count, subtracts the calibrated
stride-1 pass overhead, and reports the costs needed by the verify-width census.

Usage:
  scripts/analyze-dsv4-verify-q-kprof.py [--overhead-ns 3777] LOG.gz [...]
"""

import argparse
import gzip
import json
import re
import statistics
from collections import defaultdict


def open_log(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


def load(path):
    graphs = defaultdict(dict)
    starts = []
    flushes = defaultdict(lambda: defaultdict(list))
    with open_log(path) as fh:
        for line in fh:
            if line.startswith("KPROFG "):
                rec = json.loads(line[7:])
                graphs[rec["uid"]][rec["i"]] = rec
            elif line.startswith("KPROFS "):
                starts.append(json.loads(line[7:]))
            elif line.startswith("KPROF "):
                rec = json.loads(line[6:])
                if "error" not in rec:
                    flushes[rec["seq"]][rec["b"]].append(rec)
    seqs = sorted(flushes)
    assoc = {seq: starts[i] for i, seq in enumerate(seqs) if i < len(starts)}
    return graphs, assoc, flushes


def query_rows_from_path(path):
    match = re.search(r"(?:^|[-_/])q(\d+)(?:[-_.]|$)", path)
    if not match:
        raise ValueError("cannot infer query width from path %r (expected ...qN...)" % path)
    return int(match.group(1))


def graph_has_query_rows(nodes, qrows):
    for node in nodes.values():
        if node.get("op") != "FLASH_ATTN_EXT":
            continue
        src = node.get("src", [])
        if not src:
            continue
        ne = src[0].get("ne", [])
        if len(ne) >= 4 and ne[0] == 512 and ne[1] == qrows and ne[2] == 64:
            return True
    return False


def tag_name(node):
    name = node.get("name", "")
    src = node.get("src", [])
    if node.get("op") in ("MUL_MAT", "MUL_MAT_ID") and src:
        name = src[0].get("name", name)
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) == 3:
            name = parts[2]
    return name.lower()


def category(node):
    op = node.get("op", "")
    name = tag_name(node)
    if op == "FLASH_ATTN_EXT":
        return "flash_attention"
    if op == "MUL_MAT_ID" and "exps" in name:
        return "expert_gemv"
    if "shexp" in name:
        return "shared_expert"
    if ("indexer" in name or "compressor" in name or name.startswith(("csa_", "lid_", "hca_"))
            or op in {"LIGHTNING_INDEXER", "DSV4_INDEXED_LIGHTNING_INDEXER", "DSV4_COMPRESS",
                      "DSV4_TOP_K_MASK", "DSV4_SPARSE_PACK", "DSV4_INDEXED_SPARSE_PACK"}):
        return "indexer_compressor"
    if op == "MUL_MAT_ID":
        return "other_mul_mat_id"
    return "other"


def percentile_summary(values):
    vals = sorted(values)
    return {
        "samples": len(vals),
        "mean_ms": statistics.fmean(vals),
        "median_ms": statistics.median(vals),
        "min_ms": vals[0],
        "max_ms": vals[-1],
        "stdev_ms": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "raw_ms": vals,
    }


def analyze(path, overhead_ns):
    qrows = query_rows_from_path(path)
    graphs, assoc, flushes = load(path)

    candidates = []
    for uid, nodes in graphs.items():
        if graph_has_query_rows(nodes, qrows):
            candidates.append((len(nodes), uid, nodes))
    if not candidates:
        raise ValueError("no target graph with q=%d in %s" % (qrows, path))

    # The DSpark graph uses the same attention signature, but the full target
    # graph has more nodes (about 7.6k versus 7.1k on the pinned checkpoint).
    n_nodes, rep_uid, nodes = max(candidates)
    selected = []
    for seq, start in assoc.items():
        if start.get("n_nodes") == n_nodes:
            selected.append((seq, start.get("uid")))
    if not selected:
        raise ValueError("no computes for target n_nodes=%d in %s" % (n_nodes, path))

    samples = defaultdict(list)
    for seq, _uid in selected:
        sums = defaultdict(float)
        raw_whole = 0.0
        for recs in flushes[seq].values():
            for rec in recs:
                raw_whole += float(rec["ns"])
                ns = max(0.0, float(rec["ns"]) - overhead_ns)
                sums["whole_verify"] += ns
                ni = rec.get("node", -1)
                if ni in nodes:
                    sums[category(nodes[ni])] += ns
        for key, ns in sums.items():
            samples[key].append(ns / 1e6)
        samples["whole_verify_raw"].append(raw_whole / 1e6)

    result = {
        "path": path,
        "query_rows": qrows,
        "representative_uid": rep_uid,
        "n_nodes": n_nodes,
        "overhead_ns": overhead_ns,
        "metrics": {key: percentile_summary(vals) for key, vals in samples.items()},
    }
    for required in ("whole_verify", "flash_attention", "indexer_compressor", "expert_gemv"):
        if required not in result["metrics"]:
            raise ValueError("target graph lacks %s attribution in %s" % (required, path))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overhead-ns", type=float, default=3777.0)
    parser.add_argument("--json")
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()

    results = [analyze(path, args.overhead_ns) for path in args.logs]
    results.sort(key=lambda r: r["query_rows"])

    print("overhead subtracted: %.0f ns per stride-1 sampled pass" % args.overhead_ns)
    print("%3s %7s %3s %12s %12s %12s %12s %15s" % (
        "Q", "nodes", "n", "verify ms", "FA ms", "index/comp", "expert ms", "verify min..max"))
    for result in results:
        m = result["metrics"]
        whole = m["whole_verify"]
        print("%3d %7d %3d %12.3f %12.3f %12.3f %12.3f %7.3f..%-7.3f" % (
            result["query_rows"], result["n_nodes"], whole["samples"], whole["median_ms"],
            m["flash_attention"]["median_ms"], m["indexer_compressor"]["median_ms"],
            m["expert_gemv"]["median_ms"], whole["min_ms"], whole["max_ms"]))

    print("\nraw corrected samples (ms):")
    for result in results:
        print("Q=%d sampled=%s corrected=%s FA=%s indexer/compressor=%s expert=%s" % (
            result["query_rows"],
            ",".join("%.3f" % x for x in result["metrics"]["whole_verify_raw"]["raw_ms"]),
            ",".join("%.3f" % x for x in result["metrics"]["whole_verify"]["raw_ms"]),
            ",".join("%.3f" % x for x in result["metrics"]["flash_attention"]["raw_ms"]),
            ",".join("%.3f" % x for x in result["metrics"]["indexer_compressor"]["raw_ms"]),
            ",".join("%.3f" % x for x in result["metrics"]["expert_gemv"]["raw_ms"])))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print("wrote %s" % args.json)


if __name__ == "__main__":
    main()

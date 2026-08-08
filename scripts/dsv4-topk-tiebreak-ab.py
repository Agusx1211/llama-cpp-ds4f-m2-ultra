#!/usr/bin/env python3
"""A/B harness for the DSV4 Lightning Indexer top-k pivot tie-break policy.

The indexer score row is ReLU'd, so the top-k pivot is routinely exactly 0.0f
and the tie group is every compressed row that scored <= 0. Which members of
that group fill the 512 slots is a free parameter
(LLAMA_DSV4_TOPK_TIE=asc|desc|hash, see ggml-metal.metal). This script measures
the consequence of that choice against a *running* llama-server:

  perf   - N identical greedy requests at a fixed depth; reports per-run
           throughput, draft acceptance and transcript hash plus
           mean/median/stdev. Also the tool used to establish the pre-fix
           binary's *mean*, which the original determinism work never did.
  needle - deep-context retrieval with a verifiable answer, with the needle
           placed at a range of relative depths. Oldest-first tie-breaking
           should hurt late needles specifically, if the recency hypothesis is
           right.

The prompt generator is shared with scripts/dsv4-depth-determinism.py so the
perf numbers are directly comparable with that gate's.

Usage:
  scripts/dsv4-topk-tiebreak-ab.py perf   --depth 37k --runs 10 [--slot 0]
  scripts/dsv4-topk-tiebreak-ab.py needle --depth 37k [--positions 10]
Both write one JSON object per line to --out (default stdout) and a human
summary to stderr.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# scripts/dsv4-depth-determinism.py is not importable by name (hyphens), so load
# it by path to guarantee byte-identical prompts with the determinism gate.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "dsv4_depth_determinism",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dsv4-depth-determinism.py"))
_dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dd)

WORDS = _dd.WORDS
DEPTHS = _dd.DEPTHS
build_prompt = _dd.build_prompt


# --------------------------------------------------------------------------
# needle-in-a-haystack construction
# --------------------------------------------------------------------------

CODES = [
    "ZEBRA742", "OTTER911", "LLAMA555", "FALCON318", "MARLIN276",
    "BADGER604", "CONDOR185", "WALRUS437", "IGUANA829", "JAGUAR063",
    "PUFFIN591", "ORIOLE248", "TAPIR730", "NARWHAL152", "QUAIL486",
    "MANTIS907",
]

GATES = [
    "northern", "southern", "eastern", "western", "harbor", "valley",
    "forest", "desert", "island", "bridge", "river", "mountain",
    "ocean", "plain", "city", "village",
]


def _sections(total_chars, seed=42):
    """The same section text scripts/dsv4-depth-determinism.py generates."""
    rng = random.Random(seed)
    parts, total, idx = [], 0, 0
    while total < total_chars:
        idx += 1
        n = rng.randint(4, 8)
        sentences = []
        for _ in range(n):
            m = rng.randint(8, 16)
            ws = [rng.choice(WORDS) for _ in range(m)]
            ws[0] = ws[0].capitalize()
            sentences.append(" ".join(ws) + ".")
        p = f"Section {idx}. " + " ".join(sentences)
        total += len(p) + 2
        parts.append(p)
    return parts


def build_needle_prompt(total_chars, needles, question):
    """needles: list of (frac, gate, code). frac is the relative depth 0..1."""
    parts = _sections(total_chars)
    n = len(parts)
    for frac, gate, code in needles:
        i = min(n - 1, max(0, int(round(frac*(n - 1)))))
        fact = f" The access code for the {gate} gate is {code}."
        # append to the section body so it reads as part of the filler
        parts[i] = parts[i] + fact
    return "\n\n".join(parts) + "\n\n" + question


def needle_probes(depth, n_positions):
    """One single-needle probe per relative depth, plus two multi-needle ones."""
    chars = DEPTHS[depth]
    probes = []
    for j in range(n_positions):
        frac = j/(n_positions - 1) if n_positions > 1 else 0.5
        gate = GATES[j % len(GATES)]
        code = CODES[j % len(CODES)]
        q = (f"Question: what is the access code for the {gate} gate? "
             f"Answer with only the code.")
        probes.append({
            "name": f"single@{frac:.2f}",
            "frac": round(frac, 4),
            "kind": "single",
            "expect": [code],
            "n_predict": 24,
            "prompt": build_needle_prompt(chars, [(frac, gate, code)], q),
        })

    for k in (4, 8):
        fracs = [(j + 0.5)/k for j in range(k)]
        ns = [(f, GATES[j], CODES[j]) for j, f in enumerate(fracs)]
        q = ("Question: list the access codes for every gate mentioned above, "
             "one per line, with no other text.")
        probes.append({
            "name": f"multi{k}",
            "frac": -1.0,
            "kind": "multi",
            "expect": [c for _, _, c in ns],
            "n_predict": 16*k,
            "prompt": build_needle_prompt(chars, ns, q),
        })
    return probes


# --------------------------------------------------------------------------

def post(url, api_key, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def one_request(args, prompt, n_predict, ignore_eos):
    # perf mode pins the decoded length with ignore_eos so throughput and draft
    # acceptance are measured over the same number of steps in every arm;
    # needle mode must be allowed to stop.
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": False,
        "ignore_eos": ignore_eos,
    }
    if args.slot is not None:
        payload["id_slot"] = args.slot
    t0 = time.time()
    resp = post(args.url.rstrip("/") + "/completion", args.api_key, payload, args.timeout)
    wall = time.time() - t0
    if "content" not in resp:
        raise RuntimeError("server error: " + json.dumps(resp)[:400])
    t = resp.get("timings", {}) or {}
    return {
        "content": resp["content"],
        "sha256": hashlib.sha256(resp["content"].encode()).hexdigest()[:32],
        "wall_s": round(wall, 3),
        "prompt_n": t.get("prompt_n"),
        "predicted_n": t.get("predicted_n"),
        "tps": t.get("predicted_per_second"),
        "prompt_tps": t.get("prompt_per_second"),
        "draft_n": t.get("draft_n"),
        "draft_acc": t.get("draft_n_accepted"),
    }


def summarize(name, values):
    values = [v for v in values if v is not None]
    if not values:
        return f"{name}: n/a"
    mean = statistics.mean(values)
    med = statistics.median(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return (f"{name}: n={len(values)} mean={mean:.4f} median={med:.4f} "
            f"stdev={sd:.4f} min={min(values):.4f} max={max(values):.4f}")


def cmd_perf(args, emit):
    prompt = build_prompt(DEPTHS[args.depth])
    print(f"# perf depth={args.depth} chars={len(prompt)} runs={args.runs} "
          f"n_predict={args.n_predict} label={args.label}", file=sys.stderr, flush=True)
    tps, acc, shas = [], [], []
    for r in range(args.runs):
        try:
            res = one_request(args, prompt, args.n_predict, True)
        except Exception as e:  # keep the window moving
            print(f"run {r}: ERROR {e}", file=sys.stderr, flush=True)
            emit({"mode": "perf", "label": args.label, "depth": args.depth,
                  "run": r, "error": str(e)})
            continue
        res.pop("content")
        rate = (res["draft_acc"]/res["draft_n"]) if res.get("draft_n") else None
        rec = {"mode": "perf", "label": args.label, "depth": args.depth, "run": r,
               "accept_rate": rate, **res}
        emit(rec)
        tps.append(res["tps"])
        if rate is not None:
            acc.append(rate)
        shas.append(res["sha256"])
        print(f"run {r}: tps={res['tps']:.3f} draft={res['draft_acc']}/{res['draft_n']} "
              f"prompt_n={res['prompt_n']} sha={res['sha256']}", file=sys.stderr, flush=True)

    print(summarize("tps", tps), file=sys.stderr)
    print(summarize("accept_rate", acc), file=sys.stderr)
    print(f"distinct transcripts: {len(set(shas))}/{len(shas)}", file=sys.stderr, flush=True)
    emit({"mode": "perf-summary", "label": args.label, "depth": args.depth,
          "n": len(tps), "tps_mean": statistics.mean(tps) if tps else None,
          "tps_median": statistics.median(tps) if tps else None,
          "tps_stdev": statistics.stdev(tps) if len(tps) > 1 else 0.0,
          "acc_mean": statistics.mean(acc) if acc else None,
          "acc_stdev": statistics.stdev(acc) if len(acc) > 1 else 0.0,
          "distinct_sha": len(set(shas)), "runs": len(shas)})


def cmd_needle(args, emit):
    probes = needle_probes(args.depth, args.positions)
    print(f"# needle depth={args.depth} probes={len(probes)} label={args.label}",
          file=sys.stderr, flush=True)
    total, hits = 0, 0
    for p in probes:
        for rep in range(args.reps):
            try:
                res = one_request(args, p["prompt"], p["n_predict"], False)
            except Exception as e:
                print(f"{p['name']} rep{rep}: ERROR {e}", file=sys.stderr, flush=True)
                emit({"mode": "needle", "label": args.label, "depth": args.depth,
                      "probe": p["name"], "rep": rep, "error": str(e)})
                continue
            found = [c for c in p["expect"] if c in res["content"]]
            score = len(found)/len(p["expect"])
            total += len(p["expect"])
            hits += len(found)
            rec = {"mode": "needle", "label": args.label, "depth": args.depth,
                   "probe": p["name"], "rep": rep, "frac": p["frac"],
                   "kind": p["kind"], "expect": p["expect"],
                   "found": found, "score": score,
                   "answer": res["content"].strip()[:200],
                   "prompt_n": res["prompt_n"], "tps": res["tps"],
                   "draft_n": res["draft_n"], "draft_acc": res["draft_acc"],
                   "sha256": res["sha256"]}
            emit(rec)
            print(f"{p['name']} rep{rep}: score={score:.2f} "
                  f"({len(found)}/{len(p['expect'])}) answer={rec['answer'][:60]!r}",
                  file=sys.stderr, flush=True)
    print(f"needle total: {hits}/{total} = {hits/total if total else 0:.3f}",
          file=sys.stderr, flush=True)
    emit({"mode": "needle-summary", "label": args.label, "depth": args.depth,
          "hits": hits, "total": total,
          "rate": hits/total if total else None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["perf", "needle"])
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--depth", default="37k", choices=sorted(DEPTHS))
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--positions", type=int, default=10,
                    help="needle mode: number of relative depths to probe")
    ap.add_argument("--reps", type=int, default=1,
                    help="needle mode: repeats per probe (>1 only useful on a "
                         "nondeterministic binary)")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--slot", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", default="unlabeled",
                    help="tie policy / binary this run belongs to")
    ap.add_argument("--out", default=None, help="append JSONL here")
    args = ap.parse_args()

    fh = open(args.out, "a") if args.out else sys.stdout

    def emit(rec):
        rec["ts"] = time.time()
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    if args.mode == "perf":
        cmd_perf(args, emit)
    else:
        cmd_needle(args, emit)

    if args.out:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

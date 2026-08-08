#!/usr/bin/env python3
"""Deep-context greedy-decode determinism gate for DeepSeek V4 Flash.

Temperature-0 decode against a fixed prompt and a fixed server state must be
reproducible. It was not: once the DSV4 compressed cache passed 1024 rows
(~4k prompt tokens) the Lightning Indexer's top-k selection was resolved by a
GPU thread race, so every repeat of the same request attended a different set
of compressed keys and produced a different transcript. See
notes/2026-08-08-dsv4-depth-determinism.md.

This script drives a *running* llama-server: it sends the same request N times
and fails if the transcripts, the draft statistics, or (when the server runs
with LLAMA_SPEC_TRACE=1 and --server-log is given) the per-step trace differ.

The prompt is generated in-process from a fixed seed, so no fixture file is
needed and every invocation on every machine uses the same tokens.

Usage:
  scripts/dsv4-depth-determinism.py --url http://127.0.0.1:8080 [--api-key K]
      [--depth 12k] [--runs 3] [--n-predict 128] [--slot N]
      [--server-log /tmp/llama-server.log]

Exit status 0 = every run identical.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request

WORDS = (
    "the of and to in is that it for on with as at by from this be are was "
    "or an will can has have had not but all one two three system model data "
    "graph memory cache token decode compute kernel batch stream layer expert "
    "state page plan device host metal shader queue buffer tensor weight "
    "matrix vector index sparse dense verify draft accept reject rollback "
    "schedule allocate reserve encode submit wait sample output input context "
    "window depth width speed cost time step cycle phase result measure test "
    "check build run load store copy move read write open close start stop "
    "high low fast slow large small deep wide long short first last next "
    "prior early late many few more less best worst mean total count rate "
    "value number series table figure section chapter report note record "
    "engine method process design change effect cause reason detail case "
    "point line block chunk piece part whole group set list tree map key "
    "river mountain forest valley ocean island desert plain harbor bridge "
    "city village road path light shadow morning evening winter summer "
    "history science theory practice question answer problem solution idea"
).split()

# name -> approximate prompt characters (~5.3 chars per DSV4 token)
DEPTHS = {
    "4k":  16_000,    # ~3.0k tokens - below the radix top-k threshold
    "12k": 48_000,    # ~9.1k tokens - the cheapest depth that reproduced
    "24k": 96_000,    # ~18.1k tokens
    "37k": 200_000,   # ~37.7k tokens - the original report's prompt
}


def build_prompt(total_chars):
    rng = random.Random(42)
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
    return "\n\n".join(parts) + \
        "\n\nSummarize the recurring themes of the sections above in detail."


def post(url, api_key, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def spectrace_segments(path):
    """Split a server log into one SPECTRACE segment per request."""
    segs, cur = [], None
    opener = open
    if path.endswith(".gz"):
        import gzip
        opener = lambda p: gzip.open(p, "rt", errors="replace")  # noqa: E731
    for line in opener(path):
        if not line.startswith("SPECTRACE"):
            continue
        line = line.rstrip("\n")
        if line.startswith("SPECTRACE tgt") and " n_dec=0 " in line:
            cur = []
            segs.append(cur)
        if cur is not None:
            cur.append(line)
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--depth", default="12k", choices=sorted(DEPTHS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--slot", type=int, default=None,
                    help="pin every run to this slot (default: server picks)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--server-log", default=None,
                    help="server stderr log; when the server ran with "
                         "LLAMA_SPEC_TRACE=1 the per-step traces are compared too")
    args = ap.parse_args()

    prompt = build_prompt(DEPTHS[args.depth])
    payload = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": 0,
        "cache_prompt": False,
        "ignore_eos": True,
    }
    if args.slot is not None:
        payload["id_slot"] = args.slot

    print(f"depth={args.depth} prompt_chars={len(prompt)} runs={args.runs} "
          f"n_predict={args.n_predict}", flush=True)

    results = []
    for r in range(args.runs):
        resp = post(args.url.rstrip("/") + "/completion", args.api_key, payload, args.timeout)
        if "content" not in resp:
            print(f"run {r}: server error: {json.dumps(resp)[:400]}")
            return 2
        t = resp.get("timings", {}) or {}
        sig = {
            "sha256": hashlib.sha256(resp["content"].encode()).hexdigest(),
            "len": len(resp["content"]),
            "prompt_n": t.get("prompt_n"),
            "draft_n": t.get("draft_n"),
            "draft_n_accepted": t.get("draft_n_accepted"),
        }
        results.append(sig)
        print(f"run {r}: prompt_n={sig['prompt_n']} draft={sig['draft_n_accepted']}/{sig['draft_n']} "
              f"len={sig['len']} sha256={sig['sha256'][:32]} "
              f"tps={t.get('predicted_per_second', float('nan')):.3f}", flush=True)

    ok = all(x == results[0] for x in results)
    if not ok:
        print("FAIL: repeated identical greedy requests produced different results")
        for i, x in enumerate(results):
            if x != results[0]:
                print(f"  run {i}: {x}")
                print(f"  run 0: {results[0]}")
                break

    if args.server_log:
        segs = spectrace_segments(args.server_log)
        segs = segs[-args.runs:] if len(segs) >= args.runs else segs
        if len(segs) < args.runs:
            print(f"note: server log has {len(segs)} SPECTRACE segments for {args.runs} runs "
                  f"(is LLAMA_SPEC_TRACE=1 set?); skipping trace comparison")
        else:
            # the trace records absolute slot ids; normalize so runs that land
            # on different slots still compare
            norm = [[re.sub(r"(slot|seq)=\d+", r"\1=*", ln) for ln in s] for s in segs]
            for i in range(1, len(norm)):
                if norm[i] != norm[0]:
                    n = min(len(norm[i]), len(norm[0]))
                    d = 0
                    while d < n and norm[i][d] == norm[0][d]:
                        d += 1
                    print(f"FAIL: SPECTRACE run {i} diverges from run 0 at step record {d}")
                    print(f"  run 0: {segs[0][d] if d < len(segs[0]) else '<end>'}")
                    print(f"  run {i}: {segs[i][d] if d < len(segs[i]) else '<end>'}")
                    ok = False
                    break
            else:
                print(f"SPECTRACE: {len(norm)} segments identical "
                      f"({len(norm[0])} step records each)")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

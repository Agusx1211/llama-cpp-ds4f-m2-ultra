#!/usr/bin/env python3
"""M2 llama-server dashboard — MOCK backend.

Serves the dashboard's static files (index.html, styles.css, dashboard.js,
lib/*.mjs) AND emulates the three live server endpoints against a synthetic
but schema-exact scenario, so the dashboard can be developed and screenshotted
without the real m2 server. Stdlib only, no dependencies, no build step.

Endpoints emulated (schemas verified against tools/server/server-dashboard-
bus.{h,cpp}, server-admin-dashboard.cpp, lib/core.mjs, lib/events.mjs, and the
real wire captures in fixtures/captured-v2.json):

    GET  /m2-dashboard/events          SSE: hello + bounded replay + live tail
    GET  /m2-dashboard/cache-state     prompt-cache tier snapshot (schema v1)
    GET  /m2-dashboard/content         retained prompt-in / generation-out text
    GET  /m2-dashboard/watch           SSE: live mirror of one request's output
    GET  /m2-dashboard/cache-preview   text preview of one prompt-cache entry
    GET  /slots                        per-slot live state
    GET  /props                        model identity (once)
    GET  /internal/admin/dashboard/snapshot   request registry (lanes/queue)
    POST /internal/admin/dashboard/request-detail    {request_id} -> slot bind
    POST /internal/admin/dashboard/request-control   {request_id, action:cancel}

Design note on timestamps: every request lifecycle (run_request()) advances
its OWN logical clock `t` by the ms deltas its parameters imply (tokens/rate),
not by re-reading the wall clock at each step. That decouples "how fast the
demo visibly animates" (the real `time.sleep` between `yield`s) from "what
duration the data reports" (the `t_ms` embedded in each event) — which is how
a single mechanism produces both a 30 ms trivial request and a fabricated
44-minute agent turn without the demo ever pausing for 44 real minutes: the
long turn's early history is precomputed at startup with backdated timestamps
(seed_history()), then a slow live ticker keeps extending it in real time.

Design note on content: prompt/generation TEXT is reachable only through
/m2-dashboard/{content,watch,cache-preview}. It is never attached to a bus
event, a /slots row or the registry snapshot — the bus is metadata-only by
construction in the real server, and this mock keeps that invariant so the
dashboard can never accidentally start depending on content arriving there.

Usage:
    python3 mock_server.py [--host 0.0.0.0] [--port 8877]

See tools/m2-dashboard/MOCK.md for start/stop/restart and how to edit the
synthetic scenario.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import queue
import random
import sys
import threading
import time
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent  # tools/m2-dashboard/
N_CTX = 524288
LANES = ("fast", "normal", "low")
BYTES_PER_TOKEN = 9700  # DSV4-scale KV state size/token, mid-point of real captures

# ---- content retention caps (mirrors the real server's content store) -------
CONTENT_CAPS = {
    "in_head_tokens": 512,
    "in_tail_tokens": 256,
    "out_head_bytes": 4096,
    "out_tail_bytes": 2048,
    "max_requests": 64,
    "max_bytes": 1024 * 1024,
}
WATCH_CAPS = {"mirror_bytes": 65536, "max_streams": 2}
CACHE_PREVIEW_CAPS = {"head_tokens": 64, "tail_tokens": 32}
# the mock has no tokenizer; every tokens<->characters/bytes conversion in the
# content layer is this one approximation.
CHARS_PER_TOKEN = 4.1

MIME_MAP = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".map": "application/json",
}


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Bus: bounded ring + SSE fan-out (mirrors server_dashboard::bus)
# ---------------------------------------------------------------------------

class Bus:
    def __init__(self, capacity: int = 6000):
        self.capacity = capacity
        self.ring: list[dict] = []
        self.next_seq = 1
        self.dropped = 0
        self.lock = threading.RLock()
        self.subscribers: dict[int, "queue.Queue"] = {}

    def emit(self, kind, fields=None, t_ms=None, req=None, slot=None):
        ev = {"v": 1, "t_ms": int(t_ms) if t_ms is not None else now_ms(), "k": kind}
        if req is not None:
            ev["req"] = req
        if slot is not None:
            ev["slot"] = slot
        if fields:
            ev.update(fields)
        with self.lock:
            ev["seq"] = self.next_seq
            self.next_seq += 1
            self.ring.append(ev)
            if len(self.ring) > self.capacity:
                self.ring.pop(0)
                self.dropped += 1
            subs = list(self.subscribers.values())
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev

    def subscribe(self):
        q: "queue.Queue" = queue.Queue(maxsize=4096)
        with self.lock:
            self.subscribers[id(q)] = q
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.pop(id(q), None)

    def snapshot(self):
        with self.lock:
            return list(self.ring), self.dropped, self.next_seq


BUS = Bus()

# ---------------------------------------------------------------------------
# Mutable scenario state (slots, registry, prompt-cache tiers)
# ---------------------------------------------------------------------------

STATE_LOCK = threading.RLock()


def make_idle_slot(i):
    return {"id": i, "n_ctx": N_CTX, "speculative": True, "is_processing": False}


SLOTS = [make_idle_slot(i) for i in range(4)]
REGISTRY: dict[int, dict] = {}   # req_id -> row
BINDINGS: dict[int, int] = {}    # req_id -> slot_id
CANCEL_REQUESTED: set[int] = set()
PERSISTENT_STATE: dict = {}      # the slot-0 40+ minute turn's live counters

_id_lock = threading.Lock()
_next_req_id = 1
_next_task_id = 9000
_next_entry_id = 1


def alloc_req_id():
    global _next_req_id
    with _id_lock:
        v = _next_req_id
        _next_req_id += 1
        return v


def alloc_task_id():
    global _next_task_id
    with _id_lock:
        v = _next_task_id
        _next_task_id += 1
        return v


def alloc_entry_id():
    global _next_entry_id
    with _id_lock:
        v = _next_entry_id
        _next_entry_id += 1
        return v


# ---- slots -----------------------------------------------------------------

def slot_set_idle(slot_id):
    with STATE_LOCK:
        SLOTS[slot_id] = make_idle_slot(slot_id)


def slot_set_active(slot_id, task_id, prompt, processed, cached, decoded, max_out, remain=None):
    if remain is None:
        remain = (max_out - decoded) if max_out > 0 else -1
    with STATE_LOCK:
        SLOTS[slot_id] = {
            "id": slot_id, "n_ctx": N_CTX, "speculative": True, "is_processing": True,
            "id_task": task_id,
            "n_prompt_tokens": prompt,
            "n_prompt_tokens_processed": processed,
            "n_prompt_tokens_cache": cached,
            "params": {"n_predict": max_out if max_out > 0 else -1, "stream": True},
            "next_token": [{
                "has_next_token": True, "has_new_line": False,
                "n_remain": remain, "n_decoded": decoded,
            }],
        }


def slot_set_resident_idle(slot_id, task_id, prompt, processed, cached, decoded, max_out):
    with STATE_LOCK:
        SLOTS[slot_id] = {
            "id": slot_id, "n_ctx": N_CTX, "speculative": True, "is_processing": False,
            "id_task": task_id,
            "n_prompt_tokens": prompt,
            "n_prompt_tokens_processed": processed,
            "n_prompt_tokens_cache": cached,
            "params": {"n_predict": max_out if max_out > 0 else -1, "stream": True},
            "next_token": [{
                "has_next_token": False, "has_new_line": False,
                "n_remain": max(0, max_out - decoded) if max_out > 0 else 0, "n_decoded": decoded,
            }],
        }


def slots_json():
    with STATE_LOCK:
        return [dict(s) for s in SLOTS]


# ---- registry ----------------------------------------------------------------

def reg_add(req_id, lane, state, prompt_tokens, requested_output, start_epoch=None):
    with STATE_LOCK:
        REGISTRY[req_id] = {
            "id": str(req_id), "lane": lane, "state": state,
            "start": start_epoch if start_epoch is not None else time.time(),
            "prompt_tokens": prompt_tokens, "cache_hit_tokens": 0, "output_tokens": 0,
            "requested_output_tokens": requested_output,
        }


def reg_state(req_id, state):
    with STATE_LOCK:
        if req_id in REGISTRY:
            REGISTRY[req_id]["state"] = state


def reg_remove(req_id):
    with STATE_LOCK:
        REGISTRY.pop(req_id, None)
        BINDINGS.pop(req_id, None)
        CANCEL_REQUESTED.discard(req_id)


def reg_bind(req_id, slot_id):
    with STATE_LOCK:
        BINDINGS[req_id] = slot_id


def is_cancelled(req_id):
    with STATE_LOCK:
        return req_id in CANCEL_REQUESTED


def registry_json():
    now = time.time()
    with STATE_LOCK:
        rows = list(REGISTRY.items())
        occupied = sum(1 for s in SLOTS if s.get("is_processing"))
    lane_counts = {l: {"queued": 0, "active": 0, "oldest_wait_ms": 0} for l in LANES}
    reqs = []
    for _rid, r in rows:
        age_ms = max(0, int((now - r["start"]) * 1000))
        lane = r["lane"] if r["lane"] in lane_counts else "normal"
        reqs.append({
            "id": r["id"], "lane": lane, "state": r["state"], "age_ms": age_ms,
            "prompt_tokens": r["prompt_tokens"], "cache_hit_tokens": r.get("cache_hit_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "requested_output_tokens": r.get("requested_output_tokens"),
        })
        lc = lane_counts[lane]
        if r["state"] == "queued":
            lc["queued"] += 1
            lc["oldest_wait_ms"] = max(lc["oldest_wait_ms"], age_ms)
        else:
            lc["active"] += 1
    lanes = [{
        "id": l, "queued": lane_counts[l]["queued"], "active": lane_counts[l]["active"],
        "oldest_wait_ms": lane_counts[l]["oldest_wait_ms"], "service_deficit": 0, "bypass_count": 0,
        "predicted_start_ms": [0, 0], "claimed_permits": lane_counts[l]["active"],
        "bound_permits": lane_counts[l]["active"],
    } for l in LANES]
    ring, dropped, next_seq = BUS.snapshot()
    return {
        "schema_version": 2, "sequence": next_seq,
        "availability": {"fast_refill": True},
        "registry": {
            "active_requests": len(reqs), "occupied_slots": occupied,
            "retained_events": len(ring), "event_capacity": BUS.capacity,
            "total_events": next_seq - 1, "dropped_events": dropped,
            "claimed_permits": [0, 1, 2], "bound_permits": [0, 1, 2], "total_permits": 4,
        },
        "lanes": lanes,
        "requests": reqs,
        "timeline": [],
    }


# ---- prompt-cache tiers ------------------------------------------------------

CACHE = {
    "limit_ram_bytes": 8 * 1024 ** 3,      # 8 GiB, matches prod --cache-ram 8192
    "limit_disk_bytes": 400 * 1024 ** 3,   # 400 GiB, matches prod --cache-disk-limit 400
    "limit_tokens": 0,
    "counters": {k: 0 for k in (
        "lookups", "hits_entry", "hits_resident", "misses", "saves", "spills",
        "disk_loads", "drops", "bytes_saved", "bytes_spilled", "bytes_disk_load")},
    "entries": {},  # id -> entry
    "updated_at": time.time(),
}


CACHE_PREVIEWS: dict[int, dict] = {}   # entry id -> head/tail text preview


def cache_add_entry(tokens, tier, created_ago_s=0.0, hits=0, last_hit_ago_s=None,
                    req=None, preview=None):
    eid = alloc_entry_id()
    b = int(tokens * BYTES_PER_TOKEN * random.uniform(0.9, 1.1))
    entry = {
        "id": eid, "tokens": tokens, "tier": tier, "bytes": b,
        "created_at": time.time() - created_ago_s,
        "last_hit_at": (time.time() - last_hit_ago_s) if last_hit_ago_s is not None else None,
        "hits": hits,
        "file": (f"pc-mockA1-{eid}.lcpc" if tier == "disk" else None),
        # the request whose save created the entry; None for entries the
        # server "rebuilt from disk" at startup, where that link is lost.
        "req": req,
    }
    with STATE_LOCK:
        CACHE["entries"][eid] = entry
        if preview is not None:
            CACHE_PREVIEWS[eid] = preview
    if tier == "ram":
        cache_enforce_ram_budget(protect_id=eid)
    return eid


def cache_remove_entry(eid):
    with STATE_LOCK:
        CACHE["entries"].pop(eid, None)
        CACHE_PREVIEWS.pop(eid, None)


def cache_preview_json(eid):
    with STATE_LOCK:
        e = CACHE["entries"].get(eid)
        p = CACHE_PREVIEWS.get(eid)
        created_at = e["created_at"] if e else 0.0
    if e is None or p is None:
        return {"v": 1, "id": eid, "found": False}
    out = {
        "v": 1, "id": eid, "found": True, "tier": e["tier"], "tokens": e["tokens"],
        "head": p["head"], "tail": p["tail"],
        "head_tokens": p["head_tokens"], "tail_tokens": p["tail_tokens"],
        "elided_tokens": p["elided_tokens"], "truncated": p["truncated"],
    }
    if p.get("req") is not None:
        out["req"] = p["req"]
    out["age_ms"] = max(0, int((time.time() - created_at) * 1000))
    return out


def cache_enforce_ram_budget(protect_id=None, target_fraction=0.93):
    """Every finished request re-caches its own KV state (post_save), so
    without eviction the RAM tier grows without bound. Spill the oldest
    entries to disk once usage crosses the limit — the same pressure
    response the real server's prompt cache applies."""
    evicted = []
    with STATE_LOCK:
        limit = CACHE["limit_ram_bytes"]
        ram = [e for e in CACHE["entries"].values() if e["tier"] == "ram"]
        used = sum(e["bytes"] for e in ram)
        if used <= limit:
            return
        ram.sort(key=lambda e: e["created_at"])  # oldest first
        for e in ram:
            if used <= limit * target_fraction:
                break
            if e["id"] == protect_id:
                continue
            e["tier"] = "disk"
            e["file"] = f"pc-mockA1-{e['id']}.lcpc"
            used -= e["bytes"]
            evicted.append((e["tokens"], e["bytes"]))
    for tokens, b in evicted:
        cache_record_op("spill", tokens, b)
        BUS.emit("cache_op", {"op": "spill", "tokens": tokens, "bytes": b, "ms": round(random.uniform(400, 3500), 1)})


def cache_record_lookup(kind):
    with STATE_LOCK:
        c = CACHE["counters"]
        c["lookups"] += 1
        if kind == "resident":
            c["hits_resident"] += 1
        elif kind == "entry":
            c["hits_entry"] += 1
        else:
            c["misses"] += 1
        CACHE["updated_at"] = time.time()


def cache_record_op(op, tokens=0, b=0):
    with STATE_LOCK:
        c = CACHE["counters"]
        if op == "save":
            c["saves"] += 1
            c["bytes_saved"] += b
        elif op == "spill":
            c["spills"] += 1
            c["bytes_spilled"] += b
        elif op == "disk_load":
            c["disk_loads"] += 1
            c["bytes_disk_load"] += b
        elif op == "drop":
            c["drops"] += 1
        CACHE["updated_at"] = time.time()


def cache_state_json():
    now = time.time()
    with STATE_LOCK:
        entries = list(CACHE["entries"].values())
        counters = dict(CACHE["counters"])
        limit_ram = CACHE["limit_ram_bytes"]
        limit_disk = CACHE["limit_disk_bytes"]
        limit_tok = CACHE["limit_tokens"]
        updated_at = CACHE["updated_at"]
        with_preview = set(CACHE_PREVIEWS)
    ram_entries = [e for e in entries if e["tier"] == "ram"]
    disk_entries = [e for e in entries if e["tier"] == "disk"]
    used_ram = sum(e["bytes"] for e in ram_entries)
    used_disk = sum(e["bytes"] for e in disk_entries)
    tokens_total = sum(e["tokens"] for e in ram_entries)
    out_entries = []
    for e in sorted(entries, key=lambda e: e["id"]):
        row = {
            "id": e["id"], "tokens": e["tokens"], "tier": e["tier"], "bytes": e["bytes"],
            "age_s": max(0.0, now - e["created_at"]),
            "hits": e["hits"],
            "last_hit_s": (max(0.0, now - e["last_hit_at"]) if e["last_hit_at"] else None),
        }
        if e["file"]:
            row["file"] = e["file"]
        # v3 additive fields: which request created the entry (omitted when the
        # link is unknown) and whether /m2-dashboard/cache-preview has text.
        if e.get("req") is not None:
            row["req"] = e["req"]
        row["preview"] = e["id"] in with_preview
        out_entries.append(row)
    return {
        "v": 1, "enabled": True, "t_ms": int(updated_at * 1000),
        "age_ms": max(0, int((now - updated_at) * 1000)),
        "ram": {"used_bytes": used_ram, "limit_bytes": limit_ram, "tokens": tokens_total, "limit_tokens": limit_tok},
        "disk": {"used_bytes": used_disk, "limit_bytes": limit_disk},
        "counters": counters,
        "entries": out_entries,
    }


# ---------------------------------------------------------------------------
# Synthetic content corpus.
#
# Small hand-written corpora of plausible agent/chat traffic. Prompts are
# composed head-first from an "input" corpus; generations are produced
# incrementally from an "output" corpus by TextStream, which cycles through a
# reshuffled paragraph order so a 40-minute turn keeps producing new prose
# instead of repeating one sentence.
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PARAS = [
    "You are m2-agent, an autonomous software-engineering assistant working inside a large "
    "C++/Metal codebase (a llama.cpp fork targeting Apple M2 Ultra). You operate in a checked-out "
    "git worktree: you may read and edit files, run builds and tests, and inspect benchmark "
    "output. Never invent the contents of a file you have not read. Prefer the smallest diff that "
    "fully solves the task, and stop as soon as the task is actually done.",

    "Operating rules. (1) Read before you write - always open a file before editing it. (2) Never "
    "commit, push, or rewrite history unless you are explicitly asked to. (3) Keep the build "
    "green: `cmake --build build -j` must succeed before you report done. (4) If a test fails, fix "
    "the cause, not the assertion. (5) If the task is ambiguous, name the ambiguity and then pick "
    "the reading that changes the least existing behaviour.",

    "# Tools\n\n## read_file(path: string, offset?: int, limit?: int) -> string\nReturns the "
    "contents of `path` with 1-based line numbers, at most 2000 lines per call. Use `offset` and "
    "`limit` for anything larger. Fails if the resolved path escapes the worktree root, and fails "
    "if the file is binary.",

    "## edit_file(path: string, old: string, new: string, replace_all?: bool) -> Edit\nExact-match "
    "replacement. `old` must appear verbatim exactly once unless `replace_all` is set, and the "
    "file must have been read in this session. Returns the applied hunk so you can confirm that "
    "indentation and trailing whitespace survived the edit.",

    "## run(cmd: string, timeout_ms?: int, background?: bool) -> {stdout, stderr, code}\nRuns a "
    "shell command from the worktree root. Interactive flags are rejected. `background: true` "
    "detaches the process and returns a handle you poll with `job_status`. Build commands should "
    "pass `timeout_ms: 900000`; the Metal shader compile alone takes four minutes cold.",

    "## grep(pattern: string, path?: string, glob?: string, mode?: \"content\"|\"files\") -> "
    "string\nRipgrep over the worktree. The pattern is RE2 syntax; scope by extension with `glob` "
    "(for example `*.metal`). Prefer `mode: \"files\"` when you only need to know where something "
    "lives - it is an order of magnitude cheaper in context.",

    "## bench(model: string, preset: \"pp\"|\"tg\"|\"mixed\", n_ctx?: int) -> Report\nRuns the "
    "standard M2 benchmark harness against a built binary. This costs several minutes of wall "
    "time and takes the GPU exclusively: never run two benchmarks concurrently, and never start "
    "one without saying so in your reply first.",

    "# Style\n\nAnswer in prose, not bullet soup. Cite locations as `tools/server/server.cpp:1421`. "
    "When you touch generated or vendored code, say so explicitly. If you are asked for an opinion "
    "on a design, give one - do not hedge into a menu of options with no recommendation.",
]

AGENT_TAIL_PARAS = [
    "user: ok, that build passed. now can you work out why the MXFP4_M2 expert-plane path is ~8% "
    "slower than the dense twin on the same tensor shapes? I'd expect them inside noise.",

    "assistant: Reading `ggml/src/ggml-metal/ggml-metal-mxfp4-m2.metal` now. The dense twin uses a "
    "32-wide simdgroup reduction; the expert-plane kernel drops to a 16-wide one because `ne00` "
    "isn't a multiple of 64 once the router-selected slices are packed two-per-byte.",

    "tool_result(read_file): 214 -> const short nsg = ...\n215 -> threadgroup float shmem[NSG * "
    "32];\n216 -> // NOTE: expert planes are packed 2-per-byte, so the row stride is ne00/2\n217 -> "
    "for (short i = tiisg; i < ne00/2; i += 16) {",

    "assistant: That confirms it - the 16-wide reduction costs an extra shuffle round per row. I "
    "can pad the plane stride up to 64 and keep the packing, at a cost of about 0.4% more VRAM on "
    "the expert tensors. The alternative is a separate kernel variant, which is more code and more "
    "pipeline-cache pressure for the same win.",

    "user: do it, but put the padding behind a converter flag so existing artifacts still load.",

    "assistant: Understood. Plan: (1) add `pad_plane_stride` to the MXFP4_M2 header using reserved "
    "byte 6, which is currently always zero; (2) have the converter set it only when "
    "`--pad-expert-planes` is passed; (3) branch on it once at dispatch time in the Metal backend, "
    "not per row. Starting with the header change so the reader is tolerant before anything writes "
    "the new field.",
]

CHAT_SHORT_PARAS = [
    "user: what's the difference between --cache-ram and --cache-disk-limit again?",
    "user: does llama-server reuse KV across two requests with the same system prompt?",
    "user: quick one - is the prompt cache shared between slots or per-slot?",
    "user: summarise the spill/reload rule in two sentences please.",
    "user: what does 'wasted restore' mean on the dashboard?",
    "user: is speculative decoding on by default in this build?",
    "user: give me the curl for the slots endpoint.",
    "user: how big is one token of KV state for this model, roughly?",
]

REPO_CTX_PARAS = [
    "# Repository map (llama.cpp fork, m2-ultra lane)\n\n  ggml/src/ggml-metal/        Metal "
    "backend: kernels, pipeline cache, device caps\n  ggml/src/ggml-quants.c      reference "
    "quant/dequant, used by converter tests\n  src/llama-kv-cache.cpp      unified KV cache, "
    "defrag, seq_cp / seq_keep\n  tools/server/server.cpp     HTTP server, slot scheduler, prompt "
    "cache\n  tools/server/server-dashboard-bus.cpp   bounded event ring + SSE fan-out\n  "
    "tools/m2-dashboard/         the operator dashboard this file serves",

    "tools/server/server.cpp:1421\n```cpp\n// Reuse the longest common prefix already resident on\n"
    "// this slot before falling back to the prompt cache. Everything\n// after n_past has to be "
    "recomputed.\nsize_t n_past = common_prefix(slot.cache_tokens, prompt_tokens);\nif (n_past < "
    "slot.cache_tokens.size()) {\n    llama_memory_seq_rm(mem, slot.id, n_past, -1);\n}\n```",

    "src/llama-kv-cache.cpp:903\n```cpp\nbool llama_kv_cache::seq_cp(llama_seq_id src, llama_seq_id "
    "dst,\n                            llama_pos p0, llama_pos p1) {\n    // zero-copy share: dst "
    "points at the same cells as src\n    // until either sequence writes, at which point we "
    "split.\n    ...\n}\n```",

    "CMakePresets.json (m2 lane) sets -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON "
    "-DLLAMA_CURL=OFF -DGGML_NATIVE=ON. The prod build additionally passes -DGGML_METAL_USE_BF16=ON "
    "and CI does not, which is exactly why the two disagree on the last digit of the perplexity "
    "numbers and why the nightly comparison job has been amber for a week.",

    "Recent commits on this lane:\n  902416f gguf-m2: converter: add --keep-dense-plane for "
    "experts-only artifacts\n  a424bd8 gguf-m2: add GGML_TYPE_MXFP4_M2 expert-plane encoding\n  "
    "df630ae gguf-m2: validate E4M3_M2 row data under --check-tensors\n  f78064c Merge lane "
    "gguf-m2-e4m3: E4M3_M2 dense-plane encoding",

    "tools/server/server-dashboard-bus.h:44\n```cpp\n// The bus is metadata-only by construction: "
    "every field is a\n// number, an enum, or a short interned string. Prompt and\n// generation "
    "text must never be published here - it is exposed\n// through the content endpoints, which "
    "carry their own retention\n// caps and their own access path.\nstruct event { uint64_t seq; "
    "int64_t t_ms; kind k; /* ... */ };\n```",
]

REVIEW_PARAS = [
    "Starting with the scheduler change, because that is where the risk is concentrated. The new "
    "`claim_permit()` path takes the lane lock before the slot lock, which is the opposite order "
    "from `release_permit()` two hundred lines further down. Nothing deadlocks today only because "
    "the release path never blocks on the lane lock, but that is an accident of the current "
    "implementation rather than a property anybody wrote down.",

    "The `n_past` computation reads correctly to me. It clamps against `cache_tokens.size()` "
    "before the subtraction, so the underflow that produced the 18-exabyte allocation in the "
    "August incident cannot recur here. I would still add a debug assert, because the clamp is now "
    "the only thing standing between a bad prefix match and an allocation the size of the address "
    "space.",

    "One thing I want to flag but not block on: the retry loop around the disk restore has no "
    "backoff. If the SSD tier is genuinely unhealthy, three back-to-back restores of a 2 GiB entry "
    "will each take four seconds and then fail, so a single request can hold a slot for twelve "
    "seconds doing nothing useful. A 250 ms sleep between attempts would cost nothing on the happy "
    "path and would bound the damage on the unhappy one.",

    "The naming here is doing real work and I think it is wrong. `cache_hit_tokens` counts tokens "
    "that were *reusable*, not tokens that were actually reused - when admission subsequently "
    "clears the slot, the counter still shows the full match. That is precisely the wasted-restore "
    "case the dashboard is meant to surface, so a metric that cannot distinguish the two states is "
    "actively unhelpful. Either rename it to `cache_match_tokens` or decrement it on clear.",

    "Test coverage looks thin around the partial-restore path. There is a case for a full match "
    "and a case for no match, but nothing for a match that stops mid-checkpoint, which is the case "
    "that actually exercises the recompute-tail branch. I would add one fixture with a checkpoint "
    "gap of a few hundred tokens and assert that `n_past` lands on the checkpoint boundary rather "
    "than on the raw prefix length.",

    "Metal side: the pipeline is rebuilt on every dispatch when `ne00` changes, and `ne00` changes "
    "on essentially every request because prompts have arbitrary lengths. The pipeline cache is "
    "keyed on the full shape tuple, so the cache is effectively disabled for this kernel. Keying "
    "on the padded stride instead would collapse those thousands of distinct entries into about "
    "six, and it would explain the 40 ms first-token spike that has been blamed on the KV cache.",

    "Style nit, take it or leave it: the four-deep nesting in `handle_prompt()` is hard to follow "
    "at the point where the donor-share branch and the disk-restore branch converge. An early "
    "return once the resident case is handled would flatten it to two levels and would make the "
    "invariant - that exactly one of the four provenance paths ran - visible instead of implied.",

    "I checked the arithmetic on the byte accounting and it is consistent, but it is consistent "
    "with a definition nobody stated. `bytes_spilled` counts the size of the entry at spill time, "
    "while `bytes_saved` counts the size at save time, and those differ whenever an entry is "
    "extended between the two. Over a day of traffic the two counters drift by several percent, "
    "which looks like a leak on the dashboard and is not one.",

    "The error path when the disk tier is full deserves a second look. Right now the save is "
    "dropped silently and the request completes normally, which is the right behaviour, but there "
    "is no counter for it. Operators looking at a cache hit rate that has quietly collapsed have "
    "nothing to correlate against. A `drops_no_space` counter next to the existing `drops` would "
    "have saved a couple of hours last month.",

    "Concurrency question rather than a defect: `cache_enforce_ram_budget()` mutates entries while "
    "holding the state lock, and it can spill several gigabytes' worth of entries in one pass. "
    "Every reader of the tier snapshot blocks for the duration. Since the actual file writes are "
    "asynchronous, the pass is short in practice, but the structure invites somebody to make the "
    "write synchronous later and stall the whole server.",

    "On the whole this is a good change and I would like to see it land. The scheduler ordering is "
    "the one thing I would fix before merge; the backoff, the counter, and the naming can all be "
    "follow-ups as long as somebody writes them down. I have left the lock-ordering comment inline "
    "so it does not get lost in the review thread.",

    "One last observation about the benchmark numbers in the description. The pp/tg split was "
    "measured with a warm pipeline cache, which flatters the first-token latency by roughly the "
    "same 40 ms the pipeline-cache bug costs. If you re-measure cold, I expect the improvement to "
    "look smaller and the variance to look much worse, and it would be better to publish that "
    "number now than to have somebody discover it in production.",

    "Re-reading the diff after the rebase: the conflict resolution in `slot_selected` dropped the "
    "`sim` field from one of the two emit sites. It is still emitted on the affinity path but not "
    "on the LRU fallback, so any consumer computing a similarity histogram will silently see half "
    "the population. That is the kind of thing that survives review for months because the field "
    "is optional in the schema.",

    "The documentation change reads well and I only have one correction: it says entries are "
    "evicted 'oldest first', but the implementation sorts by creation time and skips the protected "
    "entry, which means under sustained pressure the second-oldest entry can outlive the third. "
    "That is fine behaviour; it just is not what the sentence says, and somebody will eventually "
    "write a test against the sentence.",
]

MARKDOWN_PARAS = [
    "## What I changed\n\nThree files, one behavioural change:\n\n| file | lines | why |\n| --- | "
    "--- | --- |\n| `tools/server/server.cpp` | +34 -12 | reorder the permit/slot locks |\n| "
    "`tools/server/server-dashboard-bus.cpp` | +8 -0 | emit `sim` on the LRU path too |\n| "
    "`tests/test-slot-scheduler.cpp` | +96 -0 | partial-restore fixture |\n\nThe scheduler change "
    "is the only one that can affect production behaviour.",

    "### Reproducing the stall\n\n```sh\n# terminal 1 - server with a deliberately tiny disk tier\n"
    "./build/bin/llama-server -m models/dsv4-flash.gguf \\\n    --cache-ram 512 "
    "--cache-disk-limit 2 \\\n    --parallel 4 --ctx-size 524288 --metrics\n\n# terminal 2 - two "
    "concurrent 200k-token prompts against the same prefix\nfor i in 1 2; do\n  curl -sN "
    "localhost:8080/v1/chat/completions \\\n       -H 'content-type: application/json' \\\n       "
    "-d @fixtures/deep-context.json &\ndone\nwait\n```\n\nThe second request blocks for the full "
    "duration of the first request's spill, which is the bug.",

    "### The fix\n\n```cpp\n// before: lane lock, then slot lock\nstd::lock_guard<std::mutex> "
    "lane_lk(lane.mtx);\nstd::lock_guard<std::mutex> slot_lk(slot.mtx);\n\n// after: a single "
    "scoped_lock so the ordering is fixed by the\n// compiler rather than by convention, and the "
    "release path below\n// can no longer disagree with it.\nstd::scoped_lock lk(lane.mtx, "
    "slot.mtx);\n```\n\n`std::scoped_lock` uses a deadlock-avoidance algorithm across all the "
    "mutexes it is given, so the two call sites cannot drift apart again.",

    "> **Note**\n> This changes the observable ordering of `slot_selected` and `dispatched` events "
    "under contention. Consumers that assumed `dispatched` always precedes `slot_selected` for a "
    "given request will still be correct - the guarantee is unchanged - but the wall-clock gap "
    "between them shrinks from milliseconds to microseconds, which will make some latency "
    "histograms look like they lost a mode.",

    "### Follow-ups I am not doing here\n\n1. Backoff on the disk-restore retry loop (issue #4412).\n"
    "2. `drops_no_space` counter for the full-disk path.\n3. Renaming `cache_hit_tokens` to "
    "`cache_match_tokens`, which is a breaking change to the admin snapshot schema and should go "
    "out with a schema bump rather than sneak in here.\n4. Keying the Metal pipeline cache on the "
    "padded stride - a much larger win, and a much larger change.",

    "### Benchmarks\n\n```\nmodel                     preset   before      after      delta\n"
    "dsv4-flash-0731-full.m2   pp512    1841 t/s    1852 t/s   +0.6%\ndsv4-flash-0731-full.m2   "
    "tg128    28.4 t/s    28.5 t/s   +0.4%\ndsv4-flash-0731-full.m2   mixed    on 4 slots, p99 "
    "first-token 412 ms -> 118 ms\n```\n\nThroughput is unchanged, as expected - the change only "
    "affects who waits, not how fast anything runs. The p99 first-token improvement is the whole "
    "point of the patch.",

    "```diff\n-    if (n_past > slot.cache_tokens.size()) {\n-        n_past = "
    "slot.cache_tokens.size();\n-    }\n+    n_past = std::min(n_past, slot.cache_tokens.size());\n+"
    "    GGML_ASSERT(n_past <= prompt_tokens.size() && \"prefix longer than prompt\");\n```\n\nSame "
    "semantics, but the assert documents the invariant that the clamp is protecting, which is the "
    "part that was missing when this last went wrong.",
]

SHORT_OUT_PARAS = [
    "`--cache-ram` is the in-memory budget for saved prompt-cache entries; `--cache-disk-limit` is "
    "the SSD spillover budget underneath it. Entries move down when RAM is over budget and come "
    "back up on the next matching prefix.",

    "Yes - as long as the second request's token prefix matches, the shared prefix is reused "
    "without recomputation. It is a token-level prefix match, not a string match, so a different "
    "tokenizer or a changed system prompt breaks it completely.",

    "Per-slot for the resident KV, shared for the saved entries: a slot reuses its own resident "
    "state for free, and anything else has to come through the prompt cache (or a zero-copy share "
    "from a live donor slot).",

    "It means a restore completed - sometimes several gigabytes and several seconds of it - and "
    "then admission decided the slot had to be cleared anyway, so every restored token was thrown "
    "away. It is pure waste and it is the single most expensive thing the scheduler can do.",

    "Speculative decoding is on in this build; you can see it in `/slots` as `\"speculative\": "
    "true`. Acceptance runs 0.4-0.95 depending on how predictable the output is.",

    "```sh\ncurl -s localhost:8080/slots | python3 -m json.tool\n```",

    "About 9.7 kB per token for this model at the current cache type, so a 200k-token entry is "
    "roughly 1.9 GiB. That is why the RAM tier holds only a handful of deep-context entries before "
    "it starts spilling.",
]

TOOL_PARAS = [
    "I'll start by finding where the restore path decides between the RAM and disk tiers.\n\n"
    "tool_call(grep): pattern=\"tier == .disk\" glob=\"*.cpp\" mode=\"files\"",

    "tool_result(grep): tools/server/server.cpp\ntools/server/server-prompt-cache.cpp\n"
    "tests/test-prompt-cache.cpp",

    "tool_call(read_file): path=\"tools/server/server-prompt-cache.cpp\" offset=180 limit=120",

    "tool_result(read_file): 180 -> // Load an entry back into RAM. Called with the tier lock "
    "held.\n181 -> bool prompt_cache::load(entry & e) {\n182 ->     if (e.tier != tier::disk) "
    "return true;\n183 ->     const auto t0 = ggml_time_us();\n184 ->     ...",

    "That is the whole load path, and it does not have a retry at all - the retry I was worried "
    "about is one level up, in the caller. Reading that next before I say anything else.",

    "tool_call(run): cmd=\"cmake --build build -j 2>&1 | tail -20\" timeout_ms=900000",

    "tool_result(run): [100%] Linking CXX executable ../../bin/llama-server\n[100%] Built target "
    "llama-server\ncode=0",

    "Build is green. Running the scheduler tests only, since the full suite takes eleven minutes "
    "and nothing outside the scheduler changed.\n\ntool_call(run): cmd=\"ctest --test-dir build -R "
    "slot-scheduler --output-on-failure\"",

    "tool_result(run): Test project /home/m2/llama.cpp/build\n    Start 14: test-slot-scheduler\n"
    "1/1 Test #14: test-slot-scheduler .............   Passed    3.42 sec\n\n100% tests passed, 0 "
    "tests failed out of 1\ncode=0",
]

INPUT_CORPUS = {
    "agent_deep": (AGENT_SYSTEM_PARAS, AGENT_TAIL_PARAS),
    "repo_ctx": (REPO_CTX_PARAS, AGENT_TAIL_PARAS),
    "chat_short": (CHAT_SHORT_PARAS, CHAT_SHORT_PARAS),
}

OUTPUT_CORPUS = {
    "review": REVIEW_PARAS,
    "markdown": MARKDOWN_PARAS,
    "short": SHORT_OUT_PARAS,
    "tool": TOOL_PARAS,
}


def _clip_head(text, n_chars):
    """Clip to n_chars on a word boundary, marking the cut."""
    if len(text) <= n_chars:
        return text
    cut = text[:n_chars]
    sp = cut.rfind(" ")
    if sp > n_chars * 0.6:
        cut = cut[:sp]
    return cut.rstrip() + " …"


def _compose(paras, n_chars, start=0):
    """Join whole corpus paragraphs into ~n_chars of text, rotating from
    `start`. Only clips mid-paragraph when a single paragraph already exceeds
    the budget (which is what a genuinely tiny prompt looks like)."""
    if n_chars <= 0 or not paras:
        return ""
    n = len(paras)
    if not any(len(p) <= n_chars for p in paras):
        return _clip_head(paras[start % n], n_chars)
    picked, total, i = [], 0, start
    for _ in range(n * 6):
        p = paras[i % n]
        i += 1
        add = len(p) + (2 if picked else 0)
        if total + add > n_chars:
            if picked:
                break
            continue  # this one is oversized; try the next
        picked.append(p)
        total += add
        if total >= n_chars * 0.9:
            break
    return "\n\n".join(picked) if picked else _clip_head(paras[start % n], n_chars)


def _compose_tail(paras, n_chars, start=0):
    """Same, but the text reads as the END of a longer transcript."""
    if n_chars <= 0 or not paras:
        return ""
    text = _compose(paras, max(n_chars, 240), start=start)
    if len(text) <= n_chars:
        return text
    cut = text[-n_chars:]
    sp = cut.find(" ")
    if 0 <= sp < n_chars * 0.4:
        cut = cut[sp + 1:]
    return "… " + cut.lstrip()


def make_input_content(kind, n_tokens, seed=0):
    """Head/tail view of a prompt under the in_head/in_tail token caps."""
    head_cap = CONTENT_CAPS["in_head_tokens"]
    tail_cap = CONTENT_CAPS["in_tail_tokens"]
    heads, tails = INPUT_CORPUS.get(kind) or INPUT_CORPUS["agent_deep"]
    head_tokens = min(head_cap, n_tokens)
    tail_tokens = min(tail_cap, max(0, n_tokens - head_tokens))
    elided = max(0, n_tokens - head_tokens - tail_tokens)
    return {
        "n_tokens": n_tokens,
        "head": _compose(heads, int(head_tokens * CHARS_PER_TOKEN), start=0),
        "tail": _compose_tail(tails, int(tail_tokens * CHARS_PER_TOKEN), start=seed) if tail_tokens else "",
        "head_tokens": head_tokens,
        "tail_tokens": tail_tokens,
        "elided_tokens": elided,
        "truncated": elided > 0,
    }


def cache_preview_build(tokens, kind="agent_deep", seed=0, req=None):
    """Head/tail preview of a prompt-cache entry, under the preview caps."""
    heads, tails = INPUT_CORPUS.get(kind) or INPUT_CORPUS["agent_deep"]
    head_tokens = min(CACHE_PREVIEW_CAPS["head_tokens"], tokens)
    tail_tokens = min(CACHE_PREVIEW_CAPS["tail_tokens"], max(0, tokens - head_tokens))
    elided = max(0, tokens - head_tokens - tail_tokens)
    p = {
        "head": _compose(heads, int(head_tokens * CHARS_PER_TOKEN), start=0),
        "tail": _compose_tail(tails, int(tail_tokens * CHARS_PER_TOKEN), start=seed) if tail_tokens else "",
        "head_tokens": head_tokens,
        "tail_tokens": tail_tokens,
        "elided_tokens": elided,
        "truncated": elided > 0,
    }
    if req is not None:
        p["req"] = req
    return p


class TextStream:
    """Endless plausible prose from a paragraph corpus, taken a few tokens at
    a time. The order is reshuffled every full pass so a 40-minute generation
    never settles into a visible loop."""

    def __init__(self, paras, seed=0):
        self._paras = list(paras) or ["…"]
        self._rng = random.Random(seed)
        self._order = list(range(len(self._paras)))
        self._rng.shuffle(self._order)
        self._oi = 0
        self._buf = ""
        self._pos = 0

    def _refill(self, need):
        while len(self._buf) - self._pos < need:
            idx = self._order[self._oi % len(self._order)]
            self._oi += 1
            if self._oi % len(self._order) == 0:
                self._rng.shuffle(self._order)
            self._buf += self._paras[idx] + "\n\n"
        if self._pos > 65536:  # keep the working buffer bounded forever
            self._buf = self._buf[self._pos:]
            self._pos = 0

    def take_chars(self, n):
        n = max(1, int(n))
        self._refill(n + 128)
        end = self._pos + n
        while end < len(self._buf) and not self._buf[end].isspace():
            end += 1                      # never split a word
        while end < len(self._buf) and self._buf[end].isspace():
            end += 1                      # carry the separator with this chunk
        out = self._buf[self._pos:end]
        self._pos = end
        return out

    def take_tokens(self, tokens):
        return self.take_chars(round(max(1, tokens) * CHARS_PER_TOKEN))


def make_text_stream(kind, seed=0):
    return TextStream(OUTPUT_CORPUS.get(kind) or REVIEW_PARAS, seed=seed)


class LiveText:
    """Bounded mirror of one request's generated text.

    Keeps the first `head_cap` bytes forever and the last `mirror_cap` bytes in
    a rolling window (the tail preview is derived from that window), plus the
    monotone byte/token counters /watch reports as `cursor` and `n_dec`.
    Watchers block on `cond` and resync whenever `rewind_seq` moves or they
    fall behind `base`."""

    def __init__(self, head_cap=None, tail_cap=None, mirror_cap=None):
        self.head_cap = head_cap or CONTENT_CAPS["out_head_bytes"]
        self.tail_cap = tail_cap or CONTENT_CAPS["out_tail_bytes"]
        self.mirror_cap = mirror_cap or WATCH_CAPS["mirror_bytes"]
        self.cond = threading.Condition()
        self._head = b""
        self.mirror = bytearray()
        self.base = 0          # absolute offset of mirror[0]
        self.n_bytes = 0       # bytes currently in the stream == cursor
        self.n_tokens = 0
        self.rewind_seq = 0
        self.closed = False
        self.end_reason = None

    def append(self, text, tokens):
        chunk = text.encode("utf-8")
        if not chunk:
            return
        with self.cond:
            if self.closed:
                return
            if len(self._head) < self.head_cap:
                self._head += chunk[:self.head_cap - len(self._head)]
            self.mirror += chunk
            self.n_bytes += len(chunk)
            self.n_tokens += int(tokens)
            over = len(self.mirror) - self.mirror_cap
            if over > 0:
                while over < len(self.mirror) and (self.mirror[over] & 0xC0) == 0x80:
                    over += 1               # keep the window UTF-8 aligned
                del self.mirror[:over]
                self.base += over
            self.cond.notify_all()

    def rewind(self, n_bytes, n_tokens):
        """Stop-word trim: the tail we already published turned out not to be
        part of the answer. Watchers resync from the whole mirror."""
        with self.cond:
            if self.closed:
                return 0
            n = min(int(n_bytes), len(self.mirror), self.n_bytes)
            if n <= 0:
                return 0
            del self.mirror[len(self.mirror) - n:]
            self.n_bytes -= n
            self.n_tokens = max(0, self.n_tokens - int(n_tokens))
            if self.n_bytes < len(self._head):
                self._head = self._head[:self.n_bytes]
            self.rewind_seq += 1
            self.cond.notify_all()
            return n

    def close(self, reason="finished"):
        with self.cond:
            if self.closed:
                return
            self.closed = True
            self.end_reason = reason
            self.cond.notify_all()

    def snapshot_output(self):
        with self.cond:
            n = self.n_bytes
            if n <= self.head_cap + self.tail_cap:
                # the rolling window is far larger than head+tail, so short
                # generations are still retained in full and are not truncated
                full = bytes(self.mirror)
                return {"n_tokens": self.n_tokens, "n_bytes": n,
                        "head": full.decode("utf-8", "ignore"), "tail": "",
                        "elided_bytes": 0, "truncated": False}
            head = bytes(self._head)
            tail = bytes(self.mirror[-self.tail_cap:])
            elided = max(0, n - len(head) - len(tail))
            return {"n_tokens": self.n_tokens, "n_bytes": n,
                    "head": head.decode("utf-8", "ignore"),
                    "tail": tail.decode("utf-8", "ignore"),
                    "elided_bytes": elided, "truncated": elided > 0}


# ---------------------------------------------------------------------------
# Bounded content store: at most max_requests / max_bytes, FIFO eviction.
# In-flight ("live") records are never evicted - dropping the mirror of a
# request that is still generating would break /watch.
# ---------------------------------------------------------------------------

CONTENT_LOCK = threading.RLock()
CONTENT_STORE: "OrderedDict[int, dict]" = OrderedDict()
CONTENT_EVICTED_IDS: "deque[int]" = deque(maxlen=4096)
CONTENT_EVICTED_SET: set = set()
CONTENT_STATS = {"evicted": 0, "bytes": 0}


def _content_bytes(rec):
    """4*in_head_ids + 4*in_tail_ids + len(out_head) + len(out_tail), with the
    token id arrays approximated from the head/tail token counts."""
    n = 0
    inp = rec.get("input")
    if inp:
        n += 4 * inp["head_tokens"] + 4 * inp["tail_tokens"]
    out = rec.get("output")
    if out:
        n += len(out["head"].encode("utf-8")) + len(out["tail"].encode("utf-8"))
    return n


def _content_evict_locked():
    caps = CONTENT_CAPS
    while CONTENT_STORE and (len(CONTENT_STORE) > caps["max_requests"]
                             or CONTENT_STATS["bytes"] > caps["max_bytes"]):
        victim = None
        for rid, rec in CONTENT_STORE.items():
            if rec["state"] == "live":
                continue
            victim = rid
            break
        if victim is None:
            break
        rec = CONTENT_STORE.pop(victim)
        CONTENT_STATS["bytes"] -= rec["bytes"]
        CONTENT_STATS["evicted"] += 1
        if len(CONTENT_EVICTED_IDS) == CONTENT_EVICTED_IDS.maxlen and CONTENT_EVICTED_IDS:
            CONTENT_EVICTED_SET.discard(CONTENT_EVICTED_IDS[0])
        CONTENT_EVICTED_IDS.append(victim)
        CONTENT_EVICTED_SET.add(victim)


def content_begin(req_id, slot_id, inp, live, t_ms):
    """Retain a request as in-flight. `inp` is None for requests that never
    reached prefill (queued/rejected); `live` is None with it."""
    rec = {"req": req_id, "state": "live", "slot": slot_id, "t_ms": int(t_ms),
           "input": inp, "output": None, "live": live, "bytes": 0}
    rec["bytes"] = _content_bytes(rec)
    with CONTENT_LOCK:
        old = CONTENT_STORE.pop(req_id, None)
        if old is not None:
            CONTENT_STATS["bytes"] -= old["bytes"]
        CONTENT_STORE[req_id] = rec
        CONTENT_STATS["bytes"] += rec["bytes"]
        CONTENT_EVICTED_SET.discard(req_id)
        _content_evict_locked()
    return rec


def content_finalize(req_id, reason="finished", t_ms=None):
    live = None
    with CONTENT_LOCK:
        rec = CONTENT_STORE.get(req_id)
        if rec is None:
            return
        live = rec.get("live")
        rec["state"] = "final"
        if t_ms is not None:
            rec["t_ms"] = int(t_ms)
        if live is not None:
            rec["output"] = live.snapshot_output()
            rec["live"] = None
        CONTENT_STATS["bytes"] -= rec["bytes"]
        rec["bytes"] = _content_bytes(rec)
        CONTENT_STATS["bytes"] += rec["bytes"]
        _content_evict_locked()
    if live is not None:
        live.close(reason)


def content_lookup(req_id):
    with CONTENT_LOCK:
        rec = CONTENT_STORE.get(req_id)
        if rec is not None:
            return dict(rec), None
        return None, ("evicted" if req_id in CONTENT_EVICTED_SET else "absent")


def content_store_stats():
    with CONTENT_LOCK:
        return {"requests": len(CONTENT_STORE),
                "bytes": CONTENT_STATS["bytes"],
                "evicted": CONTENT_STATS["evicted"]}


def content_json(req_id):
    rec, reason = content_lookup(req_id)
    if rec is None:
        return {"v": 1, "req": req_id, "found": False, "reason": reason,
                "caps": dict(CONTENT_CAPS), "store": content_store_stats()}
    return {
        "v": 1, "req": req_id, "found": True,
        "state": rec["state"], "slot": rec["slot"], "t_ms": rec["t_ms"],
        "caps": dict(CONTENT_CAPS), "store": content_store_stats(),
        "input": rec["input"],
        # while the request is still generating the authoritative output is the
        # /watch mirror, not a snapshot, so this is deliberately null.
        "output": None if rec["state"] == "live" else rec["output"],
    }


# ---------------------------------------------------------------------------
# Request lifecycle: one reusable, fully parametrized generator.
#
# `yield <seconds>` paces the REAL delivery of SSE frames (how fast the demo
# visibly animates); the logical clock `t` — advanced explicitly by computed
# ms deltas, never by re-reading the wall clock — is what ends up in each
# event's `t_ms`. Calling this generator to completion with real sleeps (the
# live driver) and draining it instantly with a backdated `start_ms` (seeding
# history at boot) are the same code path with different pacing.
# ---------------------------------------------------------------------------

def run_request(slot_id, **cfg):
    lane = cfg["lane"]
    n_prompt = cfg["n_prompt"]
    max_out = cfg.get("max_out", 0)
    req_id = alloc_req_id()
    task_id = alloc_task_id()

    t = float(cfg.get("start_ms") or now_ms())
    queued_at = t
    reg_add(req_id, lane, "queued", n_prompt, max_out if max_out > 0 else None, start_epoch=t / 1000.0)
    BUS.emit("queued", {"lane": lane, "n_prompt": n_prompt, "max_out": max_out}, t_ms=int(t), req=req_id)

    wait_s = random.uniform(*cfg.get("queued_wait_s", (0.05, 0.35)))
    yield wait_s
    t += wait_s * 1000.0

    if cfg.get("queue_only"):
        # never reaches prefill, so there is no tokenized input to retain: the
        # record exists with input=null so the UI can tell "not retained" from
        # "retained, but there was never any content".
        content_begin(req_id, None, None, None, int(t))
        for _ in range(cfg.get("defer_count", 2)):
            step_s = random.uniform(0.6, 1.6)
            yield step_s
            t += step_s * 1000.0
            if is_cancelled(req_id):
                BUS.emit("terminal", {"state": "cancelled", "reason": "client_cancel", "code": 500}, t_ms=int(t), req=req_id)
                content_finalize(req_id, "gone", t_ms=int(t))
                reg_remove(req_id)
                return
            BUS.emit("deferred", {"why": "capacity"}, t_ms=int(t), req=req_id)
        step_s = random.uniform(0.6, 1.2)
        yield step_s
        t += step_s * 1000.0
        qt = cfg.get("queue_terminal", {"state": "timed_out", "reason": "run_timeout", "code": 503})
        BUS.emit("terminal", qt, t_ms=int(t), req=req_id)
        content_finalize(req_id, "gone", t_ms=int(t))
        reg_remove(req_id)
        return

    BUS.emit("dispatched", {"lane": lane, "wait_ms": int(t - queued_at)}, t_ms=int(t), req=req_id)
    reg_state(req_id, "prefill")

    t += 1.0
    sel_fields = {"sel": cfg.get("sel", "lru"), "lcp": cfg.get("sel_lcp", 0), "n_task": n_prompt}
    if cfg.get("sim") is not None:
        sel_fields["sim"] = cfg["sim"]
    BUS.emit("slot_selected", sel_fields, t_ms=int(t), req=req_id, slot=slot_id)
    reg_bind(req_id, slot_id)
    slot_set_active(slot_id, task_id, n_prompt, 0, 0, 0, max_out)

    src = cfg.get("restore_src", "miss")
    restore_tokens = cfg.get("restore_tokens", 0)
    restore_bytes = cfg.get("restore_bytes", 0)
    restore_ms = cfg.get("restore_ms", round(random.uniform(0.02, 0.2), 3))
    real_wait = 0.03
    if src == "disk":
        real_wait = min(3.0, max(0.5, restore_ms / 1000.0 * 0.4))
    elif restore_ms > 200:
        real_wait = min(1.0, restore_ms / 1000.0 * 0.25)
    yield real_wait
    t += restore_ms
    rf = {"src": src, "tokens": restore_tokens, "bytes": restore_bytes, "n_task": n_prompt, "ms": restore_ms}
    if cfg.get("donor_slot") is not None:
        rf["donor_slot"] = cfg["donor_slot"]
    BUS.emit("cache_restore", rf, t_ms=int(t), req=req_id, slot=slot_id)
    cache_record_lookup("miss" if src == "miss" else ("resident" if src == "resident" else "entry"))
    if src == "disk":
        cache_record_op("disk_load", restore_tokens, restore_bytes)
        BUS.emit("cache_op", {"op": "disk_load", "tokens": restore_tokens, "bytes": restore_bytes, "ms": restore_ms}, t_ms=int(t))

    admission_outcome = cfg.get("admission_outcome")
    if admission_outcome:
        t += 0.5
        af = {"outcome": admission_outcome}
        if admission_outcome in ("ready_reuse", "ready_spanless"):
            af.update({"lcp": cfg["admission_lcp"], "resident": cfg["admission_resident"],
                       "frontier": cfg["admission_frontier"], "span_end": cfg["admission_span_end"]})
        elif admission_outcome == "ready_full_clear":
            af.update({"why": cfg["admission_why"], "resident": cfg["admission_resident"],
                       "lcp": cfg["admission_lcp"], "span_end": cfg["admission_span_end"]})
        BUS.emit("admission", af, t_ms=int(t), req=req_id, slot=slot_id)

    n_past = cfg.get("n_past", restore_tokens if admission_outcome != "ready_full_clear" else 0)
    lcp = cfg.get("lcp", n_past)
    gap_why = cfg.get("gap_why")
    t += 1.0
    psf = {"n_prompt": n_prompt, "n_past": n_past, "lcp": lcp}
    if gap_why and n_past < lcp:
        psf["gap_why"] = gap_why
    BUS.emit("prompt_start", psf, t_ms=int(t), req=req_id, slot=slot_id)
    t_prompt = t

    # Content (NOT on the bus): the request has a tokenized prompt now, so
    # retain its head/tail view and open a mirror for the generation.
    content_in = cfg.get("content_in", "agent_deep")
    out_stream = make_text_stream(cfg.get("content_out", "review"), seed=req_id * 7919)
    live = LiveText()
    content_begin(req_id, slot_id,
                  make_input_content(content_in, n_prompt, seed=req_id), live, int(t))

    # processed = NEW tokens processed so far (excludes the cached/restored
    # prefix) so cached+processed sums to n_prompt, matching the real
    # n_prompt_tokens_processed convention (core.mjs slotContextTokens()).
    slot_set_active(slot_id, task_id, n_prompt, 0, n_past, 0, max_out)

    remaining = n_prompt - n_past
    rate = cfg.get("prefill_rate", 800.0)
    ticks = max(1, cfg.get("prefill_ticks", 6)) if remaining > 0 else 0
    real_step = cfg.get("prefill_real_step", 0.4)
    done = n_past
    for i in range(ticks):
        if is_cancelled(req_id):
            BUS.emit("terminal", {"state": "cancelled", "reason": "client_cancel", "code": 500}, t_ms=int(t), req=req_id)
            content_finalize(req_id, "gone", t_ms=int(t))
            reg_remove(req_id)
            slot_set_idle(slot_id)
            return
        per = max(1, remaining // ticks)
        batch = (remaining - (done - n_past)) if i == ticks - 1 else per
        done += batch
        t += batch / rate * 1000.0
        yield real_step
        BUS.emit("prefill_progress", {"n_done": done, "n_prompt": n_prompt, "n_batch": batch, "ms": round(t - t_prompt, 1)},
                 t_ms=int(t), req=req_id, slot=slot_id)
        slot_set_active(slot_id, task_id, n_prompt, done - n_past, n_past, 0, max_out)
    reg_state(req_id, "decode")
    t_decode0 = t

    decode_tokens = cfg.get("decode_tokens", 200)
    decode_rate = cfg.get("decode_rate", 25.0)
    draft_rate = cfg.get("draft_rate", 0.0)
    draft_acc = cfg.get("draft_acc", 0.8)
    d_ticks = max(1, cfg.get("decode_ticks", 8))
    real_step_d = cfg.get("decode_real_step", 0.55)
    n = 0
    draft_n = 0
    draft_a = 0
    cancel_at = cfg.get("cancel_after_ticks")
    stall_at = cfg.get("stall_at_tick", d_ticks // 2) if cfg.get("stall") else None
    for i in range(d_ticks):
        if is_cancelled(req_id) or (cancel_at is not None and i == cancel_at):
            t += 5.0
            BUS.emit("terminal", {"state": "cancelled", "reason": "client_cancel", "code": 500}, t_ms=int(t), req=req_id)
            content_finalize(req_id, "gone", t_ms=int(t))
            reg_remove(req_id)
            slot_set_idle(slot_id)
            return
        if stall_at is not None and i == stall_at:
            yield random.uniform(6.0, 10.0)
            t += random.uniform(4.5, 6.0) * 60_000.0  # reported stall gap before the deadline fires
            BUS.emit("terminal", {"state": "timed_out", "reason": "run_stall", "code": 504}, t_ms=int(t), req=req_id)
            content_finalize(req_id, "gone", t_ms=int(t))
            reg_remove(req_id)
            slot_set_idle(slot_id)
            return
        per = max(1, decode_tokens // d_ticks)
        step = (decode_tokens - n) if i == d_ticks - 1 else per
        n += step
        draft_n += round(step * draft_rate)
        draft_a += round(step * draft_rate * draft_acc)
        t += step / decode_rate * 1000.0
        yield real_step_d
        live.append(out_stream.take_tokens(step), step)  # content only, never on the bus
        tps = decode_rate * random.uniform(0.85, 1.15)
        BUS.emit("decode_progress", {"n_dec": n, "draft_n": draft_n, "draft_a": draft_a,
                                      "ms": round(t - t_decode0, 1), "tps": round(tps, 2)},
                 t_ms=int(t), req=req_id, slot=slot_id)
        slot_set_active(slot_id, task_id, n_prompt + n, n_prompt - n_past, n_past, n, max_out,
                         remain=(max_out - n if max_out > 0 else -1))

    pp_ms = t_decode0 - t_prompt
    tg_ms = t - t_decode0
    t += 1.0
    BUS.emit("finished", {"stop": cfg.get("stop", "eos"), "n_prompt": n_prompt, "n_cached": n_past,
                           "n_dec": n, "draft_n": draft_n, "draft_a": draft_a,
                           "pp_ms": round(pp_ms, 1), "tg_ms": round(tg_ms, 1)}, t_ms=int(t), req=req_id, slot=slot_id)
    t += 1.0
    BUS.emit("terminal", {"state": "completed", "reason": "completed", "code": 600}, t_ms=int(t), req=req_id)
    content_finalize(req_id, "finished", t_ms=int(t))
    reg_remove(req_id)

    if cfg.get("post_save", True) and (n_past + n) > 200:
        tot = n_past + n
        b = int(tot * BYTES_PER_TOKEN * random.uniform(0.9, 1.1))
        cache_record_op("save", tot, b)
        BUS.emit("cache_op", {"op": "save", "tokens": tot, "bytes": b, "ms": 0.0}, t_ms=int(t) + 2)
        cache_add_entry(tot, "ram", req=req_id,
                        preview=cache_preview_build(tot, kind=content_in, seed=req_id, req=req_id))

    slot_set_resident_idle(slot_id, task_id, n_prompt + n, n_prompt - n_past, n_past, n, max_out)
    lo, hi = cfg.get("idle_after", (0.6, 2.5))
    yield random.uniform(lo, hi)
    slot_set_idle(slot_id)


def drain(gen):
    """Run a request generator to completion with no real pacing (seeding)."""
    try:
        while True:
            next(gen)
    except StopIteration:
        pass


# ---------------------------------------------------------------------------
# Scenario templates — every visual state the dashboard needs to render.
# Each returns a run_request() generator; **extra lets seed_history() pass a
# backdated start_ms through unchanged.
# ---------------------------------------------------------------------------

def tpl_trivial(slot_id, **extra):
    """Full cache hit, ~a dozen tokens out — the timeline's hardest case."""
    n_prompt = random.randint(12, 60)
    cached = n_prompt - random.randint(0, 3)
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=random.choice([0, 32, 64]),
        sel="lcp_affinity", sel_lcp=cached, sim=round(cached / n_prompt, 4),
        restore_src="resident", restore_tokens=cached, restore_bytes=0, restore_ms=round(random.uniform(0.05, 0.4), 3),
        admission_outcome="ready_spanless", admission_lcp=cached,
        admission_resident=cached + random.randint(0, 20), admission_frontier=cached + random.randint(0, 20),
        admission_span_end=n_prompt,
        n_past=cached, lcp=cached,
        decode_tokens=random.randint(3, 16), decode_rate=random.uniform(20, 45),
        draft_rate=random.choice([0, 0, 1.3]), draft_acc=random.uniform(0.6, 0.9),
        prefill_rate=3000, prefill_ticks=1, prefill_real_step=0.02,
        decode_ticks=1, decode_real_step=0.02,
        queued_wait_s=(0.01, 0.05), idle_after=(0.3, 1.2),
        # small enough that the prompt is retained whole: truncated=false, tail=""
        content_in="chat_short", content_out="short",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_short_chat(slot_id, **extra):
    """Typical short chat turn: high resident reuse, quick decode."""
    n_prompt = random.randint(2000, 16000)
    cached = int(n_prompt * random.uniform(0.85, 0.99))
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=random.choice([0, 512, 1024]),
        sel="lcp_affinity", sel_lcp=cached, sim=round(cached / n_prompt, 4),
        restore_src="resident", restore_tokens=cached, restore_bytes=0, restore_ms=round(random.uniform(0.2, 1.5), 3),
        admission_outcome="ready_spanless", admission_lcp=cached,
        admission_resident=cached, admission_frontier=cached, admission_span_end=n_prompt,
        n_past=cached, lcp=cached,
        decode_tokens=random.randint(80, 400), decode_rate=random.uniform(25, 42),
        draft_rate=1.3, draft_acc=random.uniform(0.75, 0.92),
        prefill_rate=900, prefill_ticks=random.randint(1, 3), prefill_real_step=0.3,
        decode_ticks=random.randint(4, 9), decode_real_step=random.uniform(0.4, 0.9),
        queued_wait_s=(0.02, 0.3),
        content_in="agent_deep", content_out="markdown",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_deep_context(slot_id, **extra):
    """100k+ token prompt, RAM-tier partial reuse, long-ish decode."""
    n_prompt = random.randint(60_000, 150_000)
    cached = int(n_prompt * random.uniform(0.55, 0.85))
    b = int(cached * BYTES_PER_TOKEN)
    cfg = dict(
        lane=random.choice(["normal", "low"]), n_prompt=n_prompt, max_out=0,
        sel="lcp_affinity", sel_lcp=cached, sim=round(cached / n_prompt, 4),
        restore_src="ram", restore_tokens=cached, restore_bytes=b,
        restore_ms=round(cached / 40000 * 1000 + random.uniform(50, 400), 1),
        admission_outcome="ready_reuse", admission_lcp=cached,
        admission_resident=cached, admission_frontier=cached, admission_span_end=n_prompt + random.randint(500, 2000),
        n_past=cached, lcp=cached,
        # >=1800 tokens out is ~7 kB of text, so this template always overruns
        # the 4 kB+2 kB output cap: the "generation truncated" case.
        decode_tokens=random.randint(1800, 3200), decode_rate=random.uniform(15, 30),
        draft_rate=1.3, draft_acc=random.uniform(0.5, 0.85),
        prefill_rate=780, prefill_ticks=random.randint(4, 8), prefill_real_step=random.uniform(0.5, 1.0),
        decode_ticks=random.randint(10, 16), decode_real_step=random.uniform(0.6, 1.1),
        queued_wait_s=(0.05, 0.4),
        content_in="agent_deep", content_out="review",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_ssd_restore(slot_id, **extra):
    """SSD-tier restore: slow, multi-second, then a normal decode."""
    n_prompt = random.randint(25_000, 60_000)
    tokens = int(n_prompt * random.uniform(0.9, 0.99))
    b = int(tokens * BYTES_PER_TOKEN * random.uniform(0.95, 1.05))
    ms = random.uniform(700, 4200)
    cfg = dict(
        lane=random.choice(["normal", "low"]), n_prompt=n_prompt, max_out=random.choice([0, 4096]),
        sel="lru", sel_lcp=0,
        restore_src="disk", restore_tokens=tokens, restore_bytes=b, restore_ms=round(ms, 1),
        admission_outcome="ready_reuse", admission_lcp=tokens,
        admission_resident=tokens, admission_frontier=tokens, admission_span_end=n_prompt + 500,
        n_past=tokens, lcp=tokens,
        decode_tokens=random.randint(200, 1200), decode_rate=random.uniform(18, 32),
        draft_rate=1.3, draft_acc=random.uniform(0.6, 0.85),
        prefill_rate=820, prefill_ticks=random.randint(1, 3), prefill_real_step=0.4,
        decode_ticks=random.randint(6, 12), decode_real_step=random.uniform(0.4, 0.8),
        queued_wait_s=(0.05, 0.3),
        content_in="repo_ctx", content_out="markdown",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_donor_share(slot_id, donor_slot=0, **extra):
    """Zero-copy seq_cp share from a live donor slot (near-instant, huge)."""
    n_prompt = random.randint(30_000, 80_000)
    tokens = int(n_prompt * random.uniform(0.9, 0.99))
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=0,
        sel="lru", sel_lcp=0,
        restore_src="donor", restore_tokens=tokens, restore_bytes=0, restore_ms=round(random.uniform(5, 25), 2),
        donor_slot=donor_slot,
        admission_outcome="ready_reuse", admission_lcp=tokens,
        admission_resident=tokens, admission_frontier=tokens, admission_span_end=n_prompt + 800,
        n_past=tokens, lcp=tokens,
        decode_tokens=random.randint(400, 2000), decode_rate=random.uniform(15, 28),
        draft_rate=1.3, draft_acc=random.uniform(0.55, 0.8),
        prefill_rate=800, prefill_ticks=random.randint(2, 5), prefill_real_step=0.5,
        decode_ticks=random.randint(8, 14), decode_real_step=random.uniform(0.5, 1.0),
        queued_wait_s=(0.05, 0.3),
        content_in="repo_ctx", content_out="review",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_partial_recompute(slot_id, **extra):
    """Partial restore + recomputed tail: checkpoint_gap AND divergent_tail."""
    n_prompt = random.randint(20_000, 55_000)
    n_past = int(n_prompt * random.uniform(0.55, 0.7))
    lcp = n_past + random.randint(300, 1500)
    b = int(n_past * BYTES_PER_TOKEN)
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=0,
        sel="lcp_affinity", sel_lcp=lcp, sim=round(lcp / n_prompt, 4),
        restore_src="ram", restore_tokens=n_past, restore_bytes=b, restore_ms=round(random.uniform(80, 600), 1),
        admission_outcome="ready_reuse", admission_lcp=lcp,
        admission_resident=lcp, admission_frontier=lcp, admission_span_end=n_prompt,
        n_past=n_past, lcp=lcp, gap_why="checkpoint_gap",
        decode_tokens=random.randint(150, 900), decode_rate=random.uniform(18, 35),
        draft_rate=1.3, draft_acc=random.uniform(0.55, 0.85),
        prefill_rate=850, prefill_ticks=random.randint(2, 5), prefill_real_step=0.4,
        decode_ticks=random.randint(6, 11), decode_real_step=random.uniform(0.4, 0.9),
        queued_wait_s=(0.05, 0.3),
        content_in="agent_deep", content_out="tool",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_full_recompute(slot_id, **extra):
    """Cache miss + a full clear WITH a reason (clear_code), not a cold start."""
    n_prompt = random.randint(8_000, 40_000)
    why = random.choice(["no_lcp", "family", "cache_reuse_opt", "no_cache_prompt", "mtmd"])
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=random.choice([0, 2048]),
        sel="lru", sel_lcp=0,
        restore_src="miss", restore_tokens=0, restore_bytes=0, restore_ms=round(random.uniform(0.05, 0.3), 3),
        admission_outcome="ready_full_clear", admission_why=why,
        admission_resident=random.randint(1000, 9000), admission_lcp=0, admission_span_end=n_prompt + 500,
        n_past=0, lcp=0,
        decode_tokens=random.randint(200, 1500), decode_rate=random.uniform(18, 35),
        draft_rate=random.choice([0, 1.3]), draft_acc=random.uniform(0.55, 0.85),
        prefill_rate=760, prefill_ticks=random.randint(2, 5), prefill_real_step=0.4,
        decode_ticks=random.randint(6, 11), decode_real_step=random.uniform(0.4, 0.9),
        queued_wait_s=(0.05, 0.3),
        content_in="repo_ctx", content_out="markdown",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_wasted_restore(slot_id, **extra):
    """THE key case: a slow multi-GiB restore succeeds, then admission
    clears the slot anyway, discarding every restored token."""
    n_prompt = random.randint(40_000, 100_000)
    tokens = int(n_prompt * random.uniform(0.85, 0.99))
    b = int(tokens * BYTES_PER_TOKEN)
    ms = random.uniform(900, 4300)
    why = random.choice(["no_covering_checkpoint", "rs_window"])
    cfg = dict(
        lane=random.choice(["normal", "low"]), n_prompt=n_prompt, max_out=0,
        sel="lcp_affinity", sel_lcp=tokens, sim=round(tokens / n_prompt, 4),
        restore_src=random.choice(["disk", "ram"]), restore_tokens=tokens, restore_bytes=b, restore_ms=round(ms, 1),
        admission_outcome="ready_full_clear", admission_why=why,
        admission_resident=tokens, admission_lcp=tokens, admission_span_end=n_prompt + 800,
        n_past=0, lcp=0,  # the restore was thrown away — see wastedRestore()
        decode_tokens=random.randint(300, 1800), decode_rate=random.uniform(16, 30),
        draft_rate=1.3, draft_acc=random.uniform(0.6, 0.85),
        prefill_rate=780, prefill_ticks=random.randint(3, 6), prefill_real_step=0.5,
        decode_ticks=random.randint(8, 13), decode_real_step=random.uniform(0.5, 0.9),
        queued_wait_s=(0.05, 0.3),
        content_in="agent_deep", content_out="review",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_stall(slot_id, **extra):
    """Progress stops mid-decode; the run_stall deadline eventually fires."""
    n_prompt = random.randint(30_000, 60_000)
    cached = int(n_prompt * random.uniform(0.6, 0.95))
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=0,
        sel="lru", sel_lcp=0,
        restore_src="ram", restore_tokens=cached, restore_bytes=int(cached * BYTES_PER_TOKEN),
        restore_ms=round(random.uniform(200, 3000), 1),
        admission_outcome="ready_reuse", admission_lcp=cached,
        admission_resident=cached, admission_frontier=cached, admission_span_end=n_prompt,
        n_past=cached, lcp=cached,
        decode_tokens=random.randint(500, 1200), decode_rate=random.uniform(16, 26),
        draft_rate=1.3, draft_acc=random.uniform(0.6, 0.8),
        prefill_rate=800, prefill_ticks=random.randint(2, 4), prefill_real_step=0.4,
        decode_ticks=6, decode_real_step=0.5,
        stall=True, stall_at_tick=random.randint(2, 4),
        queued_wait_s=(0.05, 0.3),
        content_in="repo_ctx", content_out="tool",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_cancelled(slot_id, **extra):
    """A request cancelled partway through decode."""
    n_prompt = random.randint(3000, 20000)
    cached = int(n_prompt * random.uniform(0.5, 0.95))
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=0,
        sel="lcp_affinity", sel_lcp=cached, sim=round(cached / n_prompt, 4),
        restore_src="resident", restore_tokens=cached, restore_bytes=0, restore_ms=round(random.uniform(0.2, 1.0), 3),
        admission_outcome="ready_spanless", admission_lcp=cached,
        admission_resident=cached, admission_frontier=cached, admission_span_end=n_prompt,
        n_past=cached, lcp=cached,
        decode_tokens=1500, decode_rate=random.uniform(20, 35),
        draft_rate=1.3, draft_acc=random.uniform(0.7, 0.9),
        prefill_rate=850, prefill_ticks=2, prefill_real_step=0.4,
        decode_ticks=10, decode_real_step=0.5,
        cancel_after_ticks=random.randint(3, 5),
        queued_wait_s=(0.05, 0.2),
        content_in="agent_deep", content_out="markdown",
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


def tpl_run_timeout_pressure(slot_id=-1, **extra):
    """'503 pressure': queued behind lane capacity, deferred repeatedly,
    never gets a slot, killed by the run_timeout deadline (code 503)."""
    n_prompt = random.randint(4000, 30000)
    cfg = dict(
        lane=random.choice(LANES), n_prompt=n_prompt, max_out=0,
        queue_only=True, defer_count=random.randint(2, 4),
        queue_terminal={"state": "timed_out", "reason": "run_timeout", "code": 503},
        queued_wait_s=(0.05, 0.2),
    )
    cfg.update(extra)
    return run_request(slot_id, **cfg)


TEMPLATES = [
    (18, tpl_trivial), (16, tpl_short_chat), (10, tpl_deep_context), (10, tpl_ssd_restore),
    (8, tpl_donor_share), (10, tpl_partial_recompute), (10, tpl_full_recompute),
    (8, tpl_wasted_restore), (5, tpl_stall), (5, tpl_cancelled),
]

# rough worst-case n_prompt per template, used only to keep the *sum* of
# concurrently-resident tokens across slots plausible against the shared
# n_ctx budget (real llama-server enforces this; the dashboard's KV-context
# tile assumes it — see core.mjs renderTiles()). Not exact, just a guard
# against three big templates landing on slots 1-3 at once.
_TEMPLATE_MAX_PROMPT = {
    tpl_trivial: 60, tpl_short_chat: 16_000, tpl_cancelled: 20_000, tpl_stall: 60_000,
    tpl_full_recompute: 40_000, tpl_partial_recompute: 55_000, tpl_ssd_restore: 60_000,
    tpl_donor_share: 80_000, tpl_wasted_restore: 100_000, tpl_deep_context: 150_000,
}


def pick_template(max_prompt=None):
    candidates = TEMPLATES
    if max_prompt is not None:
        filtered = [(w, fn) for w, fn in TEMPLATES if _TEMPLATE_MAX_PROMPT.get(fn, 0) <= max_prompt]
        candidates = filtered or [(w, fn) for w, fn in TEMPLATES if fn is tpl_trivial]
    total = sum(w for w, _ in candidates)
    r = random.uniform(0, total)
    upto = 0.0
    for w, fn in candidates:
        upto += w
        if r <= upto:
            return fn
    return candidates[-1][1]


def other_slots_context(exclude_slot):
    """Sum of resident context tokens on slots other than `exclude_slot`,
    the same accounting core.mjs's slotContextTokens() uses."""
    with STATE_LOCK:
        total = 0
        for s in SLOTS:
            if s["id"] == exclude_slot or s.get("id_task") is None:
                continue
            nt = s.get("next_token", [{}])
            decoded = nt[0].get("n_decoded", 0) if nt else 0
            total += max(s.get("n_prompt_tokens", 0),
                          s.get("n_prompt_tokens_cache", 0) + s.get("n_prompt_tokens_processed", 0) + decoded)
        return total
    return TEMPLATES[-1][1]


# ---------------------------------------------------------------------------
# Seed: a rich, immediately-visible history at boot (see module docstring for
# why this needs no real waiting even for the 44-minute agent turn).
# ---------------------------------------------------------------------------

def seed_prefill(req_id, slot_id, t0, n_past, n_prompt, rate, ticks=8):
    done = n_past
    t = t0
    remaining = n_prompt - n_past
    per = max(1, remaining // ticks)
    for i in range(ticks):
        batch = (remaining - (done - n_past)) if i == ticks - 1 else per
        done += batch
        t += batch / rate * 1000.0
        BUS.emit("prefill_progress", {"n_done": done, "n_prompt": n_prompt, "n_batch": batch, "ms": round(t - t0, 1)},
                 t_ms=int(t), req=req_id, slot=slot_id)
    return t, done


def seed_decode(req_id, slot_id, t0, n_tokens, rate, ticks, draft_rate=0.0, draft_acc=0.8):
    n = 0
    draft_n = 0
    draft_a = 0
    t = t0
    per = max(1, n_tokens // ticks)
    for i in range(ticks):
        step = (n_tokens - n) if i == ticks - 1 else per
        n += step
        draft_n += round(step * draft_rate)
        draft_a += round(step * draft_rate * draft_acc)
        t += step / rate * 1000.0
        tps = rate * random.uniform(0.85, 1.15)
        BUS.emit("decode_progress", {"n_dec": n, "draft_n": draft_n, "draft_a": draft_a,
                                      "ms": round(t - t0, 1), "tps": round(tps, 2)},
                 t_ms=int(t), req=req_id, slot=slot_id)
    return t, n, draft_n, draft_a


def _seed_preview(rng, tok, req_ok=0.75, preview_ok=0.85):
    """Preview + originating-request link for a seeded cache entry.

    Entries the server "rebuilt from disk" at startup have lost the link to the
    request that saved them, and a few have no retained text at all - both
    states the UI has to render, so both are seeded here. The linked ids come
    from a historical range (1100+) rather than this process's live id
    sequence: those requests are long gone, so following the link lands on the
    'content no longer retained' path, which is correct."""
    req = rng.randint(1100, 1480) if rng.random() < req_ok else None
    if rng.random() >= preview_ok:
        return None, req
    kind = "agent_deep" if tok > 80_000 else ("repo_ctx" if tok > 20_000 else "chat_short")
    return cache_preview_build(tok, kind=kind, seed=rng.randint(0, 9999), req=req), req


def seed_cache_population():
    rng = random.Random(20260808)  # deterministic across restarts
    ram_sizes = [212_300, 148_700, 96_100, 83_400, 61_200, 44_900, 27_300, 15_800, 9_200]
    for tok in ram_sizes:
        preview, req = _seed_preview(rng, tok, req_ok=0.85, preview_ok=0.95)
        cache_add_entry(tok, "ram", created_ago_s=rng.uniform(30, 4200),
                         hits=0 if rng.random() < 0.85 else 1,
                         last_hit_ago_s=rng.uniform(20, 3000) if rng.random() < 0.3 else None,
                         req=req, preview=preview)
    for _ in range(36):
        tok = int(rng.lognormvariate(11.6, 0.7))
        tok = max(8_000, min(320_000, tok))
        hits = rng.choices([0, 1, 2, 3, 4, 5], weights=[30, 25, 20, 15, 7, 3])[0]
        preview, req = _seed_preview(rng, tok, req_ok=0.6, preview_ok=0.8)
        cache_add_entry(tok, "disk", created_ago_s=rng.uniform(600, 200_000),
                         hits=hits, last_hit_ago_s=rng.uniform(60, 90_000) if hits > 0 else None,
                         req=req, preview=preview)
    with STATE_LOCK:
        CACHE["counters"].update({
            "lookups": 4180, "hits_entry": 2540, "hits_resident": 1120, "misses": 520,
            "saves": 2610, "spills": 178, "disk_loads": 341, "drops": 96,
            "bytes_saved": 39_400_000_000, "bytes_spilled": 5_100_000_000, "bytes_disk_load": 9_800_000_000,
        })
        CACHE["updated_at"] = time.time()


def seed_persistent_giant_turn(backdate_ms=44 * 60_000):
    """The 40+ minute agent turn on slot 0: backdated history + still live.

    `backdate_ms` is how much of the turn already happened when it is created:
    44 minutes at boot, a few minutes when the ticker rolls the turn over
    because it has filled its context."""
    now = now_ms()
    req_id = alloc_req_id()
    task_id = alloc_task_id()
    n_prompt = 281_400
    t0 = now - max(4 * 60_000, backdate_ms)

    reg_add(req_id, "normal", "decode", n_prompt, None, start_epoch=t0 / 1000.0)
    reg_bind(req_id, 0)
    BUS.emit("queued", {"lane": "normal", "n_prompt": n_prompt, "max_out": 0}, t_ms=t0, req=req_id)
    BUS.emit("dispatched", {"lane": "normal", "wait_ms": 40}, t_ms=t0 + 350, req=req_id)
    BUS.emit("slot_selected", {"sel": "lcp_affinity", "lcp": 190_000, "n_task": n_prompt, "sim": 0.675},
             t_ms=t0 + 352, req=req_id, slot=0)
    restore_tokens = 212_300
    restore_bytes = restore_tokens * BYTES_PER_TOKEN
    BUS.emit("cache_restore", {"src": "ram", "tokens": restore_tokens, "bytes": restore_bytes,
                                "n_task": n_prompt, "ms": 8123.5}, t_ms=t0 + 8480, req=req_id, slot=0)
    cache_record_lookup("entry")
    BUS.emit("admission", {"outcome": "ready_reuse", "lcp": restore_tokens, "resident": restore_tokens,
                            "frontier": restore_tokens, "span_end": n_prompt + 100}, t_ms=t0 + 8482, req=req_id, slot=0)

    n_past = 210_250
    lcp = 212_300
    t_prompt = t0 + 8550
    BUS.emit("prompt_start", {"n_prompt": n_prompt, "n_past": n_past, "lcp": lcp, "gap_why": "checkpoint_gap"},
             t_ms=t_prompt, req=req_id, slot=0)

    t_after_prefill, done = seed_prefill(req_id, 0, t_prompt, n_past, n_prompt, rate=780, ticks=8)
    reg_state(req_id, "decode")

    decode_span_ms = now - t_after_prefill
    total_dec = max(200, int(decode_span_ms / 1000 * 13.5))
    n_ticks = max(6, min(40, int(decode_span_ms / 90_000) + 6))
    t_end, n_dec, draft_n, draft_a = seed_decode(req_id, 0, t_after_prefill, total_dec, 13.5, n_ticks,
                                                  draft_rate=1.3, draft_acc=0.78)

    # Content: replay the turn's generated text into a live mirror so /watch
    # has something to stream from the instant the server comes up. Cheap -
    # only the first 4 kB and the last 64 kB survive in the mirror.
    live = LiveText()
    stream = make_text_stream("review", seed=req_id * 7919)
    left = n_dec
    while left > 0:
        step = min(400, left)
        live.append(stream.take_tokens(step), step)
        left -= step
    content_begin(req_id, 0, make_input_content("agent_deep", n_prompt, seed=req_id), live, now)

    PERSISTENT_STATE.update({
        "req_id": req_id, "task_id": task_id, "n_prompt": n_prompt, "n_past": n_past,
        "n_dec": n_dec, "draft_n": draft_n, "draft_a": draft_a, "decode_start_ms": t_after_prefill,
        "live": live, "stream": stream, "chunks": 0, "rewind_at": random.randint(12, 20),
        "last_pub_ms": now, "last_pub_dec": n_dec,
    })
    slot_set_active(0, task_id, n_prompt + n_dec, n_prompt - n_past, n_past, n_dec, 0, remain=-1)


def restart_persistent_giant_turn():
    """The turn eventually fills its context. Finish it (which closes its watch
    streams with reason "finished" and freezes its retained content) and open a
    fresh one on the same slot, so slot 0 always has a live generation to
    mirror and the KV-context tile never runs past 100%."""
    st = PERSISTENT_STATE
    old = st.get("req_id")
    if old is None:
        return
    t = now_ms()
    BUS.emit("finished", {"stop": "eos", "n_prompt": st["n_prompt"], "n_cached": st["n_past"],
                          "n_dec": st["n_dec"], "draft_n": st["draft_n"], "draft_a": st["draft_a"],
                          "pp_ms": round((st["n_prompt"] - st["n_past"]) / 780 * 1000.0, 1),
                          "tg_ms": float(max(0, t - st["decode_start_ms"]))},
             t_ms=t, req=old, slot=0)
    BUS.emit("terminal", {"state": "completed", "reason": "completed", "code": 600}, t_ms=t + 1, req=old)
    content_finalize(old, "finished", t_ms=t + 1)
    reg_remove(old)
    seed_persistent_giant_turn(backdate_ms=4 * 60_000)


def seed_history():
    now = now_ms()
    MIN = 60_000
    seed_cache_population()

    drain(tpl_stall(1, start_ms=now - 10 * MIN))
    drain(tpl_cancelled(2, start_ms=now - 9 * MIN))
    drain(tpl_run_timeout_pressure(start_ms=now - int(7.5 * MIN)))
    drain(tpl_wasted_restore(3, start_ms=now - 7 * MIN))
    drain(tpl_full_recompute(1, start_ms=now - int(5.5 * MIN)))
    drain(tpl_deep_context(2, start_ms=now - int(4.5 * MIN)))
    drain(tpl_partial_recompute(3, start_ms=now - int(3.5 * MIN)))
    drain(tpl_donor_share(1, donor_slot=0, start_ms=now - 3 * MIN))
    drain(tpl_ssd_restore(2, start_ms=now - 2 * MIN))
    drain(tpl_short_chat(3, start_ms=now - MIN))
    drain(tpl_trivial(1, start_ms=now - 9_000))
    drain(tpl_trivial(2, start_ms=now - 4_000))

    seed_persistent_giant_turn()


# ---------------------------------------------------------------------------
# Live driver: keeps the demo animating forever after boot.
# ---------------------------------------------------------------------------

class Task:
    __slots__ = ("gen", "resume_at")

    def __init__(self, gen, ready_at=0.0):
        self.gen = gen
        self.resume_at = ready_at

    def step(self, now):
        if now < self.resume_at:
            return True
        try:
            delay = next(self.gen)
        except StopIteration:
            return False
        except Exception as exc:  # keep the driver alive even if one scenario errors
            print(f"[mock-server] scenario error: {exc!r}", file=sys.stderr, flush=True)
            return False
        self.resume_at = now + max(0.0, delay or 0.0)
        return True


def persistent_giant_turn_text_ticker():
    """Extends the seeded 44-minute turn forever, ~1:1 real time.

    This is where n_dec actually advances (the bus ticker below only publishes
    it), so the metadata on /m2-dashboard/events and the text on
    /m2-dashboard/watch can never disagree. ~7-15 tokens every 0.28-0.5 s is
    roughly 20-40 tok/s, fast enough to read as live in the UI."""
    while True:
        yield random.uniform(0.28, 0.5)
        st = PERSISTENT_STATE
        live = st.get("live")
        if live is None:
            continue
        step = random.randint(7, 15)
        live.append(st["stream"].take_tokens(step), step)
        st["n_dec"] += step
        st["draft_n"] += round(step * 1.3)
        st["draft_a"] += round(step * 1.3 * 0.77)

        # Occasionally the sampler trims a partial stop word back off the end
        # of the answer; watchers have to resync rather than append.
        st["chunks"] = st.get("chunks", 0) + 1
        if st["chunks"] >= st.get("rewind_at", 32):
            st["chunks"] = 0
            st["rewind_at"] = random.randint(28, 45)
            back = random.randint(4, 12)
            if live.rewind(round(back * CHARS_PER_TOKEN), back):
                st["n_dec"] = max(0, st["n_dec"] - back)

        slot_set_active(0, st["task_id"], st["n_prompt"] + st["n_dec"], st["n_prompt"] - st["n_past"],
                         st["n_past"], st["n_dec"], 0, remain=-1)
        if st["n_prompt"] + st["n_dec"] > N_CTX - 24_000:
            restart_persistent_giant_turn()


def persistent_giant_turn_ticker():
    """Publishes the slot-0 turn's decode progress on the bus. Metadata only:
    the generated text is reachable through /m2-dashboard/watch, never here."""
    while True:
        yield random.uniform(2.5, 4.5)
        st = PERSISTENT_STATE
        if not st:
            continue
        now = now_ms()
        dt = max(1e-3, (now - st.get("last_pub_ms", now)) / 1000.0)
        d_dec = max(0, st["n_dec"] - st.get("last_pub_dec", st["n_dec"]))
        st["last_pub_ms"] = now
        st["last_pub_dec"] = st["n_dec"]
        tps = (d_dec / dt) if d_dec else random.uniform(20.0, 32.0)
        BUS.emit("decode_progress", {
            "n_dec": st["n_dec"], "draft_n": st["draft_n"], "draft_a": st["draft_a"],
            "ms": round(now - st["decode_start_ms"], 1), "tps": round(tps, 2),
        }, req=st["req_id"], slot=0)


def cache_publish_ticker():
    while True:
        yield random.uniform(1.5, 3.0)
        with STATE_LOCK:
            CACHE["updated_at"] = time.time()


def cache_jitter_loop():
    while True:
        yield random.uniform(15, 35)
        with STATE_LOCK:
            entries = list(CACHE["entries"].values())
            ram_used = sum(e["bytes"] for e in entries if e["tier"] == "ram")
            over_budget = ram_used > CACHE["limit_ram_bytes"]
        roll = 0.1 if over_budget else random.random()  # bias toward spilling under pressure
        if roll < 0.4:
            ram = [e for e in entries if e["tier"] == "ram"]
            if ram:
                e = random.choice(ram)
                with STATE_LOCK:
                    e2 = CACHE["entries"].get(e["id"])
                    if e2:
                        e2["tier"] = "disk"
                        e2["file"] = f"pc-mockA1-{e2['id']}.lcpc"
                cache_record_op("spill", e["tokens"], e["bytes"])
                BUS.emit("cache_op", {"op": "spill", "tokens": e["tokens"], "bytes": e["bytes"],
                                       "ms": round(random.uniform(500, 3800), 1)})
        elif roll < 0.7:
            disk = [e for e in entries if e["tier"] == "disk"]
            if disk:
                e = random.choice(disk)
                with STATE_LOCK:
                    e2 = CACHE["entries"].get(e["id"])
                    if e2:
                        e2["tier"] = "ram"
                        e2["file"] = None
                        e2["last_hit_at"] = time.time()
                        e2["hits"] += 1
                cache_record_op("disk_load", e["tokens"], e["bytes"])
                BUS.emit("cache_op", {"op": "disk_load", "tokens": e["tokens"], "bytes": e["bytes"],
                                       "ms": round(random.uniform(300, 3200), 1)})
        elif entries:
            e = random.choice(entries)
            cache_remove_entry(e["id"])
            cache_record_op("drop", e["tokens"], e["bytes"])
            BUS.emit("cache_op", {"op": "drop", "tokens": e["tokens"], "bytes": e["bytes"], "ms": 0.0})
            tok = random.randint(8000, 60000)
            # roughly half of the churned-in entries carry no retained text, so
            # the Cache view always has some `preview: false` rows to render.
            preview = (cache_preview_build(tok, kind="repo_ctx", seed=random.randint(0, 9999))
                       if random.random() < 0.55 else None)
            cache_add_entry(tok, random.choice(["ram", "disk"]), preview=preview)


def slot_cycle(slot_id):
    """Runs one template, then a short idle pause, forever. Leaves headroom
    against N_CTX so the KV-context tile never has to show over 100%."""
    while True:
        headroom = max(0, N_CTX - other_slots_context(slot_id) - 20_000)
        fn = pick_template(max_prompt=headroom)
        yield from fn(slot_id)
        yield random.uniform(0.4, 2.2)


def driver_loop():
    tasks = {
        "persistent": Task(persistent_giant_turn_ticker()),
        "persistent_text": Task(persistent_giant_turn_text_ticker()),
        1: Task(slot_cycle(1)),
        2: Task(slot_cycle(2)),
        3: Task(slot_cycle(3)),
        "pressure": Task(slot_cycle_pressure()),
        "cache_jitter": Task(cache_jitter_loop()),
        "cache_publish": Task(cache_publish_ticker()),
    }
    while True:
        now = time.time()
        for key, task in list(tasks.items()):
            if not task.step(now):
                # a slot_cycle()/pressure loop never raises StopIteration in
                # practice (infinite while True), but restart defensively
                if key == "persistent":
                    tasks[key] = Task(persistent_giant_turn_ticker())
                elif key == "persistent_text":
                    tasks[key] = Task(persistent_giant_turn_text_ticker())
                elif key == "pressure":
                    tasks[key] = Task(slot_cycle_pressure())
                elif key == "cache_jitter":
                    tasks[key] = Task(cache_jitter_loop())
                elif key == "cache_publish":
                    tasks[key] = Task(cache_publish_ticker())
                else:
                    tasks[key] = Task(slot_cycle(key))
        time.sleep(0.15)


def slot_cycle_pressure():
    while True:
        yield from tpl_run_timeout_pressure()
        yield random.uniform(3.0, 9.0)


# ---------------------------------------------------------------------------
# Static assets + JSON endpoints + SSE
# ---------------------------------------------------------------------------

PROPS = {}


def load_props():
    global PROPS
    try:
        PROPS = json.loads((ROOT / "fixtures" / "props.json").read_text())
    except OSError:
        PROPS = {
            "total_slots": 4, "model_alias": "dsv4-flash-0731-full.m2 (mock)",
            "build_info": "mock", "endpoint_slots": True, "endpoint_props": False,
            "endpoint_metrics": False, "is_sleeping": False,
            "default_generation_settings": {"n_ctx": N_CTX},
        }


WATCH_LOCK = threading.Lock()
WATCH_ACTIVE = 0


class Handler(BaseHTTPRequestHandler):
    server_version = "M2DashboardMock/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_static(self, path):
        rel = path.lstrip("/").split("?", 1)[0]
        if rel == "":
            rel = "index.html"
        full = (ROOT / rel).resolve()
        try:
            full.relative_to(ROOT.resolve())
        except ValueError:
            self.send_error(403)
            return
        if full.is_dir():
            full = full / "index.html"
        if not full.is_file():
            self.send_error(404)
            return
        ext = full.suffix.lower()
        ctype = MIME_MAP.get(ext) or mimetypes.guess_type(str(full))[0] or "application/octet-stream"
        data = full.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- v3 content endpoints ------------------------------------------
    def _int_param(self, query, name):
        vals = parse_qs(query).get(name) or []
        if not vals:
            return None
        try:
            return int(vals[0])
        except ValueError:
            return None

    def _send_error_obj(self, code, message, etype):
        self._send_json({"error": {"code": code, "message": message, "type": etype}}, status=code)

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _sse_write(self, obj, event=None):
        prefix = ("event: %s\n" % event) if event else ""
        self.wfile.write((prefix + "data: " + json.dumps(obj) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _handle_content(self, query):
        req = self._int_param(query, "req")
        if req is None:
            self._send_error_obj(400, "missing or non-numeric 'req' query parameter",
                                 "invalid_request_error")
            return
        self._send_json(content_json(req))

    def _handle_cache_preview(self, query):
        eid = self._int_param(query, "id")
        if eid is None:
            self._send_error_obj(400, "missing or non-numeric 'id' query parameter",
                                 "invalid_request_error")
            return
        self._send_json(cache_preview_json(eid))

    def _handle_watch(self, query):
        global WATCH_ACTIVE
        req = self._int_param(query, "req")
        if req is None:
            self._send_error_obj(400, "missing or non-numeric 'req' query parameter",
                                 "invalid_request_error")
            return
        with WATCH_LOCK:
            if WATCH_ACTIVE >= WATCH_CAPS["max_streams"]:
                self._send_error_obj(429, "too many dashboard watch streams", "unavailable_error")
                return
            WATCH_ACTIVE += 1
        try:
            self._watch_stream(req)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with WATCH_LOCK:
                WATCH_ACTIVE -= 1

    def _watch_stream(self, req):
        rec, _reason = content_lookup(req)
        live = rec.get("live") if rec else None
        inp = rec["input"] if rec else None
        slot = rec["slot"] if rec else None

        if live is not None:
            state = "live"
            with live.cond:
                cursor = live.n_bytes
                n_dec = live.n_tokens
                text = bytes(live.mirror).decode("utf-8", "ignore")
                dropped = live.base
                rseq = live.rewind_seq
        elif rec is not None:
            state = "final"
            out = rec.get("output") or {}
            cursor = out.get("n_bytes", 0)
            n_dec = out.get("n_tokens", 0)
            text = out.get("head", "")
            # bytes of the generation this frame does not carry; for a finished
            # request the missing bytes are in the middle, not at the front.
            dropped = max(0, cursor - len(text.encode("utf-8")))
            rseq = 0
        else:
            state, cursor, n_dec, text, dropped, rseq = "absent", 0, 0, "", 0, 0

        self._sse_open()
        self._sse_write({"v": 1, "k": "watch-hello", "req": req, "slot": slot, "state": state,
                         "n_dec": n_dec, "cursor": cursor, "text": text, "dropped": dropped,
                         "input": inp, "caps": dict(WATCH_CAPS)}, event="watch-hello")

        if live is None:
            self._sse_write({"v": 1, "k": "end", "req": req, "reason": "gone", "n_dec": n_dec},
                            event="watch-end")
            return

        pos = cursor
        last_write = time.time()
        while True:
            frame = None
            with live.cond:
                live.cond.wait(0.4)
                closed, end_reason = live.closed, live.end_reason
                n_bytes, base, n_dec = live.n_bytes, live.base, live.n_tokens
                if live.rewind_seq != rseq or pos < base:
                    # the text moved backwards (stop-word trim) or the rolling
                    # mirror dropped bytes we had not sent: resync wholesale.
                    rseq = live.rewind_seq
                    frame = {"v": 1, "k": "rewind", "req": req, "cursor": base,
                             "text": bytes(live.mirror).decode("utf-8", "ignore"), "n_dec": n_dec}
                    pos = n_bytes
                elif n_bytes > pos:
                    frame = {"v": 1, "k": "delta", "req": req, "cursor": n_bytes,
                             "text": bytes(live.mirror[pos - base:]).decode("utf-8", "ignore"),
                             "n_dec": n_dec}
                    pos = n_bytes
            if frame is not None:
                self._sse_write(frame)
                last_write = time.time()
            if closed:
                self._sse_write({"v": 1, "k": "end", "req": req,
                                 "reason": end_reason or "gone", "n_dec": n_dec},
                                event="watch-end")
                return
            if time.time() - last_write >= 5.0:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                last_write = time.time()

    def _handle_events(self, query):
        qs = parse_qs(query)
        after = 0
        lei = self.headers.get("Last-Event-ID")
        if lei is not None:
            try:
                after = int(lei)
            except ValueError:
                after = 0
        elif "after" in qs:
            try:
                after = int(qs["after"][0])
            except ValueError:
                after = 0

        q = BUS.subscribe()
        try:
            ring, dropped, next_seq = BUS.snapshot()
            first_seq = ring[0]["seq"] if ring else next_seq
            if after >= next_seq:
                after = first_seq - 1
            begin = max(after + 1, first_seq)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            hello = {"v": 1, "k": "hello", "first_seq": first_seq, "next_seq": next_seq, "dropped": dropped,
                     "now_ms": now_ms(), "wall_ms": now_ms()}
            self.wfile.write(("event: hello\ndata: " + json.dumps(hello) + "\n\n").encode())

            last_sent = begin - 1
            for ev in ring:
                if ev["seq"] >= begin:
                    self.wfile.write(("id: %d\ndata: %s\n\n" % (ev["seq"], json.dumps(ev))).encode())
                    last_sent = ev["seq"]
            self.wfile.flush()

            last_beat = time.time()
            while True:
                try:
                    ev = q.get(timeout=1.0)
                    if ev["seq"] > last_sent:
                        self.wfile.write(("id: %d\ndata: %s\n\n" % (ev["seq"], json.dumps(ev))).encode())
                        last_sent = ev["seq"]
                        self.wfile.flush()
                except queue.Empty:
                    pass
                now = time.time()
                if now - last_beat >= 5.0:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_beat = now
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            BUS.unsubscribe(q)

    def do_GET(self):
        parts = urlsplit(self.path)
        path = parts.path
        if path == "/m2-dashboard/events":
            self._handle_events(parts.query)
            return
        if path == "/m2-dashboard/cache-state":
            self._send_json(cache_state_json())
            return
        if path == "/m2-dashboard/content":
            self._handle_content(parts.query)
            return
        if path == "/m2-dashboard/watch":
            self._handle_watch(parts.query)
            return
        if path == "/m2-dashboard/cache-preview":
            self._handle_cache_preview(parts.query)
            return
        if path == "/slots":
            self._send_json(slots_json())
            return
        if path == "/props":
            self._send_json(PROPS)
            return
        if path == "/internal/admin/dashboard/snapshot":
            self._send_json(registry_json())
            return
        if path == "/metrics":
            self.send_error(404)
            return
        self._send_static(path)

    def do_POST(self):
        parts = urlsplit(self.path)
        path = parts.path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        if path == "/internal/admin/dashboard/request-detail":
            try:
                rid = int(body.get("request_id"))
            except (TypeError, ValueError):
                rid = None
            with STATE_LOCK:
                slot = BINDINGS.get(rid)
            if slot is None:
                self._send_json({"error": "not found"}, status=404)
            else:
                self._send_json({"registry": {"bindings": [{"slot_id": slot}]}})
            return

        if path == "/internal/admin/dashboard/request-control":
            try:
                rid = int(body.get("request_id"))
            except (TypeError, ValueError):
                rid = None
            if body.get("action") == "cancel" and rid is not None:
                with STATE_LOCK:
                    CANCEL_REQUESTED.add(rid)
            self._send_json({"status": "accepted"})
            return

        self.send_error(404)


class Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8877)
    args = ap.parse_args()

    load_props()
    print("[mock-server] seeding synthetic history...", flush=True)
    seed_history()
    print("[mock-server] seeded. starting live scenario driver...", flush=True)
    threading.Thread(target=driver_loop, daemon=True, name="scenario-driver").start()

    httpd = Server((args.host, args.port), Handler)
    print(f"[mock-server] serving {ROOT} on http://{args.host}:{args.port}/index.html", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

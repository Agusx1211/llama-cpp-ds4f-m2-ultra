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
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parent.parent  # tools/m2-dashboard/
N_CTX = 524288
LANES = ("fast", "normal", "low")
BYTES_PER_TOKEN = 9700  # DSV4-scale KV state size/token, mid-point of real captures

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


def cache_add_entry(tokens, tier, created_ago_s=0.0, hits=0, last_hit_ago_s=None):
    eid = alloc_entry_id()
    b = int(tokens * BYTES_PER_TOKEN * random.uniform(0.9, 1.1))
    entry = {
        "id": eid, "tokens": tokens, "tier": tier, "bytes": b,
        "created_at": time.time() - created_ago_s,
        "last_hit_at": (time.time() - last_hit_ago_s) if last_hit_ago_s is not None else None,
        "hits": hits,
        "file": (f"pc-mockA1-{eid}.lcpc" if tier == "disk" else None),
    }
    with STATE_LOCK:
        CACHE["entries"][eid] = entry
    if tier == "ram":
        cache_enforce_ram_budget(protect_id=eid)
    return eid


def cache_remove_entry(eid):
    with STATE_LOCK:
        CACHE["entries"].pop(eid, None)


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
        for _ in range(cfg.get("defer_count", 2)):
            step_s = random.uniform(0.6, 1.6)
            yield step_s
            t += step_s * 1000.0
            if is_cancelled(req_id):
                BUS.emit("terminal", {"state": "cancelled", "reason": "client_cancel", "code": 500}, t_ms=int(t), req=req_id)
                reg_remove(req_id)
                return
            BUS.emit("deferred", {"why": "capacity"}, t_ms=int(t), req=req_id)
        step_s = random.uniform(0.6, 1.2)
        yield step_s
        t += step_s * 1000.0
        qt = cfg.get("queue_terminal", {"state": "timed_out", "reason": "run_timeout", "code": 503})
        BUS.emit("terminal", qt, t_ms=int(t), req=req_id)
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
            reg_remove(req_id)
            slot_set_idle(slot_id)
            return
        if stall_at is not None and i == stall_at:
            yield random.uniform(6.0, 10.0)
            t += random.uniform(4.5, 6.0) * 60_000.0  # reported stall gap before the deadline fires
            BUS.emit("terminal", {"state": "timed_out", "reason": "run_stall", "code": 504}, t_ms=int(t), req=req_id)
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
    reg_remove(req_id)

    if cfg.get("post_save", True) and (n_past + n) > 200:
        tot = n_past + n
        b = int(tot * BYTES_PER_TOKEN * random.uniform(0.9, 1.1))
        cache_record_op("save", tot, b)
        BUS.emit("cache_op", {"op": "save", "tokens": tot, "bytes": b, "ms": 0.0}, t_ms=int(t) + 2)
        cache_add_entry(tot, "ram")

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
        decode_tokens=random.randint(600, 3000), decode_rate=random.uniform(15, 30),
        draft_rate=1.3, draft_acc=random.uniform(0.5, 0.85),
        prefill_rate=780, prefill_ticks=random.randint(4, 8), prefill_real_step=random.uniform(0.5, 1.0),
        decode_ticks=random.randint(10, 16), decode_real_step=random.uniform(0.6, 1.1),
        queued_wait_s=(0.05, 0.4),
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


def seed_cache_population():
    rng = random.Random(20260808)  # deterministic across restarts
    ram_sizes = [212_300, 148_700, 96_100, 83_400, 61_200, 44_900, 27_300, 15_800, 9_200]
    for tok in ram_sizes:
        cache_add_entry(tok, "ram", created_ago_s=rng.uniform(30, 4200),
                         hits=0 if rng.random() < 0.85 else 1,
                         last_hit_ago_s=rng.uniform(20, 3000) if rng.random() < 0.3 else None)
    for _ in range(36):
        tok = int(rng.lognormvariate(11.6, 0.7))
        tok = max(8_000, min(320_000, tok))
        hits = rng.choices([0, 1, 2, 3, 4, 5], weights=[30, 25, 20, 15, 7, 3])[0]
        cache_add_entry(tok, "disk", created_ago_s=rng.uniform(600, 200_000),
                         hits=hits, last_hit_ago_s=rng.uniform(60, 90_000) if hits > 0 else None)
    with STATE_LOCK:
        CACHE["counters"].update({
            "lookups": 4180, "hits_entry": 2540, "hits_resident": 1120, "misses": 520,
            "saves": 2610, "spills": 178, "disk_loads": 341, "drops": 96,
            "bytes_saved": 39_400_000_000, "bytes_spilled": 5_100_000_000, "bytes_disk_load": 9_800_000_000,
        })
        CACHE["updated_at"] = time.time()


def seed_persistent_giant_turn():
    """The 40+ minute agent turn on slot 0: backdated history + still live."""
    now = now_ms()
    req_id = alloc_req_id()
    task_id = alloc_task_id()
    n_prompt = 281_400
    t0 = now - 44 * 60_000

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

    PERSISTENT_STATE.update({
        "req_id": req_id, "task_id": task_id, "n_prompt": n_prompt, "n_past": n_past,
        "n_dec": n_dec, "draft_n": draft_n, "draft_a": draft_a, "decode_start_ms": t_after_prefill,
    })
    slot_set_active(0, task_id, n_prompt + n_dec, n_prompt - n_past, n_past, n_dec, 0, remain=-1)


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


def persistent_giant_turn_ticker():
    """Extends the seeded 44-minute turn forever, ~1:1 real time."""
    while True:
        yield random.uniform(2.5, 4.5)
        st = PERSISTENT_STATE
        if not st:
            continue
        step = random.randint(28, 46)
        st["n_dec"] += step
        st["draft_n"] += round(step * 1.3)
        st["draft_a"] += round(step * 1.3 * 0.77)
        tps = random.uniform(12.5, 15.5)
        BUS.emit("decode_progress", {
            "n_dec": st["n_dec"], "draft_n": st["draft_n"], "draft_a": st["draft_a"],
            "ms": round(now_ms() - st["decode_start_ms"], 1), "tps": round(tps, 2),
        }, req=st["req_id"], slot=0)
        slot_set_active(0, st["task_id"], st["n_prompt"] + st["n_dec"], st["n_prompt"] - st["n_past"],
                         st["n_past"], st["n_dec"], 0, remain=-1)


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
            cache_add_entry(random.randint(8000, 60000), random.choice(["ram", "disk"]))


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

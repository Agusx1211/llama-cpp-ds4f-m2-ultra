# Mock dashboard server

A persistent, synthetic-data instance of this dashboard for iterating on the
design without depending on the real m2 server (which benchmark agents
frequently take down). Stdlib Python only — no dependencies, no build step.
It serves the dashboard's static files AND emulates `/m2-dashboard/events`
(SSE), `/m2-dashboard/cache-state`, `/m2-dashboard/content`,
`/m2-dashboard/watch` (SSE), `/m2-dashboard/cache-preview`, `/slots`,
`/props`, and the admin registry/cancel routes against a synthetic scenario,
so the page runs exactly like it does against a real server — just with
fabricated numbers and fabricated text.

## Open it

```
http://<this-host-LAN-IP>:8877/index.html
http://<this-host-tailscale-IP>:8877/index.html
```

No `#key=` needed — the mock doesn't enforce the API-key middleware. Find the
current IPs with `hostname -I` (LAN) and `tailscale ip -4` on this machine.

## Start / stop / restart

The server is a single script: `mock/mock_server.py`. It binds `0.0.0.0:8877`
by default and reads the dashboard's static files straight out of this
worktree (`tools/m2-dashboard/`), the same hot-reload-from-worktree model the
real trust-lan deploy uses — edit `dashboard.js`/`styles.css`/`lib/*.mjs` and
just reload the browser tab, no restart needed for those.

```sh
cd tools/m2-dashboard/mock

# start (nohup, survives the shell/session exiting)
nohup python3 mock_server.py --host 0.0.0.0 --port 8877 > mock-server.log 2>&1 &
disown
echo $! > mock-server.pid

# stop
kill "$(cat mock-server.pid)"

# restart = stop, then start again (re-seeds a fresh scenario on boot)
# status / is it up
ss -tlnp | grep 8877          # or: curl -s localhost:8877/props | head -c80
```

Log: `tools/m2-dashboard/mock/mock-server.log` (stdout+stderr; scenario errors,
if any, print `[mock-server] scenario error: ...` there and that one
request-slot restarts on its own — the process doesn't die).

`mock-server.log` and `mock-server.pid` are gitignored — runtime artifacts,
not part of the change.

## Editing the synthetic scenario

Everything lives in `mock/mock_server.py`, no external fixture files:

- **`TEMPLATES`** (near the middle of the file) — the weighted list of
  request-shape generators picked for slots 1–3 forever. Each entry is
  `(weight, tpl_fn)`; raise/lower a weight to see more/less of a given case,
  or add a new `tpl_*` function following the existing ones as a template.
- **`tpl_*` functions** — one per visual state (trivial 30 ms hit, short chat,
  100k+-token deep context, SSD-tier restore, zero-copy donor share, partial
  restore + recomputed tail, full recompute with a `clear_code` reason, the
  **wasted-restore** case, a stalled/`run_stall`-killed request, a cancelled
  request, and the `run_timeout`/503-pressure queue-only case). Each just
  builds a `cfg` dict and hands it to `run_request()` — tune token counts,
  rates (`decode_rate` 15–45 t/s, `draft_acc` 0.4–0.95), lanes, timings, or
  the `content_in`/`content_out` corpus kinds directly in these dicts.
- **`run_request()`** — the one reusable lifecycle generator every template
  configures. Real `yield <seconds>` calls only pace how fast the demo
  *visibly* animates; the event timestamps come from an explicit logical
  clock (`t`) advanced by the computed token/rate deltas, so you can make a
  request "take" any reported duration without the demo actually waiting that
  long (see the module docstring for how this makes both the 30 ms and the
  44-minute cases work in the same run without ever sleeping 44 minutes).
- **`seed_history()`** — runs once at boot, drains every scenario with a
  *backdated* `start_ms` so the page has a rich, immediately-visible history
  (no blank Timeline on first load). `seed_persistent_giant_turn()` is the
  slot-0 44-minute agent turn specifically. Two tickers extend it forever
  afterward, ~1:1 real time: `persistent_giant_turn_text_ticker()` owns
  `n_dec` and appends 7–15 tokens of text every 0.28–0.5 s (~20–40 tok/s,
  what `/watch` streams), and `persistent_giant_turn_ticker()` only
  *publishes* that counter to the bus every 2.5–4.5 s — so the metadata and
  the text can't drift apart. When the turn fills its context the text ticker
  calls `restart_persistent_giant_turn()`, which completes it and opens a
  fresh one on the same slot.
- **`seed_cache_population()`** — the RAM/SSD prompt-cache tier entries and
  cumulative counters shown on first load, each with a text preview and a
  `req` link via `_seed_preview()`; `cache_jitter_loop()` perturbs them
  (spill/load/drop) every 15–35 s so the Cache view keeps changing.
- **`BYTES_PER_TOKEN`**, **`N_CTX`**, **`LANES`** — top-of-file constants if
  you want to change the KV-size assumption, context budget, or lane names.

After editing, just restart the process (re-seed happens on every boot).

## Request content (v3 endpoints)

Three endpoints expose request *text*. **Content is only ever reachable
through these three.** It is never attached to a `/m2-dashboard/events`
frame, a `/slots` row or the registry snapshot — the bus is metadata-only by
construction in the real server, and the mock keeps that invariant so the
dashboard can't accidentally start depending on content arriving there.

### `GET /m2-dashboard/content?req=<id>`

Head/tail view of one request's prompt and finished generation, plus the
store's caps and occupancy. `output` is `null` while `state == "live"` (use
`/watch` for an in-flight generation) and `input` is `null` for requests that
never reached prefill (the queued/`run_timeout` case). Not retained →
`200 {"found": false, "reason": "absent"|"evicted", ...}`; missing or
non-numeric `req` → `400` with an `{"error": {...}}` body.

The store holds **64 requests / 1 MiB**, FIFO, oldest first, counting into
`store.evicted`. In practice the request cap always binds first (a record is
at most ~9 kB). Live requests are skipped by the evictor — dropping the
mirror of something still generating would break `/watch` — so the slot-0
giant turn survives forever despite being the oldest record. Under the live
driver the store fills and starts evicting within about two minutes of boot,
which is what makes the UI's "content no longer retained" path reachable.

### `GET /m2-dashboard/watch?req=<id>` (SSE)

Live mirror of one in-flight generation: `event: watch-hello` (state, the
`input` block, the text so far, `cursor`, `dropped`, caps), then unnamed
`delta` frames as text arrives, then `event: watch-end`. At most **2**
concurrent streams; a 3rd gets `429`. A finished or unknown request still
gets a `watch-hello` (`state: "final"` / `"absent"`, carrying the retained
head) followed immediately by `watch-end` with `reason: "gone"`.

`rewind` frames are emitted when the generated text is trimmed backwards
(stop-word trim) or when the 64 KiB rolling mirror dropped bytes a slow
client hadn't received yet. The slot-0 turn does this every 28–45 chunks, so
watching it for ~20 s reliably exercises the path.

**Finding a watchable request id.** The persistent slot-0 agent turn is the
one that generates forever. It's the registry row bound to slot 0, and the
only one with a ~281k-token prompt:

```sh
curl -s localhost:8877/internal/admin/dashboard/snapshot \
  | python3 -c 'import json,sys; print(max(json.load(sys.stdin)["requests"], key=lambda r: r["prompt_tokens"])["id"])'
# -> e.g. 13, then:
curl -sN "localhost:8877/m2-dashboard/watch?req=13"
```

Its id is **not** stable: `restart_persistent_giant_turn()` rolls the turn
over with a fresh id once it has filled its context (roughly every 2.5 h at
~25 tok/s), so re-read it rather than hardcoding it.

### `GET /m2-dashboard/cache-preview?id=<entry_id>`

Head/tail text of one prompt-cache entry (64/32 tokens), with `req` — the
request whose save created it — omitted when that link is unknown. Unknown
id or an entry with no retained text → `200 {"found": false}`; bad `id` →
`400`. `entries[]` in `/m2-dashboard/cache-state` gained two additive fields
to drive this: `req` (omitted when unknown) and `preview` (bool).

Seeded entries link to *historical* request ids (1100+) rather than this
process's live id sequence — those requests are long gone, so following the
link correctly lands on the "not retained" path. Entries saved by requests
that actually ran in this process link to their real id and resolve.

### Editing the corpus

All of it lives in `mock/mock_server.py`, no fixture files:

- **`AGENT_SYSTEM_PARAS` / `AGENT_TAIL_PARAS` / `CHAT_SHORT_PARAS` /
  `REPO_CTX_PARAS`** — input corpora, keyed by `INPUT_CORPUS` into the kinds
  `agent_deep`, `repo_ctx`, `chat_short`. A template picks one with
  `content_in=`.
- **`REVIEW_PARAS` / `MARKDOWN_PARAS` / `SHORT_OUT_PARAS` / `TOOL_PARAS`** —
  output corpora (`OUTPUT_CORPUS`, kinds `review`, `markdown`, `short`,
  `tool`), picked with `content_out=`. `markdown` is the newline/backtick/
  long-line-heavy one for exercising the renderer.
- **`TextStream`** cycles a corpus in a reshuffled order so a 40-minute turn
  keeps producing new prose instead of looping one sentence; **`LiveText`**
  is the bounded per-request mirror (first 4 kB + last 64 kB + counters).
- **`CONTENT_CAPS` / `WATCH_CAPS` / `CACHE_PREVIEW_CAPS` / `CHARS_PER_TOKEN`**
  — top-of-file constants. The mock has no tokenizer, so `CHARS_PER_TOKEN`
  (4.1) is the *only* token↔text conversion in the content layer, and byte
  accounting approximates the real `4*len(head_ids) + 4*len(tail_ids)` from
  the head/tail token counts.

## What's covered

4 slots simultaneously busy across all three lanes with mixed phases
(prefill in progress / generating / idle-resident) plus a queued request
behind lane capacity; a 30 ms trivial full-cache-hit request and a
persistently-live 44+ minute agent turn in the same view; every prompt
provenance the schema supports (resident/zero-copy reuse, RAM-tier restore,
SSD-tier restore, donor share, partial restore + recomputed tail, full
recompute with a `clear_code` reason); the wasted-restore case (a multi-GiB
restore immediately discarded by a full clear); draft acceptance 0.4–0.95;
decode rates 15–45 t/s; prompts from a few tokens to 280k+; RAM cache tier
hovering at/over its 8 GiB limit with self-correcting spills, SSD tier with
~36 entries; and three terminal/error states (`run_stall` kill, `run_timeout`
"503 pressure" rejection that never got a slot, and a cancelled request).

On the content side: a 281k-token deep-context agent turn whose `input.head`
is a system prompt + tool definitions and whose `elided_tokens` is in the
hundreds of thousands; short chat turns retained whole (`truncated: false`,
empty `tail`); markdown/code-heavy generations; generations that overran the
4 kB+2 kB output cap (`truncated: true`, non-zero `elided_bytes`); requests
retained with `input: null` because they never reached prefill; finished
requests evicted from the store; and cache entries with and without preview
text and with and without a `req` link.

## Known deliberate mock liberties

- No API-key enforcement (simpler to reach from a browser; the real
  middleware behavior isn't part of what this mock is for).
- `/metrics` isn't implemented (404) — this is the *real* server's degraded
  state too when started without `--metrics`, so the footer's fallback
  message is exercised, not faked.
- Registry `id` and the bus `req` field are the same integer (as a string),
  rather than prod's separate `task.id:epoch` scheme — the UI treats both as
  opaque, so this doesn't change any rendering.
- No tokenizer: every token↔text conversion in the content layer is the
  single `CHARS_PER_TOKEN = 4.1` approximation, and content-store byte
  accounting approximates `4*len(in_head_ids) + 4*len(in_tail_ids)` from the
  head/tail token counts.
- `rewind.cursor` is the absolute offset at which the frame's `text` begins
  (0 = "replace everything you have"), not a byte total — the frame carries
  the authoritative text from that offset onward, which is what makes it
  usable after a backwards trim. `delta.cursor` stays the monotone count of
  bytes ever produced.
- `watch-hello.dropped` counts generated bytes the frame does not carry. For
  a `live` stream those are bytes that fell off the front of the 64 KiB
  rolling mirror; for a `final` one they're the middle of the generation,
  since the hello carries only the retained head.

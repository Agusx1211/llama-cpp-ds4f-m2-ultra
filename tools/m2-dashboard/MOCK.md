# Mock dashboard server

A persistent, synthetic-data instance of this dashboard for iterating on the
design without depending on the real m2 server (which benchmark agents
frequently take down). Stdlib Python only — no dependencies, no build step.
It serves the dashboard's static files AND emulates `/m2-dashboard/events`
(SSE), `/m2-dashboard/cache-state`, `/slots`, `/props`, and the admin
registry/cancel routes against a synthetic scenario, so the page runs exactly
like it does against a real server — just with fabricated numbers.

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
  rates (`decode_rate` 15–45 t/s, `draft_acc` 0.4–0.95), lanes, or timings
  directly in these dicts.
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
  slot-0 44-minute agent turn specifically; `persistent_giant_turn_ticker()`
  in the driver loop keeps extending it forever afterward, ~1:1 real time.
- **`seed_cache_population()`** — the RAM/SSD prompt-cache tier entries and
  cumulative counters shown on first load; `cache_jitter_loop()` perturbs
  them (spill/load/drop) every 15–35 s so the Cache view keeps changing.
- **`BYTES_PER_TOKEN`**, **`N_CTX`**, **`LANES`** — top-of-file constants if
  you want to change the KV-size assumption, context budget, or lane names.

After editing, just restart the process (re-seed happens on every boot).

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

## Known deliberate mock liberties

- No API-key enforcement (simpler to reach from a browser; the real
  middleware behavior isn't part of what this mock is for).
- `/metrics` isn't implemented (404) — this is the *real* server's degraded
  state too when started without `--metrics`, so the footer's fallback
  message is exercised, not faked.
- Registry `id` and the bus `req` field are the same integer (as a string),
  rather than prod's separate `task.id:epoch` scheme — the UI treats both as
  opaque, so this doesn't change any rendering.

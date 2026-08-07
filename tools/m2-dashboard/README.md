# M2 llama-server operations dashboard

A static, dependency-free page an operator keeps open on a second monitor while
the M2 Ultra serves DeepSeek V4 Flash. Dark-native, flat, numbers first.

On the trusted-LAN deployment the server serves this directory at
`/m2-dashboard` (`LLAMA_SERVER_TRUST_LAN=1` +
`LLAMA_SERVER_DASHBOARD_DIR=.../tools/m2-dashboard`); files hot-reload per
request, so deploying an update is a git merge on the serving worktree.

## What it shows, and where the data comes from

| View | Source | Notes |
|---|---|---|
| Generation / prompt tok/s + 4-min sparklines | `GET /slots`, counter deltas between polls | derived client-side; the server exposes no live per-slot rates |
| Per-slot rows: phase, live tok/s, progress, context tokens, cached tokens | `GET /slots` | phase = `is_processing` × `n_decoded`; context = resident tokens incl. generated |
| Lane attribution, request ids, age, queue rows | `GET /internal/admin/dashboard/snapshot` | registry per-request `state`/`output_tokens` lag live decode by design; `/slots` is the live truth |
| slot ↔ request join | `POST .../request-detail` (bindings) | one call per new request, cached; degrades silently if denied |
| Cancel button | `POST .../request-control` `{action:"cancel"}` | live mode only, with confirm |
| KV context meter | `/slots` `n_ctx` (shared unified budget) vs summed resident tokens | |
| Prefix-cache reuse | `/slots` `n_prompt_tokens_cache` | registry `cache_hit_tokens` stays 0 on prod; do not use it |
| Recent-requests tail | assembled client-side from `/slots` task transitions | only requests observed while the page is open; `+` marks a partially observed run |
| Model / build header | `GET /props` once | |
| KV-pressure counters (footer) | `GET /metrics` when enabled | prod currently runs without `--metrics`; the page degrades to a hint |

The page polls `/slots` + the registry snapshot every 2 s (15 s while the tab
is hidden), backs off exponentially to 30 s on failure, and holds the last
render dimmed instead of flashing while reconnecting.

## Authentication

Every JSON route sits behind the server API key. Open the page as
`/m2-dashboard/index.html#key=<api-key>` (bookmarkable; the fragment never
leaves the browser) or type the key into the inline form after a 401. The key
lives in page memory only — no cookies, no localStorage — which keeps the
`target/browser_security_audit_server.py` storage audit meaningful.

Note: with the API-key middleware as deployed, the *static files themselves*
also 401 without a key, so the browser cannot bootstrap the page from the
server yet. Candidate one-line server change (deliberately not made here):
exempt the `/m2-dashboard` mount from the API-key check under
`LLAMA_SERVER_TRUST_LAN`.

## Development against fixtures

```sh
cd tools/m2-dashboard
python3 -m http.server 8812
# http://localhost:8812/index.html?fixture=replay  – real prod capture (2026-08-07)
# http://localhost:8812/index.html?fixture=demo    – synthetic busy scene
```

- `fixtures/replay.json` — untouched interleaved `/slots` + registry snapshot
  samples recorded from production; `fixtures/props.json` — real `/props`
  (chat template truncated).
- `fixtures/demo.json` — synthetic 4-slot scene (regenerate with
  `node fixtures/make-demo.mjs`).
- `fixtures/state.json` / `events.json` — registry-schema fixtures used by the
  retained target-gate parser tests; the UI does not read them.

Fixture replays pre-warm the trackers so sparklines and the tail are populated
immediately.

## Layout of the code

- `dashboard.js` — transports (live/fixture), poll loop, DOM rendering.
- `lib/core.mjs` — all pure logic: normalization, phase/rate derivation,
  request-tail assembly, registry reduction, formatters, sparkline geometry.
  Node-tested in `tests/core.test.mjs`.
- `lib/schema.mjs`, `lib/live.mjs`, `lib/sse.mjs` — retained transport/parser
  modules used by the target gate probe (`target/live-probe.mjs`) and its host
  tests, not by the UI.
- `target/` — the inference-overhead target gate (see `TARGET_GATE.md`).

## Test

```sh
cd tools/m2-dashboard
npm test
python3 -m unittest discover -s target -p 'test_*.py'
```

Node's built-in runner only; no dependencies.

## Metrics that would help but are not exposed today

Candidates for tiny server-side additions (none implemented here):

- per-slot draft acceptance (`n_draft_accepted` / `n_draft_total` exist in
  `server_slot` but are not in `/slots`);
- prompt-cache tier occupancy: RAM-tier bytes, SSD-tier bytes/entries
  (`server_prompt_cache` tracks them internally);
- host RSS / memory pressure;
- `--metrics` on the prod start script would light up the KV-pressure footer
  (deployment change, not a code change).

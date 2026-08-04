# M2 Ultra admin dashboard playable vertical

This directory is a dependency-free, static dashboard for the adaptive
server. It runs against deterministic fixtures by default and can connect to
the direct single-model server's narrow request-registry admin routes.

Implemented here:

- versioned snapshot and event validation;
- immutable snapshot reduction with monotonic event sequencing;
- Last-Event-ID tracking, gap detection, truthful overflow/slow-client
  recovery causes, and bounded paced snapshot/stream reconnect attempts;
- observable retry/error state plus bounded event parsing, schema collections,
  strings, aggregate JSON bytes/nodes/depth, and the pending-event queue;
- strict known-field validation and cycle-safe bounded local control parameters;
- bounded event and timeline history;
- fixture views for lanes, requests, allocator pools, cache objects, disks,
  DSpark, capture, and operational history;
- schema-v2 bounded fast-refill configuration, cohort, cumulative member,
  deadline, and sampled one-member eligibility status;
- plain-text rendering and default prompt/output redaction;
- local pause/resume/reprioritize/cache control-intent drafting without any
  network mutation;
- zero-dependency Node tests;
- an opt-in live adapter using an API bearer and the loopback operator token;
- bounded authenticated snapshot, resumable SSE, exact-handle request detail,
  and confirmed cancel-only control views;
- queue-locked `id:epoch` cancellation with idempotent in-flight handling and
  fail-closed stale, unknown, and terminal handles;
- mandatory loopback Origin, JSON content type, custom CSRF proof, and
  duplicate-security-header rejection for both POST routes; and
- explicit unavailable markers for metrics not supplied by this vertical.

Not implemented or claimed:

- DNS, TLS, reverse-proxy configuration, or trusted lane headers;
- pause/resume, reprioritization, cache mutation, or content reveal;
- allocator, cache, disk, DSpark, capture, model, or process telemetry in the
  live registry-only view;
- browser automation or browser-storage audits; or
- a fresh dashboard/inference impact run for the cancel vertical (the existing
  read-only target gate remains available under `target/`).

## Test

```sh
cd tools/m2-dashboard
npm test
```

The tests use only Node's built-in test runner and standard library.

## Fixture preview

Serve the repository or this directory with any static HTTP server, then open
`index.html`. For example:

```sh
cd tools/m2-dashboard
python3 -m http.server 8080
```

The page initially fetches `fixtures/state.json` and `fixtures/events.json`.
Fixture mode performs no state-changing requests: its control buttons only
create a bounded local intent preview. In live mode, only request cancellation
is wired; all other controls remain visibly local drafts.

## Live detail and cancel view

Configure the direct server with an API key, an operator token of 32-256 bytes,
and localhost CORS. The operator token also controls trusted benchmark lanes,
so keep it distinct from the API bearer and never expose it to ordinary
clients:

```sh
export LLAMA_API_KEY='replace-with-api-key'
export LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN='replace-with-32-or-more-random-bytes'
export LLAMA_SERVER_TRUSTED_FAST_REFILL_MAX_MEMBERS=4
export LLAMA_SERVER_TRUSTED_FAST_REFILL_WINDOW_MS=30000

llama-server ... \
  --api-key "$LLAMA_API_KEY" \
  --cors-origins localhost \
  --cors-headers 'Authorization, Content-Type, X-Llama-Dashboard-CSRF, X-Llama-Trusted-Scheduling-Token, Last-Event-ID'
```

The explicit CORS header list is required for a dashboard served from a
different loopback port. In particular, browsers do not treat `Authorization`
as covered by a wildcard `Access-Control-Allow-Headers` response.

Serve this directory from the same loopback hostname as llama-server. Ports
may differ; hostnames must not. For example:

```sh
cd tools/m2-dashboard
python3 -m http.server 8081 --bind 127.0.0.1
```

Open `http://127.0.0.1:8081/`, enter the llama-server URL, API key, and
operator token, then choose **Connect live**. Credentials are sent only in
headers with browser credentials omitted, stay only in JavaScript memory, are
cleared from the form immediately, and are cleared from the adapter on
disconnect/page exit. They are never placed in a URL or browser storage.

The two refill variables are optional and the policy is disabled by default.
When both are valid, the compact **Bounded fast refill** card shows the runtime
configuration, current cohort, cumulative used/remaining fast members, and the
sampled monotonic deadline. `Initial selection` is the cohort's initial
selection phase; it is not the refill window. `One member: eligible at sample`
means at least one new independent fast member fit the fast-dominant cohort's
width and bounded window at the latest server sample. A same-fast request family
may use more width when demand, quota, and cohort capacity all permit it, so the
field is neither a family-size limit nor an admission promise. The browser
counts the sampled remaining time down from receipt and never reopens it without
a newer server sample. This removes indefinitely stale open status but cannot
remove snapshot or SSE transport latency; the label therefore remains explicitly
sampled rather than claiming current eligibility.

The four live endpoints are:

```text
GET /internal/admin/dashboard/snapshot
GET /internal/admin/dashboard/events  (Last-Event-ID header)
POST /internal/admin/dashboard/request-detail
POST /internal/admin/dashboard/request-control
```

All four require normal API-key middleware plus the operator header, reject
non-loopback and cross-site browser ingress, reject lane/tag headers and query
parameters, and are disabled when the operator token is absent. The POST routes
additionally require an explicit loopback `Origin`, `Content-Type:
application/json`, and `X-Llama-Dashboard-CSRF: 1`; each body is capped at 4
KiB and duplicate security headers are rejected before header-map collapse.
Router mode does not proxy these endpoints.

Request identity is the canonical `id:epoch` string from the registry. Detail
returns bounded registry metadata and explicitly empty, unretained content.
Control accepts exactly `{"action":"cancel","request_id":"id:epoch"}`. The
queue validates that full handle under its mutation lock before reusing the
existing durable cancellation path. A second cancel while the same bound
request is already cancelling is idempotent; stale, unknown, or terminal
handles cannot mutate state. An accepted operator cancellation also publishes
one `503 unavailable_error` with the fixed message `request cancelled by
dashboard operator` to the original completion waiter, so a non-streaming
client terminates instead of hanging after its slot is released.

The dependency-free target probe starts its own long completion against an
already-running direct server, discovers the new bound `id:epoch`, validates
redacted detail and negative CSRF/origin/content-type/stale-handle cases,
cancels that exact generation, checks contiguous lifecycle events, and submits
a fresh completion afterward:

```sh
export M2_DASHBOARD_BASE_URL=http://127.0.0.1:18130
export M2_DASHBOARD_ORIGIN=http://127.0.0.1:8081
export LLAMA_SERVER_BENCH_TRACE_CAPACITY=4096
python3 tools/m2-dashboard/target/admin_cancel_probe.py
```

It uses the same `LLAMA_API_KEY` and
`LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN` environment credentials as the server.
The trace-capacity variable must be present when the server starts: the probe
uses a unique authenticated benchmark tag to bind its client request to the
registry's exact runtime ID and refuses to select an unrelated concurrent
request.
Run it only against a disposable validation server: the discovered request is
cancelled intentionally.

Selecting a live request fetches its detail. **Cancel live request** presents a
request-specific browser confirmation before POSTing. Pause/resume,
reprioritization, and cache actions remain local drafts. Snapshot/event payloads
contain numeric registry facts, permit counts, refill state, lifecycle reasons,
and redacted request identities only. The strict client contract is schema
version 2.

## Client recovery contract

`AdminStateClient` still accepts two injected stream-state functions:

```text
getSnapshot() -> immutable versioned snapshot
openEvents({ lastEventId, onOpen, onEvent, onDisconnect }) -> close function
```

The live event adapter connects to authenticated SSE and honors
`Last-Event-ID`. A reconnect resumes from the reducer's last accepted event
ID. A server cursor rejection, gap, explicit overflow, malformed event, schema
mismatch, or local slow-consumer overflow discards queued deltas and fetches a
fresh snapshot before reopening the stream. Snapshot and stream-open failures
use finite retry schedules and expose the latest failure in client state.

Control intent objects for unfinished actions remain intentionally
transport-free. The live adapter separately exposes bounded `getRequestDetail`
and `cancelRequest` calls; no generic mutation dispatcher exists.

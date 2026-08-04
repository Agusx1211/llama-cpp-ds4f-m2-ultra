# M2 Ultra admin dashboard fixture foundation

This directory is a dependency-free, static dashboard for the adaptive
server. It runs against deterministic fixtures by default and can connect to
the direct single-model server's narrow read-only request-registry routes.

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
- local control-intent drafting without any network mutation;
- zero-dependency Node tests;
- an opt-in live adapter using an API bearer and the loopback operator token;
- bounded authenticated snapshot and resumable SSE request-registry views; and
- explicit unavailable markers for metrics not supplied by this vertical.

Not implemented or claimed:

- DNS, TLS, reverse-proxy configuration, or trusted lane headers;
- POST/DELETE control transport;
- request mutation/detail routes or content reveal;
- allocator, cache, disk, DSpark, capture, model, or process telemetry in the
  live registry-only view;
- browser automation or browser-storage audits; or
- dashboard/inference performance measurements.

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
It performs no state-changing requests. Fixture control buttons only create a
bounded local intent preview; the live view does not expose them.

## Live read-only view

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
  --cors-headers 'Authorization, X-Llama-Trusted-Scheduling-Token, Last-Event-ID'
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

The two live endpoints are:

```text
GET /internal/admin/dashboard/snapshot
GET /internal/admin/dashboard/events  (Last-Event-ID header)
```

Both require normal API-key middleware plus the operator header, reject
non-loopback and cross-site browser ingress, reject lane/tag headers and query
parameters, and are disabled when the operator token is absent. Router mode
does not proxy them. Snapshot/event payloads contain numeric registry facts,
permit counts, refill state, lifecycle reasons, and redacted request identities
only. The strict client contract is schema version 2.

## Client recovery contract

`AdminStateClient` accepts two injected read-only functions:

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

Control intent objects are intentionally transport-free. A later authenticated
adapter must validate authorization and CSRF policy before translating an
intent into a network mutation.

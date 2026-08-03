# M2 Ultra admin dashboard fixture foundation

This directory is a dependency-free, static foundation for the adaptive
server's future authenticated admin dashboard. It deliberately runs against
deterministic fixtures by default because the admin snapshot, event, detail,
and control routes do not exist yet.

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
- plain-text rendering and default prompt/output redaction;
- local control-intent drafting without any network mutation; and
- zero-dependency Node tests.

Not implemented or claimed:

- server routes, authentication, authorization, CORS, or CSRF protection;
- DNS, TLS, reverse-proxy configuration, or trusted lane headers;
- POST/DELETE control transport;
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

The page fetches `fixtures/state.json` and `fixtures/events.json`. It performs
no state-changing requests. Control buttons only create a bounded local intent
preview for a future authenticated adapter.

## Future integration contract

`AdminStateClient` accepts two injected read-only functions:

```text
getSnapshot() -> immutable versioned snapshot
openEvents({ lastEventId, onEvent, onDisconnect }) -> close function
```

The production event adapter must connect to an authenticated SSE route that
honors `Last-Event-ID`. A reconnect resumes from the reducer's last accepted
event ID. A gap, explicit server overflow, malformed event, schema mismatch, or
local slow-consumer overflow discards queued deltas and fetches a fresh
snapshot before reopening the stream. Snapshot and stream-open failures use
finite retry schedules and expose the latest failure in client state; this
fixture does not claim production availability behavior.

Control intent objects are intentionally transport-free. A later authenticated
adapter must validate authorization and CSRF policy before translating an
intent into a network mutation.

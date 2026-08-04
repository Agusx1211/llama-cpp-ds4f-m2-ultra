# M2 Ultra live-dashboard target gate

This gate measures the live dashboard's inference cost with one identical,
deterministic DeepSeek V4 Flash workload. It runs four counterbalanced blocks:
dashboard disconnected, connected, connected, disconnected. Each connected
block starts a fresh bounded probe and tears it down before the next block.
The harness is non-mutating apart from starting its local server and writing the
chosen artifact directory.

The checked-in example requires four samples per request and mode, rejects
coefficient of variation above 20%, and declares dashboard overhead meaningful
when connected median TTFT or elapsed time is more than 5% above disconnected.
Do not call a run an exit-gate pass if the sample count or variance gate fails.

## Contract and security

The probe uses `lib/live.mjs` and the strict shared snapshot/event parsers. It
reports both the snapshot's schema version and the parser's exported
`SCHEMA_VERSION`; it does not carry a private copy of dashboard schema v1 or v2.
When shared schema v2 is integrated, the example also requires authoritative
`fast_refill` objects in a parsed snapshot and parsed request events, including
an observed `window_open` state. State labels are derived only from the parsed
configuration, cohort, and refill fields.

Both credentials are accepted only through the process environment. They are
never accepted in a URL, command argument, config file, report, or artifact.
The probe verifies API-only and operator-only denial, query rejection, and
classification-header rejection on both live routes. It then verifies strict
snapshot schema and content redaction, future-cursor 409 recovery, bounded SSE
establishment, contiguous monotonic IDs, a resumed connection, and bounded
teardown. The harness redacts full credentials and their final four characters
from captured process output, then scans every artifact file for either form.

Keep a target-only protected environment file outside the source tree and
artifact directory. It should have mode `0600` and contain shell assignments
for these three variables:

```sh
LLAMA_API_KEY='target-api-key'
LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN='at-least-32-random-bytes'
M2_DASHBOARD_MODEL='/absolute/path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf'
```

Do not paste credential values into the queue invocation. Source the protected
file inside the queued target shell, export the values, and remove the file
after the run has been copied to its durable protected destination.

## Host checks

These checks do not load the model or establish the M2 gate:

```sh
cd tools/m2-dashboard
npm test
cd ../..
python3 -m unittest discover -s tools/m2-dashboard/target -p 'test_*.py'
```

## Reproducible queued target run

Commit the integrated refill telemetry and harness first. The target worktree
must be clean; the harness records and gates the source commit, dirty-path list,
exact server command, binary path, byte size, and SHA-256.

Synchronize the committed lane with `m2-ultra-sync.sh`, then serialize the
Release build and run through the shared M2 queue. Replace the lane and
protected-file paths, but do not put credential values on either command line:

```sh
SKILL=/home/agusx1211/workspaces/llama-cpp-m2-ultra/.agents/skills/m2-ultra-orchestrator

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label phase8-dashboard-gate-build \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/phase8-dashboard-target-gate \
  -- '/opt/homebrew/bin/cmake -S . -B build-phase8-dashboard-gate -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_FATAL_WARNINGS=OFF && /opt/homebrew/bin/cmake --build build-phase8-dashboard-gate --target llama-server -j16'

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label phase8-dashboard-gate-abba \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/phase8-dashboard-target-gate \
  -- '. /private/tmp/phase8-dashboard-gate.env && export LLAMA_API_KEY LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN M2_DASHBOARD_MODEL && export LLAMA_SERVER_TRUSTED_FAST_REFILL_MAX_MEMBERS=4 LLAMA_SERVER_TRUSTED_FAST_REFILL_WINDOW_MS=30000 && /usr/bin/python3 tools/m2-dashboard/target/ab_gate.py --config tools/m2-dashboard/target/target-gate.example.json --artifact /private/tmp/phase8-dashboard-gate-result'
```

The invocation stays in the foreground and the harness owns server/probe
cleanup. Use a new artifact path for every attempt; it refuses to overwrite an
existing path.

## Decision record

`summary.json` is authoritative for the harness decision. Supporting files
record the effective non-secret config, source/binary identity, per-request
TTFT, elapsed time, requested/reported/returned token counts, exact token IDs,
content/output hashes, per-block process samples, swap and memory-pressure
outputs, every connected live-probe result, server errors, variance, and clean
teardown. `block_order` must remain
`["disconnected", "connected", "connected", "disconnected"]` for the provided
gate.

This documentation and the host tests do not claim an M2 result. A target exit
requires a queued run whose `summary.json` says `PASS`, no credential leak, no
swap growth or critical pressure, no known server error, exact output equality,
clean teardown, sufficient samples, acceptable variance, and connected median
impact within the declared 5% threshold.

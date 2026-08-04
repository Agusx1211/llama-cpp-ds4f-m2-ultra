# Adaptive prefill prototype smoke

This is the first direct single-model Phase 3 feasibility check for DeepSeek V4
Flash on the M2 Ultra. It deliberately uses two split-KV slots: unified
multi-sequence Metal remains a separate bisection, and the 1M/64 production
geometry is a later horizontal gate.

The source at `66a16aa874a25106f4d90828170eae308e9369c8` passed this
smoke on the M2 Ultra with the pinned DeepSeek V4 Flash UD-Q8_K_XL model. Fast
output began 51.470 seconds before the 8K low request completed; isolated and
mixed 512-token outputs matched exactly. The trace recorded 49 low commits
during active fast decode, one aligned low handoff, one owner maximum, and no
lost events. The run had zero swapouts, no critical memory pressure, and clean
server teardown. This is a feasibility result for the geometry below, not a
1M/64 production-performance claim.

Build and launch on the M2 Ultra:

```sh
cmake -S . -B build-phase3-vertical-smoke \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_NATIVE=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_FATAL_WARNINGS=OFF
cmake --build build-phase3-vertical-smoke --target llama-server -j16

export DSV4_MODEL=/absolute/path/to/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
export LLAMA_API_KEY="api-$(openssl rand -hex 32)"
export LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN="$(openssl rand -hex 32)"
export LLAMA_SERVER_BENCH_TRACE_CAPACITY=4096

build-phase3-vertical-smoke/bin/llama-server \
  -m "$DSV4_MODEL" \
  --host 127.0.0.1 --port 18130 \
  -c 32768 -np 2 --no-kv-unified \
  -b 2048 -ub 2048 -t 16 -tb 16 \
  -ctk f16 -ctv f16 -ngl 999 -fa on -fit off \
  --cache-ram 0 --no-cache-idle-slots --no-context-shift --no-cache-prompt \
  --metrics --slots --api-key "$LLAMA_API_KEY"
```

From a second shell with the same two credentials:

```sh
python3 tools/server/tests/adaptive-prefill-smoke.py \
  --base-url http://127.0.0.1:18130 \
  --api-key "$LLAMA_API_KEY" \
  --scheduling-token "$LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN" \
  --artifact "/tmp/adaptive-prefill-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
```

The default runner sends an 8K trusted-low prompt plus isolated and sustained
trusted-fast 512-token decodes. It also checks an ordinary JSON lane-forgery
request, rejects either credential used alone, and reads the scheduler trace
only with both credentials. This default vertical workload remains the pinned
two-slot proof above.

The opt-in horizontal runner keeps the same two split-KV slots but replaces the
isolated/sustained 512-token pair with the shortest useful burst workload: an
isolated 8K low oracle, an isolated 128/16 fast oracle, then the identical 8K
low request alongside one to four sequential 128/16 fast requests. Restart the
server after setting the larger capacity in the server's environment:

```sh
export LLAMA_SERVER_BENCH_TRACE_CAPACITY=70000
```

Relaunch with the exact server command above so this value is present in the
server process.

Then run the client from the credentialed shell:

```sh
python3 tools/server/tests/adaptive-prefill-smoke.py \
  --base-url http://127.0.0.1:18130 \
  --api-key "$LLAMA_API_KEY" \
  --scheduling-token "$LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN" \
  --sequential-bursts --burst-count 4 --burst-interval-seconds 2 \
  --artifact "/tmp/adaptive-prefill-bursts-$(date -u +%Y%m%dT%H%M%SZ)"
```

Sequential-burst mode requires a count from one through four and a positive
interval. Before sending any workload request it requires an empty trace and
checks the server's advertised capacity against a worst-case per-token event
budget. It records each intended and actual launch time, launch lag, and TTFT.
Every burst must return before the mixed low request completes and exactly
match the isolated burst's token IDs and content; the mixed low result must
likewise exactly match its isolated oracle. The reference before/after, prompt
progress, stage/commit, one-owner, aligned-yield, active-fast chunk, trace
overflow, cardinality, and output-hash gates remain enabled.

The runner does not itself change admission. A server retaining the current
closed-cohort rule will intentionally fail to admit a later burst while the low
member remains bound. Enable a separately reviewed bounded fast-refill policy
before using this as a passing target gate. Extending the 30-second queue
deadline alone is not a fast-service result, and resident request parking is
not required for this two-slot low-plus-one-fast-at-a-time geometry.

`PASS` requires exact `tokens_predicted` and token-ID cardinality for every
request, identical token IDs and content for deterministic reference/fast
pairs, fast output before low completion, a low commit while fast decode is
active, an aligned yielded low-owner release, exact stage-to-commit lease
pairing, one prefill owner at most, and a lossless authenticated trace.

The queued lab harness additionally gates zero swapouts, zero critical-pressure
samples, zero known runtime errors, unchanged binary/source identity, and exact
server cleanup. Only after this vertical smoke passes should the same policy be
tested at `-c 1048576 -np 64`; do not add `-kvu` until its Metal bisection is
resolved.

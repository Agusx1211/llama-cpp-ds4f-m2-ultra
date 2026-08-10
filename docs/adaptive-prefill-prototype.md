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
isolated 8K low oracle, an isolated 128/4 fast oracle, then the identical 8K
low request alongside exactly four sequential 128/4 fast requests. Restart the
server after setting the larger trace and bounded-refill policy in the server's
environment:

```sh
export LLAMA_SERVER_BENCH_TRACE_CAPACITY=70000
export LLAMA_SERVER_TRUSTED_FAST_REFILL_MAX_MEMBERS=4
export LLAMA_SERVER_TRUSTED_FAST_REFILL_WINDOW_MS=30000
```

Relaunch with the exact server command above so these values are present in the
server process. Both limiter variables are required: member limits from 3
through 16 and windows from 1 through 30000 milliseconds are accepted. Missing,
malformed, partial, or out-of-range settings leave the bounded limiter disabled
while same-fast arrivals continue to refill elastically. Confirm the startup log
reports `bounded trusted fast refill limiter enabled: at most 4 fast members
within 30000 ms per cohort` before running the pinned bounded client.

To watch this prototype while playing with the sequential bursts, serve the
dependency-free read-only dashboard from a third shell:

```sh
cd tools/m2-dashboard
python3 -m http.server 8081 --bind 127.0.0.1
```

Open `http://127.0.0.1:8081/`, enter the server URL plus the API and operator
credentials above, and choose **Connect live**. The schema-v2 **Bounded fast
refill** card comes from the queue-locked request runtime snapshot. It shows the
configured maximum/window, whether a cohort is active, its dominant lane and
width limit, cumulative fast members used/remaining, and the monotonic refill
deadline. `Initial selection` reports the cohort's initial selection phase; it
does not report whether later refill is open. `One member: eligible at sample`
means at least one new independent fast member fit the latest sampled
fast-dominant cohort and bounded window. It neither limits a same-fast family to
one member nor promises admission. At the exact deadline the server sample
reports the window closed. The browser also counts a sampled open window down
from receipt and will not reopen it without a newer server sample; this bounds
staleness but cannot remove transport latency. With either limiter variable
absent or invalid, the card reports the bounded limiter's disabled state; this
does not mean elastic same-fast refill is disabled.

Then run the client from the credentialed shell:

```sh
python3 tools/server/tests/adaptive-prefill-smoke.py \
  --base-url http://127.0.0.1:18130 \
  --api-key "$LLAMA_API_KEY" \
  --scheduling-token "$LLAMA_SERVER_TRUSTED_SCHEDULING_TOKEN" \
  --sequential-bursts --burst-count 4 --burst-interval-seconds 0 \
  --burst-n-predict 4 \
  --artifact "/tmp/adaptive-prefill-bursts-$(date -u +%Y%m%dT%H%M%SZ)"
```

Sequential-burst mode requires exactly four requests with no intentional gap.
Each next request launches only after the previous HTTP response fully closes;
the zero interval prevents the low owner from staging a large idle-window chunk
before the next fast arrival. The four-token turns prove bounded interactive
refill, not sustained-generation throughput; the default vertical workload's
512-token isolated/sustained pair remains the longer-generation proof.
Before sending any workload request it requires an empty trace and
checks the server's advertised capacity against a worst-case per-token event
budget. It records each intended and actual launch time, launch lag, and TTFT.
Every burst must fully return before the mixed low request completes, exactly
match the isolated burst's token IDs and content, and, across each of the three
burst handoffs, receive a fresh low prompt-progress frame after the successor
starts and before it completes. The mixed low result must likewise exactly
match its isolated oracle. The reference before/after,
stage/commit, one-owner, aligned-yield, active-fast chunk, trace overflow,
cardinality, and output-hash gates remain enabled.

The runner does not itself change admission. The current default admits a later
same-fast burst elastically whenever physical and fast-lane width is available.
Setting both bounded-refill variables replaces that unbounded default with the
member/time budget used for the historical measurement below. The member budget
is cumulative for the live cohort: cancellation, timeout, or failure releases
physical width but does not refund a claimed member. At the pinned historical
commit below, omitting the limiter instead selected the then-current
closed-cohort rule and left a later burst queued while the low member remained
bound. Trace `cohort_id` identifies prefill ownership, not a runtime permit
epoch; the runner's refill proof is the completed burst followed by new
mixed-low progress before the next request completes. Burst 03 must complete
before the exact low response; the 02-to-03 handoff already proves low progress
while burst 03 is active, so the
runner does not require an impossible post-teardown prompt frame after a finite
low prompt has already reached its final token.

## Strict priority regression gate

The live strict-priority client exercises fast-to-fast, low-to-fast, and
normal-to-fast late arrivals through the public model IDs:

```sh
python3 tools/server/tests/strict-priority-smoke.py \
  --base-url http://127.0.0.1:8080 \
  --api-key "$LLAMA_API_KEY" \
  --artifact "/tmp/strict-priority-$(date -u +%Y%m%dT%H%M%SZ)"
```

It requires the incumbent to emit no output from the arrival's first prompt
progress through TTFT. After prompt completion, equal-fast requests must both
decode, while a lower incumbent must receive non-zero service at no more than
ten percent of the fast request's output. The default 512-token fast decode is
long enough to cross the shipped 64-iteration lower-lane keepalive interval.
Use `--observe-only` to collect a pre-change baseline without turning expected
policy violations into a non-zero client exit.

Clean commit `129fb3bba9d7706437439562b858f85f8f127c52` passed this
horizontal gate on the M2 Ultra with the pinned UD-Q8_K_XL model and exact
commands above. All four burst outputs matched the isolated oracle, the mixed
low output matched its isolated oracle, and reference-before matched
reference-after. Burst elapsed/TTFT times were 9.901/6.706 s, 10.436/7.227 s,
8.364/5.135 s, and 6.609/6.429 s; launch lag was 0.183--0.274 ms. The 35.405 s
mixed low response ended 92.810 ms after burst 03. A preceding run of the same
runtime and workload measured 35.410 s and a 92.500 ms final completion lead;
its only failure was the since-removed impossible final prompt-frame check.
The passing 106-event lossless trace recorded nine low commits during active
fast decode, three aligned low yields, and one prefill owner maximum. It used no
new swapouts, saw no critical memory pressure or runtime pressure/error metric,
and left no target process after teardown. The retained passing artifact is
`/private/tmp/phase3-fast-refill-pipeline-r2-129fb3bba9d7-20260804T122639Z`;
the SHA-256 of its verified manifest is
`9e37facf74cf64104b45ce22338867d9a973da8f9cc40a6f4580e666aff67f6e`.

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

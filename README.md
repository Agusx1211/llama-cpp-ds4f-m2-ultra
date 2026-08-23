# llama.cpp for DeepSeek V4 Flash on M2 Ultra

This is a specialized fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with one objective: execute DeepSeek V4 Flash as fast and efficiently as possible on an Apple M2 Ultra Mac.

This is not a general-purpose inference runtime. DeepSeek V4 Flash and M2 Ultra are the product. Every architecture decision, kernel, build default, and benchmark may be specialized for that exact pairing.

> [!IMPORTANT]
>
> Specialization is in progress. The repository currently inherits broad hardware and model support from upstream llama.cpp; that code should not be interpreted as a compatibility promise from this fork.

## Target

- Model: DeepSeek V4 Flash
- Hardware: Apple M2 Ultra
- Operating environment: macOS
- Primary compute backend: Metal
- Memory architecture: Apple unified memory
- Success criteria: correct inference, maximum prompt and generation throughput, low memory overhead, fast startup, and stable sustained performance

Other models and hardware are unsupported unless they help develop, validate, or operate the target. CUDA, HIP, Vulkan, SYCL, x86, non-Apple ARM, other Apple SoCs, and CPU-only execution are not optimization goals. Regressions outside the target are acceptable when they buy a meaningful target improvement.

## Optimization philosophy

- Specialize aggressively for the model graph, tensor shapes, quantization formats, M2 Ultra GPU, Metal runtime, and unified memory.
- Prefer dedicated kernels and execution paths over generic abstractions when measurement shows a benefit.
- Optimize end to end: model loading, prompt processing, token generation, dispatch, synchronization, allocation, memory traffic, and sustained execution all count.
- Require reproducible target-machine benchmarks for performance claims.
- Protect correctness. Any approximation or quality tradeoff must be explicit and measured.
- Keep useful upstream changes flowing without allowing upstream's cross-platform priorities to define this fork.

## Project status

The fork is in production on the target machine, serving eight concurrent conversations against a shared 1M-token KV budget. The table below states what it adds over mainline.

Build for the target:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_MTMD=OFF
cmake --build build --target llama-server -j
```

## Performance

DSV4 Flash 0731 as a lossless `gguf-m2` artifact (141.3 GiB), 1M context, speculation on, single lane. All figures come from one build (`97c68a32c`) on the target — a 60-core M2 Ultra with 192 GiB.

Prefill, isolated runs (three samples each, spread under 0.2%):

| prompt | prefill |
|---|---|
| 2048 | **350.0 t/s** |
| 4096 | **333.5 t/s** |

That is roughly 9 TFLOP/s against the machine's ~21.6 TFLOPS FP32 peak, at 13B active parameters per token. Raw data in `notes/2026-08-20-private-ane-prefill-experiment.md`.

Everything else is observed rather than benchmarked: 64 h of production traffic (2026-08-20 → 08-23), restricted to the 1442 requests that ran with no other lane active, 1.3M generated tokens.

| single-lane, in production | median | p90 | n |
|---|---|---|---|
| decode, prose | 24.2 t/s | 30.0 | 930 |
| decode, code | 29.9 t/s | 37.0 | 411 |
| decode, copy-heavy (edits, refactors) | 42.0 t/s | 56.2 | 84 |
| decode, all requests | 25.8 t/s | 35.2 | 1442 |
| prefill, 8k–32k prompt | 222 t/s | 292 | 23 |
| prompt-cache restore from SSD | 179k tok/s | 299k | 33 |

The production prefill row sits below the isolated figure because those requests shared the GPU with active decode; treat 350 t/s as the machine's rate and 222 t/s as what a prompt sees while the server is working.

Decode holds up with context — median t/s by prompt size: **26.8** (<1k), **25.0** (1–4k), **23.2** (4–16k), **25.1** (16–64k), **22.8** (64k+).

The cache restore row is the difference between resuming a conversation and recomputing it: the largest restore in the window rebuilt a 358k-token prefix in 1.2 s, against ~24 min to prefill it. Speculation over the same window: 0.673 pooled acceptance, median 4.21 accepted tokens per step.

Multi-lane costs more than it looks: with a second lane active, per-request decode falls to a 17.0 t/s median (n=52). Aggregate throughput still rises; individual latency does not.

## What this fork adds over mainline

DeepSeek V4 Flash itself is **not** a differentiator: the architecture, the DSpark drafter, and the `ngram-mod` / `ngram-map-k` speculators are all mainline llama.cpp. Everything below is additional, measured against the current merge base (`08659901c`, fully merged as of this commit).

| capability | mainline | here |
|---|---|---|
| Weight artifacts | stock GGUF; FP8 dense planes upcast to BF16 | `gguf-m2` (`E4M3_M2` + `MXFP4_M2`, ggml ids 90–99), bit-exact, 141.3 GiB vs 162 GB stock Q8, loader-gated by `m2.layout.version` |
| DSV4 Metal kernels | generic paths plus upstream's DSV4 ops | 83 fork-only kernels: sparse pack/compress, radix top-k, `e4m3_m2`/`mxfp4_m2` mm+mv families, lightning-indexer tails |
| Prompt cache | RAM only (`--cache-ram`), plus manual slot save/restore | adds an SSD tier (`--cache-disk`, `--cache-disk-limit`) that survives restarts — 179k tok/s median restore |
| Prefix reuse | per-slot, same conversation | cross-conversation zero-copy prefix reuse |
| Multi-lane serving | `-np` slots over a fixed KV split | elastic lanes over one shared budget with physical-page admission and deterministic victim selection |
| Scheduling | FIFO | priority lanes (`/fast`, `/normal`, `/low`) with a trusted-LAN ingress path |
| Speculation plumbing | DSpark + ngram speculators | rollback ring sized from every configured speculator, so deep and draftless configs neither crash nor fall back to per-step checkpoint save/restore |
| Observability | log lines; `--metrics` | live dashboard (SSE, per-request timelines with restore-vs-recompute provenance, cache map) and a durable request-history store (`--request-history*`) |
| Training capture | — | speculative-cycle and request-stream capture with a resync journal |

## The gguf-m2 artifact format

The fork does not serve stock GGUF files at full performance; it re-processes them offline into `*.m2.gguf` artifacts whose tensor encodings fit this fork's Metal kernels exactly. The transformation is **bit-exact**: the artifact stores the same values the source GGUF holds, verified block-by-block during conversion, and the served model produces byte-identical outputs. Two fork-owned tensor types (ggml ids 90–99 are reserved for the fork):

- `E4M3_M2` — the dense plane. DeepSeek V4 Flash's checkpoint stores most dense 2D tensors as FP8 (E4M3 × power-of-two 128×128-tile scales); public GGUFs upcast them to BF16, doubling their size and per-token read traffic. The converter recovers the FP8 form losslessly from the BF16 GGUF itself (no checkpoint download) and the kernels decode it in-register.
- `MXFP4_M2` — the routed-expert plane. Splits the 17-byte interleaved MXFP4 blocks into 16-byte-aligned code groups plus a packed 4-bit scale plane (the released expert scales use only 9 distinct values), enabling vector loads in the expert GEMV kernels.

Convert (reads the stock shards, writes one artifact, self-checks every converted block):

```sh
python3 conversion/gguf_m2_repack.py <stock-shard-*.gguf> -o model.m2.gguf
# variants: --keep-dense-plane (experts-only artifact), --keep-expert-plane (dense-only)
```

Interoperability is deliberately one-way: this fork still loads stock GGUFs, but `*.m2.gguf` artifacts declare `m2.layout.version` and the loader refuses any mismatch, while stock llama.cpp builds reject the unknown tensor types outright. A generic build can never silently misread repacked weights.

Measured on the target (M2 Ultra, DSV4 Flash 0731 UD-Q8_K_XL source, 2026-08-07): decode +4.4%, prefill −3.5%, artifact 150.7 → 141.3 GiB, outputs byte-identical. Details, alternatives considered, and the measurement ladder live in the local research notes and the commit history of the `gguf-m2` changes.

See [the build guide](docs/build.md) and its [Metal build section](docs/build.md#metal-build) for the inherited build options.

## Reproducing the deployment

This is the exact pipeline behind the numbers above, on a 192 GiB M2 Ultra. Budget ~173 GB of downloads and ~142 GiB for the artifacts; the source GGUFs can be deleted afterwards.

**1. Fetch the stock weights.** The target is Unsloth's lossless 8-bit build; the drafter is the matching DSpark checkpoint from the same repo. The `0731` in both names must agree — a DSpark built against a different checkpoint will load and draft badly.

```sh
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q8_K_XL/*" \
    --local-dir ~/unsloth/DeepSeek-V4-Flash-0731-GGUF          # 5 shards, ~162 GB
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf" \
    --local-dir ~/unsloth/DeepSeek-V4-Flash-DSpark-0731        # ~11.3 GB
```

**2. Repack both into `gguf-m2` artifacts.** Pass the shards in order; the converter self-checks every block and publishes atomically. It is I/O-heavy — do not co-run it with anything latency-sensitive.

```sh
python3 conversion/gguf_m2_repack.py \
    ~/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-0000{1,2,3,4,5}-of-00005.gguf \
    -o ~/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf          # -> 141.3 GiB

python3 conversion/gguf_m2_repack.py \
    ~/unsloth/DeepSeek-V4-Flash-DSpark-0731/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf \
    -o ~/unsloth/gguf-m2/dspark-0731.m2.gguf                   # -> ~10.3 GiB
```

**3. Serve.** This is the production invocation, eight lanes over a shared 1M-token budget:

```sh
LLAMA_DSV4_ADMISSION_VERTICAL=1 \
LLAMA_DSV4_AGGREGATE_POOL_DISABLE=1 \
GGML_E4M3_MM_NT2_MIN_N=512 \
build/bin/llama-server \
    -m  ~/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf \
    -md ~/unsloth/gguf-m2/dspark-0731.m2.gguf \
    --spec-type ngram-mod,draft-dspark \
    --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 12 --spec-ngram-mod-n-max 16 \
    --spec-draft-n-max 5 --spec-draft-n-min 1 --spec-draft-p-min 0.5 \
    -c 1048576 -np 8 --kv-unified --no-context-shift \
    -ngl 999 -ngld 999 -ctk f16 -ctv f16 -fa on -fit on -ub 2048 -b 2048 \
    --cache-ram 8192 --cache-disk ~/llama-kv-cache --cache-disk-limit 400 \
    --host 0.0.0.0 --port 8080
```

### Which flags actually carry the performance

| flag | why |
|---|---|
| `--spec-ngram-mod-n-max 16` | The single most sensitive knob. It sizes the DSV4 recurrent rollback ring, and a deep ring costs per decode step **even when the speculator emits nothing**: 64 measured −23% on prose, 16 costs −2.4% and buys +41% on copy-heavy work. Raise it only with a measurement. |
| `--spec-type ngram-mod,draft-dspark` | The gated ngram runs ahead of the drafter and discards its whole draft below `n-min`; ungated ngram variants regress prose badly. |
| `GGML_E4M3_MM_NT2_MIN_N=512` | Enables the double-tile e4m3 matmul only for 64-aligned chunks ≥512. Without the gate the NT2 path collapses in-graph — it benches faster in isolation and is slower in the real graph. |
| `LLAMA_DSV4_ADMISSION_VERTICAL=1` | Exact physical-page admission across the eight lanes. Without it, concurrent long contexts contend for sparse pages. |
| `-ub 2048 -b 2048` | Measured optimum; 4096 was not better. |
| `--cache-ram` / `--cache-disk` | The prompt-cache tiers. This is what turns a repeat 358k-token prefix into a 1.2 s restore instead of a re-prefill. |
| `--kv-unified --no-context-shift` | Required for elastic lanes to share one KV budget rather than a fixed per-slot split. |

Rollback switches for each of the riskier paths (`LLAMA_SERVER_PREFILL_PRIORITY=0`, `LLAMA_DSV4_SPARSE_RESIDENT_DISABLE=1`, `GGML_FA_MSKIP_BALLOT=0`, `GGML_METAL_SPARSE_BRIDGE_DRAIN_DISABLE=1`) each revert one optimization exactly, which is the fastest way to bisect a regression on your own hardware.

## Development

Read [AGENTS.md](AGENTS.md) before making changes. It defines target priorities, benchmark expectations, upstream merge rules, and the local research-note format.

This project's development model is purely autonomous AI. Autonomous agents are the implementers and are expected to carry work from investigation through implementation, testing, documentation, and self-review. Human direction establishes goals and supplies the target hardware; it must not be represented as human authorship of agent-produced work.

The short version:

- Target performance outranks portability and upstream architectural preferences.
- Specialized or invasive changes are welcome when evidence supports them.
- Autonomous AI development is the default operating model, not an exception.
- Cross-platform CI and support are not required.
- Performance work must include enough environment, command, baseline, result, and correctness detail to reproduce it.
- Agent-produced commits use the agent's identity and exact model name, never the directing user's personal identity.
- Commit messages are deliberately detailed records of what changed, why it changed, how it was validated, and what risks or open questions remain.

Local investigation notes belong in the ignored `/notes/` directory. Note filenames begin with the current UTC date in `YYYY-MM-DD-` form and each note is signed by the exact model that wrote it.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the intentionally lightweight contribution policy.

## Upstream relationship

This fork tracks [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). Upstream changes should be merged regularly so model formats, bug fixes, and useful runtime improvements stay current.

When priorities conflict, this fork wins: its DeepSeek V4 Flash behavior, M2 Ultra performance, target-specific kernels, build choices, README, and development rules must be preserved or deliberately adapted. Upstream is the base, not the product definition.

## License and acknowledgements

This project retains llama.cpp's MIT license and its upstream history. See [LICENSE](LICENSE). The many upstream contributors and third-party projects in that history made this specialized work possible.

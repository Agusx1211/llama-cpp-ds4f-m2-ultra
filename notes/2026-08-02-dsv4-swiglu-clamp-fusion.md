# DeepSeek V4 prompt clamp and SwiGLU fusion candidate

## Objective and status

Model: GPT-5

Status: implemented as a target-only candidate on branch
`dsv4-prompt-next-3` from integrated base
`c456d75ce5fd37d27969e3c0cf9b15d41460bbc1`. Local host compilation, the
Release test target, and three CPU whole-graph oracles pass. Apple Metal
compilation, selector tracing, correctness, and performance remain unmeasured;
no performance claim is made before serialized target validation.

The candidate recognizes DeepSeek V4's exact prompt-time routed activation
chain:

```text
gate F32 -> clamp[-inf, limit] --+
                                    +-> split SwiGLU -> F32
up F32   -> clamp[-limit, limit] --+
```

It applies both clamps inline in the existing SwiGLU element order and emits
the final F32 tensor in one Metal dispatch. It does not change the routed MoE
batch, expert selection, MXFP4 projections, F32 clamp values, exponential,
multiplication order, output layout, or model graph semantics.

## Target and measured baseline

- Hardware: Mac14,14, Apple M2 Ultra, 60 GPU cores, 192 GB unified memory.
- OS/toolchain: macOS 26.6 build 25G72, AppleClang 21.0.0.21000101, macOS SDK
  26.5.
- Build: Release, native ARM, Metal enabled with embedded metallib.
- Model: DeepSeek V4 Flash UD-Q8_K_XL, five GGUF shards; first shard
  `/Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf`.
- Runtime: full Metal offload, F16 K/V, Flash Attention, fit disabled, batch and
  physical microbatch 2048.

The current accepted prompt stack includes the compact routed-MoE worklist,
shared gate/up route preparation, and paired gate/up matrix dispatch. The most
recent paired-dispatch A/B measured 308.716679 t/s at pp512 and 342.060886 t/s
at pp4096, improvements of 0.214% and 0.243% over its two-dispatch control.
Those are historical target baselines, not measurements of this candidate.

## Ranked audit

### 1. Selected: fuse the two clamps into split SwiGLU

Every one of the 43 routed trunk layers uses a clamp limit of 7.0 and computes
two contiguous F32 `[2048, 6, n_tokens]` clamp tensors immediately before split
SwiGLU. The two clamps are independent and can run concurrently, but SwiGLU
must wait for both. The retained Metal scheduler therefore submits three
dispatches and inserts a dependency barrier before SwiGLU.

The fused kernel keeps the accepted SwiGLU workgroup geometry and adds the two
same Metal `clamp()` expressions before the existing `x / (1 + exp(-x)) * y`
calculation. Because inputs, clamp outputs, and SwiGLU inputs are all F32, no
intermediate precision or rounding step is removed. The raw-subgraph safety
check requires the final SwiGLU to be the only externally visible output.

The selector requires all of the following:

- Apple M2 Ultra;
- two adjacent `CLAMP` nodes followed by split `SWIGLU`;
- the SwiGLU inputs are exactly the two clamp outputs, in either raw-node order;
- F32 contiguous gate, up, and output tensors shaped
  `[2048, 6, 224..2048, 1]`;
- gate clamp `[-inf, limit]`, up clamp `[-limit, limit]`, and a finite positive
  shared limit; and
- graph fusion enabled and the three-node subgraph proven safe to fuse.

`GGML_METAL_DSV4_SWIGLU_CLAMP_DISABLE=1` restores the exact three-dispatch
graph in the same binary.

### 2. Rejected: combine route-map and matrix dispatch

The compact route map is produced by one 256-thread workgroup and consumed by
many matrix workgroups. A single ordinary Metal dispatch provides no global
workgroup barrier. Waiting matrix workgroups could occupy execution slots
before the map workgroup runs, while rebuilding the map per matrix workgroup
would repeat the work that the accepted shared-map optimization removed.
Neither option is a safe narrow experiment.

### 3. Rejected: reuse the gate/up route map for down projection

The map is scratch owned by the gate/up output allocation. The down projection
runs after SwiGLU, consumes a different activation, and has no allocator
lifetime guarantee keeping the earlier output storage alive. Extending that
lifetime or creating a persistent routing object is broader allocator work and
could retain large temporaries.

### 4. Deferred: pack final route counts into compact work items

Encoding each last-tile row count in the compact work item could remove the
1 KiB token-per-expert table and its matrix-kernel reads. It does not remove a
dispatch or barrier, and the table is cache-resident. Its expected benefit is
substantially smaller and less certain than eliminating two complete F32 clamp
passes.

## Static command and traffic accounting

Per routed layer, the candidate changes:

```text
two clamp dispatches + dependency barrier + SwiGLU dispatch
    -> one fused clamp/SwiGLU dispatch
```

It removes two dispatch commands and one memory barrier per layer. At pp512
that is 86 dispatches and 43 barriers. At pp4096, processed as two 2048-token
microbatches, it is 172 dispatches and 86 barriers.

Each removed clamp pass reads and writes one F32 tensor. For two clamps the
removed logical traffic is:

```text
2 * read+write * 2048 * 6 * n_tokens * 4 bytes
= 196608 * n_tokens bytes per layer
```

This is 96 MiB per layer and 4.03125 GiB across 43 layers at pp512. It is
384 MiB per layer and 32.25 GiB across 86 layer-microbatches at pp4096, before
cache effects. The graph still reserves the clamp tensor allocations, so no
allocation, RSS, or footprint reduction is claimed.

## Changes and oracle

- Add `kernel_dsv4_swiglu_clamp_f32`, using the same Metal `clamp()` intrinsic
  and SwiGLU expression as the three retained kernels.
- Add one fixed pipeline accessor and the strict raw-subgraph selector above.
- Preserve concurrency dependencies for all three fused raw nodes.
- Add the process-level kill switch above.
- Add whole-graph test cases at 224, 512, and 2048 tokens. Their random input
  range is `[-14, 14]`, so all gate and up clamp regions are exercised.

The focused test descriptor is `DSV4_SWIGLU_CLAMP_FUSION`. On Metal, the
candidate and disabled control both compare the complete graph output with the
CPU reference. Fusion debug output must report three fused ops and
`DSV4 clamp + SwiGLU` only in the candidate arm.

## Local validation

Local host: x86_64 Linux 7.0.0-28-generic, Intel Xeon E5-2630L v2, GCC 15.2.0,
CMake 4.2.3. The host lacks AVX2/FMA/BMI2, so those default backend features
were disabled after the first binary correctly exposed the host mismatch with
SIGILL. The compatible rebuild and all recorded validation commands exited
zero.

```sh
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF \
  -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF \
  -DLLAMA_CURL=OFF

cmake --build build-local --target test-backend-ops -j12

c++ -std=c++17 -fsyntax-only -w \
  -Iinclude -Iggml/include -Iggml/src -Iggml/src/ggml-metal \
  ggml/src/ggml-metal/ggml-metal-ops.cpp \
  ggml/src/ggml-metal/ggml-metal-device.cpp

build-local/bin/test-backend-ops test -b CPU \
  -o DSV4_SWIGLU_CLAMP_FUSION

git diff --check
```

Focused result: 3/3 CPU cases passed at 224, 512, and 2048 tokens. The local
host cannot compile the Metal source or exercise the M2 Ultra selector.

## Exact serialized target handoff

These commands are for the campaign director; none was run in this lane. From
the primary checkout, sync the clean committed branch:

```sh
SKILL=/home/agusx1211/workspaces/llama-cpp-m2-ultra/.agents/skills/m2-ultra-orchestrator
"$SKILL/scripts/m2-ultra-sync.sh" --lane dsv4-prompt-next-3 --base master
```

Build the matching M2 worktree:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" run \
  --label dsv4-prompt-next-3-build \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- /opt/homebrew/bin/cmake --build build-m2 \
  --target test-backend-ops test-llama-archs llama-bench llama-server -j12
```

Run the exact candidate and control whole-graph oracles:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-correctness-candidate \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_FUSION_DEBUG=2 build-m2/bin/test-backend-ops test -b MTL0 -o DSV4_SWIGLU_CLAMP_FUSION'

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-correctness-control \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_FUSION_DEBUG=2 GGML_METAL_DSV4_SWIGLU_CLAMP_DISABLE=1 build-m2/bin/test-backend-ops test -b MTL0 -o DSV4_SWIGLU_CLAMP_FUSION'
```

The candidate must compile `kernel_dsv4_swiglu_clamp_f32`, report
`fuse 3 ops`, log `DSV4 clamp + SwiGLU`, and pass all 3 cases. The control must
pass all 3 without selecting the fused pipeline.

Run broad clamp/GLU and DeepSeek V4 lifecycle correctness:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-correctness-broad \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'build-m2/bin/test-backend-ops test -b MTL0 -o CLAMP,SWIGLU,DSV4_SWIGLU_CLAMP_FUSION && build-m2/bin/test-llama-archs -a deepseek4 -s 42 && build-m2/bin/test-llama-archs -a deepseek4 -s 123'
```

Run four alternating fresh-process direct pairs. These two exact jobs are one
pair; repeat with unique `b`, `c`, and `d` labels in order
control/candidate, candidate/control, control/candidate:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-direct-candidate-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'build-m2/bin/test-backend-ops perf -b MTL0 -o DSV4_SWIGLU_CLAMP_FUSION'

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-direct-control-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'GGML_METAL_DSV4_SWIGLU_CLAMP_DISABLE=1 build-m2/bin/test-backend-ops perf -b MTL0 -o DSV4_SWIGLU_CLAMP_FUSION'
```

If direct latency improves, run at least two order-reversed full-model pairs:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-model-candidate-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'build-m2/bin/llama-bench -m /Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf -p 512,4096 -n 0 -r 3 -b 2048 -ub 2048 -ctk f16 -ctv f16 -ngl 999 -fa on -o json'

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-3-model-control-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-3 \
  -- 'GGML_METAL_DSV4_SWIGLU_CLAMP_DISABLE=1 build-m2/bin/llama-bench -m /Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf -p 512,4096 -n 0 -r 3 -b 2048 -ub 2048 -ctk f16 -ctv f16 -ngl 999 -fa on -o json'
```

## Retention gate and risks

- Reject if the embedded Metal build fails, the selector does not match the
  actual-model graph, either whole-graph arm fails, or lifecycle residuals move
  outside the established seed-42/123 values.
- Reject if direct performance regresses repeatably. Retain only if pp512 or
  pp4096 improves outside within-process standard deviation and process-order
  drift; command and traffic accounting alone is not a performance result.
- The kernel retains the existing SwiGLU workgroup geometry. Added clamp
  instructions should not change occupancy, but Apple pipeline `th_max` and
  target timing must confirm this.
- Fusion bypasses writes to two graph temporaries but does not remove their
  allocator reservations.
- Prompts whose final physical microbatch is below 224 tokens use the unchanged
  path. Decode, shared-expert SwiGLU, other shapes, other types, and other chips
  remain unchanged.

Signed-by: GPT-5
Date: 2026-08-02 UTC

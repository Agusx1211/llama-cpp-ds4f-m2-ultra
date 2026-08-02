# DeepSeek V4 prompt gate/up paired dispatch candidate

## Objective and status

Model: GPT-5

Status: implemented as a target-only candidate on branch
`dsv4-prompt-next-2` from integrated base
`abd8c6cc7a70ffc522a2273cc30a2c641cb23be9`. Local Linux C++ compilation
and the non-Metal test target pass. Apple Metal compilation, correctness, and
performance remain unmeasured and must run through the serialized M2 Ultra
queue before this candidate can be retained.

The candidate preserves the accepted shared route-map/worklist path and submits
the gate and up matrix work in one three-dimensional Metal dispatch. The third
grid dimension selects the projection; each threadgroup still computes exactly
one projection with the existing 128-thread, one-weight-stream, one-accumulator
kernel geometry. No arithmetic, precision, route preparation, allocation,
threadgroup count, activation traffic, weight traffic, or output traffic is
removed. The only expected saving is one Metal matrix-dispatch command per
gate/up pair.

## Target and baseline

- Intended hardware: Mac14,14, Apple M2 Ultra, 60 GPU cores, 192 GB unified
  memory.
- Intended OS/toolchain: macOS 26.6 build 25G72, AppleClang
  21.0.0.21000101, macOS SDK 26.5.
- Intended build: Release, native ARM, Metal enabled with embedded metallib.
- Model: DeepSeek V4 Flash UD-Q8_K_XL, five GGUF shards; first shard
  `/Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf`.
- Runtime: full Metal offload, F16 K/V, Flash Attention, fit disabled, logical
  and physical prompt batch 2048.
- Accepted shared-map baseline commit:
  `a260543b98190f3169eefba361516ce01b5e5d27`.

The accepted shared-map direct paired-operation means are 3,844.70 us at 224
tokens, 5,785.63 us at 256, 13,044.04 us at 512, 22,944.88 us at 1024, and
39,452.11 us at 2048. Its integrated full-model means are 308.071563 t/s for
pp512 and 341.317500 t/s for pp4096. These are historical baselines, not
measurements of this candidate.

## Evidence and design

DeepSeek V4 has 43 prompt gate/up pairs. The retained route-map optimization
already proves that adjacent operations share the same F32 activation and
top-6 expert IDs, have matching weight/output layouts, and can safely expose
both outputs. It prepares one compact immutable worklist and then executes two
unchanged matrix dispatches.

This candidate adds an exact `4096 -> 2048`, 256-expert/top-6 MXFP4/F32 paired
pipeline. The host binds both weight/output pairs once and dispatches the same
compact worklist over `grid.z = 2`. The template uses `threadgroup_position_in_grid.z`
only to select the weight and output buffer before the compact worklist replaces
the expert index. Thus each z-slice retains the accepted kernel's 128 threads,
64-row by 16-route tile, threadgroup memory, dequantization, reduction order,
and per-threadgroup live accumulator set. This avoids the occupancy regression
of the rejected arithmetic-fusion prototype, which made both projections live
inside one threadgroup and reduced `th_max` from 832 to 576.

At pp512, the candidate removes 43 of the 86 accepted matrix dispatch commands.
At pp4096 with two 2048-token microbatches, it removes 86 of 172. Total matrix
threadgroups are unchanged. The expected end-to-end effect is consequently
small and must clear run-order and sample noise before retention.

`GGML_METAL_DSV4_GATE_UP_DISPATCH_DISABLE=1` disables only the new paired
matrix dispatch. It leaves the accepted single map/worklist, map-to-matrix
barrier, and two independent matrix dispatches intact, providing the relevant
same-binary control.

## Rejected alternative: down-projection route-map reuse

The current compact route map resides in scratch appended to the gate/up output
allocation. Its lifetime is tied to that tensor. The later down projection
occurs after the SwiGLU activation and consumes a different activation tensor;
the gate/up output may already be dead and its allocation may be reused before
the down operation executes. There is no separately owned persistent route-map
object and no allocator lifetime guarantee connecting these operations.

Reusing that address would therefore be unsafe. Extending the gate/up tensor
lifetime would also retain its large output and scratch allocation across the
activation, which is a broader allocator/memory tradeoff. No down reuse is
implemented in this lane.

## Changes

- Add a paired flag to the Metal matrix-pipeline selector and select a unique
  paired DSV4 specialization only for an accepted shared-map gate/up pair.
- Bind the second weight and output at Metal buffer indices 7 and 8 and submit
  one matrix grid with two projection slices.
- Add compile-time paired buffer selection to the existing compact kernel while
  leaving all non-paired entry points and host dispatches unchanged.
- Add the dispatch-only environment kill switch described above.

The existing `MUL_MAT_ID_FUSION` oracle is deliberately reused. Its
`subtract=1` case compares two externally visible outputs in order, so swapped
or misbound gate/up buffers cannot pass through commutative addition. Its
existing direct-performance matrix covers 224, 256, 512, 1024, and 2048 tokens.

## Local validation

Local environment: x86_64 Linux 7.0.0-28-generic, GCC 15.2.0, CMake 4.2.3.
Metal is unavailable in this environment.

```sh
cmake -S . -B build-local -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
cmake --build build-local --target test-backend-ops -j12

c++ -std=c++17 -fsyntax-only -w \
  -Iinclude -Iggml/include -Iggml/src -Iggml/src/ggml-metal \
  ggml/src/ggml-metal/ggml-metal-ops.cpp \
  ggml/src/ggml-metal/ggml-metal-device.cpp

git diff --check
```

All commands exited zero. The incremental `test-backend-ops` target rebuild
also exited zero. The local build cannot compile the Metal specialization or
exercise the target selection.

## Serialized M2 Ultra validation commands

Run from the primary project checkout. Sync the committed lane first:

```sh
SKILL=/home/agusx1211/workspaces/llama-cpp-m2-ultra/.agents/skills/m2-ultra-orchestrator
"$SKILL/scripts/m2-ultra-sync.sh" --lane dsv4-prompt-next-2 --base master
```

Build all relevant Metal and architecture targets:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" run \
  --label dsv4-prompt-next-2-build \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- /opt/homebrew/bin/cmake --build build-m2 \
  --target test-backend-ops llama-bench llama-server test-llama-archs -j12
```

Run the exact order-sensitive candidate and control oracle. The candidate log
must show `fuse 2 ops` and compilation of
`kernel_mul_mm_id_mxfp4_f32_dsv4_n16_compact_pair`; the control must still
show `fuse 2 ops` but use `kernel_mul_mm_id_mxfp4_f32_dsv4_n16_compact`.

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-correctness-candidate \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'GGML_METAL_GRAPH_DEBUG=1 build-m2/bin/test-backend-ops test -b MTL0 -o MUL_MAT_ID_FUSION -p '\''type_a=mxfp4,type_b=f32,n_mats=256,n_used=6,b=1,m=2048,n=224,k=4096,o=2,mul=0,subtract=1'\'''

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-correctness-control \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'GGML_METAL_GRAPH_DEBUG=1 GGML_METAL_DSV4_GATE_UP_DISPATCH_DISABLE=1 build-m2/bin/test-backend-ops test -b MTL0 -o MUL_MAT_ID_FUSION -p '\''type_a=mxfp4,type_b=f32,n_mats=256,n_used=6,b=1,m=2048,n=224,k=4096,o=2,mul=0,subtract=1'\'''
```

Run the broad routed-op and DeepSeek V4 lifecycle checks:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-correctness-broad \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'build-m2/bin/test-backend-ops test -b MTL0 -o MUL_MAT_ID && build-m2/bin/test-llama-archs -a deepseek4 -s 42 && build-m2/bin/test-llama-archs -a deepseek4 -s 123'
```

Run direct candidate/control performance in four alternating fresh-process
pairs. These two queue jobs are one pair; enqueue the order
candidate/control, control/candidate, candidate/control, control/candidate with
unique labels:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-direct-candidate-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'build-m2/bin/test-backend-ops perf -b MTL0 -o MUL_MAT_ID_FUSION -p '\''type_a=mxfp4,type_b=f32,n_mats=256,n_used=6,b=1,m=2048,n=(224|256|512|1024|2048),k=4096,o=2,mul=0,subtract=0'\'''

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-direct-control-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'GGML_METAL_DSV4_GATE_UP_DISPATCH_DISABLE=1 build-m2/bin/test-backend-ops perf -b MTL0 -o MUL_MAT_ID_FUSION -p '\''type_a=mxfp4,type_b=f32,n_mats=256,n_used=6,b=1,m=2048,n=(224|256|512|1024|2048),k=4096,o=2,mul=0,subtract=0'\'''
```

If direct performance is not a regression, run at least two order-reversed
fresh-process full-model pairs; four are preferable because the expected
dispatch-only gain may be below one sample's noise:

```sh
"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-model-candidate-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'build-m2/bin/llama-bench -m /Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf -p 512,4096 -n 0 -r 3 -b 2048 -ub 2048 -ctk f16 -ctv f16 -ngl 999 -fa on -o json'

"$SKILL/scripts/m2-ultra-queue.sh" shell \
  --label dsv4-prompt-next-2-model-control-a \
  --workdir /Users/agusx1211/worktrees/llama-cpp-m2-ultra/dsv4-prompt-next-2 \
  -- 'GGML_METAL_DSV4_GATE_UP_DISPATCH_DISABLE=1 build-m2/bin/llama-bench -m /Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf -p 512,4096 -n 0 -r 3 -b 2048 -ub 2048 -ctk f16 -ctv f16 -ngl 999 -fa on -o json'
```

## Retention gate, risks, and follow-up

- Reject immediately if the embedded Metal build cannot compile the added
  inactive buffer parameters for existing specializations, if the paired
  oracle fails, or if `th_max` drops relative to the accepted compact kernel.
- Retain only if all correctness/lifecycle tests pass, direct latency is not a
  repeatable regression, and pp512 or pp4096 improves outside sample standard
  deviation and order drift. Do not infer a win solely from command-count
  reduction.
- The selector remains intentionally limited to the exact M2 Ultra shared-map
  pair. Decode, down projection, other quantizations, and other shapes are
  unchanged.
- The dispatch grid contains both independent projections but does not impose
  a synchronization or data dependency between their threadgroups. Metal may
  schedule z-slices differently than two separate dispatches; target timing is
  the deciding evidence.
- Existing non-paired Metal specializations include two compile-time-unused
  buffer parameters in the shared template. The host does not bind them outside
  the paired path, preserving the accepted command stream. The Apple compiler
  must prove them inactive during the queued build.

Signed-by: GPT-5
Date: 2026-08-02 UTC

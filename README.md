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

The fork is in production on the target machine. Beyond upstream llama.cpp it currently carries, among other things: a dedicated DSV4 Metal kernel stack; DSpark speculative decoding; elastic multi-lane serving (up to 4 concurrent conversations sharing one 512k-token KV budget with physical-page admission); a prompt cache with RAM and SSD tiers plus cross-conversation zero-copy prefix reuse; and the fork-owned `gguf-m2` weight artifact format below.

Build for the target:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_MTMD=OFF
cmake --build build --target llama-server -j
```

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

See [the build guide](docs/build.md) and its [Metal build section](docs/build.md#metal-build) for the inherited build options. Target-specific build and run commands will replace this section as they stabilize.

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

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

The first phase is to establish a reproducible DeepSeek V4 Flash baseline on the target Mac and identify the highest-cost operations. Optimization work will then concentrate on model-specific graph construction, quantization, Metal kernels, command scheduling, memory layout, and CPU/GPU overlap.

Until target artifacts and benchmark recipes are recorded, existing generic llama.cpp commands and documentation remain the operational starting point:

```sh
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
./build/bin/llama-bench --help
./build/bin/llama-cli --help
```

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

# Instructions for llama.cpp M2 Ultra

## Mission

This fork has one target: run DeepSeek V4 Flash as fast and efficiently as possible on an Apple M2 Ultra Mac.

The target has priority over every upstream compatibility goal. Optimize for this exact model, SoC, memory architecture, Metal implementation, and macOS environment. General-purpose design is not a goal.

## Scope and priorities

Use this order when requirements compete:

1. Correct DeepSeek V4 Flash inference.
2. End-to-end performance on M2 Ultra, including prompt processing, token generation, memory use, startup, and sustained operation.
3. Kernel-level performance on the M2 Ultra GPU and unified-memory system.
4. Reproducible benchmarks and a codebase that remains practical to modify.
5. Ease of merging useful upstream llama.cpp work.
6. Everything else.

CUDA, HIP, Vulkan, SYCL, other Apple chips, x86, non-Apple ARM, and CPU-only inference are unsupported unless work on them directly helps the target. It is acceptable to regress, bypass, or remove non-target behavior when that produces a meaningful target benefit. CPU work is relevant only where the M2 Ultra runtime needs it, such as orchestration, sampling, data preparation, or CPU/GPU overlap.

The inherited cross-platform code may remain because keeping upstream history mergeable is useful. Its presence does not create a support obligation.

## Engineering rules

- Measure before and after performance changes on the target machine with the target model or the closest documented representative workload.
- Keep correctness checks in every optimization loop. Compare logits or output quality where reduced precision, fusion, quantization, or altered scheduling can affect results.
- Optimize end-to-end execution rather than isolated kernels alone. Account for dispatch overhead, synchronization, memory movement, model loading, thermal behavior, and long-context behavior.
- Prefer Metal kernels and M2 Ultra-specific execution paths over generic abstractions when specialization is faster.
- Hard-coded assumptions are allowed when they are true for the target. State the assumption near the code or in a detailed note when it is not obvious.
- Aggressive and invasive changes are allowed. Simplicity is useful, but target performance takes priority over upstream architecture and broad maintainability.
- Do not add portability layers, fallback paths, or generic configuration without a demonstrated benefit to the target.
- Do not spend time repairing unrelated backends or platforms unless necessary to complete an upstream merge.
- Performance claims require the command, model/configuration, build type, relevant environment, result, and baseline. Include multiple runs when noise could change the conclusion.
- Never silently trade correctness for speed. An intentional approximation must be documented and measured.
- Read the relevant implementation and existing patterns before editing. Preserve useful upstream structure when it does not constrain the target.

## Working with agents

This project's development model is purely autonomous AI. Autonomous agents are the developers, not assistants operating under an upstream human-authorship policy. Agents are expected to investigate, design, implement, test, benchmark, document, review, and maintain the project end to end. Human input can set direction, constraints, and priorities without making the human the author of work produced by an agent.

Agents may write code, tests, documentation, notes, commit messages, pull-request material, and technical decisions. There is no requirement to open an issue first, obtain upstream maintainer interest, preserve all platforms, or prove that a human authored the design. Unless a decision would materially change the requested scope or requires external authority, agents should make a reasonable documented choice and continue autonomously.

Agents should proceed without ceremonial contribution steps. They must still:

- inspect relevant code before changing it;
- test in proportion to the risk;
- distinguish measured results from hypotheses;
- leave enough detail to reproduce performance findings;
- avoid committing, pushing, or changing external state unless the user authorizes it.

Upstream llama.cpp contribution policies apply only when intentionally proposing work back to `ggml-org/llama.cpp`. They do not govern work inside this fork.

## Commit authorship and messages

Every agent-produced commit must be authored and signed as the agent that performed the work. Never use the directing user's name, email address, GitHub noreply address, signing identity, or credentials, and never imply that the user authored agent-produced changes.

Before an authorized commit, set the identity for that commit only. Do not modify the user's global Git configuration. Use this pattern with the actual values substituted:

```sh
git -c user.name="<agent name> (<exact model name>)" \
    -c user.email="<agent-and-model-slug>@agents.invalid" \
    commit --signoff
```

The author and committer name must include both the agent product/name and the exact model identifier used for the work. The `Signed-off-by` trailer generated by `--signoff` is the required agent provenance signature. Add explicit trailers as well so the provenance remains machine-readable:

```text
Agent: <agent name>
Model: <exact model name>
Signed-off-by: <agent name> (<exact model name>) <<agent-and-model-slug>@agents.invalid>
```

Do not add `Co-authored-by` or `Signed-off-by` lines naming the directing user. If multiple agents materially contributed, name each additional agent and exact model in an `Assisted-by` trailer; the agent taking responsibility for the final review remains the author and signer.

Commit messages must be very detailed. A commit must stand alone as an engineering record for an agent that has no access to the originating conversation. Use a concise imperative subject, then a substantial body covering all applicable sections:

```text
<area>: <specific outcome>

Context:
- What problem or opportunity triggered the work.
- Relevant target hardware, model, workload, and prior behavior.

Changes:
- The concrete code, data-flow, kernel, build, test, or documentation changes.

Rationale:
- Why this design was selected.
- Important assumptions, alternatives considered, and reasons alternatives were rejected.

Validation:
- Exact commands and tests run, with meaningful results.
- Correctness comparisons and any testing that could not be run.

Performance:
- Baseline and new measurements, units, settings, and run-to-run variance.
- State "Not performance-sensitive" when the section does not apply.

Risks and tradeoffs:
- Known limitations, unsupported behavior, upstream divergence, and possible regressions.

Follow-up:
- Remaining work, unresolved questions, or "None".

Agent: <agent name>
Model: <exact model name>
Signed-off-by: <agent name> (<exact model name>) <<agent-and-model-slug>@agents.invalid>
```

Do not replace substance with boilerplate. Describe behavior before and after the commit, the reasoning chain behind non-obvious decisions, and negative or incomplete results. Include exact benchmark data for performance work and identify the corresponding `/notes/` file when it contains supporting raw data. Merge commits must also describe the upstream range, conflicts, fork-priority resolutions, validation, and any performance impact.

## Upstream integration

The expected remotes are:

- `origin`: this M2 Ultra fork;
- `upstream`: `ggml-org/llama.cpp`.

Merge upstream `master` regularly so the fork receives model support, bug fixes, format changes, runtime improvements, and structural evolution. Prefer merge commits over rebasing shared fork history so integrations and conflict resolutions remain auditable.

Use this procedure:

1. Start from a clean tree and record the current target baseline when the merge can affect runtime behavior.
2. Run `git fetch upstream` and merge `upstream/master` into the active integration branch.
3. Bring in all upstream changes by default. Do not discard upstream subsystems merely because they are outside the target unless keeping them creates a real cost.
4. Resolve conflicts in favor of this fork's DeepSeek V4 Flash and M2 Ultra behavior. Fork-owned target paths and policies have priority.
5. Treat `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, target-specific build defaults, Metal kernels, model execution changes, and benchmark tooling as fork-owned. Upstream edits to those areas must be adapted rather than allowed to erase the fork's intent.
6. If upstream changes an interface used by target-specific code, port the target code onto the new interface unless there is a measured reason to retain the old structure.
7. Build and run focused correctness tests after conflict resolution, then rerun affected target benchmarks.
8. Write a note for any non-trivial conflict, performance change, deferred repair, or deliberate divergence.

The merge commit message must follow the detailed commit policy above. In particular, list conflict resolutions that preserved fork-owned behavior and explain why each resolution remains correct for DeepSeek V4 Flash on M2 Ultra.

Being current with upstream does not mean adopting upstream priorities. Upstream provides the moving base; this fork defines the product, README, optimization criteria, and supported hardware.

## Local notes

The ignored `/notes/` directory is the project lab notebook. Agents may create and update files there freely. Notes stay local by default and can contain investigation details, raw benchmark data, failed approaches, merge decisions, and handoffs.

Every note must:

- use a filename beginning with the current UTC date as `YYYY-MM-DD-<descriptive-name>.md` (for example, `2026-08-01-metal-attention-profiling.md`);
- use a lowercase, hyphenated description after the date;
- be viewed in reverse lexical date order when reading newest first;
- identify the exact model that wrote it;
- contain enough detail for another agent to continue the work without relying on chat history.

End every note with this signature block:

```text
Signed-by: <exact model name>
Date: <YYYY-MM-DD UTC>
```

Do not sign for another model. When substantially revising an existing note, append a new signed update or create a new dated note so authorship remains clear.

Include the applicable details:

- objective and status;
- hardware, macOS, compiler, SDK, build flags, commit, model artifact, quantization, and runtime settings;
- commands and scripts used;
- observations and raw results, with units;
- baselines and run-to-run variance;
- correctness checks;
- hypotheses and why they are plausible;
- changes attempted, including failures;
- decisions, tradeoffs, and unresolved questions;
- affected files and concrete next steps.

## Contribution policy

Changes are welcome when they advance or unblock the target. A contribution does not need to be portable, preceded by an issue, split according to upstream review preferences, or supported indefinitely across upstream's backend matrix.

Keep contributions reviewable enough to validate the result. Provide correctness evidence and target benchmarks for performance-sensitive changes. Specialized code, new kernels, experimental branches, and rapid iteration are all acceptable. Compatibility work without a target benefit is out of scope.

## Useful starting points

- Metal backend: `ggml/src/ggml-metal/`
- Metal build documentation: `docs/build.md#metal-build`
- Model implementations: `src/models/`
- Benchmark tool: `tools/llama-bench/`
- Backend operation tests: `tests/test-backend-ops.cpp`
- General build documentation: `docs/build.md`

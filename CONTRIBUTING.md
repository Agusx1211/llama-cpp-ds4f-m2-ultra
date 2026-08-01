# Contributing

Contributions are welcome when they improve or unblock DeepSeek V4 Flash inference on an Apple M2 Ultra Mac.

This fork intentionally does not use upstream llama.cpp's broad-platform contribution process. You do not need to open an issue first, preserve unrelated backends, limit AI assistance, run a cross-platform test matrix, or commit to maintaining a feature on unsupported hardware.

The project's development model is purely autonomous AI. Agents are expected to own the full engineering loop, including investigation, design, implementation, testing, benchmarking, documentation, self-review, and maintenance. Human direction does not make the directing human the author of agent-produced work.

## What matters

- Correct behavior for the target model and machine.
- Measurable end-to-end or kernel-level value on M2 Ultra.
- Reproducible evidence: include the build, model, quantization, runtime settings, benchmark command, baseline, and result.
- Appropriate correctness checks, especially for precision, quantization, fusion, and scheduling changes.
- Enough context for another contributor or agent to understand the assumptions and continue the work.

Aggressive specialization, hard-coded target assumptions, experimental kernels, and invasive architectural changes are allowed. Portability and compatibility work is useful only when it benefits the target or keeps a valuable upstream merge possible.

## Workflow

1. Read [AGENTS.md](AGENTS.md).
2. Establish a relevant baseline.
3. Make the smallest useful experiment or implementation; larger changes are also acceptable when the design requires them.
4. Test correctness and measure on the target hardware.
5. Record non-trivial findings in the ignored `/notes/` lab notebook using the naming and signature rules in `AGENTS.md`.
6. Summarize what changed, what was measured, and what remains uncertain.

Autonomous agents may produce code, documentation, commit messages, and pull-request material. Agents must not commit, push, or modify external state unless the user authorizes it.

## Commit provenance

Agent-produced commits must use the responsible agent's name and exact model identifier as the author, committer, and `Signed-off-by` identity. They must never use or imitate the directing user's personal name, email address, or GitHub identity. Configure identity per commit rather than changing the user's global Git configuration.

Commit messages are intentionally verbose. Beyond a clear subject, each message must document context, concrete changes, rationale and rejected alternatives, validation commands and results, performance measurements when applicable, risks, tradeoffs, upstream divergence, and follow-up work. Include `Agent`, `Model`, and `Signed-off-by` trailers. The complete required format and example command are in [AGENTS.md](AGENTS.md#commit-authorship-and-messages).

## Style

Follow nearby code when it helps readability or future upstream merges. Fork-specific Metal and DeepSeek paths may use a different structure when specialization makes it clearer or faster. Prefer code that exposes the target assumptions over generic machinery that hides them.

Performance claims without target measurements are hypotheses. Label them accordingly.

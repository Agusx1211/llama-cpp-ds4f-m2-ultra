# DeepSeek V4 exact admission vertical

This opt-in prototype makes physical KV admission authoritative before a server
request starts prefill. It is intentionally restricted to the measured M2
Ultra geometry:

- DeepSeek V4 on the target sparse Metal allocator;
- exactly two server slots with unified elastic KV (`-np 2 -kvu`);
- context shifting and multimodal input disabled;
- full uncached prompts (the vertical forces `cache_prompt=false`);
- no global or per-request LoRA adapters; and
- an idle launch cohort. A one-slot request waits while the other slot is
  active; a parallel two-completion family is quoted and reserved atomically.

Enable it in addition to the normal target launch command:

```sh
LLAMA_DSV4_ADMISSION_VERTICAL=1 ./build/bin/llama-server \
  -m /path/to/DeepSeek-V4-Flash.gguf \
  -c 32768 -np 2 -kvu --no-context-shift
```

For each real family member, the server derives `[0, prompt + runway)` from the
tokenized prompt, the effective finite `n_predict`, and the configured
speculative width. A finite `prompt + n_predict` beyond the per-slot context is
rejected as HTTP 400. Unlimited generation reserves only the immediate target
plus speculative runway and must stop at the context boundary. Physical
pressure is deferred while another request owns the cohort; pressure with the
cohort idle returns HTTP 503.

The lower API first produces a mutation-free range quote, then atomically
reserves all family ranges. Arming the ticket does not map pages. The first
covered DSV4 preflight consumes and commits the exact reservation at the normal
last-safe point before graph submission. Destroying the ticket before that
preflight cancels the reservation; after consumption, ordinary sequence clear
owns the committed mappings.

The range planner is O(layers × family members), not O(prompt tokens). The
unified runtime must select affine compressed sparse storage; aggregate
compressed pools, prompt reuse, context shift, mixed active/admitted batches,
and configurations wider than two slots remain outside this vertical. Fixed
split KV is fully allocated at startup and therefore cannot exercise the
physical sparse-page admission contract.

Host validation:

```sh
cmake -S . -B build-host -DGGML_METAL=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target llama-server test-llama-archs \
  test-server-admission test-server-prefill test-server-request-runtime -j4
ctest --test-dir build-host --output-on-failure \
  -R 'test-server-admission|test-server-prefill|test-server-request-runtime'
./build-host/bin/test-llama-archs -a deepseek4 -s 42
```

The sparse reservation lifecycle test compiles on the host and skips without
target Metal. Run the last command from a Metal-enabled M2 Ultra build to check
quote immutability, armed-ticket cancellation, two-member preflight consumption,
quote identity, and return to baseline after sequence clear.

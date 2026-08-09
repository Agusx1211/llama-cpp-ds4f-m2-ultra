// Generates fixtures/content-v3.json: synthetic /m2-dashboard/content and
// /m2-dashboard/cache-preview bodies for the request ids and cache entry ids in
// events-v2.json, so `?fixture=v2` exercises the v3 content views offline.
//
//   node fixtures/make-content.mjs > fixtures/content-v3.json
//
// The payloads are shaped exactly like the server's (bounded head/tail with an
// explicit elision count), including a request whose content the store already
// evicted and a request that is still generating.

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const events = JSON.parse(readFileSync(join(here, "events-v2.json"), "utf8"));

const CAPS = {
    in_head_tokens: 512,
    in_tail_tokens: 256,
    out_head_bytes: 4096,
    out_tail_bytes: 2048,
    max_requests: 64,
    max_bytes: 1048576,
};

const SYSTEM = `You are a coding agent operating inside a checked-out repository.

<environment>
  workspace: /srv/work/llama-cpp-m2-ultra
  branch:    dashboard-v3
  platform:  darwin arm64 (Apple M2 Ultra, 192 GiB unified)
</environment>

<tools>
  read(path, offset?, limit?)     - read a file
  edit(path, old, new)            - exact string replacement
  bash(command, timeout_ms?)      - run a command in the workspace
  grep(pattern, glob?)            - ripgrep across the tree
</tools>

Rules:
- Read the relevant implementation before changing it.
- Never fabricate a benchmark number. Measure, then report the command.
- Keep the decode hot path allocation-free.
`;

const TOOL_RESULT = `[tool:bash] cmake --build build -j 24 --target llama-server
  [ 12%] Building CXX object tools/server/CMakeFiles/llama-server.dir/server-context.cpp.o
  [ 78%] Building CXX object tools/server/CMakeFiles/llama-server.dir/server-dashboard-bus.cpp.o
  [100%] Linking CXX executable ../../bin/llama-server
  exit 0 in 94.2s

[tool:bash] ctest --test-dir build -R server
  Test project /srv/work/llama-cpp-m2-ultra/build
        Start  1: test-server-dashboard-bus
   1/16 Test  #1: test-server-dashboard-bus ......   Passed    0.31 sec
  100% tests passed, 0 tests failed out of 16

user: good. now walk the prefill path and tell me where the 4.1 s restore is
being thrown away, and whether the fix belongs in admission or in the consult.`;

const TAIL_CONVO = `assistant: The consult happens in get_available_slot before prepare_dsv4_admission
runs, so by the time admission decides to clear, the SSD read has already been
paid for. Two candidate fixes:

  1. move the consult after admission — cheapest, but admission needs the
     restored prefix length to make its own decision, so this is circular;
  2. give admission a veto that runs *before* the consult using only the
     resident/LCP numbers it already has.

(2) is the one that actually works. Let me show you the ordering.

user: show me the ordering, with line numbers, and then stop.`;

const REVIEW_OUT = `Walking the ordering in \`tools/server/server-context.cpp\`:

1. \`get_available_slot()\` picks the slot (\`:2068\`), records the selection, and
   then — still inside the same function — runs the prompt-cache consult at
   \`:2279\` (\`ret->prompt_load(*prompt_cache, task.tokens)\`). For an SSD-tier
   entry that call is a multi-GiB \`pread\` plus a \`llama_state_seq_set_data_ext\`;
   in the capture you sent it took **4119 ms**.

2. \`prepare_dsv4_admission()\` (\`:2446\`) then evaluates \`reuse_resident\`. On the
   \`no_covering_checkpoint\` branch it takes the full-clear path at \`:2666\` and
   calls \`slot.prompt_clear()\`.

3. \`prompt_clear()\` drops \`prompt.tokens\` and the checkpoints outright. Nothing
   downstream can recover the restored prefix, so \`n_past\` lands at 0 and the
   whole 118k-token prompt is re-prefilled.

So the read, the bytes, and the 4.1 s are pure waste, and no single event
records it — you can only see it by joining \`cache_restore\` against the
\`admission\` outcome and the \`prompt_start\` landing.

**Where the fix belongs.** Not in the consult: it has no way to know what
admission will decide. Not in \`prompt_clear()\` either — by then the state is
already in the slot. It belongs in \`prepare_dsv4_admission\`, split into a
cheap predicate that runs *before* the consult:

\`\`\`cpp
// server-context.cpp, inside get_available_slot, before the consult
if (const auto veto = dsv4_admission_would_clear(*ret, task); veto) {
    // the elastic admission is going to clear this slot anyway; skipping the
    // consult saves the SSD read and the seq_cp, and costs one predicate
    SRV_TRC("skipping prompt-cache consult: admission will clear (%s)\\n",
            clear_code_name(veto.code));
    consult_cache = false;
}
\`\`\`

The predicate needs only \`resident\`, \`lcp\`, \`span_end\` and the checkpoint list —
all available before the consult. Everything else in \`reuse_resident\` is
downstream of state the consult produces, and those branches cannot fire when
the slot is about to be cleared for a *structural* reason (family admission,
\`no_lcp\`, \`mtmd\`, \`no_cache_prompt\`).

Risk: \`no_covering_checkpoint\` depends on the checkpoint list of the *restored*
state, which does not exist before the consult. That one case must stay where it
is, so this is a partial fix — it removes the family/no_lcp/mtmd/reuse-opt
wastage but not the checkpoint case, which is the one your capture hit.

Recommendation: land the veto for the structural reasons (cheap, obviously
correct), and separately teach the SSD tier to load the checkpoint index without
the state blobs so \`no_covering_checkpoint\` can also be decided pre-consult.
`;

const SHORT_CHAT_IN = `user: what does --cache-disk-limit default to, and is it per-file or total?`;
const SHORT_CHAT_OUT = `It defaults to **256** (GiB) and it is a **total** budget for the SSD tier, not a
per-file cap — see \`common/arg.cpp\`, \`--cache-disk-limit\`, which writes
\`params.cache_disk_limit_gib\`. \`-1\` disables the limit. Per-entry ceilings are
separate hard bounds in \`server-task.h\` (\`PCACHE_MAX_FILE_BYTES\`, 512 GiB).`;

const STALL_IN = `user: run the 1M-context draft-decode reproduction and report where it OOMs`;

function head(text, bytes) {
    return text.length <= bytes ? text : text.slice(0, bytes);
}

function makeInput(nTokens, headText, tailText) {
    const headTokens = Math.min(nTokens, CAPS.in_head_tokens);
    const tailTokens = nTokens > headTokens ? Math.min(nTokens - headTokens, CAPS.in_tail_tokens) : 0;
    const truncated = nTokens > headTokens + tailTokens;
    return {
        n_tokens: nTokens,
        head: headText,
        tail: tailTokens > 0 ? tailText : "",
        head_tokens: headTokens,
        tail_tokens: tailTokens,
        elided_tokens: truncated ? nTokens - headTokens - tailTokens : 0,
        truncated,
    };
}

function makeOutput(nTokens, text, realBytes = null) {
    const bytes = realBytes ?? text.length;
    const truncated = bytes > CAPS.out_head_bytes + CAPS.out_tail_bytes;
    return {
        n_tokens: nTokens,
        n_bytes: bytes,
        head: head(text, CAPS.out_head_bytes),
        tail: truncated ? text.slice(-CAPS.out_tail_bytes) : "",
        elided_bytes: truncated ? bytes - CAPS.out_head_bytes - CAPS.out_tail_bytes : 0,
        truncated,
    };
}

// a long generation, so one request exercises the output-truncation path
const LONG_OUT = Array.from({ length: 14 }, (_, i) =>
    `## Pass ${i + 1}\n\n${REVIEW_OUT}`).join("\n\n");

const store = { requests: 37, bytes: 412_344, evicted: 12 };

const requests = {
    // 101: the 281k-token deep-context agent turn, still generating
    101: {
        v: 1, req: 101, found: true, state: "live", slot: 0, t_ms: Date.now(),
        caps: CAPS, store,
        input: makeInput(281_400, SYSTEM + "\n" + TOOL_RESULT, TAIL_CONVO),
        output: null,
    },
    // 102: a short chat turn, nothing truncated on either side
    102: {
        v: 1, req: 102, found: true, state: "final", slot: 2, t_ms: Date.now() - 620_000,
        caps: CAPS, store,
        input: makeInput(14_350, SHORT_CHAT_IN, ""),
        output: makeOutput(180, SHORT_CHAT_OUT),
    },
    // 103: markdown/code-heavy output
    103: {
        v: 1, req: 103, found: true, state: "final", slot: 1, t_ms: Date.now() - 300_000,
        caps: CAPS, store,
        input: makeInput(31_900, SYSTEM, TAIL_CONVO),
        output: makeOutput(800, REVIEW_OUT),
    },
    // 104: a long generation the store had to truncate in the middle
    104: {
        v: 1, req: 104, found: true, state: "final", slot: 3, t_ms: Date.now() - 180_000,
        caps: CAPS, store,
        input: makeInput(97_200, SYSTEM + "\n" + TOOL_RESULT, TAIL_CONVO),
        output: makeOutput(2400, LONG_OUT),
    },
    // 105: hit the n_predict limit
    105: {
        v: 1, req: 105, found: true, state: "final", slot: 2, t_ms: Date.now() - 90_000,
        caps: CAPS, store,
        input: makeInput(52_400, SYSTEM, TAIL_CONVO),
        output: makeOutput(4096, LONG_OUT),
    },
    // 106: stalled and killed before producing anything
    106: {
        v: 1, req: 106, found: true, state: "live", slot: 1, t_ms: Date.now() - 400_000,
        caps: CAPS, store,
        input: makeInput(44_100, STALL_IN, ""),
        output: null,
    },
    // 107: rejected in the queue — never reached prefill, so no input captured
    107: {
        v: 1, req: 107, found: false, reason: "absent", caps: CAPS, store,
    },
    // 108: finished, but the store already evicted it
    108: {
        v: 1, req: 108, found: false, reason: "evicted", caps: CAPS, store,
    },
};

const CACHE_TEXT = {
    9: [SYSTEM, TAIL_CONVO],
    12: [SYSTEM + "\n" + TOOL_RESULT, TAIL_CONVO],
    14: [SHORT_CHAT_IN + "\n\n" + SHORT_CHAT_OUT, ""],
    3: [SYSTEM, TAIL_CONVO],
    15: [SYSTEM + "\n" + TOOL_RESULT, TAIL_CONVO],
    5: [SYSTEM, TAIL_CONVO],
    6: [SYSTEM + "\n" + TOOL_RESULT, TAIL_CONVO],
};
const CACHE_REQ = { 9: 104, 12: 101, 14: 102, 3: 103, 15: 105, 5: 108, 6: null };

const cache = {};
for (const entry of events.cache_state?.entries ?? []) {
    const [headText, tailText] = CACHE_TEXT[entry.id] ?? [SYSTEM, TAIL_CONVO];
    const headTokens = Math.min(entry.tokens, 64);
    const tailTokens = entry.tokens > headTokens ? Math.min(entry.tokens - headTokens, 32) : 0;
    const truncated = entry.tokens > headTokens + tailTokens;
    const body = {
        v: 1,
        id: entry.id,
        found: true,
        tier: entry.tier,
        tokens: entry.tokens,
        head: head(headText, 420),
        tail: tailTokens > 0 ? tailText.slice(-220) : "",
        head_tokens: headTokens,
        tail_tokens: tailTokens,
        elided_tokens: truncated ? entry.tokens - headTokens - tailTokens : 0,
        truncated,
        age_ms: 1200,
    };
    if (CACHE_REQ[entry.id]) body.req = CACHE_REQ[entry.id];
    cache[entry.id] = body;
}

process.stdout.write(JSON.stringify({ v: 1, requests, cache }, null, 1) + "\n");

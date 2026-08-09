// Persistent prompt cache (SSD tier) hardening tests.
//
// Two defects motivated these tests (notes/2026-08-08-claude-work-review.md,
// findings 2 and 3):
//
//   1. the disk fingerprint identified the target artifact and one bit for
//      "a draft model exists", while a `.lcpc` file carries *both* the target
//      and the draft sequence state - so swapping the drafter kept the same
//      fingerprint and restored incompatible state;
//   2. the `.lcpc` parser sized allocations from file-controlled lengths before
//      proving they fit in the file, so one corrupt file discovered at startup
//      could take the whole server down.
//
// The tests below exercise the real server_prompt_cache used by the server at
// startup: constructing it runs rescan_disk(), which is exactly the code path
// that must survive a directory full of garbage.

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-task.h"
#include "speculative.h"
#include "mtmd.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

//
// scratch directories
//

struct scratch_dir {
    explicit scratch_dir(const std::string & tag) {
        static std::mt19937_64 rng(0x5eed1234ull);

        path = (std::filesystem::temp_directory_path() /
                ("pcache-test-" + tag + "-" + std::to_string(rng()))).string();

        std::filesystem::create_directories(path);
    }

    ~scratch_dir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }

    scratch_dir(const scratch_dir &)             = delete;
    scratch_dir & operator=(const scratch_dir &) = delete;

    std::string path;
};

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    require(in.is_open(), "read_file: cannot open " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    require(out.is_open(), "write_file: cannot open " + path);
    if (!data.empty()) {
        out.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) data.size());
    }
    out.flush();
    require(bool(out), "write_file: write failed for " + path);
}

size_t count_files(const std::string & dir) {
    size_t n = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        n += entry.is_regular_file() ? 1 : 0;
    }
    return n;
}

//
// synthetic cache entries
//

constexpr uint64_t TEST_FINGERPRINT = 0x0123456789abcdefull;
constexpr size_t   TEST_LIMIT_TOKENS = 4096;

llama_tokens make_tokens(size_t n, int seed) {
    llama_tokens res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = (llama_token) (1 + ((i*2654435761u + (unsigned) seed) % 32000));
    }
    return res;
}

std::vector<uint8_t> make_blob(size_t n, uint8_t seed) {
    std::vector<uint8_t> res(n);
    for (size_t i = 0; i < n; ++i) {
        res[i] = (uint8_t) (seed + i*31u);
    }
    return res;
}

struct entry_spec {
    size_t n_tokens = 64;
    size_t n_main   = 4096;
    size_t n_drft   = 512;
    size_t n_ckpt   = 2;
    int    seed     = 1;
};

server_prompt_restore_domain restore_domain(
        llama_pos pos_min,
        llama_pos pos_max,
        uint32_t rollback_tokens,
        uint32_t swa_tokens = 0,
        bool rollback_bounded = true) {
    return { pos_min, pos_max, rollback_tokens, swa_tokens, rollback_bounded };
}

server_prompt_cache_state make_state(const entry_spec & spec) {
    server_prompt_cache_state state;

    state.prompt.tokens = server_tokens(make_tokens(spec.n_tokens, spec.seed), false);
    state.data.main     = make_blob(spec.n_main, (uint8_t) (0x10 + spec.seed));
    state.data.drft     = make_blob(spec.n_drft, (uint8_t) (0x20 + spec.seed));

    for (size_t i = 0; i < spec.n_ckpt; ++i) {
        common_prompt_checkpoint ckpt = {};
        ckpt.n_tokens  = (int64_t) (8*(i + 1));
        ckpt.id_task   = (int) (100 + i);
        ckpt.pos_min   = (llama_pos) (4*i);
        ckpt.pos_max   = (llama_pos) (4*i + 3);
        ckpt.data_tgt  = make_blob(128 + 16*i, (uint8_t) (0x30 + i));
        ckpt.data_dft  = make_blob( 64 +  8*i, (uint8_t) (0x40 + i));
        ckpt.data_spec = i == 0 ? std::vector<uint8_t>() : make_blob(32, (uint8_t) (0x50 + i));
        state.prompt.checkpoints.push_back(std::move(ckpt));
    }

    const llama_pos frontier = (llama_pos) spec.n_tokens - 2;
    const auto target = restore_domain(frontier, frontier, 5, 7);
    const auto draft  = restore_domain(frontier, frontier, 5, 11);
    state.coverage = server_prompt_build_restore_coverage(
            state.prompt,
            spec.n_drft > 0 ? server_prompt_restore_scope::target_and_draft
                            : server_prompt_restore_scope::target_only,
            target,
            spec.n_drft > 0 ? draft : server_prompt_restore_domain {});

    return state;
}

// spill one entry into `dir` and return the resulting file path
std::string spill_one(const std::string & dir, const entry_spec & spec, uint64_t fingerprint = TEST_FINGERPRINT) {
    server_prompt_cache cache(/*limit_size_mib =*/ 64, TEST_LIMIT_TOKENS, dir, /*disk_limit_gib =*/ 8, fingerprint);

    cache.states.push_back(make_state(spec));

    require(cache.spill_to_disk(cache.states.back()), "spill_to_disk failed");
    require(cache.states.back().on_disk(), "entry is not on disk after spill");

    return cache.states.back().file;
}

// what rescan_disk() makes of a directory, without touching anything else
size_t rescan_count(const std::string & dir, uint64_t fingerprint = TEST_FINGERPRINT) {
    server_prompt_cache cache(/*limit_size_mib =*/ 64, TEST_LIMIT_TOKENS, dir, /*disk_limit_gib =*/ 8, fingerprint);
    return cache.states.size();
}

pcache_disk_header header_of(const std::vector<uint8_t> & file) {
    require(file.size() >= sizeof(pcache_disk_header), "file too small to hold a header");
    pcache_disk_header hdr = {};
    memcpy(&hdr, file.data(), sizeof(hdr));
    return hdr;
}

// write a header back and re-seal it, so the file stays internally consistent
// apart from the field under test
void reseal(std::vector<uint8_t> & file, pcache_disk_header hdr) {
    hdr.hash_header = server_pcache_header_hash(hdr);
    memcpy(file.data(), &hdr, sizeof(hdr));
}

uint64_t coverage_hash_of(const std::vector<uint8_t> & file, const pcache_disk_header & hdr) {
    require(hdr.size_coverage == sizeof(pcache_disk_coverage) +
            hdr.n_checkpoints*sizeof(pcache_disk_checkpoint_coverage),
            "fixture has an invalid coverage size");
    require(sizeof(pcache_disk_header) + hdr.size_coverage <= file.size(),
            "fixture coverage is truncated");

    const uint8_t * data = file.data() + sizeof(pcache_disk_header);
    uint64_t hash = server_pcache_hash(data, sizeof(pcache_disk_coverage), 0);
    data += sizeof(pcache_disk_coverage);
    for (uint64_t i = 0; i < hdr.n_checkpoints; ++i) {
        hash = server_pcache_hash(data, sizeof(pcache_disk_checkpoint_coverage), hash);
        data += sizeof(pcache_disk_checkpoint_coverage);
    }
    return hash;
}

//
// tests
//

// A healthy entry must survive a spill, a restart (rescan) and a reload with
// every byte intact - this is the path the production 148 s -> 2 s repeat-prompt
// win depends on.
void test_round_trip() {
    scratch_dir dir("roundtrip");

    entry_spec spec;
    spec.n_tokens = 512;
    spec.n_main   = 256*1024;
    spec.n_drft   = 32*1024;
    spec.n_ckpt   = 3;

    const server_prompt_cache_state expected = make_state(spec);
    const auto expected_plan = server_prompt_make_restore_plan(
            expected.coverage, expected.prompt.tokens, 25, 25);
    require(expected_plan.kind == server_prompt_restore_kind::checkpoint,
            "round-trip fixture does not exercise checkpoint coverage");

    const std::string file = spill_one(dir.path, spec);
    require(std::filesystem::exists(file), "spilled file does not exist");
    require(count_files(dir.path) == 1, "spill left extra files behind (temp file not renamed?)");

    // restart the server: a fresh cache rescans the directory
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT);

    require(cache.states.size() == 1, "rescan did not restore the healthy entry");

    auto & state = cache.states.front();
    require(state.on_disk(), "restored entry is not marked on-disk");
    require(state.prompt.tokens.get_tokens() == expected.prompt.tokens.get_tokens(), "restored tokens differ");
    require(state.prompt.checkpoints.empty(), "rescan must not load checkpoints into RAM");
    require(state.coverage.scope == expected.coverage.scope, "rescan lost the restore scope");
    require(state.coverage.target.pos_min == expected.coverage.target.pos_min &&
            state.coverage.target.pos_max == expected.coverage.target.pos_max,
            "rescan changed the target frontier summary");
    require(state.coverage.target.swa_tokens == 7 && state.coverage.draft.swa_tokens == 11,
            "rescan changed target/draft SWA coverage");
    require(state.coverage.checkpoints.size() == expected.coverage.checkpoints.size(),
            "rescan did not index checkpoint coverage");
    const auto indexed_plan = server_prompt_make_restore_plan(
            state.coverage, state.prompt.tokens, 25, 25);
    require(indexed_plan.kind == expected_plan.kind &&
            indexed_plan.effective_tokens == expected_plan.effective_tokens &&
            indexed_plan.checkpoint_index == expected_plan.checkpoint_index,
            "startup coverage screening changed the restore/admission plan");

    require(cache.load_from_disk(state), "load_from_disk rejected a healthy file");

    require(state.data.main == expected.data.main, "target state blob differs after reload");
    require(state.data.drft == expected.data.drft, "draft state blob differs after reload");
    require(state.prompt.checkpoints.size() == expected.prompt.checkpoints.size(), "checkpoint count differs after reload");
    const auto loaded_plan = server_prompt_make_restore_plan(
            state.coverage, state.prompt.tokens, 25, 25);
    require(loaded_plan.kind == indexed_plan.kind &&
            loaded_plan.effective_tokens == indexed_plan.effective_tokens &&
            loaded_plan.checkpoint_index == indexed_plan.checkpoint_index,
            "full payload load changed the screened restore plan");

    auto it_a = state.prompt.checkpoints.begin();
    auto it_b = expected.prompt.checkpoints.begin();
    for (; it_a != state.prompt.checkpoints.end(); ++it_a, ++it_b) {
        const auto & a = *it_a;
        const auto & b = *it_b;
        require(a.n_tokens  == b.n_tokens,  "checkpoint n_tokens differs");
        require(a.id_task   == b.id_task,   "checkpoint id_task differs");
        require(a.pos_min   == b.pos_min,   "checkpoint pos_min differs");
        require(a.pos_max   == b.pos_max,   "checkpoint pos_max differs");
        require(a.data_tgt  == b.data_tgt,  "checkpoint data_tgt differs");
        require(a.data_dft  == b.data_dft,  "checkpoint data_dft differs");
        require(a.data_spec == b.data_spec, "checkpoint data_spec differs");
    }

    printf("  ok: healthy spill -> rescan -> reload round trip is byte-exact\n");
}

// RAM -> SSD spill on eviction and SSD limit enforcement must be unchanged.
void test_tiering_preserved() {
    scratch_dir dir("tiering");

    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT);

    entry_spec a;
    a.seed = 1;
    a.n_tokens = 32;
    entry_spec b = a;
    b.seed = 2;

    cache.states.push_back(make_state(a));
    cache.states.push_back(make_state(b));

    require(cache.size() > 0, "RAM accounting is empty before eviction");

    require(cache.evict_oldest_ram(), "evict_oldest_ram found nothing to evict");
    require(cache.states.front().on_disk(), "eviction did not spill the oldest entry to the SSD tier");
    require(!cache.states.back().on_disk(), "eviction spilled more than the oldest entry");
    require(cache.size_disk_total() > 0, "disk accounting did not grow after the spill");

    // dropping an entry must unlink its file
    const std::string file = cache.states.front().file;
    require(std::filesystem::exists(file), "spilled file missing");
    cache.drop_entry(cache.states.begin());
    require(!std::filesystem::exists(file), "drop_entry left the file behind");

    printf("  ok: RAM->SSD spill, disk accounting and drop-unlink still behave\n");
}

server_prompt make_prompt(size_t n_tokens, int seed) {
    server_prompt prompt;
    prompt.tokens = server_tokens(make_tokens(n_tokens, seed), false);
    return prompt;
}

void add_checkpoint(
        server_prompt & prompt,
        int64_t n_tokens,
        llama_pos pos_min,
        llama_pos pos_max,
        bool paired = false,
        bool speculative = false) {
    common_prompt_checkpoint checkpoint = {};
    checkpoint.n_tokens = n_tokens;
    checkpoint.pos_min  = pos_min;
    checkpoint.pos_max  = pos_max;
    checkpoint.data_tgt = make_blob(64, 0x61);
    if (paired) {
        checkpoint.data_dft = make_blob(32, 0x71);
    }
    if (speculative) {
        checkpoint.data_spec = make_blob(16, 0x81);
    }
    prompt.checkpoints.push_back(std::move(checkpoint));
}

server_prompt_restore_coverage coverage_for(
        const server_prompt & prompt,
        llama_pos pos_min,
        llama_pos pos_max,
        uint32_t rollback_tokens = 5,
        server_prompt_restore_scope scope = server_prompt_restore_scope::target_only) {
    const bool paired = scope == server_prompt_restore_scope::target_and_draft;
    const auto target = restore_domain(pos_min, pos_max, rollback_tokens, 0, true);
    return server_prompt_build_restore_coverage(
            prompt,
            scope,
            target,
            paired ? target : server_prompt_restore_domain {});
}

void test_restore_planner_boundaries_and_rollback() {
    // Full-match direct reuse consumes one token for [TAG_PROMPT_LOGITS].
    server_prompt exact = make_prompt(64, 11);
    auto exact_coverage = coverage_for(exact, 62, 62, 5);
    auto plan = server_prompt_make_restore_plan(exact_coverage, exact.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::direct, "full match was not directly restorable");
    require(plan.full_match && plan.effective_tokens == 63 && plan.recompute_tokens == 1,
            "full-match last-token adjustment is wrong");

    // Equality at the threshold is intentionally not enough: a checkpoint at
    // threshold-1 is the first one that guarantees at least one token of work.
    server_prompt boundary_bad = make_prompt(96, 12);
    add_checkpoint(boundary_bad, 64, 63, 63);
    auto bad_coverage = coverage_for(boundary_bad, 94, 94, 5);
    plan = server_prompt_make_restore_plan(bad_coverage, boundary_bad.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "checkpoint at the strict full-match threshold was admitted");

    server_prompt boundary_good = make_prompt(96, 12);
    add_checkpoint(boundary_good, 63, 62, 62);
    auto good_coverage = coverage_for(boundary_good, 94, 94, 5);
    plan = server_prompt_make_restore_plan(good_coverage, boundary_good.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::checkpoint && plan.effective_tokens == 63 &&
            plan.recompute_tokens == 1,
            "checkpoint immediately below the threshold did not land exactly");

    // Bounded recurrent rollback: the inclusive edge is valid, one token past
    // it is unusable in the absence of a covering checkpoint.
    server_prompt rollback = make_prompt(100, 13);
    auto rollback_coverage = coverage_for(rollback, 90, 98, 4);
    plan = server_prompt_make_restore_plan(rollback_coverage, rollback.tokens, 95, 97);
    require(plan.kind == server_prompt_restore_kind::direct && plan.effective_tokens == 95,
            "rollback depth edge was rejected");
    plan = server_prompt_make_restore_plan(rollback_coverage, rollback.tokens, 94, 97);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "rollback beyond the represented depth was admitted");

    auto no_rollback_coverage = server_prompt_build_restore_coverage(
            rollback,
            server_prompt_restore_scope::target_only,
            restore_domain(90, 98, 0));
    plan = server_prompt_make_restore_plan(no_rollback_coverage, rollback.tokens, 99, 100);
    require(plan.kind == server_prompt_restore_kind::direct && plan.effective_tokens == 99,
            "zero-depth bounded state rejected an exact frontier");
    plan = server_prompt_make_restore_plan(no_rollback_coverage, rollback.tokens, 98, 100);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "zero-depth bounded state admitted a one-token rollback");

    // A generated prompt commonly contains one sampled token beyond the state
    // frontier. When that whole token vector is an LCP of a longer request,
    // it is not a full request match, but the pending token still has to be
    // evaluated. The frontier, not source_tokens.size(), caps direct reuse.
    server_prompt pending = make_prompt(100, 14);
    auto pending_coverage = coverage_for(pending, 98, 98, 4);
    plan = server_prompt_make_restore_plan(pending_coverage, pending.tokens, 100, 104);
    require(plan.kind == server_prompt_restore_kind::direct &&
            plan.effective_tokens == 99 && plan.recompute_tokens == 1 && plan.pos_next == 99,
            "undecoded final token was claimed as represented state");

    auto complete_coverage = coverage_for(pending, 99, 99, 4);
    plan = server_prompt_make_restore_plan(complete_coverage, pending.tokens, 100, 104);
    require(plan.kind == server_prompt_restore_kind::direct &&
            plan.effective_tokens == 100 && plan.recompute_tokens == 0,
            "fully represented divergent prefix was truncated");

    auto impossible_coverage = complete_coverage;
    impossible_coverage.target.pos_min = 100;
    impossible_coverage.target.pos_max = 100;
    require(!server_prompt_restore_coverage_is_valid(impossible_coverage, pending.tokens),
            "frontier beyond the source position domain was accepted");

    printf("  ok: boundaries, pending-token cap, full-match adjustment and rollback edges are exact\n");
}

void test_paired_unequal_swa_thresholds() {
    const server_prompt prompt = make_prompt(100, 15);

    // Both models represent state through token 98, but their different SWA
    // widths retain different minima. This is a valid paired frontier.
    auto coverage = server_prompt_build_restore_coverage(
            prompt,
            server_prompt_restore_scope::target_and_draft,
            restore_domain(60, 98, 32, 10),
            restore_domain(40, 98, 32, 30));
    require(server_prompt_restore_coverage_is_valid(coverage, prompt.tokens),
            "paired coverage rejected unequal SWA minima at one decoded frontier");
    auto plan = server_prompt_make_restore_plan(coverage, prompt.tokens, 80, 90);
    require(plan.kind == server_prompt_restore_kind::direct && plan.effective_tokens == 80 &&
            plan.target_pos_threshold == 70 && plan.draft_pos_threshold == 50,
            "paired direct reuse did not apply independent SWA thresholds");

    // Reusing the target's narrower-SWA threshold for the draft would be a
    // false positive here: draft pos_min 60 does not cover its threshold 50.
    coverage = server_prompt_build_restore_coverage(
            prompt,
            server_prompt_restore_scope::target_and_draft,
            restore_domain(60, 98, 32, 10),
            restore_domain(60, 98, 32, 30));
    plan = server_prompt_make_restore_plan(coverage, prompt.tokens, 80, 90);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "target SWA threshold falsely qualified the draft domain");

    // Reusing the target's wider-SWA threshold for the draft would be a false
    // negative here: the draft has its own narrower window and covers 70.
    coverage = server_prompt_build_restore_coverage(
            prompt,
            server_prompt_restore_scope::target_and_draft,
            restore_domain(40, 98, 32, 30),
            restore_domain(60, 98, 32, 10));
    plan = server_prompt_make_restore_plan(coverage, prompt.tokens, 80, 90);
    require(plan.kind == server_prompt_restore_kind::direct && plan.effective_tokens == 80,
            "target SWA threshold falsely rejected the draft domain");

    printf("  ok: paired target/draft restores use independent SWA thresholds\n");
}

server_prompt make_mrope_prompt() {
    mtmd::input_chunks fixtures(mtmd_test_create_input_chunks());
    require(fixtures.size() >= 2, "mtmd test fixture has no image chunk");

    const mtmd_input_chunk * image = fixtures[1];
    size_t serialized_size = 0;
    require(mtmd_input_chunk_save(image, nullptr, 0, &serialized_size) == 0 && serialized_size > 64,
            "failed to size serialized image chunk");

    std::vector<char> serialized(serialized_size);
    require(mtmd_input_chunk_save(image, serialized.data(), serialized.size(), nullptr) == 0,
            "failed to serialize image chunk");

    // Serialized test fixture layout: version, chunk type, empty text-token
    // count, has-image flag, nx, ny, then position type. Switch NORMAL (0) to
    // MROPE (1), where this 4x4x16 image occupies 256 token cells but 4 decoder
    // positions. mtmd_input_chunk_load validates the resulting representation.
    constexpr size_t pos_type_offset =
            sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint8_t) +
            2*sizeof(uint32_t);
    const uint32_t mrope = 1;
    require(pos_type_offset + sizeof(mrope) <= serialized.size(), "serialized image layout is truncated");
    memcpy(serialized.data() + pos_type_offset, &mrope, sizeof(mrope));

    mtmd::input_chunk_ptr image_mrope(mtmd_input_chunk_load(serialized.data(), serialized.size()));
    require(image_mrope != nullptr, "failed to load synthetic M-RoPE image chunk");
    require(mtmd_input_chunk_get_n_tokens(image_mrope.get()) !=
            (size_t) mtmd_input_chunk_get_n_pos(image_mrope.get()),
            "synthetic media chunk does not have a nontrivial token/position mapping");

    server_prompt prompt;
    prompt.tokens.has_mtmd = true;
    prompt.tokens.push_back(101);
    prompt.tokens.push_back(102);
    prompt.tokens.push_back(image_mrope.get());
    prompt.tokens.push_back(103);
    prompt.tokens.push_back(104);
    return prompt;
}

void test_position_aware_frontier_mapping() {
    server_prompt prompt = make_mrope_prompt();
    require(prompt.tokens.size() == 260 && prompt.tokens.pos_next() == 8,
            "unexpected M-RoPE fixture geometry");

    // State through decoder position 6 contains 2 text tokens, all 256 image
    // cells (4 positions), and one trailing token: 259 represented tokens.
    auto coverage = coverage_for(prompt, 6, 6, 4);
    require(server_prompt_restore_coverage_is_valid(coverage, prompt.tokens),
            "legal non-1:1 frontier was rejected");
    auto plan = server_prompt_make_restore_plan(coverage, prompt.tokens, 260, 264);
    require(plan.kind == server_prompt_restore_kind::direct &&
            plan.effective_tokens == 259 && plan.pos_next == 7,
            "planner treated decoder positions as token indexes");

    // Position 3 is inside the indivisible image chunk. size_up_to_pos() must
    // not turn that into a plausible claim for all 256 media token cells.
    auto inside_media = coverage_for(prompt, 3, 3, 4);
    require(!server_prompt_restore_coverage_is_valid(inside_media, prompt.tokens),
            "frontier inside a media chunk was accepted");

    printf("  ok: restore coverage is position-aware for non-1:1 media mappings\n");
}

void test_speculative_sequence_state_clear() {
    common_speculative_deferred_state state;
    state.pending_pos      = 73;
    state.pending_embd     = { 1.0f, 2.0f, 3.0f };
    state.verify_pos_first = 70;
    state.verify_embd      = { 4.0f, 5.0f, 6.0f };
    state.verify_rows      = 1;

    state.clear();
    require(state.pending_pos == -1 && state.verify_pos_first == -1 && state.verify_rows == 0 &&
            state.verify_embd.empty() &&
            state.pending_embd == std::vector<float>({ 0.0f, 0.0f, 0.0f }),
            "sequence clear retained an EAGLE3 deferred or verify boundary");

    // Reuse the exact same absolute frontier with another conversation's
    // embedding. The old row is gone, so numeric adjacency cannot bridge the
    // conversations and only the new boundary can be serialized/used.
    state.pending_pos  = 73;
    state.pending_embd = { 9.0f, 8.0f, 7.0f };
    require(state.pending_pos == 73 &&
            state.pending_embd == std::vector<float>({ 9.0f, 8.0f, 7.0f }),
            "same-position replacement resurrected the displaced boundary embedding");

    // Public dispatch is intentionally null-safe for servers with no drafter.
    common_speculative_clear_state(nullptr, 0);

    printf("  ok: same-position EAGLE3 sequence reuse clears displaced boundary state\n");
}

void test_coverage_aware_supersession() {
    constexpr size_t boundary = 64;

    // A longer token superset without a covering checkpoint must retain the
    // useful exact boundary entry.
    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        server_prompt shorter = make_prompt(boundary, 21);
        const auto shorter_coverage = coverage_for(shorter, 62, 62, 5);
        require(cache.alloc(shorter, shorter_coverage, 128, 0) != nullptr,
                "failed to allocate the shorter boundary");

        server_prompt longer = make_prompt(128, 21);
        const auto longer_coverage = coverage_for(longer, 126, 126, 5);
        require(cache.alloc(longer, longer_coverage, 256, 0) != nullptr,
                "failed to allocate the longer entry");
        require(cache.states.size() == 2,
                "longer entry without restore coverage deleted the exact boundary");
    }

    // A checkpoint at boundary-1 can reproduce the shorter entry's effective
    // full-match landing, so supersession is safe.
    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        server_prompt shorter = make_prompt(boundary, 22);
        const auto shorter_coverage = coverage_for(shorter, 62, 62, 5);
        require(cache.alloc(shorter, shorter_coverage, 128, 0) != nullptr,
                "failed to allocate the shorter boundary");

        server_prompt longer = make_prompt(128, 22);
        add_checkpoint(longer, 63, 62, 62);
        const auto longer_coverage = coverage_for(longer, 126, 126, 5);
        require(cache.alloc(longer, longer_coverage, 256, 0) != nullptr,
                "failed to allocate checkpoint-covered replacement");
        require(cache.states.size() == 1 && cache.states.front().prompt.tokens.size() == 128,
                "valid exact checkpoint did not supersede the shorter entry");
    }

    printf("  ok: containment preserves uncovered boundaries and prunes exactly covered ones\n");
}

void test_target_draft_checkpoint_hygiene() {
    server_prompt source = make_prompt(80, 31);
    add_checkpoint(source, 48, 47, 47, /*paired=*/true, /*speculative=*/true);

    // Target-only capture strips both draft checkpoint bytes and speculative
    // implementation state.
    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        const auto coverage = coverage_for(source, 78, 78, 5, server_prompt_restore_scope::target_only);
        auto * state = cache.alloc(source, coverage, 128, 0);
        require(state != nullptr, "target-only allocation failed");
        require(state->coverage.scope == server_prompt_restore_scope::target_only && state->data.drft.empty(),
                "target-only entry retained a draft top-level state");
        require(state->prompt.checkpoints.size() == 1 &&
                state->prompt.checkpoints.front().data_dft.empty() &&
                state->prompt.checkpoints.front().data_spec.empty(),
                "target-only entry retained draft/spec checkpoint state");
    }

    // A paired entry retains the complete atomic checkpoint and advertises the
    // stronger restore domain.
    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        const auto coverage = coverage_for(source, 78, 78, 5,
                server_prompt_restore_scope::target_and_draft);
        auto * state = cache.alloc(source, coverage, 128, 64);
        require(state != nullptr, "paired allocation failed");
        require(state->coverage.scope == server_prompt_restore_scope::target_and_draft &&
                !state->prompt.checkpoints.front().data_dft.empty() &&
                !state->prompt.checkpoints.front().data_spec.empty(),
                "paired entry did not retain draft/spec checkpoint state");
    }

    // This is the donor predicate used by server-context: raw LCP alone is not
    // enough, while the target-only checkpoint makes the donor usable.
    server_prompt donor_bad = make_prompt(128, 32);
    auto donor_coverage = coverage_for(donor_bad, 126, 126, 5);
    auto plan = server_prompt_make_restore_plan(donor_coverage, donor_bad.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "uncovered live donor was qualified");
    add_checkpoint(donor_bad, 63, 62, 62, /*paired=*/true, /*speculative=*/true);
    donor_coverage = coverage_for(donor_bad, 126, 126, 5,
            server_prompt_restore_scope::target_only);
    plan = server_prompt_make_restore_plan(donor_coverage, donor_bad.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::checkpoint && plan.effective_tokens == 63,
            "coverage-qualified target-only donor was rejected");

    printf("  ok: target-only capture strips draft/spec state; paired and donor coverage stay exact\n");
}

// The fingerprint must change when anything the serialized state depends on
// changes. A value that cannot move the fingerprint cannot invalidate a file.
void test_fingerprint_covers_state_identity() {
    server_prompt_cache_fingerprint_inputs base;

    base.model_tgt.present = 1;
    base.model_tgt.path    = "/models/dsv4-flash-gguf-m2.gguf";
    base.model_tgt.size    = 153ull*1024*1024*1024;
    base.model_tgt.mtime   = 1754600000;
    base.model_tgt.probe   = 0xabcdef01;

    base.model_dft.present = 1;
    base.model_dft.path    = "/models/dsv4-drafter-0731-bf16.gguf";
    base.model_dft.size    = 3ull*1024*1024*1024;
    base.model_dft.mtime   = 1754610000;
    base.model_dft.probe   = 0x11223344;

    base.kv_type_k_tgt   = 1;
    base.kv_type_v_tgt   = 1;
    base.kv_type_k_dft   = 1;
    base.kv_type_v_dft   = 1;
    base.flash_attn_type = 1;
    base.kv_unified      = 0;
    base.swa_full        = 0;
    base.n_ctx_tgt       = 1024*1024;
    base.n_seq_max_tgt   = 4;
    base.n_ctx_dft       = 1024*1024;
    base.n_seq_max_dft   = 4;
    base.n_parallel      = 4;
    base.spec_types      = { 5 };
    base.spec_n_max      = 1;

    const uint64_t fp_base = server_prompt_cache_fingerprint(base);

    require(server_prompt_cache_fingerprint(base) == fp_base, "fingerprint is not deterministic");

    struct mutation {
        const char * name;
        void (*apply)(server_prompt_cache_fingerprint_inputs &);
    };

    const mutation mutations[] = {
        // schema / ABI generation
        { "schema version bump",       [](server_prompt_cache_fingerprint_inputs & i) { i.schema_version    += 1; } },
        { "state-seq ABI bump",        [](server_prompt_cache_fingerprint_inputs & i) { i.state_seq_version += 1; } },
        { "fork state layout bump",    [](server_prompt_cache_fingerprint_inputs & i) { i.state_seq_layout  += 1; } },
        { "session ABI bump",          [](server_prompt_cache_fingerprint_inputs & i) { i.session_version   += 1; } },

        // target artifact
        { "target path",               [](server_prompt_cache_fingerprint_inputs & i) { i.model_tgt.path  = "/models/other.gguf"; } },
        { "target size",               [](server_prompt_cache_fingerprint_inputs & i) { i.model_tgt.size  += 1; } },
        { "target mtime",              [](server_prompt_cache_fingerprint_inputs & i) { i.model_tgt.mtime += 1; } },
        { "target content probe",      [](server_prompt_cache_fingerprint_inputs & i) { i.model_tgt.probe += 1; } },

        // draft artifact - the defect this lane exists for
        { "drafter path (swap)",       [](server_prompt_cache_fingerprint_inputs & i) { i.model_dft.path  = "/models/dsv4-drafter-0731-q8_0.gguf"; } },
        { "drafter size",              [](server_prompt_cache_fingerprint_inputs & i) { i.model_dft.size  += 1; } },
        { "drafter mtime",             [](server_prompt_cache_fingerprint_inputs & i) { i.model_dft.mtime += 1; } },
        { "drafter content probe",     [](server_prompt_cache_fingerprint_inputs & i) { i.model_dft.probe += 1; } },
        { "drafter removed",           [](server_prompt_cache_fingerprint_inputs & i) { i.model_dft = {}; } },

        // KV representation
        { "KV type K (target)",        [](server_prompt_cache_fingerprint_inputs & i) { i.kv_type_k_tgt   += 1; } },
        { "KV type V (target)",        [](server_prompt_cache_fingerprint_inputs & i) { i.kv_type_v_tgt   += 1; } },
        { "KV type K (draft)",         [](server_prompt_cache_fingerprint_inputs & i) { i.kv_type_k_dft   += 1; } },
        { "KV type V (draft)",         [](server_prompt_cache_fingerprint_inputs & i) { i.kv_type_v_dft   += 1; } },
        { "flash attention mode",      [](server_prompt_cache_fingerprint_inputs & i) { i.flash_attn_type += 1; } },
        { "unified KV",                [](server_prompt_cache_fingerprint_inputs & i) { i.kv_unified       = 1; } },
        { "full SWA cache",            [](server_prompt_cache_fingerprint_inputs & i) { i.swa_full         = 1; } },

        // geometry
        { "target n_ctx",              [](server_prompt_cache_fingerprint_inputs & i) { i.n_ctx_tgt     += 1; } },
        { "target n_seq_max",          [](server_prompt_cache_fingerprint_inputs & i) { i.n_seq_max_tgt += 1; } },
        { "draft n_ctx",               [](server_prompt_cache_fingerprint_inputs & i) { i.n_ctx_dft     += 1; } },
        { "draft n_seq_max",           [](server_prompt_cache_fingerprint_inputs & i) { i.n_seq_max_dft += 1; } },
        { "n_parallel",                [](server_prompt_cache_fingerprint_inputs & i) { i.n_parallel    += 1; } },

        // speculative configuration
        { "speculative type",          [](server_prompt_cache_fingerprint_inputs & i) { i.spec_types = { 6 }; } },
        { "speculative type count",    [](server_prompt_cache_fingerprint_inputs & i) { i.spec_types.push_back(0); } },
        { "draft depth",               [](server_prompt_cache_fingerprint_inputs & i) { i.spec_n_max += 1; } },
    };

    for (const auto & m : mutations) {
        server_prompt_cache_fingerprint_inputs mutated = base;
        m.apply(mutated);
        require(server_prompt_cache_fingerprint(mutated) != fp_base,
                std::string("fingerprint did not change for: ") + m.name);
    }

    // the pre-fix fingerprint only knew "a drafter exists"; make sure that
    // collapsing the draft identity to a bit really is not what we do now
    {
        server_prompt_cache_fingerprint_inputs swapped = base;
        swapped.model_dft.path  = "/models/dsv4-drafter-pre0731.gguf";
        swapped.model_dft.size  = base.model_dft.size + 4096;
        swapped.model_dft.mtime = base.model_dft.mtime - 86400;
        swapped.model_dft.probe = ~base.model_dft.probe;
        require(server_prompt_cache_fingerprint(swapped) != fp_base,
                "a full drafter swap did not invalidate the cache");
    }

    printf("  ok: %zu independent identity fields each invalidate the cache\n",
           sizeof(mutations)/sizeof(mutations[0]));
}

// A file written by a different build/model/config must be refused, and the
// stale file must not linger in the discoverable namespace.
void test_stale_fingerprint_is_rejected() {
    scratch_dir dir("stale");

    spill_one(dir.path, entry_spec{});

    require(rescan_count(dir.path, TEST_FINGERPRINT ^ 1ull) == 0, "a stale-fingerprint file was admitted");
    require(count_files(dir.path) == 0, "a stale-fingerprint file was left on disk");

    printf("  ok: stale fingerprint drops the file at rescan\n");
}

//
// corrupt-file matrix
//

struct corruption {
    const char * name;
    // returns the bytes to write; may resize freely
    std::vector<uint8_t> (*apply)(std::vector<uint8_t>);
    bool admitted_by_rescan; // true = header survives, load_from_disk must catch it
};

std::vector<uint8_t> corrupt_empty(std::vector<uint8_t>) {
    return {};
}

std::vector<uint8_t> corrupt_truncated_header(std::vector<uint8_t> f) {
    f.resize(sizeof(pcache_disk_header)/2);
    return f;
}

std::vector<uint8_t> corrupt_junk(std::vector<uint8_t>) {
    std::vector<uint8_t> f(4096);
    for (size_t i = 0; i < f.size(); ++i) {
        f[i] = (uint8_t) (i*97u + 13u);
    }
    return f;
}

std::vector<uint8_t> corrupt_bad_magic(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.magic = 0xdeadbeefu;
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_future_version(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.version = PCACHE_DISK_VERSION + 7;
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_header_checksum(std::vector<uint8_t> f) {
    // flip a byte inside the header without re-sealing it
    f[16] ^= 0x40;
    return f;
}

std::vector<uint8_t> corrupt_coverage_checksum(std::vector<uint8_t> f) {
    // The header remains valid, but startup verifies the independently sealed
    // coverage prefix before admitting the entry.
    f[sizeof(pcache_disk_header) + offsetof(pcache_disk_coverage, target_rollback_tokens)] ^= 0x01;
    return f;
}

std::vector<uint8_t> corrupt_draft_swa_checksum(std::vector<uint8_t> f) {
    // draft_swa_tokens independently affects paired restore qualification; it
    // is part of the startup-verifiable coverage prefix, not an unsealed hint.
    f[sizeof(pcache_disk_header) + offsetof(pcache_disk_coverage, draft_swa_tokens)] ^= 0x01;
    return f;
}

std::vector<uint8_t> corrupt_coverage_scope(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    auto * coverage = reinterpret_cast<pcache_disk_coverage *>(
            f.data() + sizeof(pcache_disk_header));
    coverage->scope = 0xff;
    hdr.hash_coverage = coverage_hash_of(f, hdr);
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_coverage_components(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    require(hdr.n_checkpoints > 0, "fixture has no checkpoint coverage");
    auto * checkpoint = reinterpret_cast<pcache_disk_checkpoint_coverage *>(
            f.data() + sizeof(pcache_disk_header) + sizeof(pcache_disk_coverage));
    checkpoint->components = SERVER_PROMPT_RESTORE_COMPONENT_TARGET;
    hdr.hash_coverage = coverage_hash_of(f, hdr);
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_coverage_frontier(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    auto * coverage = reinterpret_cast<pcache_disk_coverage *>(
            f.data() + sizeof(pcache_disk_header));

    // Text positions are one-to-one in an SSD entry. A frontier equal to the
    // token count claims one decoded position beyond the serialized prompt.
    coverage->target_pos_min = (int32_t) hdr.n_tokens;
    coverage->target_pos_max = (int32_t) hdr.n_tokens;
    coverage->draft_pos_min  = (int32_t) hdr.n_tokens;
    coverage->draft_pos_max  = (int32_t) hdr.n_tokens;
    hdr.hash_coverage = coverage_hash_of(f, hdr);
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_absurd_n_tokens(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.n_tokens = 0x0000FFFFFFFFFFFFull; // ~281 T tokens -> 1.1 PiB of llama_token
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_n_tokens_over_context(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.n_tokens = TEST_LIMIT_TOKENS + 1; // plausible, but past the configured context
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_n_tokens_over_payload(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.n_tokens = TEST_LIMIT_TOKENS; // within the configured max, past this file
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_absurd_size_main(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.size_main = 1ull << 50; // 1 PiB
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_absurd_n_checkpoints(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.n_checkpoints = 1ull << 40;
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_length_overflow(std::vector<uint8_t> f) {
    auto hdr = header_of(f);
    hdr.size_main = UINT64_MAX - 8;
    hdr.size_drft = 64;
    reseal(f, hdr);
    return f;
}

std::vector<uint8_t> corrupt_truncated_payload(std::vector<uint8_t> f) {
    require(f.size() > 1024, "fixture too small to truncate");
    f.resize(f.size() - 977);
    return f;
}

std::vector<uint8_t> corrupt_trailing_bytes(std::vector<uint8_t> f) {
    f.insert(f.end(), 512, 0x5a);
    return f;
}

std::vector<uint8_t> corrupt_payload_bit_flip(std::vector<uint8_t> f) {
    // deep inside the target state blob: all lengths stay consistent, so only a
    // payload checksum can catch this
    f[sizeof(pcache_disk_header) + 1024] ^= 0x01;
    return f;
}

std::vector<uint8_t> corrupt_checkpoint_blob_length(std::vector<uint8_t> f) {
    const auto hdr = header_of(f);

    // first checkpoint blob length sits right after the fixed checkpoint fields
    const size_t off_ckpt = sizeof(pcache_disk_header) +
                            (size_t) hdr.size_coverage +
                            (size_t) hdr.n_tokens*sizeof(llama_token) +
                            (size_t) hdr.size_main +
                            (size_t) hdr.size_drft;
    const size_t off_len  = off_ckpt + sizeof(int64_t) + sizeof(int) + 2*sizeof(llama_pos);

    require(off_len + sizeof(uint64_t) <= f.size(), "fixture has no checkpoint blob length");

    const uint64_t absurd = 1ull << 45; // 32 TiB
    memcpy(f.data() + off_len, &absurd, sizeof(absurd));
    return f;
}

const corruption corruptions[] = {
    { "empty file",                    corrupt_empty,                    false },
    { "truncated header",              corrupt_truncated_header,         false },
    { "random junk",                   corrupt_junk,                     false },
    { "bad magic",                     corrupt_bad_magic,                false },
    { "future schema version",         corrupt_future_version,           false },
    { "header checksum mismatch",      corrupt_header_checksum,          false },
    { "coverage checksum mismatch",    corrupt_coverage_checksum,        false },
    { "draft SWA checksum mismatch",   corrupt_draft_swa_checksum,       false },
    { "invalid coverage scope",        corrupt_coverage_scope,           false },
    { "incomplete paired checkpoint",  corrupt_coverage_components,      false },
    { "impossible coverage frontier",  corrupt_coverage_frontier,        false },
    { "absurd n_tokens (2^48)",        corrupt_absurd_n_tokens,          false },
    { "n_tokens beyond the context",   corrupt_n_tokens_over_context,    false },
    { "n_tokens beyond the payload",   corrupt_n_tokens_over_payload,    false },
    { "absurd size_main (1 PiB)",      corrupt_absurd_size_main,         false },
    { "absurd n_checkpoints (2^40)",   corrupt_absurd_n_checkpoints,     false },
    { "length arithmetic overflow",    corrupt_length_overflow,          false },
    { "truncated payload",             corrupt_truncated_payload,        false },
    { "trailing bytes",                corrupt_trailing_bytes,           false },
    { "payload bit flip",              corrupt_payload_bit_flip,         true  },
    { "absurd checkpoint blob length", corrupt_checkpoint_blob_length,   true  },
};

// Every corruption must be handled without a crash, an unbounded allocation or
// a fatal error, and must never reach llama as restored state.
void test_corrupt_files_are_skipped() {
    scratch_dir src("fixture");

    entry_spec spec;
    spec.n_tokens = 96;
    spec.n_main   = 8192;
    spec.n_drft   = 2048;
    spec.n_ckpt   = 2;

    const std::vector<uint8_t> healthy = read_file(spill_one(src.path, spec));

    for (const auto & c : corruptions) {
        scratch_dir dir("corrupt");

        write_file(dir.path + "/pc-corrupt.lcpc", c.apply(healthy));

        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT);

        if (!c.admitted_by_rescan) {
            require(cache.states.empty(), std::string("rescan admitted a corrupt file: ") + c.name);
            require(count_files(dir.path) == 0, std::string("corrupt file was not cleaned up: ") + c.name);
        } else {
            require(cache.states.size() == 1, std::string("rescan dropped a header-valid file: ") + c.name);
            require(!cache.load_from_disk(cache.states.front()),
                    std::string("load_from_disk accepted a corrupt payload: ") + c.name);
            require(cache.states.front().data.size() == 0,
                    std::string("rejected load left state bytes behind: ") + c.name);
            require(cache.states.front().prompt.checkpoints.empty(),
                    std::string("rejected load left checkpoints behind: ") + c.name);
        }

        printf("  ok: %-30s handled cleanly (%s)\n", c.name,
               c.admitted_by_rescan ? "rejected on load" : "rejected on rescan");
    }
}

// The real operational risk: startup scans every discoverable file, so one bad
// file must not stop the server from coming up with the good ones.
void test_startup_survives_a_poisoned_directory() {
    scratch_dir src("fixture-mixed");

    entry_spec spec;
    spec.n_tokens = 96;
    spec.n_main   = 8192;
    spec.n_drft   = 2048;
    spec.n_ckpt   = 2;

    const std::vector<uint8_t> healthy = read_file(spill_one(src.path, spec));

    scratch_dir dir("poisoned");

    int idx = 0;
    for (const auto & c : corruptions) {
        write_file(dir.path + "/pc-bad-" + std::to_string(idx++) + ".lcpc", c.apply(healthy));
    }

    // plus non-cache clutter and an orphaned temp file from an interrupted spill
    write_file(dir.path + "/notes.txt", std::vector<uint8_t>(16, 'x'));
    write_file(dir.path + "/pc-orphan.lcpc.tmp", healthy);

    // ... and one healthy file, written last
    write_file(dir.path + "/pc-good.lcpc", healthy);

    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT);

    // the two header-valid corruptions survive the rescan by design; they are
    // caught when their payload is actually read
    require(cache.states.size() == 3,
            "poisoned directory did not resolve to the healthy entry plus the two payload-corrupt ones, got " +
            std::to_string(cache.states.size()));

    size_t n_loadable = 0;
    for (auto & state : cache.states) {
        n_loadable += cache.load_from_disk(state) ? 1 : 0;
    }
    require(n_loadable == 1, "expected exactly one loadable entry, got " + std::to_string(n_loadable));

    require(!std::filesystem::exists(dir.path + "/pc-orphan.lcpc.tmp"), "orphan temp file was not cleaned up");
    require(std::filesystem::exists(dir.path + "/notes.txt"), "rescan deleted an unrelated file");

    printf("  ok: %zu corrupt files + clutter still yield a working cache\n",
           sizeof(corruptions)/sizeof(corruptions[0]));
}

// The hash sits on the spill and restore hot paths; keep an eye on its cost.
void test_hash_throughput() {
    std::vector<uint8_t> buf(64ull*1024*1024);
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = (uint8_t) i;
    }

    const int64_t t0 = ggml_time_us();
    uint64_t h = 0;
    for (int i = 0; i < 4; ++i) {
        h = server_pcache_hash(buf.data(), buf.size(), h);
    }
    const int64_t t1 = ggml_time_us();

    const double gib_s = (4.0*buf.size() / (1024.0*1024.0*1024.0)) / ((t1 - t0)/1e6);

    require(h != 0, "hash returned zero for a large buffer");
    require(server_pcache_hash(buf.data(), buf.size(), 0) != server_pcache_hash(buf.data(), buf.size() - 1, 0),
            "hash ignores the trailing byte");

    printf("  ok: payload hash runs at %.1f GiB/s (informational)\n", gib_s);
}

} // namespace

int main() {
    struct test_case {
        const char * name;
        void (*fn)();
    };

    const test_case tests[] = {
        { "round trip",                 test_round_trip                          },
        { "tiering preserved",          test_tiering_preserved                   },
        { "restore plan boundaries",    test_restore_planner_boundaries_and_rollback },
        { "paired unequal SWA",         test_paired_unequal_swa_thresholds       },
        { "position-aware frontier",    test_position_aware_frontier_mapping     },
        { "speculative sequence clear", test_speculative_sequence_state_clear    },
        { "coverage supersession",      test_coverage_aware_supersession          },
        { "target/draft hygiene",       test_target_draft_checkpoint_hygiene     },
        { "fingerprint identity",       test_fingerprint_covers_state_identity   },
        { "stale fingerprint",          test_stale_fingerprint_is_rejected       },
        { "corrupt file matrix",        test_corrupt_files_are_skipped           },
        { "poisoned startup directory", test_startup_survives_a_poisoned_directory },
        { "hash throughput",            test_hash_throughput                     },
    };

    for (const auto & t : tests) {
        printf("== %s\n", t.name);
        try {
            t.fn();
        } catch (const std::exception & e) {
            fprintf(stderr, "FAILED (%s): %s\n", t.name, e.what());
            return 1;
        }
    }

    printf("all prompt cache hardening tests passed\n");

    return 0;
}

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
#include <chrono>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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

template <typename Predicate>
void wait_for_cache_io(server_prompt_cache & cache, Predicate predicate, const std::string & message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        cache.poll_io_completions();
        if (predicate()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cache.poll_io_completions();
    require(predicate(), message);
}

template <typename Predicate>
void wait_without_polling(Predicate predicate, const std::string & message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(predicate(), message);
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

server_prompt_cache_io_config integration_io_config(uint32_t max_jobs = 4) {
    server_prompt_cache_io_config config;
    config.max_jobs          = max_jobs;
    config.max_completions   = std::max<uint32_t>(8, max_jobs);
    config.max_pending_bytes = 64ull*1024*1024;
    config.error_cooldown_ms = 0;
    return config;
}

// spill one entry into `dir` and return the resulting file path
std::string spill_one(const std::string & dir, const entry_spec & spec, uint64_t fingerprint = TEST_FINGERPRINT) {
    server_prompt_cache cache(/*limit_size_mib =*/ 64, TEST_LIMIT_TOKENS, dir, /*disk_limit_gib =*/ 8, fingerprint);

    cache.states.push_back(make_state(spec));

    require(cache.spill_to_disk(cache.states.back()), "spill_to_disk failed");
    wait_for_cache_io(cache, [&] { return cache.states.back().on_disk(); },
            "async spill did not become durable");
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
    wait_for_cache_io(cache, [&] { return cache.states.front().on_disk(); },
            "evicted entry did not become durable");
    require(cache.states.front().on_disk(), "eviction did not spill the oldest entry to the SSD tier");
    require(!cache.states.back().on_disk(), "eviction spilled more than the oldest entry");
    require(cache.size_disk_total() > 0, "disk accounting did not grow after the spill");

    // dropping an entry must unlink its file
    const std::string file = cache.states.front().file;
    require(std::filesystem::exists(file), "spilled file missing");
    cache.drop_entry(cache.states.begin());
    wait_for_cache_io(cache, [&] { return !std::filesystem::exists(file); },
            "async drop did not remove the durable file");
    require(!std::filesystem::exists(file), "drop_entry left the file behind");

    printf("  ok: RAM->SSD spill, disk accounting and drop-unlink still behave\n");
}

void discard_decoded_components(server_prompt_cache_state & state) {
    state.data = {};
    state.prompt.checkpoints.clear();
}

void test_async_pending_visibility_and_idle_wake() {
    scratch_dir dir("async-visible");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());

    entry_spec spec;
    spec.n_tokens = 192;
    spec.n_main   = 128*1024;
    spec.n_drft   = 16*1024;
    spec.n_ckpt   = 3;
    const auto expected = make_state(spec);
    const auto expected_plan = server_prompt_make_restore_plan(
            expected.coverage, expected.prompt.tokens, 25, 25);

    cache.states.push_back(make_state(spec));
    auto & state = cache.states.back();
    server_prompt_cache_io_faults faults;
    faults.pause = server_prompt_cache_io_test_pause::before_write;
    cache.set_next_io_faults_for_test(faults);

    std::atomic<uint32_t> wakes{ 0 };
    cache.set_io_wake_callback([&] { wakes.fetch_add(1, std::memory_order_relaxed); });
    require(cache.spill_to_disk(state), "pending visibility spill admission failed");
    require(cache.test_wait_for_io_pause(state.io_job_id, faults.pause, 5000),
            "pending visibility writer did not reach barrier");

    require(!state.on_disk() && state.file.empty(), "queued entry advertised a non-durable file");
    require(state.io_payload && state.data.size() == 0 && state.prompt.checkpoints.empty(),
            "queued entry did not replace duplicate components with one immutable payload");
    require(cache.load_from_payload(state), "queued immutable payload was not readable");
    require(state.data.main == expected.data.main && state.data.drft == expected.data.drft,
            "queued immutable payload decoded different state bytes");
    const auto pending_plan = server_prompt_make_restore_plan(
            state.coverage, state.prompt.tokens, 25, 25);
    require(pending_plan.kind == expected_plan.kind &&
            pending_plan.effective_tokens == expected_plan.effective_tokens &&
            pending_plan.checkpoint_index == expected_plan.checkpoint_index,
            "queued payload changed authoritative restore coverage");
    discard_decoded_components(state);

    cache.test_release_io_pause(state.io_job_id);
    wait_without_polling([&] { return wakes.load(std::memory_order_relaxed) != 0; },
            "worker completion did not issue the idle wake signal");
    require(!state.on_disk() && state.io_payload,
            "wake callback reconciled owner state from the worker thread");
    cache.poll_io_completions();
    require(state.on_disk() && !state.io_payload,
            "owner poll did not publish the durable transition and release fallback RAM");

    const std::string durable_path = state.file;
    require(cache.shutdown_io(server_prompt_cache_io_shutdown_mode::graceful) ==
                    server_prompt_cache_io_status::ok,
            "source cache did not release directory ownership before restart");
    server_prompt_cache restarted(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());
    require(restarted.states.size() == 1 && restarted.states.front().file == durable_path,
            "restart did not adopt the durable async publication");
    const auto stats = restarted.io_stats();
    require(stats.initial_disk_bytes == 0 && stats.disk_committed_bytes == restarted.states.front().size_disk,
            "startup adoption did not transfer aggregate bytes into exact ownership");
    const auto restarted_plan = server_prompt_make_restore_plan(
            restarted.states.front().coverage, restarted.states.front().prompt.tokens, 25, 25);
    require(restarted.states.front().coverage.scope == expected.coverage.scope &&
            restarted_plan.kind == expected_plan.kind &&
            restarted_plan.effective_tokens == expected_plan.effective_tokens &&
            restarted_plan.checkpoint_index == expected_plan.checkpoint_index,
            "startup adoption changed authoritative restore coverage");

    printf("  ok: queued memory stays readable, idle wake is narrow, and owner poll publishes durability\n");
}

void test_async_saturation_keeps_fallback() {
    scratch_dir dir("async-saturation");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config(/*max_jobs=*/1));

    entry_spec first_spec;
    first_spec.seed   = 31;
    first_spec.n_main = 256*1024;
    entry_spec second_spec = first_spec;
    second_spec.seed = 32;
    const auto second_expected = make_state(second_spec);

    cache.states.push_back(make_state(first_spec));
    cache.states.push_back(make_state(second_spec));
    auto first  = cache.states.begin();
    auto second = std::next(first);

    server_prompt_cache_io_faults faults;
    faults.pause = server_prompt_cache_io_test_pause::before_write;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(*first), "saturation fixture first admission failed");
    require(cache.test_wait_for_io_pause(first->io_job_id, faults.pause, 5000),
            "saturation fixture did not hold the writer");

    require(!cache.spill_to_disk(*second), "job-saturated second write was unexpectedly admitted");
    require(second->io_status == server_prompt_cache_io_status::job_limit &&
            second->io_payload && second->file.empty(),
            "job saturation discarded or falsely published the second fallback");
    require(cache.load_from_payload(*second), "saturated fallback was not restorable from memory");
    require(second->data.main == second_expected.data.main && second->data.drft == second_expected.data.drft,
            "saturated fallback decoded different state bytes");
    discard_decoded_components(*second);

    cache.test_release_io_pause(first->io_job_id);
    wait_for_cache_io(cache, [&] { return first->on_disk(); }, "first saturated write did not become durable");
    require(cache.spill_to_disk(*second), "second fallback did not retry after capacity release");
    wait_for_cache_io(cache, [&] { return second->on_disk(); }, "retried second write did not become durable");

    printf("  ok: job saturation retains one immutable, restorable RAM fallback and retries it\n");
}

void test_async_stale_generation_cannot_publish() {
    scratch_dir dir("async-stale-generation");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());

    entry_spec spec;
    spec.seed   = 41;
    spec.n_main = 192*1024;
    cache.states.push_back(make_state(spec));
    auto & state = cache.states.back();

    server_prompt_cache_io_faults faults;
    faults.pause = server_prompt_cache_io_test_pause::after_publication_fence;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(state), "stale generation first admission failed");
    const uint64_t first_generation = state.io_generation;
    const uint64_t first_job        = state.io_job_id;
    require(cache.test_wait_for_io_pause(first_job, faults.pause, 5000),
            "stale generation did not reach post-fence barrier");

    // The cache normally retries only after a terminal completion. This
    // deterministic seam models an owner-side supersession while generation N
    // is fenced, so the engine must retire N and let only N+1 own the path.
    state.io_lifecycle = server_prompt_cache_io_lifecycle::write_failed;
    require(cache.spill_to_disk(state), "superseding generation admission failed");
    require(state.io_generation == first_generation + 1,
            "superseding generation was not strictly monotone");
    const auto expected_payload = state.io_payload;
    cache.test_release_io_pause(first_job);
    wait_for_cache_io(cache, [&] { return state.on_disk(); },
            "superseding generation did not become durable");
    require(state.io_generation == first_generation + 1 && state.file == state.io_path,
            "stale completion published over the current generation");
    require(expected_payload && read_file(state.file) == *expected_payload,
            "durable current generation bytes are unreadable");
    require(cache.dash_io_stale != 0, "stale generation completion was not observed by owner polling");

    printf("  ok: fenced generation N cannot publish state or unlink generation N+1\n");
}

void test_async_uncertainty_reconciles_from_memory() {
    scratch_dir dir("async-uncertain");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config(/*max_jobs=*/1));

    entry_spec spec;
    spec.seed   = 51;
    spec.n_main = 160*1024;
    const auto expected = make_state(spec);
    cache.states.push_back(make_state(spec));
    auto & state = cache.states.back();

    server_prompt_cache_io_faults faults;
    faults.fail_directory_fsync_errno = EIO;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(state), "uncertain write admission failed");

    wait_without_polling([&] { return cache.io_stats().completion_backlog != 0; },
            "uncertain write did not produce a completion");
    require(!state.on_disk() && state.io_payload,
            "worker exposed an uncertain file before owner reconciliation");
    require(cache.load_from_payload(state) && state.data.main == expected.data.main,
            "uncertain generation lost its memory-authoritative fallback");
    discard_decoded_components(state);

    // Occupy the sole job slot before the owner consumes the uncertainty.
    // Cleanup admission must retain io_reconcile_pending and retry later; it
    // must never treat capacity rejection as proof that the path is gone.
    entry_spec blocker_spec = spec;
    blocker_spec.seed = 53;
    cache.states.push_back(make_state(blocker_spec));
    auto & blocker = cache.states.back();
    faults = {};
    faults.pause = server_prompt_cache_io_test_pause::before_write;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(blocker), "uncertainty cleanup blocker admission failed");
    require(cache.test_wait_for_io_pause(blocker.io_job_id, faults.pause, 5000),
            "uncertainty cleanup blocker did not pause");

    cache.poll_io_completions();
    require(!state.on_disk() && state.io_payload && state.io_reconcile_pending && state.io_generation == 1,
            "cleanup saturation discarded fallback, path tombstone, or generation fence");
    cache.test_release_io_pause(blocker.io_job_id);
    wait_for_cache_io(cache, [&] { return state.on_disk(); },
            "uncertain generation did not clean and republish durably");
    require(state.io_generation >= 2 && cache.dash_io_uncertain == 1,
            "uncertain reconciliation did not use a newer fenced generation");

    // A missing generation has no future cleanup completion. Exercise the
    // synchronous resolved path directly so io_reconcile_pending cannot stick.
    entry_spec missing_spec = spec;
    missing_spec.seed = 52;
    cache.states.push_back(make_state(missing_spec));
    auto & missing = cache.states.back();
    faults = {};
    faults.fail_after_bytes = 0;
    faults.fail_write_errno = EIO;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(missing), "missing-generation fixture admission failed");
    wait_for_cache_io(cache,
            [&] { return missing.io_lifecycle == server_prompt_cache_io_lifecycle::write_failed; },
            "missing-generation fixture did not reach a retryable failure");
    missing.io_generation += 7; // deliberately names no engine record
    const uint64_t missing_generation = missing.io_generation;
    missing.io_lifecycle = server_prompt_cache_io_lifecycle::commit_uncertain;
    cache.test_reconcile_uncertain_for_test(missing);
    require(!missing.io_reconcile_pending && missing.io_generation == missing_generation + 1,
            "not_found reconciliation remained stuck without a future completion");
    wait_for_cache_io(cache, [&] { return missing.on_disk(); },
            "not_found reconciliation did not republish a newer generation");

    printf("  ok: directory-fsync uncertainty stays memory-authoritative, cleans, and republishes\n");
}

void test_active_uncertainty_cooldown_wakes_idle_retry() {
    scratch_dir dir("async-uncertain-cooldown");
    auto io_config = integration_io_config(/*max_jobs=*/1);
    io_config.error_cooldown_ms = 1000;
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT, io_config);

    entry_spec spec;
    spec.seed   = 54;
    spec.n_main = 160*1024;
    cache.states.push_back(make_state(spec));
    auto & state = cache.states.back();

    std::atomic<uint32_t> wakes{ 0 };
    cache.set_io_wake_callback([&] { wakes.fetch_add(1, std::memory_order_relaxed); });
    server_prompt_cache_io_faults faults;
    faults.fail_directory_fsync_errno = EIO;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(state), "cooldown uncertainty write admission failed");

    wait_without_polling([&] { return cache.io_stats().completion_backlog != 0; },
            "cooldown uncertainty write produced no completion");
    cache.poll_io_completions();
    require(state.io_reconcile_pending && state.io_payload && !state.on_disk(),
            "cooldown uncertainty lost its cleanup tombstone or RAM fallback");

    wait_without_polling([&] { return cache.io_stats().completion_backlog != 0; },
            "cooldown uncertainty cleanup produced no completion");
    wait_without_polling([&] { return wakes.load(std::memory_order_relaxed) >= 2; },
            "cooldown uncertainty cleanup callback did not finish");
    cache.poll_io_completions();
    require(!state.io_reconcile_pending && state.io_lifecycle == server_prompt_cache_io_lifecycle::write_failed &&
                    state.io_status == server_prompt_cache_io_status::cooldown && state.io_payload &&
                    state.io_generation == 1,
            "resolved uncertainty bypassed cooldown or discarded its retryable fallback");

    const uint64_t rejections_before_polling = cache.dash_io_rejections;
    for (int repetition = 0; repetition < 32; ++repetition) {
        cache.poll_io_completions();
    }
    require(cache.dash_io_rejections == rejections_before_polling &&
                    state.io_status == server_prompt_cache_io_status::cooldown && state.io_generation == 1,
            "active owner polling retried admission or inflated rejection telemetry before cooldown expiry");

    const uint32_t wakes_before_deadline = wakes.load(std::memory_order_relaxed);
    wait_without_polling(
            [&] { return wakes.load(std::memory_order_relaxed) > wakes_before_deadline; },
            "cooldown deadline did not wake the idle owner");
    require(cache.io_stats().completion_backlog == 0,
            "cooldown retry wake was accidentally supplied by an unrelated completion");

    cache.poll_io_completions();
    require(state.io_in_flight() && state.io_generation == 2 && state.io_payload,
            "deadline owner turn did not submit the same payload as a newer generation");
    wait_for_cache_io(cache, [&] { return state.on_disk(); },
            "cooldown retry did not become durable without unrelated traffic");
    require(state.io_generation == 2 && !state.io_payload,
            "durable cooldown retry retained fallback RAM or changed generations unexpectedly");

    printf("  ok: active uncertainty schedules one idle owner wake and republishes after write cooldown\n");
}

void test_async_eviction_during_write_and_tombstone_retry() {
    scratch_dir dir("async-retire");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config(/*max_jobs=*/1));

    entry_spec durable_spec;
    durable_spec.seed = 61;
    cache.states.push_back(make_state(durable_spec));
    require(cache.spill_to_disk(cache.states.back()), "durable retirement fixture spill failed");
    wait_for_cache_io(cache, [&] { return cache.states.front().on_disk(); },
            "durable retirement fixture did not publish");
    const std::string durable_path = cache.states.front().file;
    const uint64_t durable_bytes = cache.states.front().size_disk;

    entry_spec writing_spec = durable_spec;
    writing_spec.seed   = 62;
    writing_spec.n_main = 256*1024;
    cache.states.push_back(make_state(writing_spec));
    auto writing = std::prev(cache.states.end());
    server_prompt_cache_io_faults faults;
    faults.pause = server_prompt_cache_io_test_pause::before_write;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(*writing), "retirement saturation write failed");
    require(cache.test_wait_for_io_pause(writing->io_job_id, faults.pause, 5000),
            "retirement saturation write did not pause");

    cache.drop_entry(cache.states.begin());
    require(!cache.retired_states.empty() && cache.retired_states.front().io_retire_retry,
            "cleanup saturation did not retain a retryable retirement tombstone");
    require(std::filesystem::exists(durable_path), "quota was reclaimed before cleanup admission");
    require(cache.io_stats().disk_committed_bytes >= durable_bytes,
            "engine accounting released committed bytes before cleanup completion");

    const uint64_t writing_job = writing->io_job_id;
    cache.test_release_io_pause(writing_job);
    wait_for_cache_io(cache,
            [&] { return !std::filesystem::exists(durable_path) && cache.retired_states.empty(); },
            "retirement tombstone did not retry exact-path cleanup after capacity release");

    // A separate pre-fence eviction proves that logical removal can erase the
    // list entry immediately while the worker safely cancels its private temp.
    entry_spec evicted_spec = durable_spec;
    evicted_spec.seed = 63;
    cache.states.push_back(make_state(evicted_spec));
    auto evicted = std::prev(cache.states.end());
    faults.pause = server_prompt_cache_io_test_pause::before_publication_fence;
    cache.set_next_io_faults_for_test(faults);
    require(cache.spill_to_disk(*evicted), "pre-fence eviction fixture admission failed");
    require(cache.test_wait_for_io_pause(evicted->io_job_id, faults.pause, 5000),
            "pre-fence eviction fixture did not pause");
    const std::string evicted_path = evicted->io_path;
    cache.drop_entry(evicted);
    wait_for_cache_io(cache, [&] {
        return std::none_of(cache.retired_states.begin(), cache.retired_states.end(),
                [&](const auto & retired) { return retired.io_path == evicted_path; });
    }, "pre-fence eviction tombstone did not complete");
    require(!std::filesystem::exists(evicted_path), "pre-fence eviction published a retired path");

    printf("  ok: eviction cancels pre-fence work and saturated cleanup retries without early quota release\n");
}

void test_post_fence_retirement_cleanup_failures_keep_tombstones() {
    const auto exercise = [](const std::string & tag, int unlink_error, int directory_sync_error) {
        scratch_dir dir("post-fence-" + tag);
        auto io_config = integration_io_config();
        io_config.error_cooldown_ms = 60*1000;
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
                io_config);

        entry_spec spec;
        spec.seed   = unlink_error != 0 ? 64 : 65;
        spec.n_main = 192*1024;
        cache.states.push_back(make_state(spec));
        auto entry = cache.states.begin();

        server_prompt_cache_io_faults faults;
        faults.pause                      = server_prompt_cache_io_test_pause::after_publication_fence;
        faults.fail_unlink_errno          = unlink_error;
        faults.fail_directory_fsync_errno = directory_sync_error;
        cache.set_next_io_faults_for_test(faults);
        require(cache.spill_to_disk(*entry), tag + ": write admission failed");
        const uint64_t job_id = entry->io_job_id;
        const std::string path = entry->io_path;
        require(cache.test_wait_for_io_pause(job_id, faults.pause, 5000),
                tag + ": write did not cross the publication fence");

        cache.drop_entry(entry);
        require(cache.states.empty() && cache.retired_states.size() == 1,
                tag + ": post-fence logical deletion lost its tombstone");
        cache.test_release_io_pause(job_id);

        wait_without_polling([&] { return cache.io_stats().completion_backlog != 0; },
                tag + ": failed inline retirement produced no owner completion");
        require(cache.io_stats().disk_uncertain_bytes != 0,
                tag + ": failed inline retirement released disk accounting early");
        if (unlink_error != 0) {
            require(std::filesystem::exists(path), tag + ": injected unlink failure unexpectedly removed path");
        } else {
            require(!std::filesystem::exists(path),
                    tag + ": directory-fsync uncertainty did not physically unlink path");
        }

        cache.poll_io_completions();
        require(cache.retired_states.size() == 1,
                tag + ": failed post-fence cleanup completion cleared the tombstone");

        wait_for_cache_io(cache, [&] { return cache.retired_states.empty(); },
                tag + ": exact-path cleanup retry did not resolve the tombstone");
        require(!std::filesystem::exists(path) && cache.io_stats().disk_uncertain_bytes == 0,
                tag + ": resolved cleanup retained a file or uncertain charge");
    };

    exercise("unlink", EIO, 0);
    exercise("directory-fsync", 0, EIO);
    printf("  ok: post-fence unlink/fsync failures retry cleanup during write cooldown and retain tombstones until proven\n");
}

void test_async_shutdown_modes_and_corrupt_memory() {
    {
        scratch_dir dir("async-graceful");
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
                integration_io_config());
        entry_spec spec;
        spec.seed = 71;
        cache.states.push_back(make_state(spec));
        require(cache.spill_to_disk(cache.states.back()), "graceful shutdown write admission failed");
        require(cache.shutdown_io(server_prompt_cache_io_shutdown_mode::graceful) ==
                        server_prompt_cache_io_status::ok,
                "graceful shutdown failed");
        require(cache.states.back().on_disk(), "graceful shutdown did not drain accepted publication");
    }

    {
        scratch_dir dir("async-fast");
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
                integration_io_config());
        entry_spec spec;
        spec.seed = 72;
        cache.states.push_back(make_state(spec));
        auto & state = cache.states.back();
        server_prompt_cache_io_faults faults;
        faults.pause = server_prompt_cache_io_test_pause::before_write;
        cache.set_next_io_faults_for_test(faults);
        require(cache.spill_to_disk(state), "fast shutdown write admission failed");
        require(cache.test_wait_for_io_pause(state.io_job_id, faults.pause, 5000),
                "fast shutdown write did not pause");
        require(cache.shutdown_io(server_prompt_cache_io_shutdown_mode::fast) ==
                        server_prompt_cache_io_status::ok,
                "fast shutdown failed");
        require(!state.on_disk() && state.io_payload && state.has_ram_fallback(),
                "fast shutdown discarded fallback or exposed a cancelled file");
    }

    {
        scratch_dir dir("async-corrupt-memory");
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
                integration_io_config());
        entry_spec spec;
        spec.seed = 73;
        cache.states.push_back(make_state(spec));
        auto & state = cache.states.back();
        server_prompt_cache_io_faults faults;
        faults.pause = server_prompt_cache_io_test_pause::before_write;
        cache.set_next_io_faults_for_test(faults);
        require(cache.spill_to_disk(state), "corrupt-memory fixture admission failed");
        require(cache.test_wait_for_io_pause(state.io_job_id, faults.pause, 5000),
                "corrupt-memory fixture did not pause");
        const uint64_t job_id = state.io_job_id;
        auto mutable_payload = std::const_pointer_cast<std::vector<uint8_t>>(state.io_payload);
        (*mutable_payload)[sizeof(pcache_disk_header) + 7] ^= 0x80;
        require(!cache.load_from_payload(state), "corrupt pending memory passed LCPC verification");
        cache.drop_entry(cache.states.begin());
        cache.test_release_io_pause(job_id);
    }

    printf("  ok: graceful drains, fast cancels safely, and corrupt pending memory fails closed\n");
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

    const auto zero_width_equality = server_prompt_build_restore_coverage(
        boundary_bad, server_prompt_restore_scope::target_only, restore_domain(63, 94, 0, 0, false));
    plan = server_prompt_make_restore_plan(zero_width_equality, boundary_bad.tokens, 64, 64);
    require(plan.kind == server_prompt_restore_kind::unusable,
            "zero-width direct state at the landing threshold was admitted");

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

    // Exact serialized SWA coverage starts at the threshold. The key at that
    // position is already masked for the next token, so equality is sufficient
    // for direct generated-residue restore.
    coverage =
        server_prompt_build_restore_coverage(prompt, server_prompt_restore_scope::target_and_draft,
                                             restore_domain(98, 98, 32, 0), restore_domain(69, 98, 32, 30, false));
    plan = server_prompt_make_restore_plan(coverage, prompt.tokens, 100, 104);
    require(plan.kind == server_prompt_restore_kind::direct && plan.effective_tokens == 99,
            "exact serialized draft SWA boundary rejected generated residue");

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

void test_startup_cleanup_failure_disables_live_writes() {
    scratch_dir dir("startup-cleanup-failure");
    const std::string corrupt = dir.path + "/pc-unremovable.lcpc";
    write_file(corrupt, std::vector<uint8_t>(8, 0x5a));

    std::error_code ec;
    std::filesystem::permissions(
            dir.path,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
            std::filesystem::perm_options::replace,
            ec);
    require(!ec, "could not make startup cache directory read-only");

    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());

    // Restore cleanup access before any assertion can throw and before the
    // scratch fixture recursively removes the directory.
    std::filesystem::permissions(
            dir.path, std::filesystem::perms::owner_all,
            std::filesystem::perm_options::replace, ec);
    require(!ec, "could not restore startup cache directory permissions");

    require(std::filesystem::exists(corrupt),
            "startup cleanup-failure fixture unexpectedly removed the corrupt file");
    require(!cache.disk_startup_reconciled && !cache.disk_io,
            "unowned startup LCPC bytes did not disable live persistence");

    entry_spec spec;
    spec.seed = 91;
    cache.states.push_back(make_state(spec));
    require(!cache.spill_to_disk(cache.states.back()) && cache.states.back().has_ram_fallback(),
            "fail-closed startup state discarded the RAM fallback or admitted a write");

    printf("  ok: undeletable invalid startup LCPC disables writes instead of escaping quota ownership\n");
}

void test_startup_nonregular_cache_path_disables_live_writes() {
    scratch_dir dir("startup-nonregular-cache-path");
    const std::string nonregular = dir.path + "/pc-nonregular.lcpc";

    std::error_code ec;
    std::filesystem::create_directory(nonregular, ec);
    require(!ec, "could not create non-regular startup cache fixture");

    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());

    require(std::filesystem::is_directory(nonregular),
            "startup scan unexpectedly removed the non-regular cache path");
    require(!cache.disk_startup_reconciled && !cache.disk_io,
            "non-regular startup LCPC path did not disable live persistence");

    entry_spec spec;
    spec.seed = 92;
    cache.states.push_back(make_state(spec));
    require(!cache.spill_to_disk(cache.states.back()) && cache.states.back().has_ram_fallback(),
            "fail-closed non-regular startup state discarded the RAM fallback or admitted a write");

    printf("  ok: non-regular startup LCPC disables writes instead of escaping quota ownership\n");
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

struct fake_restore_sequence {
    bool    occupied = false;
    uint8_t marker   = 0;
};

struct fake_restore_engine {
    fake_restore_sequence target;
    fake_restore_sequence draft;

    std::vector<std::pair<uint8_t, server_prompt_restore_coverage>> coverage_by_marker;
    std::vector<uint8_t> fail_target;
    std::vector<uint8_t> fail_draft;
    std::vector<uint8_t> target_attempts;
    std::vector<uint8_t> draft_attempts;
    std::vector<uint8_t> captured_markers;
    size_t clears_target = 0;
    size_t clears_draft  = 0;

    server_prompt_restore_coverage coverage_for(uint8_t marker) const {
        const auto it = std::find_if(coverage_by_marker.begin(), coverage_by_marker.end(),
                [&](const auto & item) { return item.first == marker; });
        return it == coverage_by_marker.end() ?
                server_prompt_restore_coverage {} : it->second;
    }
};

llama_context * fake_context(fake_restore_sequence & sequence) {
    return reinterpret_cast<llama_context *>(&sequence);
}

server_prompt_cache::restore_test_hooks fake_restore_hooks(fake_restore_engine & engine) {
    server_prompt_cache::restore_test_hooks hooks;
    hooks.capture_coverage = [&](const server_prompt &, llama_context *, llama_context *, llama_seq_id) {
        if (!engine.target.occupied) {
            return server_prompt_restore_coverage {};
        }
        engine.captured_markers.push_back(engine.target.marker);
        return engine.coverage_for(engine.target.marker);
    };
    hooks.get_size = [](llama_context * ctx, llama_seq_id) {
        const auto & sequence = *reinterpret_cast<fake_restore_sequence *>(ctx);
        return sequence.occupied ? size_t(1) : size_t(0);
    };
    hooks.get_data = [](llama_context * ctx, uint8_t * data, size_t size, llama_seq_id) {
        const auto & sequence = *reinterpret_cast<fake_restore_sequence *>(ctx);
        if (!sequence.occupied || size != 1) {
            return size_t(0);
        }
        data[0] = sequence.marker;
        return size_t(1);
    };
    hooks.set_data = [&](llama_context * ctx, const uint8_t * data, size_t size, llama_seq_id) {
        auto & sequence = *reinterpret_cast<fake_restore_sequence *>(ctx);
        if (size != 1) {
            return size_t(0);
        }

        const uint8_t marker = data[0];
        const bool is_target = &sequence == &engine.target;
        auto & attempts = is_target ? engine.target_attempts : engine.draft_attempts;
        const auto & failures = is_target ? engine.fail_target : engine.fail_draft;
        attempts.push_back(marker);
        if (std::find(failures.begin(), failures.end(), marker) != failures.end()) {
            return size_t(0);
        }

        sequence.occupied = true;
        sequence.marker   = marker;
        return size;
    };
    hooks.clear = [&](llama_context * ctx, llama_seq_id) {
        auto & sequence = *reinterpret_cast<fake_restore_sequence *>(ctx);
        sequence.occupied = false;
        sequence.marker   = 0;
        if (&sequence == &engine.target) {
            ++engine.clears_target;
        } else {
            ++engine.clears_draft;
        }
    };
    return hooks;
}

constexpr int RESTORE_TEST_TOKEN_SEED = 701;

server_prompt_cache_state make_direct_restore_state(
        size_t n_tokens,
        uint8_t target_marker,
        bool paired = false,
        uint8_t draft_marker = 0) {
    entry_spec spec;
    spec.n_tokens = n_tokens;
    spec.n_main   = 1;
    spec.n_drft   = paired ? 1 : 0;
    spec.n_ckpt   = 0;
    spec.seed     = RESTORE_TEST_TOKEN_SEED;

    auto state = make_state(spec);
    state.data.main = { target_marker };
    state.data.drft = paired ? std::vector<uint8_t> { draft_marker } : std::vector<uint8_t> {};

    const auto target = restore_domain(0, (llama_pos) n_tokens - 2, 0, 0, false);
    const auto draft  = paired ? target : server_prompt_restore_domain {};
    state.coverage = server_prompt_build_restore_coverage(
            state.prompt,
            paired ? server_prompt_restore_scope::target_and_draft
                   : server_prompt_restore_scope::target_only,
            target,
            draft);
    return state;
}

void test_live_alternative_ranking() {
    const server_tokens request(make_tokens(140, RESTORE_TEST_TOKEN_SEED), false);

    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        const auto resident = make_direct_restore_state(50, 10);
        cache.states.push_back(make_direct_restore_state(100, 30));

        fake_restore_engine engine;
        engine.target = { true, 10 };
        engine.coverage_by_marker = {
            { 10, resident.coverage },
            { 30, cache.states.front().coverage },
        };
        cache.set_restore_test_hooks(fake_restore_hooks(engine));

        server_prompt prompt = resident.prompt.clone();
        const auto result = cache.load_best(
                prompt, request, fake_context(engine.target), nullptr, 0, 70);
        require(result == server_prompt_cache_load_result::cache_entry,
                "a longer RAM cache entry lost to the live alternative");
        require(prompt.tokens.size() == 100 && engine.target.marker == 30,
                "the longer RAM cache entry was not installed");
        require(cache.dash_last_load.source == 1 && cache.dash_last_load.n_tokens == 99,
                "the RAM winner was not reported with its effective coverage");
    }

    {
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
        const auto resident = make_direct_restore_state(50, 10);
        cache.states.push_back(make_direct_restore_state(71, 30));

        fake_restore_engine engine;
        engine.target = { true, 10 };
        engine.coverage_by_marker = {
            { 10, resident.coverage },
            { 30, cache.states.front().coverage },
        };
        cache.set_restore_test_hooks(fake_restore_hooks(engine));

        server_prompt prompt = resident.prompt.clone();
        const auto result = cache.load_best(
                prompt, request, fake_context(engine.target), nullptr, 0, 70);
        require(result == server_prompt_cache_load_result::alternative,
                "a cache tie did not keep the cheaper live alternative");
        require(prompt.tokens.size() == 50 && engine.target.marker == 10 &&
                        engine.target_attempts.empty(),
                "ranking a live alternative destructively touched the destination");
        require(cache.dash_last_load.source == 3 && cache.dash_last_load.n_tokens == 70,
                "the live alternative was not reported with its effective coverage");
    }

    {
        scratch_dir dir("live-alternative-ranking");
        server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
                integration_io_config());
        cache.states.push_back(make_direct_restore_state(90, 20));
        require(cache.spill_to_disk(cache.states.back()),
                "could not spill the longer SSD candidate");
        wait_for_cache_io(cache, [&] { return cache.states.front().on_disk(); },
                "longer SSD candidate did not become durable");

        const auto resident = make_direct_restore_state(50, 10);
        fake_restore_engine engine;
        engine.target = { true, 10 };
        engine.coverage_by_marker = {
            { 10, resident.coverage },
            { 20, cache.states.front().coverage },
        };
        cache.set_restore_test_hooks(fake_restore_hooks(engine));

        server_prompt prompt = resident.prompt.clone();
        const auto result = cache.load_best(
                prompt, request, fake_context(engine.target), nullptr, 0, 70);
        require(result == server_prompt_cache_load_result::cache_entry,
                "a longer SSD cache entry lost to the live alternative");
        require(prompt.tokens.size() == 90 && engine.target.marker == 20,
                "the longer SSD cache entry was not installed");
        require(cache.dash_last_load.source == 2 && cache.dash_last_load.n_tokens == 89,
                "the SSD winner was not reported with its effective coverage");
    }

    printf("  ok: resident, live donor, RAM, and SSD sources rank by effective coverage\n");
}

void test_failed_candidate_restores_target_only_resident() {
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);

    const auto resident = make_direct_restore_state(50, 10);
    auto bad = make_direct_restore_state(100, 30);
    cache.states.push_back(std::move(bad));

    fake_restore_engine engine;
    engine.target = { true, 10 };
    engine.draft  = { true, 99 };
    engine.coverage_by_marker = {
        { 10, resident.coverage },
        { 30, cache.states.front().coverage },
    };
    engine.fail_target = { 30 };
    cache.set_restore_test_hooks(fake_restore_hooks(engine));

    server_prompt prompt = resident.prompt.clone();
    const server_tokens request(make_tokens(140, RESTORE_TEST_TOKEN_SEED), false);
    require(cache.load(prompt, request, fake_context(engine.target), fake_context(engine.draft), 0),
            "bad target-only candidate destroyed the resident fallback");
    require(prompt.tokens.size() == resident.prompt.tokens.size(),
            "resident fallback changed the published prompt");
    require(engine.target.occupied && engine.target.marker == 10,
            "resident target bytes were not restored after candidate failure");
    require(!engine.draft.occupied,
            "target-only rollback resurrected stale draft state");
    require(engine.target_attempts == std::vector<uint8_t>({ 30, 10 }),
            "restore did not attempt the bad candidate then the resident snapshot");
    require(cache.states.empty() && cache.dash_drops == 1,
            "bad target-only entry was not retired");
    require(cache.dash_last_load.source == 0 &&
                    cache.dash_last_load.n_tokens == 49 &&
                    cache.dash_last_load.restored_scope == server_prompt_restore_scope::target_only,
            "resident rollback was not reported as reusable target-only coverage");

    printf("  ok: failed target-only candidate restores resident reuse and clears stale draft state\n");
}

void test_failed_paired_draft_restores_paired_resident() {
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);

    const auto resident = make_direct_restore_state(50, 10, true, 11);
    auto bad = make_direct_restore_state(100, 40, true, 41);
    cache.states.push_back(std::move(bad));

    fake_restore_engine engine;
    engine.target = { true, 10 };
    engine.draft  = { true, 11 };
    engine.coverage_by_marker = {
        { 10, resident.coverage },
        { 40, cache.states.front().coverage },
    };
    engine.fail_draft = { 41 };
    cache.set_restore_test_hooks(fake_restore_hooks(engine));

    server_prompt prompt = resident.prompt.clone();
    const server_tokens request(make_tokens(140, RESTORE_TEST_TOKEN_SEED), false);
    require(cache.load(prompt, request, fake_context(engine.target), fake_context(engine.draft), 0),
            "draft restore failure destroyed the paired resident fallback");
    require(engine.target.occupied && engine.target.marker == 10 &&
                    engine.draft.occupied && engine.draft.marker == 11,
            "paired resident target/draft state was not restored atomically");
    require(engine.target_attempts == std::vector<uint8_t>({ 40, 10 }) &&
                    engine.draft_attempts == std::vector<uint8_t>({ 41, 11 }),
            "paired transaction did not clear and restore both domains");
    require(std::find(engine.captured_markers.begin(), engine.captured_markers.end(), 40) ==
                    engine.captured_markers.end(),
            "candidate with a failed draft domain was coverage-validated or published");
    require(cache.states.empty() && cache.dash_last_load.source == 0,
            "failed paired entry survived or displaced the resident report");

    printf("  ok: failed draft restore cannot publish partial target state and paired resident survives\n");
}

void test_multiple_bad_candidates_fall_through_to_ssd() {
    scratch_dir dir("transactional-fallback");
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, dir.path, 8, TEST_FINGERPRINT,
            integration_io_config());

    auto durable = make_direct_restore_state(90, 20);
    cache.states.push_back(std::move(durable));
    require(cache.spill_to_disk(cache.states.back()), "could not spill valid fallback candidate");
    wait_for_cache_io(cache, [&] { return cache.states.front().on_disk(); },
            "valid fallback candidate did not become durable");

    auto mismatch = make_direct_restore_state(110, 31);
    const auto mismatch_coverage = mismatch.coverage;
    cache.states.push_back(std::move(mismatch));
    auto structural = make_direct_restore_state(120, 30);
    const auto structural_coverage = structural.coverage;
    cache.states.push_back(std::move(structural));

    const auto resident = make_direct_restore_state(50, 10);
    const auto degraded = make_direct_restore_state(70, 31);

    fake_restore_engine engine;
    engine.target = { true, 10 };
    engine.coverage_by_marker = {
        { 10, resident.coverage },
        { 20, cache.states.front().coverage },
        { 30, structural_coverage },
        { 31, degraded.coverage },
    };
    engine.fail_target = { 30 };
    cache.set_restore_test_hooks(fake_restore_hooks(engine));

    server_prompt prompt = resident.prompt.clone();
    const server_tokens request(make_tokens(150, RESTORE_TEST_TOKEN_SEED), false);
    require(cache.load(prompt, request, fake_context(engine.target), nullptr, 0),
            "multiple bad candidates prevented the valid SSD fallback");
    require(prompt.tokens.size() == 90 && engine.target.occupied && engine.target.marker == 20,
            "best still-valid SSD fallback was not installed");
    require(engine.target_attempts == std::vector<uint8_t>({ 30, 31, 20 }),
            "candidates were not attempted in descending effective-coverage order");
    require(cache.states.size() == 1 && cache.states.front().on_disk(),
            "bad candidates survived or the reusable SSD index was consumed");
    require(cache.dash_drops == 2 && cache.dash_last_load.source == 2,
            "candidate retirement or SSD hit reporting is incorrect");

    const auto attempts_before_repeat = engine.target_attempts;
    require(cache.load(prompt, request, fake_context(engine.target), nullptr, 0),
            "resident SSD result was not reusable on a repeated consultation");
    require(engine.target_attempts == attempts_before_repeat && cache.dash_drops == 2,
            "retired bad candidates were retried on a repeated consultation");
    require(mismatch_coverage.target.pos_max > degraded.coverage.target.pos_max,
            "coverage-mismatch fixture does not advertise more state than it restores");

    printf("  ok: repeated bad RAM candidates retire once and fall through to a valid SSD prefix\n");
}

void test_failed_candidate_without_fallback_misses_closed() {
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
    auto bad = make_direct_restore_state(100, 30);
    cache.states.push_back(std::move(bad));

    fake_restore_engine engine;
    engine.coverage_by_marker = {
        { 30, cache.states.front().coverage },
    };
    engine.fail_target = { 30 };
    cache.set_restore_test_hooks(fake_restore_hooks(engine));

    server_prompt prompt;
    const server_tokens request(make_tokens(140, RESTORE_TEST_TOKEN_SEED), false);
    require(!cache.load(prompt, request, fake_context(engine.target), fake_context(engine.draft), 0),
            "failed candidate with no resident or alternate fallback reported success");
    require(!engine.target.occupied && !engine.draft.occupied,
            "no-fallback failure left target or draft state published");
    require(cache.states.empty() && cache.dash_drops == 1 && cache.dash_misses == 1,
            "no-fallback failure did not retire the bad entry and record one safe miss");

    printf("  ok: true no-fallback restore failures retire the entry and miss with empty domains\n");
}

void test_discard_unpopulated_allocation() {
    server_prompt_cache cache(64, TEST_LIMIT_TOKENS, "", 0, TEST_FINGERPRINT);
    cache.states.push_back(make_direct_restore_state(64, 10));
    auto * allocated = &cache.states.back();

    require(cache.discard(allocated), "discard rejected a live allocated cache node");
    require(cache.states.empty() && cache.dash_drops == 1,
            "discard left a partially populated allocation visible");
    require(!cache.discard(allocated),
            "discard accepted a pointer after its exact node was retired");

    printf("  ok: incomplete cache allocations can be retired by exact owner-thread identity\n");
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
        { "async pending visibility",   test_async_pending_visibility_and_idle_wake },
        { "async saturation fallback",  test_async_saturation_keeps_fallback     },
        { "async stale generation",     test_async_stale_generation_cannot_publish },
        { "async uncertainty",          test_async_uncertainty_reconciles_from_memory },
        { "async uncertainty cooldown", test_active_uncertainty_cooldown_wakes_idle_retry },
        { "async retirement",           test_async_eviction_during_write_and_tombstone_retry },
        { "async post-fence cleanup",    test_post_fence_retirement_cleanup_failures_keep_tombstones },
        { "async shutdown and corruption", test_async_shutdown_modes_and_corrupt_memory },
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
        { "startup cleanup failure",     test_startup_cleanup_failure_disables_live_writes },
        { "startup non-regular path",    test_startup_nonregular_cache_path_disables_live_writes },
        { "resident transaction",        test_failed_candidate_restores_target_only_resident },
        { "paired transaction",          test_failed_paired_draft_restores_paired_resident },
        { "ranked fallback",             test_multiple_bad_candidates_fall_through_to_ssd },
        { "no-fallback miss",            test_failed_candidate_without_fallback_misses_closed },
        { "live alternative ranking",  test_live_alternative_ranking              },
        { "discard incomplete",          test_discard_unpopulated_allocation     },
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

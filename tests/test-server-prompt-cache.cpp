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

std::string only_lcpc_file(const std::string & dir) {
    std::string res;
    for (const auto & entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".lcpc") {
            require(res.empty(), "more than one .lcpc file in " + dir);
            res = entry.path().string();
        }
    }
    require(!res.empty(), "no .lcpc file in " + dir);
    return res;
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

    require(cache.load_from_disk(state), "load_from_disk rejected a healthy file");

    require(state.data.main == expected.data.main, "target state blob differs after reload");
    require(state.data.drft == expected.data.drft, "draft state blob differs after reload");
    require(state.prompt.checkpoints.size() == expected.prompt.checkpoints.size(), "checkpoint count differs after reload");

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

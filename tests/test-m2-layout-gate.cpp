// The gguf-m2 artifact layout gate (src/llama-m2-layout.h).
//
// conversion/gguf_m2_repack.py records the layout every *.m2.gguf was packed
// against; the loader refuses to decode fork-owned tensor encodings unless that
// layout is exactly the one this build decodes with. The failure this guards
// against is silent: packed E4M3/MXFP4 code bytes read under the wrong layout
// are not obviously wrong, they are plausible weights.
//
// Part 1 pins the description strings and their hashes. They are derived from
//   the decode constants, so this fails the moment one of those constants
//   changes — which is the point: the change has to be deliberate, the hashes
//   regenerated, and M2_LAYOUT_VERSION bumped on both sides, in one commit.
// Part 2 is the negative matrix against the gate function directly.
// Part 3 drives a real GGUF through llama_model_load_from_file: one artifact
//   whose layout keys disagree must be rejected *by the gate*, and an otherwise
//   identical artifact with consistent keys must get past the gate (it still
//   fails later — it is not a real model — but for some other reason).
//
// `test-m2-layout-gate --print` prints what this build now expects, for
// regenerating the pinned values after a deliberate layout change.

#include "llama-m2-layout.h"

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;

static void expect(bool cond, const std::string & what) {
    if (!cond) {
        std::printf("  FAIL: %s\n", what.c_str());
        g_fail++;
    }
}

// ---------------------------------------------------------------------------
// part 1: the layout this build accepts

// If these literals no longer match, a layout constant changed. Regenerate them
// with `test-m2-layout-gate --print`, update llama_m2_layout::hash(), and bump
// llama_m2_layout::VERSION together with M2_LAYOUT_VERSION in
// conversion/gguf_m2_repack.py — every artifact converted before the change
// must stop loading.
static const char * PINNED_DESC_E4M3 =
    "e4m3_m2:v1:block1024:codes1024+sc8+pad8:scale=e8m0-per-128:tile128x128;"
    "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384";

static const char * PINNED_DESC_NF8 =
    "nf8_m2:v1:block1024:codes1024+sb32:base=per-32-bias63:escape=e4m3;"
    "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384";

// sha256 of the two strings above, i.e. what gguf_m2_repack.py writes into
// m2.layout.hash
static const char * PINNED_HASH_E4M3 = "9c3da0a661f99161c6296131bbbdefff543fe0af0dde31553b4c850f38a86728";
static const char * PINNED_HASH_NF8  = "e2445da406f6a1478d5a41193841a3b08c402d4c27f829f68bd57c59c89eaeb9";

static void part1() {
    using namespace llama_m2_layout;

    std::printf("part 1: pinned layout\n");

    expect(description(DENSE_E4M3) == PINNED_DESC_E4M3,
            "e4m3 description drifted\n    build:  " + description(DENSE_E4M3) +
            "\n    pinned: " + PINNED_DESC_E4M3);
    expect(description(DENSE_NF8) == PINNED_DESC_NF8,
            "nf8 description drifted\n    build:  " + description(DENSE_NF8) +
            "\n    pinned: " + PINNED_DESC_NF8);

    expect(std::string(hash(DENSE_E4M3)) == PINNED_HASH_E4M3, "e4m3 hash drifted");
    expect(std::string(hash(DENSE_NF8))  == PINNED_HASH_NF8,  "nf8 hash drifted");

    // the deployed artifacts must still load
    expect(check(VERSION, PINNED_DESC_E4M3, PINNED_HASH_E4M3).empty(), "e4m3 artifact rejected");
    expect(check(VERSION, PINNED_DESC_NF8,  PINNED_HASH_NF8).empty(),  "nf8 artifact rejected");

    std::printf("  ok\n");
}

// ---------------------------------------------------------------------------
// part 2: everything that must be refused

static void reject(const char * what, uint32_t ver, const char * desc, const char * hash_hex) {
    const std::string err = llama_m2_layout::check(ver, desc, hash_hex);
    if (err.empty()) {
        std::printf("  FAIL: %s was ACCEPTED\n", what);
        g_fail++;
        return;
    }
    std::printf("  refused %-34s : %s\n", what, err.substr(0, err.find('\n')).c_str());
}

static void part2() {
    using namespace llama_m2_layout;

    std::printf("\npart 2: negative matrix\n");

    reject("older layout version",   VERSION - 1, PINNED_DESC_E4M3, PINNED_HASH_E4M3);
    reject("newer layout version",   VERSION + 1, PINNED_DESC_E4M3, PINNED_HASH_E4M3);
    reject("no description key",     VERSION, nullptr,          PINNED_HASH_E4M3);
    reject("no hash key",            VERSION, PINNED_DESC_E4M3, nullptr);
    reject("empty description",      VERSION, "",               PINNED_HASH_E4M3);
    reject("hash of the other plane", VERSION, PINNED_DESC_E4M3, PINNED_HASH_NF8);
    reject("garbage hash",           VERSION, PINNED_DESC_E4M3, "0000000000000000");

    // the reason the hash alone was never enough: a layout change that forgets
    // to bump the version. Each of these is a real, incompatible reinterpretation
    // of the packed bytes that version 1 would have waved through.
    reject("mxfp4 scale bias 116 -> 117",
            VERSION,
            "e4m3_m2:v1:block1024:codes1024+sc8+pad8:scale=e8m0-per-128:tile128x128;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias117;align16384",
            PINNED_HASH_E4M3);
    reject("e4m3 block 1024 -> 2048",
            VERSION,
            "e4m3_m2:v1:block2048:codes2048+sc16+pad8:scale=e8m0-per-128:tile128x128;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384",
            PINNED_HASH_E4M3);
    reject("e4m3 scale group 128 -> 64",
            VERSION,
            "e4m3_m2:v1:block1024:codes1024+sc16+pad8:scale=e8m0-per-64:tile128x128;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384",
            PINNED_HASH_E4M3);
    reject("nf8 bias 63 -> 64",
            VERSION,
            "nf8_m2:v1:block1024:codes1024+sb32:base=per-32-bias64:escape=e4m3;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384",
            PINNED_HASH_NF8);
    reject("alignment 16384 -> 4096",
            VERSION,
            "e4m3_m2:v1:block1024:codes1024+sc8+pad8:scale=e8m0-per-128:tile128x128;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align4096",
            PINNED_HASH_E4M3);
    reject("a plausible future layout",
            VERSION,
            "e5m2_m2:v1:block1024:codes1024+sc8+pad8:scale=e8m0-per-128:tile128x128;"
            "mxfp4_m2:v1:block2048:codes64x16+scnib64:scale=e8m0-nibble-bias116;align16384",
            PINNED_HASH_E4M3);
}

// ---------------------------------------------------------------------------
// part 3: the gate as the loader actually reaches it

static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) level;
    (void) user;
    g_log += text;
}

// a GGUF carrying one E4M3_M2 tensor and the given layout keys. Not a loadable
// model — the point is which error comes back.
static bool write_artifact(const char * path, const char * desc, const char * hash_hex) {
    ggml_init_params ip = {
        /*.mem_size   =*/ 4*ggml_tensor_overhead() + 4096,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }

    // ne[0] must be a multiple of the E4M3_M2 block size
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_E4M3_M2, llama_m2_layout::E4M3_QK, 1);
    ggml_set_name(t, "blk.0.attn_q_a.weight");
    std::memset(t->data, 0, ggml_nbytes(t));

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "llama");
    gguf_set_val_u32(g, "m2.layout.version", llama_m2_layout::VERSION);
    if (desc != nullptr) {
        gguf_set_val_str(g, "m2.layout.description", desc);
    }
    if (hash_hex != nullptr) {
        gguf_set_val_str(g, "m2.layout.hash", hash_hex);
    }
    gguf_add_tensor(g, t);

    const bool ok = gguf_write_to_file(g, path, /*only_meta =*/ false);

    gguf_free(g);
    ggml_free(ctx);

    return ok;
}

// returns true when the load failed *because of the layout gate*
static bool load_hits_gate(const char * path) {
    g_log.clear();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    llama_model * m = llama_model_load_from_file(path, mp);
    if (m != nullptr) {
        std::printf("  FAIL: %s loaded as a model, which it is not\n", path);
        g_fail++;
        llama_model_free(m);
        return false;
    }

    return g_log.find("layout gate") != std::string::npos;
}

static void part3() {
    std::printf("\npart 3: through llama_model_load_from_file\n");

    llama_log_set(log_cb, nullptr);

    const char * bad  = "test-m2-layout-gate-bad.gguf";
    const char * good = "test-m2-layout-gate-good.gguf";

    // consistent description, wrong hash: the artifact's own keys disagree
    expect(write_artifact(bad, PINNED_DESC_E4M3, PINNED_HASH_NF8), "could not write the bad artifact");
    expect(load_hits_gate(bad), "an artifact whose layout keys disagree was NOT stopped by the gate");
    std::printf("  mismatched hash  -> refused by the gate\n");

    // consistent keys: the gate must let it through (it fails later, on not
    // being a model at all — that is the control that proves part 3 discriminates)
    expect(write_artifact(good, PINNED_DESC_E4M3, PINNED_HASH_E4M3), "could not write the good artifact");
    expect(!load_hits_gate(good), "a well-formed layout was stopped by the gate");
    std::printf("  consistent keys  -> past the gate, fails later as expected\n");

    std::remove(bad);
    std::remove(good);
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    using namespace llama_m2_layout;

    if (argc > 1 && std::strcmp(argv[1], "--print") == 0) {
        const char * name[] = { "e4m3", "nf8" };
        for (int i = 0; i < DENSE_COUNT; ++i) {
            const std::string d = description((dense_plane) i);
            std::printf("%s\n  description : %s\n  pinned hash : %s\n"
                        "  recompute   : printf '%%s' '%s' | sha256sum\n\n",
                    name[i], d.c_str(), hash((dense_plane) i), d.c_str());
        }
        return 0;
    }

    part1();
    part2();
    part3();

    std::printf("\n%s\n", g_fail == 0 ? "m2 layout gate test OK" : "m2 layout gate test FAILED");

    return g_fail == 0 ? 0 : 1;
}

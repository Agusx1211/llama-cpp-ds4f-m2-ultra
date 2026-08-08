// Bit-identity gate for the Metal flash-attention split-K partial state.
//
// The vector flash-attention kernel splits the KV range over `nwg` workgroups,
// writes one partial (accumulator + running sum + running max) per workgroup to
// a temp buffer and combines them in kernel_flash_attn_ext_vec_reduce. At the
// DeepSeek V4 decode signature that temp buffer is 4 MiB per layer and it is
// written once and read once every layer, which dominates the op's device
// traffic. Lane `fa-splitk` shrinks and re-lays-out that state:
//
//   GGML_FA_NWG_FIT_DISABLE=1   restore the fixed nwg = 32 fan-out
//   GGML_FA_TMP_INTERLEAVED=1   restore the [DV4][NWG] partial layout
//   GGML_FA_SPLIT_NSG=<n>       move n key streams per workgroup into
//                               SIMDgroups (nwg /= n); NOT bit-identical
//
// Both defaults are claimed to be *bit-identical* to the historical geometry:
//
//   - the fan-out is only lowered when there are fewer KV chunks than streams,
//     so no stream's chunk list changes and the dropped workgroups contributed
//     exactly the zero partial that the reducer now supplies itself;
//   - the blocked layout reduces the partials with the same balanced binary
//     tree association that a 32-lane simd_sum() applies.
//
// Neither claim is provable by inspection (the second one depends on how the
// Metal runtime implements simd_sum), so this test measures it: it runs a set
// of DSV4-shaped flash-attention problems under each geometry in a separate
// process - the knobs are read once per process - and requires byte-identical
// outputs.
//
// GGML_FA_SPLIT_NSG is exercised too, but only reported: it changes the
// reduction association on purpose, so it is expected to differ. The test
// still requires it to be *reproducible* across runs, which is the property
// scripts/dsv4-depth-determinism.py depends on.
//
// Skips itself with status 0 when no Metal device is present.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#undef assert
#define assert(expr) do {                                                      \
    if (!(expr)) {                                                             \
        std::fprintf(stderr, "check failed at %s:%d: %s\n",                    \
                __FILE__, __LINE__, #expr);                                    \
        std::abort();                                                          \
    }                                                                          \
} while (false)

static ggml_backend_t metal_backend_init() {
    ggml_backend_load_all();

    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "?";
        if (std::strcmp(reg_name, "MTL") == 0 || std::strcmp(reg_name, "Metal") == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
        if (gpu == nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
        }
    }

    return gpu ? ggml_backend_dev_init(gpu, nullptr) : nullptr;
}

// mask shapes, chosen to cover what the DSV4 decode graph actually hands the
// kernel: a dense gathered window, a sparse top-k selection with a masked tail,
// and a fully masked row (a cleared sequence)
enum mask_mode {
    MASK_DENSE   = 0, // every key visible
    MASK_PREFIX  = 1, // a contiguous visible prefix, masked tail
    MASK_BLOCKY  = 2, // alternating visible / masked ncpsg blocks
    MASK_SPARSE  = 3, // a handful of visible keys scattered over the row
};

struct fa_case {
    int64_t kv;
    int     mask;
    const char * name;
};

// The DSV4 decode signature: DK = DV = 512, one KV head, 64 query heads, one
// query row, a one-row mask and per-head sinks. kv values bracket the observed
// geometries: 256 and 512 at shallow context (notes/2026-08-08-gpu-decode-
// attribution.md), 1280 = 768 raw + 512 gathered at depth (notes/2026-08-02-
// dsv4-sparse-fa-workgroup-rejection.md), plus a deeper raw window.
static const fa_case g_cases[] = {
    { 256,  MASK_DENSE,  "kv256/dense"   },
    { 256,  MASK_BLOCKY, "kv256/blocky"  },
    { 512,  MASK_DENSE,  "kv512/dense"   },
    { 512,  MASK_PREFIX, "kv512/prefix"  },
    { 512,  MASK_SPARSE, "kv512/sparse"  },
    { 1024, MASK_BLOCKY, "kv1024/blocky" },
    { 1280, MASK_DENSE,  "kv1280/dense"  },
    { 1280, MASK_PREFIX, "kv1280/prefix" },
    { 1280, MASK_BLOCKY, "kv1280/blocky" },
    { 2560, MASK_BLOCKY, "kv2560/blocky" },
    { 2560, MASK_SPARSE, "kv2560/sparse" },
    // not a multiple of ncpsg = 32: exercises the kvpad path together with the
    // fitted fan-out
    { 288,  MASK_DENSE,  "kv288/dense"   },
};

static const int g_n_cases = (int) (sizeof(g_cases)/sizeof(g_cases[0]));

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static void fill_mask(std::vector<ggml_fp16_t> & m, int64_t kv, int mode, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> coin(0, 1);

    const ggml_fp16_t vis = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t hid = ggml_fp32_to_fp16(-INFINITY);

    m.assign((size_t) kv, vis);

    switch (mode) {
        case MASK_DENSE:
            break;
        case MASK_PREFIX:
            for (int64_t i = kv/2; i < kv; ++i) {
                m[(size_t) i] = hid;
            }
            break;
        case MASK_BLOCKY:
            // 32 = OP_FLASH_ATTN_EXT_VEC_NCPSG, the granularity of the kernel's
            // simd_max(sm) <= -MAXHALF skip
            for (int64_t b = 0; b < kv; b += 32) {
                if (coin(rng)) {
                    for (int64_t i = b; i < std::min(b + 32, kv); ++i) {
                        m[(size_t) i] = hid;
                    }
                }
            }
            break;
        case MASK_SPARSE:
            m.assign((size_t) kv, hid);
            for (int i = 0; i < 37; ++i) {
                m[(size_t) ((int64_t) i*kv/37)] = vis;
            }
            break;
        default:
            break;
    }
}

// One flash-attention evaluation. Returns the FNV-1a hash of the f32 output and,
// when `us` is given, the median wall time of `n_perf` back-to-back evaluations.
static uint64_t run_case(ggml_backend_t backend, const fa_case & c, int n_runs, bool * stable, double * us = nullptr) {
    const int64_t DK = 512;
    const int64_t DV = 512;
    const int64_t NH = 64;

    // A single flash-attention op per command buffer measures the command
    // buffer round trip (~800 us on this box), not the kernel. Replicate the op
    // inside one graph so the per-submission cost is amortised the way it is in
    // the decode graph, where 43 of them ride one command buffer.
    const int n_rep = us ? 16 : 1;

    const size_t ctx_size = (16 + 2*(size_t) n_rep)*ggml_tensor_overhead() + ggml_graph_overhead();

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, 1,    NH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, c.kv, 1,  1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, c.kv, 1,  1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.kv, 1,  1,  1);
    ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NH);

    ggml_set_input(q);
    ggml_set_input(k);
    ggml_set_input(v);
    ggml_set_input(m);
    ggml_set_input(s);

    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * out = nullptr;
    for (int r = 0; r < n_rep; ++r) {
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/std::sqrt((float) DK), 0.0f, 0.0f);
        ggml_flash_attn_ext_add_sinks(o, s);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        ggml_set_output(o);
        ggml_build_forward_expand(gf, o);
        if (r == 0) {
            out = o;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != nullptr);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    // deterministic inputs; the seed is derived from the case so different
    // cases do not share data
    std::mt19937 rng(0xF1A5E7u + (uint32_t) c.kv*131u + (uint32_t) c.mask);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    {
        std::vector<float> hq((size_t) DK*NH);
        for (auto & x : hq) x = dist(rng);
        ggml_backend_tensor_set(q, hq.data(), 0, ggml_nbytes(q));
    }
    {
        std::vector<ggml_fp16_t> hk((size_t) DK*c.kv);
        for (auto & x : hk) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(k, hk.data(), 0, ggml_nbytes(k));

        std::vector<ggml_fp16_t> hv((size_t) DV*c.kv);
        for (auto & x : hv) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(v, hv.data(), 0, ggml_nbytes(v));
    }
    {
        std::vector<ggml_fp16_t> hm;
        fill_mask(hm, c.kv, c.mask, 0xA11CEu + (uint32_t) c.kv);
        ggml_backend_tensor_set(m, hm.data(), 0, ggml_nbytes(m));

        std::vector<float> hs((size_t) NH);
        for (auto & x : hs) x = dist(rng)*4.0f;
        ggml_backend_tensor_set(s, hs.data(), 0, ggml_nbytes(s));
    }

    const size_t n_out = (size_t) ggml_nelements(out);
    std::vector<float> first(n_out);
    std::vector<float> cur(n_out);

    *stable = true;
    for (int run = 0; run < n_runs; ++run) {
        assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(out, cur.data(), 0, n_out*sizeof(float));

        if (run == 0) {
            first = cur;
        } else if (std::memcmp(cur.data(), first.data(), n_out*sizeof(float)) != 0) {
            size_t d = 0;
            while (d < n_out && cur[d] == first[d]) {
                ++d;
            }
            std::fprintf(stderr, "%s: run %d differs from run 0 at element %zu (%.9g vs %.9g)\n",
                    c.name, run, d, first[d], cur[d]);
            *stable = false;
            break;
        }
    }

    const uint64_t h = fnv1a(first.data(), n_out*sizeof(float));

    if (us) {
        // isolated timing: the temp buffer stays hot, which is also true
        // in-graph (all 43 layers reuse the same bid_tmp region), so this ranks
        // layouts. It is NOT a substitute for the in-graph measurement.
        const int n_warm = 10;
        const int n_perf = 30;

        for (int r = 0; r < n_warm; ++r) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);

        std::vector<double> t;
        t.reserve(9);
        for (int rep = 0; rep < 9; ++rep) {
            const int64_t t0 = ggml_time_us();
            for (int r = 0; r < n_perf; ++r) {
                ggml_backend_graph_compute(backend, gf);
            }
            ggml_backend_synchronize(backend);
            t.push_back((double) (ggml_time_us() - t0)/(n_perf*n_rep));
        }
        std::sort(t.begin(), t.end());
        *us = t[t.size()/2];
    }

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return h;
}

// child: run every case under the geometry the parent selected via the
// environment, write "<hash>\n" per case to the file named by
// LLAMA_FA_SPLITK_OUT
static int run_child(const char * out_path) {
    ggml_backend_t backend = metal_backend_init();
    if (backend == nullptr) {
        return 2; // no Metal device: the parent turns this into a skip
    }

    FILE * f = std::fopen(out_path, "w");
    if (f == nullptr) {
        std::perror("fopen");
        ggml_backend_free(backend);
        return 1;
    }

    const bool perf = std::getenv("LLAMA_FA_SPLITK_PERF") != nullptr;

    bool ok = true;
    for (int i = 0; i < g_n_cases; ++i) {
        bool   stable = true;
        double us     = 0.0;
        const uint64_t h = run_case(backend, g_cases[i], 3, &stable, perf ? &us : nullptr);
        std::fprintf(f, "%016llx %d %.3f\n", (unsigned long long) h, stable ? 1 : 0, us);
        ok = ok && stable;
    }

    std::fclose(f);
    ggml_backend_free(backend);

    return ok ? 0 : 1;
}

struct arm {
    const char * name;
    const char * env[4]; // NAME=VALUE pairs, nullptr terminated
    bool         must_match_base;
};

static const arm g_arms[] = {
    { "base(nwg=32,interleaved)", { "GGML_FA_NWG_FIT_DISABLE=1", "GGML_FA_TMP_INTERLEAVED=1", nullptr, nullptr }, true  },
    { "fit(interleaved)",         { "GGML_FA_TMP_INTERLEAVED=1", nullptr, nullptr, nullptr },                     true  },
    { "blocked(no fit)",          { "GGML_FA_NWG_FIT_DISABLE=1", nullptr, nullptr, nullptr },                     true  },
    { "default(fit+blk/down)",    { nullptr, nullptr, nullptr, nullptr },                                         true  },
    { "blocked/assoc=xor",        { "GGML_FA_RED_ASSOC=xor", nullptr, nullptr, nullptr },                          false },
    { "split-nsg=2",              { "GGML_FA_SPLIT_NSG=2", nullptr, nullptr, nullptr },                           false },
    { "split-nsg=4",              { "GGML_FA_SPLIT_NSG=4", nullptr, nullptr, nullptr },                           false },
};

static const int g_n_arms = (int) (sizeof(g_arms)/sizeof(g_arms[0]));

static std::string self_path(const char * argv0) {
#ifdef __APPLE__
    char     buf[4096];
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        return std::string(buf);
    }
#endif
    return argv0 ? std::string(argv0) : std::string();
}

int main(int argc, char ** argv) {
    if (const char * out = std::getenv("LLAMA_FA_SPLITK_OUT")) {
        return run_child(out);
    }

    const std::string exe = self_path(argc > 0 ? argv[0] : nullptr);
    if (exe.empty()) {
        std::puts("metal fa split-K test: cannot locate own executable");
        return 1;
    }

    char tmpl[] = "/tmp/fa-splitk-XXXXXX";
    const char * dir = mkdtemp(tmpl);
    assert(dir != nullptr);

    std::vector<std::vector<uint64_t>> hashes((size_t) g_n_arms);
    std::vector<std::vector<bool>>     stable((size_t) g_n_arms);
    std::vector<std::vector<double>>   times ((size_t) g_n_arms);

    const bool perf = std::getenv("LLAMA_FA_SPLITK_PERF") != nullptr;

    bool ok = true;

    for (int a = 0; a < g_n_arms; ++a) {
        const std::string out = std::string(dir) + "/arm" + std::to_string(a);

        // the knobs are read once per process, so every arm needs its own exec
        for (int e = 0; e < 4 && g_arms[a].env[e]; ++e) {
            std::string kvp = g_arms[a].env[e];
            const size_t eq = kvp.find('=');
            setenv(kvp.substr(0, eq).c_str(), kvp.substr(eq + 1).c_str(), 1);
        }
        setenv("LLAMA_FA_SPLITK_OUT", out.c_str(), 1);

        std::fflush(stdout);
        const pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 1;
        }
        if (pid == 0) {
            char * const args[] = { (char *) exe.c_str(), nullptr };
            execv(exe.c_str(), args);
            _exit(127);
        }

        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            std::perror("waitpid");
            return 1;
        }

        // reset the environment for the next arm
        for (int e = 0; e < 4 && g_arms[a].env[e]; ++e) {
            std::string kvp = g_arms[a].env[e];
            unsetenv(kvp.substr(0, kvp.find('=')).c_str());
        }

        if (WIFEXITED(status) && WEXITSTATUS(status) == 2) {
            std::puts("metal fa split-K test: no Metal device, skipping");
            return 0;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::printf("arm %-24s: child failed (status %d)\n", g_arms[a].name, status);
            ok = false;
        }

        FILE * f = std::fopen(out.c_str(), "r");
        assert(f != nullptr);
        for (int i = 0; i < g_n_cases; ++i) {
            unsigned long long h = 0;
            int st = 0;
            double us = 0.0;
            if (std::fscanf(f, "%llx %d %lf", &h, &st, &us) != 3) {
                std::printf("arm %-24s: truncated result file\n", g_arms[a].name);
                ok = false;
                break;
            }
            hashes[(size_t) a].push_back((uint64_t) h);
            stable[(size_t) a].push_back(st != 0);
            times [(size_t) a].push_back(us);
        }
        std::fclose(f);
        std::remove(out.c_str());
    }
    rmdir(dir);

    if (!ok) {
        std::puts("metal fa split-K test FAILED");
        return 1;
    }

    std::printf("%-24s", "case");
    for (int a = 0; a < g_n_arms; ++a) {
        std::printf(" %-20s", g_arms[a].name);
    }
    std::printf("\n");

    for (int i = 0; i < g_n_cases; ++i) {
        std::printf("%-24s", g_cases[i].name);
        for (int a = 0; a < g_n_arms; ++a) {
            const bool same = hashes[(size_t) a][(size_t) i] == hashes[0][(size_t) i];
            std::printf(" %016llx%s%s",
                    (unsigned long long) hashes[(size_t) a][(size_t) i],
                    same ? "= " : "! ",
                    stable[(size_t) a][(size_t) i] ? "  " : "?!");
        }
        std::printf("\n");

        for (int a = 0; a < g_n_arms; ++a) {
            if (!stable[(size_t) a][(size_t) i]) {
                std::printf("  %s [%s]: NOT REPRODUCIBLE across runs\n", g_cases[i].name, g_arms[a].name);
                ok = false;
            }
            if (g_arms[a].must_match_base && hashes[(size_t) a][(size_t) i] != hashes[0][(size_t) i]) {
                std::printf("  %s [%s]: output differs from the base geometry\n", g_cases[i].name, g_arms[a].name);
                ok = false;
            }
        }
    }

    if (perf) {
        std::printf("\nisolated us/op (median of 8 x 200 evaluations)\n");
        std::printf("%-24s", "case");
        for (int a = 0; a < g_n_arms; ++a) {
            std::printf(" %-20s", g_arms[a].name);
        }
        std::printf("\n");
        for (int i = 0; i < g_n_cases; ++i) {
            std::printf("%-24s", g_cases[i].name);
            for (int a = 0; a < g_n_arms; ++a) {
                const double t0 = times[0][(size_t) i];
                const double ta = times[(size_t) a][(size_t) i];
                std::printf(" %9.2f %+6.1f%%  ", ta, t0 > 0.0 ? 100.0*(ta - t0)/t0 : 0.0);
            }
            std::printf("\n");
        }
    }

    std::printf("\nlegend: '=' identical to base, '!' differs, '?!' not reproducible\n");
    std::printf("(split-nsg arms change the reduction association on purpose and are\n"
                " only required to be reproducible)\n");

    if (!ok) {
        std::puts("metal fa split-K test FAILED");
        return 1;
    }

    std::puts("metal fa split-K test OK");
    return 0;
}

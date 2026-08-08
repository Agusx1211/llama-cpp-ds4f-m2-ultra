// MQA head re-read census for the Metal vector flash-attention kernel.
//
// DeepSeek V4 decode attention is multi-query: one 512-wide KV head serves all
// 64 query heads, and the kernel launches one threadgroup per query head. Every
// one of those threadgroups streams the same selected K/V rows, so a layer
// reads ne02 x (unique K/V bytes) to do ne02 x (a few microseconds of MACs).
//
// This tool measures that re-read at the geometries the DSV4 decode graph
// actually asks for (notes/2026-08-08-fa-splitk-partial-state.md):
//
//   raw2304    2 layers   2304 rows, a 128-row live SWA window
//   csa2816   21 layers   2304 raw + 512 gathered, all of the gathered live
//   hca2560   20 layers   2304 raw + a compressed tail (18k context)
//   hca10496  20 layers   the same family at 1M context
//   ver2816    verify     csa2816 with ne01 = 4 (speculative verification)
//
// and reports, per arm:
//
//   us          median kernel time (16 ops per graph, so the ~800 us command
//               buffer round trip is amortised the way it is in the decode
//               graph)
//   live        KV chunks the mask leaves visible (the kernel skips the rest
//               wholesale via simd_max(sm) <= -MAXHALF)
//   fabric MB   bytes the memory fabric must deliver, i.e. unique K/V bytes
//               times the number of threadgroups that read them independently
//   unique MB   the K/V bytes that exist at all
//   GB/s        fabric MB / us, i.e. what the kernel would need from the
//               fabric if nothing were cached
//
// Arms:
//   nh=N            GGML_FA_NHPTG=N: N query heads per threadgroup
//   kvh=64          one KV head per query head (ne12 = ne02): the same
//                   arithmetic with no sharing available at all. If this costs
//                   the same as kvh=1 the machine is not exploiting the MQA
//                   sharing today and head batching has the full prize; if it
//                   is much slower, the cache hierarchy already absorbs the
//                   re-read and the prize is bounded by what is left.
//   heads=N         query-head sweep, to see whether the cost is linear in the
//                   number of independent readers.
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

static const int64_t DK    = 512;
static const int64_t DV    = 512;
static const int64_t NCPSG = 32; // OP_FLASH_ATTN_EXT_VEC_NCPSG

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

// A DSV4 decode mask: everything hidden except a raw SWA window and a
// contiguous selected/compressed tail. Both are 32-aligned, which is what the
// gather path produces, so the kernel's per-chunk skip is exact and the byte
// model below has no slack.
struct fa_geom {
    const char * name;
    int64_t kv;
    int64_t ne01;    // query rows (1 = decode, 4 = speculative verify)
    int64_t win0;    // first row of the live raw window
    int64_t winn;    // live raw rows
    int64_t tail0;   // first row of the live tail (0 = no tail)
    int64_t tailn;
    bool    big;     // skip in the kvh=64 arm (working set would be > 1 GiB)
};

static const fa_geom g_geoms[] = {
    { "raw2304",  2304,  1, 1024, 128,    0,    0, false },
    { "csa2816",  2816,  1, 1024, 128, 2304,  512, false },
    { "hca2560",  2560,  1, 1024, 128, 2304,  160, false },
    { "hca10496", 10496, 1, 1024, 128, 2304, 8192, true  },
    { "ver2816",  2816,  4, 1024, 128, 2304,  512, false },
};

static const int g_n_geoms = (int) (sizeof(g_geoms)/sizeof(g_geoms[0]));

static int64_t live_chunks(const fa_geom & g) {
    int64_t n = 0;
    for (int64_t c = 0; c < (g.kv + NCPSG - 1)/NCPSG; ++c) {
        const int64_t lo = c*NCPSG;
        const int64_t hi = std::min(lo + NCPSG, g.kv);
        bool live = false;
        for (int64_t i = lo; i < hi && !live; ++i) {
            live = (i >= g.win0 && i < g.win0 + g.winn) ||
                   (g.tailn > 0 && i >= g.tail0 && i < g.tail0 + g.tailn);
        }
        n += live ? 1 : 0;
    }
    return n;
}

static uint64_t fnv1a(const void * data, size_t n) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

// one evaluation; returns the output hash and (optionally) the median us/op
static uint64_t run_geom(ggml_backend_t backend, const fa_geom & g, int64_t nqh, int64_t nkvh,
        bool perf, double * us) {
    assert(nqh % nkvh == 0);

    const int n_rep = perf ? 16 : 1;

    const size_t ctx_size = (16 + 2*(size_t) n_rep)*ggml_tensor_overhead() + ggml_graph_overhead();

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, DK, g.ne01, nqh,  1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DK, g.kv,   nkvh, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, DV, g.kv,   nkvh, 1);
    // one mask row for every query row: this keeps ne31 == 1, which is what the
    // DSV4 decode graph hands the kernel (the gathered mask is [nk,1,1,n_stream])
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, g.kv, 1, 1, 1);
    ggml_tensor * s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nqh);

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

    std::mt19937 rng(0x11A5E7u + (uint32_t) g.kv*131u + (uint32_t) nqh*7u + (uint32_t) nkvh);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    {
        std::vector<float> h((size_t) ggml_nelements(q));
        for (auto & x : h) x = dist(rng);
        ggml_backend_tensor_set(q, h.data(), 0, ggml_nbytes(q));
    }
    {
        std::vector<ggml_fp16_t> h((size_t) ggml_nelements(k));
        for (auto & x : h) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(k, h.data(), 0, ggml_nbytes(k));
        for (auto & x : h) x = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(v, h.data(), 0, ggml_nbytes(v));
    }
    {
        const ggml_fp16_t vis = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t hid = ggml_fp32_to_fp16(-INFINITY);

        std::vector<ggml_fp16_t> h((size_t) ggml_nelements(m), hid);
        for (int64_t r = 0; r < m->ne[1]; ++r) {
            for (int64_t i = 0; i < g.kv; ++i) {
                const bool live = (i >= g.win0 && i < g.win0 + g.winn) ||
                                  (g.tailn > 0 && i >= g.tail0 && i < g.tail0 + g.tailn);
                if (live) {
                    h[(size_t) (r*g.kv + i)] = vis;
                }
            }
        }
        ggml_backend_tensor_set(m, h.data(), 0, ggml_nbytes(m));

        std::vector<float> hs((size_t) nqh);
        for (auto & x : hs) x = dist(rng)*4.0f;
        ggml_backend_tensor_set(s, hs.data(), 0, ggml_nbytes(s));
    }

    const size_t n_out = (size_t) ggml_nelements(out);
    std::vector<float> res(n_out);

    assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_get(out, res.data(), 0, n_out*sizeof(float));

    const uint64_t h = fnv1a(res.data(), n_out*sizeof(float));

    if (perf) {
        for (int r = 0; r < 10; ++r) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);

        const int n_perf = 20;
        std::vector<double> t;
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

struct arm {
    const char * name;
    const char * env[2]; // NAME=VALUE pairs, nullptr terminated
    int64_t      nqh;    // query heads
    int64_t      nkvh;   // KV heads
};

// The mask-skip specialization (kernel_flash_attn_ext_vec's sparse_mask function
// constant) only engages for the exact 64-query-head / 1-KV-head / ne01 == 1 /
// ne31 == 1 signature, so the kvh and heads arms would otherwise be comparing
// two different pipelines. They all pin it off, and `nomskip` is their
// like-for-like reference at the production head count.
#define NOMSKIP "GGML_METAL_DSV4_FA_MASK_SKIP_DISABLE=1"

static const arm g_arms[] = {
    { "nh=1",      { nullptr,           nullptr }, 64,  1 },
    { "nh=2",      { "GGML_FA_NHPTG=2", nullptr }, 64,  1 },
    { "nh=4",      { "GGML_FA_NHPTG=4", nullptr }, 64,  1 },
    { "nh=8",      { "GGML_FA_NHPTG=8", nullptr }, 64,  1 },
    { "nomskip",   { NOMSKIP,           nullptr }, 64,  1 },
    { "nomsk/nh4", { NOMSKIP, "GGML_FA_NHPTG=4" }, 64,  1 },
    { "kvh=64",    { NOMSKIP,           nullptr }, 64, 64 },
    { "nomsk/nh8", { NOMSKIP, "GGML_FA_NHPTG=8" }, 64,  1 },
    { "heads=1",   { NOMSKIP,           nullptr },  1,  1 },
    { "heads=2",   { NOMSKIP,           nullptr },  2,  1 },
    { "heads=4",   { NOMSKIP,           nullptr },  4,  1 },
    { "heads=8",   { NOMSKIP,           nullptr },  8,  1 },
    { "heads=16",  { NOMSKIP,           nullptr }, 16,  1 },
    { "heads=32",  { NOMSKIP,           nullptr }, 32,  1 },
};

static const int g_n_arms = (int) (sizeof(g_arms)/sizeof(g_arms[0]));

static int run_child(const char * out_path) {
    ggml_backend_t backend = metal_backend_init();
    if (backend == nullptr) {
        return 2;
    }

    const int a = atoi(getenv("LLAMA_FA_MQA_ARM"));

    FILE * f = std::fopen(out_path, "w");
    if (f == nullptr) {
        std::perror("fopen");
        ggml_backend_free(backend);
        return 1;
    }

    for (int i = 0; i < g_n_geoms; ++i) {
        double us = 0.0;
        uint64_t h = 0;
        if (g_geoms[i].big && g_arms[a].nkvh > 1) {
            std::fprintf(f, "%016llx %.3f\n", 0ull, 0.0);
            continue;
        }
        h = run_geom(backend, g_geoms[i], g_arms[a].nqh, g_arms[a].nkvh, true, &us);
        std::fprintf(f, "%016llx %.3f\n", (unsigned long long) h, us);
    }

    std::fclose(f);
    ggml_backend_free(backend);

    return 0;
}

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
    if (const char * out = std::getenv("LLAMA_FA_MQA_OUT")) {
        return run_child(out);
    }

    const std::string exe = self_path(argc > 0 ? argv[0] : nullptr);
    if (exe.empty()) {
        std::puts("metal fa mqa census: cannot locate own executable");
        return 1;
    }

    char tmpl[] = "/tmp/fa-mqa-XXXXXX";
    const char * dir = mkdtemp(tmpl);
    assert(dir != nullptr);

    std::vector<std::vector<uint64_t>> hashes((size_t) g_n_arms);
    std::vector<std::vector<double>>   times ((size_t) g_n_arms);

    for (int a = 0; a < g_n_arms; ++a) {
        const std::string out = std::string(dir) + "/arm" + std::to_string(a);

        for (int e = 0; e < 2 && g_arms[a].env[e]; ++e) {
            std::string kvp = g_arms[a].env[e];
            const size_t eq = kvp.find('=');
            setenv(kvp.substr(0, eq).c_str(), kvp.substr(eq + 1).c_str(), 1);
        }
        setenv("LLAMA_FA_MQA_OUT", out.c_str(), 1);
        setenv("LLAMA_FA_MQA_ARM", std::to_string(a).c_str(), 1);

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

        for (int e = 0; e < 2 && g_arms[a].env[e]; ++e) {
            std::string kvp = g_arms[a].env[e];
            unsetenv(kvp.substr(0, kvp.find('=')).c_str());
        }

        if (WIFEXITED(status) && WEXITSTATUS(status) == 2) {
            std::puts("metal fa mqa census: no Metal device, skipping");
            return 0;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            std::printf("arm %-10s: child failed (status %d)\n", g_arms[a].name, status);
            return 1;
        }

        FILE * f = std::fopen(out.c_str(), "r");
        assert(f != nullptr);
        for (int i = 0; i < g_n_geoms; ++i) {
            unsigned long long h = 0;
            double us = 0.0;
            if (std::fscanf(f, "%llx %lf", &h, &us) != 2) {
                std::printf("arm %-10s: truncated result file\n", g_arms[a].name);
                return 1;
            }
            hashes[(size_t) a].push_back((uint64_t) h);
            times [(size_t) a].push_back(us);
        }
        std::fclose(f);
        std::remove(out.c_str());
    }
    rmdir(dir);

    std::printf("\n== geometry ==\n");
    std::printf("%-10s %7s %6s %6s %8s %10s %10s\n",
            "geom", "ne11", "ne01", "chunks", "live", "unique MB", "fabric MB");
    for (int i = 0; i < g_n_geoms; ++i) {
        const fa_geom & g = g_geoms[i];
        const int64_t lc = live_chunks(g);
        const double uniq = (double) lc*NCPSG*(DK + DV)*2/1e6;
        std::printf("%-10s %7lld %6lld %6lld %8lld %10.2f %10.2f\n",
                g.name, (long long) g.kv, (long long) g.ne01,
                (long long) ((g.kv + NCPSG - 1)/NCPSG), (long long) lc,
                uniq, uniq*64*g.ne01);
    }

    std::printf("\n== us/op, and fabric GB/s implied by the arm's independent readers ==\n");
    std::printf("%-10s", "geom");
    for (int a = 0; a < g_n_arms; ++a) {
        std::printf(" %-18s", g_arms[a].name);
    }
    std::printf("\n");

    for (int i = 0; i < g_n_geoms; ++i) {
        const fa_geom & g = g_geoms[i];
        const double uniq = (double) live_chunks(g)*NCPSG*(DK + DV)*2;

        std::printf("%-10s", g.name);
        for (int a = 0; a < g_n_arms; ++a) {
            const double us = times[(size_t) a][(size_t) i];
            if (us <= 0.0) {
                std::printf(" %-18s", "  -");
                continue;
            }
            // readers = threadgroups that independently stream the working set
            double readers = (double) g_arms[a].nqh*g.ne01;
            for (int e = 0; e < 2 && g_arms[a].env[e]; ++e) {
                if (std::strncmp(g_arms[a].env[e], "GGML_FA_NHPTG=", 14) == 0) {
                    readers /= atof(g_arms[a].env[e] + 14);
                }
            }
            const double bytes = g_arms[a].nkvh > 1 ? uniq*g_arms[a].nkvh*g.ne01 : uniq*readers;
            std::printf(" %8.2f %6.0f  ", us, bytes/us/1e3);
        }
        std::printf("\n");
    }

    std::printf("\n== output hash (nh arms must equal nh=1; kvh/heads arms are different problems) ==\n");
    bool ok = true;
    for (int i = 0; i < g_n_geoms; ++i) {
        std::printf("%-10s", g_geoms[i].name);
        for (int a = 0; a < 4; ++a) {
            const bool same = hashes[(size_t) a][(size_t) i] == hashes[0][(size_t) i];
            std::printf(" %016llx%s", (unsigned long long) hashes[(size_t) a][(size_t) i], same ? "= " : "! ");
            ok = ok && same;
        }
        std::printf("\n");
    }

    std::puts(ok ? "\nmetal fa mqa census OK (nh arms bit-identical)"
                 : "\nmetal fa mqa census FAILED (nh arm differs)");
    return ok ? 0 : 1;
}

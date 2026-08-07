// Exactness tests for the fork-owned GGML_TYPE_MXFP4_M2 (gguf-m2 v1 expert
// plane; see notes/2026-08-07-gguf-m2-artifact-format-design.md).
//
// MXFP4_M2 is a byte shuffle of GGML_TYPE_MXFP4 — identical E2M1 codes,
// identical E8M0 scale values (re-encoded as biased nibbles) — so the
// acceptance bar is NOT "close": the MXFP4_M2 Metal kernels must produce
// BIT-IDENTICAL outputs to the MXFP4 Metal kernels they mirror, on every
// dispatch path the expert plane can take:
//
//   - kernel_mul_mv_id_*_dsv4              (decode GEMV, NR0=4, 256e top-6)
//   - kernel_mul_mv_id_*_dsv4_weighted     (decode down+route-weight fusion)
//   - kernel_mul_mm_id_*_dsv4_n16_compact  (prefill worklist tile)
//   - ..._n16_compact_pair                 (prefill gate/up shared worklist)
//   - ..._n16_compact_weighted             (prefill down+route-weight fusion)
//   - kernel_mul_mv_id_* / kernel_mul_mm_id_* (generic geometry — the path a
//     non-256-expert artifact, e.g. a future drafter, would take)
//   - kernel_mul_mv_* / kernel_mul_mm_*    (plain MUL_MAT, test coverage)
//
// Part 0 (any host): the CPU repack + to_float of MXFP4_M2 decodes to exactly
//   the floats MXFP4's to_float produces, for data covering every code byte
//   and the full census scale range.
// Part 1 (GPU host, i.e. the M2): synthetic expert tensors with census-range
//   scales are built once as MXFP4 bytes, repacked in-process to MXFP4_M2
//   (the converter's byte shuffle), and both are run through identical
//   graphs; outputs are compared byte-for-byte. Kernel coverage is enforced
//   by capturing the Metal pipeline-compile debug log and asserting every
//   twin kernel above was actually dispatched.
//
// Extra mode (not run by ctest):
//   test-m2-mxfp4 bench-id   — MXFP4 vs MXFP4_M2 decode-GEMV wall-clock A/B
//                              at the real chained gate/down expert geometry.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// source MXFP4: 32 elems / 17 B ([1 E8M0][16 nibble-packed code bytes])
static constexpr int64_t QK17 = 32;
static constexpr int64_t BB17 = 17;
// fork MXFP4_M2: 2048 elems / 1056 B ([64x16 code bytes][32 scale-nibble B])
static constexpr int64_t QKM2 = 2048;
static constexpr int64_t SBM2 = QKM2/QK17; // 64
static constexpr int64_t BBM2 = SBM2*16 + SBM2/2; // 1056
static constexpr uint8_t SCALE_BIAS = 116; // MXFP4_M2_SCALE_BIAS (ggml-common.h)

// ---------------------------------------------------------------------------
// log capture: the Metal backend logs every pipeline compile at debug level
// ("compiling pipeline: base = '...', name = '...'"); capturing it lets the
// test prove which kernels were actually dispatched instead of trusting the
// dispatch tables.

static std::string       g_log;
static ggml_log_level    g_log_min_print = GGML_LOG_LEVEL_WARN;

static void log_cb(ggml_log_level level, const char * text, void * user) {
    (void) user;
    g_log += text;
    if (level >= g_log_min_print && level != GGML_LOG_LEVEL_DEBUG) {
        fputs(text, stderr);
    }
}

static bool expect_kernel(const char * base_name) {
    const std::string needle = std::string("base = '") + base_name + "'";
    const bool found = g_log.find(needle) != std::string::npos;
    printf("  kernel coverage: %-52s %s\n", base_name, found ? "DISPATCHED" : "MISSING");
    return found;
}

// ---------------------------------------------------------------------------
// deterministic data generation + the converter's byte shuffle

// one expert slice of MXFP4 rows: m rows of k elems = m * k/32 blocks of 17 B.
// scale bytes cycle the measured census range [118, 126]; codes take every
// byte value.
static void gen_mxfp4_rows(uint64_t seed, uint8_t * dst, int64_t k, int64_t m) {
    std::mt19937_64 rng(seed*0x9E3779B97F4A7C15ull + 0xD1B54A32D192ED03ull);
    const int64_t nb = k/QK17;
    for (int64_t r = 0; r < m; r++) {
        for (int64_t b = 0; b < nb; b++) {
            uint8_t * blk = dst + (r*nb + b)*BB17;
            const uint64_t v = rng();
            blk[0] = (uint8_t)(118 + (v % 9)); // E8M0 in [118, 126] (census)
            for (int64_t j = 0; j < 16; j += 8) {
                const uint64_t w = rng();
                memcpy(blk + 1 + j, &w, 8);
            }
        }
    }
}

// the converter's repack (encode_mxfp4_m2_slab in conversion/gguf_m2_repack.py)
static void repack_mxfp4_m2_rows(const uint8_t * src, uint8_t * dst, int64_t k, int64_t m) {
    GGML_ASSERT(k % QKM2 == 0);
    const int64_t nb17 = k/QK17;
    const int64_t nbm2 = k/QKM2;
    for (int64_t r = 0; r < m; r++) {
        const uint8_t * srow = src + r*nb17*BB17;
        uint8_t       * drow = dst + r*nbm2*BBM2;
        for (int64_t ib = 0; ib < nbm2; ib++) {
            const uint8_t * s = srow + ib*SBM2*BB17;
            uint8_t       * d = drow + ib*BBM2;
            memset(d + SBM2*16, 0, SBM2/2);
            for (int64_t sb = 0; sb < SBM2; sb++) {
                const uint8_t e = s[sb*BB17];
                GGML_ASSERT(e >= SCALE_BIAS && e <= SCALE_BIAS + 15);
                memcpy(d + 16*sb, s + sb*BB17 + 1, 16);
                d[SBM2*16 + sb/2] |= (uint8_t)((e - SCALE_BIAS) << (4*(sb % 2)));
            }
        }
    }
}

static std::mt19937 g_frng(4321);

static void fill_uniform(float * dst, size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        dst[i] = dist(rng);
    }
}

// per-token distinct expert routes
static void fill_ids(int32_t * dst, int64_t n_used, int64_t n_tokens, int64_t n_expert, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<int32_t> perm(n_expert);
    std::iota(perm.begin(), perm.end(), 0);
    for (int64_t t = 0; t < n_tokens; t++) {
        std::shuffle(perm.begin(), perm.end(), rng);
        for (int64_t j = 0; j < n_used; j++) {
            dst[t*n_used + j] = perm[j];
        }
    }
}

// ---------------------------------------------------------------------------
// part 0: CPU decode equality

static bool test_cpu_decode_equal(void) {
    const int64_t k = 3*QKM2; // 3 blocks per row
    const int64_t m = 7;
    const int64_t n = k*m;

    std::vector<uint8_t> q17(m*(k/QK17)*BB17);
    std::vector<uint8_t> qm2(m*(k/QKM2)*BBM2);
    gen_mxfp4_rows(7, q17.data(), k, m);
    // force full code coverage: overwrite the first rows with every byte value
    for (size_t i = 0; i < 256 && i < q17.size(); i++) {
        if (i % BB17 != 0) {
            q17[i] = (uint8_t) i;
        }
    }
    // and every scale in the 4-bit window [116, 131]
    for (int i = 0; i < 16; i++) {
        q17[(size_t)i*BB17] = (uint8_t)(SCALE_BIAS + i);
    }
    repack_mxfp4_m2_rows(q17.data(), qm2.data(), k, m);

    std::vector<float> d17(n);
    std::vector<float> dm2(n);
    ggml_get_type_traits(GGML_TYPE_MXFP4)->to_float(q17.data(), d17.data(), n);
    ggml_get_type_traits(GGML_TYPE_MXFP4_M2)->to_float(qm2.data(), dm2.data(), n);

    size_t n_bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&d17[i], &dm2[i], sizeof(float)) != 0) {
            if (n_bad < 8) {
                printf("  decode[%" PRId64 "]: mxfp4=%a mxfp4_m2=%a\n", i, d17[i], dm2[i]);
            }
            n_bad++;
        }
    }
    printf("%-24s: %s (%zu/%" PRId64 " decoded values differ)\n",
           "cpu-decode-equal", n_bad == 0 ? "OK" : "FAIL", n_bad, n);
    return n_bad == 0;
}

// ---------------------------------------------------------------------------
// GPU graph runners

static ggml_backend_t find_gpu_backend(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

enum id_variant {
    VAR_PLAIN,    // out = mul_mat_id(as, b, ids)
    VAR_WEIGHTED, // out = mul(mul_mat_id(as, b, ids), w)   [down + route weights]
    VAR_PAIR,     // out = add(mul_mat_id(as, b, ids), mul_mat_id(as2, b, ids))
};

struct id_case {
    const char * name;
    int64_t k, m, n_expert, n_used;
};

// set the weight tensor expert by expert from a deterministic source so both
// type runs see identical logical values without holding the full plane on
// the host. wtype must be MXFP4 or MXFP4_M2.
static void set_expert_weights(ggml_tensor * as, ggml_type wtype, const id_case & c, uint64_t seed) {
    const size_t sz17 = (size_t)(c.k/QK17)*BB17*c.m;
    std::vector<uint8_t> slab17(sz17);
    std::vector<uint8_t> slabm2;
    const size_t slice = as->nb[2];
    for (int64_t e = 0; e < c.n_expert; e++) {
        gen_mxfp4_rows(seed ^ (0xE0ull + e), slab17.data(), c.k, c.m);
        if (wtype == GGML_TYPE_MXFP4) {
            GGML_ASSERT(slice == sz17);
            ggml_backend_tensor_set(as, slab17.data(), (size_t)e*slice, slice);
        } else {
            slabm2.resize((size_t)(c.k/QKM2)*BBM2*c.m);
            GGML_ASSERT(slice == slabm2.size());
            repack_mxfp4_m2_rows(slab17.data(), slabm2.data(), c.k, c.m);
            ggml_backend_tensor_set(as, slabm2.data(), (size_t)e*slice, slice);
        }
    }
}

static std::vector<float> run_id_graph(
        ggml_backend_t backend,
        ggml_type wtype,
        const id_case & c,
        int64_t n_tokens,
        id_variant variant,
        uint64_t wseed,
        uint32_t xseed) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * as  = ggml_new_tensor_3d(ctx, wtype, c.k, c.m, c.n_expert);
    ggml_tensor * as2 = variant == VAR_PAIR ? ggml_new_tensor_3d(ctx, wtype, c.k, c.m, c.n_expert) : nullptr;
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, c.n_used, n_tokens);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.k, c.n_used, n_tokens);

    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_tensor * w   = nullptr;
    if (variant == VAR_WEIGHTED) {
        w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, c.n_used, n_tokens);
        out = ggml_mul(ctx, out, w);
    } else if (variant == VAR_PAIR) {
        out = ggml_add(ctx, out, ggml_mul_mat_id(ctx, as2, b, ids));
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    set_expert_weights(as, wtype, c, wseed);
    if (as2) {
        set_expert_weights(as2, wtype, c, wseed ^ 0xA5A5A5A5ull);
    }

    std::vector<int32_t> idsv(c.n_used*n_tokens);
    fill_ids(idsv.data(), c.n_used, n_tokens, c.n_expert, xseed + 1);
    ggml_backend_tensor_set(ids, idsv.data(), 0, idsv.size()*sizeof(int32_t));

    std::vector<float> bx(c.k*c.n_used*n_tokens);
    fill_uniform(bx.data(), bx.size(), xseed + 2);
    ggml_backend_tensor_set(b, bx.data(), 0, bx.size()*sizeof(float));

    if (w) {
        std::vector<float> wv(c.n_used*n_tokens);
        fill_uniform(wv.data(), wv.size(), xseed + 3);
        ggml_backend_tensor_set(w, wv.data(), 0, wv.size()*sizeof(float));
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(st == GGML_STATUS_SUCCESS);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return res;
}

// plain MUL_MAT on a single 2D matrix
static std::vector<float> run_plain_graph(
        ggml_backend_t backend,
        ggml_type wtype,
        int64_t k, int64_t m, int64_t n_batch,
        uint64_t wseed,
        uint32_t xseed) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, wtype, k, m);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_batch);
    ggml_tensor * out = ggml_mul_mat(ctx, a, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    std::vector<uint8_t> q17((size_t)(k/QK17)*BB17*m);
    gen_mxfp4_rows(wseed, q17.data(), k, m);
    if (wtype == GGML_TYPE_MXFP4) {
        ggml_backend_tensor_set(a, q17.data(), 0, q17.size());
    } else {
        std::vector<uint8_t> qm2((size_t)(k/QKM2)*BBM2*m);
        repack_mxfp4_m2_rows(q17.data(), qm2.data(), k, m);
        ggml_backend_tensor_set(a, qm2.data(), 0, qm2.size());
    }

    std::vector<float> bx(k*n_batch);
    fill_uniform(bx.data(), bx.size(), xseed);
    ggml_backend_tensor_set(x, bx.data(), 0, bx.size()*sizeof(float));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(st == GGML_STATUS_SUCCESS);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return res;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        num += d*d;
        den += (double) a[i] * (double) a[i];
    }
    return den > 0.0 ? num/den : num;
}

static bool bytes_equal(const char * tag, const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    size_t n_diff = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (n_diff < 4) {
                printf("    diff[%zu]: mxfp4=%a mxfp4_m2=%a\n", i, a[i], b[i]);
            }
            n_diff++;
        }
    }
    printf("%s: %s (%zu diffs / %zu values)\n", tag, n_diff == 0 ? "BIT-EXACT" : "FAIL", n_diff, a.size());
    return n_diff == 0;
}

// ---------------------------------------------------------------------------
// bench-id: MXFP4 vs MXFP4_M2 at the real chained decode expert geometry.
// 64 gate->down GEMV pairs per graph, serialized by data dependency (the
// down input is the gate output), routes rotating over all 256 experts so
// the whole 2.2 GiB working set stays DRAM-resident like production decode.

static int run_bench_id(ggml_backend_t gpu) {
    const int NPAIR  = 64;
    const int WARMUP = 3;
    const int REPS   = 10;

    const id_case gate = { "gate", 4096, 2048, 256, 6 };
    const id_case down = { "down", 2048, 4096, 256, 6 };

    printf("bench-id: %d chained gate->down pairs per graph, %d reps (median), batch 1, 256 experts top-6\n",
           NPAIR, REPS);

    double t_med[2] = {0.0, 0.0};
    const ggml_type types[2] = { GGML_TYPE_MXFP4, GGML_TYPE_MXFP4_M2 };

    for (int ti = 0; ti < 2; ti++) {
        const ggml_type wtype = types[ti];

        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(size_t)(NPAIR*6 + 16) + ggml_graph_overhead_custom(4*NPAIR + 16, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * as_gate = ggml_new_tensor_3d(ctx, wtype, gate.k, gate.m, gate.n_expert);
        ggml_tensor * as_down = ggml_new_tensor_3d(ctx, wtype, down.k, down.m, down.n_expert);

        ggml_tensor * x0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, gate.k, gate.n_used, 1);

        std::vector<ggml_tensor *> ids_g(NPAIR);
        std::vector<ggml_tensor *> ids_d(NPAIR);
        for (int j = 0; j < NPAIR; j++) {
            ids_g[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, gate.n_used, 1);
            ids_d[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, down.n_used, 1);
        }

        ggml_tensor * x = x0;
        for (int j = 0; j < NPAIR; j++) {
            x = ggml_mul_mat_id(ctx, as_gate, x, ids_g[j]); // [2048, 6, 1]
            x = ggml_mul_mat_id(ctx, as_down, x, ids_d[j]); // [4096, 6, 1]
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        GGML_ASSERT(buf != nullptr);

        set_expert_weights(as_gate, wtype, gate, 0xB100);
        set_expert_weights(as_down, wtype, down, 0xB200);

        std::vector<float> xv(gate.k*gate.n_used);
        fill_uniform(xv.data(), xv.size(), 99);
        ggml_backend_tensor_set(x0, xv.data(), 0, xv.size()*sizeof(float));

        // rotate the routes so consecutive pairs touch fresh experts
        for (int j = 0; j < NPAIR; j++) {
            int32_t g[6];
            int32_t d[6];
            for (int r = 0; r < 6; r++) {
                g[r] = (int32_t)((12*j     + r) % 256);
                d[r] = (int32_t)((12*j + 6 + r) % 256);
            }
            ggml_backend_tensor_set(ids_g[j], g, 0, sizeof(g));
            ggml_backend_tensor_set(ids_d[j], d, 0, sizeof(d));
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 4*NPAIR + 16, false);
        ggml_build_forward_expand(gf, x);

        std::vector<double> ts;
        for (int it = 0; it < WARMUP + REPS; it++) {
            const int64_t t0 = ggml_time_us();
            const ggml_status st = ggml_backend_graph_compute(gpu, gf);
            const int64_t t1 = ggml_time_us();
            GGML_ASSERT(st == GGML_STATUS_SUCCESS);
            if (it >= WARMUP) {
                ts.push_back((t1 - t0)/1e3);
            }
        }
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size()/2];
        t_med[ti] = med;

        // bytes of weight data read per graph eval: each GEMV reads 6 full
        // expert matrices
        const double row_bytes_g = wtype == GGML_TYPE_MXFP4 ? gate.k/QK17*BB17 : gate.k/QKM2*BBM2;
        const double row_bytes_d = wtype == GGML_TYPE_MXFP4 ? down.k/QK17*BB17 : down.k/QKM2*BBM2;
        const double bytes = (double)NPAIR*6.0*(row_bytes_g*gate.m + row_bytes_d*down.m);

        printf("  %-9s: median %8.3f ms/graph  (min %8.3f, max %8.3f)  %7.2f us/GEMV  weight-read %6.1f GB/s\n",
               ggml_type_name(wtype), med, ts.front(), ts.back(),
               med*1e3/(2*NPAIR), bytes/(med*1e-3)/1e9);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
    }

    printf("bench-id: time ratio mxfp4_m2/mxfp4 = %.4f  (speedup %+.1f%%)\n",
           t_med[1]/t_med[0], (t_med[0]/t_med[1] - 1.0)*100.0);
    return 0;
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    ggml_log_set(log_cb, nullptr);
    ggml_time_init();

    const bool bench_id = argc > 1 && strcmp(argv[1], "bench-id") == 0;

    bool ok = true;

    if (!bench_id) {
        ok = test_cpu_decode_equal() && ok;
    }

    ggml_backend_t gpu = find_gpu_backend();
    if (gpu == nullptr) {
        printf("no GPU backend available — skipping the Metal bit-exactness matrix "
               "(CPU decode equality above still validated)\n");
        printf("%s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    printf("GPU backend: %s\n", ggml_backend_name(gpu));

    if (bench_id) {
        return run_bench_id(gpu);
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);

    // --- the real DSV4 expert geometry (256 experts, top-6): every batch in
    // the sweep dispatches the same path for both types (dsv4 GEMV < 224
    // tokens, dsv4 worklist tile >= 224)
    {
        const id_case cases[] = {
            { "gate/up exps 4096->2048 e256u6", 4096, 2048, 256, 6 },
            { "down    exps 2048->4096 e256u6", 2048, 4096, 256, 6 },
        };
        const int64_t batches[] = { 1, 2, 4, 8, 32, 512 };

        for (const id_case & c : cases) {
            for (const int64_t n : batches) {
                const uint64_t wseed = 0x5EED0000 ^ (c.k << 16);
                const uint32_t xseed = (uint32_t)(1000 + 7*n);
                const std::vector<float> o17 = run_id_graph(gpu, GGML_TYPE_MXFP4,    c, n, VAR_PLAIN, wseed, xseed);
                const std::vector<float> om2 = run_id_graph(gpu, GGML_TYPE_MXFP4_M2, c, n, VAR_PLAIN, wseed, xseed);
                char tag[256];
                snprintf(tag, sizeof(tag), "%-34s n=%-4" PRId64 " mul_mat_id       ", c.name, n);
                ok = bytes_equal(tag, o17, om2) && ok;
            }
        }

        // fused variants at the batch sizes that select them:
        //  - down + route-weight mul: n=1 (decode mv fusion), n=512 (prompt mm fusion)
        //  - gate/up pair (two mul_mat_id sharing activations + routes): n=512
        const id_case & cg = cases[0];
        const id_case & cd = cases[1];
        for (const int64_t n : { (int64_t)1, (int64_t)512 }) {
            const std::vector<float> o17 = run_id_graph(gpu, GGML_TYPE_MXFP4,    cd, n, VAR_WEIGHTED, 0x77, (uint32_t)(2000 + n));
            const std::vector<float> om2 = run_id_graph(gpu, GGML_TYPE_MXFP4_M2, cd, n, VAR_WEIGHTED, 0x77, (uint32_t)(2000 + n));
            char tag[256];
            snprintf(tag, sizeof(tag), "%-34s n=%-4" PRId64 " down+weight fused", cd.name, n);
            ok = bytes_equal(tag, o17, om2) && ok;
        }
        {
            const int64_t n = 512;
            const std::vector<float> o17 = run_id_graph(gpu, GGML_TYPE_MXFP4,    cg, n, VAR_PAIR, 0x88, 3000);
            const std::vector<float> om2 = run_id_graph(gpu, GGML_TYPE_MXFP4_M2, cg, n, VAR_PAIR, 0x88, 3000);
            char tag[256];
            snprintf(tag, sizeof(tag), "%-34s n=%-4" PRId64 " gate/up pair     ", cg.name, n);
            ok = bytes_equal(tag, o17, om2) && ok;
        }
    }

    // --- generic (non-DSV4) expert geometry: the path a small-expert-count
    // artifact (e.g. a future drafter variant) would take. CPU reference is
    // affordable here, so both types are also NMSE-checked against it.
    {
        const id_case c = { "generic small  2048->1024 e32u4 ", 2048, 1024, 32, 4 };
        const int64_t batches[] = { 1, 2, 8, 32, 64 };

        for (const int64_t n : batches) {
            const uint64_t wseed = 0xD0A7;
            const uint32_t xseed = (uint32_t)(4000 + 3*n);
            const std::vector<float> o17 = run_id_graph(gpu, GGML_TYPE_MXFP4,    c, n, VAR_PLAIN, wseed, xseed);
            const std::vector<float> om2 = run_id_graph(gpu, GGML_TYPE_MXFP4_M2, c, n, VAR_PLAIN, wseed, xseed);
            const std::vector<float> oref = run_id_graph(cpu, GGML_TYPE_MXFP4_M2, c, n, VAR_PLAIN, wseed, xseed);

            char tag[256];
            snprintf(tag, sizeof(tag), "%-34s n=%-4" PRId64 " mul_mat_id       ", c.name, n);
            const bool beq = bytes_equal(tag, o17, om2);

            // n <= 8 keeps activations f32 end-to-end (GEMV); the mm_id tile
            // stages activations as half, which carries an inherent ~1e-7
            // NMSE vs the f32 CPU reference
            const double err     = nmse(oref, om2);
            const double err_max = n > 8 ? 1e-4 : 1e-6;
            printf("    vs-cpu nmse: %.3g %s\n", err, err < err_max ? "OK" : "FAIL");
            ok = (beq && err < err_max) && ok;
        }
    }

    // --- plain MUL_MAT twins (test coverage; the expert plane is only ever a
    // MUL_MAT_ID operand in the model graph). n in [2,8] is skipped for the
    // bit-compare: MXFP4 dispatches its mul_mv_ext kernels there while
    // MXFP4_M2 deliberately has none (unused by the target), so the paths
    // differ. n=1 compares the plain GEMVs, n>=32 the mul_mm pipelines.
    {
        const int64_t k = 4096;
        const int64_t m = 2048;
        for (const int64_t n : { (int64_t)1, (int64_t)32, (int64_t)512 }) {
            const std::vector<float> o17 = run_plain_graph(gpu, GGML_TYPE_MXFP4,    k, m, n, 0xF00D, (uint32_t)(5000 + n));
            const std::vector<float> om2 = run_plain_graph(gpu, GGML_TYPE_MXFP4_M2, k, m, n, 0xF00D, (uint32_t)(5000 + n));
            char tag[256];
            snprintf(tag, sizeof(tag), "%-34s n=%-4" PRId64 " mul_mat (plain)  ", "single matrix 4096->2048", n);
            ok = bytes_equal(tag, o17, om2) && ok;

            if (n == 1) {
                const std::vector<float> oref = run_plain_graph(cpu, GGML_TYPE_MXFP4_M2, k, m, n, 0xF00D, (uint32_t)(5000 + n));
                const double err = nmse(oref, om2);
                printf("    vs-cpu nmse: %.3g %s\n", err, err < 1e-6 ? "OK" : "FAIL");
                ok = (err < 1e-6) && ok;
            }
        }
    }

    // --- kernel coverage: prove the twins were actually dispatched (Metal
    // logs every pipeline compile; each pipeline compiles exactly once per
    // process, so every expected base name must appear in the captured log)
    if (strstr(ggml_backend_name(gpu), "Metal") != nullptr ||
        strstr(ggml_backend_name(gpu), "MTL") != nullptr) {
        printf("kernel dispatch coverage (from the pipeline-compile log):\n");
        const char * expected[] = {
            // fork twins under test
            "kernel_mul_mv_id_mxfp4_m2_f32_dsv4",
            "kernel_mul_mv_id_mxfp4_m2_f32_dsv4_weighted",
            "kernel_mul_mm_id_mxfp4_m2_f32_dsv4_n16_compact",
            "kernel_mul_mm_id_mxfp4_m2_f32_dsv4_n16_compact_pair",
            "kernel_mul_mm_id_mxfp4_m2_f32_dsv4_n16_compact_weighted",
            "kernel_mul_mv_id_mxfp4_m2_f32",
            "kernel_mul_mm_id_mxfp4_m2_f32",
            "kernel_mul_mv_mxfp4_m2_f32",
            "kernel_mul_mm_mxfp4_m2_f32",
            // baseline twins (prove the comparison ran the same paths)
            "kernel_mul_mv_id_mxfp4_f32_dsv4",
            "kernel_mul_mv_id_mxfp4_f32_dsv4_weighted",
            "kernel_mul_mm_id_mxfp4_f32_dsv4_n16_compact",
            "kernel_mul_mm_id_mxfp4_f32_dsv4_n16_compact_pair",
            "kernel_mul_mm_id_mxfp4_f32_dsv4_n16_compact_weighted",
        };
        for (const char * base : expected) {
            ok = expect_kernel(base) && ok;
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

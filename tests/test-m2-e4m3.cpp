// Exactness tests for the fork-owned GGML_TYPE_E4M3_M2 (gguf-m2 v1 dense
// plane; see notes/2026-08-07-gguf-m2-artifact-format-design.md).
//
// The gguf-m2 contract is NOT "close": for weight data whose values are
// e4m3 * 2^k (which is exactly what the converter recovers from the served
// BF16 plane), the E4M3_M2 Metal kernels must produce BIT-IDENTICAL outputs
// to the BF16 Metal kernels they mirror, at every batch size (mul_mv,
// mul_mv_ext and mul_mm paths).
//
// Part 1 (any host): decode/roundtrip exactness on the CPU reference —
//   every value decoded from an E4M3_M2 block must be exactly representable
//   in BF16 (f32 -> bf16 -> f32 roundtrip changes nothing).
// Part 2 (GPU host, i.e. the M2): for the real DSV4 dense-plane shapes and a
//   batch sweep covering all three kernel paths, mul_mat(E4M3_M2 weights, x)
//   and mul_mat(BF16 weights with identical values, x) are computed on the
//   GPU backend and compared byte-for-byte. Additionally the GPU E4M3_M2
//   result is compared against the CPU reference with an NMSE tolerance.
//
// Extra modes (not run by ctest):
//   test-m2-e4m3 bench-dense — E4M3_M2 vs BF16 wall-clock A/B at the real
//                              dense-plane shapes (serialized kernel chains).
//   test-m2-e4m3 bench-graph — E4M3_M2 vs BF16 wall-clock A/B inside a
//                              prefill-shaped MIXED-OP graph (dense mms with
//                              real concurrency structure + the real-geometry
//                              MXFP4_M2 expert MUL_MAT_ID chain). Round 1
//                              proved isolated kernel chains do NOT predict
//                              in-graph behavior on this box (mm measured
//                              1.03-1.09x isolated but -28% end-to-end); this
//                              mode is the round-2 selection instrument and
//                              was calibrated by reproducing that failure.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

constexpr int64_t QK = 1024; // QK_E4M3_M2

struct test_shape {
    const char * name;
    int64_t k;     // ne00 (row length)
    int64_t m;     // ne01 (rows)
    int64_t ne02;  // weight batch dim (attn_output_a is consumed reshaped to 3D)
};

// the real DSV4-Flash dense-plane shapes (from the served UD-Q8_K_XL headers)
static const test_shape shapes[] = {
    { "attn_q_a         (4096x1024)",   4096,  1024, 1 },
    { "attn_q_b         (1024x32768)",  1024, 32768, 1 },
    { "attn_kv          (4096x512)",    4096,   512, 1 },
    { "attn_output_a 3D (4096x1024x8)", 4096,  1024, 8 },
    { "attn_output_b    (8192x4096)",   8192,  4096, 1 },
    { "ffn_gate_shexp   (4096x2048)",   4096,  2048, 1 },
    { "ffn_down_shexp   (2048x4096)",   2048,  4096, 1 },
    { "indexer.attn_q_b (1024x8192)",   1024,  8192, 1 },
};

// batch sizes covering every kernel path: 1 -> mul_mv (_4), 2..8 -> mul_mv_ext
// (the full speculative-verify range: r1_2 at 2, r1_3 at 3/6, r1_4 at 4/7/8,
// r1_5 at 5 — every r1ptg with both full and partial last threadgroups),
// >8 -> mul_mm (16: bounds-checked store, second B tile of the NT1=2 kernel
// empty; 33: second B tile partial with 1 column; 512: direct store path)
static const int64_t batch_sizes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 512 };

static std::mt19937 rng(1234);

static void fill_uniform(float * dst, size_t n) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        dst[i] = dist(rng);
    }
}

// part 1: every decoded E4M3_M2 value must be exactly a BF16 value
static bool test_decode_bf16_exact(void) {
    const int64_t n = 64*QK;

    std::vector<float>   src(n);
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_E4M3_M2, n));
    std::vector<float>   dec(n);

    fill_uniform(src.data(), n);
    ggml_quantize_chunk(GGML_TYPE_E4M3_M2, src.data(), q.data(), 0, 1, n, nullptr);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_E4M3_M2);
    traits->to_float(q.data(), dec.data(), n);

    std::vector<ggml_bf16_t> bf(n);
    std::vector<float>       dec2(n);
    ggml_fp32_to_bf16_row(dec.data(), bf.data(), n);
    ggml_bf16_to_fp32_row(bf.data(), dec2.data(), n);

    size_t n_bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&dec[i], &dec2[i], sizeof(float)) != 0) {
            if (n_bad < 8) {
                printf("  decode[%" PRId64 "] = %a not bf16-representable (roundtrip %a)\n", i, dec[i], dec2[i]);
            }
            n_bad++;
        }
    }

    printf("%-24s: %s (%zu/%" PRId64 " values not bf16-exact)\n", "decode-bf16-exact", n_bad == 0 ? "OK" : "FAIL", n_bad, n);
    return n_bad == 0;
}

static ggml_backend_t find_gpu_backend(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// run mul_mat(w, x) on the given backend; returns the f32 result
static std::vector<float> run_mul_mat(
        ggml_backend_t backend,
        ggml_type wtype,
        const void * wdata, size_t wsize,
        const test_shape & s, int64_t n_batch,
        const float * xdata) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, wtype, s.k, s.m, s.ne02);
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, n_batch, s.ne02);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    GGML_ASSERT(ggml_nbytes(w) == wsize);
    ggml_backend_tensor_set(w, wdata, 0, wsize);
    ggml_backend_tensor_set(x, xdata, 0, ggml_nbytes(x));

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

// ---------------------------------------------------------------------------
// bench-dense (not run by ctest): E4M3_M2 vs BF16 wall-clock A/B at the real
// dense-plane shapes. Each chain serializes the GEMV/GEMM nodes by data
// dependency (the next node's activations are a view of the previous
// output), so the measurement reflects sequential kernel latency like the
// production graph, not overlapped bandwidth. Every node uses a DIFFERENT
// copy of the weight matrix, cycled so the aggregate working set (~1 GiB
// per type) stays DRAM-resident — in production every dense tensor is read
// cold once per token, so an SLC-resident rerun of one matrix would
// overstate ALU-bound behavior and hide the byte-ratio win. E4M3_M2 reads
// ~50.8% of the BF16 bytes, so the DRAM-bound expectation is a time ratio
// near 0.5-0.6.

struct bench_chain {
    const char * name;
    // alternating (k, m) pairs; single-entry chains self-feed through a view
    int64_t k0, m0;
    int64_t k1, m1; // 0 = self-chain
};

static int run_bench_dense(ggml_backend_t gpu, int64_t nb_filter, const char * chain_filter) {
    const bench_chain chains[] = {
        { "attn 4096<->8192 (output_b pair)", 4096, 8192, 8192, 4096 },
        { "shexp 4096<->2048 (gate/down)",    4096, 2048, 2048, 4096 },
        { "attn_q_b 1024->32768 (self)",      1024, 32768, 0, 0 },
    };
    const int64_t batches[] = { 1, 2, 4, 8, 512, 2048 };
    const ggml_type types[2] = { GGML_TYPE_BF16, GGML_TYPE_E4M3_M2 };

    const int WARMUP = 3;
    const int REPS   = 10;

    for (const bench_chain & c : chains) {
        if (chain_filter != nullptr && strstr(c.name, chain_filter) == nullptr) {
            continue;
        }
        // enough distinct weight copies to exceed the SLC by a wide margin
        // (BF16 bytes of one (k0,m0) matrix; target ~1 GiB per type)
        const int64_t w_bytes = c.k0*c.m0*2;
        const int NCOPY = (int)std::min<int64_t>(16, std::max<int64_t>(4, (1ll << 30)/std::max<int64_t>(1, (c.k1 ? 2 : 1)*w_bytes)));

        for (const int64_t nb : batches) {
            if (nb_filter > 0 && nb != nb_filter) {
                continue;
            }

            // node count (pairs count double); n=2048 halves it to keep the
            // activation working set inside the <=4 GiB allocation budget
            const int NREP = nb >= 2048 ? 4 : nb > 8 ? 8 : 64;

            double t_med[2] = {0.0, 0.0};

            for (int ti = 0; ti < 2; ti++) {
                const ggml_type wtype = types[ti];

                ggml_init_params ip = {
                    /*.mem_size   =*/ ggml_tensor_overhead()*(size_t)(3*NREP + 2*NCOPY + 16) + ggml_graph_overhead_custom(3*NREP + 16, false),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ggml_context * ctx = ggml_init(ip);

                const bool self = c.k1 == 0;

                std::vector<ggml_tensor *> w0(NCOPY);
                std::vector<ggml_tensor *> w1(NCOPY);
                for (int i = 0; i < NCOPY; i++) {
                    w0[i] = ggml_new_tensor_2d(ctx, wtype, c.k0, c.m0);
                    w1[i] = self ? nullptr : ggml_new_tensor_2d(ctx, wtype, c.k1, c.m1);
                }

                ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.k0, nb);
                ggml_tensor * x0 = x;

                for (int j = 0; j < NREP; j++) {
                    x = ggml_mul_mat(ctx, w0[j % NCOPY], x);            // [m0, nb]
                    if (self) {
                        // feed the first k0 outputs back in (m0 >= k0)
                        x = ggml_view_2d(ctx, x, c.k0, nb, x->nb[1], 0);
                    } else {
                        x = ggml_mul_mat(ctx, w1[j % NCOPY], x);        // [m1, nb] == [k0, nb]
                    }
                }

                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
                GGML_ASSERT(buf != nullptr);

                // weight values are irrelevant for timing; make them valid
                for (int i = 0; i < NCOPY; i++) {
                    ggml_tensor * ws[2] = { w0[i], w1[i] };
                    for (ggml_tensor * w : ws) {
                        if (w == nullptr) {
                            continue;
                        }
                        const int64_t n_w = ggml_nelements(w);
                        std::vector<float> wf(n_w);
                        fill_uniform(wf.data(), n_w);
                        if (wtype == GGML_TYPE_BF16) {
                            std::vector<ggml_bf16_t> wb(n_w);
                            ggml_fp32_to_bf16_row(wf.data(), wb.data(), n_w);
                            ggml_backend_tensor_set(w, wb.data(), 0, ggml_nbytes(w));
                        } else {
                            std::vector<uint8_t> wq(ggml_nbytes(w));
                            ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, w->ne[1], w->ne[0], nullptr);
                            ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
                        }
                    }
                }

                std::vector<float> xv(ggml_nelements(x0));
                fill_uniform(xv.data(), xv.size());
                ggml_backend_tensor_set(x0, xv.data(), 0, ggml_nbytes(x0));

                ggml_cgraph * gf = ggml_new_graph_custom(ctx, 3*NREP + 16, false);
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
                t_med[ti] = ts[ts.size()/2];

                const int n_mm = self ? NREP : 2*NREP;
                printf("  %-34s n=%-4" PRId64 " %-8s: median %8.3f ms/graph (min %8.3f, max %8.3f), %8.2f us/matmul\n",
                       c.name, nb, ggml_type_name(wtype),
                       t_med[ti], ts.front(), ts.back(), t_med[ti]*1e3/n_mm);

                ggml_backend_buffer_free(buf);
                ggml_free(ctx);
            }

            printf("  %-34s n=%-4" PRId64 " ratio e4m3_m2/bf16 = %.4f\n", c.name, nb, t_med[1]/t_med[0]);
        }
    }

    return 0;
}

// ---------------------------------------------------------------------------
// bench-graph (not run by ctest): E4M3_M2 vs BF16 dense plane inside a
// prefill-shaped mixed-op graph.
//
// Structure per "layer" (mirrors the DSV4-Flash prefill layer's dependency
// shape, which is what the concurrent hazard-tracked Metal encoder sees):
//
//   x[4096,n] -> qa = Wqa*x          [1024,n]      (dense mm)
//             -> kv = Wkv*x          [512,n]       (dense mm, CONCURRENT w/ qa)
//   qa        -> qb = Wqb*qa         [32768,n]     (dense mm)
//             -> iq = Wiq*qa         [8192,n]      (dense mm, CONCURRENT w/ qb)
//   qb[0:8192]-> ob = Wob*qb'        [4096,n]      (dense mm)
//   h = ob + x + bcast(kv) + iq[0:4096]            (join, like attn residual)
//   h -> sg = Wg*h, su = Wu*h        [2048,n]x2    (shexp gate/up, CONCURRENT
//                                                   with the MoE chain below)
//        gu = sg*su; sd = Wd*gu      [4096,n]
//   h -> ge = mul_mat_id(Eg, h, ids) [2048,6,n]    (256-expert MXFP4_M2 gate,
//                                                   real worklist geometry)
//        de = mul_mat_id(Ed, ge, ids)[4096,6,n]    (down)
//        dw = de * wts               (down+route-weight fusion shape)
//   x' = sd + dw[expert 0] + h
//
// Dense weights cycle over NCOPY distinct sets so every dense read is
// DRAM-cold (same rationale as bench-dense); the expert tensors are the REAL
// 2.2 GiB gate+down pair (one copy — production reads them from DRAM every
// layer anyway) and are byte-identical between the two dense-type runs, so
// the A/B difference is the dense plane only.
//
// usage: test-m2-e4m3 bench-graph [n] [n_layer] [nomoe]
//   n = 0 (default) sweeps { 30, 512, 2048 }; "nomoe" drops the expert chain
//   (attribution: dense-only concurrency vs mixed-op concurrency).

static void fill_mxfp4_m2_valid(ggml_tensor * as) {
    // one expert slice of valid random MXFP4_M2 blocks (codes: any byte;
    // scale nibbles biased to the census range so magnitudes stay sane),
    // replicated to every expert: identical bytes at different addresses are
    // indistinguishable for bandwidth purposes and fill 2.2 GiB in ~1 s.
    constexpr size_t BB = 1056; // block_mxfp4_m2: qs[1024] + sc[32]
    const size_t slice = as->nb[2];
    GGML_ASSERT(slice % BB == 0);

    std::vector<uint8_t> slab(slice);
    std::mt19937_64 r64(0xE4A3);
    for (size_t off = 0; off < slab.size(); off += 8) {
        const uint64_t v = r64();
        memcpy(slab.data() + off, &v, 8);
    }
    for (size_t off = 0; off + BB <= slab.size(); off += BB) {
        for (size_t i = 0; i < 32; i++) {
            const uint8_t lo = 2 + (slab[off + 1024 + i] & 3); // E8M0 118..121
            const uint8_t hi = 2 + ((slab[off + 1024 + i] >> 4) & 3);
            slab[off + 1024 + i] = (uint8_t)(lo | (hi << 4));
        }
    }
    for (int64_t e = 0; e < as->ne[2]; e++) {
        ggml_backend_tensor_set(as, slab.data(), (size_t)e*slice, slice);
    }
}

struct bg_shape {
    const char * nm;
    int64_t k, m;
};

static int run_bench_graph(ggml_backend_t gpu, int64_t n_filter, int n_layer, bool moe) {
    // the real dense-plane per-layer shapes; "oa" is attn_output_a consumed
    // reshaped 3D [4096, 1024, 8] against a PERMUTED src1 exactly like
    // deepseek4.cpp (out reshaped [o_group_dim, n_groups, nt] then
    // permute(0,2,1,3))
    static const bg_shape ds[] = {
        { "qa", 4096,  1024 },
        { "qb", 1024, 32768 },
        { "kv", 4096,   512 },
        { "iq", 1024,  8192 },
        { "ob", 8192,  4096 },
        { "sg", 4096,  2048 },
        { "su", 4096,  2048 },
        { "sd", 2048,  4096 },
        { "oa", 4096,  1024*8 }, // stored 3D [4096, 1024, 8]
    };
    constexpr int NDS   = (int)(sizeof(ds)/sizeof(ds[0]));
    constexpr int NCOPY = 2; // distinct dense sets; reuse distance stays >> SLC

    const int64_t n_experts = 256;
    const int64_t n_used    = 6;

    const int WARMUP = 3;
    const int REPS   = 10;

    // n_filter > 0 runs exactly that batch size (any value — needed to probe
    // bco=1 shapes like the 15k-prompt tail and budget-sized prefill chunks)
    const int64_t batches_all[] = { n_filter > 0 ? n_filter : 30,
                                    n_filter > 0 ? -1 : 512,
                                    n_filter > 0 ? -1 : 2048 };

    // host-side dense weight bytes, quantized once per shape, shared by copies
    std::vector<std::vector<uint8_t>>    wq(NDS);
    std::vector<std::vector<ggml_bf16_t>> wb(NDS);
    for (int s = 0; s < NDS; s++) {
        const int64_t nel = ds[s].k*ds[s].m;
        std::vector<float> wf(nel);
        fill_uniform(wf.data(), nel);
        wq[s].resize(ggml_row_size(GGML_TYPE_E4M3_M2, ds[s].k)*ds[s].m);
        ggml_quantize_chunk(GGML_TYPE_E4M3_M2, wf.data(), wq[s].data(), 0, ds[s].m, ds[s].k, nullptr);
        std::vector<float> wdec(nel);
        ggml_get_type_traits(GGML_TYPE_E4M3_M2)->to_float(wq[s].data(), wdec.data(), nel);
        wb[s].resize(nel);
        ggml_fp32_to_bf16_row(wdec.data(), wb[s].data(), nel);
    }

    // expert tensors: allocated once, shared by both dense-type runs
    ggml_context * ctx_e = nullptr;
    ggml_backend_buffer_t buf_e = nullptr;
    ggml_tensor * eg = nullptr;
    ggml_tensor * ed = nullptr;
    if (moe) {
        ggml_init_params ipe = { ggml_tensor_overhead()*8, nullptr, true };
        ctx_e = ggml_init(ipe);
        eg = ggml_new_tensor_3d(ctx_e, GGML_TYPE_MXFP4_M2, 4096, 2048, n_experts);
        ed = ggml_new_tensor_3d(ctx_e, GGML_TYPE_MXFP4_M2, 2048, 4096, n_experts);
        buf_e = ggml_backend_alloc_ctx_tensors(ctx_e, gpu);
        GGML_ASSERT(buf_e != nullptr);
        fill_mxfp4_m2_valid(eg);
        fill_mxfp4_m2_valid(ed);
        printf("bench-graph: expert plane %.2f GiB (MXFP4_M2, 256e, shared by both runs)\n",
               (double)(ggml_nbytes(eg) + ggml_nbytes(ed))/(1ll << 30));
    }

    printf("bench-graph: %d layers, %d dense copies, moe=%s, median of %d\n",
           n_layer, NCOPY, moe ? "on" : "off", REPS);

    for (const int64_t n : batches_all) {
        if (n <= 0) {
            continue;
        }

        double t_med[2] = { 0.0, 0.0 };
        const ggml_type types[2] = { GGML_TYPE_BF16, GGML_TYPE_E4M3_M2 };

        for (int ti = 0; ti < 2; ti++) {
            const ggml_type wtype = types[ti];

            // weights + graph inputs
            ggml_init_params ipw = { ggml_tensor_overhead()*(size_t)(NCOPY*NDS + 2*n_layer + 16), nullptr, true };
            ggml_context * ctx_w = ggml_init(ipw);

            ggml_tensor * W[NCOPY][NDS];
            for (int c = 0; c < NCOPY; c++) {
                for (int s = 0; s < NDS; s++) {
                    W[c][s] = strcmp(ds[s].nm, "oa") == 0
                        ? ggml_new_tensor_3d(ctx_w, wtype, ds[s].k, ds[s].m/8, 8)
                        : ggml_new_tensor_2d(ctx_w, wtype, ds[s].k, ds[s].m);
                }
            }
            ggml_tensor * x0  = ggml_new_tensor_2d(ctx_w, GGML_TYPE_F32, 4096, n);
            ggml_tensor * wts = moe ? ggml_new_tensor_3d(ctx_w, GGML_TYPE_F32, 1, n_used, n) : nullptr;
            // static K/V/mask for the flash-attention op (shared read-only by
            // all layers; prod attends up to the full context — 4096 keeps the
            // bench inside the GPU budget while making FA a first-class
            // concurrent kernel like production prefill)
            const int64_t fa_kv = 4096;
            ggml_tensor * fk = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F16, 128, fa_kv, 8, 1);
            ggml_tensor * fv = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F16, 128, fa_kv, 8, 1);
            ggml_tensor * fm = ggml_new_tensor_4d(ctx_w, GGML_TYPE_F16, fa_kv, n, 1, 1);
            std::vector<ggml_tensor *> ids(n_layer, nullptr);
            if (moe) {
                for (int l = 0; l < n_layer; l++) {
                    ids[l] = ggml_new_tensor_2d(ctx_w, GGML_TYPE_I32, n_used, n);
                }
            }

            ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, gpu);
            GGML_ASSERT(buf_w != nullptr);

            for (int c = 0; c < NCOPY; c++) {
                for (int s = 0; s < NDS; s++) {
                    if (wtype == GGML_TYPE_BF16) {
                        ggml_backend_tensor_set(W[c][s], wb[s].data(), 0, ggml_nbytes(W[c][s]));
                    } else {
                        ggml_backend_tensor_set(W[c][s], wq[s].data(), 0, ggml_nbytes(W[c][s]));
                    }
                }
            }

            {
                std::vector<float> xv(4096*n);
                fill_uniform(xv.data(), xv.size());
                ggml_backend_tensor_set(x0, xv.data(), 0, ggml_nbytes(x0));

                std::vector<float> kf(ggml_nelements(fk));
                std::vector<ggml_fp16_t> kh(kf.size());
                fill_uniform(kf.data(), kf.size());
                ggml_fp32_to_fp16_row(kf.data(), kh.data(), kf.size());
                ggml_backend_tensor_set(fk, kh.data(), 0, ggml_nbytes(fk));
                fill_uniform(kf.data(), kf.size());
                ggml_fp32_to_fp16_row(kf.data(), kh.data(), kf.size());
                ggml_backend_tensor_set(fv, kh.data(), 0, ggml_nbytes(fv));

                std::vector<ggml_fp16_t> mh(ggml_nelements(fm));
                memset(mh.data(), 0, mh.size()*sizeof(ggml_fp16_t)); // f16 zero = no masking
                ggml_backend_tensor_set(fm, mh.data(), 0, ggml_nbytes(fm));
            }
            if (moe) {
                std::vector<float> wv(n_used*n);
                fill_uniform(wv.data(), wv.size());
                ggml_backend_tensor_set(wts, wv.data(), 0, ggml_nbytes(wts));

                std::mt19937 ridg(777);
                std::vector<int32_t> iv(n_used*n);
                for (int l = 0; l < n_layer; l++) {
                    for (int64_t t = 0; t < n; t++) {
                        // 6 distinct experts per token (real router property)
                        int32_t pick[6];
                        for (int r = 0; r < 6; r++) {
                            bool dup;
                            do {
                                pick[r] = (int32_t)(ridg() % n_experts);
                                dup = false;
                                for (int p = 0; p < r; p++) {
                                    dup = dup || pick[p] == pick[r];
                                }
                            } while (dup);
                            iv[t*n_used + r] = pick[r];
                        }
                    }
                    ggml_backend_tensor_set(ids[l], iv.data(), 0, ggml_nbytes(ids[l]));
                }
            }

            // graph (intermediates via gallocr so the live set stays small)
            const size_t graph_nodes = (size_t)n_layer*48 + 16;
            ggml_init_params ipg = {
                ggml_tensor_overhead()*graph_nodes + ggml_graph_overhead_custom(graph_nodes, false),
                nullptr, true,
            };
            ggml_context * ctx_g = ggml_init(ipg);

            ggml_tensor * x = x0;
            for (int l = 0; l < n_layer; l++) {
                ggml_tensor ** Wl = W[l % NCOPY];

                ggml_tensor * qa = ggml_mul_mat(ctx_g, Wl[0], x);   // [1024, n]
                ggml_tensor * kv = ggml_mul_mat(ctx_g, Wl[2], x);   // [512, n]
                ggml_tensor * qb = ggml_mul_mat(ctx_g, Wl[1], qa);  // [32768, n]
                ggml_tensor * iq = ggml_mul_mat(ctx_g, Wl[3], qa);  // [8192, n]

                // flash attention (q = 32 heads x 128 from qb, GQA over the
                // shared 8-head K/V) — the big concurrent kernel class the
                // dense mms co-reside with during production prefill
                ggml_tensor * fq = ggml_view_3d(ctx_g, qb, 128, 32, n, 128*sizeof(float), qb->nb[1], 0);
                fq = ggml_permute(ctx_g, fq, 0, 2, 1, 3); // [128, n, 32]
                ggml_tensor * fa = ggml_flash_attn_ext(ctx_g, fq, fk, fv, fm, 1.0f/sqrtf(128.0f), 0.0f, 0.0f);
                ggml_tensor * fa2 = ggml_reshape_2d(ctx_g, fa, 4096, n);

                // attn_output_a exactly as deepseek4.cpp applies it: 3D weight
                // [4096, 1024, 8] against a permuted src1 [4096, n, 8]
                ggml_tensor * oai = ggml_view_3d(ctx_g, qb, 4096, 8, n, 4096*sizeof(float), qb->nb[1], 0);
                oai = ggml_permute(ctx_g, oai, 0, 2, 1, 3);          // [4096, n, 8]
                ggml_tensor * oa = ggml_mul_mat(ctx_g, Wl[8], oai);  // [1024, n, 8]
                oa = ggml_cont_2d(ctx_g, ggml_permute(ctx_g, oa, 0, 2, 1, 3), 8192, n);
                ggml_tensor * ob = ggml_mul_mat(ctx_g, Wl[4], oa);   // [4096, n]

                ggml_tensor * iqv = ggml_view_2d(ctx_g, iq, 4096, n, iq->nb[1], 0);
                ggml_tensor * h   = ggml_add(ctx_g,
                        ggml_add(ctx_g, ggml_add(ctx_g, ggml_add(ctx_g, ob, x), iqv), kv), fa2);

                ggml_tensor * sg = ggml_mul_mat(ctx_g, Wl[5], h);   // [2048, n]
                ggml_tensor * su = ggml_mul_mat(ctx_g, Wl[6], h);   // [2048, n]
                ggml_tensor * gu = ggml_mul(ctx_g, sg, su);
                ggml_tensor * sd = ggml_mul_mat(ctx_g, Wl[7], gu);  // [4096, n]

                if (moe) {
                    ggml_tensor * hr = ggml_reshape_3d(ctx_g, h, 4096, 1, n);
                    // gate + up adjacent with identical src1/ids/weight layout
                    // -> exercises the dsv4 pair-fusion worklist path like
                    // production (the same physical tensor twice reads the
                    // same bytes DRAM-cold either way)
                    ggml_tensor * ge = ggml_mul_mat_id(ctx_g, eg, hr, ids[l]); // [2048, 6, n]
                    ggml_tensor * ue = ggml_mul_mat_id(ctx_g, eg, hr, ids[l]); // [2048, 6, n]
                    ggml_tensor * gue = ggml_mul(ctx_g, ge, ue);
                    ggml_tensor * de = ggml_mul_mat_id(ctx_g, ed, gue, ids[l]); // [4096, 6, n]
                    ggml_tensor * dw = ggml_mul(ctx_g, de, wts);
                    ggml_tensor * mo = ggml_view_2d(ctx_g, dw, 4096, n, dw->nb[2], 0);
                    x = ggml_add(ctx_g, ggml_add(ctx_g, sd, mo), h);
                } else {
                    x = ggml_add(ctx_g, sd, h);
                }
            }

            ggml_cgraph * gf = ggml_new_graph_custom(ctx_g, graph_nodes, false);
            ggml_build_forward_expand(gf, x);

            ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gpu));
            GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));

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
            t_med[ti] = ts[ts.size()/2];

            printf("  n=%-5" PRId64 " %-8s: median %8.3f ms/graph (min %8.3f, max %8.3f), %d nodes\n",
                   n, ggml_type_name(wtype), t_med[ti], ts.front(), ts.back(), ggml_graph_n_nodes(gf));

            ggml_gallocr_free(galloc);
            ggml_free(ctx_g);
            ggml_backend_buffer_free(buf_w);
            ggml_free(ctx_w);
        }

        printf("  n=%-5" PRId64 " graph time ratio e4m3_m2/bf16 = %.4f  (dense-plane delta %+.1f%% of total)\n",
               n, t_med[1]/t_med[0], (t_med[1]/t_med[0] - 1.0)*100.0);
    }

    if (buf_e) {
        ggml_backend_buffer_free(buf_e);
    }
    if (ctx_e) {
        ggml_free(ctx_e);
    }

    return 0;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    const bool bench_dense = argc > 1 && strcmp(argv[1], "bench-dense") == 0;
    const bool bench_graph = argc > 1 && strcmp(argv[1], "bench-graph") == 0;

    bool ok = true;

    if (bench_dense || bench_graph) {
        ggml_backend_t gpu = find_gpu_backend();
        if (gpu == nullptr) {
            printf("bench requires a GPU backend\n");
            return 1;
        }
        printf("GPU backend: %s\n", ggml_backend_name(gpu));
        if (bench_graph) {
            // optional args: batch-size filter (0 = sweep), layer count, "nomoe"
            const int64_t n_filter = argc > 2 ? atoll(argv[2]) : 0;
            const int     n_layer  = argc > 3 ? atoi(argv[3]) : 8;
            const bool    moe      = !(argc > 4 && strcmp(argv[4], "nomoe") == 0);
            return run_bench_graph(gpu, n_filter, n_layer, moe);
        }
        // optional args: batch-size filter and chain-name substring filter
        // (fast A/B iteration)
        const int64_t nb_filter    = argc > 2 ? atoll(argv[2]) : 0;
        const char *  chain_filter = argc > 3 ? argv[3] : nullptr;
        return run_bench_dense(gpu, nb_filter, chain_filter);
    }

    ok = test_decode_bf16_exact() && ok;

    ggml_backend_t gpu = find_gpu_backend();
    if (gpu == nullptr) {
        printf("no GPU backend available — skipping the Metal bit-exactness matrix "
               "(decode exactness above still validated)\n");
        printf("%s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);

    printf("GPU backend: %s\n", ggml_backend_name(gpu));

    for (const test_shape & s : shapes) {
        // build the E4M3_M2 weight once per shape, and a BF16 weight holding
        // bit-identical values (the exactness premise recovered by the
        // converter, reproduced here synthetically)
        const int64_t n_w = s.k*s.m*s.ne02;

        std::vector<float> wf(n_w);
        fill_uniform(wf.data(), n_w);

        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_E4M3_M2, s.k)*s.m*s.ne02);
        ggml_quantize_chunk(GGML_TYPE_E4M3_M2, wf.data(), wq.data(), 0, s.m*s.ne02, s.k, nullptr);

        std::vector<float> wdec(n_w);
        ggml_get_type_traits(GGML_TYPE_E4M3_M2)->to_float(wq.data(), wdec.data(), n_w);

        std::vector<ggml_bf16_t> wbf(n_w);
        ggml_fp32_to_bf16_row(wdec.data(), wbf.data(), n_w);

        for (const int64_t n_batch : batch_sizes) {
            std::vector<float> x(s.k*n_batch*s.ne02);
            fill_uniform(x.data(), x.size());

            const std::vector<float> out_q  = run_mul_mat(gpu, GGML_TYPE_E4M3_M2, wq.data(),  wq.size(),                    s, n_batch, x.data());
            const std::vector<float> out_bf = run_mul_mat(gpu, GGML_TYPE_BF16,    wbf.data(), wbf.size()*sizeof(ggml_bf16_t), s, n_batch, x.data());
            const std::vector<float> out_ref = run_mul_mat(cpu, GGML_TYPE_E4M3_M2, wq.data(), wq.size(),                    s, n_batch, x.data());

            // bit-exactness vs the BF16 kernels
            size_t n_diff = 0;
            float max_abs = 0.0f;
            for (size_t i = 0; i < out_q.size(); i++) {
                if (memcmp(&out_q[i], &out_bf[i], sizeof(float)) != 0) {
                    if (n_diff < 4) {
                        printf("    diff[%zu]: e4m3=%a bf16=%a\n", i, out_q[i], out_bf[i]);
                    }
                    n_diff++;
                    max_abs = std::max(max_abs, std::fabs(out_q[i] - out_bf[i]));
                }
            }

            // sanity vs the CPU reference (different accumulation order — NMSE).
            // n_batch > 8 dispatches the mul_mm path, which stages the B
            // operand (activations) as bfloat in threadgroup memory — the
            // exact numerics of the production BF16 mul_mm today (and the
            // BIT-EXACT check above proves E4M3_M2 inherits them precisely).
            // The CPU reference keeps activations f32, so that path carries
            // an inherent ~1e-6 NMSE vs the reference; the mul_mv/ext paths
            // (n_batch <= 8) keep activations f32 and sit at ~1e-14.
            const double err     = nmse(out_ref, out_q);
            const double err_max = n_batch > 8 ? 1e-4 : 1e-6;

            const bool pass = n_diff == 0 && err < err_max;
            ok = pass && ok;

            printf("%-32s n_batch=%-4" PRId64 " vs-bf16: %s (%zu diffs, max %g)  vs-cpu nmse: %.3g %s\n",
                   s.name, n_batch,
                   n_diff == 0 ? "BIT-EXACT" : "FAIL", n_diff, (double) max_abs,
                   err, err < err_max ? "OK" : "FAIL");
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

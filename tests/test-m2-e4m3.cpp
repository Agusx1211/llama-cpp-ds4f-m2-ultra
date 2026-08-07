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
// Extra mode (not run by ctest):
//   test-m2-e4m3 bench-dense — E4M3_M2 vs BF16 wall-clock A/B at the real
//                              dense-plane shapes (serialized kernel chains).

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

int main(int argc, char ** argv) {
    ggml_time_init();

    const bool bench_dense = argc > 1 && strcmp(argv[1], "bench-dense") == 0;

    bool ok = true;

    if (bench_dense) {
        ggml_backend_t gpu = find_gpu_backend();
        if (gpu == nullptr) {
            printf("bench-dense requires a GPU backend\n");
            return 1;
        }
        printf("GPU backend: %s\n", ggml_backend_name(gpu));
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

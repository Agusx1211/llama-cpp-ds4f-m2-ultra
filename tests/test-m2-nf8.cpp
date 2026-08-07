// Exactness tests for the fork-owned GGML_TYPE_NF8_M2 (gguf-m2 dense plane,
// second encoding; see notes/2026-08-07-gguf-m2-artifact-format-design.md,
// "Round-3 candidate formats", design (b)).
//
// The gguf-m2 contract is NOT "close": for weight data whose values are
// e4m3 * 2^k (exactly what the converter recovers from the served BF16
// plane), the NF8_M2 Metal kernels must produce BIT-IDENTICAL outputs to the
// BF16 Metal kernels they mirror, at every batch size (mul_mv, mul_mv_ext
// and mul_mm paths) — INCLUDING the per-32-element-group escape blocks and
// the reserved zero code, which the weight generator here plants explicitly
// (at block-boundary groups among others) and the test asserts are present.
//
// Part 1 (any host):
//   - encoder round-trip: synthetic on-grid data (values exactly
//     e4m3 * 2^k), with planted wide groups (-> escape), zeros, signed
//     zeros and all-zero groups, must survive quantize -> decode
//     memcmp-identical, and the encoding must actually contain both fast
//     and escape groups;
//   - decode-bf16-exact: every decoded NF8_M2 value must be exactly
//     representable in BF16 (f32 -> bf16 -> f32 roundtrip changes nothing).
// Part 2 (GPU host, i.e. the M2): for the real DSV4 dense-plane shapes and a
//   batch sweep covering all three kernel paths, mul_mat(NF8_M2 weights, x)
//   and mul_mat(BF16 weights with identical values, x) are computed on the
//   GPU backend and compared byte-for-byte. Additionally the GPU NF8_M2
//   result is compared against the CPU reference with an NMSE tolerance.
//
// Extra mode (not run by ctest):
//   test-m2-nf8 bench-dense — NF8_M2 vs E4M3_M2 vs BF16 wall-clock A/B at
//                             the real dense-plane shapes (serialized kernel
//                             chains, DRAM-cold cycled weights).

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

constexpr int64_t QK  = 1024; // QK_NF8_M2
constexpr int64_t GSZ = 32;   // NF8_M2_GSZ

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
// (r1_2..r1_5 with full and partial last threadgroups, and the r1_6/7/8
// single-pass dispatch on the ne01 >= 16384 shape), >8 -> mul_mm (16/33:
// bounds-checked store, 512: direct store path)
static const int64_t batch_sizes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 16, 33, 512 };

static std::mt19937 rng(1234);

static void fill_uniform(float * dst, size_t n) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        dst[i] = dist(rng);
    }
}

// Weight generator: mostly well-scaled uniform data (fast groups), with
// planted features per 32-element group index g:
// - g % esc_period == 3 % esc_period: one element scaled by 2^-24 -> the
//                group's exponent window exceeds 16 binades -> MUST become
//                an escape group. The planted position walks the group
//                (incl. elements 0 and 31), and with esc_period coprime to
//                32 the escapes land on every group residue of every
//                1024-element block, including the first and last group of
//                a block (escape at block boundaries).
// - g % 11 == 5: two exact zeros, one of them negative (-0.0f) -> the
//                reserved zero code inside fast groups.
// - g % 13 == 8: the whole group is zeros (all-zero group).
// The exactness matrix uses esc_period = 7 (~13% escape groups — saturation
// coverage of the escape decode, far denser than reality). The bench uses
// esc_period = 997 (~0.1-0.15% incl. natural wide groups of uniform data —
// the REAL tensor class measured 0.06-0.14%): escape density is a
// first-order perf variable because different lanes of a simdgroup hold
// different 32-element groups, so a dense escape population makes nearly
// every wavefront execute BOTH decode paths.
static void fill_weights_p(float * dst, size_t n, size_t esc_period) {
    fill_uniform(dst, n);
    const size_t ngroups = n/GSZ;
    for (size_t g = 0; g < ngroups; g++) {
        float * w = dst + g*GSZ;
        if (g % esc_period == 3 % esc_period) {
            const size_t pos = (g/esc_period) % GSZ;
            w[pos] = std::ldexp(w[pos] == 0.0f ? 0.5f : w[pos], -24);
        }
        if (g % 11 == 5) {
            w[1] = 0.0f;
            w[17] = -0.0f;
        }
        if (g % 13 == 8) {
            for (int j = 0; j < GSZ; j++) {
                w[j] = (j % 2) ? 0.0f : -0.0f;
            }
        }
    }
}

static void fill_weights(float * dst, size_t n) {
    fill_weights_p(dst, n, 7);
}

// count escape groups (sb high bit) in an encoded NF8_M2 buffer
static void count_groups(const uint8_t * q, size_t qsize, size_t * n_fast, size_t * n_escape) {
    const size_t block_bytes = 1024 + 32;
    const size_t nb = qsize/block_bytes;
    *n_fast = *n_escape = 0;
    for (size_t b = 0; b < nb; b++) {
        const uint8_t * sb = q + b*block_bytes + 1024;
        for (int g = 0; g < 32; g++) {
            if (sb[g] & 0x80) {
                (*n_escape)++;
            } else {
                (*n_fast)++;
            }
        }
    }
}

// part 1a: on-grid data (values exactly e4m3 * 2^k) must round-trip
// memcmp-identical through quantize -> decode, with both group kinds present
static bool test_roundtrip_exact(void) {
    const int64_t n = 64*QK;

    // build values on the e4m3*2^k grid: per 32-group draw a k, then random
    // e4m3 codes (no NaN); plant the same features as fill_weights so wide
    // groups, signed zeros and all-zero groups are all exercised
    std::vector<float> src(n);
    std::uniform_int_distribution<int> dcode(0, 126); // e4m3 magnitude codes, 0..126 (127 = NaN)
    std::uniform_int_distribution<int> dsign(0, 1);
    std::uniform_int_distribution<int> dk(-20, -5);
    for (int64_t g = 0; g < n/GSZ; g++) {
        const int k = dk(rng);
        for (int j = 0; j < GSZ; j++) {
            const int  c  = dcode(rng);
            const int  e  = c >> 3;
            const int  m  = c & 7;
            float mag = e == 0 ? std::ldexp((float) m, -9) : std::ldexp(1.0f + m/8.0f, e - 7);
            float v   = std::ldexp(mag, k);
            src[g*GSZ + j] = dsign(rng) ? -v : v;
        }
        if (g % 7 == 3) { // widen to e4m3's full 18-binade envelope (the real
                          // pass-list wide blocks are e4m3*2^k by construction,
                          // so they always stay inside it): the smallest e4m3
                          // denormal (2^-9) next to the largest finite (448 =
                          // 1.75*2^8) spans exponents [k-9, k+8] -> width 18
                          // > 16 -> MUST escape, and must re-encode exactly
            const int pa = (int)((g/7) % GSZ);
            const int pb = (pa + 11) % GSZ;
            src[g*GSZ + pa] = std::ldexp(1.0f, k - 9);          // e4m3 denormal m=1
            src[g*GSZ + pb] = std::ldexp(-448.0f, k);           // e4m3 max finite
        }
        if (g % 11 == 5) {
            src[g*GSZ + 1] = 0.0f;
            src[g*GSZ + 17] = -0.0f;
        }
        if (g % 13 == 8) {
            for (int j = 0; j < GSZ; j++) {
                src[g*GSZ + j] = (j % 2) ? 0.0f : -0.0f;
            }
        }
    }

    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_NF8_M2, n));
    ggml_quantize_chunk(GGML_TYPE_NF8_M2, src.data(), q.data(), 0, 1, n, nullptr);

    size_t n_fast = 0, n_escape = 0;
    count_groups(q.data(), q.size(), &n_fast, &n_escape);

    std::vector<float> dec(n);
    ggml_get_type_traits(GGML_TYPE_NF8_M2)->to_float(q.data(), dec.data(), n);

    size_t n_bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&src[i], &dec[i], sizeof(float)) != 0) {
            if (n_bad < 8) {
                printf("  roundtrip[%" PRId64 "] %a -> %a (group %" PRId64 ", sb=0x%02x)\n",
                       i, src[i], dec[i], i/GSZ, q[(i/QK)*(1024 + 32) + 1024 + (i%QK)/GSZ]);
            }
            n_bad++;
        }
    }

    const bool ok = n_bad == 0 && n_escape > 0 && n_fast > 0;
    printf("%-24s: %s (%zu/%" PRId64 " values differ; %zu fast / %zu escape groups)\n",
           "roundtrip-exact", ok ? "OK" : "FAIL", n_bad, n, n_fast, n_escape);
    return ok;
}

// part 1a2: the --check-tensors validator accepts converter output and
// rejects the one invalid pattern (an e4m3 NaN code inside an escape group)
static bool test_validator(void) {
    const int64_t n = 16*QK;

    std::vector<float>   src(n);
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_NF8_M2, n));

    fill_weights(src.data(), n);
    ggml_quantize_chunk(GGML_TYPE_NF8_M2, src.data(), q.data(), 0, 1, n, nullptr);

    bool ok = ggml_validate_row_data(GGML_TYPE_NF8_M2, q.data(), q.size());
    if (!ok) {
        printf("%-24s: FAIL (valid data rejected)\n", "validator");
        return false;
    }

    // find an escape group and plant a NaN code in it
    const size_t block_bytes = 1024 + 32;
    bool planted = false;
    for (size_t b = 0; b < q.size()/block_bytes && !planted; b++) {
        uint8_t * blk = q.data() + b*block_bytes;
        for (int g = 0; g < 32; g++) {
            if (blk[1024 + g] & 0x80) {
                blk[g*32 + 7] = 0x7F; // e4m3 NaN code
                planted = true;
                break;
            }
        }
    }
    if (!planted) {
        printf("%-24s: FAIL (no escape group to corrupt)\n", "validator");
        return false;
    }

    fprintf(stderr, "(expected validator message follows)\n");
    ok = !ggml_validate_row_data(GGML_TYPE_NF8_M2, q.data(), q.size());

    printf("%-24s: %s (accepts valid data, rejects NaN code in escape group)\n", "validator", ok ? "OK" : "FAIL");
    return ok;
}

// part 1b: every decoded NF8_M2 value must be exactly a BF16 value
static bool test_decode_bf16_exact(void) {
    const int64_t n = 64*QK;

    std::vector<float>   src(n);
    std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_NF8_M2, n));
    std::vector<float>   dec(n);

    fill_weights(src.data(), n);
    ggml_quantize_chunk(GGML_TYPE_NF8_M2, src.data(), q.data(), 0, 1, n, nullptr);

    const auto * traits = ggml_get_type_traits(GGML_TYPE_NF8_M2);
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
// bench-dense (not run by ctest): NF8_M2 vs E4M3_M2 vs BF16 wall-clock A/B at
// the real dense-plane shapes. Same harness design as test-m2-e4m3
// bench-dense: dependency-serialized chains over cycled weight copies (~1 GiB
// per type) so every weight read is DRAM-cold, like production where every
// dense tensor is read cold once per token.

struct bench_chain {
    const char * name;
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
    const ggml_type types[3] = { GGML_TYPE_BF16, GGML_TYPE_E4M3_M2, GGML_TYPE_NF8_M2 };

    const int WARMUP = 3;
    const int REPS   = 10;

    for (const bench_chain & c : chains) {
        if (chain_filter != nullptr && strstr(c.name, chain_filter) == nullptr) {
            continue;
        }
        const int64_t w_bytes = c.k0*c.m0*2;
        const int NCOPY = (int)std::min<int64_t>(16, std::max<int64_t>(4, (1ll << 30)/std::max<int64_t>(1, (c.k1 ? 2 : 1)*w_bytes)));

        for (const int64_t nb : batches) {
            if (nb_filter > 0 && nb != nb_filter) {
                continue;
            }

            const int NREP = nb >= 2048 ? 4 : nb > 8 ? 8 : 64;

            double t_med[3] = {0.0, 0.0, 0.0};

            for (int ti = 0; ti < 3; ti++) {
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
                        x = ggml_view_2d(ctx, x, c.k0, nb, x->nb[1], 0);
                    } else {
                        x = ggml_mul_mat(ctx, w1[j % NCOPY], x);        // [m1, nb] == [k0, nb]
                    }
                }

                ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
                GGML_ASSERT(buf != nullptr);

                // weight values are irrelevant for timing but escape DENSITY
                // is not: use the generator at the real tensor class
                // (~0.1%; see fill_weights_p comment) — benching with the
                // matrix's 13% saturation density measured 1.3-2.3x
                // pathological ratios purely from simdgroup divergence
                for (int i = 0; i < NCOPY; i++) {
                    ggml_tensor * ws[2] = { w0[i], w1[i] };
                    for (ggml_tensor * w : ws) {
                        if (w == nullptr) {
                            continue;
                        }
                        const int64_t n_w = ggml_nelements(w);
                        std::vector<float> wf(n_w);
                        fill_weights_p(wf.data(), n_w, 997);
                        if (wtype == GGML_TYPE_BF16) {
                            std::vector<ggml_bf16_t> wb(n_w);
                            ggml_fp32_to_bf16_row(wf.data(), wb.data(), n_w);
                            ggml_backend_tensor_set(w, wb.data(), 0, ggml_nbytes(w));
                        } else {
                            std::vector<uint8_t> wq(ggml_nbytes(w));
                            ggml_quantize_chunk(wtype, wf.data(), wq.data(), 0, w->ne[1], w->ne[0], nullptr);
                            ggml_backend_tensor_set(w, wq.data(), 0, ggml_nbytes(w));
                            if (wtype == GGML_TYPE_NF8_M2 && i == 0 && w == w0[0]) {
                                size_t nf = 0, ne = 0;
                                count_groups(wq.data(), wq.size(), &nf, &ne);
                                printf("  (nf8 escape density: %.4f%%)\n", 100.0*ne/(double)(nf + ne));
                            }
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

            printf("  %-34s n=%-4" PRId64 " ratio nf8_m2/bf16 = %.4f   e4m3_m2/bf16 = %.4f   nf8_m2/e4m3_m2 = %.4f\n",
                   c.name, nb, t_med[2]/t_med[0], t_med[1]/t_med[0], t_med[2]/t_med[1]);
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
        const int64_t nb_filter    = argc > 2 ? atoll(argv[2]) : 0;
        const char *  chain_filter = argc > 3 ? argv[3] : nullptr;
        return run_bench_dense(gpu, nb_filter, chain_filter);
    }

    ok = test_roundtrip_exact() && ok;
    ok = test_validator() && ok;
    ok = test_decode_bf16_exact() && ok;

    ggml_backend_t gpu = find_gpu_backend();
    if (gpu == nullptr) {
        printf("no GPU backend available — skipping the Metal bit-exactness matrix "
               "(round-trip + decode exactness above still validated)\n");
        printf("%s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);

    printf("GPU backend: %s\n", ggml_backend_name(gpu));

    for (const test_shape & s : shapes) {
        // build the NF8_M2 weight once per shape (escape groups, zero codes
        // and all-zero groups planted by fill_weights, incl. at block
        // boundaries), and a BF16 weight holding bit-identical values
        const int64_t n_w = s.k*s.m*s.ne02;

        std::vector<float> wf(n_w);
        fill_weights(wf.data(), n_w);

        std::vector<uint8_t> wq(ggml_row_size(GGML_TYPE_NF8_M2, s.k)*s.m*s.ne02);
        ggml_quantize_chunk(GGML_TYPE_NF8_M2, wf.data(), wq.data(), 0, s.m*s.ne02, s.k, nullptr);

        size_t n_fast = 0, n_escape = 0;
        count_groups(wq.data(), wq.size(), &n_fast, &n_escape);
        if (n_escape == 0 || n_fast == 0) {
            printf("%-32s: FAIL (escape/zero coverage missing: %zu fast, %zu escape groups)\n", s.name, n_fast, n_escape);
            ok = false;
            continue;
        }

        std::vector<float> wdec(n_w);
        ggml_get_type_traits(GGML_TYPE_NF8_M2)->to_float(wq.data(), wdec.data(), n_w);

        std::vector<ggml_bf16_t> wbf(n_w);
        ggml_fp32_to_bf16_row(wdec.data(), wbf.data(), n_w);

        printf("%-32s: %zu fast / %zu escape groups (%.4f%% escape)\n",
               s.name, n_fast, n_escape, 100.0*n_escape/(double)(n_fast + n_escape));

        for (const int64_t n_batch : batch_sizes) {
            std::vector<float> x(s.k*n_batch*s.ne02);
            fill_uniform(x.data(), x.size());

            const std::vector<float> out_q  = run_mul_mat(gpu, GGML_TYPE_NF8_M2, wq.data(),  wq.size(),                       s, n_batch, x.data());
            const std::vector<float> out_bf = run_mul_mat(gpu, GGML_TYPE_BF16,   wbf.data(), wbf.size()*sizeof(ggml_bf16_t), s, n_batch, x.data());
            const std::vector<float> out_ref = run_mul_mat(cpu, GGML_TYPE_NF8_M2, wq.data(), wq.size(),                      s, n_batch, x.data());

            // bit-exactness vs the BF16 kernels
            size_t n_diff = 0;
            float max_abs = 0.0f;
            for (size_t i = 0; i < out_q.size(); i++) {
                if (memcmp(&out_q[i], &out_bf[i], sizeof(float)) != 0) {
                    if (n_diff < 4) {
                        printf("    diff[%zu]: nf8=%a bf16=%a\n", i, out_q[i], out_bf[i]);
                    }
                    n_diff++;
                    max_abs = std::max(max_abs, std::fabs(out_q[i] - out_bf[i]));
                }
            }

            // sanity vs the CPU reference (different accumulation order — NMSE).
            // n_batch > 8 dispatches the mul_mm path, which stages the B
            // operand (activations) as bfloat in threadgroup memory — the
            // exact numerics of the production BF16 mul_mm (and the BIT-EXACT
            // check above proves NF8_M2 inherits them precisely). The CPU
            // reference keeps activations f32, so that path carries an
            // inherent ~1e-6 NMSE vs the reference; the mul_mv/ext paths
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

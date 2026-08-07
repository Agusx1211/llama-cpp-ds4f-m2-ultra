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

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
// (r1 2..5), >8 -> mul_mm
static const int64_t batch_sizes[] = { 1, 2, 3, 5, 8, 16, 512 };

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

int main(void) {
    bool ok = true;

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

            // sanity vs the CPU reference (different accumulation order — NMSE)
            const double err = nmse(out_ref, out_q);

            const bool pass = n_diff == 0 && err < 1e-6;
            ok = pass && ok;

            printf("%-32s n_batch=%-4" PRId64 " vs-bf16: %s (%zu diffs, max %g)  vs-cpu nmse: %.3g %s\n",
                   s.name, n_batch,
                   n_diff == 0 ? "BIT-EXACT" : "FAIL", n_diff, (double) max_abs,
                   err, err < 1e-6 ? "OK" : "FAIL");
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    printf("%s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

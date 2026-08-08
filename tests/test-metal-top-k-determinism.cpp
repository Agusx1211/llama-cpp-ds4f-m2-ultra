// Regression test for the DeepSeek V4 Lightning Indexer top-k selection.
//
// ggml_metal_op_top_k routes k == 512 rows longer than 1024 elements to
// kernel_top_k_radix_f32_i32_impl. That kernel finds the k-th largest key with
// an MSD radix walk and then collects the partition. Because the indexer score
// row is ReLU'd, the pivot is routinely exactly 0.0f with a tie group thousands
// of elements wide, so the kernel has to *choose* which tied elements to keep.
//
// The original implementation appended both the strictly-greater and the tied
// elements to a threadgroup array using an atomic counter, i.e. in GPU arrival
// order. That made both the retained set and its order a function of thread
// scheduling: the same DeepSeek V4 Flash prompt produced different attention
// inputs, different logits and a different transcript on every run once the
// compressed cache passed 1024 rows (~4k prompt tokens). See
// notes/2026-08-08-dsv4-depth-determinism.md.
//
// This test pins the contract that replaced it:
//   1. repeated evaluations of the same row return byte-identical indices;
//   2. the result equals the unique "descending by value, ascending by source
//      index" top-k, which is a total order and therefore has no ties left.
//
// The test skips itself when no Metal device is present, so it is safe to run
// in CTest on any host.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

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

    // the Metal device reports the GPU's marketing name ("Apple M2 Ultra"),
    // so select on the backend registration (or, failing that, on any GPU)
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "?";
        // the Metal registration is named "MTL" in this tree
        if (std::strcmp(reg_name, "MTL") == 0 || std::strcmp(reg_name, "Metal") == 0) {
            return ggml_backend_dev_init(dev, nullptr);
        }
        if (gpu == nullptr && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
        }
    }

    return gpu ? ggml_backend_dev_init(gpu, nullptr) : nullptr;
}

// Reference: indices of the k largest values, ties broken by ascending index.
static std::vector<int32_t> reference_top_k(const float * row, int64_t n, int k) {
    std::vector<int32_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [row](int32_t a, int32_t b) {
        return row[a] > row[b];
    });
    idx.resize(k);
    return idx;
}

// A Lightning-Indexer-shaped score row: a few distinct positive scores, a very
// wide exact-zero tie group (the ReLU floor), and a masked tail at -inf.
static void fill_indexer_row(float * row, int64_t n, int n_positive, int n_masked, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.25f, 8.0f);

    for (int64_t i = 0; i < n; ++i) {
        row[i] = 0.0f;
    }
    for (int64_t i = n - n_masked; i < n; ++i) {
        row[i] = -INFINITY;
    }

    // spread the positives over the unmasked prefix at deterministic positions
    const int64_t span = n - n_masked;
    for (int i = 0; i < n_positive; ++i) {
        row[(int64_t) i*span/n_positive] = dist(rng);
    }
}

struct case_desc {
    int64_t     ne00;
    int64_t     ne01;
    int         k;
    int         n_positive;
    int         n_masked;
    // only the radix kernel promises the (value desc, index asc) tie-break;
    // the generic bitonic path breaks ties by network position, so it is
    // checked for reproducibility only
    bool        check_reference;
    const char * name;
};

static bool run_case(ggml_backend_t backend, const case_desc & c) {
    const size_t ctx_size = 4*ggml_tensor_overhead() + ggml_graph_overhead();

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.ne00, c.ne01);
    ggml_set_input(src);
    ggml_tensor * dst = ggml_top_k(ctx, src, c.k);
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != nullptr);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(ggml_gallocr_alloc_graph(galloc, gf));

    std::vector<float> host_src((size_t) c.ne00*c.ne01);
    for (int64_t r = 0; r < c.ne01; ++r) {
        fill_indexer_row(host_src.data() + r*c.ne00, c.ne00, c.n_positive, c.n_masked, 1234u + (uint32_t) r);
    }
    ggml_backend_tensor_set(src, host_src.data(), 0, ggml_nbytes(src));

    const size_t n_out = (size_t) c.k*c.ne01;
    std::vector<int32_t> first(n_out);
    std::vector<int32_t> cur(n_out);

    bool ok = true;
    const int n_runs = 8;
    for (int run = 0; run < n_runs; ++run) {
        assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(dst, cur.data(), 0, n_out*sizeof(int32_t));

        if (run == 0) {
            first = cur;
            continue;
        }
        if (cur != first) {
            size_t d = 0;
            while (d < n_out && cur[d] == first[d]) {
                ++d;
            }
            std::fprintf(stderr,
                    "%s: run %d differs from run 0 at element %zu (%d vs %d)\n",
                    c.name, run, d, (int) first[d], (int) cur[d]);
            ok = false;
            break;
        }
    }

    if (ok && c.check_reference) {
        for (int64_t r = 0; r < c.ne01 && ok; ++r) {
            const auto ref = reference_top_k(host_src.data() + r*c.ne00, c.ne00, c.k);
            for (int i = 0; i < c.k; ++i) {
                if (first[(size_t) r*c.k + i] != ref[i]) {
                    std::fprintf(stderr,
                            "%s: row %lld element %d = %d, reference (value desc, index asc) = %d\n",
                            c.name, (long long) r, i,
                            (int) first[(size_t) r*c.k + i], (int) ref[i]);
                    ok = false;
                    break;
                }
            }
        }
    }

    std::printf("%-28s ne00=%5lld ne01=%lld k=%d positives=%d masked=%d ref=%s : %s\n",
            c.name, (long long) c.ne00, (long long) c.ne01, c.k, c.n_positive, c.n_masked,
            c.check_reference ? "yes" : "no", ok ? "OK" : "FAILED");

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return ok;
}

int main() {
    ggml_backend_t backend = metal_backend_init();
    if (backend == nullptr) {
        std::puts("metal top-k determinism test skipped: no Metal device");
        return 0;
    }

    // ne00 > 1024 with k == 512 is exactly the routing condition for the radix
    // kernel in ggml_metal_op_top_k. 1024 stays on the bitonic path and is kept
    // as a control: it must be reproducible too.
    const case_desc cases[] = {
        { 1024, 2, 512,  64,   0, false, "bitonic-control"          },
        { 2048, 2, 512, 100,   0, true,  "radix/wide-zero-tie"      },
        { 4096, 3, 512,  50, 512, true,  "radix/zero-tie+masked"    },
        { 9472, 2, 512, 700, 256, true,  "radix/deep-context-like"  },
        { 4096, 2, 512, 900,   0, true,  "radix/mostly-positive"    },
    };

    bool ok = true;
    for (const auto & c : cases) {
        ok = run_case(backend, c) && ok;
    }

    ggml_backend_free(backend);

    if (!ok) {
        std::puts("metal top-k determinism test FAILED");
        return 1;
    }

    std::puts("metal top-k determinism test OK");
    return 0;
}

// Regression test for Metal buffer lifetime under in-flight GPU work.
//
// The Metal backend submits graphs asynchronously: ggml_backend_graph_compute
// (async form) returns as soon as the command buffers are committed, and the
// GPU keeps executing. Command buffers are created with
// commandBufferWithUnretainedReferences, so the driver holds no reference to
// the MTLBuffers they touch, and compute buffers on this box are *shared*
// buffers whose backing store is vm_allocate()d host memory that
// ggml_metal_buffer_free() vm_deallocate()s. Releasing such a buffer while the
// GPU is still reading it is a hard fault: the command buffer completes with
// MTLCommandBufferStatusError (status 5) and every buffer behind it on the
// queue reports kIOGPUCommandBufferCallbackErrorSubmissionsIgnored.
//
// This is not a theoretical hazard. ggml_backend_sched_alloc_splits() reaches
// exactly this state when a graph outgrows its reservation: it frees and
// reallocates the compute buffers at runtime, having "synchronized" only
// through the context-level ggml_metal_synchronize(), which cannot see work
// submitted straight to the shared device queue. A target-only DeepSeek V4
// Flash server faulted reproducibly at the 4th 2048-token prefill ubatch for
// this reason (notes/2026-08-08-dsv4-depth-determinism.md, defect 2;
// notes/2026-08-08-metal-realloc-safety.md).
//
// The contract pinned here: freeing a Metal buffer is safe at any time, because
// ggml_metal_buffer_free() drains the device command queue first
// ([TAG_QUEUE_DRAIN]). The test dispatches a graph that keeps the GPU busy for
// tens of milliseconds, frees the compute buffer it is reading without
// synchronizing, and then asserts that the backend is still healthy - the
// Metal backend latches has_error and fails every later graph, so a subsequent
// compute returning GGML_STATUS_SUCCESS is a sufficient check.
//
// Skips itself with status 0 when no Metal device is present.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// A chain of large matmuls: enough GPU work that the graph is guaranteed to be
// still executing when the host returns from the async submit.
struct heavy_graph {
    ggml_context *        ctx  = nullptr;
    ggml_backend_buffer_t buf  = nullptr;  // weights / inputs
    ggml_gallocr_t        gall = nullptr;  // compute buffer
    ggml_cgraph *         gf   = nullptr;

    void free() {
        if (gall) { ggml_gallocr_free(gall);        gall = nullptr; }
        if (buf)  { ggml_backend_buffer_free(buf);  buf  = nullptr; }
        if (ctx)  { ggml_free(ctx);                 ctx  = nullptr; }
    }
};

static heavy_graph build(ggml_backend_t backend, int64_t n, int depth) {
    heavy_graph g;

    const size_t ctx_size = (size_t) (2*depth + 8)*ggml_tensor_overhead() + ggml_graph_overhead();

    ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    g.ctx = ggml_init(params);
    assert(g.ctx != nullptr);

    ggml_tensor * w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n, n);
    ggml_set_input(w);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, n, n);
    ggml_set_input(x);

    ggml_tensor * cur = x;
    for (int i = 0; i < depth; ++i) {
        cur = ggml_mul_mat(g.ctx, w, cur);
    }
    ggml_set_output(cur);

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, cur);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    assert(g.buf != nullptr);

    g.gall = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    assert(g.gall != nullptr);
    assert(ggml_gallocr_alloc_graph(g.gall, g.gf));

    // small values: the point is the dispatch, not the numerics, and this keeps
    // the intermediate magnitudes finite through the chain
    std::vector<float> host((size_t) n*n, 0.001f);
    ggml_backend_tensor_set(w, host.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, host.data(), 0, ggml_nbytes(x));

    return g;
}

// Dispatch the heavy graph without waiting, then destroy the compute buffer it
// is reading. Pre-fix this faults the GPU; post-fix ggml_metal_buffer_free()
// drains the queue and the free is ordered after the graph.
static bool case_free_compute_buffer_under_inflight_graph(ggml_backend_t backend, int n_iter) {
    bool ok = true;

    for (int it = 0; it < n_iter && ok; ++it) {
        heavy_graph g = build(backend, 2048, 24);

        const enum ggml_status st = ggml_backend_graph_compute_async(backend, g.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "iteration %d: async submit returned %d\n", it, (int) st);
            g.free();
            return false;
        }

        // no ggml_backend_synchronize() here - that is the whole point
        ggml_gallocr_free(g.gall);
        g.gall = nullptr;

        ggml_backend_synchronize(backend);

        // the Metal backend latches has_error, so any later graph fails if the
        // free above faulted the GPU
        heavy_graph probe = build(backend, 256, 2);
        const enum ggml_status st_probe = ggml_backend_graph_compute(backend, probe.gf);
        if (st_probe != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                    "iteration %d: backend is in error state after freeing a compute buffer "
                    "under an in-flight graph (probe status %d)\n", it, (int) st_probe);
            ok = false;
        }
        probe.free();

        g.free();
    }

    std::printf("%-44s iters=%d : %s\n", "free-compute-buffer-under-inflight", n_iter, ok ? "OK" : "FAILED");

    return ok;
}

// Same hazard, reached through the tensor (weights) buffer rather than the
// compute buffer: ggml_backend_buffer_free() on data the running graph reads.
static bool case_free_tensor_buffer_under_inflight_graph(ggml_backend_t backend, int n_iter) {
    bool ok = true;

    for (int it = 0; it < n_iter && ok; ++it) {
        heavy_graph g = build(backend, 2048, 24);

        const enum ggml_status st = ggml_backend_graph_compute_async(backend, g.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "iteration %d: async submit returned %d\n", it, (int) st);
            g.free();
            return false;
        }

        ggml_backend_buffer_free(g.buf);
        g.buf = nullptr;

        ggml_backend_synchronize(backend);

        heavy_graph probe = build(backend, 256, 2);
        const enum ggml_status st_probe = ggml_backend_graph_compute(backend, probe.gf);
        if (st_probe != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                    "iteration %d: backend is in error state after freeing a tensor buffer "
                    "under an in-flight graph (probe status %d)\n", it, (int) st_probe);
            ok = false;
        }
        probe.free();

        g.free();
    }

    std::printf("%-44s iters=%d : %s\n", "free-tensor-buffer-under-inflight", n_iter, ok ? "OK" : "FAILED");

    return ok;
}

int main() {
    ggml_backend_t backend = metal_backend_init();
    if (backend == nullptr) {
        std::puts("metal buffer lifetime test skipped: no Metal device");
        return 0;
    }

    bool ok = true;
    ok = case_free_compute_buffer_under_inflight_graph(backend, 8) && ok;
    ok = case_free_tensor_buffer_under_inflight_graph (backend, 8) && ok;

    ggml_backend_free(backend);

    std::printf("%s\n", ok ? "OK" : "FAILED");

    return ok ? 0 : 1;
}

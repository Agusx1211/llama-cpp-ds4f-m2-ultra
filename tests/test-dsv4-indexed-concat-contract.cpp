#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>

#include <sys/wait.h>
#include <unistd.h>

static constexpr int rejection_exit_code = 125;

static void abort_as_rejection(const char *) {
    std::_Exit(rejection_exit_code);
}

static ggml_context * make_context() {
    const ggml_init_params params = {
        /*.mem_size   =*/ 2*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::abort();
    }
    return ctx;
}

static bool rejects(const std::function<void()> & fn) {
    const pid_t pid = fork();
    if (pid < 0) {
        return false;
    }
    if (pid == 0) {
        ggml_set_abort_callback(abort_as_rejection);
        fn();
        std::_Exit(0);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) != pid) {
        return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == rejection_exit_code;
}

static bool valid_output_matches_affine_concat() {
    constexpr int64_t d = 8;
    constexpr int64_t n_raw = 2;
    constexpr int64_t n_comp = 65;
    constexpr int64_t n_stream = 2;

    ggml_context * ctx = make_context();
    ggml_tensor * raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, 1, n_raw, n_stream);
    ggml_tensor * pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, d, 4*64);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n_stream);

    ggml_fp16_t * raw_data = (ggml_fp16_t *) raw->data;
    for (int64_t i = 0; i < ggml_nelements(raw); ++i) {
        raw_data[i] = ggml_fp32_to_fp16((float) (1000 + i));
    }
    ggml_fp16_t * pool_data = (ggml_fp16_t *) pool->data;
    for (int64_t i = 0; i < ggml_nelements(pool); ++i) {
        pool_data[i] = ggml_fp32_to_fp16((float) (i % 2048));
    }
    int32_t * directory = (int32_t *) ids->data;
    directory[0] = 3;
    directory[1] = 1;
    directory[2] = 0;
    directory[3] = 2;

    ggml_tensor * out = ggml_dsv4_indexed_concat(ctx, raw, pool, ids, n_comp);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 2);

    const ggml_fp16_t * out_data = (const ggml_fp16_t *) out->data;
    bool matches = true;
    for (int64_t stream = 0; stream < n_stream && matches; ++stream) {
        for (int64_t row = 0; row < n_raw + n_comp && matches; ++row) {
            const ggml_fp16_t * expected;
            if (row < n_raw) {
                expected = raw_data + d*(stream*n_raw + row);
            } else {
                const int64_t logical = row - n_raw;
                const int64_t physical = (int64_t) directory[stream*2 + logical/64]*64 + logical%64;
                expected = pool_data + d*physical;
            }
            const ggml_fp16_t * actual = out_data + d*(stream*(n_raw + n_comp) + row);
            matches = std::equal(expected, expected + d, actual);
        }
    }

    ggml_free(ctx);
    return matches;
}

static void invalid_pool_shape() {
    ggml_context * ctx = make_context();
    ggml_tensor * raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 1, 1, 1);
    ggml_tensor * pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 65);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    (void) ggml_dsv4_indexed_concat(ctx, raw, pool, ids, 1);
    ggml_free(ctx);
}

static void invalid_directory_shape() {
    ggml_context * ctx = make_context();
    ggml_tensor * raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 1, 1, 1);
    ggml_tensor * pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 128);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    (void) ggml_dsv4_indexed_concat(ctx, raw, pool, ids, 65);
    ggml_free(ctx);
}

static void invalid_directory_index() {
    ggml_context * ctx = make_context();
    ggml_tensor * raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, 1, 1, 1);
    ggml_tensor * pool = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 8, 64);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    *(int32_t *) ids->data = 1; // The only valid physical segment is zero.
    ggml_tensor * out = ggml_dsv4_indexed_concat(ctx, raw, pool, ids, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 2);
    ggml_free(ctx);
}

int main() {
    const bool pool_rejected = rejects(invalid_pool_shape);
    const bool shape_rejected = rejects(invalid_directory_shape);
    const bool index_rejected = rejects(invalid_directory_index);
    // Run the non-forking oracle last: forking after the CPU worker pool has
    // initialized can inherit library locks from helper threads.
    const bool output_matches = valid_output_matches_affine_concat();
    if (!output_matches || !pool_rejected || !shape_rejected || !index_rejected) {
        std::cerr << "indexed concat contract failed: output=" << output_matches << " pool=" << pool_rejected
                  << " directory=" << shape_rejected << " index=" << index_rejected << '\n';
        return 1;
    }

    std::cout << "indexed concat matches affine output and rejects invalid pool shape, directory shape, and segment index\n";
    return 0;
}

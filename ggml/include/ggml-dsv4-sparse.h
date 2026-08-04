#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

// Neutral procedure-table ABI for the dormant DSV4 sparse ownership path.
// Registry callers and backend wrappers use int exactly; backend-private enums
// are converted at the wrapper boundary.
enum ggml_dsv4_sparse_status {
    GGML_DSV4_SPARSE_OK = 0,
    GGML_DSV4_SPARSE_PRESSURE,
    GGML_DSV4_SPARSE_STALE,
    GGML_DSV4_SPARSE_INVALID,
    GGML_DSV4_SPARSE_OOM,
    GGML_DSV4_SPARSE_UNSUPPORTED,
};

typedef int (*ggml_dsv4_sparse_move_quote_fn)(
        struct ggml_tensor * const * sources,
        struct ggml_tensor * const * destinations,
        size_t n_tensors,
        void ** quote);
typedef int  (*ggml_dsv4_sparse_move_commit_fn)(void * quote);
typedef void (*ggml_dsv4_sparse_move_free_fn)(void * quote);

#ifdef __cplusplus
}
#endif

#pragma once

#include <stddef.h>
#include <stdint.h>

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

// Optional test-only inspection of one prepared sparse move quote. The
// callback receives the same opaque quote that the commit function consumes;
// committed is zero for the pre-commit snapshot and nonzero for the
// post-commit snapshot. The fixed pool bound covers the RAW/SWA resident
// planes while keeping this ABI allocation-free at the audit boundary.
#define GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS 8

struct ggml_dsv4_sparse_move_audit_pool {
    uint64_t pool_id;
    uint64_t generation;
    uint64_t virtual_move_count;
    uint64_t mapped_source_count;
    uint64_t source_unique_physical_count;
    uint64_t source_released_physical_count;
    uint64_t source_refcount_sum;
    uint64_t destination_page_count;
    uint64_t mapping_operation_count;
    uint64_t source_virtual_hash;
    uint64_t source_physical_hash;
    uint64_t source_refcount_hash;
    uint64_t survivor_mapping_hash;
    uint64_t free_pages;
    uint64_t mapped_mappings;
    uint64_t unique_physical_pages;
    uint64_t shared_physical_pages;
    uint64_t shared_mappings;
    uint64_t refcount_sum;
    uint32_t refcount_max;
};

struct ggml_dsv4_sparse_move_audit {
    size_t n_pools;
    struct ggml_dsv4_sparse_move_audit_pool pools[GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS];
};

typedef int (*ggml_dsv4_sparse_move_audit_fn)(
        void * quote,
        int committed,
        struct ggml_dsv4_sparse_move_audit * audit);

#ifdef __cplusplus
}
#endif

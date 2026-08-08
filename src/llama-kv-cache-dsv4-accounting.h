#pragma once

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

// Pure-host aggregation helpers kept separate from backend usage collection so
// shared placement-sparse pools can be tested without Metal.

// M2 Ultra has 192 GiB of unified memory. Below this declared compressed-K
// footprint, the affine per-stream layout is intentionally preferred because
// it avoids the placement-sparse mapping/submission cost. The declared bytes
// are virtual on the placement-sparse pool (physical pages materialize per
// used row), so the bound guards address-space/bookkeeping growth rather than
// resident memory. Measured 2026-08-08: a 4-lane unified 1M-context config
// declares slightly over 20 GiB (the earlier ~18.8 GiB estimate missed
// per-layer/per-seq padding), so the previous 20 GiB bound made the exact
// admission vertical refuse 1M configs. 28 GiB covers 4 unified lanes at 1M
// with margin; configurations beyond this bound still fall back to the
// aggregate pool and the vertical refuses to start (loudly) instead of
// degrading.
static constexpr uint64_t LLAMA_DSV4_MAX_AFFINE_COMPRESSED_BYTES = UINT64_C(28)*1024*1024*1024;

struct llama_dsv4_aggregate_selector {
    bool     unified                   = false;
    bool     sparse_supported          = false;
    bool     f16                       = false;
    bool     disabled                  = false;
    bool     forced                    = false;
    bool     affine_footprint_overflow = false;
    uint32_t n_seq_max                 = 0;
    uint64_t affine_compressed_bytes   = 0;
};

static inline bool dsv4_select_aggregate_compressed(
        const llama_dsv4_aggregate_selector & selector) {
    if (selector.disabled || !selector.unified || selector.n_seq_max <= 1 ||
            !selector.sparse_supported || !selector.f16) {
        return false;
    }

    return selector.forced || selector.affine_footprint_overflow ||
            selector.affine_compressed_bytes > LLAMA_DSV4_MAX_AFFINE_COMPRESSED_BYTES;
}

// Add count tensors with the supplied row geometry to an exact declared-byte
// total. Saturation makes overflow select the memory-saving aggregate layout.
static inline bool dsv4_affine_compressed_bytes_add(
        uint64_t & total,
        uint64_t   row_bytes,
        uint64_t   rows,
        uint64_t   n_stream,
        uint64_t   count = 1) {
    uint64_t value = row_bytes;
    for (uint64_t factor : { rows, n_stream, count }) {
        if (factor != 0 && value > std::numeric_limits<uint64_t>::max()/factor) {
            total = std::numeric_limits<uint64_t>::max();
            return false;
        }
        value *= factor;
    }
    if (total > std::numeric_limits<uint64_t>::max() - value) {
        total = std::numeric_limits<uint64_t>::max();
        return false;
    }
    total += value;
    return true;
}

static inline uint32_t dsv4_state_n_used_k_rows(
        llama_pos pos_max,
        uint32_t ratio,
        uint32_t kv_size) {
    if (pos_max < 0) {
        return 0;
    }
    const uint64_t n_rows = ((uint64_t) pos_max + 1)/ratio;
    return (uint32_t) std::min<uint64_t>(kv_size, n_rows);
}

static inline bool dsv4_sparse_usage_equal(
        const llama_dsv4_sparse_pool_usage & lhs,
        const llama_dsv4_sparse_pool_usage & rhs) {
    return lhs.pool_id               == rhs.pool_id &&
            lhs.page_size             == rhs.page_size &&
            lhs.virtual_pages         == rhs.virtual_pages &&
            lhs.physical_pages        == rhs.physical_pages &&
            lhs.free_pages            == rhs.free_pages &&
            lhs.reserved_pages        == rhs.reserved_pages &&
            lhs.mapped_mappings       == rhs.mapped_mappings &&
            lhs.unique_physical_pages == rhs.unique_physical_pages &&
            lhs.shared_physical_pages == rhs.shared_physical_pages &&
            lhs.shared_mappings       == rhs.shared_mappings &&
            lhs.refcount_sum          == rhs.refcount_sum &&
            lhs.refcount_max          == rhs.refcount_max &&
            lhs.generation            == rhs.generation &&
            lhs.cow_allocations       == rhs.cow_allocations &&
            lhs.cow_pages             == rhs.cow_pages;
}

static inline bool dsv4_sparse_pool_insert(
        std::map<uintptr_t, llama_dsv4_sparse_pool_usage> & pools,
        const llama_dsv4_sparse_pool_usage & pool) {
    const auto [it, inserted] = pools.emplace(pool.pool_id, pool);
    return inserted || dsv4_sparse_usage_equal(it->second, pool);
}

static inline void dsv4_sparse_usage_add(
        llama_dsv4_sparse_pool_usage & dst,
        const llama_dsv4_sparse_pool_usage & src) {
    const bool dst_empty = dst.virtual_pages == 0 && dst.physical_pages == 0;
    if (dst_empty) {
        dst.page_size = src.page_size;
    } else if (dst.page_size != src.page_size) {
        dst.page_size = 0;
    }
    dst.virtual_pages += src.virtual_pages;
    dst.physical_pages += src.physical_pages;
    dst.free_pages += src.free_pages;
    dst.reserved_pages += src.reserved_pages;
    dst.mapped_mappings += src.mapped_mappings;
    dst.unique_physical_pages += src.unique_physical_pages;
    dst.shared_physical_pages += src.shared_physical_pages;
    dst.shared_mappings += src.shared_mappings;
    dst.refcount_sum += src.refcount_sum;
    dst.refcount_max = std::max(dst.refcount_max, src.refcount_max);
    dst.generation = std::max(dst.generation, src.generation);
    dst.cow_allocations += src.cow_allocations;
    dst.cow_pages += src.cow_pages;
}

static inline bool dsv4_family_rebuild_sparse_total(llama_dsv4_family_usage & family) {
    std::map<uintptr_t, llama_dsv4_sparse_pool_usage> unique;
    for (const auto & pool : family.pools) {
        if (!dsv4_sparse_pool_insert(unique, pool)) {
            return false;
        }
    }

    family.pools.clear();
    family.total = {};
    for (const auto & [_, pool] : unique) {
        family.pools.push_back(pool);
        dsv4_sparse_usage_add(family.total, pool);
    }
    family.placement_sparse = !family.pools.empty();
    return true;
}

// Merge only physical pool views. DSV4's ISWA base cache has full-context
// bookkeeping but no raw-attention tensor layers, so its logical capacity must
// not inflate the actual SWA-backed RAW family.
static inline bool dsv4_family_sparse_usage_merge(
        llama_dsv4_family_usage & dst,
        const llama_dsv4_family_usage & src) {
    if (dst.family != src.family) {
        return false;
    }

    llama_dsv4_family_usage merged = dst;
    merged.pools.insert(merged.pools.end(), src.pools.begin(), src.pools.end());
    if (!dsv4_family_rebuild_sparse_total(merged)) {
        return false;
    }
    dst = std::move(merged);
    return true;
}

// A family pool is a whole-buffer view, not disjoint attribution. The same
// pool can therefore appear under CSA, HCA, and LID. Global totals count one
// canonical immutable observation and retain every viewing family in a mask.
static inline bool dsv4_memory_usage_finalize(
        llama_dsv4_memory_usage_snapshot & snapshot) {
    struct pool_view {
        llama_dsv4_memory_family family;
        uint32_t family_mask;
        llama_dsv4_sparse_pool_usage usage;
    };
    std::map<uintptr_t, pool_view> unique;
    for (const auto & family : snapshot.families) {
        for (const auto & pool : family.pools) {
            const uint32_t family_bit = 1u << family.family;
            const auto [it, inserted] = unique.emplace(
                    pool.pool_id, pool_view { family.family, family_bit, pool });
            if (!inserted) {
                if (!dsv4_sparse_usage_equal(it->second.usage, pool)) {
                    return false;
                }
                it->second.family_mask |= family_bit;
            }
        }
    }

    snapshot.sparse_total = {};
    snapshot.limiting_family = LLAMA_DSV4_MEMORY_RAW;
    snapshot.limiting_family_mask = 0;
    snapshot.limiting_pool_id = 0;
    snapshot.limiting_available_pages = 0;
    bool have_limit = false;
    for (const auto & [_, view] : unique) {
        dsv4_sparse_usage_add(snapshot.sparse_total, view.usage);
        const uint64_t available = view.usage.reserved_pages <= view.usage.free_pages ?
                view.usage.free_pages - view.usage.reserved_pages : 0;
        if (!have_limit || available < snapshot.limiting_available_pages) {
            have_limit = true;
            snapshot.limiting_family = view.family;
            snapshot.limiting_family_mask = view.family_mask;
            snapshot.limiting_pool_id = view.usage.pool_id;
            snapshot.limiting_available_pages = available;
        }
    }
    return true;
}

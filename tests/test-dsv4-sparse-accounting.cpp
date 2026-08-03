#include "llama-kv-cache-dsv4-accounting.h"

#include <cassert>
#include <cstdlib>
#include <iostream>

#undef assert
#define assert(expr) do {                                                        \
    if (!(expr)) {                                                               \
        std::cerr << "check failed at " << __FILE__ << ':' << __LINE__          \
                  << ": " #expr << '\n';                                       \
        std::abort();                                                            \
    }                                                                            \
} while (false)

static llama_dsv4_sparse_pool_usage pool(
        uintptr_t id,
        uint64_t page_size,
        uint64_t physical,
        uint64_t free,
        uint64_t reserved,
        uint64_t generation) {
    llama_dsv4_sparse_pool_usage result;
    result.pool_id = id;
    result.page_size = page_size;
    result.virtual_pages = physical*2;
    result.physical_pages = physical;
    result.free_pages = free;
    result.reserved_pages = reserved;
    result.mapped_mappings = physical - free;
    result.unique_physical_pages = physical - free;
    result.refcount_sum = result.mapped_mappings;
    result.refcount_max = result.mapped_mappings > 0;
    result.generation = generation;
    return result;
}

static void test_raw_family_merge_preserves_swa_logical_rows() {
    llama_dsv4_family_usage swa;
    swa.family = LLAMA_DSV4_MEMORY_RAW;
    swa.logical_capacity_rows = 32;
    swa.sequence_mapped_rows = { 10, 7 };
    swa.logical_mapped_rows = 17;
    swa.pools = { pool(10, 64*1024, 10, 6, 1, 3) };
    assert(dsv4_family_rebuild_sparse_total(swa));

    llama_dsv4_family_usage base;
    base.family = LLAMA_DSV4_MEMORY_RAW;
    base.logical_capacity_rows = 64;
    base.sequence_mapped_rows = { 12, 5 };
    base.logical_mapped_rows = 17;
    base.pools = {
        pool(10, 64*1024, 10, 6, 1, 3),
        pool(20, 64*1024, 8, 2, 0, 8),
    };
    assert(dsv4_family_rebuild_sparse_total(base));

    assert(dsv4_family_sparse_usage_merge(swa, base));
    assert(swa.logical_capacity_rows == 32);
    assert(swa.sequence_mapped_rows == std::vector<uint64_t>({ 10, 7 }));
    assert(swa.logical_mapped_rows == 17);
    assert(swa.pools.size() == 2);
    assert(swa.total.physical_pages == 18);
    assert(swa.total.free_pages == 8);
    assert(swa.total.reserved_pages == 1);
    assert(swa.total.generation == 8);
    assert(swa.placement_sparse);

    base.family = LLAMA_DSV4_MEMORY_CSA;
    assert(!dsv4_family_sparse_usage_merge(swa, base));
}

static void test_snapshot_total_deduplicates_cross_family_pool() {
    llama_dsv4_memory_usage_snapshot snapshot;
    auto & raw = snapshot.families[LLAMA_DSV4_MEMORY_RAW];
    raw.family = LLAMA_DSV4_MEMORY_RAW;
    raw.pools = {
        pool(10, 64*1024, 10, 6, 1, 3),
        pool(20, 64*1024, 8, 1, 1, 8),
    };
    assert(dsv4_family_rebuild_sparse_total(raw));

    auto & csa = snapshot.families[LLAMA_DSV4_MEMORY_CSA];
    csa.family = LLAMA_DSV4_MEMORY_CSA;
    csa.pools = {
        pool(20, 64*1024, 8, 1, 1, 8),
        pool(30, 64*1024, 4, 3, 1, 2),
    };
    assert(dsv4_family_rebuild_sparse_total(csa));

    snapshot.families[LLAMA_DSV4_MEMORY_HCA].family = LLAMA_DSV4_MEMORY_HCA;
    snapshot.families[LLAMA_DSV4_MEMORY_LID].family = LLAMA_DSV4_MEMORY_LID;
    assert(dsv4_memory_usage_finalize(snapshot));

    assert(snapshot.sparse_total.physical_pages == 22);
    assert(snapshot.sparse_total.free_pages == 10);
    assert(snapshot.sparse_total.reserved_pages == 3);
    assert(snapshot.sparse_total.page_size == 64*1024);
    assert(snapshot.sparse_total.generation == 8);
    assert(snapshot.limiting_pool_id == 20);
    assert(snapshot.limiting_family == LLAMA_DSV4_MEMORY_RAW);
    assert(snapshot.limiting_family_mask ==
            ((1u << LLAMA_DSV4_MEMORY_RAW) | (1u << LLAMA_DSV4_MEMORY_CSA)));
    assert(snapshot.limiting_available_pages == 0);
}

static void test_mixed_page_sizes_clear_aggregate_page_size() {
    llama_dsv4_family_usage family;
    family.family = LLAMA_DSV4_MEMORY_HCA;
    family.pools = {
        pool(1, 64*1024, 2, 1, 0, 1),
        pool(2, 16*1024, 2, 1, 0, 1),
        pool(3, 16*1024, 2, 1, 0, 1),
    };
    assert(dsv4_family_rebuild_sparse_total(family));
    assert(family.total.page_size == 0);
    assert(family.total.physical_pages == 6);
}

static void test_divergent_duplicate_pool_is_rejected() {
    llama_dsv4_family_usage family;
    family.family = LLAMA_DSV4_MEMORY_CSA;
    family.pools = {
        pool(7, 64*1024, 8, 4, 0, 2),
        pool(7, 64*1024, 8, 4, 1, 2),
    };
    assert(!dsv4_family_rebuild_sparse_total(family));

    llama_dsv4_memory_usage_snapshot snapshot;
    snapshot.families[LLAMA_DSV4_MEMORY_RAW].family = LLAMA_DSV4_MEMORY_RAW;
    snapshot.families[LLAMA_DSV4_MEMORY_RAW].pools = {
        pool(9, 64*1024, 8, 4, 0, 2),
    };
    snapshot.families[LLAMA_DSV4_MEMORY_CSA].family = LLAMA_DSV4_MEMORY_CSA;
    snapshot.families[LLAMA_DSV4_MEMORY_CSA].pools = {
        pool(9, 64*1024, 8, 4, 0, 3),
    };
    assert(!dsv4_memory_usage_finalize(snapshot));
}

static void test_empty_snapshot_and_compressed_row_boundaries() {
    llama_dsv4_memory_usage_snapshot empty;
    for (size_t i = 0; i < empty.families.size(); ++i) {
        empty.families[i].family = (llama_dsv4_memory_family) i;
    }
    assert(dsv4_memory_usage_finalize(empty));
    assert(empty.sparse_total.physical_pages == 0);
    assert(empty.limiting_family_mask == 0 && empty.limiting_pool_id == 0);

    assert(dsv4_state_n_used_k_rows(-1, 4, 16) == 0);
    assert(dsv4_state_n_used_k_rows(0, 4, 16) == 0);
    assert(dsv4_state_n_used_k_rows(3, 4, 16) == 1);
    assert(dsv4_state_n_used_k_rows(4, 4, 16) == 1);
    assert(dsv4_state_n_used_k_rows(7, 4, 16) == 2);
    assert(dsv4_state_n_used_k_rows(127, 128, 16) == 1);
    assert(dsv4_state_n_used_k_rows(128, 128, 16) == 1);
    assert(dsv4_state_n_used_k_rows(4095, 4, 16) == 16);
}

static void test_limiting_pool_ties_are_deterministic() {
    llama_dsv4_memory_usage_snapshot snapshot;
    auto & raw = snapshot.families[LLAMA_DSV4_MEMORY_RAW];
    raw.family = LLAMA_DSV4_MEMORY_RAW;
    raw.pools = { pool(50, 64*1024, 8, 3, 1, 1) };
    auto & csa = snapshot.families[LLAMA_DSV4_MEMORY_CSA];
    csa.family = LLAMA_DSV4_MEMORY_CSA;
    csa.pools = { pool(40, 64*1024, 8, 3, 1, 1) };
    snapshot.families[LLAMA_DSV4_MEMORY_HCA].family = LLAMA_DSV4_MEMORY_HCA;
    snapshot.families[LLAMA_DSV4_MEMORY_LID].family = LLAMA_DSV4_MEMORY_LID;

    assert(dsv4_memory_usage_finalize(snapshot));
    assert(snapshot.limiting_available_pages == 2);
    assert(snapshot.limiting_pool_id == 40);
    assert(snapshot.limiting_family == LLAMA_DSV4_MEMORY_CSA);
}

int main() {
    test_raw_family_merge_preserves_swa_logical_rows();
    test_snapshot_total_deduplicates_cross_family_pool();
    test_mixed_page_sizes_clear_aggregate_page_size();
    test_divergent_duplicate_pool_is_rejected();
    test_empty_snapshot_and_compressed_row_boundaries();
    test_limiting_pool_ties_are_deterministic();
    std::cout << "DSV4 sparse accounting tests passed\n";
    return 0;
}

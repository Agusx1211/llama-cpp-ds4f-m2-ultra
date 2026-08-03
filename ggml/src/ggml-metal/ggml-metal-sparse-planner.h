#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Pure-host accounting shared by the Metal allocator and its Linux tests.
// Mapping submission remains in ggml-metal-device.m; this header only plans
// exact page effects and manages reservation counters.

enum ggml_metal_sparse_plan_status {
    GGML_METAL_SPARSE_PLAN_OK = 0,
    GGML_METAL_SPARSE_PLAN_INVALID_RANGE,
    GGML_METAL_SPARSE_PLAN_INVALID_STATE,
};

struct ggml_metal_sparse_range {
    size_t offset;
    size_t size;
};

struct ggml_metal_sparse_quote {
    enum ggml_metal_sparse_plan_status status;
    uint64_t generation;
    size_t target_mappings;
    size_t new_pages;
    size_t cow_pages;
    size_t required_pages;
    size_t free_pages;
    size_t reserved_pages;
    bool feasible;
};

enum ggml_metal_sparse_ticket_state {
    GGML_METAL_SPARSE_TICKET_EMPTY = 0,
    GGML_METAL_SPARSE_TICKET_RESERVED,
    GGML_METAL_SPARSE_TICKET_COMMITTED,
    GGML_METAL_SPARSE_TICKET_ROLLED_BACK,
    GGML_METAL_SPARSE_TICKET_CANCELLED,
};

struct ggml_metal_sparse_ticket_accounting {
    uint64_t generation;
    size_t reserved_pages;
    enum ggml_metal_sparse_ticket_state state;
};

// The caller supplies zeroable scratch arrays sized n_virtual and n_physical.
// A physical page with R aliases and M selected writes costs M COW pages when
// M < R, but only M-1 when all aliases are written: one mapping can retain the
// original page. This is the same n-1 rule used by the commit path.
static inline struct ggml_metal_sparse_quote ggml_metal_sparse_plan_write(
        size_t page_size,
        size_t n_virtual,
        size_t n_physical,
        size_t n_free,
        size_t n_reserved,
        uint64_t generation,
        const uint32_t * v2p,
        const uint32_t * p_ref,
        const struct ggml_metal_sparse_range * ranges,
        size_t n_ranges,
        uint8_t * marked,
        uint32_t * marked_per_physical) {
    struct ggml_metal_sparse_quote result = {
        GGML_METAL_SPARSE_PLAN_OK,
        generation,
        0,
        0,
        0,
        0,
        n_free,
        n_reserved,
        false,
    };

    if (page_size == 0 || n_virtual > SIZE_MAX/page_size ||
            (n_ranges > 0 && ranges == NULL) ||
            (n_virtual > 0 && (v2p == NULL || marked == NULL)) ||
            (n_physical > 0 && (p_ref == NULL || marked_per_physical == NULL))) {
        result.status = GGML_METAL_SPARSE_PLAN_INVALID_STATE;
        return result;
    }

    memset(marked, 0, n_virtual*sizeof(*marked));
    memset(marked_per_physical, 0, n_physical*sizeof(*marked_per_physical));

    const size_t virtual_size = n_virtual*page_size;
    for (size_t r = 0; r < n_ranges; ++r) {
        const size_t offset = ranges[r].offset;
        const size_t size = ranges[r].size;
        if (size == 0) {
            continue;
        }
        if (offset > virtual_size || size > virtual_size - offset) {
            result.status = GGML_METAL_SPARSE_PLAN_INVALID_RANGE;
            return result;
        }

        const size_t v0 = offset/page_size;
        const size_t v1 = (offset + size - 1)/page_size;
        for (size_t v = v0; v <= v1; ++v) {
            marked[v] = 1;
        }
    }

    for (size_t v = 0; v < n_virtual; ++v) {
        if (!marked[v]) {
            continue;
        }
        ++result.target_mappings;
        const uint32_t p = v2p[v];
        if (p == UINT32_MAX) {
            ++result.new_pages;
            continue;
        }
        if (p >= n_physical || p_ref[p] == 0) {
            result.status = GGML_METAL_SPARSE_PLAN_INVALID_STATE;
            return result;
        }
        ++marked_per_physical[p];
    }

    for (size_t p = 0; p < n_physical; ++p) {
        const size_t selected = marked_per_physical[p];
        if (selected == 0) {
            continue;
        }
        if (selected > p_ref[p]) {
            result.status = GGML_METAL_SPARSE_PLAN_INVALID_STATE;
            return result;
        }
        result.cow_pages += selected < p_ref[p] ? selected : selected - 1;
    }

    result.required_pages = result.new_pages + result.cow_pages;
    result.feasible = result.status == GGML_METAL_SPARSE_PLAN_OK &&
            n_reserved <= n_free && result.required_pages <= n_free - n_reserved;
    return result;
}

static inline bool ggml_metal_sparse_accounting_try_reserve(
        size_t n_free,
        size_t * n_reserved,
        uint64_t generation,
        const struct ggml_metal_sparse_quote * quote,
        struct ggml_metal_sparse_ticket_accounting * ticket) {
    if (n_reserved == NULL || quote == NULL || ticket == NULL ||
            quote->status != GGML_METAL_SPARSE_PLAN_OK ||
            quote->generation != generation || *n_reserved > n_free ||
            quote->required_pages > n_free - *n_reserved) {
        return false;
    }

    *n_reserved += quote->required_pages;
    ticket->generation = generation;
    ticket->reserved_pages = quote->required_pages;
    ticket->state = GGML_METAL_SPARSE_TICKET_RESERVED;
    return true;
}

static inline bool ggml_metal_sparse_accounting_is_current(
        uint64_t generation,
        const struct ggml_metal_sparse_ticket_accounting * ticket) {
    return ticket != NULL && ticket->state == GGML_METAL_SPARSE_TICKET_RESERVED &&
            ticket->generation == generation;
}

static inline bool ggml_metal_sparse_accounting_finish(
        size_t * n_reserved,
        struct ggml_metal_sparse_ticket_accounting * ticket,
        enum ggml_metal_sparse_ticket_state final_state) {
    if (n_reserved == NULL || ticket == NULL ||
            (final_state != GGML_METAL_SPARSE_TICKET_COMMITTED &&
             final_state != GGML_METAL_SPARSE_TICKET_ROLLED_BACK &&
             final_state != GGML_METAL_SPARSE_TICKET_CANCELLED)) {
        return false;
    }
    if (ticket->state != GGML_METAL_SPARSE_TICKET_RESERVED) {
        return ticket->state == final_state ||
                (final_state == GGML_METAL_SPARSE_TICKET_CANCELLED &&
                 (ticket->state == GGML_METAL_SPARSE_TICKET_ROLLED_BACK ||
                  ticket->state == GGML_METAL_SPARSE_TICKET_CANCELLED));
    }
    if (*n_reserved < ticket->reserved_pages) {
        return false;
    }

    *n_reserved -= ticket->reserved_pages;
    ticket->reserved_pages = 0;
    ticket->state = final_state;
    return true;
}

#ifdef __cplusplus
}
#endif

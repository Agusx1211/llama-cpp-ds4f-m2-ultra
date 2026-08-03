#include <cstdio>

#if defined(GGML_TEST_METAL_SPARSE_RESERVATION)

#include "ggml-metal-device.h"

#include <cstdlib>

#undef assert
#define assert(expr) do {                                                        \
    if (!(expr)) {                                                               \
        std::fprintf(stderr, "check failed at %s:%d: %s\n",                    \
                __FILE__, __LINE__, #expr);                                      \
        std::abort();                                                            \
    }                                                                            \
} while (false)

static void assert_usage_equal(
        const ggml_metal_sparse_usage & lhs,
        const ggml_metal_sparse_usage & rhs) {
    assert(lhs.pool_id               == rhs.pool_id);
    assert(lhs.page_size             == rhs.page_size);
    assert(lhs.virtual_pages         == rhs.virtual_pages);
    assert(lhs.physical_pages        == rhs.physical_pages);
    assert(lhs.free_pages            == rhs.free_pages);
    assert(lhs.reserved_pages        == rhs.reserved_pages);
    assert(lhs.mapped_mappings       == rhs.mapped_mappings);
    assert(lhs.unique_physical_pages == rhs.unique_physical_pages);
    assert(lhs.shared_physical_pages == rhs.shared_physical_pages);
    assert(lhs.shared_mappings       == rhs.shared_mappings);
    assert(lhs.refcount_sum          == rhs.refcount_sum);
    assert(lhs.refcount_max          == rhs.refcount_max);
    assert(lhs.generation            == rhs.generation);
    assert(lhs.cow_allocations       == rhs.cow_allocations);
    assert(lhs.cow_pages             == rhs.cow_pages);
}

static ggml_metal_sparse_usage get_usage(ggml_metal_buffer_t buffer) {
    ggml_metal_sparse_usage result = {};
    assert(ggml_metal_buffer_sparse_get_usage(buffer, &result));
    return result;
}

static ggml_metal_sparse_pool_quote quote_one(
        ggml_metal_buffer_t buffer,
        size_t offset,
        size_t size) {
    const ggml_metal_sparse_buffer_range range = { buffer, offset, size };
    ggml_metal_sparse_pool_quote quote = {};
    size_t n_pools = 0;
    size_t limiting_pool = SIZE_MAX;
    const auto status = ggml_metal_buffers_sparse_quote(
            &range, 1, &quote, 1, &n_pools, &limiting_pool);
    assert(status == GGML_METAL_SPARSE_RESERVATION_OK);
    assert(n_pools == 1);
    assert(limiting_pool == 0);
    assert(quote.pool_id == (uintptr_t) buffer);
    return quote;
}

static ggml_metal_sparse_pool_quote reserve_one(
        ggml_metal_buffer_t buffer,
        size_t offset,
        size_t size,
        ggml_metal_sparse_reservation_t * reservation) {
    const ggml_metal_sparse_buffer_range range = { buffer, offset, size };
    ggml_metal_sparse_pool_quote quote = {};
    size_t n_pools = 0;
    size_t limiting_pool = SIZE_MAX;
    assert(ggml_metal_buffers_sparse_reserve(
            &range, 1, &quote, 1, &n_pools, &limiting_pool, reservation) ==
            GGML_METAL_SPARSE_RESERVATION_OK);
    assert(n_pools == 1 && limiting_pool == 0 && *reservation != nullptr);
    return quote;
}

static void assert_commit_delta(
        const ggml_metal_sparse_usage & before,
        const ggml_metal_sparse_usage & after,
        const ggml_metal_sparse_quote & quote) {
    assert(before.free_pages - after.free_pages == quote.required_pages);
    assert(after.unique_physical_pages - before.unique_physical_pages == quote.required_pages);
    assert(after.mapped_mappings - before.mapped_mappings == quote.new_pages);
    assert(after.cow_pages - before.cow_pages == quote.cow_pages);
    assert(after.cow_allocations - before.cow_allocations == (quote.cow_pages > 0));
    assert(after.reserved_pages == before.reserved_pages);
    assert(after.generation == before.generation + (quote.required_pages > 0));
}

int main() {
    constexpr size_t page = 64*1024;
    ggml_metal_device_t device = ggml_metal_device_get(0);
    const auto * props = ggml_metal_device_get_props(device);
    if (props == nullptr || !props->has_placement_sparse) {
        std::puts("metal sparse reservation test skipped: placement-sparse unavailable");
        return 0;
    }

    ggml_metal_sparse_init_result init = {};
    ggml_metal_buffer_t buffer = ggml_metal_buffer_init_sparse_ex(
            device, 8*page, 3*page, &init);
    if (buffer == nullptr && init.status == GGML_METAL_SPARSE_INIT_UNSUPPORTED) {
        std::puts("metal sparse reservation test skipped: placement-sparse unavailable");
        return 0;
    }
    assert(buffer != nullptr && init.status == GGML_METAL_SPARSE_INIT_OK);

    const auto initial = get_usage(buffer);
    assert(initial.page_size == page && initial.virtual_pages == 8);
    assert(initial.physical_pages == 3 && initial.free_pages == 3);
    assert(initial.reserved_pages == 0 && initial.mapped_mappings == 0);

    const auto new_quote = quote_one(buffer, 0, 1);
    assert(new_quote.write.new_pages == 1 && new_quote.write.cow_pages == 0);
    assert(new_quote.write.required_pages == 1);
    assert_usage_equal(new_quote.usage, initial);

    ggml_metal_sparse_reservation_t reservation = nullptr;
    const auto reserved_new = reserve_one(buffer, 0, 1, &reservation);
    assert(reserved_new.write.required_pages == new_quote.write.required_pages);
    assert_usage_equal(reserved_new.usage, initial);
    const auto during_new = get_usage(buffer);
    assert(during_new.reserved_pages == initial.reserved_pages + new_quote.write.required_pages);
    assert(during_new.free_pages == initial.free_pages);
    assert(ggml_metal_sparse_reservation_commit(reservation) ==
            GGML_METAL_SPARSE_RESERVATION_OK);
    ggml_metal_sparse_reservation_free(reservation);
    const auto after_new = get_usage(buffer);
    assert_commit_delta(initial, after_new, new_quote.write);

    const size_t alias_offset = page;
    const size_t alias_size = page;
    assert(ggml_metal_buffer_sparse_alias(
            buffer, 0, page, page, &alias_offset, &alias_size, 1));
    const auto aliased = get_usage(buffer);
    assert(aliased.free_pages == after_new.free_pages);
    assert(aliased.mapped_mappings == 2 && aliased.unique_physical_pages == 1);
    assert(aliased.shared_physical_pages == 1 && aliased.shared_mappings == 2);

    const auto cow_quote = quote_one(buffer, page, 1);
    assert(cow_quote.write.new_pages == 0 && cow_quote.write.cow_pages == 1);
    assert(cow_quote.write.required_pages == 1);
    reservation = nullptr;
    const auto reserved_cow = reserve_one(buffer, page, 1, &reservation);
    assert(reserved_cow.write.required_pages == cow_quote.write.required_pages);
    assert_usage_equal(reserved_cow.usage, aliased);
    assert(get_usage(buffer).reserved_pages == cow_quote.write.required_pages);
    assert(ggml_metal_sparse_reservation_commit(reservation) ==
            GGML_METAL_SPARSE_RESERVATION_OK);
    ggml_metal_sparse_reservation_free(reservation);
    const auto after_cow = get_usage(buffer);
    assert_commit_delta(aliased, after_cow, cow_quote.write);
    assert(after_cow.mapped_mappings == 2 && after_cow.unique_physical_pages == 2);
    assert(after_cow.shared_physical_pages == 0);

    reservation = nullptr;
    reserve_one(buffer, 2*page, 1, &reservation);
    const auto during_cancel = get_usage(buffer);
    assert(during_cancel.reserved_pages == 1);
    assert(ggml_metal_sparse_reservation_cancel(reservation));
    const auto after_cancel = get_usage(buffer);
    assert_usage_equal(after_cancel, after_cow);
    ggml_metal_sparse_reservation_free(reservation);

    reservation = nullptr;
    reserve_one(buffer, 3*page, 1, &reservation);
    assert(get_usage(buffer).reserved_pages == 1);
    assert(ggml_metal_sparse_reservation_rollback(reservation));
    const auto after_rollback = get_usage(buffer);
    assert_usage_equal(after_rollback, after_cow);
    ggml_metal_sparse_reservation_free(reservation);

    const ggml_metal_sparse_buffer_range pressure_ranges[] = {
        { buffer, 2*page, 1 },
        { buffer, 3*page, 1 },
    };
    ggml_metal_sparse_pool_quote pressure_quote = {};
    size_t n_pools = 0;
    size_t limiting_pool = SIZE_MAX;
    assert(ggml_metal_buffers_sparse_quote(
            pressure_ranges, 2, &pressure_quote, 1, &n_pools,
            &limiting_pool) == GGML_METAL_SPARSE_RESERVATION_PRESSURE);
    assert(n_pools == 1 && limiting_pool == 0);
    assert(pressure_quote.write.required_pages == 2 && !pressure_quote.write.feasible);
    assert_usage_equal(pressure_quote.usage, after_cow);
    assert_usage_equal(get_usage(buffer), after_cow);

    n_pools = 0;
    limiting_pool = SIZE_MAX;
    reservation = nullptr;
    assert(ggml_metal_buffers_sparse_reserve(
            pressure_ranges, 2, &pressure_quote, 1, &n_pools,
            &limiting_pool, &reservation) == GGML_METAL_SPARSE_RESERVATION_PRESSURE);
    assert(n_pools == 1 && limiting_pool == 0 && reservation == nullptr);
    assert(pressure_quote.write.required_pages == 2 && !pressure_quote.write.feasible);
    assert_usage_equal(pressure_quote.usage, after_cow);
    assert_usage_equal(get_usage(buffer), after_cow);

    ggml_metal_buffer_free(buffer);
    std::puts("metal sparse reservation device test passed");
    return 0;
}

#else

int main() {
    std::puts("metal sparse reservation test skipped: Metal backend not built");
    return 0;
}

#endif

#include "ggml-metal-device.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <vector>

#undef assert
#define assert(expr) do {                                                        \
    if (!(expr)) {                                                               \
        std::cerr << "check failed at " << __FILE__ << ':' << __LINE__          \
                  << ": " #expr << '\n';                                       \
        std::abort();                                                            \
    }                                                                            \
} while (false)

struct fixture {
    static constexpr size_t page = 64;

    std::vector<uint32_t> v2p;
    std::vector<uint32_t> refs;
    size_t free_pages;
    size_t reserved_pages = 0;
    uint64_t generation = 1;

    fixture(size_t n_virtual, size_t n_physical) :
        v2p(n_virtual, UINT32_MAX), refs(n_physical, 0), free_pages(n_physical) {}

    void alias(std::initializer_list<size_t> virtual_pages, uint32_t physical_page) {
        assert(physical_page < refs.size());
        for (size_t v : virtual_pages) {
            assert(v < v2p.size() && v2p[v] == UINT32_MAX);
            v2p[v] = physical_page;
            ++refs[physical_page];
        }
        assert(free_pages > 0);
        --free_pages;
    }

    ggml_metal_sparse_quote quote(
            const std::vector<ggml_metal_sparse_range> & ranges,
            std::vector<uint8_t> & marked,
            std::vector<uint32_t> & selected) const {
        marked.resize(v2p.size());
        selected.resize(refs.size());
        return ggml_metal_sparse_plan_write(
                page, v2p.size(), refs.size(), free_pages, reserved_pages,
                generation, v2p.data(), refs.data(), ranges.data(), ranges.size(),
                marked.data(), selected.data());
    }

    size_t commit(const ggml_metal_sparse_quote & quote, const std::vector<uint8_t> & marked) {
        std::vector<uint32_t> selected(refs.size(), 0);
        for (size_t v = 0; v < v2p.size(); ++v) {
            if (marked[v] && v2p[v] != UINT32_MAX) {
                ++selected[v2p[v]];
            }
        }
        std::vector<uint32_t> retained(refs.size());
        std::vector<uint32_t> copy_sources(v2p.size());
        assert(ggml_metal_sparse_select_cow_sources(
                v2p.size(), refs.size(), v2p.data(), refs.data(), marked.data(),
                selected.data(), retained.data(), copy_sources.data()) == GGML_METAL_SPARSE_PLAN_OK);

        size_t allocated = 0;
        for (size_t v = 0; v < v2p.size(); ++v) {
            if (!marked[v]) {
                continue;
            }
            const uint32_t old = v2p[v];
            if (old != UINT32_MAX) {
                assert(retained[old] != UINT32_MAX && v2p[retained[old]] == old);
                if (retained[old] == v) {
                    assert(copy_sources[v] == UINT32_MAX);
                    continue;
                }
                assert(copy_sources[v] == retained[old]);
            } else {
                assert(copy_sources[v] == UINT32_MAX);
            }
            uint32_t fresh = UINT32_MAX;
            for (uint32_t p = 0; p < refs.size(); ++p) {
                if (refs[p] == 0) {
                    fresh = p;
                    break;
                }
            }
            assert(fresh != UINT32_MAX && free_pages > 0);
            --free_pages;
            if (old != UINT32_MAX) {
                assert(refs[old] > 1);
                --refs[old];
            }
            v2p[v] = fresh;
            refs[fresh] = 1;
            ++allocated;
        }
        assert(allocated == quote.required_pages);
        if (allocated > 0) {
            ++generation;
        }
        return allocated;
    }
};

static bool reserve_all(
        const std::vector<std::pair<fixture *, ggml_metal_sparse_quote>> & pools,
        std::vector<ggml_metal_sparse_ticket_accounting> & tickets) {
    tickets.assign(pools.size(), {});
    for (const auto & [pool, quote] : pools) {
        if (!quote.feasible || quote.generation != pool->generation ||
                pool->reserved_pages > pool->free_pages ||
                quote.required_pages > pool->free_pages - pool->reserved_pages) {
            return false;
        }
    }
    for (size_t i = 0; i < pools.size(); ++i) {
        auto * pool = pools[i].first;
        const bool ok = ggml_metal_sparse_accounting_try_reserve(
                pool->free_pages, &pool->reserved_pages, pool->generation,
                &pools[i].second, &tickets[i]);
        assert(ok);
    }
    return true;
}

static void test_unmapped_overlap_and_rounding() {
    fixture f(8, 8);
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;

    auto q = f.quote({ { 0, 1 } }, marked, selected);
    assert(q.status == GGML_METAL_SPARSE_PLAN_OK && q.new_pages == 1 && q.cow_pages == 0);

    q = f.quote({ { 0, 64 }, { 32, 64 }, { 64, 64 } }, marked, selected);
    assert(q.target_mappings == 2 && q.required_pages == 2);

    q = f.quote({ { 63, 2 } }, marked, selected);
    assert(q.target_mappings == 2 && marked[0] && marked[1]);

    q = f.quote({ { 8*64, 1 } }, marked, selected);
    assert(q.status == GGML_METAL_SPARSE_PLAN_INVALID_RANGE && !q.feasible);
}

static void test_relative_alias_range_semantics() {
    constexpr size_t page = fixture::page;
    uint8_t marked[2] = {};

    const size_t first_offset = 0;
    const size_t one_page = page;
    assert(ggml_metal_sparse_mark_relative_ranges(
            page, page, &first_offset, &one_page, 1, marked, 1) ==
            GGML_METAL_SPARSE_PLAN_OK);
    assert(marked[0]);

    const size_t second_offset = page;
    assert(ggml_metal_sparse_mark_relative_ranges(
            page, 2*page, &second_offset, &one_page, 1, marked, 2) ==
            GGML_METAL_SPARSE_PLAN_OK);
    assert(!marked[0] && marked[1]);

    // The same offset is invalid for a one-page view. Passing an absolute
    // destination offset here was the original target-device test failure.
    assert(ggml_metal_sparse_mark_relative_ranges(
            page, page, &second_offset, &one_page, 1, marked, 1) ==
            GGML_METAL_SPARSE_PLAN_INVALID_RANGE);
}

static void test_diagnostic_status_names() {
    const auto named = [](const char * actual, const char * expected) {
        assert(std::strcmp(actual, expected) == 0);
    };
    named(ggml_metal_sparse_plan_status_name(GGML_METAL_SPARSE_PLAN_OK), "ok");
    named(ggml_metal_sparse_plan_status_name(
            GGML_METAL_SPARSE_PLAN_INVALID_RANGE), "invalid-range");
    named(ggml_metal_sparse_plan_status_name(
            GGML_METAL_SPARSE_PLAN_INVALID_STATE), "invalid-state");
    named(ggml_metal_sparse_plan_status_name(
            (ggml_metal_sparse_plan_status) 99), "unknown");

    named(ggml_metal_sparse_ticket_state_name(GGML_METAL_SPARSE_TICKET_EMPTY), "empty");
    named(ggml_metal_sparse_ticket_state_name(GGML_METAL_SPARSE_TICKET_RESERVED), "reserved");
    named(ggml_metal_sparse_ticket_state_name(GGML_METAL_SPARSE_TICKET_COMMITTED), "committed");
    named(ggml_metal_sparse_ticket_state_name(
            GGML_METAL_SPARSE_TICKET_ROLLED_BACK), "rolled-back");
    named(ggml_metal_sparse_ticket_state_name(GGML_METAL_SPARSE_TICKET_CANCELLED), "cancelled");
    named(ggml_metal_sparse_ticket_state_name(
            (ggml_metal_sparse_ticket_state) 99), "unknown");

    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_OK), "ok");
    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_PRESSURE), "pressure");
    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_STALE), "stale");
    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_INVALID), "invalid");
    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_OOM), "oom");
    named(ggml_metal_sparse_reservation_result_name(
            GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED), "unsupported");
    named(ggml_metal_sparse_reservation_result_name(
            (ggml_metal_sparse_reservation_result) 99), "unknown");
}

static void test_alias_cow_rules() {
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;

    fixture one(8, 8);
    one.alias({ 0, 1 }, 0);
    auto q = one.quote({ { 0, 1 } }, marked, selected);
    assert(q.new_pages == 0 && q.cow_pages == 1 && q.required_pages == 1);

    fixture many(8, 8);
    many.alias({ 0, 1, 2, 3 }, 0);
    q = many.quote({ { 0, 1 }, { 2*64, 1 } }, marked, selected);
    assert(q.cow_pages == 2);

    q = many.quote({ { 0, 4*64 } }, marked, selected);
    assert(q.target_mappings == 4 && q.cow_pages == 3 && q.required_pages == 3);
    const size_t before_free = many.free_pages;
    assert(many.commit(q, marked) == q.required_pages);
    assert(before_free - many.free_pages == q.required_pages);
}

static void check_cow_sources(
        std::initializer_list<size_t> aliases,
        std::initializer_list<size_t> writes,
        uint32_t expected_retained) {
    fixture f(8, 8);
    f.alias(aliases, 0);
    std::vector<ggml_metal_sparse_range> ranges;
    for (size_t v : writes) {
        ranges.push_back({ v*fixture::page, 1 });
    }
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;
    const auto q = f.quote(ranges, marked, selected);
    assert(q.status == GGML_METAL_SPARSE_PLAN_OK);
    const size_t expected_cow = writes.size() < aliases.size() ? writes.size() : writes.size() - 1;
    assert(q.cow_pages == expected_cow);

    std::vector<uint32_t> retained(f.refs.size());
    std::vector<uint32_t> copy_sources(f.v2p.size());
    assert(ggml_metal_sparse_select_cow_sources(
            f.v2p.size(), f.refs.size(), f.v2p.data(), f.refs.data(), marked.data(),
            selected.data(), retained.data(), copy_sources.data()) == GGML_METAL_SPARSE_PLAN_OK);
    assert(retained[0] == expected_retained);
    assert(f.v2p[expected_retained] == 0);
    assert(writes.size() < aliases.size() ? !marked[expected_retained] : marked[expected_retained]);
    for (size_t v : aliases) {
        if (marked[v] && v != expected_retained) {
            assert(copy_sources[v] == expected_retained);
        } else {
            assert(copy_sources[v] == UINT32_MAX);
        }
    }
}

static void test_stable_cow_sources() {
    // Partial writes retain the lowest unselected alias.
    check_cow_sources({ 0, 1 },       { 1 },       0);
    check_cow_sources({ 0, 1, 2 },    { 1, 2 },    0);
    check_cow_sources({ 0, 1, 2, 3 }, { 0, 2 },    1);

    // All-selected writes retain the lowest selected alias and every other
    // COW action reads it directly, never another new destination.
    check_cow_sources({ 0, 1 },       { 0, 1 },       0);
    check_cow_sources({ 0, 1, 2 },    { 0, 1, 2 },    0);
    check_cow_sources({ 0, 1, 2, 3 }, { 0, 1, 2, 3 }, 0);
}

static void test_insufficient_is_immutable() {
    fixture f(8, 1);
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;
    const auto before_v2p = f.v2p;
    const auto before_refs = f.refs;
    const size_t before_free = f.free_pages;
    const size_t before_reserved = f.reserved_pages;

    const auto q = f.quote({ { 0, 2*64 } }, marked, selected);
    assert(q.required_pages == 2 && !q.feasible);
    ggml_metal_sparse_ticket_accounting ticket = {};
    assert(!ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &q, &ticket));
    assert(f.v2p == before_v2p && f.refs == before_refs);
    assert(f.free_pages == before_free && f.reserved_pages == before_reserved);
}

static void test_stale_rollback_and_cancel() {
    fixture f(8, 8);
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;
    auto q = f.quote({ { 0, 64 } }, marked, selected);

    ggml_metal_sparse_ticket_accounting stale_quote = {};
    ++f.generation;
    assert(!ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &q, &stale_quote));
    assert(f.reserved_pages == 0);

    q = f.quote({ { 0, 64 } }, marked, selected);
    ggml_metal_sparse_ticket_accounting rollback = {};
    assert(ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &q, &rollback));
    assert(f.reserved_pages == 1);
    ++f.generation;
    assert(!ggml_metal_sparse_accounting_is_current(f.generation, &rollback));
    assert(ggml_metal_sparse_accounting_finish(
            &f.reserved_pages, &rollback, GGML_METAL_SPARSE_TICKET_ROLLED_BACK));
    assert(f.reserved_pages == 0);

    q = f.quote({ { 64, 64 } }, marked, selected);
    ggml_metal_sparse_ticket_accounting cancel = {};
    assert(ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &q, &cancel));
    assert(ggml_metal_sparse_accounting_finish(
            &f.reserved_pages, &cancel, GGML_METAL_SPARSE_TICKET_CANCELLED));
    assert(ggml_metal_sparse_accounting_finish(
            &f.reserved_pages, &cancel, GGML_METAL_SPARSE_TICKET_CANCELLED));
    assert(f.reserved_pages == 0);
}

static void test_commit_compatibility_including_zero_pages() {
    fixture f(8, 8);
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;
    const auto zero = f.quote({ { 0, 0 } }, marked, selected);
    assert(zero.status == GGML_METAL_SPARSE_PLAN_OK && zero.feasible);
    assert(zero.required_pages == 0 && zero.target_mappings == 0);

    ggml_metal_sparse_ticket_accounting ticket = {};
    assert(ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &zero, &ticket));
    const auto current = f.quote({ { 0, 0 } }, marked, selected);
    assert(ggml_metal_sparse_quote_commit_compatible(&zero, &current));

    auto changed = current;
    changed.target_mappings = 1;
    assert(!ggml_metal_sparse_quote_commit_compatible(&zero, &changed));
    changed = current;
    changed.feasible = false;
    assert(!ggml_metal_sparse_quote_commit_compatible(&zero, &changed));
    changed = zero;
    changed.status = GGML_METAL_SPARSE_PLAN_INVALID_STATE;
    assert(!ggml_metal_sparse_quote_commit_compatible(&changed, &current));
    assert(!ggml_metal_sparse_quote_commit_compatible(nullptr, &current));
    assert(ggml_metal_sparse_accounting_finish(
            &f.reserved_pages, &ticket, GGML_METAL_SPARSE_TICKET_COMMITTED));
}

static void test_multi_pool_failure_is_atomic() {
    fixture enough(8, 4);
    fixture short_pool(8, 1);
    std::vector<uint8_t> marked_a, marked_b;
    std::vector<uint32_t> selected_a, selected_b;
    const auto qa = enough.quote({ { 0, 2*64 } }, marked_a, selected_a);
    const auto qb = short_pool.quote({ { 0, 2*64 } }, marked_b, selected_b);
    assert(qa.feasible && !qb.feasible);

    // Aggregate reservation checks every dimension before incrementing any
    // per-pool reserved counter, matching the Metal multi-buffer transaction.
    const size_t before_a = enough.reserved_pages;
    const size_t before_b = short_pool.reserved_pages;
    std::vector<ggml_metal_sparse_ticket_accounting> tickets;
    assert(!reserve_all({ { &enough, qa }, { &short_pool, qb } }, tickets));
    assert(enough.reserved_pages == before_a && short_pool.reserved_pages == before_b);
}

static void test_quote_equals_commit() {
    fixture f(16, 12);
    f.alias({ 1, 5, 9 }, 0);
    std::vector<uint8_t> marked;
    std::vector<uint32_t> selected;
    const auto q = f.quote({ { 0, 2*64 }, { 5*64 + 1, 5*64 } }, marked, selected);
    assert(q.feasible && q.new_pages == 5 && q.cow_pages == 2);

    ggml_metal_sparse_ticket_accounting ticket = {};
    assert(ggml_metal_sparse_accounting_try_reserve(
            f.free_pages, &f.reserved_pages, f.generation, &q, &ticket));
    assert(ggml_metal_sparse_accounting_is_current(f.generation, &ticket));
    assert(f.commit(q, marked) == q.required_pages);
    assert(ggml_metal_sparse_accounting_finish(
            &f.reserved_pages, &ticket, GGML_METAL_SPARSE_TICKET_COMMITTED));
    assert(f.reserved_pages == 0);
}

int main() {
    test_diagnostic_status_names();
    test_unmapped_overlap_and_rounding();
    test_relative_alias_range_semantics();
    test_alias_cow_rules();
    test_stable_cow_sources();
    test_insufficient_is_immutable();
    test_stale_rollback_and_cancel();
    test_commit_compatibility_including_zero_pages();
    test_multi_pool_failure_is_atomic();
    test_quote_equals_commit();
    std::cout << "metal sparse planner tests passed\n";
    return 0;
}

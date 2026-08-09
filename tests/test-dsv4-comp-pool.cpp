#include "llama-dsv4-comp-pool.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace allocation_probe {
thread_local bool   enabled = false;
thread_local bool   reject  = false;
thread_local size_t calls   = 0;
}

[[gnu::noinline]] void * operator new(std::size_t size) {
    if (allocation_probe::enabled) {
        ++allocation_probe::calls;
        if (allocation_probe::reject) {
            throw std::bad_alloc();
        }
    }
    if (void * value = std::malloc(size == 0 ? 1 : size)) {
        return value;
    }
    throw std::bad_alloc();
}

[[gnu::noinline]] void * operator new[](std::size_t size) {
    return ::operator new(size);
}

[[gnu::noinline]] void operator delete(void * value) noexcept {
    std::free(value);
}

[[gnu::noinline]] void operator delete[](void * value) noexcept {
    ::operator delete(value);
}

[[gnu::noinline]] void operator delete(void * value, std::size_t) noexcept {
    std::free(value);
}

[[gnu::noinline]] void operator delete[](void * value, std::size_t) noexcept {
    ::operator delete(value);
}

namespace {

struct allocation_scope {
    explicit allocation_scope(bool reject = false) {
        expect_disabled();
        allocation_probe::calls   = 0;
        allocation_probe::reject  = reject;
        allocation_probe::enabled = true;
    }

    ~allocation_scope() {
        allocation_probe::enabled = false;
        allocation_probe::reject  = false;
    }

    size_t finish() {
        const size_t result        = allocation_probe::calls;
        allocation_probe::enabled = false;
        allocation_probe::reject  = false;
        return result;
    }

  private:
    static void expect_disabled() {
        if (allocation_probe::enabled) {
            std::abort();
        }
    }
};

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

void expect_status(llama_dsv4_comp_status actual, llama_dsv4_comp_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_dsv4_comp_status_name(expected) + ", got " +
             llama_dsv4_comp_status_name(actual));
    }
}

llama_dsv4_comp_handle_id create_handle(llama_dsv4_comp_pool & pool) {
    const auto result = pool.create_handle();
    expect_status(result.status, llama_dsv4_comp_status::ok, "create_handle");
    expect(result.handle != 0, "create_handle returned zero ID");
    return result.handle;
}

llama_dsv4_comp_handle_info handle_info(const llama_dsv4_comp_pool & pool, llama_dsv4_comp_handle_id handle) {
    llama_dsv4_comp_handle_info result;
    expect_status(pool.get_handle(handle, result), llama_dsv4_comp_status::ok, "get_handle");
    return result;
}

llama_dsv4_comp_quote quote_change(const llama_dsv4_comp_pool & pool,
                                   llama_dsv4_comp_handle_id    handle,
                                   llama_dsv4_comp_family       family,
                                   uint64_t                     rows,
                                   std::vector<uint64_t>        overwrites = {},
                                   std::vector<uint32_t>        graph_ids  = {}) {
    llama_dsv4_comp_batch_plan batch;
    batch.changes.push_back({ handle, family, rows, std::move(overwrites) });
    batch.graph_execution_ids = std::move(graph_ids);
    return pool.quote_batch(batch);
}

llama_dsv4_comp_quote commit_change(llama_dsv4_comp_pool &    pool,
                                    llama_dsv4_comp_handle_id handle,
                                    llama_dsv4_comp_family    family,
                                    uint64_t                  rows,
                                    std::vector<uint64_t>     overwrites = {}) {
    auto quote = quote_change(pool, handle, family, rows, std::move(overwrites));
    expect_status(quote.status, llama_dsv4_comp_status::ok, "quote change");
    const auto reservation = pool.try_reserve(quote);
    expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve change");
    expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit change");
    return quote;
}

void expect_family_equal(const llama_dsv4_comp_family_usage & lhs,
                         const llama_dsv4_comp_family_usage & rhs,
                         const std::string &                  message) {
#define EXPECT_FIELD(field) expect(lhs.field == rhs.field, message + ": " #field)
    EXPECT_FIELD(capacity_segments);
    EXPECT_FIELD(permanent_segments);
    EXPECT_FIELD(free_segments);
    EXPECT_FIELD(reserved_segments);
    EXPECT_FIELD(mapped_segments);
    EXPECT_FIELD(shared_segments);
    EXPECT_FIELD(cow_segments);
    EXPECT_FIELD(scratch_rows_in_use);
    EXPECT_FIELD(capacity_pages);
    EXPECT_FIELD(free_pages);
    EXPECT_FIELD(reserved_pages);
    EXPECT_FIELD(mapped_pages);
    EXPECT_FIELD(shared_pages);
    EXPECT_FIELD(cow_pages);
    EXPECT_FIELD(segment_pages_capacity);
    EXPECT_FIELD(segment_pages_free);
    EXPECT_FIELD(segment_pages_reserved);
    EXPECT_FIELD(segment_pages_mapped);
    EXPECT_FIELD(lid_pages_capacity);
    EXPECT_FIELD(lid_pages_free);
    EXPECT_FIELD(lid_pages_reserved);
    EXPECT_FIELD(lid_pages_mapped);
#undef EXPECT_FIELD
}

void expect_semantic_usage_equal(const llama_dsv4_comp_memory_usage & lhs,
                                 const llama_dsv4_comp_memory_usage & rhs,
                                 const std::string &                  message) {
    expect_family_equal(lhs.c4, rhs.c4, message + " C4");
    expect_family_equal(lhs.hca, rhs.hca, message + " HCA");
    expect(lhs.handles == rhs.handles, message + ": handles");
    expect(lhs.bindings == rhs.bindings, message + ": bindings");
    expect(lhs.resident_handles == rhs.resident_handles, message + ": resident handles");
    expect(lhs.active_tickets == rhs.active_tickets, message + ": active tickets");
}

void expect_page_partition(const llama_dsv4_comp_family_usage & usage, const std::string & family) {
    expect(usage.capacity_pages == usage.free_pages + usage.reserved_pages + usage.mapped_pages,
           family + " page partition mismatch");
    expect(usage.capacity_segments == usage.free_segments + usage.reserved_segments + usage.mapped_segments,
           family + " segment partition mismatch");
}

void test_geometry_and_permanent_ownership() {
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, 3) == 0, "C4 token 3 boundary");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, 4) == 1, "C4 token 4 boundary");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, 5) == 1, "C4 token 5 boundary");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, 127) == 0, "HCA token 127 boundary");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, 128) == 1, "HCA token 128 boundary");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, 129) == 1, "HCA token 129 boundary");
    expect(llama_dsv4_comp_segments_for_rows(63) == 1, "row 63 segment boundary");
    expect(llama_dsv4_comp_segments_for_rows(64) == 1, "row 64 segment boundary");
    expect(llama_dsv4_comp_segments_for_rows(65) == 2, "row 65 segment boundary");
    expect(llama_dsv4_comp_logical_segment(63) == 0, "logical row 63 segment");
    expect(llama_dsv4_comp_logical_segment(64) == 1, "logical row 64 segment");
    expect(llama_dsv4_comp_segment_row(63) == 63, "logical row 63 offset");
    expect(llama_dsv4_comp_segment_row(64) == 0, "logical row 64 offset");
    expect(llama_dsv4_comp_physical_row(0, 65) == 1, "zero physical-row formula");
    expect(llama_dsv4_comp_physical_row(1, 63) == 127, "scratch physical-row formula");
    expect(llama_dsv4_comp_physical_row(2, 65) == 129, "data physical-row formula");

    constexpr uint64_t max_tokens = 1048576;
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, max_tokens) == 262144, "1M C4 row geometry");
    expect(llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, max_tokens) == 8192, "1M HCA row geometry");
    expect(llama_dsv4_comp_segments_for_rows(262144) == 4096, "1M C4 segment geometry");
    expect(llama_dsv4_comp_segments_for_rows(8192) == 128, "1M HCA segment geometry");

    llama_dsv4_comp_pool pool({ 8, 4 });
    const auto           usage = pool.memory_usage_snapshot();
    expect(usage.c4.capacity_segments == 10 && usage.c4.permanent_segments == 2, "C4 permanent capacity");
    expect(usage.c4.free_segments == 8 && usage.c4.mapped_segments == 2, "C4 baseline segments");
    expect(usage.c4.segment_pages_capacity == 210, "C4 CSA page capacity");
    expect(usage.c4.segment_pages_mapped == 42, "C4 permanent CSA pages");
    expect(usage.c4.lid_pages_capacity == 63, "C4 LID page capacity");
    expect(usage.c4.lid_pages_mapped == 21, "C4 permanent LID group");
    expect(usage.c4.capacity_pages == 273 && usage.c4.free_pages == 210 && usage.c4.mapped_pages == 63,
           "C4 total page baseline");
    expect(usage.hca.capacity_segments == 6 && usage.hca.free_segments == 4 && usage.hca.mapped_segments == 2,
           "HCA baseline segments");
    expect(usage.hca.capacity_pages == 120 && usage.hca.free_pages == 80 && usage.hca.mapped_pages == 40,
           "HCA page baseline");
    expect(pool.zero_segment(llama_dsv4_comp_family::c4) == 0, "C4 zero segment");
    expect(pool.scratch_segment(llama_dsv4_comp_family::hca) == 1, "HCA scratch segment");
    expect(pool.zero_physical_row(llama_dsv4_comp_family::c4, 65) == 1, "zero row preserves segment offset");
    expect(pool.scratch_physical_row(llama_dsv4_comp_family::c4, 63) == 127, "per-graph scratch row");
    expect_page_partition(usage.c4, "C4 baseline");
    expect_page_partition(usage.hca, "HCA baseline");
}

void test_alias_shapes_and_cow_divergence() {
    {
        llama_dsv4_comp_pool pool({ 4, 2 });
        const auto           source   = create_handle(pool);
        const auto           baseline = pool.memory_usage_snapshot();
        const auto           copy     = pool.copy_handle(source);
        expect_status(copy.status, llama_dsv4_comp_status::ok, "copy empty handle");
        expect(pool.memory_usage_snapshot().c4.shared_segments == 0, "empty alias shared a data segment");
        expect_status(pool.remove_handle(copy.handle), llama_dsv4_comp_status::ok, "remove empty alias");
        expect_family_equal(pool.memory_usage_snapshot().c4, baseline.c4, "empty alias C4 baseline");
    }

    {
        llama_dsv4_comp_pool pool({ 4, 2 });
        const auto           handle = create_handle(pool);
        commit_change(pool, handle, llama_dsv4_comp_family::c4, 10);

        auto quote = quote_change(pool, handle, llama_dsv4_comp_family::c4, 11);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote unique partial append");
        expect(quote.c4.new_segments == 0 && quote.c4.cow_segments == 0 && quote.c4.total_pages() == 0,
               "unique partial append did not reuse its tail");
        auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve unique partial append");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit unique partial append");

        const auto before_overwrite = handle_info(pool, handle);
        quote                       = quote_change(pool, handle, llama_dsv4_comp_family::c4, 11, { 3 });
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote unique visible overwrite");
        expect(quote.c4.new_segments == 1 && quote.c4.cow_segments == 1,
               "unique visible overwrite did not require replacement");
        expect(quote.allocations.size() == 1 &&
                   quote.allocations[0].source_segment == before_overwrite.c4_segment_ids[0] &&
                   quote.allocations[0].populated_rows == 11,
               "unique overwrite replacement quote");
        reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve unique visible overwrite");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit unique visible overwrite");
        expect(handle_info(pool, handle).c4_segment_ids[0] != before_overwrite.c4_segment_ids[0],
               "unique overwrite retained its visible segment");
        expect(pool.memory_usage_snapshot().c4.cow_segments == 1, "unique overwrite COW accounting");
    }

    {
        llama_dsv4_comp_pool pool({ 8, 2 });
        const auto           source = create_handle(pool);
        const auto           first  = commit_change(pool, source, llama_dsv4_comp_family::c4, 10);
        expect(first.c4.new_segments == 1 && first.c4.cow_segments == 0, "partial initial allocation quote");
        expect(first.c4.segment_pages == 21 && first.c4.lid_pages == 0, "partial initial page quote");
        const auto alias = pool.copy_handle(source);
        expect_status(alias.status, llama_dsv4_comp_status::ok, "copy partial handle");
        expect(pool.memory_usage_snapshot().c4.shared_segments == 1, "partial alias refcount");
        expect(pool.memory_usage_snapshot().c4.shared_pages == 42, "partial alias shared-page accounting");

        const auto before_source = handle_info(pool, source);
        const auto before_alias  = handle_info(pool, alias.handle);
        auto       quote         = quote_change(pool, alias.handle, llama_dsv4_comp_family::c4, 11);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote partial append COW");
        expect(quote.c4.new_segments == 1 && quote.c4.cow_segments == 1, "partial append COW count");
        expect(quote.c4.segment_pages == 21 && quote.c4.lid_pages == 0, "partial append COW page delta");
        expect(quote.allocations.size() == 1 && quote.allocations[0].cow, "partial append COW vector");
        expect(quote.allocations[0].source_segment == before_alias.c4_segment_ids[0], "partial append source");
        expect(quote.allocations[0].populated_rows == 10, "partial append populated rows");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve partial append COW");
        llama_dsv4_comp_handle_info candidate;
        expect_status(pool.candidate_handle(reservation.ticket, alias.handle, candidate), llama_dsv4_comp_status::ok,
                      "partial candidate root");
        expect(candidate.visible_c4_rows == 11, "candidate partial length");
        expect(candidate.c4_segment_ids[0] != before_alias.c4_segment_ids[0], "candidate partial root did not diverge");
        expect(handle_info(pool, alias.handle).visible_c4_rows == 10, "candidate root leaked before commit");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit partial append COW");
        expect(handle_info(pool, source).c4_segment_ids == before_source.c4_segment_ids,
               "source changed after alias COW");
        expect(handle_info(pool, alias.handle).c4_segment_ids == candidate.c4_segment_ids,
               "candidate root not published");
        expect(pool.memory_usage_snapshot().c4.cow_pages == 42, "partial alias COW page accounting");
    }

    {
        llama_dsv4_comp_pool pool({ 8, 2 });
        const auto           source = create_handle(pool);
        commit_change(pool, source, llama_dsv4_comp_family::c4, 64);
        const auto alias = pool.copy_handle(source);
        auto       quote = quote_change(pool, alias.handle, llama_dsv4_comp_family::c4, 65);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote full-tail append");
        expect(quote.c4.new_segments == 1 && quote.c4.cow_segments == 0, "full segment append copied old segment");
        expect(quote.allocations[0].source_segment == LLAMA_DSV4_COMP_INVALID_SEGMENT,
               "full segment append has COW source");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve full-tail append");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit full-tail append");
        const auto source_info = handle_info(pool, source);
        const auto alias_info  = handle_info(pool, alias.handle);
        expect(source_info.c4_segment_ids[0] == alias_info.c4_segment_ids[0], "full prefix lost alias");
        expect(alias_info.c4_segment_ids.size() == 2, "full-tail append segment count");
    }

    {
        llama_dsv4_comp_pool pool({ 10, 2 });
        const auto           source = create_handle(pool);
        commit_change(pool, source, llama_dsv4_comp_family::c4, 130);
        const auto alias  = pool.copy_handle(source);
        const auto before = handle_info(pool, alias.handle);
        auto       quote  = quote_change(pool, alias.handle, llama_dsv4_comp_family::c4, 130, { 70 });
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote multi-segment overwrite");
        expect(quote.c4.new_segments == 1 && quote.c4.cow_segments == 1, "overwrite exact COW count");
        expect(quote.allocations[0].logical_segment == 1 && quote.allocations[0].populated_rows == 64,
               "overwrite COW vector");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve multi-segment overwrite");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit multi-segment overwrite");
        const auto after = handle_info(pool, alias.handle);
        expect(after.c4_segment_ids[0] == before.c4_segment_ids[0], "overwrite changed preceding segment");
        expect(after.c4_segment_ids[1] != before.c4_segment_ids[1], "overwrite did not replace target segment");
        expect(after.c4_segment_ids[2] == before.c4_segment_ids[2], "overwrite changed following segment");
        expect(handle_info(pool, source).c4_segment_ids == before.c4_segment_ids, "multi-segment source changed");
    }
}

void test_lid_group_boundary_and_hca_independence() {
    llama_dsv4_comp_pool pool({ 8, 4 });
    const auto           handle = create_handle(pool);
    for (uint64_t segment_count = 1; segment_count <= 7; ++segment_count) {
        const auto before = pool.memory_usage_snapshot();
        const auto quote =
            quote_change(pool, handle, llama_dsv4_comp_family::c4, segment_count * LLAMA_DSV4_COMP_SEGMENT_ROWS);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote C4 LID boundary");
        const uint64_t expected_lid_pages = segment_count == 3 || segment_count == 7 ? 21 : 0;
        expect(quote.c4.segment_pages == 21, "C4 append CSA page delta");
        expect(quote.c4.lid_pages == expected_lid_pages, "LID four-segment group delta");
        expect(quote.allocations.size() == 1 && quote.allocations[0].destination_segment == segment_count + 1,
               "sequential physical segment assignment");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve C4 LID boundary");
        const auto reserved = pool.memory_usage_snapshot();
        expect(reserved.c4.reserved_pages == quote.c4.total_pages(), "reserved C4 page quote mismatch");
        expect_page_partition(reserved.c4, "reserved C4 LID boundary");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit C4 LID boundary");
        const auto committed = pool.memory_usage_snapshot();
        expect(committed.c4.mapped_pages - before.c4.mapped_pages == quote.c4.total_pages(),
               "committed C4 page quote mismatch");
    }
    auto usage = pool.memory_usage_snapshot();
    expect(usage.c4.lid_pages_mapped == 63, "LID mapped page groups after physical segment eight");
    expect_page_partition(usage.c4, "C4 after LID boundary");

    commit_change(pool, handle, llama_dsv4_comp_family::hca, 10);
    const auto alias     = pool.copy_handle(handle);
    const auto c4_before = handle_info(pool, alias.handle).c4_segment_ids;
    auto       quote     = quote_change(pool, alias.handle, llama_dsv4_comp_family::hca, 11);
    expect_status(quote.status, llama_dsv4_comp_status::ok, "quote HCA tail COW");
    expect(quote.hca.new_segments == 1 && quote.hca.cow_segments == 1, "HCA COW count");
    expect(quote.hca.segment_pages == 20 && quote.hca.lid_pages == 0, "HCA page quote");
    expect(quote.c4.new_segments == 0 && quote.c4.total_pages() == 0, "HCA change touched C4 quote");
    const auto reservation = pool.try_reserve(quote);
    expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve HCA COW");
    expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit HCA COW");
    expect(handle_info(pool, alias.handle).c4_segment_ids == c4_before, "HCA COW changed C4 directory");
    expect(pool.memory_usage_snapshot().c4.shared_segments == 7, "HCA COW changed C4 refcounts");
    expect(pool.memory_usage_snapshot().hca.cow_pages == 20, "HCA COW page accounting");
}

void test_atomic_candidate_directories_and_noncontiguous_bindings() {
    llama_dsv4_comp_pool pool({ 16, 8 });
    const auto           short_handle    = create_handle(pool);
    const auto           boundary_handle = create_handle(pool);
    const auto           long_handle     = create_handle(pool);
    commit_change(pool, short_handle, llama_dsv4_comp_family::c4, 1);
    commit_change(pool, boundary_handle, llama_dsv4_comp_family::c4, 64);
    commit_change(pool, long_handle, llama_dsv4_comp_family::c4, 130);

    expect_status(pool.bind(63, short_handle), llama_dsv4_comp_status::ok, "bind slot 63");
    expect_status(pool.bind(2, boundary_handle), llama_dsv4_comp_status::ok, "bind slot 2");
    expect_status(pool.bind(41, long_handle), llama_dsv4_comp_status::ok, "bind slot 41");
    const std::vector<uint32_t> graph_order = { 41, 63, 2 };
    const auto                  current     = pool.directory_for_bindings(llama_dsv4_comp_family::c4, graph_order, 4);
    expect_status(current.status, llama_dsv4_comp_status::ok, "noncontiguous directory");
    expect(current.graph_streams == 3 && current.logical_segments == 4, "directory shape");
    const auto short_info    = handle_info(pool, short_handle);
    const auto boundary_info = handle_info(pool, boundary_handle);
    const auto long_info     = handle_info(pool, long_handle);
    expect(std::equal(long_info.c4_segment_ids.begin(), long_info.c4_segment_ids.end(), current.segment_ids.begin()),
           "graph column 0 did not follow execution 41");
    expect(current.segment_ids[3] == 0, "long directory zero padding");
    expect(current.segment_ids[4] == short_info.c4_segment_ids[0] && current.segment_ids[5] == 0,
           "graph column 1 did not follow execution 63");
    expect(current.segment_ids[8] == boundary_info.c4_segment_ids[0] && current.segment_ids[9] == 0,
           "graph column 2 did not follow execution 2");

    llama_dsv4_comp_batch_plan batch;
    batch.graph_execution_ids = graph_order;
    batch.changes             = {
        { boundary_handle, llama_dsv4_comp_family::c4,  65, {} },
        { boundary_handle, llama_dsv4_comp_family::hca, 1,  {} },
    };
    auto quote = pool.quote_batch(batch);
    expect_status(quote.status, llama_dsv4_comp_status::ok, "quote atomic multi-family update");
    expect(quote.scratch_rows == 3, "scratch quote did not follow graph width");
    const auto reservation = pool.try_reserve(quote);
    expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve atomic multi-family update");
    auto usage = pool.memory_usage_snapshot();
    expect(usage.active_tickets == 1 && usage.c4.scratch_rows_in_use == 3 && usage.hca.scratch_rows_in_use == 3,
           "ticket scratch ownership");
    llama_dsv4_comp_handle_info candidate;
    expect_status(pool.candidate_handle(reservation.ticket, boundary_handle, candidate), llama_dsv4_comp_status::ok,
                  "multi-family candidate");
    expect(candidate.visible_c4_rows == 65 && candidate.visible_hca_rows == 1, "multi-family candidate lengths");
    expect(handle_info(pool, boundary_handle).visible_c4_rows == 64, "candidate length published before commit");
    const auto ticket_directory = pool.ticket_directory(reservation.ticket, llama_dsv4_comp_family::c4, 2);
    expect_status(ticket_directory.status, llama_dsv4_comp_status::ok, "ticket candidate directory");
    expect(ticket_directory.segment_ids[5] == candidate.c4_segment_ids[1], "ticket column lacks candidate root");
    expect(current.segment_ids[9] == 0, "ordinary directory changed during reservation");

    expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok, "atomic rollback");
    usage = pool.memory_usage_snapshot();
    expect(usage.active_tickets == 0 && usage.c4.reserved_segments == 0 && usage.hca.reserved_segments == 0,
           "atomic rollback leaked reservation");
    expect(usage.c4.scratch_rows_in_use == 0 && usage.hca.scratch_rows_in_use == 0,
           "atomic rollback leaked scratch rows");
    expect(handle_info(pool, boundary_handle).visible_hca_rows == 0, "atomic rollback published HCA root");

    quote                = pool.quote_batch(batch);
    const auto committed = pool.try_reserve(quote);
    expect_status(committed.status, llama_dsv4_comp_status::ok, "reserve atomic commit");
    expect_status(pool.commit(committed.ticket), llama_dsv4_comp_status::ok, "atomic multi-family commit");
    const auto committed_info = handle_info(pool, boundary_handle);
    expect(committed_info.visible_c4_rows == 65 && committed_info.visible_hca_rows == 1,
           "atomic commit did not publish both families");

    expect_status(pool.unbind(63), llama_dsv4_comp_status::ok, "unbind resident handle");
    expect_status(pool.bind(7, short_handle), llama_dsv4_comp_status::ok, "rebind resident handle");
    llama_dsv4_comp_handle_id rebound = 0;
    expect_status(pool.get_binding(7, rebound), llama_dsv4_comp_status::ok, "lookup rebound resident handle");
    expect(rebound == short_handle, "resident handle identity depended on execution slot");
}

void test_exhaustion_immutability_and_ticket_idempotence() {
    {
        llama_dsv4_comp_pool pool({ 1, 1 });
        const auto           source = create_handle(pool);
        commit_change(pool, source, llama_dsv4_comp_family::c4, 1);
        const auto alias         = pool.copy_handle(source);
        const auto before_usage  = pool.memory_usage_snapshot();
        const auto before_source = handle_info(pool, source);
        const auto before_alias  = handle_info(pool, alias.handle);
        const auto quote         = quote_change(pool, alias.handle, llama_dsv4_comp_family::c4, 2);
        expect_status(quote.status, llama_dsv4_comp_status::capacity_exhausted, "shared tail exhaustion quote");
        expect(quote.limiting_family == llama_dsv4_comp_family::c4, "wrong limiting pool");
        expect_status(pool.try_reserve(quote).status, llama_dsv4_comp_status::capacity_exhausted,
                      "shared tail exhaustion reserve");
        const auto after_usage = pool.memory_usage_snapshot();
        expect_family_equal(after_usage.c4, before_usage.c4, "exhaustion C4 immutability");
        expect(after_usage.epoch == before_usage.epoch, "exhaustion changed pool epoch");
        expect(handle_info(pool, source).c4_segment_ids == before_source.c4_segment_ids,
               "exhaustion changed source root");
        expect(handle_info(pool, alias.handle).c4_segment_ids == before_alias.c4_segment_ids,
               "exhaustion changed alias root");
    }

    {
        llama_dsv4_comp_pool       pool({ 2, 0 });
        const auto                 handle = create_handle(pool);
        const auto                 before = pool.memory_usage_snapshot();
        llama_dsv4_comp_batch_plan batch;
        batch.changes = {
            { handle, llama_dsv4_comp_family::c4,  1, {} },
            { handle, llama_dsv4_comp_family::hca, 1, {} },
        };
        const auto quote = pool.quote_batch(batch);
        expect_status(quote.status, llama_dsv4_comp_status::capacity_exhausted, "multi-family exhaustion quote");
        expect(quote.limiting_family == llama_dsv4_comp_family::hca, "multi-family limiting pool");
        expect_status(pool.try_reserve(quote).status, llama_dsv4_comp_status::capacity_exhausted,
                      "multi-family exhaustion reserve");
        const auto after = pool.memory_usage_snapshot();
        expect_family_equal(after.c4, before.c4, "multi-family exhaustion C4 immutability");
        expect_family_equal(after.hca, before.hca, "multi-family exhaustion HCA immutability");
        expect(after.epoch == before.epoch, "multi-family exhaustion changed epoch");
    }

    {
        llama_dsv4_comp_pool pool({ 2, 2 });
        const auto           handle = create_handle(pool);
        expect_status(pool.bind(9, handle), llama_dsv4_comp_status::ok, "bind ticket handle");
        auto stale_quote = quote_change(pool, handle, llama_dsv4_comp_family::c4, 1, {}, { 9 });
        create_handle(pool);
        expect_status(pool.try_reserve(stale_quote).status, llama_dsv4_comp_status::stale_quote,
                      "ABA quote generation");

        auto quote       = quote_change(pool, handle, llama_dsv4_comp_family::c4, 1, {}, { 9 });
        auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve rollback ticket");
        const auto reserved_usage = pool.memory_usage_snapshot();
        expect(reserved_usage.c4.reserved_segments == 1 && reserved_usage.c4.scratch_rows_in_use == 1,
               "reservation accounting");
        auto forged = reservation.ticket;
        ++forged.generation;
        expect_status(pool.commit(forged), llama_dsv4_comp_status::stale_ticket, "stale ticket generation");
        expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok, "first rollback");
        expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok, "idempotent rollback");
        expect_status(pool.cancel(reservation.ticket), llama_dsv4_comp_status::ok, "rollback/cancel idempotence");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::stale_ticket,
                      "commit accepted rolled-back ticket");
        llama_dsv4_comp_handle_info terminal_candidate;
        expect_status(pool.candidate_handle(reservation.ticket, handle, terminal_candidate),
                      llama_dsv4_comp_status::stale_ticket, "terminal candidate remained visible");
        auto usage = pool.memory_usage_snapshot();
        expect(usage.c4.free_segments == 2 && usage.c4.reserved_segments == 0 && usage.c4.scratch_rows_in_use == 0,
               "rollback baseline");
        expect(handle_info(pool, handle).visible_c4_rows == 0, "rollback published candidate");

        quote       = quote_change(pool, handle, llama_dsv4_comp_family::c4, 1, {}, { 9 });
        reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve cancel ticket");
        expect_status(pool.cancel(reservation.ticket), llama_dsv4_comp_status::ok, "first cancel");
        expect_status(pool.cancel(reservation.ticket), llama_dsv4_comp_status::ok, "idempotent cancel");
        expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok, "cancel/rollback idempotence");
    }
}

void test_copy_remove_cycles_return_accounting_baseline() {
    llama_dsv4_comp_pool       pool({ 10, 6 });
    const auto                 source = create_handle(pool);
    llama_dsv4_comp_batch_plan batch;
    batch.changes = {
        { source, llama_dsv4_comp_family::c4,  130, {} },
        { source, llama_dsv4_comp_family::hca, 65,  {} },
    };
    auto quote = pool.quote_batch(batch);
    expect_status(quote.status, llama_dsv4_comp_status::ok, "quote source baseline");
    auto reservation = pool.try_reserve(quote);
    expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve source baseline");
    expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "commit source baseline");
    const auto baseline = pool.memory_usage_snapshot();

    for (int cycle = 0; cycle < 8; ++cycle) {
        const auto copy = pool.copy_handle(source);
        expect_status(copy.status, llama_dsv4_comp_status::ok, "copy cycle");
        const auto shared = pool.memory_usage_snapshot();
        expect(shared.c4.shared_segments == 3 && shared.hca.shared_segments == 2, "copy cycle refcounts");
        expect_status(pool.remove_handle(copy.handle), llama_dsv4_comp_status::ok, "remove copy cycle");
        const auto after = pool.memory_usage_snapshot();
        expect_family_equal(after.c4, baseline.c4, "copy/remove C4 baseline");
        expect_family_equal(after.hca, baseline.hca, "copy/remove HCA baseline");
    }
    expect_status(pool.remove_handle(source), llama_dsv4_comp_status::ok, "remove baseline source");
    const auto empty = pool.memory_usage_snapshot();
    expect(empty.c4.free_segments == 10 && empty.hca.free_segments == 6, "remove source did not free all data");
    expect(empty.c4.mapped_segments == 2 && empty.hca.mapped_segments == 2, "permanent ownership was released");
}

void test_resident_detach_attach_release_contract() {
    llama_dsv4_comp_pool pool({ 8, 4 });
    const auto           source = create_handle(pool);
    commit_change(pool, source, llama_dsv4_comp_family::c4, 65);
    commit_change(pool, source, llama_dsv4_comp_family::hca, 1);
    expect_status(pool.bind(3, source), llama_dsv4_comp_status::ok, "bind resident source");

    const auto quoted_usage = pool.memory_usage_snapshot();
    auto       quote        = pool.quote_detach(3);
    expect_status(quote.status, llama_dsv4_comp_status::ok, "quote resident detach");
    expect(quote.execution_id == 3 && quote.resident.pool_id != 0 && quote.resident.id == source &&
               quote.resident.handle_generation != 0 && quote.resident.lease_generation != 0,
           "resident detach quote identity");
    expect(pool.memory_usage_snapshot().epoch == quoted_usage.epoch, "resident quote mutated pool epoch");

    const auto metadata_reservation = pool.try_reserve(pool.quote_batch({}));
    expect_status(metadata_reservation.status, llama_dsv4_comp_status::ok, "reserve metadata ticket");
    const auto busy_usage = pool.memory_usage_snapshot();
    expect_status(pool.quote_detach(3).status, llama_dsv4_comp_status::busy, "detach quote ignored active ticket");
    expect(pool.memory_usage_snapshot().epoch == busy_usage.epoch, "busy detach quote mutated pool");
    expect_status(pool.cancel(metadata_reservation.ticket), llama_dsv4_comp_status::ok, "cancel metadata ticket");

    expect_status(pool.bind(5, source), llama_dsv4_comp_status::ok, "legacy second binding was rejected");
    const auto multiply_bound = pool.quote_detach(3);
    expect_status(multiply_bound.status, llama_dsv4_comp_status::handle_bound, "detach accepted multiply-bound handle");
    expect(pool.memory_usage_snapshot().bindings == 2, "failed multi-binding quote mutated bindings");
    expect_status(pool.unbind(5), llama_dsv4_comp_status::ok, "remove legacy second binding");

    const auto unrelated = create_handle(pool);
    expect_status(pool.detach(quote).status, llama_dsv4_comp_status::stale_quote, "stale resident detach quote");
    llama_dsv4_comp_handle_id still_bound = 0;
    expect_status(pool.get_binding(3, still_bound), llama_dsv4_comp_status::ok, "stale detach removed source binding");
    expect(still_bound == source && pool.memory_usage_snapshot().resident_handles == 0,
           "stale detach changed ownership");
    expect_status(pool.remove_handle(unrelated), llama_dsv4_comp_status::ok, "remove unrelated handle");

    quote                            = pool.quote_detach(3);
    const auto quoted_resident       = quote.resident;
    quote.resident.pool_id           = UINT64_MAX;
    quote.resident.id                = UINT64_MAX;
    quote.resident.handle_generation = UINT64_MAX;
    quote.resident.lease_generation  = UINT64_MAX;
    const auto detached              = pool.detach(quote);
    expect_status(detached.status, llama_dsv4_comp_status::ok, "detach resident source");
    expect(detached.resident == quoted_resident,
           "detach trusted mutable public quote fields instead of its validated plan");
    expect_status(pool.get_binding(3, still_bound), llama_dsv4_comp_status::binding_not_found,
                  "detached source remained execution-bound");
    auto usage = pool.memory_usage_snapshot();
    expect(usage.bindings == 0 && usage.resident_handles == 1, "detached resident accounting");
    expect_family_equal(usage.c4, quoted_usage.c4, "detach changed C4 ownership");
    expect_family_equal(usage.hca, quoted_usage.hca, "detach changed HCA ownership");

    llama_dsv4_comp_pool other_pool({ 1, 1 });
    const auto           other_handle = create_handle(other_pool);
    expect_status(other_pool.bind(0, other_handle), llama_dsv4_comp_status::ok, "bind other pool handle");
    const auto other_detached = other_pool.detach(other_pool.quote_detach(0));
    expect_status(other_detached.status, llama_dsv4_comp_status::ok, "detach other pool handle");
    expect_status(other_pool.attach(detached.resident, 0), llama_dsv4_comp_status::stale_handle,
                  "cross-pool resident lease attached matching local IDs");
    expect_status(other_pool.release(detached.resident), llama_dsv4_comp_status::stale_handle,
                  "cross-pool resident lease released matching local IDs");
    expect_status(other_pool.release(other_detached.resident), llama_dsv4_comp_status::ok,
                  "release other pool resident");

    expect_status(pool.bind(4, source), llama_dsv4_comp_status::handle_resident,
                  "legacy bind consumed resident handle");
    expect_status(pool.copy_handle(source).status, llama_dsv4_comp_status::handle_resident,
                  "legacy copy aliased resident handle");
    expect_status(pool.remove_handle(source), llama_dsv4_comp_status::handle_resident,
                  "legacy remove released resident handle");
    expect_status(quote_change(pool, source, llama_dsv4_comp_family::c4, 66).status,
                  llama_dsv4_comp_status::handle_resident, "batch mutation changed resident handle");

    const auto occupant = create_handle(pool);
    expect_status(pool.bind(4, occupant), llama_dsv4_comp_status::ok, "bind occupied attach target");
    const auto before_failed_attach = pool.memory_usage_snapshot();
    expect_status(pool.attach(detached.resident, 4), llama_dsv4_comp_status::slot_occupied,
                  "attach accepted occupied execution");
    usage = pool.memory_usage_snapshot();
    expect(usage.epoch == before_failed_attach.epoch && usage.resident_handles == 1 && usage.bindings == 1,
           "failed attach mutated ownership");
    expect_family_equal(usage.c4, before_failed_attach.c4, "failed attach changed C4 refs");
    expect_family_equal(usage.hca, before_failed_attach.hca, "failed attach changed HCA refs");

    expect_status(pool.attach(detached.resident, 3), llama_dsv4_comp_status::ok, "rollback resident detach");
    expect_status(pool.get_binding(3, still_bound), llama_dsv4_comp_status::ok,
                  "rollback did not restore source binding");
    expect(still_bound == source && pool.memory_usage_snapshot().resident_handles == 0,
           "rollback did not restore exact handle");
    expect_status(pool.release(detached.resident), llama_dsv4_comp_status::stale_handle,
                  "consumed resident lease released attached handle");

    const auto detached_again = pool.detach(pool.quote_detach(3));
    expect_status(detached_again.status, llama_dsv4_comp_status::ok, "second resident detach");
    expect(detached_again.resident.lease_generation > detached.resident.lease_generation,
           "resident generation did not advance across reuse");
    expect_status(pool.attach(detached.resident, 3), llama_dsv4_comp_status::stale_handle,
                  "old resident generation reattached reused handle");
    expect_status(pool.release(detached.resident), llama_dsv4_comp_status::stale_handle,
                  "old resident generation released reused handle");
    expect_status(pool.attach(detached_again.resident, 7), llama_dsv4_comp_status::ok,
                  "attach resident handle to a different execution");
    expect_status(pool.get_binding(7, still_bound), llama_dsv4_comp_status::ok,
                  "different execution did not receive resident handle");
    expect(still_bound == source, "resident handle identity depended on parked execution ID");
    expect_status(pool.release(detached_again.resident), llama_dsv4_comp_status::stale_handle,
                  "attached lease remained externally releasable");
    const auto detached_from_new_execution = pool.detach(pool.quote_detach(7));
    expect_status(detached_from_new_execution.status, llama_dsv4_comp_status::ok, "detach from replacement execution");
    expect(detached_from_new_execution.resident.lease_generation > detached_again.resident.lease_generation,
           "replacement execution detach reused resident generation");
    expect_status(pool.release(detached_from_new_execution.resident), llama_dsv4_comp_status::ok,
                  "release exact resident generation");
    expect_status(pool.release(detached_from_new_execution.resident), llama_dsv4_comp_status::stale_handle,
                  "resident generation released twice");

    usage = pool.memory_usage_snapshot();
    expect(usage.handles == 1 && usage.bindings == 1 && usage.resident_handles == 0,
           "resident release ownership baseline");
    expect(usage.c4.free_segments == 8 && usage.hca.free_segments == 4,
           "resident release leaked compressed segment refs");
    expect_status(pool.unbind(4), llama_dsv4_comp_status::ok, "unbind attach occupant");
    expect_status(pool.remove_handle(occupant), llama_dsv4_comp_status::ok, "remove attach occupant");
}

void test_prepared_resident_attach_transaction() {
    llama_dsv4_comp_pool pool({ 8, 4 });
    const auto           source = create_handle(pool);
    commit_change(pool, source, llama_dsv4_comp_family::c4, 65);
    commit_change(pool, source, llama_dsv4_comp_family::hca, 1);
    expect_status(pool.bind(3, source), llama_dsv4_comp_status::ok, "bind prepared source");
    const auto detached = pool.detach(pool.quote_detach(3));
    expect_status(detached.status, llama_dsv4_comp_status::ok, "detach prepared source");
    const auto resident_baseline = pool.memory_usage_snapshot();

    auto cancelled = pool.quote_attach(detached.resident, 7);
    expect_status(cancelled.status, llama_dsv4_comp_status::ok, "quote cancellable attach");
    expect(pool.memory_usage_snapshot().epoch == resident_baseline.epoch, "attach quote mutated epoch");
    expect_status(pool.rollback_attach(cancelled), llama_dsv4_comp_status::ok, "cancel prepared attach");
    expect_status(pool.rollback_attach(cancelled), llama_dsv4_comp_status::ok, "idempotent prepared cancel");
    expect_status(pool.commit_attach(cancelled), llama_dsv4_comp_status::stale_quote,
                  "commit replayed cancelled attach");

    llama_dsv4_comp_attach_quote allocation_failure;
    size_t                       failed_quote_allocations = 0;
    {
        allocation_scope allocations(true);
        allocation_failure       = pool.quote_attach(detached.resident, 7);
        failed_quote_allocations = allocations.finish();
    }
    expect(failed_quote_allocations == 1, "prepared attach allocation failure was not deterministic");
    expect_status(allocation_failure.status, llama_dsv4_comp_status::resource_exhausted,
                  "prepared attach allocation failure status");
    const auto after_allocation_failure = pool.memory_usage_snapshot();
    expect(after_allocation_failure.epoch == resident_baseline.epoch &&
               after_allocation_failure.bindings == resident_baseline.bindings &&
               after_allocation_failure.resident_handles == resident_baseline.resident_handles,
           "failed prepared allocation changed ownership");
    expect_family_equal(after_allocation_failure.c4, resident_baseline.c4,
                        "failed prepared allocation changed C4 accounting");
    expect_family_equal(after_allocation_failure.hca, resident_baseline.hca,
                        "failed prepared allocation changed HCA accounting");

    auto stale = pool.quote_attach(detached.resident, 7);
    expect_status(stale.status, llama_dsv4_comp_status::ok, "quote stale attach");
    const auto unrelated = create_handle(pool);
    expect_status(pool.commit_attach(stale), llama_dsv4_comp_status::stale_quote,
                  "prepared attach ignored intervening mutation");
    expect_status(pool.rollback_attach(stale), llama_dsv4_comp_status::ok,
                  "stale prepared attach did not cancel");
    expect_status(pool.remove_handle(unrelated), llama_dsv4_comp_status::ok, "remove stale mutation handle");

    const auto occupant = create_handle(pool);
    expect_status(pool.bind(7, occupant), llama_dsv4_comp_status::ok, "bind prepared occupant");
    expect_status(pool.quote_attach(detached.resident, 7).status, llama_dsv4_comp_status::slot_occupied,
                  "prepared quote accepted occupied target");
    expect_status(pool.unbind(7), llama_dsv4_comp_status::ok, "unbind prepared occupant");
    expect_status(pool.remove_handle(occupant), llama_dsv4_comp_status::ok, "remove prepared occupant");

    const auto ticket = pool.try_reserve(pool.quote_batch({}));
    expect_status(ticket.status, llama_dsv4_comp_status::ok, "reserve prepared busy ticket");
    expect_status(pool.quote_attach(detached.resident, 7).status, llama_dsv4_comp_status::busy,
                  "prepared quote ignored active ticket");
    expect_status(pool.cancel(ticket.ticket), llama_dsv4_comp_status::ok, "cancel prepared busy ticket");

    llama_dsv4_comp_pool foreign({ 1, 1 });
    auto                  prepared = pool.quote_attach(detached.resident, 7);
    expect_status(prepared.status, llama_dsv4_comp_status::ok, "quote prepared attach");
    expect_status(foreign.commit_attach(prepared), llama_dsv4_comp_status::stale_quote,
                  "foreign pool committed prepared attach");
    expect_status(foreign.rollback_attach(prepared), llama_dsv4_comp_status::stale_quote,
                  "foreign pool cancelled prepared attach");

    prepared.execution_id             = 0;
    prepared.pool_epoch               = UINT64_MAX;
    prepared.resident.pool_id         = UINT64_MAX;
    prepared.resident.lease_generation = UINT64_MAX;
    llama_dsv4_comp_status commit_status;
    size_t                 commit_allocations = 0;
    {
        allocation_scope allocations;
        commit_status      = pool.commit_attach(prepared);
        commit_allocations = allocations.finish();
    }
    expect_status(commit_status, llama_dsv4_comp_status::ok, "commit prepared attach");
    expect(commit_allocations == 0, "prepared attach commit allocated");
    llama_dsv4_comp_handle_id rebound = 0;
    expect_status(pool.get_binding(7, rebound), llama_dsv4_comp_status::ok, "prepared binding missing");
    expect(rebound == source, "prepared attach trusted mutable quote fields");
    expect_status(pool.commit_attach(prepared), llama_dsv4_comp_status::stale_quote,
                  "prepared attach commit replay succeeded");

    llama_dsv4_comp_status rollback_status;
    size_t                 rollback_allocations = 0;
    {
        allocation_scope allocations;
        rollback_status      = pool.rollback_attach(prepared);
        rollback_allocations = allocations.finish();
    }
    expect_status(rollback_status, llama_dsv4_comp_status::ok, "rollback committed attach");
    expect(rollback_allocations == 0, "committed attach rollback allocated");
    expect_status(pool.get_binding(7, rebound), llama_dsv4_comp_status::binding_not_found,
                  "attach rollback retained binding");
    auto usage = pool.memory_usage_snapshot();
    expect(usage.bindings == resident_baseline.bindings &&
               usage.resident_handles == resident_baseline.resident_handles,
           "attach rollback did not restore resident ownership");
    expect_family_equal(usage.c4, resident_baseline.c4, "attach rollback changed C4 accounting");
    expect_family_equal(usage.hca, resident_baseline.hca, "attach rollback changed HCA accounting");
    expect_status(pool.rollback_attach(prepared), llama_dsv4_comp_status::ok,
                  "committed attach rollback was not idempotent");
    expect_status(pool.commit_attach(prepared), llama_dsv4_comp_status::stale_quote,
                  "rolled-back attach recommitted");

    auto retry = pool.quote_attach(detached.resident, 5);
    expect_status(retry.status, llama_dsv4_comp_status::ok, "quote exact attach retry");
    expect_status(pool.commit_attach(retry), llama_dsv4_comp_status::ok, "commit exact attach retry");
    const auto intervening = create_handle(pool);
    expect_status(pool.rollback_attach(retry), llama_dsv4_comp_status::stale_quote,
                  "post-commit rollback ignored intervening mutation");
    expect_status(pool.remove_handle(intervening), llama_dsv4_comp_status::ok, "remove rollback mutation handle");

    const auto detached_again = pool.detach(pool.quote_detach(5));
    expect_status(detached_again.status, llama_dsv4_comp_status::ok, "detach prepared retry");
    expect(detached_again.resident.lease_generation > detached.resident.lease_generation,
           "prepared retry did not advance lease generation");
    expect_status(pool.quote_attach(detached.resident, 5).status, llama_dsv4_comp_status::stale_handle,
                  "stale lease generation prepared attach");
    expect_status(pool.release(detached_again.resident), llama_dsv4_comp_status::ok,
                  "release prepared retry lease");
}

void test_prepared_resident_detach_rollback() {
    llama_dsv4_comp_pool pool({ 16, 16 });
    const auto created = pool.create_handle();
    expect_status(created.status, llama_dsv4_comp_status::ok, "create detach rollback handle");
    expect_status(pool.bind(3, created.handle), llama_dsv4_comp_status::ok, "bind detach rollback handle");

    auto cancelled = pool.quote_detach(3);
    expect_status(cancelled.status, llama_dsv4_comp_status::ok, "quote cancellable detach");
    expect_status(pool.rollback_detach(cancelled), llama_dsv4_comp_status::ok, "cancel prepared detach");
    expect_status(pool.rollback_detach(cancelled), llama_dsv4_comp_status::ok, "repeat prepared detach cancel");
    expect_status(pool.detach(cancelled).status, llama_dsv4_comp_status::stale_quote,
                  "cancelled detach quote committed");

    auto prepared = pool.quote_detach(3);
    expect_status(prepared.status, llama_dsv4_comp_status::ok, "quote rollback detach");
    const auto resident = pool.detach(prepared);
    expect_status(resident.status, llama_dsv4_comp_status::ok, "commit rollback detach");
    llama_dsv4_comp_handle_id rebound = 0;
    expect_status(pool.get_binding(3, rebound), llama_dsv4_comp_status::binding_not_found,
                  "detached execution retained binding");
    prepared.execution_id              = 0;
    prepared.pool_epoch                = UINT64_MAX;
    prepared.resident.pool_id          = UINT64_MAX;
    prepared.resident.lease_generation = UINT64_MAX;

    size_t rollback_allocations = 0;
    {
        allocation_scope allocations;
        const auto status = pool.rollback_detach(prepared);
        rollback_allocations = allocations.finish();
        expect_status(status, llama_dsv4_comp_status::ok, "rollback committed detach");
    }
    expect(rollback_allocations == 0, "committed detach rollback allocated");
    expect_status(pool.rollback_detach(prepared), llama_dsv4_comp_status::ok,
                  "committed detach rollback was not idempotent");
    expect_status(pool.get_binding(3, rebound), llama_dsv4_comp_status::ok,
                  "detach rollback did not restore binding");
    expect(rebound == created.handle, "detach rollback restored a different handle");
    expect_status(pool.release(resident.resident), llama_dsv4_comp_status::stale_handle,
                  "rolled-back resident lease remained valid");

    const auto retried = pool.detach(pool.quote_detach(3));
    expect_status(retried.status, llama_dsv4_comp_status::ok, "retry detach after rollback");
    expect(retried.resident.lease_generation > resident.resident.lease_generation,
           "detach rollback allowed resident generation ABA");
    expect_status(pool.release(retried.resident), llama_dsv4_comp_status::ok,
                  "release retried resident handle");

    const auto other_created = pool.create_handle();
    expect_status(other_created.status, llama_dsv4_comp_status::ok, "create stale rollback handle");
    expect_status(pool.bind(4, other_created.handle), llama_dsv4_comp_status::ok, "bind stale rollback handle");
    auto stale = pool.quote_detach(4);
    expect_status(pool.detach(stale).status, llama_dsv4_comp_status::ok, "commit stale rollback detach");
    const auto mutation = pool.create_handle();
    expect_status(mutation.status, llama_dsv4_comp_status::ok, "mutate pool after detach");
    expect_status(pool.rollback_detach(stale), llama_dsv4_comp_status::stale_quote,
                  "detach rollback ignored intervening mutation");
    expect_status(pool.release(stale.resident), llama_dsv4_comp_status::ok,
                  "release stale rollback resident");
    expect_status(pool.remove_handle(mutation.handle), llama_dsv4_comp_status::ok,
                  "remove stale rollback mutation handle");
}

void test_bounded_ticket_history_and_pool_identity() {
    {
        llama_dsv4_comp_pool   pool({ 0, 0 });
        llama_dsv4_comp_ticket oldest;
        for (uint32_t transaction = 0; transaction < 4096; ++transaction) {
            const auto quote = pool.quote_batch({});
            expect_status(quote.status, llama_dsv4_comp_status::ok, "quote metadata stress ticket");
            const auto reservation = pool.try_reserve(quote);
            expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve metadata stress ticket");
            if (transaction == 0) {
                oldest = reservation.ticket;
            }

            switch (transaction % 3) {
                case 0:
                    expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "commit metadata stress ticket");
                    expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "idempotent metadata stress commit");
                    break;
                case 1:
                    expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "rollback metadata stress ticket");
                    expect_status(pool.cancel(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "idempotent metadata stress rollback");
                    break;
                case 2:
                    expect_status(pool.cancel(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "cancel metadata stress ticket");
                    expect_status(pool.rollback(reservation.ticket), llama_dsv4_comp_status::ok,
                                  "idempotent metadata stress cancel");
                    break;
            }

            const auto usage = pool.memory_usage_snapshot();
            expect(usage.active_tickets == 0, "metadata stress left an active ticket");
            expect(usage.retained_ticket_records <= LLAMA_DSV4_COMP_TICKET_TOMBSTONES,
                   "terminal ticket metadata exceeded its bound");
        }
        const auto usage = pool.memory_usage_snapshot();
        expect(usage.retained_ticket_records == LLAMA_DSV4_COMP_TICKET_TOMBSTONES,
               "terminal ticket tombstone ring did not reach its fixed bound");
        expect_status(pool.commit(oldest), llama_dsv4_comp_status::stale_ticket,
                      "pruned terminal ticket remained addressable");
    }

    llama_dsv4_comp_quote dead_pool_quote;
    {
        llama_dsv4_comp_pool pool({ 1, 0 });
        const auto           handle = create_handle(pool);
        dead_pool_quote             = quote_change(pool, handle, llama_dsv4_comp_family::c4, 1);
        expect_status(dead_pool_quote.status, llama_dsv4_comp_status::ok, "quote from short-lived pool");
    }
    for (uint32_t reuse = 0; reuse < 1024; ++reuse) {
        llama_dsv4_comp_pool pool({ 1, 0 });
        create_handle(pool);
        expect(pool.memory_usage_snapshot().epoch == dead_pool_quote.pool_epoch,
               "foreign-quote regression did not exercise equal epochs");
        expect_status(pool.try_reserve(dead_pool_quote).status, llama_dsv4_comp_status::stale_quote,
                      "destroyed-pool quote admitted by a new pool");
    }
}

void test_recreated_pool_rejects_old_handles() {
    // llama_kv_cache_dsv4::clear(true) models a clear/recreate boundary by
    // replacing its aggregate pool.  Exercise the same host-side lifecycle
    // without retaining a raw pointer into the destroyed object.
    auto old_pool = std::make_unique<llama_dsv4_comp_pool>(llama_dsv4_comp_pool_config { 8, 4 });
    const auto initial_source      = create_handle(*old_pool);
    const auto initial_survivor    = create_handle(*old_pool);
    const auto initial_destination = create_handle(*old_pool);
    expect_status(old_pool->bind(0, initial_source), llama_dsv4_comp_status::ok,
                  "bind recreated-pool source");
    expect_status(old_pool->bind(1, initial_survivor), llama_dsv4_comp_status::ok,
                  "bind recreated-pool survivor");
    expect_status(old_pool->bind(2, initial_destination), llama_dsv4_comp_status::ok,
                  "bind recreated-pool destination");

    const auto detached = old_pool->detach(old_pool->quote_detach(0));
    expect_status(detached.status, llama_dsv4_comp_status::ok, "detach before pool recreation");
    const auto old_resident = detached.resident;
    expect_status(old_pool->release(old_resident), llama_dsv4_comp_status::ok,
                  "release before pool recreation");

    // IDs 4 and 5 were created only in the old pool.  They deliberately do
    // not collide with the three constructor bindings in the replacement.
    expect_status(old_pool->unbind(2), llama_dsv4_comp_status::ok,
                  "unbind old destination before pool recreation");
    expect_status(old_pool->remove_handle(initial_destination), llama_dsv4_comp_status::ok,
                  "remove old destination before pool recreation");
    const auto old_replacement = create_handle(*old_pool);
    const auto old_destination = create_handle(*old_pool);
    expect(old_replacement > initial_destination && old_destination > old_replacement,
           "old replacement/destination IDs did not advance past constructor IDs");

    old_pool.reset();
    auto fresh_pool = std::make_unique<llama_dsv4_comp_pool>(llama_dsv4_comp_pool_config { 8, 4 });
    const auto fresh_baseline = fresh_pool->memory_usage_snapshot();
    expect(fresh_baseline.c4.capacity_segments == 10 && fresh_baseline.hca.capacity_segments == 6 &&
               fresh_baseline.c4.mapped_segments == 2 && fresh_baseline.hca.mapped_segments == 2 &&
               fresh_baseline.handles == 0 && fresh_baseline.bindings == 0 &&
               fresh_baseline.resident_handles == 0 && fresh_baseline.active_tickets == 0 &&
               fresh_baseline.retained_ticket_records == 0,
           "recreated pool did not return to constructor-empty accounting");
    expect_status(fresh_pool->attach(old_resident, 0), llama_dsv4_comp_status::stale_handle,
                  "old resident attached to recreated pool");
    expect_status(fresh_pool->release(old_resident), llama_dsv4_comp_status::stale_handle,
                  "old resident released by recreated pool");

    const auto fresh0 = create_handle(*fresh_pool);
    const auto fresh1 = create_handle(*fresh_pool);
    const auto fresh2 = create_handle(*fresh_pool);
    expect_status(fresh_pool->bind(0, fresh0), llama_dsv4_comp_status::ok,
                  "bind recreated constructor source");
    expect_status(fresh_pool->bind(1, fresh1), llama_dsv4_comp_status::ok,
                  "bind recreated constructor survivor");
    expect_status(fresh_pool->bind(2, fresh2), llama_dsv4_comp_status::ok,
                  "bind recreated constructor destination");
    const auto fresh_usage = fresh_pool->memory_usage_snapshot();
    expect(fresh_usage.c4.capacity_segments == fresh_baseline.c4.capacity_segments &&
               fresh_usage.hca.capacity_segments == fresh_baseline.hca.capacity_segments &&
               fresh_usage.c4.mapped_segments == fresh_baseline.c4.mapped_segments &&
               fresh_usage.hca.mapped_segments == fresh_baseline.hca.mapped_segments &&
               fresh_usage.handles == 3 && fresh_usage.bindings == 3 && fresh_usage.resident_handles == 0 &&
               fresh_usage.active_tickets == 0 && fresh_usage.retained_ticket_records == 0,
           "recreated constructor bindings changed pool accounting");

    llama_dsv4_comp_handle_info removed;
    expect_status(fresh_pool->get_handle(old_replacement, removed), llama_dsv4_comp_status::handle_not_found,
                  "old replacement handle survived pool recreation");
    expect_status(fresh_pool->get_handle(old_destination, removed), llama_dsv4_comp_status::handle_not_found,
                  "old destination handle survived pool recreation");
}

void test_sixty_four_handle_rounded_capacity_contract() {
    constexpr uint64_t c4_total_rows  = 262144;
    constexpr uint64_t hca_total_rows = 8192;

    const auto create_bound_handles = [](llama_dsv4_comp_pool & pool) {
        std::vector<llama_dsv4_comp_handle_id> handles;
        handles.reserve(LLAMA_DSV4_COMP_GRAPH_STREAMS);
        for (uint32_t execution_id = 0; execution_id < LLAMA_DSV4_COMP_GRAPH_STREAMS; ++execution_id) {
            const auto handle = create_handle(pool);
            expect_status(pool.bind(execution_id, handle), llama_dsv4_comp_status::ok, "bind rounded handle");
            handles.push_back(handle);
        }
        return handles;
    };
    const auto make_batch = [](const std::vector<llama_dsv4_comp_handle_id> & handles, bool include_c4,
                               bool include_hca) {
        llama_dsv4_comp_batch_plan batch;
        for (uint32_t execution_id = 0; execution_id < LLAMA_DSV4_COMP_GRAPH_STREAMS; ++execution_id) {
            batch.graph_execution_ids.push_back(execution_id);
            const bool tail = execution_id + 1 == LLAMA_DSV4_COMP_GRAPH_STREAMS;
            if (include_c4) {
                batch.changes.push_back(
                    { handles[execution_id], llama_dsv4_comp_family::c4, tail ? c4_total_rows - 63 : 1, {} });
            }
            if (include_hca) {
                batch.changes.push_back(
                    { handles[execution_id], llama_dsv4_comp_family::hca, tail ? hca_total_rows - 63 : 1, {} });
            }
        }
        return batch;
    };

    {
        llama_dsv4_comp_pool pool({ 4096, 128 });
        const auto           handles = create_bound_handles(pool);
        const auto           before  = pool.memory_usage_snapshot();

        const auto c4_quote = pool.quote_batch(make_batch(handles, true, false));
        expect_status(c4_quote.status, llama_dsv4_comp_status::capacity_exhausted,
                      "4096 C4 segments admitted rounded 64-handle aggregate");
        expect(c4_quote.limiting_family == llama_dsv4_comp_family::c4, "rounded C4 limiting family");
        expect_status(pool.try_reserve(c4_quote).status, llama_dsv4_comp_status::capacity_exhausted,
                      "rounded C4 reserve pressure");

        const auto hca_quote = pool.quote_batch(make_batch(handles, false, true));
        expect_status(hca_quote.status, llama_dsv4_comp_status::capacity_exhausted,
                      "128 HCA segments admitted rounded 64-handle aggregate");
        expect(hca_quote.limiting_family == llama_dsv4_comp_family::hca, "rounded HCA limiting family");
        expect_status(pool.try_reserve(hca_quote).status, llama_dsv4_comp_status::capacity_exhausted,
                      "rounded HCA reserve pressure");

        const auto after = pool.memory_usage_snapshot();
        expect_family_equal(after.c4, before.c4, "rounded rejection C4 immutability");
        expect_family_equal(after.hca, before.hca, "rounded rejection HCA immutability");
        expect(after.epoch == before.epoch, "rounded rejection changed pool epoch");
    }

    {
        llama_dsv4_comp_pool pool({ 4159, 191 });
        const auto           handles = create_bound_handles(pool);
        const auto           quote   = pool.quote_batch(make_batch(handles, true, true));
        expect_status(quote.status, llama_dsv4_comp_status::ok, "rounded 64-handle aggregate quote");
        expect(quote.scratch_rows == 64, "rounded aggregate scratch ownership");
        expect(quote.c4.new_segments == 4159 && quote.hca.new_segments == 191, "rounded aggregate exact segment quote");
        expect(quote.c4.lid_pages == 1040ULL * LLAMA_DSV4_COMP_LID_LAYERS, "rounded aggregate absolute LID page quote");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "rounded aggregate reserve");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "rounded aggregate commit");

        uint64_t observed_c4_rows  = 0;
        uint64_t observed_hca_rows = 0;
        for (llama_dsv4_comp_handle_id handle : handles) {
            const auto info = handle_info(pool, handle);
            observed_c4_rows += info.visible_c4_rows;
            observed_hca_rows += info.visible_hca_rows;
        }
        expect(observed_c4_rows == c4_total_rows && observed_hca_rows == hca_total_rows,
               "rounded aggregate exceeded logical 1M row totals");

        const auto usage = pool.memory_usage_snapshot();
        expect(usage.c4.free_segments == 0 && usage.hca.free_segments == 0,
               "rounded aggregate did not consume exact segment capacities");
        expect(usage.c4.lid_pages_capacity == 1041ULL * LLAMA_DSV4_COMP_LID_LAYERS &&
                   usage.c4.lid_pages_mapped == usage.c4.lid_pages_capacity,
               "rounded aggregate LID capacity/mapping");
        expect(usage.c4.free_pages == 0 && usage.hca.free_pages == 0,
               "rounded aggregate page capacity did not close exactly");
    }
}

void test_deterministic_quotes_and_full_context_boundary() {
    {
        llama_dsv4_comp_pool       pool({ 6, 2 });
        const auto                 first  = create_handle(pool);
        const auto                 second = create_handle(pool);
        llama_dsv4_comp_batch_plan forward;
        forward.changes = {
            { first,  llama_dsv4_comp_family::c4, 65, {} },
            { second, llama_dsv4_comp_family::c4, 1,  {} },
        };
        auto reverse = forward;
        std::reverse(reverse.changes.begin(), reverse.changes.end());
        const auto lhs = pool.quote_batch(forward);
        const auto rhs = pool.quote_batch(reverse);
        expect_status(lhs.status, llama_dsv4_comp_status::ok, "forward deterministic quote");
        expect_status(rhs.status, llama_dsv4_comp_status::ok, "reverse deterministic quote");
        expect(lhs.allocations.size() == rhs.allocations.size(), "deterministic allocation count");
        for (size_t i = 0; i < lhs.allocations.size(); ++i) {
            const auto & a = lhs.allocations[i];
            const auto & b = rhs.allocations[i];
            expect(a.handle == b.handle && a.family == b.family && a.logical_segment == b.logical_segment &&
                       a.source_segment == b.source_segment && a.destination_segment == b.destination_segment &&
                       a.populated_rows == b.populated_rows && a.cow == b.cow,
                   "quote depended on caller change order");
        }
    }

    {
        constexpr uint64_t         max_tokens = 1048576;
        const uint64_t             c4_rows    = llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, max_tokens);
        const uint64_t             hca_rows = llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, max_tokens);
        llama_dsv4_comp_pool       pool({ 4096, 128 });
        const auto                 handle = create_handle(pool);
        llama_dsv4_comp_batch_plan batch;
        batch.changes = {
            { handle, llama_dsv4_comp_family::c4,  c4_rows,  {} },
            { handle, llama_dsv4_comp_family::hca, hca_rows, {} },
        };
        const auto quote = pool.quote_batch(batch);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "full-context quote");
        expect(quote.c4.new_segments == 4096 && quote.c4.segment_pages == 4096ULL * 21, "full-context C4 quote");
        expect(quote.c4.lid_pages == 1024ULL * 21, "full-context LID quote");
        expect(quote.hca.new_segments == 128 && quote.hca.segment_pages == 128ULL * 20, "full-context HCA quote");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "full-context reserve");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::ok, "full-context commit");
        const auto info = handle_info(pool, handle);
        expect(info.c4_segment_ids.size() == 4096 && info.hca_segment_ids.size() == 128,
               "full-context directory lengths");
        expect(info.c4_segment_ids.back() == 4097 && info.hca_segment_ids.back() == 129,
               "full-context final legal segments");
        const auto before = pool.memory_usage_snapshot();
        expect(before.c4.lid_pages_capacity == 1025ULL * LLAMA_DSV4_COMP_LID_LAYERS &&
                   before.c4.lid_pages_mapped == before.c4.lid_pages_capacity,
               "full-context absolute LID page capacity");
        expect(before.c4.free_pages == 0 && before.hca.free_pages == 0, "full-context exact page capacity");
        const auto exhausted = quote_change(pool, handle, llama_dsv4_comp_family::c4, c4_rows + 1);
        expect_status(exhausted.status, llama_dsv4_comp_status::capacity_exhausted, "row beyond full-context capacity");
        const auto after = pool.memory_usage_snapshot();
        expect_family_equal(after.c4, before.c4, "full-context exhaustion immutability");
        expect(after.epoch == before.epoch, "full-context exhaustion changed epoch");
    }
}

void test_detached_import_abort_preserves_semantics_and_monotonicity() {
    {
        llama_dsv4_comp_pool pool({ 4, 2 });
        const auto baseline = pool.memory_usage_snapshot();
        const auto detached = create_handle(pool);
        const auto created  = pool.memory_usage_snapshot();
        expect(created.epoch > baseline.epoch, "detached handle creation did not advance epoch");
        expect_status(pool.abort_empty_detached_handle(detached, baseline.epoch),
                      llama_dsv4_comp_status::ok, "abort empty detached handle");
        const auto aborted = pool.memory_usage_snapshot();
        expect_semantic_usage_equal(aborted, baseline, "empty detached abort baseline");
        expect(aborted.epoch > created.epoch, "empty detached abort rewound epoch");
        llama_dsv4_comp_handle_info stale;
        expect_status(pool.get_handle(detached, stale), llama_dsv4_comp_status::handle_not_found,
                      "aborted empty handle remained live");
        const auto replacement = create_handle(pool);
        expect(replacement > detached, "aborted empty handle ID was reused");
        expect_status(pool.remove_handle(replacement), llama_dsv4_comp_status::ok,
                      "remove empty abort replacement");
    }

    {
        llama_dsv4_comp_pool pool({ 8, 4 });
        const auto baseline     = pool.memory_usage_snapshot();
        const auto create_epoch = baseline.epoch;
        const auto detached     = create_handle(pool);
        const auto quote        = quote_change(pool, detached, llama_dsv4_comp_family::c4, 65);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote detached import abort");
        const auto reservation = pool.try_reserve(quote);
        expect_status(reservation.status, llama_dsv4_comp_status::ok, "reserve detached import abort");
        const auto reserved = pool.memory_usage_snapshot();
        expect(reserved.active_tickets == 1 && reserved.c4.reserved_segments == 2,
               "detached import abort did not reserve exact segments");

        expect_status(pool.abort_detached_import(detached, reservation.ticket, create_epoch),
                      llama_dsv4_comp_status::ok, "abort detached import");
        const auto aborted = pool.memory_usage_snapshot();
        expect_semantic_usage_equal(aborted, baseline, "detached import abort baseline");
        expect(aborted.epoch > reserved.epoch, "detached import abort rewound epoch");
        expect(aborted.retained_ticket_records >= baseline.retained_ticket_records + 1,
               "detached import abort discarded ticket tombstone");
        expect_status(pool.commit(reservation.ticket), llama_dsv4_comp_status::stale_ticket,
                      "cancelled detached ticket became live");
        expect_status(pool.abort_detached_import(detached, reservation.ticket, create_epoch),
                      llama_dsv4_comp_status::stale_ticket, "detached import abort replay succeeded");
        llama_dsv4_comp_handle_info stale;
        expect_status(pool.get_handle(detached, stale), llama_dsv4_comp_status::handle_not_found,
                      "aborted detached handle remained live");
        const auto replacement = create_handle(pool);
        expect(replacement > detached, "aborted detached handle ID was reused");
        expect_status(pool.remove_handle(replacement), llama_dsv4_comp_status::ok,
                      "remove detached abort replacement");
    }

    {
        llama_dsv4_comp_pool pool({ 2, 1 });
        const auto baseline = pool.memory_usage_snapshot();
        llama_dsv4_comp_handle_result failed;
        {
            allocation_scope allocations(true);
            failed = pool.create_handle();
        }
        expect_status(failed.status, llama_dsv4_comp_status::resource_exhausted,
                      "detached handle allocation failure status");
        expect_semantic_usage_equal(pool.memory_usage_snapshot(), baseline,
                                    "detached handle allocation failure baseline");

        const auto handle = create_handle(pool);
        const auto quote  = quote_change(pool, handle, llama_dsv4_comp_family::c4, 1);
        expect_status(quote.status, llama_dsv4_comp_status::ok, "quote reserve allocation failure");
        const auto before_reserve = pool.memory_usage_snapshot();
        llama_dsv4_comp_reserve_result reservation;
        {
            allocation_scope allocations(true);
            reservation = pool.try_reserve(quote);
        }
        expect_status(reservation.status, llama_dsv4_comp_status::resource_exhausted,
                      "reserve allocation failure status");
        expect_semantic_usage_equal(pool.memory_usage_snapshot(), before_reserve,
                                    "reserve allocation failure baseline");
        expect(pool.memory_usage_snapshot().epoch == before_reserve.epoch,
               "reserve allocation failure advanced epoch");
        expect_status(pool.remove_handle(handle), llama_dsv4_comp_status::ok,
                      "remove allocation failure handle");
    }
}

void test_generation_checked_binding_replacement() {
    llama_dsv4_comp_pool pool({ 12, 4 });
    const auto           original = create_handle(pool);
    commit_change(pool, original, llama_dsv4_comp_family::c4, 65);
    commit_change(pool, original, llama_dsv4_comp_family::hca, 1);
    expect_status(pool.bind(3, original), llama_dsv4_comp_status::ok, "bind replacement original");
    const auto alias = pool.copy_handle(original);
    expect_status(alias.status, llama_dsv4_comp_status::ok, "copy replacement original");

    const auto replacement = create_handle(pool);
    commit_change(pool, replacement, llama_dsv4_comp_family::c4, 3);
    const auto original_info    = handle_info(pool, original);
    const auto alias_info       = handle_info(pool, alias.handle);
    const auto replacement_info = handle_info(pool, replacement);
    const auto before_failures  = pool.memory_usage_snapshot();

    expect_status(pool.replace_binding(3, original, original_info.generation + 1,
                                       replacement, replacement_info.generation),
                  llama_dsv4_comp_status::stale_handle, "replace stale original generation");
    expect_status(pool.replace_binding(3, original, original_info.generation,
                                       replacement, replacement_info.generation + 1),
                  llama_dsv4_comp_status::stale_handle, "replace stale replacement generation");
    expect_semantic_usage_equal(pool.memory_usage_snapshot(), before_failures,
                                "failed replacement semantic baseline");
    expect(pool.memory_usage_snapshot().epoch == before_failures.epoch,
           "failed replacement advanced epoch");

    llama_dsv4_comp_status status;
    size_t allocations = 0;
    {
        allocation_scope scope(true);
        status      = pool.replace_binding(3, original, original_info.generation,
                                           replacement, replacement_info.generation);
        allocations = scope.finish();
    }
    expect_status(status, llama_dsv4_comp_status::ok, "replace exact generations");
    expect(allocations == 0, "binding replacement allocated");

    llama_dsv4_comp_handle_id binding = 0;
    expect_status(pool.get_binding(3, binding), llama_dsv4_comp_status::ok,
                  "replacement binding missing");
    expect(binding == replacement, "replacement binding published wrong root");
    llama_dsv4_comp_handle_info removed;
    expect_status(pool.get_handle(original, removed), llama_dsv4_comp_status::handle_not_found,
                  "replacement retained old root");
    const auto alias_after = handle_info(pool, alias.handle);
    expect(alias_after.c4_segment_ids == alias_info.c4_segment_ids &&
               alias_after.hca_segment_ids == alias_info.hca_segment_ids &&
               alias_after.generation == alias_info.generation,
           "replacement changed COW alias root");
    const auto after = pool.memory_usage_snapshot();
    expect(after.handles == before_failures.handles - 1 && after.bindings == before_failures.bindings,
           "replacement ownership counts");
    expect(after.c4.shared_segments == 0 && after.hca.shared_segments == 0,
           "replacement did not release only the old root references");

    expect_status(pool.unbind(3), llama_dsv4_comp_status::ok, "unbind replacement root");
    expect_status(pool.remove_handle(replacement), llama_dsv4_comp_status::ok, "remove replacement root");
    expect_status(pool.remove_handle(alias.handle), llama_dsv4_comp_status::ok, "remove replacement alias");
}

}  // namespace

int main() {
    try {
        test_geometry_and_permanent_ownership();
        test_alias_shapes_and_cow_divergence();
        test_lid_group_boundary_and_hca_independence();
        test_atomic_candidate_directories_and_noncontiguous_bindings();
        test_exhaustion_immutability_and_ticket_idempotence();
        test_copy_remove_cycles_return_accounting_baseline();
        test_resident_detach_attach_release_contract();
        test_prepared_resident_detach_rollback();
        test_prepared_resident_attach_transaction();
        test_bounded_ticket_history_and_pool_identity();
        test_recreated_pool_rejects_old_handles();
        test_sixty_four_handle_rounded_capacity_contract();
        test_deterministic_quotes_and_full_context_boundary();
        test_detached_import_abort_preserves_semantics_and_monotonicity();
        test_generation_checked_binding_replacement();
    } catch (const std::exception & error) {
        std::cerr << "test-dsv4-comp-pool: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-dsv4-comp-pool: all checks passed\n";
    return 0;
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

constexpr uint32_t LLAMA_DSV4_COMP_SEGMENT_ROWS       = 64;
constexpr uint32_t LLAMA_DSV4_COMP_C4_TOKENS_PER_ROW  = 4;
constexpr uint32_t LLAMA_DSV4_COMP_HCA_TOKENS_PER_ROW = 128;
constexpr uint32_t LLAMA_DSV4_COMP_GRAPH_STREAMS      = 64;
constexpr uint32_t LLAMA_DSV4_COMP_CSA_LAYERS         = 21;
constexpr uint32_t LLAMA_DSV4_COMP_LID_LAYERS         = 21;
constexpr uint32_t LLAMA_DSV4_COMP_HCA_LAYERS         = 20;
constexpr uint32_t LLAMA_DSV4_COMP_INVALID_SEGMENT    = UINT32_MAX;
constexpr uint32_t LLAMA_DSV4_COMP_TICKET_TOMBSTONES  = 64;

using llama_dsv4_comp_handle_id = uint64_t;

enum class llama_dsv4_comp_family : uint8_t {
    none,
    c4,
    hca,
};

enum class llama_dsv4_comp_status : uint8_t {
    ok,
    invalid_argument,
    handle_not_found,
    binding_not_found,
    capacity_exhausted,
    stale_quote,
    stale_ticket,
    busy,
    slot_occupied,
    handle_bound,
};

const char * llama_dsv4_comp_status_name(llama_dsv4_comp_status status);

uint64_t llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family family, uint64_t token_count);
uint64_t llama_dsv4_comp_segments_for_rows(uint64_t row_count);
uint64_t llama_dsv4_comp_logical_segment(uint64_t logical_row);
uint32_t llama_dsv4_comp_segment_row(uint64_t logical_row);
uint64_t llama_dsv4_comp_physical_row(uint32_t physical_segment, uint64_t logical_row);

struct llama_dsv4_comp_pool_config {
    // These capacities cover allocatable data segments. Each family also owns
    // one permanently mapped zero segment and one permanently mapped scratch
    // segment with 64 graph-stream rows. This is exact rounded-segment
    // capacity: a 1M-token logical aggregate distributed arbitrarily over 64
    // handles needs up to 4096+63 C4 and 128+63 HCA data segments.
    uint32_t c4_data_segments  = 0;
    uint32_t hca_data_segments = 0;
};

struct llama_dsv4_comp_family_usage {
    uint32_t capacity_segments   = 0;
    uint32_t permanent_segments  = 0;
    uint32_t free_segments       = 0;
    uint32_t reserved_segments   = 0;
    uint32_t mapped_segments     = 0;
    uint32_t shared_segments     = 0;
    uint32_t cow_segments        = 0;
    uint32_t scratch_rows_in_use = 0;

    uint64_t capacity_pages = 0;
    uint64_t free_pages     = 0;
    uint64_t reserved_pages = 0;
    uint64_t mapped_pages   = 0;
    uint64_t shared_pages   = 0;
    uint64_t cow_pages      = 0;

    // Only C4 uses these fields. Every C4 segment consumes one CSA page per
    // layer, while four physical segments share one LID page per layer. The
    // permanent segment IDs participate in the same absolute-ID page groups.
    uint64_t segment_pages_capacity = 0;
    uint64_t segment_pages_free     = 0;
    uint64_t segment_pages_reserved = 0;
    uint64_t segment_pages_mapped   = 0;
    uint64_t lid_pages_capacity     = 0;
    uint64_t lid_pages_free         = 0;
    uint64_t lid_pages_reserved     = 0;
    uint64_t lid_pages_mapped       = 0;
};

struct llama_dsv4_comp_memory_usage {
    llama_dsv4_comp_family_usage c4;
    llama_dsv4_comp_family_usage hca;
    uint64_t                     epoch                   = 0;
    uint32_t                     handles                 = 0;
    uint32_t                     bindings                = 0;
    uint32_t                     active_tickets          = 0;
    uint32_t                     retained_ticket_records = 0;
};

struct llama_dsv4_comp_handle_info {
    llama_dsv4_comp_handle_id id               = 0;
    uint64_t                  generation       = 0;
    uint64_t                  visible_c4_rows  = 0;
    uint64_t                  visible_hca_rows = 0;
    std::vector<uint32_t>     c4_segment_ids;
    std::vector<uint32_t>     hca_segment_ids;
};

struct llama_dsv4_comp_handle_result {
    llama_dsv4_comp_status    status = llama_dsv4_comp_status::invalid_argument;
    llama_dsv4_comp_handle_id handle = 0;
};

struct llama_dsv4_comp_change {
    llama_dsv4_comp_handle_id handle           = 0;
    llama_dsv4_comp_family    family           = llama_dsv4_comp_family::none;
    uint64_t                  new_visible_rows = 0;
    std::vector<uint64_t>     overwrite_rows;
};

struct llama_dsv4_comp_batch_plan {
    std::vector<llama_dsv4_comp_change> changes;
    // Execution IDs may be sparse and unordered. Their order defines graph
    // directory columns and scratch-row ownership for this ticket.
    std::vector<uint32_t>               graph_execution_ids;
};

struct llama_dsv4_comp_allocation {
    llama_dsv4_comp_family    family              = llama_dsv4_comp_family::none;
    llama_dsv4_comp_handle_id handle              = 0;
    uint32_t                  logical_segment     = 0;
    uint32_t                  source_segment      = LLAMA_DSV4_COMP_INVALID_SEGMENT;
    uint32_t                  destination_segment = LLAMA_DSV4_COMP_INVALID_SEGMENT;
    uint32_t                  populated_rows      = 0;
    bool                      cow                 = false;
};

struct llama_dsv4_comp_family_quote {
    uint32_t new_segments  = 0;
    uint32_t cow_segments  = 0;
    uint64_t segment_pages = 0;
    uint64_t lid_pages     = 0;

    uint64_t total_pages() const { return segment_pages + lid_pages; }
};

struct llama_dsv4_comp_quote_plan;

struct llama_dsv4_comp_quote {
    llama_dsv4_comp_status                  status          = llama_dsv4_comp_status::invalid_argument;
    llama_dsv4_comp_family                  limiting_family = llama_dsv4_comp_family::none;
    llama_dsv4_comp_family_quote            c4;
    llama_dsv4_comp_family_quote            hca;
    uint32_t                                scratch_rows = 0;
    uint64_t                                pool_epoch   = 0;
    std::vector<llama_dsv4_comp_allocation> allocations;

  private:
    std::shared_ptr<const llama_dsv4_comp_quote_plan> plan;
    friend class llama_dsv4_comp_pool;
};

struct llama_dsv4_comp_ticket {
    uint64_t id         = 0;
    uint64_t generation = 0;
};

struct llama_dsv4_comp_reserve_result {
    llama_dsv4_comp_status status = llama_dsv4_comp_status::invalid_argument;
    llama_dsv4_comp_ticket ticket;
};

struct llama_dsv4_comp_directory {
    llama_dsv4_comp_status status           = llama_dsv4_comp_status::invalid_argument;
    uint32_t               logical_segments = 0;
    uint32_t               graph_streams    = 0;
    // Column-major: logical_segment + graph_stream*logical_segments.
    std::vector<uint32_t>  segment_ids;
};

class llama_dsv4_comp_pool {
  public:
    explicit llama_dsv4_comp_pool(llama_dsv4_comp_pool_config config);
    ~llama_dsv4_comp_pool();

    llama_dsv4_comp_pool(const llama_dsv4_comp_pool &)             = delete;
    llama_dsv4_comp_pool & operator=(const llama_dsv4_comp_pool &) = delete;
    llama_dsv4_comp_pool(llama_dsv4_comp_pool &&) noexcept;
    llama_dsv4_comp_pool & operator=(llama_dsv4_comp_pool &&) noexcept;

    llama_dsv4_comp_memory_usage memory_usage_snapshot() const;

    llama_dsv4_comp_handle_result create_handle();
    llama_dsv4_comp_handle_result copy_handle(llama_dsv4_comp_handle_id source);
    llama_dsv4_comp_status        remove_handle(llama_dsv4_comp_handle_id handle);
    llama_dsv4_comp_status get_handle(llama_dsv4_comp_handle_id handle, llama_dsv4_comp_handle_info & result) const;

    llama_dsv4_comp_status bind(uint32_t execution_id, llama_dsv4_comp_handle_id handle);
    llama_dsv4_comp_status unbind(uint32_t execution_id);
    llama_dsv4_comp_status get_binding(uint32_t execution_id, llama_dsv4_comp_handle_id & handle) const;

    llama_dsv4_comp_quote          quote_batch(const llama_dsv4_comp_batch_plan & batch) const;
    llama_dsv4_comp_reserve_result try_reserve(const llama_dsv4_comp_quote & quote);
    llama_dsv4_comp_status         commit(llama_dsv4_comp_ticket ticket);
    llama_dsv4_comp_status         rollback(llama_dsv4_comp_ticket ticket);
    llama_dsv4_comp_status         cancel(llama_dsv4_comp_ticket ticket);

    llama_dsv4_comp_status candidate_handle(llama_dsv4_comp_ticket        ticket,
                                            llama_dsv4_comp_handle_id     handle,
                                            llama_dsv4_comp_handle_info & result) const;

    llama_dsv4_comp_directory directory_for_bindings(llama_dsv4_comp_family        family,
                                                     const std::vector<uint32_t> & execution_ids,
                                                     uint32_t                      logical_segments) const;
    llama_dsv4_comp_directory ticket_directory(llama_dsv4_comp_ticket ticket,
                                               llama_dsv4_comp_family family,
                                               uint32_t               logical_segments) const;

    uint32_t zero_segment(llama_dsv4_comp_family family) const;
    uint32_t scratch_segment(llama_dsv4_comp_family family) const;
    uint64_t zero_physical_row(llama_dsv4_comp_family family, uint64_t logical_row) const;
    uint64_t scratch_physical_row(llama_dsv4_comp_family family, uint32_t graph_stream) const;

  private:
    struct impl;
    llama_dsv4_comp_directory make_directory(llama_dsv4_comp_family             family,
                                             const std::vector<uint32_t> &      execution_ids,
                                             uint32_t                           logical_segments,
                                             const llama_dsv4_comp_quote_plan * plan) const;
    std::unique_ptr<impl>     pimpl;
};

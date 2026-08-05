#include "llama-dsv4-comp-pool.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace {

constexpr uint32_t PERMANENT_SEGMENTS = 2;
constexpr uint32_t ZERO_SEGMENT       = 0;
constexpr uint32_t SCRATCH_SEGMENT    = 1;

enum class segment_state : uint8_t {
    free,
    reserved,
    mapped,
    permanent,
};

struct segment_record {
    segment_state state        = segment_state::free;
    uint64_t      generation   = 1;
    uint64_t      owner_ticket = 0;
    uint32_t      refs         = 0;
    bool          cow          = false;
};

struct family_state {
    llama_dsv4_comp_family      family = llama_dsv4_comp_family::none;
    std::vector<segment_record> segments;

    family_state() = default;

    family_state(llama_dsv4_comp_family family, uint32_t data_segments) : family(family) {
        segments.resize(static_cast<size_t>(data_segments) + PERMANENT_SEGMENTS);
        segments[ZERO_SEGMENT].state    = segment_state::permanent;
        segments[SCRATCH_SEGMENT].state = segment_state::permanent;
    }

    uint32_t data_segments() const { return static_cast<uint32_t>(segments.size()) - PERMANENT_SEGMENTS; }
};

struct resident_handle {
    llama_dsv4_comp_handle_id id               = 0;
    uint64_t                  generation       = 1;
    uint64_t                  visible_c4_rows  = 0;
    uint64_t                  visible_hca_rows = 0;
    std::vector<uint32_t>     c4_segment_ids;
    std::vector<uint32_t>     hca_segment_ids;
    uint64_t                  resident_generation = 0;
    bool                      resident_owned      = false;
};

struct planned_candidate {
    llama_dsv4_comp_handle_id handle                     = 0;
    llama_dsv4_comp_family    family                     = llama_dsv4_comp_family::none;
    uint64_t                  expected_handle_generation = 0;
    uint64_t                  old_visible_rows           = 0;
    uint64_t                  new_visible_rows           = 0;
    std::vector<uint32_t>     old_segment_ids;
    std::vector<uint32_t>     candidate_segment_ids;
    std::set<uint32_t>        cow_logical_segments;
};

struct planned_reservation {
    llama_dsv4_comp_allocation allocation;
    uint64_t                   expected_segment_generation = 0;
};

uint64_t allocate_pool_id() {
    static std::atomic<uint64_t> next{ 1 };
    uint64_t                     current = next.load(std::memory_order_relaxed);
    while (true) {
        if (current == 0 || current == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("DSV4 compressed-pool identity space exhausted");
        }
        if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current;
        }
    }
}

struct pool_identity {
    uint64_t id = allocate_pool_id();
};

enum class ticket_state : uint8_t {
    active,
    committed,
    rolled_back,
    cancelled,
};

enum class resident_attach_state : uint8_t {
    prepared,
    committed,
    rolled_back,
};

enum class resident_detach_state : uint8_t {
    prepared,
    committed,
    rolled_back,
};

enum class resident_release_state : uint8_t {
    prepared,
    committed,
    rolled_back,
};

uint32_t populated_rows_in_segment(uint64_t visible_rows, uint32_t logical_segment) {
    const uint64_t start = static_cast<uint64_t>(logical_segment) * LLAMA_DSV4_COMP_SEGMENT_ROWS;
    if (visible_rows <= start) {
        return 0;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(LLAMA_DSV4_COMP_SEGMENT_ROWS, visible_rows - start));
}

bool is_valid_family(llama_dsv4_comp_family family) {
    return family == llama_dsv4_comp_family::c4 || family == llama_dsv4_comp_family::hca;
}

std::vector<uint32_t> free_segment_ids(const family_state & family) {
    std::vector<uint32_t> result;
    for (uint32_t id = PERMANENT_SEGMENTS; id < family.segments.size(); ++id) {
        if (family.segments[id].state == segment_state::free) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<bool> c4_lid_occupied_groups(const family_state & c4) {
    const size_t      groups = (c4.segments.size() + 3) / 4;
    std::vector<bool> occupied(groups, false);
    for (uint32_t id = 0; id < c4.segments.size(); ++id) {
        if (c4.segments[id].state != segment_state::free) {
            occupied[id / 4] = true;
        }
    }
    return occupied;
}

const std::vector<uint32_t> & segment_ids_for(const resident_handle & handle, llama_dsv4_comp_family family) {
    return family == llama_dsv4_comp_family::c4 ? handle.c4_segment_ids : handle.hca_segment_ids;
}

uint64_t visible_rows_for(const resident_handle & handle, llama_dsv4_comp_family family) {
    return family == llama_dsv4_comp_family::c4 ? handle.visible_c4_rows : handle.visible_hca_rows;
}

void export_handle(const resident_handle & source, llama_dsv4_comp_handle_info & result) {
    result.id               = source.id;
    result.generation       = source.generation;
    result.visible_c4_rows  = source.visible_c4_rows;
    result.visible_hca_rows = source.visible_hca_rows;
    result.c4_segment_ids   = source.c4_segment_ids;
    result.hca_segment_ids  = source.hca_segment_ids;
}

}  // namespace

struct llama_dsv4_comp_quote_plan {
    std::shared_ptr<const pool_identity> owner;
    llama_dsv4_comp_status               status     = llama_dsv4_comp_status::invalid_argument;
    uint64_t                             pool_epoch = 0;
    std::vector<uint32_t>                graph_execution_ids;
    std::vector<planned_candidate>       candidates;
    std::vector<planned_reservation>     reservations;
};

struct llama_dsv4_comp_detach_plan {
    std::shared_ptr<const pool_identity> owner;
    llama_dsv4_comp_status               status                       = llama_dsv4_comp_status::invalid_argument;
    uint64_t                             pool_epoch                   = 0;
    uint32_t                             execution_id                 = UINT32_MAX;
    llama_dsv4_comp_handle_id            handle                       = 0;
    uint64_t                             expected_handle_generation   = 0;
    uint64_t                             expected_resident_generation = 0;
    uint64_t                             commit_epoch                 = 0;
    std::map<uint32_t, llama_dsv4_comp_handle_id> detached_binding;
    std::map<uint32_t, llama_dsv4_comp_handle_id>        replacement_binding;
    std::map<llama_dsv4_comp_handle_id, resident_handle> replacement_handle;
    llama_dsv4_comp_handle_id                            replacement_handle_id    = 0;
    bool                                                 preserve_empty_execution = false;
    resident_detach_state                state = resident_detach_state::prepared;
};

struct llama_dsv4_comp_attach_plan {
    std::shared_ptr<const pool_identity>                    owner;
    uint64_t                                                pool_epoch = 0;
    uint64_t                                                commit_epoch = 0;
    uint32_t                                                execution_id = UINT32_MAX;
    llama_dsv4_comp_resident_handle                         resident;
    std::map<uint32_t, llama_dsv4_comp_handle_id>           prepared_binding;
    std::map<uint32_t, llama_dsv4_comp_handle_id>           replaced_binding;
    std::map<llama_dsv4_comp_handle_id, resident_handle>    replaced_handle;
    llama_dsv4_comp_handle_id                               expected_replaced_handle = 0;
    uint64_t                                                expected_replaced_generation = 0;
    bool                                                    replace_empty = false;
    resident_attach_state                                   state = resident_attach_state::prepared;
};

struct llama_dsv4_comp_release_plan {
    std::shared_ptr<const pool_identity>                  owner;
    uint64_t                                             release_id = 0;
    llama_dsv4_comp_resident_handle                      resident;
    std::map<llama_dsv4_comp_handle_id, resident_handle> retained_handle;
    resident_release_state                               state = resident_release_state::prepared;
};

struct llama_dsv4_comp_pool::impl {
    struct ticket_record {
        uint64_t                                          generation = 0;
        ticket_state                                      state      = ticket_state::active;
        std::shared_ptr<const llama_dsv4_comp_quote_plan> plan;
    };

    explicit impl(llama_dsv4_comp_pool_config config) :
        c4(llama_dsv4_comp_family::c4, config.c4_data_segments),
        hca(llama_dsv4_comp_family::hca, config.hca_data_segments) {}

    family_state                                         c4;
    family_state                                         hca;
    std::shared_ptr<const pool_identity>                 identity = std::make_shared<const pool_identity>();
    std::map<llama_dsv4_comp_handle_id, resident_handle> handles;
    std::map<uint32_t, llama_dsv4_comp_handle_id>        bindings;
    std::map<uint64_t, ticket_record>                    tickets;
    llama_dsv4_comp_handle_id                            next_handle_id         = 1;
    uint64_t                                             next_ticket_id         = 1;
    uint64_t                                             next_ticket_generation = 1;
    uint64_t                                             active_ticket_id       = 0;
    uint64_t                                             active_release_id      = 0;
    uint64_t                                             next_release_id        = 1;
    uint32_t                                             scratch_rows_in_use    = 0;
    uint64_t                                             epoch                  = 1;

    void prune_terminal_tickets() {
        size_t terminal_count = tickets.size() - (active_ticket_id == 0 ? 0 : 1);
        for (auto it = tickets.begin(); terminal_count > LLAMA_DSV4_COMP_TICKET_TOMBSTONES && it != tickets.end();) {
            if (it->second.state == ticket_state::active) {
                ++it;
                continue;
            }
            it = tickets.erase(it);
            --terminal_count;
        }
    }

    family_state & family(llama_dsv4_comp_family value) { return value == llama_dsv4_comp_family::c4 ? c4 : hca; }

    const family_state & family(llama_dsv4_comp_family value) const {
        return value == llama_dsv4_comp_family::c4 ? c4 : hca;
    }

    bool busy() const { return active_ticket_id != 0 || active_release_id != 0; }

    bool has_resident_handles() const {
        return active_release_id != 0 || std::any_of(handles.begin(), handles.end(),
                           [](const auto & item) { return item.second.resident_owned; });
    }

    bool is_bound(llama_dsv4_comp_handle_id handle) const {
        return std::any_of(bindings.begin(), bindings.end(),
                           [&](const auto & binding) { return binding.second == handle; });
    }

    bool releasable(const std::vector<uint32_t> & ids, llama_dsv4_comp_family value) const {
        const auto & state = family(value);
        return std::all_of(ids.begin(), ids.end(), [&](uint32_t id) {
            return id < state.segments.size() && state.segments[id].state == segment_state::mapped &&
                   state.segments[id].refs > 0;
        });
    }

    void retain(const std::vector<uint32_t> & ids, llama_dsv4_comp_family value) {
        auto & state = family(value);
        for (uint32_t id : ids) {
            auto & segment = state.segments.at(id);
            if (segment.state != segment_state::mapped) {
                throw std::logic_error("retaining a non-mapped DSV4 segment");
            }
            ++segment.refs;
        }
    }

    void release(const std::vector<uint32_t> & ids, llama_dsv4_comp_family value) {
        auto & state = family(value);
        for (uint32_t id : ids) {
            auto & segment = state.segments.at(id);
            if (segment.state != segment_state::mapped || segment.refs == 0) {
                throw std::logic_error("releasing an unreferenced DSV4 segment");
            }
            --segment.refs;
            if (segment.refs == 0) {
                segment.state        = segment_state::free;
                segment.cow          = false;
                segment.owner_ticket = 0;
                ++segment.generation;
            }
        }
    }

    const ticket_record * find_ticket(llama_dsv4_comp_ticket ticket) const {
        const auto it = tickets.find(ticket.id);
        if (it == tickets.end() || it->second.generation != ticket.generation) {
            return nullptr;
        }
        return &it->second;
    }

    ticket_record * find_ticket(llama_dsv4_comp_ticket ticket) {
        const auto it = tickets.find(ticket.id);
        if (it == tickets.end() || it->second.generation != ticket.generation) {
            return nullptr;
        }
        return &it->second;
    }

    const planned_candidate * find_candidate(const llama_dsv4_comp_quote_plan & plan,
                                             llama_dsv4_comp_handle_id          handle,
                                             llama_dsv4_comp_family             value) const {
        const auto it =
            std::find_if(plan.candidates.begin(), plan.candidates.end(),
                         [&](const planned_candidate & item) { return item.handle == handle && item.family == value; });
        return it == plan.candidates.end() ? nullptr : &*it;
    }

    llama_dsv4_comp_status terminal_release(llama_dsv4_comp_ticket ticket, ticket_state terminal) {
        auto * record = find_ticket(ticket);
        if (record == nullptr) {
            return llama_dsv4_comp_status::stale_ticket;
        }
        if (record->state == ticket_state::rolled_back || record->state == ticket_state::cancelled) {
            return llama_dsv4_comp_status::ok;
        }
        if (record->state == ticket_state::committed || active_ticket_id != ticket.id) {
            return llama_dsv4_comp_status::stale_ticket;
        }

        for (const planned_reservation & reservation : record->plan->reservations) {
            auto & segment =
                family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
            if (segment.state != segment_state::reserved || segment.owner_ticket != ticket.id ||
                segment.generation != reservation.expected_segment_generation + 1) {
                return llama_dsv4_comp_status::stale_ticket;
            }
        }
        for (const planned_reservation & reservation : record->plan->reservations) {
            auto & segment =
                family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
            segment.state        = segment_state::free;
            segment.owner_ticket = 0;
            segment.cow          = false;
            ++segment.generation;
        }
        scratch_rows_in_use = 0;
        active_ticket_id    = 0;
        record->state       = terminal;
        record->plan.reset();
        ++epoch;
        prune_terminal_tickets();
        return llama_dsv4_comp_status::ok;
    }
};

const char * llama_dsv4_comp_status_name(llama_dsv4_comp_status status) {
    switch (status) {
        case llama_dsv4_comp_status::ok:
            return "ok";
        case llama_dsv4_comp_status::invalid_argument:
            return "invalid_argument";
        case llama_dsv4_comp_status::handle_not_found:
            return "handle_not_found";
        case llama_dsv4_comp_status::stale_handle:
            return "stale_handle";
        case llama_dsv4_comp_status::binding_not_found:
            return "binding_not_found";
        case llama_dsv4_comp_status::capacity_exhausted:
            return "capacity_exhausted";
        case llama_dsv4_comp_status::generation_exhausted:
            return "generation_exhausted";
        case llama_dsv4_comp_status::stale_quote:
            return "stale_quote";
        case llama_dsv4_comp_status::stale_ticket:
            return "stale_ticket";
        case llama_dsv4_comp_status::busy:
            return "busy";
        case llama_dsv4_comp_status::slot_occupied:
            return "slot_occupied";
        case llama_dsv4_comp_status::handle_bound:
            return "handle_bound";
        case llama_dsv4_comp_status::handle_resident:
            return "handle_resident";
        case llama_dsv4_comp_status::resource_exhausted:
            return "resource_exhausted";
    }
    return "unknown";
}

uint64_t llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family family, uint64_t token_count) {
    if (family == llama_dsv4_comp_family::c4) {
        return token_count / LLAMA_DSV4_COMP_C4_TOKENS_PER_ROW;
    }
    if (family == llama_dsv4_comp_family::hca) {
        return token_count / LLAMA_DSV4_COMP_HCA_TOKENS_PER_ROW;
    }
    return 0;
}

uint64_t llama_dsv4_comp_segments_for_rows(uint64_t row_count) {
    return row_count / LLAMA_DSV4_COMP_SEGMENT_ROWS + (row_count % LLAMA_DSV4_COMP_SEGMENT_ROWS != 0);
}

uint64_t llama_dsv4_comp_logical_segment(uint64_t logical_row) {
    return logical_row / LLAMA_DSV4_COMP_SEGMENT_ROWS;
}

uint32_t llama_dsv4_comp_segment_row(uint64_t logical_row) {
    return static_cast<uint32_t>(logical_row % LLAMA_DSV4_COMP_SEGMENT_ROWS);
}

uint64_t llama_dsv4_comp_physical_row(uint32_t physical_segment, uint64_t logical_row) {
    return static_cast<uint64_t>(physical_segment) * LLAMA_DSV4_COMP_SEGMENT_ROWS +
           llama_dsv4_comp_segment_row(logical_row);
}

llama_dsv4_comp_pool::llama_dsv4_comp_pool(llama_dsv4_comp_pool_config config) : pimpl(new impl(config)) {}

llama_dsv4_comp_pool::~llama_dsv4_comp_pool() {
    if (pimpl && pimpl->has_resident_handles()) {
        std::terminate();
    }
}
llama_dsv4_comp_pool::llama_dsv4_comp_pool(llama_dsv4_comp_pool &&) noexcept = default;

llama_dsv4_comp_pool & llama_dsv4_comp_pool::operator=(llama_dsv4_comp_pool && other) noexcept {
    if (this != &other) {
        if (pimpl && pimpl->has_resident_handles()) {
            std::terminate();
        }
        pimpl = std::move(other.pimpl);
    }
    return *this;
}

static llama_dsv4_comp_family_usage family_usage_for(const family_state & family, uint32_t scratch_rows_in_use) {
    llama_dsv4_comp_family_usage result;
    result.capacity_segments   = static_cast<uint32_t>(family.segments.size());
    result.permanent_segments  = PERMANENT_SEGMENTS;
    result.scratch_rows_in_use = scratch_rows_in_use;

    std::vector<bool> lid_mapped;
    std::vector<bool> lid_reserved;
    std::vector<bool> lid_shared;
    std::vector<bool> lid_cow;
    if (family.family == llama_dsv4_comp_family::c4) {
        const size_t groups = (family.segments.size() + 3) / 4;
        lid_mapped.resize(groups, false);
        lid_reserved.resize(groups, false);
        lid_shared.resize(groups, false);
        lid_cow.resize(groups, false);
    }

    for (uint32_t id = 0; id < family.segments.size(); ++id) {
        const segment_record & segment = family.segments[id];
        switch (segment.state) {
            case segment_state::free:
                ++result.free_segments;
                break;
            case segment_state::reserved:
                ++result.reserved_segments;
                break;
            case segment_state::mapped:
            case segment_state::permanent:
                ++result.mapped_segments;
                break;
        }
        if (segment.state == segment_state::mapped && segment.refs > 1) {
            ++result.shared_segments;
        }
        if (segment.state == segment_state::mapped && segment.cow) {
            ++result.cow_segments;
        }
        if (family.family == llama_dsv4_comp_family::c4) {
            const size_t group = id / 4;
            lid_mapped[group]  = lid_mapped[group] || segment.state == segment_state::mapped ||
                                segment.state == segment_state::permanent;
            lid_reserved[group] = lid_reserved[group] || segment.state == segment_state::reserved;
            lid_shared[group]   = lid_shared[group] || (segment.state == segment_state::mapped && segment.refs > 1);
            lid_cow[group]      = lid_cow[group] || (segment.state == segment_state::mapped && segment.cow);
        }
    }

    if (family.family == llama_dsv4_comp_family::c4) {
        result.segment_pages_capacity = static_cast<uint64_t>(family.segments.size()) * LLAMA_DSV4_COMP_CSA_LAYERS;
        result.segment_pages_mapped   = static_cast<uint64_t>(result.mapped_segments) * LLAMA_DSV4_COMP_CSA_LAYERS;
        result.segment_pages_reserved = static_cast<uint64_t>(result.reserved_segments) * LLAMA_DSV4_COMP_CSA_LAYERS;
        result.segment_pages_free =
            result.segment_pages_capacity - result.segment_pages_mapped - result.segment_pages_reserved;

        result.lid_pages_capacity = static_cast<uint64_t>(lid_mapped.size()) * LLAMA_DSV4_COMP_LID_LAYERS;
        result.lid_pages_mapped   = 0;
        result.lid_pages_reserved = 0;
        for (size_t group = 0; group < lid_mapped.size(); ++group) {
            if (lid_mapped[group]) {
                result.lid_pages_mapped += LLAMA_DSV4_COMP_LID_LAYERS;
            } else if (lid_reserved[group]) {
                result.lid_pages_reserved += LLAMA_DSV4_COMP_LID_LAYERS;
            }
        }
        result.lid_pages_free = result.lid_pages_capacity - result.lid_pages_mapped - result.lid_pages_reserved;
        result.capacity_pages = result.segment_pages_capacity + result.lid_pages_capacity;
        result.mapped_pages   = result.segment_pages_mapped + result.lid_pages_mapped;
        result.reserved_pages = result.segment_pages_reserved + result.lid_pages_reserved;
        result.free_pages     = result.capacity_pages - result.mapped_pages - result.reserved_pages;
        result.shared_pages =
            static_cast<uint64_t>(result.shared_segments) * LLAMA_DSV4_COMP_CSA_LAYERS +
            static_cast<uint64_t>(std::count(lid_shared.begin(), lid_shared.end(), true)) * LLAMA_DSV4_COMP_LID_LAYERS;
        result.cow_pages =
            static_cast<uint64_t>(result.cow_segments) * LLAMA_DSV4_COMP_CSA_LAYERS +
            static_cast<uint64_t>(std::count(lid_cow.begin(), lid_cow.end(), true)) * LLAMA_DSV4_COMP_LID_LAYERS;
    } else {
        result.capacity_pages = static_cast<uint64_t>(family.segments.size()) * LLAMA_DSV4_COMP_HCA_LAYERS;
        result.mapped_pages   = static_cast<uint64_t>(result.mapped_segments) * LLAMA_DSV4_COMP_HCA_LAYERS;
        result.reserved_pages = static_cast<uint64_t>(result.reserved_segments) * LLAMA_DSV4_COMP_HCA_LAYERS;
        result.free_pages     = result.capacity_pages - result.mapped_pages - result.reserved_pages;
        result.shared_pages   = static_cast<uint64_t>(result.shared_segments) * LLAMA_DSV4_COMP_HCA_LAYERS;
        result.cow_pages      = static_cast<uint64_t>(result.cow_segments) * LLAMA_DSV4_COMP_HCA_LAYERS;
    }
    return result;
}

llama_dsv4_comp_memory_usage llama_dsv4_comp_pool::memory_usage_snapshot() const {
    llama_dsv4_comp_memory_usage result;
    result.c4                      = family_usage_for(pimpl->c4, pimpl->scratch_rows_in_use);
    result.hca                     = family_usage_for(pimpl->hca, pimpl->scratch_rows_in_use);
    result.epoch                   = pimpl->epoch;
    result.handles                 = static_cast<uint32_t>(pimpl->handles.size());
    result.bindings                = static_cast<uint32_t>(pimpl->bindings.size());
    result.resident_handles        = static_cast<uint32_t>(std::count_if(
        pimpl->handles.begin(), pimpl->handles.end(), [](const auto & item) { return item.second.resident_owned; }));
    result.active_tickets          = pimpl->active_ticket_id == 0 ? 0 : 1;
    result.retained_ticket_records = static_cast<uint32_t>(pimpl->tickets.size());
    return result;
}

llama_dsv4_comp_handle_result llama_dsv4_comp_pool::create_handle() {
    if (pimpl->busy()) {
        return { llama_dsv4_comp_status::busy, 0 };
    }
    if (pimpl->next_handle_id == 0) {
        return { llama_dsv4_comp_status::generation_exhausted, 0 };
    }
    resident_handle handle;
    handle.id             = pimpl->next_handle_id;
    pimpl->next_handle_id = handle.id == std::numeric_limits<llama_dsv4_comp_handle_id>::max() ? 0 : handle.id + 1;
    pimpl->handles.emplace(handle.id, handle);
    ++pimpl->epoch;
    return { llama_dsv4_comp_status::ok, handle.id };
}

llama_dsv4_comp_handle_result llama_dsv4_comp_pool::copy_handle(llama_dsv4_comp_handle_id source) {
    if (pimpl->busy()) {
        return { llama_dsv4_comp_status::busy, 0 };
    }
    const auto source_it = pimpl->handles.find(source);
    if (source_it == pimpl->handles.end()) {
        return { llama_dsv4_comp_status::handle_not_found, 0 };
    }
    if (source_it->second.resident_owned) {
        return { llama_dsv4_comp_status::handle_resident, 0 };
    }
    if (pimpl->next_handle_id == 0) {
        return { llama_dsv4_comp_status::generation_exhausted, 0 };
    }
    resident_handle copy = source_it->second;
    copy.id                  = pimpl->next_handle_id;
    pimpl->next_handle_id    = copy.id == std::numeric_limits<llama_dsv4_comp_handle_id>::max() ? 0 : copy.id + 1;
    copy.generation      = 1;
    copy.resident_generation = 0;
    copy.resident_owned      = false;
    pimpl->retain(copy.c4_segment_ids, llama_dsv4_comp_family::c4);
    pimpl->retain(copy.hca_segment_ids, llama_dsv4_comp_family::hca);
    pimpl->handles.emplace(copy.id, copy);
    ++pimpl->epoch;
    return { llama_dsv4_comp_status::ok, copy.id };
}

llama_dsv4_comp_status llama_dsv4_comp_pool::remove_handle(llama_dsv4_comp_handle_id handle) {
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    const auto it = pimpl->handles.find(handle);
    if (it == pimpl->handles.end()) {
        return llama_dsv4_comp_status::handle_not_found;
    }
    if (it->second.resident_owned) {
        return llama_dsv4_comp_status::handle_resident;
    }
    if (pimpl->is_bound(handle)) {
        return llama_dsv4_comp_status::handle_bound;
    }
    if (!pimpl->releasable(it->second.c4_segment_ids, llama_dsv4_comp_family::c4) ||
        !pimpl->releasable(it->second.hca_segment_ids, llama_dsv4_comp_family::hca)) {
        return llama_dsv4_comp_status::stale_handle;
    }
    pimpl->release(it->second.c4_segment_ids, llama_dsv4_comp_family::c4);
    pimpl->release(it->second.hca_segment_ids, llama_dsv4_comp_family::hca);
    pimpl->handles.erase(it);
    ++pimpl->epoch;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::get_handle(llama_dsv4_comp_handle_id     handle,
                                                        llama_dsv4_comp_handle_info & result) const {
    const auto it = pimpl->handles.find(handle);
    if (it == pimpl->handles.end()) {
        return llama_dsv4_comp_status::handle_not_found;
    }
    export_handle(it->second, result);
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::bind(uint32_t execution_id, llama_dsv4_comp_handle_id handle) {
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    if (execution_id >= LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        return llama_dsv4_comp_status::invalid_argument;
    }
    const auto handle_it = pimpl->handles.find(handle);
    if (handle_it == pimpl->handles.end()) {
        return llama_dsv4_comp_status::handle_not_found;
    }
    if (handle_it->second.resident_owned) {
        return llama_dsv4_comp_status::handle_resident;
    }
    if (pimpl->bindings.count(execution_id) != 0) {
        return llama_dsv4_comp_status::slot_occupied;
    }
    pimpl->bindings.emplace(execution_id, handle);
    ++pimpl->epoch;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::unbind(uint32_t execution_id) {
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    const auto it = pimpl->bindings.find(execution_id);
    if (it == pimpl->bindings.end()) {
        return llama_dsv4_comp_status::binding_not_found;
    }
    pimpl->bindings.erase(it);
    ++pimpl->epoch;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::get_binding(uint32_t                    execution_id,
                                                         llama_dsv4_comp_handle_id & handle) const {
    const auto it = pimpl->bindings.find(execution_id);
    if (it == pimpl->bindings.end()) {
        return llama_dsv4_comp_status::binding_not_found;
    }
    handle = it->second;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_detach_quote llama_dsv4_comp_pool::quote_detach(uint32_t execution_id) const {
    return quote_detach_impl(execution_id, false);
}

llama_dsv4_comp_detach_quote llama_dsv4_comp_pool::quote_detach_preserving_empty_execution(
    uint32_t execution_id) const {
    return quote_detach_impl(execution_id, true);
}

llama_dsv4_comp_detach_quote llama_dsv4_comp_pool::quote_detach_impl(uint32_t execution_id,
                                                                     bool     preserve_empty_execution) const {
    llama_dsv4_comp_detach_quote result;
    std::shared_ptr<llama_dsv4_comp_detach_plan> plan;
    try {
        plan = std::make_shared<llama_dsv4_comp_detach_plan>();
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_comp_status::resource_exhausted;
        return result;
    }
    plan->owner                       = pimpl->identity;
    plan->pool_epoch                  = pimpl->epoch;
    plan->execution_id                = execution_id;
    result.execution_id               = execution_id;
    result.pool_epoch                 = pimpl->epoch;
    result.plan                       = plan;

    if (pimpl->busy()) {
        result.status = plan->status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (execution_id >= LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        result.status = plan->status = llama_dsv4_comp_status::invalid_argument;
        return result;
    }
    const auto binding_it = pimpl->bindings.find(execution_id);
    if (binding_it == pimpl->bindings.end()) {
        result.status = plan->status = llama_dsv4_comp_status::binding_not_found;
        return result;
    }
    const auto handle_it = pimpl->handles.find(binding_it->second);
    if (handle_it == pimpl->handles.end() || handle_it->second.resident_owned) {
        result.status = plan->status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    if (handle_it->second.resident_generation == std::numeric_limits<uint64_t>::max()) {
        result.status = plan->status = llama_dsv4_comp_status::generation_exhausted;
        return result;
    }
    if (std::count_if(pimpl->bindings.begin(), pimpl->bindings.end(),
                      [&](const auto & binding) { return binding.second == binding_it->second; }) != 1) {
        result.status = plan->status = llama_dsv4_comp_status::handle_bound;
        return result;
    }

    if (preserve_empty_execution && pimpl->next_handle_id == 0) {
        result.status = plan->status = llama_dsv4_comp_status::generation_exhausted;
        return result;
    }

    if (preserve_empty_execution) {
        try {
            resident_handle replacement;
            replacement.id = pimpl->next_handle_id;
            plan->replacement_handle.emplace(replacement.id, replacement);
            plan->replacement_binding.emplace(execution_id, replacement.id);
            plan->replacement_handle_id    = replacement.id;
            plan->preserve_empty_execution = true;
        } catch (const std::bad_alloc &) {
            result.status = plan->status = llama_dsv4_comp_status::resource_exhausted;
            return result;
        }
    }

    plan->handle                       = handle_it->first;
    plan->expected_handle_generation   = handle_it->second.generation;
    plan->expected_resident_generation = handle_it->second.resident_generation;
    result.resident                    = {
        pimpl->identity->id,
        handle_it->first,
        handle_it->second.generation,
        handle_it->second.resident_generation + 1,
    };
    result.status = plan->status = llama_dsv4_comp_status::ok;
    return result;
}

llama_dsv4_comp_resident_result llama_dsv4_comp_pool::detach(const llama_dsv4_comp_detach_quote & quote) {
    llama_dsv4_comp_resident_result result;
    if (!quote.plan || quote.plan->owner != pimpl->identity ||
        quote.plan->state != resident_detach_state::prepared) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    if (quote.plan->status != llama_dsv4_comp_status::ok) {
        result.status = quote.plan->status;
        return result;
    }
    if (pimpl->busy()) {
        result.status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (quote.plan->pool_epoch != pimpl->epoch) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    const auto binding_it = pimpl->bindings.find(quote.plan->execution_id);
    const auto handle_it  = pimpl->handles.find(quote.plan->handle);
    if (binding_it == pimpl->bindings.end() || binding_it->second != quote.plan->handle ||
        handle_it == pimpl->handles.end() || handle_it->second.resident_owned ||
        handle_it->second.generation != quote.plan->expected_handle_generation ||
        handle_it->second.resident_generation != quote.plan->expected_resident_generation) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    if (quote.plan->preserve_empty_execution &&
        (quote.plan->replacement_handle_id == 0 || quote.plan->replacement_handle_id != pimpl->next_handle_id ||
         pimpl->handles.count(quote.plan->replacement_handle_id) != 0 || quote.plan->replacement_handle.size() != 1 ||
         quote.plan->replacement_binding.size() != 1)) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }

    auto node = pimpl->bindings.extract(binding_it);
    auto retained = quote.plan->detached_binding.insert(std::move(node));
    if (!retained.inserted) {
        const auto restored = pimpl->bindings.insert(std::move(retained.node));
        (void) restored;
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    if (quote.plan->preserve_empty_execution) {
        auto replacement_handle = quote.plan->replacement_handle.extract(quote.plan->replacement_handle_id);
        auto inserted_handle    = pimpl->handles.insert(std::move(replacement_handle));
        if (!inserted_handle.inserted) {
            const auto restored_binding =
                pimpl->bindings.insert(quote.plan->detached_binding.extract(quote.plan->execution_id));
            const auto restored_handle = quote.plan->replacement_handle.insert(std::move(inserted_handle.node));
            if (!restored_binding.inserted || !restored_handle.inserted) {
                std::terminate();
            }
            result.status = llama_dsv4_comp_status::stale_quote;
            return result;
        }
        auto replacement_binding = quote.plan->replacement_binding.extract(quote.plan->execution_id);
        auto inserted_binding    = pimpl->bindings.insert(std::move(replacement_binding));
        if (!inserted_binding.inserted) {
            const auto retained_handle =
                quote.plan->replacement_handle.insert(pimpl->handles.extract(quote.plan->replacement_handle_id));
            const auto restored_binding =
                pimpl->bindings.insert(quote.plan->detached_binding.extract(quote.plan->execution_id));
            const auto retained_binding = quote.plan->replacement_binding.insert(std::move(inserted_binding.node));
            if (!retained_handle.inserted || !restored_binding.inserted || !retained_binding.inserted) {
                std::terminate();
            }
            result.status = llama_dsv4_comp_status::stale_quote;
            return result;
        }
        pimpl->next_handle_id =
            quote.plan->replacement_handle_id == std::numeric_limits<llama_dsv4_comp_handle_id>::max() ?
                0 :
                quote.plan->replacement_handle_id + 1;
    }
    handle_it->second.resident_owned = true;
    ++handle_it->second.resident_generation;
    ++pimpl->epoch;
    quote.plan->commit_epoch = pimpl->epoch;
    quote.plan->state        = resident_detach_state::committed;
    result.status   = llama_dsv4_comp_status::ok;
    result.resident = {
        pimpl->identity->id,
        handle_it->first,
        handle_it->second.generation,
        handle_it->second.resident_generation,
    };
    return result;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::rollback_detach(
        const llama_dsv4_comp_detach_quote & quote) {
    if (!quote.plan || quote.plan->owner != pimpl->identity) {
        return llama_dsv4_comp_status::stale_quote;
    }
    if (quote.plan->state == resident_detach_state::rolled_back) {
        return llama_dsv4_comp_status::ok;
    }
    if (quote.plan->state == resident_detach_state::prepared) {
        quote.plan->state = resident_detach_state::rolled_back;
        return llama_dsv4_comp_status::ok;
    }
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    if (quote.plan->commit_epoch != pimpl->epoch) {
        return llama_dsv4_comp_status::stale_quote;
    }

    const auto handle_it = pimpl->handles.find(quote.plan->handle);
    const auto detached_binding_it    = quote.plan->detached_binding.find(quote.plan->execution_id);
    const auto replacement_binding_it = pimpl->bindings.find(quote.plan->execution_id);
    const auto replacement_handle_it  = pimpl->handles.find(quote.plan->replacement_handle_id);
    const bool replacement_valid =
        !quote.plan->preserve_empty_execution ||
        (replacement_binding_it != pimpl->bindings.end() &&
         replacement_binding_it->second == quote.plan->replacement_handle_id &&
         replacement_handle_it != pimpl->handles.end() && !replacement_handle_it->second.resident_owned &&
         replacement_handle_it->second.visible_c4_rows == 0 && replacement_handle_it->second.visible_hca_rows == 0 &&
         replacement_handle_it->second.c4_segment_ids.empty() && replacement_handle_it->second.hca_segment_ids.empty());
    if (handle_it == pimpl->handles.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != quote.plan->expected_handle_generation ||
        handle_it->second.resident_generation != quote.plan->expected_resident_generation + 1 ||
        detached_binding_it == quote.plan->detached_binding.end() ||
        detached_binding_it->second != quote.plan->handle || !replacement_valid ||
        (!quote.plan->preserve_empty_execution && pimpl->bindings.count(quote.plan->execution_id) != 0) ||
        pimpl->is_bound(quote.plan->handle)) {
        return llama_dsv4_comp_status::stale_quote;
    }

    if (quote.plan->preserve_empty_execution) {
        auto binding_node     = pimpl->bindings.extract(replacement_binding_it);
        auto handle_node      = pimpl->handles.extract(replacement_handle_it);
        auto retained_binding = quote.plan->replacement_binding.insert(std::move(binding_node));
        auto retained_handle  = quote.plan->replacement_handle.insert(std::move(handle_node));
        if (!retained_binding.inserted || !retained_handle.inserted) {
            std::terminate();
        }
    }

    auto node = quote.plan->detached_binding.extract(quote.plan->execution_id);
    if (node.empty()) {
        return llama_dsv4_comp_status::stale_quote;
    }
    auto inserted = pimpl->bindings.insert(std::move(node));
    if (!inserted.inserted) {
        const auto restored = quote.plan->detached_binding.insert(std::move(inserted.node));
        if (!restored.inserted) {
            std::terminate();
        }
        if (quote.plan->preserve_empty_execution) {
            const auto replacement_handle_restored =
                pimpl->handles.insert(quote.plan->replacement_handle.extract(quote.plan->replacement_handle_id));
            const auto replacement_binding_restored =
                pimpl->bindings.insert(quote.plan->replacement_binding.extract(quote.plan->execution_id));
            if (!replacement_handle_restored.inserted || !replacement_binding_restored.inserted) {
                std::terminate();
            }
        }
        return llama_dsv4_comp_status::stale_quote;
    }
    handle_it->second.resident_owned = false;
    ++pimpl->epoch;
    quote.plan->state = resident_detach_state::rolled_back;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_attach_quote llama_dsv4_comp_pool::quote_attach(
        llama_dsv4_comp_resident_handle resident,
        uint32_t                        execution_id) const {
    llama_dsv4_comp_attach_quote result;
    result.execution_id = execution_id;
    result.pool_epoch   = pimpl->epoch;
    result.resident     = resident;

    if (pimpl->busy()) {
        result.status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (execution_id >= LLAMA_DSV4_COMP_GRAPH_STREAMS || resident.pool_id == 0 || resident.id == 0) {
        result.status = llama_dsv4_comp_status::invalid_argument;
        return result;
    }
    if (resident.pool_id != pimpl->identity->id) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    const auto handle_it = pimpl->handles.find(resident.id);
    if (handle_it == pimpl->handles.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != resident.handle_generation ||
        handle_it->second.resident_generation != resident.lease_generation) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    if (pimpl->bindings.count(execution_id) != 0) {
        result.status = llama_dsv4_comp_status::slot_occupied;
        return result;
    }
    if (pimpl->is_bound(resident.id)) {
        result.status = llama_dsv4_comp_status::handle_bound;
        return result;
    }

    try {
        auto plan          = std::make_shared<llama_dsv4_comp_attach_plan>();
        plan->owner        = pimpl->identity;
        plan->pool_epoch   = pimpl->epoch;
        plan->execution_id = execution_id;
        plan->resident     = resident;
        plan->prepared_binding.emplace(execution_id, resident.id);
        result.status = llama_dsv4_comp_status::ok;
        result.plan   = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_comp_status::resource_exhausted;
        return result;
    }
}

llama_dsv4_comp_attach_quote llama_dsv4_comp_pool::quote_attach_replacing_empty(
        llama_dsv4_comp_resident_handle resident,
        uint32_t                        execution_id) const {
    llama_dsv4_comp_attach_quote result;
    result.execution_id = execution_id;
    result.pool_epoch   = pimpl->epoch;
    result.resident     = resident;

    if (pimpl->busy()) {
        result.status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (execution_id >= LLAMA_DSV4_COMP_GRAPH_STREAMS || resident.pool_id == 0 || resident.id == 0) {
        result.status = llama_dsv4_comp_status::invalid_argument;
        return result;
    }
    if (resident.pool_id != pimpl->identity->id) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    const auto handle_it = pimpl->handles.find(resident.id);
    if (handle_it == pimpl->handles.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != resident.handle_generation ||
        handle_it->second.resident_generation != resident.lease_generation || pimpl->is_bound(resident.id)) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    const auto binding_it = pimpl->bindings.find(execution_id);
    if (binding_it == pimpl->bindings.end()) {
        result.status = llama_dsv4_comp_status::binding_not_found;
        return result;
    }
    const auto empty_it = pimpl->handles.find(binding_it->second);
    if (empty_it == pimpl->handles.end() || empty_it->second.resident_owned ||
        empty_it->second.visible_c4_rows != 0 || empty_it->second.visible_hca_rows != 0 ||
        !empty_it->second.c4_segment_ids.empty() || !empty_it->second.hca_segment_ids.empty()) {
        result.status = llama_dsv4_comp_status::slot_occupied;
        return result;
    }

    try {
        auto plan                          = std::make_shared<llama_dsv4_comp_attach_plan>();
        plan->owner                        = pimpl->identity;
        plan->pool_epoch                   = pimpl->epoch;
        plan->execution_id                 = execution_id;
        plan->resident                     = resident;
        plan->replace_empty                = true;
        plan->expected_replaced_handle     = empty_it->first;
        plan->expected_replaced_generation = empty_it->second.generation;
        plan->prepared_binding.emplace(execution_id, resident.id);
        result.status = llama_dsv4_comp_status::ok;
        result.plan   = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_comp_status::resource_exhausted;
        return result;
    }
}

llama_dsv4_comp_status llama_dsv4_comp_pool::commit_attach(const llama_dsv4_comp_attach_quote & quote) {
    if (!quote.plan || quote.plan->owner != pimpl->identity ||
        quote.plan->state != resident_attach_state::prepared) {
        return llama_dsv4_comp_status::stale_quote;
    }
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    if (quote.plan->pool_epoch != pimpl->epoch) {
        return llama_dsv4_comp_status::stale_quote;
    }
    const auto handle_it = pimpl->handles.find(quote.plan->resident.id);
    if (handle_it == pimpl->handles.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != quote.plan->resident.handle_generation ||
        handle_it->second.resident_generation != quote.plan->resident.lease_generation ||
        pimpl->is_bound(quote.plan->resident.id)) {
        return llama_dsv4_comp_status::stale_quote;
    }

    if (quote.plan->replace_empty) {
        const auto binding_it = pimpl->bindings.find(quote.plan->execution_id);
        const auto empty_it = pimpl->handles.find(quote.plan->expected_replaced_handle);
        if (binding_it == pimpl->bindings.end() || binding_it->second != quote.plan->expected_replaced_handle ||
            empty_it == pimpl->handles.end() || empty_it->second.generation != quote.plan->expected_replaced_generation ||
            empty_it->second.resident_owned || empty_it->second.visible_c4_rows != 0 ||
            empty_it->second.visible_hca_rows != 0 || !empty_it->second.c4_segment_ids.empty() ||
            !empty_it->second.hca_segment_ids.empty()) {
            return llama_dsv4_comp_status::stale_quote;
        }
        auto binding_node = pimpl->bindings.extract(binding_it);
        auto handle_node  = pimpl->handles.extract(empty_it);
        auto kept_binding = quote.plan->replaced_binding.insert(std::move(binding_node));
        auto kept_handle  = quote.plan->replaced_handle.insert(std::move(handle_node));
        if (!kept_binding.inserted || !kept_handle.inserted) {
            if (kept_binding.inserted) {
                const auto restored = pimpl->bindings.insert(
                        quote.plan->replaced_binding.extract(quote.plan->execution_id));
                (void) restored;
            } else if (!kept_binding.node.empty()) {
                const auto restored = pimpl->bindings.insert(std::move(kept_binding.node));
                (void) restored;
            }
            if (kept_handle.inserted) {
                const auto restored = pimpl->handles.insert(
                        quote.plan->replaced_handle.extract(quote.plan->expected_replaced_handle));
                (void) restored;
            } else if (!kept_handle.node.empty()) {
                const auto restored = pimpl->handles.insert(std::move(kept_handle.node));
                (void) restored;
            }
            return llama_dsv4_comp_status::stale_quote;
        }
    } else if (pimpl->bindings.count(quote.plan->execution_id) != 0) {
        return llama_dsv4_comp_status::stale_quote;
    }

    auto node = quote.plan->prepared_binding.extract(quote.plan->execution_id);
    if (node.empty()) {
        return llama_dsv4_comp_status::stale_quote;
    }
    auto inserted = pimpl->bindings.insert(std::move(node));
    if (!inserted.inserted) {
        const auto restored = quote.plan->prepared_binding.insert(std::move(inserted.node));
        (void) restored;
        if (quote.plan->replace_empty) {
            const auto binding_restored = pimpl->bindings.insert(
                    quote.plan->replaced_binding.extract(quote.plan->execution_id));
            const auto handle_restored = pimpl->handles.insert(
                    quote.plan->replaced_handle.extract(quote.plan->expected_replaced_handle));
            (void) binding_restored;
            (void) handle_restored;
        }
        return llama_dsv4_comp_status::stale_quote;
    }
    handle_it->second.resident_owned = false;
    ++pimpl->epoch;
    quote.plan->commit_epoch = pimpl->epoch;
    quote.plan->state        = resident_attach_state::committed;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::rollback_attach(const llama_dsv4_comp_attach_quote & quote) {
    if (!quote.plan || quote.plan->owner != pimpl->identity) {
        return llama_dsv4_comp_status::stale_quote;
    }
    if (quote.plan->state == resident_attach_state::rolled_back) {
        return llama_dsv4_comp_status::ok;
    }
    if (quote.plan->state == resident_attach_state::prepared) {
        quote.plan->state = resident_attach_state::rolled_back;
        return llama_dsv4_comp_status::ok;
    }
    if (pimpl->busy()) {
        return llama_dsv4_comp_status::busy;
    }
    if (quote.plan->commit_epoch != pimpl->epoch) {
        return llama_dsv4_comp_status::stale_quote;
    }

    const auto binding_it = pimpl->bindings.find(quote.plan->execution_id);
    const auto handle_it  = pimpl->handles.find(quote.plan->resident.id);
    if (binding_it == pimpl->bindings.end() || binding_it->second != quote.plan->resident.id ||
        handle_it == pimpl->handles.end() || handle_it->second.resident_owned ||
        handle_it->second.generation != quote.plan->resident.handle_generation ||
        handle_it->second.resident_generation != quote.plan->resident.lease_generation) {
        return llama_dsv4_comp_status::stale_quote;
    }

    auto node = pimpl->bindings.extract(binding_it);
    auto restored = quote.plan->prepared_binding.insert(std::move(node));
    if (!restored.inserted) {
        const auto reinserted = pimpl->bindings.insert(std::move(restored.node));
        (void) reinserted;
        return llama_dsv4_comp_status::stale_quote;
    }
    handle_it->second.resident_owned = true;
    if (quote.plan->replace_empty) {
        auto binding_restored = pimpl->bindings.insert(
                quote.plan->replaced_binding.extract(quote.plan->execution_id));
        auto handle_restored = pimpl->handles.insert(
                quote.plan->replaced_handle.extract(quote.plan->expected_replaced_handle));
        if (!binding_restored.inserted || !handle_restored.inserted) {
            std::terminate();
        }
    }
    ++pimpl->epoch;
    quote.plan->state = resident_attach_state::rolled_back;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::attach(
        llama_dsv4_comp_resident_handle resident,
        uint32_t                        execution_id) {
    const auto quote = quote_attach(resident, execution_id);
    if (quote.status != llama_dsv4_comp_status::ok) {
        return quote.status;
    }
    return commit_attach(quote);
}

llama_dsv4_comp_status llama_dsv4_comp_pool::release(llama_dsv4_comp_resident_handle resident) {
    const auto quote = prepare_release(resident);
    if (quote.status != llama_dsv4_comp_status::ok) {
        return quote.status;
    }
    return commit_release(quote);
}

llama_dsv4_comp_release_quote llama_dsv4_comp_pool::prepare_release(
        llama_dsv4_comp_resident_handle resident) {
    llama_dsv4_comp_release_quote result;
    result.resident = resident;
    if (pimpl->busy()) {
        result.status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (resident.pool_id == 0 || resident.id == 0) {
        result.status = llama_dsv4_comp_status::invalid_argument;
        return result;
    }
    if (resident.pool_id != pimpl->identity->id) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    const auto handle_it = pimpl->handles.find(resident.id);
    if (handle_it == pimpl->handles.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != resident.handle_generation ||
        handle_it->second.resident_generation != resident.lease_generation) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }
    if (pimpl->is_bound(resident.id)) {
        result.status = llama_dsv4_comp_status::handle_bound;
        return result;
    }
    if (!pimpl->releasable(handle_it->second.c4_segment_ids, llama_dsv4_comp_family::c4) ||
        !pimpl->releasable(handle_it->second.hca_segment_ids, llama_dsv4_comp_family::hca)) {
        result.status = llama_dsv4_comp_status::stale_handle;
        return result;
    }

    try {
        if (pimpl->next_release_id == 0) {
            result.status = llama_dsv4_comp_status::generation_exhausted;
            return result;
        }
        auto plan        = std::make_shared<llama_dsv4_comp_release_plan>();
        plan->owner      = pimpl->identity;
        plan->release_id = pimpl->next_release_id;
        plan->resident   = resident;
        pimpl->next_release_id = plan->release_id == std::numeric_limits<uint64_t>::max() ?
                0 : plan->release_id + 1;

        auto node = pimpl->handles.extract(handle_it);
        auto retained = plan->retained_handle.insert(std::move(node));
        if (!retained.inserted) {
            const auto restored = pimpl->handles.insert(std::move(retained.node));
            (void) restored;
            result.status = llama_dsv4_comp_status::stale_handle;
            return result;
        }
        pimpl->active_release_id = plan->release_id;
        ++pimpl->epoch;
        result.status = llama_dsv4_comp_status::ok;
        result.plan   = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_comp_status::resource_exhausted;
        return result;
    }
}

llama_dsv4_comp_status llama_dsv4_comp_pool::commit_release(
        const llama_dsv4_comp_release_quote & quote) {
    if (!quote.plan || quote.plan->owner != pimpl->identity ||
        quote.plan->state != resident_release_state::prepared ||
        quote.plan->release_id == 0 || quote.plan->release_id != pimpl->active_release_id) {
        return llama_dsv4_comp_status::stale_quote;
    }
    const auto handle_it = quote.plan->retained_handle.find(quote.plan->resident.id);
    if (handle_it == quote.plan->retained_handle.end() || !handle_it->second.resident_owned ||
        handle_it->second.generation != quote.plan->resident.handle_generation ||
        handle_it->second.resident_generation != quote.plan->resident.lease_generation ||
        !pimpl->releasable(handle_it->second.c4_segment_ids, llama_dsv4_comp_family::c4) ||
        !pimpl->releasable(handle_it->second.hca_segment_ids, llama_dsv4_comp_family::hca)) {
        return llama_dsv4_comp_status::stale_quote;
    }

    pimpl->release(handle_it->second.c4_segment_ids, llama_dsv4_comp_family::c4);
    pimpl->release(handle_it->second.hca_segment_ids, llama_dsv4_comp_family::hca);
    quote.plan->retained_handle.erase(handle_it);
    pimpl->active_release_id = 0;
    ++pimpl->epoch;
    quote.plan->state = resident_release_state::committed;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::rollback_release(
        const llama_dsv4_comp_release_quote & quote) {
    if (!quote.plan || quote.plan->owner != pimpl->identity) {
        return llama_dsv4_comp_status::stale_quote;
    }
    if (quote.plan->state == resident_release_state::rolled_back) {
        return llama_dsv4_comp_status::ok;
    }
    if (quote.plan->state != resident_release_state::prepared ||
        quote.plan->release_id == 0 || quote.plan->release_id != pimpl->active_release_id) {
        return llama_dsv4_comp_status::stale_quote;
    }
    auto node = quote.plan->retained_handle.extract(quote.plan->resident.id);
    if (node.empty()) {
        return llama_dsv4_comp_status::stale_quote;
    }
    auto restored = pimpl->handles.insert(std::move(node));
    if (!restored.inserted) {
        const auto retained = quote.plan->retained_handle.insert(std::move(restored.node));
        (void) retained;
        return llama_dsv4_comp_status::stale_quote;
    }
    pimpl->active_release_id = 0;
    ++pimpl->epoch;
    quote.plan->state = resident_release_state::rolled_back;
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_quote llama_dsv4_comp_pool::quote_batch(const llama_dsv4_comp_batch_plan & batch) const {
    llama_dsv4_comp_quote result;
    auto                  plan = std::make_shared<llama_dsv4_comp_quote_plan>();
    plan->owner                = pimpl->identity;
    plan->pool_epoch           = pimpl->epoch;
    result.pool_epoch          = pimpl->epoch;
    result.plan                = plan;

    if (pimpl->busy() || batch.graph_execution_ids.size() > LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        result.status = pimpl->busy() ? llama_dsv4_comp_status::busy : llama_dsv4_comp_status::invalid_argument;
        plan->status  = result.status;
        return result;
    }

    std::set<uint32_t> graph_ids;
    for (uint32_t execution_id : batch.graph_execution_ids) {
        if (execution_id >= LLAMA_DSV4_COMP_GRAPH_STREAMS || !graph_ids.insert(execution_id).second ||
            pimpl->bindings.count(execution_id) == 0) {
            result.status = execution_id < LLAMA_DSV4_COMP_GRAPH_STREAMS && pimpl->bindings.count(execution_id) == 0 ?
                                llama_dsv4_comp_status::binding_not_found :
                                llama_dsv4_comp_status::invalid_argument;
            plan->status  = result.status;
            return result;
        }
    }
    plan->graph_execution_ids = batch.graph_execution_ids;
    result.scratch_rows       = static_cast<uint32_t>(batch.graph_execution_ids.size());

    std::vector<llama_dsv4_comp_change> changes = batch.changes;
    std::sort(changes.begin(), changes.end(), [](const auto & lhs, const auto & rhs) {
        if (lhs.handle != rhs.handle) {
            return lhs.handle < rhs.handle;
        }
        return static_cast<uint8_t>(lhs.family) < static_cast<uint8_t>(rhs.family);
    });

    for (size_t index = 0; index < changes.size(); ++index) {
        const auto & change = changes[index];
        if (!is_valid_family(change.family) ||
            (index > 0 && changes[index - 1].handle == change.handle && changes[index - 1].family == change.family)) {
            result.status = llama_dsv4_comp_status::invalid_argument;
            plan->status  = result.status;
            return result;
        }
        const auto handle_it = pimpl->handles.find(change.handle);
        if (handle_it == pimpl->handles.end()) {
            result.status = llama_dsv4_comp_status::handle_not_found;
            plan->status  = result.status;
            return result;
        }
        if (handle_it->second.resident_owned) {
            result.status = llama_dsv4_comp_status::handle_resident;
            plan->status  = result.status;
            return result;
        }

        planned_candidate candidate;
        candidate.handle                     = change.handle;
        candidate.family                     = change.family;
        candidate.expected_handle_generation = handle_it->second.generation;
        candidate.old_visible_rows           = visible_rows_for(handle_it->second, change.family);
        candidate.new_visible_rows           = change.new_visible_rows;
        candidate.old_segment_ids            = segment_ids_for(handle_it->second, change.family);
        candidate.candidate_segment_ids      = candidate.old_segment_ids;

        if (candidate.new_visible_rows < candidate.old_visible_rows) {
            result.status = llama_dsv4_comp_status::invalid_argument;
            plan->status  = result.status;
            return result;
        }
        for (uint64_t row : change.overwrite_rows) {
            if (row >= candidate.old_visible_rows ||
                llama_dsv4_comp_logical_segment(row) > std::numeric_limits<uint32_t>::max()) {
                result.status = llama_dsv4_comp_status::invalid_argument;
                plan->status  = result.status;
                return result;
            }
            candidate.cow_logical_segments.insert(static_cast<uint32_t>(llama_dsv4_comp_logical_segment(row)));
        }

        const uint64_t required_segments = llama_dsv4_comp_segments_for_rows(candidate.new_visible_rows);
        if (required_segments > std::numeric_limits<uint32_t>::max()) {
            result.status          = llama_dsv4_comp_status::capacity_exhausted;
            result.limiting_family = change.family;
            plan->status           = result.status;
            return result;
        }
        if (required_segments > pimpl->family(change.family).data_segments()) {
            result.status          = llama_dsv4_comp_status::capacity_exhausted;
            result.limiting_family = change.family;
            plan->status           = result.status;
            return result;
        }
        if (candidate.old_visible_rows < candidate.new_visible_rows &&
            candidate.old_visible_rows % LLAMA_DSV4_COMP_SEGMENT_ROWS != 0 && !candidate.old_segment_ids.empty()) {
            const uint32_t tail     = static_cast<uint32_t>(candidate.old_segment_ids.size() - 1);
            const uint32_t physical = candidate.old_segment_ids[tail];
            if (pimpl->family(change.family).segments.at(physical).refs > 1) {
                candidate.cow_logical_segments.insert(tail);
            }
        }
        candidate.candidate_segment_ids.resize(static_cast<size_t>(required_segments), LLAMA_DSV4_COMP_INVALID_SEGMENT);
        plan->candidates.push_back(std::move(candidate));
    }

    std::vector<uint32_t> c4_free      = free_segment_ids(pimpl->c4);
    std::vector<uint32_t> hca_free     = free_segment_ids(pimpl->hca);
    size_t                c4_cursor    = 0;
    size_t                hca_cursor   = 0;
    std::vector<bool>     lid_occupied = c4_lid_occupied_groups(pimpl->c4);

    for (planned_candidate & candidate : plan->candidates) {
        auto &   free_ids = candidate.family == llama_dsv4_comp_family::c4 ? c4_free : hca_free;
        size_t & cursor   = candidate.family == llama_dsv4_comp_family::c4 ? c4_cursor : hca_cursor;
        for (uint32_t logical = 0; logical < candidate.candidate_segment_ids.size(); ++logical) {
            const bool appended_segment = logical >= candidate.old_segment_ids.size();
            const bool cow              = candidate.cow_logical_segments.count(logical) != 0;
            if (!appended_segment && !cow) {
                continue;
            }
            if (cursor >= free_ids.size()) {
                result.status          = llama_dsv4_comp_status::capacity_exhausted;
                result.limiting_family = candidate.family;
                plan->status           = result.status;
                return result;
            }

            const uint32_t destination = free_ids[cursor++];
            const uint32_t source      = cow ? candidate.old_segment_ids[logical] : LLAMA_DSV4_COMP_INVALID_SEGMENT;
            candidate.candidate_segment_ids[logical] = destination;

            planned_reservation reservation;
            reservation.allocation.family              = candidate.family;
            reservation.allocation.handle              = candidate.handle;
            reservation.allocation.logical_segment     = logical;
            reservation.allocation.source_segment      = source;
            reservation.allocation.destination_segment = destination;
            reservation.allocation.populated_rows =
                cow ? populated_rows_in_segment(candidate.old_visible_rows, logical) : 0;
            reservation.allocation.cow              = cow;
            reservation.expected_segment_generation = pimpl->family(candidate.family).segments[destination].generation;
            plan->reservations.push_back(reservation);
            result.allocations.push_back(reservation.allocation);

            llama_dsv4_comp_family_quote & family_quote =
                candidate.family == llama_dsv4_comp_family::c4 ? result.c4 : result.hca;
            ++family_quote.new_segments;
            family_quote.cow_segments += cow ? 1 : 0;
            family_quote.segment_pages += candidate.family == llama_dsv4_comp_family::c4 ? LLAMA_DSV4_COMP_CSA_LAYERS :
                                                                                           LLAMA_DSV4_COMP_HCA_LAYERS;
            if (candidate.family == llama_dsv4_comp_family::c4) {
                const size_t group = destination / 4;
                if (!lid_occupied[group]) {
                    family_quote.lid_pages += LLAMA_DSV4_COMP_LID_LAYERS;
                    lid_occupied[group] = true;
                }
            }
        }
    }

    result.status = llama_dsv4_comp_status::ok;
    plan->status  = result.status;
    return result;
}

llama_dsv4_comp_reserve_result llama_dsv4_comp_pool::try_reserve(const llama_dsv4_comp_quote & quote) {
    llama_dsv4_comp_reserve_result result;
    if (!quote.plan || quote.plan->owner != pimpl->identity) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    if (quote.plan->status != llama_dsv4_comp_status::ok) {
        result.status = quote.plan->status;
        return result;
    }
    if (pimpl->busy()) {
        result.status = llama_dsv4_comp_status::busy;
        return result;
    }
    if (quote.plan->pool_epoch != pimpl->epoch) {
        result.status = llama_dsv4_comp_status::stale_quote;
        return result;
    }
    for (const planned_candidate & candidate : quote.plan->candidates) {
        const auto handle_it = pimpl->handles.find(candidate.handle);
        if (handle_it == pimpl->handles.end() || handle_it->second.generation != candidate.expected_handle_generation) {
            result.status = llama_dsv4_comp_status::stale_quote;
            return result;
        }
    }
    for (const planned_reservation & reservation : quote.plan->reservations) {
        const segment_record & segment =
            pimpl->family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
        if (segment.state != segment_state::free || segment.generation != reservation.expected_segment_generation) {
            result.status = llama_dsv4_comp_status::stale_quote;
            return result;
        }
    }

    const uint64_t ticket_id         = pimpl->next_ticket_id++;
    const uint64_t ticket_generation = pimpl->next_ticket_generation++;
    for (const planned_reservation & reservation : quote.plan->reservations) {
        segment_record & segment =
            pimpl->family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
        segment.state        = segment_state::reserved;
        segment.owner_ticket = ticket_id;
        segment.cow          = false;
        ++segment.generation;
    }
    pimpl->tickets.emplace(ticket_id, impl::ticket_record{
                                          ticket_generation,
                                          ticket_state::active,
                                          quote.plan,
                                      });
    pimpl->active_ticket_id    = ticket_id;
    pimpl->scratch_rows_in_use = static_cast<uint32_t>(quote.plan->graph_execution_ids.size());
    ++pimpl->epoch;
    result.status = llama_dsv4_comp_status::ok;
    result.ticket = { ticket_id, ticket_generation };
    return result;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::commit(llama_dsv4_comp_ticket ticket) {
    auto * record = pimpl->find_ticket(ticket);
    if (record == nullptr) {
        return llama_dsv4_comp_status::stale_ticket;
    }
    if (record->state == ticket_state::committed) {
        return llama_dsv4_comp_status::ok;
    }
    if (record->state != ticket_state::active || pimpl->active_ticket_id != ticket.id) {
        return llama_dsv4_comp_status::stale_ticket;
    }

    for (const planned_reservation & reservation : record->plan->reservations) {
        const segment_record & segment =
            pimpl->family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
        if (segment.state != segment_state::reserved || segment.owner_ticket != ticket.id ||
            segment.generation != reservation.expected_segment_generation + 1) {
            return llama_dsv4_comp_status::stale_ticket;
        }
    }
    for (const planned_candidate & candidate : record->plan->candidates) {
        const auto handle_it = pimpl->handles.find(candidate.handle);
        if (handle_it == pimpl->handles.end() || handle_it->second.generation != candidate.expected_handle_generation) {
            return llama_dsv4_comp_status::stale_ticket;
        }
    }

    for (const planned_reservation & reservation : record->plan->reservations) {
        segment_record & segment =
            pimpl->family(reservation.allocation.family).segments.at(reservation.allocation.destination_segment);
        segment.state        = segment_state::mapped;
        segment.owner_ticket = 0;
        segment.cow          = reservation.allocation.cow;
    }
    for (const planned_candidate & candidate : record->plan->candidates) {
        pimpl->retain(candidate.candidate_segment_ids, candidate.family);
    }
    for (const planned_candidate & candidate : record->plan->candidates) {
        pimpl->release(candidate.old_segment_ids, candidate.family);
    }

    std::set<llama_dsv4_comp_handle_id> changed_handles;
    for (const planned_candidate & candidate : record->plan->candidates) {
        resident_handle & handle = pimpl->handles.at(candidate.handle);
        if (candidate.family == llama_dsv4_comp_family::c4) {
            handle.c4_segment_ids  = candidate.candidate_segment_ids;
            handle.visible_c4_rows = candidate.new_visible_rows;
        } else {
            handle.hca_segment_ids  = candidate.candidate_segment_ids;
            handle.visible_hca_rows = candidate.new_visible_rows;
        }
        changed_handles.insert(candidate.handle);
    }
    for (llama_dsv4_comp_handle_id handle : changed_handles) {
        ++pimpl->handles.at(handle).generation;
    }

    pimpl->scratch_rows_in_use = 0;
    pimpl->active_ticket_id    = 0;
    record->state              = ticket_state::committed;
    record->plan.reset();
    ++pimpl->epoch;
    pimpl->prune_terminal_tickets();
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_status llama_dsv4_comp_pool::rollback(llama_dsv4_comp_ticket ticket) {
    return pimpl->terminal_release(ticket, ticket_state::rolled_back);
}

llama_dsv4_comp_status llama_dsv4_comp_pool::cancel(llama_dsv4_comp_ticket ticket) {
    return pimpl->terminal_release(ticket, ticket_state::cancelled);
}

llama_dsv4_comp_status llama_dsv4_comp_pool::candidate_handle(llama_dsv4_comp_ticket        ticket,
                                                              llama_dsv4_comp_handle_id     handle,
                                                              llama_dsv4_comp_handle_info & result) const {
    const auto * record = pimpl->find_ticket(ticket);
    if (record == nullptr || record->state != ticket_state::active || pimpl->active_ticket_id != ticket.id) {
        return llama_dsv4_comp_status::stale_ticket;
    }
    const auto handle_it = pimpl->handles.find(handle);
    if (handle_it == pimpl->handles.end()) {
        return llama_dsv4_comp_status::handle_not_found;
    }
    resident_handle candidate = handle_it->second;
    bool            changed   = false;
    for (const planned_candidate & item : record->plan->candidates) {
        if (item.handle != handle) {
            continue;
        }
        changed = true;
        if (item.family == llama_dsv4_comp_family::c4) {
            candidate.c4_segment_ids  = item.candidate_segment_ids;
            candidate.visible_c4_rows = item.new_visible_rows;
        } else {
            candidate.hca_segment_ids  = item.candidate_segment_ids;
            candidate.visible_hca_rows = item.new_visible_rows;
        }
    }
    candidate.generation += changed ? 1 : 0;
    export_handle(candidate, result);
    return llama_dsv4_comp_status::ok;
}

llama_dsv4_comp_directory llama_dsv4_comp_pool::make_directory(llama_dsv4_comp_family             family,
                                                               const std::vector<uint32_t> &      execution_ids,
                                                               uint32_t                           logical_segments,
                                                               const llama_dsv4_comp_quote_plan * plan) const {
    const impl &              state = *pimpl;
    llama_dsv4_comp_directory result;
    if (!is_valid_family(family)) {
        result.status = llama_dsv4_comp_status::invalid_argument;
        return result;
    }
    result.logical_segments = logical_segments;
    result.graph_streams    = static_cast<uint32_t>(execution_ids.size());
    result.segment_ids.assign(static_cast<size_t>(logical_segments) * execution_ids.size(), ZERO_SEGMENT);

    for (size_t stream = 0; stream < execution_ids.size(); ++stream) {
        const auto binding_it = state.bindings.find(execution_ids[stream]);
        if (binding_it == state.bindings.end()) {
            result.status = llama_dsv4_comp_status::binding_not_found;
            result.segment_ids.clear();
            return result;
        }
        const resident_handle &       handle = state.handles.at(binding_it->second);
        const std::vector<uint32_t> * ids    = &segment_ids_for(handle, family);
        if (plan != nullptr) {
            const planned_candidate * candidate = state.find_candidate(*plan, handle.id, family);
            if (candidate != nullptr) {
                ids = &candidate->candidate_segment_ids;
            }
        }
        const size_t count = std::min<size_t>(logical_segments, ids->size());
        std::copy_n(ids->begin(), count, result.segment_ids.begin() + stream * logical_segments);
    }
    result.status = llama_dsv4_comp_status::ok;
    return result;
}

llama_dsv4_comp_directory llama_dsv4_comp_pool::directory_for_bindings(llama_dsv4_comp_family        family,
                                                                       const std::vector<uint32_t> & execution_ids,
                                                                       uint32_t logical_segments) const {
    if (execution_ids.size() > LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        return {};
    }
    return make_directory(family, execution_ids, logical_segments, nullptr);
}

llama_dsv4_comp_directory llama_dsv4_comp_pool::ticket_directory(llama_dsv4_comp_ticket ticket,
                                                                 llama_dsv4_comp_family family,
                                                                 uint32_t               logical_segments) const {
    const auto * record = pimpl->find_ticket(ticket);
    if (record == nullptr || record->state != ticket_state::active || pimpl->active_ticket_id != ticket.id) {
        llama_dsv4_comp_directory result;
        result.status = llama_dsv4_comp_status::stale_ticket;
        return result;
    }
    return make_directory(family, record->plan->graph_execution_ids, logical_segments, record->plan.get());
}

llama_dsv4_comp_directory llama_dsv4_comp_pool::ticket_directory_for(
        llama_dsv4_comp_ticket        ticket,
        llama_dsv4_comp_family        family,
        const std::vector<uint32_t> & execution_ids,
        uint32_t                      logical_segments) const {
    const auto * record = pimpl->find_ticket(ticket);
    if (record == nullptr || record->state != ticket_state::active || pimpl->active_ticket_id != ticket.id ||
        execution_ids.size() > LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        llama_dsv4_comp_directory result;
        result.status = llama_dsv4_comp_status::stale_ticket;
        return result;
    }
    return make_directory(family, execution_ids, logical_segments, record->plan.get());
}

uint32_t llama_dsv4_comp_pool::zero_segment(llama_dsv4_comp_family family) const {
    return is_valid_family(family) ? ZERO_SEGMENT : LLAMA_DSV4_COMP_INVALID_SEGMENT;
}

uint32_t llama_dsv4_comp_pool::scratch_segment(llama_dsv4_comp_family family) const {
    return is_valid_family(family) ? SCRATCH_SEGMENT : LLAMA_DSV4_COMP_INVALID_SEGMENT;
}

uint64_t llama_dsv4_comp_pool::zero_physical_row(llama_dsv4_comp_family family, uint64_t logical_row) const {
    if (!is_valid_family(family)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return llama_dsv4_comp_physical_row(ZERO_SEGMENT, logical_row);
}

uint64_t llama_dsv4_comp_pool::scratch_physical_row(llama_dsv4_comp_family family, uint32_t graph_stream) const {
    if (!is_valid_family(family) || graph_stream >= LLAMA_DSV4_COMP_GRAPH_STREAMS) {
        return std::numeric_limits<uint64_t>::max();
    }
    return llama_dsv4_comp_physical_row(SCRATCH_SEGMENT, graph_stream);
}

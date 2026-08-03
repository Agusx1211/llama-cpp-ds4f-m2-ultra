#include "llama-dsv4-comp-pool.h"

#include <algorithm>
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

struct pool_identity {};

enum class ticket_state : uint8_t {
    active,
    committed,
    rolled_back,
    cancelled,
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

    bool busy() const { return active_ticket_id != 0; }

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
        case llama_dsv4_comp_status::binding_not_found:
            return "binding_not_found";
        case llama_dsv4_comp_status::capacity_exhausted:
            return "capacity_exhausted";
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

llama_dsv4_comp_pool::~llama_dsv4_comp_pool()                                            = default;
llama_dsv4_comp_pool::llama_dsv4_comp_pool(llama_dsv4_comp_pool &&) noexcept             = default;
llama_dsv4_comp_pool & llama_dsv4_comp_pool::operator=(llama_dsv4_comp_pool &&) noexcept = default;

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
    result.active_tickets          = pimpl->active_ticket_id == 0 ? 0 : 1;
    result.retained_ticket_records = static_cast<uint32_t>(pimpl->tickets.size());
    return result;
}

llama_dsv4_comp_handle_result llama_dsv4_comp_pool::create_handle() {
    if (pimpl->busy()) {
        return { llama_dsv4_comp_status::busy, 0 };
    }
    resident_handle handle;
    handle.id = pimpl->next_handle_id++;
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
    resident_handle copy = source_it->second;
    copy.id              = pimpl->next_handle_id++;
    copy.generation      = 1;
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
    if (std::any_of(pimpl->bindings.begin(), pimpl->bindings.end(),
                    [&](const auto & binding) { return binding.second == handle; })) {
        return llama_dsv4_comp_status::handle_bound;
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
    if (pimpl->handles.count(handle) == 0) {
        return llama_dsv4_comp_status::handle_not_found;
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

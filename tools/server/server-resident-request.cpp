#include "server-resident-request.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace server_resident_request {

namespace {

bool valid_request(request_handle value) {
    return value.id != 0 && value.epoch != 0;
}

bool same_request(request_handle lhs, request_handle rhs) {
    return lhs.id == rhs.id && lhs.epoch == rhs.epoch;
}

bool request_less(request_handle lhs, request_handle rhs) {
    return lhs.id < rhs.id || (lhs.id == rhs.id && lhs.epoch < rhs.epoch);
}

}  // namespace

bool execution_lease::operator==(const execution_lease & other) const {
    return same_request(request, other.request) && slot == other.slot && generation == other.generation;
}

bool resident_lease::operator==(const resident_lease & other) const {
    return same_request(request, other.request) && slot == other.slot && generation == other.generation &&
           backend_handle == other.backend_handle;
}

bool transaction_lease::operator==(const transaction_lease & other) const {
    return same_request(parent, other.parent) && kind == other.kind && generation == other.generation;
}

bool member_snapshot::operator==(const member_snapshot & other) const {
    return same_request(request, other.request) && same_request(parent, other.parent) && state == other.state &&
           revision == other.revision && execution == other.execution && resident == other.resident;
}

bool family_snapshot::operator==(const family_snapshot & other) const {
    return same_request(parent, other.parent) && transaction == other.transaction && members == other.members;
}

bool registry_summary::operator==(const registry_summary & other) const {
    return families == other.families && requests == other.requests && bound_requests == other.bound_requests &&
           resident_requests == other.resident_requests && transitioning_requests == other.transitioning_requests &&
           terminal_requests == other.terminal_requests && occupied_execution_slots == other.occupied_execution_slots &&
           occupied_resident_slots == other.occupied_resident_slots;
}

bool registry_snapshot::operator==(const registry_snapshot & other) const {
    return families == other.families && summary == other.summary;
}

operation_result::operator bool() const {
    return code == result_code::ok;
}

registration_result::operator bool() const {
    return code == result_code::ok;
}

transaction_result::operator bool() const {
    return code == result_code::ok;
}

transition_result::operator bool() const {
    return code == result_code::ok;
}

struct registry::impl {
    struct execution_slot {
        uint64_t                       generation = 0;
        std::optional<execution_lease> lease;
    };

    struct resident_slot {
        uint64_t                      generation = 0;
        std::optional<resident_lease> lease;
    };

    using family_map = std::map<uint64_t, family_snapshot>;

    explicit impl(registry_config value) :
        config(value),
        executions(value.max_execution_slots),
        residents(value.max_resident_slots) {
        if (config.max_requests == 0 || config.max_family_members == 0 ||
            config.max_family_members > config.max_requests || config.max_execution_slots == 0 ||
            config.max_execution_slots >= no_execution_slot || config.max_resident_slots == 0 ||
            config.max_resident_slots >= no_resident_slot) {
            throw std::invalid_argument("invalid resident request registry capacity");
        }
    }

    result_code find_parent(request_handle parent, family_map::iterator & it) {
        if (!valid_request(parent)) {
            return result_code::stale_request;
        }

        it = families.find(parent.id);
        if (it != families.end()) {
            return same_request(it->second.parent, parent) ? result_code::ok : result_code::stale_request;
        }

        for (auto family = families.begin(); family != families.end(); ++family) {
            for (const member_snapshot & member : family->second.members) {
                if (member.request.id == parent.id) {
                    return same_request(member.request, parent) ? result_code::not_family_parent :
                                                                  result_code::stale_request;
                }
            }
        }
        return result_code::unknown_request;
    }

    result_code find_parent(request_handle parent, family_map::const_iterator & it) const {
        if (!valid_request(parent)) {
            return result_code::stale_request;
        }

        it = families.find(parent.id);
        if (it != families.end()) {
            return same_request(it->second.parent, parent) ? result_code::ok : result_code::stale_request;
        }

        for (auto family = families.cbegin(); family != families.cend(); ++family) {
            for (const member_snapshot & member : family->second.members) {
                if (member.request.id == parent.id) {
                    return same_request(member.request, parent) ? result_code::not_family_parent :
                                                                  result_code::stale_request;
                }
            }
        }
        return result_code::unknown_request;
    }

    result_code find_member(request_handle                           request,
                            family_map::iterator &                   family,
                            std::vector<member_snapshot>::iterator & member) {
        if (!valid_request(request)) {
            return result_code::stale_request;
        }
        for (family = families.begin(); family != families.end(); ++family) {
            for (member = family->second.members.begin(); member != family->second.members.end(); ++member) {
                if (member->request.id == request.id) {
                    return same_request(member->request, request) ? result_code::ok : result_code::stale_request;
                }
            }
        }
        return result_code::unknown_request;
    }

    result_code validate_transaction(const transaction_lease & transaction,
                                     transition_kind           expected,
                                     family_map::iterator &    family) {
        const result_code found = find_parent(transaction.parent, family);
        if (found != result_code::ok) {
            return found;
        }
        if (transaction.kind != expected || transaction.generation == 0 || !family->second.transaction ||
            !(*family->second.transaction == transaction)) {
            return result_code::stale_transaction;
        }
        return result_code::ok;
    }

    bool request_id_in_use(uint64_t id) const {
        for (const auto & family : families) {
            for (const member_snapshot & member : family.second.members) {
                if (member.request.id == id) {
                    return true;
                }
            }
        }
        return false;
    }

    static member_snapshot * find_member_in_family(family_snapshot & family, request_handle request) {
        for (member_snapshot & member : family.members) {
            if (same_request(member.request, request)) {
                return &member;
            }
        }
        return nullptr;
    }

    static const resident_commit * find_commit(const std::vector<resident_commit> & commits, request_handle request) {
        const resident_commit * found = nullptr;
        for (const resident_commit & commit : commits) {
            if (same_request(commit.request, request)) {
                if (found != nullptr) {
                    return nullptr;
                }
                found = &commit;
            }
        }
        return found;
    }

    static const restore_target * find_target(const std::vector<restore_target> & targets, request_handle request) {
        const restore_target * found = nullptr;
        for (const restore_target & target : targets) {
            if (same_request(target.request, request)) {
                if (found != nullptr) {
                    return nullptr;
                }
                found = &target;
            }
        }
        return found;
    }

    registry_summary make_summary() const {
        registry_summary result;
        result.families                 = families.size();
        result.requests                 = request_count;
        result.occupied_execution_slots = occupied_execution_slots;
        result.occupied_resident_slots  = occupied_resident_slots;
        for (const auto & family : families) {
            for (const member_snapshot & member : family.second.members) {
                switch (member.state) {
                    case lifecycle::bound:
                        ++result.bound_requests;
                        break;
                    case lifecycle::resident:
                        ++result.resident_requests;
                        break;
                    case lifecycle::suspending:
                    case lifecycle::restoring:
                        ++result.transitioning_requests;
                        break;
                    case lifecycle::terminal:
                        ++result.terminal_requests;
                        break;
                }
            }
        }
        return result;
    }

    bool is_drained() const {
        if (!families.empty() || request_count != 0 || occupied_execution_slots != 0 || occupied_resident_slots != 0) {
            return false;
        }
        const bool execution_owned = std::any_of(executions.begin(), executions.end(),
                                                 [](const execution_slot & slot) { return slot.lease.has_value(); });
        const bool resident_owned  = std::any_of(residents.begin(), residents.end(),
                                                 [](const resident_slot & slot) { return slot.lease.has_value(); });
        return !execution_owned && !resident_owned;
    }

    registry_config             config;
    mutable std::mutex          mutex;
    family_map                  families;
    std::vector<execution_slot> executions;
    std::vector<resident_slot>  residents;
    uint64_t                    next_transaction_generation = 1;
    size_t                      request_count               = 0;
    size_t                      occupied_execution_slots    = 0;
    size_t                      occupied_resident_slots     = 0;
};

registry::registry(registry_config config) : pimpl(new impl(config)) {}

registry::~registry() {
    if (!drained()) {
        std::terminate();
    }
}

registration_result registry::register_bound_family(request_handle                                 parent,
                                                    const std::vector<bound_member_registration> & members) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    if (!valid_request(parent) || members.empty() || members.size() > pimpl->config.max_family_members) {
        return { result_code::invalid_family, {} };
    }
    if (pimpl->request_count + members.size() > pimpl->config.max_requests) {
        return { result_code::request_capacity_exhausted, {} };
    }

    size_t parent_count = 0;
    for (size_t i = 0; i < members.size(); ++i) {
        const bound_member_registration & member = members[i];
        if (!valid_request(member.request)) {
            return { result_code::invalid_family, {} };
        }
        if (same_request(member.request, parent)) {
            ++parent_count;
        }
        if (member.slot >= pimpl->executions.size()) {
            return { result_code::execution_slot_out_of_range, {} };
        }
        if (pimpl->executions[member.slot].lease) {
            return { result_code::execution_slot_occupied, {} };
        }
        if (pimpl->executions[member.slot].generation == std::numeric_limits<uint64_t>::max()) {
            return { result_code::generation_exhausted, {} };
        }
        if (pimpl->request_id_in_use(member.request.id)) {
            return { result_code::duplicate_request, {} };
        }
        for (size_t prior = 0; prior < i; ++prior) {
            if (members[prior].request.id == member.request.id || members[prior].slot == member.slot) {
                return { result_code::invalid_family, {} };
            }
        }
    }
    if (parent_count != 1) {
        return { result_code::invalid_family, {} };
    }

    family_snapshot family;
    family.parent = parent;
    family.members.reserve(members.size());
    for (const bound_member_registration & registration : members) {
        member_snapshot member;
        member.request   = registration.request;
        member.parent    = parent;
        member.state     = lifecycle::bound;
        member.revision  = 1;
        member.execution = execution_lease{
            registration.request,
            registration.slot,
            pimpl->executions[registration.slot].generation + 1,
        };
        family.members.push_back(member);
    }
    std::sort(family.members.begin(), family.members.end(),
              [](const member_snapshot & lhs, const member_snapshot & rhs) {
                  return request_less(lhs.request, rhs.request);
              });

    std::vector<execution_lease> leases;
    leases.reserve(family.members.size());
    for (const member_snapshot & member : family.members) {
        leases.push_back(*member.execution);
    }

    const auto inserted = pimpl->families.emplace(parent.id, std::move(family));
    if (!inserted.second) {
        return { result_code::duplicate_request, {} };
    }
    for (const execution_lease & lease : leases) {
        impl::execution_slot & slot = pimpl->executions[lease.slot];
        slot.generation             = lease.generation;
        slot.lease                  = lease;
    }
    pimpl->request_count += leases.size();
    pimpl->occupied_execution_slots += leases.size();
    return { result_code::ok, std::move(leases) };
}

transaction_result registry::begin_suspend(request_handle parent) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           found = pimpl->find_parent(parent, family);
    if (found != result_code::ok) {
        return { found, {}, {}, {} };
    }
    if (family->second.transaction) {
        return { result_code::active_transaction, {}, {}, {} };
    }
    if (pimpl->next_transaction_generation == 0) {
        return { result_code::generation_exhausted, {}, {}, {} };
    }
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::bound || !member.execution || member.resident) {
            return { result_code::invalid_transition, {}, {}, {} };
        }
    }

    transaction_result result;
    result.transaction = { parent, transition_kind::suspend, pimpl->next_transaction_generation };
    result.executions.reserve(family->second.members.size());
    for (const member_snapshot & member : family->second.members) {
        result.executions.push_back(*member.execution);
    }

    ++pimpl->next_transaction_generation;
    family->second.transaction = result.transaction;
    for (member_snapshot & member : family->second.members) {
        member.state = lifecycle::suspending;
        ++member.revision;
    }
    return result;
}

transition_result registry::commit_suspend(const transaction_lease &            transaction,
                                           const std::vector<resident_commit> & commits) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           validated = pimpl->validate_transaction(transaction, transition_kind::suspend, family);
    if (validated != result_code::ok) {
        return { validated, {}, {} };
    }
    if (commits.size() != family->second.members.size()) {
        return { result_code::incomplete_family, {}, {} };
    }
    for (const resident_commit & commit : commits) {
        if (!valid_request(commit.request) || commit.backend_handle == 0 ||
            impl::find_member_in_family(family->second, commit.request) == nullptr) {
            return { result_code::incomplete_family, {}, {} };
        }
    }

    std::vector<resident_slot_id> free_slots;
    bool                          exhausted_free_slot = false;
    free_slots.reserve(family->second.members.size());
    for (resident_slot_id slot = 0; slot < pimpl->residents.size() && free_slots.size() < commits.size(); ++slot) {
        if (!pimpl->residents[slot].lease) {
            if (pimpl->residents[slot].generation == std::numeric_limits<uint64_t>::max()) {
                exhausted_free_slot = true;
                continue;
            }
            free_slots.push_back(slot);
        }
    }
    if (free_slots.size() != commits.size()) {
        return { free_slots.empty() && exhausted_free_slot ? result_code::generation_exhausted :
                                                             result_code::resident_capacity_exhausted,
                 {},
                 {} };
    }

    transition_result result;
    result.residents.reserve(family->second.members.size());
    for (size_t index = 0; index < family->second.members.size(); ++index) {
        const member_snapshot & member = family->second.members[index];
        const resident_commit * commit = impl::find_commit(commits, member.request);
        if (commit == nullptr) {
            return { result_code::incomplete_family, {}, {} };
        }
        const resident_slot_id slot = free_slots[index];
        result.residents.push_back({
            member.request,
            slot,
            pimpl->residents[slot].generation + 1,
            commit->backend_handle,
        });
    }

    for (size_t index = 0; index < family->second.members.size(); ++index) {
        member_snapshot &      member    = family->second.members[index];
        impl::execution_slot & execution = pimpl->executions[member.execution->slot];
        execution.lease.reset();
        --pimpl->occupied_execution_slots;

        const resident_lease & lease    = result.residents[index];
        impl::resident_slot &  resident = pimpl->residents[lease.slot];
        resident.generation             = lease.generation;
        resident.lease                  = lease;
        ++pimpl->occupied_resident_slots;

        member.execution.reset();
        member.resident = lease;
        member.state    = lifecycle::resident;
        ++member.revision;
    }
    family->second.transaction.reset();
    return result;
}

operation_result registry::rollback_suspend(const transaction_lease & transaction) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           validated = pimpl->validate_transaction(transaction, transition_kind::suspend, family);
    if (validated != result_code::ok) {
        return { validated, false };
    }
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::suspending || !member.execution || member.resident) {
            return { result_code::invalid_transition, false };
        }
    }
    for (member_snapshot & member : family->second.members) {
        member.state = lifecycle::bound;
        --member.revision;
    }
    family->second.transaction.reset();
    return { result_code::ok, true };
}

transaction_result registry::begin_restore(request_handle parent, const std::vector<restore_target> & targets) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           found = pimpl->find_parent(parent, family);
    if (found != result_code::ok) {
        return { found, {}, {}, {} };
    }
    if (family->second.transaction) {
        return { result_code::active_transaction, {}, {}, {} };
    }
    if (pimpl->next_transaction_generation == 0) {
        return { result_code::generation_exhausted, {}, {}, {} };
    }
    if (targets.size() != family->second.members.size()) {
        return { result_code::incomplete_family, {}, {}, {} };
    }
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::resident || member.execution || !member.resident) {
            return { result_code::invalid_transition, {}, {}, {} };
        }
    }
    for (size_t index = 0; index < targets.size(); ++index) {
        const restore_target & target = targets[index];
        if (!valid_request(target.request) || impl::find_member_in_family(family->second, target.request) == nullptr) {
            return { result_code::incomplete_family, {}, {}, {} };
        }
        if (target.slot >= pimpl->executions.size()) {
            return { result_code::execution_slot_out_of_range, {}, {}, {} };
        }
        if (pimpl->executions[target.slot].lease) {
            return { result_code::execution_slot_occupied, {}, {}, {} };
        }
        if (pimpl->executions[target.slot].generation == std::numeric_limits<uint64_t>::max()) {
            return { result_code::generation_exhausted, {}, {}, {} };
        }
        for (size_t prior = 0; prior < index; ++prior) {
            if (same_request(targets[prior].request, target.request) || targets[prior].slot == target.slot) {
                return { result_code::incomplete_family, {}, {}, {} };
            }
        }
    }

    transaction_result result;
    result.transaction = { parent, transition_kind::restore, pimpl->next_transaction_generation };
    result.executions.reserve(family->second.members.size());
    result.residents.reserve(family->second.members.size());
    for (const member_snapshot & member : family->second.members) {
        const restore_target * target = impl::find_target(targets, member.request);
        if (target == nullptr) {
            return { result_code::incomplete_family, {}, {}, {} };
        }
        result.executions.push_back({
            member.request,
            target->slot,
            pimpl->executions[target->slot].generation + 1,
        });
        result.residents.push_back(*member.resident);
    }

    ++pimpl->next_transaction_generation;
    family->second.transaction = result.transaction;
    for (size_t index = 0; index < family->second.members.size(); ++index) {
        member_snapshot &       member = family->second.members[index];
        const execution_lease & lease  = result.executions[index];
        impl::execution_slot &  slot   = pimpl->executions[lease.slot];
        slot.generation                = lease.generation;
        slot.lease                     = lease;
        ++pimpl->occupied_execution_slots;

        member.execution = lease;
        member.state     = lifecycle::restoring;
        ++member.revision;
    }
    return result;
}

transition_result registry::commit_restore(const transaction_lease & transaction) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           validated = pimpl->validate_transaction(transaction, transition_kind::restore, family);
    if (validated != result_code::ok) {
        return { validated, {}, {} };
    }
    transition_result result;
    result.executions.reserve(family->second.members.size());
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::restoring || !member.execution || !member.resident) {
            return { result_code::invalid_transition, {}, {} };
        }
        result.executions.push_back(*member.execution);
    }

    for (member_snapshot & member : family->second.members) {
        impl::resident_slot & slot = pimpl->residents[member.resident->slot];
        slot.lease.reset();
        --pimpl->occupied_resident_slots;
        member.resident.reset();
        member.state = lifecycle::bound;
        ++member.revision;
    }
    family->second.transaction.reset();
    return result;
}

operation_result registry::rollback_restore(const transaction_lease & transaction) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           validated = pimpl->validate_transaction(transaction, transition_kind::restore, family);
    if (validated != result_code::ok) {
        return { validated, false };
    }
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::restoring || !member.execution || !member.resident) {
            return { result_code::invalid_transition, false };
        }
    }
    for (member_snapshot & member : family->second.members) {
        impl::execution_slot & slot = pimpl->executions[member.execution->slot];
        slot.lease.reset();
        --pimpl->occupied_execution_slots;
        member.execution.reset();
        member.state = lifecycle::resident;
        --member.revision;
    }
    family->second.transaction.reset();
    return { result_code::ok, true };
}

operation_result registry::mark_terminal_family(request_handle parent) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           found = pimpl->find_parent(parent, family);
    if (found != result_code::ok) {
        return { found, false };
    }
    if (family->second.transaction) {
        return { result_code::active_transaction, false };
    }

    bool changed = false;
    for (const member_snapshot & member : family->second.members) {
        if (member.state == lifecycle::suspending || member.state == lifecycle::restoring) {
            return { result_code::invalid_transition, false };
        }
        changed = changed || member.state != lifecycle::terminal;
    }
    if (changed) {
        for (member_snapshot & member : family->second.members) {
            member.state = lifecycle::terminal;
            ++member.revision;
        }
    }
    return { result_code::ok, changed };
}

operation_result registry::confirm_terminal_execution_release(const execution_lease & lease) {
    std::lock_guard<std::mutex>            lock(pimpl->mutex);
    impl::family_map::iterator             family;
    std::vector<member_snapshot>::iterator member;
    const result_code                      found = pimpl->find_member(lease.request, family, member);
    if (found != result_code::ok) {
        return { found, false };
    }
    if (member->state != lifecycle::terminal) {
        return { result_code::not_terminal, false };
    }
    if (!member->execution || !(*member->execution == lease) || lease.slot >= pimpl->executions.size() ||
        !pimpl->executions[lease.slot].lease || !(*pimpl->executions[lease.slot].lease == lease)) {
        return { result_code::stale_execution_lease, false };
    }
    pimpl->executions[lease.slot].lease.reset();
    --pimpl->occupied_execution_slots;
    member->execution.reset();
    ++member->revision;
    return { result_code::ok, true };
}

operation_result registry::confirm_terminal_resident_release(const resident_lease & lease) {
    std::lock_guard<std::mutex>            lock(pimpl->mutex);
    impl::family_map::iterator             family;
    std::vector<member_snapshot>::iterator member;
    const result_code                      found = pimpl->find_member(lease.request, family, member);
    if (found != result_code::ok) {
        return { found, false };
    }
    if (member->state != lifecycle::terminal) {
        return { result_code::not_terminal, false };
    }
    if (!member->resident || !(*member->resident == lease) || lease.slot >= pimpl->residents.size() ||
        !pimpl->residents[lease.slot].lease || !(*pimpl->residents[lease.slot].lease == lease)) {
        return { result_code::stale_resident_lease, false };
    }
    pimpl->residents[lease.slot].lease.reset();
    --pimpl->occupied_resident_slots;
    member->resident.reset();
    ++member->revision;
    return { result_code::ok, true };
}

operation_result registry::remove_terminal_family(request_handle parent) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    impl::family_map::iterator  family;
    const result_code           found = pimpl->find_parent(parent, family);
    if (found != result_code::ok) {
        return { found, false };
    }
    if (family->second.transaction) {
        return { result_code::active_transaction, false };
    }
    for (const member_snapshot & member : family->second.members) {
        if (member.state != lifecycle::terminal) {
            return { result_code::not_terminal, false };
        }
        if (member.execution || member.resident) {
            return { result_code::resources_still_owned, false };
        }
    }
    pimpl->request_count -= family->second.members.size();
    pimpl->families.erase(family);
    return { result_code::ok, true };
}

std::optional<family_snapshot> registry::get_family(request_handle parent) const {
    std::lock_guard<std::mutex>      lock(pimpl->mutex);
    impl::family_map::const_iterator family;
    if (pimpl->find_parent(parent, family) != result_code::ok) {
        return std::nullopt;
    }
    return family->second;
}

registry_snapshot registry::snapshot() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    registry_snapshot           result;
    result.families.reserve(pimpl->families.size());
    for (const auto & family : pimpl->families) {
        result.families.push_back(family.second);
    }
    result.summary = pimpl->make_summary();
    return result;
}

registry_summary registry::summary() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->make_summary();
}

bool registry::drained() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->is_drained();
}

#ifdef SERVER_RESIDENT_REQUEST_TESTING
void registry::test_override_resident_generation(resident_slot_id slot, uint64_t generation) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (slot >= pimpl->residents.size() || pimpl->residents[slot].lease) {
        throw std::invalid_argument("resident generation seam requires a free in-range slot");
    }
    pimpl->residents[slot].generation = generation;
}
#endif

}  // namespace server_resident_request

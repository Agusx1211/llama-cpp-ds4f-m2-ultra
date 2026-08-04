#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-resident-request.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace server_resident_request;

namespace {

static_assert(!std::is_move_constructible<registry>::value, "registry ownership must not be movable");
static_assert(!std::is_move_assignable<registry>::value, "registry ownership must not be move-assignable");

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

request_handle handle(uint64_t id, uint64_t epoch) {
    return { id, epoch };
}

bool same_request(request_handle lhs, request_handle rhs) {
    return lhs.id == rhs.id && lhs.epoch == rhs.epoch;
}

family_snapshot get_required(const registry & owner, request_handle parent) {
    const auto snapshot = owner.get_family(parent);
    require(snapshot.has_value(), "family snapshot exists");
    return *snapshot;
}

member_snapshot find_required(const family_snapshot & family, request_handle request) {
    for (const member_snapshot & member : family.members) {
        if (same_request(member.request, request)) {
            return member;
        }
    }
    throw std::runtime_error("family member exists");
}

void simulate_external_release_and_remove(registry & owner, request_handle parent) {
    require(owner.mark_terminal_family(parent), "mark test family terminal");
    const family_snapshot terminal = get_required(owner, parent);
    for (const member_snapshot & member : terminal.members) {
        if (member.execution) {
            require(owner.confirm_terminal_execution_release(*member.execution), "confirm test execution release");
        }
        if (member.resident) {
            require(owner.confirm_terminal_resident_release(*member.resident), "confirm test resident release");
        }
    }
    require(owner.remove_terminal_family(parent), "remove drained test family");
}

registration_result add_singleton(registry & owner, request_handle request, execution_slot_id slot) {
    return owner.register_bound_family(request, {
                                                    { request, slot }
    });
}

transition_result suspend_singleton(registry & owner, request_handle request, backend_handle_id backend_handle) {
    const transaction_result begin = owner.begin_suspend(request);
    require(begin, "begin singleton suspend");
    return owner.commit_suspend(begin.transaction, {
                                                       { request, backend_handle }
    });
}

void test_singleton_suspend_restore_identity() {
    registry                  owner;
    const request_handle      request      = handle(101, 7);
    const registration_result registration = add_singleton(owner, request, 2);
    require(registration && registration.executions.size() == 1, "register bound singleton");
    const execution_lease original_execution = registration.executions[0];

    const transaction_result suspending = owner.begin_suspend(request);
    require(suspending && suspending.executions.size() == 1 && suspending.executions[0] == original_execution,
            "suspend retains exact execution lease");
    const member_snapshot during_suspend = find_required(get_required(owner, request), request);
    require(during_suspend.state == lifecycle::suspending && during_suspend.execution == original_execution &&
                !during_suspend.resident,
            "suspending state owns only execution");

    const transition_result resident = owner.commit_suspend(suspending.transaction, {
                                                                                        { request, 9001 }
    });
    require(resident && resident.residents.size() == 1, "commit singleton suspend");
    const resident_lease  resident_ownership = resident.residents[0];
    const member_snapshot parked             = find_required(get_required(owner, request), request);
    require(same_request(parked.request, request) && parked.state == lifecycle::resident && !parked.execution &&
                parked.resident == resident_ownership && parked.resident->backend_handle == 9001,
            "resident state preserves stable identity and opaque backend ownership");

    const transaction_result restoring = owner.begin_restore(request, {
                                                                          { request, 5 }
    });
    require(restoring && restoring.executions.size() == 1 && restoring.residents.size() == 1 &&
                restoring.residents[0] == resident_ownership,
            "restore retains resident ownership while reserving execution");
    const member_snapshot during_restore = find_required(get_required(owner, request), request);
    require(during_restore.state == lifecycle::restoring && during_restore.execution == restoring.executions[0] &&
                during_restore.resident == resident_ownership,
            "restoring state owns both sides of the transaction");

    const transition_result rebound = owner.commit_restore(restoring.transaction);
    require(rebound && rebound.executions.size() == 1 && rebound.executions[0].slot == 5, "commit singleton restore");
    const member_snapshot bound = find_required(get_required(owner, request), request);
    require(bound.state == lifecycle::bound && bound.execution == rebound.executions[0] && !bound.resident,
            "restored state owns only the new execution");

    const registry_summary summary = owner.summary();
    require(summary.families == 1 && summary.requests == 1 && summary.bound_requests == 1 &&
                summary.occupied_execution_slots == 1 && summary.occupied_resident_slots == 0,
            "singleton ownership summary");
    simulate_external_release_and_remove(owner, request);
    require(owner.drained(), "singleton registry drains before destruction");
}

void test_rollback_preserves_exact_prior_owner() {
    registry                  owner;
    const request_handle      request      = handle(11, 3);
    const registration_result registration = add_singleton(owner, request, 1);
    require(registration, "register rollback singleton");
    const execution_lease initial        = registration.executions[0];
    const family_snapshot before_suspend = get_required(owner, request);

    const transaction_result first = owner.begin_suspend(request);
    require(first, "begin failed-detach transaction");
    require(owner.commit_suspend(first.transaction, {}).code == result_code::incomplete_family,
            "incomplete detach cannot commit");
    require(find_required(get_required(owner, request), request).state == lifecycle::suspending,
            "failed commit leaves explicit transaction active");
    require(owner.rollback_suspend(first.transaction), "rollback failed detach");
    require(get_required(owner, request) == before_suspend, "suspend rollback restores the full prior snapshot");
    const member_snapshot rebound = find_required(get_required(owner, request), request);
    require(rebound.state == lifecycle::bound && rebound.execution == initial && !rebound.resident,
            "detach rollback restores exact original execution");
    require(owner.rollback_suspend(first.transaction).code == result_code::stale_transaction,
            "completed suspend transaction cannot be replayed");

    const transaction_result second = owner.begin_suspend(request);
    require(second && second.transaction.generation > first.transaction.generation, "transaction generations advance");
    const transition_result parked = owner.commit_suspend(second.transaction, {
                                                                                  { request, 55 }
    });
    require(parked, "park for restore rollback");
    const resident_lease exact_resident = parked.residents[0];

    const transaction_result restore = owner.begin_restore(request, {
                                                                        { request, 3 }
    });
    require(restore, "begin failed-restore transaction");
    require(owner.mark_terminal_family(request).code == result_code::active_transaction,
            "terminal transition cannot race restore");
    require(owner.rollback_restore(restore.transaction), "rollback failed restore");
    const member_snapshot resident = find_required(get_required(owner, request), request);
    require(resident.state == lifecycle::resident && !resident.execution && resident.resident == exact_resident,
            "restore rollback preserves exact resident lease");

    const request_handle      other  = handle(12, 1);
    const registration_result reused = add_singleton(owner, other, 3);
    require(reused && reused.executions[0].generation > restore.executions[0].generation,
            "rolled-back execution reservation is reusable with a new generation");
    simulate_external_release_and_remove(owner, request);
    simulate_external_release_and_remove(owner, other);
    require(owner.drained(), "rollback registry drains before destruction");
}

void test_family_transactions_are_atomic() {
    registry                  owner;
    const request_handle      parent       = handle(30, 2);
    const request_handle      child_a      = handle(31, 4);
    const request_handle      child_b      = handle(32, 6);
    const registration_result registration = owner.register_bound_family(parent, {
                                                                                     { child_b, 3 },
                                                                                     { parent,  1 },
                                                                                     { child_a, 2 },
    });
    require(registration && registration.executions.size() == 3, "register parent and children atomically");
    require(registration.executions[0].request.id == 30 && registration.executions[1].request.id == 31 &&
                registration.executions[2].request.id == 32,
            "family registration result is deterministic");
    require(owner.begin_suspend(child_a).code == result_code::not_family_parent,
            "child-only suspension is explicitly rejected");

    const transaction_result suspend = owner.begin_suspend(parent);
    require(suspend && suspend.executions.size() == 3, "begin whole-family suspend");
    const family_snapshot suspending = get_required(owner, parent);
    for (const member_snapshot & member : suspending.members) {
        require(member.state == lifecycle::suspending && member.execution && !member.resident,
                "every family member enters suspending together");
    }

    require(owner.commit_suspend(suspend.transaction,
                                 {
                                     { parent,  100 },
                                     { child_a, 101 },
                                     { child_a, 102 }
    })
                    .code == result_code::incomplete_family,
            "duplicate child cannot partially commit a family");
    const family_snapshot unchanged = get_required(owner, parent);
    require(unchanged == suspending, "failed family commit leaves byte-equivalent ownership snapshot");
    require(owner.rollback_suspend(suspend.transaction), "whole-family detach rollback");

    const transaction_result retry = owner.begin_suspend(parent);
    require(retry, "retry whole-family suspend");
    const transition_result resident = owner.commit_suspend(retry.transaction, {
                                                                                   { child_b, 202 },
                                                                                   { parent,  200 },
                                                                                   { child_a, 201 },
    });
    require(resident && resident.residents.size() == 3, "commit whole family to resident ownership");
    for (const member_snapshot & member : get_required(owner, parent).members) {
        require(member.state == lifecycle::resident && !member.execution && member.resident,
                "every family member becomes resident together");
    }
    const family_snapshot fully_resident = get_required(owner, parent);

    require(owner.begin_restore(parent,
                                {
                                    { parent,  4 },
                                    { child_a, 5 }
    })
                    .code == result_code::incomplete_family,
            "incomplete restore target set rejected before mutation");
    require(owner.begin_restore(parent,
                                {
                                    { parent,  4 },
                                    { child_a, 5 },
                                    { child_b, 5 }
    })
                    .code == result_code::incomplete_family,
            "duplicate family restore slot rejected before mutation");

    const std::vector<restore_target> family_targets = {
        { child_b, 6 },
        { child_a, 5 },
        { parent,  4 },
    };
    const transaction_result rolled_back_restore = owner.begin_restore(parent, family_targets);
    require(rolled_back_restore, "begin whole-family rollback restore");
    require(owner.rollback_restore(rolled_back_restore.transaction), "rollback whole-family restore");
    require(get_required(owner, parent) == fully_resident,
            "whole-family restore rollback returns the full resident snapshot");

    const transaction_result restoring = owner.begin_restore(parent, family_targets);
    require(restoring && restoring.executions.size() == 3 && restoring.residents == resident.residents,
            "begin whole-family restore retains every resident lease");
    for (const member_snapshot & member : get_required(owner, parent).members) {
        require(member.state == lifecycle::restoring && member.execution && member.resident,
                "every family member enters restoring together");
    }
    const transition_result restored = owner.commit_restore(restoring.transaction);
    require(restored && restored.executions.size() == 3, "commit whole-family restore");
    for (const member_snapshot & member : get_required(owner, parent).members) {
        require(member.state == lifecycle::bound && member.execution && !member.resident,
                "every family member becomes bound together");
    }
    simulate_external_release_and_remove(owner, parent);
    require(owner.drained(), "family registry drains before destruction");
}

void test_terminal_cleanup_and_stale_generations() {
    registry                  owner;
    const request_handle      first                = handle(41, 9);
    const registration_result initial_registration = add_singleton(owner, first, 0);
    require(initial_registration, "register terminal resident request");
    const execution_lease   first_execution = initial_registration.executions[0];
    const transition_result first_parked    = suspend_singleton(owner, first, 501);
    require(first_parked, "create first resident ownership");
    const resident_lease     first_resident = first_parked.residents[0];
    const transaction_result first_restore  = owner.begin_restore(first, {
                                                                            { first, 0 }
    });
    require(first_restore && owner.commit_restore(first_restore.transaction), "restore first resident ownership");
    const transition_result second_parked = suspend_singleton(owner, first, 502);
    require(second_parked, "create second resident ownership");
    const resident_lease second_resident = second_parked.residents[0];
    require(second_resident.slot == first_resident.slot && second_resident.generation > first_resident.generation,
            "resident slot reuse advances generation");

    require(owner.mark_terminal_family(first), "mark resident family terminal");
    require(owner.remove_terminal_family(first).code == result_code::resources_still_owned,
            "terminal record remains while resident resource is owned");
    require(owner.confirm_terminal_resident_release(first_resident).code == result_code::stale_resident_lease,
            "old resident generation cannot release reused ownership");
    require(owner.confirm_terminal_resident_release(second_resident), "confirm exact resident release");
    require(owner.confirm_terminal_resident_release(second_resident).code == result_code::stale_resident_lease,
            "resident release can be confirmed exactly once");
    require(owner.remove_terminal_family(first), "remove terminal family after cleanup");

    const request_handle      second  = handle(first.id, first.epoch + 1);
    const registration_result rebound = add_singleton(owner, second, 0);
    require(rebound && rebound.executions[0].generation > first_execution.generation,
            "execution generation survives request incarnation reuse");
    require(owner.begin_suspend(first).code == result_code::stale_request,
            "old request epoch cannot mutate new incarnation");
    require(owner.mark_terminal_family(second), "mark bound family terminal");
    const execution_lease second_execution = rebound.executions[0];
    require(owner.remove_terminal_family(second).code == result_code::resources_still_owned,
            "terminal record remains while execution resource is owned");
    require(owner.confirm_terminal_execution_release(second_execution), "confirm exact execution release");
    require(owner.confirm_terminal_execution_release(second_execution).code == result_code::stale_execution_lease,
            "execution release can be confirmed exactly once");
    require(owner.remove_terminal_family(second), "remove cleaned bound family");
    require(owner.summary().requests == 0, "terminal cleanup leaves no request metadata");
    require(owner.drained(), "terminal cleanup satisfies drained invariant");
}

void test_capacity_and_validation_fail_closed() {
    bool threw = false;
    try {
        registry_config invalid;
        invalid.max_resident_slots = 0;
        registry rejected(invalid);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    require(threw, "zero resident capacity rejected");

    registry_config config;
    config.max_requests        = 3;
    config.max_family_members  = 2;
    config.max_execution_slots = 3;
    config.max_resident_slots  = 1;
    registry owner(config);

    const request_handle parent = handle(1, 1);
    const request_handle child  = handle(2, 1);
    require(owner.register_bound_family(parent,
                                        {
                                            { parent, 0 },
                                            { child, 1 },
                                            { handle(3, 1), 2 }
    })
                    .code == result_code::invalid_family,
            "family member metadata is bounded");
    const registration_result family = owner.register_bound_family(parent, {
                                                                               { child,  1 },
                                                                               { parent, 0 }
    });
    require(family, "register capacity family");
    require(add_singleton(owner, handle(4, 1), 1).code == result_code::execution_slot_occupied,
            "occupied execution slot rejected without mutation");
    require(add_singleton(owner, handle(3, 1), 2), "fill request capacity");
    require(add_singleton(owner, handle(4, 1), 2).code == result_code::request_capacity_exhausted,
            "request metadata capacity is bounded");

    const transaction_result suspend = owner.begin_suspend(parent);
    require(suspend, "begin capacity suspend");
    require(owner.commit_suspend(suspend.transaction,
                                 {
                                     { parent, 71 },
                                     { child,  72 }
    })
                    .code == result_code::resident_capacity_exhausted,
            "resident ownership capacity is bounded");
    require(owner.rollback_suspend(suspend.transaction), "capacity failure rolls back explicitly");
    const registry_summary summary = owner.summary();
    require(summary.requests == 3 && summary.bound_requests == 3 && summary.occupied_execution_slots == 3 &&
                summary.occupied_resident_slots == 0,
            "capacity failure preserves exact ownership accounting");
    simulate_external_release_and_remove(owner, parent);
    simulate_external_release_and_remove(owner, handle(3, 1));
    require(owner.drained(), "capacity registry drains before destruction");
}

void test_resident_allocator_skips_exhausted_free_slots() {
    registry_config config;
    config.max_requests        = 1;
    config.max_family_members  = 1;
    config.max_execution_slots = 1;
    config.max_resident_slots  = 2;

    const request_handle request = handle(60, 1);
    registry             owner(config);
    owner.test_override_resident_generation(0, UINT64_MAX);
    require(add_singleton(owner, request, 0), "register generation seam request");
    const transaction_result suspend = owner.begin_suspend(request);
    require(suspend, "begin generation seam suspend");
    const transition_result resident = owner.commit_suspend(suspend.transaction, {
                                                                                     { request, 600 }
    });
    require(resident && resident.residents.size() == 1 && resident.residents[0].slot == 1,
            "allocator skips exhausted free slot and uses later slot");
    simulate_external_release_and_remove(owner, request);
    require(owner.drained(), "usable generation seam registry drains");

    registry exhausted(config);
    exhausted.test_override_resident_generation(0, UINT64_MAX);
    exhausted.test_override_resident_generation(1, UINT64_MAX);
    require(add_singleton(exhausted, request, 0), "register fully exhausted request");
    const transaction_result exhausted_suspend = exhausted.begin_suspend(request);
    require(exhausted_suspend, "begin fully exhausted suspend");
    require(exhausted
                    .commit_suspend(exhausted_suspend.transaction,
                                    {
                                        { request, 601 }
    })
                    .code == result_code::generation_exhausted,
            "generation exhaustion reported only when no usable free slot remains");
    require(exhausted.rollback_suspend(exhausted_suspend.transaction), "rollback fully exhausted suspend");
    simulate_external_release_and_remove(exhausted, request);
    require(exhausted.drained(), "fully exhausted registry drains");
}

void test_wrong_kind_illegal_states_and_preterminal_guards() {
    registry                  owner;
    const request_handle      request      = handle(70, 1);
    const registration_result registration = add_singleton(owner, request, 0);
    require(registration, "register transition guard request");
    const execution_lease initial = registration.executions[0];
    require(owner.begin_restore(request,
                                {
                                    { request, 1 }
    })
                    .code == result_code::invalid_transition,
            "bound request cannot begin restore");
    require(owner.confirm_terminal_execution_release(initial).code == result_code::not_terminal,
            "execution release cannot be confirmed before terminal state");
    require(owner.remove_terminal_family(request).code == result_code::not_terminal, "bound family cannot be removed");

    const family_snapshot    bound   = get_required(owner, request);
    const transaction_result suspend = owner.begin_suspend(request);
    require(suspend, "begin guarded suspend");
    require(owner.begin_suspend(request).code == result_code::active_transaction,
            "second suspend cannot overlap active suspend");
    require(owner.begin_restore(request,
                                {
                                    { request, 1 }
    })
                    .code == result_code::active_transaction,
            "restore cannot overlap active suspend");
    const family_snapshot suspending = get_required(owner, request);
    require(owner.commit_restore(suspend.transaction).code == result_code::stale_transaction,
            "suspend transaction cannot commit as restore");
    require(owner.rollback_restore(suspend.transaction).code == result_code::stale_transaction,
            "suspend transaction cannot roll back as restore");
    require(get_required(owner, request) == suspending, "wrong-kind suspend operations are mutation-free");
    require(owner.rollback_suspend(suspend.transaction), "roll back guarded suspend");
    require(get_required(owner, request) == bound, "guarded suspend rollback restores full snapshot");

    const transition_result resident = suspend_singleton(owner, request, 701);
    require(resident, "create guarded resident state");
    const family_snapshot before_restore = get_required(owner, request);
    require(owner.begin_suspend(request).code == result_code::invalid_transition,
            "resident request cannot begin suspend");
    require(owner.confirm_terminal_resident_release(resident.residents[0]).code == result_code::not_terminal,
            "resident release cannot be confirmed before terminal state");
    require(owner.remove_terminal_family(request).code == result_code::not_terminal,
            "resident family cannot be removed before terminal state");

    const transaction_result restore = owner.begin_restore(request, {
                                                                        { request, 1 }
    });
    require(restore, "begin guarded restore");
    require(owner.begin_suspend(request).code == result_code::active_transaction,
            "suspend cannot overlap active restore");
    require(owner.begin_restore(request,
                                {
                                    { request, 2 }
    })
                    .code == result_code::active_transaction,
            "second restore cannot overlap active restore");
    const family_snapshot restoring = get_required(owner, request);
    require(owner.commit_suspend(restore.transaction,
                                 {
                                     { request, 702 }
    })
                    .code == result_code::stale_transaction,
            "restore transaction cannot commit as suspend");
    require(owner.rollback_suspend(restore.transaction).code == result_code::stale_transaction,
            "restore transaction cannot roll back as suspend");
    require(get_required(owner, request) == restoring, "wrong-kind restore operations are mutation-free");
    require(owner.rollback_restore(restore.transaction), "roll back guarded restore");
    require(get_required(owner, request) == before_restore, "guarded restore rollback restores full snapshot");

    require(owner.mark_terminal_family(request), "terminalize guarded resident request");
    require(owner.begin_suspend(request).code == result_code::invalid_transition,
            "terminal request cannot begin suspend");
    require(owner.begin_restore(request,
                                {
                                    { request, 2 }
    })
                    .code == result_code::invalid_transition,
            "terminal request cannot begin restore");
    require(owner.confirm_terminal_resident_release(resident.residents[0]), "release guarded resident ownership");
    require(owner.remove_terminal_family(request), "remove guarded terminal request");
    require(owner.drained(), "transition guard registry drains");
}

void test_stale_tokens_cannot_cross_request_epochs() {
    registry                  owner;
    const request_handle      old_request      = handle(75, 4);
    const registration_result old_registration = add_singleton(owner, old_request, 0);
    require(old_registration, "register old request epoch");
    const execution_lease   old_execution       = old_registration.executions[0];
    const transition_result old_resident_result = suspend_singleton(owner, old_request, 750);
    require(old_resident_result, "create old resident lease");
    const resident_lease     old_resident = old_resident_result.residents[0];
    const transaction_result old_restore  = owner.begin_restore(old_request, {
                                                                                { old_request, 0 }
    });
    require(old_restore && owner.commit_restore(old_restore.transaction), "restore old request epoch");
    const execution_lease    current_old_execution = old_restore.executions[0];
    const transaction_result old_transaction       = owner.begin_suspend(old_request);
    require(old_transaction && owner.rollback_suspend(old_transaction.transaction), "retain stale transaction token");

    require(owner.mark_terminal_family(old_request), "terminalize old request epoch");
    require(owner.confirm_terminal_execution_release(current_old_execution), "release current old execution");
    require(owner.remove_terminal_family(old_request), "remove old request epoch");

    const request_handle      new_request      = handle(old_request.id, old_request.epoch + 1);
    const registration_result new_registration = add_singleton(owner, new_request, 0);
    require(new_registration, "register reused ID with new epoch");
    const transaction_result new_transaction = owner.begin_suspend(new_request);
    require(new_transaction, "begin new epoch transaction");
    const family_snapshot new_active = get_required(owner, new_request);
    require(owner.commit_suspend(old_transaction.transaction,
                                 {
                                     { old_request, 751 }
    })
                    .code == result_code::stale_request,
            "old transaction cannot commit against reused request ID");
    require(owner.rollback_suspend(old_transaction.transaction).code == result_code::stale_request,
            "old transaction cannot roll back reused request ID");
    require(owner.confirm_terminal_execution_release(old_execution).code == result_code::stale_request,
            "old execution lease cannot address reused request ID");
    require(owner.confirm_terminal_resident_release(old_resident).code == result_code::stale_request,
            "old resident lease cannot address reused request ID");
    require(get_required(owner, new_request) == new_active, "stale epoch tokens cannot mutate new active transaction");
    require(owner.rollback_suspend(new_transaction.transaction), "rollback new epoch transaction");

    require(owner.mark_terminal_family(new_request), "terminalize new request epoch");
    require(owner.confirm_terminal_execution_release(new_registration.executions[0]), "release new execution");
    require(owner.remove_terminal_family(new_request), "remove new request epoch");
    require(owner.drained(), "epoch reuse registry drains");
}

void test_nonempty_destruction_fails_closed() {
    registry empty;
    require(empty.drained(), "new registry starts drained");

    const pid_t child = fork();
    require(child >= 0, "fork lifetime invariant child");
    if (child == 0) {
        std::set_terminate([]() { std::_Exit(73); });
        {
            registry                  undrained;
            const registration_result registration = add_singleton(undrained, handle(79, 1), 0);
            if (!registration) {
                std::_Exit(74);
            }
        }
        std::_Exit(75);
    }

    int status = 0;
    require(waitpid(child, &status, 0) == child, "wait for lifetime invariant child");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 73,
            "nonempty registry destruction invokes failure-closed termination");
}

struct fake_backend {
    void retain(backend_handle_id handle) { ++references[handle]; }

    void release(backend_handle_id handle) {
        auto it = references.find(handle);
        require(it != references.end() && it->second > 0, "fake backend release owns a reference");
        --it->second;
        if (it->second == 0) {
            references.erase(it);
        }
    }

    size_t reference_count() const {
        size_t total = 0;
        for (const auto & reference : references) {
            total += reference.second;
        }
        return total;
    }

    std::map<backend_handle_id, size_t> references;
};

void test_shared_backend_references_release_exactly_once() {
    registry             owner;
    fake_backend         backend;
    const request_handle parent = handle(81, 1);
    const request_handle child  = handle(82, 1);
    require(owner.register_bound_family(parent,
                                        {
                                            { parent, 0 },
                                            { child,  1 }
    }),
            "register shared family");

    const transaction_result suspend = owner.begin_suspend(parent);
    require(suspend, "begin shared-reference suspend");
    backend.retain(700);
    backend.retain(700);
    const transition_result resident = owner.commit_suspend(suspend.transaction, {
                                                                                     { parent, 700 },
                                                                                     { child,  700 }
    });
    require(resident && resident.residents.size() == 2 && backend.reference_count() == 2,
            "each resident lease represents one retained backend reference");
    require(!owner.drained(), "resident ownership prevents registry destruction");
    require(owner.mark_terminal_family(parent), "terminalize shared-reference family");
    backend.release(resident.residents[0].backend_handle);
    require(owner.confirm_terminal_resident_release(resident.residents[0]), "confirm first family reference");
    require(owner.confirm_terminal_resident_release(resident.residents[0]).code == result_code::stale_resident_lease,
            "partial family release cannot be confirmed twice");
    require(owner.remove_terminal_family(parent).code == result_code::resources_still_owned,
            "partial family cleanup cannot remove remaining ownership");
    require(backend.reference_count() == 1 && !owner.drained(), "partial cleanup retains one backend reference");
    backend.release(resident.residents[1].backend_handle);
    require(owner.confirm_terminal_resident_release(resident.residents[1]), "confirm final family reference");
    require(backend.reference_count() == 0, "shared backend reference count returns to baseline");
    require(!owner.drained(), "terminal metadata must be removed after resources release");
    require(owner.remove_terminal_family(parent), "remove shared-reference family");

    const request_handle      bound_parent = handle(83, 1);
    const request_handle      bound_child  = handle(84, 1);
    const registration_result bound =
        owner.register_bound_family(bound_parent, {
                                                      { bound_parent, 2 },
                                                      { bound_child,  3 }
    });
    require(bound && owner.mark_terminal_family(bound_parent), "terminalize bound family");
    require(owner.confirm_terminal_execution_release(bound.executions[1]), "confirm partial bound-family release");
    require(owner.confirm_terminal_execution_release(bound.executions[1]).code == result_code::stale_execution_lease,
            "partial bound-family release cannot be confirmed twice");
    require(owner.remove_terminal_family(bound_parent).code == result_code::resources_still_owned,
            "partial bound-family cleanup cannot remove remaining execution");
    require(owner.confirm_terminal_execution_release(bound.executions[0]), "confirm final bound-family release");
    require(owner.remove_terminal_family(bound_parent), "remove bound family after mixed cleanup");
    require(owner.drained(), "fake backend and mixed-family cleanup satisfy lifetime invariant");
}

registry_snapshot run_deterministic_replay() {
    registry_config config;
    config.max_requests        = 4;
    config.max_family_members  = 3;
    config.max_execution_slots = 8;
    config.max_resident_slots  = 4;
    registry owner(config);

    const request_handle parent = handle(90, 3);
    const request_handle child  = handle(91, 5);
    require(owner.register_bound_family(parent,
                                        {
                                            { child,  4 },
                                            { parent, 2 }
    }),
            "replay registration");
    require(add_singleton(owner, handle(10, 2), 0), "replay out-of-order singleton registration");
    const transaction_result suspend = owner.begin_suspend(parent);
    require(suspend, "replay suspend begin");
    require(owner.commit_suspend(suspend.transaction,
                                 {
                                     { child,  801 },
                                     { parent, 800 }
    }),
            "replay suspend commit");
    const transaction_result restore = owner.begin_restore(parent, {
                                                                       { child,  6 },
                                                                       { parent, 5 }
    });
    require(restore, "replay restore begin");
    require(owner.commit_restore(restore.transaction), "replay restore commit");
    const registry_snapshot result = owner.snapshot();
    simulate_external_release_and_remove(owner, parent);
    simulate_external_release_and_remove(owner, handle(10, 2));
    require(owner.drained(), "replay registry drains after snapshot capture");
    return result;
}

void test_bounded_cycles_and_deterministic_snapshots() {
    const registry_snapshot first  = run_deterministic_replay();
    const registry_snapshot second = run_deterministic_replay();
    require(first == second, "identical operation streams produce identical snapshots");
    require(first.families.size() == 2 && first.families[0].parent.id == 10 && first.families[1].parent.id == 90 &&
                first.families[1].members.size() == 2 && first.families[1].members[0].request.id == 90 &&
                first.families[1].members[1].request.id == 91,
            "families and members have deterministic ordering");

    registry             owner;
    const request_handle request = handle(100, 1);
    require(add_singleton(owner, request, 0), "register cycle request");
    registry_snapshot detached;
    uint64_t          final_resident_generation = 0;
    for (uint64_t cycle = 0; cycle < 1000; ++cycle) {
        const transaction_result suspend = owner.begin_suspend(request);
        require(suspend, "cycle suspend begin");
        const transition_result resident = owner.commit_suspend(suspend.transaction, {
                                                                                         { request, 900 + cycle }
        });
        require(resident, "cycle suspend commit");
        final_resident_generation = resident.residents[0].generation;
        if (cycle == 0) {
            detached = owner.snapshot();
        }
        const transaction_result restore =
            owner.begin_restore(request, {
                                             { request, static_cast<uint32_t>(cycle % 2) }
        });
        require(restore, "cycle restore begin");
        require(owner.commit_restore(restore.transaction), "cycle restore commit");
    }
    require(detached.families[0].members[0].state == lifecycle::resident &&
                detached.families[0].members[0].resident->backend_handle == 900,
            "detached snapshot is immutable across later cycles");
    const registry_summary summary = owner.summary();
    require(summary.families == 1 && summary.requests == 1 && summary.bound_requests == 1 &&
                summary.occupied_execution_slots == 1 && summary.occupied_resident_slots == 0,
            "repeated cycles retain bounded metadata");
    require(final_resident_generation == 1000, "resident generations advance without allocating tombstones");
    simulate_external_release_and_remove(owner, request);
    require(owner.drained(), "cycle registry drains before destruction");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "singleton suspend/restore identity",        test_singleton_suspend_restore_identity               },
        { "rollback preserves prior owner",            test_rollback_preserves_exact_prior_owner             },
        { "family transactions are atomic",            test_family_transactions_are_atomic                   },
        { "terminal cleanup and stale generations",    test_terminal_cleanup_and_stale_generations           },
        { "capacity and validation fail closed",       test_capacity_and_validation_fail_closed              },
        { "resident allocator skips exhausted slots",  test_resident_allocator_skips_exhausted_free_slots    },
        { "wrong-kind and illegal transition guards",  test_wrong_kind_illegal_states_and_preterminal_guards },
        { "stale tokens cannot cross epochs",          test_stale_tokens_cannot_cross_request_epochs         },
        { "nonempty destruction fails closed",         test_nonempty_destruction_fails_closed                },
        { "shared backend references release once",    test_shared_backend_references_release_exactly_once   },
        { "bounded cycles and deterministic snapshot", test_bounded_cycles_and_deterministic_snapshots       },
    };

    try {
        for (const auto & test : tests) {
            test.second();
            std::printf("PASS: %s\n", test.first);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }

    std::printf("PASS: %zu resident request ownership tests\n", tests.size());
    return 0;
}

#pragma once

#include "server-request-registry.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace server_resident_request {

using request_handle    = server_request_registry::request_handle;
using execution_slot_id = uint32_t;
using resident_slot_id  = uint32_t;

constexpr execution_slot_id no_execution_slot = UINT32_MAX;
constexpr resident_slot_id  no_resident_slot  = UINT32_MAX;

// This component records ownership transitions only. A backend handle is an
// opaque identity for one ownership reference supplied by a future DSV4 KV
// implementation. Zero is invalid. Multiple leases may carry the same handle
// value only when the backend has retained one reference for each lease. This
// registry never detaches, attaches, copies, or releases KV state itself.
using backend_handle_id = uint64_t;

enum class lifecycle : uint8_t {
    bound = 0,
    suspending,
    resident,
    restoring,
    terminal,
};

enum class transition_kind : uint8_t {
    none = 0,
    suspend,
    restore,
};

enum class result_code : uint8_t {
    ok = 0,
    invalid_family,
    duplicate_request,
    request_capacity_exhausted,
    unknown_request,
    stale_request,
    not_family_parent,
    execution_slot_out_of_range,
    execution_slot_occupied,
    resident_capacity_exhausted,
    generation_exhausted,
    invalid_transition,
    active_transaction,
    stale_transaction,
    incomplete_family,
    stale_execution_lease,
    stale_resident_lease,
    resources_still_owned,
    not_terminal,
};

struct execution_lease {
    request_handle    request;
    execution_slot_id slot       = no_execution_slot;
    uint64_t          generation = 0;

    bool operator==(const execution_lease & other) const;
};

struct resident_lease {
    request_handle    request;
    resident_slot_id  slot           = no_resident_slot;
    uint64_t          generation     = 0;
    backend_handle_id backend_handle = 0;

    bool operator==(const resident_lease & other) const;
};

struct transaction_lease {
    request_handle  parent;
    transition_kind kind       = transition_kind::none;
    uint64_t        generation = 0;

    bool operator==(const transaction_lease & other) const;
};

struct bound_member_registration {
    request_handle    request;
    execution_slot_id slot = no_execution_slot;
};

struct resident_commit {
    request_handle    request;
    backend_handle_id backend_handle = 0;
};

struct restore_target {
    request_handle    request;
    execution_slot_id slot = no_execution_slot;
};

struct member_snapshot {
    request_handle                 request;
    request_handle                 parent;
    lifecycle                      state    = lifecycle::bound;
    uint64_t                       revision = 0;
    std::optional<execution_lease> execution;
    std::optional<resident_lease>  resident;

    bool operator==(const member_snapshot & other) const;
};

// Families and their members are returned in request-ID/epoch order. The
// snapshot owns all of its values and is unaffected by later registry changes.
struct family_snapshot {
    request_handle                   parent;
    std::optional<transaction_lease> transaction;
    std::vector<member_snapshot>     members;

    bool operator==(const family_snapshot & other) const;
};

struct registry_summary {
    size_t families                 = 0;
    size_t requests                 = 0;
    size_t bound_requests           = 0;
    size_t resident_requests        = 0;
    size_t transitioning_requests   = 0;
    size_t terminal_requests        = 0;
    size_t occupied_execution_slots = 0;
    size_t occupied_resident_slots  = 0;

    bool operator==(const registry_summary & other) const;
};

struct registry_snapshot {
    std::vector<family_snapshot> families;
    registry_summary             summary;

    bool operator==(const registry_snapshot & other) const;
};

struct registry_config {
    size_t max_requests        = 4096;
    size_t max_family_members  = 64;
    size_t max_execution_slots = 256;
    size_t max_resident_slots  = 4096;
};

struct operation_result {
    result_code code    = result_code::ok;
    bool        changed = false;

    operator bool() const;
};

struct registration_result {
    result_code                  code = result_code::ok;
    std::vector<execution_lease> executions;

    operator bool() const;
};

struct transaction_result {
    result_code                  code = result_code::ok;
    transaction_lease            transaction;
    std::vector<execution_lease> executions;
    std::vector<resident_lease>  residents;

    operator bool() const;
};

struct transition_result {
    result_code                  code = result_code::ok;
    std::vector<execution_lease> executions;
    std::vector<resident_lease>  residents;

    operator bool() const;
};

class registry {
  public:
    explicit registry(registry_config config = {});

    // Destruction is permitted only after every family and external resource
    // lease has been explicitly released and removed. Violating this invariant
    // terminates the process rather than silently discarding opaque ownership.
    ~registry();

    registry(registry &&)                  = delete;
    registry & operator=(registry &&)      = delete;
    registry(const registry &)             = delete;
    registry & operator=(const registry &) = delete;

    // A singleton is a family whose member list contains only parent. For a
    // parent with passive children, all members must be registered together.
    registration_result register_bound_family(request_handle                                 parent,
                                              const std::vector<bound_member_registration> & members);

    // begin_suspend keeps every execution lease owned while the caller tries
    // backend detachment. commit_suspend exchanges the whole family for
    // resident leases; rollback_suspend preserves the original bindings. A
    // failed commit has not accepted any supplied backend ownership reference,
    // so the caller remains responsible for releasing those references.
    transaction_result begin_suspend(request_handle parent);
    transition_result  commit_suspend(const transaction_lease &            transaction,
                                      const std::vector<resident_commit> & residents);
    operation_result   rollback_suspend(const transaction_lease & transaction);

    // begin_restore keeps every resident lease owned and atomically reserves
    // execution slots. A failed backend restore can therefore roll back to the
    // exact resident leases without exposing partially restored ownership.
    // The backend must retain the resident references on restore failure and
    // consume them only before a successful commit_restore.
    transaction_result begin_restore(request_handle parent, const std::vector<restore_target> & targets);
    transition_result  commit_restore(const transaction_lease & transaction);
    operation_result   rollback_restore(const transaction_lease & transaction);

    // Terminal state retains all resource leases. The serialized caller must
    // release the corresponding external resource first, then confirm that
    // release exactly once. Removal is forbidden until no resource is owned.
    operation_result mark_terminal_family(request_handle parent);
    operation_result confirm_terminal_execution_release(const execution_lease & lease);
    operation_result confirm_terminal_resident_release(const resident_lease & lease);
    operation_result remove_terminal_family(request_handle parent);

    std::optional<family_snapshot> get_family(request_handle parent) const;
    registry_snapshot              snapshot() const;
    registry_summary               summary() const;
    // True only when destruction is safe. Generation tombstones may remain;
    // they carry no external ownership and keep stale leases invalid.
    bool                           drained() const;

#ifdef SERVER_RESIDENT_REQUEST_TESTING
    // Deterministic seam for generation-exhaustion coverage. Production
    // builds do not expose or compile this method.
    void test_override_resident_generation(resident_slot_id slot, uint64_t generation);
#endif

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

}  // namespace server_resident_request

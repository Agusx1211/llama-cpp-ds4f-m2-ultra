#pragma once

#include "llama-snapshot-store.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace server_kv_store {

using object_key = llama_snapshot_digest;

enum class object_class : uint8_t {
    live,
    prefix,
};

enum class status : uint8_t {
    ok,
    invalid_config,
    invalid_argument,
    authority_unavailable,
    not_found,
    class_conflict,
    object_limit,
    stale_generation,
    object_in_use,
    live_quota_exceeded,
    prefix_quota_exceeded,
    blocked_by_prefix_lease,
    reconciliation_required,
    io_error,
    snapshot_error,
    commit_uncertain,
};

const char * status_name(status value);

struct config {
    // snapshot.root_path names the one physical-device pool. Object stores are
    // placed below live/ and prefix/ and inherit the remaining format limits.
    // One process-wide authority lock is held for this root until the store
    // and every read lease issued by it have been destroyed.
    llama_snapshot_store_config snapshot;
    // live_quota_bytes is also the shared physical pool ceiling. Prefix data
    // may borrow its unused capacity but can never exceed its own subquota.
    uint64_t                    live_quota_bytes   = 0;
    uint64_t                    prefix_quota_bytes = 0;
    size_t                      max_live_objects   = 64;
    size_t                      max_prefix_objects = 8192;

    struct {
        // Deterministic construction seams; not exposed through server JSON.
        bool fail_pool_parent_fsync  = false;
        bool fail_class_parent_fsync = false;
    } test_faults;
};

struct faults {
    llama_snapshot_faults snapshot;
    // Deterministic failure before a selected disposable prefix is unlinked.
    bool                  fail_prefix_delete       = false;
    // Deterministic failure after creating a new object directory but before
    // its class-directory durability fence.
    bool                  fail_object_parent_fsync = false;
};

struct stats {
    uint64_t live_committed_bytes   = 0;
    uint64_t prefix_committed_bytes = 0;
    uint64_t live_reserved_bytes    = 0;
    uint64_t prefix_reserved_bytes  = 0;
    size_t   live_objects           = 0;
    size_t   prefix_objects         = 0;
    uint64_t prefix_evictions       = 0;
    bool     reconciliation_needed  = false;
};

struct write_result {
    status                  store_status    = status::invalid_argument;
    llama_snapshot_status   snapshot_status = llama_snapshot_status::invalid_argument;
    int                     os_error        = 0;
    uint64_t                generation      = 0;
    uint64_t                charged_bytes   = 0;
    bool                    committed       = false;
    std::vector<object_key> evicted_prefixes;
};

struct erase_result {
    status                store_status    = status::invalid_argument;
    llama_snapshot_status snapshot_status = llama_snapshot_status::invalid_argument;
    int                   os_error        = 0;
    uint64_t              released_bytes  = 0;
};

struct shared_state;
struct lease_counter;

class read_lease {
  public:
    ~read_lease();

    read_lease(const read_lease &)             = delete;
    read_lease & operator=(const read_lease &) = delete;
    read_lease(read_lease &&)                  = delete;
    read_lease & operator=(read_lease &&)      = delete;

    const llama_snapshot_manifest & manifest() const;
    llama_snapshot_status           validate(int * os_error = nullptr) const;
    llama_snapshot_read_result      read_all() const;

  private:
    friend class store;
    read_lease(std::shared_ptr<shared_state>  authority,
               std::shared_ptr<lease_counter> counter,
               llama_snapshot_store_config    snapshot,
               llama_snapshot_manifest        manifest);

    std::shared_ptr<shared_state>  authority_;
    std::shared_ptr<lease_counter> counter_;
    llama_snapshot_store_config    snapshot_;
    llama_snapshot_manifest        manifest_;
};

struct open_result {
    status                      store_status    = status::invalid_argument;
    llama_snapshot_status       snapshot_status = llama_snapshot_status::invalid_argument;
    int                         os_error        = 0;
    std::unique_ptr<read_lease> lease;
};

class store {
  public:
    // Every method except read_lease destruction performs synchronous
    // filesystem work and belongs on the storage worker, never an inference
    // thread. Dropping a lease is lock-free and does not wait for filesystem;
    // the lease nevertheless retains the root authority lock.
    explicit store(config config);
    ~store();

    store(const store &)             = delete;
    store & operator=(const store &) = delete;

    status initialization_status() const;
    status reconcile();
    stats  get_stats() const;

    write_result write_generation(const object_key &                  key,
                                  object_class                        storage_class,
                                  const llama_snapshot_metadata &     metadata,
                                  const std::vector<uint8_t> &        payload,
                                  const llama_snapshot_cancel_check & cancelled = {},
                                  const faults &                      injected  = {});

    write_result write_generation_streamed(const object_key &                  key,
                                           object_class                        storage_class,
                                           const llama_snapshot_metadata &     metadata,
                                           uint64_t                            total_payload_bytes,
                                           llama_snapshot_chunk_source_i &     source,
                                           const llama_snapshot_cancel_check & cancelled    = {},
                                           const faults &                      injected     = {},
                                           const llama_snapshot_commit_fence & commit_fence = {});

    open_result acquire(const object_key &              key,
                        object_class                    storage_class,
                        const llama_snapshot_identity & expected_identity);

    // Generation equality is a compare-and-delete fence. A durable tombstone
    // makes generations for a key permanently monotonic across deletion and
    // restart, so a stale completion cannot erase a recreated continuation.
    erase_result erase(const object_key & key, object_class storage_class, uint64_t expected_generation);

  private:
    std::shared_ptr<shared_state> state_;
};

}  // namespace server_kv_store

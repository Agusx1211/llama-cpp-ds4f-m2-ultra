#pragma once

#include "ggml-dsv4-sparse.h"
#include "llama-kv-cache.h"

#include <memory>
#include <mutex>
#include <thread>
#include <vector>

//
// llama_kv_cache_iswa
//

// utilizes two instances of llama_kv_cache
//   the first instance is for the non-SWA layers of the model and the second instance is for the SWA layers

enum class llama_kv_iswa_resident_status : uint8_t {
    ok,
    invalid_sequence,
    unsupported_layout,
    slot_occupied,
    capacity_exhausted,
    stale_quote,
    stale_handle,
    generation_exhausted,
    not_quiescent,
    resource_exhausted,
    backend_error,
};

const char * llama_kv_iswa_resident_status_name(llama_kv_iswa_resident_status status);

// Test-only release audit. The raw resident release fills this from the
// exact sparse move quote that it prepares and then commits; callers never
// re-quote the resident handle. The before/after records retain the backend's
// stable source hashes and physical accounting counters for assertions.
struct llama_kv_iswa_resident_release_pool_audit {
    ggml_dsv4_sparse_move_audit_pool before = {};
    ggml_dsv4_sparse_move_audit_pool after  = {};
};

struct llama_kv_iswa_resident_release_audit {
    bool observed = false;
    int before_status = GGML_DSV4_SPARSE_UNSUPPORTED;
    int after_status  = GGML_DSV4_SPARSE_UNSUPPORTED;
    std::vector<llama_kv_iswa_resident_release_pool_audit> pools;
};

// Host-test seam for the execution-independent ownership protocol. Production
// leaves this unset and resolves the backend procedures from the tensor's
// device registry. The override is thread-local and affects resident calls
// only.
struct llama_kv_iswa_resident_backend_override {
    ggml_dsv4_sparse_move_quote_fn  quote  = nullptr;
    ggml_dsv4_sparse_move_commit_fn commit = nullptr;
    ggml_dsv4_sparse_move_free_fn   free   = nullptr;
    ggml_dsv4_sparse_move_audit_fn  audit  = nullptr;
};

void llama_kv_iswa_set_resident_backend_override_for_test(
        llama_kv_iswa_resident_backend_override backend);
void llama_kv_iswa_fail_next_resident_allocation_for_test();

enum class llama_kv_iswa_resident_lock_phase : uint8_t {
    before_lock,
    lock_contended,
    lock_acquired,
};

using llama_kv_iswa_resident_lock_hook_for_test = void (*)(llama_kv_iswa_resident_lock_phase phase, void * context);

// One-shot, thread-local synchronization seam. The hook runs immediately
// before and after the next outer ISWA resident-lock acquisition, then clears
// itself before either callback so recursive coordinator calls cannot replay
// it. Production leaves the hook unset.
void llama_kv_iswa_set_resident_lock_hook_for_test(llama_kv_iswa_resident_lock_hook_for_test hook, void * context);

struct llama_kv_iswa_resident_handle {
    uint64_t pool_id    = 0;
    uint64_t id         = 0;
    uint64_t generation = 0;

    bool operator==(const llama_kv_iswa_resident_handle & other) const {
        return pool_id == other.pool_id && id == other.id && generation == other.generation;
    }
};

struct llama_kv_iswa_resident_detach_plan;
struct llama_kv_iswa_resident_attach_plan;

struct llama_kv_iswa_resident_detach_quote {
    llama_kv_iswa_resident_status status = llama_kv_iswa_resident_status::unsupported_layout;
    llama_seq_id execution_id = -1;
    uint32_t resident_slot = UINT32_MAX;
    llama_kv_iswa_resident_handle resident;

  private:
    std::shared_ptr<llama_kv_iswa_resident_detach_plan> plan;
    friend class llama_kv_cache_iswa;
};

struct llama_kv_iswa_resident_result {
    llama_kv_iswa_resident_status status = llama_kv_iswa_resident_status::unsupported_layout;
    llama_kv_iswa_resident_handle resident;
};

struct llama_kv_iswa_resident_attach_quote {
    llama_kv_iswa_resident_status status = llama_kv_iswa_resident_status::unsupported_layout;
    llama_seq_id execution_id = -1;
    llama_kv_iswa_resident_handle resident;

  private:
    std::shared_ptr<llama_kv_iswa_resident_attach_plan> plan;
    friend class llama_kv_cache_iswa;
};

// Placement-independent K-only sequence image used by the DSV4 aggregate
// state container.  Positions are absolute and strictly increasing.  The
// tensor payload is emitted in that same logical order and contains tensor
// geometry plus row bytes, but no execution stream, cell, sparse-page, or
// resident-handle identity.
constexpr uint32_t LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA = 1;

struct llama_kv_iswa_logical_plane_state {
    uint32_t                       schema_version = LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA;
    std::vector<llama_pos>         positions;
    std::vector<llama_kv_cell_ext> extensions;
    std::vector<uint8_t>           tensor_payload;
    uint64_t                       checksum = 0;
};

struct llama_kv_iswa_logical_sequence_state {
    uint32_t                          schema_version = LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA;
    llama_kv_iswa_logical_plane_state base;
    llama_kv_iswa_logical_plane_state swa;
};

struct llama_kv_iswa_logical_import_plan;

// Move-only exclusive capability for a composite resident transaction. The
// token retains a generation-bound raw/SWA gate, so ordinary cache mutation
// and graph preparation in other threads cannot enter between component
// validation and publication. Coordinator calls on the owning thread may
// safely re-enter the recursive mutex; release/destruction is cross-thread
// safe and wakes every waiter immediately.
class llama_kv_iswa_resident_transaction {
  public:
    llama_kv_iswa_resident_transaction() = default;
    ~llama_kv_iswa_resident_transaction();

    llama_kv_iswa_resident_transaction(const llama_kv_iswa_resident_transaction &)             = delete;
    llama_kv_iswa_resident_transaction & operator=(const llama_kv_iswa_resident_transaction &) = delete;
    llama_kv_iswa_resident_transaction(llama_kv_iswa_resident_transaction && other) noexcept;
    llama_kv_iswa_resident_transaction & operator=(llama_kv_iswa_resident_transaction &&) = delete;

    explicit operator bool() const { return active; }

    // Coordinator work is same-thread so recursive calls remain well-defined,
    // but release/destruction is cross-thread safe. release() lets a consumed
    // quote drop quiescence immediately even while the quote remains alive.
    bool owned_by_current_thread() const;
    void release() noexcept;

  private:
    llama_kv_iswa_resident_transaction(std::shared_ptr<llama_kv_cache_resident_guard> guard,
                                       std::unique_lock<std::recursive_mutex>         lock);

    std::shared_ptr<llama_kv_cache_resident_guard> guard;
    std::thread::id                                owner_thread;
    uint64_t                                       generation = 0;
    bool                                           active = false;

    friend class llama_kv_cache_iswa;
};

// The sparse backend exposes one-way, generation-bound move tickets. Callers
// coordinating raw/SWA with reversible metadata must therefore attest that
// this commit is the transaction's final atomic step. The scoped enum prevents
// an unannotated prepared commit from compiling.
enum class llama_kv_iswa_resident_final_step : uint8_t {
    confirmed,
};

class llama_kv_cache_iswa : public llama_memory_i {
public:
    llama_kv_cache_iswa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        ggml_backend_buffer_type_t buft_override = nullptr,
                    uint32_t   n_resident = 0,
                         bool   logical_transactions = false);

    llama_kv_cache_iswa(
            const llama_model & model,
            const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   swa_full,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_pad,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
        ggml_backend_buffer_type_t buft_override = nullptr,
                    uint32_t   n_resident = 0,
                         bool   logical_transactions = false);

    ~llama_kv_cache_iswa();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_iswa specific API
    //

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_swa () const;

    // Canonical raw/SWA state for one DSV4 sequence.  These operations are
    // intentionally K-only: DSV4 removes V storage from both ISWA planes.
    // A prepared import builds complete replacement metadata without touching
    // the execution stream. commit_logical_sequence_import() performs only
    // validated tensor writes and noexcept metadata moves, so an occupied
    // destination remains byte-for-byte intact until the caller crosses its
    // own multi-component publication boundary.
    llama_kv_iswa_logical_sequence_state export_logical_sequence(llama_seq_id seq_id) const;
    void validate_logical_sequence(
            const llama_kv_iswa_logical_sequence_state & state,
            llama_pos                                    accepted_frontier) const;
    std::shared_ptr<llama_kv_iswa_logical_import_plan> prepare_logical_sequence_import(
            llama_seq_id                                 seq_id,
            const llama_kv_iswa_logical_sequence_state & state,
            llama_pos                                    accepted_frontier) const;
    bool validate_logical_sequence_import(
            const std::shared_ptr<llama_kv_iswa_logical_import_plan> & plan) const;
    bool commit_logical_sequence_import(
            const std::shared_ptr<llama_kv_iswa_logical_import_plan> & plan,
            const llama_kv_iswa_logical_sequence_state & state) noexcept;

    // Dormant DSV4 ownership primitive. A detach quote validates raw and SWA
    // metadata plus one free resident aperture before any mapping changes.
    // Detach transfers both planes atomically. Prepared attach allocates and
    // validates everything while the lease remains resident. Rollback is an
    // allocation-free, idempotent cancellation before commit. The backend has
    // no reverse-ticket primitive, so commit_resident_attach_final() is an
    // irreversible atomic success and must follow all reversible component
    // commits. release drops the resident page references.
    llama_kv_iswa_resident_detach_quote quote_resident_detach(llama_seq_id execution_id) const;
    llama_kv_iswa_resident_status validate_resident_detach(
            const llama_kv_iswa_resident_detach_quote & quote) const;
    llama_kv_iswa_resident_result detach_resident(const llama_kv_iswa_resident_detach_quote & quote);
    llama_kv_iswa_resident_attach_quote quote_resident_attach(llama_kv_iswa_resident_handle resident,
                                                              llama_seq_id execution_id) const;
    llama_kv_iswa_resident_status validate_resident_attach(
            const llama_kv_iswa_resident_attach_quote & quote) const;
    llama_kv_iswa_resident_status commit_resident_attach_final(
            const llama_kv_iswa_resident_attach_quote & quote,
            llama_kv_iswa_resident_final_step final_step);
    llama_kv_iswa_resident_status rollback_resident_attach(
            const llama_kv_iswa_resident_attach_quote & quote);

    // One-shot compatibility helper. Multi-component transactions must use
    // the explicit prepared/final-step API above.
    llama_kv_iswa_resident_status attach_resident(llama_kv_iswa_resident_handle resident,
                                                  llama_seq_id execution_id);
    llama_kv_iswa_resident_status release_resident(
            llama_kv_iswa_resident_handle resident,
            llama_kv_iswa_resident_release_audit * audit = nullptr);

    // Used by generic ISWA and DSV4 raw graph contexts. The opaque lease keeps
    // residency fail-closed from preparation through graph completion or
    // rollback; destruction releases it.
    std::shared_ptr<void> acquire_resident_batch_lease() const;

    // A composite DSV4 ownership transaction takes this retained exclusive
    // token only after all component quotes exist. It requires quiescence and
    // excludes graph preparation plus every raw/SWA public mutation until the
    // composite reaches a terminal commit or rollback boundary.
    llama_kv_iswa_resident_transaction acquire_resident_transaction() const;
    // True when the cache owns a resident aperture.  This distinguishes the
    // unsupported (no-aperture) layout from a resident layout whose
    // transaction is temporarily unavailable because it is not quiescent.
    bool has_resident_aperture() const;
    bool has_resident_handles() const;

private:
    const bool unified;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;

    // DSV4 logical restore needs the same quiescence boundary as resident
    // moves even when no resident aperture was requested. Generic ISWA keeps
    // this null and pays no locking or graph-lease cost.
    std::shared_ptr<llama_kv_cache_resident_guard> logical_guard;

    struct resident_impl;
    std::unique_ptr<resident_impl> resident;

    std::unique_lock<std::recursive_mutex> lock_resident() const;
    bool resident_is_quiescent() const;

    friend class llama_kv_cache_iswa_context;
};

class llama_kv_cache_iswa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_iswa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv);

    // used to create an update context
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_iswa_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_iswa_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_iswa_context specific API
    //

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_swa()  const;

private:
    llama_kv_cache_iswa * kv = nullptr;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const std::shared_ptr<void> resident_batch_lease;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_swa;

    const llama_memory_status status;
};

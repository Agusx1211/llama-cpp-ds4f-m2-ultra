#pragma once

#include "ggml-dsv4-sparse.h"
#include "llama-kv-cache.h"

#include <memory>
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

// Host-test seam for the execution-independent ownership protocol. Production
// leaves this unset and resolves the backend procedures from the tensor's
// device registry. The override is thread-local and affects resident calls
// only.
struct llama_kv_iswa_resident_backend_override {
    ggml_dsv4_sparse_move_quote_fn  quote  = nullptr;
    ggml_dsv4_sparse_move_commit_fn commit = nullptr;
    ggml_dsv4_sparse_move_free_fn   free   = nullptr;
};

void llama_kv_iswa_set_resident_backend_override_for_test(
        llama_kv_iswa_resident_backend_override backend);
void llama_kv_iswa_fail_next_resident_allocation_for_test();

struct llama_kv_iswa_resident_handle {
    uint64_t pool_id    = 0;
    uint64_t id         = 0;
    uint64_t generation = 0;

    bool operator==(const llama_kv_iswa_resident_handle & other) const {
        return pool_id == other.pool_id && id == other.id && generation == other.generation;
    }
};

struct llama_kv_iswa_resident_detach_plan;

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
                    uint32_t   n_resident = 0);

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
                    uint32_t   n_resident = 0);

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

    // Dormant DSV4 ownership primitive. A detach quote validates raw and SWA
    // metadata plus one free resident aperture before any mapping changes.
    // Detach transfers both planes atomically; attach consumes the lease into
    // any empty execution sequence, and release drops its page references.
    llama_kv_iswa_resident_detach_quote quote_resident_detach(llama_seq_id execution_id) const;
    llama_kv_iswa_resident_result detach_resident(const llama_kv_iswa_resident_detach_quote & quote);
    llama_kv_iswa_resident_status attach_resident(llama_kv_iswa_resident_handle resident,
                                                  llama_seq_id execution_id);
    llama_kv_iswa_resident_status release_resident(llama_kv_iswa_resident_handle resident);

    // Used by generic ISWA and DSV4 raw graph contexts. The opaque lease keeps
    // residency fail-closed from preparation through graph completion or
    // rollback; destruction releases it.
    std::shared_ptr<void> acquire_resident_batch_lease() const;
    bool has_resident_handles() const;

private:
    const bool unified;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_swa;

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

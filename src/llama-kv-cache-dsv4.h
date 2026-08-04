#pragma once

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-dsv4-comp-pool.h"

#include <map>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

class dsv4_sparse_transaction;
struct dsv4_sparse_range;
struct llama_kv_cache_dsv4_admission_state;

// True only for the target M2 Ultra Metal backend when placement-sparse
// buffers can provide DSV4's elastic physical page pool.
bool llama_kv_cache_dsv4_supports_elastic_metal(
        const llama_model & model,
        uint32_t n_seq_max);

// Test-only deterministic fault seam. Each injected failure is consumed by a
// successful target Metal sparse reservation before it can be committed.
void llama_kv_cache_dsv4_test_inject_physical_pressure(uint32_t count);

struct llama_dsv4_cow_preflight_test_stats {
    uint64_t source_ranges_submitted      = 0;
    uint64_t destination_ranges_submitted = 0;
    uint64_t copy_operations              = 0;
};

// Test-only observability for aggregate COW preflight ordering. Submitted
// ranges have reached the atomic sparse reservation call; copy operations have
// started moving a cache tensor segment.
void                                llama_kv_cache_dsv4_test_reset_cow_preflight_stats();
llama_dsv4_cow_preflight_test_stats llama_kv_cache_dsv4_test_get_cow_preflight_stats();

enum llama_dsv4_memory_family {
    LLAMA_DSV4_MEMORY_RAW = 0,
    LLAMA_DSV4_MEMORY_CSA,
    LLAMA_DSV4_MEMORY_HCA,
    LLAMA_DSV4_MEMORY_LID,
    LLAMA_DSV4_MEMORY_FAMILY_COUNT,
};

struct llama_dsv4_sparse_pool_usage {
    uintptr_t pool_id = 0;
    uint64_t page_size = 0;
    uint64_t virtual_pages = 0;
    uint64_t physical_pages = 0;
    uint64_t free_pages = 0;
    uint64_t reserved_pages = 0;
    uint64_t mapped_mappings = 0;
    uint64_t unique_physical_pages = 0;
    uint64_t shared_physical_pages = 0;
    uint64_t shared_mappings = 0;
    uint64_t refcount_sum = 0;
    uint32_t refcount_max = 0;
    uint64_t generation = 0;
    uint64_t cow_allocations = 0;
    uint64_t cow_pages = 0;
};

struct llama_dsv4_family_usage {
    llama_dsv4_memory_family family = LLAMA_DSV4_MEMORY_RAW;
    bool placement_sparse = false;
    uint64_t logical_capacity_rows = 0;
    uint64_t logical_mapped_rows = 0;
    std::vector<uint64_t> sequence_mapped_rows;
    std::vector<llama_dsv4_sparse_pool_usage> pools;
    llama_dsv4_sparse_pool_usage total;
};

struct llama_dsv4_memory_usage_snapshot {
    std::array<llama_dsv4_family_usage, LLAMA_DSV4_MEMORY_FAMILY_COUNT> families;
    llama_dsv4_sparse_pool_usage sparse_total;
    llama_dsv4_memory_family limiting_family = LLAMA_DSV4_MEMORY_RAW;
    uint32_t limiting_family_mask = 0;
    uintptr_t limiting_pool_id = 0;
    uint64_t limiting_available_pages = 0;
};

struct llama_dsv4_family_quote {
    llama_dsv4_memory_family family = LLAMA_DSV4_MEMORY_RAW;
    uint64_t target_mappings = 0;
    uint64_t new_pages = 0;
    uint64_t cow_pages = 0;
    uint64_t required_pages = 0;
    uint64_t physical_pages = 0;
    uint64_t free_pages = 0;
    uint64_t reserved_pages = 0;
    uint64_t logical_rows = 0;
    bool feasible = true;
};

struct llama_dsv4_batch_quote {
    std::array<llama_dsv4_family_quote, LLAMA_DSV4_MEMORY_FAMILY_COUNT> families;
    llama_dsv4_memory_family limiting_family = LLAMA_DSV4_MEMORY_RAW;
    uintptr_t limiting_pool_id = 0;
    bool feasible = true;
};

struct llama_dsv4_sparse_page_quote_test_audit {
    uint64_t generation      = 0;
    uint64_t target_mappings = 0;
    uint64_t new_pages       = 0;
    uint64_t cow_pages       = 0;
    uint64_t required_pages  = 0;
    uint64_t free_pages      = 0;
    uint64_t reserved_pages  = 0;
    bool     feasible        = false;
};

struct llama_dsv4_sparse_pool_delta_test_audit {
    // Physical deltas belong to this stable Metal pool exactly once. The mask
    // and range arrays describe every logical DSV4 family that submitted a
    // tensor range to the pool; callers must not sum duplicate family views.
    uintptr_t                                            pool_id                   = 0;
    uint32_t                                             family_mask               = 0;
    std::array<uint64_t, LLAMA_DSV4_MEMORY_FAMILY_COUNT> family_range_count        = {};
    std::array<uint64_t, LLAMA_DSV4_MEMORY_FAMILY_COUNT> family_range_bytes        = {};
    std::array<uint64_t, LLAMA_DSV4_MEMORY_FAMILY_COUNT> family_zero_offset_ranges = {};
    llama_dsv4_sparse_page_quote_test_audit              dry_quote;
    llama_dsv4_sparse_page_quote_test_audit              reserved_quote;
    llama_dsv4_sparse_pool_usage                         before;
    llama_dsv4_sparse_pool_usage                         after;
};

struct llama_dsv4_sparse_page_delta_test_audit {
    bool                                                 observed   = false;
    bool                                                 dry_quoted = false;
    bool                                                 reserved   = false;
    bool                                                 committed  = false;
    bool                                                 cancelled  = false;
    std::array<uint64_t, LLAMA_DSV4_MEMORY_FAMILY_COUNT> family_range_count = {};
    std::array<uint64_t, LLAMA_DSV4_MEMORY_FAMILY_COUNT> family_range_bytes = {};
    std::vector<llama_dsv4_sparse_pool_delta_test_audit> pools;
};

// Opt-in test-only audit of the real Metal dry-quote, reservation, and final
// pool accounting boundaries. Disabled operation performs no extra quote or
// usage snapshot work.
void                                    llama_kv_cache_dsv4_test_enable_page_delta_audit(bool enabled);
void                                    llama_kv_cache_dsv4_test_reset_page_delta_audit();
llama_dsv4_sparse_page_delta_test_audit llama_kv_cache_dsv4_test_get_page_delta_audit();

enum llama_dsv4_resident_component : uint32_t {
    LLAMA_DSV4_RESIDENT_RAW_SWA        = 1u << 0,
    LLAMA_DSV4_RESIDENT_COMPRESSED     = 1u << 1,
    LLAMA_DSV4_RESIDENT_CSA_STATE      = 1u << 2,
    LLAMA_DSV4_RESIDENT_HCA_STATE      = 1u << 3,
    LLAMA_DSV4_RESIDENT_LID_STATE      = 1u << 4,
    LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX = 1u << 5,
    LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT = 1u << 6,
};

enum class llama_dsv4_resident_scope : uint8_t {
    single_context,
    target_draft_pair,
};

enum class llama_dsv4_resident_status : uint8_t {
    ok,
    invalid_scope,
    invalid_sequence,
    unsupported_components,
    capacity_exhausted,
    slot_occupied,
    not_quiescent,
    stale_quote,
    stale_handle,
    generation_exhausted,
    resource_exhausted,
    backend_error,
};

const char * llama_dsv4_resident_status_name(llama_dsv4_resident_status status);

struct llama_dsv4_resident_detach_request {
    llama_seq_id              seq_id = -1;
    llama_dsv4_resident_scope scope  = llama_dsv4_resident_scope::single_context;
};

struct llama_dsv4_resident_handle {
    uint64_t cache_id          = 0;
    uint64_t id                = 0;
    uint64_t handle_generation = 0;
    uint64_t lease_generation  = 0;

    bool operator==(const llama_dsv4_resident_handle & other) const {
        return cache_id == other.cache_id && id == other.id &&
               handle_generation == other.handle_generation &&
               lease_generation == other.lease_generation;
    }
};

struct llama_dsv4_resident_detach_plan;
struct llama_dsv4_resident_attach_plan;
class llama_dsv4_composite_resident;

// This is a capability quote, not an ownership handle. It snapshots the
// rollback index and reports every execution-independent component before any
// detach may begin. A target+draft request also requires a higher-level paired
// context transaction; one cache instance cannot claim its counterpart.
struct llama_dsv4_resident_detach_quote {
    llama_dsv4_resident_status status                 = llama_dsv4_resident_status::unsupported_components;
    llama_seq_id               seq_id                 = -1;
    llama_dsv4_resident_scope  scope                  = llama_dsv4_resident_scope::single_context;
    uint32_t                   rollback_index         = 0;
    uint32_t                   required_components    = 0;
    uint32_t                   detachable_components  = 0;
    uint32_t                   unsupported_components = 0;
    uint32_t                   resident_state_slot    = UINT32_MAX;
    llama_dsv4_resident_handle resident;

  private:
    std::shared_ptr<llama_dsv4_resident_detach_plan> plan;
    friend class llama_dsv4_composite_resident;
};

struct llama_dsv4_resident_result {
    llama_dsv4_resident_status status = llama_dsv4_resident_status::unsupported_components;
    llama_dsv4_resident_handle resident;
};

struct llama_dsv4_resident_attach_quote {
    llama_dsv4_resident_status status       = llama_dsv4_resident_status::unsupported_components;
    llama_seq_id               execution_id = -1;
    llama_dsv4_resident_handle resident;

  private:
    std::shared_ptr<llama_dsv4_resident_attach_plan> plan;
    friend class llama_dsv4_composite_resident;
};

struct llama_dsv4_resident_usage {
    uint64_t cache_id       = 0;
    uint64_t epoch          = 0;
    uint32_t capacity       = 0;
    uint32_t occupied_slots = 0;
    uint32_t handles        = 0;
};

// Host-testable policy used by llama_kv_cache_dsv4::quote_resident_detach.
// This static policy assumes the default sequence-indexed raw/SWA layout.
// llama_kv_cache_dsv4::quote_resident_detach separately adds the opt-in target
// raw/SWA aperture when its backend quote succeeds. Compressor/state planes
// still keep the whole-sequence capability fail-closed. No sequence ID is
// reserved or parked by this function.
llama_dsv4_resident_detach_quote llama_dsv4_quote_resident_detach_layout(llama_dsv4_resident_detach_request request,
                                                                         uint32_t                           n_seq_max,
                                                                         uint32_t rollback_index,
                                                                         bool     aggregate_compressed);

class llama_dsv4_comp_state {
public:
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    stream_copy_info sc_info;

    llama_dsv4_comp_state(
            const llama_model & model,
            bool            offload,
            bool            unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            uint32_t        n_rs_seq,
            const char    * name,
        const llama_memory_i::layer_filter_cb & filter);

    void clear(llama_seq_id seq_id, bool data);
    void seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, uint32_t src_depth = 0);
    void apply_copies(const stream_copy_info & sc_info) const;

    // Copy every current and rollback plane for one sequence between separate
    // fixed allocations. Resident apertures use a different n_stream and are
    // never exposed to graph construction.
    void copy_sequence_all_depths_to(
            llama_dsv4_comp_state & destination,
            llama_seq_id            source_sequence,
            llama_seq_id            destination_sequence) const;
    bool sequence_all_depths_zero(llama_seq_id seq_id) const;

    uint32_t get_ratio()      const;
    uint32_t get_state_size() const;
    uint32_t get_n_stream()   const;
    uint32_t get_n_rs_seq()   const;
    uint32_t get_n_rows()     const;
    uint64_t state_identity() const;
    uint64_t state_generation() const;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, const std::vector<uint32_t> & rs_idx) const;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    ggml_tensor * get_kv       (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score    (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_kv_all   (ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_score_all(ggml_context * ctx, int32_t il) const;

    ggml_tensor * cpy_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;

private:
    struct layer {
        uint32_t il;

        ggml_tensor * kv;
        ggml_tensor * score;

        std::vector<ggml_tensor *> kv_stream;
        std::vector<ggml_tensor *> score_stream;
    };

    const uint32_t ratio;
    const uint32_t state_size;
    const uint32_t n_embd_state;
    const uint32_t n_stream;
    const uint32_t n_rs_seq;

    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    std::vector<layer> layers;

    std::unordered_map<int32_t, int32_t> map_layer_ids;

    mutable uint64_t generation = 1;

    size_t total_size() const;
    void bump_generation() const;
    bool has_generation_headroom(uint64_t increments) const;

    friend class llama_dsv4_composite_resident;
};

// Deterministic fault seam for cross-instance recurrent-state copies. A
// nonnegative value fails after that many successful tensor-plane copies.
void llama_dsv4_comp_state_fail_copy_after_for_test(int successful_copies);

// Backend-only whole-sequence coordinator. Production constructs it only for
// the default-off target aggregate layout. Tests instantiate the same class
// with host buffers and the raw/SWA backend override.
class llama_dsv4_composite_resident {
public:
    llama_dsv4_composite_resident(
            llama_kv_cache_iswa & raw,
            llama_dsv4_comp_pool & compressed,
            llama_dsv4_comp_state & csa_execution,
            llama_dsv4_comp_state & hca_execution,
            llama_dsv4_comp_state & lid_execution,
            llama_dsv4_comp_state & csa_resident,
            llama_dsv4_comp_state & hca_resident,
            llama_dsv4_comp_state & lid_resident,
            std::vector<uint32_t> & rollback_index,
            uint32_t & active_rollback_depth,
            uint32_t n_seq_max,
            uint32_t resident_capacity);
    ~llama_dsv4_composite_resident();

    llama_dsv4_composite_resident(const llama_dsv4_composite_resident &) = delete;
    llama_dsv4_composite_resident & operator=(const llama_dsv4_composite_resident &) = delete;

    llama_dsv4_resident_detach_quote quote_detach(llama_dsv4_resident_detach_request request) const;
    llama_dsv4_resident_result       detach(const llama_dsv4_resident_detach_quote & quote);
    llama_dsv4_resident_attach_quote quote_attach(llama_dsv4_resident_handle resident,
                                                   llama_seq_id execution_id) const;
    llama_dsv4_resident_status       attach(const llama_dsv4_resident_attach_quote & quote);
    llama_dsv4_resident_status       release(llama_dsv4_resident_handle resident);

    bool has_handles() const;
    llama_dsv4_resident_usage usage() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

//
// llama_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed caches are storage only; DSV4-specific visibility and block
// planning are handled by llama_kv_cache_dsv4_context / llm_graph_input_dsv4.
// FIXME: currently the cache only supports non-unified mode even if unified flag is passed
// FIXME: we currently conflate token_pos and buffer contents. See https://github.com/ggml-org/llama.cpp/pull/25521#discussion_r3558173819

class llama_kv_cache_dsv4 : public llama_memory_i {
public:
    llama_kv_cache_dsv4(
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
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache_dsv4();

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

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache_dsv4 specific API
    //

    llama_kv_cache_iswa * get_raw() const;
    llama_kv_cache      * get_csa() const;
    llama_kv_cache      * get_hca() const;
    llama_kv_cache      * get_lid() const;
    llama_dsv4_comp_state * get_csa_state() const;
    llama_dsv4_comp_state * get_hca_state() const;
    llama_dsv4_comp_state * get_lid_state() const;

    bool is_aggregate_compressed() const;
    uint32_t get_c4_logical_rows() const;
    uint32_t get_hca_logical_rows() const;
    llama_dsv4_comp_pool * get_comp_pool() const;

    llama_dsv4_resident_detach_quote quote_resident_detach(llama_dsv4_resident_detach_request request) const;
    llama_dsv4_resident_result       detach_resident(const llama_dsv4_resident_detach_quote & quote);
    llama_dsv4_resident_attach_quote quote_resident_attach(llama_dsv4_resident_handle resident,
                                                           llama_seq_id execution_id) const;
    llama_dsv4_resident_status       attach_resident(const llama_dsv4_resident_attach_quote & quote);
    llama_dsv4_resident_status       release_resident(llama_dsv4_resident_handle resident);
    llama_dsv4_resident_usage        resident_usage() const;

    uint32_t get_n_rs_seq() const;
    const std::vector<uint32_t> & get_rs_idx() const;
    bool set_rs_depth(uint32_t depth);
    bool set_rs_enabled(bool enabled);
    void reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches);
    llama_dsv4_memory_usage_snapshot memory_usage_snapshot() const;

    llama_kv_admission_status quote_admission(
            const llama_kv_admission_span * spans,
            size_t n_spans,
            llama_kv_admission_quote & quote);
    llama_kv_admission_status reserve_admission(
            const llama_kv_admission_span * spans,
            size_t n_spans,
            llama_kv_admission_quote & quote,
            uint64_t & id);

    bool arm_admission(uint64_t id);
    void cancel_admission(uint64_t id);

private:
    friend class llama_kv_cache_dsv4_context;

    enum admission_consume_status {
        ADMISSION_NO_MATCH,
        ADMISSION_COMMITTED,
        ADMISSION_ERROR,
    };

    uint64_t register_admission(
            std::unique_ptr<dsv4_sparse_transaction> reservation,
            const llama_kv_admission_span * spans,
            size_t n_spans,
            const llama_dsv4_batch_quote & quote,
            uint32_t limiting_family_mask);
    bool collect_admission_ranges(
            const llama_kv_admission_span * spans,
            size_t n_spans,
            std::vector<dsv4_sparse_range> & ranges,
            llama_dsv4_batch_quote & quote) const;
    admission_consume_status consume_admission(
            const std::vector<llama_ubatch> & ubatches,
            llama_dsv4_batch_quote & quote,
            uint32_t & limiting_family_mask);

    llama_hparams hparams_raw;
    llama_hparams hparams_csa;
    llama_hparams hparams_hca;
    llama_hparams hparams_lid;

    const uint32_t n_seq_max;
    const uint32_t n_rs_seq;
    uint32_t n_rs_seq_active;

    std::vector<uint32_t> rs_idx;

    std::unique_ptr<llama_kv_cache_iswa> kv_raw;
    std::unique_ptr<llama_kv_cache>      kv_csa;
    std::unique_ptr<llama_kv_cache>      kv_hca;
    std::unique_ptr<llama_kv_cache>      kv_lid;
    std::unique_ptr<llama_dsv4_comp_state> csa_state;
    std::unique_ptr<llama_dsv4_comp_state> hca_state;
    std::unique_ptr<llama_dsv4_comp_state> lid_state;
    std::unique_ptr<llama_dsv4_comp_pool> comp_pool;

    std::unique_ptr<llama_dsv4_comp_state> resident_csa_state;
    std::unique_ptr<llama_dsv4_comp_state> resident_hca_state;
    std::unique_ptr<llama_dsv4_comp_state> resident_lid_state;
    std::unique_ptr<llama_dsv4_composite_resident> composite_resident;

    bool aggregate_compressed = false;
    uint32_t c4_logical_rows = 0;
    uint32_t hca_logical_rows = 0;

    uint64_t state_identity_hash = 0;

    std::unique_ptr<llama_kv_cache_dsv4_admission_state> admission_state;

    void clear_compressed(llama_seq_id seq_id, bool data);
};

// DSV4 raw attention only uses the SWA half of kv_raw. The base half is kept
// for generic ISWA bookkeeping, but it has no DSV4 layers to expose here.
class llama_kv_cache_dsv4_raw_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base_write,
            slot_info_vec_t sinfos_swa_write,
            slot_info_vec_t sinfos_swa_read,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_ubatch> ubatches_write);

    ~llama_kv_cache_dsv4_raw_context() override = default;

    bool next() override;
    bool apply() override;
    const slot_info_vec_t::value_type & get_base_write_slot() const;
    const slot_info_vec_t::value_type & get_swa_write_slot() const;
    const slot_info_vec_t::value_type & get_base_write_slot(size_t index) const;
    const slot_info_vec_t::value_type & get_swa_write_slot(size_t index) const;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const;
    uint32_t get_n_write() const;
    uint64_t get_n_backing_rows() const;
    uint64_t get_n_backing_rows(size_t index) const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst) const;
    void set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    size_t i_next = 0;

    llama_kv_cache * kv_base = nullptr;
    llama_kv_cache * kv_swa  = nullptr;

    const std::shared_ptr<void> resident_batch_lease;

    slot_info_vec_t sinfos_base_write;
    slot_info_vec_t sinfos_write;
    slot_info_vec_t sinfos_read;
    std::vector<llama_ubatch> ubatches;
    std::vector<llama_ubatch> ubatches_write;

    const llama_memory_context_ptr ctx_base_mem;
    const llama_memory_context_ptr ctx_swa_mem;

    uint32_t n_kv = 0;

    const llama_memory_status status;
};

// DSV4 compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class llama_kv_cache_dsv4_comp_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv, uint32_t logical_n_kv = 0);

    llama_kv_cache_dsv4_comp_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches,
            uint32_t logical_n_kv = 0);

    bool next();

    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_k_pool(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    llama_kv_cache * kv;

    size_t i_cur = 0;
    slot_info_vec_t sinfos;
    std::vector<llama_ubatch> ubatches;

    uint32_t n_kv;
};

class llama_kv_cache_dsv4_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    struct comp_plan {
        // Per-ubatch recipe for updating compressor state, committing completed
        // compressed rows, and masking the compressed attention source.

        // APE row ids, i.e. pos % ratio, for the compressor-state updates.
        std::vector<int32_t> state_pos;

        // Current-ubatch source row ids and unique persistent-state
        // destination row ids for deterministic ring-state updates.
        std::vector<int32_t> state_persist_src_idxs;
        std::vector<int32_t> state_persist_dst_idxs;

        // Device-side rollback restore copies snapshot planes back to the
        // current compressor-state plane before the graph reads it.
        std::vector<int32_t> state_restore_src_idxs;
        std::vector<int32_t> state_restore_dst_idxs;

        // Device-side rollback snapshots copy rows from the graph-local
        // [persistent_state | current_ubatch_scratch] tensor into rollback
        // planes after the graph has computed current-token compressor state.
        std::vector<int32_t> state_snapshot_src_idxs;
        std::vector<int32_t> state_snapshot_dst_idxs;

        // Flattened source row ids used for state-backed commits. Source rows
        // index the graph-local [persistent_state | current_ubatch_scratch]
        // tensor. For overlapped compression the first half is previous rows
        // and the second half is current rows; a final synthetic zero/-inf row
        // may be addressed for the first block's previous half.
        std::vector<int32_t> state_read_idxs;

        // Final compressed-cache row ids written by state-backed commits.
        // A non-boundary CSA/LID decode step can target a masked scratch row.
        std::vector<int64_t> state_write_idxs;

        // Original affine logical destinations and dummy markers are retained
        // until the aggregate-pool ticket resolves them to physical rows.
        std::vector<int64_t> state_write_logical_idxs;
        std::vector<uint8_t> state_write_dummy;

        // RoPE positions for state-backed commits.
        std::vector<int32_t> state_write_pos;

        // Number of completed compressed rows visible for each query token.
        std::vector<int32_t> n_visible;

        // Number of streams used by the attention graph for this ubatch.
        int64_t n_stream = 1;

        // Graph-width for compressed rows. This can be larger than n_visible
        // so masked padding rows do not force a new graph at every CSA block.
        int64_t n_kv = 0;

        // Column-major [logical_segment, graph_stream]. Empty for the legacy
        // affine compressed-cache layout.
        std::vector<int32_t> segment_ids;
    };

    llama_kv_cache_dsv4_context(llama_memory_status status);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            llama_context * lctx,
            bool optimize,
            stream_copy_info sc_info_csa,
            stream_copy_info sc_info_hca,
            stream_copy_info sc_info_lid);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            slot_info_vec_t sinfos_raw_base_write,
            slot_info_vec_t sinfos_raw_swa_write,
            slot_info_vec_t sinfos_raw_swa_read,
            std::vector<llama_ubatch> ubatches,
            std::vector<llama_ubatch> ubatches_raw);

    virtual ~llama_kv_cache_dsv4_context();

    //
    // llama_memory_context_i
    //

    bool next()      override;
    bool apply()     override;
    bool preflight() override;
    void finish(bool success) override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;
    bool get_kv_pressure(llama_kv_pressure_info & info) const override;

    //
    // llama_kv_cache_dsv4_context specific API
    //

    const llama_kv_cache_dsv4_raw_context * get_raw() const;
    const llama_kv_cache_dsv4_comp_context * get_csa() const;
    const llama_kv_cache_dsv4_comp_context * get_hca() const;
    const llama_kv_cache_dsv4_comp_context * get_lid() const;
    const llama_dsv4_comp_state       * get_csa_state() const;
    const llama_dsv4_comp_state       * get_hca_state() const;
    const llama_dsv4_comp_state       * get_lid_state() const;

    const comp_plan & get_csa_plan() const;
    const comp_plan & get_hca_plan() const;
    const comp_plan & get_lid_plan() const;

    const comp_plan & get_csa_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_hca_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_lid_plan(const llama_ubatch & ubatch) const;
    const llama_dsv4_batch_quote & get_last_batch_quote() const;

private:
    bool reserve_batch_ranges();
    bool collect_batch_ranges(std::vector<dsv4_sparse_range> & ranges);
    bool reserve_aggregate_pool(std::vector<llama_dsv4_comp_allocation> & cow_allocations);
    void rollback_aggregate_pool();

    llama_kv_cache_dsv4 * kv = nullptr;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    std::vector<comp_plan> plans_csa;
    std::vector<comp_plan> plans_hca;
    std::vector<comp_plan> plans_lid;

    const std::unique_ptr<llama_kv_cache_dsv4_raw_context> ctx_raw;
    const llama_memory_context_ptr ctx_csa_mem;
    const llama_memory_context_ptr ctx_hca_mem;
    const llama_memory_context_ptr ctx_lid_mem;

    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_csa;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_hca;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_lid;

    llama_dsv4_comp_state * csa_state = nullptr;
    llama_dsv4_comp_state * hca_state = nullptr;
    llama_dsv4_comp_state * lid_state = nullptr;

    stream_copy_info sc_info_csa;
    stream_copy_info sc_info_hca;
    stream_copy_info sc_info_lid;

    bool reserve_plans = false;
    mutable comp_plan reserve_plan_csa;
    mutable comp_plan reserve_plan_hca;
    mutable comp_plan reserve_plan_lid;

    llama_dsv4_batch_quote last_batch_quote;

    uint32_t last_pressure_family_mask = 0;
    bool batch_ranges_reserved = false;
    llama_dsv4_comp_ticket comp_ticket;
    bool comp_ticket_active = false;
    size_t comp_finished_ubatches = 0;

    llama_memory_status status;
};

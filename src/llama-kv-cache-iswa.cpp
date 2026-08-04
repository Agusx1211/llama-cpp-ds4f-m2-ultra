#include "llama-kv-cache-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace {

uint64_t llama_kv_iswa_allocate_pool_id() {
    static std::atomic<uint64_t> next{ 1 };
    uint64_t current = next.load(std::memory_order_relaxed);
    while (true) {
        if (current == 0 || current == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("ISWA resident pool identity space exhausted");
        }
        if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current;
        }
    }
}

struct llama_kv_iswa_resident_identity {
    uint64_t id = llama_kv_iswa_allocate_pool_id();
};

struct llama_kv_iswa_batch_lease {
    explicit llama_kv_iswa_batch_lease(std::shared_ptr<llama_kv_cache_resident_guard> guard) :
        guard(std::move(guard)) {
        ++this->guard->active_batches;
    }

    ~llama_kv_iswa_batch_lease() {
        std::lock_guard<std::recursive_mutex> lock(guard->mutex);
        GGML_ASSERT(guard->active_batches > 0);
        --guard->active_batches;
    }

    std::shared_ptr<llama_kv_cache_resident_guard> guard;
};

struct llama_kv_iswa_resident_record {
    llama_seq_id source_execution = -1;
    uint32_t slot = UINT32_MAX;
    uint64_t generation = 1;
    llama_kv_cells base_cells;
    llama_kv_cells swa_cells;
    uint32_t base_head = 0;
    uint32_t swa_head = 0;
};

struct sparse_move_api {
    ggml_dsv4_sparse_move_quote_fn quote = nullptr;
    ggml_dsv4_sparse_move_commit_fn commit = nullptr;
    ggml_dsv4_sparse_move_free_fn free = nullptr;
};

thread_local sparse_move_api sparse_move_test_override;
thread_local bool sparse_move_test_fail_next_allocation = false;

sparse_move_api llama_kv_iswa_sparse_move_api(const ggml_tensor * tensor) {
    if (sparse_move_test_override.quote != nullptr ||
            sparse_move_test_override.commit != nullptr ||
            sparse_move_test_override.free != nullptr) {
        return sparse_move_test_override;
    }
    sparse_move_api result;
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return result;
    }
    auto * buft = ggml_backend_buffer_get_type(tensor->buffer);
    auto * dev = ggml_backend_buft_get_device(buft);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg == nullptr) {
        return result;
    }
    result.quote = (ggml_dsv4_sparse_move_quote_fn) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_quote");
    result.commit = (ggml_dsv4_sparse_move_commit_fn) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_commit");
    result.free = (ggml_dsv4_sparse_move_free_fn) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_free");
    return result;
}

struct backend_move_quote {
    int status = GGML_DSV4_SPARSE_UNSUPPORTED;
    ggml_dsv4_sparse_move_commit_fn commit = nullptr;
    std::shared_ptr<void> value;
};

backend_move_quote llama_kv_iswa_quote_move(
        const std::vector<ggml_tensor *> & sources,
        const std::vector<ggml_tensor *> * destinations) {
    backend_move_quote result;
    if (sources.empty() || (destinations != nullptr && destinations->size() != sources.size())) {
        result.status = GGML_DSV4_SPARSE_INVALID;
        return result;
    }
    const sparse_move_api api = llama_kv_iswa_sparse_move_api(sources.front());
    if (api.quote == nullptr || api.commit == nullptr || api.free == nullptr) {
        return result;
    }
    void * raw = nullptr;
    ggml_tensor * const * destination_data = destinations != nullptr ? destinations->data() : nullptr;
    result.status = api.quote(sources.data(), destination_data, sources.size(), &raw);
    if (result.status != GGML_DSV4_SPARSE_OK || raw == nullptr) {
        if (raw != nullptr) {
            api.free(raw);
        }
        if (result.status == GGML_DSV4_SPARSE_OK) {
            result.status = GGML_DSV4_SPARSE_INVALID;
        }
        return result;
    }
    result.commit = api.commit;
    result.value = std::shared_ptr<void>(raw, api.free);
    return result;
}

enum class sparse_move_phase {
    detach_quote,
    detach_commit,
    attach_quote,
    attach_commit,
    release_quote,
    release_commit,
};

llama_kv_iswa_resident_status llama_kv_iswa_backend_status(int status, sparse_move_phase phase) {
    switch (status) {
        case GGML_DSV4_SPARSE_OK:
            return llama_kv_iswa_resident_status::ok;
        case GGML_DSV4_SPARSE_PRESSURE:
        case GGML_DSV4_SPARSE_OOM:
            return llama_kv_iswa_resident_status::resource_exhausted;
        case GGML_DSV4_SPARSE_STALE:
            return phase == sparse_move_phase::detach_quote || phase == sparse_move_phase::detach_commit ?
                    llama_kv_iswa_resident_status::stale_quote :
                    llama_kv_iswa_resident_status::backend_error;
        case GGML_DSV4_SPARSE_INVALID:
        case GGML_DSV4_SPARSE_UNSUPPORTED:
            return phase == sparse_move_phase::detach_quote ?
                    llama_kv_iswa_resident_status::unsupported_layout :
                    llama_kv_iswa_resident_status::backend_error;
    }
    return llama_kv_iswa_resident_status::backend_error;
}

} // namespace

void llama_kv_iswa_set_resident_backend_override_for_test(
        llama_kv_iswa_resident_backend_override backend) {
    sparse_move_test_override = { backend.quote, backend.commit, backend.free };
}

void llama_kv_iswa_fail_next_resident_allocation_for_test() {
    sparse_move_test_fail_next_allocation = true;
}

struct llama_kv_cache_iswa::resident_impl {
    explicit resident_impl(uint32_t capacity) : slots(capacity, false) {}

    std::shared_ptr<llama_kv_cache_resident_guard> guard =
            std::make_shared<llama_kv_cache_resident_guard>();
    std::shared_ptr<const llama_kv_iswa_resident_identity> identity =
            std::make_shared<llama_kv_iswa_resident_identity>();
    uint64_t epoch = 1;
    uint64_t                                          next_handle_id = 1;
    std::vector<bool>                                 slots;
    std::map<uint64_t, llama_kv_iswa_resident_record> handles;
};

struct llama_kv_iswa_resident_detach_plan {
    std::shared_ptr<const llama_kv_iswa_resident_identity> owner;
    uint64_t                                               epoch         = 0;
    uint64_t                                               version[2]    = { 0, 0 };
    llama_seq_id                                           execution_id  = -1;
    uint32_t                                               resident_slot = UINT32_MAX;
    llama_kv_iswa_resident_handle                          resident;
    backend_move_quote                                     backend;
    std::map<uint64_t, llama_kv_iswa_resident_record>      prepared;
};

enum class llama_kv_iswa_resident_attach_state : uint8_t {
    prepared,
    committed,
    cancelled,
};

struct llama_kv_iswa_resident_attach_plan {
    std::shared_ptr<const llama_kv_iswa_resident_identity> owner;
    uint64_t                                               epoch      = 0;
    uint64_t                                               version[2] = { 0, 0 };
    llama_seq_id                                           execution_id = -1;
    llama_kv_iswa_resident_handle                          resident;
    uint32_t                                               resident_slot = UINT32_MAX;
    uint32_t                                               base_head = 0;
    uint32_t                                               swa_head  = 0;
    llama_kv_cells                                         base_cells;
    llama_kv_cells                                         swa_cells;
    backend_move_quote                                     backend;
    llama_kv_iswa_resident_attach_state                    state =
            llama_kv_iswa_resident_attach_state::prepared;
};

const char * llama_kv_iswa_resident_status_name(llama_kv_iswa_resident_status status) {
    switch (status) {
        case llama_kv_iswa_resident_status::ok:
            return "ok";
        case llama_kv_iswa_resident_status::invalid_sequence:     return "invalid_sequence";
        case llama_kv_iswa_resident_status::unsupported_layout:   return "unsupported_layout";
        case llama_kv_iswa_resident_status::slot_occupied:        return "slot_occupied";
        case llama_kv_iswa_resident_status::capacity_exhausted:   return "capacity_exhausted";
        case llama_kv_iswa_resident_status::stale_quote:          return "stale_quote";
        case llama_kv_iswa_resident_status::stale_handle:         return "stale_handle";
        case llama_kv_iswa_resident_status::generation_exhausted: return "generation_exhausted";
        case llama_kv_iswa_resident_status::not_quiescent:         return "not_quiescent";
        case llama_kv_iswa_resident_status::resource_exhausted:    return "resource_exhausted";
        case llama_kv_iswa_resident_status::backend_error:        return "backend_error";
    }
    return "unknown";
}

//
// llama_kv_cache_iswa
//

llama_kv_cache_iswa::llama_kv_cache_iswa(
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
    ggml_backend_buffer_type_t buft_override,
                uint32_t   n_resident) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_ubatch, n_pad, mem_other, filter, reuse, share, buft_override, n_resident) {
}

llama_kv_cache_iswa::llama_kv_cache_iswa(
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
    ggml_backend_buffer_type_t buft_override,
                uint32_t   n_resident) : unified(unified) {

    // chain filters
    const layer_filter_cb filter_base = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return !model.hparams.is_swa(il);
    };

    const layer_filter_cb filter_swa  = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return  model.hparams.is_swa(il);
    };

    const uint32_t size_base = kv_size;

    // note: the SWA cache is always padded to 256 for performance
    //       https://github.com/ggml-org/llama.cpp/issues/17037
    uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);

    // when using full-size SWA cache, we set the SWA cache size to be equal to the base cache size
    if (swa_full) {
        LLAMA_LOG_WARN("%s: using full-size SWA cache (ref: %s)\n",
                __func__, "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");

        size_swa = size_base;
    }

    LLAMA_LOG_INFO("%s: creating non-SWA KV cache, size = %u cells\n", __func__, size_base);

    llama_memory_t mem_other_base = nullptr;
    if (mem_other) {
        mem_other_base = static_cast<llama_kv_cache_iswa *>(mem_other)->get_base();
    }

    llama_memory_t mem_other_swa = nullptr;
    if (mem_other) {
        mem_other_swa = static_cast<llama_kv_cache_iswa *>(mem_other)->get_swa();
    }

    kv_base = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_base, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, mem_other_base, filter_base, reuse, share, buft_override, n_resident);

    LLAMA_LOG_INFO("%s: creating     SWA KV cache, size = %u cells\n", __func__, size_swa);

    kv_swa = std::make_unique<llama_kv_cache>(
            model, hparams, type_k, type_v,
            v_trans, offload, unified, size_swa, n_seq_max, n_pad,
            hparams.n_swa, hparams.swa_type, mem_other_swa, filter_swa, reuse, share, buft_override, n_resident);

    if (n_resident > 0) {
        resident = std::make_unique<resident_impl>(n_resident);
        kv_base->resident_bind_guard(resident->guard, 0);
        kv_swa ->resident_bind_guard(resident->guard, 1);
    }
}

llama_kv_cache_iswa::~llama_kv_cache_iswa() {
    if (resident && !resident->handles.empty()) {
        std::terminate();
    }
}

std::unique_lock<std::recursive_mutex> llama_kv_cache_iswa::lock_resident() const {
    if (resident) {
        return std::unique_lock<std::recursive_mutex>(resident->guard->mutex);
    }
    return {};
}

bool llama_kv_cache_iswa::resident_is_quiescent() const {
    return resident && kv_base->sc_info.empty() && kv_swa->sc_info.empty() &&
            resident->guard->pending_updates[0] == 0 && resident->guard->pending_updates[1] == 0 &&
            !resident->guard->update_active[0] && !resident->guard->update_active[1] &&
            resident->guard->active_batches == 0;
}

std::shared_ptr<void> llama_kv_cache_iswa::acquire_resident_batch_lease() const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident) {
        return {};
    }
    if (resident->guard->resident_transaction_active) {
        throw std::runtime_error("cannot begin ISWA graph while a resident transaction is active");
    }
    return std::make_shared<llama_kv_iswa_batch_lease>(resident->guard);
}

bool llama_kv_cache_iswa::begin_resident_transaction() const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident || resident->guard->resident_transaction_active || !resident_is_quiescent()) {
        return false;
    }
    resident->guard->resident_transaction_active = true;
    return true;
}

void llama_kv_cache_iswa::end_resident_transaction() const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident || !resident->guard->resident_transaction_active) {
        std::terminate();
    }
    resident->guard->resident_transaction_active = false;
}

bool llama_kv_cache_iswa::has_resident_handles() const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    return resident && resident->guard->parked_handles != 0;
}

void llama_kv_cache_iswa::clear(bool data) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (data && resident && !resident->handles.empty()) {
        throw std::runtime_error("cannot clear ISWA backing storage with resident handles");
    }
    kv_base->clear(data);
    kv_swa ->clear(data);
}

bool llama_kv_cache_iswa::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    bool res = true;

    res = res & kv_base->seq_rm(seq_id, p0, p1);
    res = res & kv_swa ->seq_rm(seq_id, p0, p1);

    return res;
}

void llama_kv_cache_iswa::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    kv_base->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    kv_swa ->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_iswa::seq_keep(llama_seq_id seq_id) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    kv_base->seq_keep(seq_id);
    kv_swa ->seq_keep(seq_id);
}

void llama_kv_cache_iswa::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    kv_base->seq_add(seq_id, p0, p1, shift);
    kv_swa ->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_iswa::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    kv_base->seq_div(seq_id, p0, p1, d);
    kv_swa ->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_iswa::seq_pos_min(llama_seq_id seq_id) const {
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    return kv_swa->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_iswa::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_base->memory_breakdown();
    for (const auto & buft_size : kv_swa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

llama_memory_context_ptr llama_kv_cache_iswa::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    [[maybe_unused]] auto prepare_lease = acquire_resident_batch_lease();
    GGML_UNUSED(embd_all);

    // first try simple split
    do {
        if (!unified) {
            // requires equal splits, so we skip the simple split
            break;
        }

        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // if it fails, try equal split
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_equal(n_ubatch, !unified, 0);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos_base = kv_base->prepare(ubatches);
        if (sinfos_base.empty()) {
            break;
        }

        auto sinfos_swa = kv_swa->prepare(ubatches);
        if (sinfos_swa.empty()) {
            break;
        }

        assert(sinfos_base.size() == sinfos_swa.size());

        return std::make_unique<llama_kv_cache_iswa_context>(
                this, std::move(sinfos_base), std::move(sinfos_swa), std::move(ubatches));
    } while (false);

    // TODO: if we fail again, we should attempt different splitting strategies
    //       but to do that properly, we first have to refactor the batches to be more flexible

    return std::make_unique<llama_kv_cache_iswa_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_full() {
    return std::make_unique<llama_kv_cache_iswa_context>(this);
}

llama_memory_context_ptr llama_kv_cache_iswa::init_update(llama_context * lctx, bool optimize) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    return std::make_unique<llama_kv_cache_iswa_context>(this, lctx, optimize);
}

bool llama_kv_cache_iswa::get_can_shift() const {
    return kv_base->get_can_shift() &&
           kv_swa->get_can_shift() &&
           kv_base->get_size() == kv_swa->get_size();
}

void llama_kv_cache_iswa::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (seq_id < 0 && has_resident_handles()) {
        throw std::runtime_error("ISWA resident handles are not serializable");
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_write(io, seq_id, flags);
    }

    kv_swa->state_write(io, seq_id, flags);
}

void llama_kv_cache_iswa::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (seq_id < 0 && has_resident_handles()) {
        throw std::runtime_error("ISWA resident handles cannot survive whole-cache restore");
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        kv_base->state_read(io, seq_id, flags);
    }

    kv_swa->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_kv_cache_iswa::get_base() const {
    return kv_base.get();
}

llama_kv_cache * llama_kv_cache_iswa::get_swa() const {
    return kv_swa.get();
}

llama_kv_iswa_resident_detach_quote llama_kv_cache_iswa::quote_resident_detach(llama_seq_id execution_id) const {
    llama_kv_iswa_resident_detach_quote result;
    result.execution_id = execution_id;
    try {
        [[maybe_unused]] auto resident_scope = lock_resident();
        if (!resident || execution_id < 0 || (uint32_t) execution_id >= kv_base->get_n_stream()) {
            result.status = execution_id < 0 || (uint32_t) execution_id >= kv_base->get_n_stream() ?
                                llama_kv_iswa_resident_status::invalid_sequence :
                                llama_kv_iswa_resident_status::unsupported_layout;
            return result;
        }
        if (!resident_is_quiescent()) {
            result.status = llama_kv_iswa_resident_status::not_quiescent;
            return result;
        }
        if (!kv_base->resident_execution_layout(execution_id) || !kv_swa->resident_execution_layout(execution_id)) {
            result.status = llama_kv_iswa_resident_status::unsupported_layout;
            return result;
        }
        if (resident->next_handle_id == 0) {
            result.status = llama_kv_iswa_resident_status::generation_exhausted;
            return result;
        }

        const auto free_slot = std::find(resident->slots.begin(), resident->slots.end(), false);
        if (free_slot == resident->slots.end()) {
            result.status = llama_kv_iswa_resident_status::capacity_exhausted;
            return result;
        }
        const uint32_t slot = (uint32_t) std::distance(resident->slots.begin(), free_slot);

        std::vector<ggml_tensor *> sources;
        std::vector<ggml_tensor *> destinations;
        kv_base->resident_append_views((uint32_t) execution_id, false, sources);
        kv_swa->resident_append_views((uint32_t) execution_id, false, sources);
        kv_base->resident_append_views(slot, true, destinations);
        kv_swa->resident_append_views(slot, true, destinations);
        auto backend = llama_kv_iswa_quote_move(sources, &destinations);
        if (!backend.value) {
            result.status = llama_kv_iswa_backend_status(backend.status, sparse_move_phase::detach_quote);
            return result;
        }
        if (sparse_move_test_fail_next_allocation) {
            sparse_move_test_fail_next_allocation = false;
            throw std::bad_alloc();
        }

        auto plan           = std::make_shared<llama_kv_iswa_resident_detach_plan>();
        plan->owner         = resident->identity;
        plan->epoch         = resident->epoch;
        plan->version[0]    = resident->guard->version[0];
        plan->version[1]    = resident->guard->version[1];
        plan->execution_id  = execution_id;
        plan->resident_slot = slot;
        plan->resident      = { resident->identity->id, resident->next_handle_id, 1 };
        plan->backend       = std::move(backend);

        llama_kv_iswa_resident_record record;
        record.source_execution = execution_id;
        record.slot             = slot;
        record.generation       = plan->resident.generation;
        record.base_cells       = kv_base->resident_copy_cells(execution_id);
        record.swa_cells        = kv_swa->resident_copy_cells(execution_id);
        record.base_head        = kv_base->resident_copy_head(execution_id);
        record.swa_head         = kv_swa->resident_copy_head(execution_id);
        plan->prepared.emplace(plan->resident.id, std::move(record));

        result.status        = llama_kv_iswa_resident_status::ok;
        result.resident_slot = slot;
        result.resident      = plan->resident;
        result.plan          = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_kv_iswa_resident_status::resource_exhausted;
        result.plan.reset();
        return result;
    }
}

llama_kv_iswa_resident_result llama_kv_cache_iswa::detach_resident(const llama_kv_iswa_resident_detach_quote & quote) {
    llama_kv_iswa_resident_result result;
    try {
        [[maybe_unused]] auto resident_scope = lock_resident();
        if (!resident || !quote.plan || quote.plan->owner != resident->identity ||
            quote.plan->epoch != resident->epoch || quote.plan->resident.id != resident->next_handle_id ||
            quote.plan->version[0] != resident->guard->version[0] ||
            quote.plan->version[1] != resident->guard->version[1] || !resident_is_quiescent() ||
            quote.plan->resident_slot >= resident->slots.size() ||
            resident->slots[quote.plan->resident_slot] || quote.plan->prepared.empty()) {
            result.status = llama_kv_iswa_resident_status::stale_quote;
            return result;
        }
        const int backend_status = quote.plan->backend.commit(quote.plan->backend.value.get());
        if (backend_status != GGML_DSV4_SPARSE_OK) {
            result.status = llama_kv_iswa_backend_status(backend_status, sparse_move_phase::detach_commit);
            return result;
        }

        auto node = quote.plan->prepared.extract(quote.plan->resident.id);
        GGML_ASSERT(!node.empty());
        kv_base->resident_clear_execution(quote.plan->execution_id);
        kv_swa->resident_clear_execution(quote.plan->execution_id);
        resident->slots[quote.plan->resident_slot] = true;
        const auto inserted                        = resident->handles.insert(std::move(node));
        GGML_ASSERT(inserted.inserted);
        ++resident->guard->parked_handles;
        resident->next_handle_id =
            quote.plan->resident.id == std::numeric_limits<uint64_t>::max() ? 0 : quote.plan->resident.id + 1;
        ++resident->epoch;

        result.status   = llama_kv_iswa_resident_status::ok;
        result.resident = quote.plan->resident;
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_kv_iswa_resident_status::resource_exhausted;
        return result;
    }
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::validate_resident_detach(
        const llama_kv_iswa_resident_detach_quote & quote) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident || !quote.plan || quote.plan->owner != resident->identity ||
        quote.plan->epoch != resident->epoch || quote.plan->resident.id != resident->next_handle_id ||
        quote.plan->version[0] != resident->guard->version[0] ||
        quote.plan->version[1] != resident->guard->version[1] || !resident_is_quiescent() ||
        quote.plan->resident_slot >= resident->slots.size() || resident->slots[quote.plan->resident_slot] ||
        quote.plan->prepared.empty()) {
        return llama_kv_iswa_resident_status::stale_quote;
    }
    return llama_kv_iswa_resident_status::ok;
}

llama_kv_iswa_resident_attach_quote llama_kv_cache_iswa::quote_resident_attach(
        llama_kv_iswa_resident_handle handle,
        llama_seq_id                  execution_id) const {
    llama_kv_iswa_resident_attach_quote result;
    result.execution_id = execution_id;
    result.resident     = handle;
    try {
        [[maybe_unused]] auto resident_scope = lock_resident();
        if (!resident || handle.pool_id != resident->identity->id || handle.id == 0) {
            result.status = llama_kv_iswa_resident_status::stale_handle;
            return result;
        }
        auto it = resident->handles.find(handle.id);
        if (it == resident->handles.end() || it->second.generation != handle.generation) {
            result.status = llama_kv_iswa_resident_status::stale_handle;
            return result;
        }
        if (execution_id < 0 || (uint32_t) execution_id >= kv_base->get_n_stream()) {
            result.status = llama_kv_iswa_resident_status::invalid_sequence;
            return result;
        }
        if (!resident_is_quiescent()) {
            result.status = llama_kv_iswa_resident_status::not_quiescent;
            return result;
        }
        if (!kv_base->resident_execution_layout(execution_id) || !kv_swa->resident_execution_layout(execution_id)) {
            result.status = llama_kv_iswa_resident_status::unsupported_layout;
            return result;
        }
        if (!kv_base->resident_execution_empty(execution_id) || !kv_swa->resident_execution_empty(execution_id)) {
            result.status = llama_kv_iswa_resident_status::slot_occupied;
            return result;
        }

        llama_kv_cells base_cells = it->second.base_cells;
        llama_kv_cells swa_cells  = it->second.swa_cells;
        if (!base_cells.seq_replace(it->second.source_execution, execution_id) ||
            !swa_cells.seq_replace(it->second.source_execution, execution_id)) {
            result.status = llama_kv_iswa_resident_status::backend_error;
            return result;
        }

        std::vector<ggml_tensor *> sources;
        std::vector<ggml_tensor *> destinations;
        kv_base->resident_append_views(it->second.slot, true, sources);
        kv_swa->resident_append_views(it->second.slot, true, sources);
        kv_base->resident_append_views((uint32_t) execution_id, false, destinations);
        kv_swa->resident_append_views((uint32_t) execution_id, false, destinations);
        auto backend = llama_kv_iswa_quote_move(sources, &destinations);
        if (!backend.value) {
            result.status = llama_kv_iswa_backend_status(backend.status, sparse_move_phase::attach_quote);
            return result;
        }
        if (sparse_move_test_fail_next_allocation) {
            sparse_move_test_fail_next_allocation = false;
            throw std::bad_alloc();
        }

        auto plan            = std::make_shared<llama_kv_iswa_resident_attach_plan>();
        plan->owner          = resident->identity;
        plan->epoch          = resident->epoch;
        plan->version[0]     = resident->guard->version[0];
        plan->version[1]     = resident->guard->version[1];
        plan->execution_id   = execution_id;
        plan->resident       = handle;
        plan->resident_slot  = it->second.slot;
        plan->base_head      = it->second.base_head;
        plan->swa_head       = it->second.swa_head;
        plan->base_cells     = std::move(base_cells);
        plan->swa_cells      = std::move(swa_cells);
        plan->backend        = std::move(backend);

        result.status = llama_kv_iswa_resident_status::ok;
        result.plan   = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_kv_iswa_resident_status::resource_exhausted;
        result.plan.reset();
        return result;
    }
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::commit_resident_attach_final(
        const llama_kv_iswa_resident_attach_quote & quote,
        llama_kv_iswa_resident_final_step final_step) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (final_step != llama_kv_iswa_resident_final_step::confirmed || !resident || !quote.plan ||
        quote.plan->owner != resident->identity ||
        quote.plan->state != llama_kv_iswa_resident_attach_state::prepared ||
        quote.plan->epoch != resident->epoch ||
        quote.plan->version[0] != resident->guard->version[0] ||
        quote.plan->version[1] != resident->guard->version[1] || !resident_is_quiescent() ||
        quote.plan->execution_id < 0 ||
        (uint32_t) quote.plan->execution_id >= kv_base->get_n_stream() ||
        !kv_base->resident_execution_empty(quote.plan->execution_id) ||
        !kv_swa->resident_execution_empty(quote.plan->execution_id)) {
        return llama_kv_iswa_resident_status::stale_quote;
    }

    auto it = resident->handles.find(quote.plan->resident.id);
    if (it == resident->handles.end() || it->second.generation != quote.plan->resident.generation ||
        it->second.slot != quote.plan->resident_slot || quote.plan->resident_slot >= resident->slots.size() ||
        !resident->slots[quote.plan->resident_slot]) {
        return llama_kv_iswa_resident_status::stale_quote;
    }

    const int backend_status = quote.plan->backend.commit(quote.plan->backend.value.get());
    if (backend_status != GGML_DSV4_SPARSE_OK) {
        return llama_kv_iswa_backend_status(backend_status, sparse_move_phase::attach_commit);
    }

    kv_base->resident_restore_execution(
            quote.plan->execution_id, std::move(quote.plan->base_cells), quote.plan->base_head);
    kv_swa->resident_restore_execution(
            quote.plan->execution_id, std::move(quote.plan->swa_cells), quote.plan->swa_head);
    resident->slots[quote.plan->resident_slot] = false;
    resident->handles.erase(it);
    GGML_ASSERT(resident->guard->parked_handles > 0);
    --resident->guard->parked_handles;
    ++resident->epoch;
    quote.plan->state = llama_kv_iswa_resident_attach_state::committed;
    return llama_kv_iswa_resident_status::ok;
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::validate_resident_attach(
        const llama_kv_iswa_resident_attach_quote & quote) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident || !quote.plan || quote.plan->owner != resident->identity ||
        quote.plan->state != llama_kv_iswa_resident_attach_state::prepared ||
        quote.plan->epoch != resident->epoch || quote.plan->version[0] != resident->guard->version[0] ||
        quote.plan->version[1] != resident->guard->version[1] || !resident_is_quiescent() ||
        quote.plan->execution_id < 0 || (uint32_t) quote.plan->execution_id >= kv_base->get_n_stream() ||
        !kv_base->resident_execution_empty(quote.plan->execution_id) ||
        !kv_swa->resident_execution_empty(quote.plan->execution_id)) {
        return llama_kv_iswa_resident_status::stale_quote;
    }
    const auto it = resident->handles.find(quote.plan->resident.id);
    if (it == resident->handles.end() || it->second.generation != quote.plan->resident.generation ||
        it->second.slot != quote.plan->resident_slot || quote.plan->resident_slot >= resident->slots.size() ||
        !resident->slots[quote.plan->resident_slot]) {
        return llama_kv_iswa_resident_status::stale_quote;
    }
    return llama_kv_iswa_resident_status::ok;
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::rollback_resident_attach(
        const llama_kv_iswa_resident_attach_quote & quote) {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!resident || !quote.plan || quote.plan->owner != resident->identity) {
        return llama_kv_iswa_resident_status::stale_quote;
    }
    if (quote.plan->state == llama_kv_iswa_resident_attach_state::cancelled) {
        return llama_kv_iswa_resident_status::ok;
    }
    if (quote.plan->state == llama_kv_iswa_resident_attach_state::committed) {
        // The backend has no reverse move ticket. Never pretend that a
        // successful final-step commit was rolled back.
        return llama_kv_iswa_resident_status::stale_quote;
    }
    quote.plan->state = llama_kv_iswa_resident_attach_state::cancelled;
    return llama_kv_iswa_resident_status::ok;
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::attach_resident(
        llama_kv_iswa_resident_handle handle,
        llama_seq_id                  execution_id) {
    const auto quote = quote_resident_attach(handle, execution_id);
    if (quote.status != llama_kv_iswa_resident_status::ok) {
        return quote.status;
    }
    return commit_resident_attach_final(quote, llama_kv_iswa_resident_final_step::confirmed);
}

llama_kv_iswa_resident_status llama_kv_cache_iswa::release_resident(llama_kv_iswa_resident_handle handle) {
    try {
        [[maybe_unused]] auto resident_scope = lock_resident();
        if (!resident || handle.pool_id != resident->identity->id || handle.id == 0) {
            return llama_kv_iswa_resident_status::stale_handle;
        }
        auto it = resident->handles.find(handle.id);
        if (it == resident->handles.end() || it->second.generation != handle.generation) {
            return llama_kv_iswa_resident_status::stale_handle;
        }

        std::vector<ggml_tensor *> sources;
        kv_base->resident_append_views(it->second.slot, true, sources);
        kv_swa->resident_append_views(it->second.slot, true, sources);
        auto backend = llama_kv_iswa_quote_move(sources, nullptr);
        if (!backend.value) {
            return llama_kv_iswa_backend_status(backend.status, sparse_move_phase::release_quote);
        }
        const int backend_status = backend.commit(backend.value.get());
        if (backend_status != GGML_DSV4_SPARSE_OK) {
            return llama_kv_iswa_backend_status(backend_status, sparse_move_phase::release_commit);
        }

        const uint32_t slot   = it->second.slot;
        resident->slots[slot] = false;
        resident->handles.erase(it);
        GGML_ASSERT(resident->guard->parked_handles > 0);
        --resident->guard->parked_handles;
        ++resident->epoch;
        return llama_kv_iswa_resident_status::ok;
    } catch (const std::bad_alloc &) {
        return llama_kv_iswa_resident_status::resource_exhausted;
    }
}

//
// llama_kv_cache_iswa_context
//

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(llama_memory_status status) : status(status) {}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv) :
    kv(kv),
    resident_batch_lease(kv->acquire_resident_batch_lease()),
    ctx_base(kv->get_base()->init_full()),
    ctx_swa (kv->get_swa ()->init_full()),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv(kv),
    resident_batch_lease(),
    ctx_base(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa (kv->get_swa ()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context::llama_kv_cache_iswa_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    kv(kv),
    ubatches(std::move(ubatches)),
    resident_batch_lease(kv->acquire_resident_batch_lease()),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_base(new llama_kv_cache_context(kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa (new llama_kv_cache_context(kv->get_swa (), std::move(sinfos_swa),  this->ubatches)),
    status(llama_memory_status_combine(ctx_base->get_status(), ctx_swa->get_status())) {
}

llama_kv_cache_iswa_context:: ~llama_kv_cache_iswa_context() = default;

bool llama_kv_cache_iswa_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_base->next();
    ctx_swa ->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_iswa_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    [[maybe_unused]] auto resident_scope = kv->lock_resident();

    bool res = true;

    res = res & ctx_base->apply();
    res = res & ctx_swa ->apply();

    return res;
}

llama_memory_status llama_kv_cache_iswa_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_iswa_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_base() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_base.get());
}

const llama_kv_cache_context * llama_kv_cache_iswa_context::get_swa()  const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return static_cast<const llama_kv_cache_context *>(ctx_swa.get());
}

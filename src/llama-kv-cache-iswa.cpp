#include "llama-kv-cache-iswa.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>

namespace {

uint64_t logical_hash_u64(uint64_t hash, uint64_t value) {
    for (uint32_t i = 0; i < 8; ++i) {
        hash ^= (value >> (8*i)) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t logical_hash_bytes(uint64_t hash, const uint8_t * data, size_t size) {
    hash = logical_hash_u64(hash, size);
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t logical_plane_checksum(const llama_kv_iswa_logical_plane_state & state) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = logical_hash_u64(hash, state.schema_version);
    hash = logical_hash_u64(hash, state.positions.size());
    for (llama_pos pos : state.positions) {
        hash = logical_hash_u64(hash, static_cast<uint64_t>(pos));
    }
    hash = logical_hash_u64(hash, state.extensions.size());
    for (const llama_kv_cell_ext & ext : state.extensions) {
        hash = logical_hash_u64(hash, static_cast<uint64_t>(ext.x));
        hash = logical_hash_u64(hash, static_cast<uint64_t>(ext.y));
    }
    return logical_hash_bytes(hash, state.tensor_payload.data(), state.tensor_payload.size());
}

class logical_vector_writer final : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        if (size > std::numeric_limits<size_t>::max() - data.size()) {
            throw std::overflow_error("ISWA logical state size overflow");
        }
        const size_t old_size = data.size();
        data.resize(old_size + size);
        if (size != 0) {
            std::memcpy(data.data() + old_size, src, size);
        }
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > std::numeric_limits<size_t>::max() - data.size()) {
            throw std::overflow_error("ISWA logical tensor state size overflow");
        }
        const size_t old_size = data.size();
        data.resize(old_size + size);
        if (size != 0) {
            ggml_backend_tensor_get(tensor, data.data() + old_size, offset, size);
        }
    }

    size_t n_bytes() override { return data.size(); }

    std::vector<uint8_t> data;
};

class logical_vector_reader final : public llama_io_read_i {
public:
    explicit logical_vector_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        require(size);
        if (size != 0) {
            std::memcpy(dst, data.data() + offset, size);
        }
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        require(size);
        if (size != 0) {
            ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        }
        offset += size;
    }

    void skip(size_t size) {
        require(size);
        offset += size;
    }

    size_t n_bytes() override { return offset; }

private:
    void require(size_t size) const {
        if (offset > data.size() || size > data.size() - offset) {
            throw std::runtime_error("truncated ISWA logical tensor payload");
        }
    }

    const std::vector<uint8_t> & data;
    size_t                       offset = 0;
};

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
    ggml_dsv4_sparse_move_audit_fn audit = nullptr;
};

thread_local sparse_move_api sparse_move_test_override;
thread_local bool sparse_move_test_fail_next_allocation = false;

struct resident_lock_test_hook {
    llama_kv_iswa_resident_lock_hook_for_test callback = nullptr;
    void *                                    context  = nullptr;
};

thread_local resident_lock_test_hook resident_lock_hook;

sparse_move_api llama_kv_iswa_sparse_move_api(const ggml_tensor * tensor, bool need_audit) {
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
    if (need_audit) {
        result.audit = (ggml_dsv4_sparse_move_audit_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_metal_dsv4_sparse_move_audit");
    }
    return result;
}

struct backend_move_quote {
    int status = GGML_DSV4_SPARSE_UNSUPPORTED;
    ggml_dsv4_sparse_move_commit_fn commit = nullptr;
    ggml_dsv4_sparse_move_audit_fn audit = nullptr;
    std::shared_ptr<void> value;
};

backend_move_quote llama_kv_iswa_quote_move(
        const std::vector<ggml_tensor *> & sources,
        const std::vector<ggml_tensor *> * destinations,
        bool need_audit = false) {
    backend_move_quote result;
    if (sources.empty() || (destinations != nullptr && destinations->size() != sources.size())) {
        result.status = GGML_DSV4_SPARSE_INVALID;
        return result;
    }
    const sparse_move_api api = llama_kv_iswa_sparse_move_api(sources.front(), need_audit);
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
    result.audit = api.audit;
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

struct resident_move_audit_snapshot {
    int status = GGML_DSV4_SPARSE_UNSUPPORTED;
    size_t n_pools = 0;
    std::array<ggml_dsv4_sparse_move_audit_pool, GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS> pools = {};
};

resident_move_audit_snapshot llama_kv_iswa_capture_move_audit(
        const backend_move_quote & backend,
        int committed) {
    resident_move_audit_snapshot result;
    if (!backend.audit || !backend.value) {
        return result;
    }
    ggml_dsv4_sparse_move_audit audit = {};
    try {
        result.status = backend.audit(backend.value.get(), committed, &audit);
    } catch (...) {
        result.status = GGML_DSV4_SPARSE_UNSUPPORTED;
        return result;
    }
    if (result.status != GGML_DSV4_SPARSE_OK ||
            audit.n_pools > GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS) {
        result.n_pools = 0;
        return result;
    }
    result.n_pools = audit.n_pools;
    std::copy(audit.pools, audit.pools + audit.n_pools, result.pools.begin());
    return result;
}

void llama_kv_iswa_publish_move_audit(
        const resident_move_audit_snapshot & before,
        const resident_move_audit_snapshot & after,
        llama_kv_iswa_resident_release_audit * result) {
    if (result == nullptr) {
        return;
    }
    result->before_status = before.status;
    result->after_status  = after.status;
    result->pools.clear();
    result->observed = false;
    if (before.status != GGML_DSV4_SPARSE_OK || after.status != GGML_DSV4_SPARSE_OK) {
        return;
    }
    result->pools.reserve(GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS);
    for (size_t i = 0; i < before.n_pools; ++i) {
        const auto & before_pool = before.pools[i];
        const auto it = std::find_if(after.pools.begin(), after.pools.begin() + after.n_pools,
                                     [&](const auto & pool) { return pool.pool_id == before_pool.pool_id; });
        if (it == after.pools.begin() + after.n_pools) {
            result->pools.clear();
            return;
        }
        result->pools.push_back({ before_pool, *it });
    }
    if (result->pools.size() != after.n_pools) {
        result->pools.clear();
        return;
    }
    result->observed = true;
}

} // namespace

enum class llama_kv_iswa_logical_import_state : uint8_t {
    prepared,
    committed,
};

struct llama_kv_iswa_logical_import_plan {
    struct plane {
        uint32_t                  stream = UINT32_MAX;
        uint32_t                  head = 0;
        llama_kv_cells            cells;
        llama_kv_cache::slot_info slot;
    };

    const llama_kv_cache_iswa * owner = nullptr;
    llama_seq_id                destination = -1;
    uint64_t                    version[2] = { 0, 0 };
    uint64_t                    transaction_generation = 0;
    uint64_t                    base_checksum = 0;
    uint64_t                    swa_checksum = 0;
    plane                       base;
    plane                       swa;
    llama_kv_iswa_logical_import_state state = llama_kv_iswa_logical_import_state::prepared;
};

void llama_kv_iswa_set_resident_backend_override_for_test(
        llama_kv_iswa_resident_backend_override backend) {
    sparse_move_test_override = { backend.quote, backend.commit, backend.free, backend.audit };
}

void llama_kv_iswa_fail_next_resident_allocation_for_test() {
    sparse_move_test_fail_next_allocation = true;
}

void llama_kv_iswa_set_resident_lock_hook_for_test(llama_kv_iswa_resident_lock_hook_for_test hook, void * context) {
    resident_lock_hook = { hook, context };
}

struct llama_kv_cache_iswa::resident_impl {
    resident_impl(uint32_t capacity, std::shared_ptr<llama_kv_cache_resident_guard> guard) :
        guard(std::move(guard)), slots(capacity, false) {
        GGML_ASSERT(this->guard != nullptr);
    }

    std::shared_ptr<llama_kv_cache_resident_guard> guard;
    std::shared_ptr<const llama_kv_iswa_resident_identity> identity =
            std::make_shared<llama_kv_iswa_resident_identity>();
    uint64_t epoch = 1;
    uint64_t                                          next_handle_id = 1;
    std::vector<bool>                                 slots;
    std::map<uint64_t, llama_kv_iswa_resident_record> handles;
};

llama_kv_iswa_resident_transaction::llama_kv_iswa_resident_transaction(
    std::shared_ptr<llama_kv_cache_resident_guard> guard,
    std::unique_lock<std::recursive_mutex>         lock) :
    guard(std::move(guard)),
    owner_thread(std::this_thread::get_id()),
    active(true) {
    GGML_ASSERT(this->guard != nullptr && lock.owns_lock());
    GGML_ASSERT(!this->guard->resident_transaction_active);
    if (this->guard->resident_transaction_generation == std::numeric_limits<uint64_t>::max()) {
        std::terminate();
    }
    generation = ++this->guard->resident_transaction_generation;
    this->guard->resident_transaction_active = true;
    this->guard->resident_transaction_owner  = owner_thread;
}

llama_kv_iswa_resident_transaction::llama_kv_iswa_resident_transaction(
    llama_kv_iswa_resident_transaction && other) noexcept :
    guard(std::move(other.guard)),
    owner_thread(other.owner_thread),
    generation(other.generation),
    active(other.active) {
    other.active = false;
}

llama_kv_iswa_resident_transaction::~llama_kv_iswa_resident_transaction() {
    release();
}

bool llama_kv_iswa_resident_transaction::owned_by_current_thread() const {
    return active && owner_thread == std::this_thread::get_id();
}

void llama_kv_iswa_resident_transaction::release() noexcept {
    if (!active) {
        return;
    }
    auto retained_guard = guard;
    if (!retained_guard) {
        std::terminate();
    }
    {
        std::lock_guard<std::recursive_mutex> lock(retained_guard->mutex);
        if (!retained_guard->resident_transaction_active ||
                retained_guard->resident_transaction_generation != generation ||
                retained_guard->resident_transaction_owner != owner_thread) {
            std::terminate();
        }
        retained_guard->resident_transaction_active = false;
        retained_guard->resident_transaction_owner  = {};
        active = false;
    }
    retained_guard->transaction_released.notify_all();
}

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
                uint32_t   n_resident,
                     bool   logical_transactions) :
    llama_kv_cache_iswa(model, model.hparams, type_k, type_v, v_trans, offload, swa_full, unified,
            kv_size, n_seq_max, n_ubatch, n_pad, mem_other, filter, reuse, share, buft_override, n_resident,
            logical_transactions) {
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
                uint32_t   n_resident,
                     bool   logical_transactions) : unified(unified) {

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

    if (n_resident > 0 || logical_transactions) {
        logical_guard = std::make_shared<llama_kv_cache_resident_guard>();
        kv_base->resident_bind_guard(logical_guard, 0);
        kv_swa ->resident_bind_guard(logical_guard, 1);
    }
    if (n_resident > 0) {
        resident = std::make_unique<resident_impl>(n_resident, logical_guard);
    }
}

llama_kv_cache_iswa::~llama_kv_cache_iswa() {
    if (resident && !resident->handles.empty()) {
        std::terminate();
    }
}

std::unique_lock<std::recursive_mutex> llama_kv_cache_iswa::lock_resident() const {
    if (logical_guard) {
        const auto hook    = resident_lock_hook;
        resident_lock_hook = {};
        if (hook.callback == nullptr) {
            std::unique_lock<std::recursive_mutex> result(logical_guard->mutex);
            logical_guard->transaction_released.wait(result, [&] {
                return !logical_guard->resident_transaction_active ||
                        logical_guard->resident_transaction_owner == std::this_thread::get_id();
            });
            return result;
        }

        hook.callback(llama_kv_iswa_resident_lock_phase::before_lock, hook.context);
        std::unique_lock<std::recursive_mutex> result(logical_guard->mutex, std::try_to_lock);
        bool contended = !result.owns_lock();
        if (!result.owns_lock()) {
            hook.callback(llama_kv_iswa_resident_lock_phase::lock_contended, hook.context);
            result.lock();
        }
        if (logical_guard->resident_transaction_active &&
                logical_guard->resident_transaction_owner != std::this_thread::get_id()) {
            if (!contended) {
                hook.callback(llama_kv_iswa_resident_lock_phase::lock_contended, hook.context);
                contended = true;
            }
            logical_guard->transaction_released.wait(result, [&] {
                return !logical_guard->resident_transaction_active ||
                        logical_guard->resident_transaction_owner == std::this_thread::get_id();
            });
        }
        hook.callback(llama_kv_iswa_resident_lock_phase::lock_acquired, hook.context);
        return result;
    }
    return {};
}

bool llama_kv_cache_iswa::resident_is_quiescent() const {
    return logical_guard && kv_base->sc_info.empty() && kv_swa->sc_info.empty() &&
            logical_guard->pending_updates[0] == 0 && logical_guard->pending_updates[1] == 0 &&
            !logical_guard->update_active[0] && !logical_guard->update_active[1] &&
            logical_guard->active_batches == 0;
}

std::shared_ptr<void> llama_kv_cache_iswa::acquire_resident_batch_lease() const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!logical_guard) {
        return {};
    }
    if (logical_guard->resident_transaction_active) {
        throw std::runtime_error("cannot begin ISWA graph while a resident transaction is active");
    }
    return std::make_shared<llama_kv_iswa_batch_lease>(logical_guard);
}

llama_kv_iswa_resident_transaction llama_kv_cache_iswa::acquire_resident_transaction() const {
    auto resident_scope = lock_resident();
    if (!logical_guard || logical_guard->resident_transaction_active || !resident_is_quiescent()) {
        return {};
    }
    return llama_kv_iswa_resident_transaction(logical_guard, std::move(resident_scope));
}

bool llama_kv_cache_iswa::has_resident_aperture() const {
    return resident != nullptr;
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
    [[maybe_unused]] auto resident_scope = lock_resident();
    // the base cache is a superset of the SWA cache, so we can just check the SWA cache
    return kv_swa->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_iswa::seq_pos_max(llama_seq_id seq_id) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
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

llama_kv_iswa_logical_sequence_state llama_kv_cache_iswa::export_logical_sequence(
        llama_seq_id seq_id) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (seq_id < 0 || (uint32_t) seq_id >= kv_base->n_seq_max) {
        throw std::runtime_error("ISWA logical export sequence is out of range");
    }

    const auto export_plane = [&](const llama_kv_cache & cache) {
        llama_kv_iswa_logical_plane_state result;
        result.schema_version = LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA;

        const uint32_t stream = cache.seq_to_stream.at((size_t) seq_id);
        const auto &   cells  = cache.v_cells.at(stream);
        std::vector<std::pair<llama_pos, uint32_t>> logical_cells;
        logical_cells.reserve(cells.get_used());
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
                continue;
            }
            if (llama_hparams::is_masked_swa(
                        cache.n_swa, cache.swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id))) {
                continue;
            }
            logical_cells.emplace_back(cells.pos_get(i), i);
        }
        std::sort(logical_cells.begin(), logical_cells.end(), [](const auto & lhs, const auto & rhs) {
            return lhs.first != rhs.first ? lhs.first < rhs.first : lhs.second < rhs.second;
        });
        for (size_t i = 1; i < logical_cells.size(); ++i) {
            if (logical_cells[i - 1].first == logical_cells[i].first) {
                throw std::runtime_error("ISWA logical export contains duplicate absolute positions");
            }
        }

        llama_kv_cache::cell_ranges_t ranges;
        ranges.strm = stream;
        for (const auto & [pos, cell] : logical_cells) {
            result.positions.push_back(pos);
            if (cache.hparams.n_pos_per_embd() > 1) {
                result.extensions.push_back(cells.ext_get(cell));
            }
            if (!ranges.data.empty() && ranges.data.back().second == cell) {
                ++ranges.data.back().second;
            } else {
                ranges.data.emplace_back(cell, cell + 1);
            }
        }

        for (const auto & layer : cache.layers) {
            if (layer.v != nullptr) {
                throw std::runtime_error("ISWA canonical DSV4 state requires K-only storage");
            }
        }
        logical_vector_writer writer;
        cache.state_write_data(writer, ranges);
        result.tensor_payload = std::move(writer.data);
        result.checksum       = logical_plane_checksum(result);
        return result;
    };

    llama_kv_iswa_logical_sequence_state result;
    result.schema_version = LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA;
    result.base           = export_plane(*kv_base);
    result.swa            = export_plane(*kv_swa);
    return result;
}

void llama_kv_cache_iswa::validate_logical_sequence(
        const llama_kv_iswa_logical_sequence_state & state,
        llama_pos                                    accepted_frontier) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (state.schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA) {
        throw std::runtime_error("ISWA logical sequence schema mismatch");
    }
    if (accepted_frontier < -1) {
        throw std::runtime_error("ISWA logical sequence frontier is invalid");
    }

    const auto validate_plane = [&](const llama_kv_cache & cache,
                                    const llama_kv_iswa_logical_plane_state & plane) {
        if (plane.schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA) {
            throw std::runtime_error("ISWA logical plane schema mismatch");
        }
        if (plane.positions.size() > cache.get_size()) {
            throw std::runtime_error("ISWA logical plane exceeds cache capacity");
        }
        for (size_t i = 0; i < plane.positions.size(); ++i) {
            if (plane.positions[i] < 0 || plane.positions[i] > accepted_frontier ||
                    (i != 0 && plane.positions[i - 1] >= plane.positions[i])) {
                throw std::runtime_error("ISWA logical positions are not a strict absolute range");
            }
        }
        if (accepted_frontier < 0) {
            if (!plane.positions.empty()) {
                throw std::runtime_error("ISWA empty frontier has logical rows");
            }
        } else {
            // Every currently visible row is mandatory. Derive the exact
            // contiguous suffix from the cache's own SWA mask; stop after one
            // capacity plus one so a hostile frontier cannot force an
            // unbounded scan.
            llama_pos first = accepted_frontier;
            uint64_t expected_count = 1;
            while (first > 0 && !llama_hparams::is_masked_swa(
                        cache.n_swa, cache.swa_type, first - 1, accepted_frontier)) {
                if (expected_count == cache.get_size()) {
                    throw std::runtime_error("ISWA logical frontier exceeds plane capacity");
                }
                --first;
                ++expected_count;
            }
            if (plane.positions.size() != expected_count) {
                throw std::runtime_error("ISWA logical plane does not cover the accepted frontier");
            }
            for (size_t i = 0; i < plane.positions.size(); ++i) {
                if (plane.positions[i] != first + (llama_pos) i) {
                    throw std::runtime_error("ISWA logical plane has a sparse accepted tail");
                }
            }
        }
        const bool has_extensions = cache.hparams.n_pos_per_embd() > 1;
        if (plane.extensions.size() != (has_extensions ? plane.positions.size() : 0)) {
            throw std::runtime_error("ISWA logical extension coverage mismatch");
        }
        if (logical_plane_checksum(plane) != plane.checksum) {
            throw std::runtime_error("ISWA logical plane checksum mismatch");
        }
        for (const auto & layer : cache.layers) {
            if (layer.v != nullptr) {
                throw std::runtime_error("ISWA logical DSV4 validation requires K-only storage");
            }
        }

        logical_vector_reader reader(plane.tensor_payload);
        uint32_t v_trans_ref = 0;
        uint32_t n_layer_ref = 0;
        reader.read(&v_trans_ref, sizeof(v_trans_ref));
        reader.read(&n_layer_ref, sizeof(n_layer_ref));
        if (v_trans_ref != (cache.v_trans ? 1u : 0u) || n_layer_ref != cache.layers.size()) {
            throw std::runtime_error("ISWA logical tensor envelope mismatch");
        }
        for (const auto & layer : cache.layers) {
            int32_t  type_ref     = -1;
            uint64_t row_size_ref = 0;
            reader.read(&type_ref, sizeof(type_ref));
            reader.read(&row_size_ref, sizeof(row_size_ref));
            const ggml_tensor * tensor   = layer.k_stream.at(0);
            const uint64_t      row_size = ggml_row_size(tensor->type, cache.hparams.n_embd_k_gqa(layer.il));
            if (type_ref != (int32_t) tensor->type || row_size_ref != row_size) {
                throw std::runtime_error("ISWA logical K tensor geometry mismatch");
            }
            if (row_size != 0 && plane.positions.size() > std::numeric_limits<size_t>::max()/row_size) {
                throw std::runtime_error("ISWA logical K payload size overflow");
            }
            reader.skip(plane.positions.size()*row_size);
        }
        if (reader.n_bytes() != plane.tensor_payload.size()) {
            throw std::runtime_error("ISWA logical tensor payload has trailing bytes");
        }
    };

    validate_plane(*kv_base, state.base);
    validate_plane(*kv_swa,  state.swa);
}

std::shared_ptr<llama_kv_iswa_logical_import_plan>
llama_kv_cache_iswa::prepare_logical_sequence_import(
        llama_seq_id                                 seq_id,
        const llama_kv_iswa_logical_sequence_state & state,
        llama_pos                                    accepted_frontier) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    if (!logical_guard || !logical_guard->resident_transaction_active ||
            seq_id < 0 || (uint32_t) seq_id >= kv_base->n_seq_max) {
        throw std::runtime_error("ISWA logical import requires a retained quiescent transaction");
    }
    validate_logical_sequence(state, accepted_frontier);

    auto result = std::make_shared<llama_kv_iswa_logical_import_plan>();
    result->owner         = this;
    result->destination   = seq_id;
    result->version[0]    = logical_guard->version[0];
    result->version[1]    = logical_guard->version[1];
    result->transaction_generation = logical_guard->resident_transaction_generation;
    result->base_checksum = state.base.checksum;
    result->swa_checksum  = state.swa.checksum;

    const auto prepare_plane = [&](const llama_kv_cache & cache,
                                   const llama_kv_iswa_logical_plane_state & logical,
                                   llama_kv_iswa_logical_import_plan::plane & prepared) {
        if (cache.other != nullptr || cache.n_stream <= 1 ||
                cache.seq_to_stream.at((size_t) seq_id) != (uint32_t) seq_id) {
            throw std::runtime_error("ISWA logical import requires exclusive per-sequence streams");
        }
        const auto & current = cache.v_cells.at((uint32_t) seq_id);
        if (current.get_has_shift()) {
            throw std::runtime_error("ISWA logical import cannot replace shifted metadata");
        }
        for (uint32_t i = 0; i < current.size(); ++i) {
            if (!current.is_empty(i) &&
                    (current.seq_count(i) != 1 || !current.seq_has(i, seq_id))) {
                throw std::runtime_error("ISWA logical import stream contains foreign sequence ownership");
            }
        }

        prepared.stream = (uint32_t) seq_id;
        prepared.head   = (uint32_t) logical.positions.size();
        prepared.cells.resize(cache.get_size());
        prepared.slot.s0 = prepared.stream;
        prepared.slot.s1 = prepared.stream;
        prepared.slot.resize(1);
        prepared.slot.strm[0] = prepared.stream;
        prepared.slot.idxs[0].resize(logical.positions.size());
        for (size_t i = 0; i < logical.positions.size(); ++i) {
            prepared.slot.idxs[0][i] = (uint32_t) i;
            prepared.cells.pos_set((uint32_t) i, logical.positions[i]);
            if (!logical.extensions.empty()) {
                prepared.cells.ext_set((uint32_t) i, logical.extensions[i]);
            }
            prepared.cells.seq_add((uint32_t) i, seq_id);
        }
    };

    prepare_plane(*kv_base, state.base, result->base);
    prepare_plane(*kv_swa,  state.swa,  result->swa);
    return result;
}

bool llama_kv_cache_iswa::validate_logical_sequence_import(
        const std::shared_ptr<llama_kv_iswa_logical_import_plan> & plan) const {
    [[maybe_unused]] auto resident_scope = lock_resident();
    return plan && plan->owner == this &&
            plan->state == llama_kv_iswa_logical_import_state::prepared &&
            logical_guard && logical_guard->resident_transaction_active &&
            plan->transaction_generation == logical_guard->resident_transaction_generation &&
            plan->destination >= 0 && (uint32_t) plan->destination < kv_base->n_seq_max &&
            plan->version[0] == logical_guard->version[0] &&
            plan->version[1] == logical_guard->version[1];
}

bool llama_kv_cache_iswa::commit_logical_sequence_import(
        const std::shared_ptr<llama_kv_iswa_logical_import_plan> & plan,
        const llama_kv_iswa_logical_sequence_state & state) noexcept {
    try {
        [[maybe_unused]] auto resident_scope = lock_resident();
        if (!validate_logical_sequence_import(plan) ||
                state.schema_version != LLAMA_KV_ISWA_LOGICAL_STATE_SCHEMA ||
                state.base.checksum != plan->base_checksum || state.swa.checksum != plan->swa_checksum ||
                logical_plane_checksum(state.base) != state.base.checksum ||
                logical_plane_checksum(state.swa)  != state.swa.checksum ||
                state.base.positions.size() != plan->base.slot.idxs[0].size() ||
                state.swa.positions.size()  != plan->swa.slot.idxs[0].size()) {
            return false;
        }

        const auto write_plane = [](llama_kv_cache & cache,
                                    llama_kv_iswa_logical_import_plan::plane & prepared,
                                    const llama_kv_iswa_logical_plane_state & logical) {
            logical_vector_reader reader(logical.tensor_payload);
            if (!cache.state_read_data(reader, prepared.stream,
                                       (uint32_t) logical.positions.size(), prepared.slot) ||
                    reader.n_bytes() != logical.tensor_payload.size()) {
                // Every envelope field and byte count was checked during the
                // prepare phase. Reaching this point would mean an internal
                // invariant changed while the retained transaction was held;
                // continuing could expose a half-written execution stream.
                std::terminate();
            }
        };
        write_plane(*kv_base, plan->base, state.base);
        write_plane(*kv_swa,  plan->swa,  state.swa);

        static_assert(std::is_nothrow_move_assignable_v<llama_kv_cells>);
        kv_base->resident_note_mutation();
        kv_swa ->resident_note_mutation();
        kv_base->v_cells[plan->base.stream] = std::move(plan->base.cells);
        kv_swa ->v_cells[plan->swa.stream]  = std::move(plan->swa.cells);
        kv_base->v_heads[plan->base.stream] = plan->base.head;
        kv_swa ->v_heads[plan->swa.stream]  = plan->swa.head;
        plan->state = llama_kv_iswa_logical_import_state::committed;
        return true;
    } catch (...) {
        // Publication is designed to be no-throw after the complete prepare
        // and checksum checks above. A backend throwing through its C entry
        // point is not a recoverable cache-state failure.
        std::terminate();
    }
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

llama_kv_iswa_resident_status llama_kv_cache_iswa::release_resident(
        llama_kv_iswa_resident_handle handle,
        llama_kv_iswa_resident_release_audit * audit) {
    try {
        if (audit != nullptr) {
            *audit = {};
            audit->pools.reserve(GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS);
        }
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
        auto backend = llama_kv_iswa_quote_move(sources, nullptr, audit != nullptr);
        if (!backend.value) {
            return llama_kv_iswa_backend_status(backend.status, sparse_move_phase::release_quote);
        }
        const auto before_audit = llama_kv_iswa_capture_move_audit(backend, 0);
        if (audit != nullptr) {
            audit->before_status = before_audit.status;
        }
        const int backend_status = backend.commit(backend.value.get());
        if (backend_status != GGML_DSV4_SPARSE_OK) {
            return llama_kv_iswa_backend_status(backend_status, sparse_move_phase::release_commit);
        }
        const auto after_audit = llama_kv_iswa_capture_move_audit(backend, 1);
        llama_kv_iswa_publish_move_audit(before_audit, after_audit, audit);

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

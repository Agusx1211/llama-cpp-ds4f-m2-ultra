#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-dsv4-accounting.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "../ggml/src/ggml-metal/ggml-metal-device.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;

static constexpr uint32_t DSV4_STATE_MAGIC         = 0x34565344; // DSV4
static constexpr uint32_t DSV4_STATE_VERSION       = 2;
static constexpr uint32_t DSV4_STATE_MODE_FULL     = 0;
static constexpr uint32_t DSV4_STATE_MODE_PARTIAL  = 1;
static constexpr uint32_t DSV4_K_CACHE_STATE_VER   = 2;
static constexpr uint32_t DSV4_COMP_STATE_VER      = 1;

static std::atomic<uint32_t> dsv4_test_pressure_count = 0;

static uint64_t dsv4_hash_u64(uint64_t hash, uint64_t value) {
    // FNV-1a over an explicit little-endian integer encoding. This keeps the
    // persistent identity independent of host padding and object layout.
    for (uint32_t i = 0; i < 8; ++i) {
        hash ^= (value >> (8*i)) & 0xffu;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t dsv4_hash_string(uint64_t hash, const std::string & value) {
    hash = dsv4_hash_u64(hash, value.size());
    for (unsigned char c : value) {
        hash ^= c;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t dsv4_hash_tensor(uint64_t hash, const ggml_tensor * tensor) {
    hash = dsv4_hash_u64(hash, tensor->type);
    for (uint32_t i = 0; i < GGML_MAX_DIMS; ++i) {
        hash = dsv4_hash_u64(hash, tensor->ne[i]);
    }
    return hash;
}

static uint64_t dsv4_hash_cache(uint64_t hash, const llama_kv_cache * cache) {
    hash = dsv4_hash_u64(hash, cache->get_size());
    hash = dsv4_hash_u64(hash, cache->get_n_stream());
    // DSV4 turns every participating cache into K-only MLA storage, so its V
    // pointers are null and the absence of V is itself part of the schema. K
    // types are hashed per tensor below; some raw halves legitimately have no
    // participating layers, so generic type_k()/type_v() cannot be called.
    hash = dsv4_hash_string(hash, "k-only");
    const auto layer_ids = cache->get_layer_ids();
    hash = dsv4_hash_u64(hash, layer_ids.size());
    for (uint32_t il : layer_ids) {
        hash = dsv4_hash_u64(hash, il);
        hash = dsv4_hash_tensor(hash, cache->get_k_storage(il));
    }
    return hash;
}

void llama_kv_cache_dsv4_test_inject_physical_pressure(uint32_t count) {
    dsv4_test_pressure_count.store(count, std::memory_order_release);
}

static bool dsv4_test_consume_physical_pressure() {
    uint32_t current = dsv4_test_pressure_count.load(std::memory_order_acquire);
    while (current != 0) {
        if (dsv4_test_pressure_count.compare_exchange_weak(
                    current, current - 1,
                    std::memory_order_acq_rel,
                    std::memory_order_acquire)) {
            return true;
        }
    }
    return false;
}

using dsv4_sparse_buft_fn = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t, uint32_t);

static ggml_backend_buffer_type_t dsv4_sparse_buft(
        const llama_model & model,
        uint32_t n_seq_max) {
    if (n_seq_max == 0) {
        return nullptr;
    }

    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        auto * dev = model.dev_layer(il);
        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg == nullptr) {
            continue;
        }
        auto * get_buft = (dsv4_sparse_buft_fn) ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_metal_dsv4_sparse_buffer_type");
        if (get_buft != nullptr) {
            if (auto * buft = get_buft(dev, n_seq_max)) {
                return buft;
            }
        }
    }

    return nullptr;
}

bool llama_kv_cache_dsv4_supports_elastic_metal(
        const llama_model & model,
        uint32_t n_seq_max) {
    return dsv4_sparse_buft(model, n_seq_max) != nullptr;
}

static uint32_t dsv4_comp_size(uint32_t kv_size, uint32_t ratio) {
    return std::max<uint32_t>(1, (kv_size + ratio - 1)/ratio);
}

static void * dsv4_backend_proc(const ggml_tensor * tensor, const char * name) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    auto * buft = ggml_backend_buffer_get_type(tensor->buffer);
    auto * dev = ggml_backend_buft_get_device(buft);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

static int dsv4_sparse_unmap_tensor_range(
        ggml_tensor * tensor,
        size_t offset,
        size_t size) {
    using unmap_fn = int (*)(ggml_tensor *, size_t, size_t);
    auto * unmap = (unmap_fn) dsv4_backend_proc(
            tensor, "ggml_backend_metal_dsv4_sparse_unmap_tensor_range");
    return unmap ? unmap(tensor, offset, size) : 0;
}

static void dsv4_clear_tensor_stream(ggml_tensor * tensor, uint32_t stream) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[3] == 1);
    GGML_ASSERT(stream < (uint32_t) tensor->ne[2]);

    const size_t stream_size = tensor->nb[2];
    const bool page_aligned = stream_size % (64*1024) == 0 &&
            (uintptr_t) tensor->data % (64*1024) == 0;
    const int sparse = page_aligned ? dsv4_sparse_unmap_tensor_range(
            tensor, stream*stream_size, stream_size) : 0;
    if (sparse < 0) {
        throw std::runtime_error("failed to reclaim DSV4 sparse tensor stream");
    }
    if (sparse > 0) {
        return;
    }
    ggml_backend_tensor_memset(tensor, 0, stream*stream_size, stream_size);
}

struct dsv4_sparse_range {
    ggml_tensor * tensor;
    size_t offset;
    size_t size;
    llama_dsv4_memory_family family = LLAMA_DSV4_MEMORY_RAW;
};

static bool dsv4_sparse_append_k_rows(
        const llama_kv_cache * kv,
        std::vector<int64_t> rows,
        llama_dsv4_memory_family family,
        std::vector<dsv4_sparse_range> & ranges) {
    if (rows.empty()) {
        return true;
    }

    const int64_t n_rows_total = (int64_t) kv->get_size()*kv->get_n_stream();
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    if (rows.front() < 0 || rows.back() >= n_rows_total) {
        return false;
    }

    std::vector<std::pair<int64_t, int64_t>> runs;
    for (size_t i = 0; i < rows.size();) {
        const int64_t begin = rows[i];
        int64_t end = begin + 1;
        while (++i < rows.size() && rows[i] == end) {
            ++end;
        }
        runs.emplace_back(begin, end);
    }

    ranges.reserve(ranges.size() + kv->get_layer_ids().size()*runs.size());
    for (uint32_t il : kv->get_layer_ids()) {
        ggml_tensor * tensor = kv->get_k_storage(il);
        for (const auto & [begin, end] : runs) {
            ranges.push_back({
                tensor,
                (size_t) begin*tensor->nb[1],
                (size_t) (end - begin)*tensor->nb[1],
                family,
            });
        }
    }
    return true;
}

static bool dsv4_sparse_append_slot(
        const llama_kv_cache * kv,
        const llama_kv_cache::slot_info & sinfo,
        llama_dsv4_memory_family family,
        std::vector<dsv4_sparse_range> & ranges) {
    std::vector<int64_t> rows;
    for (size_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t idx : sinfo.idxs[s]) {
            rows.push_back((int64_t) sinfo.strm[s]*kv->get_size() + idx);
        }
    }
    return dsv4_sparse_append_k_rows(kv, std::move(rows), family, ranges);
}

class dsv4_sparse_transaction {
public:
    enum prepare_status {
        SUCCESS,
        PRESSURE,
        ERROR,
    };

    enum failure_phase {
        PHASE_NONE,
        PHASE_PROC_LOOKUP,
        PHASE_USAGE,
        PHASE_RESERVE,
        PHASE_POOL_ACCOUNTING,
        PHASE_COMMIT,
    };

    ~dsv4_sparse_transaction() {
        if (ticket != nullptr) {
            cancel(ticket);
            free_ticket(ticket);
        }
    }

    prepare_status reserve_ranges(
            const std::vector<dsv4_sparse_range> & ranges,
            llama_dsv4_batch_quote & result) {
        n_ranges = ranges.size();
        for (size_t i = 0; i < result.families.size(); ++i) {
            result.families[i].family = (llama_dsv4_memory_family) i;
        }
        if (ranges.empty()) {
            return SUCCESS;
        }

        using reserve_fn = ggml_metal_sparse_reservation_result (*)(
                ggml_tensor * const *, const size_t *, const size_t *, size_t,
                ggml_metal_sparse_pool_quote *, size_t, size_t *, size_t *, void **);
        using commit_fn = ggml_metal_sparse_reservation_result (*)(void *);
        using cancel_fn = bool (*)(void *);
        using free_fn = void (*)(void *);
        using usage_fn = int (*)(const ggml_tensor *, ggml_metal_sparse_usage *);

        reserve_fn reserve = nullptr;
        ggml_tensor * proc_tensor = nullptr;
        for (const auto & range : ranges) {
            reserve = (reserve_fn) dsv4_backend_proc(
                    range.tensor, "ggml_backend_metal_dsv4_sparse_reserve_tensor_ranges");
            if (reserve != nullptr) {
                proc_tensor = range.tensor;
                break;
            }
        }
        if (reserve == nullptr) {
            return SUCCESS;
        }

        commit = (commit_fn) dsv4_backend_proc(
                proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_commit");
        cancel = (cancel_fn) dsv4_backend_proc(
                proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_cancel");
        free_ticket = (free_fn) dsv4_backend_proc(
                proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_free");
        auto * usage = (usage_fn) dsv4_backend_proc(
                proc_tensor, "ggml_backend_metal_dsv4_sparse_tensor_usage");
        usage_proc_available = usage != nullptr;
        if (commit == nullptr || cancel == nullptr || free_ticket == nullptr || usage == nullptr) {
            phase = PHASE_PROC_LOOKUP;
            return ERROR;
        }

        std::vector<ggml_tensor *> tensors;
        std::vector<size_t> offsets;
        std::vector<size_t> sizes;
        tensors.reserve(ranges.size());
        offsets.reserve(ranges.size());
        sizes.reserve(ranges.size());
        std::map<uintptr_t, uint32_t> pool_family_masks;
        for (const auto & range : ranges) {
            tensors.push_back(range.tensor);
            offsets.push_back(range.offset);
            sizes.push_back(range.size);
            ggml_metal_sparse_usage snapshot = {};
            const int snapshot_result = usage(range.tensor, &snapshot);
            if (snapshot_result < 0) {
                phase = PHASE_USAGE;
                return ERROR;
            }
            if (snapshot_result > 0) {
                ++n_sparse_ranges;
                pool_family_masks[snapshot.pool_id] |= 1u << range.family;
            }
        }

        // The Metal registry exposes the sparse transaction procedures for
        // ordinary buffers too. No sparse ranges is therefore a successful
        // no-op, not an unsupported reservation failure.
        if (n_sparse_ranges == 0) {
            return SUCCESS;
        }

        std::vector<ggml_metal_sparse_pool_quote> pools(ranges.size());
        size_t n_pools = 0;
        size_t limiting_pool = SIZE_MAX;
        const auto status = reserve(
                tensors.data(), offsets.data(), sizes.data(), tensors.size(),
                pools.data(), pools.size(), &n_pools, &limiting_pool, &ticket);
        backend_status = status;
        ticket_issued = ticket != nullptr;
        n_quoted_pools = n_pools;
        pools.resize(n_pools);

        for (const auto & pool : pools) {
            const auto found = pool_family_masks.find(pool.pool_id);
            if (found == pool_family_masks.end() || found->second == 0) {
                phase = PHASE_POOL_ACCOUNTING;
                failure_pool_id = pool.pool_id;
                return ERROR;
            }
            const uint32_t family_mask = found->second;
            if ((family_mask & (family_mask - 1)) != 0) {
                ++n_shared_family_pools;
            }
            const auto representative = (llama_dsv4_memory_family) __builtin_ctz(family_mask);
            auto & family = result.families[representative];
            family.target_mappings += pool.write.target_mappings;
            family.new_pages += pool.write.new_pages;
            family.cow_pages += pool.write.cow_pages;
            family.required_pages += pool.write.required_pages;
            family.physical_pages += pool.usage.physical_pages;
            family.free_pages += pool.usage.free_pages;
            family.reserved_pages += pool.usage.reserved_pages;
            family.feasible = family.feasible && pool.write.feasible;
        }
        if (limiting_pool < pools.size()) {
            result.limiting_pool_id = pools[limiting_pool].pool_id;
            const auto found = pool_family_masks.find(result.limiting_pool_id);
            if (found != pool_family_masks.end() && found->second != 0) {
                limiting_family_mask_value = found->second;
                result.limiting_family = (llama_dsv4_memory_family) __builtin_ctz(found->second);
            }
        }
        result.feasible = status == GGML_METAL_SPARSE_RESERVATION_OK;

        // Consume the test fault only after the real backend has produced a
        // valid quote and ticket. Returning pressure here exercises the same
        // cancellation and propagation path as capacity exhaustion, without
        // committing a mapping or submitting a graph.
        if (status == GGML_METAL_SPARSE_RESERVATION_OK && ticket != nullptr &&
                !pools.empty() && dsv4_test_consume_physical_pressure()) {
            const auto & pool = pools.front();
            const uint32_t family_mask = pool_family_masks.at(pool.pool_id);
            limiting_family_mask_value = family_mask;
            result.limiting_pool_id = pool.pool_id;
            result.limiting_family = (llama_dsv4_memory_family) __builtin_ctz(family_mask);
            result.families[result.limiting_family].feasible = false;
            result.feasible = false;
            backend_status = GGML_METAL_SPARSE_RESERVATION_PRESSURE;
            return PRESSURE;
        }

        if (status == GGML_METAL_SPARSE_RESERVATION_PRESSURE) {
            return PRESSURE;
        }
        if (status != GGML_METAL_SPARSE_RESERVATION_OK || ticket == nullptr) {
            phase = PHASE_RESERVE;
            return ERROR;
        }
        return SUCCESS;
    }

    ggml_metal_sparse_reservation_result commit_ranges() {
        if (ticket == nullptr) {
            return GGML_METAL_SPARSE_RESERVATION_OK;
        }
        const auto status = commit(ticket);
        backend_status = status;
        if (status != GGML_METAL_SPARSE_RESERVATION_OK) {
            phase = PHASE_COMMIT;
        }
        free_ticket(ticket);
        ticket = nullptr;
        return status;
    }

    void log_failure(const char * caller) const {
        LLAMA_LOG_ERROR(
                "%s: DSV4 sparse reservation failed: phase=%s backend_status=%s"
                " ranges=%zu sparse_ranges=%zu pools=%zu ticket=%s"
                " failure_pool=%" PRIuPTR " shared_family_pools=%zu"
                " procs={commit=%d,cancel=%d,free=%d,usage=%d}\n",
                caller, failure_phase_name(phase),
                ggml_metal_sparse_reservation_result_name(backend_status),
                n_ranges, n_sparse_ranges, n_quoted_pools,
                ticket_issued ? "issued" : "none", failure_pool_id,
                n_shared_family_pools, commit != nullptr, cancel != nullptr,
                free_ticket != nullptr, usage_proc_available);
    }

    uint32_t limiting_family_mask() const {
        return limiting_family_mask_value;
    }

    size_t sparse_range_count() const {
        return n_sparse_ranges;
    }

    size_t shared_family_pool_count() const {
        return n_shared_family_pools;
    }

private:
    static const char * failure_phase_name(failure_phase value) {
        switch (value) {
            case PHASE_NONE:            return "none";
            case PHASE_PROC_LOOKUP:     return "proc-lookup";
            case PHASE_USAGE:           return "usage";
            case PHASE_RESERVE:         return "reserve";
            case PHASE_POOL_ACCOUNTING: return "pool-accounting";
            case PHASE_COMMIT:          return "commit";
        }
        return "unknown";
    }

    void * ticket = nullptr;
    ggml_metal_sparse_reservation_result (*commit)(void *) = nullptr;
    bool (*cancel)(void *) = nullptr;
    void (*free_ticket)(void *) = nullptr;
    failure_phase phase = PHASE_NONE;
    ggml_metal_sparse_reservation_result backend_status = GGML_METAL_SPARSE_RESERVATION_OK;
    size_t n_ranges = 0;
    size_t n_sparse_ranges = 0;
    size_t n_quoted_pools = 0;
    size_t n_shared_family_pools = 0;
    uintptr_t failure_pool_id = 0;
    uint32_t limiting_family_mask_value = 0;
    bool ticket_issued = false;
    bool usage_proc_available = false;
};

static int64_t dsv4_stream_offset(uint32_t n_stream, llama_seq_id seq_id, uint32_t size) {
    if (n_stream <= 1) {
        return 0;
    }
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        throw std::runtime_error("DSV4 sequence id out of stream range");
    }

    return (int64_t) seq_id*size;
}

static bool dsv4_ubatch_has_coupled(const llama_ubatch & ubatch) {
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static bool dsv4_token_has_seq(const llama_ubatch & ubatch, uint32_t i, llama_seq_id seq_id) {
    for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
        if (ubatch.seq_id[i][s] == seq_id) {
            return true;
        }
    }

    return false;
}

static llama_ubatch dsv4_build_raw_write_ubatch(const llama_ubatch & ubatch) {
    if (!dsv4_ubatch_has_coupled(ubatch)) {
        return ubatch;
    }
    if (ubatch.embd) {
        throw std::runtime_error("DSV4 coupled embedding ubatches are not supported");
    }

    std::vector<uint32_t> counts(ubatch.n_seqs_unq, 0);
    uint32_t n_tokens = 0;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                ++counts[s];
                ++n_tokens;
            }
        }
    }

    if (n_tokens == 0) {
        return ubatch;
    }

    const uint32_t n_seq_tokens = counts[0];
    for (uint32_t s = 1; s < counts.size(); ++s) {
        if (counts[s] != n_seq_tokens) {
            throw std::runtime_error("DSV4 coupled raw writes require equal sequence lengths");
        }
    }

    auto data = std::make_shared<llama_ubatch::data_t>();
    data->pos.resize((size_t) n_tokens*ubatch.n_pos);
    data->n_seq_id.reserve(n_tokens);
    data->seq_id.reserve(n_tokens);
    data->seq_id_data.reserve(n_tokens);
    data->seq_id_unq.assign(ubatch.seq_id_unq, ubatch.seq_id_unq + ubatch.n_seqs_unq);
    data->seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data->output.assign(n_tokens, 0);
    if (ubatch.token) {
        data->token.reserve(n_tokens);
    }

    for (uint32_t s = 0; s < data->seq_id_unq.size(); ++s) {
        data->seq_idx[data->seq_id_unq[s]] = s;
    }

    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        const llama_seq_id seq_id = ubatch.seq_id_unq[s];
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (!dsv4_token_has_seq(ubatch, i, seq_id)) {
                continue;
            }

            const uint32_t dst = data->n_seq_id.size();
            if (ubatch.token) {
                data->token.push_back(ubatch.token[i]);
            }
            for (uint32_t p = 0; p < ubatch.n_pos; ++p) {
                data->pos[(size_t) p*n_tokens + dst] = ubatch.pos[(size_t) p*ubatch.n_tokens + i];
            }
            data->n_seq_id.push_back(1);
            data->seq_id_data.push_back(seq_id);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        data->seq_id.push_back(&data->seq_id_data[i]);
    }

    llama_ubatch res {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ n_tokens,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seqs       =*/ ubatch.n_seqs_unq,
        /*.n_seqs_unq   =*/ ubatch.n_seqs_unq,
        /*.n_pos        =*/ ubatch.n_pos,
        /*.token        =*/ data->token.empty() ? nullptr : data->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ data->pos.data(),
        /*.n_seq_id     =*/ data->n_seq_id.data(),
        /*.seq_id       =*/ data->seq_id.data(),
        /*.seq_id_unq   =*/ data->seq_id_unq.data(),
        /*.seq_idx      =*/ data->seq_idx.data(),
        /*.output       =*/ data->output.data(),
        /*.data         =*/ data,
    };

    return res;
}

static std::vector<llama_ubatch> dsv4_build_raw_write_ubatches(const std::vector<llama_ubatch> & ubatches) {
    std::vector<llama_ubatch> res;
    res.reserve(ubatches.size());
    for (const llama_ubatch & ubatch : ubatches) {
        res.push_back(dsv4_build_raw_write_ubatch(ubatch));
    }
    return res;
}

static bool dsv4_batch_has_coupled(const llama_batch & batch) {
    if (!batch.n_seq_id) {
        return false;
    }

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] > 1) {
            return true;
        }
    }

    return false;
}

static int64_t dsv4_comp_graph_n_stream(const llama_ubatch & ubatch, uint32_t n_stream) {
    // Coupled sequence sets must stay in one graph stream because their
    // compressed state is shared. Independent per-seq state can fan out.
    if (n_stream <= 1 || ubatch.n_seqs_unq <= 1 || dsv4_ubatch_has_coupled(ubatch)) {
        return 1;
    }

    return ubatch.n_seqs_unq;
}

static void dsv4_state_src_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t     & s0,
        uint32_t     & ns) {
    if (seq_id >= 0 && n_stream > 1) {
        if ((uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = (uint32_t) seq_id;
        ns = 1;
        return;
    }

    s0 = 0;
    ns = seq_id >= 0 ? 1 : n_stream;
}

static void dsv4_state_dst_stream_range(
        uint32_t       n_stream,
        llama_seq_id   seq_id,
        uint32_t       ns,
        uint32_t     & s0) {
    if (seq_id >= 0) {
        if (ns != 1) {
            throw std::runtime_error("DSV4 sequence state stream count mismatch");
        }
        if (n_stream > 1 && (uint32_t) seq_id >= n_stream) {
            throw std::runtime_error("DSV4 state sequence id out of stream range");
        }

        s0 = n_stream > 1 ? (uint32_t) seq_id : 0;
        return;
    }

    if (ns != n_stream) {
        throw std::runtime_error("DSV4 full state stream count mismatch");
    }

    s0 = 0;
}

static void dsv4_state_write_tensor_streams(
        llama_io_write_i & io,
        ggml_tensor      * tensor,
        uint32_t           tensor_rows,
        uint32_t           n_rows,
        uint32_t           s0,
        uint32_t           ns,
        const std::vector<uint32_t> * stream_ids = nullptr) {
    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    io.write(&type_i,   sizeof(type_i));
    io.write(&ne0,      sizeof(ne0));
    io.write(&rows,     sizeof(rows));
    io.write(&row_size, sizeof(row_size));

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    if (stream_ids && stream_ids->size() != ns) {
        throw std::runtime_error("DSV4 state tensor stream map size mismatch");
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t stream = stream_ids ? (*stream_ids)[s] : s0 + s;
        if ((int64_t) stream >= tensor->ne[2]) {
            throw std::runtime_error("DSV4 state tensor stream out of range");
        }
        const size_t offset = (size_t) stream*stream_stride;
        io.write_tensor(tensor, offset, size);
    }
}

static void dsv4_state_read_tensor_streams(
        llama_io_read_i & io,
        ggml_tensor     * tensor,
        uint32_t          tensor_rows,
        uint32_t          n_rows,
        uint32_t          s0,
        uint32_t          ns) {
    int32_t  type_i_ref;
    uint64_t ne0_ref;
    uint64_t rows_ref;
    uint64_t row_size_ref;

    io.read(&type_i_ref,   sizeof(type_i_ref));
    io.read(&ne0_ref,      sizeof(ne0_ref));
    io.read(&rows_ref,     sizeof(rows_ref));
    io.read(&row_size_ref, sizeof(row_size_ref));

    const int32_t  type_i   = (int32_t) tensor->type;
    const uint64_t ne0      = tensor->ne[0];
    const uint64_t rows     = n_rows;
    const uint64_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);

    if (type_i != type_i_ref || ne0 != ne0_ref || rows != rows_ref || row_size != row_size_ref) {
        throw std::runtime_error("DSV4 state tensor metadata mismatch");
    }
    if (n_rows > tensor_rows) {
        throw std::runtime_error("DSV4 state tensor row count exceeds storage");
    }

    const size_t stream_stride = (size_t) tensor_rows*row_size;
    const size_t size          = (size_t) n_rows*row_size;
    if (size == 0) {
        return;
    }

    for (uint32_t s = 0; s < ns; ++s) {
        const size_t offset = (size_t) (s0 + s)*stream_stride;
        io.read_tensor(tensor, offset, size);
    }
}

static void dsv4_state_write_k_cache(
        llama_io_write_i    & io,
        const llama_kv_cache * kv,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags,
        uint32_t              n_rows) {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(kv->get_n_stream(), seq_id, s0, ns);

    const uint32_t version = DSV4_K_CACHE_STATE_VER;
    const uint32_t kv_size = kv->get_size();
    const auto layer_ids = kv->get_layer_ids();
    const uint32_t n_layer = layer_ids.size();

    if (n_rows > kv_size) {
        throw std::runtime_error("DSV4 K-cache state row count exceeds cache size");
    }

    io.write(&version, sizeof(version));
    io.write(&n_rows,  sizeof(n_rows));
    io.write(&ns,      sizeof(ns));
    io.write(&n_layer, sizeof(n_layer));

    for (uint32_t il : layer_ids) {
        io.write(&il, sizeof(il));
        dsv4_state_write_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows, s0, ns);
    }
}

static void dsv4_state_read_k_cache(
        llama_io_read_i  & io,
        llama_kv_cache   * kv,
        llama_seq_id       seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t n_rows_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,     sizeof(version));
    io.read(&n_rows_ref,  sizeof(n_rows_ref));
    io.read(&ns,          sizeof(ns));
    io.read(&n_layer_ref, sizeof(n_layer_ref));

    if (version != 1 && version != DSV4_K_CACHE_STATE_VER) {
        throw std::runtime_error("DSV4 K-cache state version mismatch");
    }

    const uint32_t kv_size = kv->get_size();
    if (version == 1 && n_rows_ref != kv_size) {
        LLAMA_LOG_INFO("kv size ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }
    if (n_rows_ref > kv_size) {
        LLAMA_LOG_INFO("kv rows ref %d kv %d\n", n_rows_ref, kv_size);
        throw std::runtime_error("DSV4 K-cache state size mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(kv->get_n_stream(), seq_id, ns, s0);

    const auto layer_ids = kv->get_layer_ids();
    if (n_layer_ref != layer_ids.size()) {
        throw std::runtime_error("DSV4 K-cache layer count mismatch");
    }

    for (uint32_t il : layer_ids) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != il) {
            throw std::runtime_error("DSV4 K-cache layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, kv->get_k_storage(il), kv_size, n_rows_ref, s0, ns);
    }
}

static std::string dsv4_plan_positions(const std::vector<int32_t> & values) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            ss << ", ";
        }
        ss << values[i];
    }
    ss << "]";
    return ss.str();
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        uint32_t n_rs_seq_alloc,
        const std::vector<uint32_t> & rs_idx) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);

    if (n_rs_seq > n_rs_seq_alloc) {
        throw std::runtime_error("DSV4 active rollback depth exceeds allocated state layout");
    }

    // n_stream is the persistent cache/state layout; plan.n_stream is the
    // graph view for this ubatch and can be a subset of those streams.
    if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
        throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
    }

    const int64_t state_rows = (int64_t) state_size*n_stream;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
    // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
    // the "previous-window" gather indices for every block, followed by the
    // "current-window" indices for every block. Collect them separately here and
    // append cur after prev once the loop has visited all completed blocks
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;
    std::map<llama_seq_id, uint32_t> state_write_counts;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            curr_token_idx_map[std::make_pair(ubatch.seq_id[i][s], ubatch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return (int32_t) (state_rows + ubatch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
        return (int32_t) (stream_off + pos%state_size);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i];

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const int32_t state_idx = (int32_t) (stream_off + pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            if ((pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);

            plan.state_write_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_logical_idxs.push_back(cache_off + pos/ratio);
            plan.state_write_dummy.push_back(0);
            plan.state_write_pos.push_back((int32_t) source_start);
            ++state_write_counts[seq_id];

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    if (ratio == DSV4_CSA_RATIO && !plan.state_pos.empty()) {
        assert(kv_size > 0);

        // Pad each stream to the reserve plan's block count.
        const auto append_dummy_block = [&](llama_seq_id seq_id, uint32_t i) {
            const int64_t cache_off = dsv4_stream_offset(n_stream, seq_id, kv_size);
            const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

            plan.state_write_idxs.push_back(cache_off + kv_size - 1);
            plan.state_write_logical_idxs.push_back(cache_off + kv_size - 1);
            plan.state_write_dummy.push_back(1);
            plan.state_write_pos .push_back(0);

            if (overlap) {
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(source_idx);
                    overlap_cur_reads .push_back(source_idx);
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(source_idx);
                }
            }
        };

        if (dsv4_ubatch_has_coupled(ubatch)) {
            if (plan.state_write_idxs.empty()) {
                uint32_t i = 0;
                while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(ubatch.seq_id[i][0], i);
            }
        } else {
            const uint32_t n_blocks = (std::max<uint32_t>(1, ubatch.n_seq_tokens) + ratio - 1)/ratio;

            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id_unq[s];
                const uint32_t n_writes = state_write_counts[seq_id];
                if (n_writes >= n_blocks) {
                    continue;
                }
                if (n_writes + 1 != n_blocks) {
                    throw std::runtime_error("DSV4 CSA sequence positions are not contiguous");
                }

                uint32_t i = 0;
                while (i < ubatch.n_tokens && (ubatch.pos[i] < 0 || !dsv4_token_has_seq(ubatch, i, seq_id))) {
                    ++i;
                }
                assert(i < ubatch.n_tokens);
                append_dummy_block(seq_id, i);
            }
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    plan.n_kv = GGML_PAD(plan.n_kv, 256u);

    std::sort(persist_rows.begin(), persist_rows.end(),
            [](const persist_row & a, const persist_row & b) {
                return a.dst < b.dst;
            });

    for (const persist_row & row : persist_rows) {
        plan.state_persist_src_idxs.push_back(row.src);
        plan.state_persist_dst_idxs.push_back(row.dst);
    }


    if (n_rs_seq > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
                continue;
            }

            const int64_t stream_off = dsv4_stream_offset(n_stream, seq_id, state_size);
            const uint32_t rollback = (uint32_t) seq_id < rs_idx.size() ? rs_idx[seq_id] : 0;
            // Keep the restore graph fixed-width when no rollback is pending.
            const int64_t src_plane = rollback > 0 && rollback <= n_rs_seq ? (int64_t) rollback*state_rows : 0;
            for (uint32_t r = 0; r < state_size; ++r) {
                plan.state_restore_src_idxs.push_back((int32_t) (src_plane + stream_off + r));
                plan.state_restore_dst_idxs.push_back((int32_t) (stream_off + r));
            }

            std::vector<uint32_t> token_idxs;
            token_idxs.reserve(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (dsv4_token_has_seq(ubatch, i, seq_id)) {
                    token_idxs.push_back(i);
                }
            }
            if (token_idxs.empty()) {
                continue;
            }

            const uint32_t n_seq_tokens = (uint32_t) token_idxs.size();
            // Snapshot execution is bounded by the active depth, but appended
            // token rows follow every plane in the allocated state tensor.
            const int64_t scratch_off = (int64_t) state_rows*(1 + n_rs_seq_alloc);
            for (uint32_t d = 1; d <= n_rs_seq; ++d) {
                const int64_t dst_plane = (int64_t) d*state_rows;

                for (uint32_t r = 0; r < state_size; ++r) {
                    int32_t src;
                    if (d <= n_seq_tokens) {
                        const uint32_t prefix = n_seq_tokens - d;
                        src = (int32_t) (stream_off + r);

                        for (uint32_t j = 0; j < prefix; ++j) {
                            const uint32_t i_tok = token_idxs[j];
                            if (ubatch.pos[i_tok] >= 0 && (uint32_t) (ubatch.pos[i_tok]%state_size) == r) {
                                src = (int32_t) (scratch_off + i_tok);
                            }
                        }
                    } else {
                        const int64_t src_plane = (int64_t) (d - n_seq_tokens)*state_rows;
                        src = (int32_t) (src_plane + stream_off + r);
                    }

                    plan.state_snapshot_src_idxs.push_back(src);
                    plan.state_snapshot_dst_idxs.push_back((int32_t) (dst_plane + stream_off + r));
                }
            }
        }
    }

    static const bool debug = []() {
        const char * env = getenv("LLAMA_DSV4_COMPRESS_DEBUG");
        return env && atoi(env) > 0;
    }();

    if (debug) {
        LLAMA_LOG_INFO("%s: ratio=%u, n_tokens=%u, state_persist_dst=%s, state_write_pos=%s\n",
                __func__, ratio, ubatch.n_tokens,
                dsv4_plan_positions(plan.state_persist_dst_idxs).c_str(),
                dsv4_plan_positions(plan.state_write_pos).c_str());
    }

    return plan;
}

static std::vector<llama_kv_cache_dsv4_context::comp_plan> dsv4_build_comp_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq,
        uint32_t n_rs_seq_alloc,
        const std::vector<uint32_t> & rs_idx) {
    std::vector<llama_kv_cache_dsv4_context::comp_plan> plans;
    plans.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(
                ubatch, ratio, overlap, state_size, kv_size, n_stream, n_rs_seq, n_rs_seq_alloc, rs_idx));
    }

    return plans;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_comp_sinfos(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_stream,
        bool aggregate) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (const llama_ubatch & ubatch : ubatches) {
        if (aggregate) {
            llama_kv_cache::slot_info sinfo;
            sinfo.s0 = 0;
            sinfo.s1 = 0;
            sinfo.resize(1);
            sinfo.strm[0] = 0;
            sinfo.idxs[0].resize(1, 0);
            sinfos.push_back(std::move(sinfo));
            continue;
        }
        if (n_stream <= 1 && ubatch.n_seqs_unq > 1) {
            throw std::runtime_error("DSV4 single compressed stream cannot serve multiple sequences");
        }

        const uint32_t ns = (uint32_t) dsv4_comp_graph_n_stream(ubatch, n_stream);
        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = n_stream > 1 ? LLAMA_MAX_SEQ : 0;
        sinfo.s1 = 0;
        sinfo.resize(ns);

        for (uint32_t s = 0; s < ns; ++s) {
            const llama_seq_id seq_id = n_stream > 1 ? ubatch.seq_id_unq[s] : 0;
            const uint32_t strm = (uint32_t) dsv4_stream_offset(n_stream, seq_id, 1);

            sinfo.s0 = std::min(sinfo.s0, strm);
            sinfo.s1 = std::max(sinfo.s1, strm);
            sinfo.strm[s] = strm;
            sinfo.idxs[s].resize(1, 0);
        }

        if (n_stream > 1 && sinfo.s1 - sinfo.s0 + 1 != ns) {
            throw std::runtime_error("DSV4 compressed streams are not contiguous in ubatch");
        }

        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache::slot_info_vec_t dsv4_build_raw_read_sinfos(
        const llama_kv_cache::slot_info_vec_t & sinfos_write,
        const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.reserve(ubatches.size());

    for (size_t i = 0; i < ubatches.size(); ++i) {
        const llama_ubatch & ubatch = ubatches[i];
        const auto & sinfo_write = sinfos_write[i];

        if (!dsv4_ubatch_has_coupled(ubatch)) {
            sinfos.push_back(sinfo_write);
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[0][0];
        uint32_t i_stream = 0;
        for (; i_stream < sinfo_write.n_stream(); ++i_stream) {
            if (sinfo_write.strm[i_stream] == seq_id) {
                break;
            }
        }
        if (i_stream == sinfo_write.n_stream()) {
            throw std::runtime_error("DSV4 raw write stream not found for coupled read");
        }

        llama_kv_cache::slot_info sinfo;
        sinfo.s0 = sinfo_write.strm[i_stream];
        sinfo.s1 = sinfo_write.strm[i_stream];
        sinfo.resize(1);
        sinfo.strm[0] = sinfo_write.strm[i_stream];
        sinfo.idxs[0] = sinfo_write.idxs[i_stream];
        sinfos.push_back(std::move(sinfo));
    }

    return sinfos;
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_reserve_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_stream,
        uint32_t n_rs_seq) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_stream = dsv4_comp_graph_n_stream(ubatch, n_stream);
    plan.n_kv = kv_size;

    if (ubatch.n_tokens == 0) {
        return plan;
    }

    const uint32_t n_seqs       = std::max<uint32_t>(1, ubatch.n_seqs);
    const uint32_t n_seq_tokens = std::max<uint32_t>(1, ubatch.n_seq_tokens);
    const uint64_t n_blocks_u64 = (uint64_t) n_seqs*((n_seq_tokens + ratio - 1)/ratio);
    const size_t n_blocks = (size_t) std::max<uint64_t>(1, n_blocks_u64);
    GGML_ASSERT((uint64_t) n_blocks == std::max<uint64_t>(1, n_blocks_u64));

    const uint64_t state_rows = (uint64_t) state_size*n_stream;
    const size_t n_persist = (size_t) std::min<uint64_t>(ubatch.n_tokens, state_rows);
    const size_t n_restore = n_rs_seq > 0 ? (size_t) state_size*std::max<uint32_t>(1, ubatch.n_seqs_unq) : 0;
    const size_t n_snapshot = (size_t) n_rs_seq*state_size*std::max<uint32_t>(1, ubatch.n_seqs_unq);

    plan.state_pos .resize(ubatch.n_tokens);
    plan.state_persist_src_idxs.resize(n_persist);
    plan.state_persist_dst_idxs.resize(n_persist);
    plan.state_restore_src_idxs.resize(n_restore);
    plan.state_restore_dst_idxs.resize(n_restore);
    plan.state_snapshot_src_idxs.resize(n_snapshot);
    plan.state_snapshot_dst_idxs.resize(n_snapshot);
    plan.state_read_idxs .resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_logical_idxs.resize(n_blocks);
    plan.state_write_dummy.resize(n_blocks);
    plan.state_write_pos .resize(n_blocks);

    return plan;
}

static void dsv4_make_k_only(llama_hparams & hparams) {
    // llama_kv_cache uses hparams.is_mla() to allocate K-only storage.
    hparams.n_embd_head_k_mla_impl = hparams.n_embd_head_k();
    hparams.n_embd_head_v_mla_impl = hparams.n_embd_head_k();
}

//
// llama_dsv4_comp_state
//

llama_dsv4_comp_state::llama_dsv4_comp_state(
        const llama_model & model,
                bool        offload,
                bool        unified,
            uint32_t        n_seq_max,
            uint32_t        ratio,
            uint32_t        state_size,
            uint32_t        n_embd_state,
            uint32_t        n_rs_seq,
        const char    * name,
        const llama_memory_i::layer_filter_cb & filter) :
    ratio(ratio),
    state_size(state_size),
    n_embd_state(n_embd_state),
    n_stream(unified ? 1 : n_seq_max),
    n_rs_seq(n_rs_seq) {
    const llama_hparams & hparams = model.hparams;

    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream*(1 + n_rs_seq))*hparams.n_layer()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for DSV4 compressor state");
        }

        const uint32_t n_planes = n_stream*(1 + n_rs_seq);
        ggml_tensor * kv    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);
        ggml_tensor * score = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd_state, state_size, n_planes);

        ggml_format_name(kv,    "dsv4_%s_state_kv_l%d",    name, il);
        ggml_format_name(score, "dsv4_%s_state_score_l%d", name, il);

        std::vector<ggml_tensor *> kv_stream;
        std::vector<ggml_tensor *> score_stream;

        for (uint32_t s = 0; s < n_planes; ++s) {
            kv_stream.push_back(ggml_view_2d(ctx, kv, n_embd_state, state_size, kv->nb[1], s*kv->nb[2]));
            score_stream.push_back(ggml_view_2d(ctx, score, n_embd_state, state_size, score->nb[1], s*score->nb[2]));
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, kv, score, std::move(kv_stream), std::move(score_stream) });
    }

    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for DSV4 compressor state");
        }

        ggml_backend_buffer_clear(buf, 0);

        LLAMA_LOG_INFO("%s: %10s DSV4 %s state buffer size = %8.2f MiB\n",
                __func__, ggml_backend_buffer_name(buf), name, ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    LLAMA_LOG_INFO("%s: %s ratio = %u, state = %u x %u, streams = %u, rs_seq = %u, layers = %zu, size = %7.2f MiB\n",
            __func__, name, ratio, state_size, n_embd_state, n_stream, n_rs_seq, layers.size(), total_size()/1024.0/1024.0);
}

void llama_dsv4_comp_state::clear(llama_seq_id seq_id, bool data) {
    if (!data) {
        return;
    }

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_stream);

        for (const auto & layer : layers) {
            for (uint32_t d = 0; d <= n_rs_seq; ++d) {
                const uint32_t stream = d*n_stream + (uint32_t) seq_id;
                dsv4_clear_tensor_stream(layer.kv,    stream);
                dsv4_clear_tensor_stream(layer.score, stream);
            }
        }
        return;
    }

    for (auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_clear(buf.get(), 0);
    }
}

void llama_dsv4_comp_state::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, uint32_t src_depth) {
    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < n_stream);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_stream);
    GGML_ASSERT(src_depth <= n_rs_seq);

    if (seq_id_src == seq_id_dst) {
        return;
    }

    clear(seq_id_dst, true);

    sc_info.ssrc.push_back(src_depth*n_stream + (uint32_t) seq_id_src);
    sc_info.sdst.push_back((uint32_t) seq_id_dst);
}

void llama_dsv4_comp_state::apply_copies(const stream_copy_info & sc_info) const {
    for (size_t i = 0; i < sc_info.ssrc.size(); ++i) {
        const uint32_t ssrc = sc_info.ssrc[i];
        const uint32_t sdst = sc_info.sdst[i];

        for (const auto & layer : layers) {
            ggml_backend_tensor_copy(layer.kv_stream[ssrc], layer.kv_stream[sdst]);
            ggml_backend_tensor_copy(layer.score_stream[ssrc], layer.score_stream[sdst]);
        }
    }
}

uint32_t llama_dsv4_comp_state::get_ratio() const {
    return ratio;
}

uint32_t llama_dsv4_comp_state::get_state_size() const {
    return state_size;
}

uint32_t llama_dsv4_comp_state::get_n_stream() const {
    return n_stream;
}

uint32_t llama_dsv4_comp_state::get_n_rs_seq() const {
    return n_rs_seq;
}

uint32_t llama_dsv4_comp_state::get_n_rows() const {
    return state_size*n_stream;
}

uint64_t llama_dsv4_comp_state::state_identity() const {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash = dsv4_hash_u64(hash, ratio);
    hash = dsv4_hash_u64(hash, state_size);
    hash = dsv4_hash_u64(hash, n_embd_state);
    hash = dsv4_hash_u64(hash, n_stream);
    hash = dsv4_hash_u64(hash, n_rs_seq);
    hash = dsv4_hash_u64(hash, layers.size());
    for (const auto & layer : layers) {
        hash = dsv4_hash_u64(hash, layer.il);
        hash = dsv4_hash_tensor(hash, layer.kv);
        hash = dsv4_hash_tensor(hash, layer.score);
    }
    return hash;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_dsv4_comp_state::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [_, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        ret[buft] += ggml_backend_buffer_get_size(buf.get());
    }
    return ret;
}

void llama_dsv4_comp_state::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags,
        const std::vector<uint32_t> & rs_idx) const {
    GGML_UNUSED(flags);

    uint32_t s0;
    uint32_t ns;
    dsv4_state_src_stream_range(n_stream, seq_id, s0, ns);

    std::vector<uint32_t> stream_ids(ns);
    for (uint32_t s = 0; s < ns; ++s) {
        const uint32_t seq = seq_id >= 0 ? (uint32_t) seq_id : s0 + s;
        if (seq >= rs_idx.size() || rs_idx[seq] > n_rs_seq) {
            throw std::runtime_error("DSV4 recurrent state rollback index out of range");
        }
        stream_ids[s] = rs_idx[seq]*n_stream + s0 + s;
    }

    const uint32_t version      = DSV4_COMP_STATE_VER;
    const uint32_t n_layer      = layers.size();

    io.write(&version,      sizeof(version));
    io.write(&ratio,        sizeof(ratio));
    io.write(&state_size,   sizeof(state_size));
    io.write(&n_embd_state, sizeof(n_embd_state));
    io.write(&ns,           sizeof(ns));
    io.write(&n_layer,      sizeof(n_layer));

    for (const auto & layer : layers) {
        io.write(&layer.il, sizeof(layer.il));

        dsv4_state_write_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns, &stream_ids);
        dsv4_state_write_tensor_streams(io, layer.score, state_size, state_size, s0, ns, &stream_ids);
    }
}

void llama_dsv4_comp_state::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t version;
    uint32_t ratio_ref;
    uint32_t state_size_ref;
    uint32_t n_embd_state_ref;
    uint32_t ns;
    uint32_t n_layer_ref;

    io.read(&version,          sizeof(version));
    io.read(&ratio_ref,        sizeof(ratio_ref));
    io.read(&state_size_ref,   sizeof(state_size_ref));
    io.read(&n_embd_state_ref, sizeof(n_embd_state_ref));
    io.read(&ns,               sizeof(ns));
    io.read(&n_layer_ref,      sizeof(n_layer_ref));

    if (version != DSV4_COMP_STATE_VER) {
        throw std::runtime_error("DSV4 compressor state version mismatch");
    }
    if (ratio_ref != ratio || state_size_ref != state_size || n_embd_state_ref != n_embd_state) {
        throw std::runtime_error("DSV4 compressor state metadata mismatch");
    }
    if (n_layer_ref != layers.size()) {
        throw std::runtime_error("DSV4 compressor state layer count mismatch");
    }

    uint32_t s0;
    dsv4_state_dst_stream_range(n_stream, seq_id, ns, s0);

    for (const auto & layer : layers) {
        uint32_t il_ref;
        io.read(&il_ref, sizeof(il_ref));
        if (il_ref != layer.il) {
            throw std::runtime_error("DSV4 compressor state layer id mismatch");
        }

        dsv4_state_read_tensor_streams(io, layer.kv,    state_size, state_size, s0, ns);
        dsv4_state_read_tensor_streams(io, layer.score, state_size, state_size, s0, ns);
    }
}

ggml_tensor * llama_dsv4_comp_state::get_kv_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].kv;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_score_all(ggml_context * ctx, int32_t il) const {
    const int32_t ids = map_layer_ids.at(il);
    ggml_tensor * state = layers[ids].score;

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows()*(1 + n_rs_seq), state->nb[1], 0);
}

ggml_tensor * llama_dsv4_comp_state::get_kv(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_kv_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::get_score(ggml_context * ctx, int32_t il) const {
    ggml_tensor * state = get_score_all(ctx, il);
    const size_t row_size = ggml_row_size(state->type, state->ne[0]);

    return ggml_view_2d(ctx, state, state->ne[0], get_n_rows(), state->nb[1], 0*row_size);
}

ggml_tensor * llama_dsv4_comp_state::cpy_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_kv_all(ctx, il), cur, idxs);
}

ggml_tensor * llama_dsv4_comp_state::cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_score_all(ctx, il), cur, idxs);
}

size_t llama_dsv4_comp_state::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

//
// llama_kv_cache_dsv4
//

llama_kv_cache_dsv4::llama_kv_cache_dsv4(
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
    const  layer_reuse_cb & reuse) :
    hparams_raw(model.hparams),
    hparams_csa(model.hparams),
    hparams_hca(model.hparams),
    hparams_lid(model.hparams),
    n_seq_max(n_seq_max),
    n_rs_seq(n_rs_seq),
    n_rs_seq_active(n_rs_seq),
    rs_idx(n_seq_max, 0) {

    const layer_filter_cb filter_raw = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return true;
    };

    ggml_backend_buffer_type_t sparse_buft = nullptr;
    if (unified && n_seq_max > 1) {
        sparse_buft = dsv4_sparse_buft(model, n_seq_max);
        if (sparse_buft == nullptr) {
            throw std::runtime_error("DSV4 unified context requires target Metal sparse pages");
        }
    }

    c4_logical_rows  = GGML_PAD(dsv4_comp_size(kv_size, DSV4_CSA_RATIO), 256u);
    hca_logical_rows = GGML_PAD(dsv4_comp_size(kv_size, DSV4_HCA_RATIO), 256u);

    // The aggregate layout is target-only and currently requires the F16
    // indexed concat path. Keep the affine layout for other cache formats.
    aggregate_compressed = unified && n_seq_max > 1 && sparse_buft != nullptr && type_k == GGML_TYPE_F16 &&
            std::getenv("LLAMA_DSV4_AGGREGATE_POOL_DISABLE") == nullptr;

    uint32_t c4_storage_rows  = c4_logical_rows;
    uint32_t hca_storage_rows = hca_logical_rows;
    ggml_backend_buffer_type_t compressed_buft = sparse_buft;
    if (aggregate_compressed) {
        const uint32_t tail_segments = n_seq_max - 1;
        llama_dsv4_comp_pool_config config {
            (uint32_t) llama_dsv4_comp_segments_for_rows(c4_logical_rows) + tail_segments,
            (uint32_t) llama_dsv4_comp_segments_for_rows(hca_logical_rows) + tail_segments,
        };
        c4_storage_rows  = (config.c4_data_segments  + 2)*LLAMA_DSV4_COMP_SEGMENT_ROWS;
        hca_storage_rows = (config.hca_data_segments + 2)*LLAMA_DSV4_COMP_SEGMENT_ROWS;
        comp_pool = std::make_unique<llama_dsv4_comp_pool>(config);
        for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
            const auto handle = comp_pool->create_handle();
            if (handle.status != llama_dsv4_comp_status::ok ||
                comp_pool->bind(seq, handle.handle) != llama_dsv4_comp_status::ok) {
                throw std::runtime_error("failed to initialize DSV4 aggregate compressed-pool binding");
            }
        }
        compressed_buft = dsv4_sparse_buft(model, 1);
        if (compressed_buft == nullptr) {
            throw std::runtime_error("DSV4 aggregate compressed pool requires target Metal sparse storage");
        }
        LLAMA_LOG_INFO("%s: aggregate DSV4 compressed pool enabled: C4 logical=%u storage=%u rows, HCA logical=%u storage=%u rows, slots=%u\n",
                __func__, c4_logical_rows, c4_storage_rows, hca_logical_rows, hca_storage_rows, n_seq_max);
    }

    // Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
    const bool unified_raw = false;

    hparams_raw.n_layer_nextn = 0;
    hparams_csa.n_layer_nextn = 0;
    hparams_hca.n_layer_nextn = 0;
    hparams_lid.n_layer_nextn = 0;

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    dsv4_make_k_only(hparams_raw);

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr);

    dsv4_make_k_only(hparams_csa);
    dsv4_make_k_only(hparams_hca);

    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_k_swa  = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_swa  = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
    dsv4_make_k_only(hparams_lid);

    const layer_filter_cb filter_csa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_hca = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    const bool unified_compressed = aggregate_compressed;

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_csa = std::make_unique<llama_kv_cache>(
            model, hparams_csa, type_k, type_v,
            v_trans, offload, unified_compressed, c4_storage_rows, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr, compressed_buft);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressed KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_HCA_RATIO));

    kv_hca = std::make_unique<llama_kv_cache>(
            model, hparams_hca, type_k, type_v,
            v_trans, offload, unified_compressed, hca_storage_rows, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_hca, nullptr, nullptr, compressed_buft);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer KV cache, size = %u cells\n",
            __func__, dsv4_comp_size(kv_size, DSV4_CSA_RATIO));

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            v_trans, offload, unified_compressed, c4_storage_rows, n_seq_max, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr, compressed_buft);

    LLAMA_LOG_INFO("%s: creating DSV4 CSA compressor state\n", __func__);

    csa_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, false, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.n_embd_head_k(), n_rs_seq, "csa", filter_csa);

    LLAMA_LOG_INFO("%s: creating DSV4 HCA compressor state\n", __func__);

    hca_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, false, n_seq_max, DSV4_HCA_RATIO, DSV4_HCA_RATIO,
            model.hparams.n_embd_head_k(), n_rs_seq, "hca", filter_hca);

    LLAMA_LOG_INFO("%s: creating DSV4 lightning-indexer compressor state\n", __func__);

    lid_state = std::make_unique<llama_dsv4_comp_state>(
            model, offload, false, n_seq_max, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
            2*model.hparams.indexer_head_size, n_rs_seq, "lid", filter_csa);

    // Bind sequence snapshots to the stable model metadata and every cache /
    // compressor geometry that controls interpretation of their tensor bytes.
    // llama.cpp does not currently expose a persistent model-content digest,
    // so this deliberately identifies the artifact geometry, not its weights.
    uint64_t identity = UINT64_C(14695981039346656037);
    identity = dsv4_hash_string(identity, "llama.cpp-dsv4-sequence-state-v2");
    identity = dsv4_hash_u64(identity, model.arch);
    identity = dsv4_hash_u64(identity, model.size());
    identity = dsv4_hash_u64(identity, model.n_elements());
    identity = dsv4_hash_u64(identity, model.n_tensors());
    identity = dsv4_hash_u64(identity, n_seq_max);
    identity = dsv4_hash_u64(identity, n_rs_seq);
    identity = dsv4_hash_u64(identity, hparams_raw.n_ctx_train);
    identity = dsv4_hash_u64(identity, hparams_raw.n_embd);
    identity = dsv4_hash_u64(identity, hparams_raw.n_layer());
    identity = dsv4_hash_u64(identity, hparams_raw.n_swa);
    identity = dsv4_hash_u64(identity, hparams_raw.swa_type);
    identity = dsv4_hash_u64(identity, hparams_raw.indexer_n_head);
    identity = dsv4_hash_u64(identity, hparams_raw.indexer_head_size);
    identity = dsv4_hash_u64(identity, hparams_raw.indexer_top_k);
    identity = dsv4_hash_u64(identity, hparams_raw.indexer_block_size);
    identity = dsv4_hash_u64(identity, hparams_raw.indexer_local_blocks);
    for (uint32_t il = 0; il < hparams_raw.n_layer(); ++il) {
        identity = dsv4_hash_u64(identity, hparams_raw.n_head(il));
        identity = dsv4_hash_u64(identity, hparams_raw.n_head_kv(il));
        identity = dsv4_hash_u64(identity, hparams_raw.n_embd_k_gqa(il));
        identity = dsv4_hash_u64(identity, hparams_raw.n_embd_v_gqa(il));
        identity = dsv4_hash_u64(identity, hparams_raw.is_swa(il));
        identity = dsv4_hash_u64(identity, hparams_raw.dsv4_compress_ratios[il]);
    }
    std::map<std::string, std::string> metadata(model.gguf_kv.begin(), model.gguf_kv.end());
    identity = dsv4_hash_u64(identity, metadata.size());
    for (const auto & [key, value] : metadata) {
        identity = dsv4_hash_string(identity, key);
        identity = dsv4_hash_string(identity, value);
    }
    identity = dsv4_hash_cache(identity, kv_raw->get_base());
    identity = dsv4_hash_cache(identity, kv_raw->get_swa());
    identity = dsv4_hash_cache(identity, kv_csa.get());
    identity = dsv4_hash_cache(identity, kv_hca.get());
    identity = dsv4_hash_cache(identity, kv_lid.get());
    identity = dsv4_hash_u64(identity, csa_state->state_identity());
    identity = dsv4_hash_u64(identity, hca_state->state_identity());
    identity = dsv4_hash_u64(identity, lid_state->state_identity());
    state_identity_hash = identity;

    // DSV4 attention reads compressed-K / compressor-state rows that the current
    // graph does not necessarily overwrite; uninitialized buffer contents would
    // otherwise leak in (instance-specific garbage) and corrupt recall. Zero all
    // compressed buffers up front so reads of un-written rows are deterministic.
    clear_compressed(-1, true);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    const bool raw_per_seq  = kv_raw->get_base()->get_n_stream() != 1;
    const bool comp_per_seq = csa_state->get_n_stream() > 1;
    const bool has_coupled = dsv4_batch_has_coupled(balloc.get_batch());

    const auto make_context = [&](std::vector<llama_ubatch> ubatches) -> llama_memory_context_ptr {
        auto ubatches_raw = dsv4_build_raw_write_ubatches(ubatches);

        auto sinfos_raw_base_write = kv_raw->get_base()->prepare(ubatches_raw);
        if (sinfos_raw_base_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_write = kv_raw->get_swa()->prepare(ubatches_raw);
        if (sinfos_raw_swa_write.empty()) {
            return nullptr;
        }

        auto sinfos_raw_swa_read = dsv4_build_raw_read_sinfos(sinfos_raw_swa_write, ubatches);

        return std::make_unique<llama_kv_cache_dsv4_context>(
                this,
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                std::move(ubatches),
                std::move(ubatches_raw));
    };

    // Match llama_kv_cache_iswa splitting when DSV4 compressed state does not
    // require per-sequence graph layout.
    do {
        if (raw_per_seq || comp_per_seq) {
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
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    // When raw or compressed state is per-sequence, independent sequences can
    // share an equal-length ubatch. Coupled sequence sets still serialize until
    // DSV4 has explicit shared-state handling for compressed streams.
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;
            if (has_coupled) {
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                ubatch = balloc.split_equal(n_ubatch, raw_per_seq || comp_per_seq, 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        if (auto ctx = make_context(std::move(ubatches))) {
            return ctx;
        }
    } while (false);

    return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_full() {
    return std::make_unique<llama_kv_cache_dsv4_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv4_context>(
            this,
            lctx,
            optimize,
            std::move(csa_state->sc_info),
            std::move(hca_state->sc_info),
            std::move(lid_state->sc_info));
}

bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}

void llama_kv_cache_dsv4::clear(bool data) {
    kv_raw->clear(data);
    clear_compressed(-1, true); // DSV4 compressed buffers must never expose stale/uninit rows
}

bool llama_kv_cache_dsv4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        return false;
    }

    if (p0 > 0) {
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
            return false;
        }

        const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);
        if (p0 > pos_max) {
            bool res = true;

            res = res & kv_raw->seq_rm(seq_id, p0, -1);
            if (!aggregate_compressed) {
                res = res & kv_csa->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);
                res = res & kv_hca->seq_rm(seq_id, p0/DSV4_HCA_RATIO, -1);
                res = res & kv_lid->seq_rm(seq_id, p0/DSV4_CSA_RATIO, -1);
            }

            return res;
        }

        if (n_rs_seq_active == 0) {
            return false;
        }

        const llama_pos rollback = pos_max - (p0 - 1);
        if (rollback < 1 || rollback > (llama_pos) n_rs_seq_active) {
            return false;
        }

        const bool res = kv_raw->seq_rm(seq_id, p0, p1);
        if (res) {
            rs_idx[seq_id] = (uint32_t) rollback;
        }

        return res;
    }

    const bool res = kv_raw->seq_rm(seq_id, p0, p1);

    if (res) {
        clear_compressed(seq_id, true);
    }

    return res;
}

void llama_kv_cache_dsv4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "DSV4 only supports full sequence copies");

    kv_raw->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    if (aggregate_compressed && seq_id_src != seq_id_dst) {
        llama_dsv4_comp_handle_id source = 0;
        llama_dsv4_comp_handle_id destination = 0;
        GGML_ASSERT(comp_pool->get_binding((uint32_t) seq_id_src, source) == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->get_binding((uint32_t) seq_id_dst, destination) == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->unbind((uint32_t) seq_id_dst) == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->remove_handle(destination) == llama_dsv4_comp_status::ok);
        const auto copy = comp_pool->copy_handle(source);
        GGML_ASSERT(copy.status == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->bind((uint32_t) seq_id_dst, copy.handle) == llama_dsv4_comp_status::ok);
    } else if (!aggregate_compressed) {
        kv_csa->seq_cp(seq_id_src, seq_id_dst, -1, -1);
        kv_hca->seq_cp(seq_id_src, seq_id_dst, -1, -1);
        kv_lid->seq_cp(seq_id_src, seq_id_dst, -1, -1);
    }

    const uint32_t src_depth = rs_idx[seq_id_src];
    csa_state->seq_cp(seq_id_src, seq_id_dst, src_depth);
    hca_state->seq_cp(seq_id_src, seq_id_dst, src_depth);
    lid_state->seq_cp(seq_id_src, seq_id_dst, src_depth);

    if (seq_id_src != seq_id_dst) {
        rs_idx[seq_id_dst] = 0;
    }
}

void llama_kv_cache_dsv4::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    kv_raw->seq_keep(seq_id);

    for (llama_seq_id id = 0; id < (llama_seq_id) n_seq_max; ++id) {
        if (id == seq_id) {
            continue;
        }

        kv_raw->seq_rm(id, -1, -1);
        clear_compressed(id, true);
    }
}

void llama_kv_cache_dsv4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_raw->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsv4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_raw->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsv4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    // The raw SWA cache may contain a wider window, but the compressed DSV4
    // state cannot be rolled back within that window. Report only the current
    // boundary so server-context uses checkpoints for rollback.
    return kv_raw->seq_pos_max(seq_id);
}

llama_pos llama_kv_cache_dsv4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return kv_raw->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv4::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = kv_raw->memory_breakdown();
    for (const auto & buft_size : kv_csa->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_hca->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : kv_lid->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : csa_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : hca_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    for (const auto & buft_size : lid_state->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

static llama_dsv4_sparse_pool_usage dsv4_convert_sparse_usage(
        const ggml_metal_sparse_usage & src) {
    return {
        src.pool_id,
        src.page_size,
        src.virtual_pages,
        src.physical_pages,
        src.free_pages,
        src.reserved_pages,
        src.mapped_mappings,
        src.unique_physical_pages,
        src.shared_physical_pages,
        src.shared_mappings,
        src.refcount_sum,
        src.refcount_max,
        src.generation,
        src.cow_allocations,
        src.cow_pages,
    };
}

struct dsv4_sparse_buffer_snapshot {
    int status = 0;
    ggml_metal_sparse_usage usage = {};
};

using dsv4_sparse_buffer_snapshots =
        std::map<ggml_backend_buffer_t, dsv4_sparse_buffer_snapshot>;

static llama_dsv4_family_usage dsv4_family_usage(
        const llama_kv_cache * cache,
        llama_dsv4_memory_family family,
        dsv4_sparse_buffer_snapshots & snapshots) {
    llama_dsv4_family_usage result;
    result.family = family;
    result.logical_capacity_rows = (uint64_t) cache->get_size()*cache->get_n_stream();
    result.sequence_mapped_rows.reserve(cache->get_n_stream());
    for (uint32_t seq = 0; seq < cache->get_n_stream(); ++seq) {
        const uint64_t used = cache->get_cells((llama_seq_id) seq).get_used();
        result.sequence_mapped_rows.push_back(used);
        result.logical_mapped_rows += used;
    }

    using usage_fn = int (*)(const ggml_tensor *, ggml_metal_sparse_usage *);
    std::map<uintptr_t, llama_dsv4_sparse_pool_usage> unique;
    for (uint32_t il : cache->get_layer_ids()) {
        ggml_tensor * tensor = cache->get_k_storage(il);
        const auto [it, inserted] = snapshots.try_emplace(tensor->buffer);
        if (inserted) {
            auto * usage = (usage_fn) dsv4_backend_proc(
                    tensor, "ggml_backend_metal_dsv4_sparse_tensor_usage");
            if (usage != nullptr) {
                it->second.status = usage(tensor, &it->second.usage);
            }
        }
        const auto & snapshot = it->second;
        if (snapshot.status < 0) {
            throw std::runtime_error("failed to read DSV4 sparse usage snapshot");
        }
        if (snapshot.status > 0) {
            if (!dsv4_sparse_pool_insert(unique, dsv4_convert_sparse_usage(snapshot.usage))) {
                throw std::runtime_error("inconsistent DSV4 sparse pool usage snapshot");
            }
        }
    }
    for (const auto & [_, pool] : unique) {
        result.pools.push_back(pool);
        dsv4_sparse_usage_add(result.total, pool);
    }
    result.placement_sparse = !result.pools.empty();
    return result;
}

static void dsv4_set_compressed_logical_usage(
        llama_dsv4_family_usage & result,
        const llama_kv_cache * cache,
        const llama_kv_cache_iswa * raw,
        uint32_t ratio) {
    result.logical_capacity_rows = (uint64_t) cache->get_size()*cache->get_n_stream();
    result.logical_mapped_rows = 0;
    result.sequence_mapped_rows.clear();
    result.sequence_mapped_rows.reserve(cache->get_n_stream());
    for (uint32_t seq = 0; seq < cache->get_n_stream(); ++seq) {
        const uint64_t used = dsv4_state_n_used_k_rows(
                raw->seq_pos_max((llama_seq_id) seq), ratio, cache->get_size());
        result.sequence_mapped_rows.push_back(used);
        result.logical_mapped_rows += used;
    }
}

llama_dsv4_memory_usage_snapshot llama_kv_cache_dsv4::memory_usage_snapshot() const {
    llama_dsv4_memory_usage_snapshot result;
    dsv4_sparse_buffer_snapshots snapshots;
    result.families[LLAMA_DSV4_MEMORY_RAW] =
            dsv4_family_usage(kv_raw->get_swa(), LLAMA_DSV4_MEMORY_RAW, snapshots);
    const bool raw_pools_merged = dsv4_family_sparse_usage_merge(
            result.families[LLAMA_DSV4_MEMORY_RAW],
            dsv4_family_usage(kv_raw->get_base(), LLAMA_DSV4_MEMORY_RAW, snapshots));
    if (!raw_pools_merged) {
        throw std::runtime_error("inconsistent RAW DSV4 sparse pool usage snapshot");
    }
    result.families[LLAMA_DSV4_MEMORY_CSA] =
            dsv4_family_usage(kv_csa.get(), LLAMA_DSV4_MEMORY_CSA, snapshots);
    result.families[LLAMA_DSV4_MEMORY_HCA] =
            dsv4_family_usage(kv_hca.get(), LLAMA_DSV4_MEMORY_HCA, snapshots);
    result.families[LLAMA_DSV4_MEMORY_LID] =
            dsv4_family_usage(kv_lid.get(), LLAMA_DSV4_MEMORY_LID, snapshots);

    dsv4_set_compressed_logical_usage(
            result.families[LLAMA_DSV4_MEMORY_CSA], kv_csa.get(), kv_raw.get(), DSV4_CSA_RATIO);
    dsv4_set_compressed_logical_usage(
            result.families[LLAMA_DSV4_MEMORY_HCA], kv_hca.get(), kv_raw.get(), DSV4_HCA_RATIO);
    dsv4_set_compressed_logical_usage(
            result.families[LLAMA_DSV4_MEMORY_LID], kv_lid.get(), kv_raw.get(), DSV4_CSA_RATIO);

    if (aggregate_compressed) {
        const auto set_aggregate_logical = [&](llama_dsv4_family_usage & usage, uint32_t logical_rows, uint32_t ratio) {
            usage.logical_capacity_rows = logical_rows;
            usage.logical_mapped_rows = 0;
            usage.sequence_mapped_rows.clear();
            usage.sequence_mapped_rows.reserve(n_seq_max);
            for (uint32_t seq = 0; seq < n_seq_max; ++seq) {
                const uint64_t used = dsv4_state_n_used_k_rows(
                        kv_raw->seq_pos_max((llama_seq_id) seq), ratio, logical_rows);
                usage.sequence_mapped_rows.push_back(used);
                usage.logical_mapped_rows += used;
            }
        };
        set_aggregate_logical(result.families[LLAMA_DSV4_MEMORY_CSA], c4_logical_rows, DSV4_CSA_RATIO);
        set_aggregate_logical(result.families[LLAMA_DSV4_MEMORY_HCA], hca_logical_rows, DSV4_HCA_RATIO);
        set_aggregate_logical(result.families[LLAMA_DSV4_MEMORY_LID], c4_logical_rows, DSV4_CSA_RATIO);
    }

    if (!dsv4_memory_usage_finalize(result)) {
        throw std::runtime_error("inconsistent shared DSV4 sparse pool usage snapshot");
    }
    return result;
}

void llama_kv_cache_dsv4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (aggregate_compressed && !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("full DSV4 state write is not yet supported by the aggregate compressed pool");
    }
    const bool partial_only = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    const uint32_t magic   = DSV4_STATE_MAGIC;
    const uint32_t version = DSV4_STATE_VERSION;
    const uint32_t mode    = partial_only ? DSV4_STATE_MODE_PARTIAL : DSV4_STATE_MODE_FULL;

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&mode,    sizeof(mode));
    io.write(&state_identity_hash, sizeof(state_identity_hash));

    kv_raw->state_write(io, seq_id, flags);

    if (!partial_only) {
        const llama_pos pos_max = seq_id >= 0 ? kv_raw->seq_pos_max(seq_id) : -1;

        //FIXME : note that we conflate token positions with rows, which is not true for multi-modal case.
        const uint32_t n_rows_csa = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_csa->get_size()) : kv_csa->get_size();
        const uint32_t n_rows_hca = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_HCA_RATIO, kv_hca->get_size()) : kv_hca->get_size();
        const uint32_t n_rows_lid = seq_id >= 0 ?
            dsv4_state_n_used_k_rows(pos_max, DSV4_CSA_RATIO, kv_lid->get_size()) : kv_lid->get_size();

        dsv4_state_write_k_cache(io, kv_csa.get(), seq_id, flags, n_rows_csa);
        dsv4_state_write_k_cache(io, kv_hca.get(), seq_id, flags, n_rows_hca);
        dsv4_state_write_k_cache(io, kv_lid.get(), seq_id, flags, n_rows_lid);
    }

    csa_state->state_write(io, seq_id, flags, rs_idx);
    hca_state->state_write(io, seq_id, flags, rs_idx);
    lid_state->state_write(io, seq_id, flags, rs_idx);
}

void llama_kv_cache_dsv4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (aggregate_compressed && !(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("full DSV4 state restore is not yet supported by the aggregate compressed pool");
    }
    uint32_t magic;
    uint32_t version;
    uint32_t mode = DSV4_STATE_MODE_FULL;
    uint64_t identity;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));

    if (magic != DSV4_STATE_MAGIC) {
        throw std::runtime_error("DSV4 state magic mismatch");
    }
    if (version != DSV4_STATE_VERSION) {
        throw std::runtime_error("DSV4 state version mismatch");
    }

    io.read(&mode, sizeof(mode));
    if (mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL) {
        throw std::runtime_error("DSV4 state mode mismatch");
    }

    const bool partial_only = mode == DSV4_STATE_MODE_PARTIAL;
    if (partial_only != !!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("DSV4 state flags mismatch");
    }

    io.read(&identity, sizeof(identity));
    if (identity != state_identity_hash) {
        throw std::runtime_error("DSV4 state model/cache geometry identity mismatch");
    }

    if (!partial_only) {
        if (seq_id >= 0) {
            if (!seq_rm(seq_id, -1, -1)) {
                throw std::runtime_error("failed to clear DSV4 restore destination");
            }
        } else {
            clear(true);
        }
    }

    kv_raw->state_read(io, seq_id, flags);

    if (!partial_only) {
        dsv4_state_read_k_cache(io, kv_csa.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_hca.get(), seq_id, flags);
        dsv4_state_read_k_cache(io, kv_lid.get(), seq_id, flags);
    }

    csa_state->state_read(io, seq_id, flags);
    hca_state->state_read(io, seq_id, flags);
    lid_state->state_read(io, seq_id, flags);

    if (seq_id >= 0) {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }
}

llama_kv_cache_iswa * llama_kv_cache_dsv4::get_raw() const {
    return kv_raw.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_csa() const {
    return kv_csa.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_hca() const {
    return kv_hca.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_lid() const {
    return kv_lid.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_csa_state() const {
    return csa_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_hca_state() const {
    return hca_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_lid_state() const {
    return lid_state.get();
}

bool llama_kv_cache_dsv4::is_aggregate_compressed() const {
    return aggregate_compressed;
}

uint32_t llama_kv_cache_dsv4::get_c4_logical_rows() const {
    return c4_logical_rows;
}

uint32_t llama_kv_cache_dsv4::get_hca_logical_rows() const {
    return hca_logical_rows;
}

llama_dsv4_comp_pool * llama_kv_cache_dsv4::get_comp_pool() const {
    return comp_pool.get();
}

uint32_t llama_kv_cache_dsv4::get_n_rs_seq() const {
    return n_rs_seq_active;
}

const std::vector<uint32_t> & llama_kv_cache_dsv4::get_rs_idx() const {
    return rs_idx;
}

bool llama_kv_cache_dsv4::set_rs_enabled(bool enabled) {
    return set_rs_depth(enabled ? n_rs_seq : 0);
}

bool llama_kv_cache_dsv4::set_rs_depth(uint32_t depth) {
    if (depth > n_rs_seq) {
        throw std::runtime_error("DSV4 active rollback depth exceeds allocated capacity");
    }
    if (n_rs_seq_active == depth) {
        return false;
    }

    n_rs_seq_active = depth;
    std::fill(rs_idx.begin(), rs_idx.end(), 0);
    return true;
}

void llama_kv_cache_dsv4::reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches) {
    if (n_rs_seq_active == 0) {
        return;
    }

    for (const llama_ubatch & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id[i][s];
                if (seq_id >= 0 && (uint32_t) seq_id < n_seq_max) {
                    rs_idx[seq_id] = 0;
                }
            }
        }
    }
}

void llama_kv_cache_dsv4::clear_compressed(llama_seq_id seq_id, bool data) {
    if (aggregate_compressed && seq_id < 0) {
        if (data) {
            kv_csa->clear(true);
            kv_hca->clear(true);
            kv_lid->clear(true);
        }
        const auto usage = comp_pool->memory_usage_snapshot();
        comp_pool = std::make_unique<llama_dsv4_comp_pool>(llama_dsv4_comp_pool_config {
                usage.c4.capacity_segments - usage.c4.permanent_segments,
                usage.hca.capacity_segments - usage.hca.permanent_segments,
        });
        for (uint32_t execution_id = 0; execution_id < n_seq_max; ++execution_id) {
            const auto handle = comp_pool->create_handle();
            GGML_ASSERT(handle.status == llama_dsv4_comp_status::ok);
            GGML_ASSERT(comp_pool->bind(execution_id, handle.handle) == llama_dsv4_comp_status::ok);
        }
    } else if (aggregate_compressed) {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);
        llama_dsv4_comp_handle_id old_handle = 0;
        GGML_ASSERT(comp_pool->get_binding((uint32_t) seq_id, old_handle) == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->unbind((uint32_t) seq_id) == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->remove_handle(old_handle) == llama_dsv4_comp_status::ok);
        const auto replacement = comp_pool->create_handle();
        GGML_ASSERT(replacement.status == llama_dsv4_comp_status::ok);
        GGML_ASSERT(comp_pool->bind((uint32_t) seq_id, replacement.handle) == llama_dsv4_comp_status::ok);
    } else if (seq_id < 0) {
        kv_csa->clear(data);
        kv_hca->clear(data);
        kv_lid->clear(data);
    } else {
        GGML_ASSERT((uint32_t) seq_id < n_seq_max);

        const auto clear_seq = [seq_id, data](llama_kv_cache * kv) {
            kv->seq_rm(seq_id, -1, -1);

            if (data) {
                for (uint32_t il : kv->get_layer_ids()) {
                    dsv4_clear_tensor_stream(kv->get_k_storage(il), (uint32_t) seq_id);
                }
            }
        };

        clear_seq(kv_csa.get());
        clear_seq(kv_hca.get());
        clear_seq(kv_lid.get());
    }

    csa_state->clear(seq_id, data);
    hca_state->clear(seq_id, data);
    lid_state->clear(seq_id, data);

    if (seq_id >= 0) {
        rs_idx[seq_id] = 0;
    } else {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
    }
}

//
// llama_kv_cache_dsv4_raw_context
//

static llama_kv_cache::slot_info dsv4_build_full_sinfo(const llama_kv_cache * kv) {
    const uint32_t n_stream = kv->get_n_stream();

    llama_kv_cache::slot_info sinfo;
    sinfo.s0 = 0;
    sinfo.s1 = n_stream - 1;
    sinfo.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfo.strm[s] = s;
        sinfo.idxs[s].resize(1, 0);
    }

    return sinfo;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) :
    kv_base(kv->get_base()),
    kv_swa(kv->get_swa()),
    ctx_base_mem(nullptr),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
    sinfos_read.push_back(dsv4_build_full_sinfo(kv_swa));
    sinfos_write = sinfos_read;
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv_base(kv->get_base()),
    kv_swa(kv->get_swa()),
    ctx_base_mem(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa_mem(kv->get_swa()->init_update(lctx, optimize)),
    n_kv(kv_swa->get_size()),
    status(llama_memory_status_combine(ctx_base_mem->get_status(), ctx_swa_mem->get_status())) {
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base_write,
        slot_info_vec_t sinfos_swa_write,
        slot_info_vec_t sinfos_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_write) :
    kv_base(kv->get_base()),
    kv_swa(kv->get_swa()),
    sinfos_base_write(std::move(sinfos_base_write)),
    sinfos_write(std::move(sinfos_swa_write)),
    sinfos_read(std::move(sinfos_swa_read)),
    ubatches(std::move(ubatches)),
    ubatches_write(std::move(ubatches_write)),
    ctx_base_mem(std::make_unique<llama_kv_cache_context>(
                kv_base, this->sinfos_base_write, this->ubatches_write)),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_kv_cache_dsv4_raw_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (ctx_base_mem) {
        ctx_base_mem->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_raw_context::apply() {
    bool res = true;

    if (ctx_base_mem) {
        res = res & ctx_base_mem->apply();
    }
    if (ctx_swa_mem) {
        res = res & ctx_swa_mem->apply();
    }
    if (!ubatches_write.empty()) {
        kv_swa->apply_ubatch(sinfos_write[i_next], ubatches_write[i_next]);
        n_kv = kv_swa->get_n_kv(sinfos_read[i_next]);
    }

    return res;
}

const llama_kv_cache_dsv4_raw_context::slot_info_vec_t::value_type &
llama_kv_cache_dsv4_raw_context::get_base_write_slot() const {
    return get_base_write_slot(i_next);
}

const llama_kv_cache_dsv4_raw_context::slot_info_vec_t::value_type &
llama_kv_cache_dsv4_raw_context::get_swa_write_slot() const {
    return get_swa_write_slot(i_next);
}

const llama_kv_cache_dsv4_raw_context::slot_info_vec_t::value_type &
llama_kv_cache_dsv4_raw_context::get_base_write_slot(size_t index) const {
    return sinfos_base_write.at(index);
}

const llama_kv_cache_dsv4_raw_context::slot_info_vec_t::value_type &
llama_kv_cache_dsv4_raw_context::get_swa_write_slot(size_t index) const {
    return sinfos_write.at(index);
}

llama_memory_status llama_kv_cache_dsv4_raw_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_raw_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_write() const {
    if (ubatches_write.empty()) {
        return 0;
    }

    return ubatches_write[i_next].n_tokens;
}

uint64_t llama_kv_cache_dsv4_raw_context::get_n_backing_rows() const {
    return get_n_backing_rows(i_next);
}

uint64_t llama_kv_cache_dsv4_raw_context::get_n_backing_rows(size_t index) const {
    if (ubatches_write.empty()) {
        return 0;
    }
    uint64_t result = 0;
    for (const auto & idxs : sinfos_write.at(index).idxs) {
        result += idxs.size();
    }
    return result;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv_swa->get_k(ctx, il, n_kv, sinfos_read[i_next]);
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const auto & sinfo = sinfos_write[i_next];

    if (k_cur->ne[2] == k_idxs->ne[0]) {
        return kv_swa->cpy_k(ctx, k_cur, k_idxs, il, sinfo);
    }

    // k_idxs may be expanded to one block per stream while k_cur is only
    // the token block. Keep zero deps on all copies so each write executes.
    const int64_t n_fanout = (int64_t) sinfo.size()*sinfo.n_stream();

    GGML_ASSERT(sinfo.n_stream() > 1);
    GGML_ASSERT(k_cur->ne[2] == (int64_t) sinfo.size());
    GGML_ASSERT(k_idxs->ne[0] == n_fanout);

    ggml_tensor * res = nullptr;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        ggml_tensor * k_idxs_s = ggml_view_1d(ctx, k_idxs, sinfo.size(), s*sinfo.size()*ggml_element_size(k_idxs));
        ggml_tensor * cur = kv_swa->cpy_k(ctx, k_cur, k_idxs_s, il, sinfo);
        if (res == nullptr) {
            res = cur;
        } else {
            res = ggml_add(ctx, res, ggml_sub(ctx, cur, cur));
        }
    }

    return res;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatches_write.empty() ? ubatch.n_tokens : ubatches_write[i_next].n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_rot(ggml_context * ctx) const {
    return kv_swa->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_idxs(ggml_tensor * dst) const {
    kv_swa->set_input_k_idxs(dst, &ubatches_write[i_next], sinfos_write[i_next]);
}

void llama_kv_cache_dsv4_raw_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv_swa->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_rot(ggml_tensor * dst) const {
    kv_swa->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_comp_context
//

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(
        llama_kv_cache * kv,
        uint32_t logical_n_kv) : kv(kv), n_kv(logical_n_kv ? logical_n_kv : kv->get_size()) {
    const uint32_t n_stream = kv->get_n_stream();

    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(
        llama_kv_cache * kv,
        slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches,
        uint32_t logical_n_kv) :
    kv(kv),
    sinfos(std::move(sinfos)),
    ubatches(std::move(ubatches)),
    n_kv(logical_n_kv ? logical_n_kv : kv->get_size()) {
}

bool llama_kv_cache_dsv4_comp_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

uint32_t llama_kv_cache_dsv4_comp_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k_pool(ggml_context * ctx, int32_t il) const {
    ggml_tensor * storage = kv->get_k_storage(il);
    return ggml_view_2d(ctx, storage, storage->ne[0], storage->ne[1], storage->nb[1], 0);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_comp_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_context
//

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv) :
    kv(kv),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw())),
    ctx_csa_mem(kv->get_csa()->init_full()),
    ctx_hca_mem(kv->get_hca()->init_full()),
    ctx_lid_mem(kv->get_lid()->init_full()),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_csa(), kv->get_c4_logical_rows())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_hca(), kv->get_hca_logical_rows())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(kv->get_lid(), kv->get_c4_logical_rows())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    reserve_plans(true),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        llama_context * lctx,
        bool optimize,
        stream_copy_info sc_info_csa,
        stream_copy_info sc_info_hca,
        stream_copy_info sc_info_lid) :
    kv(kv),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw(), lctx, optimize)),
    ctx_csa_mem(kv->get_csa()->init_update(lctx, optimize)),
    ctx_hca_mem(kv->get_hca()->init_update(lctx, optimize)),
    ctx_lid_mem(kv->get_lid()->init_update(lctx, optimize)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    sc_info_csa(std::move(sc_info_csa)),
    sc_info_hca(std::move(sc_info_hca)),
    sc_info_lid(std::move(sc_info_lid)),
    status(llama_memory_status_combine(
                llama_memory_status_combine(
                    llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                    llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status())),
                this->sc_info_csa.empty() && this->sc_info_hca.empty() && this->sc_info_lid.empty() ?
                    LLAMA_MEMORY_STATUS_NO_UPDATE : LLAMA_MEMORY_STATUS_SUCCESS)) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        slot_info_vec_t sinfos_raw_base_write,
        slot_info_vec_t sinfos_raw_swa_write,
        slot_info_vec_t sinfos_raw_swa_read,
        std::vector<llama_ubatch> ubatches,
        std::vector<llama_ubatch> ubatches_raw) :
    kv(kv),
    ubatches(std::move(ubatches)),
    plans_csa(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_csa_state()->get_state_size(), kv->get_c4_logical_rows(), kv->get_csa_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_csa_state()->get_n_rs_seq(), kv->get_rs_idx())),
    plans_hca(dsv4_build_comp_plans(this->ubatches, DSV4_HCA_RATIO, false,
                kv->get_hca_state()->get_state_size(), kv->get_hca_logical_rows(), kv->get_hca_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_hca_state()->get_n_rs_seq(), kv->get_rs_idx())),
    plans_lid(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_lid_state()->get_state_size(), kv->get_c4_logical_rows(), kv->get_lid_state()->get_n_stream(),
                kv->get_n_rs_seq(), kv->get_lid_state()->get_n_rs_seq(), kv->get_rs_idx())),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(
                kv->get_raw(),
                std::move(sinfos_raw_base_write),
                std::move(sinfos_raw_swa_write),
                std::move(sinfos_raw_swa_read),
                this->ubatches,
                std::move(ubatches_raw))),
    ctx_csa_mem(nullptr),
    ctx_hca_mem(nullptr),
    ctx_lid_mem(nullptr),
    ctx_csa(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_csa(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_csa()->get_n_stream(), kv->is_aggregate_compressed()),
                this->ubatches, kv->get_c4_logical_rows())),
    ctx_hca(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_hca(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_hca()->get_n_stream(), kv->is_aggregate_compressed()),
                this->ubatches, kv->get_hca_logical_rows())),
    ctx_lid(std::make_unique<llama_kv_cache_dsv4_comp_context>(
                kv->get_lid(),
                dsv4_build_comp_sinfos(this->ubatches, kv->get_lid()->get_n_stream(), kv->is_aggregate_compressed()),
                this->ubatches, kv->get_c4_logical_rows())),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(ctx_raw->get_status()) {
    kv->reset_rs_idx_for_ubatches(this->ubatches);
}

llama_kv_cache_dsv4_context::~llama_kv_cache_dsv4_context() {
    rollback_aggregate_pool();
}

bool llama_kv_cache_dsv4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_raw->next();
    ctx_csa->next();
    ctx_hca->next();
    ctx_lid->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

static std::vector<uint32_t> dsv4_aggregate_graph_execution_ids(const llama_ubatch & ubatch, int64_t n_stream) {
    std::vector<uint32_t> result;
    result.reserve(n_stream);
    if (n_stream == 1) {
        if (ubatch.n_seqs_unq == 0 || ubatch.seq_id_unq[0] < 0) {
            throw std::runtime_error("DSV4 aggregate graph has no execution sequence");
        }
        result.push_back((uint32_t) ubatch.seq_id_unq[0]);
        return result;
    }
    if ((int64_t) ubatch.n_seqs_unq != n_stream) {
        throw std::runtime_error("DSV4 aggregate graph stream count mismatch");
    }
    for (uint32_t i = 0; i < ubatch.n_seqs_unq; ++i) {
        if (ubatch.seq_id_unq[i] < 0) {
            throw std::runtime_error("DSV4 aggregate graph has invalid execution sequence");
        }
        result.push_back((uint32_t) ubatch.seq_id_unq[i]);
    }
    return result;
}

static void dsv4_copy_pool_segment(
        llama_kv_cache * cache,
        uint32_t source_segment,
        uint32_t destination_segment,
        uint32_t populated_rows) {
    if (populated_rows == 0) {
        return;
    }
    for (uint32_t il : cache->get_layer_ids()) {
        ggml_tensor * tensor = cache->get_k_storage(il);
        const size_t row_size = tensor->nb[1];
        std::vector<uint8_t> bytes((size_t) populated_rows*row_size);
        const size_t source_offset = (size_t) source_segment*LLAMA_DSV4_COMP_SEGMENT_ROWS*row_size;
        const size_t destination_offset = (size_t) destination_segment*LLAMA_DSV4_COMP_SEGMENT_ROWS*row_size;
        ggml_backend_tensor_get(tensor, bytes.data(), source_offset, bytes.size());
        ggml_backend_tensor_set(tensor, bytes.data(), destination_offset, bytes.size());
    }
}

bool llama_kv_cache_dsv4_context::reserve_aggregate_pool() {
    if (!kv->is_aggregate_compressed() || ubatches.empty()) {
        return true;
    }

    llama_dsv4_comp_pool * pool = kv->get_comp_pool();
    GGML_ASSERT(pool != nullptr);

    struct pending_change {
        llama_dsv4_comp_change change;
        std::set<uint64_t> overwrites;
    };
    std::map<std::pair<llama_dsv4_comp_handle_id, llama_dsv4_comp_family>, pending_change> changes;
    std::set<uint32_t> all_execution_ids;

    const auto collect = [&](const llama_ubatch & ubatch, const comp_plan & plan,
                             llama_dsv4_comp_family family, uint32_t logical_rows) {
        for (uint32_t i = 0; i < ubatch.n_seqs_unq; ++i) {
            const uint32_t execution_id = (uint32_t) ubatch.seq_id_unq[i];
            all_execution_ids.insert(execution_id);
            llama_dsv4_comp_handle_id handle = 0;
            if (pool->get_binding(execution_id, handle) != llama_dsv4_comp_status::ok) {
                throw std::runtime_error("DSV4 aggregate execution binding is missing");
            }
            llama_dsv4_comp_handle_info info;
            if (pool->get_handle(handle, info) != llama_dsv4_comp_status::ok) {
                throw std::runtime_error("DSV4 aggregate resident handle is missing");
            }
            const uint64_t old_visible = family == llama_dsv4_comp_family::c4 ?
                    info.visible_c4_rows : info.visible_hca_rows;
            auto & pending = changes[{ handle, family }];
            pending.change.handle = handle;
            pending.change.family = family;
            pending.change.new_visible_rows = std::max(pending.change.new_visible_rows, old_visible);
        }

        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            if (ubatch.pos[i] < 0) {
                continue;
            }
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                const uint32_t execution_id = (uint32_t) ubatch.seq_id[i][s];
                llama_dsv4_comp_handle_id handle = 0;
                GGML_ASSERT(pool->get_binding(execution_id, handle) == llama_dsv4_comp_status::ok);
                const uint64_t ratio = family == llama_dsv4_comp_family::c4 ?
                        LLAMA_DSV4_COMP_C4_TOKENS_PER_ROW : LLAMA_DSV4_COMP_HCA_TOKENS_PER_ROW;
                auto & pending = changes.at({ handle, family });
                pending.change.new_visible_rows = std::max<uint64_t>(
                        pending.change.new_visible_rows, ((uint64_t) ubatch.pos[i] + 1)/ratio);
            }
        }

        GGML_ASSERT(plan.state_write_logical_idxs.size() == plan.state_write_dummy.size());
        for (size_t i = 0; i < plan.state_write_logical_idxs.size(); ++i) {
            if (plan.state_write_dummy[i]) {
                continue;
            }
            const uint64_t flat = (uint64_t) plan.state_write_logical_idxs[i];
            const uint32_t execution_id = (uint32_t) (flat/logical_rows);
            const uint64_t logical_row = flat%logical_rows;
            llama_dsv4_comp_handle_id handle = 0;
            GGML_ASSERT(pool->get_binding(execution_id, handle) == llama_dsv4_comp_status::ok);
            llama_dsv4_comp_handle_info info;
            GGML_ASSERT(pool->get_handle(handle, info) == llama_dsv4_comp_status::ok);
            const uint64_t old_visible = family == llama_dsv4_comp_family::c4 ?
                    info.visible_c4_rows : info.visible_hca_rows;
            if (logical_row < old_visible) {
                changes.at({ handle, family }).overwrites.insert(logical_row);
            }
        }
    };

    for (size_t i = 0; i < ubatches.size(); ++i) {
        collect(ubatches[i], plans_csa[i], llama_dsv4_comp_family::c4, kv->get_c4_logical_rows());
        collect(ubatches[i], plans_hca[i], llama_dsv4_comp_family::hca, kv->get_hca_logical_rows());
    }

    llama_dsv4_comp_batch_plan batch;
    batch.graph_execution_ids.assign(all_execution_ids.begin(), all_execution_ids.end());
    for (auto & [_, pending] : changes) {
        pending.change.overwrite_rows.assign(pending.overwrites.begin(), pending.overwrites.end());
        batch.changes.push_back(std::move(pending.change));
    }

    const llama_dsv4_comp_quote quote = pool->quote_batch(batch);
    if (quote.status == llama_dsv4_comp_status::capacity_exhausted) {
        status = LLAMA_MEMORY_STATUS_KV_PHYSICAL_PRESSURE;
        last_pressure_family_mask = quote.limiting_family == llama_dsv4_comp_family::c4 ?
                (1u << LLAMA_DSV4_MEMORY_CSA) | (1u << LLAMA_DSV4_MEMORY_LID) :
                (1u << LLAMA_DSV4_MEMORY_HCA);
        return false;
    }
    if (quote.status != llama_dsv4_comp_status::ok) {
        throw std::runtime_error(std::string("DSV4 aggregate quote failed: ") +
                llama_dsv4_comp_status_name(quote.status));
    }
    const llama_dsv4_comp_reserve_result reserved = pool->try_reserve(quote);
    if (reserved.status != llama_dsv4_comp_status::ok) {
        if (reserved.status == llama_dsv4_comp_status::capacity_exhausted) {
            status = LLAMA_MEMORY_STATUS_KV_PHYSICAL_PRESSURE;
            return false;
        }
        throw std::runtime_error(std::string("DSV4 aggregate reservation failed: ") +
                llama_dsv4_comp_status_name(reserved.status));
    }
    comp_ticket = reserved.ticket;
    comp_ticket_active = true;

    for (const llama_dsv4_comp_allocation & allocation : quote.allocations) {
        if (!allocation.cow) {
            continue;
        }
        if (allocation.family == llama_dsv4_comp_family::c4) {
            dsv4_copy_pool_segment(kv->get_csa(), allocation.source_segment,
                    allocation.destination_segment, allocation.populated_rows);
            dsv4_copy_pool_segment(kv->get_lid(), allocation.source_segment,
                    allocation.destination_segment, allocation.populated_rows);
        } else {
            dsv4_copy_pool_segment(kv->get_hca(), allocation.source_segment,
                    allocation.destination_segment, allocation.populated_rows);
        }
    }

    const auto resolve = [&](comp_plan & plan, const llama_ubatch & ubatch,
                             llama_dsv4_comp_family family, uint32_t logical_rows) {
        const std::vector<uint32_t> execution_ids = dsv4_aggregate_graph_execution_ids(ubatch, plan.n_stream);
        const uint32_t logical_segments = (uint32_t) ((plan.n_kv + LLAMA_DSV4_COMP_SEGMENT_ROWS - 1)/
                LLAMA_DSV4_COMP_SEGMENT_ROWS);
        const llama_dsv4_comp_directory directory = pool->ticket_directory_for(
                comp_ticket, family, execution_ids, logical_segments);
        if (directory.status != llama_dsv4_comp_status::ok) {
            throw std::runtime_error("failed to build DSV4 aggregate graph directory");
        }
        plan.segment_ids.assign(directory.segment_ids.begin(), directory.segment_ids.end());

        for (size_t i = 0; i < plan.state_write_idxs.size(); ++i) {
            const uint64_t flat = (uint64_t) plan.state_write_logical_idxs[i];
            const uint32_t execution_id = (uint32_t) (flat/logical_rows);
            if (plan.state_write_dummy[i]) {
                const auto it = std::find(execution_ids.begin(), execution_ids.end(), execution_id);
                if (it == execution_ids.end()) {
                    throw std::runtime_error("DSV4 dummy write execution stream is absent");
                }
                plan.state_write_idxs[i] = (int64_t) pool->scratch_physical_row(
                        family, (uint32_t) std::distance(execution_ids.begin(), it));
                continue;
            }
            const uint64_t logical_row = flat%logical_rows;
            llama_dsv4_comp_handle_id handle = 0;
            GGML_ASSERT(pool->get_binding(execution_id, handle) == llama_dsv4_comp_status::ok);
            llama_dsv4_comp_handle_info candidate;
            GGML_ASSERT(pool->candidate_handle(comp_ticket, handle, candidate) == llama_dsv4_comp_status::ok);
            const auto & ids = family == llama_dsv4_comp_family::c4 ?
                    candidate.c4_segment_ids : candidate.hca_segment_ids;
            const uint64_t logical_segment = llama_dsv4_comp_logical_segment(logical_row);
            if (logical_segment >= ids.size()) {
                throw std::runtime_error("DSV4 aggregate write has no reserved segment");
            }
            plan.state_write_idxs[i] = (int64_t) llama_dsv4_comp_physical_row(
                    ids[logical_segment], logical_row);
        }
    };

    for (size_t i = 0; i < ubatches.size(); ++i) {
        resolve(plans_csa[i], ubatches[i], llama_dsv4_comp_family::c4, kv->get_c4_logical_rows());
        resolve(plans_lid[i], ubatches[i], llama_dsv4_comp_family::c4, kv->get_c4_logical_rows());
        resolve(plans_hca[i], ubatches[i], llama_dsv4_comp_family::hca, kv->get_hca_logical_rows());
    }
    return true;
}

void llama_kv_cache_dsv4_context::rollback_aggregate_pool() {
    if (!comp_ticket_active) {
        return;
    }
    llama_dsv4_comp_pool * pool = kv->get_comp_pool();
    if (pool->rollback(comp_ticket) != llama_dsv4_comp_status::ok) {
        LLAMA_LOG_ERROR("%s: failed to rollback DSV4 aggregate compressed-pool ticket\n", __func__);
    }
    comp_ticket_active = false;
}

bool llama_kv_cache_dsv4_context::reserve_batch_ranges() {
    if (!reserve_aggregate_pool()) {
        return false;
    }

    std::vector<dsv4_sparse_range> ranges;
    for (size_t i = 0; i < ubatches.size(); ++i) {
        if (!dsv4_sparse_append_slot(
                    kv->get_raw()->get_base(), ctx_raw->get_base_write_slot(i),
                    LLAMA_DSV4_MEMORY_RAW, ranges) ||
                !dsv4_sparse_append_slot(
                    kv->get_raw()->get_swa(), ctx_raw->get_swa_write_slot(i),
                    LLAMA_DSV4_MEMORY_RAW, ranges) ||
                !dsv4_sparse_append_k_rows(
                    kv->get_csa(), plans_csa.at(i).state_write_idxs,
                    LLAMA_DSV4_MEMORY_CSA, ranges) ||
                !dsv4_sparse_append_k_rows(
                    kv->get_hca(), plans_hca.at(i).state_write_idxs,
                    LLAMA_DSV4_MEMORY_HCA, ranges) ||
                !dsv4_sparse_append_k_rows(
                    kv->get_lid(), plans_lid.at(i).state_write_idxs,
                    LLAMA_DSV4_MEMORY_LID, ranges)) {
            rollback_aggregate_pool();
            return false;
        }

        last_batch_quote.families[LLAMA_DSV4_MEMORY_RAW].logical_rows +=
                ctx_raw->get_n_backing_rows(i);
    }

    dsv4_sparse_transaction reservation;
    const auto reserve_status = reservation.reserve_ranges(ranges, last_batch_quote);
    if (reserve_status == dsv4_sparse_transaction::PRESSURE) {
        status = LLAMA_MEMORY_STATUS_KV_PHYSICAL_PRESSURE;
        last_pressure_family_mask = reservation.limiting_family_mask();
        const auto & limiting = last_batch_quote.families[last_batch_quote.limiting_family];
        LLAMA_LOG_WARN("%s: DSV4 elastic Metal page pressure before batch submission: family_mask=0x%x pool=%" PRIuPTR
                " need=%" PRIu64 " (new=%" PRIu64 ", COW=%" PRIu64
                ") free=%" PRIu64 " reserved=%" PRIu64 " capacity=%" PRIu64
                " pages ubatches=%zu sparse_ranges=%zu shared_family_pools=%zu\n",
                __func__, reservation.limiting_family_mask(),
                last_batch_quote.limiting_pool_id,
                limiting.required_pages, limiting.new_pages, limiting.cow_pages,
                limiting.free_pages, limiting.reserved_pages, limiting.physical_pages,
                ubatches.size(), reservation.sparse_range_count(), reservation.shared_family_pool_count());
        rollback_aggregate_pool();
        return false;
    }
    if (reserve_status != dsv4_sparse_transaction::SUCCESS) {
        reservation.log_failure(__func__);
        rollback_aggregate_pool();
        return false;
    }
    if (reservation.commit_ranges() != GGML_METAL_SPARSE_RESERVATION_OK) {
        reservation.log_failure(__func__);
        rollback_aggregate_pool();
        return false;
    }

    batch_ranges_reserved = true;
    return true;
}

void llama_kv_cache_dsv4_context::finish(bool success) {
    if (!comp_ticket_active) {
        return;
    }
    if (!success) {
        rollback_aggregate_pool();
        return;
    }
    ++comp_finished_ubatches;
    if (comp_finished_ubatches < ubatches.size()) {
        return;
    }
    const llama_dsv4_comp_status commit_status = kv->get_comp_pool()->commit(comp_ticket);
    if (commit_status != llama_dsv4_comp_status::ok) {
        comp_ticket_active = false;
        throw std::runtime_error(std::string("failed to commit DSV4 aggregate compressed-pool ticket: ") +
                llama_dsv4_comp_status_name(commit_status));
    }
    comp_ticket_active = false;
}

bool llama_kv_cache_dsv4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    if (!preflight()) {
        return false;
    }

    bool res = true;

    res = res & ctx_raw->apply();

    if (ctx_csa_mem) {
        res = res & ctx_csa_mem->apply();
        res = res & ctx_hca_mem->apply();
        res = res & ctx_lid_mem->apply();
    }

    if (ubatches.empty()) {
        csa_state->apply_copies(sc_info_csa);
        hca_state->apply_copies(sc_info_hca);
        lid_state->apply_copies(sc_info_lid);
    }

    return res;
}

bool llama_kv_cache_dsv4_context::preflight() {
    assert(!llama_memory_status_is_fail(status));

    if (ubatches.empty() || batch_ranges_reserved) {
        return true;
    }

    last_batch_quote = {};
    return reserve_batch_ranges();
}

llama_memory_status llama_kv_cache_dsv4_context::get_status() const {
    return status;
}

bool llama_kv_cache_dsv4_context::get_kv_pressure(llama_kv_pressure_info & info) const {
    if (status != LLAMA_MEMORY_STATUS_KV_PHYSICAL_PRESSURE) {
        return false;
    }

    const auto & limiting = last_batch_quote.families[last_batch_quote.limiting_family];
    info = {
        (uint32_t) last_batch_quote.limiting_family,
        last_pressure_family_mask,
        (uint64_t) last_batch_quote.limiting_pool_id,
        limiting.required_pages,
        limiting.new_pages,
        limiting.cow_pages,
        limiting.free_pages,
        limiting.reserved_pages,
        limiting.physical_pages,
    };
    return true;
}

const llama_ubatch & llama_kv_cache_dsv4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsv4_raw_context * llama_kv_cache_dsv4_context::get_raw() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_raw.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_csa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_csa.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_hca() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_hca.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_lid() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_lid.get();
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_csa_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return csa_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_hca_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return hca_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_lid_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return lid_state;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_csa.empty()) {
        return empty;
    }

    return plans_csa[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_hca.empty()) {
        return empty;
    }

    return plans_hca[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_lid.empty()) {
        return empty;
    }

    return plans_lid[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_csa_plan();
    }

    reserve_plan_csa = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            csa_state->get_state_size(), get_csa()->get_n_kv(), csa_state->get_n_stream(), csa_state->get_n_rs_seq());

    if (kv->is_aggregate_compressed()) {
        reserve_plan_csa.segment_ids.resize(
                (size_t) ((reserve_plan_csa.n_kv + 63)/64)*reserve_plan_csa.n_stream);
    }

    return reserve_plan_csa;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_hca_plan();
    }

    reserve_plan_hca = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_HCA_RATIO, false,
            hca_state->get_state_size(), get_hca()->get_n_kv(), hca_state->get_n_stream(), hca_state->get_n_rs_seq());

    if (kv->is_aggregate_compressed()) {
        reserve_plan_hca.segment_ids.resize(
                (size_t) ((reserve_plan_hca.n_kv + 63)/64)*reserve_plan_hca.n_stream);
    }

    return reserve_plan_hca;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_lid_plan();
    }

    reserve_plan_lid = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            lid_state->get_state_size(), get_lid()->get_n_kv(), lid_state->get_n_stream(), lid_state->get_n_rs_seq());

    if (kv->is_aggregate_compressed()) {
        reserve_plan_lid.segment_ids.resize(
                (size_t) ((reserve_plan_lid.n_kv + 63)/64)*reserve_plan_lid.n_stream);
    }

    return reserve_plan_lid;
}

const llama_dsv4_batch_quote & llama_kv_cache_dsv4_context::get_last_batch_quote() const {
    return last_batch_quote;
}

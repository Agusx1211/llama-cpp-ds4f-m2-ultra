#include "llama-kv-cache-dsv4.h"

#include "../ggml/src/ggml-metal/ggml-metal-device.h"
#include "ggml-backend.h"
#include "llama-batch.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-kv-cache-dsv4-accounting.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
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

// Host-overhead profiler (LLAMA_HOST_PROFILE=1). Sub-phase attribution of the
// per-decode DSV4 sparse page reservation, which llama_context reports as the
// "pre" phase. Disabled cost is one predicted branch on a function-local
// static; see src/llama-context.cpp for the record format.
static bool dsv4_hp_enabled() {
    static const bool enabled = []() {
        const char * v = getenv("LLAMA_HOST_PROFILE");
        return v != nullptr && atoi(v) != 0;
    }();
    return enabled;
}

static std::atomic<uint32_t> dsv4_test_pressure_count = 0;
static std::atomic<uint64_t> dsv4_test_cow_source_ranges      = 0;
static std::atomic<uint64_t> dsv4_test_cow_destination_ranges = 0;
static std::atomic<uint64_t> dsv4_test_cow_copy_operations    = 0;
static std::atomic<bool>                       dsv4_test_page_delta_audit_enabled = false;
static std::mutex                              dsv4_test_page_delta_audit_mutex;
static llama_dsv4_sparse_page_delta_test_audit dsv4_test_page_delta_audit;
static thread_local int                        dsv4_test_comp_state_copy_after = -1;

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

void llama_kv_cache_dsv4_test_reset_cow_preflight_stats() {
    dsv4_test_cow_source_ranges.store(0, std::memory_order_release);
    dsv4_test_cow_destination_ranges.store(0, std::memory_order_release);
    dsv4_test_cow_copy_operations.store(0, std::memory_order_release);
}

llama_dsv4_cow_preflight_test_stats llama_kv_cache_dsv4_test_get_cow_preflight_stats() {
    return {
        dsv4_test_cow_source_ranges.load(std::memory_order_acquire),
        dsv4_test_cow_destination_ranges.load(std::memory_order_acquire),
        dsv4_test_cow_copy_operations.load(std::memory_order_acquire),
    };
}

void llama_kv_cache_dsv4_test_enable_page_delta_audit(bool enabled) {
    dsv4_test_page_delta_audit_enabled.store(enabled, std::memory_order_relaxed);
}

void llama_kv_cache_dsv4_test_reset_page_delta_audit() {
    std::lock_guard<std::mutex> lock(dsv4_test_page_delta_audit_mutex);
    dsv4_test_page_delta_audit = {};
}

llama_dsv4_sparse_page_delta_test_audit llama_kv_cache_dsv4_test_get_page_delta_audit() {
    std::lock_guard<std::mutex> lock(dsv4_test_page_delta_audit_mutex);
    return dsv4_test_page_delta_audit;
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

static uint64_t dsv4_affine_compressed_bytes(
        const llama_model & model,
                  ggml_type type_k,
                   uint32_t n_seq_max,
                   uint32_t c4_rows,
                   uint32_t hca_rows,
    const llama_memory_i::layer_filter_cb & filter,
                       bool & overflow) {
    uint64_t total = 0;
    overflow = false;

    auto add = [&](uint32_t width, uint32_t rows) {
        if (!dsv4_affine_compressed_bytes_add(
                    total, ggml_row_size(type_k, width), rows, n_seq_max)) {
            overflow = true;
        }
    };

    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        const uint32_t ratio = model.hparams.dsv4_compress_ratios[il];
        if (ratio == DSV4_CSA_RATIO) {
            add(model.hparams.n_embd_k_gqa(il), c4_rows);
            add(model.hparams.indexer_head_size, c4_rows);
        } else if (ratio == DSV4_HCA_RATIO) {
            add(model.hparams.n_embd_k_gqa(il), hca_rows);
        }
    }

    return total;
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

// env: LLAMA_DSV4_DEBUG_ZERO_CLEARED_SEQ=1 - diagnostic seam for the
// deep-context determinism investigation. A cleared DSV4 sequence normally
// *unmaps* its sparse pages, so the next request that writes those rows is
// handed recycled physical pages carrying the previous tenant's bytes, while
// the very first request in a process gets pages the driver zero-fills. Any
// read of a row that the current request has not written therefore produces
// different results on a cold and a warm slot. With this knob the clear
// memsets instead of unmapping and additionally zeroes the raw base/SWA K
// streams, i.e. a warm slot is made byte-identical to a cold one.
static bool dsv4_debug_zero_cleared_seq() {
    static const bool en = []() {
        const char * v = getenv("LLAMA_DSV4_DEBUG_ZERO_CLEARED_SEQ");
        return v != nullptr && atoi(v) != 0;
    }();
    return en;
}

static void dsv4_clear_tensor_stream(ggml_tensor * tensor, uint32_t stream) {
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[3] == 1);
    GGML_ASSERT(stream < (uint32_t) tensor->ne[2]);

    const size_t stream_size = tensor->nb[2];
    const bool page_aligned = !dsv4_debug_zero_cleared_seq() &&
            stream_size % (64*1024) == 0 &&
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

static llama_dsv4_sparse_pool_usage dsv4_convert_sparse_usage(const ggml_metal_sparse_usage & src) {
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

static llama_dsv4_sparse_page_quote_test_audit dsv4_convert_sparse_quote(const ggml_metal_sparse_quote & src) {
    return {
        src.generation,     src.target_mappings, src.new_pages,      src.cow_pages,
        src.required_pages, src.free_pages,      src.reserved_pages, src.feasible,
    };
}

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

static bool dsv4_sparse_append_k_row_run(
        const llama_kv_cache * kv,
        int64_t row_begin,
        int64_t row_end,
        llama_dsv4_memory_family family,
        std::vector<dsv4_sparse_range> & ranges) {
    if (row_begin == row_end) {
        return true;
    }
    const int64_t n_rows_total = (int64_t) kv->get_size()*kv->get_n_stream();
    if (row_begin < 0 || row_end <= row_begin || row_end > n_rows_total) {
        return false;
    }
    ranges.reserve(ranges.size() + kv->get_layer_ids().size());
    for (uint32_t il : kv->get_layer_ids()) {
        ggml_tensor * tensor = kv->get_k_storage(il);
        ranges.push_back({
            tensor,
            (size_t) row_begin*tensor->nb[1],
            (size_t) (row_end - row_begin)*tensor->nb[1],
            family,
        });
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

// Steady-state decode writes into pages that are already mapped and privately
// owned, so the sparse reservation it performs is a no-op that still costs
// O(virtual pages in the pool) to plan. This probe answers "no reservation
// needed" in O(pages in the ranges); see
// ggml_metal_buffers_sparse_ranges_resident for why skipping is exact.
// Returns 1 (skip), 0 (reserve) or -1 (probe unavailable / invalid).
static int dsv4_sparse_ranges_resident(const std::vector<dsv4_sparse_range> & ranges) {
    static const bool disabled = []() {
        const char * v = getenv("LLAMA_DSV4_SPARSE_RESIDENT_DISABLE");
        return v != nullptr && atoi(v) != 0;
    }();
    if (disabled || ranges.empty()) {
        return -1;
    }

    using resident_fn = int (*)(ggml_tensor * const *, const size_t *, const size_t *, size_t);

    resident_fn resident = nullptr;
    for (const auto & range : ranges) {
        resident = (resident_fn) dsv4_backend_proc(
                range.tensor, "ggml_backend_metal_dsv4_sparse_ranges_resident");
        if (resident != nullptr) {
            break;
        }
    }
    if (resident == nullptr) {
        return -1;
    }

    std::vector<ggml_tensor *> tensors;
    std::vector<size_t>        offsets;
    std::vector<size_t>        sizes;
    tensors.reserve(ranges.size());
    offsets.reserve(ranges.size());
    sizes.reserve(ranges.size());
    for (const auto & range : ranges) {
        tensors.push_back(range.tensor);
        offsets.push_back(range.offset);
        sizes.push_back(range.size);
    }

    return resident(tensors.data(), offsets.data(), sizes.data(), tensors.size());
}

using dsv4_sparse_quote_fn   = ggml_metal_sparse_reservation_result (*)(ggml_tensor * const *,
                                                                      const size_t *,
                                                                      const size_t *,
                                                                      size_t,
                                                                      ggml_metal_sparse_pool_quote *,
                                                                      size_t,
                                                                      size_t *,
                                                                      size_t *);
using dsv4_sparse_reserve_fn = ggml_metal_sparse_reservation_result (*)(ggml_tensor * const *,
                                                                        const size_t *,
                                                                        const size_t *,
                                                                        size_t,
                                                                        ggml_metal_sparse_pool_quote *,
                                                                        size_t,
                                                                        size_t *,
                                                                        size_t *,
                                                                        void **);
using dsv4_sparse_usage_fn   = int (*)(const ggml_tensor *, ggml_metal_sparse_usage *);

struct dsv4_sparse_transaction_audit_state {
    llama_dsv4_sparse_page_delta_test_audit                      result;
    std::map<uintptr_t, llama_dsv4_sparse_pool_delta_test_audit> range_metadata;
    std::map<uintptr_t, ggml_tensor *>                           pool_tensors;
    bool                                                         finalized = false;
};

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
        bool cancelled = false;
        if (ticket != nullptr) {
            cancelled = cancel(ticket);
            free_ticket(ticket);
        }
        if (audit_state != nullptr && !audit_state->finalized) {
            finish_audit(false, cancelled);
        }
    }

    prepare_status quote_ranges(
            const std::vector<dsv4_sparse_range> & ranges,
            llama_dsv4_batch_quote & result) {
        return prepare_ranges(ranges, result, false);
    }

    prepare_status reserve_ranges(
            const std::vector<dsv4_sparse_range> & ranges,
            llama_dsv4_batch_quote & result) {
        return prepare_ranges(ranges, result, true);
    }

private:
    prepare_status prepare_ranges(
            const std::vector<dsv4_sparse_range> & ranges,
            llama_dsv4_batch_quote & result,
            bool reserve_ticket) {
        const bool hp = dsv4_hp_enabled();
        int64_t    hp_t = hp ? ggml_time_us() : 0;
        const auto hp_mark = [hp, &hp_t](int64_t & acc) {
            if (hp) {
                const int64_t t = ggml_time_us();
                acc += t - hp_t;
                hp_t = t;
            }
        };

        n_ranges = ranges.size();
        if (reserve_ticket && dsv4_test_page_delta_audit_enabled.load(std::memory_order_relaxed)) {
            audit_state = std::make_unique<dsv4_sparse_transaction_audit_state>();
        }
        for (size_t i = 0; i < result.families.size(); ++i) {
            result.families[i].family = (llama_dsv4_memory_family) i;
        }
        if (ranges.empty()) {
            return SUCCESS;
        }

        using commit_fn = ggml_metal_sparse_reservation_result (*)(void *);
        using cancel_fn = bool (*)(void *);
        using free_fn   = void (*)(void *);

        dsv4_sparse_quote_fn quote = nullptr;
        ggml_tensor * proc_tensor = nullptr;
        for (const auto & range : ranges) {
            quote = (dsv4_sparse_quote_fn) dsv4_backend_proc(
                range.tensor, "ggml_backend_metal_dsv4_sparse_quote_tensor_ranges");
            if (quote != nullptr) {
                proc_tensor = range.tensor;
                break;
            }
        }
        if (quote == nullptr) {
            return SUCCESS;
        }

        dsv4_sparse_reserve_fn reserve = nullptr;
        if (reserve_ticket) {
            reserve = (dsv4_sparse_reserve_fn) dsv4_backend_proc(
                    proc_tensor, "ggml_backend_metal_dsv4_sparse_reserve_tensor_ranges");
            commit = (commit_fn) dsv4_backend_proc(
                    proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_commit");
            cancel = (cancel_fn) dsv4_backend_proc(
                    proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_cancel");
            free_ticket = (free_fn) dsv4_backend_proc(
                    proc_tensor, "ggml_backend_metal_dsv4_sparse_reservation_free");
        }
        usage = (dsv4_sparse_usage_fn) dsv4_backend_proc(proc_tensor, "ggml_backend_metal_dsv4_sparse_tensor_usage");
        usage_proc_available = usage != nullptr;
        if (usage == nullptr || (reserve_ticket &&
                (reserve == nullptr || commit == nullptr || cancel == nullptr || free_ticket == nullptr))) {
            phase = PHASE_PROC_LOOKUP;
            return ERROR;
        }

        hp_mark(hp_proc);

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
            if (audit_state != nullptr) {
                ++audit_state->result.family_range_count[range.family];
                audit_state->result.family_range_bytes[range.family] += range.size;
            }
            ggml_metal_sparse_usage snapshot = {};
            const int snapshot_result = usage(range.tensor, &snapshot);
            if (snapshot_result < 0) {
                phase = PHASE_USAGE;
                return ERROR;
            }
            if (snapshot_result > 0) {
                ++n_sparse_ranges;
                pool_family_masks[snapshot.pool_id] |= 1u << range.family;
                if (audit_state != nullptr) {
                    audit_state->pool_tensors.try_emplace(snapshot.pool_id, range.tensor);
                    auto & pool = audit_state->range_metadata[snapshot.pool_id];
                    ++pool.family_range_count[range.family];
                    pool.family_range_bytes[range.family] += range.size;
                    pool.family_zero_offset_ranges[range.family] += range.offset == 0;
                }
            }
        }

        // The Metal registry exposes the sparse transaction procedures for
        // ordinary buffers too. No sparse ranges is therefore a successful
        // no-op, not an unsupported reservation failure.
        if (n_sparse_ranges == 0) {
            return SUCCESS;
        }

        if (audit_state != nullptr) {
            std::vector<ggml_metal_sparse_pool_quote> dry_pools(ranges.size());
            size_t                                    dry_n_pools       = 0;
            size_t                                    dry_limiting_pool = SIZE_MAX;
            const auto dry_status = quote(tensors.data(), offsets.data(), sizes.data(), tensors.size(),
                                          dry_pools.data(), dry_pools.size(), &dry_n_pools, &dry_limiting_pool);
            GGML_UNUSED(dry_limiting_pool);
            dry_pools.resize(dry_n_pools);
            audit_state->result.observed = true;
            audit_state->result.dry_quoted =
                dry_status == GGML_METAL_SPARSE_RESERVATION_OK || dry_status == GGML_METAL_SPARSE_RESERVATION_PRESSURE;
            for (const auto & pool : dry_pools) {
                const auto found    = pool_family_masks.find(pool.pool_id);
                const auto metadata = audit_state->range_metadata.find(pool.pool_id);
                if (found == pool_family_masks.end() || found->second == 0 ||
                    metadata == audit_state->range_metadata.end()) {
                    phase           = PHASE_POOL_ACCOUNTING;
                    failure_pool_id = pool.pool_id;
                    return ERROR;
                }
                auto entry        = metadata->second;
                entry.pool_id     = pool.pool_id;
                entry.family_mask = found->second;
                entry.dry_quote   = dsv4_convert_sparse_quote(pool.write);
                entry.before      = dsv4_convert_sparse_usage(pool.usage);
                audit_state->result.pools.push_back(std::move(entry));
            }
        }

        hp_mark(hp_usage);

        std::vector<ggml_metal_sparse_pool_quote> pools(ranges.size());
        size_t n_pools = 0;
        size_t limiting_pool = SIZE_MAX;

        hp_mark(hp_pools);

        const auto status = reserve_ticket ? reserve(
                tensors.data(), offsets.data(), sizes.data(), tensors.size(),
                pools.data(), pools.size(), &n_pools, &limiting_pool, &ticket) : quote(
                tensors.data(), offsets.data(), sizes.data(), tensors.size(),
                pools.data(), pools.size(), &n_pools, &limiting_pool);
        hp_mark(hp_call);

        backend_status = status;
        ticket_issued = ticket != nullptr;
        n_quoted_pools = n_pools;
        pools.resize(n_pools);
        if (audit_state != nullptr) {
            audit_state->result.reserved = status == GGML_METAL_SPARSE_RESERVATION_OK && ticket != nullptr;
        }

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
            if (audit_state != nullptr) {
                const auto audit_pool = std::find_if(
                    audit_state->result.pools.begin(), audit_state->result.pools.end(),
                    [&](const auto & entry) { return entry.pool_id == pool.pool_id; });
                if (audit_pool == audit_state->result.pools.end()) {
                    phase           = PHASE_POOL_ACCOUNTING;
                    failure_pool_id = pool.pool_id;
                    return ERROR;
                }
                audit_pool->reserved_quote = dsv4_convert_sparse_quote(pool.write);
                audit_pool->before         = dsv4_convert_sparse_usage(pool.usage);
            }
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

        hp_mark(hp_acct);

        // Consume the test fault only after the real backend has produced a
        // valid quote and ticket. Returning pressure here exercises the same
        // cancellation and propagation path as capacity exhaustion, without
        // committing a mapping or submitting a graph.
        if (reserve_ticket && status == GGML_METAL_SPARSE_RESERVATION_OK && ticket != nullptr &&
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
        if (status != GGML_METAL_SPARSE_RESERVATION_OK || (reserve_ticket && ticket == nullptr)) {
            phase = PHASE_RESERVE;
            return ERROR;
        }
        return SUCCESS;
    }

public:

    ggml_metal_sparse_reservation_result commit_ranges() {
        if (ticket == nullptr) {
            return GGML_METAL_SPARSE_RESERVATION_OK;
        }
        const auto status = commit(ticket);
        backend_status = status;
        if (status != GGML_METAL_SPARSE_RESERVATION_OK) {
            phase = PHASE_COMMIT;
        }
        if (audit_state != nullptr) {
            finish_audit(status == GGML_METAL_SPARSE_RESERVATION_OK, false);
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

    size_t quoted_pool_count() const {
        return n_quoted_pools;
    }

    // host-overhead profiler accumulators (LLAMA_HOST_PROFILE=1), microseconds
    int64_t hp_proc  = 0; // backend proc-address lookups
    int64_t hp_usage = 0; // per-range tensor -> pool usage snapshots
    int64_t hp_pools = 0; // per-call quote/usage output vector allocation
    int64_t hp_call  = 0; // the Metal sparse quote/reserve call itself
    int64_t hp_acct  = 0; // per-pool family accounting

private:
    void finish_audit(bool committed_value, bool cancelled_value) {
        GGML_ASSERT(audit_state != nullptr);
        auto & audit     = audit_state->result;
        audit.committed = committed_value;
        audit.cancelled = cancelled_value;
        for (auto & pool : audit.pools) {
            const auto found = audit_state->pool_tensors.find(pool.pool_id);
            if (found == audit_state->pool_tensors.end() || usage == nullptr) {
                continue;
            }
            ggml_metal_sparse_usage snapshot = {};
            if (usage(found->second, &snapshot) > 0) {
                pool.after = dsv4_convert_sparse_usage(snapshot);
            }
        }
        {
            std::lock_guard<std::mutex> lock(dsv4_test_page_delta_audit_mutex);
            dsv4_test_page_delta_audit = audit;
        }
        audit_state->finalized = true;
    }

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
    dsv4_sparse_usage_fn usage = nullptr;
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
    std::unique_ptr<dsv4_sparse_transaction_audit_state> audit_state;
};

struct llama_kv_cache_dsv4_admission_state {
    struct entry {
        bool armed = false;
        std::vector<llama_kv_admission_span> spans;
        std::unique_ptr<dsv4_sparse_transaction> reservation;
        llama_dsv4_batch_quote quote;
        uint32_t limiting_family_mask = 0;
    };

    uint64_t next_id = 1;
    std::map<uint64_t, entry> entries;
};

static void dsv4_public_admission_quote(
        const llama_dsv4_batch_quote & source,
        llama_kv_admission_status status,
        uint32_t limiting_family_mask,
        llama_kv_admission_quote & result) {
    result = {};
    result.status = status;
    result.limiting_family = source.limiting_family;
    result.limiting_family_mask = limiting_family_mask;
    result.limiting_pool_id = source.limiting_pool_id;
    for (const auto & family : source.families) {
        result.target_mappings += family.target_mappings;
        result.required_pages += family.required_pages;
        result.new_pages += family.new_pages;
        result.cow_pages += family.cow_pages;
        result.free_pages += family.free_pages;
        result.reserved_pages += family.reserved_pages;
        result.physical_pages += family.physical_pages;
    }
}

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

    // A pending rollback restores the current plane exactly once, in the
    // first ubatch that touches its sequence. Plans are built up front, so use
    // a planning-local copy rather than replaying the original rs_idx in every
    // later ubatch.
    std::vector<uint32_t> pending(rs_idx);
    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(
                ubatch, ratio, overlap, state_size, kv_size, n_stream, n_rs_seq, n_rs_seq_alloc, pending));

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            if (seq_id >= 0 && (size_t) seq_id < pending.size()) {
                pending[seq_id] = 0;
            }
        }
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
    bump_generation();

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
    if (!sc_info.ssrc.empty()) {
        bump_generation();
    }
    for (size_t i = 0; i < sc_info.ssrc.size(); ++i) {
        const uint32_t ssrc = sc_info.ssrc[i];
        const uint32_t sdst = sc_info.sdst[i];

        for (const auto & layer : layers) {
            ggml_backend_tensor_copy(layer.kv_stream[ssrc], layer.kv_stream[sdst]);
            ggml_backend_tensor_copy(layer.score_stream[ssrc], layer.score_stream[sdst]);
        }
    }
}

void llama_dsv4_comp_state_fail_copy_after_for_test(int successful_copies) {
    dsv4_test_comp_state_copy_after = successful_copies;
}

void llama_dsv4_comp_state::copy_sequence_all_depths_to(
        llama_dsv4_comp_state & destination,
        llama_seq_id            source_sequence,
        llama_seq_id            destination_sequence) const {
    if (source_sequence < 0 || (uint32_t) source_sequence >= n_stream ||
        destination_sequence < 0 || (uint32_t) destination_sequence >= destination.n_stream) {
        throw std::invalid_argument("DSV4 resident state sequence out of range");
    }
    if (this == &destination || ratio != destination.ratio || state_size != destination.state_size ||
        n_embd_state != destination.n_embd_state || n_rs_seq != destination.n_rs_seq ||
        layers.size() != destination.layers.size()) {
        throw std::invalid_argument("DSV4 resident state geometry mismatch");
    }
    for (size_t i = 0; i < layers.size(); ++i) {
        if (layers[i].il != destination.layers[i].il || layers[i].kv->type != destination.layers[i].kv->type ||
            layers[i].score->type != destination.layers[i].score->type ||
            layers[i].kv->ne[0] != destination.layers[i].kv->ne[0] ||
            layers[i].kv->ne[1] != destination.layers[i].kv->ne[1] ||
            layers[i].score->ne[0] != destination.layers[i].score->ne[0] ||
            layers[i].score->ne[1] != destination.layers[i].score->ne[1]) {
            throw std::invalid_argument("DSV4 resident state layer mismatch");
        }
    }

    destination.clear(destination_sequence, true);
    const auto copy_tensor = [](ggml_tensor * source, ggml_tensor * target) {
        if (dsv4_test_comp_state_copy_after == 0) {
            throw std::runtime_error("injected DSV4 resident state copy failure");
        }
        ggml_backend_tensor_copy(source, target);
        if (dsv4_test_comp_state_copy_after > 0) {
            --dsv4_test_comp_state_copy_after;
        }
    };
    for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
        const uint32_t source_stream = depth*n_stream + (uint32_t) source_sequence;
        const uint32_t destination_stream = depth*destination.n_stream + (uint32_t) destination_sequence;
        for (size_t i = 0; i < layers.size(); ++i) {
            copy_tensor(layers[i].kv_stream[source_stream], destination.layers[i].kv_stream[destination_stream]);
            copy_tensor(layers[i].score_stream[source_stream], destination.layers[i].score_stream[destination_stream]);
        }
    }
}

bool llama_dsv4_comp_state::sequence_all_depths_zero(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_stream) {
        return false;
    }
    std::array<uint8_t, 4096> bytes;
    for (uint32_t depth = 0; depth <= n_rs_seq; ++depth) {
        const uint32_t stream = depth*n_stream + (uint32_t) seq_id;
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : { layer.kv_stream[stream], layer.score_stream[stream] }) {
                const size_t size = ggml_nbytes(tensor);
                for (size_t offset = 0; offset < size; offset += bytes.size()) {
                    const size_t chunk = std::min(bytes.size(), size - offset);
                    ggml_backend_tensor_get(tensor, bytes.data(), offset, chunk);
                    if (std::any_of(bytes.begin(), bytes.begin() + chunk,
                                    [](uint8_t value) { return value != 0; })) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
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

uint64_t llama_dsv4_comp_state::state_generation() const {
    return generation;
}

void llama_dsv4_comp_state::bump_generation() const {
    if (generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("DSV4 recurrent-state generation exhausted");
    }
    ++generation;
}

bool llama_dsv4_comp_state::has_generation_headroom(uint64_t increments) const {
    return increments <= std::numeric_limits<uint64_t>::max() - generation;
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

    bump_generation();

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

const char * llama_dsv4_resident_status_name(llama_dsv4_resident_status status) {
    switch (status) {
        case llama_dsv4_resident_status::ok:
            return "ok";
        case llama_dsv4_resident_status::invalid_scope:
            return "invalid_scope";
        case llama_dsv4_resident_status::invalid_sequence:
            return "invalid_sequence";
        case llama_dsv4_resident_status::unsupported_components:
            return "unsupported_components";
        case llama_dsv4_resident_status::capacity_exhausted: return "capacity_exhausted";
        case llama_dsv4_resident_status::slot_occupied:      return "slot_occupied";
        case llama_dsv4_resident_status::not_quiescent:      return "not_quiescent";
        case llama_dsv4_resident_status::stale_quote:        return "stale_quote";
        case llama_dsv4_resident_status::stale_handle:       return "stale_handle";
        case llama_dsv4_resident_status::generation_exhausted: return "generation_exhausted";
        case llama_dsv4_resident_status::resource_exhausted: return "resource_exhausted";
        case llama_dsv4_resident_status::backend_error:      return "backend_error";
    }
    return "unknown";
}

llama_dsv4_resident_detach_quote llama_dsv4_quote_resident_detach_layout(llama_dsv4_resident_detach_request request,
                                                                         uint32_t                           n_seq_max,
                                                                         uint32_t rollback_index,
                                                                         bool     aggregate_compressed) {
    llama_dsv4_resident_detach_quote result;
    result.seq_id              = request.seq_id;
    result.scope               = request.scope;
    result.rollback_index      = rollback_index;
    result.required_components = LLAMA_DSV4_RESIDENT_RAW_SWA | LLAMA_DSV4_RESIDENT_COMPRESSED |
                                 LLAMA_DSV4_RESIDENT_CSA_STATE | LLAMA_DSV4_RESIDENT_HCA_STATE |
                                 LLAMA_DSV4_RESIDENT_LID_STATE | LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX;
    if (request.scope == llama_dsv4_resident_scope::target_draft_pair) {
        result.required_components |= LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT;
    }

    if (request.scope != llama_dsv4_resident_scope::single_context &&
        request.scope != llama_dsv4_resident_scope::target_draft_pair) {
        result.status                 = llama_dsv4_resident_status::invalid_scope;
        result.unsupported_components = result.required_components;
        return result;
    }

    if (request.seq_id < 0 || (uint32_t) request.seq_id >= n_seq_max) {
        result.status                 = llama_dsv4_resident_status::invalid_sequence;
        result.unsupported_components = result.required_components;
        return result;
    }

    // The scalar rollback selector can travel in a future resident handle.
    // Aggregate compressed roots already have an execution-independent owner.
    // The static policy covers the default raw/SWA layout; the live cache
    // quote may add its opt-in sparse aperture. Every state plane remains
    // fixed to per-sequence tensor slices, and paired target/draft ownership
    // needs a coordinator.
    result.detachable_components = LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX;
    if (aggregate_compressed) {
        result.detachable_components |= LLAMA_DSV4_RESIDENT_COMPRESSED;
    }
    result.unsupported_components = result.required_components & ~result.detachable_components;
    result.status                 = result.unsupported_components == 0 ? llama_dsv4_resident_status::ok :
                                                                         llama_dsv4_resident_status::unsupported_components;
    return result;
}

namespace {

uint64_t dsv4_allocate_resident_cache_id() {
    static std::atomic<uint64_t> next{ 1 };
    uint64_t current = next.load(std::memory_order_relaxed);
    while (true) {
        if (current == 0 || current == std::numeric_limits<uint64_t>::max()) {
            throw std::overflow_error("DSV4 resident cache identity space exhausted");
        }
        if (next.compare_exchange_weak(current, current + 1, std::memory_order_relaxed)) {
            return current;
        }
    }
}

llama_dsv4_resident_status dsv4_resident_status_from_raw(llama_kv_iswa_resident_status status) {
    switch (status) {
        case llama_kv_iswa_resident_status::ok:                   return llama_dsv4_resident_status::ok;
        case llama_kv_iswa_resident_status::invalid_sequence:     return llama_dsv4_resident_status::invalid_sequence;
        case llama_kv_iswa_resident_status::unsupported_layout:   return llama_dsv4_resident_status::unsupported_components;
        case llama_kv_iswa_resident_status::slot_occupied:        return llama_dsv4_resident_status::slot_occupied;
        case llama_kv_iswa_resident_status::capacity_exhausted:   return llama_dsv4_resident_status::capacity_exhausted;
        case llama_kv_iswa_resident_status::stale_quote:          return llama_dsv4_resident_status::stale_quote;
        case llama_kv_iswa_resident_status::stale_handle:         return llama_dsv4_resident_status::stale_handle;
        case llama_kv_iswa_resident_status::generation_exhausted: return llama_dsv4_resident_status::generation_exhausted;
        case llama_kv_iswa_resident_status::not_quiescent:        return llama_dsv4_resident_status::not_quiescent;
        case llama_kv_iswa_resident_status::resource_exhausted:   return llama_dsv4_resident_status::resource_exhausted;
        case llama_kv_iswa_resident_status::backend_error:        return llama_dsv4_resident_status::backend_error;
    }
    return llama_dsv4_resident_status::backend_error;
}

llama_dsv4_resident_status dsv4_resident_status_from_compressed(llama_dsv4_comp_status status) {
    switch (status) {
        case llama_dsv4_comp_status::ok:                   return llama_dsv4_resident_status::ok;
        case llama_dsv4_comp_status::capacity_exhausted:   return llama_dsv4_resident_status::capacity_exhausted;
        case llama_dsv4_comp_status::generation_exhausted: return llama_dsv4_resident_status::generation_exhausted;
        case llama_dsv4_comp_status::stale_quote:          return llama_dsv4_resident_status::stale_quote;
        case llama_dsv4_comp_status::stale_ticket:         return llama_dsv4_resident_status::stale_quote;
        case llama_dsv4_comp_status::stale_handle:
        case llama_dsv4_comp_status::handle_not_found:
        case llama_dsv4_comp_status::binding_not_found:    return llama_dsv4_resident_status::stale_handle;
        case llama_dsv4_comp_status::busy:                 return llama_dsv4_resident_status::not_quiescent;
        case llama_dsv4_comp_status::slot_occupied:
        case llama_dsv4_comp_status::handle_bound:
        case llama_dsv4_comp_status::handle_resident:      return llama_dsv4_resident_status::slot_occupied;
        case llama_dsv4_comp_status::resource_exhausted:   return llama_dsv4_resident_status::resource_exhausted;
        case llama_dsv4_comp_status::invalid_argument:     return llama_dsv4_resident_status::backend_error;
    }
    return llama_dsv4_resident_status::backend_error;
}

struct dsv4_resident_identity {
    uint64_t id = dsv4_allocate_resident_cache_id();
};

enum class dsv4_composite_plan_state : uint8_t {
    prepared,
    committed,
    rolled_back,
};

struct dsv4_composite_record {
    llama_dsv4_resident_handle       handle;
    llama_kv_iswa_resident_handle    raw;
    llama_dsv4_comp_resident_handle  compressed;
    uint32_t                         state_slot = UINT32_MAX;
    uint32_t                         rollback_index = 0;
    uint32_t                         active_rollback_depth = 0;
};

class dsv4_raw_transaction_scope {
public:
  explicit dsv4_raw_transaction_scope(llama_kv_cache_iswa & raw) : token(raw.acquire_resident_transaction()) {}

  explicit operator bool() const { return static_cast<bool>(token); }

private:
  llama_kv_iswa_resident_transaction token;
};

} // namespace

struct llama_dsv4_resident_detach_plan {
    std::shared_ptr<const dsv4_resident_identity> owner;
    uint64_t                                      epoch = 0;
    llama_seq_id                                  execution_id = -1;
    uint32_t                                      state_slot = UINT32_MAX;
    uint32_t                                      rollback_index = 0;
    uint32_t                                      active_rollback_depth = 0;
    std::array<uint64_t, 3>                       state_generation = {};
    llama_kv_iswa_resident_detach_quote           raw;
    llama_dsv4_comp_detach_quote                  compressed;
    std::map<uint64_t, dsv4_composite_record>     prepared_record;
    dsv4_composite_plan_state                     state = dsv4_composite_plan_state::prepared;
};

struct llama_dsv4_resident_attach_plan {
    std::shared_ptr<const dsv4_resident_identity> owner;
    uint64_t                                      epoch = 0;
    llama_seq_id                                  execution_id = -1;
    llama_dsv4_resident_handle                    resident;
    std::array<uint64_t, 3>                       state_generation = {};
    llama_kv_iswa_resident_attach_quote           raw;
    llama_dsv4_comp_attach_quote                  compressed;
    dsv4_composite_plan_state                     state = dsv4_composite_plan_state::prepared;
};

struct llama_dsv4_composite_resident::impl {
    impl(
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
            uint32_t resident_capacity) :
        raw(raw),
        compressed(compressed),
        execution{ &csa_execution, &hca_execution, &lid_execution },
        resident{ &csa_resident, &hca_resident, &lid_resident },
        rollback_index(rollback_index),
        active_rollback_depth(active_rollback_depth),
        n_seq_max(n_seq_max),
        slots(resident_capacity, false) {
        if (resident_capacity == 0 || rollback_index.size() != n_seq_max ||
            csa_execution.get_n_stream() != n_seq_max || hca_execution.get_n_stream() != n_seq_max ||
            lid_execution.get_n_stream() != n_seq_max || csa_resident.get_n_stream() != resident_capacity ||
            hca_resident.get_n_stream() != resident_capacity || lid_resident.get_n_stream() != resident_capacity ||
            csa_execution.get_n_rs_seq() != csa_resident.get_n_rs_seq() ||
            hca_execution.get_n_rs_seq() != hca_resident.get_n_rs_seq() ||
            lid_execution.get_n_rs_seq() != lid_resident.get_n_rs_seq()) {
            throw std::invalid_argument("invalid DSV4 composite resident geometry");
        }
    }

    void clear_state(std::array<llama_dsv4_comp_state *, 3> states, llama_seq_id seq) const {
        for (auto * state : states) {
            state->clear(seq, true);
        }
    }

    bool state_is_empty(std::array<llama_dsv4_comp_state *, 3> states, llama_seq_id seq) const {
        return std::all_of(states.begin(), states.end(),
                [seq](const llama_dsv4_comp_state * state) { return state->sequence_all_depths_zero(seq); });
    }

    void copy_state(
            std::array<llama_dsv4_comp_state *, 3> source,
            std::array<llama_dsv4_comp_state *, 3> destination,
            llama_seq_id source_seq,
            llama_seq_id destination_seq) const {
        for (size_t i = 0; i < source.size(); ++i) {
            source[i]->copy_sequence_all_depths_to(*destination[i], source_seq, destination_seq);
        }
    }

    llama_kv_cache_iswa & raw;
    llama_dsv4_comp_pool & compressed;
    std::array<llama_dsv4_comp_state *, 3> execution;
    std::array<llama_dsv4_comp_state *, 3> resident;
    std::vector<uint32_t> & rollback_index;
    uint32_t & active_rollback_depth;
    uint32_t n_seq_max;
    std::shared_ptr<const dsv4_resident_identity> identity =
            std::make_shared<const dsv4_resident_identity>();
    mutable std::recursive_mutex mutex;
    std::vector<bool> slots;
    std::map<uint64_t, dsv4_composite_record> handles;
    uint64_t epoch = 1;
    uint64_t next_handle_id = 1;
    uint64_t next_handle_generation = 1;
    uint64_t next_lease_generation = 1;
};

class llama_dsv4_composite_resident::aggregate_clear_transaction {
public:
    explicit aggregate_clear_transaction(impl & state) :
        coordinator_lock(state.mutex),
        raw_transaction(state.raw),
        permitted(static_cast<bool>(raw_transaction) && state.handles.empty() &&
                  !state.raw.has_resident_handles() &&
                  state.compressed.memory_usage_snapshot().resident_handles == 0) {
    }

    explicit operator bool() const {
        return permitted;
    }

private:
    std::unique_lock<std::recursive_mutex> coordinator_lock;
    dsv4_raw_transaction_scope             raw_transaction;
    bool                                   permitted = false;
};

llama_dsv4_composite_resident::llama_dsv4_composite_resident(
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
        uint32_t resident_capacity) :
    pimpl(std::make_unique<impl>(raw, compressed, csa_execution, hca_execution, lid_execution,
            csa_resident, hca_resident, lid_resident, rollback_index, active_rollback_depth,
            n_seq_max, resident_capacity)) {
}

llama_dsv4_composite_resident::~llama_dsv4_composite_resident() {
    if (pimpl && !pimpl->handles.empty()) {
        std::terminate();
    }
}

std::unique_ptr<llama_dsv4_composite_resident::aggregate_clear_transaction>
llama_dsv4_composite_resident::acquire_aggregate_clear_transaction() {
    return std::make_unique<aggregate_clear_transaction>(*pimpl);
}

llama_dsv4_resident_detach_quote llama_dsv4_composite_resident::quote_detach(
        llama_dsv4_resident_detach_request request) const {
    llama_dsv4_resident_detach_quote result;
    result.seq_id = request.seq_id;
    result.scope  = request.scope;
    result.required_components = LLAMA_DSV4_RESIDENT_RAW_SWA | LLAMA_DSV4_RESIDENT_COMPRESSED |
            LLAMA_DSV4_RESIDENT_CSA_STATE | LLAMA_DSV4_RESIDENT_HCA_STATE |
            LLAMA_DSV4_RESIDENT_LID_STATE | LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX;
    result.unsupported_components = result.required_components;
    if (request.scope != llama_dsv4_resident_scope::single_context) {
        result.status = request.scope == llama_dsv4_resident_scope::target_draft_pair ?
                llama_dsv4_resident_status::unsupported_components : llama_dsv4_resident_status::invalid_scope;
        if (request.scope == llama_dsv4_resident_scope::target_draft_pair) {
            result.required_components |= LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT;
            result.unsupported_components = result.required_components;
        }
        return result;
    }
    if (request.seq_id < 0 || (uint32_t) request.seq_id >= pimpl->n_seq_max) {
        result.status = llama_dsv4_resident_status::invalid_sequence;
        return result;
    }

    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    const auto free_slot = std::find(pimpl->slots.begin(), pimpl->slots.end(), false);
    if (free_slot == pimpl->slots.end()) {
        result.status = llama_dsv4_resident_status::capacity_exhausted;
        return result;
    }
    if (pimpl->next_handle_id == 0 || pimpl->next_handle_generation == 0 ||
        pimpl->next_lease_generation == 0) {
        result.status = llama_dsv4_resident_status::generation_exhausted;
        return result;
    }
    const uint32_t rollback = pimpl->rollback_index[(uint32_t) request.seq_id];
    if (rollback > pimpl->execution[0]->get_n_rs_seq() ||
        pimpl->active_rollback_depth > pimpl->execution[0]->get_n_rs_seq()) {
        result.status = llama_dsv4_resident_status::backend_error;
        return result;
    }
    if (std::any_of(pimpl->resident.begin(), pimpl->resident.end(),
                    [](const llama_dsv4_comp_state * state) { return !state->has_generation_headroom(2); }) ||
        std::any_of(pimpl->execution.begin(), pimpl->execution.end(),
                    [](const llama_dsv4_comp_state * state) { return !state->has_generation_headroom(1); })) {
        result.status = llama_dsv4_resident_status::generation_exhausted;
        return result;
    }

    auto raw = pimpl->raw.quote_resident_detach(request.seq_id);
    if (raw.status != llama_kv_iswa_resident_status::ok) {
        result.status = dsv4_resident_status_from_raw(raw.status);
        return result;
    }
    auto compressed = pimpl->compressed.quote_detach_preserving_empty_execution((uint32_t) request.seq_id);
    if (compressed.status != llama_dsv4_comp_status::ok) {
        result.status = dsv4_resident_status_from_compressed(compressed.status);
        return result;
    }

    try {
        auto plan                   = std::make_shared<llama_dsv4_resident_detach_plan>();
        plan->owner                 = pimpl->identity;
        plan->epoch                 = pimpl->epoch;
        plan->execution_id          = request.seq_id;
        plan->state_slot            = (uint32_t) std::distance(pimpl->slots.begin(), free_slot);
        plan->rollback_index        = rollback;
        plan->active_rollback_depth = pimpl->active_rollback_depth;
        for (size_t i = 0; i < pimpl->execution.size(); ++i) {
            plan->state_generation[i] = pimpl->execution[i]->state_generation();
        }
        plan->raw                   = std::move(raw);
        plan->compressed            = std::move(compressed);

        dsv4_composite_record record;
        record.handle = {
            pimpl->identity->id,
            pimpl->next_handle_id,
            pimpl->next_handle_generation,
            pimpl->next_lease_generation,
        };
        record.raw                   = plan->raw.resident;
        record.compressed            = plan->compressed.resident;
        record.state_slot            = plan->state_slot;
        record.rollback_index        = rollback;
        record.active_rollback_depth = pimpl->active_rollback_depth;
        plan->prepared_record.emplace(record.handle.id, record);

        pimpl->next_handle_id = record.handle.id == std::numeric_limits<uint64_t>::max() ?
                0 : record.handle.id + 1;
        pimpl->next_handle_generation = record.handle.handle_generation == std::numeric_limits<uint64_t>::max() ?
                0 : record.handle.handle_generation + 1;
        pimpl->next_lease_generation = record.handle.lease_generation == std::numeric_limits<uint64_t>::max() ?
                0 : record.handle.lease_generation + 1;

        result.status                 = llama_dsv4_resident_status::ok;
        result.rollback_index         = rollback;
        result.detachable_components  = result.required_components;
        result.unsupported_components = 0;
        result.resident_state_slot    = plan->state_slot;
        result.resident               = record.handle;
        result.plan                   = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_resident_status::resource_exhausted;
        return result;
    }
}

llama_dsv4_resident_result llama_dsv4_composite_resident::detach(
        const llama_dsv4_resident_detach_quote & quote) {
    llama_dsv4_resident_result result;
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (!quote.plan || quote.plan->owner != pimpl->identity ||
        quote.plan->state != dsv4_composite_plan_state::prepared ||
        quote.plan->epoch != pimpl->epoch || quote.plan->execution_id < 0 ||
        (uint32_t) quote.plan->execution_id >= pimpl->n_seq_max ||
        quote.plan->state_slot >= pimpl->slots.size() || pimpl->slots[quote.plan->state_slot] ||
        pimpl->rollback_index[(uint32_t) quote.plan->execution_id] != quote.plan->rollback_index ||
        pimpl->active_rollback_depth != quote.plan->active_rollback_depth ||
        quote.plan->prepared_record.empty()) {
        result.status = llama_dsv4_resident_status::stale_quote;
        return result;
    }
    dsv4_raw_transaction_scope transaction(pimpl->raw);
    if (!transaction) {
        result.status = llama_dsv4_resident_status::not_quiescent;
        return result;
    }
    if (pimpl->raw.validate_resident_detach(quote.plan->raw) != llama_kv_iswa_resident_status::ok ||
        !std::equal(quote.plan->state_generation.begin(), quote.plan->state_generation.end(),
                pimpl->execution.begin(),
                [](uint64_t generation, const llama_dsv4_comp_state * state) {
                    return generation == state->state_generation();
                })) {
        result.status = llama_dsv4_resident_status::stale_quote;
        return result;
    }

    try {
        pimpl->copy_state(pimpl->execution, pimpl->resident,
                quote.plan->execution_id, (llama_seq_id) quote.plan->state_slot);
    } catch (const std::bad_alloc &) {
        pimpl->clear_state(pimpl->resident, (llama_seq_id) quote.plan->state_slot);
        result.status = llama_dsv4_resident_status::resource_exhausted;
        return result;
    } catch (...) {
        pimpl->clear_state(pimpl->resident, (llama_seq_id) quote.plan->state_slot);
        result.status = llama_dsv4_resident_status::backend_error;
        return result;
    }

    const auto compressed = pimpl->compressed.detach(quote.plan->compressed);
    if (compressed.status != llama_dsv4_comp_status::ok) {
        pimpl->clear_state(pimpl->resident, (llama_seq_id) quote.plan->state_slot);
        result.status = dsv4_resident_status_from_compressed(compressed.status);
        return result;
    }
    const auto rollback = [&]() {
        if (pimpl->compressed.rollback_detach(quote.plan->compressed) != llama_dsv4_comp_status::ok) {
            std::terminate();
        }
        pimpl->clear_state(pimpl->resident, (llama_seq_id) quote.plan->state_slot);
        quote.plan->state = dsv4_composite_plan_state::rolled_back;
    };
    llama_kv_iswa_resident_result raw;
    try {
        raw = pimpl->raw.detach_resident(quote.plan->raw);
    } catch (const std::bad_alloc &) {
        rollback();
        result.status = llama_dsv4_resident_status::resource_exhausted;
        return result;
    } catch (...) {
        rollback();
        result.status = llama_dsv4_resident_status::backend_error;
        return result;
    }
    if (raw.status != llama_kv_iswa_resident_status::ok) {
        rollback();
        result.status = dsv4_resident_status_from_raw(raw.status);
        return result;
    }

    auto node = quote.plan->prepared_record.extract(quote.plan->prepared_record.begin());
    if (node.empty()) {
        std::terminate();
    }
    const auto inserted = pimpl->handles.insert(std::move(node));
    if (!inserted.inserted) {
        std::terminate();
    }
    pimpl->slots[quote.plan->state_slot] = true;
    pimpl->clear_state(pimpl->execution, quote.plan->execution_id);
    pimpl->rollback_index[(uint32_t) quote.plan->execution_id] = 0;
    ++pimpl->epoch;
    quote.plan->state = dsv4_composite_plan_state::committed;
    result.status     = llama_dsv4_resident_status::ok;
    result.resident   = inserted.position->second.handle;
    return result;
}

llama_dsv4_resident_attach_quote llama_dsv4_composite_resident::quote_attach(
        llama_dsv4_resident_handle resident,
        llama_seq_id execution_id) const {
    llama_dsv4_resident_attach_quote result;
    result.execution_id = execution_id;
    result.resident     = resident;
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (resident.cache_id != pimpl->identity->id || resident.id == 0) {
        result.status = llama_dsv4_resident_status::stale_handle;
        return result;
    }
    const auto record = pimpl->handles.find(resident.id);
    if (record == pimpl->handles.end() || !(record->second.handle == resident)) {
        result.status = llama_dsv4_resident_status::stale_handle;
        return result;
    }
    if (execution_id < 0 || (uint32_t) execution_id >= pimpl->n_seq_max) {
        result.status = llama_dsv4_resident_status::invalid_sequence;
        return result;
    }
    if (pimpl->rollback_index[(uint32_t) execution_id] != 0 ||
        pimpl->active_rollback_depth != record->second.active_rollback_depth ||
        !pimpl->state_is_empty(pimpl->execution, execution_id)) {
        result.status = llama_dsv4_resident_status::slot_occupied;
        return result;
    }
    if (std::any_of(pimpl->execution.begin(), pimpl->execution.end(),
                    [](const llama_dsv4_comp_state * state) { return !state->has_generation_headroom(2); }) ||
        std::any_of(pimpl->resident.begin(), pimpl->resident.end(),
                    [](const llama_dsv4_comp_state * state) { return !state->has_generation_headroom(1); })) {
        result.status = llama_dsv4_resident_status::generation_exhausted;
        return result;
    }

    auto raw = pimpl->raw.quote_resident_attach(record->second.raw, execution_id);
    if (raw.status != llama_kv_iswa_resident_status::ok) {
        result.status = dsv4_resident_status_from_raw(raw.status);
        return result;
    }
    llama_dsv4_comp_handle_id destination_root = 0;
    const auto binding = pimpl->compressed.get_binding((uint32_t) execution_id, destination_root);
    auto compressed = binding == llama_dsv4_comp_status::binding_not_found ?
            pimpl->compressed.quote_attach(record->second.compressed, (uint32_t) execution_id) :
            pimpl->compressed.quote_attach_replacing_empty(record->second.compressed, (uint32_t) execution_id);
    if (compressed.status != llama_dsv4_comp_status::ok) {
        result.status = dsv4_resident_status_from_compressed(compressed.status);
        return result;
    }

    try {
        auto plan          = std::make_shared<llama_dsv4_resident_attach_plan>();
        plan->owner        = pimpl->identity;
        plan->epoch        = pimpl->epoch;
        plan->execution_id = execution_id;
        plan->resident     = resident;
        for (size_t i = 0; i < pimpl->execution.size(); ++i) {
            plan->state_generation[i] = pimpl->execution[i]->state_generation();
        }
        plan->raw          = std::move(raw);
        plan->compressed   = std::move(compressed);
        result.status      = llama_dsv4_resident_status::ok;
        result.plan        = std::move(plan);
        return result;
    } catch (const std::bad_alloc &) {
        result.status = llama_dsv4_resident_status::resource_exhausted;
        return result;
    }
}

llama_dsv4_resident_status llama_dsv4_composite_resident::attach(
        const llama_dsv4_resident_attach_quote & quote) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (!quote.plan || quote.plan->owner != pimpl->identity ||
        quote.plan->state != dsv4_composite_plan_state::prepared || quote.plan->epoch != pimpl->epoch ||
        quote.plan->execution_id < 0 || (uint32_t) quote.plan->execution_id >= pimpl->n_seq_max) {
        return llama_dsv4_resident_status::stale_quote;
    }
    const auto record = pimpl->handles.find(quote.plan->resident.id);
    if (record == pimpl->handles.end() || !(record->second.handle == quote.plan->resident) ||
        pimpl->rollback_index[(uint32_t) quote.plan->execution_id] != 0 ||
        pimpl->active_rollback_depth != record->second.active_rollback_depth) {
        return llama_dsv4_resident_status::stale_quote;
    }
    dsv4_raw_transaction_scope transaction(pimpl->raw);
    if (!transaction) {
        return llama_dsv4_resident_status::not_quiescent;
    }
    if (pimpl->raw.validate_resident_attach(quote.plan->raw) != llama_kv_iswa_resident_status::ok ||
        !std::equal(quote.plan->state_generation.begin(), quote.plan->state_generation.end(),
                pimpl->execution.begin(),
                [](uint64_t generation, const llama_dsv4_comp_state * state) {
                    return generation == state->state_generation();
                })) {
        return llama_dsv4_resident_status::stale_quote;
    }

    const auto compressed = pimpl->compressed.commit_attach(quote.plan->compressed);
    if (compressed != llama_dsv4_comp_status::ok) {
        return dsv4_resident_status_from_compressed(compressed);
    }
    const auto rollback = [&]() {
        pimpl->clear_state(pimpl->execution, quote.plan->execution_id);
        if (pimpl->compressed.rollback_attach(quote.plan->compressed) != llama_dsv4_comp_status::ok) {
            std::terminate();
        }
        quote.plan->state = dsv4_composite_plan_state::rolled_back;
    };
    try {
        pimpl->copy_state(pimpl->resident, pimpl->execution,
                (llama_seq_id) record->second.state_slot, quote.plan->execution_id);
    } catch (const std::bad_alloc &) {
        rollback();
        return llama_dsv4_resident_status::resource_exhausted;
    } catch (...) {
        rollback();
        return llama_dsv4_resident_status::backend_error;
    }

    llama_kv_iswa_resident_status raw;
    try {
        raw = pimpl->raw.commit_resident_attach_final(
                quote.plan->raw, llama_kv_iswa_resident_final_step::confirmed);
    } catch (const std::bad_alloc &) {
        rollback();
        return llama_dsv4_resident_status::resource_exhausted;
    } catch (...) {
        rollback();
        return llama_dsv4_resident_status::backend_error;
    }
    if (raw != llama_kv_iswa_resident_status::ok) {
        rollback();
        return dsv4_resident_status_from_raw(raw);
    }

    pimpl->rollback_index[(uint32_t) quote.plan->execution_id] = record->second.rollback_index;
    pimpl->clear_state(pimpl->resident, (llama_seq_id) record->second.state_slot);
    pimpl->slots[record->second.state_slot] = false;
    pimpl->handles.erase(record);
    ++pimpl->epoch;
    quote.plan->state = dsv4_composite_plan_state::committed;
    return llama_dsv4_resident_status::ok;
}

llama_dsv4_resident_status llama_dsv4_composite_resident::release(
        llama_dsv4_resident_handle resident,
        llama_kv_iswa_resident_release_audit * audit) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (resident.cache_id != pimpl->identity->id || resident.id == 0) {
        return llama_dsv4_resident_status::stale_handle;
    }
    const auto record = pimpl->handles.find(resident.id);
    if (record == pimpl->handles.end() || !(record->second.handle == resident)) {
        return llama_dsv4_resident_status::stale_handle;
    }
    if (std::any_of(pimpl->resident.begin(), pimpl->resident.end(),
                    [](const llama_dsv4_comp_state * state) { return !state->has_generation_headroom(1); })) {
        return llama_dsv4_resident_status::generation_exhausted;
    }
    dsv4_raw_transaction_scope transaction(pimpl->raw);
    if (!transaction) {
        return llama_dsv4_resident_status::not_quiescent;
    }
    const auto compressed = pimpl->compressed.prepare_release(record->second.compressed);
    if (compressed.status != llama_dsv4_comp_status::ok) {
        return dsv4_resident_status_from_compressed(compressed.status);
    }
    const auto rollback = [&]() {
        if (pimpl->compressed.rollback_release(compressed) != llama_dsv4_comp_status::ok) {
            std::terminate();
        }
    };
    llama_kv_iswa_resident_status raw;
    try {
        raw = pimpl->raw.release_resident(record->second.raw, audit);
    } catch (const std::bad_alloc &) {
        rollback();
        return llama_dsv4_resident_status::resource_exhausted;
    } catch (...) {
        rollback();
        return llama_dsv4_resident_status::backend_error;
    }
    if (raw != llama_kv_iswa_resident_status::ok) {
        rollback();
        return dsv4_resident_status_from_raw(raw);
    }
    if (pimpl->compressed.commit_release(compressed) != llama_dsv4_comp_status::ok) {
        std::terminate();
    }
    pimpl->clear_state(pimpl->resident, (llama_seq_id) record->second.state_slot);
    pimpl->slots[record->second.state_slot] = false;
    pimpl->handles.erase(record);
    ++pimpl->epoch;
    return llama_dsv4_resident_status::ok;
}

bool llama_dsv4_composite_resident::has_handles() const {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    return !pimpl->handles.empty();
}

llama_dsv4_resident_usage llama_dsv4_composite_resident::usage() const {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    llama_dsv4_resident_usage result;
    result.cache_id       = pimpl->identity->id;
    result.epoch          = pimpl->epoch;
    result.capacity       = (uint32_t) pimpl->slots.size();
    result.occupied_slots = (uint32_t) std::count(pimpl->slots.begin(), pimpl->slots.end(), true);
    result.handles        = (uint32_t) pimpl->handles.size();
    return result;
}

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
    rs_idx(n_seq_max, 0),
    admission_state(std::make_unique<llama_kv_cache_dsv4_admission_state>()) {

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

    const bool aggregate_capable = unified && n_seq_max > 1 &&
            sparse_buft != nullptr && type_k == GGML_TYPE_F16;
    bool affine_footprint_overflow = false;
    const uint64_t affine_compressed_bytes = aggregate_capable ?
            dsv4_affine_compressed_bytes(
                model, type_k, n_seq_max, c4_logical_rows, hca_logical_rows,
                filter, affine_footprint_overflow) : 0;
    const llama_dsv4_aggregate_selector selector {
        /*.unified                   =*/ unified,
        /*.sparse_supported          =*/ sparse_buft != nullptr,
        /*.f16                       =*/ type_k == GGML_TYPE_F16,
        /*.disabled                  =*/ std::getenv("LLAMA_DSV4_AGGREGATE_POOL_DISABLE") != nullptr,
        /*.forced                    =*/ std::getenv("LLAMA_DSV4_AGGREGATE_POOL_FORCE") != nullptr,
        /*.affine_footprint_overflow =*/ affine_footprint_overflow,
        /*.n_seq_max                 =*/ n_seq_max,
        /*.affine_compressed_bytes   =*/ affine_compressed_bytes,
    };

    // The aggregate layout is target-only and currently requires the F16
    // indexed concat path. Prefer affine storage while its exact declared K
    // tensors fit the M2 Ultra product budget: Metal System Trace shows that
    // aggregate sparse mapping fragments a four-stream request into thousands
    // of Compute/Blit intervals even though GPU-active work is unchanged.
    aggregate_compressed = dsv4_select_aggregate_compressed(selector);
    if (selector.unified && selector.n_seq_max > 1 && selector.sparse_supported && selector.f16 &&
            !selector.disabled && !aggregate_compressed) {
        LLAMA_LOG_INFO("%s: affine DSV4 compressed storage selected: footprint=%.2f MiB, aggregate threshold=%.2f MiB\n",
                __func__, affine_compressed_bytes/(1024.0*1024.0),
                LLAMA_DSV4_MAX_AFFINE_COMPRESSED_BYTES/(1024.0*1024.0));
    }

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
        LLAMA_LOG_INFO("%s: aggregate selection: affine footprint=%s%.2f MiB, threshold=%.2f MiB%s\n",
                __func__, affine_footprint_overflow ? ">" : "",
                affine_compressed_bytes/(1024.0*1024.0),
                LLAMA_DSV4_MAX_AFFINE_COMPRESSED_BYTES/(1024.0*1024.0),
                selector.forced ? ", forced" : "");
    }

    // Keep DSV4 KV/state streams per sequence even when public KV mode is unified.
    const bool unified_raw = false;

    hparams_raw.n_layer_nextn = 0;
    hparams_csa.n_layer_nextn = 0;
    hparams_hca.n_layer_nextn = 0;
    hparams_lid.n_layer_nextn = 0;

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    dsv4_make_k_only(hparams_raw);

    // Raw/SWA resident apertures double virtual address space but alias the
    // same physical pages while parked. A divisor of two therefore keeps one
    // full execution allocation plus sparse COW slack, rather than sizing the
    // heap as if only one of n_seq_max execution streams could be active.
    const bool composite_resident_enabled = std::getenv("LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE") != nullptr;
    const bool raw_resident_enabled = composite_resident_enabled ||
            std::getenv("LLAMA_DSV4_RAW_RESIDENT_ENABLE") != nullptr;
    ggml_backend_buffer_type_t raw_resident_buft = raw_resident_enabled && offload && n_seq_max > 1 ?
            dsv4_sparse_buft(model, 2) : nullptr;
    const uint32_t raw_resident_slots = raw_resident_buft != nullptr ? n_seq_max : 0;
    if (raw_resident_enabled && raw_resident_buft == nullptr) {
        LLAMA_LOG_WARN("%s: raw/SWA resident ownership requested without target sparse storage\n", __func__);
    }

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            v_trans, offload, swa_full, unified_raw, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr, raw_resident_buft, raw_resident_slots);

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

    if (composite_resident_enabled) {
        if (!aggregate_compressed || raw_resident_slots == 0) {
            LLAMA_LOG_WARN("%s: composite DSV4 resident ownership requires target aggregate and raw sparse apertures\n",
                    __func__);
        } else {
            LLAMA_LOG_INFO("%s: creating default-off DSV4 composite resident state aperture, slots = %u\n",
                    __func__, raw_resident_slots);
            resident_csa_state = std::make_unique<llama_dsv4_comp_state>(
                    model, offload, false, raw_resident_slots, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
                    2*model.hparams.n_embd_head_k(), n_rs_seq, "resident_csa", filter_csa);
            resident_hca_state = std::make_unique<llama_dsv4_comp_state>(
                    model, offload, false, raw_resident_slots, DSV4_HCA_RATIO, DSV4_HCA_RATIO,
                    model.hparams.n_embd_head_k(), n_rs_seq, "resident_hca", filter_hca);
            resident_lid_state = std::make_unique<llama_dsv4_comp_state>(
                    model, offload, false, raw_resident_slots, DSV4_CSA_RATIO, 2*DSV4_CSA_RATIO,
                    2*model.hparams.indexer_head_size, n_rs_seq, "resident_lid", filter_csa);
        }
    }

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
    if (resident_csa_state) {
        composite_resident = std::make_unique<llama_dsv4_composite_resident>(
                *kv_raw, *comp_pool, *csa_state, *hca_state, *lid_state,
                *resident_csa_state, *resident_hca_state, *resident_lid_state,
                rs_idx, n_rs_seq_active, n_seq_max, resident_csa_state->get_n_stream());
    }
}

llama_kv_cache_dsv4::~llama_kv_cache_dsv4() = default;

bool llama_kv_cache_dsv4::collect_admission_ranges(
        const llama_kv_admission_span * spans,
        size_t n_spans,
        std::vector<dsv4_sparse_range> & ranges,
        llama_dsv4_batch_quote & quote) const {
    if (aggregate_compressed || spans == nullptr || n_spans == 0 || n_spans > n_seq_max) {
        return false;
    }

    const auto append_span = [&](const llama_kv_admission_span & span) {
        const int64_t raw_base_offset = dsv4_stream_offset(
                kv_raw->get_base()->get_n_stream(), span.seq_id, kv_raw->get_base()->get_size());
        const int64_t raw_swa_offset = dsv4_stream_offset(
                kv_raw->get_swa()->get_n_stream(), span.seq_id, kv_raw->get_swa()->get_size());
        // SWA is a sliding-window ring buffer of size_swa cells per stream (see
        // llama_kv_cache find_slot ring wrap). Reserving the full linear prompt
        // range [pos_begin, pos_end) here overflows size_swa for any prompt
        // larger than the window and turns a feasible request into 503. Reserve
        // the whole per-stream window instead: it is a safe superset of every
        // ring cell the window can touch and always fits (n_rows_total =
        // size_swa * n_stream).
        const int64_t raw_swa_size = kv_raw->get_swa()->get_size();
        if (!dsv4_sparse_append_k_row_run(
                    kv_raw->get_base(), raw_base_offset + span.pos_begin, raw_base_offset + span.pos_end,
                    LLAMA_DSV4_MEMORY_RAW, ranges) ||
                !dsv4_sparse_append_k_row_run(
                    kv_raw->get_swa(), raw_swa_offset, raw_swa_offset + raw_swa_size,
                    LLAMA_DSV4_MEMORY_RAW, ranges)) {
            return false;
        }
        quote.families[LLAMA_DSV4_MEMORY_RAW].logical_rows += span.pos_end - span.pos_begin;

        const auto append_compressed = [&](const llama_kv_cache * cache, uint32_t ratio,
                                           llama_dsv4_memory_family family, bool needs_dummy) {
            const int64_t stream_offset = dsv4_stream_offset(
                    cache->get_n_stream(), span.seq_id, cache->get_size());
            const int64_t row_begin = span.pos_begin/ratio;
            const int64_t row_end = span.pos_end/ratio;
            if (!dsv4_sparse_append_k_row_run(
                        cache, stream_offset + row_begin, stream_offset + row_end, family, ranges)) {
                return false;
            }

            // CSA/LID use the last per-stream row as the masked destination
            // for every non-boundary compressor update. A [0, end) admission
            // containing any token therefore reserves that fixed scratch row.
            const int64_t dummy_row = stream_offset + cache->get_size() - 1;
            const bool dummy_in_run = row_begin <= cache->get_size() - 1 && row_end > cache->get_size() - 1;
            if (needs_dummy && !dummy_in_run &&
                    !dsv4_sparse_append_k_row_run(cache, dummy_row, dummy_row + 1, family, ranges)) {
                return false;
            }
            quote.families[family].logical_rows += row_end - row_begin;
            return true;
        };

        return append_compressed(kv_csa.get(), DSV4_CSA_RATIO, LLAMA_DSV4_MEMORY_CSA, true) &&
               append_compressed(kv_hca.get(), DSV4_HCA_RATIO, LLAMA_DSV4_MEMORY_HCA, false) &&
               append_compressed(kv_lid.get(), DSV4_CSA_RATIO, LLAMA_DSV4_MEMORY_LID, true);
    };

    for (size_t i = 0; i < n_spans; ++i) {
        if (!append_span(spans[i])) {
            return false;
        }
    }
    return true;
}

llama_kv_admission_status llama_kv_cache_dsv4::quote_admission(
        const llama_kv_admission_span * spans,
        size_t n_spans,
        llama_kv_admission_quote & quote) {
    llama_dsv4_batch_quote internal_quote = {};
    std::vector<dsv4_sparse_range> ranges;
    if (!collect_admission_ranges(spans, n_spans, ranges, internal_quote)) {
        dsv4_public_admission_quote(internal_quote, LLAMA_KV_ADMISSION_UNSUPPORTED, 0, quote);
        return quote.status;
    }

    dsv4_sparse_transaction transaction;
    const auto prepare_status = transaction.quote_ranges(ranges, internal_quote);
    llama_kv_admission_status public_status = LLAMA_KV_ADMISSION_ERROR;
    if (transaction.sparse_range_count() == 0) {
        public_status = LLAMA_KV_ADMISSION_UNSUPPORTED;
    } else if (prepare_status == dsv4_sparse_transaction::SUCCESS) {
        public_status = LLAMA_KV_ADMISSION_FEASIBLE;
    } else if (prepare_status == dsv4_sparse_transaction::PRESSURE) {
        public_status = LLAMA_KV_ADMISSION_PRESSURE;
    }
    dsv4_public_admission_quote(internal_quote, public_status, transaction.limiting_family_mask(), quote);
    return quote.status;
}

llama_kv_admission_status llama_kv_cache_dsv4::reserve_admission(
        const llama_kv_admission_span * spans,
        size_t n_spans,
        llama_kv_admission_quote & quote,
        uint64_t & id) {
    llama_dsv4_batch_quote internal_quote = {};
    std::vector<dsv4_sparse_range> ranges;
    id = 0;
    if (!collect_admission_ranges(spans, n_spans, ranges, internal_quote)) {
        dsv4_public_admission_quote(internal_quote, LLAMA_KV_ADMISSION_UNSUPPORTED, 0, quote);
        return quote.status;
    }

    auto reservation = std::make_unique<dsv4_sparse_transaction>();
    const auto prepare_status = reservation->reserve_ranges(ranges, internal_quote);
    llama_kv_admission_status public_status = LLAMA_KV_ADMISSION_ERROR;
    if (reservation->sparse_range_count() == 0) {
        public_status = LLAMA_KV_ADMISSION_UNSUPPORTED;
    } else if (prepare_status == dsv4_sparse_transaction::SUCCESS) {
        public_status = LLAMA_KV_ADMISSION_FEASIBLE;
    } else if (prepare_status == dsv4_sparse_transaction::PRESSURE) {
        public_status = LLAMA_KV_ADMISSION_PRESSURE;
    }
    dsv4_public_admission_quote(internal_quote, public_status, reservation->limiting_family_mask(), quote);
    if (public_status == LLAMA_KV_ADMISSION_FEASIBLE) {
        id = register_admission(std::move(reservation), spans, n_spans,
                internal_quote, quote.limiting_family_mask);
    }
    return quote.status;
}

uint64_t llama_kv_cache_dsv4::register_admission(
        std::unique_ptr<dsv4_sparse_transaction> reservation,
        const llama_kv_admission_span * spans,
        size_t n_spans,
        const llama_dsv4_batch_quote & quote,
        uint32_t limiting_family_mask) {
    GGML_ASSERT(reservation != nullptr);
    GGML_ASSERT(spans != nullptr);
    GGML_ASSERT(n_spans > 0);

    const uint64_t id = admission_state->next_id++;
    llama_kv_cache_dsv4_admission_state::entry entry;
    entry.spans.assign(spans, spans + n_spans);
    entry.reservation = std::move(reservation);
    entry.quote = quote;
    entry.limiting_family_mask = limiting_family_mask;
    const bool inserted = admission_state->entries.emplace(id, std::move(entry)).second;
    GGML_ASSERT(inserted);
    return id;
}

bool llama_kv_cache_dsv4::arm_admission(uint64_t id) {
    const auto found = admission_state->entries.find(id);
    if (found == admission_state->entries.end()) {
        return false;
    }
    found->second.armed = true;
    // Commit the sparse reservation now, while the ticket generation is current.
    // Deferring the commit to consume_admission (init_batch) lets the pool
    // generation advance via concurrent slot + speculative decodes between reserve
    // and the first prefill, so the Metal sparse commit rejects the ticket as
    // stale (result=stale reason=ticket-current; e.g. ticket_generation=6 vs
    // current_generation=27) and the memory preflight fails with ret=-2 -- the
    // transient HTTP 500 "Compute error". commit_ranges() is idempotent (a no-op
    // once the ticket is freed), so consume_admission's later call is harmless.
    if (found->second.reservation->commit_ranges() != GGML_METAL_SPARSE_RESERVATION_OK) {
        admission_state->entries.erase(found);
        return false;
    }
    return true;
}

void llama_kv_cache_dsv4::cancel_admission(uint64_t id) {
    admission_state->entries.erase(id);
}

// env: LLAMA_DSV4_ADMISSION_DEBUG=1 - one line per admission lifecycle event
// (arm / consume / re-map). Diagnostic for [TAG_DSV4_ADMISSION_REMAP].
static bool dsv4_admission_debug() {
    static const bool en = []() {
        const char * v = getenv("LLAMA_DSV4_ADMISSION_DEBUG");
        return v != nullptr && atoi(v) != 0;
    }();
    return en;
}

// env: LLAMA_DSV4_ADMISSION_REMAP=0 - restore the pre-fix behaviour, i.e. trust
// the arm-time commit and never re-map at consume. Diagnostic seam only; the
// pre-fix behaviour loses the first batch's compressed KV (see the tag below).
static bool dsv4_admission_remap_enabled() {
    static const bool en = []() {
        const char * v = getenv("LLAMA_DSV4_ADMISSION_REMAP");
        return v == nullptr || atoi(v) != 0;
    }();
    return en;
}

llama_kv_cache_dsv4::admission_consume_status llama_kv_cache_dsv4::consume_admission(
        const std::vector<llama_ubatch> & ubatches,
        llama_dsv4_batch_quote & quote,
        uint32_t & limiting_family_mask) {
    if (aggregate_compressed || ubatches.empty()) {
        return ADMISSION_NO_MATCH;
    }

    for (auto it = admission_state->entries.begin(); it != admission_state->entries.end(); ++it) {
        auto & entry = it->second;
        if (!entry.armed) {
            continue;
        }

        bool covered = true;
        for (const llama_ubatch & ubatch : ubatches) {
            for (uint32_t i = 0; i < ubatch.n_tokens && covered; ++i) {
                for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[i][s];
                    const llama_pos pos = ubatch.pos[i];
                    const bool token_covered = std::any_of(
                            entry.spans.begin(), entry.spans.end(),
                            [&](const llama_kv_admission_span & span) {
                                return span.seq_id == seq_id && pos >= span.pos_begin && pos < span.pos_end;
                            });
                    if (!token_covered) {
                        covered = false;
                        break;
                    }
                }
            }
            if (!covered) {
                break;
            }
        }
        if (!covered) {
            continue;
        }

        const auto commit_status = entry.reservation->commit_ranges();
        if (commit_status != GGML_METAL_SPARSE_RESERVATION_OK) {
            quote = entry.quote;
            limiting_family_mask = entry.limiting_family_mask;
            admission_state->entries.erase(it);
            return ADMISSION_ERROR;
        }

        // [TAG_DSV4_ADMISSION_REMAP]
        //
        // arm_admission() maps this admission's pages at task-launch time.
        // The slot then starts its prompt and llama-server issues
        // `slot.mem.seq_rm(slot.id, pos_next, -1)` (server-context.cpp, prompt
        // start) with pos_next == 0 for a fresh prompt. That lands in
        // llama_kv_cache_dsv4::seq_rm -> clear_compressed(seq, true) ->
        // dsv4_clear_tensor_stream(), which for a page-aligned stream *unmaps*
        // the whole per-stream range of kv_csa / kv_hca / kv_lid instead of
        // memsetting it - i.e. it destroys exactly the mapping this admission
        // just committed. The reservation's ticket was freed by that first
        // commit, so commit_ranges() above is a no-op and nothing ever maps
        // those pages again; the batch then writes its compressed K rows into
        // unmapped sparse pages, where the writes are discarded and the reads
        // return zero.
        //
        // That is the whole defect: the compressed KV rows of the FIRST batch
        // of every request (n_batch/compress_ratio rows, i.e. 512 rows at
        // -b 2048 with the 4:1 CSA/LID ratio) are permanently zero, in both
        // the CSA plane and the Lightning Indexer plane. Every later batch
        // takes the ordinary per-batch reservation path and is unaffected.
        //
        // Re-verify that the mapping is still there, and re-map with a FRESH
        // transaction if it is not. A fresh transaction cannot be rejected as
        // stale (which is why arm_admission commits early in the first place),
        // and it costs one reservation per request. If it cannot be re-made,
        // report NO_MATCH so the caller falls back to the ordinary per-batch
        // reservation, which is the same path every subsequent batch uses.
        const bool remap_enabled = dsv4_admission_remap_enabled();

        int resident = 1;
        std::vector<dsv4_sparse_range> ranges;
        if (remap_enabled || dsv4_admission_debug()) {
            llama_dsv4_batch_quote probe_quote = {};
            if (collect_admission_ranges(entry.spans.data(), entry.spans.size(), ranges, probe_quote)) {
                // 1 = every page mapped and exclusively owned, 0 = not, -1 =
                // probe unavailable. Anything other than a positive answer is
                // treated as "must re-map": the cost is one reservation per
                // request and the alternative is silently losing the batch.
                resident = dsv4_sparse_ranges_resident(ranges);
            }
        }

        if (resident != 1 && !remap_enabled) {
            if (dsv4_admission_debug()) {
                LLAMA_LOG_WARN("%s: DSV4 admission mapping was dropped after arm (resident=%d ranges=%zu);"
                        " re-map disabled, this batch's compressed KV will be lost\n",
                        __func__, resident, ranges.size());
            }
        } else if (resident != 1) {
            dsv4_sparse_transaction remap;
            llama_dsv4_batch_quote remap_quote = {};
            const auto reserve_status = remap.reserve_ranges(ranges, remap_quote);
            const bool remapped = reserve_status == dsv4_sparse_transaction::SUCCESS &&
                    remap.commit_ranges() == GGML_METAL_SPARSE_RESERVATION_OK;
            if (dsv4_admission_debug()) {
                LLAMA_LOG_WARN("%s: DSV4 admission mapping was dropped after arm; re-map %s"
                        " (ranges=%zu sparse_ranges=%zu)\n",
                        __func__, remapped ? "ok" : "FAILED",
                        ranges.size(), remap.sparse_range_count());
            }
            if (!remapped) {
                if (reserve_status == dsv4_sparse_transaction::PRESSURE) {
                    limiting_family_mask = remap.limiting_family_mask();
                }
                admission_state->entries.erase(it);
                return ADMISSION_NO_MATCH;
            }
        } else if (dsv4_admission_debug()) {
            LLAMA_LOG_WARN("%s: DSV4 admission mapping intact (resident=%d ranges=%zu)\n",
                    __func__, resident, ranges.size());
        }

        quote = entry.quote;
        limiting_family_mask = entry.limiting_family_mask;
        admission_state->entries.erase(it);
        return ADMISSION_COMMITTED;
    }

    return ADMISSION_NO_MATCH;
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    // Cover raw prepare's temporary metadata mutations and hand off without a
    // detach window to the raw graph context's own lifetime lease.
    [[maybe_unused]] auto raw_prepare_lease = kv_raw->acquire_resident_batch_lease();

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
    if (composite_resident && composite_resident->has_handles()) {
        throw std::runtime_error("cannot clear DSV4 cache with composite resident handles");
    }
    composite_resident.reset();
    kv_raw->clear(data);
    clear_compressed(-1, true); // DSV4 compressed buffers must never expose stale/uninit rows
    if (resident_csa_state) {
        resident_csa_state->clear(-1, true);
        resident_hca_state->clear(-1, true);
        resident_lid_state->clear(-1, true);
        composite_resident = std::make_unique<llama_dsv4_composite_resident>(
                *kv_raw, *comp_pool, *csa_state, *hca_state, *lid_state,
                *resident_csa_state, *resident_hca_state, *resident_lid_state,
                rs_idx, n_rs_seq_active, n_seq_max, resident_csa_state->get_n_stream());
    }
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

        // Snapshot planes are relative to one accepted state. Stacking a
        // second removal before the first restore executes would reinterpret
        // them relative to already-truncated raw KV and cannot compose.
        if (rs_idx[seq_id] != 0) {
            return false;
        }

        const bool res = kv_raw->seq_rm(seq_id, p0, p1);
        if (res) {
            rs_idx[seq_id] = (uint32_t) rollback;
        }

        return res;
    }

    const auto remove_and_clear = [&]() {
        const bool res = kv_raw->seq_rm(seq_id, p0, p1);

        if (res) {
            clear_compressed(seq_id, true);

            // diagnostic only - see dsv4_debug_zero_cleared_seq()
            if (dsv4_debug_zero_cleared_seq() && seq_id >= 0) {
                for (llama_kv_cache * raw : { kv_raw->get_base(), kv_raw->get_swa() }) {
                    if (raw == nullptr || (uint32_t) seq_id >= raw->get_n_stream()) {
                        continue;
                    }
                    for (uint32_t il : raw->get_layer_ids()) {
                        dsv4_clear_tensor_stream(raw->get_k_storage(il), (uint32_t) seq_id);
                    }
                }
            }
        }

        return res;
    };

    if (seq_id < 0 && aggregate_compressed) {
        if (composite_resident) {
            auto aggregate_transaction = composite_resident->acquire_aggregate_clear_transaction();
            if (!aggregate_transaction || !*aggregate_transaction) {
                return false;
            }
            return remove_and_clear();
        }

        if (kv_raw->has_resident_aperture()) {
            dsv4_raw_transaction_scope raw_transaction(*kv_raw);
            if (!raw_transaction || kv_raw->has_resident_handles() ||
                comp_pool->memory_usage_snapshot().resident_handles != 0) {
                return false;
            }
            return remove_and_clear();
        }

        if (kv_raw->has_resident_handles() || comp_pool->memory_usage_snapshot().resident_handles != 0) {
            // The raw-only resident mode has no composite coordinator to hold
            // the resident transaction.  Reject before touching raw storage;
            // callers can release the external resident lease and retry.
            return false;
        }
    }

    return remove_and_clear();
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
    for (const auto * state : { resident_csa_state.get(), resident_hca_state.get(), resident_lid_state.get() }) {
        if (state == nullptr) {
            continue;
        }
        for (const auto & buft_size : state->memory_breakdown()) {
            mb[buft_size.first] += buft_size.second;
        }
    }
    return mb;
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
    // The lease closes the gap between the fail-closed handle check and the
    // DSV4 header write. A new detach cannot land until the complete state
    // operation reaches its terminal boundary.
    if (seq_id < 0 && composite_resident && composite_resident->has_handles()) {
        throw std::runtime_error("DSV4 composite resident handles are not serializable");
    }
    [[maybe_unused]] auto raw_state_lease =
            seq_id < 0 ? kv_raw->acquire_resident_batch_lease() : nullptr;
    if (seq_id < 0 && kv_raw->has_resident_handles()) {
        throw std::runtime_error("DSV4 raw/SWA resident handles are not serializable");
    }
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
    if (seq_id < 0 && composite_resident && composite_resident->has_handles()) {
        throw std::runtime_error("DSV4 composite resident handles cannot survive whole-cache restore");
    }
    [[maybe_unused]] auto raw_state_lease =
            seq_id < 0 ? kv_raw->acquire_resident_batch_lease() : nullptr;
    if (seq_id < 0 && kv_raw->has_resident_handles()) {
        throw std::runtime_error("DSV4 raw/SWA resident handles cannot survive whole-cache restore");
    }
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

llama_dsv4_resident_detach_quote llama_kv_cache_dsv4::quote_resident_detach(
    llama_dsv4_resident_detach_request request) const {
    if (composite_resident) {
        return composite_resident->quote_detach(request);
    }
    const uint32_t rollback_index =
        request.seq_id >= 0 && (uint32_t) request.seq_id < n_seq_max ? rs_idx[request.seq_id] : 0;
    auto result = llama_dsv4_quote_resident_detach_layout(
            request, n_seq_max, rollback_index, aggregate_compressed);
    if (result.status == llama_dsv4_resident_status::invalid_scope ||
            result.status == llama_dsv4_resident_status::invalid_sequence) {
        return result;
    }
    if (kv_raw->quote_resident_detach(request.seq_id).status == llama_kv_iswa_resident_status::ok) {
        result.detachable_components |= LLAMA_DSV4_RESIDENT_RAW_SWA;
        result.unsupported_components = result.required_components & ~result.detachable_components;
        result.status = result.unsupported_components == 0 ? llama_dsv4_resident_status::ok :
                                                             llama_dsv4_resident_status::unsupported_components;
    }
    return result;
}

llama_dsv4_resident_result llama_kv_cache_dsv4::detach_resident(
        const llama_dsv4_resident_detach_quote & quote) {
    if (!composite_resident) {
        return {};
    }
    return composite_resident->detach(quote);
}

llama_dsv4_resident_attach_quote llama_kv_cache_dsv4::quote_resident_attach(
        llama_dsv4_resident_handle resident,
        llama_seq_id execution_id) const {
    if (!composite_resident) {
        return {};
    }
    return composite_resident->quote_attach(resident, execution_id);
}

llama_dsv4_resident_status llama_kv_cache_dsv4::attach_resident(
        const llama_dsv4_resident_attach_quote & quote) {
    return composite_resident ? composite_resident->attach(quote) :
                                llama_dsv4_resident_status::unsupported_components;
}

llama_dsv4_resident_status llama_kv_cache_dsv4::release_resident(
        llama_dsv4_resident_handle resident,
        llama_kv_iswa_resident_release_audit * audit) {
    return composite_resident ? composite_resident->release(resident, audit) :
                                llama_dsv4_resident_status::unsupported_components;
}

llama_dsv4_resident_usage llama_kv_cache_dsv4::resident_usage() const {
    return composite_resident ? composite_resident->usage() : llama_dsv4_resident_usage{};
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
        llama_dsv4_comp_pool replacement(llama_dsv4_comp_pool_config {
                usage.c4.capacity_segments - usage.c4.permanent_segments,
                usage.hca.capacity_segments - usage.hca.permanent_segments,
        });
        for (uint32_t execution_id = 0; execution_id < n_seq_max; ++execution_id) {
            const auto handle = replacement.create_handle();
            GGML_ASSERT(handle.status == llama_dsv4_comp_status::ok);
            GGML_ASSERT(replacement.bind(execution_id, handle.handle) == llama_dsv4_comp_status::ok);
        }
        // Keep the coordinator's compressed-pool reference valid.  Replacing
        // the unique_ptr leaves a dangling reference even when no resident
        // lease is outstanding; move assignment swaps the pool implementation
        // in place after the transaction above has proved that no resident
        // handles remain.
        *comp_pool = std::move(replacement);
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
    resident_batch_lease(kv->acquire_resident_batch_lease()),
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
    resident_batch_lease(),
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
    resident_batch_lease(kv->acquire_resident_batch_lease()),
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
        dsv4_test_cow_copy_operations.fetch_add(1, std::memory_order_relaxed);
        ggml_tensor * tensor = cache->get_k_storage(il);
        const size_t row_size = tensor->nb[1];
        std::vector<uint8_t> bytes((size_t) populated_rows*row_size);
        const size_t source_offset = (size_t) source_segment*LLAMA_DSV4_COMP_SEGMENT_ROWS*row_size;
        const size_t destination_offset = (size_t) destination_segment*LLAMA_DSV4_COMP_SEGMENT_ROWS*row_size;
        ggml_backend_tensor_get(tensor, bytes.data(), source_offset, bytes.size());
        ggml_backend_tensor_set(tensor, bytes.data(), destination_offset, bytes.size());
    }
}

bool llama_kv_cache_dsv4_context::reserve_aggregate_pool(std::vector<llama_dsv4_comp_allocation> & cow_allocations) {
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
        if (allocation.cow && allocation.populated_rows > 0) {
            cow_allocations.push_back(allocation);
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
    const bool hp = dsv4_hp_enabled();
    const int64_t hp_t0 = hp ? ggml_time_us() : 0;
    int64_t hp_t = hp_t0;
    int64_t hp_adm = 0, hp_aggr = 0, hp_cow = 0, hp_coll = 0;
    int64_t hp_resv = 0, hp_comm = 0, hp_copy = 0;
    const auto hp_mark = [hp, &hp_t](int64_t & acc) {
        if (hp) {
            const int64_t t = ggml_time_us();
            acc += t - hp_t;
            hp_t = t;
        }
    };

    const auto admission_status = kv->consume_admission(ubatches, last_batch_quote, last_pressure_family_mask);
    if (admission_status == llama_kv_cache_dsv4::ADMISSION_COMMITTED) {
        batch_ranges_reserved = true;
        if (hp) {
            const int64_t t1 = ggml_time_us();
            fprintf(stderr, "HOSTPROF {\"op\":\"pref\",\"t\":%" PRId64 ",\"tot\":%" PRId64
                    ",\"path\":\"admitted\",\"nub\":%zu}\n", t1, t1 - hp_t0, ubatches.size());
        }
        return true;
    }
    if (admission_status == llama_kv_cache_dsv4::ADMISSION_ERROR) {
        return false;
    }

    hp_mark(hp_adm);

    std::vector<llama_dsv4_comp_allocation> cow_allocations;
    if (!reserve_aggregate_pool(cow_allocations)) {
        return false;
    }

    hp_mark(hp_aggr);

    std::vector<dsv4_sparse_range> ranges;
    uint64_t                       cow_source_ranges      = 0;
    uint64_t                       cow_destination_ranges = 0;
    const auto append_cow_segment = [&ranges](const llama_kv_cache * cache, uint32_t segment, uint32_t populated_rows,
                                              llama_dsv4_memory_family family, uint64_t & appended_ranges) {
        std::vector<int64_t> rows;
        rows.reserve(populated_rows);
        const int64_t first = (int64_t) segment * LLAMA_DSV4_COMP_SEGMENT_ROWS;
        for (uint32_t row = 0; row < populated_rows; ++row) {
            rows.push_back(first + row);
        }
        const size_t old_size = ranges.size();
        if (!dsv4_sparse_append_k_rows(cache, std::move(rows), family, ranges)) {
            return false;
        }
        appended_ranges += ranges.size() - old_size;
        return true;
    };
    const auto append_cow_cache = [&](const llama_kv_cache * cache, const llama_dsv4_comp_allocation & allocation,
                                      llama_dsv4_memory_family family) {
        return append_cow_segment(cache, allocation.source_segment, allocation.populated_rows, family,
                                  cow_source_ranges) &&
               append_cow_segment(cache, allocation.destination_segment, allocation.populated_rows, family,
                                  cow_destination_ranges);
    };

    for (const llama_dsv4_comp_allocation & allocation : cow_allocations) {
        const bool appended = allocation.family == llama_dsv4_comp_family::c4 ?
                                  append_cow_cache(kv->get_csa(), allocation, LLAMA_DSV4_MEMORY_CSA) &&
                                      append_cow_cache(kv->get_lid(), allocation, LLAMA_DSV4_MEMORY_LID) :
                                  append_cow_cache(kv->get_hca(), allocation, LLAMA_DSV4_MEMORY_HCA);
        if (!appended) {
            rollback_aggregate_pool();
            return false;
        }
    }

    hp_mark(hp_cow);

    if (!collect_batch_ranges(ranges)) {
        rollback_aggregate_pool();
        return false;
    }

    hp_mark(hp_coll);

    // Fast path. With no COW allocation pending and every target page already
    // mapped and exclusively owned, the reservation below would quote
    // required_pages = 0 and its commit would perform no mapping operation, no
    // generation bump and no net accounting change - while still paying the
    // planner's full scan of the pool's virtual and physical page tables,
    // twice (quote and commit re-quote), on every decode step. Skipping it is
    // observationally identical.
    //
    // The test seams are excluded: the page-delta audit expects a transaction
    // to observe, and an injected physical-pressure fault is consumed inside
    // the reservation.
    if (cow_allocations.empty() &&
            !dsv4_test_page_delta_audit_enabled.load(std::memory_order_relaxed) &&
            dsv4_test_pressure_count.load(std::memory_order_relaxed) == 0 &&
            dsv4_sparse_ranges_resident(ranges) > 0) {
        batch_ranges_reserved = true;
        if (hp) {
            hp_mark(hp_resv);
            const int64_t t1 = ggml_time_us();
            fprintf(stderr, "HOSTPROF {\"op\":\"pref\",\"t\":%" PRId64 ",\"tot\":%" PRId64
                    ",\"path\":\"resident\",\"nub\":%zu,\"nr\":%zu"
                    ",\"adm\":%" PRId64 ",\"aggr\":%" PRId64 ",\"cow\":%" PRId64
                    ",\"coll\":%" PRId64 ",\"resv\":%" PRId64 "}\n",
                    t1, t1 - hp_t0, ubatches.size(), ranges.size(),
                    hp_adm, hp_aggr, hp_cow, hp_coll, hp_resv);
        }
        return true;
    }

    dsv4_sparse_transaction reservation;
    const auto reserve_status = reservation.reserve_ranges(ranges, last_batch_quote);
    hp_mark(hp_resv);
    if (cow_source_ranges > 0) {
        dsv4_test_cow_source_ranges.fetch_add(cow_source_ranges, std::memory_order_relaxed);
        dsv4_test_cow_destination_ranges.fetch_add(cow_destination_ranges, std::memory_order_relaxed);
    }
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

    hp_mark(hp_comm);

    for (const llama_dsv4_comp_allocation & allocation : cow_allocations) {
        if (allocation.family == llama_dsv4_comp_family::c4) {
            dsv4_copy_pool_segment(kv->get_csa(), allocation.source_segment, allocation.destination_segment,
                                   allocation.populated_rows);
            dsv4_copy_pool_segment(kv->get_lid(), allocation.source_segment, allocation.destination_segment,
                                   allocation.populated_rows);
        } else {
            dsv4_copy_pool_segment(kv->get_hca(), allocation.source_segment, allocation.destination_segment,
                                   allocation.populated_rows);
        }
    }

    batch_ranges_reserved = true;

    if (hp) {
        hp_mark(hp_copy);
        const int64_t t1 = ggml_time_us();
        fprintf(stderr, "HOSTPROF {\"op\":\"pref\",\"t\":%" PRId64 ",\"tot\":%" PRId64
                ",\"path\":\"reserve\",\"nub\":%zu,\"nr\":%zu,\"nsr\":%zu,\"npool\":%zu"
                ",\"adm\":%" PRId64 ",\"aggr\":%" PRId64 ",\"cow\":%" PRId64 ",\"coll\":%" PRId64
                ",\"resv\":%" PRId64 ",\"comm\":%" PRId64 ",\"copy\":%" PRId64
                ",\"r_proc\":%" PRId64 ",\"r_usage\":%" PRId64 ",\"r_pools\":%" PRId64
                ",\"r_call\":%" PRId64 ",\"r_acct\":%" PRId64 "}\n",
                t1, t1 - hp_t0, ubatches.size(), ranges.size(),
                reservation.sparse_range_count(), reservation.quoted_pool_count(),
                hp_adm, hp_aggr, hp_cow, hp_coll, hp_resv, hp_comm, hp_copy,
                reservation.hp_proc, reservation.hp_usage, reservation.hp_pools,
                reservation.hp_call, reservation.hp_acct);
    }

    return true;
}

bool llama_kv_cache_dsv4_context::collect_batch_ranges(std::vector<dsv4_sparse_range> & ranges) {
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
            return false;
        }

        last_batch_quote.families[LLAMA_DSV4_MEMORY_RAW].logical_rows +=
                ctx_raw->get_n_backing_rows(i);
    }
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

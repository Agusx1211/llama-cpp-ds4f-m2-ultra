#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"
#include "models/models.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace allocation_probe {
thread_local bool   enabled = false;
thread_local bool   reject  = false;
thread_local size_t calls   = 0;
}

[[gnu::noinline]] void * operator new(std::size_t size) {
    if (allocation_probe::enabled) {
        ++allocation_probe::calls;
        if (allocation_probe::reject) {
            throw std::bad_alloc();
        }
    }
    if (void * value = std::malloc(size == 0 ? 1 : size)) {
        return value;
    }
    throw std::bad_alloc();
}

[[gnu::noinline]] void * operator new[](std::size_t size) {
    return ::operator new(size);
}

[[gnu::noinline]] void operator delete(void * value) noexcept {
    std::free(value);
}

[[gnu::noinline]] void operator delete[](void * value) noexcept {
    ::operator delete(value);
}

[[gnu::noinline]] void operator delete(void * value, std::size_t) noexcept {
    std::free(value);
}

[[gnu::noinline]] void operator delete[](void * value, std::size_t) noexcept {
    ::operator delete(value);
}

#undef assert
#define assert(expr) do {                                                        \
    if (!(expr)) {                                                               \
        std::cerr << "check failed at " << __FILE__ << ':' << __LINE__          \
                  << ": " #expr << '\n';                                       \
        std::abort();                                                            \
    }                                                                            \
} while (false)

namespace {

struct allocation_scope {
    explicit allocation_scope(bool reject = false) {
        if (allocation_probe::enabled) {
            std::abort();
        }
        allocation_probe::calls   = 0;
        allocation_probe::reject  = reject;
        allocation_probe::enabled = true;
    }

    ~allocation_scope() {
        allocation_probe::enabled = false;
        allocation_probe::reject  = false;
    }

    size_t finish() {
        const size_t result        = allocation_probe::calls;
        allocation_probe::enabled = false;
        allocation_probe::reject  = false;
        return result;
    }
};

using iswa_final_commit_fn = decltype(&llama_kv_cache_iswa::commit_resident_attach_final);
static_assert(!std::is_invocable_v<
              iswa_final_commit_fn,
              llama_kv_cache_iswa *,
              const llama_kv_iswa_resident_attach_quote &>);
static_assert(std::is_invocable_r_v<
              llama_kv_iswa_resident_status,
              iswa_final_commit_fn,
              llama_kv_cache_iswa *,
              const llama_kv_iswa_resident_attach_quote &,
              llama_kv_iswa_resident_final_step>);

static_assert(GGML_DSV4_SPARSE_OK == 0);
static_assert(GGML_DSV4_SPARSE_PRESSURE == 1);
static_assert(GGML_DSV4_SPARSE_STALE == 2);
static_assert(GGML_DSV4_SPARSE_INVALID == 3);
static_assert(GGML_DSV4_SPARSE_OOM == 4);
static_assert(GGML_DSV4_SPARSE_UNSUPPORTED == 5);

struct fake_move {
    struct entry {
        ggml_tensor * source;
        ggml_tensor * destination;
        std::vector<uint8_t> bytes;
    };
    std::vector<entry> tensors;
};

int quote_status = GGML_DSV4_SPARSE_OK;
int commit_status = GGML_DSV4_SPARSE_OK;
int live_quotes = 0;
int commit_calls = 0;

int fake_quote(
        ggml_tensor * const * sources,
        ggml_tensor * const * destinations,
        size_t count,
        void ** quote) {
    *quote = nullptr;
    if (quote_status != GGML_DSV4_SPARSE_OK) {
        return quote_status;
    }

    auto move = std::make_unique<fake_move>();
    move->tensors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        assert(sources[i] != nullptr);
        assert(destinations == nullptr ||
                ggml_nbytes(sources[i]) == ggml_nbytes(destinations[i]));
        fake_move::entry entry = { sources[i], destinations ? destinations[i] : nullptr, {} };
        entry.bytes.resize(ggml_nbytes(sources[i]));
        ggml_backend_tensor_get(sources[i], entry.bytes.data(), 0, entry.bytes.size());
        move->tensors.push_back(std::move(entry));
    }
    *quote = move.release();
    ++live_quotes;
    return GGML_DSV4_SPARSE_OK;
}

int fake_commit(void * raw) {
    ++commit_calls;
    if (commit_status != GGML_DSV4_SPARSE_OK) {
        return commit_status;
    }

    auto * move = static_cast<fake_move *>(raw);
    for (auto & entry : move->tensors) {
        if (entry.destination != nullptr) {
            ggml_backend_tensor_set(entry.destination, entry.bytes.data(), 0, entry.bytes.size());
        }
        std::memset(entry.bytes.data(), 0, entry.bytes.size());
        ggml_backend_tensor_set(entry.source, entry.bytes.data(), 0, entry.bytes.size());
    }
    return GGML_DSV4_SPARSE_OK;
}

void fake_free(void * raw) {
    delete static_cast<fake_move *>(raw);
    --live_quotes;
}

static_assert(std::is_same_v<decltype(&fake_quote), ggml_dsv4_sparse_move_quote_fn>);
static_assert(std::is_same_v<decltype(&fake_commit), ggml_dsv4_sparse_move_commit_fn>);
static_assert(std::is_same_v<decltype(&fake_free), ggml_dsv4_sparse_move_free_fn>);

struct backend_scope {
    backend_scope() {
        quote_status = GGML_DSV4_SPARSE_OK;
        commit_status = GGML_DSV4_SPARSE_OK;
        commit_calls = 0;
        llama_kv_iswa_set_resident_backend_override_for_test({ fake_quote, fake_commit, fake_free });
    }

    ~backend_scope() {
        llama_kv_iswa_set_resident_backend_override_for_test({});
        assert(live_quotes == 0);
    }
};

struct vector_writer : llama_io_write_i {
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t old_size = data.size();
        data.resize(old_size + size);
        ggml_backend_tensor_get(tensor, data.data() + old_size, offset, size);
    }

    size_t n_bytes() override { return data.size(); }

    std::vector<uint8_t> data;
};

struct vector_reader : llama_io_read_i {
    explicit vector_reader(std::vector<uint8_t> data, size_t fail_after = SIZE_MAX) :
        data(std::move(data)), fail_after(fail_after) {
    }

    void read(void * dst, size_t size) override {
        check(size);
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        check(size);
        ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        offset += size;
    }

    size_t n_bytes() override { return offset; }

    void check(size_t size) {
        if (offset > fail_after || size > fail_after - offset ||
                offset > data.size() || size > data.size() - offset) {
            throw std::runtime_error("injected state read failure");
        }
    }

    std::vector<uint8_t> data;
    size_t fail_after;
    size_t offset = 0;
};

llama_hparams make_hparams() {
    llama_hparams hparams = {};
    hparams.n_layer_all = 2;
    hparams.n_embd = 8;
    hparams.n_embd_head_k_full = 8;
    hparams.n_embd_head_v_full = 8;
    hparams.n_embd_head_k_swa = 8;
    hparams.n_embd_head_v_swa = 8;
    hparams.n_head_arr[0] = hparams.n_head_arr[1] = 1;
    hparams.n_head_kv_arr[0] = hparams.n_head_kv_arr[1] = 1;
    hparams.is_swa_impl[0] = 0;
    hparams.is_swa_impl[1] = 1;
    hparams.n_swa = 4;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_type = LLAMA_ROPE_TYPE_NONE;
    return hparams;
}

struct cache_fixture {
    llama_model_llama model;
    llama_hparams hparams;
    llama_kv_cache_iswa cache;

    explicit cache_fixture(uint32_t resident_slots = 2) :
        model(llama_model_default_params()),
        hparams(make_hparams()),
        cache(init_model(), hparams,
                GGML_TYPE_F32, GGML_TYPE_F32,
                false, false, true, false,
                8, 2, 1, 1,
                nullptr, nullptr, nullptr, nullptr, nullptr, resident_slots) {
    }

    const llama_model & init_model() {
        model.arch = LLM_ARCH_LLAMA;
        model.hparams = hparams;
        return model;
    }

    llama_kv_cache::slot_info slot(llama_seq_id seq, uint32_t index) {
        llama_kv_cache::slot_info result;
        result.s0 = result.s1 = (uint32_t) seq;
        result.strm = { seq };
        result.idxs = { { index } };
        return result;
    }

    static llama_ubatch token_ubatch(llama_seq_id seq, llama_pos pos) {
        llama_batch_allocr balloc(1);
        llama_ubatch ubatch = balloc.ubatch_reserve(1, 1);
        ubatch.data->seq_id_data = { seq };
        ubatch.data->seq_id[0] = ubatch.data->seq_id_data.data();
        ubatch.seq_id = ubatch.data->seq_id.data();
        ubatch.pos[0] = pos;
        ubatch.n_seq_id[0] = 1;
        ubatch.seq_id[0][0] = seq;
        ubatch.seq_id_unq[0] = seq;
        return ubatch;
    }

    void put_metadata(llama_seq_id seq, uint32_t base_index, uint32_t swa_index, llama_pos pos) {
        llama_ubatch ubatch = token_ubatch(seq, pos);
        cache.get_base()->apply_ubatch(slot(seq, base_index), ubatch);
        cache.get_swa ()->apply_ubatch(slot(seq, swa_index), ubatch);
    }

    uint32_t next_slot_index(llama_kv_cache * kv, llama_seq_id seq) {
        llama_ubatch ubatch = token_ubatch(seq, 0);
        const auto next = kv->find_slot(ubatch, false);
        assert(!next.empty());
        return next.idxs[0][0];
    }

    ggml_tensor * base_k() { return cache.get_base()->get_k_storage(0); }
    ggml_tensor * swa_k () { return cache.get_swa ()->get_k_storage(1); }

    static std::vector<uint8_t> stream_bytes(ggml_tensor * tensor, uint32_t stream) {
        const size_t size = tensor->nb[2];
        std::vector<uint8_t> result(size);
        ggml_backend_tensor_get(tensor, result.data(), stream*size, size);
        return result;
    }

    static void fill_stream(ggml_tensor * tensor, uint32_t stream, uint8_t value) {
        const size_t size = tensor->nb[2];
        std::vector<uint8_t> bytes(size, value);
        ggml_backend_tensor_set(tensor, bytes.data(), stream*size, size);
    }
};

llama_kv_iswa_resident_status quote_on_thread(llama_kv_cache_iswa & cache, llama_seq_id seq) {
    std::atomic<int> status { -1 };
    std::thread worker([&] {
        status.store((int) cache.quote_resident_detach(seq).status, std::memory_order_relaxed);
    });
    worker.join();
    return (llama_kv_iswa_resident_status) status.load(std::memory_order_relaxed);
}

template<class F>
void assert_throws(F && fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) {
        threw = true;
    }
    assert(threw);
}

llama_kv_iswa_resident_handle park_sequence(cache_fixture & f, llama_seq_id seq = 0) {
    quote_status = GGML_DSV4_SPARSE_OK;
    commit_status = GGML_DSV4_SPARSE_OK;
    auto result = f.cache.detach_resident(f.cache.quote_resident_detach(seq));
    assert(result.status == llama_kv_iswa_resident_status::ok);
    return result.resident;
}

void test_batch_graph_quiescence_lifecycle() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 7, 7, 42);

    auto quote = f.cache.quote_resident_detach(0);
    assert(quote.status == llama_kv_iswa_resident_status::ok);
    {
        auto prepare_lease = f.cache.acquire_resident_batch_lease();
        assert(quote_on_thread(f.cache, 0) == llama_kv_iswa_resident_status::not_quiescent);
        const int commits_before = commit_calls;
        std::atomic<int> detach_status { -1 };
        std::thread worker([&] {
            detach_status.store((int) f.cache.detach_resident(quote).status, std::memory_order_relaxed);
        });
        worker.join();
        assert((llama_kv_iswa_resident_status) detach_status.load(std::memory_order_relaxed) ==
                llama_kv_iswa_resident_status::stale_quote);
        assert(commit_calls == commits_before);
    }
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);

    f.cache.clear(false);
    auto ubatch = cache_fixture::token_ubatch(0, 9);
    auto base_slot = f.slot(0, 0);
    auto swa_slot = f.slot(0, 0);
    {
        auto batch = std::make_unique<llama_kv_cache_iswa_context>(
                &f.cache,
                llama_kv_cache::slot_info_vec_t { base_slot },
                llama_kv_cache::slot_info_vec_t { swa_slot },
                std::vector<llama_ubatch> { ubatch });
        assert(quote_on_thread(f.cache, 0) == llama_kv_iswa_resident_status::not_quiescent);
        assert(batch->apply());
        // Metadata is applied, but graph submission/rollback ownership is held
        // until the context itself reaches its terminal lifetime boundary.
        assert(quote_on_thread(f.cache, 0) == llama_kv_iswa_resident_status::not_quiescent);
    }
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);

    {
        llama_kv_cache_dsv4_raw_context raw_graph(&f.cache);
        assert(quote_on_thread(f.cache, 0) == llama_kv_iswa_resident_status::not_quiescent);
    }
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);

    // Full abandonment before apply is also a terminal boundary.
    {
        auto abandoned = std::make_unique<llama_kv_cache_iswa_context>(
                &f.cache,
                llama_kv_cache::slot_info_vec_t { f.slot(0, 1) },
                llama_kv_cache::slot_info_vec_t { f.slot(0, 1) },
                std::vector<llama_ubatch> { cache_fixture::token_ubatch(0, 10) });
        assert(quote_on_thread(f.cache, 0) == llama_kv_iswa_resident_status::not_quiescent);
    }
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);
}

void test_unique_update_ownership_and_retry() {
    backend_scope backend;

    {
        cache_fixture f;
        f.put_metadata(0, 0, 0, 3);
        f.cache.seq_cp(0, 1, -1, -1);
        auto first = f.cache.init_update(nullptr, false);
        auto second = f.cache.init_update(nullptr, false);
        assert(first->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        assert(second->get_status() == LLAMA_MEMORY_STATUS_FAILED_PREPARE);

        // Appended work is outside the first immutable prefix.
        f.cache.seq_cp(0, 1, -1, -1);
        assert(first->apply());
        llama_kv_cache_fail_stream_copy_after_for_test(0);
        assert(first->apply()); // consumed update is a no-op, not a replay
        llama_kv_cache_fail_stream_copy_after_for_test(-1);
        assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::not_quiescent);
        second.reset();
        first.reset();

        auto appended = f.cache.init_update(nullptr, false);
        assert(appended->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        assert(appended->apply());
        assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);
    }

    // Destroying the owner first permits a fresh retry even while the rejected
    // contender remains live; destroying in the opposite order changes none of
    // the queued work.
    {
        cache_fixture f;
        f.put_metadata(0, 0, 0, 3);
        f.cache.seq_cp(0, 1, -1, -1);
        auto owner = f.cache.init_update(nullptr, false);
        auto rejected = f.cache.init_update(nullptr, false);
        owner.reset();
        auto retry = f.cache.init_update(nullptr, false);
        assert(retry->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        rejected.reset();
        assert(retry->apply());
    }

    for (int fail_after : { 0, 1 }) {
        cache_fixture f;
        f.put_metadata(0, 0, 0, 3);
        f.fill_stream(f.base_k(), 0, 0x31);
        f.fill_stream(f.swa_k(), 0, 0x72);
        f.cache.seq_cp(0, 1, -1, -1);
        auto update = f.cache.init_update(nullptr, false);
        llama_kv_cache_fail_stream_copy_after_for_test(fail_after);
        assert_throws([&] { (void) update->apply(); });
        assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::not_quiescent);
        assert(update->apply());
        assert(f.stream_bytes(f.base_k(), 1) == f.stream_bytes(f.base_k(), 0));
        assert(f.stream_bytes(f.swa_k(), 1) == f.stream_bytes(f.swa_k(), 0));
        assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);
    }
}

void test_real_cycle_ring_end_and_clear_false() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 7, 7, 42); // both heads become the valid size() sentinel
    f.fill_stream(f.base_k(), 0, 0x31);
    f.fill_stream(f.swa_k(), 0, 0x72);
    const auto base_before = f.stream_bytes(f.base_k(), 0);
    const auto swa_before = f.stream_bytes(f.swa_k(), 0);

    auto quote = f.cache.quote_resident_detach(0);
    assert(quote.status == llama_kv_iswa_resident_status::ok);
    auto detached = f.cache.detach_resident(quote);
    assert(detached.status == llama_kv_iswa_resident_status::ok);
    assert(f.cache.get_base()->get_cells(0).get_used() == 0);
    assert(f.cache.get_swa()->get_cells(0).get_used() == 0);

    // Parked data and metadata must survive an execution-only clear.
    f.cache.clear(false);
    f.fill_stream(f.base_k(), 1, 0xa5); // mapped/non-zero but metadata-empty destination
    f.fill_stream(f.swa_k(), 1, 0xb6);
    assert(f.cache.attach_resident(detached.resident, 1) == llama_kv_iswa_resident_status::ok);
    assert(f.cache.get_base()->get_cells(1).seq_has(7, 1));
    assert(f.cache.get_swa()->get_cells(1).seq_has(7, 1));
    assert(f.stream_bytes(f.base_k(), 1) == base_before);
    assert(f.stream_bytes(f.swa_k(), 1) == swa_before);
    assert(f.next_slot_index(f.cache.get_base(), 1) == 0);
    assert(f.next_slot_index(f.cache.get_swa(), 1) == 0);
    assert(f.cache.attach_resident(detached.resident, 0) == llama_kv_iswa_resident_status::stale_handle);

    // A second complete cycle must return to the same data/refcount baseline.
    auto quote2 = f.cache.quote_resident_detach(1);
    assert(quote2.status == llama_kv_iswa_resident_status::ok);
    auto detached2 = f.cache.detach_resident(quote2);
    assert(detached2.status == llama_kv_iswa_resident_status::ok);
    assert(f.cache.attach_resident(detached2.resident, 0) == llama_kv_iswa_resident_status::ok);
    assert(f.stream_bytes(f.base_k(), 0) == base_before);
    assert(f.stream_bytes(f.swa_k(), 0) == swa_before);
    assert(f.next_slot_index(f.cache.get_base(), 0) == 0);
    assert(f.next_slot_index(f.cache.get_swa(), 0) == 0);
}

void test_state_io_rejects_parked_handles_before_io() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 3, 3, 12);
    f.fill_stream(f.base_k(), 0, 0x3c);
    f.fill_stream(f.swa_k(), 0, 0x7d);
    const auto base_before = f.stream_bytes(f.base_k(), 0);
    const auto swa_before = f.stream_bytes(f.swa_k(), 0);

    vector_writer valid;
    f.cache.state_write(valid, -1, 0);
    assert(!valid.data.empty());

    const auto handle = park_sequence(f);
    vector_writer rejected_write;
    assert_throws([&] { f.cache.state_write(rejected_write, -1, 0); });
    assert(rejected_write.n_bytes() == 0);

    vector_writer rejected_base_write;
    assert_throws([&] { f.cache.get_base()->state_write(rejected_base_write, -1, 0); });
    assert(rejected_base_write.n_bytes() == 0);

    std::vector<std::vector<uint8_t>> inputs;
    inputs.push_back(valid.data); // otherwise successful
    inputs.push_back(std::vector<uint8_t>(valid.data.begin(), valid.data.begin() + valid.data.size()/2)); // truncated
    inputs.push_back(valid.data); // corrupt
    inputs.back()[0] ^= 0xff;
    for (auto & input : inputs) {
        vector_reader reader(std::move(input));
        assert_throws([&] { f.cache.state_read(reader, -1, 0); });
        assert(reader.n_bytes() == 0);
    }

    // Failure injection points that would land in the base plane and between
    // planes are also rejected before the first byte or tensor mutation.
    for (size_t fail_after : { size_t(0), valid.data.size()/2 }) {
        vector_reader reader(valid.data, fail_after);
        assert_throws([&] { f.cache.state_read(reader, -1, 0); });
        assert(reader.n_bytes() == 0);
    }
    vector_reader direct_base(valid.data, 0);
    assert_throws([&] { f.cache.get_base()->state_read(direct_base, -1, 0); });
    assert(direct_base.n_bytes() == 0);
    vector_reader direct_swa(valid.data, valid.data.size()/2);
    assert_throws([&] { f.cache.get_swa()->state_read(direct_swa, -1, 0); });
    assert(direct_swa.n_bytes() == 0);

    assert_throws([&] { f.cache.get_base()->clear(true); });
    assert_throws([&] { f.cache.get_swa()->clear(true); });
    assert(f.cache.attach_resident(handle, 0) == llama_kv_iswa_resident_status::ok);
    assert(f.cache.get_base()->get_cells(0).seq_has(3, 0));
    assert(f.cache.get_swa()->get_cells(0).seq_has(3, 0));
    assert(f.stream_bytes(f.base_k(), 0) == base_before);
    assert(f.stream_bytes(f.swa_k(), 0) == swa_before);
}

void test_quote_staleness_and_no_backend_mutation() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 7, 7, 42);
    f.fill_stream(f.base_k(), 0, 0x19);
    const auto before = f.stream_bytes(f.base_k(), 0);

    auto quote = f.cache.quote_resident_detach(0);
    assert(quote.status == llama_kv_iswa_resident_status::ok);
    const int commits_before = commit_calls;
    assert(f.cache.seq_rm(0, 999, 1000)); // no cells changed, but the mutation boundary advanced
    assert(f.cache.detach_resident(quote).status == llama_kv_iswa_resident_status::stale_quote);
    assert(commit_calls == commits_before);
    assert(f.stream_bytes(f.base_k(), 0) == before);

    auto token_quote = f.cache.quote_resident_detach(0);
    assert(token_quote.status == llama_kv_iswa_resident_status::ok);
    f.put_metadata(0, 6, 6, 43);
    assert(f.cache.detach_resident(token_quote).status == llama_kv_iswa_resident_status::stale_quote);

    auto clear_quote = f.cache.quote_resident_detach(0);
    assert(clear_quote.status == llama_kv_iswa_resident_status::ok);
    f.cache.clear(false);
    assert(f.cache.detach_resident(clear_quote).status == llama_kv_iswa_resident_status::stale_quote);
}

void test_pending_copy_fail_closed() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 0, 0, 3);
    f.cache.seq_cp(0, 1, -1, -1);
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::not_quiescent);
    assert(f.cache.quote_resident_detach(1).status == llama_kv_iswa_resident_status::not_quiescent);

    {
        auto update = f.cache.init_update(nullptr, false);
        assert(update->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
        assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::not_quiescent);
    }
    // Dropping an unapplied update context re-exposes, rather than loses, work.
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::not_quiescent);

    auto update = f.cache.init_update(nullptr, false);
    assert(update->apply());
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::ok);
    assert(f.cache.quote_resident_detach(1).status == llama_kv_iswa_resident_status::ok);
}

void test_backend_failures_are_atomic_and_retryable() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 0, 0, 5);
    f.fill_stream(f.base_k(), 0, 0x44);
    const auto source_before = f.stream_bytes(f.base_k(), 0);

    quote_status = GGML_DSV4_SPARSE_PRESSURE;
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::resource_exhausted);
    assert(live_quotes == 0);

    quote_status = GGML_DSV4_SPARSE_OK;
    llama_kv_iswa_fail_next_resident_allocation_for_test();
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::resource_exhausted);
    assert(live_quotes == 0);

    auto quote = f.cache.quote_resident_detach(0);
    assert(quote.status == llama_kv_iswa_resident_status::ok);
    commit_status = GGML_DSV4_SPARSE_OOM;
    assert(f.cache.detach_resident(quote).status == llama_kv_iswa_resident_status::resource_exhausted);
    assert(f.stream_bytes(f.base_k(), 0) == source_before);
    assert(f.cache.get_base()->get_cells(0).get_used() == 1);

    commit_status = GGML_DSV4_SPARSE_OK;
    auto detached = f.cache.detach_resident(quote);
    assert(detached.status == llama_kv_iswa_resident_status::ok);
    assert(f.cache.release_resident(detached.resident) == llama_kv_iswa_resident_status::ok);
}

void test_prepared_attach_final_step_contract() {
    backend_scope backend;
    cache_fixture f;
    f.put_metadata(0, 7, 7, 42);
    f.fill_stream(f.base_k(), 0, 0x29);
    f.fill_stream(f.swa_k(), 0, 0x6a);
    const auto base_before = f.stream_bytes(f.base_k(), 0);
    const auto swa_before  = f.stream_bytes(f.swa_k(), 0);
    const auto handle      = park_sequence(f);
    assert(f.cache.has_resident_handles());

    auto cancelled = f.cache.quote_resident_attach(handle, 1);
    assert(cancelled.status == llama_kv_iswa_resident_status::ok);
    const int commits_before_cancel = commit_calls;
    llama_kv_iswa_resident_status cancel_status;
    size_t                        cancel_allocations = 0;
    {
        allocation_scope allocations;
        cancel_status      = f.cache.rollback_resident_attach(cancelled);
        cancel_allocations = allocations.finish();
    }
    assert(cancel_status == llama_kv_iswa_resident_status::ok);
    assert(cancel_allocations == 0);
    assert(f.cache.rollback_resident_attach(cancelled) == llama_kv_iswa_resident_status::ok);
    assert(f.cache.commit_resident_attach_final(
                   cancelled, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::stale_quote);
    assert(commit_calls == commits_before_cancel);
    assert(f.cache.has_resident_handles());

    llama_kv_iswa_resident_attach_quote allocation_failure;
    size_t                              failed_quote_allocations = 0;
    {
        allocation_scope allocations(true);
        allocation_failure       = f.cache.quote_resident_attach(handle, 1);
        failed_quote_allocations = allocations.finish();
    }
    assert(failed_quote_allocations == 1);
    assert(allocation_failure.status == llama_kv_iswa_resident_status::resource_exhausted);
    assert(f.cache.has_resident_handles());
    assert(f.cache.get_base()->get_cells(1).get_used() == 0);
    assert(f.cache.get_swa()->get_cells(1).get_used() == 0);

    llama_kv_iswa_fail_next_resident_allocation_for_test();
    const auto post_backend_allocation_failure = f.cache.quote_resident_attach(handle, 1);
    assert(post_backend_allocation_failure.status == llama_kv_iswa_resident_status::resource_exhausted);
    assert(f.cache.has_resident_handles());

    auto stale = f.cache.quote_resident_attach(handle, 1);
    assert(stale.status == llama_kv_iswa_resident_status::ok);
    const int commits_before_stale = commit_calls;
    assert(f.cache.seq_rm(1, 999, 1000));
    assert(f.cache.commit_resident_attach_final(
                   stale, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::stale_quote);
    assert(commit_calls == commits_before_stale);
    assert(f.cache.rollback_resident_attach(stale) == llama_kv_iswa_resident_status::ok);

    f.put_metadata(1, 0, 0, 8);
    assert(f.cache.quote_resident_attach(handle, 1).status == llama_kv_iswa_resident_status::slot_occupied);
    f.cache.clear(false);

    cache_fixture foreign;
    auto          prepared = f.cache.quote_resident_attach(handle, 1);
    assert(prepared.status == llama_kv_iswa_resident_status::ok);
    assert(foreign.cache.commit_resident_attach_final(
                   prepared, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::stale_quote);
    assert(foreign.cache.rollback_resident_attach(prepared) == llama_kv_iswa_resident_status::stale_quote);

    commit_status = GGML_DSV4_SPARSE_OOM;
    const int commits_before_failure = commit_calls;
    assert(f.cache.commit_resident_attach_final(
                   prepared, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::resource_exhausted);
    assert(commit_calls == commits_before_failure + 1);
    assert(f.cache.has_resident_handles());
    assert(f.cache.get_base()->get_cells(1).get_used() == 0);
    assert(f.cache.get_swa()->get_cells(1).get_used() == 0);
    assert(f.cache.rollback_resident_attach(prepared) == llama_kv_iswa_resident_status::ok);
    assert(f.cache.rollback_resident_attach(prepared) == llama_kv_iswa_resident_status::ok);
    commit_status = GGML_DSV4_SPARSE_OK;
    assert(f.cache.commit_resident_attach_final(
                   prepared, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::stale_quote);

    auto retry = f.cache.quote_resident_attach(handle, 1);
    assert(retry.status == llama_kv_iswa_resident_status::ok);
    retry.execution_id       = 0;
    retry.resident.pool_id   = UINT64_MAX;
    retry.resident.id        = UINT64_MAX;
    retry.resident.generation = UINT64_MAX;
    llama_kv_iswa_resident_status final_status;
    size_t                        final_allocations = 0;
    {
        allocation_scope allocations;
        final_status = f.cache.commit_resident_attach_final(
                retry, llama_kv_iswa_resident_final_step::confirmed);
        final_allocations = allocations.finish();
    }
    assert(final_status == llama_kv_iswa_resident_status::ok);
    assert(final_allocations == 0);
    assert(!f.cache.has_resident_handles());
    assert(f.cache.get_base()->get_cells(1).seq_has(7, 1));
    assert(f.cache.get_swa()->get_cells(1).seq_has(7, 1));
    assert(f.stream_bytes(f.base_k(), 1) == base_before);
    assert(f.stream_bytes(f.swa_k(), 1) == swa_before);

    const int commits_after_success = commit_calls;
    assert(f.cache.rollback_resident_attach(retry) == llama_kv_iswa_resident_status::stale_quote);
    assert(f.stream_bytes(f.base_k(), 1) == base_before);
    assert(f.stream_bytes(f.swa_k(), 1) == swa_before);
    assert(f.cache.commit_resident_attach_final(
                   retry, llama_kv_iswa_resident_final_step::confirmed) ==
           llama_kv_iswa_resident_status::stale_quote);
    assert(commit_calls == commits_after_success);
    assert(f.cache.quote_resident_attach(handle, 0).status == llama_kv_iswa_resident_status::stale_handle);
}

struct backend_status_case {
    int backend;
    llama_kv_iswa_resident_status detach_quote;
    llama_kv_iswa_resident_status detach_commit;
    llama_kv_iswa_resident_status later_phase;
};

void assert_seq_present(const cache_fixture & f, llama_seq_id seq) {
    assert(f.cache.get_base()->get_cells(seq).seq_has(0, seq));
    assert(f.cache.get_swa ()->get_cells(seq).seq_has(0, seq));
}

void test_backend_status_mapping_by_transaction_phase() {
    backend_scope backend;
    const backend_status_case cases[] = {
        { GGML_DSV4_SPARSE_OK,
          llama_kv_iswa_resident_status::ok,
          llama_kv_iswa_resident_status::ok,
          llama_kv_iswa_resident_status::ok },
        { GGML_DSV4_SPARSE_PRESSURE,
          llama_kv_iswa_resident_status::resource_exhausted,
          llama_kv_iswa_resident_status::resource_exhausted,
          llama_kv_iswa_resident_status::resource_exhausted },
        { GGML_DSV4_SPARSE_STALE,
          llama_kv_iswa_resident_status::stale_quote,
          llama_kv_iswa_resident_status::stale_quote,
          llama_kv_iswa_resident_status::backend_error },
        { GGML_DSV4_SPARSE_INVALID,
          llama_kv_iswa_resident_status::unsupported_layout,
          llama_kv_iswa_resident_status::backend_error,
          llama_kv_iswa_resident_status::backend_error },
        { GGML_DSV4_SPARSE_OOM,
          llama_kv_iswa_resident_status::resource_exhausted,
          llama_kv_iswa_resident_status::resource_exhausted,
          llama_kv_iswa_resident_status::resource_exhausted },
        { GGML_DSV4_SPARSE_UNSUPPORTED,
          llama_kv_iswa_resident_status::unsupported_layout,
          llama_kv_iswa_resident_status::backend_error,
          llama_kv_iswa_resident_status::backend_error },
        { 77,
          llama_kv_iswa_resident_status::backend_error,
          llama_kv_iswa_resident_status::backend_error,
          llama_kv_iswa_resident_status::backend_error },
    };

    for (const auto & status : cases) {
        {
            cache_fixture f;
            f.put_metadata(0, 0, 0, 5);
            quote_status = status.backend;
            auto quote = f.cache.quote_resident_detach(0);
            assert(quote.status == status.detach_quote);
            assert_seq_present(f, 0);
        }

        {
            cache_fixture f;
            f.put_metadata(0, 0, 0, 5);
            quote_status = GGML_DSV4_SPARSE_OK;
            commit_status = GGML_DSV4_SPARSE_OK;
            auto quote = f.cache.quote_resident_detach(0);
            assert(quote.status == llama_kv_iswa_resident_status::ok);
            commit_status = status.backend;
            auto result = f.cache.detach_resident(quote);
            assert(result.status == status.detach_commit);
            if (status.backend == GGML_DSV4_SPARSE_OK) {
                assert(f.cache.release_resident(result.resident) == llama_kv_iswa_resident_status::ok);
            } else {
                assert_seq_present(f, 0);
                commit_status = GGML_DSV4_SPARSE_OK;
                result = f.cache.detach_resident(quote);
                assert(result.status == llama_kv_iswa_resident_status::ok);
                assert(f.cache.release_resident(result.resident) == llama_kv_iswa_resident_status::ok);
            }
        }

        for (bool fail_commit : { false, true }) {
            cache_fixture f;
            f.put_metadata(0, 0, 0, 5);
            f.fill_stream(f.base_k(), 0, 0x42);
            f.fill_stream(f.swa_k(), 0, 0x81);
            const auto base_before = f.stream_bytes(f.base_k(), 0);
            const auto swa_before = f.stream_bytes(f.swa_k(), 0);
            const auto handle = park_sequence(f);
            if (fail_commit) {
                commit_status = status.backend;
            } else {
                quote_status = status.backend;
            }
            const auto result = f.cache.attach_resident(handle, 1);
            assert(result == status.later_phase);
            if (status.backend == GGML_DSV4_SPARSE_OK) {
                assert(f.stream_bytes(f.base_k(), 1) == base_before);
                assert(f.stream_bytes(f.swa_k(), 1) == swa_before);
            } else {
                quote_status = GGML_DSV4_SPARSE_OK;
                commit_status = GGML_DSV4_SPARSE_OK;
                assert(f.cache.attach_resident(handle, 1) == llama_kv_iswa_resident_status::ok);
                assert(f.stream_bytes(f.base_k(), 1) == base_before);
                assert(f.stream_bytes(f.swa_k(), 1) == swa_before);
            }
        }

        for (bool fail_commit : { false, true }) {
            cache_fixture f;
            f.put_metadata(0, 0, 0, 5);
            const auto handle = park_sequence(f);
            if (fail_commit) {
                commit_status = status.backend;
            } else {
                quote_status = status.backend;
            }
            const auto result = f.cache.release_resident(handle);
            assert(result == status.later_phase);
            if (status.backend != GGML_DSV4_SPARSE_OK) {
                quote_status = GGML_DSV4_SPARSE_OK;
                commit_status = GGML_DSV4_SPARSE_OK;
                assert(f.cache.release_resident(handle) == llama_kv_iswa_resident_status::ok);
            }
        }
    }
}

void test_cross_pool_handle_rejected() {
    backend_scope backend;
    cache_fixture a;
    cache_fixture b;
    a.put_metadata(0, 0, 0, 1);
    auto detached = a.cache.detach_resident(a.cache.quote_resident_detach(0));
    assert(detached.status == llama_kv_iswa_resident_status::ok);
    assert(b.cache.attach_resident(detached.resident, 0) == llama_kv_iswa_resident_status::stale_handle);
    assert(a.cache.release_resident(detached.resident) == llama_kv_iswa_resident_status::ok);
}

void test_disabled_path_does_not_offer_residency() {
    backend_scope backend;
    cache_fixture f(0);
    f.put_metadata(0, 0, 0, 1);
    assert(f.cache.quote_resident_detach(0).status == llama_kv_iswa_resident_status::unsupported_layout);
    assert(commit_calls == 0);
    assert(f.cache.seq_rm(0, -1, -1));
    assert(f.cache.get_base()->get_cells(0).get_used() == 0);
    assert(f.cache.get_swa()->get_cells(0).get_used() == 0);
}

} // namespace

int main() {
    test_batch_graph_quiescence_lifecycle();
    test_unique_update_ownership_and_retry();
    test_real_cycle_ring_end_and_clear_false();
    test_state_io_rejects_parked_handles_before_io();
    test_quote_staleness_and_no_backend_mutation();
    test_pending_copy_fail_closed();
    test_backend_failures_are_atomic_and_retryable();
    test_prepared_attach_final_step_contract();
    test_backend_status_mapping_by_transaction_phase();
    test_cross_pool_handle_rejected();
    test_disabled_path_does_not_offer_residency();
    assert(std::strcmp(llama_kv_iswa_resident_status_name(
            llama_kv_iswa_resident_status::resource_exhausted), "resource_exhausted") == 0);
    std::cout << "ISWA resident transaction tests passed\n";
    return 0;
}

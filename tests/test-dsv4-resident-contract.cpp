#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"
#include "models/models.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace allocation_probe {
thread_local bool enabled = false;
thread_local bool reject  = false;
thread_local size_t calls = 0;
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

[[gnu::noinline]] void * operator new[](std::size_t size) { return ::operator new(size); }
[[gnu::noinline]] void operator delete(void * value) noexcept { std::free(value); }
[[gnu::noinline]] void operator delete[](void * value) noexcept { ::operator delete(value); }
[[gnu::noinline]] void operator delete(void * value, std::size_t) noexcept { std::free(value); }
[[gnu::noinline]] void operator delete[](void * value, std::size_t) noexcept { ::operator delete(value); }

namespace {

void expect(bool condition, const std::string & message);

struct allocation_scope {
    explicit allocation_scope(bool reject = false) {
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

struct fake_move {
    struct entry {
        ggml_tensor * source;
        ggml_tensor * destination;
        std::vector<uint8_t> bytes;
    };
    std::vector<entry> tensors;
};

int fake_quote_status  = GGML_DSV4_SPARSE_OK;
int fake_commit_status = GGML_DSV4_SPARSE_OK;
int fake_live_quotes   = 0;
int fake_commit_calls  = 0;
bool fake_quote_throws = false;
bool fake_commit_throws = false;

struct fake_commit_pause {
    std::promise<void>       entered;
    std::shared_future<void> resume;
};

thread_local fake_commit_pause * fake_commit_pause_override = nullptr;

int fake_quote(
        ggml_tensor * const * sources,
        ggml_tensor * const * destinations,
        size_t count,
        void ** quote) {
    *quote = nullptr;
    if (fake_quote_throws) {
        throw std::runtime_error("injected raw resident quote exception");
    }
    if (fake_quote_status != GGML_DSV4_SPARSE_OK) {
        return fake_quote_status;
    }
    auto move = std::make_unique<fake_move>();
    move->tensors.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        fake_move::entry entry = { sources[i], destinations ? destinations[i] : nullptr, {} };
        entry.bytes.resize(ggml_nbytes(sources[i]));
        ggml_backend_tensor_get(sources[i], entry.bytes.data(), 0, entry.bytes.size());
        move->tensors.push_back(std::move(entry));
    }
    *quote = move.release();
    ++fake_live_quotes;
    return GGML_DSV4_SPARSE_OK;
}

int fake_commit(void * raw) {
    ++fake_commit_calls;
    if (fake_commit_pause_override != nullptr) {
        auto * pause               = fake_commit_pause_override;
        fake_commit_pause_override = nullptr;
        pause->entered.set_value();
        pause->resume.wait();
    }
    if (fake_commit_throws) {
        throw std::runtime_error("injected raw resident commit exception");
    }
    if (fake_commit_status != GGML_DSV4_SPARSE_OK) {
        return fake_commit_status;
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
    --fake_live_quotes;
}

struct backend_scope {
    backend_scope() {
        fake_quote_status  = GGML_DSV4_SPARSE_OK;
        fake_commit_status = GGML_DSV4_SPARSE_OK;
        fake_commit_calls  = 0;
        fake_quote_throws  = false;
        fake_commit_throws = false;
        llama_kv_iswa_set_resident_backend_override_for_test({ fake_quote, fake_commit, fake_free });
    }
    ~backend_scope() {
        llama_kv_iswa_set_resident_backend_override_for_test({});
        if (fake_live_quotes != 0) {
            std::abort();
        }
    }
};

struct vector_writer : llama_io_write_i {
    void write(const void * src, size_t size) override {
        const auto * bytes = static_cast<const uint8_t *>(src);
        data.insert(data.end(), bytes, bytes + size);
    }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t old = data.size();
        data.resize(old + size);
        ggml_backend_tensor_get(tensor, data.data() + old, offset, size);
    }
    size_t n_bytes() override { return data.size(); }
    std::vector<uint8_t> data;
};

struct vector_reader : llama_io_read_i {
    explicit vector_reader(const std::vector<uint8_t> & data) : data(data) {}

    void read(void * dst, size_t size) override {
        expect(offset <= data.size() && size <= data.size() - offset, "state reader overflow");
        std::memcpy(dst, data.data() + offset, size);
        offset += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t tensor_offset, size_t size) override {
        expect(offset <= data.size() && size <= data.size() - offset, "tensor reader overflow");
        ggml_backend_tensor_set(tensor, data.data() + offset, tensor_offset, size);
        offset += size;
    }

    size_t n_bytes() override { return offset; }

    const std::vector<uint8_t> & data;
    size_t                       offset = 0;
};

struct resident_lock_probe {
    std::promise<void> before_lock;
    std::promise<void> lock_contended;
    std::promise<void> lock_acquired;
    std::exception_ptr error;
};

void resident_lock_probe_hook(llama_kv_iswa_resident_lock_phase phase, void * context) {
    auto & probe = *static_cast<resident_lock_probe *>(context);
    if (phase == llama_kv_iswa_resident_lock_phase::before_lock) {
        probe.before_lock.set_value();
    } else if (phase == llama_kv_iswa_resident_lock_phase::lock_contended) {
        probe.lock_contended.set_value();
    } else {
        probe.lock_acquired.set_value();
    }
}

template <typename Action> std::thread start_resident_lock_probe(resident_lock_probe & probe, Action action) {
    return std::thread([&probe, action = std::move(action)]() mutable {
        try {
            llama_kv_iswa_set_resident_lock_hook_for_test(resident_lock_probe_hook, &probe);
            action();
        } catch (...) {
            probe.error = std::current_exception();
        }
    });
}

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

struct composite_fixture {
    llama_model_llama model;
    llama_hparams hparams;
    llama_kv_cache_iswa raw;
    llama_dsv4_comp_pool compressed;
    llama_dsv4_comp_state csa_execution;
    llama_dsv4_comp_state hca_execution;
    llama_dsv4_comp_state lid_execution;
    llama_dsv4_comp_state csa_resident;
    llama_dsv4_comp_state hca_resident;
    llama_dsv4_comp_state lid_resident;
    std::vector<uint32_t> rollback_index;
    uint32_t active_rollback_depth = 2;
    llama_dsv4_composite_resident composite;
    llama_dsv4_comp_handle_id roots[2] = { 0, 0 };

    explicit composite_fixture(uint32_t capacity = 1) :
        model(llama_model_default_params()),
        hparams(make_hparams()),
        raw(init_model(), hparams, GGML_TYPE_F32, GGML_TYPE_F32,
                false, false, true, false, 8, 2, 1, 1,
                nullptr, nullptr, nullptr, nullptr, nullptr, 2),
        compressed({ 16, 8 }),
        csa_execution(init_model(), false, false, 2, 4, 8, 8, 2, "test_csa", nullptr),
        hca_execution(init_model(), false, false, 2, 128, 4, 8, 2, "test_hca", nullptr),
        lid_execution(init_model(), false, false, 2, 4, 8, 8, 2, "test_lid", nullptr),
        csa_resident(init_model(), false, false, capacity, 4, 8, 8, 2, "resident_csa", nullptr),
        hca_resident(init_model(), false, false, capacity, 128, 4, 8, 2, "resident_hca", nullptr),
        lid_resident(init_model(), false, false, capacity, 4, 8, 8, 2, "resident_lid", nullptr),
        rollback_index(2, 0),
        composite(raw, compressed, csa_execution, hca_execution, lid_execution,
                csa_resident, hca_resident, lid_resident, rollback_index,
                active_rollback_depth, 2, capacity) {
        for (uint32_t seq = 0; seq < 2; ++seq) {
            const auto created = compressed.create_handle();
            expect(created.status == llama_dsv4_comp_status::ok, "create aggregate root");
            roots[seq] = created.handle;
            expect(compressed.bind(seq, roots[seq]) == llama_dsv4_comp_status::ok, "bind aggregate root");
        }
    }

    const llama_model & init_model() {
        model.arch = LLM_ARCH_LLAMA;
        model.hparams = hparams;
        return model;
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

    static llama_kv_cache::slot_info slot(llama_seq_id seq, uint32_t index) {
        llama_kv_cache::slot_info result;
        result.s0 = result.s1 = (uint32_t) seq;
        result.strm = { seq };
        result.idxs = { { index } };
        return result;
    }

    void put_raw(llama_seq_id seq, uint32_t index, llama_pos pos) {
        auto ubatch = token_ubatch(seq, pos);
        raw.get_base()->apply_ubatch(slot(seq, index), ubatch);
        raw.get_swa()->apply_ubatch(slot(seq, index), ubatch);
    }

    static void fill_stream(ggml_tensor * tensor, uint32_t stream, uint8_t value) {
        const size_t size = tensor->nb[2];
        std::vector<uint8_t> bytes(size, value);
        ggml_backend_tensor_set(tensor, bytes.data(), stream*size, size);
    }

    static std::vector<uint8_t> stream_bytes(ggml_tensor * tensor, uint32_t stream) {
        const size_t size = tensor->nb[2];
        std::vector<uint8_t> bytes(size);
        ggml_backend_tensor_get(tensor, bytes.data(), stream*size, size);
        return bytes;
    }

    void seed_compressed_root(llama_dsv4_comp_handle_id root = 0) {
        if (root == 0) {
            root = roots[0];
        }
        llama_dsv4_comp_batch_plan batch;
        batch.graph_execution_ids = { root == roots[1] ? 1u : 0u };
        batch.changes = {
            { root, llama_dsv4_comp_family::c4, 65, {} },
            { root, llama_dsv4_comp_family::hca, 129, {} },
        };
        const auto quote = compressed.quote_batch(batch);
        expect(quote.status == llama_dsv4_comp_status::ok, "quote seeded aggregate root");
        const auto reservation = compressed.try_reserve(quote);
        expect(reservation.status == llama_dsv4_comp_status::ok, "reserve seeded aggregate root");
        expect(compressed.commit(reservation.ticket) == llama_dsv4_comp_status::ok, "commit seeded aggregate root");
    }

    std::array<llama_dsv4_comp_state *, 3> execution_states() {
        return { &csa_execution, &hca_execution, &lid_execution };
    }

    std::array<llama_dsv4_comp_state *, 3> resident_states() {
        return { &csa_resident, &hca_resident, &lid_resident };
    }
};

void expect(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool has(uint32_t mask, llama_dsv4_resident_component component) {
    return (mask & component) != 0;
}

void test_current_layout_fails_closed() {
    const auto affine =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::single_context }, 4, 3, false);
    expect(affine.status == llama_dsv4_resident_status::unsupported_components,
           "affine whole-sequence quote unexpectedly succeeded");
    expect(affine.rollback_index == 3 && affine.detachable_components == LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX,
           "affine capability quote lost rollback metadata");
    expect(has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_RAW_SWA) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_COMPRESSED) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_CSA_STATE) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_HCA_STATE) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_LID_STATE),
           "affine quote omitted a fixed sequence component");

    const auto aggregate =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::single_context }, 4, 3, true);
    expect(aggregate.status == llama_dsv4_resident_status::unsupported_components,
           "aggregate compressed ownership made the whole-sequence quote succeed");
    expect(has(aggregate.detachable_components, LLAMA_DSV4_RESIDENT_COMPRESSED) &&
               !has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_COMPRESSED),
           "aggregate quote did not expose compressed capability");
    expect(has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_RAW_SWA) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_CSA_STATE) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_HCA_STATE) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_LID_STATE),
           "aggregate quote hid whole-sequence blockers");

    const auto paired =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::target_draft_pair }, 4, 3, true);
    expect(paired.status == llama_dsv4_resident_status::unsupported_components &&
               has(paired.required_components, LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT) &&
               has(paired.unsupported_components, LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT),
           "single backend context claimed target/draft atomic ownership");

    const auto invalid =
        llama_dsv4_quote_resident_detach_layout({ 4, llama_dsv4_resident_scope::single_context }, 4, 99, true);
    expect(invalid.status == llama_dsv4_resident_status::invalid_sequence && invalid.detachable_components == 0,
           "invalid sequence exposed detachable ownership");

    const auto invalid_scope = llama_dsv4_quote_resident_detach_layout(
        { 2, static_cast<llama_dsv4_resident_scope>(UINT8_MAX) }, 4, 3, true);
    expect(invalid_scope.status == llama_dsv4_resident_status::invalid_scope &&
               invalid_scope.detachable_components == 0 &&
               invalid_scope.unsupported_components == invalid_scope.required_components,
           "invalid scope exposed detachable ownership");
}

void fill_state_sequence(llama_dsv4_comp_state & state, llama_seq_id seq, uint8_t family_seed) {
    for (uint32_t il = 0; il < 2; ++il) {
        ggml_init_params params = {
            /*.mem_size   =*/ 4*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        expect(ctx != nullptr, "create state view context");
        ggml_tensor * kv    = state.get_kv_all(ctx, (int32_t) il);
        ggml_tensor * score = state.get_score_all(ctx, (int32_t) il);
        const size_t plane_size = (size_t) state.get_state_size()*kv->nb[1];
        for (uint32_t depth = 0; depth <= state.get_n_rs_seq(); ++depth) {
            const uint32_t stream = depth*state.get_n_stream() + (uint32_t) seq;
            std::vector<uint8_t> kv_bytes(plane_size, (uint8_t) (family_seed + 11*depth + 3*il));
            std::vector<uint8_t> score_bytes(plane_size, (uint8_t) (family_seed + 71 + 7*depth + 5*il));
            ggml_backend_tensor_set(kv, kv_bytes.data(), stream*plane_size, plane_size);
            ggml_backend_tensor_set(score, score_bytes.data(), stream*plane_size, plane_size);
        }
        ggml_free(ctx);
    }
}

std::vector<uint8_t> state_snapshot(
        llama_dsv4_comp_state & state,
        llama_seq_id seq,
        uint32_t depth) {
    std::vector<uint32_t> selected(state.get_n_stream(), 0);
    selected[(uint32_t) seq] = depth;
    vector_writer writer;
    state.state_write(writer, seq, 0, selected);
    return writer.data;
}

using state_snapshots = std::array<std::array<std::vector<uint8_t>, 3>, 3>;

state_snapshots snapshot_all_depths(
        const std::array<llama_dsv4_comp_state *, 3> & states,
        llama_seq_id seq) {
    state_snapshots result;
    for (size_t family = 0; family < states.size(); ++family) {
        for (uint32_t depth = 0; depth <= 2; ++depth) {
            result[family][depth] = state_snapshot(*states[family], seq, depth);
        }
    }
    return result;
}

void expect_state_snapshots(
        const std::array<llama_dsv4_comp_state *, 3> & states,
        llama_seq_id seq,
        const state_snapshots & expected,
        const std::string & phase) {
    for (size_t family = 0; family < states.size(); ++family) {
        for (uint32_t depth = 0; depth <= 2; ++depth) {
            expect(state_snapshot(*states[family], seq, depth) == expected[family][depth],
                    phase + " state mismatch at family " + std::to_string(family) +
                    " depth " + std::to_string(depth));
        }
    }
}

void expect_states_zero(
        const std::array<llama_dsv4_comp_state *, 3> & states,
        llama_seq_id seq,
        const std::string & phase) {
    for (auto * state : states) {
        expect(state->sequence_all_depths_zero(seq), phase + " left recurrent state bytes");
    }
}

void prepare_source(composite_fixture & fixture) {
    fixture.put_raw(0, 7, 42);
    composite_fixture::fill_stream(fixture.raw.get_base()->get_k_storage(0), 0, 0x31);
    composite_fixture::fill_stream(fixture.raw.get_swa()->get_k_storage(1), 0, 0x72);
    fixture.seed_compressed_root();
    uint8_t seed = 0x10;
    for (auto * state : fixture.execution_states()) {
        fill_state_sequence(*state, 0, seed);
        seed = (uint8_t) (seed + 0x20);
    }
    fixture.rollback_index[0] = 2;
}

bool same_comp_pages(
        const llama_dsv4_comp_memory_usage & lhs,
        const llama_dsv4_comp_memory_usage & rhs) {
    return lhs.c4.capacity_pages == rhs.c4.capacity_pages &&
           lhs.c4.free_pages == rhs.c4.free_pages && lhs.c4.mapped_pages == rhs.c4.mapped_pages &&
           lhs.c4.shared_pages == rhs.c4.shared_pages && lhs.c4.cow_pages == rhs.c4.cow_pages &&
           lhs.hca.capacity_pages == rhs.hca.capacity_pages &&
           lhs.hca.free_pages == rhs.hca.free_pages && lhs.hca.mapped_pages == rhs.hca.mapped_pages &&
           lhs.hca.shared_pages == rhs.hca.shared_pages && lhs.hca.cow_pages == rhs.hca.cow_pages;
}

void test_composite_detach_attach_and_repeated_accounting() {
    backend_scope backend;
    composite_fixture fixture;
    prepare_source(fixture);
    const auto source_state = snapshot_all_depths(fixture.execution_states(), 0);
    const auto base_bytes = composite_fixture::stream_bytes(fixture.raw.get_base()->get_k_storage(0), 0);
    const auto swa_bytes  = composite_fixture::stream_bytes(fixture.raw.get_swa()->get_k_storage(1), 0);

    llama_dsv4_resident_status allocation_status;
    {
        allocation_scope allocations(true);
        allocation_status = fixture.composite.quote_detach(
                { 0, llama_dsv4_resident_scope::single_context }).status;
    }
    expect(allocation_status == llama_dsv4_resident_status::resource_exhausted,
            "detach quote allocation failure was not fail-before-mutation");
    auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
    expect(quote.status == llama_dsv4_resident_status::ok && quote.unsupported_components == 0,
            "complete composite quote failed");
    expect(quote.rollback_index == 2 && quote.resident_state_slot == 0,
            "composite quote lost rollback or state slot");
    quote.seq_id = 1;
    quote.rollback_index = 0;
    quote.resident.cache_id = UINT64_MAX;
    llama_dsv4_resident_result detached;
    size_t detach_allocations = 0;
    {
        allocation_scope allocations(true);
        detached = fixture.composite.detach(quote);
        detach_allocations = allocations.finish();
    }
    expect(detached.status == llama_dsv4_resident_status::ok && detach_allocations == 0,
            "composite detach was not allocation-free");
    expect(fixture.composite.usage().handles == 1 && fixture.composite.usage().occupied_slots == 1,
            "composite detach did not publish exact ownership");
    expect(fixture.rollback_index[0] == 0, "detach retained execution rollback index");
    expect_states_zero(fixture.execution_states(), 0, "detach source");
    expect(fixture.raw.get_base()->get_cells(0).get_used() == 0 &&
            fixture.raw.get_swa()->get_cells(0).get_used() == 0, "detach retained raw execution metadata");
    llama_dsv4_comp_handle_id binding = 0;
    expect(fixture.compressed.get_binding(0, binding) == llama_dsv4_comp_status::ok && binding != fixture.roots[0],
           "detach did not install a fresh compressed execution root");
    llama_dsv4_comp_handle_info replacement;
    expect(fixture.compressed.get_handle(binding, replacement) == llama_dsv4_comp_status::ok &&
               replacement.visible_c4_rows == 0 && replacement.visible_hca_rows == 0 &&
               replacement.c4_segment_ids.empty() && replacement.hca_segment_ids.empty(),
           "detach replacement compressed root was not constructor-empty");
    const auto detached_usage = fixture.compressed.memory_usage_snapshot();
    expect(detached_usage.resident_handles == 1 && detached_usage.bindings == 2 && detached_usage.handles == 3,
           "detach did not retain history alongside the empty execution root");
    llama_dsv4_comp_batch_plan fresh_graph;
    fresh_graph.graph_execution_ids = { 0 };
    expect(fixture.compressed.quote_batch(fresh_graph).status == llama_dsv4_comp_status::ok,
           "fresh graph could not address the detached source execution");
    expect(fixture.composite.quote_detach({ 1, llama_dsv4_resident_scope::single_context }).status ==
            llama_dsv4_resident_status::capacity_exhausted, "state aperture exhaustion was not fail-closed");

    auto tampered = detached.resident;
    ++tampered.lease_generation;
    expect(fixture.composite.quote_attach(tampered, 1).status == llama_dsv4_resident_status::stale_handle,
            "tampered composite lease was accepted");
    composite_fixture foreign;
    expect(foreign.composite.quote_attach(detached.resident, 0).status == llama_dsv4_resident_status::stale_handle,
            "foreign composite lease was accepted");

    {
        allocation_scope allocations(true);
        allocation_status = fixture.composite.quote_attach(detached.resident, 1).status;
    }
    expect(allocation_status == llama_dsv4_resident_status::resource_exhausted,
            "attach quote allocation failure was not fail-before-mutation");
    auto attach = fixture.composite.quote_attach(detached.resident, 1);
    expect(attach.status == llama_dsv4_resident_status::ok, "different-ID composite attach quote failed");
    attach.execution_id = 0;
    attach.resident.id = UINT64_MAX;
    llama_dsv4_resident_status attached;
    size_t attach_allocations = 0;
    {
        allocation_scope allocations(true);
        attached = fixture.composite.attach(attach);
        attach_allocations = allocations.finish();
    }
    expect(attached == llama_dsv4_resident_status::ok && attach_allocations == 0,
            "composite attach was not allocation-free");
    expect(fixture.composite.usage().handles == 0 && fixture.composite.usage().occupied_slots == 0,
            "attach retained composite ownership");
    expect(fixture.rollback_index[1] == 2, "attach lost rollback index");
    expect_state_snapshots(fixture.execution_states(), 1, source_state, "different-ID attach");
    expect_states_zero(fixture.resident_states(), 0, "attached resident slot");
    expect(composite_fixture::stream_bytes(fixture.raw.get_base()->get_k_storage(0), 1) == base_bytes &&
            composite_fixture::stream_bytes(fixture.raw.get_swa()->get_k_storage(1), 1) == swa_bytes,
            "different-ID attach changed raw/SWA bytes");
    expect(fixture.composite.attach(attach) == llama_dsv4_resident_status::stale_quote,
            "replayed attach quote succeeded");
    expect(fixture.composite.quote_attach(detached.resident, 0).status == llama_dsv4_resident_status::stale_handle,
            "consumed handle survived attach");

    const auto stable = fixture.compressed.memory_usage_snapshot();
    llama_seq_id current = 1;
    uint64_t previous_handle_generation = detached.resident.handle_generation;
    uint64_t previous_lease_generation  = detached.resident.lease_generation;
    for (uint32_t cycle = 0; cycle < 4; ++cycle) {
        const llama_seq_id destination = 1 - current;
        auto cycle_detach = fixture.composite.detach(
                fixture.composite.quote_detach({ current, llama_dsv4_resident_scope::single_context }));
        expect(cycle_detach.status == llama_dsv4_resident_status::ok, "repeated detach failed");
        expect(cycle_detach.resident.handle_generation > previous_handle_generation &&
                cycle_detach.resident.lease_generation > previous_lease_generation,
                "composite generations were not monotonic");
        previous_handle_generation = cycle_detach.resident.handle_generation;
        previous_lease_generation  = cycle_detach.resident.lease_generation;
        expect(fixture.composite.attach(
                fixture.composite.quote_attach(cycle_detach.resident, destination)) ==
                llama_dsv4_resident_status::ok, "repeated attach failed");
        const auto after = fixture.compressed.memory_usage_snapshot();
        expect(
            same_comp_pages(stable, after) && after.resident_handles == 0 && after.bindings == 2 && after.handles == 2,
            "repeated cycle changed compressed page accounting");
        expect(fixture.composite.usage().handles == 0 && fixture.composite.usage().occupied_slots == 0,
                "repeated cycle leaked composite accounting");
        expect_state_snapshots(fixture.execution_states(), destination, source_state, "repeated attach");
        current = destination;
    }
}

void test_failure_rollback_and_retry() {
    backend_scope backend;

    {
        composite_fixture fixture;
        prepare_source(fixture);
        const auto source_state = snapshot_all_depths(fixture.execution_states(), 0);
        auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
        fake_commit_status = GGML_DSV4_SPARSE_OOM;
        llama_dsv4_resident_result failed;
        size_t                     rollback_allocations = 0;
        {
            allocation_scope allocations(true);
            failed               = fixture.composite.detach(quote);
            rollback_allocations = allocations.finish();
        }
        expect(failed.status == llama_dsv4_resident_status::resource_exhausted && rollback_allocations == 0,
               "raw failure after compressed detach was not allocation-free rolled back");
        fake_commit_status = GGML_DSV4_SPARSE_OK;
        llama_dsv4_comp_handle_id binding = 0;
        expect(fixture.compressed.get_binding(0, binding) == llama_dsv4_comp_status::ok &&
                binding == fixture.roots[0], "compressed detach rollback lost source root");
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 0,
                "compressed detach rollback stranded ownership");
        expect_state_snapshots(fixture.execution_states(), 0, source_state, "failed detach source");
        expect_states_zero(fixture.resident_states(), 0, "failed detach resident");
        expect(fixture.composite.detach(quote).status == llama_dsv4_resident_status::stale_quote,
                "rolled-back detach quote replayed");
        fake_commit_throws = true;
        expect(fixture.composite.detach(
                fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context })).status ==
                llama_dsv4_resident_status::backend_error, "exceptional raw detach was not rolled back");
        fake_commit_throws = false;
        expect(fixture.compressed.get_binding(0, binding) == llama_dsv4_comp_status::ok &&
                binding == fixture.roots[0] && fixture.compressed.memory_usage_snapshot().resident_handles == 0,
                "exceptional raw detach stranded compressed ownership");
    }

    {
        composite_fixture fixture;
        prepare_source(fixture);
        llama_dsv4_comp_state_fail_copy_after_for_test(1);
        expect(fixture.composite.detach(
                fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context })).status ==
                llama_dsv4_resident_status::backend_error, "partial recurrent detach copy was accepted");
        llama_dsv4_comp_state_fail_copy_after_for_test(-1);
        expect_states_zero(fixture.resident_states(), 0, "failed recurrent detach copy");

        auto detached = fixture.composite.detach(
                fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }));
        expect(detached.status == llama_dsv4_resident_status::ok, "detach retry after state fault failed");
        const auto blank_root = fixture.roots[1];

        auto state_fault = fixture.composite.quote_attach(detached.resident, 1);
        llama_dsv4_comp_state_fail_copy_after_for_test(1);
        expect(fixture.composite.attach(state_fault) == llama_dsv4_resident_status::backend_error,
                "partial recurrent attach copy was accepted");
        llama_dsv4_comp_state_fail_copy_after_for_test(-1);
        llama_dsv4_comp_handle_id binding = 0;
        expect(fixture.compressed.get_binding(1, binding) == llama_dsv4_comp_status::ok && binding == blank_root,
                "state-copy attach rollback lost empty destination root");
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 1,
                "state-copy attach rollback lost resident root");
        expect_states_zero(fixture.execution_states(), 1, "failed recurrent attach destination");

        auto raw_fault = fixture.composite.quote_attach(detached.resident, 1);
        fake_commit_status = GGML_DSV4_SPARSE_OOM;
        expect(fixture.composite.attach(raw_fault) == llama_dsv4_resident_status::resource_exhausted,
                "raw attach failure was not reported");
        fake_commit_status = GGML_DSV4_SPARSE_OK;
        expect(fixture.compressed.get_binding(1, binding) == llama_dsv4_comp_status::ok && binding == blank_root,
                "raw attach rollback lost empty destination root");
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 1,
                "raw attach rollback lost retryable root");
        expect_states_zero(fixture.execution_states(), 1, "failed raw attach destination");
        auto exceptional_raw = fixture.composite.quote_attach(detached.resident, 1);
        fake_commit_throws = true;
        expect(fixture.composite.attach(exceptional_raw) == llama_dsv4_resident_status::backend_error,
                "exceptional raw attach was not rolled back");
        fake_commit_throws = false;
        expect(fixture.compressed.get_binding(1, binding) == llama_dsv4_comp_status::ok && binding == blank_root &&
                fixture.compressed.memory_usage_snapshot().resident_handles == 1,
                "exceptional raw attach stranded compressed ownership");
        expect_states_zero(fixture.execution_states(), 1, "exceptional raw attach destination");
        expect(fixture.composite.attach(fixture.composite.quote_attach(detached.resident, 1)) ==
                llama_dsv4_resident_status::ok, "attach retry failed");
    }
}

void test_detached_source_accepts_fresh_work_and_clear() {
    backend_scope     backend;
    composite_fixture fixture;
    const auto        baseline = fixture.compressed.memory_usage_snapshot();
    prepare_source(fixture);

    const auto detached =
        fixture.composite.detach(fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }));
    expect(detached.status == llama_dsv4_resident_status::ok, "fresh-source detach failed");

    llama_dsv4_comp_handle_id replacement = 0;
    expect(
        fixture.compressed.get_binding(0, replacement) == llama_dsv4_comp_status::ok && replacement != fixture.roots[0],
        "fresh source has no replacement compressed root");

    llama_dsv4_comp_batch_plan batch;
    batch.graph_execution_ids = { 0 };
    batch.changes             = {
        { replacement, llama_dsv4_comp_family::c4,  1, {} },
        { replacement, llama_dsv4_comp_family::hca, 1, {} },
    };
    const auto quote = fixture.compressed.quote_batch(batch);
    expect(quote.status == llama_dsv4_comp_status::ok, "fresh source graph quote failed");
    const auto reservation = fixture.compressed.try_reserve(quote);
    expect(reservation.status == llama_dsv4_comp_status::ok &&
               fixture.compressed.commit(reservation.ticket) == llama_dsv4_comp_status::ok,
           "fresh source graph commit failed");

    fixture.put_raw(0, 1, 7);
    fill_state_sequence(fixture.csa_execution, 0, 0x91);
    expect(fixture.raw.seq_rm(0, -1, -1), "fresh source raw seq_rm failed");
    for (auto * state : fixture.execution_states()) {
        state->clear(0, true);
    }

    // Mirrors aggregate clear_compressed(seq_id): the bound fresh root can be
    // removed and replaced without touching the independently resident root.
    expect(fixture.compressed.unbind(0) == llama_dsv4_comp_status::ok &&
               fixture.compressed.remove_handle(replacement) == llama_dsv4_comp_status::ok,
           "fresh source compressed clear could not remove its root");
    const auto cleared = fixture.compressed.create_handle();
    expect(cleared.status == llama_dsv4_comp_status::ok &&
               fixture.compressed.bind(0, cleared.handle) == llama_dsv4_comp_status::ok,
           "fresh source compressed clear could not install an empty root");
    const auto during = fixture.compressed.memory_usage_snapshot();
    expect(during.resident_handles == 1 && during.bindings == 2 && during.handles == 3,
           "fresh source clear changed resident ownership accounting");

    expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::ok,
           "resident release after fresh source reuse failed");
    const auto after = fixture.compressed.memory_usage_snapshot();
    expect(after.resident_handles == 0 && after.bindings == 2 && after.handles == 2 && same_comp_pages(baseline, after),
           "fresh source lifecycle did not return to baseline accounting");
}

void test_busy_stale_occupied_and_release() {
    backend_scope backend;

    {
        composite_fixture fixture;
        prepare_source(fixture);
        auto graph = fixture.raw.acquire_resident_batch_lease();
        expect(fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }).status ==
                llama_dsv4_resident_status::not_quiescent, "active graph did not block composite quote");
        graph.reset();

        auto stale = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
        fixture.rollback_index[0] = 1;
        expect(fixture.composite.detach(stale).status == llama_dsv4_resident_status::stale_quote,
                "rollback-index mutation did not stale composite quote");
        fixture.rollback_index[0] = 2;

        auto detached = fixture.composite.detach(
                fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }));
        expect(detached.status == llama_dsv4_resident_status::ok, "detach for occupied tests failed");
        fixture.put_raw(1, 0, 7);
        expect(fixture.composite.quote_attach(detached.resident, 1).status ==
                llama_dsv4_resident_status::slot_occupied, "occupied raw destination was accepted");
        fixture.raw.clear(false);
        fill_state_sequence(fixture.csa_execution, 1, 0x99);
        expect(fixture.composite.quote_attach(detached.resident, 1).status ==
                llama_dsv4_resident_status::slot_occupied, "occupied state destination was accepted");
        fixture.csa_execution.clear(1, true);
        fixture.seed_compressed_root(fixture.roots[1]);
        expect(fixture.composite.quote_attach(detached.resident, 1).status ==
                llama_dsv4_resident_status::slot_occupied, "occupied compressed destination was accepted");

        llama_dsv4_resident_status allocation_status;
        {
            allocation_scope allocations(true);
            allocation_status = fixture.composite.release(detached.resident);
        }
        expect(allocation_status == llama_dsv4_resident_status::resource_exhausted &&
                fixture.compressed.memory_usage_snapshot().resident_handles == 1 &&
                fixture.composite.usage().handles == 1, "release allocation failure stranded ownership");
        fake_quote_status = GGML_DSV4_SPARSE_OOM;
        expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::resource_exhausted,
                "fallible raw release was not rolled back");
        fake_quote_status = GGML_DSV4_SPARSE_OK;
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 1 &&
                fixture.composite.usage().handles == 1, "failed release stranded partial ownership");
        fake_quote_throws = true;
        expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::backend_error,
                "raw release exception was not rolled back");
        fake_quote_throws = false;
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 1 &&
                fixture.composite.usage().handles == 1, "exceptional release stranded partial ownership");
        expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::ok,
                "release retry failed");
        expect(fixture.compressed.memory_usage_snapshot().resident_handles == 0 &&
                fixture.composite.usage().handles == 0 && fixture.composite.usage().occupied_slots == 0,
                "release did not return exact accounting");
        expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::stale_handle,
                "release was not exact-once");
    }

    {
        composite_fixture fixture;
        prepare_source(fixture);
        auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
        fixture.put_raw(1, 1, 8);
        expect(fixture.composite.detach(quote).status == llama_dsv4_resident_status::stale_quote,
                "raw generation mutation did not stale composite quote");
    }

    {
        composite_fixture fixture;
        prepare_source(fixture);
        auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
        fixture.csa_execution.clear(1, true);
        expect(fixture.composite.detach(quote).status == llama_dsv4_resident_status::stale_quote,
                "recurrent-state generation mutation did not stale detach quote");
    }

    {
        composite_fixture fixture;
        prepare_source(fixture);
        const auto detached = fixture.composite.detach(
                fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }));
        expect(detached.status == llama_dsv4_resident_status::ok, "detach for state attach staleness failed");
        auto quote = fixture.composite.quote_attach(detached.resident, 1);
        fixture.csa_execution.clear(1, true);
        expect(fixture.composite.attach(quote) == llama_dsv4_resident_status::stale_quote,
                "recurrent-state generation mutation did not stale attach quote");
        expect(fixture.composite.attach(fixture.composite.quote_attach(detached.resident, 1)) ==
                llama_dsv4_resident_status::ok, "fresh attach after state staleness failed");
    }

    {
        composite_fixture fixture;
        prepare_source(fixture);
        auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
        const auto mutation = fixture.compressed.create_handle();
        expect(mutation.status == llama_dsv4_comp_status::ok, "create staleness mutation");
        expect(fixture.composite.detach(quote).status == llama_dsv4_resident_status::stale_quote,
                "compressed mutation did not stale composite quote");
        expect(fixture.compressed.remove_handle(mutation.handle) == llama_dsv4_comp_status::ok,
                "remove staleness mutation");
    }
}

void test_transaction_blocks_graph_and_destruction_fails_closed() {
    backend_scope backend;
    composite_fixture fixture;
    bool graph_rejected = false;
    {
        auto transaction = fixture.raw.acquire_resident_transaction();
        expect(static_cast<bool>(transaction), "acquire raw transaction guard");
        try {
            (void) fixture.raw.acquire_resident_batch_lease();
        } catch (const std::exception &) {
            graph_rejected = true;
        }
    }
    expect(graph_rejected, "resident transaction allowed a new graph lease");

    const auto expect_excluded = [&](auto action, const std::string & phase) {
        resident_lock_probe probe;
        auto                before_lock_future    = probe.before_lock.get_future();
        auto                lock_contended_future = probe.lock_contended.get_future();
        auto                lock_acquired_future  = probe.lock_acquired.get_future();
        std::thread         worker;
        {
            auto transaction = fixture.raw.acquire_resident_transaction();
            expect(static_cast<bool>(transaction), phase + " transaction acquisition failed");
            worker = start_resident_lock_probe(probe, action);
            before_lock_future.wait();
            lock_contended_future.wait();
        }
        lock_acquired_future.wait();
        worker.join();
        if (probe.error) {
            std::rethrow_exception(probe.error);
        }
    };

    expect_excluded([&]() { fixture.raw.clear(false); }, "raw clear");
    expect_excluded([&]() { (void) fixture.raw.seq_rm(0, -1, -1); }, "raw seq_rm");
    expect_excluded(
        [&]() {
            vector_writer writer;
            fixture.raw.state_write(writer, 0, 0);
        },
        "raw state_write");
    vector_writer serialized;
    fixture.raw.state_write(serialized, 0, 0);
    expect_excluded(
        [&]() {
            vector_reader reader(serialized.data);
            fixture.raw.state_read(reader, 0, 0);
        },
        "raw state_read");
    expect_excluded(
        [&]() {
            auto graph = fixture.raw.acquire_resident_batch_lease();
            expect(graph != nullptr, "graph lease missing after transaction release");
        },
        "raw graph");

#if defined(__unix__) || defined(__APPLE__)
    const pid_t child = fork();
    expect(child >= 0, "fork composite destruction check");
    if (child == 0) {
        backend_scope child_backend;
        {
            composite_fixture child_fixture;
            prepare_source(child_fixture);
            const auto detached = child_fixture.composite.detach(
                    child_fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context }));
            if (detached.status != llama_dsv4_resident_status::ok) {
                _exit(2);
            }
        }
        _exit(0);
    }
    int status = 0;
    expect(waitpid(child, &status, 0) == child, "wait composite destruction check");
    expect(WIFSIGNALED(status), "composite destruction did not fail closed");
#endif
}

void test_composite_detach_retains_raw_transaction() {
    backend_scope     backend;
    composite_fixture fixture;
    prepare_source(fixture);

    vector_writer serialized;
    fixture.raw.state_write(serialized, 1, 0);
    auto quote = fixture.composite.quote_detach({ 0, llama_dsv4_resident_scope::single_context });
    expect(quote.status == llama_dsv4_resident_status::ok, "composite exclusion detach quote failed");

    std::promise<void> resume_commit;
    fake_commit_pause  commit_pause;
    commit_pause.resume = resume_commit.get_future().share();
    auto commit_entered = commit_pause.entered.get_future();

    llama_dsv4_resident_result detached;
    std::exception_ptr         detach_error;
    std::thread                detach_worker([&]() {
        try {
            fake_commit_pause_override = &commit_pause;
            detached                   = fixture.composite.detach(quote);
            fake_commit_pause_override = nullptr;
        } catch (...) {
            fake_commit_pause_override = nullptr;
            detach_error               = std::current_exception();
        }
    });
    commit_entered.wait();

    constexpr size_t                                 operation_count = 5;
    std::array<resident_lock_probe, operation_count> probes;
    std::array<std::future<void>, operation_count>   before_lock;
    std::array<std::future<void>, operation_count>   lock_contended;
    std::array<std::future<void>, operation_count>   lock_acquired;
    for (size_t i = 0; i < operation_count; ++i) {
        before_lock[i]    = probes[i].before_lock.get_future();
        lock_contended[i] = probes[i].lock_contended.get_future();
        lock_acquired[i]  = probes[i].lock_acquired.get_future();
    }

    std::array<std::thread, operation_count> workers = {
        start_resident_lock_probe(probes[0], [&]() { fixture.raw.clear(false); }),
        start_resident_lock_probe(probes[1], [&]() { (void) fixture.raw.seq_rm(1, -1, -1); }),
        start_resident_lock_probe(probes[2],
                                  [&]() {
                                      vector_writer writer;
                                      fixture.raw.state_write(writer, 1, 0);
                                  }),
        start_resident_lock_probe(probes[3],
                                  [&]() {
                                      vector_reader reader(serialized.data);
                                      fixture.raw.state_read(reader, 1, 0);
                                  }),
        start_resident_lock_probe(probes[4],
                                  [&]() {
                                      auto   graph   = fixture.raw.acquire_resident_batch_lease();
                                      expect(graph != nullptr, "composite exclusion graph lease missing after detach");
                                  }),
    };

    for (size_t i = 0; i < operation_count; ++i) {
        before_lock[i].wait();
        lock_contended[i].wait();
    }
    resume_commit.set_value();
    detach_worker.join();
    if (detach_error) {
        std::rethrow_exception(detach_error);
    }
    expect(detached.status == llama_dsv4_resident_status::ok && fixture.composite.usage().handles == 1 &&
               fixture.composite.usage().occupied_slots == 1,
           "composite detach did not publish after retained raw transaction");

    for (size_t i = 0; i < operation_count; ++i) {
        lock_acquired[i].wait();
        workers[i].join();
        if (probes[i].error) {
            std::rethrow_exception(probes[i].error);
        }
    }

    llama_dsv4_comp_handle_id binding = 0;
    expect(fixture.compressed.get_binding(0, binding) == llama_dsv4_comp_status::ok && binding != fixture.roots[0],
           "composite exclusion detach did not publish the empty execution root");
    expect(fixture.composite.release(detached.resident) == llama_dsv4_resident_status::ok,
           "composite exclusion resident release failed");
}

void test_recurrent_logical_all_depth_roundtrip_and_validation() {
    backend_scope backend;
    composite_fixture fixture;
    uint8_t seed = 0x11;
    for (auto * state : fixture.execution_states()) {
        fill_state_sequence(*state, 0, seed);
        fill_state_sequence(*state, 1, (uint8_t) (seed + 0x61));
        seed = (uint8_t) (seed + 0x20);
    }
    const auto source_snapshots      = snapshot_all_depths(fixture.execution_states(), 0);
    const auto destination_snapshots = snapshot_all_depths(fixture.execution_states(), 1);

    std::array<llama_dsv4_recurrent_sequence_state, 3> logical;
    auto states = fixture.execution_states();
    for (size_t family = 0; family < states.size(); ++family) {
        logical[family] = states[family]->export_sequence_all_depths(0);
        states[family]->validate_sequence_all_depths(logical[family]);
        expect(logical[family].n_rs_seq == 2 && !logical[family].kv.empty() &&
                   logical[family].kv.front().chunks.size() == 3,
               "recurrent logical export omitted a rollback plane");

        const auto expect_rejected = [&](llama_dsv4_recurrent_sequence_state malformed,
                                         const std::string & phase) {
            const uint64_t generation = states[family]->state_generation();
            bool threw = false;
            try {
                states[family]->validate_sequence_all_depths(malformed);
            } catch (const std::exception &) {
                threw = true;
            }
            expect(threw, phase + " was accepted");
            expect(states[family]->state_generation() == generation,
                   phase + " changed recurrent generation");
            for (uint32_t depth = 0; depth <= 2; ++depth) {
                expect(state_snapshot(*states[family], 1, depth) == destination_snapshots[family][depth],
                       phase + " changed destination bytes");
            }
        };

        auto malformed = logical[family];
        malformed.state_identity ^= UINT64_C(1);
        expect_rejected(std::move(malformed), "wrong recurrent identity");

        malformed = logical[family];
        malformed.kv.front().chunks.front().bytes.pop_back();
        expect_rejected(std::move(malformed), "truncated recurrent plane");

        malformed = logical[family];
        std::swap(malformed.kv.front().chunks[0], malformed.kv.front().chunks[1]);
        expect_rejected(std::move(malformed), "reordered recurrent ranges");

        malformed = logical[family];
        malformed.score.front().chunks[1] = malformed.score.front().chunks[0];
        expect_rejected(std::move(malformed), "duplicate recurrent range");
    }

    for (size_t family = 0; family < states.size(); ++family) {
        const uint64_t generation = states[family]->state_generation();
        states[family]->import_sequence_all_depths(1, logical[family]);
        expect(states[family]->state_generation() == generation + 1,
               "recurrent logical import generation");
    }
    expect_state_snapshots(states, 1, source_snapshots, "recurrent logical roundtrip");
}

}  // namespace

int main() {
    try {
        test_current_layout_fails_closed();
        test_composite_detach_attach_and_repeated_accounting();
        test_failure_rollback_and_retry();
        test_detached_source_accepts_fresh_work_and_clear();
        test_busy_stale_occupied_and_release();
        test_transaction_blocks_graph_and_destruction_fails_closed();
        test_composite_detach_retains_raw_transaction();
        test_recurrent_logical_all_depth_roundtrip_and_validation();
    } catch (const std::exception & error) {
        std::cerr << "test-dsv4-resident-contract: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-dsv4-resident-contract: all checks passed\n";
    return 0;
}

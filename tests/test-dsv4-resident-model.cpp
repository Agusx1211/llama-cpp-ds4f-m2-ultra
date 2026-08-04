#include "ggml-backend.h"
#include "llama-kv-cache-dsv4.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t N_CTX        = 512;
constexpr uint32_t N_BATCH      = 512;
constexpr uint32_t N_UBATCH     = 512;
constexpr uint32_t N_SEQ_MAX    = 3;
constexpr uint32_t N_RS_SEQ     = 5;
constexpr uint32_t N_OUTPUTS    = 3;
constexpr double   NMSE_LIMIT   = 1.0e-8;
constexpr double   MAXABS_LIMIT = 1.0e-5;

constexpr std::array<uint32_t, 6> BOUNDARIES = { 3, 4, 5, 127, 128, 129 };

using model_ptr   = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct graph_counter {
    uint64_t evaluations = 0;
    uint64_t asks        = 0;
};

struct phase_evaluations {
    uint64_t prompt_evaluations  = 0;
    uint64_t parked_evaluations  = 0;
    uint64_t resumed_evaluations = 0;
    uint64_t post_evaluations    = 0;
    uint64_t prompt_asks         = 0;
    uint64_t parked_asks         = 0;
    uint64_t resumed_asks        = 0;
    uint64_t post_asks           = 0;
};

bool count_graph_evaluations(ggml_tensor * /* tensor */, bool ask, void * user_data) {
    auto * counter = static_cast<graph_counter *>(user_data);
    if (ask) {
        ++counter->asks;
    } else {
        ++counter->evaluations;
    }
    return true;
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

bool env_is_one(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct resident_cleanup {
    llama_kv_cache_dsv4 *      memory = nullptr;
    llama_dsv4_resident_handle resident;
    bool                       active = false;

    explicit resident_cleanup(llama_kv_cache_dsv4 * memory) : memory(memory) {}

    void adopt(const llama_dsv4_resident_handle & handle) {
        expect(!active, "resident cleanup already owns a handle");
        resident = handle;
        active   = true;
    }

    void disarm() { active = false; }

    ~resident_cleanup() noexcept {
        if (!active) {
            return;
        }
        try {
            const auto status = memory->release_resident(resident);
            if (status != llama_dsv4_resident_status::ok) {
                std::fprintf(stderr, "resident-model cleanup failed status=%u\n", (unsigned) status);
                std::terminate();
            }
        } catch (...) {
            std::fprintf(stderr, "resident-model cleanup threw while releasing a live handle\n");
            std::terminate();
        }
    }
};

struct row_input {
    uint32_t     role     = 0;
    llama_seq_id sequence = -1;
    llama_pos    position = 0;
    llama_token  token    = 0;
    bool         output   = false;
};

struct phase_logits {
    std::array<std::vector<float>, 3> values;
};

struct decode_ledger {
    std::array<std::vector<uint32_t>, N_SEQ_MAX> writes;

    void record(llama_seq_id sequence, llama_pos position) {
        expect(sequence >= 0 && sequence < (llama_seq_id) N_SEQ_MAX, "ledger sequence out of range");
        expect(position >= 0, "ledger position out of range");
        auto & sequence_writes = writes[(size_t) sequence];
        if ((size_t) position >= sequence_writes.size()) {
            sequence_writes.resize((size_t) position + 1, 0);
        }
        ++sequence_writes[(size_t) position];
    }

    uint32_t count(llama_seq_id sequence, llama_pos position) const {
        if (sequence < 0 || sequence >= (llama_seq_id) N_SEQ_MAX || position < 0) {
            return 0;
        }
        const auto & sequence_writes = writes[(size_t) sequence];
        return (size_t) position < sequence_writes.size() ? sequence_writes[(size_t) position] : 0;
    }
};

struct model_plan {
    uint32_t                                n_vocab = 0;
    std::array<std::vector<llama_token>, 3> tokens;

    llama_token token(uint32_t role, uint32_t position) const {
        expect(role < tokens.size() && position < tokens[role].size(), "token plan index out of range");
        return tokens[role][position];
    }
};

model_plan make_model_plan(uint32_t n_vocab) {
    expect(n_vocab > 1, "model vocabulary is too small");
    model_plan plan;
    plan.n_vocab                    = n_vocab;
    constexpr uint32_t MAX_POSITION = 140;
    for (uint32_t role = 0; role < plan.tokens.size(); ++role) {
        plan.tokens[role].resize(MAX_POSITION);
        for (uint32_t position = 0; position < MAX_POSITION; ++position) {
            const uint64_t value = UINT64_C(0x9e3779b97f4a7c15) + UINT64_C(0x100000001b3) * (role + 1) +
                                   UINT64_C(0x5851f42d4c957f2d) * (position + 1);
            plan.tokens[role][position] = 1 + (llama_token) (value % (n_vocab - 1));
        }
    }
    return plan;
}

context_ptr make_context(llama_model * model, graph_counter & counter) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = N_CTX;
    params.n_batch              = N_BATCH;
    params.n_ubatch             = N_UBATCH;
    params.n_seq_max            = N_SEQ_MAX;
    params.n_rs_seq             = N_RS_SEQ;
    params.n_outputs_max        = N_OUTPUTS;
    params.kv_unified           = true;
    params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.no_perf              = true;
    params.n_threads            = 16;
    params.n_threads_batch      = 16;
    params.cb_eval              = count_graph_evaluations;
    params.cb_eval_user_data    = &counter;

    context_ptr context(llama_init_from_model(model, params), llama_free);
    expect(context != nullptr, "failed to create exact-model context");
    expect(llama_n_ctx(context.get()) == N_CTX, "context n_ctx mismatch");
    expect(llama_n_batch(context.get()) == N_BATCH, "context n_batch mismatch");
    expect(llama_n_ubatch(context.get()) == N_UBATCH, "context n_ubatch mismatch");
    expect(llama_n_seq_max(context.get()) == N_SEQ_MAX, "context n_seq_max mismatch");
    expect(llama_n_rs_seq(context.get()) == N_RS_SEQ, "context n_rs_seq mismatch");
    return context;
}

phase_logits decode_rows(llama_context *                context,
                         const std::vector<row_input> & rows,
                         uint32_t                       n_vocab,
                         decode_ledger *                ledger = nullptr) {
    expect(!rows.empty() && rows.size() <= N_BATCH, "invalid decode row count");
    uint32_t n_outputs = 0;
    for (const auto & row : rows) {
        expect(row.sequence >= 0 && row.sequence < (llama_seq_id) N_SEQ_MAX, "decode sequence out of range");
        expect(row.role < 3, "decode role out of range");
        n_outputs += row.output ? 1 : 0;
    }
    expect(n_outputs > 0 && n_outputs <= N_OUTPUTS, "invalid decode output count");

    llama_batch batch = llama_batch_init((int32_t) rows.size(), 0, N_SEQ_MAX);
    batch.n_tokens    = (int32_t) rows.size();
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        if (ledger != nullptr) {
            ledger->record(row.sequence, row.position);
        }
        batch.token[i]     = row.token;
        batch.pos[i]       = row.position;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = row.sequence;
        batch.logits[i]    = row.output ? 1 : 0;
    }

    const int32_t result = llama_decode(context, batch);
    llama_batch_free(batch);
    expect(result == LLAMA_DECODE_SUCCESS, "exact-model decode failed");
    llama_synchronize(context);

    phase_logits output;
    uint32_t     output_index = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        if (!row.output) {
            continue;
        }
        // Positive indices are original batch-token indices; output_ids maps
        // those indices to compact output rows internally.
        const float * logits = llama_get_logits_ith(context, (int32_t) i);
        expect(logits != nullptr, "missing exact-model logits");
        output.values[row.role].assign(logits, logits + n_vocab);
        ++output_index;
    }
    expect(output_index == n_outputs, "exact-model output row count changed during capture");
    return output;
}

phase_logits decode_history(llama_context *    context,
                            const model_plan & plan,
                            uint32_t           boundary,
                            llama_seq_id       source_sequence,
                            llama_seq_id       survivor_sequence,
                            decode_ledger *    ledger = nullptr) {
    std::vector<row_input> rows;
    rows.reserve(2 * (boundary + 1));
    for (uint32_t position = 0; position <= boundary; ++position) {
        rows.push_back({ 0, source_sequence, (llama_pos) position, plan.token(0, position), position == boundary });
    }
    for (uint32_t position = 0; position <= boundary; ++position) {
        rows.push_back({ 1, survivor_sequence, (llama_pos) position, plan.token(1, position), position == boundary });
    }
    return decode_rows(context, rows, plan.n_vocab, ledger);
}

void expect_prompt_ledger(const decode_ledger & ledger,
                          llama_seq_id          source_sequence,
                          llama_seq_id          survivor_sequence,
                          uint32_t              boundary) {
    for (llama_seq_id sequence = 0; sequence < (llama_seq_id) N_SEQ_MAX; ++sequence) {
        const bool expected = sequence == source_sequence || sequence == survivor_sequence;
        for (uint32_t position = 0; position <= boundary; ++position) {
            expect(ledger.count(sequence, (llama_pos) position) == (expected ? 1u : 0u),
                   "prompt sequence ledger was not written exactly once");
        }
        if (expected) {
            expect(ledger.writes[(size_t) sequence].size() == boundary + 1,
                   "prompt sequence ledger contains an unexpected position");
        } else {
            expect(ledger.writes[(size_t) sequence].empty(), "prompt sequence ledger populated an unrelated sequence");
        }
    }
}

void expect_exact_positions(const decode_ledger &                                 ledger,
                            const std::array<std::vector<llama_pos>, N_SEQ_MAX> & expected,
                            const char *                                          phase) {
    for (llama_seq_id sequence = 0; sequence < (llama_seq_id) N_SEQ_MAX; ++sequence) {
        std::vector<uint32_t> expected_counts;
        for (const llama_pos position : expected[(size_t) sequence]) {
            expect(position >= 0, std::string(phase) + " expected a negative decode position");
            if ((size_t) position >= expected_counts.size()) {
                expected_counts.resize((size_t) position + 1, 0);
            }
            ++expected_counts[(size_t) position];
        }
        const auto & actual_counts = ledger.writes[(size_t) sequence];
        const size_t n_positions   = std::max(expected_counts.size(), actual_counts.size());
        for (size_t position = 0; position < n_positions; ++position) {
            const uint32_t expected_count = position < expected_counts.size() ? expected_counts[position] : 0;
            const uint32_t actual_count   = position < actual_counts.size() ? actual_counts[position] : 0;
            expect(actual_count == expected_count, std::string(phase) + " decode position ledger mismatch");
        }
    }
}

phase_logits decode_parked(llama_context * context, const model_plan & plan, uint32_t boundary) {
    return decode_rows(context,
                       {
                           { 1, 1, (llama_pos) boundary + 1, plan.token(1, boundary + 1), true },
                           { 2, 0, 0,                        plan.token(2, 0),            true },
    },
                       plan.n_vocab);
}

phase_logits decode_resumed(llama_context *    context,
                            const model_plan & plan,
                            uint32_t           boundary,
                            llama_seq_id       source_sequence,
                            decode_ledger *    ledger = nullptr) {
    return decode_rows(context,
                       {
                           { 0, source_sequence, (llama_pos) boundary + 1, plan.token(0, boundary + 1), true },
                           { 1, 1,               (llama_pos) boundary + 2, plan.token(1, boundary + 2), true },
                           { 2, 0,               1,                        plan.token(2, 1),            true },
    },
                       plan.n_vocab, ledger);
}

phase_logits decode_post_release(llama_context * context, const model_plan & plan, uint32_t boundary) {
    return decode_rows(context,
                       {
                           { 1, 1, (llama_pos) boundary + 3, plan.token(1, boundary + 3), true },
                           { 2, 0, 2,                        plan.token(2, 2),            true },
    },
                       plan.n_vocab);
}

int logits_argmax(const std::vector<float> & logits) {
    expect(!logits.empty(), "empty logits vector");
    return (int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

void compare_logits(const std::vector<float> & expected,
                    const std::vector<float> & actual,
                    const char *               phase,
                    uint32_t                   boundary,
                    uint32_t                   role) {
    expect(expected.size() == actual.size(), "logit vector size mismatch");
    double squared_error   = 0.0;
    double expected_energy = 0.0;
    double max_absolute    = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        expect(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite exact-model logits");
        const double delta = (double) actual[i] - expected[i];
        squared_error += delta * delta;
        expected_energy += (double) expected[i] * expected[i];
        max_absolute = std::max(max_absolute, std::abs(delta));
    }
    const double error        = expected_energy > 0.0 ? squared_error / expected_energy : squared_error;
    const int    expected_top = logits_argmax(expected);
    const int    actual_top   = logits_argmax(actual);
    std::fprintf(stderr, "resident-model boundary=%u phase=%s role=%u nmse=%.3e maxabs=%.3e argmax=%d/%d\n", boundary,
                 phase, role, error, max_absolute, expected_top, actual_top);
    expect(expected_top == actual_top, "exact-model argmax changed across resident transaction");
    expect(error <= NMSE_LIMIT, "exact-model NMSE exceeded resident gate");
    expect(max_absolute <= MAXABS_LIMIT, "exact-model max-absolute error exceeded resident gate");
}

void compare_phase(const phase_logits & expected, const phase_logits & actual, const char * phase, uint32_t boundary) {
    for (uint32_t role = 0; role < expected.values.size(); ++role) {
        if (!expected.values[role].empty()) {
            expect(!actual.values[role].empty(), "candidate omitted expected phase logits");
            compare_logits(expected.values[role], actual.values[role], phase, boundary, role);
        } else {
            expect(actual.values[role].empty(), "candidate emitted an unexpected phase logit row");
        }
    }
}

void compare_sparse_pool(const llama_dsv4_sparse_pool_usage & expected,
                         const llama_dsv4_sparse_pool_usage & actual,
                         const char *                         phase,
                         uint32_t                             boundary) {
    const auto same = [&](auto lhs, auto rhs, const char * field) {
        if (lhs != rhs) {
            char message[256];
            std::snprintf(message, sizeof(message), "%s boundary %u sparse field %s changed", phase, boundary, field);
            fail(message);
        }
    };
    same(expected.pool_id, actual.pool_id, "pool_id");
    same(expected.page_size, actual.page_size, "page_size");
    same(expected.virtual_pages, actual.virtual_pages, "virtual_pages");
    same(expected.physical_pages, actual.physical_pages, "physical_pages");
    same(expected.free_pages, actual.free_pages, "free_pages");
    same(expected.reserved_pages, actual.reserved_pages, "reserved_pages");
    same(expected.mapped_mappings, actual.mapped_mappings, "mapped_mappings");
    same(expected.unique_physical_pages, actual.unique_physical_pages, "unique_physical_pages");
    same(expected.shared_physical_pages, actual.shared_physical_pages, "shared_physical_pages");
    same(expected.shared_mappings, actual.shared_mappings, "shared_mappings");
    same(expected.refcount_sum, actual.refcount_sum, "refcount_sum");
    same(expected.refcount_max, actual.refcount_max, "refcount_max");
    same(expected.cow_allocations, actual.cow_allocations, "cow_allocations");
    same(expected.cow_pages, actual.cow_pages, "cow_pages");
}

void validate_sparse_pool(const llama_dsv4_sparse_pool_usage & pool, const char * phase, uint32_t boundary) {
    expect(pool.unique_physical_pages + pool.free_pages + pool.reserved_pages == pool.physical_pages,
           std::string(phase) + " sparse physical-page conservation is inconsistent");
    expect(pool.unique_physical_pages <= pool.physical_pages,
           std::string(phase) + " sparse unique pages exceed capacity");
    expect(pool.mapped_mappings == pool.refcount_sum,
           std::string(phase) + " sparse mapping/refcount accounting is inconsistent");
    expect(pool.shared_physical_pages <= pool.unique_physical_pages && pool.shared_mappings <= pool.mapped_mappings,
           std::string(phase) + " sparse sharing accounting is inconsistent");
    expect(pool.refcount_max == 0 || pool.refcount_sum >= pool.refcount_max,
           std::string(phase) + " sparse refcount summary is inconsistent");
    (void) boundary;
}

void compare_family_usage(const llama_dsv4_family_usage & expected,
                          const llama_dsv4_family_usage & actual,
                          const char *                    phase,
                          uint32_t                        boundary) {
    expect(expected.family == actual.family && expected.placement_sparse == actual.placement_sparse,
           std::string(phase) + " family identity changed");
    expect(expected.logical_capacity_rows == actual.logical_capacity_rows &&
               expected.logical_mapped_rows == actual.logical_mapped_rows &&
               expected.sequence_mapped_rows == actual.sequence_mapped_rows,
           std::string(phase) + " logical family usage changed");
    expect(expected.pools.size() == actual.pools.size(), std::string(phase) + " sparse pool count changed");
    for (size_t i = 0; i < expected.pools.size(); ++i) {
        compare_sparse_pool(expected.pools[i], actual.pools[i], phase, boundary);
    }
    compare_sparse_pool(expected.total, actual.total, phase, boundary);
}

void compare_memory_baseline(const llama_dsv4_memory_usage_snapshot & expected,
                             const llama_dsv4_memory_usage_snapshot & actual,
                             const char *                             phase,
                             uint32_t                                 boundary) {
    expect(expected.limiting_family == actual.limiting_family &&
               expected.limiting_family_mask == actual.limiting_family_mask &&
               expected.limiting_pool_id == actual.limiting_pool_id &&
               expected.limiting_available_pages == actual.limiting_available_pages,
           std::string(phase) + " limiting memory resource changed");
    for (size_t i = 0; i < expected.families.size(); ++i) {
        compare_family_usage(expected.families[i], actual.families[i], phase, boundary);
    }
    compare_sparse_pool(expected.sparse_total, actual.sparse_total, phase, boundary);
}

void compare_memory_physical(const llama_dsv4_memory_usage_snapshot & expected,
                             const llama_dsv4_memory_usage_snapshot & actual,
                             const char *                             phase,
                             uint32_t                                 boundary) {
    expect(expected.limiting_family == actual.limiting_family &&
               expected.limiting_family_mask == actual.limiting_family_mask &&
               expected.limiting_pool_id == actual.limiting_pool_id &&
               expected.limiting_available_pages == actual.limiting_available_pages,
           std::string(phase) + " limiting sparse resource changed unexpectedly");
    for (size_t i = 0; i < expected.families.size(); ++i) {
        expect(expected.families[i].pools.size() == actual.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed");
        for (size_t p = 0; p < expected.families[i].pools.size(); ++p) {
            compare_sparse_pool(expected.families[i].pools[p], actual.families[i].pools[p], phase, boundary);
        }
        compare_sparse_pool(expected.families[i].total, actual.families[i].total, phase, boundary);
    }
    compare_sparse_pool(expected.sparse_total, actual.sparse_total, phase, boundary);
}

void expect_sparse_generation_advance(const llama_dsv4_memory_usage_snapshot & before,
                                      const llama_dsv4_memory_usage_snapshot & after,
                                      const char *                             phase,
                                      uint32_t                                 boundary) {
    const auto check = [&](const llama_dsv4_sparse_pool_usage & lhs, const llama_dsv4_sparse_pool_usage & rhs) {
        expect(lhs.pool_id == rhs.pool_id, std::string(phase) + " sparse pool identity changed during move");
        expect(rhs.generation > lhs.generation,
               "boundary " + std::to_string(boundary) + " " + phase + " sparse generation did not advance");
    };
    expect(before.families.size() == after.families.size(),
           std::string(phase) + " sparse family count changed during move");
    for (size_t i = 0; i < before.families.size(); ++i) {
        expect(before.families[i].pools.size() == after.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed during move");
        for (size_t p = 0; p < before.families[i].pools.size(); ++p) {
            const auto & lhs = before.families[i].pools[p];
            const auto & rhs = after.families[i].pools[p];
            if (before.families[i].family == LLAMA_DSV4_MEMORY_RAW) {
                check(lhs, rhs);
            } else {
                expect(lhs.pool_id == rhs.pool_id && lhs.generation == rhs.generation,
                       std::string(phase) + " untouched sparse pool generation changed");
            }
        }
        const auto & lhs = before.families[i].total;
        const auto & rhs = after.families[i].total;
        if (before.families[i].family == LLAMA_DSV4_MEMORY_RAW) {
            check(lhs, rhs);
        } else {
            expect(lhs.pool_id == rhs.pool_id && lhs.generation == rhs.generation,
                   std::string(phase) + " untouched sparse family generation changed");
        }
    }
    check(before.sparse_total, after.sparse_total);
}

void expect_sparse_move(const llama_dsv4_memory_usage_snapshot & before,
                        const llama_dsv4_memory_usage_snapshot & after,
                        const char *                             phase,
                        uint32_t                                 boundary) {
    compare_memory_physical(before, after, phase, boundary);
    expect_sparse_generation_advance(before, after, phase, boundary);
}

void expect_sparse_release_delta(const llama_dsv4_memory_usage_snapshot & expected_before,
                                 const llama_dsv4_memory_usage_snapshot & expected_after,
                                 const llama_dsv4_memory_usage_snapshot & actual_before,
                                 const llama_dsv4_memory_usage_snapshot & actual_after,
                                 const char *                             phase,
                                 uint32_t                                 boundary) {
    const auto check = [&](const llama_dsv4_sparse_pool_usage & expected_lhs,
                           const llama_dsv4_sparse_pool_usage & expected_rhs,
                           const llama_dsv4_sparse_pool_usage & actual_lhs,
                           const llama_dsv4_sparse_pool_usage & actual_rhs, bool moved) {
        expect(expected_lhs.page_size == actual_lhs.page_size &&
                   expected_lhs.virtual_pages == actual_lhs.virtual_pages &&
                   expected_lhs.physical_pages == actual_lhs.physical_pages,
               std::string(phase) + " sparse release pool geometry differs from oracle");
        expect(actual_lhs.pool_id == actual_rhs.pool_id, std::string(phase) + " sparse release pool identity changed");
        expect(actual_lhs.physical_pages == actual_rhs.physical_pages &&
                   actual_lhs.virtual_pages == actual_rhs.virtual_pages &&
                   actual_lhs.page_size == actual_rhs.page_size &&
                   actual_lhs.reserved_pages == actual_rhs.reserved_pages,
               std::string(phase) + " sparse release capacity/reservation changed unexpectedly");
        if (moved) {
            expect(actual_rhs.generation > actual_lhs.generation,
                   "boundary " + std::to_string(boundary) + " " + phase + " sparse release generation did not advance");
        } else {
            expect(actual_rhs.generation == actual_lhs.generation,
                   std::string(phase) + " untouched sparse release generation changed");
        }
        expect(actual_rhs.cow_allocations == actual_lhs.cow_allocations && actual_rhs.cow_pages == actual_lhs.cow_pages,
               std::string(phase) + " sparse release COW accounting changed");
        expect(
            actual_rhs.free_pages - actual_lhs.free_pages == expected_rhs.free_pages - expected_lhs.free_pages &&
                actual_lhs.mapped_mappings - actual_rhs.mapped_mappings ==
                    expected_lhs.mapped_mappings - expected_rhs.mapped_mappings &&
                actual_lhs.unique_physical_pages - actual_rhs.unique_physical_pages ==
                    expected_lhs.unique_physical_pages - expected_rhs.unique_physical_pages &&
                actual_lhs.shared_physical_pages - actual_rhs.shared_physical_pages ==
                    expected_lhs.shared_physical_pages - expected_rhs.shared_physical_pages &&
                actual_lhs.shared_mappings - actual_rhs.shared_mappings ==
                    expected_lhs.shared_mappings - expected_rhs.shared_mappings &&
                actual_lhs.refcount_sum - actual_rhs.refcount_sum ==
                    expected_lhs.refcount_sum - expected_rhs.refcount_sum &&
                actual_lhs.refcount_max - actual_rhs.refcount_max ==
                    expected_lhs.refcount_max - expected_rhs.refcount_max,
            "boundary " + std::to_string(boundary) + " " + phase + " sparse release mapping delta differs from oracle");
    };
    expect(expected_before.families.size() == expected_after.families.size() &&
               actual_before.families.size() == actual_after.families.size() &&
               expected_before.families.size() == actual_before.families.size(),
           std::string(phase) + " sparse release family geometry changed");
    for (size_t i = 0; i < actual_before.families.size(); ++i) {
        expect(expected_before.families[i].pools.size() == expected_after.families[i].pools.size() &&
                   actual_before.families[i].pools.size() == actual_after.families[i].pools.size() &&
                   expected_before.families[i].pools.size() == actual_before.families[i].pools.size(),
               std::string(phase) + " sparse release pool geometry changed");
        const bool moved =
            expected_after.families[i].total.free_pages != expected_before.families[i].total.free_pages ||
            expected_after.families[i].total.mapped_mappings != expected_before.families[i].total.mapped_mappings;
        check(expected_before.families[i].total, expected_after.families[i].total, actual_before.families[i].total,
              actual_after.families[i].total, moved);
        for (size_t p = 0; p < actual_before.families[i].pools.size(); ++p) {
            const bool pool_moved =
                expected_after.families[i].pools[p].free_pages != expected_before.families[i].pools[p].free_pages ||
                expected_after.families[i].pools[p].mapped_mappings !=
                    expected_before.families[i].pools[p].mapped_mappings;
            check(expected_before.families[i].pools[p], expected_after.families[i].pools[p],
                  actual_before.families[i].pools[p], actual_after.families[i].pools[p], pool_moved);
        }
    }
    const bool total_moved =
        expected_after.sparse_total.free_pages != expected_before.sparse_total.free_pages ||
        expected_after.sparse_total.mapped_mappings != expected_before.sparse_total.mapped_mappings;
    check(expected_before.sparse_total, expected_after.sparse_total, actual_before.sparse_total,
          actual_after.sparse_total, total_moved);
}

void validate_memory_snapshot(const llama_dsv4_memory_usage_snapshot & snapshot,
                              const char *                             phase,
                              uint32_t                                 boundary) {
    for (const auto & family : snapshot.families) {
        expect(family.logical_mapped_rows <= family.logical_capacity_rows,
               std::string(phase) + " logical family rows exceed capacity");
        validate_sparse_pool(family.total, phase, boundary);
        for (const auto & pool : family.pools) {
            validate_sparse_pool(pool, phase, boundary);
        }
    }
    validate_sparse_pool(snapshot.sparse_total, phase, boundary);
}

void expect_sparse_counters_monotonic(const llama_dsv4_memory_usage_snapshot & before,
                                      const llama_dsv4_memory_usage_snapshot & after,
                                      const char *                             phase,
                                      uint32_t                                 boundary) {
    (void) boundary;
    const auto check = [&](const llama_dsv4_sparse_pool_usage & lhs, const llama_dsv4_sparse_pool_usage & rhs) {
        expect(lhs.pool_id == rhs.pool_id, std::string(phase) + " sparse pool identity changed");
        expect(rhs.generation >= lhs.generation, std::string(phase) + " sparse generation regressed");
        expect(rhs.cow_allocations >= lhs.cow_allocations,
               std::string(phase) + " sparse COW allocation counter regressed");
        expect(rhs.cow_pages >= lhs.cow_pages, std::string(phase) + " sparse COW page counter regressed");
    };
    expect(before.families.size() == after.families.size(), std::string(phase) + " sparse family count changed");
    for (size_t i = 0; i < before.families.size(); ++i) {
        check(before.families[i].total, after.families[i].total);
        expect(before.families[i].pools.size() == after.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed");
        for (size_t p = 0; p < before.families[i].pools.size(); ++p) {
            check(before.families[i].pools[p], after.families[i].pools[p]);
        }
    }
    check(before.sparse_total, after.sparse_total);
}

uint32_t logical_row_ratio(llama_dsv4_memory_family family) {
    switch (family) {
        case LLAMA_DSV4_MEMORY_RAW:
            return 1;
        case LLAMA_DSV4_MEMORY_CSA:
        case LLAMA_DSV4_MEMORY_LID:
            return LLAMA_DSV4_COMP_C4_TOKENS_PER_ROW;
        case LLAMA_DSV4_MEMORY_HCA:
            return LLAMA_DSV4_COMP_HCA_TOKENS_PER_ROW;
        case LLAMA_DSV4_MEMORY_FAMILY_COUNT:
            break;
    }
    fail("invalid DSV4 memory family in logical-row expectation");
}

uint64_t expected_logical_rows(const llama_dsv4_family_usage & family, llama_pos position) {
    if (position < 0) {
        return 0;
    }
    const uint64_t rows = ((uint64_t) position + 1) / logical_row_ratio(family.family);
    return std::min<uint64_t>(rows, family.logical_capacity_rows);
}

void expect_logical_rows(const llama_dsv4_memory_usage_snapshot & snapshot,
                         const std::array<llama_pos, N_SEQ_MAX> & positions,
                         const char *                             phase,
                         uint32_t                                 boundary) {
    for (const auto & family : snapshot.families) {
        expect(family.sequence_mapped_rows.size() == N_SEQ_MAX,
               std::string(phase) + " sequence logical-row geometry changed");
        uint64_t expected_total = 0;
        for (size_t sequence = 0; sequence < positions.size(); ++sequence) {
            const uint64_t expected = expected_logical_rows(family, positions[sequence]);
            expect(family.sequence_mapped_rows[sequence] == expected,
                   "boundary " + std::to_string(boundary) + " " + phase + " sequence logical-row count mismatch");
            expected_total += expected;
        }
        expect(family.logical_mapped_rows == expected_total,
               "boundary " + std::to_string(boundary) + " " + phase + " logical-row total mismatch");
    }
}

llama_dsv4_comp_handle_id comp_binding(const llama_dsv4_comp_pool * pool, uint32_t execution_id, const char * phase) {
    llama_dsv4_comp_handle_id handle = 0;
    expect(pool->get_binding(execution_id, handle) == llama_dsv4_comp_status::ok && handle != 0,
           std::string(phase) + " compressed execution binding is missing");
    return handle;
}

llama_dsv4_comp_handle_info comp_handle_info(const llama_dsv4_comp_pool * pool,
                                             llama_dsv4_comp_handle_id    handle,
                                             const char *                 phase) {
    llama_dsv4_comp_handle_info info;
    expect(pool->get_handle(handle, info) == llama_dsv4_comp_status::ok,
           std::string(phase) + " compressed handle metadata is missing");
    return info;
}

void expect_comp_handle_info_equal(const llama_dsv4_comp_handle_info & expected,
                                   const llama_dsv4_comp_handle_info & actual,
                                   const char *                        phase) {
    expect(expected.id == actual.id && expected.generation == actual.generation &&
               expected.visible_c4_rows == actual.visible_c4_rows &&
               expected.visible_hca_rows == actual.visible_hca_rows &&
               expected.c4_segment_ids == actual.c4_segment_ids && expected.hca_segment_ids == actual.hca_segment_ids,
           std::string(phase) + " compressed survivor handle metadata changed");
}

void expect_comp_release_delta(const llama_dsv4_comp_memory_usage &               before,
                               const llama_dsv4_comp_memory_usage &               after,
                               const llama_dsv4_comp_handle_info &                released,
                               const std::array<llama_dsv4_comp_handle_info, 3> & survivors,
                               const char *                                       phase,
                               uint32_t                                           boundary) {
    (void) boundary;
    const auto check = [&](const llama_dsv4_comp_family_usage & lhs, const llama_dsv4_comp_family_usage & rhs,
                           const std::vector<uint32_t> & released_segments,
                           const std::vector<uint32_t> & survivor_segments) {
        std::set<uint32_t> survivor_set(survivor_segments.begin(), survivor_segments.end());
        uint32_t           expected_freed_segments = 0;
        for (const uint32_t segment : released_segments) {
            expect(segment >= 2, std::string(phase) + " resident release referenced a permanent segment");
            expect(survivor_set.count(segment) == 0,
                   std::string(phase) + " resident release would change shared segment accounting");
            ++expected_freed_segments;
        }

        expect(lhs.capacity_segments == rhs.capacity_segments && lhs.permanent_segments == rhs.permanent_segments &&
                   lhs.capacity_pages == rhs.capacity_pages &&
                   lhs.segment_pages_capacity == rhs.segment_pages_capacity &&
                   lhs.lid_pages_capacity == rhs.lid_pages_capacity,
               std::string(phase) + " compressed capacity geometry changed during release");
        expect(lhs.reserved_segments == rhs.reserved_segments && lhs.reserved_pages == rhs.reserved_pages &&
                   lhs.segment_pages_reserved == rhs.segment_pages_reserved &&
                   lhs.lid_pages_reserved == rhs.lid_pages_reserved,
               std::string(phase) + " compressed reservations changed during release");
        expect(lhs.shared_segments == rhs.shared_segments && lhs.shared_pages == rhs.shared_pages &&
                   lhs.cow_segments == rhs.cow_segments && lhs.cow_pages == rhs.cow_pages &&
                   lhs.scratch_rows_in_use == rhs.scratch_rows_in_use,
               std::string(phase) + " compressed shared/COW accounting changed during release");

        expect(lhs.mapped_segments - rhs.mapped_segments == expected_freed_segments &&
                   rhs.free_segments - lhs.free_segments == expected_freed_segments,
               std::string(phase) + " compressed segment release delta is incorrect");
        const bool     is_c4 = lhs.segment_pages_capacity != 0;
        const uint64_t expected_segment_pages =
            (uint64_t) expected_freed_segments * (is_c4 ? LLAMA_DSV4_COMP_CSA_LAYERS : LLAMA_DSV4_COMP_HCA_LAYERS);
        if (is_c4) {
            expect(lhs.segment_pages_mapped - rhs.segment_pages_mapped == expected_segment_pages &&
                       rhs.segment_pages_free - lhs.segment_pages_free == expected_segment_pages,
                   std::string(phase) + " compressed segment-page release delta is incorrect");
        } else {
            expect(lhs.segment_pages_capacity == rhs.segment_pages_capacity &&
                       lhs.segment_pages_free == rhs.segment_pages_free &&
                       lhs.segment_pages_reserved == rhs.segment_pages_reserved &&
                       lhs.segment_pages_mapped == rhs.segment_pages_mapped,
                   std::string(phase) + " HCA segment-page accounting changed during release");
        }

        std::set<uint32_t> released_groups;
        if (lhs.segment_pages_capacity != 0) {
            for (const uint32_t segment : released_segments) {
                released_groups.insert(segment / 4);
            }
        }
        uint32_t expected_freed_lid_groups = 0;
        for (const uint32_t group : released_groups) {
            // Segment IDs 0 and 1 are permanent zero/scratch mappings, so
            // their LID group remains mapped even when all data segments in
            // the group are released.
            if (group == 0) {
                continue;
            }
            const bool survivor_group = std::any_of(survivor_segments.begin(), survivor_segments.end(),
                                                    [group](uint32_t segment) { return segment / 4 == group; });
            if (!survivor_group) {
                ++expected_freed_lid_groups;
            }
        }
        const uint64_t expected_lid_pages = (uint64_t) expected_freed_lid_groups * LLAMA_DSV4_COMP_LID_LAYERS;
        if (!is_c4) {
            expect(lhs.lid_pages_capacity == rhs.lid_pages_capacity && lhs.lid_pages_free == rhs.lid_pages_free &&
                       lhs.lid_pages_reserved == rhs.lid_pages_reserved && lhs.lid_pages_mapped == rhs.lid_pages_mapped,
                   std::string(phase) + " HCA LID-page accounting changed during release");
        }
        expect(lhs.lid_pages_mapped - rhs.lid_pages_mapped == expected_lid_pages &&
                   rhs.lid_pages_free - lhs.lid_pages_free == expected_lid_pages,
               std::string(phase) + " compressed LID-page release delta is incorrect");
        expect(lhs.mapped_pages - rhs.mapped_pages == expected_segment_pages + expected_lid_pages &&
                   rhs.free_pages - lhs.free_pages == expected_segment_pages + expected_lid_pages,
               std::string(phase) + " compressed total-page release delta is incorrect");
    };
    std::vector<uint32_t> c4_survivors = survivors[0].c4_segment_ids;
    c4_survivors.insert(c4_survivors.end(), survivors[1].c4_segment_ids.begin(), survivors[1].c4_segment_ids.end());
    c4_survivors.insert(c4_survivors.end(), survivors[2].c4_segment_ids.begin(), survivors[2].c4_segment_ids.end());
    std::vector<uint32_t> hca_survivors = survivors[0].hca_segment_ids;
    hca_survivors.insert(hca_survivors.end(), survivors[1].hca_segment_ids.begin(), survivors[1].hca_segment_ids.end());
    hca_survivors.insert(hca_survivors.end(), survivors[2].hca_segment_ids.begin(), survivors[2].hca_segment_ids.end());
    check(before.c4, after.c4, released.c4_segment_ids, c4_survivors);
    check(before.hca, after.hca, released.hca_segment_ids, hca_survivors);
}

void expect_comp_transition(const llama_dsv4_comp_memory_usage & before,
                            const llama_dsv4_comp_memory_usage & after,
                            const char *                         phase,
                            uint32_t                             boundary,
                            uint64_t                             epoch_delta = 1) {
    (void) boundary;
    expect(after.epoch == before.epoch + epoch_delta,
           std::string(phase) + " compressed pool epoch advanced by an unexpected amount");
    expect(after.active_tickets == before.active_tickets &&
               after.retained_ticket_records == before.retained_ticket_records,
           std::string(phase) + " compressed ticket accounting changed unexpectedly");
}

void compare_comp_family(const llama_dsv4_comp_family_usage & expected,
                         const llama_dsv4_comp_family_usage & actual,
                         const char *                         phase,
                         uint32_t                             boundary) {
    (void) phase;
    (void) boundary;
#define CHECK_COMP_FIELD(field) expect(expected.field == actual.field, "compressed " #field " changed unexpectedly")
    CHECK_COMP_FIELD(capacity_segments);
    CHECK_COMP_FIELD(permanent_segments);
    CHECK_COMP_FIELD(free_segments);
    CHECK_COMP_FIELD(reserved_segments);
    CHECK_COMP_FIELD(mapped_segments);
    CHECK_COMP_FIELD(shared_segments);
    CHECK_COMP_FIELD(cow_segments);
    CHECK_COMP_FIELD(scratch_rows_in_use);
    CHECK_COMP_FIELD(capacity_pages);
    CHECK_COMP_FIELD(free_pages);
    CHECK_COMP_FIELD(reserved_pages);
    CHECK_COMP_FIELD(mapped_pages);
    CHECK_COMP_FIELD(shared_pages);
    CHECK_COMP_FIELD(cow_pages);
    CHECK_COMP_FIELD(segment_pages_capacity);
    CHECK_COMP_FIELD(segment_pages_free);
    CHECK_COMP_FIELD(segment_pages_reserved);
    CHECK_COMP_FIELD(segment_pages_mapped);
    CHECK_COMP_FIELD(lid_pages_capacity);
    CHECK_COMP_FIELD(lid_pages_free);
    CHECK_COMP_FIELD(lid_pages_reserved);
    CHECK_COMP_FIELD(lid_pages_mapped);
#undef CHECK_COMP_FIELD
}

void compare_comp_baseline(const llama_dsv4_comp_memory_usage & expected,
                           const llama_dsv4_comp_memory_usage & actual,
                           const char *                         phase,
                           uint32_t                             boundary) {
    compare_comp_family(expected.c4, actual.c4, phase, boundary);
    compare_comp_family(expected.hca, actual.hca, phase, boundary);
    expect(expected.handles == actual.handles && expected.bindings == actual.bindings &&
               expected.resident_handles == actual.resident_handles &&
               expected.active_tickets == actual.active_tickets &&
               expected.retained_ticket_records == actual.retained_ticket_records,
           std::string(phase) + " compressed ownership counts changed unexpectedly");
}

void expect_resident_usage(const llama_dsv4_resident_usage & usage,
                           uint32_t                          occupied,
                           uint32_t                          handles,
                           const char *                      phase,
                           uint32_t                          boundary) {
    expect(usage.cache_id != 0 && usage.capacity == N_SEQ_MAX && usage.occupied_slots == occupied &&
               usage.handles == handles,
           std::string(phase) + " resident usage mismatch at boundary " + std::to_string(boundary));
}

void expect_resident_transition(const llama_dsv4_resident_usage & before,
                                const llama_dsv4_resident_usage & after,
                                const char *                      phase,
                                uint32_t                          boundary) {
    expect(before.cache_id != 0 && before.cache_id == after.cache_id,
           std::string(phase) + " resident cache identity changed");
    expect(after.epoch > before.epoch, std::string(phase) + " resident epoch did not advance");
    expect(after.capacity == before.capacity, std::string(phase) + " resident capacity changed");
    (void) boundary;
}

void expect_context_empty(llama_context *                          context,
                          llama_kv_cache_dsv4 *                    memory,
                          const llama_dsv4_memory_usage_snapshot & memory_baseline,
                          const llama_dsv4_comp_memory_usage &     comp_baseline,
                          uint32_t                                 boundary) {
    llama_memory_clear(llama_get_memory(context), true);
    compare_memory_baseline(memory_baseline, memory->memory_usage_snapshot(), "empty", boundary);
    compare_comp_baseline(comp_baseline, memory->get_comp_pool()->memory_usage_snapshot(), "empty", boundary);
    expect_resident_usage(memory->resident_usage(), 0, 0, "empty", boundary);
}

void run_boundary(llama_model * model, uint32_t n_vocab, uint32_t boundary) {
    const model_plan                 plan = make_model_plan(n_vocab);
    phase_logits                     oracle_prompt;
    phase_logits                     oracle_parked;
    phase_logits                     oracle_resumed;
    phase_logits                     oracle_post;
    phase_evaluations                oracle_evaluations;
    llama_dsv4_memory_usage_snapshot oracle_before_release_memory;
    llama_dsv4_memory_usage_snapshot oracle_release_memory;

    // Keep only one context alive at a time. The model weights remain shared,
    // while the two context compute buffers never coexist in this gate.
    {
        graph_counter counter;
        context_ptr   oracle = make_context(model, counter);
        auto *        memory = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(oracle.get()));
        expect(memory != nullptr, "oracle memory is not DSV4");
        expect(memory->is_aggregate_compressed(), "oracle did not select aggregate compressed storage");
        expect_resident_usage(memory->resident_usage(), 0, 0, "oracle creation", boundary);
        const auto oracle_memory_baseline = memory->memory_usage_snapshot();
        const auto oracle_comp_baseline   = memory->get_comp_pool()->memory_usage_snapshot();

        decode_ledger  oracle_prompt_ledger;
        const uint64_t oracle_before_prompt      = counter.evaluations;
        const uint64_t oracle_asks_before_prompt = counter.asks;
        oracle_prompt = decode_history(oracle.get(), plan, boundary, 2, 1, &oracle_prompt_ledger);
        oracle_evaluations.prompt_evaluations = counter.evaluations - oracle_before_prompt;
        oracle_evaluations.prompt_asks        = counter.asks - oracle_asks_before_prompt;
        expect_prompt_ledger(oracle_prompt_ledger, 2, 1, boundary);
        const uint64_t oracle_before_parked       = counter.evaluations;
        const uint64_t oracle_asks_before_parked  = counter.asks;
        oracle_parked                             = decode_parked(oracle.get(), plan, boundary);
        oracle_evaluations.parked_evaluations     = counter.evaluations - oracle_before_parked;
        oracle_evaluations.parked_asks            = counter.asks - oracle_asks_before_parked;
        const uint64_t oracle_before_resumed      = counter.evaluations;
        const uint64_t oracle_asks_before_resumed = counter.asks;
        decode_ledger  oracle_resumed_ledger;
        oracle_resumed = decode_resumed(oracle.get(), plan, boundary, 2, &oracle_resumed_ledger);
        oracle_evaluations.resumed_evaluations = counter.evaluations - oracle_before_resumed;
        oracle_evaluations.resumed_asks        = counter.asks - oracle_asks_before_resumed;
        expect_exact_positions(oracle_resumed_ledger,
                               { std::vector<llama_pos>{ 1 }, std::vector<llama_pos>{ (llama_pos) boundary + 2 },
                                 std::vector<llama_pos>{ (llama_pos) boundary + 1 } },
                               "oracle resumed");
        oracle_before_release_memory = memory->memory_usage_snapshot();
        expect(llama_memory_seq_rm(llama_get_memory(oracle.get()), 2, -1, -1), "oracle source release failed");
        oracle_release_memory = memory->memory_usage_snapshot();
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 2) == -1,
               "oracle released source position ledger mismatch");
        const uint64_t oracle_before_post      = counter.evaluations;
        const uint64_t oracle_asks_before_post = counter.asks;
        oracle_post                            = decode_post_release(oracle.get(), plan, boundary);
        oracle_evaluations.post_evaluations    = counter.evaluations - oracle_before_post;
        oracle_evaluations.post_asks           = counter.asks - oracle_asks_before_post;
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 2) == -1,
               "oracle source remained populated after release");
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 1) == (llama_pos) boundary + 3,
               "oracle survivor position ledger mismatch");
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 0) == 2,
               "oracle replacement position ledger mismatch");
        expect_context_empty(oracle.get(), memory, oracle_memory_baseline, oracle_comp_baseline, boundary);
    }

    graph_counter counter;
    context_ptr   candidate = make_context(model, counter);
    auto *        memory    = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(candidate.get()));
    expect(memory != nullptr, "candidate memory is not DSV4");
    expect(memory->is_aggregate_compressed(), "candidate did not select aggregate compressed storage");
    resident_cleanup cleanup(memory);
    auto *           comp_pool = memory->get_comp_pool();
    expect(comp_pool != nullptr, "candidate aggregate compressed pool is missing");
    const auto initial_binding0       = comp_binding(comp_pool, 0, "candidate creation");
    const auto initial_binding1       = comp_binding(comp_pool, 1, "candidate creation");
    const auto initial_binding2       = comp_binding(comp_pool, 2, "candidate creation");
    const auto resident_before_detach = memory->resident_usage();
    expect_resident_usage(resident_before_detach, 0, 0, "candidate creation", boundary);
    const auto memory_baseline = memory->memory_usage_snapshot();
    const auto comp_baseline   = memory->get_comp_pool()->memory_usage_snapshot();
    const auto rs_idx_baseline = memory->get_rs_idx();
    expect(rs_idx_baseline.size() == N_SEQ_MAX, "rollback index geometry mismatch");
    const auto expect_rs_idx = [&](const char * phase) {
        expect(memory->get_rs_idx() == rs_idx_baseline,
               std::string(phase) + " rollback index changed during resident transaction");
    };
    validate_memory_snapshot(memory_baseline, "candidate creation", boundary);
    decode_ledger      prompt_ledger;
    const uint64_t     candidate_before_prompt      = counter.evaluations;
    const uint64_t     candidate_asks_before_prompt = counter.asks;
    const phase_logits candidate_prompt = decode_history(candidate.get(), plan, boundary, 0, 1, &prompt_ledger);
    expect(counter.evaluations - candidate_before_prompt == oracle_evaluations.prompt_evaluations &&
               counter.asks - candidate_asks_before_prompt == oracle_evaluations.prompt_asks,
           "candidate prompt graph-evaluation count differs from oracle");
    expect_prompt_ledger(prompt_ledger, 0, 1, boundary);
    compare_phase(oracle_prompt, candidate_prompt, "prompt", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == (llama_pos) boundary,
           "candidate source prompt position ledger mismatch");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary,
           "candidate survivor prompt position ledger mismatch");
    expect_rs_idx("prompt");
    const auto memory_before_detach           = memory->memory_usage_snapshot();
    const auto comp_before_detach             = memory->get_comp_pool()->memory_usage_snapshot();
    const auto survivor_binding_before_detach = comp_binding(comp_pool, 1, "prompt");
    expect_logical_rows(memory_before_detach, { (llama_pos) boundary, (llama_pos) boundary, -1 }, "prompt", boundary);
    expect(survivor_binding_before_detach == initial_binding1, "survivor compressed binding changed during prompt");
    const auto source_info_before_detach   = comp_handle_info(comp_pool, initial_binding0, "prompt");
    const auto survivor_info_before_detach = comp_handle_info(comp_pool, survivor_binding_before_detach, "prompt");
    validate_memory_snapshot(memory_before_detach, "prompt", boundary);

    const uint64_t callback_before_detach = counter.evaluations;
    const auto     quote = memory->quote_resident_detach({ 0, llama_dsv4_resident_scope::single_context });
    expect(quote.status == llama_dsv4_resident_status::ok && quote.unsupported_components == 0,
           "candidate composite detach quote failed");
    expect(quote.seq_id == 0 && quote.scope == llama_dsv4_resident_scope::single_context &&
               quote.rollback_index == rs_idx_baseline[0] && quote.required_components == quote.detachable_components &&
               quote.resident_state_slot < N_SEQ_MAX && quote.resident.cache_id == resident_before_detach.cache_id &&
               quote.resident.id != 0,
           "candidate composite detach quote metadata mismatch");
    const auto detached = memory->detach_resident(quote);
    expect(detached.status == llama_dsv4_resident_status::ok, "candidate composite detach failed");
    expect(detached.resident == quote.resident, "candidate composite detach changed resident handle metadata");
    cleanup.adopt(detached.resident);
    expect(counter.evaluations == callback_before_detach, "resident detach changed graph callback count");
    const auto resident_after_detach = memory->resident_usage();
    expect_resident_transition(resident_before_detach, resident_after_detach, "detached", boundary);
    expect_resident_usage(resident_after_detach, 1, 1, "detached", boundary);
    expect_rs_idx("detached");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == -1,
           "detached source execution sequence remained populated");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary,
           "resident detach altered the survivor sequence");
    expect(comp_binding(comp_pool, 0, "detached") != initial_binding0 &&
               comp_binding(comp_pool, 1, "detached") == survivor_binding_before_detach &&
               comp_binding(comp_pool, 2, "detached") == initial_binding2,
           "resident detach altered survivor or destination compressed bindings");
    expect_comp_handle_info_equal(survivor_info_before_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "detached"), "detached");
    expect_comp_handle_info_equal(source_info_before_detach, comp_handle_info(comp_pool, initial_binding0, "detached"),
                                  "detached source");
    const auto memory_after_detach = memory->memory_usage_snapshot();
    const auto comp_after_detach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_detach.handles == comp_before_detach.handles + 1 &&
               comp_after_detach.bindings == comp_before_detach.bindings &&
               comp_after_detach.resident_handles == comp_before_detach.resident_handles + 1,
           "compressed resident detach ownership accounting mismatch");
    expect_comp_transition(comp_before_detach, comp_after_detach, "detached", boundary);
    compare_comp_family(comp_before_detach.c4, comp_after_detach.c4, "detached", boundary);
    compare_comp_family(comp_before_detach.hca, comp_after_detach.hca, "detached", boundary);
    expect_sparse_move(memory_before_detach, memory_after_detach, "detached", boundary);
    expect_logical_rows(memory_after_detach, { -1, (llama_pos) boundary, -1 }, "detached", boundary);
    validate_memory_snapshot(memory_after_detach, "detached", boundary);

    const uint64_t     candidate_before_parked      = counter.evaluations;
    const uint64_t     candidate_asks_before_parked = counter.asks;
    const phase_logits candidate_parked             = decode_parked(candidate.get(), plan, boundary);
    expect(counter.evaluations - candidate_before_parked == oracle_evaluations.parked_evaluations &&
               counter.asks - candidate_asks_before_parked == oracle_evaluations.parked_asks,
           "candidate parked graph-evaluation count differs from oracle");
    compare_phase(oracle_parked, candidate_parked, "parked", boundary);
    const auto memory_after_parked         = memory->memory_usage_snapshot();
    const auto comp_before_attach          = memory->get_comp_pool()->memory_usage_snapshot();
    const auto resident_before_attach      = memory->resident_usage();
    const auto survivor_info_before_attach = comp_handle_info(comp_pool, survivor_binding_before_detach, "parked");
    expect_sparse_counters_monotonic(memory_after_detach, memory_after_parked, "parked", boundary);
    expect_logical_rows(memory_after_parked, { 0, (llama_pos) boundary + 1, -1 }, "parked", boundary);
    validate_memory_snapshot(memory_after_parked, "parked", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == 0,
           "fresh replacement work did not start at position zero");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 1,
           "survivor position changed during parked phase");

    const uint64_t callback_before_attach = counter.evaluations;
    const auto     attach_quote           = memory->quote_resident_attach(detached.resident, 2);
    expect(attach_quote.status == llama_dsv4_resident_status::ok, "candidate composite attach quote failed");
    expect(attach_quote.execution_id == 2 && attach_quote.resident == detached.resident,
           "candidate composite attach quote metadata mismatch");
    const auto attach_status = memory->attach_resident(attach_quote);
    if (attach_status == llama_dsv4_resident_status::ok) {
        cleanup.disarm();
    }
    expect(attach_status == llama_dsv4_resident_status::ok, "candidate composite attach failed");
    expect(counter.evaluations == callback_before_attach, "resident attach changed graph callback count");
    const auto resident_after_attach = memory->resident_usage();
    expect_resident_transition(resident_before_attach, resident_after_attach, "attached", boundary);
    expect_resident_usage(resident_after_attach, 0, 0, "attached", boundary);
    expect_rs_idx("attached");
    const auto memory_after_attach = memory->memory_usage_snapshot();
    const auto comp_after_attach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_attach.handles + 1 == comp_before_attach.handles &&
               comp_after_attach.bindings == comp_before_attach.bindings &&
               comp_after_attach.resident_handles + 1 == comp_before_attach.resident_handles,
           "compressed resident attach ownership accounting mismatch");
    expect_comp_transition(comp_before_attach, comp_after_attach, "attached", boundary);
    compare_comp_family(comp_before_attach.c4, comp_after_attach.c4, "attached", boundary);
    compare_comp_family(comp_before_attach.hca, comp_after_attach.hca, "attached", boundary);
    expect_sparse_move(memory_after_parked, memory_after_attach, "attached", boundary);
    expect_logical_rows(memory_after_attach, { 0, (llama_pos) boundary + 1, (llama_pos) boundary }, "attached",
                        boundary);
    validate_memory_snapshot(memory_after_attach, "attached", boundary);
    expect(comp_binding(comp_pool, 1, "attached") == survivor_binding_before_detach &&
               comp_binding(comp_pool, 2, "attached") == initial_binding0,
           "resident attach did not bind source to destination or preserve survivor binding");
    expect_comp_handle_info_equal(survivor_info_before_attach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "attached"), "attached");
    expect_comp_handle_info_equal(source_info_before_detach, comp_handle_info(comp_pool, initial_binding0, "attached"),
                                  "attached source");
    const auto stale_attach = memory->quote_resident_attach(detached.resident, 2);
    expect(stale_attach.status == llama_dsv4_resident_status::stale_handle,
           "consumed resident handle was accepted after attach");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == (llama_pos) boundary,
           "attached source position ledger was recomputed or lost");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 1,
           "resident attach altered the survivor sequence");

    const uint64_t     candidate_before_resumed      = counter.evaluations;
    const uint64_t     candidate_asks_before_resumed = counter.asks;
    decode_ledger      resumed_ledger;
    const phase_logits candidate_resumed = decode_resumed(candidate.get(), plan, boundary, 2, &resumed_ledger);
    // Ownership APIs have no callback into the model graph. Equal callback
    // deltas for resumed execution are the strongest public no-replay signal;
    // backend work hidden inside one graph evaluation is not externally visible.
    expect(counter.evaluations - candidate_before_resumed == oracle_evaluations.resumed_evaluations &&
               counter.asks - candidate_asks_before_resumed == oracle_evaluations.resumed_asks,
           "candidate resumed graph-evaluation count differs from oracle (possible prompt replay)");
    expect_exact_positions(resumed_ledger,
                           { std::vector<llama_pos>{ 1 }, std::vector<llama_pos>{ (llama_pos) boundary + 2 },
                             std::vector<llama_pos>{ (llama_pos) boundary + 1 } },
                           "candidate resumed");
    compare_phase(oracle_resumed, candidate_resumed, "resumed", boundary);
    const auto memory_after_resumed = memory->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_after_attach, memory_after_resumed, "resumed", boundary);
    expect_logical_rows(memory_after_resumed, { 1, (llama_pos) boundary + 2, (llama_pos) boundary + 1 }, "resumed",
                        boundary);
    validate_memory_snapshot(memory_after_resumed, "resumed", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == (llama_pos) boundary + 1,
           "resumed source position ledger mismatch");
    const auto survivor_info_before_second_detach =
        comp_handle_info(comp_pool, survivor_binding_before_detach, "resumed");

    const auto     comp_before_second_detach     = memory->get_comp_pool()->memory_usage_snapshot();
    const auto     resident_before_second_detach = memory->resident_usage();
    const uint64_t callback_before_release       = counter.evaluations;
    const auto     second_quote = memory->quote_resident_detach({ 2, llama_dsv4_resident_scope::single_context });
    expect(second_quote.status == llama_dsv4_resident_status::ok, "second composite detach quote failed");
    expect(second_quote.seq_id == 2 && second_quote.scope == llama_dsv4_resident_scope::single_context &&
               second_quote.rollback_index == rs_idx_baseline[2] &&
               second_quote.required_components == second_quote.detachable_components &&
               second_quote.resident_state_slot < N_SEQ_MAX &&
               second_quote.resident.cache_id == resident_after_attach.cache_id && second_quote.resident.id != 0,
           "second composite detach quote metadata mismatch");
    const auto second_detached = memory->detach_resident(second_quote);
    expect(second_detached.status == llama_dsv4_resident_status::ok, "second composite detach failed");
    cleanup.adopt(second_detached.resident);
    const auto resident_after_second_detach = memory->resident_usage();
    expect_resident_transition(resident_before_second_detach, resident_after_second_detach, "second detached",
                               boundary);
    expect_resident_usage(resident_after_second_detach, 1, 1, "second detached", boundary);
    expect(comp_binding(comp_pool, 2, "second detached") != initial_binding0 &&
               comp_binding(comp_pool, 1, "second detached") == survivor_binding_before_detach,
           "second resident detach altered destination or survivor compressed binding");
    expect_comp_handle_info_equal(survivor_info_before_second_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "second detached"),
                                  "second detached");
    const auto released_source_info_before_release = comp_handle_info(comp_pool, initial_binding0, "second detached");
    const auto replacement_info_before_release =
        comp_handle_info(comp_pool, comp_binding(comp_pool, 0, "second detached"), "second detached");
    const auto destination_info_before_release =
        comp_handle_info(comp_pool, comp_binding(comp_pool, 2, "second detached"), "second detached");
    const auto memory_after_second_detach = memory->memory_usage_snapshot();
    const auto comp_after_second_detach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_second_detach.handles == comp_before_second_detach.handles + 1 &&
               comp_after_second_detach.bindings == comp_before_second_detach.bindings &&
               comp_after_second_detach.resident_handles == comp_before_second_detach.resident_handles + 1,
           "compressed second resident detach ownership accounting mismatch");
    expect_comp_transition(comp_before_second_detach, comp_after_second_detach, "second detached", boundary);
    compare_comp_family(comp_before_second_detach.c4, comp_after_second_detach.c4, "second detached", boundary);
    compare_comp_family(comp_before_second_detach.hca, comp_after_second_detach.hca, "second detached", boundary);
    expect_sparse_move(memory_after_resumed, memory_after_second_detach, "second detached", boundary);
    expect_logical_rows(memory_after_second_detach, { 1, (llama_pos) boundary + 2, -1 }, "second detached", boundary);
    const auto release_status = memory->release_resident(second_detached.resident);
    if (release_status == llama_dsv4_resident_status::ok) {
        cleanup.disarm();
    }
    expect(release_status == llama_dsv4_resident_status::ok, "composite resident release failed");
    const auto stale_second_attach = memory->quote_resident_attach(second_detached.resident, 2);
    expect(stale_second_attach.status == llama_dsv4_resident_status::stale_handle,
           "released resident handle was accepted after release");
    expect(memory->release_resident(second_detached.resident) == llama_dsv4_resident_status::stale_handle,
           "released resident handle was accepted for a second release");
    expect(comp_binding(comp_pool, 2, "released") != initial_binding0 &&
               comp_binding(comp_pool, 1, "released") == survivor_binding_before_detach,
           "resident release altered destination or survivor compressed binding");
    expect_comp_handle_info_equal(survivor_info_before_second_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "released"), "released");
    llama_dsv4_comp_handle_info released_source_info;
    expect(comp_pool->get_handle(initial_binding0, released_source_info) == llama_dsv4_comp_status::handle_not_found,
           "released source compressed handle remained addressable");
    expect(counter.evaluations == callback_before_release, "resident release changed graph callback count");
    const auto resident_after_release = memory->resident_usage();
    expect_resident_transition(resident_after_second_detach, resident_after_release, "released", boundary);
    expect_resident_usage(resident_after_release, 0, 0, "released", boundary);
    expect_rs_idx("released");
    const auto memory_after_release = memory->memory_usage_snapshot();
    const auto comp_after_release   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_release.handles + 1 == comp_after_second_detach.handles &&
               comp_after_release.bindings == comp_after_second_detach.bindings &&
               comp_after_release.resident_handles + 1 == comp_after_second_detach.resident_handles,
           "compressed resident release ownership accounting mismatch");
    expect_comp_transition(comp_after_second_detach, comp_after_release, "released", boundary, 2);
    expect_comp_release_delta(
        comp_after_second_detach, comp_after_release, released_source_info_before_release,
        { replacement_info_before_release, survivor_info_before_second_detach, destination_info_before_release },
        "released", boundary);
    expect_sparse_release_delta(oracle_before_release_memory, oracle_release_memory, memory_after_second_detach,
                                memory_after_release, "released", boundary);
    expect_sparse_counters_monotonic(memory_after_second_detach, memory_after_release, "released", boundary);
    expect_logical_rows(memory_after_release, { 1, (llama_pos) boundary + 2, -1 }, "released", boundary);
    validate_memory_snapshot(memory_after_release, "released", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == -1,
           "released source execution sequence remained populated");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 2,
           "resident release altered the survivor sequence");

    const uint64_t     candidate_before_post      = counter.evaluations;
    const uint64_t     candidate_asks_before_post = counter.asks;
    const phase_logits candidate_post             = decode_post_release(candidate.get(), plan, boundary);
    expect(counter.evaluations - candidate_before_post == oracle_evaluations.post_evaluations &&
               counter.asks - candidate_asks_before_post == oracle_evaluations.post_asks,
           "candidate post-release graph-evaluation count differs from oracle");
    const auto memory_after_post = memory->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_after_release, memory_after_post, "post-release", boundary);
    expect_logical_rows(memory_after_post, { 2, (llama_pos) boundary + 3, -1 }, "post-release", boundary);
    validate_memory_snapshot(memory_after_post, "post-release", boundary);
    compare_phase(oracle_post, candidate_post, "post-release", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 3,
           "survivor position ledger mismatch after release");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == 2,
           "replacement position ledger mismatch after release");

    llama_memory_clear(llama_get_memory(candidate.get()), true);
    const auto final_memory = memory->memory_usage_snapshot();
    const auto final_comp   = memory->get_comp_pool()->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_baseline, final_memory, "final", boundary);
    compare_memory_baseline(memory_baseline, final_memory, "final", boundary);
    expect(final_comp.epoch >= comp_baseline.epoch, "compressed pool epoch regressed after clear");
    compare_comp_baseline(comp_baseline, final_comp, "final", boundary);
    expect_resident_usage(memory->resident_usage(), 0, 0, "final", boundary);
    expect_rs_idx("final");
}

struct backend_scope {
    backend_scope() { llama_backend_init(); }

    ~backend_scope() { llama_backend_free(); }
};

bool silent_progress(float /* progress */, void * /* user_data */) {
    return true;
}

bool parse_model_path(int argc, char ** argv, std::string & path) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        std::printf("usage: %s --model <first-shard.gguf>\n", argv[0]);
        return false;
    }
    if (argc != 3 || (std::strcmp(argv[1], "--model") != 0 && std::strcmp(argv[1], "-m") != 0)) {
        std::fprintf(stderr, "usage: %s --model <first-shard.gguf>\n", argv[0]);
        return false;
    }
    path = argv[2];
    return !path.empty();
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    if (!parse_model_path(argc, argv, model_path)) {
        return argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0) ? 0 : 2;
    }
    if (!env_is_one("LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE") || !env_is_one("LLAMA_DSV4_AGGREGATE_POOL_FORCE") ||
        std::getenv("LLAMA_DSV4_AMX_COEXEC") != nullptr) {
        std::fprintf(stderr,
                     "resident-model requires LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE=1, "
                     "LLAMA_DSV4_AGGREGATE_POOL_FORCE=1, and no LLAMA_DSV4_AMX_COEXEC\n");
        return 2;
    }

    try {
        backend_scope      backend;
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 999;
        model_params.split_mode         = LLAMA_SPLIT_MODE_LAYER;
        model_params.progress_callback  = silent_progress;
        model_ptr model(llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        expect(model != nullptr, "failed to load exact DeepSeek V4 Flash model");
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        expect(n_vocab > 1, "exact model vocabulary is empty");
        std::fprintf(stderr,
                     "resident-model start n_ctx=%u n_batch=%u n_ubatch=%u n_seq_max=%u n_rs_seq=%u "
                     "n_outputs_max=%u flash_attn=1 boundaries=6\n",
                     N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX, N_RS_SEQ, N_OUTPUTS);
        for (uint32_t boundary : BOUNDARIES) {
            run_boundary(model.get(), n_vocab, boundary);
        }
        std::fprintf(stderr, "resident-model complete boundaries=6\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "resident-model failure: %s\n", error.what());
        return 1;
    }
}

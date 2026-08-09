#include "../src/llama-ext.h"
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static constexpr uint32_t MAX_ROLLBACK_DEPTH = 5;

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams      = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch, (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(llama_context *                  ctx,
                          const std::vector<llama_token> & tokens,
                          uint32_t                         start,
                          uint32_t                         count,
                          int32_t                          output_idx = -1) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos    = start + i;
        const bool     output = output_idx < 0 ? i + 1 == count : i == (uint32_t) output_idx;
        common_batch_add(batch, tokens[pos], pos, { 0 }, output);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_one(llama_context * ctx, llama_token tok, llama_pos pos) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_batch_add(batch, tok, pos, { 0 }, true);
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_token dsv4_test_token(int n_vocab, llama_seq_id seq_id, llama_pos pos) {
    return (llama_token) ((11*(uint32_t) pos + 37*(uint32_t) seq_id + 3) % (uint32_t) n_vocab);
}

static bool dsv4_decode_sequence_range(
        llama_context * ctx,
        int             n_vocab,
        llama_seq_id    seq_id,
        llama_pos       pos_begin,
        uint32_t        count,
        bool            output) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const llama_pos pos = pos_begin + (llama_pos) i;
        common_batch_add(batch, dsv4_test_token(n_vocab, seq_id, pos), pos, { seq_id }, output);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool dsv4_decode_replay_batch(
        llama_context *                 ctx,
        int                             n_vocab,
        const std::vector<llama_pos> &  pos_begin,
        uint32_t                        count) {
    llama_batch batch = llama_batch_init((int32_t) pos_begin.size()*count, 0, 1);
    for (uint32_t s = 0; s < pos_begin.size(); ++s) {
        for (uint32_t i = 0; i < count; ++i) {
            const llama_pos pos = pos_begin[s] + (llama_pos) i;
            common_batch_add(batch, dsv4_test_token(n_vocab, (llama_seq_id) s, pos), pos,
                    { (llama_seq_id) s }, true);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool dsv4_compare_replay_logits(
        llama_context * ctx_actual,
        llama_context * ctx_expected,
        int             n_vocab,
        uint32_t        n_outputs,
        const char *    label) {
    constexpr float eps = 1e-7f;

    double   sse       = 0.0;
    double   energy    = 0.0;
    float    max_abs   = 0.0f;
    uint32_t first_out = n_outputs;
    int      first_tok = -1;

    for (uint32_t i = 0; i < n_outputs; ++i) {
        const float * actual   = llama_get_logits_ith(ctx_actual, i);
        const float * expected = llama_get_logits_ith(ctx_expected, i);
        if (actual == nullptr || expected == nullptr) {
            fprintf(stderr, "%s : %s missing logits at output %u\n", __func__, label, i);
            return false;
        }
        for (int tok = 0; tok < n_vocab; ++tok) {
            if (!std::isfinite(actual[tok]) || !std::isfinite(expected[tok])) {
                fprintf(stderr, "%s : %s produced non-finite logits at output %u token %d\n",
                        __func__, label, i, tok);
                return false;
            }

            const double delta = (double) actual[tok] - expected[tok];
            const float abs_delta = (float) std::fabs(delta);
            sse += delta*delta;
            energy += (double) expected[tok]*expected[tok];
            max_abs = std::max(max_abs, abs_delta);
            if (abs_delta > eps && first_out == n_outputs) {
                first_out = i;
                first_tok = tok;
            }
        }
    }

    const double nmse = energy > 0.0 ? sse/energy : (sse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    fprintf(stderr, "%s : %s NMSE %.3e, max abs %.3e\n", __func__, label, nmse, (double) max_abs);
    if (max_abs > eps) {
        fprintf(stderr, "%s : %s first mismatch at output %u token %d\n",
                __func__, label, first_out, first_tok);
        return false;
    }

    return true;
}

// DSV4 plans all ubatches before executing any of them. A pending rollback
// must therefore be consumed by only the first ubatch that touches its
// sequence. This is the production shape: four independent lanes, different
// rejection depths, requested unified KV, and a replay long enough to span
// several equal-split ubatches.
//
// The incremental case also protects the bounded snapshot shift-register:
// deeper planes must continue to advance when the latest decode contributes
// fewer tokens than the rollback capacity (normal single-token generation).
static bool test_dsv4_multi_seq_rollback(const common_params & params, llama_model * model, int n_vocab) {
    constexpr uint32_t  n_seqs     = 4;
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_replay   = 40;
    constexpr uint32_t  n_rs_seq   = 8;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = n_rs_seq;
        cparams.n_ctx      = 512;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = params.kv_unified;
        return llama_context_ptr(llama_init_from_model(model, cparams));
    };

    llama_context_ptr ctx_roll = make_ctx_multi();
    llama_context_ptr ctx_ref  = make_ctx_multi();
    if (ctx_roll == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to initialize four-sequence contexts\n", __func__);
        return false;
    }
    if (llama_n_rs_seq(ctx_roll.get()) < n_seqs) {
        fprintf(stderr, "%s : need at least %u rollback snapshots, context has %u\n",
                __func__, n_seqs, llama_n_rs_seq(ctx_roll.get()));
        return false;
    }

    std::vector<llama_pos> replay_pos(n_seqs);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t depth = s + 1;
        const llama_pos p0 = (llama_pos) n_prompt - (llama_pos) depth;
        replay_pos[s] = p0;

        if (!dsv4_decode_sequence_range(ctx_roll.get(), n_vocab, (llama_seq_id) s, 0, p0, false) ||
            !dsv4_decode_sequence_range(ctx_ref.get(),  n_vocab, (llama_seq_id) s, 0, p0, false) ||
            !dsv4_decode_sequence_range(ctx_roll.get(), n_vocab, (llama_seq_id) s, p0, depth, false) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1)) {
            fprintf(stderr, "%s : failed to prepare sequence %u at rollback depth %u\n", __func__, s, depth);
            return false;
        }
    }

    if (!dsv4_decode_replay_batch(ctx_roll.get(), n_vocab, replay_pos, n_replay) ||
        !dsv4_decode_replay_batch(ctx_ref.get(),  n_vocab, replay_pos, n_replay)) {
        fprintf(stderr, "%s : four-sequence replay decode failed\n", __func__);
        return false;
    }
    if (!dsv4_compare_replay_logits(
                ctx_roll.get(), ctx_ref.get(), n_vocab, n_seqs*n_replay,
                params.kv_unified ? "four-sequence requested-unified replay" : "four-sequence split replay")) {
        return false;
    }

    // Four one-token calls must shift the depth-four snapshot back by four
    // states. Replaying the same four-token batch then matches a context that
    // never advanced beyond the rollback boundary.
    llama_memory_clear(llama_get_memory(ctx_roll.get()), true);
    llama_memory_clear(llama_get_memory(ctx_ref.get()),  true);
    constexpr llama_seq_id chain_seq = 2;
    constexpr llama_pos    chain_pos = 12;
    constexpr uint32_t     chain_depth = 4;
    if (!dsv4_decode_sequence_range(ctx_roll.get(), n_vocab, chain_seq, 0, chain_pos, false) ||
        !dsv4_decode_sequence_range(ctx_ref.get(),  n_vocab, chain_seq, 0, chain_pos, false)) {
        fprintf(stderr, "%s : failed to prepare incremental snapshot prefix\n", __func__);
        return false;
    }
    for (uint32_t i = 0; i < chain_depth; ++i) {
        if (!dsv4_decode_sequence_range(
                    ctx_roll.get(), n_vocab, chain_seq, chain_pos + (llama_pos) i, 1, false)) {
            fprintf(stderr, "%s : failed incremental snapshot step %u\n", __func__, i);
            return false;
        }
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), chain_seq, chain_pos, -1)) {
        fprintf(stderr, "%s : failed depth-four incremental rollback\n", __func__);
        return false;
    }
    const std::vector<llama_pos> chain_replay = { chain_pos };
    if (!dsv4_decode_replay_batch(ctx_roll.get(), n_vocab, chain_replay, chain_depth) ||
        !dsv4_decode_replay_batch(ctx_ref.get(),  n_vocab, chain_replay, chain_depth) ||
        !dsv4_compare_replay_logits(
                ctx_roll.get(), ctx_ref.get(), n_vocab, chain_depth, "incremental snapshot chain")) {
        return false;
    }

    // A second partial removal cannot be composed while the first restore is
    // still pending. Once a replay consumes it, a new rollback is valid.
    llama_memory_clear(llama_get_memory(ctx_roll.get()), true);
    constexpr llama_seq_id pending_seq = 3;
    constexpr llama_pos    pending_pos = 16;
    if (!dsv4_decode_sequence_range(ctx_roll.get(), n_vocab, pending_seq, 0, n_prompt, false) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), pending_seq, pending_pos, -1)) {
        fprintf(stderr, "%s : failed to arm pending rollback\n", __func__);
        return false;
    }
    if (llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), pending_seq, pending_pos - 1, -1)) {
        fprintf(stderr, "%s : accepted a second partial removal while rollback was pending\n", __func__);
        return false;
    }
    if (!dsv4_decode_sequence_range(ctx_roll.get(), n_vocab, pending_seq, pending_pos, 1, false) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), pending_seq, pending_pos, -1)) {
        fprintf(stderr, "%s : rollback was not reusable after its restore was consumed\n", __func__);
        return false;
    }

    fprintf(stderr, "%s : four-sequence rollback, incremental snapshot chain, and single-use guard passed\n",
            __func__);
    return true;
}

static bool copy_logits(llama_context *      ctx,
                        int                  n_vocab,
                        uint32_t             depth,
                        const char *         phase,
                        std::vector<float> & logits,
                        int32_t              logits_idx = 0) {
    const float * ptr = llama_get_logits_ith(ctx, logits_idx);
    if (ptr == nullptr) {
        fprintf(stderr, "%s : depth %u missing %s logits\n", __func__, depth, phase);
        return false;
    }

    logits.assign(ptr, ptr + n_vocab);
    return true;
}

static int logits_argmax(const std::vector<float> & logits) {
    return (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static bool compare_checkpoint_logits(const std::vector<float> & actual,
                                      const std::vector<float> & expected,
                                      uint32_t                   depth,
                                      const char *               label) {
    constexpr float eps     = 1e-5f;
    float           max_abs = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            fprintf(stderr, "%s : depth %u %s produced non-finite logits at token %zu (%g, %g)\n", __func__, depth,
                    label, i, (double) actual[i], (double) expected[i]);
            return false;
        }

        const float delta = std::fabs(actual[i] - expected[i]);
        max_abs           = std::max(max_abs, delta);
        if (delta > eps) {
            fprintf(stderr, "%s : depth %u %s mismatch at token %zu (%g != %g, abs %g)\n", __func__, depth, label, i,
                    (double) actual[i], (double) expected[i], (double) delta);
            return false;
        }
    }

    fprintf(stderr, "%s : depth %u %s max abs %.3e\n", __func__, depth, label, (double) max_abs);
    return true;
}

static bool compare_distributions(const std::vector<float> & actual,
                                  const std::vector<float> & expected,
                                  uint32_t                   depth,
                                  const char *               label) {
    double sse     = 0.0;
    double energy  = 0.0;
    float  max_abs = 0.0f;

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            fprintf(stderr, "%s : depth %u %s produced non-finite logits at token %zu (%g, %g)\n", __func__, depth,
                    label, i, (double) actual[i], (double) expected[i]);
            return false;
        }

        const double delta = (double) actual[i] - expected[i];
        sse += delta * delta;
        energy += (double) expected[i] * expected[i];
        max_abs = std::max(max_abs, (float) std::fabs(delta));
    }

    const double nmse = energy > 0.0 ? sse / energy : (sse == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    const int    actual_argmax   = logits_argmax(actual);
    const int    expected_argmax = logits_argmax(expected);

    fprintf(stderr, "%s : depth %u %s NMSE %.3e, max abs %.3e, argmax %d/%d\n", __func__, depth, label, nmse,
            (double) max_abs, actual_argmax, expected_argmax);
    if (actual_argmax != expected_argmax || nmse > 1e-8) {
        fprintf(stderr, "%s : depth %u %s changed the target distribution\n", __func__, depth, label);
        return false;
    }

    return true;
}

static bool rollback_trial(llama_context *                  ctx,
                           const std::vector<llama_token> & tokens,
                           uint32_t                         depth,
                           bool                             expected,
                           const char *                     phase) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    const uint32_t  n_tokens     = (uint32_t) tokens.size();
    const uint32_t  prefix_count = n_tokens - (depth + 1);
    const llama_pos rollback_pos = (llama_pos) prefix_count + 1;
    if (!decode_tokens(ctx, tokens, 0, prefix_count) || !decode_tokens(ctx, tokens, prefix_count, depth + 1)) {
        fprintf(stderr, "%s : %s failed to decode depth %u transition trial\n", __func__, phase, depth);
        return false;
    }

    const bool removed = llama_memory_seq_rm(mem, 0, rollback_pos, -1);
    llama_memory_clear(mem, true);
    if (removed != expected) {
        fprintf(stderr, "%s : %s depth %u rollback returned %d, expected %d\n", __func__, phase, depth, (int) removed,
                (int) expected);
        return false;
    }

    fprintf(stderr, "%s : %s depth %u rollback result %d as expected\n", __func__, phase, depth, (int) removed);
    return true;
}

static bool test_active_depth_transitions(llama_context * ctx, const std::vector<llama_token> & tokens) {
    llama_set_rs_rollback_depth(ctx, 5);
    if (!rollback_trial(ctx, tokens, 5, true, "active depth 5")) {
        return false;
    }

    llama_set_rs_rollback_depth(ctx, 1);
    if (!rollback_trial(ctx, tokens, 1, true, "active depth 1")) {
        return false;
    }

    llama_set_rs_rollback_depth(ctx, 0);
    if (!rollback_trial(ctx, tokens, 1, false, "active depth 0")) {
        return false;
    }

    llama_set_rs_rollback_depth(ctx, 5);
    fprintf(stderr, "%s : active rollback depth transition 5 -> 1 -> 0 -> 5 passed\n", __func__);
    return true;
}

static bool run_rollback_depth(llama_context *                  ctx_src,
                               llama_context *                  ctx_dst,
                               llama_context *                  ctx_ref,
                               llama_context *                  ctx_dirty,
                               const std::vector<llama_token> & tokens,
                               int                              n_vocab,
                               uint32_t                         depth) {
    llama_memory_clear(llama_get_memory(ctx_src), true);
    llama_memory_clear(llama_get_memory(ctx_dst), true);
    llama_memory_clear(llama_get_memory(ctx_ref), true);
    llama_memory_clear(llama_get_memory(ctx_dirty), true);

    const uint32_t    n_tokens     = (uint32_t) tokens.size();
    const uint32_t    prefix_count = n_tokens - (depth + 1);
    const llama_pos   accepted_pos = (llama_pos) prefix_count;
    const llama_pos   replay_pos   = accepted_pos + 1;
    const llama_token replay_tok   = tokens[replay_pos];

    // Reference the verification boundary with the same batched prefix but
    // evaluate the accepted and first rejected tokens one at a time.
    if (!decode_tokens(ctx_ref, tokens, 0, prefix_count) || !decode_one(ctx_ref, tokens[accepted_pos], accepted_pos) ||
        !decode_one(ctx_ref, replay_tok, replay_pos)) {
        fprintf(stderr, "%s : depth %u failed to decode sequential reference\n", __func__, depth);
        return false;
    }
    std::vector<float> logits_ref;
    if (!copy_logits(ctx_ref, n_vocab, depth, "sequential reference", logits_ref)) {
        return false;
    }

    // Measure ordinary two-token verification at the same positions. This
    // separates Metal batch-shape drift from rollback snapshot corruption.
    if (!decode_tokens(ctx_dst, tokens, 0, prefix_count) || !decode_tokens(ctx_dst, tokens, prefix_count, 2, 1)) {
        fprintf(stderr, "%s : depth %u failed to decode two-token batch control\n", __func__, depth);
        return false;
    }
    std::vector<float> logits_batch_control;
    if (!copy_logits(ctx_dst, n_vocab, depth, "two-token batch control", logits_batch_control, 1)) {
        return false;
    }
    llama_memory_clear(llama_get_memory(ctx_dst), true);

    // Evaluate one accepted token followed by `depth` speculative tokens in a
    // single batch. Request logits for the first token that will be rejected.
    if (!decode_tokens(ctx_src, tokens, 0, prefix_count) ||
        !decode_tokens(ctx_src, tokens, prefix_count, depth + 1, 1)) {
        fprintf(stderr, "%s : depth %u failed to decode speculative batch\n", __func__, depth);
        return false;
    }
    std::vector<float> logits_batch;
    if (!copy_logits(ctx_src, n_vocab, depth, "batched reference", logits_batch, 1)) {
        return false;
    }

    if (!llama_memory_seq_rm(llama_get_memory(ctx_src), 0, replay_pos, -1)) {
        fprintf(stderr, "%s : depth %u rollback failed at position %d\n", __func__, depth, (int) replay_pos);
        return false;
    }

    // Save while the source has a pending depth-specific recurrent snapshot,
    // then restore into a clean destination before replaying the first reject.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src, 0, 0);
    ckpt.load_tgt(ctx_dst, 0, 0);

    if (!decode_one(ctx_src, replay_tok, replay_pos) || !decode_one(ctx_dst, replay_tok, replay_pos)) {
        fprintf(stderr, "%s : depth %u clean checkpoint replay failed\n", __func__, depth);
        return false;
    }

    std::vector<float> logits_src;
    std::vector<float> logits_dst;
    if (!copy_logits(ctx_src, n_vocab, depth, "source replay", logits_src) ||
        !copy_logits(ctx_dst, n_vocab, depth, "clean checkpoint replay", logits_dst)) {
        return false;
    }
    bool all_ok = true;
    all_ok      = compare_checkpoint_logits(logits_src, logits_dst, depth, "clean checkpoint") && all_ok;
    all_ok = compare_distributions(logits_batch_control, logits_ref, depth, "two-token batch vs sequential") && all_ok;
    all_ok = compare_distributions(logits_batch, logits_batch_control, depth, "speculative batch vs two-token batch") &&
             all_ok;
    all_ok = compare_distributions(logits_src, logits_ref, depth, "rollback vs sequential") && all_ok;
    all_ok = compare_distributions(logits_src, logits_batch, depth, "rollback vs speculative batch") && all_ok;

    // Load the same checkpoint into a destination whose state planes and
    // rollback index contain a different speculative history at this depth.
    std::vector<llama_token> noise = tokens;
    for (auto & tok : noise) {
        tok = (tok + (llama_token) depth + 1) % n_vocab;
        if (tok < 0) {
            tok = 0;
        }
    }
    if (!decode_tokens(ctx_dirty, noise, 0, n_tokens) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, replay_pos, -1)) {
        fprintf(stderr, "%s : depth %u failed to prepare dirty checkpoint destination\n", __func__, depth);
        return false;
    }

    ckpt.load_tgt(ctx_dirty, 0, 0);
    if (!decode_one(ctx_dirty, replay_tok, replay_pos)) {
        fprintf(stderr, "%s : depth %u dirty checkpoint replay failed\n", __func__, depth);
        return false;
    }

    std::vector<float> logits_dirty;
    if (!copy_logits(ctx_dirty, n_vocab, depth, "dirty checkpoint replay", logits_dirty)) {
        return false;
    }
    all_ok = compare_checkpoint_logits(logits_src, logits_dirty, depth, "dirty checkpoint") && all_ok;

    if (!all_ok) {
        fprintf(stderr, "%s : depth %u recurrent rollback oracle failed\n", __func__, depth);
        return false;
    }

    fprintf(stderr, "%s : depth %u recurrent rollback oracle passed\n", __func__, depth);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict     = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *          model      = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        fprintf(stderr, "%s : model has no vocabulary\n", __func__);
        return 1;
    }

    llama_context_ptr ctx_src(make_ctx(params, model));
    llama_context_ptr ctx_dst(make_ctx(params, model));
    llama_context_ptr ctx_ref(make_ctx(params, model));
    llama_context_ptr ctx_dirty(make_ctx(params, model));
    if (ctx_src == nullptr || ctx_dst == nullptr || ctx_ref == nullptr || ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    if (n_rs_seq == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        return 0;
    }
    if (n_rs_seq < MAX_ROLLBACK_DEPTH) {
        fprintf(stderr, "%s : need at least %u rollback snapshots, context has %u\n", __func__, MAX_ROLLBACK_DEPTH,
                n_rs_seq);
        return 1;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps over the lazy dog beside the river", true);
    }

    const uint32_t required_tokens = MAX_ROLLBACK_DEPTH + 2;
    while (tokens.size() < required_tokens) {
        const llama_token prev = tokens.empty() ? 0 : tokens.back();
        tokens.push_back((prev + 1) % n_vocab);
    }
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }

    char       arch[32] = {};
    const bool is_dsv4  = llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) >= 0 &&
                         std::strcmp(arch, "deepseek4") == 0;
    if (is_dsv4 && !test_active_depth_transitions(ctx_src.get(), tokens)) {
        return 1;
    }
    if (!is_dsv4) {
        fprintf(stderr, "%s : active-depth transition check is DSV4-specific; running rollback oracle for %s\n",
                __func__, arch[0] != '\0' ? arch : "unknown architecture");
    }

    llama_set_rs_rollback_depth(ctx_src.get(), MAX_ROLLBACK_DEPTH);
    llama_set_rs_rollback_depth(ctx_dst.get(), MAX_ROLLBACK_DEPTH);
    llama_set_rs_rollback_depth(ctx_ref.get(), MAX_ROLLBACK_DEPTH);
    llama_set_rs_rollback_depth(ctx_dirty.get(), MAX_ROLLBACK_DEPTH);

    for (uint32_t depth = 1; depth <= MAX_ROLLBACK_DEPTH; ++depth) {
        if (!run_rollback_depth(ctx_src.get(), ctx_dst.get(), ctx_ref.get(), ctx_dirty.get(), tokens, n_vocab, depth)) {
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback depths 1-%u restored successfully\n", __func__, MAX_ROLLBACK_DEPTH);

    if (is_dsv4) {
        // Release the four single-sequence schedulers before constructing two
        // four-lane contexts for the target-model regression.
        ctx_src.reset();
        ctx_dst.reset();
        ctx_ref.reset();
        ctx_dirty.reset();
        if (!test_dsv4_multi_seq_rollback(params, model, n_vocab)) {
            return 1;
        }
    }

    return 0;
}

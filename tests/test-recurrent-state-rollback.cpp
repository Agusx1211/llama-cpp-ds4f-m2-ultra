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
    if (!compare_checkpoint_logits(logits_src, logits_dst, depth, "clean checkpoint") ||
        !compare_distributions(logits_src, logits_ref, depth, "rollback vs sequential") ||
        !compare_distributions(logits_src, logits_batch, depth, "rollback vs batched")) {
        return false;
    }

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
    if (!copy_logits(ctx_dirty, n_vocab, depth, "dirty checkpoint replay", logits_dirty) ||
        !compare_checkpoint_logits(logits_src, logits_dirty, depth, "dirty checkpoint")) {
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
    return 0;
}

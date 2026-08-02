#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <vector>

static llama_context * make_ctx(const common_params & params, llama_model * model) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  (uint32_t) (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, (uint32_t) (cparams.n_rs_seq + 1));
    return llama_init_from_model(model, cparams);
}

static bool decode_tokens(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        uint32_t start,
        uint32_t count) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = start + i;
        common_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == count);
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

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_ptr ctx_src(make_ctx(params, model));
    llama_context_ptr ctx_dst(make_ctx(params, model));
    llama_context_ptr ctx_ref(make_ctx(params, model));
    if (ctx_src == nullptr || ctx_dst == nullptr || ctx_ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    if (llama_n_rs_seq(ctx_src.get()) == 0) {
        fprintf(stderr, "%s : skipping because n_rs_seq is disabled\n", __func__);
        return 0;
    }

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    } else {
        tokens = common_tokenize(ctx_src.get(), "The quick brown fox jumps", true);
    }
    const uint32_t n_rs_seq = llama_n_rs_seq(ctx_src.get());
    if (tokens.size() > n_rs_seq + 1) {
        tokens.resize(n_rs_seq + 1);
    }
    if (tokens.size() < 3) {
        fprintf(stderr, "%s : not enough prompt tokens\n", __func__);
        return 1;
    }
    const uint32_t    n_tokens = tokens.size();
    const llama_token last_tok = tokens.back();
    const llama_pos   last_pos = (llama_pos) n_tokens - 1;

    const llama_pos verify_pos = last_pos - 1;

    // Match speculative verification: establish an existing prefix, evaluate
    // two continuation tokens together, then reject and replay only the last.
    // The reference evaluates those continuation tokens one at a time.
    if (!decode_tokens(ctx_ref.get(), tokens, 0, verify_pos) ||
        !decode_one(ctx_ref.get(), tokens[verify_pos], verify_pos) ||
        !decode_one(ctx_ref.get(), last_tok, last_pos)) {
        fprintf(stderr, "%s : failed to decode reference\n", __func__);
        return 1;
    }
    const float * logits_ref_ptr = llama_get_logits_ith(ctx_ref.get(), 0);
    if (logits_ref_ptr == nullptr) {
        fprintf(stderr, "%s : missing reference logits\n", __func__);
        return 1;
    }
    const std::vector<float> logits_ref(logits_ref_ptr, logits_ref_ptr + n_vocab);

    // Decode the full prompt on the source, then roll back the last position.
    // Rollback leaves the recurrent memory in a snapshot state (rs_idx != 0).
    if (!decode_tokens(ctx_src.get(), tokens, 0, verify_pos) ||
        !decode_tokens(ctx_src.get(), tokens, verify_pos, 2)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }
    const float * logits_batch_ptr = llama_get_logits_ith(ctx_src.get(), -1);
    if (logits_batch_ptr == nullptr) {
        fprintf(stderr, "%s : missing batched logits\n", __func__);
        return 1;
    }
    const std::vector<float> logits_batch(logits_batch_ptr, logits_batch_ptr + n_vocab);
    if (!llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, last_pos, -1)) {
        fprintf(stderr, "%s : rollback failed\n", __func__);
        return 1;
    }

    // Save the rolled-back state and restore it into a fresh context.
    common_prompt_checkpoint ckpt;
    ckpt.update_tgt(ctx_src.get(), 0, 0);
    ckpt.load_tgt(ctx_dst.get(), 0, 0);

    // Replay the rolled-back token on both contexts and compare logits.
    if (!decode_one(ctx_src.get(), last_tok, last_pos) ||
        !decode_one(ctx_dst.get(), last_tok, last_pos)) {
        fprintf(stderr, "%s : replay failed\n", __func__);
        return 1;
    }

    const float * logits_src = llama_get_logits_ith(ctx_src.get(), 0);
    const float * logits_dst = llama_get_logits_ith(ctx_dst.get(), 0);
    if (logits_src == nullptr || logits_dst == nullptr) {
        fprintf(stderr, "%s : missing logits\n", __func__);
        return 1;
    }

    constexpr float eps = 1e-5f;
    double ref_sse = 0.0;
    double ref_energy = 0.0;
    double batch_sse = 0.0;
    double batch_energy = 0.0;
    float ref_max_abs = 0.0f;
    float batch_max_abs = 0.0f;
    int src_argmax = 0;
    int ref_argmax = 0;
    int batch_argmax = 0;
    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dst[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dst[i]);
            return 1;
        }
        const double ref_delta = (double) logits_src[i] - logits_ref[i];
        const double batch_delta = (double) logits_src[i] - logits_batch[i];
        ref_sse += ref_delta*ref_delta;
        ref_energy += (double) logits_ref[i]*logits_ref[i];
        batch_sse += batch_delta*batch_delta;
        batch_energy += (double) logits_batch[i]*logits_batch[i];
        ref_max_abs = std::max(ref_max_abs, (float) std::fabs(ref_delta));
        batch_max_abs = std::max(batch_max_abs, (float) std::fabs(batch_delta));
        if (logits_src[i] > logits_src[src_argmax]) {
            src_argmax = i;
        }
        if (logits_ref[i] > logits_ref[ref_argmax]) {
            ref_argmax = i;
        }
        if (logits_batch[i] > logits_batch[batch_argmax]) {
            batch_argmax = i;
        }
    }
    fprintf(stderr,
            "%s : rollback vs sequential NMSE %.3e, max abs %.3e, argmax %d/%d; "
            "rollback vs batched NMSE %.3e, max abs %.3e, argmax %d/%d\n",
            __func__, ref_sse/ref_energy, (double) ref_max_abs, src_argmax, ref_argmax,
            batch_sse/batch_energy, (double) batch_max_abs, src_argmax, batch_argmax);
    if (src_argmax != ref_argmax || src_argmax != batch_argmax ||
        ref_sse/ref_energy > 1e-8 || batch_sse/batch_energy > 1e-8) {
        fprintf(stderr, "%s : rollback changed the target distribution\n", __func__);
        return 1;
    }

    // Repeat the load into a context that already has its own rollback state:
    // groups 1..n_rs_seq hold a *different* prompt's history, and rs_idx[0] is
    // non-zero at load time. The restore must wipe that state and still match.
    llama_context_ptr ctx_dirty(make_ctx(params, model));
    if (ctx_dirty == nullptr) {
        fprintf(stderr, "%s : failed to init dirty ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> noise = tokens;
    for (auto & t : noise) {
        t = (t + 1) % n_vocab;
        if (t < 0) {
            t = 0;
        }
    }
    if (!decode_tokens(ctx_dirty.get(), noise, 0, n_tokens)) {
        fprintf(stderr, "%s : dirty prompt decode failed\n", __func__);
        return 1;
    }
    if (!llama_memory_seq_rm(llama_get_memory(ctx_dirty.get()), 0, last_pos, -1)) {
        fprintf(stderr, "%s : dirty rollback failed\n", __func__);
        return 1;
    }

    ckpt.load_tgt(ctx_dirty.get(), 0, 0);

    if (!decode_one(ctx_dirty.get(), last_tok, last_pos)) {
        fprintf(stderr, "%s : dirty replay failed\n", __func__);
        return 1;
    }

    const float * logits_dirty = llama_get_logits_ith(ctx_dirty.get(), 0);
    if (logits_dirty == nullptr) {
        fprintf(stderr, "%s : missing dirty logits\n", __func__);
        return 1;
    }

    for (int i = 0; i < n_vocab; ++i) {
        if (std::fabs(logits_src[i] - logits_dirty[i]) > eps) {
            fprintf(stderr, "%s : dirty-ctx logits mismatch at token %d (%g != %g)\n",
                    __func__, i, (double) logits_src[i], (double) logits_dirty[i]);
            return 1;
        }
    }

    fprintf(stderr, "%s : recurrent rollback checkpoint restored successfully\n", __func__);
    return 0;
}

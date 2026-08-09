#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

// LLAMA_DSV4_SPARSE_ROUTE_DEBUG=<n>: log the first <n> compressed-attention
// routing decisions, one line each, so a routing change can be confirmed
// without a full GGML_METAL_KPROF window. Diagnostic only, default off.
// WARN because llama-server's log callback drops llama-internal INFO.
static void dsv4_sparse_route_debug(const char * path, int64_t ne01, int64_t n_csa, int il, bool draft) {
    static const int64_t budget = []() {
        const char * v = std::getenv("LLAMA_DSV4_SPARSE_ROUTE_DEBUG");
        return v != nullptr ? atoll(v) : (int64_t) 0;
    }();
    if (budget <= 0) {
        return;
    }
    static int64_t emitted = 0;
    if (emitted >= budget) {
        return;
    }
    ++emitted;
    LLAMA_LOG_WARN("SPARSEROUTE path=%s ne01=%lld n_csa=%lld il=%d draft=%d\n",
            path, (long long) ne01, (long long) n_csa, il, (int) draft);
}

static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

void llama_model_deepseek4::load_arch_hparams(llama_model_loader & ml) {
    // The released model appends one MTP decoder block after the 43-layer
    // trunk. Read this before sizing any per-layer metadata through n_layer().
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    ml.get_key(LLM_KV_BLOCK_SIZE, hparams.n_dspark_block_size, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");
    if (hparams.n_layer_nextn > 1 && hparams.n_dspark_block_size == 0) {
        throw std::runtime_error("DeepSeek-V4 supports exactly one MTP block");
    }
    if (hparams.n_dspark_block_size > 0) {
        if (hparams.n_layer_nextn != 3) {
            throw std::runtime_error("DeepSeek-V4 DSpark requires exactly three draft blocks");
        }
        if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false) || target_layer_ids.size() != 3) {
            throw std::runtime_error("DeepSeek-V4 DSpark requires three target layer ids");
        }
    }

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                     hparams.dsv4_hash_layer_count);

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4 compress_ratios is shorter than block_count");
    }
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4 loader currently expects sqrtsoftplus MoE scoring");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    if (hparams.n_dspark_block_size > 0) {
        // Official DSpark fuses the mean states after three target layers and
        // emits one standard hidden-width row from its encoder/confidence head.
        hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size()*hparams.n_embd;
        hparams.n_embd_nextn_impl = hparams.n_embd;
    } else {
        // A trunk-only target still has to expose its complete four-stream
        // state when the MTP block is supplied as a separate draft model.
        hparams.n_embd_nextn_impl = hparams.dsv4_hc_mult*hparams.n_embd;
    }

    if (hparams.n_layer_nextn > 0) {
        for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 0) {
                throw std::runtime_error("DeepSeek-V4 MTP block must use uncompressed sliding-window attention");
            }
            hparams.is_swa_impl[il] = true;
        }
    }

    switch (hparams.n_layer()) {
        case 43: type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deepseek4::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    // A speculative sidecar can omit the target embedding and output matrices.
    // They are resolved through ctx_other when the draft context is created.
    const bool is_sidecar = hparams.n_layer_nextn > 0 && ml.get_weight("blk.0.attn_norm.weight") == nullptr;
    const int root_flags = is_sidecar ? TENSOR_NOT_REQUIRED : 0;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, root_flags);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, root_flags);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, root_flags);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN, "weight"),    {hc_dim, hc_mult}, root_flags);
    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE, "weight"),  {hc_mult}, root_flags);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, root_flags);

    // Speculative GGUFs deliberately omit the target's blocks 0..42.
    const int trunk_flags = is_sidecar ? TENSOR_NOT_REQUIRED : 0;
    const int mtp_flags = ml.load_mtp ? 0 : TENSOR_SKIP;

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const int flags = i < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank * o_groups}, flags);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        const int64_t ratio = hparams.dsv4_compress_ratios[i];
        if (ratio != 0) {
            const int64_t coff = ratio == 4 ? 2 : 1;

            layer.attn_comp_wkv   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,   "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, coff * n_embd_head}, flags);
            layer.attn_comp_ape   = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_APE,   "weight", i), {coff * n_embd_head, ratio}, flags);
            layer.attn_comp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM,  "weight", i), {n_embd_head}, flags);

            if (ratio == 4) {
                const int64_t n_embd_indexer = hparams.indexer_head_size;

                layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, hparams.indexer_n_head}, flags);
                layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

                layer.indexer_comp_wkv   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WKV,   "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", i), {n_embd, 2 * n_embd_indexer}, flags);
                layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", i), {2 * n_embd_indexer, ratio}, flags);
                layer.indexer_comp_norm  = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_NORM,  "weight", i), {n_embd_indexer}, flags);
            } else if (ratio != 128) {
                throw std::runtime_error("DeepSeek-V4 loader only supports compression ratios 0, 4, and 128");
            }
        }

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, flags);
        if ((uint32_t) i < hparams.dsv4_hash_layer_count) {
            layer.ffn_gate_tid2eid = create_tensor(tn(LLM_TENSOR_FFN_GATE_TID2EID, "weight", i), {n_expert_used, n_vocab}, flags);
        } else {
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias", i), {n_expert}, flags);
        }
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (i >= n_layer) {
            const int nextn_flags = hparams.n_dspark_block_size > 0 ? flags | TENSOR_NOT_REQUIRED : flags;
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2*n_embd, n_embd}, nextn_flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, nextn_flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, nextn_flags);
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", i), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, nextn_flags);
            // Bundled conversion keeps the MTP-specific mixer here. MTP-only
            // sidecars expose the same three tensors through the root head.
            layer.nextn.hc_head_fn       = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_FN,       "weight", i), {hc_dim, hc_mult}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.hc_head_base     = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_BASE,     "weight", i), {hc_mult}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.hc_head_scale    = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_SCALE,    "weight", i), {1}, flags | TENSOR_NOT_REQUIRED);

            if (hparams.n_dspark_block_size > 0 && i == n_layer) {
                layer.nextn.dspark_main_proj = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ, "weight", i),
                        {(int64_t) target_layer_ids.size()*n_embd, n_embd}, flags);
                layer.nextn.dspark_main_norm = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM, "weight", i),
                        {n_embd}, flags);
            }
        }
    }

    if (hparams.n_dspark_block_size > 0) {
        const int dspark_flags = ml.load_mtp ? 0 : TENSOR_SKIP;
        const ggml_tensor * markov_meta = ml.get_tensor_meta("markov_w1.weight");
        if (markov_meta == nullptr) {
            throw std::runtime_error("DeepSeek-V4 DSpark sidecar is missing its Markov head");
        }
        const int64_t rank = markov_meta->ne[0];
        dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {rank, n_vocab}, dspark_flags);
        dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), {rank, n_vocab}, dspark_flags);
        dspark_conf_proj = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), {n_embd + rank, 1}, dspark_flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek4::build_arch_graph(const llm_graph_params & params) const {
    if (hparams.n_dspark_block_size > 0) {
        switch (params.gtype) {
            case LLM_GRAPH_TYPE_ENCODER:
                return std::make_unique<graph_dspark_encoder>(*this, params);
            case LLM_GRAPH_TYPE_DECODER_MTP:
                return std::make_unique<graph_dspark>(*this, params);
            default:
                GGML_ABORT("DeepSeek-V4 DSpark sidecar requires encoder or MTP decoder graph");
        }
    }
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(
        ggml_context * ctx,
        ggml_tensor  * t,
        int64_t        ne0,
        int64_t        ne1,
        int64_t        i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

struct dsv4_state_tensors {
    ggml_tensor * kv;
    ggml_tensor * score;
};

static dsv4_state_tensors dsv4_build_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il) {
    dsv4_state_tensors restored = {
        state->get_kv_all(ctx, il),
        state->get_score_all(ctx, il),
    };

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return restored;
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, restored.kv, inp.state_restore_src_idxs);
    restored.kv = state->cpy_kv(ctx, kv_rows, inp.state_restore_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, restored.score, inp.state_restore_src_idxs);
    restored.score = state->cpy_score(ctx, score_rows, inp.state_restore_dst_idxs, il);

    return restored;
}

static dsv4_state_tensors dsv4_build_state_snapshot(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        ggml_tensor * source_kv,
        ggml_tensor * source_score,
        int32_t il) {
    if (inp.state_snapshot_src_idxs == nullptr || inp.state_snapshot_dst_idxs == nullptr ||
            source_kv == nullptr || source_score == nullptr) {
        return {};
    }

    ggml_tensor * kv_rows = ggml_get_rows(ctx, source_kv, inp.state_snapshot_src_idxs);
    ggml_tensor * kv = state->cpy_kv(ctx, kv_rows, inp.state_snapshot_dst_idxs, il);

    ggml_tensor * score_rows = ggml_get_rows(ctx, source_score, inp.state_snapshot_src_idxs);
    ggml_tensor * score = state->cpy_score(ctx, score_rows, inp.state_snapshot_dst_idxs, il);

    return { kv, score };
}

static constexpr int64_t DSV4_CSA_RATIO  = 4;
static constexpr int64_t DSV4_HCA_RATIO  = 128;

static ggml_tensor * dsv4_hc_mean(ggml_context * ctx, ggml_tensor * x) {
    const int64_t hc = x->ne[1];

    ggml_tensor * result = ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], 0);
    for (int64_t ih = 1; ih < hc; ++ih) {
        result = ggml_add(ctx, result,
                ggml_view_2d(ctx, x, x->ne[0], x->ne[2], x->nb[2], ih*x->nb[1]));
    }

    return ggml_scale(ctx, result, 1.0f/hc);
}

static void dsv4_build_dspark_markov_head(
        llm_graph_context & g,
        const llama_model & model,
        ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto & res = g.res;

    GGML_ASSERT(model.dspark_markov_w1 && model.dspark_markov_w2 && model.dspark_conf_proj);

    ggml_tensor * base = res->t_logits;
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tokens = base->ne[1];
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t block_size = model.hparams.n_dspark_block_size;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    const int64_t block_tokens = n_tokens/n_blocks;
    // Graph reservation probes synthetic token batches up to n_ubatch. Real
    // DSpark decoding is clamped to its trained block size by the driver.
    if (block_tokens > block_size) {
        return;
    }

    const size_t token_stride = (size_t) block_tokens*tokens->nb[0];
    const size_t logits_stride = (size_t) block_tokens*base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * hidden = res->t_embd;
    ggml_tensor * logits_all = nullptr;
    ggml_tensor * conf_all = nullptr;

    for (int64_t i = 0; i < block_tokens; ++i) {
        ggml_tensor * markov = ggml_get_rows(ctx0, model.dspark_markov_w1, prev);
        ggml_tensor * bias = ggml_mul_mat(ctx0, model.dspark_markov_w2, markov);

        ggml_tensor * logits_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks,
                logits_stride, i*base->nb[1]);
        logits_i = ggml_add(ctx0, logits_i, bias);
        logits_all = logits_all ? ggml_concat(ctx0, logits_all, logits_i, 1) : logits_i;

        ggml_tensor * hidden_i = ggml_view_2d(ctx0, hidden, hidden->ne[0], n_blocks,
                (size_t) block_tokens*hidden->nb[1], i*hidden->nb[1]);
        ggml_tensor * conf_inp = ggml_concat(ctx0, ggml_cont(ctx0, hidden_i), markov, 0);
        ggml_tensor * conf = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, model.dspark_conf_proj, conf_inp));
        conf_all = conf_all ? ggml_concat(ctx0, conf_all, conf, 1) : conf;

        if (i + 1 < block_tokens) {
            prev = ggml_argmax(ctx0, logits_i);
        }
    }

    ggml_tensor * logits = ggml_reshape_3d(ctx0, logits_all, n_vocab, n_blocks, block_tokens);
    logits = ggml_cont(ctx0, ggml_permute(ctx0, logits, 0, 2, 1, 3));
    logits = ggml_reshape_2d(ctx0, logits, n_vocab, n_tokens);

    ggml_tensor * conf = ggml_reshape_3d(ctx0, conf_all, 1, n_blocks, block_tokens);
    conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
    conf = ggml_reshape_2d(ctx0, conf, 1, n_tokens);
    conf = ggml_repeat(ctx0, conf, hidden);

    res->t_h_nextn = conf;
    res->t_logits = logits;
    ggml_build_forward_expand(g.gf, conf);
    ggml_build_forward_expand(g.gf, logits);
}

static ggml_tensor * dsv4_hc_affine(
        ggml_context * ctx,
        ggml_tensor  * x,
        ggml_tensor  * scale,
        ggml_tensor  * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * weights,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    if (cparams.fused_dsv4_hc_pre && il >= 0) {
        ggml_tensor * result = ggml_dsv4_hc_pre(ctx0, x, weights);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
        return result;
    }

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_sinkhorn(
        ggml_tensor * comb,
        int           il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]. Sinkhorn follows the reference:
    // row softmax over dst, one column normalization, then repeated row/column normalization.
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_pre(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    ggml_tensor * result = build_hc_pre(x, pre, il);
    return result;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == hparams.dsv4_hc_mult);

    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, x, residual, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_hc_pre(x, pre, -1);
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * comp = nullptr;
    if (cparams.fused_dsv4_compress) {
        comp = ggml_dsv4_compress(
                ctx0, kv_state, score_state, state_read_idxs, DSV4_HCA_RATIO, false);
        res->add_fused_node({LLM_FUSED_OP_DSV4_COMPRESS, comp, il});
    } else {
        ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
        kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
        cb(kv, name, il);

        ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
        score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
        cb(score, name, il);

        ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv, 1, 0, 2, 3));
        ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx0, scores);
        comp = ggml_mul(ctx0, values, weights);
        comp = ggml_sum_rows(ctx0, comp);
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    }
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t ratio,
        int64_t n_embd_head,
        const char * name,
        int il) const {
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * comp = nullptr;
    if (cparams.fused_dsv4_compress) {
        comp = ggml_dsv4_compress(
                ctx0, kv_state, score_state, state_read_idxs, (int32_t) ratio, true);
        res->add_fused_node({LLM_FUSED_OP_DSV4_COMPRESS, comp, il});
    } else {
        kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
        score_state = dsv4_append_zero_row(ctx0, score_state, true);

        const int64_t n_read = ratio*n_blocks;
        ggml_tensor * kv_rows = ggml_get_rows(ctx0, kv_state, state_read_idxs);
        ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

        ggml_tensor * kv_prev = ggml_cont(ctx0,
                ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
        kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
        cb(kv_prev, name, il);

        ggml_tensor * score_prev = ggml_cont(ctx0,
                ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
        score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
        cb(score_prev, name, il);

        ggml_tensor * kv_cur = ggml_cont(ctx0,
                ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                    n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
        kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

        ggml_tensor * score_cur = ggml_cont(ctx0,
                ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                    n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
        score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

        ggml_tensor * values = ggml_concat(ctx0, kv_prev, kv_cur, 1);
        ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);
        values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
        scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

        ggml_tensor * weights = ggml_soft_max(ctx0, scores);
        comp = ggml_mul(ctx0, values, weights);
        comp = ggml_sum_rows(ctx0, comp);
        comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    }
    cb(comp, name, il);

    comp = build_norm(comp, norm, nullptr, LLM_NORM_RMS, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

ggml_tensor * llama_model_deepseek4::graph::build_lid_top_k(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const auto & inp_lid = inp_dsv4->get_lid();
    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot();
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(inp_lid.k_rot);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = build_lora_mm(layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(indexer_q_pe, "lid_q_pe", il);

    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    indexer_q = llama_mul_mat_hadamard(ctx0, indexer_q, inp_lid.k_rot);
    cb(indexer_q, "lid_q_rot", il);

    ggml_tensor * indexer_weights = build_lora_mm(layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    const bool indexed = inp_lid.segment_ids != nullptr;
    ggml_tensor * indexer_k = indexed ? inp_dsv4->mctx->get_lid()->get_k_pool(ctx0, il) :
            inp_dsv4->mctx->get_lid()->get_k(ctx0, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= (indexed ? indexer_k->ne[1] : indexer_k->ne[2]));

    if (!indexed) {
        indexer_k = ggml_view_4d(ctx0, indexer_k,
                indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
                indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    }
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexed ? inp_lid.kq_mask->ne[3] : indexer_k->ne[3];
    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    ggml_tensor * indexer_score = nullptr;
    if (cparams.fused_lid || indexed) {
        indexer_score = indexed ? ggml_dsv4_indexed_lightning_indexer(
                ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask, inp_lid.segment_ids) :
                ggml_lightning_indexer(ctx0, indexer_q, indexer_k, indexer_weights, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
        res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, indexer_score, il});
    } else {
        indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
        cb(indexer_q, "lid_q", il);
        indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
        cb(indexer_k, "lid_k", il);

        ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
        cb(indexer_kq, "lid_kq", il);

        indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
        cb(indexer_kq, "lid_kq", il);

        indexer_score = ggml_relu(ctx0, indexer_kq);
        indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
        indexer_score = ggml_sum_rows(ctx0, indexer_score);
        indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
        cb(indexer_score, "lid_score", il);

        indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
        cb(indexer_score, "lid_score_masked", il);
    }

    const uint32_t n_top_k = indexer_score->ne[0] < hparams.indexer_top_k ? indexer_score->ne[0] : hparams.indexer_top_k;
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char * name,
        int il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

ggml_tensor * llama_model_deepseek4::graph::build_csa_lid_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_csa = inp_dsv4->get_csa();
    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    const bool indexed = inp_csa.segment_ids != nullptr;
    ggml_tensor * csa_k = indexed ? inp_dsv4->mctx->get_csa()->get_k_pool(ctx0, il) :
            inp_dsv4->mctx->get_csa()->get_k(ctx0, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= (indexed ? csa_k->ne[1] : csa_k->ne[2]));

    if (!indexed) {
        csa_k = ggml_view_4d(ctx0, csa_k,
                csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
                csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    }
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    const bool sparse_k_type = raw_k->type == csa_k->type &&
            (raw_k->type == GGML_TYPE_F16 || raw_k->type == GGML_TYPE_Q8_0);

    // Selecting every compressed row is equivalent to using the visibility
    // mask directly. Avoid building the Lightning Indexer and TOP_K graph until
    // the compressed cache grows beyond the selection size. Auto probes are
    // the only exception because they must materialize their fused ops.
    ggml_tensor * top_k = nullptr;
    if (cparams.auto_fdsv4_aux || cparams.auto_fdsv4_sparse ||
            n_csa > (int64_t) hparams.indexer_top_k) {
        top_k = build_lid_top_k(model, inp_dsv4, qr, cur, inp_pos, il);
    }

    // At sufficiently deep single-token decode, gather only the Lightning
    // Indexer's selected compressed keys. This preserves the regular head layout
    // used by Metal's vector Flash Attention while making its KV work fixed-size.
    static const bool sparse_parallel_enabled =
            std::getenv("GGML_DSV4_SPARSE_PARALLEL_DISABLE") == nullptr;
    static const bool sparse_gather_enabled =
            std::getenv("GGML_DSV4_SPARSE_GATHER_DISABLE") == nullptr;
    const int64_t comp_n_stream = indexed ? inp_csa.kq_mask->ne[3] : csa_k->ne[3];
    const bool sparse_parallel_decode = sparse_parallel_enabled &&
            comp_n_stream > 1 && q->ne[2] == comp_n_stream;
    const bool gather_decode = sparse_gather_enabled &&
            cparams.fused_dsv4_sparse && cparams.flash_attn &&
            sparse_k_type &&
            (q->ne[2] == 1 || sparse_parallel_decode) && top_k &&
            top_k->ne[1] == 1 && top_k->ne[3] == comp_n_stream &&
            n_csa >= 2*(int64_t) hparams.indexer_top_k;
    if (gather_decode) {
        dsv4_sparse_route_debug("gather", q->ne[2], n_csa, il, hparams.n_dspark_block_size > 0);
        const int64_t n_stream = comp_n_stream;
        const int64_t n_raw = raw_k->type == GGML_TYPE_Q8_0 ?
                std::min<int64_t>(hparams.n_swa, raw_k->ne[2]) : 0;
        ggml_tensor * packed = indexed ? ggml_dsv4_indexed_sparse_pack(
                ctx0, raw_k, csa_k, raw_mask, inp_csa.kq_mask, top_k, inp_csa.segment_ids, n_raw) :
                ggml_dsv4_sparse_pack(ctx0, raw_k, csa_k, raw_mask, inp_csa.kq_mask, top_k, n_raw);
        cb(packed, "csa_gathered_pack", il);
        res->add_fused_node({LLM_FUSED_OP_DSV4_SPARSE_PACK, packed, il});

        ggml_tensor * k_sel;
        ggml_tensor * kq_mask;
        if (n_raw > 0) {
            const int64_t nk = n_raw + top_k->ne[0];
            k_sel = ggml_view_4d(ctx0, packed, csa_k->ne[0], 1, nk, n_stream,
                    csa_k->ne[0]*sizeof(ggml_fp16_t), csa_k->ne[0]*sizeof(ggml_fp16_t), packed->nb[1], 0);
            kq_mask = ggml_view_4d(ctx0, packed, nk, 1, 1, n_stream,
                    nk*sizeof(ggml_fp16_t), nk*sizeof(ggml_fp16_t), packed->nb[1],
                    csa_k->ne[0]*nk*sizeof(ggml_fp16_t));
        } else {
            const int64_t nk = top_k->ne[0];
            ggml_tensor * gathered = ggml_view_4d(ctx0, packed, csa_k->ne[0], 1, nk, n_stream,
                    csa_k->ne[0]*sizeof(ggml_fp16_t), csa_k->ne[0]*sizeof(ggml_fp16_t), packed->nb[1], 0);
            cb(gathered, "csa_gathered_k", il);

            k_sel = ggml_concat(ctx0, raw_k, gathered, 2);

            ggml_tensor * comp_mask = ggml_view_4d(ctx0, packed, nk, 1, 1, n_stream,
                    nk*sizeof(ggml_fp16_t), nk*sizeof(ggml_fp16_t), packed->nb[1],
                    csa_k->ne[0]*nk*sizeof(ggml_fp16_t));
            cb(comp_mask, "csa_gathered_mask", il);

            kq_mask = ggml_concat(ctx0, raw_mask, comp_mask, 0);
        }
        cb(k_sel, "csa_k_selected", il);
        cb(kq_mask, "csa_lid_kq_mask", il);

        if (sparse_parallel_decode) {
            const int64_t n_head   = q->ne[1];
            const int64_t nk       = k_sel->ne[2];

            ggml_tensor * q_fa = ggml_reshape_4d(ctx0, q, q->ne[0], 1, n_head, n_stream);
            ggml_tensor * k_fa = ggml_view_4d(ctx0, k_sel, k_sel->ne[0], nk, 1, n_stream,
                    k_sel->nb[2], k_sel->nb[3], k_sel->nb[3], 0);
            ggml_tensor * out = ggml_flash_attn_ext(ctx0, q_fa, k_fa, k_fa, kq_mask, kq_scale,
                    hparams.f_max_alibi_bias,
                    hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
            ggml_flash_attn_ext_add_sinks(out, sinks);
            ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, out, il});
            out = ggml_reshape_2d(ctx0, out, q->ne[0]*n_head, n_stream);
            ggml_build_forward_expand(gf, out);

            if (k_rot) {
                out = llama_mul_mat_hadamard(ctx0, out, k_rot);
            }
            cb(out, "attn_csa_lid_gathered_parallel", il);
            return out;
        }

        ggml_tensor * out = build_attn_mha(
                q, k_sel, k_sel, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_csa_lid_gathered", il);
        return out;
    }

    // The Lightning Indexer selects a different compressed working set for every
    // prefill token. Reinterpret the 64 attention heads as query rows of one MQA
    // problem per token so Metal's tiled flash-attention kernel consumes only
    // those keys. Auto probing forces this branch once during setup so unsupported
    // layer devices can disable it.
    const bool sparse_probe = cparams.auto_fdsv4_sparse;

    // GGML_DSV4_SPARSE_PACK_MIN_TOKENS: the smallest query-row count routed to
    // the packed sparse-attention path. Below it the fallback masks the whole
    // compressed range instead, and that fallback is the ONLY consumption path
    // whose cost depends on the selection being *contiguous*:
    // FC_flash_attn_ext_vec_sparse_mask skips a K-row group only when all NE
    // entries in it are -inf.
    //
    // The historical literal 8 left a gap at 2..7 query rows, which is exactly
    // the speculative-verify shape under --spec-draft-n-max 5 and, per the
    // indexer-zero-rows census, ~80% of decode selections (252 of 315 had
    // ne01 = 4). ne01 = 1 already takes the gather path above; prefill
    // (ne01 = n_ubatch) already packs. So 2..7 was the one shape paying the
    // scatter penalty. See notes/2026-08-08-sparse-pack-routing.md.
    //
    // The DSpark drafter is a DeepSeek V4 model too and runs this same builder,
    // so a process-wide threshold changes the drafter's numerics as well as the
    // target's - and a drafter that produces different tokens changes draft
    // acceptance, which moves end-to-end t/s far more than the attention cost
    // this knob is meant to control. GGML_DSV4_SPARSE_PACK_MIN_TOKENS_DRAFT is
    // therefore a separate threshold for `n_dspark_block_size > 0` models so the
    // two effects can be separated. Default 8 for both.
    static const int64_t sparse_pack_min_tokens_tgt = []() {
        const char * v = std::getenv("GGML_DSV4_SPARSE_PACK_MIN_TOKENS");
        const int64_t n = v != nullptr ? atoll(v) : 8;
        return n < 2 ? (int64_t) 2 : n;
    }();
    static const int64_t sparse_pack_min_tokens_drf = []() {
        const char * v = std::getenv("GGML_DSV4_SPARSE_PACK_MIN_TOKENS_DRAFT");
        const int64_t n = v != nullptr ? atoll(v) : 8;
        return n < 2 ? (int64_t) 2 : n;
    }();
    const bool is_dspark_draft = hparams.n_dspark_block_size > 0;
    const int64_t sparse_pack_min_tokens =
            is_dspark_draft ? sparse_pack_min_tokens_drf : sparse_pack_min_tokens_tgt;

    // The pack kernel stages the whole per-token working set in threadgroup
    // memory (kernel_dsv4_sparse_pack / kernel_dsv4_indexed_sparse_pack in
    // ggml-metal.metal, `constexpr int max_selected = 128 + 512`). That bound
    // is this target's n_swa + indexer_top_k exactly, so it holds for DSV4
    // Flash - but it is a silent threadgroup-memory overrun if a future model
    // widens either, and it is now reachable from decode as well as prefill.
    // Fall back to the mask path rather than corrupt memory.
    constexpr int64_t sparse_pack_max_selected = 128 + 512;
    const int64_t sparse_pack_nk = top_k != nullptr ?
            std::min<int64_t>(hparams.n_swa, raw_k->ne[2]) + top_k->ne[0] : 0;
    const bool sparse_pack_fits = top_k != nullptr && sparse_pack_nk <= sparse_pack_max_selected;

    const bool sparse_prefill = q->ne[2] >= sparse_pack_min_tokens &&
            n_csa > (int64_t) hparams.indexer_top_k;
    if (cparams.fused_dsv4_sparse && cparams.flash_attn &&
            sparse_k_type && sparse_pack_fits &&
            (sparse_probe || sparse_prefill)) {
        dsv4_sparse_route_debug("pack", q->ne[2], n_csa, il, is_dspark_draft);
        GGML_ASSERT(top_k);
        const int64_t n_stream = comp_n_stream;
        const int64_t nq       = q->ne[2]/n_stream;
        const int64_t nt       = q->ne[2];
        const int64_t n_head   = q->ne[1];
        const int64_t n_raw    = std::min<int64_t>(hparams.n_swa, raw_k->ne[2]);

        GGML_ASSERT(q->ne[0] == raw_k->ne[0]);
        GGML_ASSERT(raw_k->ne[1] == 1);
        GGML_ASSERT(indexed || csa_k->ne[1] == 1);
        GGML_ASSERT(raw_k->ne[3] == n_stream);
        GGML_ASSERT(raw_mask->ne[1] == nq && raw_mask->ne[3] == n_stream);
        GGML_ASSERT(inp_csa.kq_mask->ne[1] == nq && inp_csa.kq_mask->ne[3] == n_stream);

        ggml_tensor * packed = indexed ? ggml_dsv4_indexed_sparse_pack(ctx0, raw_k, csa_k, raw_mask,
                inp_csa.kq_mask, top_k, inp_csa.segment_ids, n_raw) :
                ggml_dsv4_sparse_pack(ctx0, raw_k, csa_k, raw_mask, inp_csa.kq_mask, top_k, n_raw);
        cb(packed, "csa_sparse_pack", il);
        res->add_fused_node({LLM_FUSED_OP_DSV4_SPARSE_PACK, packed, il});

        const int64_t nk = n_raw + top_k->ne[0];
        ggml_tensor * k_sel = ggml_view_4d(ctx0, packed, q->ne[0], nk, 1, nt,
                q->ne[0]*sizeof(ggml_fp16_t), q->ne[0]*nk*sizeof(ggml_fp16_t), packed->nb[1], 0);
        ggml_tensor * mask_sel = ggml_view_4d(ctx0, packed, nk, 1, 1, nt,
                nk*sizeof(ggml_fp16_t), nk*sizeof(ggml_fp16_t), packed->nb[1],
                q->ne[0]*nk*sizeof(ggml_fp16_t));
        cb(k_sel, "csa_sparse_k", il);
        cb(mask_sel, "csa_sparse_mask", il);

        ggml_tensor * q_fa = ggml_reshape_4d(ctx0, q, q->ne[0], n_head, 1, nt);
        ggml_tensor * out = ggml_flash_attn_ext(ctx0, q_fa, k_sel, k_sel, mask_sel, kq_scale,
                hparams.f_max_alibi_bias, hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_flash_attn_ext_add_sinks_rows(out, sinks);
        ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
        res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, out, il});
        cb(out, "attn_csa_lid_sparse_fa", il);
        out = ggml_reshape_2d(ctx0, out, q->ne[0]*n_head, nt);
        ggml_build_forward_expand(gf, out);

        if (k_rot) {
            out = llama_mul_mat_hadamard(ctx0, out, k_rot);
        }
        cb(out, "attn_csa_lid_sparse", il);
        return out;
    }

    dsv4_sparse_route_debug("mask", q->ne[2], n_csa, il, is_dspark_draft);

    ggml_tensor * k_all = indexed ? ggml_dsv4_indexed_concat(
            ctx0, raw_k, csa_k, inp_csa.segment_ids, n_csa) :
            ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * csa_mask = inp_csa.kq_mask;
    ggml_tensor * kq_mask  = nullptr;
    const bool fused_mask_types = raw_mask->type == GGML_TYPE_F16 && csa_mask->type == GGML_TYPE_F16;
    if (top_k && cparams.fused_dsv4_top_k_mask && fused_mask_types) {
        kq_mask = ggml_dsv4_top_k_mask(ctx0, raw_mask, csa_mask, top_k);
        cb(kq_mask, "csa_top_k_mask", il);
        res->add_fused_node({LLM_FUSED_OP_DSV4_TOP_K_MASK, kq_mask, il});
    } else {
        if (top_k) {
            csa_mask = build_top_k_mask(csa_mask, top_k, "csa_top_k_mask", il);
        } else {
            cb(csa_mask, "csa_top_k_mask", il);
        }
        kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    }

    cb(kq_mask, "csa_lid_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_hca_attention(
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    const auto & inp_hca = inp_dsv4->get_hca();
    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp_attn->self_k_rot;
    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    const bool indexed = inp_hca.segment_ids != nullptr;
    ggml_tensor * hca_k = indexed ? inp_dsv4->mctx->get_hca()->get_k_pool(ctx0, il) :
            inp_dsv4->mctx->get_hca()->get_k(ctx0, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= (indexed ? hca_k->ne[1] : hca_k->ne[2]));

    if (!indexed) {
        hca_k = ggml_view_4d(ctx0, hca_k,
                hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
                hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    }
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = indexed ? ggml_dsv4_indexed_concat(
            ctx0, raw_k, hca_k, inp_hca.segment_ids, n_hca) :
            ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * raw_mask = inp_attn->get_kq_mask();
    ggml_tensor * hca_mask = inp_hca.kq_mask;

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, hca_mask, 0);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_raw_attention(
        llm_graph_input_dsv4_raw * inp_attn,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float kq_scale,
        int il) const {
    GGML_ASSERT(hparams.is_swa(il));

    ggml_tensor * k_rot = inp_attn->self_k_rot;

    if (k_rot) {
        q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
        kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    const llama_kv_cache_dsv4_raw_context * mctx_cur = inp_attn->mctx;

    ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();

    ggml_tensor * k = mctx_cur->get_k(ctx0, il);

    ggml_tensor * out = build_attn_mha(q, k, k, nullptr, kq_mask, sinks, nullptr, kq_scale, il);
    if (k_rot) {
        out = llama_mul_mat_hadamard(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, inp_dsv4, nullptr, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    return build_attention_impl(model, nullptr, inp_mtp, cur, inp_pos, il);
}

ggml_tensor * llama_model_deepseek4::graph::build_attention_impl(
        const llama_model & model,
        llm_graph_input_dsv4 * inp_dsv4,
        llm_graph_input_attn_k_iswa * inp_mtp,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    GGML_ASSERT((inp_dsv4 == nullptr) != (inp_mtp == nullptr));

    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4 ? inp_dsv4->get_raw() : nullptr;

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const bool use_compress_rope = hparams.dsv4_compress_ratios[il] != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(q_pe, "q_pe", il);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];
    GGML_ASSERT(inp_dsv4 || ratio == 0);

    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    ggml_tensor * hca_source_kv   = nullptr;
    ggml_tensor * hca_source_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        hca_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape = layer.attn_comp_ape;

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, ape, inp_dsv4->get_hca().state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);

    }

    if (ratio == DSV4_CSA_RATIO && inp_dsv4->get_csa().state_pos) {
        ggml_tensor * csa_state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = build_lora_mm(layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape = layer.attn_comp_ape;

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, csa_ape, inp_dsv4->get_csa().state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_csa().state_write_idxs);

        const auto * csa_state = inp_dsv4->mctx->get_csa_state();
        const dsv4_state_tensors csa_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_csa(), csa_state, il);
        ggml_tensor * csa_base_kv = dsv4_view_2d(
                ctx0, csa_restored.kv, csa_restored.kv->ne[0], csa_state->get_n_rows(), 0);
        ggml_tensor * csa_base_score = dsv4_view_2d(
                ctx0, csa_restored.score, csa_restored.score->ne[0], csa_state->get_n_rows(), 0);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0, csa_base_kv, csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0, csa_base_score, csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_dsv4->get_csa().state_read_idxs,
                inp_dsv4->get_csa().state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_dsv4->get_csa().k_rot) {
            kv_comp_csa_state = llama_mul_mat_hadamard(ctx0, kv_comp_csa_state, inp_dsv4->get_csa().k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_tensor * csa_k_write = inp_dsv4->mctx->get_csa()->cpy_k(ctx0,
                    kv_comp_csa_state, inp_dsv4->get_csa().state_write_idxs, il);
        cb(csa_k_write, "csa_k_write", il);
        ggml_build_forward_expand(gf, csa_k_write);

        ggml_tensor * csa_snapshot_source_kv = ggml_concat(ctx0,
                csa_restored.kv, csa_state_kv, 1);
        ggml_tensor * csa_snapshot_source_score = ggml_concat(ctx0,
                csa_restored.score, csa_state_score, 1);

        const dsv4_state_tensors csa_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_csa(), csa_state, csa_snapshot_source_kv, csa_snapshot_source_score, il);
        if (csa_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.kv);
        }
        if (csa_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, csa_snapshot.score);
        }

        ggml_tensor * csa_persist_kv = ggml_get_rows(ctx0, csa_state_kv, inp_dsv4->get_csa().state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_dsv4->get_csa().state_persist_src_idxs);

        csa_state_kv = inp_dsv4->mctx->get_csa_state()->cpy_kv(ctx0,
                csa_persist_kv, inp_dsv4->get_csa().state_persist_dst_idxs, il);
        csa_state_score = inp_dsv4->mctx->get_csa_state()->cpy_score(ctx0,
                csa_persist_score, inp_dsv4->get_csa().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = build_lora_mm(layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = build_lora_mm(layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape = layer.indexer_comp_ape;

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, lid_ape, inp_dsv4->get_lid().state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_dsv4->get_lid().state_write_idxs);

        const auto * lid_state = inp_dsv4->mctx->get_lid_state();
        const dsv4_state_tensors lid_restored = dsv4_build_state_restore(
                ctx0, inp_dsv4->get_lid(), lid_state, il);
        ggml_tensor * lid_base_kv = dsv4_view_2d(
                ctx0, lid_restored.kv, lid_restored.kv->ne[0], lid_state->get_n_rows(), 0);
        ggml_tensor * lid_base_score = dsv4_view_2d(
                ctx0, lid_restored.score, lid_restored.score->ne[0], lid_state->get_n_rows(), 0);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0, lid_base_kv, lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0, lid_base_score, lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_dsv4->get_lid().state_read_idxs,
                inp_dsv4->get_lid().state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_dsv4->get_lid().k_rot) {
            kv_comp_lid_state = llama_mul_mat_hadamard(ctx0, kv_comp_lid_state, inp_dsv4->get_lid().k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_tensor * lid_k_write = inp_dsv4->mctx->get_lid()->cpy_k(ctx0,
                    kv_comp_lid_state, inp_dsv4->get_lid().state_write_idxs, il);
        cb(lid_k_write, "lid_k_write", il);
        ggml_build_forward_expand(gf, lid_k_write);

        ggml_tensor * lid_snapshot_source_kv = ggml_concat(ctx0,
                lid_restored.kv, lid_state_kv, 1);
        ggml_tensor * lid_snapshot_source_score = ggml_concat(ctx0,
                lid_restored.score, lid_state_score, 1);

        const dsv4_state_tensors lid_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_lid(), lid_state, lid_snapshot_source_kv, lid_snapshot_source_score, il);
        if (lid_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.kv);
        }
        if (lid_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, lid_snapshot.score);
        }

        ggml_tensor * lid_persist_kv = ggml_get_rows(ctx0, lid_state_kv, inp_dsv4->get_lid().state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_dsv4->get_lid().state_persist_src_idxs);

        lid_state_kv = inp_dsv4->mctx->get_lid_state()->cpy_kv(ctx0,
                lid_persist_kv, inp_dsv4->get_lid().state_persist_dst_idxs, il);
        lid_state_score = inp_dsv4->mctx->get_lid_state()->cpy_score(ctx0,
                lid_persist_score, inp_dsv4->get_lid().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    const llama_dsv4_comp_state * hca_state = nullptr;
    dsv4_state_tensors hca_restored = {};
    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state = inp_dsv4->mctx->get_hca_state();
        hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        ggml_tensor * hca_base_kv = dsv4_view_2d(
                ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
        ggml_tensor * hca_base_score = dsv4_view_2d(
                ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

        hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
        hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_dsv4->get_hca().state_read_idxs,
                inp_dsv4->get_hca().state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_dsv4->get_hca().k_rot) {
            kv_comp_hca = llama_mul_mat_hadamard(ctx0, kv_comp_hca, inp_dsv4->get_hca().k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, inp_dsv4->mctx->get_hca()->cpy_k(ctx0,
                    kv_comp_hca, inp_dsv4->get_hca().state_write_idxs, il));
    }

    if (ratio == DSV4_HCA_RATIO && inp_dsv4->get_hca().state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        if (hca_state == nullptr) {
            hca_state = inp_dsv4->mctx->get_hca_state();
        }
        if (hca_restored.kv == nullptr) {
            hca_restored = dsv4_build_state_restore(ctx0, inp_dsv4->get_hca(), hca_state, il);
        }
        if (hca_source_kv == nullptr || hca_source_score == nullptr) {
            ggml_tensor * hca_base_kv = dsv4_view_2d(
                    ctx0, hca_restored.kv, hca_restored.kv->ne[0], hca_state->get_n_rows(), 0);
            ggml_tensor * hca_base_score = dsv4_view_2d(
                    ctx0, hca_restored.score, hca_restored.score->ne[0], hca_state->get_n_rows(), 0);

            hca_source_kv = ggml_concat(ctx0, hca_base_kv, hca_state_kv, 1);
            hca_source_score = ggml_concat(ctx0, hca_base_score, hca_state_score, 1);
        }

        ggml_tensor * hca_snapshot_source_kv = ggml_concat(ctx0,
                hca_restored.kv, hca_state_kv, 1);
        ggml_tensor * hca_snapshot_source_score = ggml_concat(ctx0,
                hca_restored.score, hca_state_score, 1);

        const dsv4_state_tensors hca_snapshot = dsv4_build_state_snapshot(
                ctx0, inp_dsv4->get_hca(), hca_state, hca_snapshot_source_kv, hca_snapshot_source_score, il);
        if (hca_snapshot.kv != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.kv);
        }
        if (hca_snapshot.score != nullptr) {
            ggml_build_forward_expand(gf, hca_snapshot.score);
        }

        ggml_tensor * hca_persist_kv = ggml_get_rows(ctx0, hca_state_kv, inp_dsv4->get_hca().state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_dsv4->get_hca().state_persist_src_idxs);

        hca_state_kv = inp_dsv4->mctx->get_hca_state()->cpy_kv(ctx0,
                hca_persist_kv, inp_dsv4->get_hca().state_persist_dst_idxs, il);
        hca_state_score = inp_dsv4->mctx->get_hca_state()->cpy_score(ctx0,
                hca_persist_score, inp_dsv4->get_hca().state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    ggml_tensor * out = nullptr;
    if (inp_mtp) {
        out = build_attn(inp_mtp,
                nullptr, nullptr, nullptr,
                q, kv, nullptr, nullptr,
                layer.attn_sinks, nullptr,
                1.0f/sqrtf(float(n_embd_head)), il);
        cb(out, "attn_raw", il);
    } else if (ratio == DSV4_CSA_RATIO &&
            inp_dsv4->get_csa().kq_mask &&
            inp_dsv4->get_lid().kq_mask &&
            inp_dsv4->get_lid().k_rot) {
        out = build_csa_lid_attention(model, inp_dsv4, inp_attn, q, kv, qr, cur, inp_pos, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else if (ratio == DSV4_HCA_RATIO &&
            inp_dsv4->get_hca().kq_mask) {
        out = build_hca_attention(inp_dsv4, inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    } else {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks,
                1.0f/sqrtf(float(n_embd_head)), il);
    }

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

ggml_tensor * llama_model_deepseek4::graph::build_dspark_attention(
        const llama_model & model,
        llm_graph_input_attn_k_iswa * inp_attn,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups = hparams.dsv4_o_group_count;
    const int64_t n_heads_group = n_head/n_groups;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim = n_heads_group*n_embd_head;
    const int64_t nt = cur->ne[1];

    GGML_ASSERT(hparams.dsv4_compress_ratios[il] == 0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());
    GGML_ASSERT(n_head % n_groups == 0);

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "dspark_qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head, 0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "dspark_q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head), 0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "dspark_kv", il);

    ggml_tensor * out = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            q, kv, nullptr, nullptr, layer.attn_sinks, nullptr,
            1.0f/sqrtf(float(n_embd_head)), il);

    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head, 0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));
    out_pe = ggml_rope_ext_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
            freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "dspark_attn_out", il);
    return out;
}

llama_model_deepseek4::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = dsv4_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const auto & layer = model.layers[il];
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count) {
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, res->t_inp_tokens);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp;
        if (params.dsv4_amx_mode == LLM_DSV4_AMX_DISABLED) {
            ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, nullptr, nullptr,
                    layer.ffn_gate_shexp, nullptr, nullptr,
                    layer.ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);
        } else {
            // The direct AMX worker fills this context-owned Metal-shared leaf
            // before the existing routed+shared add is submitted. All layer-local
            // leaf records reuse one external buffer under exact callback ordering.
            ffn_shexp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
            cb(ffn_shexp, "ffn_amx_out", il);

            if (params.dsv4_amx_mode == LLM_DSV4_AMX_VALIDATE) {
                ggml_tensor * reference = build_ffn(cur,
                        layer.ffn_up_shexp, nullptr, nullptr,
                        layer.ffn_gate_shexp, nullptr, nullptr,
                        layer.ffn_down_shexp, nullptr, nullptr,
                        nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
                cb(reference, "ffn_amx_ref", il);

                // Keep both tensors live until after the retained Metal result is
                // available. The callback waits for AMX at reference; this diff is
                // correctness-only and is absent from ordinary coexecution graphs.
                ggml_tensor * validation = ggml_sub(ctx0, reference, ffn_shexp);
                cb(validation, "ffn_amx_validation", il);
                ggml_build_forward_expand(gf, validation);
            }
        }

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = dsv4_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (cparams.embeddings_nextn) {
        ggml_tensor * h_nextn = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        cb(h_nextn, "h_nextn", -1);
        res->t_h_nextn = h_nextn;
        ggml_build_forward_expand(gf, h_nextn);
    }

    if (inp_out_ids) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

llama_model_deepseek4::graph_dspark_encoder::graph_dspark_encoder(
        const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    GGML_ASSERT(layer.nextn.dspark_main_proj && layer.nextn.dspark_main_norm);

    auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = build_lora_mm(layer.nextn.dspark_main_proj, inp->embd);
    cb(cur, "dspark_main_proj", il);
    cur = build_norm(cur, layer.nextn.dspark_main_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "dspark_main_norm", il);

    res->add_input(std::move(inp));
    res->t_h_nextn = cur;
    ggml_build_forward_expand(gf, cur);
}

llama_model_deepseek4::graph_dspark::graph_dspark(
        const llama_model & model,
        const llm_graph_params & params) : graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 3);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();
    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        ggml_tensor * main_x = inp->embd;
        res->add_input(std::move(inp));

        for (uint32_t stage = 0; stage < hparams.n_layer_nextn; ++stage) {
            const int il = hparams.n_layer() + stage;
            const auto & layer = model.layers[il];

            ggml_tensor * kv = build_lora_mm(layer.wkv, main_x);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_tokens,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head_nope));
            kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
            cb(kv, "dspark_main_kv", il);

            if (inp_attn->self_k_rot_swa) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }
            ggml_build_forward_expand(gf,
                    inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = main_x;
        ggml_build_forward_expand(gf, main_x);
        return;
    }

    GGML_ASSERT(cparams.ctx_other != nullptr);
    const llama_model * target = llama_get_model(cparams.ctx_other);
    ggml_tensor * tok_embd = model.tok_embd ? model.tok_embd : target->tok_embd;
    ggml_tensor * output_norm = model.output_norm ? model.output_norm : target->output_norm;
    ggml_tensor * output = model.output ? model.output : target->output;
    GGML_ASSERT(tok_embd && output_norm && output);

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp_tokens);
    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "dspark_hc_init", -1);

    for (uint32_t stage = 0; stage < hparams.n_layer_nextn; ++stage) {
        const int il = hparams.n_layer() + stage;
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(inpL,
                layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_dspark_attention(model, inp_attn, cur, inp_pos, il);
        inpL = build_hc_post(cur, residual, post, comb, il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);

        ggml_tensor * shared_out = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);

        cur = ggml_add(ctx0, moe_out, shared_out);
        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "dspark_stage_out", il);
    }

    const int il_last = hparams.n_layer_all - 1;
    const auto & last = model.layers[il_last];
    ggml_tensor * hc_fn = last.nextn.hc_head_fn ? last.nextn.hc_head_fn : model.hc_head_fn;
    ggml_tensor * hc_base = last.nextn.hc_head_base ? last.nextn.hc_head_base : model.hc_head_base;
    ggml_tensor * hc_scale = last.nextn.hc_head_scale ? last.nextn.hc_head_scale : model.hc_head_scale;
    GGML_ASSERT(hc_fn && hc_base && hc_scale);

    ggml_tensor * cur = build_hc_head(inpL, hc_fn, hc_scale, hc_base);
    cb(cur, "dspark_hc_head", -1);

    // Confidence is trained on the collapsed hidden state before output norm.
    res->t_embd = cur;
    ggml_tensor * norm = build_norm(cur, output_norm, nullptr, LLM_NORM_RMS, -1);
    cur = ggml_mul_mat(ctx0, output, norm);
    cb(cur, "dspark_output", -1);
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);

    dsv4_build_dspark_markov_head(*this, model, inp_tokens);
}

// DSV4's native MTP head is a full, uncompressed-SWA V4 block. Unlike the
// common 4096-wide NextN heads, it consumes and produces all four 4096-wide
// hyper-connection streams so the draft context can continue the trained state.
llama_model_deepseek4::graph_mtp::graph_mtp(
        const llama_model & model,
        const llm_graph_params & params) : graph(params) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "DSV4 MTP requires one NextN block");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t n_embd_h = hparams.n_embd_nextn();

    GGML_ASSERT(n_embd_h == hc*n_embd);
    GGML_ASSERT(layer.nextn.eh_proj);
    GGML_ASSERT(layer.nextn.enorm);
    GGML_ASSERT(layer.nextn.hnorm);
    ggml_tensor * mtp_head_norm = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    ggml_tensor * mtp_hc_head_fn = layer.nextn.hc_head_fn
            ? layer.nextn.hc_head_fn
            : model.hc_head_fn;
    ggml_tensor * mtp_hc_head_base = layer.nextn.hc_head_base
            ? layer.nextn.hc_head_base
            : model.hc_head_base;
    ggml_tensor * mtp_hc_head_scale = layer.nextn.hc_head_scale
            ? layer.nextn.hc_head_scale
            : model.hc_head_scale;
    GGML_ASSERT(mtp_head_norm);
    GGML_ASSERT(mtp_hc_head_fn && mtp_hc_head_base && mtp_hc_head_scale);

    auto inp = std::make_unique<llm_graph_input_embd_h>(n_embd_h);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    // Graph reservation can probe an embedding-only shape. Real DSV4 MTP
    // decoding always supplies token ids plus a separate wide hidden row.
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_h, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd = ubatch.token
            ? ggml_get_rows(ctx0, tok_embd_w, inp->tokens)
            : inp->embd;
    cb(tok_embd, "mtp_tok_embd", il);
    ggml_tensor * h = inp->h;
    res->add_input(std::move(inp));

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();
    ggml_build_forward_expand(gf, inp_attn->self_kq_mask);

    ggml_tensor * e = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e, "mtp_enorm", il);

    ggml_tensor * h3 = ggml_reshape_3d(ctx0, h, n_embd, hc, n_tokens);
    h3 = build_norm(h3, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h3, "mtp_hnorm", il);

    ggml_tensor * e3 = ggml_reshape_3d(ctx0, e, n_embd, 1, n_tokens);
    e3 = ggml_repeat_4d(ctx0, e3, n_embd, hc, n_tokens, 1);
    ggml_tensor * concat = ggml_concat(ctx0, e3, h3, 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * inpL = ggml_mul_mat(ctx0, layer.nextn.eh_proj, concat);
    cb(inpL, "mtp_eh_proj", il);

    ggml_tensor * residual = inpL;
    ggml_tensor * post = nullptr;
    ggml_tensor * comb = nullptr;

    ggml_tensor * cur = build_hc_pre(inpL,
            layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_attn_pre", il);

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);
    cur = build_attention(model, inp_dsv4, cur, inp_pos, il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_hc_attn_post", il);

    residual = inpL;
    cur = build_hc_pre(inpL,
            layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
            &post, &comb, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    ggml_build_forward_expand(gf, residual);
    ggml_build_forward_expand(gf, post);
    ggml_build_forward_expand(gf, comb);

    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_ffn_norm", il);

    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr);
    cb(moe_out, "mtp_ffn_moe_out", il);

    ggml_tensor * ffn_shexp = build_ffn(cur,
            layer.ffn_up_shexp, nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(ffn_shexp, "mtp_ffn_shexp", il);

    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    cb(cur, "mtp_ffn_out", il);

    inpL = build_hc_post(cur, residual, post, comb, il);
    cb(inpL, "mtp_block_out", il);

    ggml_tensor * h_out = ggml_reshape_2d(ctx0, inpL, n_embd_h, n_tokens);
    ggml_tensor * h_capture = h_out;
    if (inp_out_ids && cparams.embeddings_nextn_masked) {
        h_capture = ggml_get_rows(ctx0, h_capture, inp_out_ids);
    }
    cb(h_capture, "h_nextn", -1);
    res->t_h_nextn = h_capture;
    ggml_build_forward_expand(gf, h_capture);

    if (inp_out_ids) {
        h_out = ggml_get_rows(ctx0, h_out, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, h_out, n_embd, hc, n_outputs);
    }

    cur = build_hc_head(inpL,
            mtp_hc_head_fn,
            mtp_hc_head_scale,
            mtp_hc_head_base);
    cb(cur, "mtp_hc_head", -1);

    cur = build_norm(cur, mtp_head_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "mtp_shared_head_norm", -1);
    res->t_embd = cur;

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    GGML_ASSERT(head_w && "DeepSeek-V4 MTP is missing its shared LM head");
    cur = ggml_mul_mat(ctx0, head_w, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

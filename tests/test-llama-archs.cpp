#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-kv-cache-dsv4-accounting.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(LLAMA_DECODE_SUCCESS == 0, "decode success ABI changed");
static_assert(LLAMA_DECODE_KV_LOGICAL_FULL == 1, "logical KV-full ABI changed");
static_assert(LLAMA_DECODE_ABORTED == 2, "decode abort ABI changed");
static_assert(LLAMA_DECODE_KV_PHYSICAL_PRESSURE == 3, "physical pressure result must remain distinct");

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-v/--verbose]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool dsv4_mtp = false) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = arch == LLM_ARCH_DEEPSEEK4 ? 256 : 128;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 64;
        n_layer = dsv4_mtp ? 3 : 2;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    }

    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    if (dsv4_mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                     true);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                        7.0f);
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,            uint32_t(1));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,              uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,       10000.0f);
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,
                dsv4_mtp ? std::vector<uint32_t>({4, 128, 0}) : std::vector<uint32_t>({4, 128}));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                  uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,    uint32_t(1));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,                1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                        uint32_t(0));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

struct test_context_config {
    uint32_t n_ctx       = 0;
    uint32_t n_batch     = 0;
    uint32_t n_ubatch    = 64;
    uint32_t n_seq_max   = 1;
    bool     kv_unified  = false;
    bool     load_mtp    = false;
    enum llama_flash_attn_type flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context_type ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data = nullptr;
};

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const test_context_config & config = {}) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.load_mtp = config.load_mtp;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config.n_ctx;
    if (config.n_batch > 0) {
        ctx_params.n_batch = config.n_batch;
    }
    ctx_params.n_seq_max = config.n_seq_max;
    ctx_params.kv_unified = config.kv_unified;
    ctx_params.ctx_type = config.ctx_type;
    ctx_params.cb_eval = config.cb_eval;
    ctx_params.cb_eval_user_data = config.cb_eval_user_data;
    ctx_params.flash_attn_type = config.flash_attn_type;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = config.n_ubatch;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static void test_minimax_m3_dense_fallback(const size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_MINIMAX_M3, true);
    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    test_context_config config;
    config.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    const size_t device_count = ggml_backend_dev_count();
    GGML_ASSERT(device_count > 0);
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto model_and_ctx = get_model_and_ctx(
                gguf_ctx.get(), nullptr, seed, {dev}, LLAMA_SPLIT_MODE_LAYER, false, config);
        const std::vector<float> logits = get_logits(
                model_and_ctx.first.get(), model_and_ctx.second.get(), tokens);

        GGML_ASSERT(!logits.empty());
        printf("MiniMax-M3 forced-no-FA dense fallback (%s): OK\n", ggml_backend_dev_description(dev));
    }
}

static void test_dsv4_mtp_device(
        const size_t seed, const std::vector<ggml_backend_dev_t> & devices, const char * device_label) {
    const std::vector<llama_token> tokens = get_tokens(8, 128, seed);

    test_context_config trunk_config;
    trunk_config.n_ctx = 256;
    trunk_config.n_batch = 4*tokens.size();
    trunk_config.n_ubatch = tokens.size();

    auto trunk_gguf = get_gguf_ctx(LLM_ARCH_DEEPSEEK4, true);
    auto trunk = get_model_and_ctx(
            trunk_gguf.get(), nullptr, seed, devices, LLAMA_SPLIT_MODE_LAYER, false, trunk_config);
    llama_set_embeddings_nextn(trunk.second.get(), true, false);
    const std::vector<float> logits_trunk = get_logits(trunk.first.get(), trunk.second.get(), tokens);
    const int32_t n_embd_nextn = llama_model_n_embd_nextn(trunk.first.get());
    GGML_ASSERT(n_embd_nextn == 4*llama_model_n_embd(trunk.first.get()));
    const size_t target_prompt_output_bytes = llama_get_output_buffer_size(trunk.second.get());
    const size_t legacy_prompt_output_bytes =
            (tokens.size()*llama_vocab_n_tokens(llama_model_get_vocab(trunk.first.get())) +
             trunk_config.n_batch*(size_t) n_embd_nextn)*sizeof(float);
    const char * compact_disable = std::getenv("LLAMA_DSV4_NEXTN_OUTPUT_COMPACT_DISABLE");
    const bool compact_enabled = !(compact_disable && std::strcmp(compact_disable, "1") == 0);
    if (compact_enabled) {
        GGML_ASSERT(target_prompt_output_bytes < legacy_prompt_output_bytes &&
                "unmasked DSV4 NextN output retained configured batch capacity");
    } else {
        GGML_ASSERT(target_prompt_output_bytes >= legacy_prompt_output_bytes &&
                "DSV4 NextN output compact control did not retain configured batch capacity");
    }
    const float * target_h = llama_get_embeddings_nextn(trunk.second.get());
    GGML_ASSERT(target_h);

    test_context_config mtp_config = trunk_config;
    mtp_config.load_mtp = true;
    auto mtp_gguf = get_gguf_ctx(LLM_ARCH_DEEPSEEK4, true, true);
    auto target = get_model_and_ctx(
            mtp_gguf.get(), nullptr, seed, devices, LLAMA_SPLIT_MODE_LAYER, false, mtp_config);

    const std::vector<float> logits_mtp_target = get_logits(target.first.get(), target.second.get(), tokens);
    const double target_nmse = nmse(logits_trunk, logits_mtp_target);
    GGML_ASSERT(target_nmse < 1e-12 && "bundled MTP metadata changed DSV4 trunk logits");

    GGML_ASSERT(n_embd_nextn == llama_model_n_embd_nextn(target.first.get()));

    llama_context_params draft_params = llama_context_default_params();
    draft_params.n_ctx = 256;
    draft_params.n_batch = tokens.size();
    draft_params.n_ubatch = tokens.size();
    draft_params.n_seq_max = 1;
    draft_params.n_threads = 4;
    draft_params.n_threads_batch = 4;
    draft_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    draft_params.n_rs_seq = 1;

    llama_context_ptr draft(llama_init_from_model(target.first.get(), draft_params));
    GGML_ASSERT(draft);
    llama_set_embeddings_nextn(draft.get(), true, true);

    llama_batch batch = llama_batch_init(tokens.size(), n_embd_nextn, 1);
    std::vector<llama_token> draft_tokens(tokens);
    batch.token = draft_tokens.data();
    batch.n_tokens = tokens.size();
    std::memset(batch.embd, 0, (size_t) tokens.size()*n_embd_nextn*sizeof(float));

    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
        if (i > 0) {
            std::memcpy(
                    batch.embd + i*(size_t) n_embd_nextn,
                    target_h + (i - 1)*(size_t) n_embd_nextn,
                    (size_t) n_embd_nextn*sizeof(float));
        }
    }

    const auto make_draft_context = [&]() {
        llama_context_ptr result(llama_init_from_model(target.first.get(), draft_params));
        GGML_ASSERT(result);
        llama_set_embeddings_nextn(result.get(), true, true);
        return result;
    };

    const auto decode_mtp_rows = [&](llama_context * ctx, size_t pos_beg, size_t pos_end, llama_token last_token = LLAMA_TOKEN_NULL) {
        GGML_ASSERT(pos_beg < pos_end && pos_end <= tokens.size());

        const size_t n_rows = pos_end - pos_beg;
        llama_batch rows = llama_batch_init(n_rows, n_embd_nextn, 1);
        std::vector<llama_token> row_tokens(tokens.begin() + pos_beg, tokens.begin() + pos_end);
        if (last_token != LLAMA_TOKEN_NULL) {
            row_tokens.back() = last_token;
        }
        rows.token = row_tokens.data();
        rows.n_tokens = n_rows;

        for (size_t i = 0; i < n_rows; ++i) {
            const size_t pos = pos_beg + i;
            rows.pos[i] = pos;
            rows.n_seq_id[i] = 1;
            rows.seq_id[i][0] = 0;
            rows.logits[i] = i + 1 == n_rows;

            float * h = rows.embd + i*(size_t) n_embd_nextn;
            if (pos == 0) {
                std::memset(h, 0, (size_t) n_embd_nextn*sizeof(float));
            } else {
                std::memcpy(h, target_h + (pos - 1)*(size_t) n_embd_nextn,
                        (size_t) n_embd_nextn*sizeof(float));
            }
        }

        GGML_ASSERT(llama_decode(ctx, rows) == 0);

        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(target.first.get()));
        const float * logits = llama_get_logits_ith(ctx, -1);
        GGML_ASSERT(logits);
        std::vector<float> result(logits, logits + n_vocab);

        rows.token = nullptr;
        llama_batch_free(rows);
        return result;
    };

    // The production control removes the separately evaluated committed row
    // and later catches up [P, P+1] in one batch. The optimized path retains P
    // and catches up only P+1. Verify both accepted continuation and rejection
    // rollback against the control geometry on the synthetic DSV4 MTP model.
    auto accept_control = make_draft_context();
    auto accept_reuse   = make_draft_context();
    decode_mtp_rows(accept_control.get(), 0, 5);
    decode_mtp_rows(accept_reuse.get(),   0, 5);

    const std::vector<float> accept_control_p6 = decode_mtp_rows(accept_control.get(), 5, 7);
    decode_mtp_rows(accept_reuse.get(), 5, 6);
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(accept_reuse.get()), 0) == 5);
    const std::vector<float> accept_reuse_p6 = decode_mtp_rows(accept_reuse.get(), 6, 7);
    const double retain_nmse = nmse(accept_control_p6, accept_reuse_p6);
    GGML_ASSERT(retain_nmse < 1e-10 && "retained DSV4 MTP row changed proposal catch-up logits");

    const std::vector<float> accept_control_p7 = decode_mtp_rows(accept_control.get(), 7, 8);
    const std::vector<float> accept_reuse_p7   = decode_mtp_rows(accept_reuse.get(),   7, 8);
    const double accept_nmse = nmse(accept_control_p7, accept_reuse_p7);
    GGML_ASSERT(accept_nmse < 1e-10 && "retained DSV4 MTP proposal changed accepted continuation logits");

    auto reject_control = make_draft_context();
    auto reject_reuse   = make_draft_context();
    decode_mtp_rows(reject_control.get(), 0, 5);
    decode_mtp_rows(reject_reuse.get(),   0, 5);
    decode_mtp_rows(reject_control.get(), 5, 7);
    decode_mtp_rows(reject_reuse.get(),   5, 6);
    decode_mtp_rows(reject_reuse.get(),   6, 7);

    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(reject_control.get()), 0, 6, -1));
    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(reject_reuse.get()),   0, 6, -1));
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(reject_control.get()), 0) == 5);
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(reject_reuse.get()),   0) == 5);

    const llama_token replacement = (tokens[6] + 1) % 128;
    const std::vector<float> reject_control_p6 = decode_mtp_rows(reject_control.get(), 6, 7, replacement);
    const std::vector<float> reject_reuse_p6   = decode_mtp_rows(reject_reuse.get(),   6, 7, replacement);
    const double reject_nmse = nmse(reject_control_p6, reject_reuse_p6);
    GGML_ASSERT(reject_nmse < 1e-10 && "retained DSV4 MTP row changed rejection rollback logits");

    GGML_ASSERT(llama_decode(draft.get(), batch) == 0);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const float * logits = llama_get_logits_ith(draft.get(), i);
        const float * h = llama_get_embeddings_nextn_ith(draft.get(), i);
        GGML_ASSERT(logits && h);
        for (int32_t j = 0; j < llama_vocab_n_tokens(llama_model_get_vocab(target.first.get())); ++j) {
            GGML_ASSERT(std::isfinite(logits[j]));
        }
        for (int32_t j = 0; j < n_embd_nextn; ++j) {
            GGML_ASSERT(std::isfinite(h[j]));
        }
    }

    // The prompt-sized target buffer must release its VM-backed excess when
    // generation transitions to a much smaller verifier batch.
    llama_memory_clear(llama_get_memory(trunk.second.get()), true);
    llama_batch one = llama_batch_init(1, 0, 1);
    common_batch_add(one, tokens[0], 0, {0}, true);
    GGML_ASSERT(llama_decode(trunk.second.get(), one) == 0);
    const size_t target_decode_output_bytes = llama_get_output_buffer_size(trunk.second.get());
    if (compact_enabled) {
        GGML_ASSERT(target_decode_output_bytes < target_prompt_output_bytes &&
                "DSV4 target output buffer did not shrink after prompt processing");
    } else {
        GGML_ASSERT(target_decode_output_bytes == target_prompt_output_bytes &&
                "DSV4 target output compact control did not retain prompt capacity");
    }
    GGML_ASSERT(std::isfinite(llama_get_logits_ith(trunk.second.get(), 0)[0]));
    GGML_ASSERT(std::isfinite(llama_get_embeddings_nextn_ith(trunk.second.get(), 0)[0]));
    llama_batch_free(one);

    batch.token = nullptr;
    llama_batch_free(batch);
    printf("  DSV4 MTP graph (%s): target NMSE %.3e, hidden width %d, finite draft rows %zu, "
           "retain/accept/reject NMSE %.3e/%.3e/%.3e, target output bytes prompt/decode %zu/%zu (%s)\n",
            device_label, target_nmse, n_embd_nextn, tokens.size(), retain_nmse, accept_nmse, reject_nmse,
            target_prompt_output_bytes, target_decode_output_bytes, compact_enabled ? "compact" : "control");
}

static void test_dsv4_mtp(const size_t seed) {
    test_dsv4_mtp_device(seed, {}, "CPU");

    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu) {
        test_dsv4_mtp_device(seed, { gpu }, ggml_backend_dev_description(gpu));
    }
}

static bool dsv4_test_has_elastic_pages(ggml_backend_dev_t dev, uint32_t n_seq_max) {
    using sparse_buft_fn = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t, uint32_t);
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto * get_buft = reg ? (sparse_buft_fn) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_metal_dsv4_sparse_buffer_type") : nullptr;
    return get_buft && get_buft(dev, n_seq_max) != nullptr;
}

struct dsv4_test_context {
    llama_model_ptr   model;
    llama_context_ptr lctx;
    llama_batch       batch = {};
    uint32_t          n_vocab;

    dsv4_test_context(
            size_t seed,
            ggml_backend_dev_t dev,
            uint32_t n_seq_max,
            bool kv_unified,
            uint32_t batch_capacity,
            uint32_t max_seq_ids = 1,
            uint32_t n_ctx_total = 0,
            uint32_t n_ubatch = 0,
            ggml_backend_sched_eval_callback cb_eval = nullptr,
            void * cb_eval_user_data = nullptr) {
        test_context_config config;
        config.n_ctx = n_ctx_total > 0 ? n_ctx_total : 256*n_seq_max;
        config.n_batch = batch_capacity;
        config.n_seq_max = n_seq_max;
        config.kv_unified = kv_unified;
        config.n_ubatch = n_ubatch > 0 ? n_ubatch : batch_capacity;
        config.cb_eval = cb_eval;
        config.cb_eval_user_data = cb_eval_user_data;

        gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_DEEPSEEK4, true);
        auto loaded = get_model_and_ctx(
                gguf_ctx.get(), nullptr, seed, {dev}, LLAMA_SPLIT_MODE_LAYER, false, config);

        model = std::move(loaded.first);
        lctx = std::move(loaded.second);
        GGML_ASSERT(llama_n_ctx(lctx.get()) == config.n_ctx);
        const uint32_t expected_seq_ctx = kv_unified && dsv4_test_has_elastic_pages(dev, n_seq_max)
            ? config.n_ctx
            : config.n_ctx/n_seq_max;
        GGML_ASSERT(llama_n_ctx_seq(lctx.get()) == expected_seq_ctx);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        batch = llama_batch_init(batch_capacity, 0, max_seq_ids);
    }

    ~dsv4_test_context() {
        llama_batch_free(batch);
    }

    void clear_batch() {
        batch.n_tokens = 0;
    }

    void add(llama_token token, llama_pos pos, const std::vector<llama_seq_id> & seq_ids) {
        common_batch_add(batch, token, pos, seq_ids, true);
    }

    void decode(const char * label) {
        if (llama_decode(lctx.get(), batch) != 0) {
            throw std::runtime_error(std::string("failed to decode DSV4 ") + label + " batch");
        }
    }

    void append_logits(int32_t output_idx, std::vector<float> & dst) {
        const float * logits = llama_get_logits_ith(lctx.get(), output_idx);
        if (logits == nullptr) {
            throw std::runtime_error("failed to retrieve DSV4 test logits");
        }

        dst.insert(dst.end(), logits, logits + n_vocab);
    }
};

static bool dsv4_count_graph_nodes(ggml_tensor * /* tensor */, bool ask, void * user_data) {
    if (!ask) {
        ++*static_cast<uint32_t *>(user_data);
    }
    return true;
}

static std::vector<uint8_t> dsv4_sequence_state(llama_context * lctx, llama_seq_id seq_id) {
    std::vector<uint8_t> result(llama_state_seq_get_size(lctx, seq_id));
    const size_t written = llama_state_seq_get_data(lctx, result.data(), result.size(), seq_id);
    if (written != result.size()) {
        throw std::runtime_error("failed to capture DSV4 sequence state");
    }
    return result;
}

static bool dsv4_family_usage_equal(
        const llama_dsv4_family_usage & lhs,
        const llama_dsv4_family_usage & rhs) {
    if (lhs.family != rhs.family ||
            lhs.placement_sparse != rhs.placement_sparse ||
            lhs.logical_capacity_rows != rhs.logical_capacity_rows ||
            lhs.logical_mapped_rows != rhs.logical_mapped_rows ||
            lhs.sequence_mapped_rows != rhs.sequence_mapped_rows ||
            lhs.pools.size() != rhs.pools.size() ||
            !dsv4_sparse_usage_equal(lhs.total, rhs.total)) {
        return false;
    }
    for (size_t i = 0; i < lhs.pools.size(); ++i) {
        if (!dsv4_sparse_usage_equal(lhs.pools[i], rhs.pools[i])) {
            return false;
        }
    }
    return true;
}

static bool dsv4_usage_snapshot_equal(
        const llama_dsv4_memory_usage_snapshot & lhs,
        const llama_dsv4_memory_usage_snapshot & rhs) {
    if (!dsv4_sparse_usage_equal(lhs.sparse_total, rhs.sparse_total) ||
            lhs.limiting_family != rhs.limiting_family ||
            lhs.limiting_family_mask != rhs.limiting_family_mask ||
            lhs.limiting_pool_id != rhs.limiting_pool_id ||
            lhs.limiting_available_pages != rhs.limiting_available_pages) {
        return false;
    }
    for (size_t i = 0; i < lhs.families.size(); ++i) {
        if (!dsv4_family_usage_equal(lhs.families[i], rhs.families[i])) {
            return false;
        }
    }
    return true;
}

static bool test_dsv4_physical_pressure(size_t seed, ggml_backend_dev_t dev) {
    if (!dsv4_test_has_elastic_pages(dev, 2)) {
        return true;
    }

    uint32_t graph_nodes = 0;
    dsv4_test_context test(seed, dev, 2, true, 4, 1, 256, 1,
            dsv4_count_graph_nodes, &graph_nodes);
    GGML_ASSERT(llama_n_ubatch(test.lctx.get()) == 1);
    auto * memory = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(test.lctx.get()));
    GGML_ASSERT(memory != nullptr);

    const auto tokens = get_tokens(4, 128, seed + 20);
    test.add(tokens[0], 0, { 0 });
    test.add(tokens[1], 0, { 1 });
    test.decode("physical-pressure baseline");

    std::vector<float> logits_before;
    logits_before.reserve(2*test.n_vocab);
    test.append_logits(0, logits_before);
    test.append_logits(1, logits_before);

    const auto usage_before = memory->memory_usage_snapshot();
    const auto seq_0_before = dsv4_sequence_state(test.lctx.get(), 0);
    const auto seq_1_before = dsv4_sequence_state(test.lctx.get(), 1);
    const llama_pos seq_0_min = llama_memory_seq_pos_min(llama_get_memory(test.lctx.get()), 0);
    const llama_pos seq_0_max = llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 0);
    const llama_pos seq_1_min = llama_memory_seq_pos_min(llama_get_memory(test.lctx.get()), 1);
    const llama_pos seq_1_max = llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 1);

    test.clear_batch();
    test.add(tokens[2], 1, { 0 });
    test.add(tokens[3], 1, { 1 });
    const std::array<llama_token, 2> batch_tokens = { test.batch.token[0], test.batch.token[1] };
    const std::array<llama_pos, 2> batch_positions = { test.batch.pos[0], test.batch.pos[1] };
    const std::array<llama_seq_id, 2> batch_sequences = {
        test.batch.seq_id[0][0], test.batch.seq_id[1][0]
    };
    const std::array<int8_t, 2> batch_outputs = { test.batch.logits[0], test.batch.logits[1] };

    graph_nodes = 0;
    llama_kv_cache_dsv4_test_inject_physical_pressure(1);
    const int32_t pressure_result = llama_decode(test.lctx.get(), test.batch);
    GGML_ASSERT(pressure_result == LLAMA_DECODE_KV_PHYSICAL_PRESSURE);
    GGML_ASSERT(graph_nodes == 0 && "physical pressure reached graph submission");

    llama_kv_pressure_info pressure = {};
    GGML_ASSERT(llama_get_last_kv_pressure(test.lctx.get(), &pressure));
    GGML_ASSERT(pressure.limiting_family < LLAMA_DSV4_MEMORY_FAMILY_COUNT);
    GGML_ASSERT(pressure.limiting_family_mask != 0 && pressure.limiting_pool_id != 0);
    GGML_ASSERT(pressure.reserved_pages <= pressure.free_pages);
    GGML_ASSERT(pressure.physical_pages >= pressure.free_pages);

    GGML_ASSERT(dsv4_usage_snapshot_equal(usage_before, memory->memory_usage_snapshot()));
    GGML_ASSERT(seq_0_before == dsv4_sequence_state(test.lctx.get(), 0));
    GGML_ASSERT(seq_1_before == dsv4_sequence_state(test.lctx.get(), 1));
    GGML_ASSERT(llama_memory_seq_pos_min(llama_get_memory(test.lctx.get()), 0) == seq_0_min);
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 0) == seq_0_max);
    GGML_ASSERT(llama_memory_seq_pos_min(llama_get_memory(test.lctx.get()), 1) == seq_1_min);
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 1) == seq_1_max);

    const std::array<llama_token, 2> batch_tokens_after = { test.batch.token[0], test.batch.token[1] };
    const std::array<llama_pos, 2> batch_positions_after = { test.batch.pos[0], test.batch.pos[1] };
    const std::array<llama_seq_id, 2> batch_sequences_after = {
        test.batch.seq_id[0][0], test.batch.seq_id[1][0]
    };
    const std::array<int8_t, 2> batch_outputs_after = { test.batch.logits[0], test.batch.logits[1] };
    GGML_ASSERT(batch_tokens == batch_tokens_after);
    GGML_ASSERT(batch_positions == batch_positions_after);
    GGML_ASSERT(batch_sequences == batch_sequences_after);
    GGML_ASSERT(batch_outputs == batch_outputs_after);

    std::vector<float> logits_after;
    logits_after.reserve(logits_before.size());
    test.append_logits(0, logits_after);
    test.append_logits(1, logits_after);
    GGML_ASSERT(logits_before == logits_after);

    GGML_ASSERT(llama_decode(test.lctx.get(), test.batch) == LLAMA_DECODE_SUCCESS);
    GGML_ASSERT(!llama_get_last_kv_pressure(test.lctx.get(), &pressure));
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 0) == 1);
    GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(test.lctx.get()), 1) == 1);
    return true;
}

static std::vector<float> run_dsv4_isolated(
        size_t seed,
        ggml_backend_dev_t dev,
        const std::vector<llama_token> & tokens,
        uint32_t n_prompt) {
    dsv4_test_context test(seed, dev, 1, false, 256);

    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(tokens[pos], pos, {0});
    }

    std::vector<float> result;
    result.reserve(tokens.size()*test.n_vocab);

    test.decode("isolated prompt");
    for (uint32_t i = 0; i < n_prompt; ++i) {
        test.append_logits(i, result);
    }

    for (uint32_t pos = n_prompt; pos < tokens.size(); ++pos) {
        test.clear_batch();
        test.add(tokens[pos], pos, {0});
        test.decode("isolated decode");
        test.append_logits(0, result);
    }

    return result;
}

struct dsv4_parallel_result {
    std::vector<float> seq_a;
    std::vector<float> seq_b;
};

static dsv4_parallel_result run_dsv4_parallel(
        size_t seed,
        ggml_backend_dev_t dev,
        const std::vector<llama_token> & tokens_a,
        const std::vector<llama_token> & tokens_b,
        uint32_t n_prompt,
        bool kv_unified,
        uint32_t n_seq_max,
        llama_seq_id seq_a,
        llama_seq_id seq_b) {
    GGML_ASSERT(tokens_a.size() == tokens_b.size());
    GGML_ASSERT(seq_a >= 0 && (uint32_t) seq_a < n_seq_max);
    GGML_ASSERT(seq_b >= 0 && (uint32_t) seq_b < n_seq_max);

    dsv4_test_context test(seed, dev, n_seq_max, kv_unified, 2*n_prompt);
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(tokens_a[pos], pos, {seq_a});
    }
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(tokens_b[pos], pos, {seq_b});
    }

    dsv4_parallel_result result;
    result.seq_a.reserve(tokens_a.size()*test.n_vocab);
    result.seq_b.reserve(tokens_b.size()*test.n_vocab);

    test.decode("parallel prompt");
    for (uint32_t i = 0; i < n_prompt; ++i) {
        test.append_logits(i, result.seq_a);
        test.append_logits(n_prompt + i, result.seq_b);
    }

    for (uint32_t pos = n_prompt; pos < tokens_a.size(); ++pos) {
        test.clear_batch();
        test.add(tokens_a[pos], pos, {seq_a});
        test.add(tokens_b[pos], pos, {seq_b});
        test.decode("parallel decode");
        test.append_logits(0, result.seq_a);
        test.append_logits(1, result.seq_b);
    }

    return result;
}

enum class dsv4_prefix_setup {
    COUPLED,
    COPIED,
    RESTORED,
};

static dsv4_parallel_result run_dsv4_shared_prefix(
        size_t seed,
        ggml_backend_dev_t dev,
        const std::vector<llama_token> & tokens_a,
        const std::vector<llama_token> & tokens_b,
        uint32_t n_prompt,
        bool kv_unified,
        dsv4_prefix_setup setup) {
    GGML_ASSERT(tokens_a.size() == tokens_b.size());
    GGML_ASSERT(std::equal(tokens_a.begin(), tokens_a.begin() + n_prompt, tokens_b.begin()));

    const uint32_t max_seq_ids = setup == dsv4_prefix_setup::COUPLED ? 2 : 1;
    dsv4_test_context test(seed, dev, 2, kv_unified, n_prompt, max_seq_ids);
    const std::vector<llama_seq_id> prefix_seq_ids =
            setup == dsv4_prefix_setup::COUPLED ? std::vector<llama_seq_id>{0, 1} : std::vector<llama_seq_id>{0};

    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(tokens_a[pos], pos, prefix_seq_ids);
    }

    dsv4_parallel_result result;
    result.seq_a.reserve(tokens_a.size()*test.n_vocab);
    result.seq_b.reserve(tokens_b.size()*test.n_vocab);

    test.decode("shared prefix");
    for (uint32_t i = 0; i < n_prompt; ++i) {
        test.append_logits(i, result.seq_a);
        test.append_logits(i, result.seq_b);
    }

    if (setup == dsv4_prefix_setup::COPIED) {
        llama_memory_seq_cp(llama_get_memory(test.lctx.get()), 0, 1, -1, -1);
    } else if (setup == dsv4_prefix_setup::RESTORED) {
        std::vector<uint8_t> state(llama_state_seq_get_size(test.lctx.get(), 0));
        const size_t n_copied = llama_state_seq_get_data(
                test.lctx.get(), state.data(), state.size(), 0);
        if (n_copied != state.size()) {
            throw std::runtime_error("failed to save complete DSV4 sequence state");
        }

        const size_t n_restored = llama_state_seq_set_data(
                test.lctx.get(), state.data(), state.size(), 1);
        if (n_restored != state.size()) {
            throw std::runtime_error("failed to restore complete DSV4 sequence state");
        }
    }

    for (uint32_t pos = n_prompt; pos < tokens_a.size(); ++pos) {
        test.clear_batch();
        test.add(tokens_a[pos], pos, {0});
        test.add(tokens_b[pos], pos, {1});
        test.decode("shared-prefix decode");
        test.append_logits(0, result.seq_a);
        test.append_logits(1, result.seq_b);
    }

    return result;
}

static dsv4_parallel_result run_dsv4_asymmetric(
        size_t seed,
        ggml_backend_dev_t dev,
        const std::vector<llama_token> & tokens_a,
        const std::vector<llama_token> & tokens_b,
        uint32_t n_prompt_a,
        uint32_t n_prompt_b,
        bool kv_unified) {
    GGML_ASSERT(tokens_a.size() - n_prompt_a == tokens_b.size() - n_prompt_b);

    dsv4_test_context test(seed, dev, 2, kv_unified, n_prompt_a + n_prompt_b);
    for (uint32_t pos = 0; pos < n_prompt_a; ++pos) {
        test.add(tokens_a[pos], pos, {0});
    }
    for (uint32_t pos = 0; pos < n_prompt_b; ++pos) {
        test.add(tokens_b[pos], pos, {1});
    }

    dsv4_parallel_result result;
    result.seq_a.reserve(tokens_a.size()*test.n_vocab);
    result.seq_b.reserve(tokens_b.size()*test.n_vocab);

    test.decode("asymmetric prompt");
    for (uint32_t i = 0; i < n_prompt_a; ++i) {
        test.append_logits(i, result.seq_a);
    }
    for (uint32_t i = 0; i < n_prompt_b; ++i) {
        test.append_logits(n_prompt_a + i, result.seq_b);
    }

    const uint32_t n_decode = tokens_a.size() - n_prompt_a;
    for (uint32_t i = 0; i < n_decode; ++i) {
        test.clear_batch();
        test.add(tokens_a[n_prompt_a + i], n_prompt_a + i, {0});
        test.add(tokens_b[n_prompt_b + i], n_prompt_b + i, {1});
        test.decode("asymmetric decode");
        test.append_logits(0, result.seq_a);
        test.append_logits(1, result.seq_b);
    }

    return result;
}

static dsv4_parallel_result run_dsv4_reused_slot(
        size_t seed,
        ggml_backend_dev_t dev,
        const std::vector<llama_token> & discarded,
        const std::vector<llama_token> & survivor,
        const std::vector<llama_token> & replacement,
        uint32_t n_prompt,
        uint32_t n_decode_before_reuse,
        bool kv_unified) {
    GGML_ASSERT(discarded.size() >= n_prompt + n_decode_before_reuse);
    GGML_ASSERT(survivor.size() == replacement.size() + n_decode_before_reuse);

    dsv4_test_context test(seed, dev, 2, kv_unified, 2*n_prompt);
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(discarded[pos], pos, {0});
    }
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(survivor[pos], pos, {1});
    }

    dsv4_parallel_result result;
    result.seq_a.reserve(replacement.size()*test.n_vocab);
    result.seq_b.reserve(survivor.size()*test.n_vocab);

    test.decode("reuse prompt");
    for (uint32_t i = 0; i < n_prompt; ++i) {
        test.append_logits(n_prompt + i, result.seq_b);
    }

    for (uint32_t i = 0; i < n_decode_before_reuse; ++i) {
        const uint32_t pos = n_prompt + i;
        test.clear_batch();
        test.add(discarded[pos], pos, {0});
        test.add(survivor[pos], pos, {1});
        test.decode("pre-reuse decode");
        test.append_logits(1, result.seq_b);
    }

    if (!llama_memory_seq_rm(llama_get_memory(test.lctx.get()), 0, -1, -1)) {
        throw std::runtime_error("failed to remove DSV4 sequence before slot reuse");
    }

    test.clear_batch();
    for (uint32_t pos = 0; pos < n_prompt; ++pos) {
        test.add(replacement[pos], pos, {0});
    }
    test.decode("replacement prompt");
    for (uint32_t i = 0; i < n_prompt; ++i) {
        test.append_logits(i, result.seq_a);
    }

    const uint32_t n_decode_after_reuse = replacement.size() - n_prompt;
    for (uint32_t i = 0; i < n_decode_after_reuse; ++i) {
        const uint32_t replacement_pos = n_prompt + i;
        const uint32_t survivor_pos = n_prompt + n_decode_before_reuse + i;
        test.clear_batch();
        test.add(replacement[replacement_pos], replacement_pos, {0});
        test.add(survivor[survivor_pos], survivor_pos, {1});
        test.decode("post-reuse decode");
        test.append_logits(0, result.seq_a);
        test.append_logits(1, result.seq_b);
    }

    return result;
}

static bool compare_dsv4_trace(
        const char * mode,
        const char * sequence,
        const std::vector<float> & expected,
        const std::vector<float> & actual) {
    if (expected.size() != actual.size()) {
        fprintf(stderr, "DSV4 %s %s trace size mismatch: %zu != %zu\n",
                mode, sequence, expected.size(), actual.size());
        return false;
    }

    double sum_squared_error = 0.0;
    double sum_squared_ref = 0.0;
    double max_abs_error = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double error = (double) actual[i] - expected[i];
        sum_squared_error += error*error;
        sum_squared_ref += (double) expected[i]*expected[i];
        max_abs_error = std::max(max_abs_error, std::abs(error));
    }

    const double trace_nmse = sum_squared_ref > 0.0 ? sum_squared_error/sum_squared_ref : sum_squared_error;
    printf("DSV4 %-7s seq %s: NMSE = %.3e, max abs = %.3e\n", mode, sequence, trace_nmse, max_abs_error);

    return trace_nmse <= 1.0e-8 && max_abs_error <= 1.0e-5;
}

static bool test_dsv4_elastic_borrow(size_t seed, ggml_backend_dev_t dev) {
    static constexpr uint32_t n_ctx_total = 512;
    static constexpr uint32_t n_tokens = 300;

    if (!dsv4_test_has_elastic_pages(dev, 2)) {
        return true;
    }

    const std::vector<llama_token> tokens = get_tokens(n_tokens, 128, seed + 10);
    const auto run = [&](uint32_t n_seq_max, bool kv_unified) {
        dsv4_test_context test(seed, dev, n_seq_max, kv_unified, n_tokens, 1, n_ctx_total);
        for (uint32_t pos = 0; pos < n_tokens; ++pos) {
            test.add(tokens[pos], pos, {0});
        }
        test.decode("elastic borrowing");

        std::vector<float> result;
        result.reserve(test.n_vocab);
        test.append_logits(n_tokens - 1, result);
        return result;
    };

    const std::vector<float> isolated = run(1, false);
    const std::vector<float> elastic  = run(2, true);
    return compare_dsv4_trace("elastic-borrow", "A", isolated, elastic);
}

static bool test_dsv4_parallel_device(size_t seed, ggml_backend_dev_t dev) {
    static constexpr uint32_t n_vocab = 128;
    static constexpr uint32_t n_prompt = 128;
    static constexpr uint32_t n_decode = 8;

    const std::vector<llama_token> tokens_a = get_tokens(n_prompt + n_decode, n_vocab, seed + 1);
    const std::vector<llama_token> tokens_b = get_tokens(n_prompt + n_decode, n_vocab, seed + 2);

    const std::vector<float> isolated_a = run_dsv4_isolated(seed, dev, tokens_a, n_prompt);
    const std::vector<float> isolated_b = run_dsv4_isolated(seed, dev, tokens_b, n_prompt);

    struct parallel_case {
        const char * name;
        bool kv_unified;
        uint32_t n_seq_max;
        llama_seq_id seq_a;
        llama_seq_id seq_b;
    };

    const parallel_case cases[] = {
        { "split-contiguous",   false, 2, 0, 1 },
        { "requested-unified-contiguous",          true,  2, 0, 1 },
        { "split-sparse",       false, 4, 0, 2 },
        { "requested-unified-sparse",              true,  4, 0, 2 },
    };

    bool all_ok = true;
    all_ok = test_dsv4_elastic_borrow(seed, dev) && all_ok;
    all_ok = test_dsv4_physical_pressure(seed, dev) && all_ok;
    for (const parallel_case & test_case : cases) {
        const dsv4_parallel_result parallel = run_dsv4_parallel(
                seed, dev, tokens_a, tokens_b, n_prompt,
                test_case.kv_unified, test_case.n_seq_max, test_case.seq_a, test_case.seq_b);
        all_ok = compare_dsv4_trace(test_case.name, "A", isolated_a, parallel.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(test_case.name, "B", isolated_b, parallel.seq_b) && all_ok;
    }

    std::vector<llama_token> coupled_a = get_tokens(n_prompt + n_decode, n_vocab, seed + 3);
    std::vector<llama_token> coupled_b = coupled_a;
    const std::vector<llama_token> suffix_b = get_tokens(n_decode, n_vocab, seed + 4);
    std::copy(suffix_b.begin(), suffix_b.end(), coupled_b.begin() + n_prompt);

    const std::vector<float> isolated_coupled_a = run_dsv4_isolated(seed, dev, coupled_a, n_prompt);
    const std::vector<float> isolated_coupled_b = run_dsv4_isolated(seed, dev, coupled_b, n_prompt);

    for (bool kv_unified : { false, true }) {
        const char * mode = kv_unified ?
            "requested-unified-coupled" : "split-coupled";
        const dsv4_parallel_result coupled = run_dsv4_shared_prefix(
                seed, dev, coupled_a, coupled_b, n_prompt, kv_unified, dsv4_prefix_setup::COUPLED);
        all_ok = compare_dsv4_trace(mode, "A", isolated_coupled_a, coupled.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(mode, "B", isolated_coupled_b, coupled.seq_b) && all_ok;
    }

    for (bool kv_unified : { false, true }) {
        const char * mode = kv_unified ?
            "requested-unified-copied" : "split-copied";
        const dsv4_parallel_result copied = run_dsv4_shared_prefix(
                seed, dev, coupled_a, coupled_b, n_prompt, kv_unified, dsv4_prefix_setup::COPIED);
        all_ok = compare_dsv4_trace(mode, "A", isolated_coupled_a, copied.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(mode, "B", isolated_coupled_b, copied.seq_b) && all_ok;
    }

    for (bool kv_unified : { false, true }) {
        const char * mode = kv_unified ?
            "requested-unified-restored" : "split-restored";
        const dsv4_parallel_result restored = run_dsv4_shared_prefix(
                seed, dev, coupled_a, coupled_b, n_prompt, kv_unified, dsv4_prefix_setup::RESTORED);
        all_ok = compare_dsv4_trace(mode, "A", isolated_coupled_a, restored.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(mode, "B", isolated_coupled_b, restored.seq_b) && all_ok;
    }

    static constexpr uint32_t n_short_prompt = 64;
    const std::vector<llama_token> asymmetric_a = get_tokens(n_prompt + n_decode, n_vocab, seed + 5);
    const std::vector<llama_token> asymmetric_b = get_tokens(n_short_prompt + n_decode, n_vocab, seed + 6);
    const std::vector<float> isolated_asymmetric_a = run_dsv4_isolated(seed, dev, asymmetric_a, n_prompt);
    const std::vector<float> isolated_asymmetric_b = run_dsv4_isolated(seed, dev, asymmetric_b, n_short_prompt);

    for (bool kv_unified : { false, true }) {
        const char * mode = kv_unified ?
            "requested-unified-asymmetric" : "split-asymmetric";
        const dsv4_parallel_result asymmetric = run_dsv4_asymmetric(
                seed, dev, asymmetric_a, asymmetric_b, n_prompt, n_short_prompt, kv_unified);
        all_ok = compare_dsv4_trace(mode, "A", isolated_asymmetric_a, asymmetric.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(mode, "B", isolated_asymmetric_b, asymmetric.seq_b) && all_ok;
    }

    static constexpr uint32_t n_decode_before_reuse = 4;
    const std::vector<llama_token> discarded = get_tokens(
            n_prompt + n_decode_before_reuse, n_vocab, seed + 7);
    const std::vector<llama_token> survivor = get_tokens(
            n_prompt + n_decode_before_reuse + n_decode, n_vocab, seed + 8);
    const std::vector<llama_token> replacement = get_tokens(
            n_prompt + n_decode, n_vocab, seed + 9);
    const std::vector<float> isolated_survivor = run_dsv4_isolated(seed, dev, survivor, n_prompt);
    const std::vector<float> isolated_replacement = run_dsv4_isolated(seed, dev, replacement, n_prompt);

    for (bool kv_unified : { false, true }) {
        const char * mode = kv_unified ?
            "requested-unified-reuse" : "split-reuse";
        const dsv4_parallel_result reused = run_dsv4_reused_slot(
                seed, dev, discarded, survivor, replacement,
                n_prompt, n_decode_before_reuse, kv_unified);
        all_ok = compare_dsv4_trace(mode, "replacement", isolated_replacement, reused.seq_a) && all_ok;
        all_ok = compare_dsv4_trace(mode, "survivor", isolated_survivor, reused.seq_b) && all_ok;
    }

    return all_ok;
}

static int test_dsv4_parallel(size_t seed) {
    bool all_ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        printf("DSV4 parallel lifecycle on %s\n", ggml_backend_dev_description(dev));
        all_ok = test_dsv4_parallel_device(seed, dev) && all_ok;
    }

    return all_ok ? 0 : 1;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MINIMAX_M3) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    FILE * file = tmpfile(); // Can be null on Windows without administrator privileges.
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file != nullptr && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file);
                        rewind(file);

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        const int backends_result = test_backends(arch, seed, log_level);
        if (backends_result != 0) {
            return backends_result;
        }
        if (arch == LLM_ARCH_UNKNOWN || arch == LLM_ARCH_MINIMAX_M3) {
            test_minimax_m3_dense_fallback(seed);
        }
        if (arch == LLM_ARCH_UNKNOWN || arch == LLM_ARCH_DEEPSEEK4) {
            test_dsv4_mtp(seed);
            return test_dsv4_parallel(seed);
        }
        return 0;
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}

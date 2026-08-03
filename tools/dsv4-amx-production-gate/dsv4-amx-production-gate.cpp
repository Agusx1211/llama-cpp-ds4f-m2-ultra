#include "build-info.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int32_t DSV4_GATE_TOKENS = 2048;

struct backend_scope {
    backend_scope() { llama_backend_init(); }

    ~backend_scope() { shutdown(); }

    void shutdown() {
        if (active) {
            llama_backend_free();
            active = false;
        }
    }

    bool active = true;
};

using model_ptr   = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

int fail(const char * event, const char * reason) {
    std::fprintf(stderr, "dsv4_amx_gate_driver event=%s outcome=fail reason=%s\n", event, reason);
    return 1;
}

bool env_is_one(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 5 || std::strcmp(argv[1], "--model") != 0 || std::strcmp(argv[3], "--expected-commit") != 0) {
        std::fprintf(stderr, "usage: %s --model <first-shard.gguf> --expected-commit <full-sha>\n", argv[0]);
        return 2;
    }

    const std::string model_path      = argv[2];
    const std::string expected_commit = argv[4];
    const std::string build_commit    = llama_commit();
    if (expected_commit.size() != 40 || build_commit == "unknown" ||
        expected_commit.compare(0, build_commit.size(), build_commit) != 0) {
        return fail("provenance", "build_commit_mismatch");
    }
    if (!env_is_one("LLAMA_DSV4_AMX_COEXEC") || !env_is_one("LLAMA_DSV4_AMX_COEXEC_VALIDATE") ||
        env_is_one("LLAMA_DSV4_AMX_COEXEC_DISABLE")) {
        return fail("environment", "exact_validation_opt_in_missing");
    }

    std::fprintf(stderr,
                 "dsv4_amx_gate_driver event=start outcome=begin source_commit=%s build_commit=%s tokens=%d "
                 "submissions=1 benchmark=0\n",
                 expected_commit.c_str(), build_commit.c_str(), DSV4_GATE_TOKENS);

    backend_scope      backend;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 999;
    model_params.split_mode         = LLAMA_SPLIT_MODE_LAYER;
    model_params.main_gpu           = 0;
    model_params.load_mtp           = false;

    model_ptr model(llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
    if (!model) {
        return fail("model_load", "model_load_failed");
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = DSV4_GATE_TOKENS;
    context_params.n_batch              = DSV4_GATE_TOKENS;
    context_params.n_ubatch             = DSV4_GATE_TOKENS;
    context_params.n_seq_max            = 1;
    context_params.n_rs_seq             = 0;
    context_params.n_outputs_max        = 1;
    context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.no_perf              = true;

    context_ptr context(llama_init_from_model(model.get(), context_params), llama_free);
    if (!context) {
        return fail("context_create", "context_create_failed");
    }
    if (llama_n_ctx(context.get()) != DSV4_GATE_TOKENS || llama_n_batch(context.get()) != DSV4_GATE_TOKENS ||
        llama_n_ubatch(context.get()) != DSV4_GATE_TOKENS || llama_n_seq_max(context.get()) != 1) {
        return fail("context_contract", "runtime_context_shape_mismatch");
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model.get());
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        return fail("model_contract", "empty_vocabulary");
    }

    std::vector<llama_token> tokens(DSV4_GATE_TOKENS);
    for (int32_t i = 0; i < DSV4_GATE_TOKENS; ++i) {
        tokens[static_cast<size_t>(i)] = i % n_vocab;
    }
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos >= 0 && bos < n_vocab) {
        tokens[0] = bos;
    }

    std::fprintf(stderr, "dsv4_amx_gate_driver event=decode outcome=begin tokens=%d submissions=1\n", DSV4_GATE_TOKENS);
    const llama_batch batch = llama_batch_get_one(tokens.data(), DSV4_GATE_TOKENS);
    if (llama_decode(context.get(), batch) != 0) {
        return fail("decode", "llama_decode_failed");
    }
    llama_synchronize(context.get());
    std::fprintf(stderr, "dsv4_amx_gate_driver event=decode outcome=pass tokens=%d submissions=1\n", DSV4_GATE_TOKENS);

    context.reset();
    std::fprintf(stderr, "dsv4_amx_gate_driver event=context_teardown outcome=pass\n");
    model.reset();
    std::fprintf(stderr, "dsv4_amx_gate_driver event=model_teardown outcome=pass\n");
    backend.shutdown();
    std::fprintf(stderr, "dsv4_amx_gate_driver event=backend_teardown outcome=pass\n");
    std::fprintf(stderr, "dsv4_amx_gate_driver event=complete outcome=pass tokens=%d submissions=1 benchmark=0\n",
                 DSV4_GATE_TOKENS);
    return 0;
}

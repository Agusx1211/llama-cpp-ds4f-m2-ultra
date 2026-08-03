#include "../src/llama-cparams.h"
#include "../src/llama-dsv4-amx.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

void expect(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool close(float lhs, float rhs) {
    return std::abs(lhs - rhs) <= 1e-6f * std::max(1.0f, std::abs(rhs));
}

float swiglu_reference(float gate, float up, float limit) {
    if (limit > 1e-6f) {
        gate = std::min(gate, limit);
        up   = std::max(-limit, std::min(up, limit));
    }
    return gate / (1.0f + std::exp(-gate)) * up;
}

bool dummy_eval_callback(ggml_tensor *, bool, void *) {
    return false;
}

void test_swiglu() {
    const std::vector<float> gate_source = { 20.0f, -20.0f, 2.0f, -2.0f, 0.0f };
    const std::vector<float> up_source   = { 20.0f, -20.0f, 0.5f, -0.5f, 3.0f };

    auto gate = gate_source;
    auto up   = up_source;
    llama_dsv4_amx_apply_swiglu(gate.data(), up.data(), gate.size(), 10.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        expect(close(gate[i], swiglu_reference(gate_source[i], up_source[i], 10.0f)), "clamped SwiGLU mismatch");
    }
    expect(up[0] == 10.0f, "positive up clamp mismatch");
    expect(up[1] == -10.0f, "negative up clamp mismatch");

    gate = gate_source;
    up   = up_source;
    llama_dsv4_amx_apply_swiglu(gate.data(), up.data(), gate.size(), 0.0f);
    for (size_t i = 0; i < gate.size(); ++i) {
        expect(close(gate[i], swiglu_reference(gate_source[i], up_source[i], 0.0f)), "unclamped SwiGLU mismatch");
        expect(up[i] == up_source[i], "disabled clamp modified up projection");
    }
}

void test_scratch_contract() {
    expect(llama_dsv4_amx_scratch_bytes(0) == 0, "zero-token scratch mismatch");
    expect(llama_dsv4_amx_scratch_bytes(2048) == 64ULL * 1024ULL * 1024ULL, "production scratch mismatch");
    expect(llama_dsv4_amx_output_bytes(2048) == 32ULL * 1024ULL * 1024ULL, "shared output size mismatch");
    expect(llama_dsv4_amx_incremental_context_bytes(2048) == 96ULL * 1024ULL * 1024ULL,
           "incremental context size mismatch");
}

void test_oracle_stage_gate() {
    constexpr double nmse_limit    = 1e-12;
    constexpr double max_abs_limit = 5e-4;
    expect(llama_dsv4_amx_oracle_stage_passes(true, true, 1.0, nmse_limit, max_abs_limit),
           "exact oracle limits were rejected");
    expect(!llama_dsv4_amx_oracle_stage_passes(
               true, true, 1.0, std::nextafter(nmse_limit, std::numeric_limits<double>::infinity()), max_abs_limit),
           "oracle accepted NMSE above the probe limit");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, true, 1.0, nmse_limit,
                                               std::nextafter(max_abs_limit, std::numeric_limits<double>::infinity())),
           "oracle accepted max_abs above the probe limit");
    expect(!llama_dsv4_amx_oracle_stage_passes(false, true, 1.0, 0.0, 0.0), "oracle accepted a nonfinite stage");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, false, 1.0, 0.0, 0.0), "oracle accepted an all-zero stage");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, true, 0.0, 0.0, 0.0), "oracle accepted zero reference L2");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, true, std::numeric_limits<double>::infinity(), 0.0, 0.0),
           "oracle accepted infinite reference L2");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, true, 1.0, std::numeric_limits<double>::infinity(), 0.0),
           "oracle accepted nonfinite NMSE");
    expect(!llama_dsv4_amx_oracle_stage_passes(true, true, 1.0, 0.0, std::numeric_limits<double>::quiet_NaN()),
           "oracle accepted nonfinite max_abs");
}

void test_metal_telemetry_gate() {
    // Metal divergence is telemetry under the F32 quality policy. Arbitrarily
    // large finite differences remain structurally valid.
    expect(llama_dsv4_amx_telemetry_stage_is_valid(true, true, true, 1.0, 10.0, 100.0),
           "finite Metal divergence was treated as a rejection");
    expect(!llama_dsv4_amx_telemetry_stage_is_valid(false, true, true, 1.0, 0.0, 0.0),
           "nonfinite direct output passed the telemetry health gate");
    expect(!llama_dsv4_amx_telemetry_stage_is_valid(true, false, true, 1.0, 0.0, 0.0),
           "all-zero direct output passed the telemetry health gate");
    expect(!llama_dsv4_amx_telemetry_stage_is_valid(true, true, false, 1.0, 0.0, 0.0),
           "nonfinite Metal reference passed the telemetry health gate");
    expect(!llama_dsv4_amx_telemetry_stage_is_valid(true, true, true, 0.0, 0.0, 0.0),
           "zero Metal reference L2 passed the telemetry health gate");
    expect(
        !llama_dsv4_amx_telemetry_stage_is_valid(true, true, true, 1.0, std::numeric_limits<double>::quiet_NaN(), 0.0),
        "nonfinite telemetry metric passed the health gate");
    expect(!llama_dsv4_amx_telemetry_stage_is_valid(true, true, true, 1.0, -1.0, 0.0),
           "negative telemetry NMSE passed the health gate");
}

void test_context_gates() {
    llama_cparams params = {};
    params.ctx_type      = LLAMA_CONTEXT_TYPE_DEFAULT;
    params.n_ubatch      = 2048;
    params.n_seq_max     = 1;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_ELIGIBLE, "eligible context was rejected");

    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_TYPE, "MTP context was not excluded");
    params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    params.n_rs_seq = 1;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_RECURRENT,
           "recurrent default context was not excluded");
    params.n_rs_seq = 0;
    params.n_ubatch = 1024;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_UBATCH,
           "unsupported ubatch was not excluded");
    params.n_ubatch  = 2048;
    params.n_seq_max = 2;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_MULTI_SEQUENCE,
           "multi-sequence context was not excluded");
    params.n_seq_max         = 1;
    params.pipeline_parallel = true;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_PIPELINE,
           "pipeline context was not excluded");
    params.pipeline_parallel = false;
    params.cb_eval           = dummy_eval_callback;
    expect(llama_dsv4_amx_context_gate_for(params) == LLAMA_DSV4_AMX_CONTEXT_CALLBACK,
           "user callback context was not excluded");
}

void test_callback_order() {
    llama_dsv4_amx_callback_order order;
    order.begin();
    expect(order.ask(0, true) == LLAMA_DSV4_AMX_CALLBACK_OK, "first callback ask failed");
    expect(order.ask(0, true) == LLAMA_DSV4_AMX_CALLBACK_PENDING_AFTER, "pending after was not detected");
    expect(order.after(0, false) == LLAMA_DSV4_AMX_CALLBACK_AFTER_MISMATCH, "after mismatch was not detected");
    expect(order.after(0, true) == LLAMA_DSV4_AMX_CALLBACK_OK, "first callback after failed");
    expect(order.ask(0, true) == LLAMA_DSV4_AMX_CALLBACK_DUPLICATE, "duplicate start was not detected");
    expect(order.ask(1, false) == LLAMA_DSV4_AMX_CALLBACK_ORDER, "out-of-order end was not detected");
    expect(order.ask(0, false) == LLAMA_DSV4_AMX_CALLBACK_OK, "first end ask failed");
    expect(order.after(0, false) == LLAMA_DSV4_AMX_CALLBACK_OK, "first end after failed");
    expect(!order.complete(), "partial callback stream was accepted");

    for (int layer = 1; layer < 43; ++layer) {
        expect(order.ask(layer, true) == LLAMA_DSV4_AMX_CALLBACK_OK, "ordered start ask failed");
        expect(order.after(layer, true) == LLAMA_DSV4_AMX_CALLBACK_OK, "ordered start after failed");
        expect(order.ask(layer, false) == LLAMA_DSV4_AMX_CALLBACK_OK, "ordered end ask failed");
        expect(order.after(layer, false) == LLAMA_DSV4_AMX_CALLBACK_OK, "ordered end after failed");
    }
    expect(order.complete(), "complete exact callback stream was rejected");
    expect(order.starts_seen() == ((1ULL << 43) - 1), "start bitset mismatch");
    expect(order.ends_seen() == ((1ULL << 43) - 1), "end bitset mismatch");

    order.begin();
    expect(order.after(0, true) == LLAMA_DSV4_AMX_CALLBACK_AFTER_WITHOUT_ASK, "after without ask was not detected");
    expect(order.ask(-1, true) == LLAMA_DSV4_AMX_CALLBACK_LAYER_RANGE, "negative layer was not rejected");
    expect(order.ask(43, true) == LLAMA_DSV4_AMX_CALLBACK_LAYER_RANGE, "overflow layer was not rejected");
}

void test_shared_output_binding() {
    ggml_backend_ptr backend(ggml_backend_cpu_init());
    expect(backend != nullptr, "CPU backend initialization failed");
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend.get());
    ggml_backend_buffer_ptr    buffer(ggml_backend_buft_alloc_buffer(buft, llama_dsv4_amx_output_bytes(2048)));
    expect(buffer != nullptr, "shared output contract buffer allocation failed");

    ggml_init_params params = {
        ggml_tensor_overhead() * 48,
        nullptr,
        true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    expect(ctx != nullptr, "shared output metadata context initialization failed");

    std::unordered_set<ggml_tensor *>         tensors;
    std::unordered_set<void *>                data;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    for (int layer = 0; layer < 43; ++layer) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4096, 2048);
        expect(llama_dsv4_amx_output_tensor_is_exact(tensor), "exact producerless output tensor was rejected");
        expect(llama_dsv4_amx_bind_output_tensor(buffer.get(), tensor), "external output tensor binding failed");
        tensors.emplace(tensor);
        data.emplace(tensor->data);
        buffers.emplace(tensor->buffer);
    }
    expect(tensors.size() == 43, "output tensor metadata was unexpectedly shared");
    expect(data.size() == 1, "output tensors did not reuse one data allocation");
    expect(buffers.size() == 1 && *buffers.begin() == buffer.get(), "output tensors did not reuse one buffer");
    expect(ggml_backend_buffer_get_size(buffer.get()) == 32ULL * 1024ULL * 1024ULL,
           "external output buffer retained more than one layer");

    ggml_tensor * malformed = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4096, 1024);
    expect(!llama_dsv4_amx_output_tensor_is_exact(malformed), "malformed output shape passed the poison gate");
    expect(!llama_dsv4_amx_bind_output_tensor(buffer.get(), malformed), "malformed output tensor was accepted");
    expect(malformed->buffer == nullptr && malformed->data == nullptr, "rejected output tensor was mutated");

    ggml_tensor * produced = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4096, 2048);
    produced->src[0]       = malformed;
    expect(!llama_dsv4_amx_output_tensor_is_exact(produced), "produced output passed the poison gate");
    expect(!llama_dsv4_amx_bind_output_tensor(buffer.get(), produced), "produced output tensor was accepted");

    ggml_tensor * operation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4096, 2048);
    operation->op           = GGML_OP_ADD;
    expect(!llama_dsv4_amx_output_tensor_is_exact(operation), "operation output passed the poison gate");
    expect(!llama_dsv4_amx_bind_output_tensor(buffer.get(), operation), "operation output tensor was accepted");
}

void test_graph_reuse_mode() {
    llm_graph_params lhs = {};
    llm_graph_params rhs = {};
    expect(lhs.allow_reuse(rhs), "identical graph parameters are not reusable");
    rhs.dsv4_amx_mode = LLM_DSV4_AMX_COEXEC;
    expect(!lhs.allow_reuse(rhs), "disabled and AMX graphs reused the same topology");
    lhs.dsv4_amx_mode = LLM_DSV4_AMX_COEXEC;
    expect(lhs.allow_reuse(rhs), "identical AMX graph parameters are not reusable");
    rhs.dsv4_amx_mode = LLM_DSV4_AMX_VALIDATE;
    expect(!lhs.allow_reuse(rhs), "coexecution and validation graphs reused the same topology");
}

#if !defined(_WIN32)
struct saved_env {
    explicit saved_env(const char * name) : name(name) {
        const char * value = std::getenv(name);
        if (value != nullptr) {
            present   = true;
            old_value = value;
        }
    }

    ~saved_env() {
        if (present) {
            setenv(name.c_str(), old_value.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }

    std::string name;
    bool        present = false;
    std::string old_value;
};

void test_disable_wins() {
    saved_env save_enable("LLAMA_DSV4_AMX_COEXEC");
    saved_env save_disable("LLAMA_DSV4_AMX_COEXEC_DISABLE");

    unsetenv("LLAMA_DSV4_AMX_COEXEC");
    unsetenv("LLAMA_DSV4_AMX_COEXEC_DISABLE");
    expect(!llama_dsv4_amx_requested(), "AMX path was enabled by default");

    setenv("LLAMA_DSV4_AMX_COEXEC", "1", 1);
    expect(llama_dsv4_amx_requested(), "explicit AMX opt-in was ignored");

    setenv("LLAMA_DSV4_AMX_COEXEC_DISABLE", "1", 1);
    expect(!llama_dsv4_amx_requested(), "explicit AMX disable did not win");

    setenv("LLAMA_DSV4_AMX_COEXEC_DISABLE", "0", 1);
    expect(llama_dsv4_amx_requested(), "non-active disable unexpectedly won");
}
#endif

}  // namespace

int main() {
    test_swiglu();
    test_scratch_contract();
    test_oracle_stage_gate();
    test_metal_telemetry_gate();
    test_context_gates();
    test_callback_order();
    test_shared_output_binding();
    test_graph_reuse_mode();
#if !defined(_WIN32)
    test_disable_wins();
#endif
    return 0;
}

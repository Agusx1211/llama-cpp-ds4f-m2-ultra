#pragma once

#include "ggml-backend.h"
#include "llama-graph.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct llama_cparams;
struct llama_model;
class llama_dsv4_amx_context;

enum llama_dsv4_amx_context_gate : uint8_t {
    LLAMA_DSV4_AMX_CONTEXT_ELIGIBLE = 0,
    LLAMA_DSV4_AMX_CONTEXT_TYPE,
    LLAMA_DSV4_AMX_CONTEXT_RECURRENT,
    LLAMA_DSV4_AMX_CONTEXT_UBATCH,
    LLAMA_DSV4_AMX_CONTEXT_MULTI_SEQUENCE,
    LLAMA_DSV4_AMX_CONTEXT_PIPELINE,
    LLAMA_DSV4_AMX_CONTEXT_CALLBACK,
};

enum llama_dsv4_amx_callback_error : uint8_t {
    LLAMA_DSV4_AMX_CALLBACK_OK = 0,
    LLAMA_DSV4_AMX_CALLBACK_LAYER_RANGE,
    LLAMA_DSV4_AMX_CALLBACK_PENDING_AFTER,
    LLAMA_DSV4_AMX_CALLBACK_ORDER,
    LLAMA_DSV4_AMX_CALLBACK_DUPLICATE,
    LLAMA_DSV4_AMX_CALLBACK_AFTER_WITHOUT_ASK,
    LLAMA_DSV4_AMX_CALLBACK_AFTER_MISMATCH,
};

// Shared by the production callback and host-side contract tests. The two
// 43-bit masks make callback uniqueness explicit rather than inferred from a
// final count.
class llama_dsv4_amx_callback_order {
  public:
    void begin();

    llama_dsv4_amx_callback_error ask(int layer, bool start);
    llama_dsv4_amx_callback_error after(int layer, bool start);

    bool     complete() const;
    int      expected_layer() const;
    bool     expected_start() const;
    uint64_t starts_seen() const;
    uint64_t ends_seen() const;

  private:
    uint64_t starts        = 0;
    uint64_t ends          = 0;
    int      layer         = 0;
    bool     start         = true;
    bool     pending       = false;
    int      pending_layer = -1;
    bool     pending_start = false;
};

class llama_dsv4_amx_model {
  public:
    ~llama_dsv4_amx_model();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    explicit llama_dsv4_amx_model(std::unique_ptr<impl> pimpl);
    void ensure_packed(const llama_model & model, bool run_oracle) const;

    friend std::shared_ptr<const llama_dsv4_amx_model> llama_dsv4_amx_model_create(const llama_model & model);
    friend std::unique_ptr<llama_dsv4_amx_context>     llama_dsv4_amx_context_create(
            const llama_model &,
            const llama_cparams &,
            const std::vector<ggml_backend_t> &,
            const std::vector<ggml_backend_buffer_type_t> &);
    friend class llama_dsv4_amx_context;
};

class llama_dsv4_amx_context {
  public:
    ~llama_dsv4_amx_context();

    llm_dsv4_amx_mode mode_for(llm_graph_type gtype, int64_t n_tokens, bool has_lora) const;
    void              set_scheduler(ggml_backend_sched_t sched);

    // Both graphs bind their producerless output leaves to context-owned storage.
    // The executable graph also becomes the next scheduler callback binding.
    void prepare_graph(ggml_cgraph * graph, llm_dsv4_amx_mode mode, bool executable);

    bool        owns_graph(const ggml_cgraph * graph) const;
    bool        begin_graph(const ggml_cgraph * graph);
    ggml_status finish_graph(ggml_status scheduler_status);
    void        shutdown();

    static bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data);

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;

    explicit llama_dsv4_amx_context(std::unique_ptr<impl> pimpl);

    friend std::unique_ptr<llama_dsv4_amx_context> llama_dsv4_amx_context_create(
        const llama_model &                             model,
        const llama_cparams &                           cparams,
        const std::vector<ggml_backend_t> &             backends,
        const std::vector<ggml_backend_buffer_type_t> & buffer_types);
};

// All environment parsing is centralized here. The explicit disable always
// wins, including over validation mode.
bool llama_dsv4_amx_requested();

llama_dsv4_amx_context_gate llama_dsv4_amx_context_gate_for(const llama_cparams & cparams);
const char *                llama_dsv4_amx_context_gate_name(llama_dsv4_amx_context_gate gate);
const char *                llama_dsv4_amx_callback_error_name(llama_dsv4_amx_callback_error error);

std::shared_ptr<const llama_dsv4_amx_model> llama_dsv4_amx_model_create(const llama_model & model);

std::unique_ptr<llama_dsv4_amx_context> llama_dsv4_amx_context_create(
    const llama_model &                             model,
    const llama_cparams &                           cparams,
    const std::vector<ggml_backend_t> &             backends,
    const std::vector<ggml_backend_buffer_type_t> & buffer_types);

// These helpers are shared by the worker and focused host-side contract tests.
void   llama_dsv4_amx_apply_swiglu(float * gate_hidden, float * up, size_t count, float clamp_limit);
size_t llama_dsv4_amx_scratch_bytes(int64_t tokens);
size_t llama_dsv4_amx_output_bytes(int64_t tokens);
size_t llama_dsv4_amx_incremental_context_bytes(int64_t tokens);
bool   llama_dsv4_amx_oracle_stage_passes(bool finite, bool nonzero, double reference_l2, double nmse, double max_abs);
bool   llama_dsv4_amx_telemetry_stage_is_valid(bool   direct_finite,
                                               bool   direct_nonzero,
                                               bool   reference_finite,
                                               double reference_l2,
                                               double nmse,
                                               double max_abs);
bool   llama_dsv4_amx_output_tensor_is_exact(const ggml_tensor * tensor);
bool   llama_dsv4_amx_bind_output_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor);

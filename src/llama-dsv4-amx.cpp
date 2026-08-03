#include "llama-dsv4-amx.h"

#include "ggml-cpp.h"
#include "ggml.h"
#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(LLAMA_DSV4_AMX_APPLE_ARM64)
#    include "llama-dsv4-amx-aarch64.h"

#    include <Accelerate/Accelerate.h>
#    include <dispatch/dispatch.h>
#    include <sys/sysctl.h>
#endif

namespace {

using clock_type = std::chrono::steady_clock;

constexpr int64_t DSV4_AMX_EMBD                 = 4096;
constexpr int64_t DSV4_AMX_FF                   = 2048;
constexpr int64_t DSV4_AMX_TOKENS               = 2048;
constexpr int     DSV4_AMX_LAYERS               = 43;
constexpr int64_t DSV4_AMX_PANEL_COLS           = 128;
constexpr int64_t DSV4_AMX_K_BLOCK              = 1024;
constexpr size_t  DSV4_AMX_MATRIX_ELEMENTS      = 8388608ULL;
constexpr size_t  DSV4_AMX_SOURCE_BYTES         = DSV4_AMX_MATRIX_ELEMENTS * sizeof(ggml_bf16_t);
constexpr size_t  DSV4_AMX_PACKED_VALUES        = 1082130432ULL;
constexpr size_t  DSV4_AMX_PACKED_BYTES         = DSV4_AMX_PACKED_VALUES * sizeof(float);
constexpr size_t  AMX_PAIR_ALIGNMENT            = 128;
constexpr double  DSV4_AMX_ORACLE_NMSE_LIMIT    = 1e-12;
constexpr double  DSV4_AMX_ORACLE_MAX_ABS_LIMIT = 5e-4;

bool env_is_one(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool validation_requested() {
    return env_is_one("LLAMA_DSV4_AMX_COEXEC_VALIDATE");
}

bool timing_requested() {
    return env_is_one("LLAMA_DSV4_AMX_COEXEC_TIMING");
}

bool oracle_requested() {
    return validation_requested() || env_is_one("LLAMA_DSV4_AMX_COEXEC_ORACLE");
}

template <typename... Args>
void audit_record(const char * event, const char * outcome, const char * format, Args... args) {
    char detail[2048];
    if constexpr (sizeof...(Args) == 0) {
        std::snprintf(detail, sizeof(detail), "%s", format);
    } else {
        std::snprintf(detail, sizeof(detail), format, args...);
    }
    LLAMA_LOG_INFO("dsv4_amx_audit event=%s outcome=%s %s\n", event, outcome, detail);
}

double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool has_target_shape(const llama_model & model) {
    return model.arch == LLM_ARCH_DEEPSEEK4 && model.hparams.n_layer() == DSV4_AMX_LAYERS &&
           model.hparams.n_embd == DSV4_AMX_EMBD && model.hparams.n_ff_exp == DSV4_AMX_FF &&
           model.hparams.n_expert_shared == 1 && model.hparams.n_dspark_block_size == 0;
}

void require_tensor_shape(const ggml_tensor * tensor, int64_t ne0, int64_t ne1, int il, const char * role) {
    char expected_name[64];
    std::snprintf(expected_name, sizeof(expected_name), "blk.%d.ffn_%s_shexp.weight", il, role);
    if (tensor == nullptr || tensor->type != GGML_TYPE_BF16 || tensor->ne[0] != ne0 || tensor->ne[1] != ne1 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor) || tensor->op != GGML_OP_NONE ||
        tensor->view_src != nullptr || std::strcmp(tensor->name, expected_name) != 0) {
        throw std::runtime_error("DeepSeek V4 AMX requires contiguous BF16 " + std::string(role) +
                                 " weight with the exact target identity and shape at layer " + std::to_string(il));
    }
}

bool has_exact_binary_sources(const ggml_tensor * tensor,
                              ggml_op             op,
                              const ggml_tensor * first,
                              const ggml_tensor * second,
                              bool                unordered) {
    if (tensor == nullptr || tensor->op != op) {
        return false;
    }
    const bool sources_match = tensor->src[0] == first && tensor->src[1] == second;
    const bool swapped_match = unordered && tensor->src[0] == second && tensor->src[1] == first;
    if (!sources_match && !swapped_match) {
        return false;
    }
    for (size_t i = 2; i < GGML_MAX_SRC; ++i) {
        if (tensor->src[i] != nullptr) {
            return false;
        }
    }
    return true;
}

std::vector<ggml_tensor *> direct_consumers(ggml_cgraph * graph, const ggml_tensor * source) {
    std::vector<ggml_tensor *> result;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        for (const ggml_tensor * input : node->src) {
            if (input == source) {
                result.push_back(node);
            }
        }
    }
    return result;
}

#if defined(LLAMA_DSV4_AMX_APPLE_ARM64)

using metal_tensor_host_ptr_fn = void * (*) (const ggml_tensor * tensor);
using metal_buft_is_shared_fn  = bool (*)(ggml_backend_buffer_type_t buft);

std::string cpu_brand() {
    size_t size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", nullptr, &size, nullptr, 0) != 0 || size == 0) {
        return {};
    }

    std::string brand(size, '\0');
    if (sysctlbyname("machdep.cpu.brand_string", brand.data(), &size, nullptr, 0) != 0) {
        return {};
    }
    brand.resize(std::strlen(brand.c_str()));
    return brand;
}

ggml_backend_reg_t metal_registry_for_device(ggml_backend_dev_t device) {
    ggml_backend_reg_t registry = device == nullptr ? nullptr : ggml_backend_dev_backend_reg(device);
    if (registry == nullptr || std::strcmp(ggml_backend_reg_name(registry), "MTL") != 0) {
        return nullptr;
    }
    return registry;
}

ggml_backend_reg_t metal_registry_for_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    return metal_registry_for_device(ggml_backend_buft_get_device(buft));
}

metal_tensor_host_ptr_fn resolve_tensor_host_ptr(ggml_backend_reg_t registry) {
    return registry == nullptr ? nullptr :
                                 reinterpret_cast<metal_tensor_host_ptr_fn>(ggml_backend_reg_get_proc_address(
                                     registry, "ggml_backend_metal_get_tensor_host_ptr"));
}

const ggml_bf16_t * require_shared_bf16_pointer(const ggml_tensor * tensor, int il, const char * role) {
    ggml_backend_reg_t registry     = metal_registry_for_tensor(tensor);
    auto               get_host_ptr = resolve_tensor_host_ptr(registry);
    void *             pointer      = get_host_ptr == nullptr ? nullptr : get_host_ptr(tensor);
    if (pointer == nullptr) {
        throw std::runtime_error("DeepSeek V4 AMX requires Metal-shared host-visible " + std::string(role) +
                                 " weight storage at layer " + std::to_string(il));
    }
    return static_cast<const ggml_bf16_t *>(pointer);
}

class direct_amx_weight {
  public:
    direct_amx_weight(const ggml_bf16_t * source, int64_t output_cols, int64_t input_cols) :
        output_cols_(output_cols),
        input_cols_(input_cols) {
        if (source == nullptr || output_cols_ % DSV4_AMX_PANEL_COLS != 0 || DSV4_AMX_PANEL_COLS % 64 != 0) {
            throw std::invalid_argument("invalid direct AMX packed-weight dimensions");
        }

        packed_count_    = static_cast<size_t>(output_cols_ * input_cols_);
        void *    memory = nullptr;
        const int error  = posix_memalign(&memory, AMX_PAIR_ALIGNMENT, packed_bytes());
        if (error != 0) {
            throw std::runtime_error("failed to allocate aligned direct AMX weights: " +
                                     std::string(std::strerror(error)));
        }
        packed_.reset(static_cast<float *>(memory));
        validate_alignment();

        for (int64_t panel = 0; panel < panel_count(); ++panel) {
            const int64_t first_col   = panel * DSV4_AMX_PANEL_COLS;
            float *       destination = packed_.get() + panel * input_cols_ * DSV4_AMX_PANEL_COLS;
            for (int64_t k = 0; k < input_cols_; ++k) {
                for (int64_t col = 0; col < DSV4_AMX_PANEL_COLS; ++col) {
                    const size_t source_index = static_cast<size_t>((first_col + col) * input_cols_ + k);
                    destination[k * DSV4_AMX_PANEL_COLS + col] = ggml_bf16_to_fp32(source[source_index]);
                }
            }
        }
        validate_source(source);
    }

    size_t packed_bytes() const { return packed_count_ * sizeof(float); }

    void multiply_transposed(const float * activation_t, float * output, int64_t rows) const {
        if (activation_t == nullptr || output == nullptr || rows != DSV4_AMX_TOKENS || rows % 16 != 0) {
            throw std::invalid_argument("direct AMX multiply requires the exact 2048-row production shape");
        }
        validate_alignment();
        worker_context context = { this, activation_t, output, rows };
        dispatch_apply_f(static_cast<size_t>(panel_count()), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                         &context, worker_entry);
    }

    // Factored Accelerate oracle over the exact production F32 pack. This is a
    // conventional SGEMM reference and deliberately shares no implementation
    // with the raw AMX instruction loop.
    void multiply_reference(const float * activation, float * output, int64_t rows) const {
        if (activation == nullptr || output == nullptr || rows <= 0) {
            throw std::invalid_argument("direct AMX reference received invalid storage");
        }
        for (int64_t panel = 0; panel < panel_count(); ++panel) {
            const float * weight_panel = packed_.get() + panel * input_cols_ * DSV4_AMX_PANEL_COLS;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(rows),
                        static_cast<int>(DSV4_AMX_PANEL_COLS), static_cast<int>(input_cols_), 1.0f, activation,
                        static_cast<int>(input_cols_), weight_panel, static_cast<int>(DSV4_AMX_PANEL_COLS), 0.0f,
                        output + panel * DSV4_AMX_PANEL_COLS, static_cast<int>(output_cols_));
        }
    }

  private:
    struct aligned_free {
        void operator()(float * pointer) const { std::free(pointer); }
    };

    struct worker_context {
        const direct_amx_weight * weight;
        const float *             activation_t;
        float *                   output;
        int64_t                   rows;
    };

    static uint64_t fma32_operand(int bank, int x_byte_offset, bool clear_accumulator) {
        return (static_cast<uint64_t>(bank) << 20) | (static_cast<uint64_t>(x_byte_offset) << 10) |
               (clear_accumulator ? (1ULL << 27) : 0);
    }

    static uint64_t z_operand(const float * address, int row) {
        return reinterpret_cast<uint64_t>(address) | (static_cast<uint64_t>(row) << 56);
    }

    void validate_alignment() const {
        const size_t row_bytes   = static_cast<size_t>(DSV4_AMX_PANEL_COLS) * sizeof(float);
        const size_t panel_bytes = static_cast<size_t>(input_cols_) * row_bytes;
        if (packed_ == nullptr || reinterpret_cast<uintptr_t>(packed_.get()) % AMX_PAIR_ALIGNMENT != 0 ||
            row_bytes % AMX_PAIR_ALIGNMENT != 0 || panel_bytes % AMX_PAIR_ALIGNMENT != 0) {
            throw std::runtime_error("direct AMX paired-load alignment invariant failed");
        }
    }

    void validate_source(const ggml_bf16_t * source) const {
        for (int64_t col = 0; col < output_cols_; ++col) {
            const int64_t panel     = col / DSV4_AMX_PANEL_COLS;
            const int64_t panel_col = col % DSV4_AMX_PANEL_COLS;
            for (int64_t k = 0; k < input_cols_; ++k) {
                const size_t   source_index  = static_cast<size_t>(col * input_cols_ + k);
                const size_t   packed_index  = static_cast<size_t>(panel * input_cols_ * DSV4_AMX_PANEL_COLS +
                                                                   k * DSV4_AMX_PANEL_COLS + panel_col);
                const uint32_t expected_bits = static_cast<uint32_t>(source[source_index].bits) << 16;
                float          expected;
                std::memcpy(&expected, &expected_bits, sizeof(expected));
                if (!std::isfinite(expected) ||
                    std::memcmp(&packed_.get()[packed_index], &expected, sizeof(float)) != 0) {
                    throw std::runtime_error("direct AMX packed weight failed exact BF16 expansion/index validation");
                }
            }
        }
    }

    static void worker_entry(void * opaque, size_t panel) {
        const auto & context = *static_cast<worker_context *>(opaque);
        AMX_SET();
        context.weight->multiply_panel(context.activation_t, context.output, context.rows, static_cast<int64_t>(panel));
        AMX_CLR();
    }

    void multiply_panel(const float * activation_t, float * output, int64_t rows, int64_t panel) const {
        constexpr uint64_t ldx_pair  = 1ULL << 62;
        const int64_t      first_col = panel * DSV4_AMX_PANEL_COLS;
        const float *      packed    = packed_.get() + panel * input_cols_ * DSV4_AMX_PANEL_COLS;

        for (int64_t k_base = 0; k_base < input_cols_; k_base += DSV4_AMX_K_BLOCK) {
            const int64_t k_count = std::min<int64_t>(DSV4_AMX_K_BLOCK, input_cols_ - k_base);
            for (int64_t row_base = 0; row_base < rows; row_base += 16) {
                for (int64_t panel_col = 0; panel_col < DSV4_AMX_PANEL_COLS; panel_col += 64) {
                    if (k_base != 0) {
                        for (int bank = 0; bank < 4; ++bank) {
                            for (int row = 0; row < 16; ++row) {
                                const float * partial =
                                    output + (row_base + row) * output_cols_ + first_col + panel_col + 16 * bank;
                                AMX_LDZ(z_operand(partial, 4 * row + bank));
                            }
                        }
                    }

                    for (int64_t k = 0; k < k_count; ++k) {
                        const bool clear_accumulator = k_base == 0 && k == 0;
                        AMX_LDY(reinterpret_cast<uint64_t>(activation_t + (k_base + k) * rows + row_base));
                        const float * weight_row = packed + (k_base + k) * DSV4_AMX_PANEL_COLS + panel_col;
                        AMX_LDX(reinterpret_cast<uint64_t>(weight_row) | ldx_pair);
                        AMX_LDX(reinterpret_cast<uint64_t>(weight_row + 32) | (2ULL << 56) | ldx_pair);
                        AMX_FMA32(fma32_operand(0, 0, clear_accumulator));
                        AMX_FMA32(fma32_operand(1, 64, clear_accumulator));
                        AMX_FMA32(fma32_operand(2, 128, clear_accumulator));
                        AMX_FMA32(fma32_operand(3, 192, clear_accumulator));
                    }

                    for (int bank = 0; bank < 4; ++bank) {
                        for (int row = 0; row < 16; ++row) {
                            float * destination =
                                output + (row_base + row) * output_cols_ + first_col + panel_col + 16 * bank;
                            AMX_STZ(z_operand(destination, 4 * row + bank));
                        }
                    }
                }
            }
        }
    }

    int64_t panel_count() const { return output_cols_ / DSV4_AMX_PANEL_COLS; }

    int64_t                              output_cols_;
    int64_t                              input_cols_;
    size_t                               packed_count_ = 0;
    std::unique_ptr<float, aligned_free> packed_;
};

struct packed_layer {
    std::unique_ptr<direct_amx_weight> gate;
    std::unique_ptr<direct_amx_weight> up;
    std::unique_ptr<direct_amx_weight> down;
    float                              clamp_limit = 0.0f;
};

struct worker_timing {
    double gate_ms             = 0.0;
    double up_ms               = 0.0;
    double activation_ms       = 0.0;
    double hidden_transpose_ms = 0.0;
    double down_ms             = 0.0;
};

enum amx_stage {
    AMX_STAGE_GATE_UP,
    AMX_STAGE_HIDDEN,
    AMX_STAGE_DOWN,
};

class amx_stage_observer {
  public:
    virtual ~amx_stage_observer()                                                    = default;
    virtual void observe(amx_stage stage, const float * first, const float * second) = 0;
};

void execute_packed_layer(const packed_layer & weights,
                          float *              input_t,
                          float *              gate_hidden,
                          float *              up,
                          float *              output,
                          worker_timing &      timing,
                          amx_stage_observer * observer) {
    auto begin = clock_type::now();
    weights.gate->multiply_transposed(input_t, gate_hidden, DSV4_AMX_TOKENS);
    timing.gate_ms = elapsed_ms(begin, clock_type::now());

    begin = clock_type::now();
    weights.up->multiply_transposed(input_t, up, DSV4_AMX_TOKENS);
    timing.up_ms = elapsed_ms(begin, clock_type::now());
    if (observer != nullptr) {
        observer->observe(AMX_STAGE_GATE_UP, gate_hidden, up);
    }

    begin = clock_type::now();
    llama_dsv4_amx_apply_swiglu(gate_hidden, up, static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_FF),
                                weights.clamp_limit);
    timing.activation_ms = elapsed_ms(begin, clock_type::now());
    if (observer != nullptr) {
        observer->observe(AMX_STAGE_HIDDEN, gate_hidden, nullptr);
    }

    begin = clock_type::now();
    vDSP_mtrans(gate_hidden, 1, input_t, 1, static_cast<vDSP_Length>(DSV4_AMX_FF),
                static_cast<vDSP_Length>(DSV4_AMX_TOKENS));
    timing.hidden_transpose_ms = elapsed_ms(begin, clock_type::now());

    begin = clock_type::now();
    weights.down->multiply_transposed(input_t, output, DSV4_AMX_TOKENS);
    timing.down_ms = elapsed_ms(begin, clock_type::now());
    std::atomic_thread_fence(std::memory_order_release);
    if (observer != nullptr) {
        observer->observe(AMX_STAGE_DOWN, output, nullptr);
    }
}

struct oracle_stage_stats {
    bool   finite       = true;
    bool   nonzero      = false;
    double nmse         = 0.0;
    double max_abs      = 0.0;
    double reference_l2 = 0.0;
};

constexpr std::array<int64_t, 4> ORACLE_ROWS = { 0, 15, 16, DSV4_AMX_TOKENS - 1 };

float deterministic_oracle_value(int64_t row, int64_t col) {
    uint32_t bits = static_cast<uint32_t>(row) * 0x9e3779b9U;
    bits ^= static_cast<uint32_t>(col) * 0x85ebca6bU;
    bits ^= bits >> 16;
    return (static_cast<int32_t>(bits & 2047U) - 1024) * (1.0f / 32768.0f);
}

oracle_stage_stats compare_oracle_stage(const float * actual, int64_t cols, const std::vector<float> & reference) {
    oracle_stage_stats result;
    const size_t       total = static_cast<size_t>(DSV4_AMX_TOKENS * cols);
    for (size_t i = 0; i < total; ++i) {
        result.finite  = result.finite && std::isfinite(actual[i]);
        result.nonzero = result.nonzero || actual[i] != 0.0f;
    }
    double squared_error = 0.0;
    for (size_t selected = 0; selected < ORACLE_ROWS.size(); ++selected) {
        const float * actual_row    = actual + ORACLE_ROWS[selected] * cols;
        const float * reference_row = reference.data() + selected * cols;
        for (int64_t col = 0; col < cols; ++col) {
            const double lhs   = actual_row[col];
            const double rhs   = reference_row[col];
            result.finite      = result.finite && std::isfinite(lhs) && std::isfinite(rhs);
            const double error = lhs - rhs;
            squared_error += error * error;
            result.reference_l2 += rhs * rhs;
            result.max_abs = std::max(result.max_abs, std::abs(error));
        }
    }
    result.nmse =
        result.reference_l2 > 0.0 ? squared_error / result.reference_l2 : std::numeric_limits<double>::infinity();
    return result;
}

void require_oracle_stage(int layer, const char * stage, int64_t cols, const oracle_stage_stats & stats) {
    const bool reference_l2_valid = std::isfinite(stats.reference_l2) && stats.reference_l2 > 0.0;
    const bool metrics_finite     = std::isfinite(stats.nmse) && std::isfinite(stats.max_abs);
    const bool pass =
        llama_dsv4_amx_oracle_stage_passes(stats.finite, stats.nonzero, stats.reference_l2, stats.nmse, stats.max_abs);
    audit_record("oracle", pass ? "pass" : "fail",
                 "layer=%d stage=%s finite=%d nonzero=%d reference_l2_valid=%d metrics_finite=%d "
                 "full_actual_elements=%zu compared=%zu nmse=%.9e nmse_limit=%.9e max_abs=%.9e "
                 "max_abs_limit=%.9e reference_l2=%.9e",
                 layer, stage, stats.finite ? 1 : 0, stats.nonzero ? 1 : 0, reference_l2_valid ? 1 : 0,
                 metrics_finite ? 1 : 0, static_cast<size_t>(DSV4_AMX_TOKENS * cols),
                 ORACLE_ROWS.size() * static_cast<size_t>(cols), stats.nmse, DSV4_AMX_ORACLE_NMSE_LIMIT, stats.max_abs,
                 DSV4_AMX_ORACLE_MAX_ABS_LIMIT, stats.reference_l2);
    if (!pass) {
        throw std::runtime_error("DeepSeek V4 AMX deterministic oracle failed at layer " + std::to_string(layer) +
                                 " stage " + stage);
    }
}

class production_oracle_observer final : public amx_stage_observer {
  public:
    production_oracle_observer(int                        layer,
                               const std::vector<float> & gate,
                               const std::vector<float> & up,
                               const std::vector<float> & hidden,
                               const std::vector<float> & down) :
        layer_(layer),
        gate_(gate),
        up_(up),
        hidden_(hidden),
        down_(down) {}

    void observe(amx_stage stage, const float * first, const float * second) override {
        switch (stage) {
            case AMX_STAGE_GATE_UP:
                require_oracle_stage(layer_, "gate", DSV4_AMX_FF, compare_oracle_stage(first, DSV4_AMX_FF, gate_));
                require_oracle_stage(layer_, "up", DSV4_AMX_FF, compare_oracle_stage(second, DSV4_AMX_FF, up_));
                break;
            case AMX_STAGE_HIDDEN:
                require_oracle_stage(layer_, "post_swiglu", DSV4_AMX_FF,
                                     compare_oracle_stage(first, DSV4_AMX_FF, hidden_));
                break;
            case AMX_STAGE_DOWN:
                require_oracle_stage(layer_, "down", DSV4_AMX_EMBD, compare_oracle_stage(first, DSV4_AMX_EMBD, down_));
                break;
        }
    }

  private:
    int                        layer_;
    const std::vector<float> & gate_;
    const std::vector<float> & up_;
    const std::vector<float> & hidden_;
    const std::vector<float> & down_;
};

void run_production_oracle(const std::vector<packed_layer> & layers) {
    if (layers.size() != DSV4_AMX_LAYERS) {
        throw std::runtime_error("DeepSeek V4 AMX oracle requires all 43 packed layers");
    }

    std::vector<float> input_template(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_EMBD));
    std::vector<float> selected_input(ORACLE_ROWS.size() * static_cast<size_t>(DSV4_AMX_EMBD));
    for (int64_t k = 0; k < DSV4_AMX_EMBD; ++k) {
        for (int64_t row = 0; row < DSV4_AMX_TOKENS; ++row) {
            input_template[static_cast<size_t>(k * DSV4_AMX_TOKENS + row)] = deterministic_oracle_value(row, k);
        }
        for (size_t selected = 0; selected < ORACLE_ROWS.size(); ++selected) {
            selected_input[selected * DSV4_AMX_EMBD + k] =
                input_template[static_cast<size_t>(k * DSV4_AMX_TOKENS + ORACLE_ROWS[selected])];
        }
    }

    std::vector<float> input_t(input_template.size());
    std::vector<float> gate_hidden(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_FF));
    std::vector<float> up(gate_hidden.size());
    std::vector<float> output(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_EMBD));
    std::vector<float> gate_reference(ORACLE_ROWS.size() * static_cast<size_t>(DSV4_AMX_FF));
    std::vector<float> up_reference(gate_reference.size());
    std::vector<float> hidden_reference(gate_reference.size());
    std::vector<float> down_reference(ORACLE_ROWS.size() * static_cast<size_t>(DSV4_AMX_EMBD));

    audit_record("oracle", "begin", "layers=%zu rows=%zu nmse_limit=%.9e max_abs_limit=%.9e", layers.size(),
                 ORACLE_ROWS.size(), DSV4_AMX_ORACLE_NMSE_LIMIT, DSV4_AMX_ORACLE_MAX_ABS_LIMIT);
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        const packed_layer & weights = layers[layer];
        weights.gate->multiply_reference(selected_input.data(), gate_reference.data(), ORACLE_ROWS.size());
        weights.up->multiply_reference(selected_input.data(), up_reference.data(), ORACLE_ROWS.size());
        hidden_reference              = gate_reference;
        std::vector<float> up_clamped = up_reference;
        llama_dsv4_amx_apply_swiglu(hidden_reference.data(), up_clamped.data(), hidden_reference.size(),
                                    weights.clamp_limit);
        weights.down->multiply_reference(hidden_reference.data(), down_reference.data(), ORACLE_ROWS.size());

        input_t = input_template;
        worker_timing              timing;
        production_oracle_observer observer(static_cast<int>(layer), gate_reference, up_reference, hidden_reference,
                                            down_reference);
        execute_packed_layer(weights, input_t.data(), gate_hidden.data(), up.data(), output.data(), timing, &observer);
    }
    audit_record("oracle", "complete", "layers=%zu stage_records=%zu", layers.size(), layers.size() * 4);
}

class amx_worker {
  public:
    explicit amx_worker(const std::vector<packed_layer> & layers) :
        layers_(layers),
        input_t_(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_EMBD)),
        gate_hidden_(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_FF)),
        up_(static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_FF)),
        thread_(&amx_worker::thread_main, this) {}

    ~amx_worker() { shutdown(); }

    float * input_t() { return input_t_.data(); }

    void start(int layer, float * output) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopping_ || ready_ || !done_) {
            throw std::runtime_error("DeepSeek V4 AMX worker received overlapping jobs");
        }
        if (layer < 0 || layer >= static_cast<int>(layers_.size()) || output == nullptr) {
            throw std::runtime_error("DeepSeek V4 AMX worker received an invalid job");
        }
        layer_  = layer;
        output_ = output;
        error_.clear();
        timing_ = {};
        done_   = false;
        ready_  = true;
        cv_.notify_all();
    }

    bool wait(worker_timing & timing, std::string & error) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return done_ || stopping_; });
        timing = timing_;
        error  = error_;
        return error.empty() && done_;
    }

    bool active() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ready_ || !done_;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
            cv_.notify_all();
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

  private:
    void thread_main() {
        for (;;) {
            int     layer  = -1;
            float * output = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return ready_ || stopping_; });
                if (stopping_ && !ready_) {
                    return;
                }
                layer  = layer_;
                output = output_;
                ready_ = false;
            }

            worker_timing timing;
            std::string   error;
            try {
                const packed_layer & weights = layers_.at(static_cast<size_t>(layer));
                execute_packed_layer(weights, input_t_.data(), gate_hidden_.data(), up_.data(), output, timing,
                                     nullptr);
            } catch (const std::exception & exception) {
                error = exception.what();
                if (output != nullptr) {
                    std::fill(output, output + DSV4_AMX_TOKENS * DSV4_AMX_EMBD,
                              std::numeric_limits<float>::quiet_NaN());
                }
            } catch (...) {
                error = "unknown exception in DeepSeek V4 AMX worker";
                if (output != nullptr) {
                    std::fill(output, output + DSV4_AMX_TOKENS * DSV4_AMX_EMBD,
                              std::numeric_limits<float>::quiet_NaN());
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                timing_ = timing;
                error_  = std::move(error);
                done_   = true;
                output_ = nullptr;
                cv_.notify_all();
            }
        }
    }

    const std::vector<packed_layer> & layers_;
    std::vector<float>                input_t_;
    std::vector<float>                gate_hidden_;
    std::vector<float>                up_;

    mutable std::mutex      mutex_;
    std::condition_variable cv_;
    bool                    stopping_ = false;
    bool                    ready_    = false;
    bool                    done_     = true;
    int                     layer_    = -1;
    float *                 output_   = nullptr;
    worker_timing           timing_;
    std::string             error_;
    std::thread             thread_;
};

#endif  // LLAMA_DSV4_AMX_APPLE_ARM64

}  // namespace

void llama_dsv4_amx_apply_swiglu(float * gate_hidden, float * up, size_t count, float clamp_limit) {
    if (gate_hidden == nullptr || up == nullptr) {
        throw std::invalid_argument("DeepSeek V4 AMX SwiGLU received a null buffer");
    }
    for (size_t i = 0; i < count; ++i) {
        float gate     = gate_hidden[i];
        float up_value = up[i];
        if (clamp_limit > 1e-6f) {
            gate     = std::min(gate, clamp_limit);
            up_value = std::max(-clamp_limit, std::min(up_value, clamp_limit));
            up[i]    = up_value;
        }
        gate_hidden[i] = gate / (1.0f + std::exp(-gate)) * up_value;
    }
}

size_t llama_dsv4_amx_scratch_bytes(int64_t tokens) {
    if (tokens <= 0) {
        return 0;
    }
    return static_cast<size_t>(tokens) * static_cast<size_t>(DSV4_AMX_EMBD + 2 * DSV4_AMX_FF) * sizeof(float);
}

size_t llama_dsv4_amx_output_bytes(int64_t tokens) {
    if (tokens <= 0) {
        return 0;
    }
    return static_cast<size_t>(tokens) * static_cast<size_t>(DSV4_AMX_EMBD) * sizeof(float);
}

size_t llama_dsv4_amx_incremental_context_bytes(int64_t tokens) {
    return llama_dsv4_amx_scratch_bytes(tokens) + llama_dsv4_amx_output_bytes(tokens);
}

bool llama_dsv4_amx_oracle_stage_passes(bool finite, bool nonzero, double reference_l2, double nmse, double max_abs) {
    return finite && nonzero && std::isfinite(reference_l2) && reference_l2 > 0.0 && std::isfinite(nmse) &&
           nmse <= DSV4_AMX_ORACLE_NMSE_LIMIT && std::isfinite(max_abs) && max_abs <= DSV4_AMX_ORACLE_MAX_ABS_LIMIT;
}

bool llama_dsv4_amx_telemetry_stage_is_valid(bool   direct_finite,
                                             bool   direct_nonzero,
                                             bool   reference_finite,
                                             double reference_l2,
                                             double nmse,
                                             double max_abs) {
    return direct_finite && direct_nonzero && reference_finite && std::isfinite(reference_l2) && reference_l2 > 0.0 &&
           std::isfinite(nmse) && nmse >= 0.0 && std::isfinite(max_abs) && max_abs >= 0.0;
}

bool llama_dsv4_amx_output_tensor_is_exact(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || tensor->op != GGML_OP_NONE ||
        tensor->view_src != nullptr || tensor->ne[0] != DSV4_AMX_EMBD || tensor->ne[1] != DSV4_AMX_TOKENS ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor) ||
        ggml_nbytes(tensor) != llama_dsv4_amx_output_bytes(DSV4_AMX_TOKENS)) {
        return false;
    }
    for (const ggml_tensor * source : tensor->src) {
        if (source != nullptr) {
            return false;
        }
    }
    return true;
}

bool llama_dsv4_amx_bind_output_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (buffer == nullptr || tensor == nullptr || !llama_dsv4_amx_output_tensor_is_exact(tensor) ||
        tensor->buffer != nullptr || tensor->data != nullptr) {
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    if (base == nullptr || ggml_backend_buffer_get_size(buffer) < llama_dsv4_amx_output_bytes(DSV4_AMX_TOKENS) ||
        ggml_backend_tensor_alloc(buffer, tensor, base) != GGML_STATUS_SUCCESS) {
        return false;
    }
    return tensor->buffer == buffer && tensor->data == base;
}

bool llama_dsv4_amx_requested() {
    return env_is_one("LLAMA_DSV4_AMX_COEXEC") && !env_is_one("LLAMA_DSV4_AMX_COEXEC_DISABLE");
}

llama_dsv4_amx_context_gate llama_dsv4_amx_context_gate_for(const llama_cparams & cparams) {
    if (cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT) {
        return LLAMA_DSV4_AMX_CONTEXT_TYPE;
    }
    if (cparams.n_rs_seq != 0) {
        return LLAMA_DSV4_AMX_CONTEXT_RECURRENT;
    }
    if (cparams.n_ubatch != DSV4_AMX_TOKENS) {
        return LLAMA_DSV4_AMX_CONTEXT_UBATCH;
    }
    if (cparams.n_seq_max != 1) {
        return LLAMA_DSV4_AMX_CONTEXT_MULTI_SEQUENCE;
    }
    if (cparams.pipeline_parallel) {
        return LLAMA_DSV4_AMX_CONTEXT_PIPELINE;
    }
    if (cparams.cb_eval != nullptr) {
        return LLAMA_DSV4_AMX_CONTEXT_CALLBACK;
    }
    return LLAMA_DSV4_AMX_CONTEXT_ELIGIBLE;
}

const char * llama_dsv4_amx_context_gate_name(llama_dsv4_amx_context_gate gate) {
    switch (gate) {
        case LLAMA_DSV4_AMX_CONTEXT_ELIGIBLE:
            return "eligible";
        case LLAMA_DSV4_AMX_CONTEXT_TYPE:
            return "context_type";
        case LLAMA_DSV4_AMX_CONTEXT_RECURRENT:
            return "recurrent_state";
        case LLAMA_DSV4_AMX_CONTEXT_UBATCH:
            return "unsupported_ubatch";
        case LLAMA_DSV4_AMX_CONTEXT_MULTI_SEQUENCE:
            return "multi_sequence";
        case LLAMA_DSV4_AMX_CONTEXT_PIPELINE:
            return "pipeline_parallel";
        case LLAMA_DSV4_AMX_CONTEXT_CALLBACK:
            return "user_callback";
    }
    return "unknown";
}

const char * llama_dsv4_amx_callback_error_name(llama_dsv4_amx_callback_error error) {
    switch (error) {
        case LLAMA_DSV4_AMX_CALLBACK_OK:
            return "ok";
        case LLAMA_DSV4_AMX_CALLBACK_LAYER_RANGE:
            return "layer_range";
        case LLAMA_DSV4_AMX_CALLBACK_PENDING_AFTER:
            return "pending_after";
        case LLAMA_DSV4_AMX_CALLBACK_ORDER:
            return "order";
        case LLAMA_DSV4_AMX_CALLBACK_DUPLICATE:
            return "duplicate";
        case LLAMA_DSV4_AMX_CALLBACK_AFTER_WITHOUT_ASK:
            return "after_without_ask";
        case LLAMA_DSV4_AMX_CALLBACK_AFTER_MISMATCH:
            return "after_mismatch";
    }
    return "unknown";
}

void llama_dsv4_amx_callback_order::begin() {
    starts        = 0;
    ends          = 0;
    layer         = 0;
    start         = true;
    pending       = false;
    pending_layer = -1;
    pending_start = false;
}

llama_dsv4_amx_callback_error llama_dsv4_amx_callback_order::ask(int event_layer, bool event_start) {
    if (event_layer < 0 || event_layer >= DSV4_AMX_LAYERS) {
        return LLAMA_DSV4_AMX_CALLBACK_LAYER_RANGE;
    }
    if (pending) {
        return LLAMA_DSV4_AMX_CALLBACK_PENDING_AFTER;
    }
    const uint64_t bit = 1ULL << event_layer;
    if ((event_start ? starts : ends) & bit) {
        return LLAMA_DSV4_AMX_CALLBACK_DUPLICATE;
    }
    if (event_layer != layer || event_start != start) {
        return LLAMA_DSV4_AMX_CALLBACK_ORDER;
    }
    pending       = true;
    pending_layer = event_layer;
    pending_start = event_start;
    return LLAMA_DSV4_AMX_CALLBACK_OK;
}

llama_dsv4_amx_callback_error llama_dsv4_amx_callback_order::after(int event_layer, bool event_start) {
    if (!pending) {
        return LLAMA_DSV4_AMX_CALLBACK_AFTER_WITHOUT_ASK;
    }
    if (event_layer != pending_layer || event_start != pending_start) {
        return LLAMA_DSV4_AMX_CALLBACK_AFTER_MISMATCH;
    }
    const uint64_t bit = 1ULL << event_layer;
    if (event_start) {
        starts |= bit;
        start = false;
    } else {
        ends |= bit;
        ++layer;
        start = true;
    }
    pending       = false;
    pending_layer = -1;
    pending_start = false;
    return LLAMA_DSV4_AMX_CALLBACK_OK;
}

bool llama_dsv4_amx_callback_order::complete() const {
    constexpr uint64_t all_layers = (1ULL << DSV4_AMX_LAYERS) - 1;
    return !pending && layer == DSV4_AMX_LAYERS && start && starts == all_layers && ends == all_layers;
}

int llama_dsv4_amx_callback_order::expected_layer() const {
    return layer;
}

bool llama_dsv4_amx_callback_order::expected_start() const {
    return start;
}

uint64_t llama_dsv4_amx_callback_order::starts_seen() const {
    return starts;
}

uint64_t llama_dsv4_amx_callback_order::ends_seen() const {
    return ends;
}

struct llama_dsv4_amx_model::impl {
    mutable std::mutex mutex;
#if defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    std::vector<packed_layer> layers;
    bool                      oracle_complete = false;
#endif
};

llama_dsv4_amx_model::llama_dsv4_amx_model(std::unique_ptr<impl> pimpl) : pimpl(std::move(pimpl)) {}

llama_dsv4_amx_model::~llama_dsv4_amx_model() = default;

std::shared_ptr<const llama_dsv4_amx_model> llama_dsv4_amx_model_create(const llama_model & model) {
    if (!llama_dsv4_amx_requested()) {
        if (env_is_one("LLAMA_DSV4_AMX_COEXEC")) {
            audit_record("eligibility", "fallback", "scope=model reason=explicit_disable");
        }
        return nullptr;
    }
    if (model.arch != LLM_ARCH_DEEPSEEK4) {
        audit_record("eligibility", "fallback", "scope=model reason=model_arch");
        return nullptr;
    }
    if (!has_target_shape(model)) {
        audit_record("eligibility", "fallback", "scope=model reason=sidecar_or_shape");
        return nullptr;
    }

#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    audit_record("eligibility", "fallback", "scope=model reason=platform");
    return nullptr;
#else
    const std::string brand = cpu_brand();
    if (brand != "Apple M2 Ultra") {
        audit_record("eligibility", "fallback", "scope=model reason=cpu_brand detected=%s", brand.c_str());
        return nullptr;
    }
    if (model.devices.size() != 1 || model.n_gpu_layers() <= model.hparams.n_layer_all ||
        model.has_tensor_overrides()) {
        audit_record("eligibility", "fallback",
                     "scope=model reason=offload devices=%zu gpu_layers=%d total_layers=%u overrides=%d",
                     model.devices.size(), model.n_gpu_layers(), model.hparams.n_layer_all,
                     model.has_tensor_overrides() ? 1 : 0);
        return nullptr;
    }
    ggml_backend_reg_t model_registry = metal_registry_for_device(model.devices[0].dev);
    if (model_registry == nullptr || resolve_tensor_host_ptr(model_registry) == nullptr) {
        audit_record("eligibility", "fallback", "scope=model reason=metal_host_pointer_proc");
        return nullptr;
    }

    audit_record("eligibility", "eligible", "scope=model lazy_pack=1 layers=%d", DSV4_AMX_LAYERS);
    return std::shared_ptr<const llama_dsv4_amx_model>(
        new llama_dsv4_amx_model(std::make_unique<llama_dsv4_amx_model::impl>()));
#endif
}

void llama_dsv4_amx_model::ensure_packed(const llama_model & model, bool run_oracle) const {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(model);
    GGML_UNUSED(run_oracle);
    audit_record("pack", "fallback", "reason=platform");
#else
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->layers.empty()) {
        audit_record("pack", "begin", "layers=%d expected_bytes=%zu", DSV4_AMX_LAYERS, DSV4_AMX_PACKED_BYTES);
        std::vector<packed_layer> layers;
        layers.reserve(DSV4_AMX_LAYERS);
        std::unordered_set<const ggml_tensor *> source_tensors;
        source_tensors.reserve(3 * DSV4_AMX_LAYERS);
        std::vector<std::pair<uintptr_t, uintptr_t>> source_ranges;
        source_ranges.reserve(3 * DSV4_AMX_LAYERS);
        size_t     source_bytes  = 0;
        size_t     packed_bytes  = 0;
        size_t     packed_values = 0;
        const auto begin         = clock_type::now();

        try {
            for (int il = 0; il < DSV4_AMX_LAYERS; ++il) {
                const llama_layer & layer = model.layers.at(static_cast<size_t>(il));
                require_tensor_shape(layer.ffn_gate_shexp, DSV4_AMX_EMBD, DSV4_AMX_FF, il, "gate");
                require_tensor_shape(layer.ffn_up_shexp, DSV4_AMX_EMBD, DSV4_AMX_FF, il, "up");
                require_tensor_shape(layer.ffn_down_shexp, DSV4_AMX_FF, DSV4_AMX_EMBD, il, "down");
                if (!source_tensors.emplace(layer.ffn_gate_shexp).second ||
                    !source_tensors.emplace(layer.ffn_up_shexp).second ||
                    !source_tensors.emplace(layer.ffn_down_shexp).second) {
                    throw std::runtime_error("DeepSeek V4 AMX layer binding reused a shared-expert weight tensor");
                }
                const float clamp_limit = model.hparams.swiglu_clamp_shexp[static_cast<size_t>(il)];
                if (!std::isfinite(clamp_limit) || clamp_limit < 0.0f) {
                    throw std::runtime_error("DeepSeek V4 AMX layer has an invalid split-SwiGLU clamp");
                }

                struct source_binding {
                    const char *        role;
                    const ggml_tensor * tensor;
                    const ggml_bf16_t * data;
                };

                const std::array<source_binding, 3> sources = {
                    source_binding{ "gate", layer.ffn_gate_shexp,
                                   require_shared_bf16_pointer(layer.ffn_gate_shexp, il, "gate") },
                    source_binding{ "up",   layer.ffn_up_shexp,
                                   require_shared_bf16_pointer(layer.ffn_up_shexp,   il, "up")   },
                    source_binding{ "down", layer.ffn_down_shexp,
                                   require_shared_bf16_pointer(layer.ffn_down_shexp, il, "down") },
                };
                for (const source_binding & source : sources) {
                    const size_t bytes = ggml_nbytes(source.tensor);
                    const auto   first = reinterpret_cast<uintptr_t>(source.data);
                    if (ggml_nelements(source.tensor) != DSV4_AMX_MATRIX_ELEMENTS || bytes != DSV4_AMX_SOURCE_BYTES ||
                        first == 0 || first > std::numeric_limits<uintptr_t>::max() - bytes) {
                        throw std::runtime_error("DeepSeek V4 AMX source tensor violated its exact storage range");
                    }
                    const uintptr_t last = first + bytes;
                    for (const auto & existing : source_ranges) {
                        if (first < existing.second && existing.first < last) {
                            throw std::runtime_error("DeepSeek V4 AMX source weight ranges overlap or alias");
                        }
                    }
                    source_ranges.emplace_back(first, last);
                    source_bytes += bytes;
                }

                packed_layer packed;
                packed.gate = std::make_unique<direct_amx_weight>(sources[0].data, DSV4_AMX_FF, DSV4_AMX_EMBD);
                packed.up   = std::make_unique<direct_amx_weight>(sources[1].data, DSV4_AMX_FF, DSV4_AMX_EMBD);
                packed.down = std::make_unique<direct_amx_weight>(sources[2].data, DSV4_AMX_EMBD, DSV4_AMX_FF);
                uint32_t clamp_bits;
                std::memcpy(&clamp_bits, &clamp_limit, sizeof(clamp_bits));
                for (const source_binding & source : sources) {
                    audit_record("pack_tensor", "pass",
                                 "layer=%d role=%s tensor=%p source=%p source_bytes=%zu name=%s elements=%zu "
                                 "exact_bf16_bits=1 mapping=output_input_to_panel_k_col clamp_bits=0x%08x",
                                 il, source.role, static_cast<const void *>(source.tensor),
                                 static_cast<const void *>(source.data), ggml_nbytes(source.tensor),
                                 source.tensor->name, static_cast<size_t>(ggml_nelements(source.tensor)), clamp_bits);
                }
                packed.clamp_limit = clamp_limit;
                packed_bytes += packed.gate->packed_bytes() + packed.up->packed_bytes() + packed.down->packed_bytes();
                packed_values +=
                    (packed.gate->packed_bytes() + packed.up->packed_bytes() + packed.down->packed_bytes()) /
                    sizeof(float);
                layers.emplace_back(std::move(packed));
            }
        } catch (const std::exception & exception) {
            audit_record("pack", "fail", "reason=%s", exception.what());
            throw;
        }

        const double pack_ms = elapsed_ms(begin, clock_type::now());
        if (packed_bytes != DSV4_AMX_PACKED_BYTES || packed_values != DSV4_AMX_PACKED_VALUES ||
            source_bytes != 3 * DSV4_AMX_LAYERS * DSV4_AMX_SOURCE_BYTES || layers.size() != DSV4_AMX_LAYERS ||
            source_tensors.size() != 3 * DSV4_AMX_LAYERS || source_ranges.size() != 3 * DSV4_AMX_LAYERS) {
            audit_record("pack", "fail",
                         "reason=contract bytes=%zu values=%zu source_bytes=%zu layers=%zu unique_tensors=%zu "
                         "unique_sources=%zu",
                         packed_bytes, packed_values, source_bytes, layers.size(), source_tensors.size(),
                         source_ranges.size());
            throw std::runtime_error("DeepSeek V4 AMX pack violated its exact size contract");
        }
        audit_record("pack_contract", "complete",
                     "layers=%zu matrices=%zu unique_tensors=%zu unique_sources=%zu source_bytes=%zu values=%zu "
                     "exact_bf16_expand=1 orientation=output_input_to_panel_k_col clamp_metadata=1 clamp_layers=%d",
                     layers.size(), source_tensors.size(), source_tensors.size(), source_ranges.size(), source_bytes,
                     packed_values, DSV4_AMX_LAYERS);
        pimpl->layers = std::move(layers);
        audit_record("pack", "complete", "layers=%zu bytes=%zu ms=%.3f", pimpl->layers.size(), packed_bytes, pack_ms);
    }

    if (run_oracle && !pimpl->oracle_complete) {
        try {
            run_production_oracle(pimpl->layers);
            pimpl->oracle_complete = true;
        } catch (const std::exception & exception) {
            audit_record("oracle", "fail", "scope=all_layers reason=%s", exception.what());
            throw;
        }
    }
#endif
}

struct llama_dsv4_amx_context::impl {
#if defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    struct event {
        int  layer;
        bool start;
    };

    struct binding {
        ggml_tensor * norm      = nullptr;
        ggml_tensor * output    = nullptr;
        ggml_tensor * reference = nullptr;
    };

    const llama_model *                         model = nullptr;
    std::shared_ptr<const llama_dsv4_amx_model> model_pack;
    ggml_backend_sched_t                        sched         = nullptr;
    ggml_backend_t                              metal_backend = nullptr;
    ggml_backend_buffer_type_t                  metal_buft    = nullptr;
    metal_tensor_host_ptr_fn                    get_host_ptr  = nullptr;
    bool                                        validate      = false;
    bool                                        timing        = false;
    std::unique_ptr<amx_worker>                 worker;
    ggml_backend_buffer_ptr                     output_buffer;
    size_t                                      output_buffer_bytes = 0;

    const ggml_cgraph *                      active_graph = nullptr;
    llm_dsv4_amx_mode                        active_mode  = LLM_DSV4_AMX_DISABLED;
    std::array<binding, DSV4_AMX_LAYERS>     bindings;
    std::unordered_map<ggml_tensor *, event> events;

    bool                          poisoned = false;
    std::string                   poison_reason;
    bool                          submission_started = false;
    int                           completed_layers   = 0;
    uint64_t                      submission_index   = 0;
    llama_dsv4_amx_callback_order callback_order;
    double                        graph_transpose_ms = 0.0;
    double                        graph_wait_ms      = 0.0;
    worker_timing                 graph_worker_timing;

    void poison(const std::string & reason, ggml_tensor * output = nullptr) {
        // Failure callbacks can arrive while the AMX worker still owns the
        // shared output. Join before writing poison so fail-closed handling is
        // data-race-free as well as the accepted callback path.
        if (worker && worker->active()) {
            worker_timing ignored_timing;
            std::string   ignored_error;
            worker->wait(ignored_timing, ignored_error);
            std::atomic_thread_fence(std::memory_order_acquire);
        }
        if (!poisoned) {
            poison_reason = reason;
        }
        poisoned             = true;
        bool output_poisoned = false;
        if (llama_dsv4_amx_output_tensor_is_exact(output) && output_buffer && output->buffer == output_buffer.get() &&
            output->data == ggml_backend_buffer_get_base(output_buffer.get()) && get_host_ptr != nullptr) {
            void * pointer = get_host_ptr(output);
            if (pointer == output->data) {
                float * data = static_cast<float *>(pointer);
                std::fill(data, data + ggml_nelements(output), std::numeric_limits<float>::quiet_NaN());
                output_poisoned = true;
            }
        }
        audit_record("runtime", "poison", "output_poisoned=%d reason=%s", output_poisoned ? 1 : 0,
                     poison_reason.c_str());
    }

    float * host_pointer(ggml_tensor * tensor, const char * role, int layer) {
        void * pointer = tensor == nullptr || get_host_ptr == nullptr ? nullptr : get_host_ptr(tensor);
        if (pointer == nullptr) {
            throw std::runtime_error("DeepSeek V4 AMX could not access Metal-shared " + std::string(role) +
                                     " tensor at layer " + std::to_string(layer));
        }
        return static_cast<float *>(pointer);
    }

    void bind_output_leaf(ggml_tensor * tensor, int layer) {
        if (!llama_dsv4_amx_output_tensor_is_exact(tensor)) {
            throw std::runtime_error("DeepSeek V4 AMX output is not an exact producerless F32 leaf at layer " +
                                     std::to_string(layer));
        }
        if (tensor->buffer != nullptr || tensor->data != nullptr) {
            throw std::runtime_error("DeepSeek V4 AMX output leaf was allocated before explicit binding at layer " +
                                     std::to_string(layer));
        }
        if (!output_buffer) {
            const size_t bytes = ggml_backend_buft_get_alloc_size(metal_buft, tensor);
            if (bytes != llama_dsv4_amx_output_bytes(DSV4_AMX_TOKENS)) {
                throw std::runtime_error("DeepSeek V4 AMX output buffer violated the exact 32 MiB contract");
            }
            output_buffer.reset(ggml_backend_buft_alloc_buffer(metal_buft, bytes));
            if (!output_buffer || ggml_backend_buffer_get_base(output_buffer.get()) == nullptr ||
                ggml_backend_buffer_get_size(output_buffer.get()) != bytes) {
                throw std::runtime_error("DeepSeek V4 AMX could not allocate its context-owned shared output buffer");
            }
            output_buffer_bytes = bytes;
            audit_record("context_output", "allocated", "bytes=%zu base=%p", bytes,
                         ggml_backend_buffer_get_base(output_buffer.get()));
        }

        void * base = ggml_backend_buffer_get_base(output_buffer.get());
        if (!llama_dsv4_amx_bind_output_tensor(output_buffer.get(), tensor) || get_host_ptr(tensor) != base) {
            throw std::runtime_error("DeepSeek V4 AMX shared output leaf binding failed at layer " +
                                     std::to_string(layer));
        }
    }

    void reset_submission_counters() {
        submission_started = true;
        completed_layers   = 0;
        callback_order.begin();
        graph_transpose_ms  = 0.0;
        graph_wait_ms       = 0.0;
        graph_worker_timing = {};
    }

    void record_metal_telemetry(int layer, const float * direct, const float * reference) {
        double       squared_error    = 0.0;
        double       reference_l2     = 0.0;
        double       max_abs          = 0.0;
        bool         direct_finite    = true;
        bool         direct_nonzero   = false;
        bool         reference_finite = true;
        const size_t count            = static_cast<size_t>(DSV4_AMX_TOKENS * DSV4_AMX_EMBD);
        for (size_t i = 0; i < count; ++i) {
            const double lhs   = direct[i];
            const double rhs   = reference[i];
            direct_finite      = direct_finite && std::isfinite(lhs);
            direct_nonzero     = direct_nonzero || lhs != 0.0;
            reference_finite   = reference_finite && std::isfinite(rhs);
            const double error = lhs - rhs;
            squared_error += error * error;
            reference_l2 += rhs * rhs;
            max_abs = std::max(max_abs, std::abs(error));
        }
        const double nmse = reference_l2 > 0.0 ? squared_error / reference_l2 : std::numeric_limits<double>::infinity();
        const bool   reference_l2_valid = std::isfinite(reference_l2) && reference_l2 > 0.0;
        const bool   metrics_finite     = std::isfinite(nmse) && std::isfinite(max_abs);
        const bool   valid = llama_dsv4_amx_telemetry_stage_is_valid(direct_finite, direct_nonzero, reference_finite,
                                                                     reference_l2, nmse, max_abs);
        audit_record("metal_telemetry", valid ? "recorded" : "fail",
                     "layer=%d stage=down direct_finite=%d direct_nonzero=%d reference_finite=%d "
                     "reference_l2_valid=%d metrics_finite=%d nmse=%.9e max_abs=%.9e reference_l2=%.9e "
                     "legacy_nmse_limit=1.000000000e-03 within_legacy_nmse=%d divergence_rejects=0",
                     layer, direct_finite ? 1 : 0, direct_nonzero ? 1 : 0, reference_finite ? 1 : 0,
                     reference_l2_valid ? 1 : 0, metrics_finite ? 1 : 0, nmse, max_abs, reference_l2,
                     metrics_finite && nmse <= 1e-3 ? 1 : 0);
        if (!valid) {
            throw std::runtime_error("DeepSeek V4 AMX retained-Metal telemetry health gate failed at layer " +
                                     std::to_string(layer));
        }
    }

    bool callback(ggml_tensor * tensor, bool ask) {
        const auto found = events.find(tensor);
        if (found == events.end()) {
            return poisoned && ask;
        }
        const event current = found->second;
        binding &   bound   = bindings.at(static_cast<size_t>(current.layer));
        if (ask) {
            if (poisoned) {
                return true;
            }
            const llama_dsv4_amx_callback_error error = callback_order.ask(current.layer, current.start);
            audit_record("callback", error == LLAMA_DSV4_AMX_CALLBACK_OK ? "ask" : "fail",
                         "submission=%llu layer=%d edge=%s error=%s", static_cast<unsigned long long>(submission_index),
                         current.layer, current.start ? "start" : "end", llama_dsv4_amx_callback_error_name(error));
            if (error != LLAMA_DSV4_AMX_CALLBACK_OK) {
                poison("DeepSeek V4 AMX callback ask invariant failed: " +
                           std::string(llama_dsv4_amx_callback_error_name(error)),
                       bound.output);
            }
            return true;
        }

        try {
            if (poisoned) {
                return false;
            }
            const llama_dsv4_amx_callback_error order_error = callback_order.after(current.layer, current.start);
            audit_record("callback", order_error == LLAMA_DSV4_AMX_CALLBACK_OK ? "after" : "fail",
                         "submission=%llu layer=%d edge=%s error=%s", static_cast<unsigned long long>(submission_index),
                         current.layer, current.start ? "start" : "end",
                         llama_dsv4_amx_callback_error_name(order_error));
            if (order_error != LLAMA_DSV4_AMX_CALLBACK_OK) {
                throw std::runtime_error("DeepSeek V4 AMX callback after invariant failed: " +
                                         std::string(llama_dsv4_amx_callback_error_name(order_error)));
            }
            if (current.start) {
                if (worker->active()) {
                    throw std::runtime_error("DeepSeek V4 AMX norm callback observed an unfinished prior layer");
                }
                if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != DSV4_AMX_EMBD ||
                    tensor->ne[1] != DSV4_AMX_TOKENS || !ggml_is_contiguous(tensor)) {
                    throw std::runtime_error("DeepSeek V4 AMX norm tensor violated the exact F32 production shape");
                }
                if (!llama_dsv4_amx_output_tensor_is_exact(bound.output) ||
                    bound.output->buffer != output_buffer.get() ||
                    bound.output->data != ggml_backend_buffer_get_base(output_buffer.get())) {
                    throw std::runtime_error("DeepSeek V4 AMX output leaf violated the exact F32 production shape");
                }

                const float * source = host_pointer(bound.norm, "norm", current.layer);
                float *       output = host_pointer(bound.output, "output", current.layer);
                const auto    begin  = clock_type::now();
                vDSP_mtrans(source, 1, worker->input_t(), 1, static_cast<vDSP_Length>(DSV4_AMX_EMBD),
                            static_cast<vDSP_Length>(DSV4_AMX_TOKENS));
                graph_transpose_ms += elapsed_ms(begin, clock_type::now());
                worker->start(current.layer, output);
                return true;
            }

            const auto    wait_begin = clock_type::now();
            worker_timing worker_result;
            std::string   worker_error;
            if (!worker->wait(worker_result, worker_error)) {
                throw std::runtime_error("DeepSeek V4 AMX worker failed: " + worker_error);
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            graph_wait_ms += elapsed_ms(wait_begin, clock_type::now());
            graph_worker_timing.gate_ms += worker_result.gate_ms;
            graph_worker_timing.up_ms += worker_result.up_ms;
            graph_worker_timing.activation_ms += worker_result.activation_ms;
            graph_worker_timing.hidden_transpose_ms += worker_result.hidden_transpose_ms;
            graph_worker_timing.down_ms += worker_result.down_ms;

            if (active_mode == LLM_DSV4_AMX_VALIDATE) {
                const float * direct    = host_pointer(bound.output, "output", current.layer);
                const float * reference = host_pointer(bound.reference, "retained reference", current.layer);
                record_metal_telemetry(current.layer, direct, reference);
            }
            audit_record("output_visibility", "ready",
                         "submission=%llu layer=%d shared_buffer=1 upload=0 worker_join=1 release_acquire=1",
                         static_cast<unsigned long long>(submission_index), current.layer);
            ++completed_layers;
            return true;
        } catch (const std::exception & exception) {
            poison(exception.what(), bound.output);
            LLAMA_LOG_ERROR("%s: %s\n", __func__, poison_reason.c_str());
            return false;
        } catch (...) {
            poison("unknown exception in DeepSeek V4 AMX scheduler callback");
            LLAMA_LOG_ERROR("%s: %s\n", __func__, poison_reason.c_str());
            return false;
        }
    }
#endif
};

llama_dsv4_amx_context::llama_dsv4_amx_context(std::unique_ptr<impl> pimpl) : pimpl(std::move(pimpl)) {}

llama_dsv4_amx_context::~llama_dsv4_amx_context() {
    shutdown();
}

std::unique_ptr<llama_dsv4_amx_context> llama_dsv4_amx_context_create(
    const llama_model &                             model,
    const llama_cparams &                           cparams,
    const std::vector<ggml_backend_t> &             backends,
    const std::vector<ggml_backend_buffer_type_t> & buffer_types) {
    if (!llama_dsv4_amx_requested() || !has_target_shape(model)) {
        return nullptr;
    }

#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(cparams);
    GGML_UNUSED(backends);
    GGML_UNUSED(buffer_types);
    audit_record("context", "fallback", "reason=platform");
    return nullptr;
#else
    const auto & model_pack = model.dsv4_amx_model();
    if (model_pack == nullptr) {
        audit_record("context", "fallback", "reason=model_ineligible");
        return nullptr;
    }
    const llama_dsv4_amx_context_gate context_gate = llama_dsv4_amx_context_gate_for(cparams);
    if (context_gate != LLAMA_DSV4_AMX_CONTEXT_ELIGIBLE) {
        audit_record("context", "fallback", "reason=%s", llama_dsv4_amx_context_gate_name(context_gate));
        return nullptr;
    }
    if (backends.size() != buffer_types.size()) {
        audit_record("context", "fallback", "reason=backend_configuration");
        return nullptr;
    }

    ggml_backend_t             metal_backend  = nullptr;
    ggml_backend_reg_t         metal_registry = nullptr;
    ggml_backend_buffer_type_t metal_buft     = nullptr;
    for (size_t i = 0; i < backends.size(); ++i) {
        ggml_backend_dev_t device   = ggml_backend_get_device(backends[i]);
        ggml_backend_reg_t registry = metal_registry_for_device(device);
        if (registry != nullptr) {
            if (metal_backend != nullptr) {
                audit_record("context", "fallback", "reason=multiple_metal_backends");
                return nullptr;
            }
            metal_backend  = backends[i];
            metal_registry = registry;
            metal_buft     = buffer_types[i];
        }
    }
    if (metal_backend == nullptr) {
        audit_record("context", "fallback", "reason=no_metal_backend");
        return nullptr;
    }

    auto buft_is_shared = reinterpret_cast<metal_buft_is_shared_fn>(
        ggml_backend_reg_get_proc_address(metal_registry, "ggml_backend_metal_buffer_type_is_shared"));
    auto get_host_ptr = resolve_tensor_host_ptr(metal_registry);
    if (buft_is_shared == nullptr || !buft_is_shared(metal_buft) || get_host_ptr == nullptr) {
        audit_record("context", "fallback", "reason=metal_shared_storage");
        return nullptr;
    }

    auto result_impl           = std::make_unique<llama_dsv4_amx_context::impl>();
    result_impl->model         = &model;
    result_impl->model_pack    = model_pack;
    result_impl->metal_backend = metal_backend;
    result_impl->metal_buft    = metal_buft;
    result_impl->get_host_ptr  = get_host_ptr;
    result_impl->validate      = validation_requested();
    result_impl->timing        = timing_requested();

    audit_record("context", "eligible", "lazy_pack=1 lazy_worker=1 validation=%d timing=%d",
                 result_impl->validate ? 1 : 0, result_impl->timing ? 1 : 0);
    return std::unique_ptr<llama_dsv4_amx_context>(new llama_dsv4_amx_context(std::move(result_impl)));
#endif
}

void llama_dsv4_amx_context::set_scheduler(ggml_backend_sched_t sched) {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(sched);
#else
    if (sched == nullptr) {
        throw std::runtime_error("DeepSeek V4 AMX received a null scheduler");
    }
    pimpl->sched        = sched;
    pimpl->active_graph = nullptr;
    pimpl->active_mode  = LLM_DSV4_AMX_DISABLED;
    pimpl->events.clear();
#endif
}

llm_dsv4_amx_mode llama_dsv4_amx_context::mode_for(llm_graph_type gtype, int64_t n_tokens, bool has_lora) const {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(gtype);
    GGML_UNUSED(n_tokens);
    GGML_UNUSED(has_lora);
    return LLM_DSV4_AMX_DISABLED;
#else
    if (gtype != LLM_GRAPH_TYPE_DEFAULT || n_tokens != DSV4_AMX_TOKENS || has_lora) {
        audit_record("graph_mode", "fallback", "graph_type=%d tokens=%lld lora=%d", static_cast<int>(gtype),
                     static_cast<long long>(n_tokens), has_lora ? 1 : 0);
        return LLM_DSV4_AMX_DISABLED;
    }
    const llm_dsv4_amx_mode mode = pimpl->validate ? LLM_DSV4_AMX_VALIDATE : LLM_DSV4_AMX_COEXEC;
    audit_record("graph_mode", "eligible", "mode=%d tokens=%lld lora=0 pack_ready=%d", static_cast<int>(mode),
                 static_cast<long long>(n_tokens), pimpl->worker ? 1 : 0);
    return mode;
#endif
}

void llama_dsv4_amx_context::prepare_graph(ggml_cgraph * graph, llm_dsv4_amx_mode mode, bool executable) {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(graph);
    GGML_UNUSED(mode);
    GGML_UNUSED(executable);
#else
    if (graph == nullptr) {
        throw std::runtime_error("DeepSeek V4 AMX received a null graph");
    }
    if (pimpl->sched == nullptr) {
        throw std::runtime_error("DeepSeek V4 AMX graph prepared before scheduler initialization");
    }
    if (mode == LLM_DSV4_AMX_DISABLED) {
        if (executable) {
            pimpl->active_graph = nullptr;
            pimpl->active_mode  = mode;
            pimpl->events.clear();
            audit_record("bindings", "fallback", "mode=disabled executable=1");
        }
        return;
    }
    if (executable && !pimpl->worker) {
        try {
            pimpl->model_pack->ensure_packed(*pimpl->model, oracle_requested());
            pimpl->worker = std::make_unique<amx_worker>(pimpl->model_pack->pimpl->layers);
            audit_record("context", "allocated", "worker_scratch_bytes=%zu oracle=%d",
                         llama_dsv4_amx_scratch_bytes(DSV4_AMX_TOKENS), oracle_requested() ? 1 : 0);
        } catch (const std::exception & exception) {
            pimpl->poison("DeepSeek V4 AMX lazy initialization failed: " + std::string(exception.what()));
            throw;
        }
    }

    std::array<impl::binding, DSV4_AMX_LAYERS>     bindings;
    std::unordered_map<ggml_tensor *, impl::event> events;
    std::unordered_set<ggml_tensor *>              output_tensors;
    events.reserve(2 * DSV4_AMX_LAYERS);
    output_tensors.reserve(DSV4_AMX_LAYERS);
    size_t direct_consumer_edges = 0;
    size_t routed_joins          = 0;
    size_t telemetry_consumers   = 0;
    char   name[64];
    for (int il = 0; il < DSV4_AMX_LAYERS; ++il) {
        auto find = [&](const char * base) {
            std::snprintf(name, sizeof(name), "%s-%d", base, il);
            return ggml_graph_get_tensor(graph, name);
        };
        impl::binding & bound    = bindings[static_cast<size_t>(il)];
        bound.norm               = find("ffn_norm");
        bound.output             = find("ffn_amx_out");
        bound.reference          = mode == LLM_DSV4_AMX_VALIDATE ? find("ffn_amx_ref") : nullptr;
        ggml_tensor * moe        = find("ffn_moe_out");
        ggml_tensor * join       = find("ffn_out");
        ggml_tensor * validation = mode == LLM_DSV4_AMX_VALIDATE ? find("ffn_amx_validation") : nullptr;
        ggml_tensor * completion = mode == LLM_DSV4_AMX_VALIDATE ? bound.reference : moe;
        if (bound.norm == nullptr || bound.output == nullptr || moe == nullptr || join == nullptr ||
            completion == nullptr || (mode == LLM_DSV4_AMX_VALIDATE && bound.reference == nullptr)) {
            throw std::runtime_error("DeepSeek V4 AMX graph is missing a required layer binding at layer " +
                                     std::to_string(il));
        }
        if (!has_exact_binary_sources(join, GGML_OP_ADD, moe, bound.output, true)) {
            throw std::runtime_error("DeepSeek V4 AMX graph has an invalid routed/shared join at layer " +
                                     std::to_string(il));
        }
        const auto consumers = direct_consumers(graph, bound.output);
        if (mode == LLM_DSV4_AMX_VALIDATE) {
            if (!has_exact_binary_sources(validation, GGML_OP_SUB, bound.reference, bound.output, false) ||
                consumers.size() != 2 || std::count(consumers.begin(), consumers.end(), join) != 1 ||
                std::count(consumers.begin(), consumers.end(), validation) != 1) {
                throw std::runtime_error("DeepSeek V4 AMX validation graph has unexpected output consumers at layer " +
                                         std::to_string(il));
            }
            ++telemetry_consumers;
        } else if (consumers.size() != 1 || consumers[0] != join) {
            throw std::runtime_error("DeepSeek V4 AMX graph did not join its output exactly once at layer " +
                                     std::to_string(il));
        }
        direct_consumer_edges += consumers.size();
        ++routed_joins;
        pimpl->bind_output_leaf(bound.output, il);
        ggml_backend_sched_set_tensor_backend(pimpl->sched, bound.output, pimpl->metal_backend);
        const bool inserted_start = events.emplace(bound.norm, impl::event{ il, true }).second;
        const bool inserted_end   = events.emplace(completion, impl::event{ il, false }).second;
        output_tensors.emplace(bound.output);
        if (!inserted_start || !inserted_end) {
            throw std::runtime_error("DeepSeek V4 AMX graph contains duplicate callback tensors at layer " +
                                     std::to_string(il));
        }
    }
    const size_t expected_consumer_edges =
        static_cast<size_t>(DSV4_AMX_LAYERS) * (mode == LLM_DSV4_AMX_VALIDATE ? 2 : 1);
    if (events.size() != 2 * DSV4_AMX_LAYERS || output_tensors.size() != DSV4_AMX_LAYERS ||
        routed_joins != DSV4_AMX_LAYERS ||
        telemetry_consumers != (mode == LLM_DSV4_AMX_VALIDATE ? DSV4_AMX_LAYERS : 0) ||
        direct_consumer_edges != expected_consumer_edges ||
        pimpl->output_buffer_bytes != llama_dsv4_amx_output_bytes(DSV4_AMX_TOKENS)) {
        throw std::runtime_error("DeepSeek V4 AMX graph binding cardinality or lifetime contract failed");
    }
    audit_record("bindings", "complete",
                 "executable=%d layers=%d callback_events=%zu tensor_metadata=%zu unique_buffers=1 unique_data=1 "
                 "producerless=1 callback_edges=norm_completion visible_before_consumer=1 routed_join_exact=1 "
                 "routed_joins=%zu telemetry_consumers=%zu direct_consumer_edges=%zu external_output_bytes=%zu "
                 "gallocr_output_bytes=0",
                 executable ? 1 : 0, DSV4_AMX_LAYERS, events.size(), output_tensors.size(), routed_joins,
                 telemetry_consumers, direct_consumer_edges, pimpl->output_buffer_bytes);

    if (executable) {
        pimpl->bindings           = bindings;
        pimpl->events             = std::move(events);
        pimpl->active_graph       = graph;
        pimpl->active_mode        = mode;
        pimpl->submission_started = false;
        audit_record("graph_reuse", "bound", "graph=%p mode=%d prior_submissions=%llu", graph, static_cast<int>(mode),
                     static_cast<unsigned long long>(pimpl->submission_index));
    }
#endif
}

bool llama_dsv4_amx_context::owns_graph(const ggml_cgraph * graph) const {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(graph);
    return false;
#else
    return graph != nullptr && pimpl->active_graph == graph && pimpl->active_mode != LLM_DSV4_AMX_DISABLED;
#endif
}

bool llama_dsv4_amx_context::begin_graph(const ggml_cgraph * graph) {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(graph);
    return false;
#else
    if (!owns_graph(graph) || pimpl->poisoned || pimpl->submission_started || !pimpl->worker ||
        pimpl->worker->active()) {
        pimpl->poison("DeepSeek V4 AMX graph could not begin from a clean callback state");
        return false;
    }
    pimpl->reset_submission_counters();
    ++pimpl->submission_index;
    audit_record("graph_reuse", "submit", "graph=%p mode=%d submission=%llu reused=%d", graph,
                 static_cast<int>(pimpl->active_mode), static_cast<unsigned long long>(pimpl->submission_index),
                 pimpl->submission_index > 1 ? 1 : 0);
    return true;
#endif
}

bool llama_dsv4_amx_context::eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * context = static_cast<llama_dsv4_amx_context *>(user_data);
    if (context == nullptr) {
        return false;
    }
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    GGML_UNUSED(tensor);
    GGML_UNUSED(ask);
    return false;
#else
    return context->pimpl->callback(tensor, ask);
#endif
}

ggml_status llama_dsv4_amx_context::finish_graph(ggml_status scheduler_status) {
#if !defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    return scheduler_status;
#else
    const bool submission_started = pimpl->submission_started;
    if (pimpl->worker->active()) {
        worker_timing ignored_timing;
        std::string   error;
        if (!pimpl->worker->wait(ignored_timing, error)) {
            pimpl->poison("DeepSeek V4 AMX worker failed after scheduler exit: " + error);
        }
    }
    if (scheduler_status != GGML_STATUS_SUCCESS) {
        pimpl->poison("scheduler failed while a DeepSeek V4 AMX graph was active");
    }
    if (submission_started) {
        if (pimpl->timing) {
            LLAMA_LOG_INFO(
                "%s: layers=%d transpose=%.3f ms gate=%.3f ms up=%.3f ms activation=%.3f ms "
                "hidden_transpose=%.3f ms down=%.3f ms route_wait=%.3f ms\n",
                __func__, pimpl->completed_layers, pimpl->graph_transpose_ms, pimpl->graph_worker_timing.gate_ms,
                pimpl->graph_worker_timing.up_ms, pimpl->graph_worker_timing.activation_ms,
                pimpl->graph_worker_timing.hidden_transpose_ms, pimpl->graph_worker_timing.down_ms,
                pimpl->graph_wait_ms);
        } else {
            LLAMA_LOG_DEBUG(
                "%s: layers=%d transpose=%.3f ms gate=%.3f ms up=%.3f ms activation=%.3f ms "
                "hidden_transpose=%.3f ms down=%.3f ms route_wait=%.3f ms\n",
                __func__, pimpl->completed_layers, pimpl->graph_transpose_ms, pimpl->graph_worker_timing.gate_ms,
                pimpl->graph_worker_timing.up_ms, pimpl->graph_worker_timing.activation_ms,
                pimpl->graph_worker_timing.hidden_transpose_ms, pimpl->graph_worker_timing.down_ms,
                pimpl->graph_wait_ms);
        }
        pimpl->submission_started = false;
    }
    const bool exact_callbacks = pimpl->callback_order.complete();
    audit_record("callbacks", exact_callbacks ? "complete" : "fail",
                 "submission=%llu expected_layer=%d expected_edge=%s starts=0x%llx ends=0x%llx completed_layers=%d",
                 static_cast<unsigned long long>(pimpl->submission_index), pimpl->callback_order.expected_layer(),
                 pimpl->callback_order.expected_start() ? "start" : "end",
                 static_cast<unsigned long long>(pimpl->callback_order.starts_seen()),
                 static_cast<unsigned long long>(pimpl->callback_order.ends_seen()), pimpl->completed_layers);
    if (!submission_started || pimpl->poisoned || pimpl->completed_layers != DSV4_AMX_LAYERS || !exact_callbacks) {
        if (!pimpl->poisoned) {
            pimpl->poison(submission_started ? "DeepSeek V4 AMX graph violated exact callback completion" :
                                               "DeepSeek V4 AMX graph completed without beginning callback state");
        }
        LLAMA_LOG_ERROR("%s: context poisoned: %s\n", __func__, pimpl->poison_reason.c_str());
        return GGML_STATUS_FAILED;
    }
    audit_record("graph", "complete", "submission=%llu mode=%d layers=%d",
                 static_cast<unsigned long long>(pimpl->submission_index), static_cast<int>(pimpl->active_mode),
                 pimpl->completed_layers);
    return scheduler_status;
#endif
}

void llama_dsv4_amx_context::shutdown() {
#if defined(LLAMA_DSV4_AMX_APPLE_ARM64)
    if (pimpl && pimpl->worker) {
        pimpl->worker->shutdown();
    }
#endif
}

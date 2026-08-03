#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#if defined(__APPLE__)
#    include <Accelerate/Accelerate.h>
#    include <mach/mach.h>
#else
#    include <cblas.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr int64_t DSV4_EMBD         = 4096;
constexpr int64_t DSV4_FF           = 2048;
constexpr int64_t DSV4_EXPERTS      = 256;
constexpr int64_t DSV4_EXPERTS_USED = 6;
constexpr float   DSV4_CLAMP_LIMIT  = 10.0f;

struct options {
    int64_t     tokens      = 2048;
    int         warmup      = 2;
    int         runs        = 4;
    float       clamp_limit = DSV4_CLAMP_LIMIT;
    std::string mode        = "all";
};

struct output_stats {
    bool   finite   = true;
    double checksum = 0.0;
    double l2       = 0.0;
};

struct error_stats {
    double nmse    = 0.0;
    double max_abs = 0.0;
};

struct overlap_timing {
    double wall_ms  = 0.0;
    double cpu_ms   = 0.0;
    double metal_ms = 0.0;
};

struct raw_run {
    int         iteration = 0;
    std::string order;
    std::string mode;
    double      wall_ms  = 0.0;
    double      cpu_ms   = 0.0;
    double      metal_ms = 0.0;
};

static double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static uint64_t current_rss_bytes() {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
#else
    return 0;
#endif
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static float deterministic_value(uint64_t index, uint32_t seed, float scale) {
    const uint32_t bits     = mix32(static_cast<uint32_t>(index) ^ mix32(static_cast<uint32_t>(index >> 32) + seed));
    const int32_t  centered = static_cast<int32_t>(bits & 0xffffU) - 32768;
    return scale * static_cast<float>(centered) / 32768.0f;
}

static output_stats summarize(const std::vector<float> & values) {
    output_stats result;
    for (size_t i = 0; i < values.size(); ++i) {
        const double value = values[i];
        result.finite      = result.finite && std::isfinite(value);
        result.checksum += value * (1.0 + static_cast<double>(i % 251) / 251.0);
        result.l2 += value * value;
    }
    return result;
}

static error_stats compare_outputs(const std::vector<float> & reference, const std::vector<float> & candidate) {
    GGML_ASSERT(reference.size() == candidate.size());
    double error_sq     = 0.0;
    double reference_sq = 0.0;
    double max_abs      = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            return { std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity() };
        }
        const double error = static_cast<double>(reference[i]) - candidate[i];
        error_sq += error * error;
        reference_sq += static_cast<double>(reference[i]) * reference[i];
        max_abs = std::max(max_abs, std::abs(error));
    }
    return { reference_sq == 0.0 ? error_sq : error_sq / reference_sq, max_abs };
}

static double median(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

static bool env_truthy(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--tokens 128|256|512|1024|2048] [--warmup N] [--runs N] "
                 "[--clamp-limit F] [--mode all|candidate|sequential]\n"
                 "       LLAMA_DSV4_AMX_COEXEC_DISABLE=1 forces the sequential all-Metal control.\n",
                 argv0);
}

static bool parse_options(int argc, char ** argv, options & opts) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char * flag) -> const char * {
            if (++i >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                return nullptr;
            }
            return argv[i];
        };

        if (std::strcmp(argv[i], "--tokens") == 0) {
            const char * value = next("--tokens");
            if (value == nullptr) {
                return false;
            }
            opts.tokens = std::strtoll(value, nullptr, 10);
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            const char * value = next("--warmup");
            if (value == nullptr) {
                return false;
            }
            opts.warmup = std::atoi(value);
        } else if (std::strcmp(argv[i], "--runs") == 0) {
            const char * value = next("--runs");
            if (value == nullptr) {
                return false;
            }
            opts.runs = std::atoi(value);
        } else if (std::strcmp(argv[i], "--clamp-limit") == 0) {
            const char * value = next("--clamp-limit");
            if (value == nullptr) {
                return false;
            }
            opts.clamp_limit = std::strtof(value, nullptr);
        } else if (std::strcmp(argv[i], "--mode") == 0) {
            const char * value = next("--mode");
            if (value == nullptr) {
                return false;
            }
            opts.mode = value;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    const std::vector<int64_t> valid_tokens = { 128, 256, 512, 1024, 2048 };
    const bool tokens_ok = std::find(valid_tokens.begin(), valid_tokens.end(), opts.tokens) != valid_tokens.end();
    const bool mode_ok   = opts.mode == "all" || opts.mode == "candidate" || opts.mode == "sequential";
    return tokens_ok && mode_ok && opts.warmup >= 0 && opts.runs > 0 && std::isfinite(opts.clamp_limit) &&
           opts.clamp_limit >= 0.0f;
}

class cpu_shared_ffn {
  public:
    cpu_shared_ffn(int64_t tokens, float clamp_limit) : tokens_(tokens), clamp_limit_(clamp_limit) {
        const size_t gate_elements = DSV4_EMBD * DSV4_FF;
        const size_t down_elements = DSV4_FF * DSV4_EMBD;

        gate_bf16_.resize(gate_elements);
        up_bf16_.resize(gate_elements);
        down_bf16_.resize(down_elements);
        initialize_bf16(gate_bf16_, 0x4101U);
        initialize_bf16(up_bf16_, 0x4102U);
        initialize_bf16(down_bf16_, 0x4103U);

        input_.resize(tokens_ * DSV4_EMBD);
        gate_.resize(tokens_ * DSV4_FF);
        up_.resize(tokens_ * DSV4_FF);
        hidden_.resize(tokens_ * DSV4_FF);
        output_.resize(tokens_ * DSV4_EMBD);
        for (size_t i = 0; i < input_.size(); ++i) {
            input_[i] = deterministic_value(i, 0x5111U, 0.125f);
        }

        const uint64_t rss_before = current_rss_bytes();
        const auto     begin      = clock_type::now();
        gate_f32_.resize(gate_elements);
        up_f32_.resize(gate_elements);
        down_f32_.resize(down_elements);
        ggml_bf16_to_fp32_row(gate_bf16_.data(), gate_f32_.data(), gate_elements);
        ggml_bf16_to_fp32_row(up_bf16_.data(), up_f32_.data(), gate_elements);
        ggml_bf16_to_fp32_row(down_bf16_.data(), down_f32_.data(), down_elements);
        pack_ms_                 = elapsed_ms(begin, clock_type::now());
        const uint64_t rss_after = current_rss_bytes();
        pack_rss_delta_          = rss_after >= rss_before ? rss_after - rss_before : 0;
    }

    void run() {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(tokens_), static_cast<int>(DSV4_FF),
                    static_cast<int>(DSV4_EMBD), 1.0f, input_.data(), static_cast<int>(DSV4_EMBD), gate_f32_.data(),
                    static_cast<int>(DSV4_EMBD), 0.0f, gate_.data(), static_cast<int>(DSV4_FF));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(tokens_), static_cast<int>(DSV4_FF),
                    static_cast<int>(DSV4_EMBD), 1.0f, input_.data(), static_cast<int>(DSV4_EMBD), up_f32_.data(),
                    static_cast<int>(DSV4_EMBD), 0.0f, up_.data(), static_cast<int>(DSV4_FF));

        for (size_t i = 0; i < hidden_.size(); ++i) {
            if (clamp_limit_ > 1e-6f) {
                gate_[i] = std::min(gate_[i], clamp_limit_);
                up_[i]   = std::clamp(up_[i], -clamp_limit_, clamp_limit_);
            }
            hidden_[i] = (gate_[i] / (1.0f + std::exp(-gate_[i]))) * up_[i];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(tokens_), static_cast<int>(DSV4_EMBD),
                    static_cast<int>(DSV4_FF), 1.0f, hidden_.data(), static_cast<int>(DSV4_FF), down_f32_.data(),
                    static_cast<int>(DSV4_FF), 0.0f, output_.data(), static_cast<int>(DSV4_EMBD));
    }

    double time_run_ms() {
        const auto begin = clock_type::now();
        run();
        return elapsed_ms(begin, clock_type::now());
    }

    const std::vector<ggml_bf16_t> & gate_bf16() const { return gate_bf16_; }

    const std::vector<ggml_bf16_t> & up_bf16() const { return up_bf16_; }

    const std::vector<ggml_bf16_t> & down_bf16() const { return down_bf16_; }

    const std::vector<float> & input() const { return input_; }

    const std::vector<float> & gate() const { return gate_; }

    const std::vector<float> & up() const { return up_; }

    const std::vector<float> & hidden() const { return hidden_; }

    const std::vector<float> & output() const { return output_; }

    double pack_ms() const { return pack_ms_; }

    uint64_t pack_rss_delta() const { return pack_rss_delta_; }

    size_t bf16_weight_bytes() const {
        return (gate_bf16_.size() + up_bf16_.size() + down_bf16_.size()) * sizeof(ggml_bf16_t);
    }

    size_t f32_pack_bytes() const { return (gate_f32_.size() + up_f32_.size() + down_f32_.size()) * sizeof(float); }

  private:
    static void initialize_bf16(std::vector<ggml_bf16_t> & values, uint32_t seed) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = ggml_fp32_to_bf16(deterministic_value(i, seed, 0.015625f));
        }
    }

    int64_t                  tokens_;
    float                    clamp_limit_;
    std::vector<ggml_bf16_t> gate_bf16_;
    std::vector<ggml_bf16_t> up_bf16_;
    std::vector<ggml_bf16_t> down_bf16_;
    std::vector<float>       gate_f32_;
    std::vector<float>       up_f32_;
    std::vector<float>       down_f32_;
    std::vector<float>       input_;
    std::vector<float>       gate_;
    std::vector<float>       up_;
    std::vector<float>       hidden_;
    std::vector<float>       output_;
    double                   pack_ms_        = 0.0;
    uint64_t                 pack_rss_delta_ = 0;
};

class metal_graphs {
  public:
    metal_graphs(ggml_backend_t backend, const cpu_shared_ffn & cpu, int64_t tokens, float clamp_limit) :
        backend_(backend),
        tokens_(tokens) {
        constexpr size_t       graph_nodes   = 256;
        const ggml_init_params weight_params = {
            ggml_tensor_overhead() * 8,
            nullptr,
            true,
        };
        const ggml_init_params compute_params = {
            ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(graph_nodes, false) * 3,
            nullptr,
            true,
        };
        weights_.reset(ggml_init(weight_params));
        compute_.reset(ggml_init(compute_params));
        if (!weights_ || !compute_) {
            throw std::runtime_error("failed to create ggml contexts");
        }

        route_gate_  = ggml_new_tensor_3d(weights_.get(), GGML_TYPE_MXFP4, DSV4_EMBD, DSV4_FF, DSV4_EXPERTS);
        route_up_    = ggml_new_tensor_3d(weights_.get(), GGML_TYPE_MXFP4, DSV4_EMBD, DSV4_FF, DSV4_EXPERTS);
        route_down_  = ggml_new_tensor_3d(weights_.get(), GGML_TYPE_MXFP4, DSV4_FF, DSV4_EMBD, DSV4_EXPERTS);
        shared_gate_ = ggml_new_tensor_2d(weights_.get(), GGML_TYPE_BF16, DSV4_EMBD, DSV4_FF);
        shared_up_   = ggml_new_tensor_2d(weights_.get(), GGML_TYPE_BF16, DSV4_EMBD, DSV4_FF);
        shared_down_ = ggml_new_tensor_2d(weights_.get(), GGML_TYPE_BF16, DSV4_FF, DSV4_EMBD);

        route_input_   = ggml_new_tensor_3d(compute_.get(), GGML_TYPE_F32, DSV4_EMBD, 1, tokens_);
        ids_base_      = ggml_new_tensor_2d(compute_.get(), GGML_TYPE_I32, DSV4_EXPERTS, tokens_);
        ids_           = ggml_view_2d(compute_.get(), ids_base_, DSV4_EXPERTS_USED, tokens_, ids_base_->nb[1], 0);
        route_weights_ = ggml_new_tensor_3d(compute_.get(), GGML_TYPE_F32, 1, DSV4_EXPERTS_USED, tokens_);
        shared_input_  = ggml_new_tensor_2d(compute_.get(), GGML_TYPE_F32, DSV4_EMBD, tokens_);

        routed_up_out_   = ggml_mul_mat_id(compute_.get(), route_up_, route_input_, ids_);
        routed_gate_out_ = ggml_mul_mat_id(compute_.get(), route_gate_, route_input_, ids_);
        if (clamp_limit > 1e-6f) {
            routed_up_out_   = ggml_clamp(compute_.get(), routed_up_out_, -clamp_limit, clamp_limit);
            routed_gate_out_ = ggml_clamp(compute_.get(), routed_gate_out_, -INFINITY, clamp_limit);
        }
        routed_hidden_out_   = ggml_swiglu_split(compute_.get(), routed_gate_out_, routed_up_out_);
        routed_down_out_     = ggml_mul_mat_id(compute_.get(), route_down_, routed_hidden_out_, ids_);
        ggml_tensor * routed = ggml_mul(compute_.get(), routed_down_out_, route_weights_);
        ggml_tensor * routed_lanes[DSV4_EXPERTS_USED];
        for (int64_t i = 0; i < DSV4_EXPERTS_USED; ++i) {
            routed_lanes[i] =
                ggml_view_2d(compute_.get(), routed, DSV4_EMBD, tokens_, routed->nb[2], i * routed->nb[1]);
        }
        routed_out_ = routed_lanes[0];
        for (int64_t i = 1; i < DSV4_EXPERTS_USED; ++i) {
            routed_out_ = ggml_add(compute_.get(), routed_out_, routed_lanes[i]);
        }

        shared_gate_out_ = ggml_mul_mat(compute_.get(), shared_gate_, shared_input_);
        shared_up_out_   = ggml_mul_mat(compute_.get(), shared_up_, shared_input_);
        if (clamp_limit > 1e-6f) {
            shared_up_out_   = ggml_clamp(compute_.get(), shared_up_out_, -clamp_limit, clamp_limit);
            shared_gate_out_ = ggml_clamp(compute_.get(), shared_gate_out_, -INFINITY, clamp_limit);
        }
        shared_hidden_out_ = ggml_swiglu_split(compute_.get(), shared_gate_out_, shared_up_out_);
        shared_out_        = ggml_mul_mat(compute_.get(), shared_down_, shared_hidden_out_);
        all_out_           = ggml_add(compute_.get(), routed_out_, shared_out_);

        route_graph_  = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        shared_graph_ = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        all_graph_    = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        ggml_build_forward_expand(route_graph_, routed_out_);
        ggml_build_forward_expand(shared_graph_, shared_out_);
        ggml_build_forward_expand(all_graph_, all_out_);

        weights_buffer_.reset(ggml_backend_alloc_ctx_tensors(weights_.get(), backend_));
        compute_buffer_.reset(ggml_backend_alloc_ctx_tensors(compute_.get(), backend_));
        if (!weights_buffer_ || !compute_buffer_) {
            throw std::runtime_error("failed to allocate Metal probe buffers");
        }
        ggml_backend_buffer_set_usage(weights_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        initialize(cpu);
        validate_graph_support(route_graph_, "routed");
        validate_graph_support(shared_graph_, "shared");
        validate_graph_support(all_graph_, "all-metal");
    }

    double time_route_ms() { return time_graph(route_graph_); }

    double time_shared_ms() { return time_graph(shared_graph_); }

    double time_all_ms() { return time_graph(all_graph_); }

    void launch_route_async() {
        const ggml_status status = ggml_backend_graph_compute_async(backend_, route_graph_);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Metal routed graph submission failed");
        }
    }

    void synchronize() { ggml_backend_synchronize(backend_); }

    std::vector<float> get_shared_output() const { return get_tensor(shared_out_); }

    std::vector<float> get_shared_gate() const { return get_tensor(shared_gate_out_); }

    std::vector<float> get_shared_up() const { return get_tensor(shared_up_out_); }

    std::vector<float> get_shared_hidden() const { return get_tensor(shared_hidden_out_); }

    std::vector<float> get_routed_output() const { return get_tensor(routed_out_); }

    std::vector<float> get_routed_gate() const { return get_tensor(routed_gate_out_); }

    std::vector<float> get_routed_up() const { return get_tensor(routed_up_out_); }

    std::vector<float> get_routed_hidden() const { return get_tensor(routed_hidden_out_); }

    std::vector<float> get_routed_down() const { return get_tensor(routed_down_out_); }

    std::vector<float> get_all_output() const { return get_tensor(all_out_); }

    size_t routed_weight_bytes() const {
        return ggml_nbytes(route_gate_) + ggml_nbytes(route_up_) + ggml_nbytes(route_down_);
    }

  private:
    static std::vector<float> get_tensor(const ggml_tensor * tensor) {
        std::vector<float> result(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
        return result;
    }

    static void initialize_mxfp4(ggml_tensor * tensor, uint32_t seed) {
        constexpr int64_t block_elements = 32;
        float             source[block_elements];
        for (int64_t i = 0; i < block_elements; ++i) {
            source[i] = deterministic_value(i, seed, 0.0625f);
        }

        const size_t         block_bytes = ggml_row_size(GGML_TYPE_MXFP4, block_elements);
        std::vector<uint8_t> packed_block(block_bytes);
        const size_t         written =
            ggml_quantize_chunk(GGML_TYPE_MXFP4, source, packed_block.data(), 0, 1, block_elements, nullptr);
        if (written != block_bytes) {
            throw std::runtime_error("failed to create deterministic MXFP4 block");
        }

        constexpr size_t     blocks_per_chunk = 1U << 20;
        std::vector<uint8_t> chunk(block_bytes * blocks_per_chunk);
        std::memcpy(chunk.data(), packed_block.data(), block_bytes);
        for (size_t initialized = block_bytes; initialized < chunk.size();) {
            const size_t copy = std::min(initialized, chunk.size() - initialized);
            std::memcpy(chunk.data() + initialized, chunk.data(), copy);
            initialized += copy;
        }

        const size_t tensor_bytes = ggml_nbytes(tensor);
        for (size_t offset = 0; offset < tensor_bytes; offset += chunk.size()) {
            ggml_backend_tensor_set(tensor, chunk.data(), offset, std::min(chunk.size(), tensor_bytes - offset));
        }
    }

    void initialize(const cpu_shared_ffn & cpu) {
        initialize_mxfp4(route_gate_, 0x7101U);
        initialize_mxfp4(route_up_, 0x7102U);
        initialize_mxfp4(route_down_, 0x7103U);
        ggml_backend_tensor_set(shared_gate_, cpu.gate_bf16().data(), 0, ggml_nbytes(shared_gate_));
        ggml_backend_tensor_set(shared_up_, cpu.up_bf16().data(), 0, ggml_nbytes(shared_up_));
        ggml_backend_tensor_set(shared_down_, cpu.down_bf16().data(), 0, ggml_nbytes(shared_down_));

        ggml_backend_tensor_set(route_input_, cpu.input().data(), 0, ggml_nbytes(route_input_));

        std::vector<int32_t> ids(ggml_nelements(ids_base_), 0);
        for (int64_t token = 0; token < tokens_; ++token) {
            for (int64_t lane = 0; lane < DSV4_EXPERTS_USED; ++lane) {
                ids[token * DSV4_EXPERTS + lane] = static_cast<int32_t>((token * 37 + lane * 43) % DSV4_EXPERTS);
            }
        }
        ggml_backend_tensor_set(ids_base_, ids.data(), 0, ggml_nbytes(ids_base_));

        std::vector<float> route_weights(ggml_nelements(route_weights_), 1.0f / DSV4_EXPERTS_USED);
        ggml_backend_tensor_set(route_weights_, route_weights.data(), 0, ggml_nbytes(route_weights_));
        ggml_backend_tensor_set(shared_input_, cpu.input().data(), 0, ggml_nbytes(shared_input_));
        ggml_backend_synchronize(backend_);
    }

    void validate_graph_support(ggml_cgraph * graph, const char * name) {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend_, node)) {
                throw std::runtime_error(std::string("Metal does not support ") + name + " node " + ggml_op_desc(node));
            }
        }
    }

    double time_graph(ggml_cgraph * graph) {
        const auto        begin  = clock_type::now();
        const ggml_status status = ggml_backend_graph_compute(backend_, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Metal graph execution failed");
        }
        return elapsed_ms(begin, clock_type::now());
    }

    ggml_backend_t          backend_;
    int64_t                 tokens_;
    ggml_context_ptr        weights_;
    ggml_context_ptr        compute_;
    ggml_backend_buffer_ptr weights_buffer_;
    ggml_backend_buffer_ptr compute_buffer_;
    ggml_tensor *           route_gate_        = nullptr;
    ggml_tensor *           route_up_          = nullptr;
    ggml_tensor *           route_down_        = nullptr;
    ggml_tensor *           shared_gate_       = nullptr;
    ggml_tensor *           shared_up_         = nullptr;
    ggml_tensor *           shared_down_       = nullptr;
    ggml_tensor *           route_input_       = nullptr;
    ggml_tensor *           ids_base_          = nullptr;
    ggml_tensor *           ids_               = nullptr;
    ggml_tensor *           route_weights_     = nullptr;
    ggml_tensor *           shared_input_      = nullptr;
    ggml_tensor *           routed_gate_out_   = nullptr;
    ggml_tensor *           routed_up_out_     = nullptr;
    ggml_tensor *           routed_hidden_out_ = nullptr;
    ggml_tensor *           routed_down_out_   = nullptr;
    ggml_tensor *           routed_out_        = nullptr;
    ggml_tensor *           shared_gate_out_   = nullptr;
    ggml_tensor *           shared_up_out_     = nullptr;
    ggml_tensor *           shared_hidden_out_ = nullptr;
    ggml_tensor *           shared_out_        = nullptr;
    ggml_tensor *           all_out_           = nullptr;
    ggml_cgraph *           route_graph_       = nullptr;
    ggml_cgraph *           shared_graph_      = nullptr;
    ggml_cgraph *           all_graph_         = nullptr;
};

static overlap_timing run_overlap(cpu_shared_ffn & cpu, metal_graphs & metal) {
    std::atomic<bool>      ready(false);
    std::atomic<bool>      go(false);
    clock_type::time_point cpu_end;

    auto cpu_future = std::async(std::launch::async, [&]() {
        ready.store(true, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        cpu.run();
        cpu_end = clock_type::now();
    });
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto begin = clock_type::now();
    go.store(true, std::memory_order_release);
    metal.launch_route_async();
    metal.synchronize();
    const auto metal_end = clock_type::now();
    cpu_future.get();
    const auto end = std::max(cpu_end, metal_end);
    return { elapsed_ms(begin, end), elapsed_ms(begin, cpu_end), elapsed_ms(begin, metal_end) };
}

static double run_sequential(cpu_shared_ffn & cpu, metal_graphs & metal, bool cpu_first) {
    const auto begin = clock_type::now();
    if (cpu_first) {
        cpu.run();
        metal.time_route_ms();
    } else {
        metal.time_route_ms();
        cpu.run();
    }
    return elapsed_ms(begin, clock_type::now());
}

static void print_raw(const raw_run & run) {
    std::printf("raw\titeration=%d\torder=%s\tmode=%s\twall_ms=%.6f\tcpu_ms=%.6f\tmetal_ms=%.6f\n", run.iteration,
                run.order.c_str(), run.mode.c_str(), run.wall_ms, run.cpu_ms, run.metal_ms);
}

}  // namespace

int main(int argc, char ** argv) try {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        usage(argv[0]);
        return 2;
    }

    const bool disabled = env_truthy("LLAMA_DSV4_AMX_COEXEC_DISABLE");
    if (disabled) {
        opts.mode = "sequential";
    }

    ggml_backend_load_all();
    ggml_backend_dev_t metal_device   = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_reg_t metal_registry = metal_device == nullptr ? nullptr : ggml_backend_dev_backend_reg(metal_device);
    if (metal_registry == nullptr || std::strcmp(ggml_backend_reg_name(metal_registry), "MTL") != 0) {
        std::fprintf(stderr, "llama-dsv4-amx-probe requires the Metal backend\n");
        return 2;
    }
    ggml_backend_ptr metal_backend(ggml_backend_dev_init(metal_device, nullptr));
    if (!metal_backend) {
        std::fprintf(stderr, "failed to initialize Metal backend\n");
        return 2;
    }

    cpu_shared_ffn cpu(opts.tokens, opts.clamp_limit);
    metal_graphs   metal(metal_backend.get(), cpu, opts.tokens, opts.clamp_limit);

    std::printf(
        "probe\tcommit=%s\tdevice=%s\ttokens=%lld\twarmup=%d\truns=%d\tclamp_limit=%.9g\tmode=%s\tdisabled=%d\n",
        ggml_commit(), ggml_backend_dev_description(metal_device), static_cast<long long>(opts.tokens), opts.warmup,
        opts.runs, opts.clamp_limit, opts.mode.c_str(), disabled ? 1 : 0);
    std::printf(
        "shape\tcpu_shared=BF16_source_to_F32_pack:weights[4096,2048]+[4096,2048]+[2048,4096],input[4096,%lld]"
        "\tmetal_shared=BF16:weights[4096,2048]+[4096,2048]+[2048,4096],input[4096,%lld]"
        "\tmetal_routed=MXFP4:weights[4096,2048,256]+[4096,2048,256]+[2048,4096,256],"
        "input[4096,1,%lld],ids[6,%lld]\trouted_scope=complete_ffn\n",
        static_cast<long long>(opts.tokens), static_cast<long long>(opts.tokens), static_cast<long long>(opts.tokens),
        static_cast<long long>(opts.tokens));
    std::printf(
        "cpu_path\tprovider=Accelerate\tapi=cblas_sgemm\tsteady_state_calls=3"
        "\thardware_accelerator=unverified\tverification=profile_Apple_AMX_counters_or_disassembly\n");
    std::printf(
        "pack\tone_time_ms=%.6f\tbf16_source_bytes=%zu\tf32_pack_bytes=%zu\trss_delta_bytes=%llu"
        "\tsteady_state_excludes_pack=1\n",
        cpu.pack_ms(), cpu.bf16_weight_bytes(), cpu.f32_pack_bytes(),
        static_cast<unsigned long long>(cpu.pack_rss_delta()));
    std::printf("memory\tmetal_routed_weight_bytes=%zu\tmxfp4_init=deterministic_repeated_nonzero_blocks\n",
                metal.routed_weight_bytes());

    for (int i = 0; i < opts.warmup; ++i) {
        cpu.run();
        metal.time_route_ms();
        metal.time_shared_ms();
        metal.time_all_ms();
        if (opts.mode != "sequential") {
            run_overlap(cpu, metal);
        }
    }

    cpu.run();
    metal.time_all_ms();
    const std::vector<float> metal_gate   = metal.get_shared_gate();
    const std::vector<float> metal_up     = metal.get_shared_up();
    const std::vector<float> metal_hidden = metal.get_shared_hidden();
    const std::vector<float> metal_shared = metal.get_shared_output();
    const std::vector<float> metal_routed = metal.get_routed_output();
    const std::vector<float> metal_all    = metal.get_all_output();
    std::vector<float>       expected_all(metal_all.size());
    for (size_t i = 0; i < expected_all.size(); ++i) {
        expected_all[i] = metal_shared[i] + metal_routed[i];
    }
    const output_stats cpu_stats           = summarize(cpu.output());
    const output_stats shared_stats        = summarize(metal_shared);
    const output_stats routed_stats        = summarize(metal_routed);
    const output_stats all_stats           = summarize(metal_all);
    const output_stats routed_gate_stats   = summarize(metal.get_routed_gate());
    const output_stats routed_up_stats     = summarize(metal.get_routed_up());
    const output_stats routed_hidden_stats = summarize(metal.get_routed_hidden());
    const output_stats routed_down_stats   = summarize(metal.get_routed_down());
    const error_stats  gate_error          = compare_outputs(cpu.gate(), metal_gate);
    const error_stats  up_error            = compare_outputs(cpu.up(), metal_up);
    const error_stats  hidden_error        = compare_outputs(cpu.hidden(), metal_hidden);
    const error_stats  shared_error        = compare_outputs(cpu.output(), metal_shared);
    const error_stats  all_error           = compare_outputs(expected_all, metal_all);
    std::printf(
        "correctness\tcpu_finite=%d\tmetal_shared_finite=%d\tmetal_routed_finite=%d\tmetal_all_finite=%d"
        "\tcpu_checksum=%.9e\tmetal_shared_checksum=%.9e\tmetal_routed_checksum=%.9e\tmetal_all_checksum=%.9e"
        "\tcpu_l2=%.9e\tmetal_shared_l2=%.9e\tmetal_routed_l2=%.9e\tmetal_all_l2=%.9e"
        "\trouted_gate_finite=%d\trouted_gate_checksum=%.9e\trouted_gate_l2=%.9e"
        "\trouted_up_finite=%d\trouted_up_checksum=%.9e\trouted_up_l2=%.9e"
        "\trouted_hidden_finite=%d\trouted_hidden_checksum=%.9e\trouted_hidden_l2=%.9e"
        "\trouted_down_finite=%d\trouted_down_checksum=%.9e\trouted_down_l2=%.9e"
        "\tgate_nmse=%.9e\tgate_max_abs=%.9e\tup_nmse=%.9e\tup_max_abs=%.9e"
        "\thidden_nmse=%.9e\thidden_max_abs=%.9e\tshared_nmse=%.9e\tshared_max_abs=%.9e"
        "\tall_add_nmse=%.9e\tall_add_max_abs=%.9e\n",
        cpu_stats.finite ? 1 : 0, shared_stats.finite ? 1 : 0, routed_stats.finite ? 1 : 0, all_stats.finite ? 1 : 0,
        cpu_stats.checksum, shared_stats.checksum, routed_stats.checksum, all_stats.checksum, cpu_stats.l2,
        shared_stats.l2, routed_stats.l2, all_stats.l2, routed_gate_stats.finite ? 1 : 0, routed_gate_stats.checksum,
        routed_gate_stats.l2, routed_up_stats.finite ? 1 : 0, routed_up_stats.checksum, routed_up_stats.l2,
        routed_hidden_stats.finite ? 1 : 0, routed_hidden_stats.checksum, routed_hidden_stats.l2,
        routed_down_stats.finite ? 1 : 0, routed_down_stats.checksum, routed_down_stats.l2, gate_error.nmse,
        gate_error.max_abs, up_error.nmse, up_error.max_abs, hidden_error.nmse, hidden_error.max_abs, shared_error.nmse,
        shared_error.max_abs, all_error.nmse, all_error.max_abs);
    const bool routed_valid = routed_stats.finite && routed_stats.l2 > 0.0 && routed_gate_stats.finite &&
                              routed_gate_stats.l2 > 0.0 && routed_up_stats.finite && routed_up_stats.l2 > 0.0 &&
                              routed_hidden_stats.finite && routed_hidden_stats.l2 > 0.0 && routed_down_stats.finite &&
                              routed_down_stats.l2 > 0.0;
    if (!cpu_stats.finite || !shared_stats.finite || !all_stats.finite || !routed_valid || gate_error.nmse > 1e-3 ||
        up_error.nmse > 1e-3 || hidden_error.nmse > 1e-3 || shared_error.nmse > 1e-3 || all_error.nmse > 1e-12) {
        std::fprintf(stderr, "correctness gate failed\n");
        return 1;
    }

    std::vector<raw_run> raw;
    for (int iteration = 0; iteration < opts.runs; ++iteration) {
        // Four adjacent repetitions use the A/B/B/A order. A runs the solo
        // and control cases before the overlap candidate; B reverses it.
        const bool        forward = iteration % 4 == 0 || iteration % 4 == 3;
        const std::string order   = forward ? "A" : "B";
        if (opts.mode == "sequential") {
            const double all_ms = metal.time_all_ms();
            raw.push_back({ iteration, order, "metal_all_control", all_ms, 0.0, all_ms });
            print_raw(raw.back());
            continue;
        }
        if (opts.mode == "candidate") {
            const overlap_timing timing = run_overlap(cpu, metal);
            raw.push_back({ iteration, order, "cpu_metal_overlap", timing.wall_ms, timing.cpu_ms, timing.metal_ms });
            print_raw(raw.back());
            continue;
        }

        auto run_measurement = [&](const std::string & name) {
            raw_run result = { iteration, order, name, 0.0, 0.0, 0.0 };
            if (name == "cpu_shared_solo") {
                result.wall_ms = result.cpu_ms = cpu.time_run_ms();
            } else if (name == "metal_routed_solo") {
                result.wall_ms = result.metal_ms = metal.time_route_ms();
            } else if (name == "metal_shared_solo") {
                result.wall_ms = result.metal_ms = metal.time_shared_ms();
            } else if (name == "metal_all_control") {
                result.wall_ms = result.metal_ms = metal.time_all_ms();
            } else if (name == "cpu_metal_sequential") {
                result.wall_ms = run_sequential(cpu, metal, forward);
            } else if (name == "cpu_metal_overlap") {
                const overlap_timing timing = run_overlap(cpu, metal);
                result.wall_ms              = timing.wall_ms;
                result.cpu_ms               = timing.cpu_ms;
                result.metal_ms             = timing.metal_ms;
            } else {
                GGML_ABORT("unknown DSV4 AMX probe measurement");
            }
            raw.push_back(result);
            print_raw(raw.back());
        };

        const std::vector<std::string> sequence =
            forward ? std::vector<std::string>{ "cpu_shared_solo",   "metal_routed_solo",    "metal_shared_solo",
                                                "metal_all_control", "cpu_metal_sequential", "cpu_metal_overlap" } :
                      std::vector<std::string>{ "cpu_metal_overlap", "cpu_metal_sequential", "metal_all_control",
                                                "metal_shared_solo", "metal_routed_solo",    "cpu_shared_solo" };
        for (const std::string & name : sequence) {
            run_measurement(name);
        }
    }

    if (opts.mode == "all") {
        auto values = [&](const char * mode, auto member) {
            std::vector<double> result;
            for (const raw_run & run : raw) {
                if (run.mode == mode) {
                    result.push_back(run.*member);
                }
            }
            return result;
        };
        const double cpu_solo      = median(values("cpu_shared_solo", &raw_run::wall_ms));
        const double route_solo    = median(values("metal_routed_solo", &raw_run::wall_ms));
        const double shared_solo   = median(values("metal_shared_solo", &raw_run::wall_ms));
        const double metal_all     = median(values("metal_all_control", &raw_run::wall_ms));
        const double sequential    = median(values("cpu_metal_sequential", &raw_run::wall_ms));
        const double overlap       = median(values("cpu_metal_overlap", &raw_run::wall_ms));
        const double overlap_cpu   = median(values("cpu_metal_overlap", &raw_run::cpu_ms));
        const double overlap_metal = median(values("cpu_metal_overlap", &raw_run::metal_ms));
        const double cpu_flops     = 6.0 * DSV4_EMBD * DSV4_FF * opts.tokens;
        const double route_flops   = 6.0 * DSV4_EMBD * DSV4_FF * DSV4_EXPERTS_USED * opts.tokens;
        std::printf(
            "summary\tcpu_shared_solo_ms=%.6f\tmetal_shared_solo_ms=%.6f"
            "\tmetal_routed_solo_ms=%.6f\tsolo_cpu_plus_routed_ms=%.6f"
            "\tcpu_metal_sequential_wall_ms=%.6f\tmetal_all_control_ms=%.6f"
            "\toverlap_wall_ms=%.6f\toverlap_cpu_ms=%.6f\toverlap_metal_ms=%.6f"
            "\tcpu_slowdown=%.6f\tmetal_slowdown=%.6f\toverlap_efficiency=%.6f"
            "\tcpu_shared_tflops=%.6f\tmetal_routed_tflops=%.6f"
            "\tbreak_even_vs_metal_all=%d\n",
            cpu_solo, shared_solo, route_solo, cpu_solo + route_solo, sequential, metal_all, overlap, overlap_cpu,
            overlap_metal, overlap_cpu / cpu_solo, overlap_metal / route_solo, (cpu_solo + route_solo) / overlap,
            cpu_flops / (cpu_solo * 1.0e9), route_flops / (route_solo * 1.0e9), overlap < metal_all ? 1 : 0);
    }

    ggml_quantize_free();
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "llama-dsv4-amx-probe: %s\n", error.what());
    return 1;
}

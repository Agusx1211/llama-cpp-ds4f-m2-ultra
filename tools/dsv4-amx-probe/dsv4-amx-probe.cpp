// Measurement-only Apple M2 Ultra direct-AMX/Metal overlap probe. The AMX
// kernel was implemented independently from the public algorithm description
// in arXiv:2606.25426 because the inspected inferc artifact has no declared
// software license.

#include "corsix-amx-aarch64.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

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
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr int64_t DSV4_EMBD          = 4096;
constexpr int64_t DSV4_FF            = 2048;
constexpr int64_t DSV4_EXPERTS       = 256;
constexpr int64_t DSV4_EXPERTS_USED  = 6;
constexpr int64_t DSV4_LAYERS        = 43;
constexpr float   DSV4_CLAMP_LIMIT   = 10.0f;
constexpr size_t  AMX_PAIR_ALIGNMENT = 128;

struct options {
    int64_t     tokens      = 2048;
    int64_t     panel_cols  = 128;
    int64_t     k_block     = 1024;
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
    bool     finite   = true;
    uint64_t bit_diff = 0;
    double   nmse     = 0.0;
    double   max_abs  = 0.0;
};

struct shared_timing {
    double transpose_input_ms  = 0.0;
    double gate_ms             = 0.0;
    double up_ms               = 0.0;
    double activation_ms       = 0.0;
    double transpose_hidden_ms = 0.0;
    double down_ms             = 0.0;
    double total_ms            = 0.0;
};

struct overlap_timing {
    double        wall_ms              = 0.0;
    double        direct_completion_ms = 0.0;
    double        metal_ms             = 0.0;
    shared_timing direct;
};

struct raw_run {
    int           iteration = 0;
    std::string   order;
    std::string   mode;
    double        wall_ms              = 0.0;
    double        direct_completion_ms = 0.0;
    double        metal_ms             = 0.0;
    shared_timing direct;
};

static double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static uint64_t current_rss_bytes() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0;
    }
    return info.resident_size;
}

static std::string cpu_brand() {
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
    error_stats result;
    double      error_sq     = 0.0;
    double      reference_sq = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(candidate[i])) {
            result.finite  = false;
            result.nmse    = std::numeric_limits<double>::infinity();
            result.max_abs = std::numeric_limits<double>::infinity();
            return result;
        }
        uint32_t reference_bits;
        uint32_t candidate_bits;
        std::memcpy(&reference_bits, &reference[i], sizeof(reference_bits));
        std::memcpy(&candidate_bits, &candidate[i], sizeof(candidate_bits));
        result.bit_diff += reference_bits != candidate_bits;
        const double error = static_cast<double>(reference[i]) - candidate[i];
        error_sq += error * error;
        reference_sq += static_cast<double>(reference[i]) * reference[i];
        result.max_abs = std::max(result.max_abs, std::abs(error));
    }
    result.nmse = reference_sq == 0.0 ? error_sq : error_sq / reference_sq;
    return result;
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
                 "usage: %s [--tokens 128|224|256|512|1024|2048] [--panel-cols N] [--k-block N] "
                 "[--warmup N] [--runs N (multiple of 4)] [--clamp-limit F] [--mode all|candidate|sequential]\n"
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
        } else if (std::strcmp(argv[i], "--panel-cols") == 0) {
            const char * value = next("--panel-cols");
            if (value == nullptr) {
                return false;
            }
            opts.panel_cols = std::strtoll(value, nullptr, 10);
        } else if (std::strcmp(argv[i], "--k-block") == 0) {
            const char * value = next("--k-block");
            if (value == nullptr) {
                return false;
            }
            opts.k_block = std::strtoll(value, nullptr, 10);
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

    const std::vector<int64_t> valid_tokens = { 128, 224, 256, 512, 1024, 2048 };
    const bool tokens_ok = std::find(valid_tokens.begin(), valid_tokens.end(), opts.tokens) != valid_tokens.end();
    const bool mode_ok   = opts.mode == "all" || opts.mode == "candidate" || opts.mode == "sequential";
    return tokens_ok && mode_ok && opts.panel_cols >= 64 && opts.panel_cols % 64 == 0 &&
           DSV4_EMBD % opts.panel_cols == 0 && DSV4_FF % opts.panel_cols == 0 && opts.k_block > 0 &&
           opts.k_block % 16 == 0 && opts.warmup >= 0 && opts.runs > 0 && opts.runs % 4 == 0 &&
           std::isfinite(opts.clamp_limit) && opts.clamp_limit >= 0.0f;
}

class probe_data {
  public:
    explicit probe_data(int64_t tokens) {
        const size_t gate_elements = DSV4_EMBD * DSV4_FF;
        const size_t down_elements = DSV4_FF * DSV4_EMBD;

        gate_bf16_.resize(gate_elements);
        up_bf16_.resize(gate_elements);
        down_bf16_.resize(down_elements);
        initialize_bf16(gate_bf16_, 0x4101U);
        initialize_bf16(up_bf16_, 0x4102U);
        initialize_bf16(down_bf16_, 0x4103U);

        input_.resize(static_cast<size_t>(tokens * DSV4_EMBD));
        for (size_t i = 0; i < input_.size(); ++i) {
            input_[i] = deterministic_value(i, 0x5111U, 0.125f);
        }
    }

    const std::vector<ggml_bf16_t> & gate_bf16() const { return gate_bf16_; }

    const std::vector<ggml_bf16_t> & up_bf16() const { return up_bf16_; }

    const std::vector<ggml_bf16_t> & down_bf16() const { return down_bf16_; }

    const std::vector<float> & input() const { return input_; }

    size_t bf16_weight_bytes() const {
        return (gate_bf16_.size() + up_bf16_.size() + down_bf16_.size()) * sizeof(ggml_bf16_t);
    }

  private:
    static void initialize_bf16(std::vector<ggml_bf16_t> & values, uint32_t seed) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = ggml_fp32_to_bf16(deterministic_value(i, seed, 0.015625f));
        }
    }

    std::vector<ggml_bf16_t> gate_bf16_;
    std::vector<ggml_bf16_t> up_bf16_;
    std::vector<ggml_bf16_t> down_bf16_;
    std::vector<float>       input_;
};

class direct_amx_weight {
  public:
    direct_amx_weight(const std::vector<ggml_bf16_t> & source,
                      int64_t                          output_cols,
                      int64_t                          input_cols,
                      int64_t                          panel_cols,
                      int64_t                          k_block) :
        output_cols_(output_cols),
        input_cols_(input_cols),
        panel_cols_(panel_cols),
        k_block_(k_block) {
        if (output_cols_ % panel_cols_ != 0 || panel_cols_ % 64 != 0) {
            throw std::invalid_argument("direct AMX output dimension must be divisible by the panel width");
        }
        if (source.size() != static_cast<size_t>(output_cols_ * input_cols_)) {
            throw std::invalid_argument("direct AMX source weight has the wrong size");
        }

        packed_count_              = source.size();
        void *    memory           = nullptr;
        const int allocation_error = posix_memalign(&memory, AMX_PAIR_ALIGNMENT, packed_bytes());
        if (allocation_error != 0) {
            throw std::runtime_error(std::string("failed to allocate aligned direct AMX weights: ") +
                                     std::strerror(allocation_error));
        }
        packed_.reset(static_cast<float *>(memory));
        validate_packed_alignment();
        for (int64_t panel = 0; panel < panel_count(); ++panel) {
            const int64_t first_col   = panel * panel_cols_;
            float *       destination = packed_.get() + panel * input_cols_ * panel_cols_;
            for (int64_t k = 0; k < input_cols_; ++k) {
                for (int64_t col = 0; col < panel_cols_; ++col) {
                    const auto source_index            = static_cast<size_t>((first_col + col) * input_cols_ + k);
                    destination[k * panel_cols_ + col] = ggml_bf16_to_fp32(source[source_index]);
                }
            }
        }
    }

    void multiply_transposed(const float * activation_t, float * output, int64_t rows) const {
        if (rows % 16 != 0) {
            throw std::invalid_argument("direct AMX row count must be a multiple of 16");
        }
        validate_packed_alignment();
        worker_context context = { this, activation_t, output, rows };
        dispatch_apply_f(static_cast<size_t>(panel_count()), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                         &context, worker_entry);
    }

    size_t packed_bytes() const { return packed_count_ * sizeof(float); }

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

    static bool is_pair_aligned(const float * address) {
        return reinterpret_cast<uintptr_t>(address) % AMX_PAIR_ALIGNMENT == 0;
    }

    void validate_packed_alignment() const {
        const size_t row_bytes   = static_cast<size_t>(panel_cols_) * sizeof(float);
        const size_t panel_bytes = static_cast<size_t>(input_cols_) * row_bytes;
        if (packed_ == nullptr || !is_pair_aligned(packed_.get()) || row_bytes % AMX_PAIR_ALIGNMENT != 0 ||
            panel_bytes % AMX_PAIR_ALIGNMENT != 0 || 32 * sizeof(float) != AMX_PAIR_ALIGNMENT) {
            throw std::runtime_error("direct AMX paired-load alignment invariant failed");
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
        const int64_t      first_col = panel * panel_cols_;
        const float *      packed    = packed_.get() + panel * input_cols_ * panel_cols_;

        for (int64_t k_base = 0; k_base < input_cols_; k_base += k_block_) {
            const int64_t k_count = std::min(k_block_, input_cols_ - k_base);
            for (int64_t row_base = 0; row_base < rows; row_base += 16) {
                for (int64_t panel_col = 0; panel_col < panel_cols_; panel_col += 64) {
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
                        // validate_packed_alignment() proves both paired LDX addresses are 128-byte aligned.
                        const float * weight_row = packed + (k_base + k) * panel_cols_ + panel_col;
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

    int64_t panel_count() const { return output_cols_ / panel_cols_; }

    int64_t                              output_cols_;
    int64_t                              input_cols_;
    int64_t                              panel_cols_;
    int64_t                              k_block_;
    size_t                               packed_count_ = 0;
    std::unique_ptr<float, aligned_free> packed_;
};

class direct_amx_shared_ffn {
  public:
    direct_amx_shared_ffn(const probe_data & data,
                          int64_t            tokens,
                          float              clamp_limit,
                          int64_t            panel_cols,
                          int64_t            k_block) :
        input_(data.input()),
        tokens_(tokens),
        clamp_limit_(clamp_limit) {
        const uint64_t rss_before = current_rss_bytes();
        const auto     begin      = clock_type::now();
        gate_weight_ = std::make_unique<direct_amx_weight>(data.gate_bf16(), DSV4_FF, DSV4_EMBD, panel_cols, k_block);
        up_weight_   = std::make_unique<direct_amx_weight>(data.up_bf16(), DSV4_FF, DSV4_EMBD, panel_cols, k_block);
        down_weight_ = std::make_unique<direct_amx_weight>(data.down_bf16(), DSV4_EMBD, DSV4_FF, panel_cols, k_block);
        pack_ms_     = elapsed_ms(begin, clock_type::now());
        const uint64_t rss_after = current_rss_bytes();
        pack_rss_delta_          = rss_after >= rss_before ? rss_after - rss_before : 0;

        input_t_.resize(static_cast<size_t>(tokens_ * DSV4_EMBD));
        gate_.resize(static_cast<size_t>(tokens_ * DSV4_FF));
        up_.resize(static_cast<size_t>(tokens_ * DSV4_FF));
        hidden_.resize(static_cast<size_t>(tokens_ * DSV4_FF));
        hidden_t_.resize(static_cast<size_t>(tokens_ * DSV4_FF));
        output_.resize(static_cast<size_t>(tokens_ * DSV4_EMBD));
    }

    shared_timing run() {
        shared_timing result;
        const auto    total_begin = clock_type::now();

        auto begin = clock_type::now();
        transpose(input_.data(), input_t_.data(), tokens_, DSV4_EMBD);
        result.transpose_input_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        gate_weight_->multiply_transposed(input_t_.data(), gate_.data(), tokens_);
        result.gate_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        up_weight_->multiply_transposed(input_t_.data(), up_.data(), tokens_);
        result.up_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        for (size_t i = 0; i < hidden_.size(); ++i) {
            if (clamp_limit_ > 1e-6f) {
                gate_[i] = std::min(gate_[i], clamp_limit_);
                up_[i]   = std::clamp(up_[i], -clamp_limit_, clamp_limit_);
            }
            hidden_[i] = gate_[i] / (1.0f + std::exp(-gate_[i])) * up_[i];
        }
        result.activation_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        transpose(hidden_.data(), hidden_t_.data(), tokens_, DSV4_FF);
        result.transpose_hidden_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        down_weight_->multiply_transposed(hidden_t_.data(), output_.data(), tokens_);
        result.down_ms  = elapsed_ms(begin, clock_type::now());
        result.total_ms = elapsed_ms(total_begin, clock_type::now());
        return result;
    }

    const std::vector<float> & gate() const { return gate_; }

    const std::vector<float> & up() const { return up_; }

    const std::vector<float> & hidden() const { return hidden_; }

    const std::vector<float> & output() const { return output_; }

    double pack_ms() const { return pack_ms_; }

    uint64_t pack_rss_delta() const { return pack_rss_delta_; }

    size_t packed_f32_bytes() const {
        return gate_weight_->packed_bytes() + up_weight_->packed_bytes() + down_weight_->packed_bytes();
    }

  private:
    static void transpose(const float * source, float * destination, int64_t rows, int64_t cols) {
        vDSP_mtrans(source, 1, destination, 1, static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
    }

    const std::vector<float> &         input_;
    int64_t                            tokens_;
    float                              clamp_limit_;
    std::unique_ptr<direct_amx_weight> gate_weight_;
    std::unique_ptr<direct_amx_weight> up_weight_;
    std::unique_ptr<direct_amx_weight> down_weight_;
    std::vector<float>                 input_t_;
    std::vector<float>                 gate_;
    std::vector<float>                 up_;
    std::vector<float>                 hidden_;
    std::vector<float>                 hidden_t_;
    std::vector<float>                 output_;
    double                             pack_ms_        = 0.0;
    uint64_t                           pack_rss_delta_ = 0;
};

class metal_graphs {
  public:
    metal_graphs(ggml_backend_t backend, const probe_data & data, int64_t tokens, float clamp_limit) :
        backend_(backend),
        tokens_(tokens) {
        constexpr size_t       graph_nodes   = 256;
        constexpr size_t       graph_count   = 4;
        const ggml_init_params weight_params = {
            ggml_tensor_overhead() * 8,
            nullptr,
            true,
        };
        const ggml_init_params compute_params = {
            ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(graph_nodes, false) * graph_count,
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
            routed_lanes[i] = ggml_view_2d(compute_.get(), routed, DSV4_EMBD, tokens_, routed->nb[2],
                                           static_cast<size_t>(i) * routed->nb[1]);
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

        route_graph_                  = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        shared_graph_                 = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        all_graph_                    = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        routed_down_validation_graph_ = ggml_new_graph_custom(compute_.get(), graph_nodes, false);
        ggml_build_forward_expand(route_graph_, routed_out_);
        ggml_build_forward_expand(shared_graph_, shared_out_);
        ggml_build_forward_expand(all_graph_, all_out_);
        ggml_build_forward_expand(routed_down_validation_graph_, routed_down_out_);

        weights_buffer_.reset(ggml_backend_alloc_ctx_tensors(weights_.get(), backend_));
        compute_buffer_.reset(ggml_backend_alloc_ctx_tensors(compute_.get(), backend_));
        if (!weights_buffer_ || !compute_buffer_) {
            throw std::runtime_error("failed to allocate Metal probe buffers");
        }
        ggml_backend_buffer_set_usage(weights_buffer_.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        initialize(data);
        validate_graph_support(route_graph_, "routed");
        validate_graph_support(shared_graph_, "shared");
        validate_graph_support(all_graph_, "all-metal");
        validate_graph_support(routed_down_validation_graph_, "routed-down validation");
    }

    double time_route_ms() { return time_graph(route_graph_); }

    double time_shared_ms() { return time_graph(shared_graph_); }

    double time_all_ms() { return time_graph(all_graph_); }

    void materialize_routed_down_for_validation() { compute_graph(routed_down_validation_graph_); }

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
        std::vector<float> result(static_cast<size_t>(ggml_nelements(tensor)));
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
        return result;
    }

    static void initialize_mxfp4(ggml_tensor * tensor, uint32_t seed) {
        constexpr int64_t block_elements = 32;
        float             source[block_elements];
        for (int64_t i = 0; i < block_elements; ++i) {
            source[i] = deterministic_value(static_cast<uint64_t>(i), seed, 0.0625f);
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

    void initialize(const probe_data & data) {
        initialize_mxfp4(route_gate_, 0x7101U);
        initialize_mxfp4(route_up_, 0x7102U);
        initialize_mxfp4(route_down_, 0x7103U);
        ggml_backend_tensor_set(shared_gate_, data.gate_bf16().data(), 0, ggml_nbytes(shared_gate_));
        ggml_backend_tensor_set(shared_up_, data.up_bf16().data(), 0, ggml_nbytes(shared_up_));
        ggml_backend_tensor_set(shared_down_, data.down_bf16().data(), 0, ggml_nbytes(shared_down_));

        ggml_backend_tensor_set(route_input_, data.input().data(), 0, ggml_nbytes(route_input_));

        std::vector<int32_t> ids(static_cast<size_t>(ggml_nelements(ids_base_)), 0);
        for (int64_t token = 0; token < tokens_; ++token) {
            for (int64_t lane = 0; lane < DSV4_EXPERTS_USED; ++lane) {
                ids[static_cast<size_t>(token * DSV4_EXPERTS + lane)] =
                    static_cast<int32_t>((token * 37 + lane * 43) % DSV4_EXPERTS);
            }
        }
        ggml_backend_tensor_set(ids_base_, ids.data(), 0, ggml_nbytes(ids_base_));

        std::vector<float> route_weights(static_cast<size_t>(ggml_nelements(route_weights_)), 1.0f / DSV4_EXPERTS_USED);
        ggml_backend_tensor_set(route_weights_, route_weights.data(), 0, ggml_nbytes(route_weights_));
        ggml_backend_tensor_set(shared_input_, data.input().data(), 0, ggml_nbytes(shared_input_));
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

    void compute_graph(ggml_cgraph * graph) {
        const ggml_status status = ggml_backend_graph_compute(backend_, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Metal graph execution failed");
        }
    }

    double time_graph(ggml_cgraph * graph) {
        const auto begin = clock_type::now();
        compute_graph(graph);
        return elapsed_ms(begin, clock_type::now());
    }

    ggml_backend_t          backend_;
    int64_t                 tokens_;
    ggml_context_ptr        weights_;
    ggml_context_ptr        compute_;
    ggml_backend_buffer_ptr weights_buffer_;
    ggml_backend_buffer_ptr compute_buffer_;
    ggml_tensor *           route_gate_                   = nullptr;
    ggml_tensor *           route_up_                     = nullptr;
    ggml_tensor *           route_down_                   = nullptr;
    ggml_tensor *           shared_gate_                  = nullptr;
    ggml_tensor *           shared_up_                    = nullptr;
    ggml_tensor *           shared_down_                  = nullptr;
    ggml_tensor *           route_input_                  = nullptr;
    ggml_tensor *           ids_base_                     = nullptr;
    ggml_tensor *           ids_                          = nullptr;
    ggml_tensor *           route_weights_                = nullptr;
    ggml_tensor *           shared_input_                 = nullptr;
    ggml_tensor *           routed_gate_out_              = nullptr;
    ggml_tensor *           routed_up_out_                = nullptr;
    ggml_tensor *           routed_hidden_out_            = nullptr;
    ggml_tensor *           routed_down_out_              = nullptr;
    ggml_tensor *           routed_out_                   = nullptr;
    ggml_tensor *           shared_gate_out_              = nullptr;
    ggml_tensor *           shared_up_out_                = nullptr;
    ggml_tensor *           shared_hidden_out_            = nullptr;
    ggml_tensor *           shared_out_                   = nullptr;
    ggml_tensor *           all_out_                      = nullptr;
    ggml_cgraph *           route_graph_                  = nullptr;
    ggml_cgraph *           shared_graph_                 = nullptr;
    ggml_cgraph *           all_graph_                    = nullptr;
    ggml_cgraph *           routed_down_validation_graph_ = nullptr;
};

static overlap_timing run_overlap(direct_amx_shared_ffn & direct, metal_graphs & metal) {
    std::atomic<bool>      ready(false);
    std::atomic<bool>      go(false);
    clock_type::time_point direct_end;
    shared_timing          direct_timing;

    auto direct_future = std::async(std::launch::async, [&]() {
        ready.store(true, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        direct_timing = direct.run();
        direct_end    = clock_type::now();
    });
    while (!ready.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    const auto begin = clock_type::now();
    go.store(true, std::memory_order_release);
    metal.launch_route_async();
    metal.synchronize();
    const auto metal_end = clock_type::now();
    direct_future.get();
    const auto end = std::max(direct_end, metal_end);
    return { elapsed_ms(begin, end), elapsed_ms(begin, direct_end), elapsed_ms(begin, metal_end), direct_timing };
}

static overlap_timing run_sequential(direct_amx_shared_ffn & direct, metal_graphs & metal, bool direct_first) {
    const auto             begin = clock_type::now();
    clock_type::time_point direct_end;
    shared_timing          direct_timing;
    double                 metal_ms;
    if (direct_first) {
        direct_timing = direct.run();
        direct_end    = clock_type::now();
        metal_ms      = metal.time_route_ms();
    } else {
        metal_ms      = metal.time_route_ms();
        direct_timing = direct.run();
        direct_end    = clock_type::now();
    }
    return { elapsed_ms(begin, clock_type::now()), elapsed_ms(begin, direct_end), metal_ms, direct_timing };
}

static void print_raw(const raw_run & run) {
    std::printf(
        "raw\titeration=%d\torder=%s\tmode=%s\twall_ms=%.6f\tdirect_completion_ms=%.6f\tmetal_ms=%.6f"
        "\tdirect_total_ms=%.6f\ttranspose_input_ms=%.6f\tgate_ms=%.6f\tup_ms=%.6f"
        "\tactivation_ms=%.6f\ttranspose_hidden_ms=%.6f\tdown_ms=%.6f\n",
        run.iteration, run.order.c_str(), run.mode.c_str(), run.wall_ms, run.direct_completion_ms, run.metal_ms,
        run.direct.total_ms, run.direct.transpose_input_ms, run.direct.gate_ms, run.direct.up_ms,
        run.direct.activation_ms, run.direct.transpose_hidden_ms, run.direct.down_ms);
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

    const std::string brand = cpu_brand();
    if (brand.find("Apple M2 Ultra") == std::string::npos) {
        std::fprintf(stderr, "llama-dsv4-amx-probe requires Apple M2 Ultra; detected '%s'\n", brand.c_str());
        return 2;
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

    const uint64_t                         rss_start = current_rss_bytes();
    probe_data                             data(opts.tokens);
    const uint64_t                         rss_after_source = current_rss_bytes();
    std::unique_ptr<direct_amx_shared_ffn> direct;
    const bool                             candidate_enabled = !disabled && opts.mode != "sequential";
    if (candidate_enabled) {
        direct =
            std::make_unique<direct_amx_shared_ffn>(data, opts.tokens, opts.clamp_limit, opts.panel_cols, opts.k_block);
    }
    const uint64_t rss_after_direct = current_rss_bytes();
    metal_graphs   metal(metal_backend.get(), data, opts.tokens, opts.clamp_limit);
    const uint64_t rss_after_metal = current_rss_bytes();

    std::printf(
        "probe\tcommit=%s\tdevice=%s\tcpu=%s\ttokens=%lld\tpanel_cols=%lld\tk_block=%lld"
        "\twarmup=%d\truns=%d\tclamp_limit=%.9g\tmode=%s\tenv_disabled=%d\tcandidate_enabled=%d\n",
        ggml_commit(), ggml_backend_dev_description(metal_device), brand.c_str(), static_cast<long long>(opts.tokens),
        static_cast<long long>(opts.panel_cols), static_cast<long long>(opts.k_block), opts.warmup, opts.runs,
        opts.clamp_limit, opts.mode.c_str(), disabled ? 1 : 0, candidate_enabled ? 1 : 0);
    std::printf(
        "shape\tdirect_shared=BF16_source_to_reordered_F32:weights[4096,2048]+[4096,2048]+[2048,4096],"
        "input[4096,%lld]\tmetal_shared=BF16:weights[4096,2048]+[4096,2048]+[2048,4096],input[4096,%lld]"
        "\tmetal_routed=MXFP4:weights[4096,2048,256]+[4096,2048,256]+[2048,4096,256],"
        "input[4096,1,%lld],ids[6,%lld]\trouted_scope=complete_ffn\n",
        static_cast<long long>(opts.tokens), static_cast<long long>(opts.tokens), static_cast<long long>(opts.tokens),
        static_cast<long long>(opts.tokens));
    std::printf(
        "direct_path\tprovider=raw_apple_amx_fma32\tsteady_state_projections=3"
        "\ttranspose_input=vDSP_mtrans\ttranspose_hidden=vDSP_mtrans\tprivate_isa=1"
        "\tcandidate_output_join_implemented=0\toverlap_worker_start_overhead_timed=0"
        "\tpanel_dispatch_overhead_timed=1\n");
    std::printf(
        "pack\tcandidate_allocated=%d\tone_time_ms=%.6f\tbf16_source_bytes=%zu\tdirect_f32_bytes=%zu"
        "\trss_delta_bytes=%llu\tprojected_43_layer_direct_pack_bytes=%zu\tsteady_state_excludes_pack=1\n",
        direct ? 1 : 0, direct ? direct->pack_ms() : 0.0, data.bf16_weight_bytes(),
        direct ? direct->packed_f32_bytes() : 0, static_cast<unsigned long long>(direct ? direct->pack_rss_delta() : 0),
        direct ? direct->packed_f32_bytes() * static_cast<size_t>(DSV4_LAYERS) : size_t{ 0 });
    std::printf(
        "memory\trss_start_bytes=%llu\trss_after_source_bytes=%llu\trss_after_direct_bytes=%llu"
        "\trss_after_metal_bytes=%llu\tmetal_routed_weight_bytes=%zu"
        "\tmxfp4_init=deterministic_repeated_nonzero_blocks\n",
        static_cast<unsigned long long>(rss_start), static_cast<unsigned long long>(rss_after_source),
        static_cast<unsigned long long>(rss_after_direct), static_cast<unsigned long long>(rss_after_metal),
        metal.routed_weight_bytes());

    for (int i = 0; i < opts.warmup; ++i) {
        if (direct) {
            direct->run();
        }
        metal.time_route_ms();
        metal.time_shared_ms();
        metal.time_all_ms();
        if (direct && opts.mode != "sequential") {
            run_overlap(*direct, metal);
        }
    }

    if (direct) {
        direct->run();
    }
    metal.time_all_ms();
    metal.materialize_routed_down_for_validation();
    std::printf("validation\tgraph=routed_down_materialization\ttimed=0\n");
    std::vector<float> metal_gate   = metal.get_shared_gate();
    std::vector<float> metal_up     = metal.get_shared_up();
    std::vector<float> metal_hidden = metal.get_shared_hidden();
    std::vector<float> metal_shared = metal.get_shared_output();
    std::vector<float> metal_routed = metal.get_routed_output();
    std::vector<float> metal_all    = metal.get_all_output();
    std::vector<float> expected_metal_all(metal_all.size());
    for (size_t i = 0; i < expected_metal_all.size(); ++i) {
        expected_metal_all[i] = metal_shared[i] + metal_routed[i];
    }

    const output_stats shared_stats        = summarize(metal_shared);
    const output_stats routed_stats        = summarize(metal_routed);
    const output_stats all_stats           = summarize(metal_all);
    const output_stats routed_gate_stats   = summarize(metal.get_routed_gate());
    const output_stats routed_up_stats     = summarize(metal.get_routed_up());
    const output_stats routed_hidden_stats = summarize(metal.get_routed_hidden());
    const output_stats routed_down_stats   = summarize(metal.get_routed_down());
    const error_stats  metal_add_error     = compare_outputs(expected_metal_all, metal_all);

    output_stats direct_stats;
    error_stats  gate_error;
    error_stats  up_error;
    error_stats  hidden_error;
    error_stats  shared_error;
    if (direct) {
        direct_stats = summarize(direct->output());
        gate_error   = compare_outputs(direct->gate(), metal_gate);
        up_error     = compare_outputs(direct->up(), metal_up);
        hidden_error = compare_outputs(direct->hidden(), metal_hidden);
        shared_error = compare_outputs(direct->output(), metal_shared);
    }

    std::printf(
        "correctness\tdirect_checked=%d\tdirect_finite=%d\tmetal_shared_finite=%d\tmetal_routed_finite=%d"
        "\tmetal_all_finite=%d\tdirect_checksum=%.9e\tmetal_shared_checksum=%.9e"
        "\tmetal_routed_checksum=%.9e\tmetal_all_checksum=%.9e\tdirect_l2=%.9e\tmetal_shared_l2=%.9e"
        "\tmetal_routed_l2=%.9e\tmetal_all_l2=%.9e\trouted_gate_finite=%d\trouted_gate_checksum=%.9e"
        "\trouted_gate_l2=%.9e\trouted_up_finite=%d\trouted_up_checksum=%.9e\trouted_up_l2=%.9e"
        "\trouted_hidden_finite=%d\trouted_hidden_checksum=%.9e\trouted_hidden_l2=%.9e"
        "\trouted_down_finite=%d\trouted_down_checksum=%.9e\trouted_down_l2=%.9e"
        "\tgate_bit_diff=%llu\tgate_nmse=%.9e\tgate_max_abs=%.9e"
        "\tup_bit_diff=%llu\tup_nmse=%.9e\tup_max_abs=%.9e"
        "\thidden_bit_diff=%llu\thidden_nmse=%.9e\thidden_max_abs=%.9e"
        "\tshared_bit_diff=%llu\tshared_nmse=%.9e\tshared_max_abs=%.9e"
        "\tmetal_add_bit_diff=%llu\tmetal_add_nmse=%.9e\tmetal_add_max_abs=%.9e\n",
        direct ? 1 : 0, direct_stats.finite ? 1 : 0, shared_stats.finite ? 1 : 0, routed_stats.finite ? 1 : 0,
        all_stats.finite ? 1 : 0, direct_stats.checksum, shared_stats.checksum, routed_stats.checksum,
        all_stats.checksum, direct_stats.l2, shared_stats.l2, routed_stats.l2, all_stats.l2,
        routed_gate_stats.finite ? 1 : 0, routed_gate_stats.checksum, routed_gate_stats.l2,
        routed_up_stats.finite ? 1 : 0, routed_up_stats.checksum, routed_up_stats.l2,
        routed_hidden_stats.finite ? 1 : 0, routed_hidden_stats.checksum, routed_hidden_stats.l2,
        routed_down_stats.finite ? 1 : 0, routed_down_stats.checksum, routed_down_stats.l2,
        static_cast<unsigned long long>(gate_error.bit_diff), gate_error.nmse, gate_error.max_abs,
        static_cast<unsigned long long>(up_error.bit_diff), up_error.nmse, up_error.max_abs,
        static_cast<unsigned long long>(hidden_error.bit_diff), hidden_error.nmse, hidden_error.max_abs,
        static_cast<unsigned long long>(shared_error.bit_diff), shared_error.nmse, shared_error.max_abs,
        static_cast<unsigned long long>(metal_add_error.bit_diff), metal_add_error.nmse, metal_add_error.max_abs);

    const bool routed_valid = routed_stats.finite && routed_stats.l2 > 0.0 && routed_gate_stats.finite &&
                              routed_gate_stats.l2 > 0.0 && routed_up_stats.finite && routed_up_stats.l2 > 0.0 &&
                              routed_hidden_stats.finite && routed_hidden_stats.l2 > 0.0 && routed_down_stats.finite &&
                              routed_down_stats.l2 > 0.0;
    bool correct = shared_stats.finite && all_stats.finite && routed_valid && metal_add_error.finite &&
                   metal_add_error.nmse <= 1e-12;
    if (direct) {
        correct = correct && direct_stats.finite && gate_error.finite && up_error.finite && hidden_error.finite &&
                  shared_error.finite && gate_error.nmse <= 1e-3 && up_error.nmse <= 1e-3 &&
                  hidden_error.nmse <= 1e-3 && shared_error.nmse <= 1e-3;
    }
    std::printf(
        "correctness_gate\tpass=%d\tdirect_checked=%d\tshared_stage_nmse_limit=1e-3"
        "\tmetal_add_nmse_limit=1e-12\tcandidate_join_checked=0\n",
        correct ? 1 : 0, direct ? 1 : 0);
    if (!correct) {
        std::fprintf(stderr, "correctness gate failed; performance timing suppressed\n");
        return 1;
    }

    std::vector<float> direct_reference;
    std::vector<float> routed_reference;
    if (direct && opts.mode != "sequential") {
        direct_reference = direct->output();
        routed_reference = metal_routed;
    }
    auto release_validation_copy = [](std::vector<float> & values) {
        std::vector<float>().swap(values);
    };
    release_validation_copy(metal_gate);
    release_validation_copy(metal_up);
    release_validation_copy(metal_hidden);
    release_validation_copy(metal_shared);
    release_validation_copy(metal_routed);
    release_validation_copy(metal_all);
    release_validation_copy(expected_metal_all);
    std::printf("memory\tphase=after_validation_release\tcurrent_rss_bytes=%llu\n",
                static_cast<unsigned long long>(current_rss_bytes()));
    auto check_overlap_outputs = [&](const char * phase, int iteration, double measured_wall_ms) {
        const error_stats direct_overlap_error = compare_outputs(direct_reference, direct->output());
        const error_stats routed_overlap_error = compare_outputs(routed_reference, metal.get_routed_output());
        const bool        pass                 = direct_overlap_error.finite && routed_overlap_error.finite &&
                          direct_overlap_error.bit_diff == 0 && routed_overlap_error.bit_diff == 0 &&
                          direct_overlap_error.nmse <= 1e-12 && routed_overlap_error.nmse <= 1e-12;
        std::printf(
            "overlap_validation\tphase=%s\titeration=%d\tpass=%d\tmeasured_wall_ms=%.6f\tdirect_bit_diff=%llu"
            "\tdirect_nmse=%.9e\tdirect_max_abs=%.9e\trouted_bit_diff=%llu\trouted_nmse=%.9e"
            "\trouted_max_abs=%.9e\n",
            phase, iteration, pass ? 1 : 0, measured_wall_ms,
            static_cast<unsigned long long>(direct_overlap_error.bit_diff), direct_overlap_error.nmse,
            direct_overlap_error.max_abs, static_cast<unsigned long long>(routed_overlap_error.bit_diff),
            routed_overlap_error.nmse, routed_overlap_error.max_abs);
        return pass;
    };
    if (direct) {
        const overlap_timing timing = run_overlap(*direct, metal);
        if (!check_overlap_outputs("before_timing", -1, timing.wall_ms)) {
            std::fprintf(stderr, "overlap reproducibility gate failed; performance timing suppressed\n");
            return 1;
        }
    }

    std::vector<raw_run> raw;
    for (int iteration = 0; iteration < opts.runs; ++iteration) {
        // Four adjacent repetitions use the A/B/B/A order. A runs the solo
        // and control cases before the overlap candidate; B reverses it.
        const bool        forward = iteration % 4 == 0 || iteration % 4 == 3;
        const std::string order   = forward ? "A" : "B";
        if (opts.mode == "sequential") {
            const double all_ms = metal.time_all_ms();
            raw.push_back({ iteration, order, "metal_all_control", all_ms, 0.0, all_ms, {} });
            print_raw(raw.back());
            continue;
        }
        if (opts.mode == "candidate") {
            const overlap_timing timing = run_overlap(*direct, metal);
            if (!check_overlap_outputs("measured", iteration, timing.wall_ms)) {
                throw std::runtime_error("measured overlap changed direct or routed output");
            }
            raw.push_back({ iteration, order, "direct_amx_metal_overlap", timing.wall_ms, timing.direct_completion_ms,
                            timing.metal_ms, timing.direct });
            print_raw(raw.back());
            continue;
        }

        auto run_measurement = [&](const std::string & name) {
            raw_run result = { iteration, order, name, 0.0, 0.0, 0.0, {} };
            if (name == "direct_amx_shared_solo") {
                result.direct  = direct->run();
                result.wall_ms = result.direct_completion_ms = result.direct.total_ms;
            } else if (name == "metal_routed_solo") {
                result.wall_ms = result.metal_ms = metal.time_route_ms();
            } else if (name == "metal_shared_solo") {
                result.wall_ms = result.metal_ms = metal.time_shared_ms();
            } else if (name == "metal_all_control") {
                result.wall_ms = result.metal_ms = metal.time_all_ms();
            } else if (name == "direct_amx_metal_sequential") {
                const overlap_timing timing = run_sequential(*direct, metal, forward);
                result.wall_ms              = timing.wall_ms;
                result.direct_completion_ms = timing.direct_completion_ms;
                result.metal_ms             = timing.metal_ms;
                result.direct               = timing.direct;
            } else if (name == "direct_amx_metal_overlap") {
                const overlap_timing timing = run_overlap(*direct, metal);
                if (!check_overlap_outputs("measured", iteration, timing.wall_ms)) {
                    throw std::runtime_error("measured overlap changed direct or routed output");
                }
                result.wall_ms              = timing.wall_ms;
                result.direct_completion_ms = timing.direct_completion_ms;
                result.metal_ms             = timing.metal_ms;
                result.direct               = timing.direct;
            } else {
                GGML_ABORT("unknown DSV4 direct AMX probe measurement");
            }
            raw.push_back(result);
            print_raw(raw.back());
        };

        const std::vector<std::string> sequence =
            forward ? std::vector<std::string>{ "direct_amx_shared_solo",      "metal_routed_solo",
                                                "metal_shared_solo",           "metal_all_control",
                                                "direct_amx_metal_sequential", "direct_amx_metal_overlap" } :
                      std::vector<std::string>{ "direct_amx_metal_overlap", "direct_amx_metal_sequential",
                                                "metal_all_control",        "metal_shared_solo",
                                                "metal_routed_solo",        "direct_amx_shared_solo" };
        for (const std::string & name : sequence) {
            run_measurement(name);
        }
    }

    if (direct) {
        const overlap_timing timing = run_overlap(*direct, metal);
        if (!check_overlap_outputs("after_timing", -1, timing.wall_ms)) {
            std::fprintf(stderr, "post-timing overlap reproducibility gate failed\n");
            return 1;
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
        auto direct_values = [&](const char * mode, double shared_timing::* member) {
            std::vector<double> result;
            for (const raw_run & run : raw) {
                if (run.mode == mode) {
                    result.push_back(run.direct.*member);
                }
            }
            return result;
        };
        const double direct_solo = median(values("direct_amx_shared_solo", &raw_run::wall_ms));
        const double route_solo  = median(values("metal_routed_solo", &raw_run::wall_ms));
        const double shared_solo = median(values("metal_shared_solo", &raw_run::wall_ms));
        const double metal_all   = median(values("metal_all_control", &raw_run::wall_ms));
        const double sequential  = median(values("direct_amx_metal_sequential", &raw_run::wall_ms));
        const double overlap     = median(values("direct_amx_metal_overlap", &raw_run::wall_ms));
        const double overlap_direct_completion =
            median(values("direct_amx_metal_overlap", &raw_run::direct_completion_ms));
        const double overlap_metal = median(values("direct_amx_metal_overlap", &raw_run::metal_ms));
        const double overlap_direct_internal =
            median(direct_values("direct_amx_metal_overlap", &shared_timing::total_ms));
        const double direct_flops = 6.0 * DSV4_EMBD * DSV4_FF * static_cast<double>(opts.tokens);
        const double route_flops  = 6.0 * DSV4_EMBD * DSV4_FF * DSV4_EXPERTS_USED * static_cast<double>(opts.tokens);
        std::printf(
            "summary\tdirect_amx_shared_solo_ms=%.6f\tmetal_shared_solo_ms=%.6f"
            "\tmetal_routed_solo_ms=%.6f\tsolo_direct_plus_routed_ms=%.6f"
            "\tdirect_amx_metal_sequential_wall_ms=%.6f\tmetal_all_control_ms=%.6f"
            "\toverlap_wall_ms=%.6f\toverlap_direct_completion_ms=%.6f\toverlap_metal_completion_ms=%.6f"
            "\toverlap_direct_internal_ms=%.6f\tdirect_slowdown=%.6f\tdirect_internal_slowdown=%.6f"
            "\tmetal_slowdown=%.6f\toverlap_efficiency=%.6f\tdirect_shared_tflops=%.6f"
            "\tmetal_routed_tflops=%.6f\tdirect_solo_transpose_input_ms=%.6f"
            "\tdirect_solo_transpose_hidden_ms=%.6f\toverlap_transpose_input_ms=%.6f"
            "\toverlap_transpose_hidden_ms=%.6f\tcomparison_scope=pre_join_branch_feasibility"
            "\tend_to_end_comparable=0\tprejoin_branch_overlap_fits_under_metal_all=%d\n",
            direct_solo, shared_solo, route_solo, direct_solo + route_solo, sequential, metal_all, overlap,
            overlap_direct_completion, overlap_metal, overlap_direct_internal, overlap_direct_completion / direct_solo,
            overlap_direct_internal / direct_solo, overlap_metal / route_solo, (direct_solo + route_solo) / overlap,
            direct_flops / (direct_solo * 1.0e9), route_flops / (route_solo * 1.0e9),
            median(direct_values("direct_amx_shared_solo", &shared_timing::transpose_input_ms)),
            median(direct_values("direct_amx_shared_solo", &shared_timing::transpose_hidden_ms)),
            median(direct_values("direct_amx_metal_overlap", &shared_timing::transpose_input_ms)),
            median(direct_values("direct_amx_metal_overlap", &shared_timing::transpose_hidden_ms)),
            overlap < metal_all ? 1 : 0);
    }

    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    std::printf(
        "memory_final\tcurrent_rss_bytes=%llu\tmax_rss_bytes=%llu\tswaps=%lld\tminor_faults=%lld"
        "\tmajor_faults=%lld\tblock_inputs=%lld\tblock_outputs=%lld\n",
        static_cast<unsigned long long>(current_rss_bytes()), static_cast<unsigned long long>(usage.ru_maxrss),
        static_cast<long long>(usage.ru_nswap), static_cast<long long>(usage.ru_minflt),
        static_cast<long long>(usage.ru_majflt), static_cast<long long>(usage.ru_inblock),
        static_cast<long long>(usage.ru_oublock));

    ggml_quantize_free();
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "llama-dsv4-amx-probe: %s\n", error.what());
    return 1;
}

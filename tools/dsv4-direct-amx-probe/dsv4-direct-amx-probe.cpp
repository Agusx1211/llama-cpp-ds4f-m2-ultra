// Measurement-only Apple M1-M3 AMX F32 probe. The cache blocking and panel
// geometry follow the public algorithm description in arXiv:2606.25426; this
// implementation was written independently because the paper's inferc source
// repository did not declare a license when inspected on 2026-08-03.

#include "corsix-amx-aarch64.h"
#include "ggml.h"

#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

constexpr int64_t DSV4_EMBD          = 4096;
constexpr int64_t DSV4_FF            = 2048;
constexpr size_t  AMX_PAIR_ALIGNMENT = 128;

struct options {
    int64_t tokens     = 2048;
    int64_t panel_cols = 128;
    int64_t k_block    = 1024;
    int     warmup     = 2;
    int     runs       = 8;
    float   clamp      = 10.0f;
};

struct stage_timing {
    double transpose_input_ms  = 0.0;
    double gate_ms             = 0.0;
    double up_ms               = 0.0;
    double activation_ms       = 0.0;
    double transpose_hidden_ms = 0.0;
    double down_ms             = 0.0;
    double total_ms            = 0.0;
};

struct error_stats {
    bool     finite   = true;
    uint64_t bit_diff = 0;
    double   nmse     = 0.0;
    double   max_abs  = 0.0;
    double   checksum = 0.0;
};

static double elapsed_ms(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

static uint64_t current_rss_bytes() {
    mach_task_basic_info_data_t info;
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

static double median(std::vector<double> values) {
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

static uint32_t mix32(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

static float deterministic_value(uint64_t index, uint32_t seed, float scale) {
    const uint32_t bits = mix32(static_cast<uint32_t>(index) ^ mix32(static_cast<uint32_t>(index >> 32) + seed));
    return scale * static_cast<float>(static_cast<int32_t>(bits & 0xffffU) - 32768) / 32768.0f;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s [--tokens 224|512|1024|2048] [--panel-cols N] [--k-block N] "
                 "[--warmup N] [--runs N] [--clamp F]\n",
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
        } else if (std::strcmp(argv[i], "--clamp") == 0) {
            const char * value = next("--clamp");
            if (value == nullptr) {
                return false;
            }
            opts.clamp = std::strtof(value, nullptr);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    const std::vector<int64_t> token_cases = { 224, 512, 1024, 2048 };
    const bool tokens_ok = std::find(token_cases.begin(), token_cases.end(), opts.tokens) != token_cases.end();
    return tokens_ok && opts.panel_cols >= 64 && opts.panel_cols % 64 == 0 && DSV4_EMBD % opts.panel_cols == 0 &&
           DSV4_FF % opts.panel_cols == 0 && opts.k_block > 0 && opts.k_block % 16 == 0 && opts.warmup >= 0 &&
           opts.runs > 0 && std::isfinite(opts.clamp) && opts.clamp >= 0.0f;
}

static void transpose(const float * source, float * destination, int64_t rows, int64_t cols) {
    vDSP_mtrans(source, 1, destination, 1, static_cast<vDSP_Length>(cols), static_cast<vDSP_Length>(rows));
}

static void swiglu(const float * gate, const float * up, float * output, size_t count, float clamp) {
    for (size_t i = 0; i < count; ++i) {
        const float gate_value = clamp > 1e-6f ? std::min(gate[i], clamp) : gate[i];
        const float up_value   = clamp > 1e-6f ? std::clamp(up[i], -clamp, clamp) : up[i];
        output[i]              = gate_value / (1.0f + std::exp(-gate_value)) * up_value;
    }
}

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

class probe {
  public:
    explicit probe(const options & opts) : opts_(opts) {
        initialize_bf16(gate_bf16_, DSV4_FF, DSV4_EMBD, 0x4101U);
        initialize_bf16(up_bf16_, DSV4_FF, DSV4_EMBD, 0x4102U);
        initialize_bf16(down_bf16_, DSV4_EMBD, DSV4_FF, 0x4103U);
        input_.resize(static_cast<size_t>(opts_.tokens * DSV4_EMBD));
        for (size_t i = 0; i < input_.size(); ++i) {
            input_[i] = deterministic_value(i, 0x5101U, 0.125f);
        }

        const uint64_t oracle_rss_before = current_rss_bytes();
        const auto     oracle_begin      = clock_type::now();
        expand_bf16(gate_bf16_, gate_f32_);
        expand_bf16(up_bf16_, up_f32_);
        expand_bf16(down_bf16_, down_f32_);
        oracle_expand_ms_               = elapsed_ms(oracle_begin, clock_type::now());
        const uint64_t oracle_rss_after = current_rss_bytes();
        oracle_rss_delta_ = oracle_rss_after >= oracle_rss_before ? oracle_rss_after - oracle_rss_before : 0;

        const uint64_t pack_rss_before = current_rss_bytes();
        const auto     pack_begin      = clock_type::now();
        gate_amx_ =
            std::make_unique<direct_amx_weight>(gate_bf16_, DSV4_FF, DSV4_EMBD, opts_.panel_cols, opts_.k_block);
        up_amx_ = std::make_unique<direct_amx_weight>(up_bf16_, DSV4_FF, DSV4_EMBD, opts_.panel_cols, opts_.k_block);
        down_amx_ =
            std::make_unique<direct_amx_weight>(down_bf16_, DSV4_EMBD, DSV4_FF, opts_.panel_cols, opts_.k_block);
        pack_ms_                      = elapsed_ms(pack_begin, clock_type::now());
        const uint64_t pack_rss_after = current_rss_bytes();
        pack_rss_delta_               = pack_rss_after >= pack_rss_before ? pack_rss_after - pack_rss_before : 0;

        const size_t ff_elements     = static_cast<size_t>(opts_.tokens * DSV4_FF);
        const size_t output_elements = static_cast<size_t>(opts_.tokens * DSV4_EMBD);
        input_t_.resize(input_.size());
        direct_gate_.resize(ff_elements);
        direct_up_.resize(ff_elements);
        direct_hidden_.resize(ff_elements);
        hidden_t_.resize(ff_elements);
        direct_output_.resize(output_elements);
        accelerate_gate_.resize(ff_elements);
        accelerate_up_.resize(ff_elements);
        accelerate_hidden_.resize(ff_elements);
        accelerate_output_.resize(output_elements);
    }

    probe(const probe &)             = delete;
    probe & operator=(const probe &) = delete;

    stage_timing run_direct() {
        stage_timing result;
        const auto   total_begin = clock_type::now();

        auto begin = clock_type::now();
        transpose(input_.data(), input_t_.data(), opts_.tokens, DSV4_EMBD);
        result.transpose_input_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        gate_amx_->multiply_transposed(input_t_.data(), direct_gate_.data(), opts_.tokens);
        result.gate_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        up_amx_->multiply_transposed(input_t_.data(), direct_up_.data(), opts_.tokens);
        result.up_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        swiglu(direct_gate_.data(), direct_up_.data(), direct_hidden_.data(), direct_hidden_.size(), opts_.clamp);
        result.activation_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        transpose(direct_hidden_.data(), hidden_t_.data(), opts_.tokens, DSV4_FF);
        result.transpose_hidden_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        down_amx_->multiply_transposed(hidden_t_.data(), direct_output_.data(), opts_.tokens);
        result.down_ms  = elapsed_ms(begin, clock_type::now());
        result.total_ms = elapsed_ms(total_begin, clock_type::now());
        return result;
    }

    stage_timing run_accelerate() {
        stage_timing result;
        const auto   total_begin = clock_type::now();
        auto         begin       = total_begin;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(opts_.tokens), DSV4_FF, DSV4_EMBD, 1.0f,
                    input_.data(), DSV4_EMBD, gate_f32_.data(), DSV4_EMBD, 0.0f, accelerate_gate_.data(), DSV4_FF);
        result.gate_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(opts_.tokens), DSV4_FF, DSV4_EMBD, 1.0f,
                    input_.data(), DSV4_EMBD, up_f32_.data(), DSV4_EMBD, 0.0f, accelerate_up_.data(), DSV4_FF);
        result.up_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        swiglu(accelerate_gate_.data(), accelerate_up_.data(), accelerate_hidden_.data(), accelerate_hidden_.size(),
               opts_.clamp);
        result.activation_ms = elapsed_ms(begin, clock_type::now());

        begin = clock_type::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(opts_.tokens), DSV4_EMBD, DSV4_FF, 1.0f,
                    accelerate_hidden_.data(), DSV4_FF, down_f32_.data(), DSV4_FF, 0.0f, accelerate_output_.data(),
                    DSV4_EMBD);
        result.down_ms  = elapsed_ms(begin, clock_type::now());
        result.total_ms = elapsed_ms(total_begin, clock_type::now());
        return result;
    }

    error_stats gate_error() const { return compare(accelerate_gate_, direct_gate_); }

    error_stats up_error() const { return compare(accelerate_up_, direct_up_); }

    error_stats hidden_error() const { return compare(accelerate_hidden_, direct_hidden_); }

    error_stats output_error() const { return compare(accelerate_output_, direct_output_); }

    size_t bf16_bytes() const {
        return (gate_bf16_.size() + up_bf16_.size() + down_bf16_.size()) * sizeof(ggml_bf16_t);
    }

    size_t oracle_f32_bytes() const { return (gate_f32_.size() + up_f32_.size() + down_f32_.size()) * sizeof(float); }

    size_t packed_f32_bytes() const {
        return gate_amx_->packed_bytes() + up_amx_->packed_bytes() + down_amx_->packed_bytes();
    }

    double oracle_expand_ms() const { return oracle_expand_ms_; }

    uint64_t oracle_rss_delta() const { return oracle_rss_delta_; }

    double pack_ms() const { return pack_ms_; }

    uint64_t pack_rss_delta() const { return pack_rss_delta_; }

  private:
    static void initialize_bf16(std::vector<ggml_bf16_t> & values, int64_t rows, int64_t cols, uint32_t seed) {
        values.resize(static_cast<size_t>(rows * cols));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = ggml_fp32_to_bf16(deterministic_value(i, seed, 0.015625f));
        }
    }

    static void expand_bf16(const std::vector<ggml_bf16_t> & source, std::vector<float> & destination) {
        destination.resize(source.size());
        ggml_bf16_to_fp32_row(source.data(), destination.data(), static_cast<int64_t>(source.size()));
    }

    static error_stats compare(const std::vector<float> & reference, const std::vector<float> & candidate) {
        if (reference.size() != candidate.size()) {
            throw std::logic_error("output sizes differ");
        }
        error_stats result;
        double      error_sq     = 0.0;
        double      reference_sq = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            result.finite = result.finite && std::isfinite(reference[i]) && std::isfinite(candidate[i]);
            uint32_t reference_bits;
            uint32_t candidate_bits;
            std::memcpy(&reference_bits, &reference[i], sizeof(reference_bits));
            std::memcpy(&candidate_bits, &candidate[i], sizeof(candidate_bits));
            result.bit_diff += reference_bits != candidate_bits;
            const double error = static_cast<double>(reference[i]) - candidate[i];
            error_sq += error * error;
            reference_sq += static_cast<double>(reference[i]) * reference[i];
            result.max_abs = std::max(result.max_abs, std::abs(error));
            result.checksum += candidate[i] * (1.0 + static_cast<double>(i % 251) / 251.0);
        }
        result.nmse = reference_sq == 0.0 ? error_sq : error_sq / reference_sq;
        return result;
    }

    options opts_;

    std::vector<ggml_bf16_t>           gate_bf16_;
    std::vector<ggml_bf16_t>           up_bf16_;
    std::vector<ggml_bf16_t>           down_bf16_;
    std::vector<float>                 gate_f32_;
    std::vector<float>                 up_f32_;
    std::vector<float>                 down_f32_;
    std::unique_ptr<direct_amx_weight> gate_amx_;
    std::unique_ptr<direct_amx_weight> up_amx_;
    std::unique_ptr<direct_amx_weight> down_amx_;

    std::vector<float> input_;
    std::vector<float> input_t_;
    std::vector<float> direct_gate_;
    std::vector<float> direct_up_;
    std::vector<float> direct_hidden_;
    std::vector<float> hidden_t_;
    std::vector<float> direct_output_;
    std::vector<float> accelerate_gate_;
    std::vector<float> accelerate_up_;
    std::vector<float> accelerate_hidden_;
    std::vector<float> accelerate_output_;

    double   oracle_expand_ms_ = 0.0;
    uint64_t oracle_rss_delta_ = 0;
    double   pack_ms_          = 0.0;
    uint64_t pack_rss_delta_   = 0;
};

static void print_error(const char * stage, const error_stats & error) {
    std::printf("correctness\tstage=%s\tfinite=%d\tbit_diff=%llu\tnmse=%.9e\tmax_abs=%.9e\tchecksum=%.9e\n", stage,
                error.finite ? 1 : 0, static_cast<unsigned long long>(error.bit_diff), error.nmse, error.max_abs,
                error.checksum);
}

static void append_timing(std::vector<stage_timing> & timings,
                          const stage_timing &        timing,
                          int                         iteration,
                          const char *                order,
                          const char *                arm) {
    timings.push_back(timing);
    std::printf(
        "raw\titeration=%d\torder=%s\tarm=%s\ttotal_ms=%.6f\ttranspose_input_ms=%.6f\tgate_ms=%.6f"
        "\tup_ms=%.6f\tactivation_ms=%.6f\ttranspose_hidden_ms=%.6f\tdown_ms=%.6f\n",
        iteration, order, arm, timing.total_ms, timing.transpose_input_ms, timing.gate_ms, timing.up_ms,
        timing.activation_ms, timing.transpose_hidden_ms, timing.down_ms);
}

static double stage_median(const std::vector<stage_timing> & timings, double stage_timing::* member) {
    std::vector<double> values;
    values.reserve(timings.size());
    for (const stage_timing & timing : timings) {
        values.push_back(timing.*member);
    }
    return median(std::move(values));
}

static double projection_tflops(int64_t tokens, double milliseconds) {
    const double operations = 2.0 * static_cast<double>(tokens) * DSV4_EMBD * DSV4_FF;
    return operations / milliseconds / 1.0e9;
}

}  // namespace

int main(int argc, char ** argv) try {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        usage(argv[0]);
        return 2;
    }

    const std::string brand = cpu_brand();
    if (brand.find("Apple M2 Ultra") == std::string::npos) {
        std::fprintf(stderr, "llama-dsv4-direct-amx-probe requires Apple M2 Ultra; detected '%s'\n", brand.c_str());
        return 2;
    }

    probe benchmark(opts);
    std::printf(
        "probe\tcommit=%s\ttokens=%lld\tpanel_cols=%lld\tk_block=%lld\twarmup=%d\truns=%d\tclamp=%.9g"
        "\tprivate_isa=apple_amx_fma32\toracle=accelerate_cblas_sgemm\tcpu=%s\n",
        ggml_commit(), static_cast<long long>(opts.tokens), static_cast<long long>(opts.panel_cols),
        static_cast<long long>(opts.k_block), opts.warmup, opts.runs, opts.clamp, brand.c_str());
    std::printf(
        "shape\tgate_up=Mx4096_by_2048x4096T\tdown=Mx2048_by_4096x2048T\trows=%lld"
        "\tactivation=clamped_silu_times_up\ttranspose=vDSP_mtrans\n",
        static_cast<long long>(opts.tokens));
    std::printf(
        "pack\tbf16_source_bytes=%zu\tdirect_f32_bytes=%zu\tdirect_ms=%.6f\tdirect_rss_delta=%llu"
        "\toracle_f32_bytes=%zu\toracle_expand_ms=%.6f\toracle_rss_delta=%llu"
        "\tprojected_43_layer_direct_bytes=%zu\tsteady_state_excludes_weight_pack=1\n",
        benchmark.bf16_bytes(), benchmark.packed_f32_bytes(), benchmark.pack_ms(),
        static_cast<unsigned long long>(benchmark.pack_rss_delta()), benchmark.oracle_f32_bytes(),
        benchmark.oracle_expand_ms(), static_cast<unsigned long long>(benchmark.oracle_rss_delta()),
        benchmark.packed_f32_bytes() * 43);

    for (int i = 0; i < opts.warmup; ++i) {
        benchmark.run_direct();
        benchmark.run_accelerate();
    }

    benchmark.run_direct();
    benchmark.run_accelerate();
    const error_stats gate_error   = benchmark.gate_error();
    const error_stats up_error     = benchmark.up_error();
    const error_stats hidden_error = benchmark.hidden_error();
    const error_stats output_error = benchmark.output_error();
    print_error("gate", gate_error);
    print_error("up", up_error);
    print_error("hidden", hidden_error);
    print_error("output", output_error);
    const bool correct = gate_error.finite && up_error.finite && hidden_error.finite && output_error.finite &&
                         gate_error.nmse <= 1e-12 && up_error.nmse <= 1e-12 && hidden_error.nmse <= 1e-12 &&
                         output_error.nmse <= 1e-12 && gate_error.max_abs <= 5e-4 && up_error.max_abs <= 5e-4 &&
                         hidden_error.max_abs <= 5e-4 && output_error.max_abs <= 5e-4;
    std::printf(
        "correctness_gate\tpass=%d\tbit_exact_all=%d\tnmse_limit=1e-12\tmax_abs_limit=5e-4\n", correct ? 1 : 0,
        gate_error.bit_diff == 0 && up_error.bit_diff == 0 && hidden_error.bit_diff == 0 && output_error.bit_diff == 0 ?
            1 :
            0);
    if (!correct) {
        std::fprintf(stderr, "direct AMX correctness gate failed; performance timing suppressed\n");
        return 1;
    }

    std::vector<stage_timing> direct_timings;
    std::vector<stage_timing> accelerate_timings;
    for (int iteration = 0; iteration < opts.runs; ++iteration) {
        const bool   direct_first = iteration % 4 == 0 || iteration % 4 == 3;
        const char * order        = direct_first ? "A" : "B";
        if (direct_first) {
            append_timing(direct_timings, benchmark.run_direct(), iteration, order, "direct_amx");
            append_timing(accelerate_timings, benchmark.run_accelerate(), iteration, order, "accelerate");
        } else {
            append_timing(accelerate_timings, benchmark.run_accelerate(), iteration, order, "accelerate");
            append_timing(direct_timings, benchmark.run_direct(), iteration, order, "direct_amx");
        }
    }

    const double direct_total     = stage_median(direct_timings, &stage_timing::total_ms);
    const double accelerate_total = stage_median(accelerate_timings, &stage_timing::total_ms);
    const double direct_gate      = stage_median(direct_timings, &stage_timing::gate_ms);
    const double direct_up        = stage_median(direct_timings, &stage_timing::up_ms);
    const double direct_down      = stage_median(direct_timings, &stage_timing::down_ms);
    const double accelerate_gate  = stage_median(accelerate_timings, &stage_timing::gate_ms);
    const double accelerate_up    = stage_median(accelerate_timings, &stage_timing::up_ms);
    const double accelerate_down  = stage_median(accelerate_timings, &stage_timing::down_ms);
    std::printf(
        "summary\tarm=direct_amx\ttotal_ms=%.6f\ttranspose_input_ms=%.6f\tgate_ms=%.6f\tup_ms=%.6f"
        "\tactivation_ms=%.6f\ttranspose_hidden_ms=%.6f\tdown_ms=%.6f\n",
        direct_total, stage_median(direct_timings, &stage_timing::transpose_input_ms), direct_gate, direct_up,
        stage_median(direct_timings, &stage_timing::activation_ms),
        stage_median(direct_timings, &stage_timing::transpose_hidden_ms), direct_down);
    std::printf(
        "summary\tarm=accelerate\ttotal_ms=%.6f\tgate_ms=%.6f\tup_ms=%.6f\tactivation_ms=%.6f"
        "\tdown_ms=%.6f\n",
        accelerate_total, accelerate_gate, accelerate_up,
        stage_median(accelerate_timings, &stage_timing::activation_ms), accelerate_down);
    std::printf("projection\tname=gate\tdirect_tflops=%.6f\taccelerate_tflops=%.6f\tdirect_over_accelerate=%.6f\n",
                projection_tflops(opts.tokens, direct_gate), projection_tflops(opts.tokens, accelerate_gate),
                accelerate_gate / direct_gate);
    std::printf("projection\tname=up\tdirect_tflops=%.6f\taccelerate_tflops=%.6f\tdirect_over_accelerate=%.6f\n",
                projection_tflops(opts.tokens, direct_up), projection_tflops(opts.tokens, accelerate_up),
                accelerate_up / direct_up);
    std::printf("projection\tname=down\tdirect_tflops=%.6f\taccelerate_tflops=%.6f\tdirect_over_accelerate=%.6f\n",
                projection_tflops(opts.tokens, direct_down), projection_tflops(opts.tokens, accelerate_down),
                accelerate_down / direct_down);
    std::printf("decision\taccelerate_over_direct=%.6f\tdirect_faster=%d\tmeasurement_scope=solo_shared_ffn\n",
                accelerate_total / direct_total, direct_total < accelerate_total ? 1 : 0);
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "fatal: %s\n", error.what());
    return 1;
}

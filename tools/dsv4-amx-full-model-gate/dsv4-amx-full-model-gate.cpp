#include "build-info.h"
#include "llama.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern char ** environ;

namespace {

constexpr int32_t  PROMPT_TOKENS       = 2048;
constexpr int32_t  CONTINUATION_TOKENS = 64;
constexpr int32_t  LOGICAL_TOKENS      = PROMPT_TOKENS + CONTINUATION_TOKENS;
constexpr int32_t  CONTEXT_ALIGNMENT   = 256;
constexpr int32_t  RUNTIME_N_CTX   = ((LOGICAL_TOKENS + CONTEXT_ALIGNMENT - 1) / CONTEXT_ALIGNMENT) * CONTEXT_ALIGNMENT;
constexpr int32_t  CONTEXT_PADDING = RUNTIME_N_CTX - LOGICAL_TOKENS;
constexpr int32_t  NO_INPUT_TOKEN  = std::numeric_limits<int32_t>::min();
constexpr uint32_t FORMAT_VERSION  = 1;
constexpr uint32_t PROMPT_VERSION  = 1;
constexpr uint32_t ENDIAN_MARKER   = 0x01020304U;

static_assert(sizeof(llama_token) == sizeof(int32_t), "capture format requires 32-bit tokens");
static_assert(sizeof(float) == sizeof(uint32_t), "capture format requires 32-bit floats");
static_assert(std::numeric_limits<float>::is_iec559, "capture format requires IEEE-754 floats");
static_assert(LOGICAL_TOKENS == 2112 && RUNTIME_N_CTX == 2304 && CONTEXT_PADDING == 192,
              "full-model gate context geometry changed");

constexpr std::array<char, 8> FILE_MAGIC     = { 'D', 'S', 'V', '4', 'L', 'G', '0', '1' };
constexpr std::array<char, 8> TEACHER_MARKER = { 'T', 'C', 'H', 'L', 'O', 'G', '0', '1' };
constexpr std::array<char, 8> GREEDY_MARKER  = { 'G', 'R', 'Y', 'L', 'O', 'G', '0', '1' };
constexpr std::array<char, 8> FOOTER_MARKER  = { 'D', 'S', 'V', '4', 'D', 'O', 'N', 'E' };

constexpr const char * PROMPT_SEED =
    "A careful systems researcher is validating a language model optimization on one fixed computer. "
    "The researcher records the source revision, model identity, numerical outputs, and every correctness invariant. ";
constexpr const char * PROMPT_FILLER =
    "Continue the reproducible analysis in precise natural language. Compare the control and candidate under identical "
    "inputs, preserve complete evidence, reject nonfinite data, and do not infer performance from a correctness run. ";
constexpr const char * TEACHER_TEXT =
    "The final report explains that deterministic evidence, exact provenance, and conservative numerical gates are "
    "required before any timing experiment can begin. ";

enum class run_mode : uint32_t {
    control   = 0,
    candidate = 1,
};

const char * mode_name(run_mode mode) {
    return mode == run_mode::control ? "control" : "candidate";
}

struct options {
    run_mode    mode = run_mode::control;
    std::string model;
    std::string expected_commit;
    std::string output;
    bool        have_mode = false;
};

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

int fail(run_mode mode, const char * event, const char * reason) {
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=%s outcome=fail mode=%s reason=%s\n", event, mode_name(mode),
                 reason);
    return 1;
}

bool starts_with(const std::string & value, const char * prefix) {
    return value.compare(0, std::strlen(prefix), prefix) == 0;
}

bool exact_environment(run_mode mode) {
    static constexpr std::array<const char *, 7> prefixes = {
        "GGML_", "LLAMA_", "METAL_", "MTL_", "DYLD_", "ACCELERATE_", "OMP_",
    };
    std::map<std::string, std::string> expected = {
        { "LLAMA_DSV4_AMX_COEXEC", "1" }
    };
    if (mode == run_mode::control) {
        expected.emplace("LLAMA_DSV4_AMX_COEXEC_DISABLE", "1");
    }

    std::map<std::string, std::string> actual;
    for (char ** item = environ; item != nullptr && *item != nullptr; ++item) {
        const std::string entry(*item);
        const size_t      separator = entry.find('=');
        if (separator == std::string::npos) {
            continue;
        }
        const std::string key      = entry.substr(0, separator);
        bool              relevant = false;
        for (const char * prefix : prefixes) {
            relevant = relevant || starts_with(key, prefix);
        }
        if (relevant) {
            actual[key] = entry.substr(separator + 1);
        }
    }
    return actual == expected;
}

bool parse_options(int argc, char ** argv, options & result) {
    if (argc != 9) {
        return false;
    }
    for (int i = 1; i < argc; i += 2) {
        const std::string key   = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--mode" && !result.have_mode) {
            if (value == "control") {
                result.mode = run_mode::control;
            } else if (value == "candidate") {
                result.mode = run_mode::candidate;
            } else {
                return false;
            }
            result.have_mode = true;
        } else if (key == "--model" && result.model.empty()) {
            result.model = value;
        } else if (key == "--expected-commit" && result.expected_commit.empty()) {
            result.expected_commit = value;
        } else if (key == "--output" && result.output.empty()) {
            result.output = value;
        } else {
            return false;
        }
    }
    return result.have_mode && !result.model.empty() && !result.expected_commit.empty() && !result.output.empty();
}

bool tokenize(const llama_vocab *        vocab,
              const std::string &        text,
              bool                       add_special,
              std::vector<llama_token> & tokens) {
    const int32_t required =
        llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, true);
    if (required >= 0 || required == std::numeric_limits<int32_t>::min()) {
        return false;
    }
    tokens.resize(static_cast<size_t>(-required));
    const int32_t written = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                                           static_cast<int32_t>(tokens.size()), add_special, true);
    return written == static_cast<int32_t>(tokens.size());
}

uint64_t token_hash(const std::vector<llama_token> & tokens) {
    uint64_t result = 14695981039346656037ULL;
    for (llama_token token : tokens) {
        const uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8) {
            result ^= (value >> shift) & 0xffU;
            result *= 1099511628211ULL;
        }
    }
    return result;
}

bool make_inputs(const llama_vocab *        vocab,
                 std::vector<llama_token> & prompt,
                 std::vector<llama_token> & teacher,
                 size_t &                   seed_tokens,
                 size_t &                   filler_tokens) {
    std::vector<llama_token> filler;
    std::vector<llama_token> teacher_seed;
    if (!tokenize(vocab, PROMPT_SEED, true, prompt) || !tokenize(vocab, PROMPT_FILLER, false, filler) ||
        !tokenize(vocab, TEACHER_TEXT, false, teacher_seed) || prompt.empty() || filler.empty() ||
        teacher_seed.empty() || prompt.size() >= PROMPT_TOKENS) {
        return false;
    }
    seed_tokens   = prompt.size();
    filler_tokens = filler.size();
    size_t index  = 0;
    while (prompt.size() < PROMPT_TOKENS) {
        prompt.push_back(filler[index++ % filler.size()]);
    }
    teacher.reserve(CONTINUATION_TOKENS);
    for (int32_t i = 0; i < CONTINUATION_TOKENS; ++i) {
        teacher.push_back(teacher_seed[static_cast<size_t>(i) % teacher_seed.size()]);
    }
    return prompt.size() == PROMPT_TOKENS && teacher.size() == CONTINUATION_TOKENS;
}

class capture_writer {
  public:
    capture_writer() = default;

    ~capture_writer() {
        if (file_ != nullptr) {
            std::fclose(file_);
        }
    }

    bool open(const std::string &              path,
              run_mode                         mode,
              const std::string &              commit,
              int32_t                          n_vocab,
              const std::vector<llama_token> & prompt,
              const std::vector<llama_token> & teacher) {
        if (std::FILE * existing = std::fopen(path.c_str(), "rb")) {
            std::fclose(existing);
            return false;
        }
        file_ = std::fopen(path.c_str(), "wb");
        if (file_ == nullptr) {
            return false;
        }
        const std::array<uint32_t, 9> fields = {
            FORMAT_VERSION,
            ENDIAN_MARKER,
            static_cast<uint32_t>(mode),
            static_cast<uint32_t>(n_vocab),
            static_cast<uint32_t>(prompt.size()),
            static_cast<uint32_t>(teacher.size()),
            2U,
            static_cast<uint32_t>(CONTINUATION_TOKENS + 1),
            PROMPT_VERSION,
        };
        return write(FILE_MAGIC.data(), FILE_MAGIC.size()) && write(fields.data(), sizeof(fields)) &&
               write(commit.data(), commit.size()) && write(prompt.data(), prompt.size() * sizeof(prompt[0])) &&
               write(teacher.data(), teacher.size() * sizeof(teacher[0]));
    }

    bool begin_path(bool greedy) {
        const auto & marker = greedy ? GREEDY_MARKER : TEACHER_MARKER;
        return write(marker.data(), marker.size());
    }

    bool write_logits(int32_t step, int32_t input_token, int32_t top1, const float * logits, int32_t n_vocab) {
        const std::array<int32_t, 3> fields = { step, input_token, top1 };
        return write(fields.data(), sizeof(fields)) && write(logits, static_cast<size_t>(n_vocab) * sizeof(float));
    }

    bool finish(size_t & bytes) {
        if (!write(FOOTER_MARKER.data(), FOOTER_MARKER.size()) || std::fflush(file_) != 0) {
            return false;
        }
        const long position = std::ftell(file_);
        if (position < 0 || std::fclose(file_) != 0) {
            file_ = nullptr;
            return false;
        }
        file_ = nullptr;
        bytes = static_cast<size_t>(position);
        return true;
    }

  private:
    bool write(const void * data, size_t bytes) {
        return file_ != nullptr && bytes > 0 && std::fwrite(data, 1, bytes, file_) == bytes;
    }

    std::FILE * file_ = nullptr;
};

context_ptr make_context(llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = RUNTIME_N_CTX;
    params.n_batch              = PROMPT_TOKENS;
    params.n_ubatch             = PROMPT_TOKENS;
    params.n_seq_max            = 1;
    params.n_rs_seq             = 0;
    params.n_outputs_max        = 1;
    params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.no_perf              = true;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

bool capture_current_logits(llama_context *  context,
                            capture_writer & writer,
                            int32_t          step,
                            int32_t          input_token,
                            int32_t          n_vocab,
                            llama_token &    top1) {
    llama_synchronize(context);
    const float * logits = llama_get_logits_ith(context, -1);
    if (logits == nullptr || n_vocab <= 0) {
        return false;
    }
    int32_t best = 0;
    for (int32_t token = 0; token < n_vocab; ++token) {
        if (!std::isfinite(logits[token])) {
            return false;
        }
        if (token == 0 || logits[token] > logits[best]) {
            best = token;
        }
    }
    top1 = best;
    return writer.write_logits(step, input_token, best, logits, n_vocab);
}

bool run_path(run_mode                         mode,
              const char *                     path_name,
              bool                             greedy,
              llama_model *                    model,
              const std::vector<llama_token> & prompt,
              const std::vector<llama_token> & teacher,
              int32_t                          n_vocab,
              capture_writer &                 writer) {
    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=path outcome=begin mode=%s path=%s prompt_tokens=%d steps=%d\n",
                 mode_name(mode), path_name, PROMPT_TOKENS, CONTINUATION_TOKENS);
    context_ptr context = make_context(model);
    if (!context) {
        std::fprintf(stderr,
                     "dsv4_amx_full_gate_driver event=context_contract outcome=fail mode=%s path=%s "
                     "stage=create reason=context_create_failed expected_n_ctx=%d logical_tokens=%d\n",
                     mode_name(mode), path_name, RUNTIME_N_CTX, LOGICAL_TOKENS);
        return false;
    }
    const uint32_t actual_n_ctx     = llama_n_ctx(context.get());
    const uint32_t actual_n_batch   = llama_n_batch(context.get());
    const uint32_t actual_n_ubatch  = llama_n_ubatch(context.get());
    const uint32_t actual_n_seq_max = llama_n_seq_max(context.get());
    if (actual_n_ctx != RUNTIME_N_CTX || actual_n_batch != PROMPT_TOKENS || actual_n_ubatch != PROMPT_TOKENS ||
        actual_n_seq_max != 1) {
        std::fprintf(stderr,
                     "dsv4_amx_full_gate_driver event=context_contract outcome=fail mode=%s path=%s "
                     "stage=validate reason=runtime_shape_mismatch expected_n_ctx=%d actual_n_ctx=%u "
                     "expected_n_batch=%d actual_n_batch=%u expected_n_ubatch=%d actual_n_ubatch=%u "
                     "expected_n_seq_max=1 actual_n_seq_max=%u logical_tokens=%d\n",
                     mode_name(mode), path_name, RUNTIME_N_CTX, actual_n_ctx, PROMPT_TOKENS, actual_n_batch,
                     PROMPT_TOKENS, actual_n_ubatch, actual_n_seq_max, LOGICAL_TOKENS);
        return false;
    }
    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=context_contract outcome=pass mode=%s path=%s "
                 "n_ctx=%u n_batch=%u n_ubatch=%u n_seq_max=%u logical_tokens=%d context_padding=%d\n",
                 mode_name(mode), path_name, actual_n_ctx, actual_n_batch, actual_n_ubatch, actual_n_seq_max,
                 LOGICAL_TOKENS, CONTEXT_PADDING);

    if (!writer.begin_path(greedy)) {
        std::fprintf(stderr,
                     "dsv4_amx_full_gate_driver event=capture outcome=fail mode=%s path=%s phase=path_marker "
                     "step=0 reason=write_failed\n",
                     mode_name(mode), path_name);
        return false;
    }
    const int32_t prompt_decode_result =
        llama_decode(context.get(), llama_batch_get_one(const_cast<llama_token *>(prompt.data()), PROMPT_TOKENS));
    if (prompt_decode_result != 0) {
        std::fprintf(stderr,
                     "dsv4_amx_full_gate_driver event=decode outcome=fail mode=%s path=%s phase=prompt step=0 "
                     "tokens=%d return_code=%d reason=llama_decode_failed\n",
                     mode_name(mode), path_name, PROMPT_TOKENS, prompt_decode_result);
        return false;
    }

    llama_token top1 = LLAMA_TOKEN_NULL;
    if (!capture_current_logits(context.get(), writer, 0, NO_INPUT_TOKEN, n_vocab, top1)) {
        std::fprintf(stderr,
                     "dsv4_amx_full_gate_driver event=capture outcome=fail mode=%s path=%s phase=prompt_logits "
                     "step=0 reason=logit_capture_failed\n",
                     mode_name(mode), path_name);
        return false;
    }
    for (int32_t step = 0; step < CONTINUATION_TOKENS; ++step) {
        llama_token input = greedy ? top1 : teacher[static_cast<size_t>(step)];
        if (input < 0 || input >= n_vocab) {
            std::fprintf(stderr,
                         "dsv4_amx_full_gate_driver event=input outcome=fail mode=%s path=%s phase=continuation "
                         "step=%d token=%d n_vocab=%d reason=token_out_of_range\n",
                         mode_name(mode), path_name, step + 1, input, n_vocab);
            return false;
        }
        const int32_t decode_result = llama_decode(context.get(), llama_batch_get_one(&input, 1));
        if (decode_result != 0) {
            std::fprintf(stderr,
                         "dsv4_amx_full_gate_driver event=decode outcome=fail mode=%s path=%s phase=continuation "
                         "step=%d tokens=1 input=%d return_code=%d reason=llama_decode_failed\n",
                         mode_name(mode), path_name, step + 1, input, decode_result);
            return false;
        }
        if (!capture_current_logits(context.get(), writer, step + 1, input, n_vocab, top1)) {
            std::fprintf(stderr,
                         "dsv4_amx_full_gate_driver event=capture outcome=fail mode=%s path=%s "
                         "phase=continuation_logits step=%d input=%d reason=logit_capture_failed\n",
                         mode_name(mode), path_name, step + 1, input);
            return false;
        }
    }

    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=path outcome=pass mode=%s path=%s prompt_tokens=%d steps=%d "
                 "vectors=%d\n",
                 mode_name(mode), path_name, PROMPT_TOKENS, CONTINUATION_TOKENS, CONTINUATION_TOKENS + 1);
    context.reset();
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=context_teardown outcome=pass mode=%s path=%s\n",
                 mode_name(mode), path_name);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        std::fprintf(stderr,
                     "usage: %s --mode <control|candidate> --model <first-shard.gguf> "
                     "--expected-commit <full-sha> --output <capture.bin>\n",
                     argv[0]);
        return 2;
    }

    const std::string build_commit = llama_commit();
    if (opts.expected_commit.size() != 40 || build_commit == "unknown" ||
        opts.expected_commit.compare(0, build_commit.size(), build_commit) != 0) {
        return fail(opts.mode, "provenance", "build_commit_mismatch");
    }
    if (!exact_environment(opts.mode)) {
        return fail(opts.mode, "environment", "runtime_environment_mismatch");
    }

    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=start outcome=begin mode=%s source_commit=%s build_commit=%s "
                 "n_ctx=%d logical_tokens=%d context_alignment=%d context_padding=%d "
                 "n_batch=%d n_ubatch=%d n_seq_max=1 n_rs_seq=0 flash_attn=1 n_gpu_layers=999 "
                 "split_mode=layer main_gpu=0 load_mtp=0 n_outputs_max=1 no_perf=1 prompt_tokens=%d steps=%d "
                 "paths=2 validation=0 timing=0 benchmark=0\n",
                 mode_name(opts.mode), opts.expected_commit.c_str(), build_commit.c_str(), RUNTIME_N_CTX,
                 LOGICAL_TOKENS, CONTEXT_ALIGNMENT, CONTEXT_PADDING, PROMPT_TOKENS, PROMPT_TOKENS, PROMPT_TOKENS,
                 CONTINUATION_TOKENS);

    backend_scope      backend;
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 999;
    model_params.split_mode         = LLAMA_SPLIT_MODE_LAYER;
    model_params.main_gpu           = 0;
    model_params.load_mtp           = false;

    model_ptr model(llama_model_load_from_file(opts.model.c_str(), model_params), llama_model_free);
    if (!model) {
        return fail(opts.mode, "model_load", "model_load_failed");
    }
    const llama_vocab * vocab   = llama_model_get_vocab(model.get());
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);
    if (n_vocab <= 0) {
        return fail(opts.mode, "model_contract", "empty_vocabulary");
    }
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=model_load outcome=pass mode=%s n_vocab=%d\n",
                 mode_name(opts.mode), n_vocab);

    std::vector<llama_token> prompt;
    std::vector<llama_token> teacher;
    size_t                   seed_tokens   = 0;
    size_t                   filler_tokens = 0;
    if (!make_inputs(vocab, prompt, teacher, seed_tokens, filler_tokens)) {
        return fail(opts.mode, "prompt", "tokenization_failed");
    }
    const uint64_t prompt_hash  = token_hash(prompt);
    const uint64_t teacher_hash = token_hash(teacher);
    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=prompt outcome=pass mode=%s prompt_version=%u tokens=%zu "
                 "teacher_tokens=%zu seed_tokens=%zu filler_tokens=%zu prompt_hash=0x%016llx "
                 "teacher_hash=0x%016llx padded=1\n",
                 mode_name(opts.mode), PROMPT_VERSION, prompt.size(), teacher.size(), seed_tokens, filler_tokens,
                 static_cast<unsigned long long>(prompt_hash), static_cast<unsigned long long>(teacher_hash));

    capture_writer writer;
    if (!writer.open(opts.output, opts.mode, opts.expected_commit, n_vocab, prompt, teacher)) {
        return fail(opts.mode, "output", "capture_open_failed");
    }
    if (!run_path(opts.mode, "teacher", false, model.get(), prompt, teacher, n_vocab, writer)) {
        return fail(opts.mode, "teacher", "path_failed");
    }
    if (!run_path(opts.mode, "greedy", true, model.get(), prompt, teacher, n_vocab, writer)) {
        return fail(opts.mode, "greedy", "path_failed");
    }

    size_t output_bytes = 0;
    if (!writer.finish(output_bytes)) {
        return fail(opts.mode, "output", "capture_finalize_failed");
    }
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=output outcome=pass mode=%s bytes=%zu paths=2 vectors=%d\n",
                 mode_name(opts.mode), output_bytes, 2 * (CONTINUATION_TOKENS + 1));

    model.reset();
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=model_teardown outcome=pass mode=%s\n", mode_name(opts.mode));
    backend.shutdown();
    std::fprintf(stderr, "dsv4_amx_full_gate_driver event=backend_teardown outcome=pass mode=%s\n",
                 mode_name(opts.mode));
    std::fprintf(stderr,
                 "dsv4_amx_full_gate_driver event=complete outcome=pass mode=%s prompt_tokens=%d steps=%d paths=2 "
                 "vectors=%d validation=0 benchmark=0\n",
                 mode_name(opts.mode), PROMPT_TOKENS, CONTINUATION_TOKENS, 2 * (CONTINUATION_TOKENS + 1));
    return 0;
}

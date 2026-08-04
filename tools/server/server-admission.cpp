#include "server-admission.h"

#include <algorithm>

server_dsv4_admission_plan server_dsv4_plan_admission(
        uint64_t prompt_tokens,
        int32_t request_n_predict,
        int32_t global_n_predict,
        uint32_t speculative_n_max,
        uint64_t n_ctx_slot) {
    server_dsv4_admission_plan result;
    result.prompt_tokens = prompt_tokens;
    result.effective_n_predict = request_n_predict != -1 ? request_n_predict : global_n_predict;

    if (prompt_tokens == 0 || n_ctx_slot == 0 || request_n_predict < -1 || global_n_predict < -1) {
        return result;
    }
    if (prompt_tokens > n_ctx_slot) {
        result.status = server_dsv4_admission_status::context_overflow;
        return result;
    }

    const uint64_t runway_cap = uint64_t { 1 } + speculative_n_max;
    if (result.effective_n_predict >= 0) {
        const uint64_t output_tokens = (uint32_t) result.effective_n_predict;
        if (output_tokens > n_ctx_slot - prompt_tokens) {
            result.status = server_dsv4_admission_status::context_overflow;
            return result;
        }
        result.decode_runway = std::min(output_tokens, runway_cap);
    } else {
        if (prompt_tokens == n_ctx_slot) {
            result.status = server_dsv4_admission_status::context_overflow;
            return result;
        }
        result.decode_runway = std::min(runway_cap, n_ctx_slot - prompt_tokens);
    }

    result.span_tokens = prompt_tokens + result.decode_runway;
    result.status = server_dsv4_admission_status::ok;
    return result;
}

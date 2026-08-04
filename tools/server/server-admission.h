#pragma once

#include <cstdint>

enum class server_dsv4_admission_status {
    ok,
    invalid,
    context_overflow,
};

struct server_dsv4_admission_plan {
    server_dsv4_admission_status status = server_dsv4_admission_status::invalid;
    uint64_t prompt_tokens = 0;
    uint64_t decode_runway = 0;
    uint64_t span_tokens = 0;
    int32_t effective_n_predict = -1;
};

// Derive the logical span that physical DSV4 admission must cover. A finite
// output budget is authoritative for logical context overflow; an unlimited
// request gets only the immediate target/speculative decode runway and must
// still stop at the context boundary during generation.
server_dsv4_admission_plan server_dsv4_plan_admission(
        uint64_t prompt_tokens,
        int32_t request_n_predict,
        int32_t global_n_predict,
        uint32_t speculative_n_max,
        uint64_t n_ctx_slot);

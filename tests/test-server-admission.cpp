#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-admission.h"

#include <cassert>

int main() {
    {
        const auto plan = server_dsv4_plan_admission(100, 32, -1, 4, 1024);
        assert(plan.status == server_dsv4_admission_status::ok);
        assert(plan.effective_n_predict == 32);
        assert(plan.decode_runway == 5);
        assert(plan.span_tokens == 105);
    }
    {
        const auto plan = server_dsv4_plan_admission(100, -1, 3, 8, 1024);
        assert(plan.status == server_dsv4_admission_status::ok);
        assert(plan.decode_runway == 3);
        assert(plan.span_tokens == 103);
    }
    {
        const auto plan = server_dsv4_plan_admission(1022, -1, -1, 8, 1024);
        assert(plan.status == server_dsv4_admission_status::ok);
        assert(plan.decode_runway == 2);
        assert(plan.span_tokens == 1024);
    }
    {
        const auto plan = server_dsv4_plan_admission(1000, 25, -1, 2, 1024);
        assert(plan.status == server_dsv4_admission_status::context_overflow);
    }
    {
        const auto plan = server_dsv4_plan_admission(1024, 0, -1, 2, 1024);
        assert(plan.status == server_dsv4_admission_status::ok);
        assert(plan.decode_runway == 0);
        assert(plan.span_tokens == 1024);
    }
    {
        const auto invalid = server_dsv4_plan_admission(0, 1, -1, 0, 1024);
        assert(invalid.status == server_dsv4_admission_status::invalid);
    }
    {
        const auto unlimited_full = server_dsv4_plan_admission(1024, -1, -1, 0, 1024);
        assert(unlimited_full.status == server_dsv4_admission_status::context_overflow);
    }
    {
        const auto bounded = server_dsv4_plan_admission(1, -1, -1, UINT32_MAX, 1024);
        assert(bounded.status == server_dsv4_admission_status::ok);
        assert(bounded.decode_runway == 1023);
        assert(bounded.span_tokens == 1024);
    }
    return 0;
}

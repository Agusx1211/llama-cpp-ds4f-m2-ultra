#include "llama-kv-cache-dsv4.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void expect(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool has(uint32_t mask, llama_dsv4_resident_component component) {
    return (mask & component) != 0;
}

void test_current_layout_fails_closed() {
    const auto affine =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::single_context }, 4, 3, false);
    expect(affine.status == llama_dsv4_resident_status::unsupported_components,
           "affine whole-sequence quote unexpectedly succeeded");
    expect(affine.rollback_index == 3 && affine.detachable_components == LLAMA_DSV4_RESIDENT_ROLLBACK_INDEX,
           "affine capability quote lost rollback metadata");
    expect(has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_RAW_SWA) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_COMPRESSED) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_CSA_STATE) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_HCA_STATE) &&
               has(affine.unsupported_components, LLAMA_DSV4_RESIDENT_LID_STATE),
           "affine quote omitted a fixed sequence component");

    const auto aggregate =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::single_context }, 4, 3, true);
    expect(aggregate.status == llama_dsv4_resident_status::unsupported_components,
           "aggregate compressed ownership made the whole-sequence quote succeed");
    expect(has(aggregate.detachable_components, LLAMA_DSV4_RESIDENT_COMPRESSED) &&
               !has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_COMPRESSED),
           "aggregate quote did not expose compressed capability");
    expect(has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_RAW_SWA) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_CSA_STATE) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_HCA_STATE) &&
               has(aggregate.unsupported_components, LLAMA_DSV4_RESIDENT_LID_STATE),
           "aggregate quote hid whole-sequence blockers");

    const auto paired =
        llama_dsv4_quote_resident_detach_layout({ 2, llama_dsv4_resident_scope::target_draft_pair }, 4, 3, true);
    expect(paired.status == llama_dsv4_resident_status::unsupported_components &&
               has(paired.required_components, LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT) &&
               has(paired.unsupported_components, LLAMA_DSV4_RESIDENT_PAIRED_CONTEXT),
           "single backend context claimed target/draft atomic ownership");

    const auto invalid =
        llama_dsv4_quote_resident_detach_layout({ 4, llama_dsv4_resident_scope::single_context }, 4, 99, true);
    expect(invalid.status == llama_dsv4_resident_status::invalid_sequence && invalid.detachable_components == 0,
           "invalid sequence exposed detachable ownership");

    const auto invalid_scope = llama_dsv4_quote_resident_detach_layout(
        { 2, static_cast<llama_dsv4_resident_scope>(UINT8_MAX) }, 4, 3, true);
    expect(invalid_scope.status == llama_dsv4_resident_status::invalid_scope &&
               invalid_scope.detachable_components == 0 &&
               invalid_scope.unsupported_components == invalid_scope.required_components,
           "invalid scope exposed detachable ownership");
}

}  // namespace

int main() {
    try {
        test_current_layout_fails_closed();
    } catch (const std::exception & error) {
        std::cerr << "test-dsv4-resident-contract: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-dsv4-resident-contract: all checks passed\n";
    return 0;
}

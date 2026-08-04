#include "llama-snapshot-metal-layout.h"

#include "llama-snapshot-store.h"

#include <algorithm>
#include <limits>
#include <new>
#include <utility>

const char * llama_snapshot_metal_layout_status_name(llama_snapshot_metal_layout_status status) {
    switch (status) {
        case llama_snapshot_metal_layout_status::ok:
            return "ok";
        case llama_snapshot_metal_layout_status::invalid_argument:
            return "invalid_argument";
        case llama_snapshot_metal_layout_status::too_many_regions:
            return "too_many_regions";
        case llama_snapshot_metal_layout_status::out_of_range:
            return "out_of_range";
        case llama_snapshot_metal_layout_status::output_too_small:
            return "output_too_small";
        case llama_snapshot_metal_layout_status::allocation_failed:
            return "allocation_failed";
    }
    return "unknown";
}

llama_snapshot_metal_layout_status llama_snapshot_metal_layout::build(
    const std::vector<llama_snapshot_metal_region> & input,
    llama_snapshot_metal_layout &                    output) noexcept {
    if (input.size() > LLAMA_SNAPSHOT_METAL_MAX_REGIONS) {
        return llama_snapshot_metal_layout_status::too_many_regions;
    }

    uint64_t expected_offset = 0;
    for (const llama_snapshot_metal_region & region : input) {
        if (region.bytes == 0 || region.logical_offset != expected_offset ||
            region.resource_offset > std::numeric_limits<uint64_t>::max() - region.bytes ||
            region.bytes > LLAMA_SNAPSHOT_MAX_BYTES || expected_offset > LLAMA_SNAPSHOT_MAX_BYTES - region.bytes) {
            return llama_snapshot_metal_layout_status::invalid_argument;
        }
        expected_offset += region.bytes;
    }

    try {
        llama_snapshot_metal_layout prepared;
        prepared.regions       = input;
        prepared.logical_bytes = expected_offset;
        output                 = std::move(prepared);
        return llama_snapshot_metal_layout_status::ok;
    } catch (const std::bad_alloc &) {
        return llama_snapshot_metal_layout_status::allocation_failed;
    } catch (...) {
        return llama_snapshot_metal_layout_status::invalid_argument;
    }
}

llama_snapshot_metal_map_result llama_snapshot_metal_layout::map(uint64_t                    logical_offset,
                                                                 uint64_t                    bytes,
                                                                 llama_snapshot_metal_span * spans,
                                                                 size_t span_capacity) const noexcept {
    llama_snapshot_metal_map_result result;
    if (logical_offset > logical_bytes || bytes > logical_bytes - logical_offset ||
        (span_capacity != 0 && spans == nullptr)) {
        result.status = llama_snapshot_metal_layout_status::out_of_range;
        return result;
    }
    if (bytes == 0) {
        result.status = llama_snapshot_metal_layout_status::ok;
        return result;
    }

    uint64_t remaining    = bytes;
    uint64_t cursor       = logical_offset;
    size_t   output_index = 0;
    for (const llama_snapshot_metal_region & region : regions) {
        const uint64_t region_end = region.logical_offset + region.bytes;
        if (cursor >= region_end) {
            continue;
        }
        if (cursor < region.logical_offset) {
            result.status = llama_snapshot_metal_layout_status::invalid_argument;
            return result;
        }

        const uint64_t within = cursor - region.logical_offset;
        const uint64_t count  = std::min<uint64_t>(remaining, region.bytes - within);
        if (output_index < span_capacity) {
            spans[output_index] = {
                region.resource_index,
                region.resource_offset + within,
                cursor - logical_offset,
                count,
            };
        }
        ++output_index;
        cursor += count;
        remaining -= count;
        if (remaining == 0) {
            result.required_spans = output_index;
            result.status         = output_index <= span_capacity ? llama_snapshot_metal_layout_status::ok :
                                                                    llama_snapshot_metal_layout_status::output_too_small;
            return result;
        }
    }

    result.status         = llama_snapshot_metal_layout_status::invalid_argument;
    result.required_spans = output_index;
    return result;
}

uint64_t llama_snapshot_metal_layout::total_bytes() const noexcept {
    return logical_bytes;
}

size_t llama_snapshot_metal_layout::region_count() const noexcept {
    return regions.size();
}

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr uint32_t LLAMA_SNAPSHOT_METAL_MAX_REGIONS = 4096;

enum class llama_snapshot_metal_layout_status : uint8_t {
    ok,
    invalid_argument,
    too_many_regions,
    out_of_range,
    output_too_small,
    allocation_failed,
};

const char * llama_snapshot_metal_layout_status_name(llama_snapshot_metal_layout_status status);

// One immutable runtime resource range in the serialized logical byte stream.
// Logical offsets must be contiguous and ascending. Resource indices are
// caller-defined bindings to concrete private Metal tensors.
struct llama_snapshot_metal_region {
    uint32_t resource_index  = 0;
    uint64_t logical_offset  = 0;
    uint64_t resource_offset = 0;
    uint64_t bytes           = 0;
};

// A mapped subrange of one resource. chunk_offset is relative to the requested
// chunk and therefore becomes the offset within one shared staging buffer.
struct llama_snapshot_metal_span {
    uint32_t resource_index  = 0;
    uint64_t resource_offset = 0;
    uint64_t chunk_offset    = 0;
    uint64_t bytes           = 0;
};

struct llama_snapshot_metal_map_result {
    llama_snapshot_metal_layout_status status         = llama_snapshot_metal_layout_status::invalid_argument;
    size_t                             required_spans = 0;
};

class llama_snapshot_metal_layout {
  public:
    static llama_snapshot_metal_layout_status build(const std::vector<llama_snapshot_metal_region> & regions,
                                                    llama_snapshot_metal_layout &                    output) noexcept;

    llama_snapshot_metal_map_result map(uint64_t                    logical_offset,
                                        uint64_t                    bytes,
                                        llama_snapshot_metal_span * spans,
                                        size_t                      span_capacity) const noexcept;

    uint64_t total_bytes() const noexcept;
    size_t   region_count() const noexcept;

  private:
    std::vector<llama_snapshot_metal_region> regions;
    uint64_t                                 logical_bytes = 0;
};

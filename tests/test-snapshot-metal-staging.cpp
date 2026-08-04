#include "llama-snapshot-metal-layout.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(GGML_TEST_METAL_SNAPSHOT_STAGING)
#    include "ggml-metal-snapshot.h"
#endif

namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

void expect_layout_status(llama_snapshot_metal_layout_status actual,
                          llama_snapshot_metal_layout_status expected,
                          const std::string &                message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_snapshot_metal_layout_status_name(expected) + ", got " +
             llama_snapshot_metal_layout_status_name(actual));
    }
}

void test_logical_layout_mapping() {
    const std::vector<llama_snapshot_metal_region> regions = {
        { 0, 0, 11, 3 },
        { 1, 3, 7,  5 },
        { 0, 8, 20, 4 },
    };
    llama_snapshot_metal_layout layout;
    expect_layout_status(llama_snapshot_metal_layout::build(regions, layout), llama_snapshot_metal_layout_status::ok,
                         "build logical layout");
    expect(layout.total_bytes() == 12 && layout.region_count() == 3, "logical layout bounds");

    std::array<llama_snapshot_metal_span, 3> spans{};
    const auto                               mapped = layout.map(2, 8, spans.data(), spans.size());
    expect_layout_status(mapped.status, llama_snapshot_metal_layout_status::ok, "map cross-resource chunk");
    expect(mapped.required_spans == 3, "cross-resource span count");
    expect(spans[0].resource_index == 0 && spans[0].resource_offset == 13 && spans[0].chunk_offset == 0 &&
               spans[0].bytes == 1,
           "first mapped span");
    expect(spans[1].resource_index == 1 && spans[1].resource_offset == 7 && spans[1].chunk_offset == 1 &&
               spans[1].bytes == 5,
           "second mapped span");
    expect(spans[2].resource_index == 0 && spans[2].resource_offset == 20 && spans[2].chunk_offset == 6 &&
               spans[2].bytes == 2,
           "third mapped span");

    const auto short_output = layout.map(2, 8, spans.data(), 2);
    expect_layout_status(short_output.status, llama_snapshot_metal_layout_status::output_too_small,
                         "short span output");
    expect(short_output.required_spans == 3, "required span count on short output");
    expect_layout_status(layout.map(11, 2, spans.data(), spans.size()).status,
                         llama_snapshot_metal_layout_status::out_of_range, "logical range overflow");
    expect_layout_status(layout.map(12, 0, nullptr, 0).status, llama_snapshot_metal_layout_status::ok,
                         "empty terminal mapping");

    llama_snapshot_metal_layout rejected;
    auto                        gap = regions;
    gap[1].logical_offset           = 4;
    expect_layout_status(llama_snapshot_metal_layout::build(gap, rejected),
                         llama_snapshot_metal_layout_status::invalid_argument, "layout gap");
    auto zero     = regions;
    zero[0].bytes = 0;
    expect_layout_status(llama_snapshot_metal_layout::build(zero, rejected),
                         llama_snapshot_metal_layout_status::invalid_argument, "zero-sized region");
    const std::vector<llama_snapshot_metal_region> resource_overflow = {
        { 0, 0, std::numeric_limits<uint64_t>::max(), 1 },
    };
    expect_layout_status(llama_snapshot_metal_layout::build(resource_overflow, rejected),
                         llama_snapshot_metal_layout_status::invalid_argument, "resource offset overflow");
    std::vector<llama_snapshot_metal_region> excessive(LLAMA_SNAPSHOT_METAL_MAX_REGIONS + 1);
    expect_layout_status(llama_snapshot_metal_layout::build(excessive, rejected),
                         llama_snapshot_metal_layout_status::too_many_regions, "region count bound");

    llama_snapshot_metal_layout empty;
    expect_layout_status(llama_snapshot_metal_layout::build({}, empty), llama_snapshot_metal_layout_status::ok,
                         "empty layout");
    expect(empty.total_bytes() == 0 && empty.region_count() == 0, "empty layout shape");
}

#if defined(GGML_TEST_METAL_SNAPSHOT_STAGING)

void expect_metal_status(enum ggml_metal_snapshot_status actual,
                         enum ggml_metal_snapshot_status expected,
                         const std::string &             message) {
    if (actual != expected) {
        fail(message + ": expected " + ggml_metal_snapshot_status_name(expected) + ", got " +
             ggml_metal_snapshot_status_name(actual));
    }
}

struct options {
    uint64_t bytes            = 5 * 64 * 1024 + 37;
    uint64_t chunk_bytes      = 64 * 1024;
    uint32_t iterations       = 1;
    bool     benchmark        = false;
    bool     require_m2_ultra = false;
};

uint64_t parse_u64(const char * value, const char * option) {
    char * end                      = nullptr;
    errno                           = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        fail(std::string("invalid value for ") + option);
    }
    return parsed;
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--benchmark") {
            result.benchmark = true;
        } else if (argument == "--require-m2-ultra") {
            result.require_m2_ultra = true;
        } else if ((argument == "--bytes" || argument == "--chunk-bytes" || argument == "--iterations") &&
                   index + 1 < argc) {
            const uint64_t value = parse_u64(argv[++index], argument.c_str());
            if (argument == "--bytes") {
                result.bytes = value;
            } else if (argument == "--chunk-bytes") {
                result.chunk_bytes = value;
            } else {
                expect(value <= UINT32_MAX, "iteration count exceeds uint32");
                result.iterations = static_cast<uint32_t>(value);
            }
        } else {
            fail("unknown or incomplete argument: " + argument);
        }
    }
    expect(result.bytes >= 3 && result.bytes <= SIZE_MAX, "benchmark bytes out of range");
    expect(result.chunk_bytes != 0 && result.chunk_bytes <= result.bytes && result.chunk_bytes <= SIZE_MAX,
           "benchmark chunk bytes out of range");
    expect(result.bytes / result.chunk_bytes + (result.bytes % result.chunk_bytes != 0) >= 3,
           "at least three chunks are required to prove two-slot backpressure");
    expect(result.iterations != 0, "benchmark iterations must be nonzero");
    return result;
}

struct metal_resource {
    ggml_metal_buffer_t buffer = nullptr;
    ggml_tensor *       tensor = nullptr;
};

struct active_transfer {
    ggml_metal_snapshot_token token{};
    uint64_t                  logical_offset = 0;
    size_t                    bytes          = 0;
};

struct mapped_copies {
    std::array<ggml_metal_snapshot_copy, 3> copies{};
    size_t                                  count = 0;

    const ggml_metal_snapshot_copy * data() const { return copies.data(); }

    ggml_metal_snapshot_copy * data() { return copies.data(); }

    size_t size() const { return count; }
};

mapped_copies map_copies(const llama_snapshot_metal_layout &   layout,
                         const std::array<metal_resource, 3> & resources,
                         uint64_t                              logical_offset,
                         size_t                                bytes) {
    std::array<llama_snapshot_metal_span, 3> spans{};
    const auto                               mapped = layout.map(logical_offset, bytes, spans.data(), spans.size());
    expect_layout_status(mapped.status, llama_snapshot_metal_layout_status::ok, "map Metal transfer");
    mapped_copies result;
    result.count = mapped.required_spans;
    for (size_t index = 0; index < mapped.required_spans; ++index) {
        expect(spans[index].resource_index < resources.size(), "mapped resource index");
        const metal_resource & resource = resources[spans[index].resource_index];
        expect(spans[index].resource_offset <= SIZE_MAX && spans[index].chunk_offset <= SIZE_MAX &&
                   spans[index].bytes <= SIZE_MAX,
               "mapped Metal span exceeds size_t");
        result.copies[index] = {
            resource.buffer,
            resource.tensor,
            static_cast<size_t>(spans[index].resource_offset),
            static_cast<size_t>(spans[index].chunk_offset),
            static_cast<size_t>(spans[index].bytes),
        };
    }
    return result;
}

void run_readback_pipeline(ggml_metal_snapshot_staging_t         staging,
                           const llama_snapshot_metal_layout &   layout,
                           const std::array<metal_resource, 3> & resources,
                           size_t                                chunk_bytes,
                           std::vector<uint8_t> &                output) {
    const uint64_t total_bytes = layout.total_bytes();
    output.resize(static_cast<size_t>(total_bytes));
    std::vector<active_transfer> active;
    active.reserve(GGML_METAL_SNAPSHOT_STAGING_SLOTS);
    uint64_t   submitted             = 0;
    uint64_t   completed             = 0;
    bool       observed_backpressure = false;
    const auto deadline              = std::chrono::steady_clock::now() + std::chrono::seconds(120);

    while (completed < total_bytes) {
        while (submitted < total_bytes) {
            const size_t bytes  = static_cast<size_t>(std::min<uint64_t>(chunk_bytes, total_bytes - submitted));
            const auto   copies = map_copies(layout, resources, submitted, bytes);
            ggml_metal_snapshot_token token{};
            const auto status = ggml_metal_snapshot_submit_readback(staging, copies.data(), copies.size(), &token);
            if (status == GGML_METAL_SNAPSHOT_STATUS_WOULD_BLOCK) {
                observed_backpressure = true;
                break;
            }
            expect_metal_status(status, GGML_METAL_SNAPSHOT_STATUS_OK, "submit private readback");
            active.push_back({ token, submitted, bytes });
            submitted += bytes;
        }

        bool made_progress = false;
        for (size_t index = 0; index < active.size();) {
            ggml_metal_snapshot_poll_result poll{};
            const auto                      status = ggml_metal_snapshot_poll(staging, active[index].token, &poll);
            if (status == GGML_METAL_SNAPSHOT_STATUS_NOT_READY) {
                ++index;
                continue;
            }
            expect_metal_status(status, GGML_METAL_SNAPSHOT_STATUS_OK, "poll private readback");
            expect(poll.direction == GGML_METAL_SNAPSHOT_DIRECTION_READBACK && poll.data != nullptr &&
                       poll.size == active[index].bytes,
                   "private readback completion metadata");
            std::memcpy(output.data() + active[index].logical_offset, poll.data, poll.size);
            expect_metal_status(ggml_metal_snapshot_release(staging, active[index].token),
                                GGML_METAL_SNAPSHOT_STATUS_OK, "release readback slot");
            expect_metal_status(ggml_metal_snapshot_poll(staging, active[index].token, &poll),
                                GGML_METAL_SNAPSHOT_STATUS_STALE_TRANSFER, "released readback token");
            completed += active[index].bytes;
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(index));
            made_progress = true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            fail("private readback pipeline timed out");
        }
        if (!made_progress) {
            std::this_thread::yield();
        }
    }
    expect(observed_backpressure, "readback never proved two-slot backpressure");
}

void run_restore_pipeline(ggml_metal_snapshot_staging_t         staging,
                          const llama_snapshot_metal_layout &   layout,
                          const std::array<metal_resource, 3> & resources,
                          size_t                                chunk_bytes,
                          const std::vector<uint8_t> &          source) {
    const uint64_t total_bytes = layout.total_bytes();
    expect(source.size() == total_bytes, "restore source size");
    std::vector<active_transfer> active;
    active.reserve(GGML_METAL_SNAPSHOT_STAGING_SLOTS);
    uint64_t   submitted             = 0;
    uint64_t   completed             = 0;
    bool       observed_backpressure = false;
    const auto deadline              = std::chrono::steady_clock::now() + std::chrono::seconds(120);

    while (completed < total_bytes) {
        while (submitted < total_bytes) {
            const size_t bytes  = static_cast<size_t>(std::min<uint64_t>(chunk_bytes, total_bytes - submitted));
            const auto   copies = map_copies(layout, resources, submitted, bytes);
            ggml_metal_snapshot_token token{};
            const auto status = ggml_metal_snapshot_submit_restore(staging, source.data() + submitted, bytes,
                                                                   copies.data(), copies.size(), &token);
            if (status == GGML_METAL_SNAPSHOT_STATUS_WOULD_BLOCK) {
                observed_backpressure = true;
                break;
            }
            expect_metal_status(status, GGML_METAL_SNAPSHOT_STATUS_OK, "submit private restore");
            active.push_back({ token, submitted, bytes });
            submitted += bytes;
        }

        bool made_progress = false;
        for (size_t index = 0; index < active.size();) {
            ggml_metal_snapshot_poll_result poll{};
            const auto                      status = ggml_metal_snapshot_poll(staging, active[index].token, &poll);
            if (status == GGML_METAL_SNAPSHOT_STATUS_NOT_READY) {
                ++index;
                continue;
            }
            expect_metal_status(status, GGML_METAL_SNAPSHOT_STATUS_OK, "poll private restore");
            expect(poll.direction == GGML_METAL_SNAPSHOT_DIRECTION_RESTORE && poll.data == nullptr &&
                       poll.size == active[index].bytes,
                   "private restore completion metadata");
            expect_metal_status(ggml_metal_snapshot_release(staging, active[index].token),
                                GGML_METAL_SNAPSHOT_STATUS_OK, "release restore slot");
            completed += active[index].bytes;
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(index));
            made_progress = true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            fail("private restore pipeline timed out");
        }
        if (!made_progress) {
            std::this_thread::yield();
        }
    }
    expect(observed_backpressure, "restore never proved two-slot backpressure");
}

void run_metal_gate(const options & run_options) {
    ggml_metal_device_t device = ggml_metal_device_get(0);
    if (device == nullptr) {
        if (run_options.require_m2_ultra) {
            fail("required M2 Ultra Metal device is unavailable");
        }
        std::puts("snapshot Metal staging test skipped: no Metal device");
        return;
    }
    const ggml_metal_device_props * properties = ggml_metal_device_get_props(device);
    if (run_options.require_m2_ultra && properties->device_id != GGML_METAL_DEVICE_M2_ULTRA) {
        fail("Metal device is not the required M2 Ultra");
    }
    expect(properties->has_unified_memory, "snapshot staging requires unified memory");

    const uint64_t                           region0          = run_options.bytes / 3;
    const uint64_t                           region1          = (run_options.bytes - region0) / 2;
    const uint64_t                           region2          = run_options.bytes - region0 - region1;
    const std::array<uint64_t, 3>            region_bytes     = { region0, region1, region2 };
    const std::array<uint64_t, 3>            resource_offsets = { 17, 31, 47 };
    std::vector<llama_snapshot_metal_region> regions;
    uint64_t                                 logical_offset = 0;
    for (uint32_t index = 0; index < region_bytes.size(); ++index) {
        regions.push_back({ index, logical_offset, resource_offsets[index], region_bytes[index] });
        logical_offset += region_bytes[index];
    }
    llama_snapshot_metal_layout layout;
    expect_layout_status(llama_snapshot_metal_layout::build(regions, layout), llama_snapshot_metal_layout_status::ok,
                         "build target Metal layout");

    ggml_init_params params = {
        /*.mem_size   =*/4 * ggml_tensor_overhead(),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * context = ggml_init(params);
    expect(context != nullptr, "create Metal test tensor context");
    std::array<metal_resource, 3>       resources{};
    std::array<std::vector<uint8_t>, 3> initial;
    for (uint32_t index = 0; index < resources.size(); ++index) {
        const size_t tensor_bytes = static_cast<size_t>(resource_offsets[index] + region_bytes[index]);
        resources[index].buffer   = ggml_metal_buffer_init(device, tensor_bytes, false);
        expect(resources[index].buffer != nullptr, "allocate private Metal resource");
        resources[index].tensor       = ggml_new_tensor_1d(context, GGML_TYPE_I8, tensor_bytes);
        resources[index].tensor->data = ggml_metal_buffer_get_base(resources[index].buffer);
        initial[index].assign(tensor_bytes, static_cast<uint8_t>(0xa0 + index));
        for (size_t byte = 0; byte < region_bytes[index]; ++byte) {
            initial[index][static_cast<size_t>(resource_offsets[index]) + byte] =
                static_cast<uint8_t>(11 + index * 29 + byte * 37 + byte / 7);
        }
        ggml_metal_buffer_set_tensor(resources[index].buffer, resources[index].tensor, initial[index].data(), 0,
                                     tensor_bytes);
    }

    std::vector<uint8_t> expected(static_cast<size_t>(run_options.bytes));
    logical_offset = 0;
    for (uint32_t index = 0; index < resources.size(); ++index) {
        std::copy_n(initial[index].data() + resource_offsets[index], static_cast<size_t>(region_bytes[index]),
                    expected.data() + logical_offset);
        logical_offset += region_bytes[index];
    }

    ggml_metal_snapshot_staging_t staging =
        ggml_metal_snapshot_staging_init(device, static_cast<size_t>(run_options.chunk_bytes));
    expect(staging != nullptr, "allocate two shared Metal staging slots");

    const size_t first_chunk_bytes =
        static_cast<size_t>(std::min<uint64_t>(run_options.chunk_bytes, layout.total_bytes()));
    auto invalid_copies                          = map_copies(layout, resources, 0, first_chunk_bytes);
    invalid_copies.copies.front().staging_offset = 1;
    ggml_metal_snapshot_token invalid_token{};
    expect_metal_status(
        ggml_metal_snapshot_submit_readback(staging, invalid_copies.data(), invalid_copies.size(), &invalid_token),
        GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT, "reject staging gap");
    const auto valid_copies = map_copies(layout, resources, 0, first_chunk_bytes);
    expect_metal_status(ggml_metal_snapshot_submit_restore(staging, expected.data(), first_chunk_bytes - 1,
                                                           valid_copies.data(), valid_copies.size(), &invalid_token),
                        GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT, "reject restore size mismatch");
    expect(ggml_metal_snapshot_get_stats(staging).slots_in_use == 0, "invalid transfer occupied staging slot");

    std::vector<uint8_t> snapshot;
    run_readback_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), snapshot);
    expect(snapshot == expected, "private-to-shared snapshot bytes");
    for (const metal_resource & resource : resources) {
        ggml_metal_buffer_clear(resource.buffer, 0);
    }
    run_restore_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), snapshot);
    std::vector<uint8_t> restored;
    run_readback_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), restored);
    expect(restored == expected, "shared-to-private restored bytes");

    double readback_seconds = 0;
    double restore_seconds  = 0;
    if (run_options.benchmark) {
        auto start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < run_options.iterations; ++iteration) {
            run_readback_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), snapshot);
        }
        readback_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        expect(snapshot == expected, "benchmark readback bytes");
        start = std::chrono::steady_clock::now();
        for (uint32_t iteration = 0; iteration < run_options.iterations; ++iteration) {
            run_restore_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), snapshot);
        }
        restore_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        run_readback_pipeline(staging, layout, resources, static_cast<size_t>(run_options.chunk_bytes), restored);
        expect(restored == expected, "benchmark restored bytes");
    }

    const ggml_metal_snapshot_stats stats = ggml_metal_snapshot_get_stats(staging);
    expect(stats.staging_bytes == run_options.chunk_bytes && stats.slot_high_watermark == 2 &&
               stats.slots_in_use == 0 && stats.failed_transfers == 0 && stats.would_block_count != 0,
           "Metal staging accounting");
    if (run_options.benchmark) {
        const double gib =
            static_cast<double>(run_options.bytes) * run_options.iterations / static_cast<double>(uint64_t{ 1 } << 30);
        std::printf(
            "snapshot_metal_benchmark={\"device\":\"%s\",\"bytes\":%llu,\"chunk_bytes\":%llu,"
            "\"iterations\":%u,\"slots\":2,\"readback_seconds\":%.9f,\"readback_gib_s\":%.6f,"
            "\"restore_seconds\":%.9f,\"restore_gib_s\":%.6f,\"would_block\":%llu}\n",
            properties->name, static_cast<unsigned long long>(run_options.bytes),
            static_cast<unsigned long long>(run_options.chunk_bytes), run_options.iterations, readback_seconds,
            gib / readback_seconds, restore_seconds, gib / restore_seconds,
            static_cast<unsigned long long>(stats.would_block_count));
    }

    ggml_metal_snapshot_staging_free(staging);
    for (metal_resource & resource : resources) {
        ggml_metal_buffer_free(resource.buffer);
    }
    ggml_free(context);
}

#endif

}  // namespace

int main(int argc, char ** argv) {
    try {
        test_logical_layout_mapping();
#if defined(GGML_TEST_METAL_SNAPSHOT_STAGING)
        run_metal_gate(parse_options(argc, argv));
#else
        (void) argc;
        (void) argv;
        std::puts("snapshot Metal staging test skipped: Metal backend not built");
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-snapshot-metal-staging: %s\n", error.what());
        return 1;
    }
    std::puts("test-snapshot-metal-staging: all checks passed");
    return 0;
}

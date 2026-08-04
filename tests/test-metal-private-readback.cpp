#include <cstdio>

#if defined(GGML_TEST_METAL_PRIVATE_READBACK)

#include "ggml-metal-device.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#undef assert
#define assert(expr) do {                                                        \
    if (!(expr)) {                                                               \
        std::fprintf(stderr, "check failed at %s:%d: %s\n",                    \
                __FILE__, __LINE__, #expr);                                      \
        std::abort();                                                            \
    }                                                                            \
} while (false)

int main() {
    ggml_metal_device_t device = ggml_metal_device_get(0);
    if (device == nullptr) {
        std::puts("metal private readback test skipped: no Metal device");
        return 0;
    }

    const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
    assert(page_size > 0 && page_size % 32 == 0);

    const size_t tensor_size = 64*page_size;
    ggml_metal_buffer_t buffer = ggml_metal_buffer_init(device, tensor_size, false);
    assert(buffer != nullptr);

    ggml_init_params params = {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, tensor_size);
    tensor->data = ggml_metal_buffer_get_base(buffer);

    void * source_raw = nullptr;
    void * destination_raw = nullptr;
    assert(posix_memalign(&source_raw, page_size, tensor_size) == 0);
    assert(posix_memalign(&destination_raw, page_size, tensor_size + page_size) == 0);

    auto * source = static_cast<uint8_t *>(source_raw);
    auto * destination_base = static_cast<uint8_t *>(destination_raw);

    constexpr uint8_t sentinel = 0xa5;
    const size_t destination_offset = 5*page_size/32;
    uint8_t * destination = destination_base + destination_offset;
    assert((uintptr_t) destination % page_size == destination_offset);

    // Exercise manual Objective-C ownership of both temporary no-copy wrappers
    // with 1 GiB of cumulative traffic while keeping live storage
    // bounded to a few MiB.
    constexpr int iterations = 512;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (size_t i = 0; i < tensor_size; ++i) {
            source[i] = (uint8_t) (11 + 37*i + iteration);
        }
        std::memset(destination_base, sentinel, tensor_size + page_size);

        ggml_metal_buffer_set_tensor(buffer, tensor, source, 0, tensor_size);
        ggml_metal_buffer_get_tensor(buffer, tensor, destination, 0, tensor_size);

        for (size_t i = 0; i < destination_offset; ++i) {
            assert(destination_base[i] == sentinel);
        }
        assert(std::memcmp(destination, source, tensor_size) == 0);
        for (size_t i = destination_offset + tensor_size; i < tensor_size + page_size; ++i) {
            assert(destination_base[i] == sentinel);
        }
    }

    std::free(destination_raw);
    std::free(source_raw);
    ggml_free(ctx);
    ggml_metal_buffer_free(buffer);

    std::puts("metal private unaligned readback test passed");
    return 0;
}

#else

int main() {
    std::puts("metal private readback test skipped: Metal backend not built");
    return 0;
}

#endif

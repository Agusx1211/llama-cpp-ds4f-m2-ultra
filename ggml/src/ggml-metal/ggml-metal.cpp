#include "ggml-metal.h"

#include "ggml-dsv4-sparse.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"

#include <mutex>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

static_assert((int) GGML_DSV4_SPARSE_OK          == (int) GGML_METAL_SPARSE_RESERVATION_OK);
static_assert((int) GGML_DSV4_SPARSE_PRESSURE    == (int) GGML_METAL_SPARSE_RESERVATION_PRESSURE);
static_assert((int) GGML_DSV4_SPARSE_STALE       == (int) GGML_METAL_SPARSE_RESERVATION_STALE);
static_assert((int) GGML_DSV4_SPARSE_INVALID     == (int) GGML_METAL_SPARSE_RESERVATION_INVALID);
static_assert((int) GGML_DSV4_SPARSE_OOM         == (int) GGML_METAL_SPARSE_RESERVATION_OOM);
static_assert((int) GGML_DSV4_SPARSE_UNSUPPORTED == (int) GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED);

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

// forward declaration
static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);
static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ NULL,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ NULL,
};

static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    return buffer != nullptr &&
          (buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer);
}

// Target-specific unified-memory escape hatch for the production DSV4 AMX
// experiment. Keep Metal buffer types non-host to preserve scheduler transfer
// and dependency semantics; callers must opt in and validate each tensor here.
static void * ggml_backend_metal_get_tensor_host_ptr(const ggml_tensor * tensor) {
    if (tensor == NULL || tensor->buffer == NULL || tensor->data == NULL ||
        !ggml_backend_buffer_is_metal(tensor->buffer)) {
        return NULL;
    }

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) tensor->buffer->context;
    if (!ggml_metal_buffer_is_shared(ctx)) {
        return NULL;
    }

    return tensor->data;
}

static bool ggml_backend_metal_buffer_type_is_shared(ggml_backend_buffer_type_t buft) {
    return buft != NULL && buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name;
}

//
// buffer types
//

struct ggml_backend_metal_buffer_type {
    int device;
    std::string name;
    uint32_t sparse_divisor = 1;
};

struct ggml_backend_metal_buffer_type_deleter {
    void operator()(ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);
    if (res == nullptr) {
        const auto * props = ggml_metal_device_get_props(ctx_dev);
        GGML_LOG_ERROR("%s: Metal %s buffer allocation failed: requested=%zu bytes (%.2f MiB), maxBufferLength=%zu bytes (%.2f MiB)\n",
                __func__, shared ? "shared" : "private", size, size/1024.0/1024.0,
                props->max_buffer_size, props->max_buffer_size/1024.0/1024.0);
        return nullptr;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
                res += ggml_metal_op_mul_mat_id_extra_work(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                new ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// DSV4 placement-sparse buffer type. The virtual allocation retains one full
// affine cache stream per configured sequence. A heap sized for one aggregate
// context (plus page-granularity scratch slack) backs only pages touched by
// active sequences.

static const char * ggml_backend_metal_buffer_type_dsv4_sparse_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_dsv4_sparse_alloc_buffer(
        ggml_backend_buffer_type_t buft,
        size_t size) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *) buft->context;
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) buft->device->context;

    const size_t page = 64*1024;
    // Each DSV4 compressed-cache family spans 20 layer tensors. A fresh
    // virtual stream touches at least one sparse page in every tensor, even
    // when its byte-proportional share is much smaller than a page.
    const size_t slack = std::max<size_t>(32*1024*1024,
            (size_t) ctx->sparse_divisor*20*page);
    const size_t aggregate = std::min(size,
            GGML_PAD((size + ctx->sparse_divisor - 1)/ctx->sparse_divisor, page));
    const size_t physical = aggregate + std::min(slack, size - aggregate);

    ggml_metal_buffer_t res = ggml_metal_buffer_init_sparse(ctx_dev, size, physical);
    if (res == NULL) {
        return NULL;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_metal_buffer_private_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_dsv4_sparse_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64*1024;
}

static size_t ggml_backend_metal_buffer_type_dsv4_sparse_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) buft->device->context;
    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_dsv4_sparse_get_alloc_size(
        ggml_backend_buffer_type_t buft,
        const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_dsv4_sparse_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

struct ggml_backend_metal_sparse_buft_bundle {
    ggml_backend_metal_buffer_type_ptr context;
    ggml_backend_buffer_type buft;
};

static ggml_backend_buffer_type_t ggml_backend_metal_dsv4_sparse_buffer_type(
        ggml_backend_dev_t dev,
        uint32_t n_stream) {
    if (dev == NULL || n_stream == 0) {
        return NULL;
    }

    ggml_metal_device_t ctx_dev = (ggml_metal_device_t) dev->context;
    const ggml_metal_device_props * props = ggml_metal_device_get_props(ctx_dev);
    if (!props->has_placement_sparse || props->device_id != GGML_METAL_DEVICE_M2_ULTRA) {
        return NULL;
    }

    static std::mutex mutex;
    static std::map<std::pair<int, uint32_t>, std::unique_ptr<ggml_backend_metal_sparse_buft_bundle>> bundles;
    std::lock_guard<std::mutex> lock(mutex);

    const auto key = std::make_pair(props->device, n_stream);
    auto it = bundles.find(key);
    if (it != bundles.end()) {
        return &it->second->buft;
    }

    auto bundle = std::make_unique<ggml_backend_metal_sparse_buft_bundle>();
    bundle->context.reset(new ggml_backend_metal_buffer_type {
        /*.device          =*/ props->device,
        /*.name            =*/ GGML_METAL_NAME + std::to_string(props->device) + "_DSV4Sparse" + std::to_string(n_stream),
        /*.sparse_divisor  =*/ n_stream,
    });
    bundle->buft = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_metal_buffer_type_dsv4_sparse_get_name,
            /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_dsv4_sparse_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_metal_buffer_type_dsv4_sparse_get_alignment,
            /* .get_max_size     = */ ggml_backend_metal_buffer_type_dsv4_sparse_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_dsv4_sparse_get_alloc_size,
            /* .is_host          = */ ggml_backend_metal_buffer_type_dsv4_sparse_is_host,
        },
        /* .device  = */ dev,
        /* .context = */ bundle->context.get(),
    };

    ggml_backend_buffer_type_t result = &bundle->buft;
    bundles.emplace(key, std::move(bundle));
    return result;
}

static bool ggml_backend_metal_dsv4_sparse_map_tensor_ranges(
        ggml_tensor * const * tensors,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges) {
    if (n_ranges == 0) {
        return true;
    }

    ggml_backend_buffer_t backend_buffer = tensors[0]->buffer;
    if (backend_buffer == NULL || !ggml_backend_buffer_is_metal(backend_buffer)) {
        return true;
    }
    ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) backend_buffer->context;
    if (!ggml_metal_buffer_is_sparse(buffer)) {
        return true;
    }

    std::vector<size_t> absolute_offsets(n_ranges);
    for (size_t i = 0; i < n_ranges; ++i) {
        if (tensors[i] == NULL || tensors[i]->buffer != backend_buffer) {
            return false;
        }
        const ggml_metal_buffer_id bid = ggml_metal_buffer_get_id(buffer, tensors[i]);
        if (bid.metal == NULL || offsets[i] > ggml_nbytes(tensors[i]) ||
                sizes[i] > ggml_nbytes(tensors[i]) - offsets[i]) {
            return false;
        }
        absolute_offsets[i] = bid.offs + offsets[i];
    }

    return ggml_metal_buffer_sparse_map_write(
            buffer, absolute_offsets.data(), sizes, n_ranges);
}

// Returns 0 for a non-sparse tensor, 1 on success, and -1 on an error.
static int ggml_backend_metal_dsv4_sparse_tensor_usage(
        const ggml_tensor * tensor,
        ggml_metal_sparse_usage * usage) {
    if (tensor == nullptr || tensor->buffer == nullptr || usage == nullptr ||
            !ggml_backend_buffer_is_metal(tensor->buffer)) {
        return 0;
    }
    ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) tensor->buffer->context;
    if (!ggml_metal_buffer_is_sparse(buffer)) {
        return 0;
    }
    return ggml_metal_buffer_sparse_get_usage(buffer, usage) ? 1 : -1;
}

static ggml_metal_sparse_reservation_result ggml_backend_metal_dsv4_sparse_prepare_tensor_ranges(
        ggml_tensor * const * tensors,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges,
        ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool,
        ggml_metal_sparse_reservation_t * reservation) {
    if (tensors == nullptr || offsets == nullptr || sizes == nullptr || n_ranges == 0) {
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }

    std::vector<ggml_metal_sparse_buffer_range> absolute;
    absolute.reserve(n_ranges);
    for (size_t i = 0; i < n_ranges; ++i) {
        ggml_tensor * tensor = tensors[i];
        if (tensor == nullptr || tensor->buffer == nullptr ||
                offsets[i] > ggml_nbytes(tensor) ||
                sizes[i] > ggml_nbytes(tensor) - offsets[i]) {
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        if (!ggml_backend_buffer_is_metal(tensor->buffer)) {
            continue;
        }
        ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) tensor->buffer->context;
        if (!ggml_metal_buffer_is_sparse(buffer)) {
            continue;
        }
        const ggml_metal_buffer_id bid = ggml_metal_buffer_get_id(buffer, tensor);
        if (bid.metal == nullptr) {
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        absolute.push_back({ buffer, bid.offs + offsets[i], sizes[i] });
    }
    if (absolute.empty()) {
        // Registry procedures are available to ordinary Metal buffers too.
        // A range set containing no placement-sparse storage needs no ticket
        // and is a successful no-op, including reserve calls.
        if (n_pools != nullptr) {
            *n_pools = 0;
        }
        if (limiting_pool != nullptr) {
            *limiting_pool = SIZE_MAX;
        }
        if (reservation != nullptr) {
            *reservation = nullptr;
        }
        return GGML_METAL_SPARSE_RESERVATION_OK;
    }

    if (reservation != nullptr) {
        return ggml_metal_buffers_sparse_reserve(
                absolute.data(), absolute.size(), pools, pool_capacity,
                n_pools, limiting_pool, reservation);
    }
    return ggml_metal_buffers_sparse_quote(
            absolute.data(), absolute.size(), pools, pool_capacity,
            n_pools, limiting_pool);
}

static ggml_metal_sparse_reservation_result ggml_backend_metal_dsv4_sparse_quote_tensor_ranges(
        ggml_tensor * const * tensors,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges,
        ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool) {
    return ggml_backend_metal_dsv4_sparse_prepare_tensor_ranges(
            tensors, offsets, sizes, n_ranges, pools, pool_capacity,
            n_pools, limiting_pool, nullptr);
}

static ggml_metal_sparse_reservation_result ggml_backend_metal_dsv4_sparse_reserve_tensor_ranges(
        ggml_tensor * const * tensors,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges,
        ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool,
        void ** reservation) {
    ggml_metal_sparse_reservation_t ticket = nullptr;
    const auto result = ggml_backend_metal_dsv4_sparse_prepare_tensor_ranges(
            tensors, offsets, sizes, n_ranges, pools, pool_capacity,
            n_pools, limiting_pool, &ticket);
    if (reservation != nullptr) {
        *reservation = ticket;
    } else if (ticket != nullptr) {
        ggml_metal_sparse_reservation_free(ticket);
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }
    return result;
}

static ggml_metal_sparse_reservation_result ggml_backend_metal_dsv4_sparse_reservation_commit(
        void * reservation) {
    return ggml_metal_sparse_reservation_commit(
            (ggml_metal_sparse_reservation_t) reservation);
}

static bool ggml_backend_metal_dsv4_sparse_reservation_rollback(void * reservation) {
    return ggml_metal_sparse_reservation_rollback(
            (ggml_metal_sparse_reservation_t) reservation);
}

static bool ggml_backend_metal_dsv4_sparse_reservation_cancel(void * reservation) {
    return ggml_metal_sparse_reservation_cancel(
            (ggml_metal_sparse_reservation_t) reservation);
}

static void ggml_backend_metal_dsv4_sparse_reservation_free(void * reservation) {
    ggml_metal_sparse_reservation_free((ggml_metal_sparse_reservation_t) reservation);
}

enum ggml_backend_metal_dsv4_sparse_move_test_failure {
    GGML_METAL_DSV4_MOVE_FAIL_NONE = -1,
    GGML_METAL_DSV4_MOVE_FAIL_RESERVE_ALLOC,
    GGML_METAL_DSV4_MOVE_FAIL_PUSH_ALLOC,
    GGML_METAL_DSV4_MOVE_FAIL_PUBLISH_ALLOC,
    GGML_METAL_DSV4_MOVE_FAIL_LENGTH,
    GGML_METAL_DSV4_MOVE_FAIL_PUBLISH_UNEXPECTED,
};

static thread_local int ggml_backend_metal_dsv4_sparse_move_fail_stage =
        GGML_METAL_DSV4_MOVE_FAIL_NONE;

// Test stages 0/1/2 inject allocation failures at reserve/push/publication;
// stage 3 injects length_error at reserve and stage 4 an unexpected exception
// after the backend quote exists, exercising the final cleanup barrier.
static void ggml_backend_metal_dsv4_sparse_move_fail_for_test(int stage) {
    ggml_backend_metal_dsv4_sparse_move_fail_stage = stage;
}

static void ggml_backend_metal_dsv4_sparse_move_maybe_fail(int point) {
    const int failure = ggml_backend_metal_dsv4_sparse_move_fail_stage;
    if (failure == point ||
            (failure == GGML_METAL_DSV4_MOVE_FAIL_LENGTH &&
             point == GGML_METAL_DSV4_MOVE_FAIL_RESERVE_ALLOC) ||
            (failure == GGML_METAL_DSV4_MOVE_FAIL_PUBLISH_UNEXPECTED &&
             point == GGML_METAL_DSV4_MOVE_FAIL_PUBLISH_ALLOC)) {
        ggml_backend_metal_dsv4_sparse_move_fail_stage = GGML_METAL_DSV4_MOVE_FAIL_NONE;
        if (failure == GGML_METAL_DSV4_MOVE_FAIL_LENGTH) {
            throw std::length_error("injected sparse move length failure");
        }
        if (failure == GGML_METAL_DSV4_MOVE_FAIL_PUBLISH_UNEXPECTED) {
            throw std::runtime_error("injected sparse move unexpected failure");
        }
        throw std::bad_alloc();
    }
}

static int ggml_backend_metal_dsv4_sparse_move_tensor_rows_quote(
        ggml_tensor * const * sources,
        ggml_tensor * const * destinations,
        size_t n_tensors,
        void ** quote) {
    if (quote != nullptr) {
        *quote = nullptr;
    }
    if (sources == nullptr || n_tensors == 0 || quote == nullptr) {
        return GGML_DSV4_SPARSE_INVALID;
    }

    using move_vector = std::vector<ggml_metal_sparse_buffer_move>;
    move_vector moves;
    if (n_tensors > SIZE_MAX / sizeof(move_vector::value_type) || n_tensors > moves.max_size()) {
        return GGML_DSV4_SPARSE_INVALID;
    }

    ggml_metal_sparse_move_t result = nullptr;
    const auto fail = [&](int status) {
        if (result != nullptr) {
            ggml_metal_sparse_move_free(result);
            result = nullptr;
        }
        *quote = nullptr;
        return status;
    };
    try {
        ggml_backend_metal_dsv4_sparse_move_maybe_fail(0);
        moves.reserve(n_tensors);
        for (size_t i = 0; i < n_tensors; ++i) {
            ggml_tensor * source = sources[i];
            ggml_tensor * destination = destinations != nullptr ? destinations[i] : nullptr;
            if (source == nullptr || source->buffer == nullptr ||
                    !ggml_backend_buffer_is_metal(source->buffer) ||
                    (destination != nullptr &&
                     (destination->buffer != source->buffer ||
                      ggml_nbytes(destination) != ggml_nbytes(source)))) {
                return GGML_DSV4_SPARSE_INVALID;
            }
            ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) source->buffer->context;
            if (!ggml_metal_buffer_is_sparse(buffer)) {
                return GGML_DSV4_SPARSE_UNSUPPORTED;
            }
            const ggml_metal_buffer_id source_id = ggml_metal_buffer_get_id(buffer, source);
            ggml_metal_sparse_buffer_range destination_range = {};
            if (destination != nullptr) {
                const ggml_metal_buffer_id destination_id = ggml_metal_buffer_get_id(buffer, destination);
                destination_range = { buffer, destination_id.offs, ggml_nbytes(destination) };
            }
            ggml_backend_metal_dsv4_sparse_move_maybe_fail(1);
            moves.push_back({
                { buffer, source_id.offs, ggml_nbytes(source) },
                destination_range,
            });
        }

        const auto status = ggml_metal_buffers_sparse_move_quote(
                moves.data(), moves.size(), &result);
        if (status == GGML_METAL_SPARSE_RESERVATION_OK && result != nullptr) {
            ggml_backend_metal_dsv4_sparse_move_maybe_fail(2);
            *quote = result;
            return GGML_DSV4_SPARSE_OK;
        }
        return fail(status == GGML_METAL_SPARSE_RESERVATION_OK ?
                GGML_DSV4_SPARSE_INVALID : (int) status);
    } catch (const std::length_error &) {
        return fail(GGML_DSV4_SPARSE_INVALID);
    } catch (const std::bad_array_new_length &) {
        return fail(GGML_DSV4_SPARSE_INVALID);
    } catch (const std::bad_alloc &) {
        return fail(GGML_DSV4_SPARSE_OOM);
    } catch (...) {
        return fail(GGML_DSV4_SPARSE_INVALID);
    }
}

static int ggml_backend_metal_dsv4_sparse_move_tensor_rows_commit(
        void * quote) {
    return (int) ggml_metal_sparse_move_commit((ggml_metal_sparse_move_t) quote);
}

static void ggml_backend_metal_dsv4_sparse_move_tensor_rows_free(void * quote) {
    ggml_metal_sparse_move_free((ggml_metal_sparse_move_t) quote);
}

// Returns 0 for a non-sparse tensor, 1 on success, and -1 on a sparse error.
static int ggml_backend_metal_dsv4_sparse_alias_tensor_rows(
        const ggml_tensor * src,
        ggml_tensor * dst,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges) {
    if (src == NULL || dst == NULL || src->buffer == NULL || src->buffer != dst->buffer ||
            !ggml_backend_buffer_is_metal(src->buffer)) {
        return 0;
    }

    ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) src->buffer->context;
    if (!ggml_metal_buffer_is_sparse(buffer)) {
        return 0;
    }
    if (ggml_nbytes(src) != ggml_nbytes(dst)) {
        return -1;
    }

    const ggml_metal_buffer_id src_bid = ggml_metal_buffer_get_id(buffer, src);
    const ggml_metal_buffer_id dst_bid = ggml_metal_buffer_get_id(buffer, dst);
    const bool ok = ggml_metal_buffer_sparse_alias(
            buffer, src_bid.offs, dst_bid.offs, ggml_nbytes(src),
            offsets, sizes, n_ranges);
    return ok ? 1 : -1;
}

// Returns 0 for a non-sparse tensor, 1 on success, and -1 on a sparse error.
static int ggml_backend_metal_dsv4_sparse_unmap_tensor_range(
        ggml_tensor * tensor,
        size_t offset,
        size_t size) {
    if (tensor == NULL || tensor->buffer == NULL || !ggml_backend_buffer_is_metal(tensor->buffer)) {
        return 0;
    }

    ggml_metal_buffer_t buffer = (ggml_metal_buffer_t) tensor->buffer->context;
    if (!ggml_metal_buffer_is_sparse(buffer)) {
        return 0;
    }

    const ggml_metal_buffer_id bid = ggml_metal_buffer_get_id(buffer, tensor);
    if (offset > ggml_nbytes(tensor) || size > ggml_nbytes(tensor) - offset) {
        return -1;
    }
    const bool ok = ggml_metal_buffer_sparse_unmap(buffer, bid.offs + offset, size);
    return ok ? 1 : -1;
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name         = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer     = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .get_alignment    = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size     = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size   = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .is_host          = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_name(ctx);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    return ggml_metal_cpy_tensor_async(ctx_src, ctx_dst, src, dst);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static void ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

static void ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ true,
    };
}

static ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_supports_op(ctx_dev, op);
}

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_dsv4_sparse_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_metal_reg * ggml_backend_metal_reg_t;

static ggml_backend_metal_reg_t ggml_backend_metal_reg_init(void) {
    ggml_backend_metal_reg_t ctx = new struct ggml_backend_metal_reg;

    return ctx;
}

static void ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_metal_reg_deleter {
    void operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_buffer_type") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_buffer_type;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_map_tensor_ranges") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_map_tensor_ranges;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_tensor_usage") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_tensor_usage;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_quote_tensor_ranges") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_quote_tensor_ranges;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_reserve_tensor_ranges") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_reserve_tensor_ranges;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_reservation_commit") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_reservation_commit;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_reservation_rollback") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_reservation_rollback;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_reservation_cancel") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_reservation_cancel;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_reservation_free") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_reservation_free;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_quote") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_move_tensor_rows_quote;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_commit") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_move_tensor_rows_commit;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_move_tensor_rows_free") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_move_tensor_rows_free;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_move_fail_for_test") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_move_fail_for_test;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_alias_tensor_rows") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_alias_tensor_rows;
    }
    if (strcmp(name, "ggml_backend_metal_dsv4_sparse_unmap_tensor_range") == 0) {
        return (void *)ggml_backend_metal_dsv4_sparse_unmap_tensor_range;
    }
    if (strcmp(name, "ggml_backend_metal_get_tensor_host_ptr") == 0) {
        return (void *)ggml_backend_metal_get_tensor_host_ptr;
    }
    if (strcmp(name, "ggml_backend_metal_buffer_type_is_shared") == 0) {
        return (void *)ggml_backend_metal_buffer_type_is_shared;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_count,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device),
    };
}

static void ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)

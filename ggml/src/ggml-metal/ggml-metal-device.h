#pragma once

#include "ggml.h"
#include "ggml-dsv4-sparse.h"
#include "ggml-metal-sparse-planner.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(void);
void ggml_metal_cv_free(ggml_metal_cv_t cv);

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx);
void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);

struct ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    int nsg;

    int nr0;
    int nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline);

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// GGML_METAL_KPROF: per-kernel GPU attribution (opt-in, measurement only)
//
// The M2 Ultra only supports MTLCounterSamplingPointAtStageBoundary (verified:
// DrawBoundary/DispatchBoundary/TileDispatchBoundary/BlitBoundary all report
// unsupported), so per-dispatch GPU timestamps are not available. Instead, when
// kprof is active the encoder is split into one compute pass per node (or per
// GGML_METAL_KPROF nodes) and each pass samples the GPU timestamp counter at
// its start and end boundary. Splitting costs one encoder boundary per segment;
// the stride sweep (GGML_METAL_KPROF=1,2,4,8,...) calibrates that overhead.
//

// non-zero when GGML_METAL_KPROF is set to a positive stride
int  ggml_metal_kprof_stride(void);

// end the current compute pass and begin a new counter-sampled one.
// `raw_node_idx` is the graph node index that starts the new segment.
// no-op (returns -1) when kprof is inactive.
int  ggml_metal_encoder_kprof_split(ggml_metal_encoder_t encoder, int raw_node_idx);

// resolve every completed segment recorded since the last flush and emit one
// `KPROF ` JSONL record per segment on stderr. Must be called only after the
// owning command buffers have completed.
void ggml_metal_kprof_flush(void);

//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose);

void ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_swiglu_clamp (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc_pre_norm  (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_router       (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_sparse_pack  (ggml_metal_library_t lib, enum ggml_type type, bool indexed);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, enum ggml_type type, bool tail);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, const struct ggml_tensor * op, int ssm_conv_bs);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nxpsg, int r1ptg);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, int ne02, int ne20, bool compact);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, const struct ggml_tensor * op, bool compact, bool paired, bool weighted);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, const struct ggml_tensor * op, bool weighted);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, int n);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix       (ggml_metal_library_t lib, bool radix8);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge       (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse );
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, enum ggml_type type);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, const struct ggml_tensor * op, bool tiled);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal       (ggml_metal_library_t lib, const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

typedef void * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
typedef struct ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
void ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

enum ggml_metal_device_id {
    GGML_METAL_DEVICE_GENERIC = 0,

    GGML_METAL_DEVICE_M1,
    GGML_METAL_DEVICE_M1_PRO,
    GGML_METAL_DEVICE_M1_MAX,
    GGML_METAL_DEVICE_M1_ULTRA,
    GGML_METAL_DEVICE_M2,
    GGML_METAL_DEVICE_M2_PRO,
    GGML_METAL_DEVICE_M2_MAX,
    GGML_METAL_DEVICE_M2_ULTRA,
    GGML_METAL_DEVICE_M3,
    GGML_METAL_DEVICE_M3_PRO,
    GGML_METAL_DEVICE_M3_MAX,
    GGML_METAL_DEVICE_M3_ULTRA,
    GGML_METAL_DEVICE_M4,
    GGML_METAL_DEVICE_M4_PRO,
    GGML_METAL_DEVICE_M4_MAX,
    GGML_METAL_DEVICE_M5,
    GGML_METAL_DEVICE_M5_PRO,
    GGML_METAL_DEVICE_M5_MAX,
    GGML_METAL_DEVICE_M5_ULTRA,
};

struct ggml_metal_device_props {
    int device;
    char name[128];
    char desc[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool has_placement_sparse;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;

    enum ggml_metal_device_id device_id;

    int op_offload_min_batch_size;
};

typedef struct ggml_metal_event * ggml_metal_event_t;

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
void ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(int device);
void ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(int device);

void * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
void * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
void ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op);

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;

enum ggml_metal_sparse_init_status {
    GGML_METAL_SPARSE_INIT_OK = 0,
    GGML_METAL_SPARSE_INIT_UNSUPPORTED,
    GGML_METAL_SPARSE_INIT_INVALID_SIZE,
    GGML_METAL_SPARSE_INIT_CPU_BUFFER,
    GGML_METAL_SPARSE_INIT_MTL_BUFFER,
    GGML_METAL_SPARSE_INIT_PLACEMENT_HEAP,
    GGML_METAL_SPARSE_INIT_COMMAND_QUEUE,
    GGML_METAL_SPARSE_INIT_SHARED_EVENT,
    GGML_METAL_SPARSE_INIT_LOCK,
    GGML_METAL_SPARSE_INIT_CPU_V2P_TABLE,
    GGML_METAL_SPARSE_INIT_CPU_REFCOUNT_TABLE,
    GGML_METAL_SPARSE_INIT_CPU_FREE_TABLE,
    GGML_METAL_SPARSE_INIT_RESIDENCY_SET,
};

struct ggml_metal_sparse_init_result {
    enum ggml_metal_sparse_init_status status;
    size_t requested_virtual_bytes;
    size_t requested_physical_bytes;
    size_t aligned_virtual_bytes;
    size_t aligned_physical_bytes;
    size_t device_max_buffer_length;
};

struct ggml_metal_sparse_usage {
    uintptr_t pool_id;
    size_t page_size;
    size_t virtual_pages;
    size_t physical_pages;
    size_t free_pages;
    size_t reserved_pages;
    size_t mapped_mappings;
    size_t unique_physical_pages;
    size_t shared_physical_pages;
    size_t shared_mappings;
    size_t refcount_sum;
    uint32_t refcount_max;
    uint64_t generation;
    uint64_t cow_allocations;
    uint64_t cow_pages;
};

struct ggml_metal_sparse_buffer_range {
    ggml_metal_buffer_t buffer;
    size_t offset;
    size_t size;
};

struct ggml_metal_sparse_buffer_move {
    struct ggml_metal_sparse_buffer_range source;
    // A null destination buffer releases the source mapping.
    struct ggml_metal_sparse_buffer_range destination;
};

struct ggml_metal_sparse_pool_quote {
    uintptr_t pool_id;
    struct ggml_metal_sparse_usage usage;
    struct ggml_metal_sparse_quote write;
};

enum ggml_metal_sparse_reservation_result {
    GGML_METAL_SPARSE_RESERVATION_OK = 0,
    GGML_METAL_SPARSE_RESERVATION_PRESSURE,
    GGML_METAL_SPARSE_RESERVATION_STALE,
    GGML_METAL_SPARSE_RESERVATION_INVALID,
    GGML_METAL_SPARSE_RESERVATION_OOM,
    GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED,
};

static inline const char * ggml_metal_sparse_reservation_result_name(
        enum ggml_metal_sparse_reservation_result result) {
    switch (result) {
        case GGML_METAL_SPARSE_RESERVATION_OK:          return "ok";
        case GGML_METAL_SPARSE_RESERVATION_PRESSURE:    return "pressure";
        case GGML_METAL_SPARSE_RESERVATION_STALE:       return "stale";
        case GGML_METAL_SPARSE_RESERVATION_INVALID:     return "invalid";
        case GGML_METAL_SPARSE_RESERVATION_OOM:         return "oom";
        case GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED: return "unsupported";
    }
    return "unknown";
}

typedef struct ggml_metal_sparse_reservation * ggml_metal_sparse_reservation_t;
typedef struct ggml_metal_sparse_move * ggml_metal_sparse_move_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_init_sparse(
        ggml_metal_device_t dev,
        size_t virtual_size,
        size_t physical_size);
ggml_metal_buffer_t ggml_metal_buffer_init_sparse_ex(
        ggml_metal_device_t dev,
        size_t virtual_size,
        size_t physical_size,
        struct ggml_metal_sparse_init_result * result);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_sparse(ggml_metal_buffer_t buf);

bool ggml_metal_buffer_sparse_get_usage(
        ggml_metal_buffer_t buf,
        struct ggml_metal_sparse_usage * usage);

// Largest number of distinct sparse pools a residency probe can span. The DSV4
// decode batch touches one pool per memory family; the bound exists only so the
// probe can order its locks on the stack without allocating.
#define GGML_METAL_SPARSE_RESIDENT_MAX_POOLS 64

// Cheap predicate for "this write needs no reservation at all". Returns 1 when
// every page covered by the ranges is already mapped to a physical page that
// this virtual mapping owns exclusively (v2p != UINT32_MAX and p_ref == 1), 0
// when a reservation is required, and -1 on invalid input.
//
// A quote over such a range set yields new_pages = cow_pages = required_pages =
// 0, and committing that reservation performs no mapping operation, no
// generation bump, no COW copy and no net change to the pool's reserved-page
// accounting. The probe is therefore an exact substitute for the full
// quote/reserve/commit cycle in that case, and costs O(pages in ranges)
// instead of O(virtual pages in the pool).
int ggml_metal_buffers_sparse_ranges_resident(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges);

enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_quote(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges,
        struct ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool);

enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_reserve(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges,
        struct ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool,
        ggml_metal_sparse_reservation_t * reservation);

enum ggml_metal_sparse_reservation_result ggml_metal_sparse_reservation_commit(
        ggml_metal_sparse_reservation_t reservation);
bool ggml_metal_sparse_reservation_rollback(ggml_metal_sparse_reservation_t reservation);
bool ggml_metal_sparse_reservation_cancel  (ggml_metal_sparse_reservation_t reservation);
void ggml_metal_sparse_reservation_free    (ggml_metal_sparse_reservation_t reservation);

// A move quote validates every source/destination aperture and snapshots all
// sparse-pool generations without mutating mappings. Commit locks every pool
// in address order, rejects stale generations, preallocates all operation
// storage, then transfers ownership without a partial-failure path.
enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_move_quote(
        const struct ggml_metal_sparse_buffer_move * moves,
        size_t n_moves,
        ggml_metal_sparse_move_t * quote);
enum ggml_metal_sparse_reservation_result ggml_metal_sparse_move_commit(
        ggml_metal_sparse_move_t quote);
void ggml_metal_sparse_move_free(ggml_metal_sparse_move_t quote);
int ggml_metal_sparse_move_audit(
        ggml_metal_sparse_move_t quote,
        int committed,
        struct ggml_dsv4_sparse_move_audit * audit);

bool ggml_metal_buffer_sparse_map_write(
        ggml_metal_buffer_t buf,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges);

bool ggml_metal_buffer_sparse_alias(
        ggml_metal_buffer_t buf,
        size_t src_offset,
        size_t dst_offset,
        size_t size,
        // Selected byte ranges relative to the source/destination views, not
        // absolute offsets within the backing buffer.
        const size_t * relative_offsets,
        const size_t * sizes,
        size_t n_ranges);

bool ggml_metal_buffer_sparse_unmap(
        ggml_metal_buffer_t buf,
        size_t offset,
        size_t size);

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * src, struct ggml_tensor * dst);
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif

#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <inttypes.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

// Private ABI shared with ggml-metal-device.m. Keeping it out of the public
// Metal headers avoids exposing this target-only instrumentation prototype.
struct ggml_metal_encoder_profile_result {
    uint64_t fingerprint;
    uint64_t pipeline_fingerprint;
    uint64_t scalar_fingerprint;
    uint64_t buffer_fingerprint;
    uint64_t threadgroup_memory_fingerprint;
    uint64_t dispatch_fingerprint;
    uint64_t barrier_fingerprint;

    uint64_t n_commands;
    uint64_t n_pipelines;
    uint64_t n_scalars;
    uint64_t n_buffers;
    uint64_t n_threadgroup_memories;
    uint64_t n_dispatches;
    uint64_t n_barriers;

    uint64_t scalar_bytes;
    uint64_t projected_plan_bytes;
};

void ggml_metal_encoder_profile_begin(void);
struct ggml_metal_encoder_profile_result ggml_metal_encoder_profile_end(void);

struct ggml_metal_encode_profile_sample {
    bool valid;
    bool committed;

    uint64_t uid;
    int n_cb;
    int idx_start;
    int idx_end;
    int n_raw_nodes;
    int n_filtered_nodes;
    bool use_fusion;
    bool use_concurrency;

    int64_t prepare_us;
    int64_t encode_us;
    int64_t finish_us;
    int64_t commit_us;
    int64_t total_us;

    struct ggml_metal_encoder_profile_result commands;
};

struct ggml_metal_encode_profile_segment {
    bool valid;

    uint64_t uid;
    int n_cb;
    int idx_start;
    int idx_end;
    bool use_fusion;
    bool use_concurrency;

    struct ggml_metal_encoder_profile_result reference;
    struct ggml_metal_encode_profile_sample last;

    uint64_t samples;
    uint64_t matches;
    uint64_t changes;

    int64_t prepare_us;
    int64_t encode_us;
    int64_t finish_us;
    int64_t commit_us;
    int64_t total_us;
};

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;
    bool use_dsv4_split_tuning;
    bool use_dsv4_encode_profile;

    int dsv4_encode_profile_interval;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // Opt-in instrumentation state. Each worker owns its indexed sample while
    // encoding; the caller compares and logs samples only after dispatch_apply.
    bool dsv4_encode_profile_active;
    struct ggml_metal_encode_profile_sample dsv4_encode_profile_samples[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    struct ggml_metal_encode_profile_segment dsv4_encode_profile_segments[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    uint64_t dsv4_encode_profile_graphs;
    uint64_t dsv4_encode_profile_key_hits;
    uint64_t dsv4_encode_profile_key_misses;
    uint64_t dsv4_encode_profile_replay_hits;
    uint64_t dsv4_encode_profile_replay_misses;
    uint64_t dsv4_encode_profile_changes;
    int64_t dsv4_encode_profile_graph_host_us;

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

static bool ggml_metal_encode_profile_commands_equal(
        const struct ggml_metal_encoder_profile_result * lhs,
        const struct ggml_metal_encoder_profile_result * rhs) {
    return lhs->fingerprint                    == rhs->fingerprint &&
           lhs->pipeline_fingerprint           == rhs->pipeline_fingerprint &&
           lhs->scalar_fingerprint             == rhs->scalar_fingerprint &&
           lhs->buffer_fingerprint             == rhs->buffer_fingerprint &&
           lhs->threadgroup_memory_fingerprint == rhs->threadgroup_memory_fingerprint &&
           lhs->dispatch_fingerprint           == rhs->dispatch_fingerprint &&
           lhs->barrier_fingerprint            == rhs->barrier_fingerprint &&
           lhs->n_commands                     == rhs->n_commands &&
           lhs->n_pipelines                    == rhs->n_pipelines &&
           lhs->n_scalars                      == rhs->n_scalars &&
           lhs->n_buffers                      == rhs->n_buffers &&
           lhs->n_threadgroup_memories         == rhs->n_threadgroup_memories &&
           lhs->n_dispatches                   == rhs->n_dispatches &&
           lhs->n_barriers                     == rhs->n_barriers &&
           lhs->scalar_bytes                   == rhs->scalar_bytes &&
           lhs->projected_plan_bytes           == rhs->projected_plan_bytes;
}

static bool ggml_metal_encode_profile_key_equal(
        const struct ggml_metal_encode_profile_segment * segment,
        const struct ggml_metal_encode_profile_sample * sample) {
    return segment->valid &&
           segment->uid             == sample->uid &&
           segment->n_cb            == sample->n_cb &&
           segment->idx_start       == sample->idx_start &&
           segment->idx_end         == sample->idx_end &&
           segment->use_fusion      == sample->use_fusion &&
           segment->use_concurrency == sample->use_concurrency;
}

static void ggml_metal_encode_profile_diff_add(char * dst, size_t size, const char * field) {
    const size_t used = strlen(dst);
    if (used >= size) {
        return;
    }

    snprintf(dst + used, size - used, "%s%s", used > 0 ? "," : "", field);
}

static void ggml_metal_encode_profile_diff(
        const struct ggml_metal_encoder_profile_result * reference,
        const struct ggml_metal_encoder_profile_result * current,
        char * dst,
        size_t size) {
    dst[0] = '\0';

    if (reference->pipeline_fingerprint != current->pipeline_fingerprint ||
            reference->n_pipelines != current->n_pipelines) {
        ggml_metal_encode_profile_diff_add(dst, size, "pipeline");
    }
    if (reference->scalar_fingerprint != current->scalar_fingerprint ||
            reference->n_scalars != current->n_scalars ||
            reference->scalar_bytes != current->scalar_bytes) {
        ggml_metal_encode_profile_diff_add(dst, size, "scalar");
    }
    if (reference->buffer_fingerprint != current->buffer_fingerprint ||
            reference->n_buffers != current->n_buffers) {
        ggml_metal_encode_profile_diff_add(dst, size, "buffer+offset");
    }
    if (reference->threadgroup_memory_fingerprint != current->threadgroup_memory_fingerprint ||
            reference->n_threadgroup_memories != current->n_threadgroup_memories) {
        ggml_metal_encode_profile_diff_add(dst, size, "tgmem");
    }
    if (reference->dispatch_fingerprint != current->dispatch_fingerprint ||
            reference->n_dispatches != current->n_dispatches) {
        ggml_metal_encode_profile_diff_add(dst, size, "grid");
    }
    if (reference->barrier_fingerprint != current->barrier_fingerprint ||
            reference->n_barriers != current->n_barriers) {
        ggml_metal_encode_profile_diff_add(dst, size, "barrier");
    }
    if (dst[0] == '\0' && reference->fingerprint != current->fingerprint) {
        ggml_metal_encode_profile_diff_add(dst, size, "overall-order");
    }
    if (dst[0] == '\0') {
        snprintf(dst, size, "none");
    }
}

static void ggml_metal_encode_profile_log_segment(
        const struct ggml_metal_encode_profile_segment * segment,
        int cb_idx,
        const char * reason,
        const char * diff) {
    const struct ggml_metal_encode_profile_sample * sample = &segment->last;
    const struct ggml_metal_encoder_profile_result * commands = &sample->commands;
    const uint64_t comparisons = segment->matches + segment->changes;
    const double stable_pct = comparisons > 0 ? 100.0*segment->matches/comparisons : 0.0;
    const double samples = segment->samples > 0 ? (double) segment->samples : 1.0;

    GGML_LOG_INFO("dsv4_encode_profile: uid=%" PRIu64 " cb=%d/%d range=%d:%d reason=%s diff=%s samples=%" PRIu64
            " matches=%" PRIu64 " changes=%" PRIu64 " stable=%.2f%% raw=%d filtered=%d committed=%s\n",
            sample->uid, cb_idx, sample->n_cb, sample->idx_start, sample->idx_end, reason, diff,
            segment->samples, segment->matches, segment->changes, stable_pct,
            sample->n_raw_nodes, sample->n_filtered_nodes, sample->committed ? "true" : "false");
    GGML_LOG_INFO("dsv4_encode_profile: timing_us prepare=%" PRId64 "/%.1f encode=%" PRId64 "/%.1f finish=%" PRId64
            "/%.1f commit=%" PRId64 "/%.1f total=%" PRId64 "/%.1f commands=%" PRIu64
            " projected_bytes=%" PRIu64 " scalar_bytes=%" PRIu64 "\n",
            sample->prepare_us, segment->prepare_us/samples,
            sample->encode_us,  segment->encode_us/samples,
            sample->finish_us,  segment->finish_us/samples,
            sample->commit_us,  segment->commit_us/samples,
            sample->total_us,   segment->total_us/samples,
            commands->n_commands, commands->projected_plan_bytes, commands->scalar_bytes);
    GGML_LOG_INFO("dsv4_encode_profile: fingerprints all=%016" PRIx64 " pipeline=%016" PRIx64 " scalar=%016" PRIx64
            " buffer_offset=%016" PRIx64 " tgmem=%016" PRIx64 " grid=%016" PRIx64 " barrier=%016" PRIx64
            " counts=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            commands->fingerprint, commands->pipeline_fingerprint, commands->scalar_fingerprint,
            commands->buffer_fingerprint, commands->threadgroup_memory_fingerprint,
            commands->dispatch_fingerprint, commands->barrier_fingerprint,
            commands->n_pipelines, commands->n_scalars, commands->n_buffers,
            commands->n_threadgroup_memories, commands->n_dispatches, commands->n_barriers);
}

static void ggml_metal_encode_profile_log_summary(ggml_metal_t ctx, const char * reason) {
    const uint64_t key_samples = ctx->dsv4_encode_profile_key_hits + ctx->dsv4_encode_profile_key_misses;
    const uint64_t replay_samples = ctx->dsv4_encode_profile_replay_hits + ctx->dsv4_encode_profile_replay_misses;
    const double key_hit_pct = key_samples > 0 ? 100.0*ctx->dsv4_encode_profile_key_hits/key_samples : 0.0;
    const double replay_hit_pct = replay_samples > 0 ? 100.0*ctx->dsv4_encode_profile_replay_hits/replay_samples : 0.0;
    const double graph_host_us = ctx->dsv4_encode_profile_graphs > 0 ?
            (double) ctx->dsv4_encode_profile_graph_host_us/ctx->dsv4_encode_profile_graphs : 0.0;

    GGML_LOG_INFO("dsv4_encode_profile: summary=%s graphs=%" PRIu64 " graph_host_us_avg=%.1f key_hits=%" PRIu64
            " key_misses=%" PRIu64 " key_hit=%.2f%% replay_hits=%" PRIu64 " replay_misses=%" PRIu64
            " replay_hit=%.2f%% same_key_changes=%" PRIu64 "\n",
            reason, ctx->dsv4_encode_profile_graphs, graph_host_us,
            ctx->dsv4_encode_profile_key_hits, ctx->dsv4_encode_profile_key_misses, key_hit_pct,
            ctx->dsv4_encode_profile_replay_hits, ctx->dsv4_encode_profile_replay_misses, replay_hit_pct,
            ctx->dsv4_encode_profile_changes);
}

static void ggml_metal_encode_profile_update(
        ggml_metal_t ctx,
        int cb_idx,
        bool periodic) {
    struct ggml_metal_encode_profile_sample * sample = &ctx->dsv4_encode_profile_samples[cb_idx];
    struct ggml_metal_encode_profile_segment * segment = &ctx->dsv4_encode_profile_segments[cb_idx];
    if (!sample->valid) {
        return;
    }

    const bool same_key = ggml_metal_encode_profile_key_equal(segment, sample);
    const bool exact = same_key && ggml_metal_encode_profile_commands_equal(&segment->reference, &sample->commands);

    char diff[96];
    const char * reason = "periodic";
    if (!same_key) {
        memset(segment, 0, sizeof(*segment));
        segment->valid           = true;
        segment->uid             = sample->uid;
        segment->n_cb            = sample->n_cb;
        segment->idx_start       = sample->idx_start;
        segment->idx_end         = sample->idx_end;
        segment->use_fusion      = sample->use_fusion;
        segment->use_concurrency = sample->use_concurrency;
        segment->reference       = sample->commands;

        ctx->dsv4_encode_profile_key_misses++;
        ctx->dsv4_encode_profile_replay_misses++;
        snprintf(diff, sizeof(diff), "new-key");
        reason = "new-key";
    } else {
        ctx->dsv4_encode_profile_key_hits++;
        if (exact) {
            segment->matches++;
            ctx->dsv4_encode_profile_replay_hits++;
            snprintf(diff, sizeof(diff), "none");
        } else {
            segment->changes++;
            ctx->dsv4_encode_profile_replay_misses++;
            ctx->dsv4_encode_profile_changes++;
            ggml_metal_encode_profile_diff(&segment->reference, &sample->commands, diff, sizeof(diff));
            reason = "changed";
        }
    }

    segment->last = *sample;
    segment->samples++;
    segment->prepare_us += sample->prepare_us;
    segment->encode_us  += sample->encode_us;
    segment->finish_us  += sample->finish_us;
    segment->commit_us  += sample->commit_us;
    segment->total_us   += sample->total_us;

    if (!same_key || !exact || periodic) {
        ggml_metal_encode_profile_log_segment(segment, cb_idx, reason, diff);
    }
}

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion            = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency       = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;
    res->use_dsv4_split_tuning = getenv("GGML_METAL_DSV4_SPLIT_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_DSV4_ENCODE_PROFILE");
        res->use_dsv4_encode_profile = val != NULL;
        res->dsv4_encode_profile_interval = val ? atoi(val) : 0;
        if (res->use_dsv4_encode_profile && res->dsv4_encode_profile_interval <= 0) {
            res->dsv4_encode_profile_interval = 64;
        }
    }

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");
    if (res->use_dsv4_encode_profile) {
        GGML_LOG_INFO("%s: DSV4 encode profile = true (interval = %d)\n",
                __func__, res->dsv4_encode_profile_interval);
    }

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    res->dsv4_encode_profile_active = false;
    memset(res->dsv4_encode_profile_samples, 0, sizeof(res->dsv4_encode_profile_samples));
    memset(res->dsv4_encode_profile_segments, 0, sizeof(res->dsv4_encode_profile_segments));

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->use_dsv4_encode_profile && ctx->dsv4_encode_profile_graphs > 0) {
        for (int cb_idx = 0; cb_idx <= GGML_METAL_MAX_COMMAND_BUFFERS; ++cb_idx) {
            if (ctx->dsv4_encode_profile_segments[cb_idx].valid) {
                ggml_metal_encode_profile_log_segment(
                        &ctx->dsv4_encode_profile_segments[cb_idx], cb_idx, "final", "none");
            }
        }
        ggml_metal_encode_profile_log_summary(ctx, "final");
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    // Number of nodes encoded by the main thread. A DSV4 decode graph on the
    // M2 Ultra has about 6.8k raw nodes (about 4k after Metal fusion). Its
    // generic 10% split delays the first submission enough to leave a
    // repeatable GPU bubble. A measured 512-node prefix starts the GPU earlier
    // while still covering async encoding of the remainder. Keep small and
    // non-target graphs on the established generic policy.
    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    const bool tune_dsv4_split = ctx->use_dsv4_split_tuning &&
            props_dev->device_id == GGML_METAL_DEVICE_M2_ULTRA && gf->n_nodes >= 3000;
    const int n_main = tune_dsv4_split ? 512 : MAX(64, 0.1*gf->n_nodes);
    const bool profile_dsv4_encode = ctx->use_dsv4_encode_profile && tune_dsv4_split && gf->uid != 0;
    const int64_t profile_graph_start_us = profile_dsv4_encode ? ggml_time_us() : 0;
    if (ctx->debug_graph > 0) {
        GGML_LOG_DEBUG("%s: nodes = %d, main-thread nodes = %d, DSV4 split tuning = %s\n",
                __func__, gf->n_nodes, n_main, tune_dsv4_split ? "true" : "false");
    }

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // Capture and graph-debug encoders can add commands that are not part
        // of the stable production decode program. Leave those runs out of the
        // replay-feasibility fingerprint.
        ctx->dsv4_encode_profile_active = profile_dsv4_encode &&
                !use_capture && !ctx->capture_started && ctx->debug_graph == 0;
        if (ctx->dsv4_encode_profile_active) {
            memset(ctx->dsv4_encode_profile_samples, 0, sizeof(ctx->dsv4_encode_profile_samples));
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        if (ctx->dsv4_encode_profile_active) {
            const int64_t graph_host_us = ggml_time_us() - profile_graph_start_us;
            ctx->dsv4_encode_profile_graphs++;
            ctx->dsv4_encode_profile_graph_host_us += graph_host_us;

            const bool periodic = ctx->dsv4_encode_profile_graphs % ctx->dsv4_encode_profile_interval == 0;
            for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
                ggml_metal_encode_profile_update(ctx, cb_idx, periodic);
            }

            if (periodic) {
                ggml_metal_encode_profile_log_summary(ctx, "periodic");
            }
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    return GGML_STATUS_ABORTED;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        const bool profile_dsv4_encode = ctx->dsv4_encode_profile_active;
        const int64_t profile_start_us = profile_dsv4_encode ? ggml_time_us() : 0;
        if (profile_dsv4_encode) {
            ggml_metal_encoder_profile_begin();
        }

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency,
            ctx->capture_compute,
            ctx->debug_graph,
            ctx->debug_fusion);

        const int64_t profile_prepared_us = profile_dsv4_encode ? ggml_time_us() : 0;
        const int n_filtered_nodes = profile_dsv4_encode ? ggml_metal_op_n_nodes(ctx_op) : 0;

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        const int64_t profile_encoded_us = profile_dsv4_encode ? ggml_time_us() : 0;

        ggml_metal_op_free(ctx_op);

        const int64_t profile_finished_us = profile_dsv4_encode ? ggml_time_us() : 0;

        bool committed = false;
        int64_t profile_committed_us = profile_finished_us;
        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
            committed = true;
            profile_committed_us = profile_dsv4_encode ? ggml_time_us() : profile_finished_us;
        }

        if (profile_dsv4_encode) {
            struct ggml_metal_encode_profile_sample * sample = &ctx->dsv4_encode_profile_samples[cb_idx];

            sample->valid           = true;
            sample->committed       = committed;
            sample->uid             = ctx->gf->uid;
            sample->n_cb            = n_cb_l;
            sample->idx_start        = idx_start;
            sample->idx_end          = idx_end;
            sample->n_raw_nodes      = idx_end - idx_start;
            sample->n_filtered_nodes = n_filtered_nodes;
            sample->use_fusion      = ctx->use_fusion;
            sample->use_concurrency = ctx->use_concurrency;
            sample->prepare_us      = profile_prepared_us - profile_start_us;
            sample->encode_us       = profile_encoded_us - profile_prepared_us;
            sample->finish_us       = profile_finished_us - profile_encoded_us;
            sample->commit_us       = profile_committed_us - profile_finished_us;
            sample->total_us        = profile_committed_us - profile_start_us;
            sample->commands        = ggml_metal_encoder_profile_end();
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}

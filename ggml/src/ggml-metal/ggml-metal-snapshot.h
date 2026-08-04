#pragma once

#include "ggml-metal-device.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_SNAPSHOT_STAGING_SLOTS 2

enum ggml_metal_snapshot_status {
    GGML_METAL_SNAPSHOT_STATUS_OK = 0,
    GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT,
    GGML_METAL_SNAPSHOT_STATUS_WOULD_BLOCK,
    GGML_METAL_SNAPSHOT_STATUS_NOT_READY,
    GGML_METAL_SNAPSHOT_STATUS_STALE_TRANSFER,
    GGML_METAL_SNAPSHOT_STATUS_RESOURCE_ERROR,
    GGML_METAL_SNAPSHOT_STATUS_COMMAND_ERROR,
    GGML_METAL_SNAPSHOT_STATUS_STOPPED,
};

const char * ggml_metal_snapshot_status_name(enum ggml_metal_snapshot_status status);

enum ggml_metal_snapshot_direction {
    GGML_METAL_SNAPSHOT_DIRECTION_NONE = 0,
    GGML_METAL_SNAPSHOT_DIRECTION_READBACK,
    GGML_METAL_SNAPSHOT_DIRECTION_RESTORE,
};

// One contiguous copy between a private Metal tensor and a shared staging
// slot. staging_offset values must form a gap-free range beginning at zero.
struct ggml_metal_snapshot_copy {
    ggml_metal_buffer_t        buffer;
    const struct ggml_tensor * tensor;
    size_t                     tensor_offset;
    size_t                     staging_offset;
    size_t                     size;
};

struct ggml_metal_snapshot_token {
    uint32_t slot;
    uint64_t generation;
};

struct ggml_metal_snapshot_poll_result {
    enum ggml_metal_snapshot_direction direction;
    const void *                       data;
    size_t                             size;
    int32_t                            command_buffer_status;
    int64_t                            command_error;
};

struct ggml_metal_snapshot_stats {
    size_t   staging_bytes;
    uint32_t slots_in_use;
    uint32_t slot_high_watermark;
    uint64_t readbacks_submitted;
    uint64_t restores_submitted;
    uint64_t readbacks_completed;
    uint64_t restores_completed;
    uint64_t failed_transfers;
    uint64_t readback_bytes;
    uint64_t restore_bytes;
    uint64_t would_block_count;
};

typedef struct ggml_metal_snapshot_staging * ggml_metal_snapshot_staging_t;

// The device and every bound private buffer must outlive the staging object
// and all submitted transfers. Submit, poll, and release are thread-safe;
// free requires external caller quiescence. Exactly two shared MTLBuffers are
// allocated.
ggml_metal_snapshot_staging_t ggml_metal_snapshot_staging_init(ggml_metal_device_t device, size_t staging_bytes);
void                          ggml_metal_snapshot_staging_free(ggml_metal_snapshot_staging_t staging);

// Submissions are appended to the device's command queue after prior graph
// work and never wait for GPU completion. WOULD_BLOCK means both fixed slots
// are in-flight or completed but not yet released by the caller.
enum ggml_metal_snapshot_status ggml_metal_snapshot_submit_readback(ggml_metal_snapshot_staging_t           staging,
                                                                    const struct ggml_metal_snapshot_copy * copies,
                                                                    size_t                                  copy_count,
                                                                    struct ggml_metal_snapshot_token *      token);

// The source bytes are copied into a free shared MTLBuffer before the private
// restore blit is committed. Completion only means the GPU consumed that slot.
enum ggml_metal_snapshot_status ggml_metal_snapshot_submit_restore(ggml_metal_snapshot_staging_t           staging,
                                                                   const void *                            source,
                                                                   size_t                                  source_size,
                                                                   const struct ggml_metal_snapshot_copy * copies,
                                                                   size_t                                  copy_count,
                                                                   struct ggml_metal_snapshot_token *      token);

enum ggml_metal_snapshot_status ggml_metal_snapshot_poll(ggml_metal_snapshot_staging_t            staging,
                                                         struct ggml_metal_snapshot_token         token,
                                                         struct ggml_metal_snapshot_poll_result * result);

// A successful or failed completed transfer retains its slot until release.
// Readback data returned by poll remains valid through that release. In-flight
// transfers cannot be released.
enum ggml_metal_snapshot_status ggml_metal_snapshot_release(ggml_metal_snapshot_staging_t    staging,
                                                            struct ggml_metal_snapshot_token token);

struct ggml_metal_snapshot_stats ggml_metal_snapshot_get_stats(ggml_metal_snapshot_staging_t staging);

#ifdef __cplusplus
}
#endif

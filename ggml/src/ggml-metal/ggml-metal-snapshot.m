#include "ggml-metal-snapshot.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dispatch/dispatch.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

enum ggml_metal_snapshot_slot_state {
    GGML_METAL_SNAPSHOT_SLOT_FREE = 0,
    GGML_METAL_SNAPSHOT_SLOT_IN_FLIGHT,
    GGML_METAL_SNAPSHOT_SLOT_COMPLETE,
    GGML_METAL_SNAPSHOT_SLOT_FAILED,
};

struct ggml_metal_snapshot_slot {
    id<MTLBuffer>                         buffer;
    enum ggml_metal_snapshot_slot_state   state;
    enum ggml_metal_snapshot_direction    direction;
    uint64_t                              generation;
    size_t                                size;
    int32_t                               command_buffer_status;
    int64_t                               command_error;
};

struct ggml_metal_snapshot_staging {
    id<MTLDevice>                          metal_device;
    id<MTLCommandQueue>                    queue;
    size_t                                 staging_bytes;
    pthread_mutex_t                        mutex;
    dispatch_group_t                       completions;
    bool                                   stopping;
    struct ggml_metal_snapshot_slot        slots[GGML_METAL_SNAPSHOT_STAGING_SLOTS];
    struct ggml_metal_snapshot_stats       stats;
};

const char * ggml_metal_snapshot_status_name(enum ggml_metal_snapshot_status status) {
    switch (status) {
        case GGML_METAL_SNAPSHOT_STATUS_OK:               return "ok";
        case GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT: return "invalid_argument";
        case GGML_METAL_SNAPSHOT_STATUS_WOULD_BLOCK:      return "would_block";
        case GGML_METAL_SNAPSHOT_STATUS_NOT_READY:        return "not_ready";
        case GGML_METAL_SNAPSHOT_STATUS_STALE_TRANSFER:   return "stale_transfer";
        case GGML_METAL_SNAPSHOT_STATUS_RESOURCE_ERROR:   return "resource_error";
        case GGML_METAL_SNAPSHOT_STATUS_COMMAND_ERROR:    return "command_error";
        case GGML_METAL_SNAPSHOT_STATUS_STOPPED:          return "stopped";
    }
    return "unknown";
}

static enum ggml_metal_snapshot_status ggml_metal_snapshot_validate_copies(
        ggml_metal_snapshot_staging_t            staging,
        const struct ggml_metal_snapshot_copy *  copies,
        size_t                                   copy_count,
        size_t *                                 transfer_size) {
    if (copies == NULL || copy_count == 0 || transfer_size == NULL) {
        return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }

    size_t expected_offset = 0;
    for (size_t index = 0; index < copy_count; ++index) {
        const struct ggml_metal_snapshot_copy * copy = &copies[index];
        if (copy->buffer == NULL || copy->tensor == NULL || copy->size == 0 ||
            copy->staging_offset != expected_offset || ggml_metal_buffer_is_shared(copy->buffer)) {
            return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
        }
        const size_t tensor_size = ggml_nbytes(copy->tensor);
        if (copy->tensor_offset > tensor_size || copy->size > tensor_size - copy->tensor_offset ||
            copy->size > staging->staging_bytes - expected_offset) {
            return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
        }
        const struct ggml_metal_buffer_id buffer_id = ggml_metal_buffer_get_id(copy->buffer, copy->tensor);
        if (buffer_id.metal == nil || copy->tensor_offset > SIZE_MAX - buffer_id.offs ||
            [(id<MTLBuffer>) buffer_id.metal device] != staging->metal_device) {
            return GGML_METAL_SNAPSHOT_STATUS_RESOURCE_ERROR;
        }
        expected_offset += copy->size;
    }
    *transfer_size = expected_offset;
    return GGML_METAL_SNAPSHOT_STATUS_OK;
}

static struct ggml_metal_snapshot_slot * ggml_metal_snapshot_find_free_slot(
        ggml_metal_snapshot_staging_t staging,
        uint32_t *                    slot_index) {
    for (uint32_t index = 0; index < GGML_METAL_SNAPSHOT_STAGING_SLOTS; ++index) {
        if (staging->slots[index].state == GGML_METAL_SNAPSHOT_SLOT_FREE) {
            *slot_index = index;
            return &staging->slots[index];
        }
    }
    return NULL;
}

static void ggml_metal_snapshot_reset_slot(
        ggml_metal_snapshot_staging_t staging,
        struct ggml_metal_snapshot_slot * slot) {
    slot->state                 = GGML_METAL_SNAPSHOT_SLOT_FREE;
    slot->direction             = GGML_METAL_SNAPSHOT_DIRECTION_NONE;
    slot->size                  = 0;
    slot->command_buffer_status = 0;
    slot->command_error         = 0;
    --staging->stats.slots_in_use;
}

ggml_metal_snapshot_staging_t ggml_metal_snapshot_staging_init(
        ggml_metal_device_t device,
        size_t              staging_bytes) {
    if (device == NULL || staging_bytes == 0) {
        return NULL;
    }
    id<MTLDevice> metal_device = ggml_metal_device_get_obj(device);
    id<MTLCommandQueue> queue  = ggml_metal_device_get_queue(device);
    if (metal_device == nil || queue == nil || staging_bytes > (size_t) [metal_device maxBufferLength]) {
        return NULL;
    }

    ggml_metal_snapshot_staging_t staging = calloc(1, sizeof(struct ggml_metal_snapshot_staging));
    if (staging == NULL) {
        return NULL;
    }
    if (pthread_mutex_init(&staging->mutex, NULL) != 0) {
        free(staging);
        return NULL;
    }
    staging->completions = dispatch_group_create();
    if (staging->completions == NULL) {
        pthread_mutex_destroy(&staging->mutex);
        free(staging);
        return NULL;
    }
    staging->metal_device        = metal_device;
    staging->queue               = queue;
    staging->staging_bytes       = staging_bytes;
    staging->stats.staging_bytes = staging_bytes;

    @autoreleasepool {
        for (uint32_t index = 0; index < GGML_METAL_SNAPSHOT_STAGING_SLOTS; ++index) {
            staging->slots[index].buffer =
                [metal_device newBufferWithLength:staging_bytes options:MTLResourceStorageModeShared];
            if (staging->slots[index].buffer == nil || [staging->slots[index].buffer contents] == NULL) {
                ggml_metal_snapshot_staging_free(staging);
                return NULL;
            }
        }
    }
    return staging;
}

void ggml_metal_snapshot_staging_free(ggml_metal_snapshot_staging_t staging) {
    if (staging == NULL) {
        return;
    }
    pthread_mutex_lock(&staging->mutex);
    staging->stopping = true;
    pthread_mutex_unlock(&staging->mutex);

    dispatch_group_wait(staging->completions, DISPATCH_TIME_FOREVER);
    for (uint32_t index = 0; index < GGML_METAL_SNAPSHOT_STAGING_SLOTS; ++index) {
        [staging->slots[index].buffer release];
    }
    dispatch_release(staging->completions);
    pthread_mutex_destroy(&staging->mutex);
    free(staging);
}

static enum ggml_metal_snapshot_status ggml_metal_snapshot_submit(
        ggml_metal_snapshot_staging_t            staging,
        enum ggml_metal_snapshot_direction       direction,
        const void *                             source,
        size_t                                   source_size,
        const struct ggml_metal_snapshot_copy *  copies,
        size_t                                   copy_count,
        struct ggml_metal_snapshot_token *       token) {
    if (staging == NULL || token == NULL ||
        (direction == GGML_METAL_SNAPSHOT_DIRECTION_RESTORE && source == NULL)) {
        return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    size_t transfer_size = 0;
    const enum ggml_metal_snapshot_status validated =
        ggml_metal_snapshot_validate_copies(staging, copies, copy_count, &transfer_size);
    if (validated != GGML_METAL_SNAPSHOT_STATUS_OK ||
        (direction == GGML_METAL_SNAPSHOT_DIRECTION_RESTORE && source_size != transfer_size)) {
        return validated == GGML_METAL_SNAPSHOT_STATUS_OK ? GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT : validated;
    }

    pthread_mutex_lock(&staging->mutex);
    if (staging->stopping) {
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_STOPPED;
    }
    uint32_t slot_index = 0;
    struct ggml_metal_snapshot_slot * slot = ggml_metal_snapshot_find_free_slot(staging, &slot_index);
    if (slot == NULL) {
        ++staging->stats.would_block_count;
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_WOULD_BLOCK;
    }

    slot->generation = slot->generation == UINT64_MAX ? 1 : slot->generation + 1;
    slot->state      = GGML_METAL_SNAPSHOT_SLOT_IN_FLIGHT;
    slot->direction  = direction;
    slot->size       = transfer_size;
    ++staging->stats.slots_in_use;
    if (staging->stats.slots_in_use > staging->stats.slot_high_watermark) {
        staging->stats.slot_high_watermark = staging->stats.slots_in_use;
    }
    *token = (struct ggml_metal_snapshot_token) { slot_index, slot->generation };

    enum ggml_metal_snapshot_status result = GGML_METAL_SNAPSHOT_STATUS_OK;
    @autoreleasepool {
        if (direction == GGML_METAL_SNAPSHOT_DIRECTION_RESTORE) {
            memcpy([slot->buffer contents], source, transfer_size);
        }

        id<MTLCommandBuffer> command_buffer = [staging->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder   = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            result = GGML_METAL_SNAPSHOT_STATUS_COMMAND_ERROR;
        } else {
            for (size_t index = 0; index < copy_count; ++index) {
                const struct ggml_metal_snapshot_copy * copy = &copies[index];
                struct ggml_metal_buffer_id buffer_id = ggml_metal_buffer_get_id(copy->buffer, copy->tensor);
                buffer_id.offs += copy->tensor_offset;
                if (direction == GGML_METAL_SNAPSHOT_DIRECTION_READBACK) {
                    [encoder copyFromBuffer:(id<MTLBuffer>) buffer_id.metal
                               sourceOffset:buffer_id.offs
                                   toBuffer:slot->buffer
                          destinationOffset:copy->staging_offset
                                       size:copy->size];
                } else {
                    const size_t destination_offset = buffer_id.offs;
                    if (!ggml_metal_buffer_sparse_map_write(
                            copy->buffer, &destination_offset, &copy->size, 1)) {
                        result = GGML_METAL_SNAPSHOT_STATUS_RESOURCE_ERROR;
                        break;
                    }
                    [encoder copyFromBuffer:slot->buffer
                               sourceOffset:copy->staging_offset
                                   toBuffer:(id<MTLBuffer>) buffer_id.metal
                          destinationOffset:buffer_id.offs
                                       size:copy->size];
                }
            }
            [encoder endEncoding];
        }

        if (result == GGML_METAL_SNAPSHOT_STATUS_OK) {
            const uint64_t generation = slot->generation;
            dispatch_group_enter(staging->completions);
            [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                @autoreleasepool {
                    pthread_mutex_lock(&staging->mutex);
                    struct ggml_metal_snapshot_slot * completed_slot = &staging->slots[slot_index];
                    if (completed_slot->generation == generation &&
                        completed_slot->state == GGML_METAL_SNAPSHOT_SLOT_IN_FLIGHT) {
                        const MTLCommandBufferStatus command_status = [completed status];
                        completed_slot->command_buffer_status = (int32_t) command_status;
                        NSError * error = [completed error];
                        completed_slot->command_error = error == nil ? 0 : (int64_t) [error code];
                        if (command_status == MTLCommandBufferStatusCompleted) {
                            completed_slot->state = GGML_METAL_SNAPSHOT_SLOT_COMPLETE;
                            if (completed_slot->direction == GGML_METAL_SNAPSHOT_DIRECTION_READBACK) {
                                ++staging->stats.readbacks_completed;
                                staging->stats.readback_bytes += completed_slot->size;
                            } else {
                                ++staging->stats.restores_completed;
                                staging->stats.restore_bytes += completed_slot->size;
                            }
                        } else {
                            completed_slot->state = GGML_METAL_SNAPSHOT_SLOT_FAILED;
                            ++staging->stats.failed_transfers;
                        }
                    }
                    pthread_mutex_unlock(&staging->mutex);
                }
                dispatch_group_leave(staging->completions);
            }];
            [command_buffer commit];
            if (direction == GGML_METAL_SNAPSHOT_DIRECTION_READBACK) {
                ++staging->stats.readbacks_submitted;
            } else {
                ++staging->stats.restores_submitted;
            }
        }
    }

    if (result != GGML_METAL_SNAPSHOT_STATUS_OK) {
        ggml_metal_snapshot_reset_slot(staging, slot);
    }
    pthread_mutex_unlock(&staging->mutex);
    return result;
}

enum ggml_metal_snapshot_status ggml_metal_snapshot_submit_readback(
        ggml_metal_snapshot_staging_t            staging,
        const struct ggml_metal_snapshot_copy *  copies,
        size_t                                   copy_count,
        struct ggml_metal_snapshot_token *       token) {
    return ggml_metal_snapshot_submit(
        staging, GGML_METAL_SNAPSHOT_DIRECTION_READBACK, NULL, 0, copies, copy_count, token);
}

enum ggml_metal_snapshot_status ggml_metal_snapshot_submit_restore(
        ggml_metal_snapshot_staging_t            staging,
        const void *                             source,
        size_t                                   source_size,
        const struct ggml_metal_snapshot_copy *  copies,
        size_t                                   copy_count,
        struct ggml_metal_snapshot_token *       token) {
    return ggml_metal_snapshot_submit(
        staging, GGML_METAL_SNAPSHOT_DIRECTION_RESTORE, source, source_size, copies, copy_count, token);
}

enum ggml_metal_snapshot_status ggml_metal_snapshot_poll(
        ggml_metal_snapshot_staging_t             staging,
        struct ggml_metal_snapshot_token          token,
        struct ggml_metal_snapshot_poll_result *  result) {
    if (staging == NULL || result == NULL || token.slot >= GGML_METAL_SNAPSHOT_STAGING_SLOTS ||
        token.generation == 0) {
        return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&staging->mutex);
    const struct ggml_metal_snapshot_slot * slot = &staging->slots[token.slot];
    if (slot->state == GGML_METAL_SNAPSHOT_SLOT_FREE || slot->generation != token.generation) {
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_STALE_TRANSFER;
    }
    if (slot->state == GGML_METAL_SNAPSHOT_SLOT_IN_FLIGHT) {
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_NOT_READY;
    }
    *result = (struct ggml_metal_snapshot_poll_result) {
        slot->direction,
        slot->state == GGML_METAL_SNAPSHOT_SLOT_COMPLETE &&
                slot->direction == GGML_METAL_SNAPSHOT_DIRECTION_READBACK ?
            [slot->buffer contents] : NULL,
        slot->size,
        slot->command_buffer_status,
        slot->command_error,
    };
    const enum ggml_metal_snapshot_status status = slot->state == GGML_METAL_SNAPSHOT_SLOT_COMPLETE ?
                                                        GGML_METAL_SNAPSHOT_STATUS_OK :
                                                        GGML_METAL_SNAPSHOT_STATUS_COMMAND_ERROR;
    pthread_mutex_unlock(&staging->mutex);
    return status;
}

enum ggml_metal_snapshot_status ggml_metal_snapshot_release(
        ggml_metal_snapshot_staging_t    staging,
        struct ggml_metal_snapshot_token token) {
    if (staging == NULL || token.slot >= GGML_METAL_SNAPSHOT_STAGING_SLOTS || token.generation == 0) {
        return GGML_METAL_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&staging->mutex);
    struct ggml_metal_snapshot_slot * slot = &staging->slots[token.slot];
    if (slot->state == GGML_METAL_SNAPSHOT_SLOT_FREE || slot->generation != token.generation) {
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_STALE_TRANSFER;
    }
    if (slot->state == GGML_METAL_SNAPSHOT_SLOT_IN_FLIGHT) {
        pthread_mutex_unlock(&staging->mutex);
        return GGML_METAL_SNAPSHOT_STATUS_NOT_READY;
    }
    ggml_metal_snapshot_reset_slot(staging, slot);
    pthread_mutex_unlock(&staging->mutex);
    return GGML_METAL_SNAPSHOT_STATUS_OK;
}

struct ggml_metal_snapshot_stats ggml_metal_snapshot_get_stats(
        ggml_metal_snapshot_staging_t staging) {
    struct ggml_metal_snapshot_stats result = { 0 };
    if (staging == NULL) {
        return result;
    }
    pthread_mutex_lock(&staging->mutex);
    result = staging->stats;
    pthread_mutex_unlock(&staging->mutex);
    return result;
}

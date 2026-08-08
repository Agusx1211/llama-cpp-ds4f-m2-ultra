#import "ggml-metal-device.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define GGML_METAL_PROFILE_UNLIKELY(x) __builtin_expect(!!(x), 0)

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

struct ggml_metal_library {
    id<MTLLibrary> obj;

    ggml_metal_device_t dev;
    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if a precompiled library is embedded
    // - otherwise check if the library is in the bundle
    // - if not found, load and compile the source
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metallib\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        // The embedded bytes live for the process lifetime, so the dispatch
        // object must not copy or release their backing storage.
        dispatch_data_t lib_data = dispatch_data_create(
            ggml_metallib_start,
            ggml_metallib_end - ggml_metallib_start,
            dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
            ^{});
        library = [device newLibraryWithData:lib_data error:&error];
        dispatch_release(lib_data);

        if (!library || error) {
            GGML_LOG_ERROR("%s: error loading embedded metallib: %s\n", __func__, [[error description] UTF8String]);
            return nil;
        }
#else
        NSString * src = nil;

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif // GGML_METAL_EMBED_LIBRARY

#if !GGML_METAL_EMBED_LIBRARY
        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

                if (ggml_metal_device_get_props(dev)->has_tensor) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
                }

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj       = library;
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->obj       = library;
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
    return lib->dev;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [lib->obj newFunctionWithName:base_func];
        } else {
            mtl_function = [lib->obj newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

enum ggml_metal_encoder_profile_opcode {
    GGML_METAL_ENCODER_PROFILE_PIPELINE,
    GGML_METAL_ENCODER_PROFILE_SCALAR,
    GGML_METAL_ENCODER_PROFILE_BUFFER,
    GGML_METAL_ENCODER_PROFILE_THREADGROUP_MEMORY,
    GGML_METAL_ENCODER_PROFILE_DISPATCH,
    GGML_METAL_ENCODER_PROFILE_BARRIER,
};

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

struct ggml_metal_encoder_scalar_record {
    uint32_t command_ordinal;
    int32_t argument_index;
    uint32_t size;
    uint32_t data_offset;
};

struct ggml_metal_encoder_scalar_capture {
    uint64_t thread_id;
    uint32_t n_records;
    uint32_t n_bytes;
    uint32_t truncated;
    uint32_t reserved;
    struct ggml_metal_encoder_scalar_record * records;
    uint8_t * bytes;
};

void ggml_metal_encoder_profile_begin(bool capture_scalar_delta);
struct ggml_metal_encoder_profile_result ggml_metal_encoder_profile_end(
        struct ggml_metal_encoder_scalar_capture * scalar_capture);

struct ggml_metal_encoder_profile_tls {
    bool enabled;
    uint64_t sequence;
    struct ggml_metal_encoder_profile_result result;
    struct ggml_metal_encoder_scalar_capture scalar_capture;
};

static _Thread_local struct ggml_metal_encoder_profile_tls ggml_metal_encoder_profile;

static const uint64_t GGML_METAL_ENCODER_PROFILE_FNV_OFFSET = 14695981039346656037ULL;
static const uint64_t GGML_METAL_ENCODER_PROFILE_FNV_PRIME  = 1099511628211ULL;

// The observed DSV4 worker segment records about 4k scalar commands and 400 KiB
// of scalar data. Keep the one-run diagnostic bounded well above that workload.
static const uint32_t GGML_METAL_ENCODER_PROFILE_MAX_SCALARS      = 16384;
static const uint32_t GGML_METAL_ENCODER_PROFILE_MAX_SCALAR_BYTES = 1024*1024;

static void ggml_metal_encoder_profile_hash(uint64_t * hash, const void * data, size_t size) {
    const uint8_t * bytes = data;

    for (size_t i = 0; i < size; ++i) {
        *hash ^= bytes[i];
        *hash *= GGML_METAL_ENCODER_PROFILE_FNV_PRIME;
    }
}

static void ggml_metal_encoder_profile_command(
        struct ggml_metal_encoder_profile_tls * profile,
        enum ggml_metal_encoder_profile_opcode opcode,
        uint64_t * category_fingerprint) {
    const uint64_t sequence = profile->sequence++;

    ggml_metal_encoder_profile_hash(&profile->result.fingerprint, &opcode, sizeof(opcode));
    ggml_metal_encoder_profile_hash(&profile->result.fingerprint, &sequence, sizeof(sequence));
    ggml_metal_encoder_profile_hash(category_fingerprint, &opcode, sizeof(opcode));
    ggml_metal_encoder_profile_hash(category_fingerprint, &sequence, sizeof(sequence));

    profile->result.n_commands++;
}

static void ggml_metal_encoder_profile_value(
        struct ggml_metal_encoder_profile_tls * profile,
        uint64_t * category_fingerprint,
        const void * data,
        size_t size) {
    ggml_metal_encoder_profile_hash(&profile->result.fingerprint, data, size);
    ggml_metal_encoder_profile_hash(category_fingerprint, data, size);
}

static void ggml_metal_encoder_profile_record_scalar(
        struct ggml_metal_encoder_profile_tls * profile,
        const void * data,
        size_t size,
        int idx) {
    struct ggml_metal_encoder_scalar_capture * capture = &profile->scalar_capture;
    if (capture->records == NULL || capture->bytes == NULL || capture->truncated) {
        return;
    }

    if (profile->sequence > UINT32_MAX ||
            capture->n_records >= GGML_METAL_ENCODER_PROFILE_MAX_SCALARS ||
            size > GGML_METAL_ENCODER_PROFILE_MAX_SCALAR_BYTES ||
            capture->n_bytes > GGML_METAL_ENCODER_PROFILE_MAX_SCALAR_BYTES - size) {
        capture->truncated = true;
        return;
    }

    struct ggml_metal_encoder_scalar_record * record = &capture->records[capture->n_records++];
    record->command_ordinal = (uint32_t) profile->sequence;
    record->argument_index = idx;
    record->size = (uint32_t) size;
    record->data_offset = capture->n_bytes;

    memcpy(capture->bytes + capture->n_bytes, data, size);
    capture->n_bytes += (uint32_t) size;
}

void ggml_metal_encoder_profile_begin(bool capture_scalar_delta) {
    GGML_ASSERT(!ggml_metal_encoder_profile.enabled);

    memset(&ggml_metal_encoder_profile, 0, sizeof(ggml_metal_encoder_profile));
    ggml_metal_encoder_profile.enabled = true;

    struct ggml_metal_encoder_profile_result * result = &ggml_metal_encoder_profile.result;
    result->fingerprint                    = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->pipeline_fingerprint           = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->scalar_fingerprint             = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->buffer_fingerprint             = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->threadgroup_memory_fingerprint = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->dispatch_fingerprint           = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;
    result->barrier_fingerprint            = GGML_METAL_ENCODER_PROFILE_FNV_OFFSET;

    if (capture_scalar_delta) {
        struct ggml_metal_encoder_scalar_capture * capture = &ggml_metal_encoder_profile.scalar_capture;
        capture->thread_id = (uint64_t) (uintptr_t) pthread_self();
        capture->records = malloc(
                GGML_METAL_ENCODER_PROFILE_MAX_SCALARS*sizeof(struct ggml_metal_encoder_scalar_record));
        capture->bytes = malloc(GGML_METAL_ENCODER_PROFILE_MAX_SCALAR_BYTES);
        if (capture->records == NULL || capture->bytes == NULL) {
            free(capture->records);
            free(capture->bytes);
            capture->records = NULL;
            capture->bytes = NULL;
            capture->truncated = true;
        }
    }
}

struct ggml_metal_encoder_profile_result ggml_metal_encoder_profile_end(
        struct ggml_metal_encoder_scalar_capture * scalar_capture) {
    GGML_ASSERT(ggml_metal_encoder_profile.enabled);
    GGML_ASSERT(scalar_capture != NULL);

    ggml_metal_encoder_profile.enabled = false;

    *scalar_capture = ggml_metal_encoder_profile.scalar_capture;
    memset(&ggml_metal_encoder_profile.scalar_capture, 0, sizeof(ggml_metal_encoder_profile.scalar_capture));

    return ggml_metal_encoder_profile.result;
}

struct ggml_metal_encoder_profile_pipeline_command {
    uint8_t opcode;
    uintptr_t pipeline;
};

struct ggml_metal_encoder_profile_scalar_command {
    uint8_t opcode;
    int idx;
    size_t size;
};

struct ggml_metal_encoder_profile_buffer_command {
    uint8_t opcode;
    int idx;
    uintptr_t buffer;
    size_t offset;
};

struct ggml_metal_encoder_profile_threadgroup_memory_command {
    uint8_t opcode;
    int idx;
    size_t size;
};

struct ggml_metal_encoder_profile_dispatch_command {
    uint8_t opcode;
    int dimensions[6];
};

struct ggml_metal_encoder_profile_barrier_command {
    uint8_t opcode;
};

//
// GGML_METAL_KPROF - per-segment GPU timestamps via stage-boundary counter sampling
//

// the device caps one counter sample buffer at 32768 B = 4096 timestamps = 2048
// segments (measured: newCounterSampleBufferWithDescriptor rejects anything
// larger with "Expected range: 8 -> 32768"), so a batch chains several of them
#define GGML_METAL_KPROF_SEG_PER_SB 2048
#define GGML_METAL_KPROF_MAX_SB     32
#define GGML_METAL_KPROF_MAX_SEGMENTS (GGML_METAL_KPROF_SEG_PER_SB*GGML_METAL_KPROF_MAX_SB)

struct ggml_metal_kprof_batch {
    id<MTLCounterSampleBuffer> sb[GGML_METAL_KPROF_MAX_SB];
    int    n_sb;
    int  * nodes;   // raw graph node index that starts each segment
    int    n_seg;
    int    cap;
    struct ggml_metal_kprof_batch * next;
};

static id<MTLCounterSampleBuffer> ggml_metal_kprof_new_sb(id<MTLDevice> dev) {
    id<MTLCounterSampleBuffer> sb = nil;

    @autoreleasepool {
        MTLCounterSampleBufferDescriptor * d = [[MTLCounterSampleBufferDescriptor alloc] init];

        for (id<MTLCounterSet> cs in [dev counterSets]) {
            if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) {
                d.counterSet = cs;
                break;
            }
        }

        d.sampleCount = 2*GGML_METAL_KPROF_SEG_PER_SB;
        d.storageMode = MTLStorageModeShared;
        d.label       = @"ggml-kprof";

        NSError * err = nil;
        sb = d.counterSet ? [dev newCounterSampleBufferWithDescriptor:d error:&err] : nil;

        if (!sb) {
            fprintf(stderr, "%s: kprof: newCounterSampleBuffer failed (%s)\n", __func__,
                    err ? [[err localizedDescription] UTF8String] : "no timestamp counter set");
        }

        [d release];
    }

    return sb;
}

static struct {
    int    stride;          // 0 = disabled
    bool   initialized;
    _Atomic int seq;        // monotonically increasing flush sequence
    struct ggml_metal_kprof_batch * pending;
    pthread_mutex_t mtx;
} g_kprof = {
    /*.stride      =*/ 0,
    /*.initialized =*/ false,
    /*.seq         =*/ 0,
    /*.pending     =*/ NULL,
    /*.mtx         =*/ PTHREAD_MUTEX_INITIALIZER,
};

int ggml_metal_kprof_stride(void) {
    if (!g_kprof.initialized) {
        // benign race: every writer computes the same value
        const char * v = getenv("GGML_METAL_KPROF");
        g_kprof.stride      = v ? atoi(v) : 0;
        if (g_kprof.stride < 0) {
            g_kprof.stride = 0;
        }
        g_kprof.initialized = true;
    }

    return g_kprof.stride;
}

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;

    // Points to thread-local profiling state only while an opt-in recording
    // scope is active. It does not retain any Metal object.
    struct ggml_metal_encoder_profile_tls * profile;

    // kprof state (NULL/nil unless GGML_METAL_KPROF is active)
    id<MTLCommandBuffer> cmd_buf;
    bool concurrent;
    struct ggml_metal_kprof_batch * kprof;
};

static void ggml_metal_encoder_kprof_begin(ggml_metal_encoder_t encoder, int raw_node_idx) {
    struct ggml_metal_kprof_batch * b = encoder->kprof;

    const int isb = b->n_seg / GGML_METAL_KPROF_SEG_PER_SB;
    const int islot = b->n_seg % GGML_METAL_KPROF_SEG_PER_SB;

    if (b->n_seg < b->cap && islot == 0 && isb == b->n_sb) {
        id<MTLCounterSampleBuffer> sb = ggml_metal_kprof_new_sb([encoder->cmd_buf device]);
        if (sb) {
            b->sb[b->n_sb++] = sb;
        }
    }

    @autoreleasepool {
        if (b->n_seg >= b->cap || isb >= b->n_sb) {
            // out of slots: fall back to a plain pass so encoding still completes
            if (encoder->concurrent) {
                encoder->obj = [encoder->cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
            } else {
                encoder->obj = [encoder->cmd_buf computeCommandEncoder];
            }
            [encoder->obj retain];
            return;
        }

        MTLComputePassDescriptor * desc = [MTLComputePassDescriptor computePassDescriptor];

        desc.dispatchType = encoder->concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

        desc.sampleBufferAttachments[0].sampleBuffer               = b->sb[isb];
        desc.sampleBufferAttachments[0].startOfEncoderSampleIndex  = 2*islot + 0;
        desc.sampleBufferAttachments[0].endOfEncoderSampleIndex    = 2*islot + 1;

        b->nodes[b->n_seg] = raw_node_idx;
        b->n_seg++;

        encoder->obj = [encoder->cmd_buf computeCommandEncoderWithDescriptor:desc];
        [encoder->obj retain];
    }
}

int ggml_metal_encoder_kprof_split(ggml_metal_encoder_t encoder, int raw_node_idx) {
    if (encoder->kprof == NULL) {
        return -1;
    }

    [encoder->obj endEncoding];
    [encoder->obj release];

    const int seg = encoder->kprof->n_seg;

    ggml_metal_encoder_kprof_begin(encoder, raw_node_idx);

    return seg;
}

void ggml_metal_kprof_flush(void) {
    if (ggml_metal_kprof_stride() == 0) {
        return;
    }

    pthread_mutex_lock(&g_kprof.mtx);
    struct ggml_metal_kprof_batch * head = g_kprof.pending;
    g_kprof.pending = NULL;
    pthread_mutex_unlock(&g_kprof.mtx);

    if (head == NULL) {
        return;
    }

    const int seq = g_kprof.seq++;

    int ibatch = 0;

    while (head) {
        struct ggml_metal_kprof_batch * b = head;
        head = b->next;
        const int batch = ibatch++;

        for (int isb = 0; isb < b->n_sb; ++isb) {
            const int seg0 = isb*GGML_METAL_KPROF_SEG_PER_SB;
            const int nseg = MIN(b->n_seg - seg0, GGML_METAL_KPROF_SEG_PER_SB);

            if (nseg <= 0) {
                break;
            }

            NSData * data = [b->sb[isb] resolveCounterRange:NSMakeRange(0, 2*nseg)];

            if (data && [data length] >= 2*(NSUInteger)nseg*sizeof(MTLCounterResultTimestamp)) {
                const MTLCounterResultTimestamp * ts = (const MTLCounterResultTimestamp *) [data bytes];

                for (int i = 0; i < nseg; ++i) {
                    const uint64_t t0 = ts[2*i + 0].timestamp;
                    const uint64_t t1 = ts[2*i + 1].timestamp;

                    if (t0 == MTLCounterErrorValue || t1 == MTLCounterErrorValue || t1 < t0) {
                        continue;
                    }

                    fprintf(stderr, "KPROF {\"seq\":%d,\"b\":%d,\"seg\":%d,\"node\":%d,\"t0\":%llu,\"ns\":%llu}\n",
                            seq, batch, seg0 + i, b->nodes[seg0 + i],
                            (unsigned long long) t0, (unsigned long long) (t1 - t0));
                }
            } else {
                fprintf(stderr, "KPROF {\"seq\":%d,\"b\":%d,\"error\":\"resolve failed\",\"sb\":%d,\"n_seg\":%d}\n",
                        seq, batch, isb, nseg);
            }

            [b->sb[isb] release];
        }

        free(b->nodes);
        free(b);
    }

    fflush(stderr);
}

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    res->cmd_buf    = cmd_buf;
    res->concurrent = concurrent;

    if (ggml_metal_kprof_stride() > 0) {
        struct ggml_metal_kprof_batch * b = calloc(1, sizeof(struct ggml_metal_kprof_batch));

        b->cap   = GGML_METAL_KPROF_MAX_SEGMENTS;
        b->nodes = calloc(b->cap, sizeof(int));
        b->n_seg = 0;
        b->n_sb  = 0;

        res->kprof = b;

        ggml_metal_encoder_kprof_begin(res, -1);

        if (b->n_sb == 0) {
            // counter sample buffers unavailable: kprof_begin already opened a
            // plain compute pass, so only the bookkeeping has to be undone
            free(b->nodes);
            free(b);
            res->kprof = NULL;
        }

        res->profile = ggml_metal_encoder_profile.enabled ? &ggml_metal_encoder_profile : NULL;

        return res;
    }

    {
        if (concurrent) {
            res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
        } else {
            res->obj = [cmd_buf computeCommandEncoder];
        }

        [res->obj retain];
    }

    res->profile = ggml_metal_encoder_profile.enabled ? &ggml_metal_encoder_profile : NULL;

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    if (encoder->kprof) {
        struct ggml_metal_kprof_batch * b = encoder->kprof;

        pthread_mutex_lock(&g_kprof.mtx);
        b->next = g_kprof.pending;
        g_kprof.pending = b;
        pthread_mutex_unlock(&g_kprof.mtx);

        encoder->kprof = NULL;
    }

    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.pipeline_fingerprint;
        const uintptr_t pipeline_obj = (uintptr_t) pipeline.pipeline->obj;

        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_PIPELINE, fingerprint);
        ggml_metal_encoder_profile_value(profile, fingerprint, &pipeline_obj, sizeof(pipeline_obj));

        profile->result.n_pipelines++;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_pipeline_command);
    }

    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.scalar_fingerprint;

        ggml_metal_encoder_profile_record_scalar(profile, data, size, idx);
        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_SCALAR, fingerprint);
        ggml_metal_encoder_profile_value(profile, fingerprint, &idx, sizeof(idx));
        ggml_metal_encoder_profile_value(profile, fingerprint, &size, sizeof(size));
        ggml_metal_encoder_profile_value(profile, fingerprint, data, size);

        profile->result.n_scalars++;
        profile->result.scalar_bytes += size;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_scalar_command) + size;
    }

    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.buffer_fingerprint;
        const uintptr_t buffer_obj = (uintptr_t) buffer.metal;

        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_BUFFER, fingerprint);
        ggml_metal_encoder_profile_value(profile, fingerprint, &idx, sizeof(idx));
        ggml_metal_encoder_profile_value(profile, fingerprint, &buffer_obj, sizeof(buffer_obj));
        ggml_metal_encoder_profile_value(profile, fingerprint, &buffer.offs, sizeof(buffer.offs));

        profile->result.n_buffers++;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_buffer_command);
    }

    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.threadgroup_memory_fingerprint;

        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_THREADGROUP_MEMORY, fingerprint);
        ggml_metal_encoder_profile_value(profile, fingerprint, &idx, sizeof(idx));
        ggml_metal_encoder_profile_value(profile, fingerprint, &size, sizeof(size));

        profile->result.n_threadgroup_memories++;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_threadgroup_memory_command);
    }

    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.dispatch_fingerprint;
        const int dimensions[6] = { tg0, tg1, tg2, tptg0, tptg1, tptg2 };

        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_DISPATCH, fingerprint);
        ggml_metal_encoder_profile_value(profile, fingerprint, dimensions, sizeof(dimensions));

        profile->result.n_dispatches++;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_dispatch_command);
    }

    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    if (GGML_METAL_PROFILE_UNLIKELY(encoder->profile != NULL)) {
        struct ggml_metal_encoder_profile_tls * profile = encoder->profile;
        uint64_t * fingerprint = &profile->result.barrier_fingerprint;

        ggml_metal_encoder_profile_command(profile, GGML_METAL_ENCODER_PROFILE_BARRIER, fingerprint);

        profile->result.n_barriers++;
        profile->result.projected_plan_bytes += sizeof(struct ggml_metal_encoder_profile_barrier_command);
    }

    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;
    int loops_per_s;
    int time_per_loop_ms;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
static void ggml_metal_dummy_work(ggml_metal_device_t dev) {
    if (dev->mtl_queue == nil) {
        return;
    }

    @autoreleasepool {
        // perform a minimal dummy operation on the GPU
        id<MTLBuffer> buf = [dev->mtl_device newBufferWithLength:1 options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf range:NSMakeRange(0, 1) value:0];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [buf release];
    }
}
#endif

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    res->time_per_loop_ms = 5;
    res->loops_per_s = 1000/res->time_per_loop_ms;

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  usleep(res->time_per_loop_ms * 1000);
              }
        }
#endif
    });

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        // workaround for residency set memory not being released if no GPU operation occurs
        // https://developer.apple.com/forums/thread/839089
        // https://github.com/ggml-org/llama.cpp/issues/25937
        ggml_metal_dummy_work(dev);
    }
#endif

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

static enum ggml_metal_device_id ggml_metal_device_id_parse(const char * name) {
    if (!name) {
        return GGML_METAL_DEVICE_GENERIC;
    }

    static const char prefix[] = "Apple ";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return GGML_METAL_DEVICE_GENERIC;
    }
    const char * suffix = name + sizeof(prefix) - 1;

    static const struct {
        const char * name;
        enum ggml_metal_device_id id;
    } table[] = {
        {"M1",       GGML_METAL_DEVICE_M1},
        {"M1 Pro",   GGML_METAL_DEVICE_M1_PRO},
        {"M1 Max",   GGML_METAL_DEVICE_M1_MAX},
        {"M1 Ultra", GGML_METAL_DEVICE_M1_ULTRA},
        {"M2",       GGML_METAL_DEVICE_M2},
        {"M2 Pro",   GGML_METAL_DEVICE_M2_PRO},
        {"M2 Max",   GGML_METAL_DEVICE_M2_MAX},
        {"M2 Ultra", GGML_METAL_DEVICE_M2_ULTRA},
        {"M3",       GGML_METAL_DEVICE_M3},
        {"M3 Pro",   GGML_METAL_DEVICE_M3_PRO},
        {"M3 Max",   GGML_METAL_DEVICE_M3_MAX},
        {"M3 Ultra", GGML_METAL_DEVICE_M3_ULTRA},
        {"M4",       GGML_METAL_DEVICE_M4},
        {"M4 Pro",   GGML_METAL_DEVICE_M4_PRO},
        {"M4 Max",   GGML_METAL_DEVICE_M4_MAX},
        {"M5",       GGML_METAL_DEVICE_M5},
        {"M5 Pro",   GGML_METAL_DEVICE_M5_PRO},
        {"M5 Max",   GGML_METAL_DEVICE_M5_MAX},
        {"M5 Ultra", GGML_METAL_DEVICE_M5_ULTRA},
    };

    for (size_t i = 0; i < sizeof(table)/sizeof(table[0]); ++i) {
        if (strcmp(suffix, table[i].name) == 0) {
            return table[i].id;
        }
    }
    return GGML_METAL_DEVICE_GENERIC;
}

ggml_metal_device_t ggml_metal_device_init(int device) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
        dev->mtl_device = MTLCreateSystemDefaultDevice();

        if (dev->mtl_device) {
            dev->mtl_queue = [dev->mtl_device newCommandQueue];
            if (dev->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            dev->addr_virt = 0x000000400ULL;

            dev->props.device = device;
            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
            if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                dev->props.has_bfloat = false;
            }

            dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                dev->props.has_tensor = false;
            }

            dev->props.has_placement_sparse = false;
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
            if (@available(macOS 26.4, *)) {
                dev->props.has_placement_sparse = dev->mtl_device.supportsPlacementSparse;
            }
#endif

            // note: disable the tensor API by default for old chips because with the current implementation it is not useful
            // - M2 Ultra:   ~5% slower
            // - M4, M4 Max: no significant difference
            //
            // TODO: try to update the tensor API kernels to at least match the simdgroup performance
            if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                ![[dev->mtl_device name] containsString:@"M5"] &&
                ![[dev->mtl_device name] containsString:@"M6"] &&
                ![[dev->mtl_device name] containsString:@"A19"] &&
                ![[dev->mtl_device name] containsString:@"A20"]) {
                GGML_LOG_INFO("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                dev->props.has_tensor = false;
            }

            // double-check that the tensor API compiles
            if (dev->props.has_tensor) {
                const char * src_tensor_f16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                    dev->props.has_tensor = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
            if (dev->props.has_tensor && dev->props.has_bfloat) {
                const char * src_tensor_bf16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                    dev->props.has_bfloat = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            // In case of eGPU, shared memory may be preferable.
            dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.device_id = ggml_metal_device_id_parse([[dev->mtl_device name] UTF8String]);

            dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
            if (@available(macOS 10.12, iOS 16.0, *)) {
                dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
            } else {
                dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
            }

            snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
            snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", [[dev->mtl_device name] UTF8String]);

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            if (dev->props.use_residency_sets) {
                dev->rsets = ggml_metal_rsets_init(dev);
            } else {
                dev->rsets = nil;
            }

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: placement sparse      = %s\n", __func__, dev->props.has_placement_sparse    ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    ggml_metal_rsets_free(dev->rsets);

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    [event release];

    free(ev);

    GGML_UNUSED(dev);
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
    if (!res) {
        GGML_ABORT("%s: failed to wait for event\n", __func__);
    }

    GGML_UNUSED(dev);
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_SILU_BACK:
            return (op->src[0]->type == GGML_TYPE_F32) &&
                (op->src[1]->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F32) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_are_same_shape(op->src[0], op->src[1]);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONCAT:
            {
                const enum ggml_type src0_type = op->src[0]->type;
                const enum ggml_type src1_type = op->src[1]->type;
                if (src0_type != src1_type || src0_type != op->type) {
                    return false;
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_I8:
                    case GGML_TYPE_I16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_I64:
                        return true;
                    case GGML_TYPE_BF16:
                        return has_bfloat;
                    default:
                        return false;
                }
            }
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) && (op->src[0]->type == op->src[1]->type);
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_COL2IM_1D:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                op->type == op->src[0]->type &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op);
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D_DW:
            return op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }

            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
        case GGML_OP_ROLL:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    break;
                case GGML_TYPE_BF16:
                    if (!has_bfloat) {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_DSV4_HC_COMB:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == 24 &&
                op->src[1]->ne[0] >= 3 &&
                op->src[2]->ne[0] == 24 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]);
        case GGML_OP_DSV4_COMPRESS:
            return op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_I32 &&
                op->type         == GGML_TYPE_F32 &&
                (ggml_get_op_params_i32(op, 1) ? 2 : 1)*ggml_get_op_params_i32(op, 0) <= 128 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous(op->src[2]);
        case GGML_OP_DSV4_TOP_K_MASK:
            return op->src[0]->type == GGML_TYPE_F16 &&
                op->src[1]->type == GGML_TYPE_F16 &&
                op->src[2]->type == GGML_TYPE_I32 &&
                op->type         == GGML_TYPE_F16 &&
                op->src[2]->ne[0] <= op->src[1]->ne[0] &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op);
        case GGML_OP_DSV4_HC_PRE:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[1] == 4 &&
                op->src[1]->ne[0] == 4 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]);
        case GGML_OP_DSV4_HC_POST:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[1]->ne[1] == 4 &&
                op->src[2]->ne[0] == 4 &&
                op->src[3]->ne[0] == 4 &&
                op->src[3]->ne[1] == 4 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]) &&
                ggml_is_contiguous_rows(op->src[3]);
        case GGML_OP_DSV4_SPARSE_PACK:
            {
                const bool indexed = op->src[5] != NULL;
                const bool common =
                    (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_Q8_0) &&
                    op->src[1]->type == op->src[0]->type &&
                    op->src[2]->type == GGML_TYPE_F16 &&
                    op->src[3]->type == GGML_TYPE_F16 &&
                    op->src[4]->type == GGML_TYPE_I32 &&
                    op->type         == GGML_TYPE_F16 &&
                    op->src[0]->ne[0] == 512 &&
                    op->src[0]->ne[2] <= INT32_MAX &&
                    op->src[3]->ne[0] > 0 && op->src[3]->ne[0] <= INT32_MAX &&
                    op->src[4]->ne[0] <= op->src[3]->ne[0] &&
                    ggml_get_op_params_i32(op, 0) <= 128 &&
                    ggml_get_op_params_i32(op, 0) + op->src[4]->ne[0] <= 128 + 512 &&
                    ggml_is_contiguous_rows(op->src[0]) &&
                    ggml_is_contiguous_rows(op->src[1]) &&
                    ggml_is_contiguous_rows(op->src[2]) &&
                    ggml_is_contiguous_rows(op->src[3]) &&
                    ggml_is_contiguous_rows(op->src[4]) &&
                    ggml_is_contiguous(op);
                if (!common) {
                    return false;
                }
                if (!indexed) {
                    return op->src[1]->ne[1] == 1 &&
                        op->src[1]->ne[2] == op->src[3]->ne[0] &&
                        op->src[1]->ne[3] == op->src[0]->ne[3];
                }
                return op->src[5]->type == GGML_TYPE_I32 &&
                    op->src[1]->ne[1] > 0 && op->src[1]->ne[1] % 64 == 0 &&
                    op->src[1]->ne[1]/64 <= INT32_MAX &&
                    op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
                    op->src[5]->ne[0] == (op->src[3]->ne[0] + 63)/64 &&
                    op->src[5]->ne[1] == op->src[0]->ne[3] &&
                    op->src[5]->ne[2] == 1 && op->src[5]->ne[3] == 1 &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op->src[5]);
            }
        case GGML_OP_DSV4_INDEXED_CONCAT:
            {
                const int64_t n_comp = ggml_get_op_params_i32(op, 0);
                return op->src[0]->type == GGML_TYPE_F16 &&
                    op->src[1]->type == GGML_TYPE_F16 &&
                    op->src[2]->type == GGML_TYPE_I32 &&
                    op->type         == GGML_TYPE_F16 &&
                    op->src[0]->ne[0] > 0 && op->src[0]->ne[0] % 4 == 0 &&
                    op->src[0]->ne[0]/4 <= INT32_MAX &&
                    op->src[0]->ne[1] == 1 && op->src[0]->ne[3] > 0 &&
                    op->src[0]->ne[2] <= INT32_MAX &&
                    op->src[1]->ne[0] == op->src[0]->ne[0] &&
                    op->src[1]->ne[1] > 0 && op->src[1]->ne[1] % 64 == 0 &&
                    op->src[1]->ne[1]/64 <= INT32_MAX &&
                    op->src[1]->ne[2] == 1 && op->src[1]->ne[3] == 1 &&
                    n_comp >= 0 && op->src[0]->ne[2] <= INT32_MAX - n_comp &&
                    op->src[2]->ne[0] == (n_comp + 63)/64 &&
                    op->src[2]->ne[1] == op->src[0]->ne[3] &&
                    op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1 &&
                    op->ne[0] == op->src[0]->ne[0] && op->ne[1] == 1 &&
                    op->ne[2] == op->src[0]->ne[2] + n_comp &&
                    op->ne[3] == op->src[0]->ne[3] &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op->src[1]) &&
                    ggml_is_contiguous(op->src[2]) &&
                    ggml_is_contiguous(op);
            }
        case GGML_OP_LIGHTNING_INDEXER:
            return has_simdgroup_mm &&
                op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_Q8_0) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == 128 &&
                op->src[0]->ne[1] == 64 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]) &&
                ggml_is_contiguous_rows(op->src[3]);
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0;
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            // fork: E4M3_M2/NF8_M2 have plain MUL_MAT kernels only (mirrors of
            // the BF16 kernels; require bfloat + simdgroup mm for the exact
            // mul_mm path)
            if (op->src[0]->type == GGML_TYPE_E4M3_M2 || op->src[0]->type == GGML_TYPE_NF8_M2) {
                return op->op == GGML_OP_MUL_MAT &&
                    has_simdgroup_reduction && has_simdgroup_mm && has_bfloat &&
                    op->src[1]->type == GGML_TYPE_F32;
            }
            return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q1_0:
                           case GGML_TYPE_Q2_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            // fork: no get_rows kernels for the gguf-m2 types (dense plane is
            // only ever a MUL_MAT operand, expert plane only a MUL_MAT_ID one)
            return op->src[0]->type != GGML_TYPE_NVFP4 &&
                   op->src[0]->type != GGML_TYPE_E4M3_M2 &&
                   op->src[0]->type != GGML_TYPE_MXFP4_M2 &&
                   op->src[0]->type != GGML_TYPE_NF8_M2;
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type == GGML_TYPE_F16) {
                    return op->type == GGML_TYPE_F16;
                }

                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool is_sparse;
    bool owned;

    // DSV4 placement-sparse buffers retain the ordinary affine virtual
    // address layout while committing only a bounded pool of 64 KiB tiles.
    // The CPU tables implement aliasing and copy-on-write for sequence copies.
    id sparse_heap;
    id sparse_queue;
    id sparse_event;
    NSLock * sparse_lock;
    uint64_t sparse_event_value;
    size_t sparse_page_size;
    size_t sparse_n_virtual;
    size_t sparse_n_physical;
    size_t sparse_n_free;
    size_t sparse_n_reserved;
    uint64_t sparse_generation;
    uint64_t sparse_cow_allocations;
    uint64_t sparse_cow_pages;
    uint32_t * sparse_v2p;
    uint32_t * sparse_p_ref;
    uint32_t * sparse_free;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers + (buf->is_sparse ? 1 : 0);

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }
        if (buf->is_sparse) {
            [buf->rset addAllocation:buf->sparse_heap];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset commit];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

static const char * ggml_metal_sparse_init_status_name(enum ggml_metal_sparse_init_status status) {
    switch (status) {
        case GGML_METAL_SPARSE_INIT_OK:                     return "ok";
        case GGML_METAL_SPARSE_INIT_UNSUPPORTED:            return "placement-sparse unsupported";
        case GGML_METAL_SPARSE_INIT_INVALID_SIZE:           return "invalid size";
        case GGML_METAL_SPARSE_INIT_CPU_BUFFER:             return "CPU buffer wrapper";
        case GGML_METAL_SPARSE_INIT_MTL_BUFFER:             return "MTLBuffer";
        case GGML_METAL_SPARSE_INIT_PLACEMENT_HEAP:         return "placement heap";
        case GGML_METAL_SPARSE_INIT_COMMAND_QUEUE:          return "MTL4 command queue";
        case GGML_METAL_SPARSE_INIT_SHARED_EVENT:           return "shared event";
        case GGML_METAL_SPARSE_INIT_LOCK:                   return "CPU lock";
        case GGML_METAL_SPARSE_INIT_CPU_V2P_TABLE:          return "CPU virtual-to-physical table";
        case GGML_METAL_SPARSE_INIT_CPU_REFCOUNT_TABLE:     return "CPU refcount table";
        case GGML_METAL_SPARSE_INIT_CPU_FREE_TABLE:         return "CPU free-page table";
        case GGML_METAL_SPARSE_INIT_RESIDENCY_SET:          return "residency set";
    }
    return "unknown";
}

ggml_metal_buffer_t ggml_metal_buffer_init_sparse_ex(
        ggml_metal_device_t dev,
        size_t virtual_size,
        size_t physical_size,
        struct ggml_metal_sparse_init_result * result) {
    struct ggml_metal_sparse_init_result local = {
        /*.status                   =*/ GGML_METAL_SPARSE_INIT_UNSUPPORTED,
        /*.requested_virtual_bytes  =*/ virtual_size,
        /*.requested_physical_bytes =*/ physical_size,
        /*.aligned_virtual_bytes    =*/ 0,
        /*.aligned_physical_bytes   =*/ 0,
        /*.device_max_buffer_length =*/ dev != NULL ? (size_t) dev->mtl_device.maxBufferLength : 0,
    };
    if (result != NULL) {
        *result = local;
    }

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        const size_t page_size = 64*1024;
        if (dev == NULL || virtual_size == 0 || physical_size == 0 ||
                virtual_size > SIZE_MAX - (page_size - 1) ||
                physical_size > SIZE_MAX - (page_size - 1)) {
            local.status = GGML_METAL_SPARSE_INIT_INVALID_SIZE;
            goto fail_without_buffer;
        }
        virtual_size  = GGML_PAD(virtual_size,  page_size);
        physical_size = GGML_PAD(physical_size, page_size);
        physical_size = physical_size < virtual_size ? physical_size : virtual_size;
        local.aligned_virtual_bytes = virtual_size;
        local.aligned_physical_bytes = physical_size;

        if (virtual_size > SIZE_MAX - page_size) {
            local.status = GGML_METAL_SPARSE_INIT_INVALID_SIZE;
            goto fail_without_buffer;
        }

        if (!dev->props.has_placement_sparse) {
            local.status = GGML_METAL_SPARSE_INIT_UNSUPPORTED;
            goto fail_without_buffer;
        }

        ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));
        if (res == NULL) {
            local.status = GGML_METAL_SPARSE_INIT_CPU_BUFFER;
            goto fail_without_buffer;
        }

        res->dev                  = dev;
        const uintptr_t addr_raw  = atomic_fetch_add_explicit(
                &dev->addr_virt, virtual_size + page_size, memory_order_relaxed);
        res->all_data             = (void *) GGML_PAD(addr_raw, page_size);
        res->all_size             = virtual_size;
        res->is_shared            = false;
        res->is_sparse            = true;
        res->owned                = true;
        res->n_buffers            = 1;
        res->sparse_page_size     = page_size;
        res->sparse_n_virtual     = virtual_size/page_size;
        res->sparse_n_physical    = physical_size/page_size;
        res->sparse_n_free        = res->sparse_n_physical;
        res->sparse_n_reserved    = 0;
        res->sparse_generation    = 1;
        res->sparse_event_value   = 0;

        res->buffers[0].data = res->all_data;
        res->buffers[0].size = virtual_size;
        res->buffers[0].metal = [dev->mtl_device newBufferWithLength:virtual_size
                                                             options:MTLResourceStorageModePrivate
                                             placementSparsePageSize:MTLSparsePageSize64];
        if (res->buffers[0].metal == nil) {
            local.status = GGML_METAL_SPARSE_INIT_MTL_BUFFER;
            goto fail;
        }

        MTLHeapDescriptor * desc = [[MTLHeapDescriptor alloc] init];
        desc.type = MTLHeapTypePlacement;
        desc.storageMode = MTLStorageModePrivate;
        desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
        desc.maxCompatiblePlacementSparsePageSize = MTLSparsePageSize64;
        desc.size = physical_size;
        res->sparse_heap = [dev->mtl_device newHeapWithDescriptor:desc];
        [desc release];
        if (res->sparse_heap == nil) {
            local.status = GGML_METAL_SPARSE_INIT_PLACEMENT_HEAP;
            goto fail;
        }

        res->sparse_queue = [dev->mtl_device newMTL4CommandQueue];
        if (res->sparse_queue == nil) {
            local.status = GGML_METAL_SPARSE_INIT_COMMAND_QUEUE;
            goto fail;
        }
        res->sparse_event = [dev->mtl_device newSharedEvent];
        if (res->sparse_event == nil) {
            local.status = GGML_METAL_SPARSE_INIT_SHARED_EVENT;
            goto fail;
        }
        res->sparse_lock  = [[NSLock alloc] init];
        if (res->sparse_lock == nil) {
            local.status = GGML_METAL_SPARSE_INIT_LOCK;
            goto fail;
        }

        res->sparse_v2p   = malloc(res->sparse_n_virtual  * sizeof(uint32_t));
        if (res->sparse_v2p == NULL) {
            local.status = GGML_METAL_SPARSE_INIT_CPU_V2P_TABLE;
            goto fail;
        }
        res->sparse_p_ref = calloc(res->sparse_n_physical, sizeof(uint32_t));
        if (res->sparse_p_ref == NULL) {
            local.status = GGML_METAL_SPARSE_INIT_CPU_REFCOUNT_TABLE;
            goto fail;
        }
        res->sparse_free  = malloc(res->sparse_n_physical * sizeof(uint32_t));
        if (res->sparse_free == NULL) {
            local.status = GGML_METAL_SPARSE_INIT_CPU_FREE_TABLE;
            goto fail;
        }

        for (size_t i = 0; i < res->sparse_n_virtual; ++i) {
            res->sparse_v2p[i] = UINT32_MAX;
        }
        for (size_t i = 0; i < res->sparse_n_physical; ++i) {
            // Pop low-numbered heap tiles first so fresh mappings coalesce.
            res->sparse_free[res->sparse_n_physical - 1 - i] = (uint32_t) i;
        }

        res->use_residency_sets = dev->props.use_residency_sets;
        if (!ggml_metal_buffer_rset_init(res)) {
            local.status = GGML_METAL_SPARSE_INIT_RESIDENCY_SET;
            goto fail;
        }

        ggml_metal_device_rsets_add(dev, res->rset);

        local.status = GGML_METAL_SPARSE_INIT_OK;
        if (result != NULL) {
            *result = local;
        }
        GGML_LOG_INFO("%s: DSV4 sparse buffer = %.2f MiB physical / %.2f MiB virtual, page = 64 KiB, maxBufferLength = %.2f MiB\n",
                __func__, physical_size/1024.0/1024.0, virtual_size/1024.0/1024.0,
                local.device_max_buffer_length/1024.0/1024.0);
        return res;

fail:
        GGML_LOG_ERROR("%s: DSV4 sparse init failed at %s: requested virtual=%zu bytes (%.2f MiB), physical=%zu bytes (%.2f MiB); aligned virtual=%zu bytes, physical=%zu bytes; device maxBufferLength=%zu bytes (%.2f MiB)\n",
                __func__, ggml_metal_sparse_init_status_name(local.status),
                local.requested_virtual_bytes, local.requested_virtual_bytes/1024.0/1024.0,
                local.requested_physical_bytes, local.requested_physical_bytes/1024.0/1024.0,
                local.aligned_virtual_bytes, local.aligned_physical_bytes,
                local.device_max_buffer_length, local.device_max_buffer_length/1024.0/1024.0);
        ggml_metal_buffer_rset_free(res);
        [res->buffers[0].metal release];
        [res->sparse_heap release];
        [res->sparse_queue release];
        [res->sparse_event release];
        [res->sparse_lock release];
        free(res->sparse_v2p);
        free(res->sparse_p_ref);
        free(res->sparse_free);
        free(res);
        if (result != NULL) {
            *result = local;
        }
        return NULL;

fail_without_buffer:
        GGML_LOG_ERROR("%s: DSV4 sparse init failed at %s: requested virtual=%zu bytes (%.2f MiB), physical=%zu bytes (%.2f MiB); aligned virtual=%zu bytes, physical=%zu bytes; device maxBufferLength=%zu bytes (%.2f MiB)\n",
                __func__, ggml_metal_sparse_init_status_name(local.status),
                local.requested_virtual_bytes, local.requested_virtual_bytes/1024.0/1024.0,
                local.requested_physical_bytes, local.requested_physical_bytes/1024.0/1024.0,
                local.aligned_virtual_bytes, local.aligned_physical_bytes,
                local.device_max_buffer_length, local.device_max_buffer_length/1024.0/1024.0);
        if (result != NULL) {
            *result = local;
        }
        return NULL;
    }
#endif

    GGML_UNUSED(dev);
    GGML_UNUSED(virtual_size);
    GGML_UNUSED(physical_size);
    return NULL;
}

ggml_metal_buffer_t ggml_metal_buffer_init_sparse(
        ggml_metal_device_t dev,
        size_t virtual_size,
        size_t physical_size) {
    return ggml_metal_buffer_init_sparse_ex(dev, virtual_size, physical_size, NULL);
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
struct ggml_metal_sparse_write_action {
    uint32_t vtile;
    uint32_t ptile;
    uint32_t copy_src_vtile;
};

static void ggml_metal_sparse_submit(
        ggml_metal_buffer_t buf,
        const MTL4UpdateSparseBufferMappingOperation * operations,
        size_t n_operations,
        const struct ggml_metal_sparse_write_action * writes,
        size_t n_writes) {
    GGML_ASSERT(n_operations > 0);

    id<MTL4CommandQueue> sparse_queue = (id<MTL4CommandQueue>) buf->sparse_queue;
    id<MTLSharedEvent> event = (id<MTLSharedEvent>) buf->sparse_event;
    id<MTLHeap> heap = (id<MTLHeap>) buf->sparse_heap;
    id<MTLBuffer> metal = buf->buffers[0].metal;

    // Mapping changes must follow all earlier graph work and precede all later
    // graph work on the backend's legacy queue. Shared events bridge that queue
    // with the MTL4 queue that owns placement-sparse mapping operations.
    const uint64_t before_value = ++buf->sparse_event_value;
    id<MTLCommandBuffer> before = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];
    [before encodeSignalEvent:event value:before_value];
    [before commit];

    [sparse_queue waitForEvent:event value:before_value];
    [sparse_queue updateBufferMappings:metal
                                  heap:heap
                            operations:operations
                                 count:n_operations];

    const uint64_t after_value = ++buf->sparse_event_value;
    [sparse_queue signalEvent:event value:after_value];

    id<MTLCommandBuffer> after = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];
    [after encodeWaitForEvent:event value:after_value];

    if (n_writes > 0) {
        id<MTLBlitCommandEncoder> blit = [after blitCommandEncoder];

        // Fresh heap tiles can contain recycled cache data. Zero newly mapped
        // pages before exposing them to a graph.
        for (size_t i = 0; i < n_writes; ++i) {
            if (writes[i].copy_src_vtile == UINT32_MAX) {
                [blit fillBuffer:metal
                           range:NSMakeRange((size_t) writes[i].vtile*buf->sparse_page_size,
                                             buf->sparse_page_size)
                           value:0];
            }
        }

        // Every COW action reads the stable alias deliberately retained on the
        // original physical page, so copies are independent of action order.
        for (size_t i = 0; i < n_writes; ++i) {
            if (writes[i].copy_src_vtile != UINT32_MAX) {
                [blit copyFromBuffer:metal
                        sourceOffset:(size_t) writes[i].copy_src_vtile*buf->sparse_page_size
                            toBuffer:metal
                   destinationOffset:(size_t) writes[i].vtile*buf->sparse_page_size
                                size:buf->sparse_page_size];
            }
        }

        [blit endEncoding];
    }

    [after commit];
}

struct ggml_metal_sparse_reservation_entry {
    ggml_metal_buffer_t buffer;
    struct ggml_metal_sparse_range * ranges;
    size_t n_ranges;
    uint8_t * marked;
    uint32_t * marked_per_physical;
    uint32_t * retained_by_physical;
    uint32_t * copy_source_by_virtual;
    struct ggml_metal_sparse_quote quote;
    struct ggml_metal_sparse_ticket_accounting accounting;
    struct ggml_metal_sparse_write_action * writes;
    MTL4UpdateSparseBufferMappingOperation * operations;
};

struct ggml_metal_sparse_reservation {
    struct ggml_metal_sparse_reservation_entry * entries;
    size_t n_entries;
};

static int ggml_metal_sparse_buffer_range_compare(const void * lhs, const void * rhs) {
    const struct ggml_metal_sparse_buffer_range * l = lhs;
    const struct ggml_metal_sparse_buffer_range * r = rhs;
    const uintptr_t lb = (uintptr_t) l->buffer;
    const uintptr_t rb = (uintptr_t) r->buffer;
    if (lb != rb) {
        return lb < rb ? -1 : 1;
    }
    if (l->offset != r->offset) {
        return l->offset < r->offset ? -1 : 1;
    }
    if (l->size != r->size) {
        return l->size < r->size ? -1 : 1;
    }
    return 0;
}

struct ggml_metal_sparse_move_entry {
    ggml_metal_buffer_t buffer;
    struct ggml_metal_sparse_page_move * moves;
    size_t n_moves;
    uint64_t generation;
    uint32_t * source_physical;
    uint32_t * source_refcount;
    uint8_t * selected;
    MTL4UpdateSparseBufferMappingOperation * operations;
    size_t n_operations;
};

struct ggml_metal_sparse_move {
    struct ggml_metal_sparse_move_entry * entries;
    size_t n_entries;
    bool consumed;
};

static bool ggml_metal_sparse_move_trace_enabled(void) {
    const char * value = getenv("LLAMA_DSV4_RESIDENT_SPARSE_MOVE_TRACE");
    return value != NULL && (strcmp(value, "1") == 0 || strcmp(value, "true") == 0);
}

static int ggml_metal_sparse_buffer_move_compare(const void * lhs, const void * rhs) {
    const struct ggml_metal_sparse_buffer_move * l = lhs;
    const struct ggml_metal_sparse_buffer_move * r = rhs;
    const uintptr_t lb = (uintptr_t) l->source.buffer;
    const uintptr_t rb = (uintptr_t) r->source.buffer;
    if (lb != rb) {
        return lb < rb ? -1 : 1;
    }
    if (l->source.offset != r->source.offset) {
        return l->source.offset < r->source.offset ? -1 : 1;
    }
    if (l->destination.offset != r->destination.offset) {
        return l->destination.offset < r->destination.offset ? -1 : 1;
    }
    return 0;
}

static void ggml_metal_sparse_move_destroy(ggml_metal_sparse_move_t move) {
    if (move == NULL) {
        return;
    }
    for (size_t i = 0; i < move->n_entries; ++i) {
        free(move->entries[i].moves);
        free(move->entries[i].source_physical);
        free(move->entries[i].source_refcount);
        free(move->entries[i].selected);
        free(move->entries[i].operations);
    }
    free(move->entries);
    free(move);
}

static void ggml_metal_sparse_move_lock(const ggml_metal_sparse_move_t move) {
    for (size_t i = 0; i < move->n_entries; ++i) {
        [move->entries[i].buffer->sparse_lock lock];
    }
}

static void ggml_metal_sparse_move_unlock(const ggml_metal_sparse_move_t move) {
    for (size_t i = move->n_entries; i-- > 0;) {
        [move->entries[i].buffer->sparse_lock unlock];
    }
}

static enum ggml_metal_sparse_reservation_result ggml_metal_sparse_move_prepare(
        const struct ggml_metal_sparse_buffer_move * input,
        size_t n_input,
        ggml_metal_sparse_move_t * result) {
    if (result != NULL) {
        *result = NULL;
    }
    if (input == NULL || n_input == 0 || result == NULL ||
            n_input > SIZE_MAX/sizeof(*input)) {
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }

    struct ggml_metal_sparse_buffer_move * sorted = malloc(n_input*sizeof(*sorted));
    ggml_metal_sparse_move_t move = calloc(1, sizeof(*move));
    if (sorted == NULL || move == NULL) {
        free(sorted);
        free(move);
        return GGML_METAL_SPARSE_RESERVATION_OOM;
    }
    memcpy(sorted, input, n_input*sizeof(*sorted));
    qsort(sorted, n_input, sizeof(*sorted), ggml_metal_sparse_buffer_move_compare);

    for (size_t i = 0; i < n_input; ++i) {
        const struct ggml_metal_sparse_buffer_move * item = &sorted[i];
        ggml_metal_buffer_t buf = item->source.buffer;
        if (buf == NULL || !buf->is_sparse || item->source.size == 0 ||
                item->source.offset % buf->sparse_page_size != 0 ||
                item->source.size % buf->sparse_page_size != 0 ||
                item->source.offset > buf->all_size ||
                item->source.size > buf->all_size - item->source.offset ||
                (item->destination.buffer != NULL &&
                 (item->destination.buffer != buf ||
                  item->destination.size != item->source.size ||
                  item->destination.offset % buf->sparse_page_size != 0 ||
                  item->destination.offset > buf->all_size ||
                  item->destination.size > buf->all_size - item->destination.offset))) {
            free(sorted);
            free(move);
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        if (i == 0 || sorted[i - 1].source.buffer != buf) {
            ++move->n_entries;
        }
    }

    move->entries = calloc(move->n_entries, sizeof(*move->entries));
    if (move->entries == NULL) {
        free(sorted);
        free(move);
        return GGML_METAL_SPARSE_RESERVATION_OOM;
    }

    size_t ie = 0;
    for (size_t i = 0; i < n_input;) {
        size_t end = i + 1;
        while (end < n_input && sorted[end].source.buffer == sorted[i].source.buffer) {
            ++end;
        }
        struct ggml_metal_sparse_move_entry * entry = &move->entries[ie++];
        entry->buffer = sorted[i].source.buffer;
        for (size_t j = i; j < end; ++j) {
            const size_t pages = sorted[j].source.size/entry->buffer->sparse_page_size;
            if (entry->n_moves > SIZE_MAX - pages) {
                free(sorted);
                ggml_metal_sparse_move_destroy(move);
                return GGML_METAL_SPARSE_RESERVATION_INVALID;
            }
            entry->n_moves += pages;
        }
        if (entry->n_moves > SIZE_MAX/sizeof(*entry->moves) ||
                entry->n_moves > SIZE_MAX/sizeof(*entry->source_physical) ||
                entry->n_moves > SIZE_MAX/sizeof(*entry->source_refcount) ||
                entry->n_moves > SIZE_MAX/(3*sizeof(*entry->operations))) {
            free(sorted);
            ggml_metal_sparse_move_destroy(move);
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        entry->moves = malloc(entry->n_moves*sizeof(*entry->moves));
        entry->source_physical = malloc(entry->n_moves*sizeof(*entry->source_physical));
        entry->source_refcount = malloc(entry->n_moves*sizeof(*entry->source_refcount));
        entry->selected = calloc(entry->buffer->sparse_n_virtual, sizeof(*entry->selected));
        entry->operations = calloc(3*entry->n_moves, sizeof(*entry->operations));
        if (entry->moves == NULL || entry->source_physical == NULL ||
                entry->source_refcount == NULL || entry->selected == NULL || entry->operations == NULL) {
            free(sorted);
            ggml_metal_sparse_move_destroy(move);
            return GGML_METAL_SPARSE_RESERVATION_OOM;
        }
        size_t im = 0;
        for (size_t j = i; j < end; ++j) {
            const size_t page = entry->buffer->sparse_page_size;
            const size_t source = sorted[j].source.offset/page;
            const size_t destination = sorted[j].destination.buffer != NULL ?
                    sorted[j].destination.offset/page : SIZE_MAX;
            const size_t pages = sorted[j].source.size/page;
            for (size_t p = 0; p < pages; ++p) {
                entry->moves[im++] = (struct ggml_metal_sparse_page_move) {
                    source + p,
                    destination == SIZE_MAX ? SIZE_MAX : destination + p,
                };
            }
        }
        GGML_ASSERT(im == entry->n_moves);
        i = end;
    }
    free(sorted);

    enum ggml_metal_sparse_reservation_result status = GGML_METAL_SPARSE_RESERVATION_OK;
    ggml_metal_sparse_move_lock(move);
    for (size_t i = 0; i < move->n_entries; ++i) {
        struct ggml_metal_sparse_move_entry * entry = &move->entries[i];
        entry->generation = entry->buffer->sparse_generation;
        const enum ggml_metal_sparse_plan_status plan = ggml_metal_sparse_plan_page_move(
                entry->buffer->sparse_n_virtual, entry->buffer->sparse_n_physical,
                entry->buffer->sparse_v2p, entry->buffer->sparse_p_ref,
                entry->moves, entry->n_moves, entry->source_physical,
                entry->selected);
        if (plan != GGML_METAL_SPARSE_PLAN_OK) {
            status = GGML_METAL_SPARSE_RESERVATION_INVALID;
            break;
        }
        for (size_t j = 0; j < entry->n_moves; ++j) {
            const uint32_t physical = entry->source_physical[j];
            entry->source_refcount[j] = physical == UINT32_MAX ? 0 : entry->buffer->sparse_p_ref[physical];
        }
    }
    ggml_metal_sparse_move_unlock(move);
    if (status != GGML_METAL_SPARSE_RESERVATION_OK) {
        ggml_metal_sparse_move_destroy(move);
        return status;
    }
    *result = move;
    return status;
}

// Entries are grouped after sorting by stable buffer address. Every aggregate
// quote, reserve, commit, rollback, and cancel therefore acquires pool locks in
// the same order and releases them in reverse order.
static void ggml_metal_sparse_lock_entries(
        const struct ggml_metal_sparse_reservation * reservation) {
    for (size_t i = 0; i < reservation->n_entries; ++i) {
        [reservation->entries[i].buffer->sparse_lock lock];
    }
}

static void ggml_metal_sparse_unlock_entries(
        const struct ggml_metal_sparse_reservation * reservation) {
    for (size_t i = reservation->n_entries; i-- > 0;) {
        [reservation->entries[i].buffer->sparse_lock unlock];
    }
}

static void ggml_metal_sparse_get_usage_locked(
        ggml_metal_buffer_t buf,
        struct ggml_metal_sparse_usage * usage) {
    *usage = (struct ggml_metal_sparse_usage) {
        /*.pool_id               =*/ (uintptr_t) buf,
        /*.page_size             =*/ buf->sparse_page_size,
        /*.virtual_pages         =*/ buf->sparse_n_virtual,
        /*.physical_pages        =*/ buf->sparse_n_physical,
        /*.free_pages            =*/ buf->sparse_n_free,
        /*.reserved_pages        =*/ buf->sparse_n_reserved,
        /*.mapped_mappings       =*/ 0,
        /*.unique_physical_pages =*/ 0,
        /*.shared_physical_pages =*/ 0,
        /*.shared_mappings       =*/ 0,
        /*.refcount_sum          =*/ 0,
        /*.refcount_max          =*/ 0,
        /*.generation            =*/ buf->sparse_generation,
        /*.cow_allocations       =*/ buf->sparse_cow_allocations,
        /*.cow_pages             =*/ buf->sparse_cow_pages,
    };

    for (size_t p = 0; p < buf->sparse_n_physical; ++p) {
        const uint32_t refs = buf->sparse_p_ref[p];
        if (refs == 0) {
            continue;
        }
        ++usage->unique_physical_pages;
        usage->mapped_mappings += refs;
        usage->refcount_sum += refs;
        usage->refcount_max = refs > usage->refcount_max ? refs : usage->refcount_max;
        if (refs > 1) {
            ++usage->shared_physical_pages;
            usage->shared_mappings += refs;
        }
    }
    GGML_ASSERT(usage->unique_physical_pages + usage->free_pages == usage->physical_pages);
    GGML_ASSERT(usage->mapped_mappings == usage->refcount_sum);
}

static void ggml_metal_sparse_reservation_destroy(
        ggml_metal_sparse_reservation_t reservation) {
    if (reservation == NULL) {
        return;
    }
    for (size_t i = 0; i < reservation->n_entries; ++i) {
        free(reservation->entries[i].ranges);
        free(reservation->entries[i].marked);
        free(reservation->entries[i].marked_per_physical);
        free(reservation->entries[i].retained_by_physical);
        free(reservation->entries[i].copy_source_by_virtual);
        free(reservation->entries[i].writes);
        free(reservation->entries[i].operations);
    }
    free(reservation->entries);
    free(reservation);
}

static enum ggml_metal_sparse_reservation_result ggml_metal_sparse_prepare(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges,
        bool reserve,
        struct ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool,
        ggml_metal_sparse_reservation_t * result) {
    if (n_pools != NULL) {
        *n_pools = 0;
    }
    if (limiting_pool != NULL) {
        *limiting_pool = SIZE_MAX;
    }
    if (result != NULL) {
        *result = NULL;
    }
    if (ranges == NULL || n_ranges == 0 || n_pools == NULL ||
            (reserve && result == NULL)) {
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }

    struct ggml_metal_sparse_buffer_range * sorted = malloc(n_ranges*sizeof(*sorted));
    ggml_metal_sparse_reservation_t reservation = calloc(1, sizeof(*reservation));
    if (sorted == NULL || reservation == NULL) {
        free(sorted);
        free(reservation);
        return GGML_METAL_SPARSE_RESERVATION_OOM;
    }
    memcpy(sorted, ranges, n_ranges*sizeof(*sorted));
    qsort(sorted, n_ranges, sizeof(*sorted), ggml_metal_sparse_buffer_range_compare);

    for (size_t i = 0; i < n_ranges; ++i) {
        if (sorted[i].buffer == NULL || !sorted[i].buffer->is_sparse) {
            free(sorted);
            free(reservation);
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        if (i == 0 || sorted[i].buffer != sorted[i - 1].buffer) {
            ++reservation->n_entries;
        }
    }
    *n_pools = reservation->n_entries;
    if (pools != NULL && pool_capacity < reservation->n_entries) {
        free(sorted);
        free(reservation);
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }

    reservation->entries = calloc(reservation->n_entries, sizeof(*reservation->entries));
    if (reservation->entries == NULL) {
        free(sorted);
        free(reservation);
        return GGML_METAL_SPARSE_RESERVATION_OOM;
    }

    size_t ie = 0;
    for (size_t i = 0; i < n_ranges;) {
        size_t end = i + 1;
        while (end < n_ranges && sorted[end].buffer == sorted[i].buffer) {
            ++end;
        }
        struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[ie++];
        entry->buffer = sorted[i].buffer;
        entry->n_ranges = end - i;
        entry->ranges = malloc(entry->n_ranges*sizeof(*entry->ranges));
        entry->marked = calloc(entry->buffer->sparse_n_virtual, sizeof(*entry->marked));
        entry->marked_per_physical = calloc(
                entry->buffer->sparse_n_physical, sizeof(*entry->marked_per_physical));
        entry->retained_by_physical = malloc(
                entry->buffer->sparse_n_physical*sizeof(*entry->retained_by_physical));
        entry->copy_source_by_virtual = malloc(
                entry->buffer->sparse_n_virtual*sizeof(*entry->copy_source_by_virtual));
        if (entry->ranges == NULL || entry->marked == NULL || entry->marked_per_physical == NULL ||
                entry->retained_by_physical == NULL || entry->copy_source_by_virtual == NULL) {
            free(sorted);
            ggml_metal_sparse_reservation_destroy(reservation);
            return GGML_METAL_SPARSE_RESERVATION_OOM;
        }
        for (size_t j = i; j < end; ++j) {
            entry->ranges[j - i] = (struct ggml_metal_sparse_range) {
                sorted[j].offset,
                sorted[j].size,
            };
        }
        i = end;
    }
    free(sorted);

    enum ggml_metal_sparse_reservation_result status = GGML_METAL_SPARSE_RESERVATION_OK;
    size_t limiting = SIZE_MAX;
    size_t limiting_margin = SIZE_MAX;
    bool limiting_has_deficit = false;
    ggml_metal_sparse_lock_entries(reservation);
    for (size_t i = 0; i < reservation->n_entries; ++i) {
        struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
        ggml_metal_buffer_t buf = entry->buffer;
        entry->quote = ggml_metal_sparse_plan_write(
                buf->sparse_page_size, buf->sparse_n_virtual, buf->sparse_n_physical,
                buf->sparse_n_free, buf->sparse_n_reserved, buf->sparse_generation,
                buf->sparse_v2p, buf->sparse_p_ref, entry->ranges, entry->n_ranges,
                entry->marked, entry->marked_per_physical);
        if (entry->quote.status == GGML_METAL_SPARSE_PLAN_OK) {
            entry->quote.status = ggml_metal_sparse_select_cow_sources(
                    buf->sparse_n_virtual, buf->sparse_n_physical,
                    buf->sparse_v2p, buf->sparse_p_ref,
                    entry->marked, entry->marked_per_physical,
                    entry->retained_by_physical, entry->copy_source_by_virtual);
            entry->quote.feasible = entry->quote.status == GGML_METAL_SPARSE_PLAN_OK &&
                    entry->quote.feasible;
        }
        if (entry->quote.status != GGML_METAL_SPARSE_PLAN_OK) {
            GGML_LOG_ERROR(
                    "%s: result=%s entry=%zu pool=%p plan_status=%s"
                    " generation=%llu ranges=%zu\n",
                    __func__, ggml_metal_sparse_reservation_result_name(
                            GGML_METAL_SPARSE_RESERVATION_INVALID),
                    i, (void *) buf,
                    ggml_metal_sparse_plan_status_name(entry->quote.status),
                    (unsigned long long) buf->sparse_generation, entry->n_ranges);
            status = GGML_METAL_SPARSE_RESERVATION_INVALID;
        } else if (!entry->quote.feasible && status == GGML_METAL_SPARSE_RESERVATION_OK) {
            status = GGML_METAL_SPARSE_RESERVATION_PRESSURE;
        }

        const size_t available = buf->sparse_n_reserved <= buf->sparse_n_free ?
                buf->sparse_n_free - buf->sparse_n_reserved : 0;
        const bool has_deficit = entry->quote.required_pages > available;
        const size_t margin = has_deficit ?
                entry->quote.required_pages - available :
                available - entry->quote.required_pages;
        if (limiting == SIZE_MAX ||
                (has_deficit && !limiting_has_deficit) ||
                (has_deficit == limiting_has_deficit &&
                 (has_deficit ? margin > limiting_margin : margin < limiting_margin))) {
            limiting = i;
            limiting_margin = margin;
            limiting_has_deficit = has_deficit;
        }
        if (pools != NULL) {
            pools[i].pool_id = (uintptr_t) buf;
            ggml_metal_sparse_get_usage_locked(buf, &pools[i].usage);
            pools[i].write = entry->quote;
        }
    }

    if (status == GGML_METAL_SPARSE_RESERVATION_OK && reserve) {
        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            const size_t n_actions = entry->quote.required_pages;
            if (n_actions > 0) {
                entry->writes = calloc(n_actions, sizeof(*entry->writes));
                entry->operations = calloc(n_actions, sizeof(*entry->operations));
                if (entry->writes == NULL || entry->operations == NULL) {
                    status = GGML_METAL_SPARSE_RESERVATION_OOM;
                    break;
                }
            }
        }
    }
    if (status == GGML_METAL_SPARSE_RESERVATION_OK && reserve) {
        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            if (!ggml_metal_sparse_accounting_try_reserve(
                        entry->buffer->sparse_n_free, &entry->buffer->sparse_n_reserved,
                        entry->buffer->sparse_generation, &entry->quote, &entry->accounting)) {
                GGML_LOG_ERROR(
                        "%s: result=%s entry=%zu pool=%p plan_status=%s feasible=%d"
                        " quote_generation=%llu current_generation=%llu"
                        " required=%zu free=%zu reserved=%zu ticket_state=%s\n",
                        __func__, ggml_metal_sparse_reservation_result_name(
                                GGML_METAL_SPARSE_RESERVATION_STALE),
                        i, (void *) entry->buffer,
                        ggml_metal_sparse_plan_status_name(entry->quote.status),
                        (int) entry->quote.feasible,
                        (unsigned long long) entry->quote.generation,
                        (unsigned long long) entry->buffer->sparse_generation,
                        entry->quote.required_pages, entry->buffer->sparse_n_free,
                        entry->buffer->sparse_n_reserved,
                        ggml_metal_sparse_ticket_state_name(entry->accounting.state));
                status = GGML_METAL_SPARSE_RESERVATION_STALE;
                for (size_t j = 0; j < i; ++j) {
                    ggml_metal_sparse_accounting_finish(
                            &reservation->entries[j].buffer->sparse_n_reserved,
                            &reservation->entries[j].accounting,
                            GGML_METAL_SPARSE_TICKET_ROLLED_BACK);
                }
                break;
            }
        }
    }
    ggml_metal_sparse_unlock_entries(reservation);

    if (limiting_pool != NULL) {
        *limiting_pool = limiting;
    }
    if (status != GGML_METAL_SPARSE_RESERVATION_OK || !reserve) {
        ggml_metal_sparse_reservation_destroy(reservation);
        return status;
    }

    *result = reservation;
    return GGML_METAL_SPARSE_RESERVATION_OK;
}
#endif

bool ggml_metal_buffer_sparse_map_write(
        ggml_metal_buffer_t buf,
        const size_t * offsets,
        const size_t * sizes,
        size_t n_ranges) {
    if (buf == NULL || !buf->is_sparse || n_ranges == 0) {
        return true;
    }
    if (offsets == NULL || sizes == NULL) {
        return false;
    }

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        struct ggml_metal_sparse_buffer_range * writes = malloc(n_ranges*sizeof(*writes));
        if (writes == NULL) {
            return false;
        }
        for (size_t i = 0; i < n_ranges; ++i) {
            writes[i] = (struct ggml_metal_sparse_buffer_range) { buf, offsets[i], sizes[i] };
        }
        ggml_metal_sparse_reservation_t reservation = NULL;
        size_t n_pools = 0;
        const enum ggml_metal_sparse_reservation_result reserve_status =
                ggml_metal_buffers_sparse_reserve(
                        writes, n_ranges, NULL, 0, &n_pools, NULL, &reservation);
        free(writes);
        if (reserve_status != GGML_METAL_SPARSE_RESERVATION_OK) {
            if (reserve_status == GGML_METAL_SPARSE_RESERVATION_PRESSURE) {
                GGML_LOG_WARN("%s: DSV4 sparse heap reservation pressure\n", __func__);
            }
            return false;
        }
        const enum ggml_metal_sparse_reservation_result commit_status =
                ggml_metal_sparse_reservation_commit(reservation);
        ggml_metal_sparse_reservation_free(reservation);
        return commit_status == GGML_METAL_SPARSE_RESERVATION_OK;
    }
#endif

    return false;
}

bool ggml_metal_buffer_sparse_get_usage(
        ggml_metal_buffer_t buf,
        struct ggml_metal_sparse_usage * usage) {
    if (buf == NULL || usage == NULL || !buf->is_sparse) {
        return false;
    }
    [buf->sparse_lock lock];
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    ggml_metal_sparse_get_usage_locked(buf, usage);
#else
    memset(usage, 0, sizeof(*usage));
#endif
    [buf->sparse_lock unlock];
    return true;
}

int ggml_metal_buffers_sparse_ranges_resident(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        if (ranges == NULL || n_ranges == 0) {
            return -1;
        }

        // Collect the distinct pools in address order. ggml_metal_sparse_prepare
        // sorts its entries the same way, so taking the locks in this order
        // cannot deadlock against a concurrent reservation or move.
        ggml_metal_buffer_t bufs[GGML_METAL_SPARSE_RESIDENT_MAX_POOLS];
        size_t n_bufs = 0;
        for (size_t i = 0; i < n_ranges; ++i) {
            ggml_metal_buffer_t buf = ranges[i].buffer;
            if (buf == NULL || !buf->is_sparse) {
                return -1;
            }
            size_t j = 0;
            while (j < n_bufs && (uintptr_t) bufs[j] < (uintptr_t) buf) {
                ++j;
            }
            if (j < n_bufs && bufs[j] == buf) {
                continue;
            }
            if (n_bufs == GGML_METAL_SPARSE_RESIDENT_MAX_POOLS) {
                // Out of lock slots: report "needs a reservation" and let the
                // caller take the general path.
                return 0;
            }
            for (size_t k = n_bufs; k > j; --k) {
                bufs[k] = bufs[k - 1];
            }
            bufs[j] = buf;
            ++n_bufs;
        }

        for (size_t i = 0; i < n_bufs; ++i) {
            [bufs[i]->sparse_lock lock];
        }

        int resident = 1;
        for (size_t i = 0; i < n_ranges; ++i) {
            ggml_metal_buffer_t buf = ranges[i].buffer;
            const size_t size = ranges[i].size;
            if (size == 0) {
                continue;
            }
            const size_t offset = ranges[i].offset;
            const size_t virtual_size = buf->sparse_n_virtual*buf->sparse_page_size;
            if (offset > virtual_size || size > virtual_size - offset) {
                resident = -1;
                break;
            }
            const size_t v0 = offset/buf->sparse_page_size;
            const size_t v1 = (offset + size - 1)/buf->sparse_page_size;
            for (size_t v = v0; v <= v1; ++v) {
                const uint32_t p = buf->sparse_v2p[v];
                // p_ref != 1 covers both a page shared with another virtual
                // mapping (COW required) and two pages of this same range set
                // aliasing one physical page (also COW).
                if (p == UINT32_MAX || p >= buf->sparse_n_physical ||
                        buf->sparse_p_ref[p] != 1) {
                    resident = 0;
                    break;
                }
            }
            if (resident != 1) {
                break;
            }
        }

        for (size_t i = 0; i < n_bufs; ++i) {
            [bufs[i]->sparse_lock unlock];
        }

        return resident;
    }
#endif
    GGML_UNUSED(ranges);
    GGML_UNUSED(n_ranges);
    return 0;
}

enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_quote(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges,
        struct ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        return ggml_metal_sparse_prepare(
                ranges, n_ranges, false, pools, pool_capacity, n_pools,
                limiting_pool, NULL);
    }
#endif
    GGML_UNUSED(ranges);
    GGML_UNUSED(n_ranges);
    GGML_UNUSED(pools);
    GGML_UNUSED(pool_capacity);
    GGML_UNUSED(n_pools);
    GGML_UNUSED(limiting_pool);
    return GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED;
}

enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_reserve(
        const struct ggml_metal_sparse_buffer_range * ranges,
        size_t n_ranges,
        struct ggml_metal_sparse_pool_quote * pools,
        size_t pool_capacity,
        size_t * n_pools,
        size_t * limiting_pool,
        ggml_metal_sparse_reservation_t * reservation) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        return ggml_metal_sparse_prepare(
                ranges, n_ranges, true, pools, pool_capacity, n_pools,
                limiting_pool, reservation);
    }
#endif
    GGML_UNUSED(ranges);
    GGML_UNUSED(n_ranges);
    GGML_UNUSED(pools);
    GGML_UNUSED(pool_capacity);
    GGML_UNUSED(n_pools);
    GGML_UNUSED(limiting_pool);
    if (reservation != NULL) {
        *reservation = NULL;
    }
    return GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED;
}

enum ggml_metal_sparse_reservation_result ggml_metal_buffers_sparse_move_quote(
        const struct ggml_metal_sparse_buffer_move * moves,
        size_t n_moves,
        ggml_metal_sparse_move_t * quote) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        return ggml_metal_sparse_move_prepare(moves, n_moves, quote);
    }
#endif
    GGML_UNUSED(moves);
    GGML_UNUSED(n_moves);
    if (quote != NULL) {
        *quote = NULL;
    }
    return GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED;
}

enum ggml_metal_sparse_reservation_result ggml_metal_sparse_move_commit(
        ggml_metal_sparse_move_t move) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        if (move == NULL || move->consumed) {
            return GGML_METAL_SPARSE_RESERVATION_INVALID;
        }
        ggml_metal_sparse_move_lock(move);
        enum ggml_metal_sparse_reservation_result status = GGML_METAL_SPARSE_RESERVATION_OK;
        const bool trace = ggml_metal_sparse_move_trace_enabled();
        for (size_t i = 0; i < move->n_entries; ++i) {
            struct ggml_metal_sparse_move_entry * entry = &move->entries[i];
            ggml_metal_buffer_t buf = entry->buffer;
            if (entry->generation != buf->sparse_generation ||
                    ggml_metal_sparse_plan_page_move(
                        buf->sparse_n_virtual, buf->sparse_n_physical,
                        buf->sparse_v2p, buf->sparse_p_ref,
                        entry->moves, entry->n_moves, entry->source_physical,
                        entry->selected) != GGML_METAL_SPARSE_PLAN_OK) {
                if (trace) {
                    const size_t source_first = entry->n_moves > 0 ? entry->moves[0].source : SIZE_MAX;
                    const size_t source_last = entry->n_moves > 0 ? entry->moves[entry->n_moves - 1].source : SIZE_MAX;
                    fprintf(stderr,
                            "resident-model diagnostic sparse-move phase=commit status=stale "
                            "entry=%zu pool=%llu generation_before=%llu generation_current=%llu "
                            "moves=%zu source_first=%zu source_last=%zu\n",
                            i, (unsigned long long) (uintptr_t) buf,
                            (unsigned long long) entry->generation,
                            (unsigned long long) buf->sparse_generation,
                            entry->n_moves, source_first, source_last);
                }
                status = GGML_METAL_SPARSE_RESERVATION_STALE;
                break;
            }

            size_t io = 0;
            // Unmap overwritten destinations first. The source mapping still
            // owns its page, so even a source/destination alias cannot free it.
            for (size_t j = 0; j < entry->n_moves; ++j) {
                const size_t destination = entry->moves[j].destination;
                if (destination != SIZE_MAX && buf->sparse_v2p[destination] != UINT32_MAX) {
                    entry->operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                        /*.mode        =*/ MTLSparseTextureMappingModeUnmap,
                        /*.bufferRange =*/ NSMakeRange(destination, 1),
                        /*.heapOffset  =*/ 0,
                    };
                }
            }
            for (size_t j = 0; j < entry->n_moves; ++j) {
                if (entry->source_physical[j] == UINT32_MAX) {
                    continue;
                }
                entry->operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                    /*.mode        =*/ MTLSparseTextureMappingModeUnmap,
                    /*.bufferRange =*/ NSMakeRange(entry->moves[j].source, 1),
                    /*.heapOffset  =*/ 0,
                };
            }
            for (size_t j = 0; j < entry->n_moves; ++j) {
                const size_t destination = entry->moves[j].destination;
                const uint32_t source_p = entry->source_physical[j];
                if (destination == SIZE_MAX || source_p == UINT32_MAX) {
                    continue;
                }
                entry->operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                    /*.mode        =*/ MTLSparseTextureMappingModeMap,
                    /*.bufferRange =*/ NSMakeRange(destination, 1),
                    /*.heapOffset  =*/ source_p,
                };
            }
            GGML_ASSERT(io <= 3*entry->n_moves);
            entry->n_operations = io;
            if (trace) {
                size_t mapped_sources = 0;
                size_t destination_pages = 0;
                size_t destination_first = SIZE_MAX;
                size_t destination_last = SIZE_MAX;
                for (size_t j = 0; j < entry->n_moves; ++j) {
                    if (entry->source_physical[j] != UINT32_MAX) {
                        ++mapped_sources;
                    }
                    if (entry->moves[j].destination != SIZE_MAX) {
                        if (destination_first == SIZE_MAX) {
                            destination_first = entry->moves[j].destination;
                        }
                        destination_last = entry->moves[j].destination;
                        ++destination_pages;
                    }
                }
                const size_t source_first = entry->n_moves > 0 ? entry->moves[0].source : SIZE_MAX;
                const size_t source_last = entry->n_moves > 0 ? entry->moves[entry->n_moves - 1].source : SIZE_MAX;
                fprintf(stderr,
                        "resident-model diagnostic sparse-move phase=commit status=quoted "
                        "entry=%zu pool=%llu generation_before=%llu moves=%zu mapped_sources=%zu "
                        "source_first=%zu source_last=%zu destination_pages=%zu destination_first=%zu "
                        "destination_last=%zu mapping_operations=%zu\n",
                        i, (unsigned long long) (uintptr_t) buf,
                        (unsigned long long) entry->generation,
                        entry->n_moves, mapped_sources,
                        source_first, source_last,
                        destination_pages, destination_first, destination_last, entry->n_operations);
            }
        }

        if (status == GGML_METAL_SPARSE_RESERVATION_OK) {
            // All allocations and validation completed before the first table
            // changes. apply_page_move has no failure path.
            for (size_t i = 0; i < move->n_entries; ++i) {
                struct ggml_metal_sparse_move_entry * entry = &move->entries[i];
                ggml_metal_buffer_t buf = entry->buffer;
                ggml_metal_sparse_apply_page_move(
                        buf->sparse_n_physical, buf->sparse_v2p,
                        buf->sparse_p_ref, buf->sparse_free, &buf->sparse_n_free,
                        entry->moves, entry->source_physical, entry->n_moves);
            }
            for (size_t i = 0; i < move->n_entries; ++i) {
                struct ggml_metal_sparse_move_entry * entry = &move->entries[i];
                if (entry->n_operations == 0) {
                    continue;
                }
                ggml_metal_sparse_submit(
                        entry->buffer, entry->operations, entry->n_operations,
                        NULL, 0);
                ++entry->buffer->sparse_generation;
                if (trace) {
                    fprintf(stderr,
                            "resident-model diagnostic sparse-move phase=commit status=committed "
                            "entry=%zu pool=%llu generation_after=%llu mapping_operations=%zu\n",
                            i, (unsigned long long) (uintptr_t) entry->buffer,
                            (unsigned long long) entry->buffer->sparse_generation,
                            entry->n_operations);
                }
            }
            move->consumed = true;
        }
        ggml_metal_sparse_move_unlock(move);
        return status;
    }
#endif
    GGML_UNUSED(move);
    return GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED;
}

static uint64_t ggml_metal_sparse_audit_hash_mix(uint64_t hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
    return hash;
}

int ggml_metal_sparse_move_audit(
        ggml_metal_sparse_move_t move,
        int committed,
        struct ggml_dsv4_sparse_move_audit * audit) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        if (move == NULL || audit == NULL || move->consumed == (committed == 0)) {
            return GGML_DSV4_SPARSE_INVALID;
        }
        if (move->n_entries > GGML_DSV4_SPARSE_MOVE_AUDIT_MAX_POOLS) {
            return GGML_DSV4_SPARSE_INVALID;
        }
        memset(audit, 0, sizeof(*audit));
        ggml_metal_sparse_move_lock(move);
        audit->n_pools = move->n_entries;
        for (size_t i = 0; i < move->n_entries; ++i) {
            struct ggml_metal_sparse_move_entry * entry = &move->entries[i];
            ggml_metal_buffer_t buf = entry->buffer;
            struct ggml_dsv4_sparse_move_audit_pool * pool = &audit->pools[i];
            struct ggml_metal_sparse_usage usage;
            ggml_metal_sparse_get_usage_locked(buf, &usage);
            pool->pool_id = (uint64_t) (uintptr_t) buf;
            pool->generation = buf->sparse_generation;
            pool->virtual_move_count = entry->n_moves;
            pool->destination_page_count = 0;
            pool->mapping_operation_count = committed ? entry->n_operations : 0;
            pool->source_virtual_hash = UINT64_C(1469598103934665603);
            pool->source_physical_hash = UINT64_C(1469598103934665603);
            pool->source_refcount_hash = UINT64_C(1469598103934665603);
            pool->survivor_mapping_hash = UINT64_C(1469598103934665603);

            for (size_t j = 0; j < entry->n_moves; ++j) {
                const size_t source = entry->moves[j].source;
                const size_t destination = entry->moves[j].destination;
                pool->source_virtual_hash = ggml_metal_sparse_audit_hash_mix(
                        pool->source_virtual_hash, source);
                pool->source_virtual_hash = ggml_metal_sparse_audit_hash_mix(
                        pool->source_virtual_hash, destination);
                if (destination != SIZE_MAX) {
                    ++pool->destination_page_count;
                }
                const uint32_t physical = entry->source_physical[j];
                if (physical == UINT32_MAX) {
                    continue;
                }
                ++pool->mapped_source_count;
                pool->source_physical_hash = ggml_metal_sparse_audit_hash_mix(
                        pool->source_physical_hash, physical);
                const uint32_t refcount = entry->source_refcount[j];
                size_t occurrences = 0;
                bool first = true;
                for (size_t k = 0; k < entry->n_moves; ++k) {
                    if (entry->source_physical[k] == physical) {
                        ++occurrences;
                        if (k < j) {
                            first = false;
                        }
                    }
                }
                if (first) {
                    ++pool->source_unique_physical_count;
                    pool->source_refcount_sum += refcount;
                    if (refcount == occurrences) {
                        ++pool->source_released_physical_count;
                    }
                    pool->source_refcount_hash = ggml_metal_sparse_audit_hash_mix(
                            pool->source_refcount_hash, physical);
                    pool->source_refcount_hash = ggml_metal_sparse_audit_hash_mix(
                            pool->source_refcount_hash, refcount);
                    pool->source_refcount_hash = ggml_metal_sparse_audit_hash_mix(
                            pool->source_refcount_hash, occurrences);
                }
            }
            for (size_t page = 0; page < buf->sparse_n_virtual; ++page) {
                if (entry->selected[page] != 0) {
                    continue;
                }
                pool->survivor_mapping_hash = ggml_metal_sparse_audit_hash_mix(
                        pool->survivor_mapping_hash, page);
                pool->survivor_mapping_hash = ggml_metal_sparse_audit_hash_mix(
                        pool->survivor_mapping_hash, buf->sparse_v2p[page]);
            }
            pool->free_pages = usage.free_pages;
            pool->mapped_mappings = usage.mapped_mappings;
            pool->unique_physical_pages = usage.unique_physical_pages;
            pool->shared_physical_pages = usage.shared_physical_pages;
            pool->shared_mappings = usage.shared_mappings;
            pool->refcount_sum = usage.refcount_sum;
            pool->refcount_max = usage.refcount_max;
        }
        ggml_metal_sparse_move_unlock(move);
        return GGML_DSV4_SPARSE_OK;
    }
#endif
    GGML_UNUSED(move);
    GGML_UNUSED(committed);
    GGML_UNUSED(audit);
    return GGML_DSV4_SPARSE_UNSUPPORTED;
}

void ggml_metal_sparse_move_free(ggml_metal_sparse_move_t move) {
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    ggml_metal_sparse_move_destroy(move);
#else
    GGML_UNUSED(move);
#endif
}

enum ggml_metal_sparse_reservation_result ggml_metal_sparse_reservation_commit(
        ggml_metal_sparse_reservation_t reservation) {
    if (reservation == NULL) {
        return GGML_METAL_SPARSE_RESERVATION_INVALID;
    }

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        ggml_metal_sparse_lock_entries(reservation);

        enum ggml_metal_sparse_reservation_result status = GGML_METAL_SPARSE_RESERVATION_OK;
        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            ggml_metal_buffer_t buf = entry->buffer;
            if (!ggml_metal_sparse_accounting_is_current(
                        buf->sparse_generation, &entry->accounting)) {
                GGML_LOG_ERROR(
                        "%s: result=%s reason=ticket-current entry=%zu pool=%p"
                        " ticket_state=%s ticket_generation=%llu current_generation=%llu"
                        " ticket_pages=%zu free=%zu reserved=%zu\n",
                        __func__, ggml_metal_sparse_reservation_result_name(
                                GGML_METAL_SPARSE_RESERVATION_STALE),
                        i, (void *) buf,
                        ggml_metal_sparse_ticket_state_name(entry->accounting.state),
                        (unsigned long long) entry->accounting.generation,
                        (unsigned long long) buf->sparse_generation,
                        entry->accounting.reserved_pages, buf->sparse_n_free,
                        buf->sparse_n_reserved);
                status = GGML_METAL_SPARSE_RESERVATION_STALE;
                break;
            }

            const size_t other_reserved = buf->sparse_n_reserved - entry->accounting.reserved_pages;
            const struct ggml_metal_sparse_quote current = ggml_metal_sparse_plan_write(
                    buf->sparse_page_size, buf->sparse_n_virtual, buf->sparse_n_physical,
                    buf->sparse_n_free, other_reserved, buf->sparse_generation,
                    buf->sparse_v2p, buf->sparse_p_ref, entry->ranges, entry->n_ranges,
                    entry->marked, entry->marked_per_physical);
            if (!ggml_metal_sparse_quote_commit_compatible(&entry->quote, &current)) {
                GGML_LOG_ERROR(
                        "%s: result=%s reason=requote entry=%zu pool=%p ticket_state=%s"
                        " expected_generation=%llu current_generation=%llu"
                        " current_plan_status=%s current_feasible=%d"
                        " expected={target=%zu,new=%zu,cow=%zu,required=%zu}"
                        " current={target=%zu,new=%zu,cow=%zu,required=%zu}"
                        " free=%zu reserved_total=%zu reserved_by_ticket=%zu"
                        " reserved_other=%zu\n",
                        __func__, ggml_metal_sparse_reservation_result_name(
                                GGML_METAL_SPARSE_RESERVATION_STALE),
                        i, (void *) buf,
                        ggml_metal_sparse_ticket_state_name(entry->accounting.state),
                        (unsigned long long) entry->quote.generation,
                        (unsigned long long) current.generation,
                        ggml_metal_sparse_plan_status_name(current.status),
                        (int) current.feasible,
                        entry->quote.target_mappings, entry->quote.new_pages,
                        entry->quote.cow_pages, entry->quote.required_pages,
                        current.target_mappings, current.new_pages,
                        current.cow_pages, current.required_pages,
                        buf->sparse_n_free, buf->sparse_n_reserved,
                        entry->accounting.reserved_pages, other_reserved);
                status = GGML_METAL_SPARSE_RESERVATION_STALE;
                break;
            }
        }

        if (status != GGML_METAL_SPARSE_RESERVATION_OK) {
            for (size_t i = 0; i < reservation->n_entries; ++i) {
                struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
                ggml_metal_sparse_accounting_finish(
                        &entry->buffer->sparse_n_reserved, &entry->accounting,
                        GGML_METAL_SPARSE_TICKET_ROLLED_BACK);
            }
            ggml_metal_sparse_unlock_entries(reservation);
            return status;
        }

        // All generations and preallocated action arrays have been validated
        // while every pool lock is held. No later pool can fail after an
        // earlier pool's CPU mapping table has changed.
        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            ggml_metal_buffer_t buf = entry->buffer;
            size_t iw = 0;
            for (size_t v = 0; v < buf->sparse_n_virtual; ++v) {
                if (!entry->marked[v]) {
                    continue;
                }
                const uint32_t old_p = buf->sparse_v2p[v];
                const uint32_t copy_src = entry->copy_source_by_virtual[v];
                if (old_p != UINT32_MAX) {
                    GGML_ASSERT(old_p < buf->sparse_n_physical);
                    const uint32_t retained = entry->retained_by_physical[old_p];
                    GGML_ASSERT(retained != UINT32_MAX);
                    GGML_ASSERT(retained < buf->sparse_n_virtual);
                    GGML_ASSERT(buf->sparse_v2p[retained] == old_p);
                    if (retained == v) {
                        GGML_ASSERT(copy_src == UINT32_MAX);
                        continue;
                    }
                    GGML_ASSERT(copy_src == retained);
                } else {
                    GGML_ASSERT(copy_src == UINT32_MAX);
                }

                GGML_ASSERT(buf->sparse_n_free > 0);
                const uint32_t new_p = buf->sparse_free[--buf->sparse_n_free];
                if (old_p != UINT32_MAX) {
                    GGML_ASSERT(buf->sparse_p_ref[old_p] > 1);
                    --buf->sparse_p_ref[old_p];
                }

                buf->sparse_v2p[v] = new_p;
                buf->sparse_p_ref[new_p] = 1;
                entry->writes[iw] = (struct ggml_metal_sparse_write_action) {
                    /*.vtile          =*/ (uint32_t) v,
                    /*.ptile          =*/ new_p,
                    /*.copy_src_vtile =*/ copy_src,
                };
                entry->operations[iw] = (MTL4UpdateSparseBufferMappingOperation) {
                    /*.mode        =*/ MTLSparseTextureMappingModeMap,
                    /*.bufferRange =*/ NSMakeRange(v, 1),
                    /*.heapOffset  =*/ new_p,
                };
                ++iw;
            }
            GGML_ASSERT(iw == entry->quote.required_pages);
        }

        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            ggml_metal_buffer_t buf = entry->buffer;
            if (entry->quote.required_pages > 0) {
                ggml_metal_sparse_submit(
                        buf, entry->operations, entry->quote.required_pages,
                        entry->writes, entry->quote.required_pages);
            }
            buf->sparse_cow_allocations += entry->quote.cow_pages > 0;
            buf->sparse_cow_pages += entry->quote.cow_pages;
            if (entry->quote.required_pages > 0) {
                ++buf->sparse_generation;
            }
            GGML_ASSERT(ggml_metal_sparse_accounting_finish(
                    &buf->sparse_n_reserved, &entry->accounting,
                    GGML_METAL_SPARSE_TICKET_COMMITTED));
        }

        ggml_metal_sparse_unlock_entries(reservation);
        return GGML_METAL_SPARSE_RESERVATION_OK;
    }
#endif

    return GGML_METAL_SPARSE_RESERVATION_UNSUPPORTED;
}

static bool ggml_metal_sparse_reservation_release(
        ggml_metal_sparse_reservation_t reservation,
        enum ggml_metal_sparse_ticket_state final_state) {
    if (reservation == NULL) {
        return false;
    }
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        bool result = true;
        ggml_metal_sparse_lock_entries(reservation);
        for (size_t i = 0; i < reservation->n_entries; ++i) {
            struct ggml_metal_sparse_reservation_entry * entry = &reservation->entries[i];
            result = ggml_metal_sparse_accounting_finish(
                    &entry->buffer->sparse_n_reserved, &entry->accounting,
                    final_state) && result;
        }
        ggml_metal_sparse_unlock_entries(reservation);
        return result;
    }
#endif
    return false;
}

bool ggml_metal_sparse_reservation_rollback(ggml_metal_sparse_reservation_t reservation) {
    return ggml_metal_sparse_reservation_release(
            reservation, GGML_METAL_SPARSE_TICKET_ROLLED_BACK);
}

bool ggml_metal_sparse_reservation_cancel(ggml_metal_sparse_reservation_t reservation) {
    return ggml_metal_sparse_reservation_release(
            reservation, GGML_METAL_SPARSE_TICKET_CANCELLED);
}

void ggml_metal_sparse_reservation_free(ggml_metal_sparse_reservation_t reservation) {
    if (reservation == NULL) {
        return;
    }
#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    ggml_metal_sparse_reservation_cancel(reservation);
    ggml_metal_sparse_reservation_destroy(reservation);
#else
    GGML_UNUSED(reservation);
#endif
}

bool ggml_metal_buffer_sparse_alias(
        ggml_metal_buffer_t buf,
        size_t src_offset,
        size_t dst_offset,
        size_t size,
        const size_t * relative_offsets,
        const size_t * sizes,
        size_t n_ranges) {
    if (buf == NULL || !buf->is_sparse) {
        return false;
    }
    if (n_ranges > 0 && (relative_offsets == NULL || sizes == NULL)) {
        return false;
    }

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        const size_t page = buf->sparse_page_size;
        if (src_offset % page != 0 || dst_offset % page != 0 || size % page != 0 ||
                src_offset > buf->all_size || size > buf->all_size - src_offset ||
                dst_offset > buf->all_size || size > buf->all_size - dst_offset) {
            GGML_LOG_ERROR(
                    "%s: invalid alias views: src=%zu dst=%zu size=%zu"
                    " buffer_size=%zu page=%zu\n",
                    __func__, src_offset, dst_offset, size, buf->all_size, page);
            return false;
        }
        if (src_offset == dst_offset) {
            return true;
        }
        // The alias implementation snapshots physical source pages before it
        // replaces the destination mapping. DSV4 streams are disjoint; reject
        // unsupported overlapping views so a destination unmap cannot recycle
        // a page that is still referenced by the source snapshot.
        if (!(src_offset + size <= dst_offset || dst_offset + size <= src_offset)) {
            GGML_LOG_ERROR(
                    "%s: overlapping alias views are unsupported: src=%zu dst=%zu size=%zu\n",
                    __func__, src_offset, dst_offset, size);
            return false;
        }

        [buf->sparse_lock lock];

        const size_t n_tiles = size/page;
        uint8_t * selected = malloc(n_tiles*sizeof(*selected));
        uint32_t * alias_p = malloc(n_tiles*sizeof(uint32_t));
        if (selected == NULL || alias_p == NULL) {
            free(selected);
            free(alias_p);
            [buf->sparse_lock unlock];
            return false;
        }
        for (size_t i = 0; i < n_tiles; ++i) {
            alias_p[i] = UINT32_MAX;
        }

        const enum ggml_metal_sparse_plan_status range_status =
                ggml_metal_sparse_mark_relative_ranges(
                        page, size, relative_offsets, sizes, n_ranges, selected, n_tiles);
        const bool valid = range_status == GGML_METAL_SPARSE_PLAN_OK;

        size_t n_unmap = 0;
        size_t n_map = 0;
        if (valid) {
            const size_t src_v0 = src_offset/page;
            const size_t dst_v0 = dst_offset/page;
            for (size_t t = 0; t < n_tiles; ++t) {
                if (buf->sparse_v2p[dst_v0 + t] != UINT32_MAX) {
                    ++n_unmap;
                }
                if (selected[t]) {
                    alias_p[t] = buf->sparse_v2p[src_v0 + t];
                    if (alias_p[t] != UINT32_MAX) {
                        ++n_map;
                    }
                }
            }
        }

        if (!valid) {
            GGML_LOG_ERROR(
                    "%s: invalid relative alias ranges: status=%s view_size=%zu"
                    " page=%zu ranges=%zu\n",
                    __func__, ggml_metal_sparse_plan_status_name(range_status),
                    size, page, n_ranges);
            free(selected);
            free(alias_p);
            [buf->sparse_lock unlock];
            return false;
        }

        const size_t n_operations = n_unmap + n_map;
        if (n_operations == 0) {
            free(selected);
            free(alias_p);
            [buf->sparse_lock unlock];
            return true;
        }

        MTL4UpdateSparseBufferMappingOperation * operations = calloc(n_operations, sizeof(*operations));
        if (operations == NULL) {
            free(selected);
            free(alias_p);
            [buf->sparse_lock unlock];
            return false;
        }

        size_t io = 0;
        const size_t dst_v0 = dst_offset/page;
        for (size_t t = 0; t < n_tiles; ++t) {
            const size_t v = dst_v0 + t;
            const uint32_t p = buf->sparse_v2p[v];
            if (p == UINT32_MAX) {
                continue;
            }
            operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                /*.mode        =*/ MTLSparseTextureMappingModeUnmap,
                /*.bufferRange =*/ NSMakeRange(v, 1),
                /*.heapOffset  =*/ 0,
            };
            buf->sparse_v2p[v] = UINT32_MAX;
            GGML_ASSERT(buf->sparse_p_ref[p] > 0);
            if (--buf->sparse_p_ref[p] == 0) {
                buf->sparse_free[buf->sparse_n_free++] = p;
            }
        }

        for (size_t t = 0; t < n_tiles; ++t) {
            const uint32_t p = alias_p[t];
            if (p == UINT32_MAX) {
                continue;
            }
            const size_t v = dst_v0 + t;
            operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                /*.mode        =*/ MTLSparseTextureMappingModeMap,
                /*.bufferRange =*/ NSMakeRange(v, 1),
                /*.heapOffset  =*/ p,
            };
            buf->sparse_v2p[v] = p;
            ++buf->sparse_p_ref[p];
        }
        GGML_ASSERT(io == n_operations);

        ggml_metal_sparse_submit(buf, operations, n_operations, NULL, 0);
        ++buf->sparse_generation;

        free(selected);
        free(alias_p);
        free(operations);
        [buf->sparse_lock unlock];
        return true;
    }
#endif

    return false;
}

bool ggml_metal_buffer_sparse_unmap(
        ggml_metal_buffer_t buf,
        size_t offset,
        size_t size) {
    if (buf == NULL || !buf->is_sparse || size == 0) {
        return buf != NULL && buf->is_sparse;
    }

#if TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260400
    if (@available(macOS 26.4, *)) {
        const size_t page = buf->sparse_page_size;
        if (offset % page != 0 || size % page != 0 ||
                offset > buf->all_size || size > buf->all_size - offset) {
            return false;
        }

        [buf->sparse_lock lock];

        const size_t v0 = offset/page;
        const size_t nv = size/page;
        size_t n_operations = 0;
        for (size_t i = 0; i < nv; ++i) {
            n_operations += buf->sparse_v2p[v0 + i] != UINT32_MAX;
        }
        if (n_operations == 0) {
            [buf->sparse_lock unlock];
            return true;
        }

        MTL4UpdateSparseBufferMappingOperation * operations = calloc(n_operations, sizeof(*operations));
        if (operations == NULL) {
            [buf->sparse_lock unlock];
            return false;
        }

        size_t io = 0;
        for (size_t i = 0; i < nv; ++i) {
            const size_t v = v0 + i;
            const uint32_t p = buf->sparse_v2p[v];
            if (p == UINT32_MAX) {
                continue;
            }
            operations[io++] = (MTL4UpdateSparseBufferMappingOperation) {
                /*.mode        =*/ MTLSparseTextureMappingModeUnmap,
                /*.bufferRange =*/ NSMakeRange(v, 1),
                /*.heapOffset  =*/ 0,
            };
            buf->sparse_v2p[v] = UINT32_MAX;
            GGML_ASSERT(buf->sparse_p_ref[p] > 0);
            if (--buf->sparse_p_ref[p] == 0) {
                buf->sparse_free[buf->sparse_n_free++] = p;
            }
        }
        GGML_ASSERT(io == n_operations);

        ggml_metal_sparse_submit(buf, operations, n_operations, NULL, 0);
        ++buf->sparse_generation;
        free(operations);
        [buf->sparse_lock unlock];
        return true;
    }
#endif

    return false;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    ggml_metal_device_rsets_rm(buf->dev, buf->rset);

    ggml_metal_buffer_rset_free(buf);

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    if (buf->is_sparse) {
        [buf->sparse_heap release];
        [buf->sparse_queue release];
        [buf->sparse_event release];
        [buf->sparse_lock release];
        free(buf->sparse_v2p);
        free(buf->sparse_p_ref);
        free(buf->sparse_free);
    }

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf != NULL && buf->is_shared;
}

bool ggml_metal_buffer_is_sparse(ggml_metal_buffer_t buf) {
    return buf != NULL && buf->is_sparse;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        const size_t dst_offset = bid_dst.offs;
        if (!ggml_metal_buffer_sparse_map_write(buf, &dst_offset, &size, 1)) {
            GGML_ABORT("failed to back sparse Metal tensor write");
        }

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        const size_t dst_offset = bid_dst.offs;
        if (!ggml_metal_buffer_sparse_map_write(buf, &dst_offset, &size, 1)) {
            GGML_ABORT("failed to back sparse Metal tensor upload");
        }

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        [buf_src release];

        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    if (size == 0) {
        return;
    }
    GGML_ASSERT(data != NULL);

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // Metal requires no-copy host buffers to cover whole pages. Keep the
        // caller's requested bytes at their original offset within the mapped
        // range so arbitrary host destinations remain valid.
        const size_t size_page = sysconf(_SC_PAGESIZE);
        GGML_ASSERT(size_page > 0);

        const uintptr_t data_addr = (uintptr_t) data;
        const size_t data_offset = data_addr % size_page;
        void * data_page = (void *) (data_addr - data_offset);

        GGML_ASSERT(size <= SIZE_MAX - data_offset);
        size_t size_mapped = size + data_offset;
        const size_t size_remainder = size_mapped % size_page;
        if (size_remainder != 0) {
            const size_t size_padding = size_page - size_remainder;
            GGML_ASSERT(size_mapped <= SIZE_MAX - size_padding);
            size_mapped += size_padding;
        }
        GGML_ASSERT(size_mapped <= (size_t) [buf->dev->mtl_device maxBufferLength]);

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_page
                                                               length:size_mapped
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:data_offset
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        [buf_dst release];
    }
}

bool ggml_metal_buffer_cpy_tensor(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t)src->buffer->context;

    const size_t size = ggml_nbytes(src);

    // if both buffers are shared, we can use memcpy directly
    if (buf_dst->is_shared && buf_src->is_shared) {
        memcpy(dst->data, src->data, size);
        return true;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf_src, src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf_dst, dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        const size_t dst_offset = bid_dst.offs;
        if (!ggml_metal_buffer_sparse_map_write(buf_dst, &dst_offset, &size, 1)) {
            return false;
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }

    return true;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    if (buf->is_sparse) {
        GGML_ASSERT(value == 0 && "DSV4 sparse buffers only support zero clear");
        const bool ok = ggml_metal_buffer_sparse_unmap(buf, 0, buf->all_size);
        GGML_ASSERT(ok);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

constexpr uint32_t LLAMA_SNAPSHOT_FORMAT_VERSION       = 1;
constexpr uint32_t LLAMA_SNAPSHOT_CHUNK_FORMAT_VERSION = 1;
constexpr uint32_t LLAMA_SNAPSHOT_DEVICE_QUEUES        = 1;
constexpr uint64_t LLAMA_SNAPSHOT_CHUNK_ALIGNMENT      = 64ULL * 1024;
constexpr uint64_t LLAMA_SNAPSHOT_MAX_CHUNK_BYTES      = 64ULL * 1024 * 1024;
constexpr uint32_t LLAMA_SNAPSHOT_MAX_CHUNKS           = 16384;
constexpr uint64_t LLAMA_SNAPSHOT_MAX_MANIFEST_BYTES   = 1024ULL * 1024;
constexpr uint64_t LLAMA_SNAPSHOT_MAX_BYTES            = 1024ULL * 1024 * 1024 * 1024;
constexpr uint32_t LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES   = 256;

using llama_snapshot_digest = std::array<uint8_t, 32>;

llama_snapshot_digest llama_snapshot_sha256(const void * data, size_t size);
std::string           llama_snapshot_digest_hex(const llama_snapshot_digest & digest);

enum class llama_snapshot_status : uint8_t {
    ok,
    invalid_argument,
    no_current_generation,
    generation_exists,
    stale_generation,
    identity_mismatch,
    device_mismatch,
    manifest_too_large,
    chunk_too_large,
    format_error,
    truncated,
    trailing_data,
    missing_chunk,
    checksum_mismatch,
    cancelled,
    no_space,
    short_write,
    io_error,
    commit_uncertain,
};

const char * llama_snapshot_status_name(llama_snapshot_status status);

struct llama_snapshot_identity {
    std::string           architecture;
    llama_snapshot_digest model_artifact_digest = {};
    llama_snapshot_digest tokenizer_digest      = {};
    llama_snapshot_digest chat_template_digest  = {};
    llama_snapshot_digest runtime_build_digest  = {};
    llama_snapshot_digest target_kv_digest      = {};
    llama_snapshot_digest draft_kv_digest       = {};
    llama_snapshot_digest rope_digest           = {};
    llama_snapshot_digest lora_digest           = {};
    std::string           target_kv_type;
    std::string           draft_kv_type;
    uint32_t              context_size       = 0;
    uint32_t              raw_window         = 0;
    uint32_t              c4_ratio           = 0;
    uint32_t              hca_ratio          = 0;
    uint32_t              dsv4_state_version = 0;
    uint32_t              rollback_depth     = 0;
};

bool operator==(const llama_snapshot_identity & lhs, const llama_snapshot_identity & rhs);
bool operator!=(const llama_snapshot_identity & lhs, const llama_snapshot_identity & rhs);

struct llama_snapshot_metadata {
    uint64_t                snapshot_generation = 0;
    uint64_t                request_generation  = 0;
    llama_snapshot_identity identity;
};

struct llama_snapshot_store_config {
    std::string root_path;
    std::string physical_device_id;
    // This dormant core is deliberately serialized. A later worker layer may
    // own one queue per physical device, but one store never advertises more.
    uint32_t    physical_device_queues = LLAMA_SNAPSHOT_DEVICE_QUEUES;
    uint64_t    chunk_payload_bytes    = 4ULL * 1024 * 1024;
    uint32_t    max_chunks             = LLAMA_SNAPSHOT_MAX_CHUNKS;
    uint64_t    max_manifest_bytes     = LLAMA_SNAPSHOT_MAX_MANIFEST_BYTES;
    uint64_t    max_snapshot_bytes     = LLAMA_SNAPSHOT_MAX_BYTES;
};

enum class llama_snapshot_write_fault : uint8_t {
    none,
    no_space,
    zero_progress,
};

struct llama_snapshot_faults {
    // Successful writes are capped to this size to exercise ordinary short
    // POSIX writes and retry behavior deterministically.
    uint64_t                   max_write_size              = std::numeric_limits<uint64_t>::max();
    // Once this many bytes have reached the filesystem, the selected fault is
    // returned before any later byte is written.
    uint64_t                   fail_after_bytes            = std::numeric_limits<uint64_t>::max();
    llama_snapshot_write_fault write_fault                 = llama_snapshot_write_fault::none;
    bool                       fail_before_manifest_commit = false;
    // Test-only deterministic race seams around the publication fence. The
    // store calls these after its final ordinary cancellation check.
    std::function<void()>      before_manifest_commit_fence;
    std::function<void()>      after_manifest_commit_fence;
    // Test-only crash seam. Ordinary errors and cancellation always clean up.
    bool                       preserve_failed_generation  = false;
};

using llama_snapshot_cancel_check = std::function<bool(uint32_t durable_chunks)>;
// Called exactly once immediately before current.manifest publication. A true
// result establishes that cancellation is too late; false aborts publication.
using llama_snapshot_commit_fence = std::function<bool()>;

struct llama_snapshot_chunk_info {
    uint32_t              index          = 0;
    uint64_t              logical_offset = 0;
    uint64_t              payload_bytes  = 0;
    llama_snapshot_digest checksum       = {};
};

struct llama_snapshot_manifest {
    uint32_t                               format_version      = 0;
    uint64_t                               snapshot_generation = 0;
    uint64_t                               request_generation  = 0;
    llama_snapshot_identity                identity;
    std::string                            physical_device_id;
    uint32_t                               physical_device_queues = 0;
    uint64_t                               chunk_payload_limit    = 0;
    uint64_t                               total_payload_bytes    = 0;
    std::vector<llama_snapshot_chunk_info> chunks;
};

struct llama_snapshot_write_result {
    llama_snapshot_status status     = llama_snapshot_status::invalid_argument;
    uint64_t              generation = 0;
    int                   os_error   = 0;
    bool                  committed  = false;
};

struct llama_snapshot_open_result {
    llama_snapshot_status   status = llama_snapshot_status::invalid_argument;
    llama_snapshot_manifest manifest;
    int                     os_error = 0;
};

struct llama_snapshot_read_result {
    llama_snapshot_status status = llama_snapshot_status::invalid_argument;
    std::vector<uint8_t>  payload;
    int                   os_error = 0;
};

struct llama_snapshot_read_into_result {
    llama_snapshot_status status        = llama_snapshot_status::invalid_argument;
    uint64_t              payload_bytes = 0;
    int                   os_error      = 0;
};

struct llama_snapshot_chunk_source_result {
    llama_snapshot_status status   = llama_snapshot_status::invalid_argument;
    const uint8_t *       data     = nullptr;
    uint64_t              size     = 0;
    int                   os_error = 0;
};

// A streamed write acquires chunks in ascending logical order. Every
// successful acquire is paired with exactly one release after the store has
// finished hashing and writing that view. Implementations may block in
// acquire() while a bounded staging producer makes the requested chunk ready.
class llama_snapshot_chunk_source_i {
  public:
    virtual ~llama_snapshot_chunk_source_i() = default;

    virtual llama_snapshot_chunk_source_result acquire(
            uint32_t index, uint64_t logical_offset, uint64_t size) noexcept = 0;
    virtual void release(uint32_t index) noexcept = 0;
};

struct llama_snapshot_cleanup_result {
    llama_snapshot_status status   = llama_snapshot_status::invalid_argument;
    uint32_t              removed  = 0;
    int                   os_error = 0;
};

class llama_snapshot_store {
  public:
    // Calls are serialized by the single physical-device owner. This core has
    // no internal worker or concurrency layer.
    explicit llama_snapshot_store(llama_snapshot_store_config config);

    const llama_snapshot_store_config & config() const;

    llama_snapshot_write_result write_generation(const llama_snapshot_metadata &     metadata,
                                                 const std::vector<uint8_t> &        payload,
                                                 const llama_snapshot_cancel_check & cancelled = {},
                                                 const llama_snapshot_faults &       faults    = {});

    // Worker-facing streaming seam. It preserves the same publication and
    // cleanup protocol as write_generation() without requiring a full payload
    // vector. Calls remain externally serialized per store.
    llama_snapshot_write_result write_generation_streamed(
            const llama_snapshot_metadata &     metadata,
            uint64_t                            total_payload_bytes,
            llama_snapshot_chunk_source_i &     source,
            const llama_snapshot_cancel_check & cancelled = {},
            const llama_snapshot_faults &       faults       = {},
            const llama_snapshot_commit_fence & commit_fence = {});

    llama_snapshot_open_result    open_current(const llama_snapshot_identity & expected_identity) const;
    llama_snapshot_status         validate(const llama_snapshot_manifest & manifest, int * os_error = nullptr) const;
    llama_snapshot_read_result    read_chunk(const llama_snapshot_manifest & manifest, uint32_t chunk_index) const;
    llama_snapshot_read_into_result read_chunk_into(const llama_snapshot_manifest & manifest,
                                                    uint32_t chunk_index,
                                                    uint8_t * destination,
                                                    size_t destination_size) const;
    llama_snapshot_read_result    read_all(const llama_snapshot_manifest & manifest) const;
    llama_snapshot_cleanup_result cleanup_temporary_generations();

  private:
    llama_snapshot_store_config cfg;
};

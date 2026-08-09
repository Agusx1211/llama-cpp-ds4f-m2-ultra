#pragma once

#include "llama-kv-cache-dsv4.h"
#include "llama-snapshot-store.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

// This is a standalone persistence format for the placement-independent DSV4
// logical transaction.  It is deliberately not wired into llama state v3:
// callers export one canonical host state, while the codec adds at most one
// fixed-size payload staging buffer instead of constructing another monolithic
// serialized image.
constexpr uint32_t LLAMA_DSV4_SEGMENT_FORMAT_VERSION       = 1;
constexpr uint32_t LLAMA_DSV4_SEGMENT_CHUNK_FORMAT_VERSION = 1;
constexpr uint64_t LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES        = 4ULL * 1024 * 1024;
constexpr uint32_t LLAMA_DSV4_SEGMENT_MAX_CHUNKS           = 65536;
constexpr uint64_t LLAMA_DSV4_SEGMENT_MAX_MANIFEST_BYTES   = 64ULL * 1024 * 1024;
// The default byte and chunk ceilings describe the same 256 GiB envelope.
// Fragmentation across many logical fields may reach the chunk ceiling first;
// custom configs must not advertise more bytes than max_chunks can carry.
constexpr uint64_t LLAMA_DSV4_SEGMENT_MAX_STATE_BYTES =
    LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES * LLAMA_DSV4_SEGMENT_MAX_CHUNKS;

enum class llama_dsv4_segment_status : uint8_t {
    ok,
    invalid_argument,
    unsupported_version,
    invalid_schema,
    identity_mismatch,
    coverage_mismatch,
    state_too_large,
    manifest_too_large,
    too_many_chunks,
    no_current_manifest,
    missing_parent,
    missing_manifest,
    missing_chunk,
    malformed,
    truncated,
    trailing_data,
    checksum_mismatch,
    path_security,
    owner_busy,
    resource_exhausted,
    no_space,
    short_write,
    io_error,
    commit_uncertain,
    injected_failure,
    import_rejected,
};

const char * llama_dsv4_segment_status_name(llama_dsv4_segment_status status);

struct llama_dsv4_segment_identity {
    // Phase one's stable geometry/GGUF identity. This is not a weight digest.
    uint64_t              geometry_identity     = 0;
    // Required caller-supplied digest of the complete model artifact.
    llama_snapshot_digest model_artifact_digest = {};
};

bool operator==(const llama_dsv4_segment_identity & lhs, const llama_dsv4_segment_identity & rhs);
bool operator!=(const llama_dsv4_segment_identity & lhs, const llama_dsv4_segment_identity & rhs);

struct llama_dsv4_segment_prefix_metadata {
    uint64_t              token_count  = 0;
    uint32_t              radix_depth  = 0;
    llama_snapshot_digest token_digest = {};
};

struct llama_dsv4_segment_store_config {
    // Must be absolute. The store creates or validates an exact 0700,
    // same-owner directory and keeps a nonblocking exclusive owner lock.
    std::string root_path;
    uint32_t    max_chunks         = LLAMA_DSV4_SEGMENT_MAX_CHUNKS;
    uint64_t    max_manifest_bytes = LLAMA_DSV4_SEGMENT_MAX_MANIFEST_BYTES;
    uint64_t    max_state_bytes    = LLAMA_DSV4_SEGMENT_MAX_STATE_BYTES;
};

struct llama_dsv4_segment_info {
    llama_snapshot_digest semantic_key       = {};
    llama_snapshot_digest content_digest     = {};
    uint64_t              logical_offset     = 0;
    uint64_t              payload_bytes      = 0;
    bool                  reused_from_parent = false;
};

struct llama_dsv4_segment_manifest {
    uint32_t                             format_version = 0;
    llama_snapshot_digest                digest         = {};
    llama_dsv4_segment_identity          identity;
    llama_dsv4_segment_prefix_metadata   prefix;
    bool                                 has_parent             = false;
    llama_snapshot_digest                parent_digest          = {};
    uint64_t                             logical_fingerprint    = 0;
    uint64_t                             logical_payload_bytes  = 0;
    uint64_t                             encoded_manifest_bytes = 0;
    uint32_t                             reused_segments        = 0;
    std::vector<llama_dsv4_segment_info> segments;
};

struct llama_dsv4_segment_measurement {
    llama_dsv4_segment_status status                   = llama_dsv4_segment_status::invalid_argument;
    uint64_t                  logical_payload_bytes    = 0;
    uint64_t                  encoded_manifest_bytes   = 0;
    uint32_t                  segment_count            = 0;
    // Maximum temporary payload-plus-header allocation used by codec I/O.
    // The caller-owned canonical state is intentionally excluded.
    uint64_t                  peak_codec_scratch_bytes = 0;
};

struct llama_dsv4_segment_faults {
    // Fail before publishing the segment at this zero-based ordinal. Earlier
    // immutable segments may remain as safe orphans for reconciliation.
    uint32_t fail_before_segment          = std::numeric_limits<uint32_t>::max();
    bool     fail_before_manifest_publish = false;
    bool     fail_before_ref_publish      = false;
    // The ref has been renamed and is readable, but its directory durability
    // fence is skipped and commit_uncertain is returned.
    bool     fail_after_ref_rename        = false;
};

struct llama_dsv4_segment_publish_result {
    llama_dsv4_segment_status   status = llama_dsv4_segment_status::invalid_argument;
    llama_dsv4_segment_manifest manifest;
    bool                        committed                = false;
    uint32_t                    segments_created         = 0;
    uint32_t                    segments_reused          = 0;
    uint64_t                    peak_codec_scratch_bytes = 0;
    int                         os_error                 = 0;
};

struct llama_dsv4_segment_open_result {
    llama_dsv4_segment_status   status = llama_dsv4_segment_status::invalid_argument;
    llama_dsv4_segment_manifest manifest;
    int                         os_error = 0;
};

struct llama_dsv4_segment_load_result {
    llama_dsv4_segment_status         status = llama_dsv4_segment_status::invalid_argument;
    llama_dsv4_logical_sequence_state state;
    uint64_t                          peak_codec_scratch_bytes = 0;
    uint32_t                          segments_read            = 0;
    int                               os_error                 = 0;
};

struct llama_dsv4_segment_restore_result {
    llama_dsv4_segment_status       status                   = llama_dsv4_segment_status::invalid_argument;
    llama_dsv4_logical_state_status import_status            = llama_dsv4_logical_state_status::invalid_schema;
    uint64_t                        peak_codec_scratch_bytes = 0;
    uint32_t                        segments_read            = 0;
    int                             os_error                 = 0;
};

struct llama_dsv4_segment_reconcile_result {
    llama_dsv4_segment_status status                  = llama_dsv4_segment_status::invalid_argument;
    uint32_t                  manifests_scanned       = 0;
    uint32_t                  segments_scanned        = 0;
    uint32_t                  manifests_removed       = 0;
    uint32_t                  segments_removed        = 0;
    uint32_t                  temporary_files_removed = 0;
    int                       os_error                = 0;
};

struct llama_dsv4_segment_store_impl;

class llama_dsv4_segment_store {
  public:
    explicit llama_dsv4_segment_store(llama_dsv4_segment_store_config config);
    ~llama_dsv4_segment_store();

    llama_dsv4_segment_store(const llama_dsv4_segment_store &)             = delete;
    llama_dsv4_segment_store & operator=(const llama_dsv4_segment_store &) = delete;

    const llama_dsv4_segment_store_config & config() const;

    // Counting pass: visits only vector sizes and structural metadata. It does
    // not copy a tensor payload or create a second whole-state image.
    llama_dsv4_segment_measurement measure(const llama_dsv4_logical_sequence_state &  state,
                                           const llama_dsv4_segment_identity &        identity,
                                           const llama_dsv4_segment_prefix_metadata & prefix = {}) const;

    llama_dsv4_segment_publish_result publish(const std::string &                        ref_name,
                                              const llama_dsv4_logical_sequence_state &  state,
                                              const llama_dsv4_segment_identity &        identity,
                                              const llama_dsv4_segment_prefix_metadata & prefix = {},
                                              const llama_dsv4_segment_manifest *        parent = nullptr,
                                              const llama_dsv4_segment_faults &          faults = {});

    llama_dsv4_segment_open_result open_current(const std::string &                 ref_name,
                                                const llama_dsv4_segment_identity & expected_identity) const;
    llama_dsv4_segment_open_result open_manifest(const llama_snapshot_digest &       digest,
                                                 const llama_dsv4_segment_identity & expected_identity) const;

    // Authenticates every referenced immutable segment. No logical state is
    // allocated and no cache mutation occurs.
    llama_dsv4_segment_status validate(const llama_dsv4_segment_manifest & manifest, int * os_error = nullptr) const;

    // Allocates exactly one canonical phase-one state after manifest bounds
    // and all declared shapes have been checked. Codec scratch remains capped.
    llama_dsv4_segment_load_result load(const llama_dsv4_segment_manifest & manifest,
                                        const llama_dsv4_segment_identity & expected_identity) const;

    // Fully loads/authenticates before asking phase one for its transactional
    // quote. Every store-side failure therefore precedes destination mutation.
    llama_dsv4_segment_restore_result restore(const llama_dsv4_segment_manifest & manifest,
                                              const llama_dsv4_segment_identity & expected_identity,
                                              llama_kv_cache_dsv4 &               cache,
                                              llama_seq_id                        destination) const;

    // Current refs are always roots. Additional immutable manifests may be
    // pinned explicitly. Scanning and deletion are descriptor-relative and
    // restricted to recognized digest/temp names under the private namespace.
    llama_dsv4_segment_reconcile_result reconcile(const std::vector<llama_snapshot_digest> & pinned_manifests = {},
                                                  bool                                       dry_run          = false);

  private:
    std::unique_ptr<llama_dsv4_segment_store_impl> pimpl;
};

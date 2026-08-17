#pragma once

#include "server-capture.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace server_capture {

// The on-disk format is deliberately separate from cycle_observation's
// process layout.  This lets the inference ABI evolve without making old
// shards unreadable, and makes it possible to prove that compact records do
// not contain token IDs or request-owned content.
constexpr uint32_t CAPTURE_STORE_FORMAT_VERSION  = 1;
// Record format 2: sixteen proposal slots (was five) and a proposer byte in
// place of the retired dspark_mode byte. No format-1 shards were ever written
// by a deployed server, so no migration path exists or is needed.
constexpr uint32_t CAPTURE_RECORD_FORMAT_VERSION = 2;
constexpr uint32_t CAPTURE_SHARD_FORMAT_VERSION  = 1;
using capture_digest                             = std::array<uint8_t, 32>;
constexpr size_t CAPTURE_SHARD_HEADER_BYTES      = 72;
constexpr size_t CAPTURE_SHARD_FOOTER_BYTES      = sizeof(capture_digest);
constexpr size_t CAPTURE_MANIFEST_HEADER_BYTES   = 80;
constexpr size_t CAPTURE_MANIFEST_FOOTER_BYTES   = sizeof(capture_digest);
constexpr size_t CAPTURE_RECORD_HEADER_BYTES     = 16;
constexpr size_t CAPTURE_COMPACT_RECORD_BYTES    = 192;
constexpr size_t CAPTURE_RICH_RECORD_BYTES       = 264;
constexpr size_t CAPTURE_MAX_RING_CAPACITY       = 1U << 20;
constexpr size_t CAPTURE_MAX_SHARD_BYTES         = 64U * 1024U * 1024U;
constexpr size_t CAPTURE_MAX_MANIFEST_BYTES      = 16U * 1024U * 1024U;

enum class capture_store_status : uint8_t {
    ok = 0,
    invalid_argument,
    disabled,
    stopped,
    no_manifest,
    malformed_manifest,
    truncated,
    trailing_data,
    checksum_mismatch,
    shard_missing,
    shard_corrupt,
    manifest_too_large,
    shard_too_large,
    too_many_records,
    too_many_shards,
    no_space,
    short_write,
    io_error,
    commit_uncertain,
    cancelled,
    deletion_failed,
    path_security,
};

const char * capture_store_status_name(capture_store_status status) noexcept;

// The salted, non-reversible request tag persisted in cycle records. Exposed
// so a sidecar request log written with the same salt can carry the same tag
// and let offline training join cycle records to their request streams
// without either file containing the raw request ID.
capture_digest capture_request_tag(const std::array<uint8_t, 16> & salt, uint64_t request_id);

enum class capture_store_phase : uint8_t {
    before_shard_write = 0,
    after_shard_write,
    before_shard_rename,
    after_shard_rename,
    before_manifest_write,
    after_manifest_write,
    before_manifest_rename,
    after_manifest_rename,
    before_retention_delete,
    after_retention_delete,
};

// A cancellation callback is called only on the background worker.  Returning
// true cancels the current flush/write operation.  The callback is never
// consulted by try_enqueue(), so inference cannot be stalled by a slow or
// hostile callback.
using capture_store_cancel_check = std::function<bool(capture_store_phase phase)>;

enum class capture_write_fault : uint8_t {
    none = 0,
    no_space,
    zero_progress,
};

struct capture_store_faults {
    // Successful writes are capped to this number of bytes to exercise
    // deterministic short POSIX writes.
    uint64_t            max_write_size   = UINT64_MAX;
    // Once this many bytes have reached the filesystem, write_fault is
    // returned before a later byte is written.
    uint64_t            fail_after_bytes = UINT64_MAX;
    capture_write_fault write_fault      = capture_write_fault::none;

    // These seams model a process crash.  The caller should destroy the
    // store and construct a new one to exercise recovery.  Files are kept so
    // that the recovery path sees the same state a crash would leave.
    bool crash_before_shard_rename    = false;
    bool crash_after_shard_rename     = false;
    bool crash_before_manifest_rename = false;
    bool crash_after_manifest_rename  = false;

    bool preserve_failed_files = false;
    bool slow_worker           = false;

    // Deterministic test seams for OS CSPRNG admission. There is no
    // clock/address/pid fallback for a generated identity salt.
    bool fail_csprng              = false;
    bool csprng_returns_zero      = false;
    bool force_zero_manifest_salt = false;

    // Deterministic producer-lifecycle barrier, called after an admission
    // lifetime token is claimed and before accepting is checked. The callback
    // must not throw; try_enqueue remains noexcept.
    std::function<void()> before_enqueue_accept;

    // Deterministic producer-lifecycle barrier, called after accepting is
    // observed true and immediately before the SPSC push. The callback must
    // not throw; try_enqueue remains noexcept.
    std::function<void()> before_enqueue_push;

    // Deterministic flush lifecycle seam, called after a flush request is
    // registered and before its wake is posted. The callback must not throw.
    std::function<void()> after_flush_registration;

    // Deterministic filesystem fault seams. Each enabled point fails every
    // matching operation with EIO so tests can prove terminal ordering and
    // orphan/tombstone recovery without relying on host filesystem failures.
    bool fail_file_fsync                       = false;
    bool fail_tombstone_fsync                  = false;
    bool fail_directory_fsync                  = false;
    bool fail_post_publication_directory_fsync = false;
    bool fail_shard_rename                     = false;
    bool fail_manifest_rename                  = false;
    bool fail_fstat                            = false;
    bool fail_unlink                           = false;

    // Optional deterministic worker barrier. It is called immediately before
    // the worker waits for the next wake token, and exceptions fail the store.
    std::function<void()> before_worker_wait;
};

struct capture_store_config {
    std::string root_path;
    mode        capture_mode         = mode::compact;
    bool        require_private_root = true;

    // The ring is allocated once.  It is the only object touched by the
    // producer after construction; a full ring drops the newest observation.
    size_t ring_capacity = 1024;

    // A shard is finalized before either bound is exceeded.  Limits include
    // framing and checksum overhead, not just records.
    uint32_t max_shard_records = 1024;
    uint64_t max_shard_bytes   = 1024 * 1024;

    // Retention is evaluated oldest-first after every committed shard.
    uint32_t max_retained_shards  = 64;
    uint64_t max_retained_records = 64 * 1024;
    uint64_t max_retained_bytes   = 64 * 1024 * 1024;
    uint64_t max_retained_age_ns  = 0;  // zero disables age retention
    uint64_t max_manifest_bytes   = 1024 * 1024;

    // Rich records include the five proposal token IDs and target correction
    // ID.  They are never emitted unless both capture_mode==sampled_rich and
    // this explicit opt-in is true.  A value of one samples every cycle.
    bool     allow_sampled_rich = false;
    uint32_t rich_sample_every  = 1;

    // Request IDs are never written.  This fixed salt is mixed into the
    // persisted 64-bit request tag, allowing correlation without exposing a
    // directly reversible identity.  Deployments should set a random salt.
    std::array<uint8_t, 16> identity_salt = {};

    capture_store_faults       faults;
    capture_store_cancel_check cancel_check;
};

struct capture_shard_info {
    uint64_t       sequence           = 0;
    uint64_t       first_monotonic_ns = 0;
    uint64_t       last_monotonic_ns  = 0;
    uint32_t       record_count       = 0;
    uint64_t       byte_count         = 0;
    capture_digest checksum           = {};
};

struct capture_manifest {
    uint32_t                        format_version = 0;
    uint64_t                        generation     = 0;
    mode                            capture_mode   = mode::off;
    std::array<uint8_t, 16>         identity_salt  = {};
    uint64_t                        total_records  = 0;
    uint64_t                        total_bytes    = 0;
    std::vector<capture_shard_info> shards;
};

struct capture_store_result {
    capture_store_status status    = capture_store_status::invalid_argument;
    int                  os_error  = 0;
    bool                 committed = false;
};

struct capture_store_stats {
    ring_stats ring;
    uint64_t   committed_shards    = 0;
    uint64_t   committed_records   = 0;
    uint64_t   committed_bytes     = 0;
    uint64_t   failed_writes       = 0;
    uint64_t   cancelled_writes    = 0;
    uint64_t   dropped_after_stop  = 0;
    uint64_t   dropped_on_shutdown = 0;
    bool       worker_running      = false;
    bool       worker_failed       = false;
};

// The store owns one background consumer.  Construction performs crash
// recovery before starting the worker; destruction is equivalent to
// shutdown(true).  The owning capture_store object must outlive every method
// call: shutdown() may run concurrently with producers that already entered
// try_enqueue(), but destruction requires external lifetime quiescence.  The
// try_enqueue() operation is noexcept, non-blocking, allocation-free after
// construction, and safe to call until shutdown begins while the owner lives.
class capture_store {
  public:
    explicit capture_store(capture_store_config config);
    ~capture_store();

    capture_store(const capture_store &)             = delete;
    capture_store & operator=(const capture_store &) = delete;

    bool try_enqueue(const cycle_observation & observation) noexcept;

    // Flush waits for all observations accepted before this call to reach a
    // committed shard.  It may block the caller, but never the producer.
    capture_store_result flush();

    // Requests a draining shutdown.  If drain is false, unread observations
    // are dropped and no new shard is started.  Calling shutdown repeatedly is
    // safe and returns the first terminal status.
    capture_store_result shutdown(bool drain = true);

    // Re-read and validate the current manifest.  This method does not start
    // a worker and is useful for restart/recovery tests and diagnostics.
    capture_store_result inspect(capture_manifest & manifest) const;

    // Validate every referenced shard checksum and size.  It is deliberately
    // separate from inspect() so a restart can publish a manifest before
    // paying the cost of reading all shard payloads.
    capture_store_result validate(const capture_manifest & manifest) const;

    capture_store_stats          stats() const noexcept;
    const capture_store_config & config() const noexcept;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

}  // namespace server_capture

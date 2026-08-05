#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace server_capture {

// The journal is deliberately a small, fixed replay suffix.  It is not a
// general request log and it has no hot-path integration.  Exactly 128
// accepted token slots are retained; one latest rejected proposal may be
// retained separately as a boundary diagnostic and is never replayed.
constexpr size_t   RESYNC_JOURNAL_MAX_EVENTS       = 128;
constexpr uint32_t RESYNC_JOURNAL_FORMAT_VERSION   = 1;
constexpr size_t   RESYNC_JOURNAL_HEADER_BYTES     = 96;
constexpr size_t   RESYNC_JOURNAL_EVENT_BYTES      = 60;
constexpr size_t   RESYNC_JOURNAL_FOOTER_BYTES     = 32;
constexpr size_t   RESYNC_JOURNAL_MAX_FILE_BYTES   = RESYNC_JOURNAL_HEADER_BYTES +
                                                   RESYNC_JOURNAL_MAX_EVENTS * RESYNC_JOURNAL_EVENT_BYTES +
                                                   RESYNC_JOURNAL_EVENT_BYTES +
                                                   RESYNC_JOURNAL_FOOTER_BYTES;
constexpr uint64_t RESYNC_JOURNAL_FIRST_EVENT_SEQ  = 1;
constexpr uint64_t RESYNC_JOURNAL_FIRST_REPLAY_SEQ = 1;

using resync_digest = std::array<uint8_t, 32>;

enum class resync_journal_status : uint8_t {
    ok = 0,
    invalid_argument,
    path_security,
    owner_busy,
    malformed,
    unsupported_version,
    truncated,
    trailing_data,
    checksum_mismatch,
    sequence_gap,
    sequence_overflow,
    token_position_gap,
    token_position_overflow,
    invalid_commit,
    out_of_order_commit,
    duplicate_commit,
    commit_conflict,
    invalid_event,
    too_many_events,
    io_error,
    no_space,
    short_write,
    commit_uncertain,
};

const char * resync_journal_status_name(resync_journal_status status) noexcept;

enum class resync_token_outcome : uint8_t {
    accepted = 0,
    rejected = 1,
};

// This identity is copied from the capture-store commit result by the owner
// that knows a capture observation has reached a committed shard.  The
// journal never opens or re-validates that shard: retention may delete it
// later, while the identity remains useful for replay provenance.
struct resync_capture_commit {
    uint64_t generation          = 0;
    uint64_t shard_sequence      = 0;
    uint32_t record_index        = 0;
    uint64_t observation_sequence = 0;
};

// Input event supplied by the owner after the corresponding capture commit.
// Token IDs are intentionally exact here: an exact replay suffix cannot be
// reconstructed from a hash.  They are written only when persistence is
// explicitly enabled and the private-root policy passes; capture-store
// background shards never receive these events.
struct resync_token_event_input {
    uint64_t              token_position = 0;
    int32_t               token_id       = 0;
    resync_token_outcome  outcome        = resync_token_outcome::accepted;
};

// Durable event returned by inspect().  event_sequence advances for every
// physical event.  replay_sequence advances only for accepted tokens;
// rejected proposals have replay_sequence==0 and are diagnostics only.
struct resync_journal_event {
    uint64_t             event_sequence       = 0;
    uint64_t             replay_sequence      = 0;
    uint64_t             token_position       = 0;
    int32_t              token_id             = 0;
    uint64_t             observation_sequence = 0;
    uint64_t             capture_generation   = 0;
    uint64_t             capture_shard        = 0;
    uint32_t             capture_record       = 0;
    resync_token_outcome outcome              = resync_token_outcome::accepted;
};

struct resync_journal_snapshot {
    // event_count/replay_count are the exact accepted replay suffix.  A
    // rejected proposal is kept separately and never consumes this 128-slot
    // replay capacity.
    uint32_t              event_count         = 0;
    uint32_t              replay_count        = 0;
    uint64_t              next_event_sequence = RESYNC_JOURNAL_FIRST_EVENT_SEQ;
    uint64_t              next_replay_sequence = RESYNC_JOURNAL_FIRST_REPLAY_SEQ;
    uint64_t              next_token_position = 0;
    resync_capture_commit last_commit;
    std::array<resync_journal_event, RESYNC_JOURNAL_MAX_EVENTS> events = {};

    bool                  has_rejection_boundary = false;
    resync_journal_event  rejection_boundary;

    // Ordered exact accepted-token suffix used to rebuild model state.  This
    // is derived from events and excludes rejected proposal diagnostics.
    std::array<resync_journal_event, RESYNC_JOURNAL_MAX_EVENTS> replay_tokens = {};
};

struct resync_journal_config {
    // Persistence is opt-in.  With persist=false the same bounded semantics
    // apply in memory and no filesystem path is required.
    bool        persist            = false;
    std::string root_path;
    bool        require_private_root = true;
    // Required for persist=true because exact token IDs are sensitive.  An
    // in-memory journal does not cross a process boundary and may be used by
    // a caller that has already accepted that narrower exposure.
    bool        allow_sensitive_tokens = false;

    // Deterministic publication failures used only by the focused lifecycle
    // tests. Persistent mode rejects require_private_root=false; exact token
    // IDs must never be written below a shared root.
    struct test_faults {
        bool fail_file_fsync      = false;
        bool fail_directory_fsync = false;
        bool crash_before_rename  = false;
        bool crash_after_rename   = false;
    } faults;

    // Per-stream token positions normally start at zero.  Tests and callers
    // restoring a known stream may seed a non-zero base; persisted snapshots
    // carry the resulting watermark forward.
    uint64_t    initial_token_position = 0;
};

struct resync_journal_result {
    resync_journal_status status    = resync_journal_status::invalid_argument;
    int                   os_error  = 0;
    bool                  committed = false;
};

// A synchronous, mutex-serialized prototype.  It is intentionally dormant:
// callers must invoke append_committed() from a background/control owner only
// after capture_store reports committed=true.  No inference path references
// this class, and append never blocks an inference producer by contract.  The
// owner must externally quiesce all calls before destruction; the internal
// mutex does not extend object lifetime.
class resync_journal {
  public:
    explicit resync_journal(resync_journal_config config);
    ~resync_journal();

    resync_journal(const resync_journal &)             = delete;
    resync_journal & operator=(const resync_journal &) = delete;

    // Appends one committed observation.  count==0 is a valid metadata-only
    // commit and does not advance any token sequence.  A batch may contain an
    // accepted prefix of up to 128 tokens followed by at most one final
    // rejected proposal (129 physical inputs in that one call).
    // Accepted token positions must continue from next_token_position.
    resync_journal_result append_committed(const resync_capture_commit &   commit,
                                           const resync_token_event_input * events,
                                           size_t                           count);

    // Returns the last valid bounded state.  Corrupt persisted state fails
    // closed; no best-effort partial replay is exposed.
    resync_journal_result inspect(resync_journal_snapshot & snapshot) const;

    const resync_journal_config & config() const noexcept;

  private:
    struct impl;
    std::unique_ptr<impl> data;
};

}  // namespace server_capture

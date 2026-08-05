#include "server-capture-resync-journal.h"
#include "server-capture-sha256.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace server_capture {
namespace {

constexpr std::array<uint8_t, 8> JOURNAL_MAGIC = {
    { 'S', 'C', 'R', 'J', 'N', 'L', '0', '1' }
};
constexpr char JOURNAL_NAME[] = "resync.journal";
constexpr char TEMP_NAME[]    = ".resync.journal.tmp";
constexpr char LOCK_NAME[]    = ".resync.journal.lock";
constexpr mode_t PRIVATE_FILE_MODE = S_IRUSR | S_IWUSR;
constexpr mode_t PRIVATE_DIRECTORY_MODE = S_IRWXU;

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

bool read_u32(const std::vector<uint8_t> & bytes, size_t & cursor, uint32_t & value) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < 4) {
        return false;
    }
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[cursor++]) << shift;
    }
    return true;
}

bool read_u64(const std::vector<uint8_t> & bytes, size_t & cursor, uint64_t & value) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < 8) {
        return false;
    }
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[cursor++]) << shift;
    }
    return true;
}

bool read_bytes(const std::vector<uint8_t> & bytes, size_t & cursor, void * destination, size_t size) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < size) {
        return false;
    }
    std::memcpy(destination, bytes.data() + cursor, size);
    cursor += size;
    return true;
}

void append_zeroes(std::vector<uint8_t> & bytes, size_t count) {
    bytes.insert(bytes.end(), count, 0);
}

bool path_has_traversal(const fs::path & path) {
    if (!path.is_absolute() || path == path.root_path()) {
        return true;
    }
    for (const fs::path & component : path) {
        if (component == "." || component == "..") {
            return true;
        }
    }
    return false;
}

// This is intentionally kept in lockstep with server-capture-store's
// open_private_root(): absolute no-follow walking, owner check, and exact
// 0700 enforcement when requested.  An existing leaf must already be exact;
// only a leaf created during this walk may be chmod'd into the private mode.
// The capture helper is currently private to that translation unit. Keeping
// the invariants identical avoids granting this dormant sidecar a weaker path
// authority until a shared helper can be extracted without changing the
// capture-store ABI.
resync_journal_status open_private_root(const fs::path & root, bool require_private, int & descriptor, int & os_error) {
    descriptor = -1;
    os_error  = 0;
    if (path_has_traversal(root)) {
        os_error = EINVAL;
        return resync_journal_status::path_security;
    }

    int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (current < 0) {
        os_error = errno;
        return resync_journal_status::io_error;
    }
    bool created_leaf = false;
    for (auto component_it = root.begin(); component_it != root.end(); ++component_it) {
        const fs::path & component = *component_it;
        const bool       is_leaf   = std::next(component_it) == root.end();
        if (component == root.root_path()) {
            continue;
        }
        const std::string name = component.string();
        int               next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        bool              created = false;
        if (next < 0 && errno == ENOENT) {
            const int mkdir_result = ::mkdirat(current, name.c_str(), PRIVATE_DIRECTORY_MODE);
            if (mkdir_result != 0) {
                if (errno != EEXIST) {
                    os_error = errno;
                    ::close(current);
                    return resync_journal_status::io_error;
                }
            } else {
                created = true;
            }
            if (is_leaf) {
                created_leaf = created;
            }
            next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (next < 0) {
            os_error = errno;
            ::close(current);
            return resync_journal_status::path_security;
        }
        struct stat status = {};
        if (::fstat(next, &status) != 0 || !S_ISDIR(status.st_mode)) {
            os_error = errno == 0 ? ENOTDIR : errno;
            ::close(next);
            ::close(current);
            return resync_journal_status::path_security;
        }
        ::close(current);
        current = next;
    }

    struct stat status = {};
    if (::fstat(current, &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::geteuid()) {
        os_error = errno == 0 ? EACCES : errno;
        ::close(current);
        return resync_journal_status::path_security;
    }
    if (require_private) {
        if (created_leaf) {
            if (::fchmod(current, PRIVATE_DIRECTORY_MODE) != 0 || ::fstat(current, &status) != 0 ||
                (status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
                os_error = errno == 0 ? EACCES : errno;
                ::close(current);
                return resync_journal_status::path_security;
            }
        } else if ((status.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
            os_error = errno == 0 ? EACCES : errno;
            ::close(current);
            return resync_journal_status::path_security;
        }
    }
    descriptor = current;
    return resync_journal_status::ok;
}

resync_journal_status acquire_owner_lock(int root_fd, int & descriptor, int & os_error) {
    descriptor = -1;
    os_error  = 0;
    const int lock = ::openat(root_fd, LOCK_NAME, O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, PRIVATE_FILE_MODE);
    if (lock < 0) {
        os_error = errno;
        return errno == EACCES || errno == ELOOP ? resync_journal_status::path_security :
                                                   resync_journal_status::io_error;
    }
    struct stat status = {};
    if (::fstat(lock, &status) != 0 || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        ::fchmod(lock, PRIVATE_FILE_MODE) != 0 || ::fstat(lock, &status) != 0 ||
        (status.st_mode & 07777) != PRIVATE_FILE_MODE) {
        os_error = errno == 0 ? EACCES : errno;
        ::close(lock);
        return resync_journal_status::path_security;
    }
    int flock_result = 0;
    do {
        flock_result = ::flock(lock, LOCK_EX | LOCK_NB);
    } while (flock_result != 0 && errno == EINTR);
    if (flock_result != 0) {
        os_error = errno;
        ::close(lock);
        return errno == EWOULDBLOCK || errno == EAGAIN ? resync_journal_status::owner_busy :
                                                          resync_journal_status::io_error;
    }
    descriptor = lock;
    return resync_journal_status::ok;
}

resync_journal_result result(resync_journal_status status, int os_error = 0, bool committed = false) {
    return { status, os_error, committed };
}

bool equal_commit(const resync_capture_commit & left, const resync_capture_commit & right) noexcept {
    return left.generation == right.generation && left.shard_sequence == right.shard_sequence &&
           left.record_index == right.record_index && left.observation_sequence == right.observation_sequence;
}

int compare_commit(const resync_capture_commit & left, const resync_capture_commit & right) noexcept {
    if (left.generation != right.generation) {
        return left.generation < right.generation ? -1 : 1;
    }
    if (left.shard_sequence != right.shard_sequence) {
        return left.shard_sequence < right.shard_sequence ? -1 : 1;
    }
    if (left.record_index != right.record_index) {
        return left.record_index < right.record_index ? -1 : 1;
    }
    if (left.observation_sequence != right.observation_sequence) {
        return left.observation_sequence < right.observation_sequence ? -1 : 1;
    }
    return 0;
}

resync_capture_commit event_commit(const resync_journal_event & event) noexcept {
    return { event.capture_generation, event.capture_shard, event.capture_record, event.observation_sequence };
}

struct journal_state {
    uint32_t accepted_count       = 0;
    uint64_t next_event_sequence  = RESYNC_JOURNAL_FIRST_EVENT_SEQ;
    uint64_t next_replay_sequence = RESYNC_JOURNAL_FIRST_REPLAY_SEQ;
    uint64_t next_token_position  = 0;
    resync_capture_commit last_commit;
    uint64_t               last_commit_event_sequence = 0;
    std::array<resync_journal_event, RESYNC_JOURNAL_MAX_EVENTS> events = {};
    bool                  boundary_present = false;
    resync_journal_event  rejection_boundary;
};

void append_event_bytes(std::vector<uint8_t> & bytes, const resync_journal_event & event) {
    append_u64(bytes, event.event_sequence);
    append_u64(bytes, event.replay_sequence);
    append_u64(bytes, event.token_position);
    append_u32(bytes, static_cast<uint32_t>(event.token_id));
    append_u64(bytes, event.observation_sequence);
    append_u64(bytes, event.capture_generation);
    append_u64(bytes, event.capture_shard);
    append_u32(bytes, event.capture_record);
    bytes.push_back(static_cast<uint8_t>(event.outcome));
    append_zeroes(bytes, 3);
}

bool read_event(const std::vector<uint8_t> & bytes, size_t & cursor, resync_journal_event & event) noexcept {
    uint32_t token_id = 0;
    uint32_t outcome  = 0;
    if (!read_u64(bytes, cursor, event.event_sequence) || !read_u64(bytes, cursor, event.replay_sequence) ||
        !read_u64(bytes, cursor, event.token_position) || !read_u32(bytes, cursor, token_id) ||
        !read_u64(bytes, cursor, event.observation_sequence) || !read_u64(bytes, cursor, event.capture_generation) ||
        !read_u64(bytes, cursor, event.capture_shard) || !read_u32(bytes, cursor, event.capture_record) ||
        !read_bytes(bytes, cursor, &outcome, 1)) {
        return false;
    }
    event.token_id = static_cast<int32_t>(token_id);
    event.outcome  = static_cast<resync_token_outcome>(outcome & 0xffU);
    uint8_t reserved[3] = {};
    if (!read_bytes(bytes, cursor, reserved, sizeof(reserved))) {
        return false;
    }
    return reserved[0] == 0 && reserved[1] == 0 && reserved[2] == 0;
}

std::vector<uint8_t> serialize_state(const journal_state & state) {
    std::vector<uint8_t> bytes;
    bytes.reserve(RESYNC_JOURNAL_HEADER_BYTES +
                  static_cast<size_t>(state.accepted_count + (state.boundary_present ? 1U : 0U)) *
                      RESYNC_JOURNAL_EVENT_BYTES +
                  RESYNC_JOURNAL_FOOTER_BYTES);
    bytes.insert(bytes.end(), JOURNAL_MAGIC.begin(), JOURNAL_MAGIC.end());
    append_u32(bytes, RESYNC_JOURNAL_FORMAT_VERSION);
    append_u32(bytes, 0);  // flags/reserved
    append_u32(bytes, state.accepted_count);
    append_u32(bytes, 0);  // reserved
    append_u64(bytes, state.next_event_sequence);
    append_u64(bytes, state.next_replay_sequence);
    append_u64(bytes, state.next_token_position);
    append_u64(bytes, state.last_commit.generation);
    append_u64(bytes, state.last_commit.shard_sequence);
    append_u32(bytes, state.last_commit.record_index);
    append_u32(bytes, 0);  // reserved
    append_u64(bytes, state.last_commit.observation_sequence);
    append_u32(bytes, state.boundary_present ? 1U : 0U);
    append_u32(bytes, 0);  // reserved
    append_u64(bytes, state.last_commit_event_sequence);
    for (size_t index = 0; index < state.accepted_count; ++index) {
        append_event_bytes(bytes, state.events[index]);
    }
    if (state.boundary_present) {
        append_event_bytes(bytes, state.rejection_boundary);
    }
    const resync_digest checksum = capture_sha256::hash(bytes.data(), bytes.size());
    bytes.insert(bytes.end(), checksum.begin(), checksum.end());
    return bytes;
}

resync_journal_result parse_state(const std::vector<uint8_t> & bytes, journal_state & state,
                                  uint64_t initial_token_position) {
    if (bytes.size() < RESYNC_JOURNAL_HEADER_BYTES + RESYNC_JOURNAL_FOOTER_BYTES) {
        return result(resync_journal_status::truncated);
    }
    if (bytes.size() > RESYNC_JOURNAL_MAX_FILE_BYTES) {
        return result(resync_journal_status::trailing_data);
    }

    size_t cursor = 0;
    std::array<uint8_t, 8> magic = {};
    uint32_t version = 0;
    uint32_t flags   = 0;
    uint32_t count   = 0;
    uint32_t reserved = 0;
    uint32_t boundary = 0;
    if (!read_bytes(bytes, cursor, magic.data(), magic.size()) || !read_u32(bytes, cursor, version) ||
        !read_u32(bytes, cursor, flags) || !read_u32(bytes, cursor, count) || !read_u32(bytes, cursor, reserved)) {
        return result(resync_journal_status::malformed);
    }
    if (!std::equal(magic.begin(), magic.end(), JOURNAL_MAGIC.begin())) {
        return result(resync_journal_status::malformed);
    }
    if (version != RESYNC_JOURNAL_FORMAT_VERSION) {
        return result(resync_journal_status::unsupported_version);
    }
    if (flags != 0 || reserved != 0 || count > RESYNC_JOURNAL_MAX_EVENTS ||
        !read_u64(bytes, cursor, state.next_event_sequence) || !read_u64(bytes, cursor, state.next_replay_sequence) ||
        !read_u64(bytes, cursor, state.next_token_position) || !read_u64(bytes, cursor, state.last_commit.generation) ||
        !read_u64(bytes, cursor, state.last_commit.shard_sequence) || !read_u32(bytes, cursor, state.last_commit.record_index) ||
        !read_u32(bytes, cursor, reserved) || !read_u64(bytes, cursor, state.last_commit.observation_sequence) ||
        reserved != 0 || !read_u32(bytes, cursor, boundary) || !read_u32(bytes, cursor, reserved) || reserved != 0 ||
        !read_u64(bytes, cursor, state.last_commit_event_sequence) || boundary > 1) {
        return result(resync_journal_status::malformed);
    }
    const size_t expected_size = RESYNC_JOURNAL_HEADER_BYTES +
                                 (static_cast<size_t>(count) + (boundary != 0 ? 1U : 0U)) * RESYNC_JOURNAL_EVENT_BYTES +
                                 RESYNC_JOURNAL_FOOTER_BYTES;
    if (bytes.size() < expected_size) {
        return result(resync_journal_status::truncated);
    }
    if (bytes.size() > expected_size) {
        return result(resync_journal_status::trailing_data);
    }
    if (cursor != RESYNC_JOURNAL_HEADER_BYTES) {
        return result(resync_journal_status::malformed);
    }
    const size_t checksum_offset = bytes.size() - RESYNC_JOURNAL_FOOTER_BYTES;
    resync_digest expected_checksum = {};
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(checksum_offset), expected_checksum.size(),
                expected_checksum.begin());
    const capture_sha256::digest computed_checksum = capture_sha256::hash(bytes.data(), checksum_offset);
    if (!capture_sha256::equal(expected_checksum, computed_checksum)) {
        return result(resync_journal_status::checksum_mismatch);
    }

    state.accepted_count = count;
    state.boundary_present = boundary != 0;
    for (size_t index = 0; index < count; ++index) {
        if (!read_event(bytes, cursor, state.events[index])) {
            return result(resync_journal_status::malformed);
        }
        if (state.events[index].outcome != resync_token_outcome::accepted || state.events[index].replay_sequence == 0 ||
            state.events[index].capture_generation == 0 || state.events[index].capture_shard == 0 ||
            state.events[index].observation_sequence == 0) {
            return result(resync_journal_status::invalid_event);
        }
    }
    if (state.boundary_present) {
        if (!read_event(bytes, cursor, state.rejection_boundary) ||
            state.rejection_boundary.outcome != resync_token_outcome::rejected ||
            state.rejection_boundary.replay_sequence != 0 || state.rejection_boundary.capture_generation == 0 ||
            state.rejection_boundary.capture_shard == 0 || state.rejection_boundary.observation_sequence == 0) {
            return result(resync_journal_status::invalid_event);
        }
    }
    if (cursor != checksum_offset) {
        return result(resync_journal_status::trailing_data);
    }
    if (state.next_event_sequence == 0 || state.next_replay_sequence == 0) {
        return result(resync_journal_status::sequence_overflow);
    }

    // Accepted replay sequences and positions are contiguous within the
    // retained suffix. Physical event sequences may contain gaps because
    // rejected diagnostics are outside this ring, but must remain increasing.
    if (count != 0) {
        uint64_t expected_replay = state.events[0].replay_sequence;
        uint64_t expected_position = state.events[0].token_position;
        uint64_t previous_event = state.events[0].event_sequence;
        if (expected_replay == 0 || previous_event == 0) {
            return result(resync_journal_status::sequence_gap);
        }
        const uint64_t replay_offset = expected_replay - RESYNC_JOURNAL_FIRST_REPLAY_SEQ;
        if (initial_token_position > UINT64_MAX - replay_offset ||
            expected_position != initial_token_position + replay_offset) {
            return result(resync_journal_status::token_position_gap);
        }
        for (size_t index = 0; index < count; ++index) {
            const resync_journal_event & event = state.events[index];
            if (index != 0) {
                if (event.event_sequence <= previous_event) {
                    return result(resync_journal_status::sequence_gap);
                }
                previous_event = event.event_sequence;
                if (expected_replay == UINT64_MAX) {
                    return result(resync_journal_status::sequence_overflow);
                }
                ++expected_replay;
                if (event.replay_sequence != expected_replay) {
                    return result(resync_journal_status::sequence_gap);
                }
            }
            if (event.token_position != expected_position) {
                return result(resync_journal_status::token_position_gap);
            }
            if (expected_position == UINT64_MAX) {
                return result(resync_journal_status::token_position_overflow);
            }
            ++expected_position;
        }
        if (state.next_token_position != expected_position) {
            return result(resync_journal_status::token_position_gap);
        }
        if (expected_replay == UINT64_MAX) {
            return result(resync_journal_status::sequence_overflow);
        }
        if (state.next_replay_sequence != expected_replay + 1) {
            return result(resync_journal_status::sequence_gap);
        }
        const resync_journal_event & last = state.events[count - 1];
        if (last.capture_generation == 0 || last.capture_shard == 0 || last.observation_sequence == 0) {
            return result(resync_journal_status::invalid_commit);
        }
        if (count < RESYNC_JOURNAL_MAX_EVENTS && state.events[0].replay_sequence != RESYNC_JOURNAL_FIRST_REPLAY_SEQ) {
            return result(resync_journal_status::sequence_gap);
        }
    } else {
        if (state.next_replay_sequence != RESYNC_JOURNAL_FIRST_REPLAY_SEQ ||
            state.next_token_position != initial_token_position) {
            return result(resync_journal_status::sequence_gap);
        }
        if (!state.boundary_present && state.next_event_sequence != RESYNC_JOURNAL_FIRST_EVENT_SEQ) {
            return result(resync_journal_status::sequence_gap);
        }
        if (state.boundary_present && state.next_event_sequence <= RESYNC_JOURNAL_FIRST_EVENT_SEQ) {
            return result(resync_journal_status::sequence_gap);
        }
    }
    if (state.boundary_present) {
        const uint64_t boundary_sequence = state.rejection_boundary.event_sequence;
        if (boundary_sequence == 0 || boundary_sequence >= state.next_event_sequence) {
            return result(resync_journal_status::sequence_gap);
        }
        if (state.rejection_boundary.token_position < initial_token_position ||
            state.rejection_boundary.token_position > state.next_token_position) {
            return result(resync_journal_status::token_position_gap);
        }
        if (count == 0) {
            if (state.rejection_boundary.token_position != initial_token_position) {
                return result(resync_journal_status::token_position_gap);
            }
        } else {
            const auto first = state.events.begin();
            const auto last  = first + static_cast<std::ptrdiff_t>(count);
            const auto upper = std::lower_bound(first, last, boundary_sequence,
                                                [](const resync_journal_event & event, uint64_t sequence) {
                                                    return event.event_sequence < sequence;
                                                });
            if (upper == first) {
                if (state.rejection_boundary.token_position > state.events[0].token_position) {
                    return result(resync_journal_status::token_position_gap);
                }
            } else if (upper == last) {
                if (state.rejection_boundary.token_position != state.next_token_position) {
                    return result(resync_journal_status::token_position_gap);
                }
            } else if (state.rejection_boundary.token_position != upper->token_position) {
                return result(resync_journal_status::token_position_gap);
            }
        }
        for (size_t index = 0; index < count; ++index) {
            if (state.events[index].event_sequence == state.rejection_boundary.event_sequence) {
                return result(resync_journal_status::sequence_gap);
            }
        }
    }

    std::array<const resync_journal_event *, RESYNC_JOURNAL_MAX_EVENTS + 1> retained_events = {};
    size_t retained_count = 0;
    for (size_t index = 0; index < count; ++index) {
        retained_events[retained_count++] = &state.events[index];
    }
    if (state.boundary_present) {
        retained_events[retained_count++] = &state.rejection_boundary;
    }
    std::sort(retained_events.begin(), retained_events.begin() + static_cast<std::ptrdiff_t>(retained_count),
              [](const resync_journal_event * left, const resync_journal_event * right) {
                  return left->event_sequence < right->event_sequence;
              });
    for (size_t index = 1; index < retained_count; ++index) {
        if (compare_commit(event_commit(*retained_events[index - 1]), event_commit(*retained_events[index])) > 0) {
            return result(resync_journal_status::invalid_commit);
        }
    }
    if (retained_count != 0 && state.last_commit_event_sequence == 0 &&
        compare_commit(event_commit(*retained_events[retained_count - 1]), state.last_commit) >= 0) {
        // A zero event watermark denotes a metadata-only commit.  It must be
        // strictly newer than the latest physical event; equal identities
        // with different event counts are conflicts, not duplicates.
        return result(resync_journal_status::invalid_commit);
    }
    uint64_t retained_latest_sequence = count != 0 ? state.events[count - 1].event_sequence : 0;
    if (state.boundary_present) {
        retained_latest_sequence = std::max(retained_latest_sequence, state.rejection_boundary.event_sequence);
    }
    if (state.last_commit_event_sequence != 0 && state.last_commit_event_sequence != retained_latest_sequence) {
        return result(resync_journal_status::sequence_gap);
    }
    if (state.last_commit_event_sequence != 0) {
        const resync_journal_event * latest = nullptr;
        for (size_t index = 0; index < count; ++index) {
            if (state.events[index].event_sequence == state.last_commit_event_sequence) {
                latest = &state.events[index];
                break;
            }
        }
        if (state.boundary_present && state.rejection_boundary.event_sequence == state.last_commit_event_sequence) {
            latest = &state.rejection_boundary;
        }
        if (latest == nullptr ||
            latest->capture_generation != state.last_commit.generation || latest->capture_shard != state.last_commit.shard_sequence ||
            latest->capture_record != state.last_commit.record_index ||
            latest->observation_sequence != state.last_commit.observation_sequence) {
            return result(resync_journal_status::invalid_commit);
        }
    } else if (state.last_commit.generation == 0 || state.last_commit.shard_sequence == 0 ||
               state.last_commit.observation_sequence == 0) {
        // A zero watermark is only valid for a pristine image; persisted
        // snapshots with no token event still carry a non-zero metadata-only
        // commit identity.
        if (count != 0 || state.boundary_present) {
            return result(resync_journal_status::invalid_commit);
        }
    }
    if (count != 0 && state.next_event_sequence <= state.events[count - 1].event_sequence) {
        return result(resync_journal_status::sequence_gap);
    }
    return result(resync_journal_status::ok, 0, true);
}

resync_journal_result read_private_file(int root_fd, std::vector<uint8_t> & bytes) {
    struct stat status = {};
    if (::fstatat(root_fd, JOURNAL_NAME, &status, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno == ENOENT) {
            bytes.clear();
            return result(resync_journal_status::ok, 0, true);
        }
        return result(resync_journal_status::path_security, errno);
    }
    if (S_ISLNK(status.st_mode) || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (status.st_mode & 07777) != PRIVATE_FILE_MODE) {
        return result(resync_journal_status::path_security, EACCES);
    }
    if (status.st_size <= 0) {
        return result(resync_journal_status::truncated);
    }
    if (static_cast<uint64_t>(status.st_size) > RESYNC_JOURNAL_MAX_FILE_BYTES) {
        return result(resync_journal_status::trailing_data, EFBIG);
    }
    const int descriptor = ::openat(root_fd, JOURNAL_NAME, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) {
        return result(resync_journal_status::io_error, errno);
    }
    bytes.assign(static_cast<size_t>(status.st_size), 0);
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            const int error = count == 0 ? EIO : errno;
            ::close(descriptor);
            return result(resync_journal_status::truncated, error);
        }
        offset += static_cast<size_t>(count);
    }
    if (::close(descriptor) != 0) {
        return result(resync_journal_status::io_error, errno);
    }
    return result(resync_journal_status::ok, 0, true);
}

resync_journal_result validate_private_file_at(int root_fd, bool allow_missing) {
    struct stat status = {};
    if (::fstatat(root_fd, JOURNAL_NAME, &status, AT_SYMLINK_NOFOLLOW) != 0) {
        if (allow_missing && errno == ENOENT) {
            return result(resync_journal_status::ok, 0, true);
        }
        return result(resync_journal_status::path_security, errno);
    }
    if (S_ISLNK(status.st_mode) || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (status.st_mode & 07777) != PRIVATE_FILE_MODE) {
        return result(resync_journal_status::path_security, EACCES);
    }
    return result(resync_journal_status::ok, 0, true);
}

resync_journal_result write_private_file(int root_fd, const std::vector<uint8_t> & bytes,
                                          const resync_journal_config::test_faults & faults) {
    const resync_journal_result final_status = validate_private_file_at(root_fd, true);
    if (final_status.status != resync_journal_status::ok) {
        return final_status;
    }
    int  descriptor = ::openat(root_fd, TEMP_NAME,
                               O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK,
                               PRIVATE_FILE_MODE);
    bool created_temp = descriptor >= 0;
    if (descriptor < 0 && errno == EEXIST) {
        descriptor = ::openat(root_fd, TEMP_NAME, O_WRONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
    }
    if (descriptor < 0) {
        const int error = errno;
        return result(error == ELOOP || error == EISDIR ? resync_journal_status::path_security :
                                                          (error == ENOSPC ? resync_journal_status::no_space :
                                                                              resync_journal_status::io_error),
                      error);
    }
    struct stat temporary_status = {};
    if (::fstat(descriptor, &temporary_status) != 0 || !S_ISREG(temporary_status.st_mode) ||
        temporary_status.st_uid != ::geteuid() ||
        (!created_temp && (temporary_status.st_mode & 07777) != PRIVATE_FILE_MODE)) {
        const int error = errno == 0 ? EACCES : errno;
        ::close(descriptor);
        if (created_temp) {
            ::unlinkat(root_fd, TEMP_NAME, 0);
        }
        return result(resync_journal_status::path_security, error);
    }
    if (created_temp) {
        if (::fchmod(descriptor, PRIVATE_FILE_MODE) != 0 || ::fstat(descriptor, &temporary_status) != 0 ||
            (temporary_status.st_mode & 07777) != PRIVATE_FILE_MODE) {
            const int error = errno == 0 ? EACCES : errno;
            ::close(descriptor);
            ::unlinkat(root_fd, TEMP_NAME, 0);
            return result(resync_journal_status::io_error, error);
        }
    }
    if (::ftruncate(descriptor, 0) != 0) {
        const int error = errno;
        ::close(descriptor);
        if (created_temp) {
            ::unlinkat(root_fd, TEMP_NAME, 0);
        }
        return result(error == ENOSPC ? resync_journal_status::no_space : resync_journal_status::io_error, error);
    }
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0) {
            const int error = count == 0 ? EIO : errno;
            ::close(descriptor);
            ::unlinkat(root_fd, TEMP_NAME, 0);
            return result(count == 0 ? resync_journal_status::short_write
                                     : (error == ENOSPC ? resync_journal_status::no_space
                                                        : resync_journal_status::io_error),
                         error);
        }
        offset += static_cast<size_t>(count);
    }
    if (faults.fail_file_fsync || ::fsync(descriptor) != 0) {
        const int error = faults.fail_file_fsync ? EIO : errno;
        ::close(descriptor);
        ::unlinkat(root_fd, TEMP_NAME, 0);
        return result(resync_journal_status::io_error, error);
    }
    if (::close(descriptor) != 0) {
        const int error = errno;
        ::unlinkat(root_fd, TEMP_NAME, 0);
        return result(resync_journal_status::io_error, error);
    }
    if (faults.crash_before_rename) {
        return result(resync_journal_status::commit_uncertain, 0, false);
    }
    const resync_journal_result final_before_rename = validate_private_file_at(root_fd, true);
    if (final_before_rename.status != resync_journal_status::ok) {
        ::unlinkat(root_fd, TEMP_NAME, 0);
        return final_before_rename;
    }
    if (::renameat(root_fd, TEMP_NAME, root_fd, JOURNAL_NAME) != 0) {
        const int error = errno;
        ::unlinkat(root_fd, TEMP_NAME, 0);
        return result(error == ENOSPC ? resync_journal_status::no_space : resync_journal_status::io_error, error);
    }
    // A successful rename publishes a complete checksum-verified snapshot. If
    // this directory fence fails, the new file is visible but durability is
    // uncertain; recovery will validate either the old or new complete image.
    if (faults.crash_after_rename) {
        return result(resync_journal_status::commit_uncertain, 0, true);
    }
    if (faults.fail_directory_fsync || ::fsync(root_fd) != 0) {
        return result(resync_journal_status::commit_uncertain, faults.fail_directory_fsync ? EIO : errno, true);
    }
    return result(resync_journal_status::ok, 0, true);
}

resync_journal_result validate_input(const resync_capture_commit & commit,
                                     const resync_token_event_input * events,
                                     size_t count) {
    if (commit.generation == 0 || commit.shard_sequence == 0 || commit.observation_sequence == 0) {
        return result(resync_journal_status::invalid_commit);
    }
    if (count > RESYNC_JOURNAL_MAX_EVENTS + 1U) {
        return result(resync_journal_status::too_many_events);
    }
    if (count != 0 && events == nullptr) {
        return result(resync_journal_status::invalid_argument);
    }
    bool saw_rejection = false;
    for (size_t index = 0; index < count; ++index) {
        const resync_token_event_input & event = events[index];
        if (event.outcome != resync_token_outcome::accepted && event.outcome != resync_token_outcome::rejected) {
            return result(resync_journal_status::invalid_event);
        }
        if (event.outcome == resync_token_outcome::rejected) {
            if (saw_rejection || index + 1 != count) {
                return result(resync_journal_status::invalid_event);
            }
            saw_rejection = true;
        } else if (saw_rejection) {
            return result(resync_journal_status::invalid_event);
        }
    }
    if (count == RESYNC_JOURNAL_MAX_EVENTS + 1U && !saw_rejection) {
        return result(resync_journal_status::too_many_events);
    }
    return result(resync_journal_status::ok);
}

bool event_matches_input(const resync_journal_event & event,
                         const resync_capture_commit & commit,
                         const resync_token_event_input & input) noexcept {
    return event.token_position == input.token_position && event.token_id == input.token_id &&
           event.outcome == input.outcome && event.capture_generation == commit.generation &&
           event.capture_shard == commit.shard_sequence && event.capture_record == commit.record_index &&
           event.observation_sequence == commit.observation_sequence;
}

size_t collect_commit_events(const journal_state & state, const resync_capture_commit & commit,
                             std::array<const resync_journal_event *, RESYNC_JOURNAL_MAX_EVENTS + 1> & matches) noexcept {
    size_t count = 0;
    for (size_t index = 0; index < state.accepted_count; ++index) {
        const resync_journal_event & event = state.events[index];
        const bool same = event.capture_generation == commit.generation && event.capture_shard == commit.shard_sequence &&
                          event.capture_record == commit.record_index &&
                          event.observation_sequence == commit.observation_sequence;
        if (same) {
            matches[count++] = &event;
        }
    }
    if (state.boundary_present &&
        state.rejection_boundary.capture_generation == commit.generation &&
        state.rejection_boundary.capture_shard == commit.shard_sequence &&
        state.rejection_boundary.capture_record == commit.record_index &&
        state.rejection_boundary.observation_sequence == commit.observation_sequence) {
        matches[count++] = &state.rejection_boundary;
    }
    std::sort(matches.begin(), matches.begin() + static_cast<std::ptrdiff_t>(count),
              [](const resync_journal_event * left, const resync_journal_event * right) {
                  return left->event_sequence < right->event_sequence;
              });
    return count;
}

void append_to_ring(journal_state & state, const resync_journal_event & event) {
    // Only accepted tokens consume the exact replay suffix. Rejected
    // proposals are kept in one separate latest-boundary slot.
    if (event.outcome == resync_token_outcome::rejected) {
        state.rejection_boundary = event;
        state.boundary_present   = true;
        return;
    }
    if (state.accepted_count < RESYNC_JOURNAL_MAX_EVENTS) {
        state.events[state.accepted_count++] = event;
    } else {
        std::move(state.events.begin() + 1, state.events.end(), state.events.begin());
        state.events.back() = event;
    }
}

void fill_snapshot(const journal_state & state, resync_journal_snapshot & snapshot) {
    snapshot = {};
    snapshot.event_count          = state.accepted_count;
    snapshot.next_event_sequence  = state.next_event_sequence;
    snapshot.next_replay_sequence = state.next_replay_sequence;
    snapshot.next_token_position  = state.next_token_position;
    snapshot.last_commit          = state.last_commit;
    snapshot.has_rejection_boundary = state.boundary_present;
    if (state.boundary_present) {
        snapshot.rejection_boundary = state.rejection_boundary;
    }
    for (size_t index = 0; index < state.accepted_count; ++index) {
        snapshot.events[index] = state.events[index];
        snapshot.replay_tokens[snapshot.replay_count++] = state.events[index];
    }
}

}  // namespace

const char * resync_journal_status_name(resync_journal_status status) noexcept {
    switch (status) {
        case resync_journal_status::ok:
            return "ok";
        case resync_journal_status::invalid_argument:
            return "invalid_argument";
        case resync_journal_status::path_security:
            return "path_security";
        case resync_journal_status::owner_busy:
            return "owner_busy";
        case resync_journal_status::malformed:
            return "malformed";
        case resync_journal_status::unsupported_version:
            return "unsupported_version";
        case resync_journal_status::truncated:
            return "truncated";
        case resync_journal_status::trailing_data:
            return "trailing_data";
        case resync_journal_status::checksum_mismatch:
            return "checksum_mismatch";
        case resync_journal_status::sequence_gap:
            return "sequence_gap";
        case resync_journal_status::sequence_overflow:
            return "sequence_overflow";
        case resync_journal_status::token_position_gap:
            return "token_position_gap";
        case resync_journal_status::token_position_overflow:
            return "token_position_overflow";
        case resync_journal_status::invalid_commit:
            return "invalid_commit";
        case resync_journal_status::out_of_order_commit:
            return "out_of_order_commit";
        case resync_journal_status::duplicate_commit:
            return "duplicate_commit";
        case resync_journal_status::commit_conflict:
            return "commit_conflict";
        case resync_journal_status::invalid_event:
            return "invalid_event";
        case resync_journal_status::too_many_events:
            return "too_many_events";
        case resync_journal_status::io_error:
            return "io_error";
        case resync_journal_status::no_space:
            return "no_space";
        case resync_journal_status::short_write:
            return "short_write";
        case resync_journal_status::commit_uncertain:
            return "commit_uncertain";
    }
    return "unknown";
}

struct resync_journal::impl {
    explicit impl(resync_journal_config input) : config(std::move(input)) {
        state.next_token_position = config.initial_token_position;
        if (!config.persist) {
            return;
        }
        if (!config.allow_sensitive_tokens || config.root_path.empty() || !config.require_private_root) {
            throw std::invalid_argument(
                    "persistent resync journal requires explicit sensitive-token opt-in, private root, and root path");
        }
        int os_error = 0;
        const resync_journal_status status =
            open_private_root(fs::path(config.root_path), true, root_fd, os_error);
        if (status != resync_journal_status::ok) {
            throw std::invalid_argument("resync journal root failed: " +
                                        std::string(resync_journal_status_name(status)));
        }
        const resync_journal_status lock_status = acquire_owner_lock(root_fd, lock_fd, os_error);
        if (lock_status != resync_journal_status::ok) {
            ::close(root_fd);
            root_fd = -1;
            throw std::invalid_argument("resync journal owner lock failed: " +
                                        std::string(resync_journal_status_name(lock_status)));
        }
        std::vector<uint8_t> bytes;
        const resync_journal_result read_result = read_private_file(root_fd, bytes);
        if (read_result.status != resync_journal_status::ok) {
            startup_status = read_result.status;
            startup_error  = read_result.os_error;
            return;
        }
        if (bytes.empty()) {
            return;
        }
        const resync_journal_result parse_result = parse_state(bytes, state, config.initial_token_position);
        startup_status = parse_result.status;
        startup_error  = parse_result.os_error;
    }

    ~impl() {
        if (lock_fd >= 0) {
            ::flock(lock_fd, LOCK_UN);
            ::close(lock_fd);
        }
        if (root_fd >= 0) {
            ::close(root_fd);
        }
    }

    resync_journal_config config;
    mutable std::mutex    mutex;
    journal_state          state;
    resync_journal_status  startup_status = resync_journal_status::ok;
    int                    startup_error  = 0;
    int                    lock_fd        = -1;
    int                    root_fd        = -1;
};

resync_journal::resync_journal(resync_journal_config config) : data(std::make_unique<impl>(std::move(config))) {}

resync_journal::~resync_journal() = default;

resync_journal_result resync_journal::append_committed(const resync_capture_commit &   commit,
                                                       const resync_token_event_input * events,
                                                       size_t                           count) {
    impl & implementation = *data;
    std::lock_guard<std::mutex> lock(implementation.mutex);
    if (implementation.startup_status != resync_journal_status::ok) {
        return result(implementation.startup_status, implementation.startup_error);
    }
    const resync_journal_result input_result = validate_input(commit, events, count);
    if (input_result.status != resync_journal_status::ok) {
        return input_result;
    }
    if (implementation.state.last_commit.generation != 0) {
        if (compare_commit(commit, implementation.state.last_commit) < 0) {
            return result(resync_journal_status::out_of_order_commit);
        }
        std::array<const resync_journal_event *, RESYNC_JOURNAL_MAX_EVENTS + 1> existing = {};
        const size_t existing_count = collect_commit_events(implementation.state, commit, existing);
        if (existing_count != 0) {
            if (existing_count != count) {
                return result(resync_journal_status::commit_conflict);
            }
            for (size_t index = 0; index < count; ++index) {
                if (!event_matches_input(*existing[index], commit, events[index])) {
                    return result(resync_journal_status::commit_conflict);
                }
            }
            return result(resync_journal_status::duplicate_commit);
        }
        if (equal_commit(commit, implementation.state.last_commit)) {
            // The previous commit had zero events, so it cannot have been
            // found in the physical ring.  A retry with non-zero events is a
            // conflict; a retry with zero events is an idempotent duplicate.
            return count == 0 ? result(resync_journal_status::duplicate_commit)
                              : result(resync_journal_status::commit_conflict);
        }
    }

    if (count != 0) {
        if (implementation.state.next_event_sequence > UINT64_MAX - count) {
            return result(resync_journal_status::sequence_overflow);
        }
        size_t accepted_count = 0;
        for (size_t index = 0; index < count; ++index) {
            if (events[index].outcome == resync_token_outcome::accepted) {
                ++accepted_count;
            }
        }
        if (accepted_count != 0 && implementation.state.next_replay_sequence > UINT64_MAX - accepted_count) {
            return result(resync_journal_status::sequence_overflow);
        }
        uint64_t expected_position = implementation.state.next_token_position;
        for (size_t index = 0; index < count; ++index) {
            if (events[index].token_position != expected_position) {
                return result(resync_journal_status::token_position_gap);
            }
            const bool accepted = events[index].outcome == resync_token_outcome::accepted;
            if (accepted) {
                if (expected_position == UINT64_MAX) {
                    return result(resync_journal_status::token_position_overflow);
                }
                ++expected_position;
            }
        }
    }

    journal_state candidate = implementation.state;
    candidate.last_commit  = commit;
    candidate.last_commit_event_sequence = 0;
    uint64_t expected_position = candidate.next_token_position;
    for (size_t index = 0; index < count; ++index) {
        resync_journal_event event;
        event.event_sequence       = candidate.next_event_sequence++;
        event.replay_sequence      = 0;
        event.token_position       = events[index].token_position;
        event.token_id             = events[index].token_id;
        event.observation_sequence = commit.observation_sequence;
        event.capture_generation   = commit.generation;
        event.capture_shard        = commit.shard_sequence;
        event.capture_record       = commit.record_index;
        event.outcome              = events[index].outcome;
        if (event.outcome == resync_token_outcome::accepted) {
            event.replay_sequence = candidate.next_replay_sequence++;
            expected_position++;
        }
        append_to_ring(candidate, event);
        candidate.last_commit_event_sequence = event.event_sequence;
    }
    candidate.next_token_position = expected_position;

    if (implementation.config.persist) {
        const std::vector<uint8_t> bytes = serialize_state(candidate);
        const resync_journal_result write_result =
                write_private_file(implementation.root_fd, bytes, implementation.config.faults);
        if (write_result.status != resync_journal_status::ok &&
            write_result.status != resync_journal_status::commit_uncertain) {
            return write_result;
        }
        if (write_result.status == resync_journal_status::commit_uncertain && !write_result.committed) {
            return write_result;
        }
        implementation.state = std::move(candidate);
        implementation.startup_status = resync_journal_status::ok;
        return write_result;
    }
    implementation.state = std::move(candidate);
    return result(resync_journal_status::ok, 0, true);
}

resync_journal_result resync_journal::inspect(resync_journal_snapshot & snapshot) const {
    const impl & implementation = *data;
    std::lock_guard<std::mutex> lock(implementation.mutex);
    if (implementation.startup_status != resync_journal_status::ok) {
        snapshot = {};
        return result(implementation.startup_status, implementation.startup_error);
    }
    fill_snapshot(implementation.state, snapshot);
    return result(resync_journal_status::ok, 0, true);
}

const resync_journal_config & resync_journal::config() const noexcept {
    return data->config;
}

}  // namespace server_capture

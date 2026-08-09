#include "server-request-history.h"

#include "server-capture-sha256.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace server_request_history {
namespace {

namespace fs = std::filesystem;
using digest = server_capture::capture_sha256::digest;

constexpr mode_t PRIVATE_DIRECTORY_MODE = S_IRWXU;
constexpr mode_t PRIVATE_FILE_MODE      = S_IRUSR | S_IWUSR;
constexpr size_t FILE_HEADER_PREFIX_BYTES = 40;
constexpr size_t FILE_HEADER_BYTES        = FILE_HEADER_PREFIX_BYTES + sizeof(digest);
constexpr size_t FRAME_PREFIX_BYTES       = 40;
constexpr size_t FRAME_CHECKSUM_BYTES     = sizeof(digest);
constexpr uint64_t MAX_FRAME_PAYLOAD      = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_METADATA_STRING    = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_RECORD_FILES       = 10ULL * 1000ULL * 1000ULL;
constexpr uint64_t MAX_PENDING_BYTES      = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr char FILE_MAGIC[8]              = { 'L', 'L', 'R', 'H', 'I', 'S', 'T', '1' };
constexpr char FRAME_MAGIC[4]             = { 'L', 'R', 'F', '1' };
constexpr const char * OWNER_LOCK_NAME    = ".request-history.owner.lock";

enum class frame_type : uint16_t {
    initial_metadata = 1,
    request_body     = 2,
    response_body    = 3,
    response_chunk   = 4,
    usage            = 5,
    terminal         = 6,
};

constexpr uint32_t FRAME_FIRST_SEGMENT = 1U << 0;
constexpr uint32_t FRAME_LAST_SEGMENT  = 1U << 1;

uint64_t system_wall_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

uint64_t steady_ns() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

result make_result(status value, int os_error = 0, bool durable = false) noexcept {
    return { value, os_error, durable };
}

status errno_status(int value) noexcept {
    if (value == ENOSPC || value == EDQUOT) {
        return status::no_space;
    }
    return status::io_error;
}

result errno_result(int value) noexcept {
    return make_result(errno_status(value), value);
}

bool checked_add(uint64_t left, uint64_t right, uint64_t & output) noexcept {
    if (left > UINT64_MAX - right) {
        return false;
    }
    output = left + right;
    return true;
}

void append_u16(std::vector<uint8_t> & bytes, uint16_t value) {
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8U));
}

void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

void append_i32(std::vector<uint8_t> & bytes, int32_t value) {
    append_u32(bytes, static_cast<uint32_t>(value));
}

void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

bool read_u16(const uint8_t * bytes, size_t size, size_t & offset, uint16_t & value) noexcept {
    if (offset > size || size - offset < 2) {
        return false;
    }
    value = static_cast<uint16_t>(bytes[offset]) |
            static_cast<uint16_t>(static_cast<uint16_t>(bytes[offset + 1]) << 8U);
    offset += 2;
    return true;
}

bool read_u32(const uint8_t * bytes, size_t size, size_t & offset, uint32_t & value) noexcept {
    if (offset > size || size - offset < 4) {
        return false;
    }
    value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[offset++]) << shift;
    }
    return true;
}

bool read_i32(const uint8_t * bytes, size_t size, size_t & offset, int32_t & value) noexcept {
    uint32_t raw = 0;
    if (!read_u32(bytes, size, offset, raw)) {
        return false;
    }
    value = static_cast<int32_t>(raw);
    return true;
}

bool read_u64(const uint8_t * bytes, size_t size, size_t & offset, uint64_t & value) noexcept {
    if (offset > size || size - offset < 8) {
        return false;
    }
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[offset++]) << shift;
    }
    return true;
}

bool append_string(std::vector<uint8_t> & bytes, const std::string & value) {
    if (value.size() > UINT32_MAX) {
        return false;
    }
    append_u32(bytes, static_cast<uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
    return true;
}

bool read_string(const uint8_t * bytes, size_t size, size_t & offset, std::string & value) {
    uint32_t length = 0;
    if (!read_u32(bytes, size, offset, length) || length > MAX_METADATA_STRING || offset > size || size - offset < length) {
        return false;
    }
    value.assign(reinterpret_cast<const char *>(bytes + offset), length);
    offset += length;
    return true;
}

bool path_has_traversal(const fs::path & path) {
    if (!path.is_absolute() || path.empty() || path == path.root_path()) {
        return true;
    }
    for (const fs::path & component : path) {
        if (component == "." || component == "..") {
            return true;
        }
    }
    return false;
}

result fsync_directory_fd(int root_fd, const config * cfg = nullptr) noexcept {
    if (cfg != nullptr && cfg->test_faults.fail_directory_fsync) {
        return make_result(status::io_error, EIO);
    }
    for (;;) {
        if (::fsync(root_fd) == 0) {
            return make_result(status::ok);
        }
        if (errno != EINTR) {
            return errno_result(errno);
        }
    }
}

result open_private_root(const fs::path & root, bool require_private, bool create, int & descriptor) {
    descriptor = -1;
    if (path_has_traversal(root)) {
        return make_result(status::path_security, EINVAL);
    }
    int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (current < 0) {
        return errno_result(errno);
    }
    bool root_created = false;
    for (const fs::path & component : root) {
        if (component == root.root_path()) {
            continue;
        }
        const std::string name = component.string();
        int next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        bool created = false;
        if (next < 0 && errno == ENOENT && create) {
            if (::mkdirat(current, name.c_str(), PRIVATE_DIRECTORY_MODE) == 0) {
                created = true;
            } else {
                const int error = errno;
                if (error != EEXIST) {
                    ::close(current);
                    return errno_result(error);
                }
            }
            next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (next < 0) {
            const int error = errno;
            ::close(current);
            return make_result(error == ENOENT ? status::not_found : status::path_security, error);
        }
        struct stat info = {};
        if (::fstat(next, &info) != 0 || !S_ISDIR(info.st_mode)) {
            const int error = errno == 0 ? EACCES : errno;
            ::close(next);
            ::close(current);
            return make_result(status::path_security, error);
        }
        if (created) {
            const result synced = fsync_directory_fd(current);
            if (synced.value != status::ok) {
                ::close(next);
                ::close(current);
                return synced;
            }
        }
        // Keep the descriptor walk authoritative if a path component is
        // exchanged between mkdir/open/stat by another process.
        struct stat named_info = {};
        struct stat opened_info = {};
        if (::fstatat(current, name.c_str(), &named_info, AT_SYMLINK_NOFOLLOW) != 0) {
            const int error = errno;
            ::close(next);
            ::close(current);
            return make_result(status::path_security, error);
        }
        if (::fstat(next, &opened_info) != 0) {
            const int error = errno;
            ::close(next);
            ::close(current);
            return make_result(status::path_security, error);
        }
        if (!S_ISDIR(named_info.st_mode) || named_info.st_dev != opened_info.st_dev ||
            named_info.st_ino != opened_info.st_ino) {
            ::close(next);
            ::close(current);
            return make_result(status::path_security, EAGAIN);
        }
        root_created = created;
        ::close(current);
        current = next;
    }
    struct stat info = {};
    if (::fstat(current, &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != ::geteuid()) {
        const int error = errno == 0 ? EACCES : errno;
        ::close(current);
        return make_result(status::path_security, error);
    }
    if (require_private) {
        // A pre-existing broad root may already have exposed prior content.
        // Never silently repair it. chmod is safe only for the directory this
        // call created, before any history file is written.
        if ((root_created && ::fchmod(current, PRIVATE_DIRECTORY_MODE) != 0) ||
            ::fstat(current, &info) != 0 || info.st_uid != ::geteuid() ||
            (info.st_mode & 07777) != PRIVATE_DIRECTORY_MODE) {
            const int error = errno == 0 ? EACCES : errno;
            ::close(current);
            return make_result(status::path_security, error);
        }
    }
    descriptor = current;
    return make_result(status::ok);
}

bool same_inode(const struct stat & left, const struct stat & right) noexcept {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}

result private_regular_info_fd(int descriptor, struct stat & info) noexcept {
    if (::fstat(descriptor, &info) != 0) {
        return errno_result(errno);
    }
    if (!S_ISREG(info.st_mode) || info.st_uid != ::geteuid() || (info.st_mode & 07777) != PRIVATE_FILE_MODE) {
        return make_result(status::path_security, EACCES);
    }
    return make_result(status::ok);
}

result private_regular_info_at(int root_fd, const std::string & name, struct stat & info, bool allow_missing = false) noexcept {
    if (::fstatat(root_fd, name.c_str(), &info, AT_SYMLINK_NOFOLLOW) != 0) {
        if (allow_missing && errno == ENOENT) return make_result(status::not_found, ENOENT);
        return make_result(status::path_security, errno);
    }
    if (S_ISLNK(info.st_mode) || !S_ISREG(info.st_mode) || info.st_uid != ::geteuid() ||
        (info.st_mode & 07777) != PRIVATE_FILE_MODE) {
        return make_result(status::path_security, EACCES);
    }
    return make_result(status::ok);
}

result verify_name_matches_inode(int root_fd, const std::string & name, const struct stat & expected) noexcept {
    struct stat named = {};
    const result valid = private_regular_info_at(root_fd, name, named);
    if (valid.value != status::ok) return valid;
    return same_inode(named, expected) ? make_result(status::ok) : make_result(status::path_security, EAGAIN);
}

result validate_private_regular_at(int root_fd, const std::string & name, bool allow_missing = false) noexcept {
    struct stat info = {};
    const result checked = private_regular_info_at(root_fd, name, info, allow_missing);
    return allow_missing && checked.value == status::not_found ? make_result(status::ok) : checked;
}

result rename_no_replace(int root_fd, const std::string & source, const std::string & destination) noexcept {
#if defined(__linux__) && defined(SYS_renameat2)
    if (::syscall(SYS_renameat2, root_fd, source.c_str(), root_fd, destination.c_str(), 1U) == 0) {
        return make_result(status::ok);
    }
    return make_result(errno == EEXIST || errno == ELOOP ? status::path_security : errno_status(errno), errno);
#elif defined(__APPLE__)
    if (::renameatx_np(root_fd, source.c_str(), root_fd, destination.c_str(), RENAME_EXCL) == 0) {
        return make_result(status::ok);
    }
    return make_result(errno == EEXIST || errno == ELOOP ? status::path_security : errno_status(errno), errno);
#else
    // The fork's production target is macOS and host validation runs Linux.
    // Keep a conservative fallback for other build hosts.
    if (::linkat(root_fd, source.c_str(), root_fd, destination.c_str(), 0) != 0) {
        return make_result(errno == EEXIST || errno == ELOOP ? status::path_security : errno_status(errno), errno);
    }
    if (::unlinkat(root_fd, source.c_str(), 0) != 0) return errno_result(errno);
    return make_result(status::ok);
#endif
}

result unlink_if_same(int root_fd, const std::string & name, const struct stat & expected) noexcept;

enum class file_kind : uint8_t { none, open, claim, complete, incomplete, quarantine };

struct file_identity {
    file_kind kind = file_kind::none;
    uint64_t  wall_ns = 0;
    uint64_t  id = 0;
};

bool parse_file_name(const std::string & name, file_identity & identity) noexcept;
std::string format_file_name(file_kind kind, uint64_t wall_ns, uint64_t id);

result claim_name_if_same(
        int root_fd,
        const std::string & source,
        const struct stat & expected,
        std::string & claim) noexcept {
    file_identity identity;
    if (!parse_file_name(source, identity)) return make_result(status::path_security, EINVAL);
    try {
        claim = format_file_name(file_kind::claim, identity.wall_ns, identity.id);
    } catch (...) {
        return make_result(status::io_error, ENOMEM);
    }
    if (identity.kind == file_kind::claim) {
        return verify_name_matches_inode(root_fd, source, expected);
    }
    result moved = rename_no_replace(root_fd, source, claim);
    if (moved.value != status::ok) return moved;
    result verified = verify_name_matches_inode(root_fd, claim, expected);
    if (verified.value == status::ok) return verified;

    // Never delete a namespace entry that did not resolve to the expected
    // inode. Restore it only without replacement; otherwise leave the unique
    // claim visible for fail-closed operator reconciliation.
    (void) rename_no_replace(root_fd, claim, source);
    return verified;
}

result rename_if_same(
        int root_fd,
        const std::string & source,
        const std::string & destination,
        const struct stat & expected) noexcept {
    struct stat target = {};
    result target_state = private_regular_info_at(root_fd, destination, target, true);
    if (target_state.value != status::not_found) {
        return target_state.value == status::ok ? make_result(status::path_security, EEXIST) : target_state;
    }
    std::string claim;
    result claimed = claim_name_if_same(root_fd, source, expected, claim);
    if (claimed.value != status::ok) return claimed;
    result renamed = rename_no_replace(root_fd, claim, destination);
    if (renamed.value != status::ok) {
        (void) rename_no_replace(root_fd, claim, source);
        return renamed;
    }
    result final_matches = verify_name_matches_inode(root_fd, destination, expected);
    if (final_matches.value != status::ok) {
        // The final name was exchanged after our no-replace rename. Never
        // delete the replacement; fail closed and force reconciliation.
        return final_matches;
    }
    return make_result(status::ok);
}

result unlink_if_same(
        int root_fd,
        const std::string & name,
        const struct stat & expected) noexcept {
    std::string claim;
    result claimed = claim_name_if_same(root_fd, name, expected, claim);
    if (claimed.value != status::ok) return claimed;
    if (::unlinkat(root_fd, claim.c_str(), 0) != 0) return errno_result(errno);
    return make_result(status::ok);
}

result acquire_owner_lock(int root_fd, int & lock_fd) {
    // Use the already-open, verified root directory inode as the cooperative
    // lock authority. A live authority can no longer be replaced by swapping
    // a named child lock file and creating a second independent lock domain.
    lock_fd = -1;
    if (::flock(root_fd, LOCK_EX | LOCK_NB) != 0) {
        const int error = errno;
        return make_result(error == EWOULDBLOCK || error == EAGAIN ? status::owner_busy : status::io_error, error);
    }
    return make_result(status::ok);
}

struct write_state {
    uint64_t bytes_written = 0;
    bool     eintr_used    = false;
};

result write_all(int descriptor, const uint8_t * bytes, size_t size, const config & cfg, write_state & state) {
    while (size != 0) {
        if (cfg.test_faults.inject_eintr_once && !state.eintr_used) {
            state.eintr_used = true;
            continue;
        }
        if (state.bytes_written >= cfg.test_faults.fail_after_bytes) {
            return make_result(cfg.test_faults.fail_no_space ? status::no_space : status::short_write,
                               cfg.test_faults.fail_no_space ? ENOSPC : EIO);
        }
        uint64_t amount = std::min<uint64_t>(size, cfg.test_faults.max_write_size);
        if (cfg.test_faults.fail_after_bytes != UINT64_MAX) {
            amount = std::min<uint64_t>(amount, cfg.test_faults.fail_after_bytes - state.bytes_written);
        }
        if (amount == 0 || amount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return make_result(cfg.test_faults.fail_no_space ? status::no_space : status::short_write,
                               cfg.test_faults.fail_no_space ? ENOSPC : EIO);
        }
        const ssize_t written = ::write(descriptor, bytes, static_cast<size_t>(amount));
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno_result(errno);
        }
        if (written == 0) {
            return make_result(status::short_write, EIO);
        }
        const size_t advanced = static_cast<size_t>(written);
        bytes += advanced;
        size -= advanced;
        state.bytes_written += advanced;
    }
    return make_result(status::ok);
}

result pread_exact(int descriptor, uint8_t * bytes, size_t size, uint64_t offset, bool allow_eof, bool & eof) {
    eof = false;
    size_t done = 0;
    while (done < size) {
        if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - done) {
            return make_result(status::too_large, EOVERFLOW);
        }
        const ssize_t amount = ::pread(descriptor, bytes + done, size - done, static_cast<off_t>(offset + done));
        if (amount < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno_result(errno);
        }
        if (amount == 0) {
            if (allow_eof && done == 0) {
                eof = true;
                return make_result(status::ok);
            }
            return make_result(status::truncated);
        }
        done += static_cast<size_t>(amount);
    }
    return make_result(status::ok);
}

std::vector<uint8_t> make_file_header(uint64_t record_id, uint64_t wall_ns, uint64_t monotonic_ns) {
    std::vector<uint8_t> bytes;
    bytes.reserve(FILE_HEADER_BYTES);
    bytes.insert(bytes.end(), FILE_MAGIC, FILE_MAGIC + sizeof(FILE_MAGIC));
    append_u32(bytes, REQUEST_HISTORY_FORMAT_VERSION);
    append_u32(bytes, static_cast<uint32_t>(FILE_HEADER_BYTES));
    append_u64(bytes, record_id);
    append_u64(bytes, wall_ns);
    append_u64(bytes, monotonic_ns);
    const digest checksum = server_capture::capture_sha256::hash(bytes.data(), bytes.size());
    bytes.insert(bytes.end(), checksum.begin(), checksum.end());
    return bytes;
}

result parse_file_header(const uint8_t * bytes, size_t size, record_metadata & metadata) {
    if (size != FILE_HEADER_BYTES || std::memcmp(bytes, FILE_MAGIC, sizeof(FILE_MAGIC)) != 0) {
        return make_result(status::malformed);
    }
    size_t offset = sizeof(FILE_MAGIC);
    uint32_t version = 0;
    uint32_t header_size = 0;
    if (!read_u32(bytes, size, offset, version) || !read_u32(bytes, size, offset, header_size) ||
        version != REQUEST_HISTORY_FORMAT_VERSION || header_size != FILE_HEADER_BYTES ||
        !read_u64(bytes, size, offset, metadata.record_id) ||
        !read_u64(bytes, size, offset, metadata.started_wall_ns) ||
        !read_u64(bytes, size, offset, metadata.started_monotonic_ns) || offset != FILE_HEADER_PREFIX_BYTES) {
        return make_result(status::malformed);
    }
    digest stored = {};
    std::copy(bytes + offset, bytes + size, stored.begin());
    const digest expected = server_capture::capture_sha256::hash(bytes, FILE_HEADER_PREFIX_BYTES);
    return server_capture::capture_sha256::equal(stored, expected) ? make_result(status::ok) :
                                                                    make_result(status::checksum_mismatch);
}

std::vector<uint8_t> make_initial_payload(
        const std::string & method,
        const std::string & path,
        const std::string & model,
        const std::string & profile,
        uint64_t request_bytes) {
    std::vector<uint8_t> payload;
    append_u32(payload, 1);
    if (!append_string(payload, method) || !append_string(payload, path) ||
        !append_string(payload, model) || !append_string(payload, profile)) {
        return {};
    }
    append_u64(payload, request_bytes);
    return payload;
}

result parse_initial_payload(const std::vector<uint8_t> & payload, record_metadata & metadata) {
    size_t offset = 0;
    uint32_t version = 0;
    uint64_t declared_request_bytes = 0;
    if (!read_u32(payload.data(), payload.size(), offset, version) || version != 1 ||
        !read_string(payload.data(), payload.size(), offset, metadata.method) ||
        !read_string(payload.data(), payload.size(), offset, metadata.path) ||
        !read_string(payload.data(), payload.size(), offset, metadata.requested_model) ||
        !read_string(payload.data(), payload.size(), offset, metadata.requested_profile) ||
        !read_u64(payload.data(), payload.size(), offset, declared_request_bytes) || offset != payload.size()) {
        return make_result(status::malformed);
    }
    metadata.request_bytes = declared_request_bytes;
    return make_result(status::ok);
}

std::vector<uint8_t> make_terminal_payload(
        outcome terminal_outcome,
        int http_status,
        bool streaming,
        bool recovered,
        bool transport_complete,
        bool transport_complete_known,
        uint64_t ended_wall_ns,
        uint64_t duration_ns,
        uint64_t request_bytes,
        uint64_t response_bytes,
        uint64_t chunks,
        const std::string & usage_json) {
    std::vector<uint8_t> payload;
    append_u32(payload, 1);
    append_u16(payload, static_cast<uint16_t>(terminal_outcome));
    uint16_t flags = 0;
    if (streaming) {
        flags |= 1U;
    }
    if (recovered) {
        flags |= 2U;
    }
    if (transport_complete_known) flags |= 4U;
    if (transport_complete) flags |= 8U;
    append_u16(payload, flags);
    append_i32(payload, http_status);
    append_u64(payload, ended_wall_ns);
    append_u64(payload, duration_ns);
    append_u64(payload, request_bytes);
    append_u64(payload, response_bytes);
    append_u64(payload, chunks);
    if (!append_string(payload, usage_json)) {
        return {};
    }
    return payload;
}

result parse_terminal_payload(const std::vector<uint8_t> & payload, record_metadata & metadata) {
    size_t offset = 0;
    uint32_t version = 0;
    uint16_t raw_outcome = 0;
    uint16_t flags = 0;
    int32_t http_status = 0;
    if (!read_u32(payload.data(), payload.size(), offset, version) || version != 1 ||
        !read_u16(payload.data(), payload.size(), offset, raw_outcome) || raw_outcome > static_cast<uint16_t>(outcome::recovered_incomplete) ||
        !read_u16(payload.data(), payload.size(), offset, flags) || (flags & ~15U) != 0 ||
        ((flags & 8U) != 0 && (flags & 4U) == 0) ||
        !read_i32(payload.data(), payload.size(), offset, http_status) ||
        !read_u64(payload.data(), payload.size(), offset, metadata.ended_wall_ns) ||
        !read_u64(payload.data(), payload.size(), offset, metadata.duration_ns) ||
        !read_u64(payload.data(), payload.size(), offset, metadata.request_bytes) ||
        !read_u64(payload.data(), payload.size(), offset, metadata.response_bytes) ||
        !read_u64(payload.data(), payload.size(), offset, metadata.response_chunks) ||
        !read_string(payload.data(), payload.size(), offset, metadata.usage_json) || offset != payload.size()) {
        return make_result(status::malformed);
    }
    metadata.http_status      = http_status;
    metadata.terminal_outcome = static_cast<outcome>(raw_outcome);
    metadata.streaming        = (flags & 1U) != 0;
    metadata.recovered        = (flags & 2U) != 0;
    metadata.transport_complete_known = (flags & 4U) != 0;
    metadata.transport_complete = (flags & 8U) != 0;
    metadata.terminal_present = true;
    return make_result(status::ok);
}

std::vector<uint8_t> make_frame_prefix(frame_type type, uint32_t flags, uint64_t sequence, uint64_t wall_ns, uint64_t size) {
    std::vector<uint8_t> bytes;
    bytes.reserve(FRAME_PREFIX_BYTES);
    bytes.insert(bytes.end(), FRAME_MAGIC, FRAME_MAGIC + sizeof(FRAME_MAGIC));
    append_u16(bytes, REQUEST_HISTORY_FRAME_VERSION);
    append_u16(bytes, static_cast<uint16_t>(type));
    append_u32(bytes, flags);
    append_u32(bytes, static_cast<uint32_t>(FRAME_PREFIX_BYTES));
    append_u64(bytes, sequence);
    append_u64(bytes, wall_ns);
    append_u64(bytes, size);
    return bytes;
}

struct parsed_frame {
    frame_type           type = frame_type::initial_metadata;
    uint32_t             flags = 0;
    uint64_t             sequence = 0;
    uint64_t             wall_ns = 0;
    std::vector<uint8_t> payload;
};

result read_frame(int descriptor, uint64_t file_size, uint64_t & offset, uint64_t max_payload, parsed_frame & frame, bool & eof) {
    eof = false;
    if (offset == file_size) {
        eof = true;
        return make_result(status::ok);
    }
    if (offset > file_size || file_size - offset < FRAME_PREFIX_BYTES) {
        return make_result(status::truncated);
    }
    std::vector<uint8_t> prefix(FRAME_PREFIX_BYTES);
    bool physical_eof = false;
    result current = pread_exact(descriptor, prefix.data(), prefix.size(), offset, false, physical_eof);
    if (current.value != status::ok) {
        return current;
    }
    if (std::memcmp(prefix.data(), FRAME_MAGIC, sizeof(FRAME_MAGIC)) != 0) {
        return make_result(status::malformed);
    }
    size_t cursor = sizeof(FRAME_MAGIC);
    uint16_t version = 0;
    uint16_t raw_type = 0;
    uint32_t prefix_size = 0;
    uint64_t payload_size = 0;
    if (!read_u16(prefix.data(), prefix.size(), cursor, version) || version != REQUEST_HISTORY_FRAME_VERSION ||
        !read_u16(prefix.data(), prefix.size(), cursor, raw_type) ||
        raw_type < static_cast<uint16_t>(frame_type::initial_metadata) || raw_type > static_cast<uint16_t>(frame_type::terminal) ||
        !read_u32(prefix.data(), prefix.size(), cursor, frame.flags) ||
        !read_u32(prefix.data(), prefix.size(), cursor, prefix_size) || prefix_size != FRAME_PREFIX_BYTES ||
        !read_u64(prefix.data(), prefix.size(), cursor, frame.sequence) ||
        !read_u64(prefix.data(), prefix.size(), cursor, frame.wall_ns) ||
        !read_u64(prefix.data(), prefix.size(), cursor, payload_size) || cursor != FRAME_PREFIX_BYTES) {
        return make_result(status::malformed);
    }
    if (payload_size > max_payload || payload_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return make_result(status::too_large);
    }
    uint64_t frame_bytes = 0;
    if (!checked_add(FRAME_PREFIX_BYTES + FRAME_CHECKSUM_BYTES, payload_size, frame_bytes) ||
        offset > file_size || frame_bytes > file_size - offset) {
        return make_result(status::truncated);
    }
    frame.type = static_cast<frame_type>(raw_type);
    frame.payload.resize(static_cast<size_t>(payload_size));
    if (!frame.payload.empty()) {
        current = pread_exact(descriptor, frame.payload.data(), frame.payload.size(), offset + FRAME_PREFIX_BYTES, false, physical_eof);
        if (current.value != status::ok) {
            return current;
        }
    }
    digest stored = {};
    current = pread_exact(descriptor, stored.data(), stored.size(), offset + FRAME_PREFIX_BYTES + payload_size, false, physical_eof);
    if (current.value != status::ok) {
        return current;
    }
    std::vector<uint8_t> checksum_input;
    checksum_input.reserve(prefix.size() + frame.payload.size());
    checksum_input.insert(checksum_input.end(), prefix.begin(), prefix.end());
    checksum_input.insert(checksum_input.end(), frame.payload.begin(), frame.payload.end());
    const digest expected = server_capture::capture_sha256::hash(checksum_input.data(), checksum_input.size());
    if (!server_capture::capture_sha256::equal(stored, expected)) {
        return make_result(status::checksum_mismatch);
    }
    offset += frame_bytes;
    return make_result(status::ok);
}

bool parse_20_digits(const std::string & text, size_t offset, uint64_t & value) noexcept {
    if (offset > text.size() || text.size() - offset < 20) {
        return false;
    }
    uint64_t parsed = 0;
    for (size_t i = 0; i < 20; ++i) {
        const char c = text[offset + i];
        if (c < '0' || c > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(c - '0');
        if (parsed > (UINT64_MAX - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    value = parsed;
    return true;
}

bool parse_file_name(const std::string & name, file_identity & identity) noexcept {
    struct prefix_info { const char * prefix; file_kind kind; const char * suffix; };
    constexpr prefix_info prefixes[] = {
        { ".open-",      file_kind::open,       ".llrh.tmp" },
        { ".claim-",     file_kind::claim,      ".llrh.tmp" },
        { "request-",    file_kind::complete,   ".llrh" },
        { "incomplete-", file_kind::incomplete, ".llrh" },
        { "quarantine-", file_kind::quarantine, ".llrh" },
    };
    for (const prefix_info & current : prefixes) {
        const size_t prefix_size = std::strlen(current.prefix);
        const size_t suffix_size = std::strlen(current.suffix);
        if (name.size() != prefix_size + 20 + 1 + 20 + suffix_size || name.rfind(current.prefix, 0) != 0 ||
            name[prefix_size + 20] != '-' || name.compare(name.size() - suffix_size, suffix_size, current.suffix) != 0) {
            continue;
        }
        uint64_t wall_ns = 0;
        uint64_t id = 0;
        if (!parse_20_digits(name, prefix_size, wall_ns) || !parse_20_digits(name, prefix_size + 21, id) || id == 0) {
            return false;
        }
        identity = { current.kind, wall_ns, id };
        return true;
    }
    return false;
}

std::string format_file_name(file_kind kind, uint64_t wall_ns, uint64_t id) {
    const char * prefix = kind == file_kind::open ? ".open-" :
                          kind == file_kind::claim ? ".claim-" :
                          kind == file_kind::complete ? "request-" :
                          kind == file_kind::incomplete ? "incomplete-" : "quarantine-";
    const char * suffix = (kind == file_kind::open || kind == file_kind::claim) ? ".llrh.tmp" : ".llrh";
    char output[128] = {};
    std::snprintf(output, sizeof(output), "%s%020llu-%020llu%s", prefix,
                  static_cast<unsigned long long>(wall_ns), static_cast<unsigned long long>(id), suffix);
    return output;
}

result list_directory(int root_fd, std::vector<std::string> & names, uint64_t max_entries = MAX_RECORD_FILES) {
    // dup() would share the directory stream offset with root_fd, causing a
    // later retention/recovery scan to start at EOF. Open "." relative to the
    // trusted descriptor to get an independent file description.
    const int duplicate = ::openat(root_fd, ".", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (duplicate < 0) {
        return errno_result(errno);
    }
    DIR * directory = ::fdopendir(duplicate);
    if (directory == nullptr) {
        const int error = errno;
        ::close(duplicate);
        return errno_result(error);
    }
    errno = 0;
    for (dirent * entry = ::readdir(directory); entry != nullptr; entry = ::readdir(directory)) {
        if (names.size() >= max_entries) {
            ::closedir(directory);
            return make_result(status::too_large);
        }
        names.emplace_back(entry->d_name);
    }
    const int scan_error = errno;
    const int close_error = ::closedir(directory) == 0 ? 0 : errno;
    if (scan_error != 0 || close_error != 0) {
        return errno_result(scan_error != 0 ? scan_error : close_error);
    }
    return make_result(status::ok);
}

struct parse_options {
    uint64_t max_frame_payload = MAX_FRAME_PAYLOAD;
    uint64_t max_total_body    = UINT64_MAX;
    bool     collect_bodies    = false;
    bool     allow_incomplete  = false;
};

struct parse_output {
    record value;
    uint64_t last_good_offset = 0;
    uint64_t observed_request_bytes = 0;
    uint64_t observed_response_bytes = 0;
    uint64_t observed_chunks = 0;
    uint64_t declared_request_bytes = 0;
    bool     saw_initial = false;
    bool     chunk_open = false;
};

result append_bounded(std::string & destination, const std::vector<uint8_t> & payload, uint64_t max_total, uint64_t & total) {
    uint64_t next = 0;
    if (!checked_add(total, payload.size(), next) || next > max_total ||
        next > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return make_result(status::too_large);
    }
    destination.append(reinterpret_cast<const char *>(payload.data()), payload.size());
    total = next;
    return make_result(status::ok);
}

result parse_record_fd(int descriptor, uint64_t file_size, const parse_options & options, parse_output & output) {
    if (file_size < FILE_HEADER_BYTES) {
        return make_result(status::truncated);
    }
    std::vector<uint8_t> header(FILE_HEADER_BYTES);
    bool eof = false;
    result current = pread_exact(descriptor, header.data(), header.size(), 0, false, eof);
    if (current.value != status::ok) {
        return current;
    }
    current = parse_file_header(header.data(), header.size(), output.value.metadata);
    if (current.value != status::ok) {
        return current;
    }
    output.value.metadata.file_bytes = file_size;
    uint64_t offset = FILE_HEADER_BYTES;
    output.last_good_offset = offset;
    uint64_t expected_sequence = 1;
    uint64_t collected_total = 0;
    while (offset < file_size) {
        parsed_frame frame;
        const uint64_t before = offset;
        current = read_frame(descriptor, file_size, offset, options.max_frame_payload, frame, eof);
        if (current.value != status::ok) {
            output.last_good_offset = before;
            return current;
        }
        if (eof) {
            break;
        }
        if (frame.sequence != expected_sequence++) {
            return make_result(status::malformed);
        }
        if (output.value.metadata.terminal_present) {
            return make_result(status::trailing_data);
        }
        switch (frame.type) {
            case frame_type::initial_metadata:
                if (output.saw_initial || frame.sequence != 1 || frame.flags != 0) {
                    return make_result(status::malformed);
                }
                current = parse_initial_payload(frame.payload, output.value.metadata);
                if (current.value != status::ok) {
                    return current;
                }
                output.declared_request_bytes = output.value.metadata.request_bytes;
                output.saw_initial = true;
                break;
            case frame_type::request_body:
                if (!output.saw_initial || output.chunk_open || frame.flags != 0) {
                    return make_result(status::malformed);
                }
                if (!checked_add(output.observed_request_bytes, frame.payload.size(), output.observed_request_bytes)) {
                    return make_result(status::too_large);
                }
                if (options.collect_bodies) {
                    current = append_bounded(output.value.request_body, frame.payload, options.max_total_body, collected_total);
                    if (current.value != status::ok) return current;
                }
                break;
            case frame_type::response_body:
                if (!output.saw_initial || output.chunk_open || frame.flags != 0) {
                    return make_result(status::malformed);
                }
                if (!checked_add(output.observed_response_bytes, frame.payload.size(), output.observed_response_bytes)) {
                    return make_result(status::too_large);
                }
                if (options.collect_bodies) {
                    current = append_bounded(output.value.response_body, frame.payload, options.max_total_body, collected_total);
                    if (current.value != status::ok) return current;
                }
                break;
            case frame_type::response_chunk: {
                if (!output.saw_initial || (frame.flags & ~(FRAME_FIRST_SEGMENT | FRAME_LAST_SEGMENT)) != 0 ||
                    (!output.chunk_open && (frame.flags & FRAME_FIRST_SEGMENT) == 0) ||
                    (output.chunk_open && (frame.flags & FRAME_FIRST_SEGMENT) != 0)) {
                    return make_result(status::malformed);
                }
                if (!checked_add(output.observed_response_bytes, frame.payload.size(), output.observed_response_bytes)) {
                    return make_result(status::too_large);
                }
                if (!output.chunk_open) {
                    output.chunk_open = true;
                    if (options.collect_bodies) {
                        output.value.response_stream_chunks.emplace_back();
                    }
                }
                if (options.collect_bodies) {
                    current = append_bounded(output.value.response_stream_chunks.back(), frame.payload,
                                             options.max_total_body, collected_total);
                    if (current.value != status::ok) return current;
                }
                if ((frame.flags & FRAME_LAST_SEGMENT) != 0) {
                    output.chunk_open = false;
                    ++output.observed_chunks;
                }
                break;
            }
            case frame_type::usage:
                if (!output.saw_initial || output.chunk_open || frame.flags != 0 ||
                    frame.payload.size() > MAX_METADATA_STRING) {
                    return make_result(status::malformed);
                }
                output.value.metadata.usage_json.assign(
                    reinterpret_cast<const char *>(frame.payload.data()), frame.payload.size());
                break;
            case frame_type::terminal:
                if (!output.saw_initial || output.chunk_open || frame.flags != 0) {
                    return make_result(status::malformed);
                }
                current = parse_terminal_payload(frame.payload, output.value.metadata);
                if (current.value != status::ok) {
                    return current;
                }
                break;
        }
        output.last_good_offset = offset;
    }
    if (!output.saw_initial || output.chunk_open) {
        return make_result(status::malformed);
    }
    if (output.value.metadata.terminal_present) {
        const bool recovered_prefix = output.value.metadata.terminal_outcome == outcome::recovered_incomplete &&
                                      output.observed_request_bytes <= output.declared_request_bytes;
        if (output.declared_request_bytes != output.observed_request_bytes && !recovered_prefix) {
            return make_result(status::malformed);
        }
        if (output.value.metadata.request_bytes != output.observed_request_bytes ||
            output.value.metadata.response_bytes != output.observed_response_bytes ||
            output.value.metadata.response_chunks != output.observed_chunks) {
            return make_result(status::malformed);
        }
    } else {
        output.value.metadata.request_bytes   = output.observed_request_bytes;
        output.value.metadata.response_bytes  = output.observed_response_bytes;
        output.value.metadata.response_chunks = output.observed_chunks;
        if (!options.allow_incomplete) {
            return make_result(status::truncated);
        }
    }
    return make_result(status::ok);
}

result open_record_at(int root_fd, const std::string & name, uint64_t max_bytes, int & descriptor, uint64_t & size) {
    descriptor = ::openat(root_fd, name.c_str(), O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) {
        return make_result(errno == ENOENT ? status::not_found : errno == ELOOP ? status::path_security : errno_status(errno), errno);
    }
    struct stat info = {};
    const result valid = private_regular_info_fd(descriptor, info);
    if (valid.value != status::ok) {
        ::close(descriptor);
        descriptor = -1;
        return valid;
    }
    const result named = verify_name_matches_inode(root_fd, name, info);
    if (named.value != status::ok) {
        ::close(descriptor);
        descriptor = -1;
        return named;
    }
    if (info.st_size < 0) {
        ::close(descriptor);
        descriptor = -1;
        return make_result(status::io_error, EIO);
    }
    size = static_cast<uint64_t>(info.st_size);
    if (size > max_bytes) {
        ::close(descriptor);
        descriptor = -1;
        return make_result(status::too_large);
    }
    return make_result(status::ok);
}

result close_fd(int & descriptor) noexcept {
    if (descriptor < 0) {
        return make_result(status::ok);
    }
    int error = 0;
    if (::close(descriptor) != 0) {
        error = errno;
    }
    descriptor = -1;
    return error == 0 ? make_result(status::ok) : errno_result(error);
}

bool valid_config(const config & cfg) {
    if (cfg.root_path.empty()) {
        return true;
    }
    const fs::path root(cfg.root_path);
    return !path_has_traversal(root) && cfg.max_retained_bytes != 0 &&
           cfg.max_retained_files != 0 && cfg.max_retained_files <= MAX_RECORD_FILES &&
           cfg.max_frame_payload >= 256 && cfg.max_frame_payload <= MAX_FRAME_PAYLOAD &&
           cfg.max_pending_bytes >= cfg.max_frame_payload && cfg.max_pending_bytes <= MAX_PENDING_BYTES &&
           cfg.max_stream_pending_frames_per_session != 0 && cfg.max_stream_pending_frames_per_session <= 64 &&
           cfg.terminal_ack_timeout_ms != 0 && cfg.terminal_ack_timeout_ms <= 10ULL * 60ULL * 1000ULL &&
           cfg.test_faults.max_write_size != 0;
}

}  // namespace

const char * status_name(status value) noexcept {
    switch (value) {
        case status::ok: return "ok";
        case status::disabled: return "disabled";
        case status::invalid_config: return "invalid_config";
        case status::stopped: return "stopped";
        case status::timeout: return "timeout";
        case status::owner_busy: return "owner_busy";
        case status::path_security: return "path_security";
        case status::not_found: return "not_found";
        case status::malformed: return "malformed";
        case status::truncated: return "truncated";
        case status::trailing_data: return "trailing_data";
        case status::checksum_mismatch: return "checksum_mismatch";
        case status::too_large: return "too_large";
        case status::no_space: return "no_space";
        case status::short_write: return "short_write";
        case status::io_error: return "io_error";
        case status::commit_uncertain: return "commit_uncertain";
    }
    return "unknown";
}

const char * outcome_name(outcome value) noexcept {
    switch (value) {
        case outcome::complete: return "complete";
        case outcome::aborted: return "aborted";
        case outcome::error: return "error";
        case outcome::recovered_incomplete: return "recovered_incomplete";
    }
    return "unknown";
}

struct store::impl {
    struct acknowledgement {
        std::mutex              mutex;
        std::condition_variable condition;
        bool                    ready = false;
        result                  value = make_result(status::stopped);

        void complete(result next) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (ready) return;
                value = next;
                ready = true;
            }
            condition.notify_all();
        }

        result wait(uint64_t timeout_ms) {
            std::unique_lock<std::mutex> lock(mutex);
            if (!condition.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return ready; })) {
                return make_result(status::timeout);
            }
            return value;
        }
    };

    enum class event_kind : uint8_t { begin, request, response, usage, finish, barrier, stop };

    struct event {
        event_kind kind = event_kind::barrier;
        uint64_t id = 0;
        uint64_t wall_ns = 0;
        uint64_t monotonic_ns = 0;
        std::string method;
        std::string path;
        std::string model;
        std::string profile;
        std::vector<uint8_t> payload;
        uint64_t declared_request_bytes = 0;
        uint32_t frame_flags = 0;
        bool stream_chunk = false;
        int http_status = 0;
        outcome terminal_outcome = outcome::recovered_incomplete;
        bool streaming = false;
        bool transport_complete = false;
        bool transport_complete_known = false;
        uint64_t charge = 0;
        bool stream_reservation = false;
        std::shared_ptr<acknowledgement> ack;
    };

    struct active_record {
        int fd = -1;
        struct stat inode = {};
        uint64_t id = 0;
        uint64_t started_wall_ns = 0;
        uint64_t started_monotonic_ns = 0;
        uint64_t next_frame_sequence = 1;
        uint64_t request_bytes = 0;
        uint64_t response_bytes = 0;
        uint64_t response_chunks = 0;
        uint64_t file_bytes = 0;
        std::string usage_json;
        std::string temporary_name;
    };

    struct retained_file {
        std::string name;
        uint64_t wall_ns = 0;
        uint64_t id = 0;
        uint64_t bytes = 0;
        struct stat inode = {};
    };

    using retained_key = std::pair<uint64_t, uint64_t>;

    explicit impl(config input) : cfg(std::move(input)) {
        if (!valid_config(cfg)) {
            throw std::invalid_argument("invalid request-history configuration");
        }
        if (cfg.root_path.empty()) {
            startup = make_result(status::disabled);
            return;
        }
        result opened = open_private_root(fs::path(cfg.root_path), cfg.require_private_root, true, root_fd);
        if (opened.value != status::ok) {
            throw std::runtime_error(std::string("request-history root: ") + status_name(opened.value));
        }
        result locked = acquire_owner_lock(root_fd, owner_lock_fd);
        if (locked.value != status::ok) {
            close_fd(root_fd);
            throw std::runtime_error(std::string("request-history owner lock: ") + status_name(locked.value));
        }
        result recovered = recover_open_records();
        if (recovered.value != status::ok) {
            close_fd(owner_lock_fd);
            close_fd(root_fd);
            throw std::runtime_error(std::string("request-history recovery: ") + status_name(recovered.value));
        }
        result retained = enforce_retention(now_wall_ns());
        if (retained.value != status::ok && !retained.durable) {
            close_fd(owner_lock_fd);
            close_fd(root_fd);
            throw std::runtime_error(std::string("request-history retention: ") + status_name(retained.value));
        }
        startup = make_result(status::ok);
        accepting = true;
        worker = std::thread(&impl::run, this);
    }

    ~impl() {
        if (worker.joinable()) {
            (void) shutdown(true);
            // Public completion waits are bounded. Object destruction is the
            // final lifetime barrier and must never detach a writer that still
            // owns descriptors into this object.
            if (worker.joinable()) worker.join();
        }
        close_fd(owner_lock_fd);
        close_fd(root_fd);
    }

    uint64_t now_wall_ns() const noexcept {
        try {
            return cfg.wall_time_ns ? cfg.wall_time_ns() : system_wall_ns();
        } catch (...) {
            return system_wall_ns();
        }
    }

    uint64_t now_monotonic_ns() const noexcept {
        try {
            return cfg.monotonic_time_ns ? cfg.monotonic_time_ns() : steady_ns();
        } catch (...) {
            return steady_ns();
        }
    }

    bool enqueue(event next, bool ingress = false) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (!accepting || failed || (ingress && !ingress_open)) {
            return false;
        }
        queue.push_back(std::move(next));
        queue_ready.notify_one();
        return true;
    }

    bool enqueue_payload(event next, const uint8_t * bytes, size_t size) {
        next.charge = size;
        std::unique_lock<std::mutex> lock(queue_mutex);
        if (!accepting || failed) return false;
        const auto capacity_available = [&] {
            if (!accepting || failed) return true;
            if (next.stream_reservation) {
                return pending_stream_frames[next.id] < cfg.max_stream_pending_frames_per_session;
            }
            return pending_bulk_bytes <= cfg.max_pending_bytes - next.charge;
        };
        if (!capacity_available()) {
            ++counter_backpressure_waits;
            queue_space.wait(lock, [&] { return capacity_available() || cancel_capacity_waits; });
        }
        if (!accepting || failed || (!capacity_available() && cancel_capacity_waits)) return false;

        // Wait before copying. A blocked HTTP producer therefore holds only
        // the handler's original entity, not an extra unaccounted frame.
        next.payload.assign(bytes, bytes + size);
        if (next.stream_reservation) {
            ++pending_stream_frames[next.id];
        } else {
            pending_bulk_bytes += next.charge;
        }
        pending_total_bytes += next.charge;
        peak_pending_bytes = std::max(peak_pending_bytes, pending_total_bytes);
        try {
            queue.push_back(std::move(next));
        } catch (...) {
            if (next.stream_reservation) {
                auto found = pending_stream_frames.find(next.id);
                if (found != pending_stream_frames.end() && found->second != 0 && --found->second == 0) {
                    pending_stream_frames.erase(found);
                }
            } else {
                pending_bulk_bytes -= next.charge;
            }
            pending_total_bytes -= next.charge;
            throw;
        }
        queue_ready.notify_one();
        return true;
    }

    void release_pending(const event & current) {
        std::lock_guard<std::mutex> lock(queue_mutex);
        if (current.stream_reservation) {
            auto found = pending_stream_frames.find(current.id);
            if (found != pending_stream_frames.end() && found->second != 0 && --found->second == 0) {
                pending_stream_frames.erase(found);
            }
        } else {
            pending_bulk_bytes -= current.charge;
        }
        pending_total_bytes -= current.charge;
        queue_space.notify_all();
    }

    result flush(uint64_t timeout_ms) {
        if (startup.value == status::disabled) return startup;
        auto ack = std::make_shared<acknowledgement>();
        event next;
        next.kind = event_kind::barrier;
        next.ack = ack;
        if (!enqueue(std::move(next))) {
            return current_failure();
        }
        return ack->wait(timeout_ms);
    }

    void stop_accepting() {
        std::lock_guard<std::mutex> lock(queue_mutex);
        ingress_open = false;
        cancel_capacity_waits = true;
        queue_space.notify_all();
    }

    result shutdown(bool drain) {
        if (startup.value == status::disabled) return startup;
        // This mutex covers the acknowledgement wait and join as well as stop
        // publication. Concurrent shutdown callers therefore share one result
        // and can never race two std::thread::join() calls.
        std::unique_lock<std::mutex> shutdown_lock(shutdown_mutex);
        if (shutdown_complete) return shutdown_result;
        if (!shutdown_started) {
            shutdown_started = true;
            std::lock_guard<std::mutex> queue_lock(queue_mutex);
            accepting = false;
            if (!drain) {
                for (event & queued : queue) {
                    if (queued.ack) queued.ack->complete(make_result(status::stopped));
                    if (queued.stream_reservation) {
                        auto found = pending_stream_frames.find(queued.id);
                        if (found != pending_stream_frames.end() && found->second != 0 && --found->second == 0) {
                            pending_stream_frames.erase(found);
                        }
                    } else {
                        pending_bulk_bytes -= queued.charge;
                    }
                    pending_total_bytes -= queued.charge;
                }
                queue.clear();
            }
            auto ack = std::make_shared<acknowledgement>();
            stop_ack = ack;
            if (failed || worker_exited) {
                result failure;
                {
                    std::lock_guard<std::mutex> stats_lock(stats_mutex);
                    failure = last_failure;
                }
                ack->complete(failure.value == status::ok ? make_result(status::stopped) : failure);
            } else {
                event stop_event;
                stop_event.kind = event_kind::stop;
                stop_event.ack = std::move(ack);
                queue.push_back(std::move(stop_event));
                queue_ready.notify_one();
            }
            queue_space.notify_all();
        }
        result stopped = stop_ack->wait(cfg.terminal_ack_timeout_ms);
        if (worker.joinable() && stopped.value != status::timeout) {
            worker.join();
        }
        if (stopped.value == status::timeout) {
            ++counter_terminal_timeouts;
            return stopped;
        }
        shutdown_result = stopped;
        shutdown_complete = true;
        return shutdown_result;
    }

    result current_failure() const {
        bool is_failed = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            is_failed = failed;
        }
        std::lock_guard<std::mutex> lock(stats_mutex);
        return is_failed ? last_failure : make_result(status::stopped);
    }

    void record_failure(result failure) {
        std::lock_guard<std::mutex> lock(stats_mutex);
        last_failure = failure;
    }

    result write_frame(active_record & active, frame_type type, uint32_t flags, const std::vector<uint8_t> & payload) {
        if (payload.size() > cfg.max_frame_payload) {
            return make_result(status::too_large);
        }
        try {
            if (cfg.test_faults.before_frame_write) cfg.test_faults.before_frame_write();
        } catch (...) {
            return make_result(status::io_error, EFAULT);
        }
        std::vector<uint8_t> bytes = make_frame_prefix(
            type, flags, active.next_frame_sequence, now_wall_ns(), payload.size());
        bytes.insert(bytes.end(), payload.begin(), payload.end());
        const digest checksum = server_capture::capture_sha256::hash(bytes.data(), bytes.size());
        result written = write_all(active.fd, bytes.data(), bytes.size(), cfg, writer_state);
        if (written.value == status::ok) {
            written = write_all(active.fd, checksum.data(), checksum.size(), cfg, writer_state);
        }
        if (written.value != status::ok) {
            return written;
        }
        ++active.next_frame_sequence;
        uint64_t frame_bytes = 0;
        if (!checked_add(bytes.size(), checksum.size(), frame_bytes) ||
            !checked_add(active.file_bytes, frame_bytes, active.file_bytes)) {
            return make_result(status::too_large, EOVERFLOW);
        }
        return make_result(status::ok);
    }

    result process_begin(const event & current) {
        if (active.count(current.id) != 0) return make_result(status::malformed);
        active_record record;
        record.id = current.id;
        record.started_wall_ns = current.wall_ns;
        record.started_monotonic_ns = current.monotonic_ns;
        record.temporary_name = format_file_name(file_kind::open, current.wall_ns, current.id);
        record.fd = ::openat(root_fd, record.temporary_name.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, PRIVATE_FILE_MODE);
        if (record.fd < 0) {
            return make_result(errno == EEXIST || errno == ELOOP ? status::path_security : errno_status(errno), errno);
        }
        if (::fchmod(record.fd, PRIVATE_FILE_MODE) != 0) {
            const int error = errno;
            close_fd(record.fd);
            return errno_result(error);
        }
        result valid = private_regular_info_fd(record.fd, record.inode);
        if (valid.value != status::ok) {
            close_fd(record.fd);
            return valid;
        }
        std::vector<uint8_t> header = make_file_header(current.id, current.wall_ns, current.monotonic_ns);
        result written = write_all(record.fd, header.data(), header.size(), cfg, writer_state);
        if (written.value != status::ok) {
            close_fd(record.fd);
            return written;
        }
        record.file_bytes = header.size();
        std::vector<uint8_t> initial = make_initial_payload(
            current.method, current.path, current.model, current.profile, current.declared_request_bytes);
        if (initial.empty() || initial.size() > cfg.max_frame_payload) {
            close_fd(record.fd);
            return make_result(status::too_large);
        }
        written = write_frame(record, frame_type::initial_metadata, 0, initial);
        if (written.value != status::ok) {
            close_fd(record.fd);
            return written;
        }
        active.emplace(current.id, std::move(record));
        return make_result(status::ok);
    }

    result process_body(const event & current) {
        auto found = active.find(current.id);
        if (found == active.end()) return make_result(status::malformed);
        active_record & record = found->second;
        const frame_type type = current.kind == event_kind::request ? frame_type::request_body :
                                current.stream_chunk ? frame_type::response_chunk : frame_type::response_body;
        result written = write_frame(record, type, current.frame_flags, current.payload);
        if (written.value != status::ok) return written;
        if (current.kind == event_kind::request) {
            if (!checked_add(record.request_bytes, current.payload.size(), record.request_bytes)) return make_result(status::too_large);
        } else {
            if (!checked_add(record.response_bytes, current.payload.size(), record.response_bytes)) return make_result(status::too_large);
            if (current.stream_chunk && (current.frame_flags & FRAME_LAST_SEGMENT) != 0) ++record.response_chunks;
        }
        return make_result(status::ok);
    }

    result process_usage(const event & current) {
        auto found = active.find(current.id);
        if (found == active.end()) return make_result(status::malformed);
        if (current.payload.size() > cfg.max_frame_payload) return make_result(status::too_large);
        result written = write_frame(found->second, frame_type::usage, 0, current.payload);
        if (written.value == status::ok) {
            found->second.usage_json.assign(reinterpret_cast<const char *>(current.payload.data()), current.payload.size());
        }
        return written;
    }

    result sync_file(int descriptor) {
        if (cfg.test_faults.fail_file_fsync) return make_result(status::io_error, EIO);
        for (;;) {
            if (::fsync(descriptor) == 0) return make_result(status::ok);
            if (errno != EINTR) return errno_result(errno);
        }
    }

    result publish(active_record & record, outcome terminal_outcome) {
        if (record.file_bytes > cfg.max_retained_bytes) {
            return make_result(status::too_large);
        }
        if (retained_index_dirty) {
            result reconciled = rebuild_retained_index();
            if (reconciled.value != status::ok) return reconciled;
        }
        result synced = sync_file(record.fd);
        if (synced.value != status::ok) return synced;
        struct stat current_inode = {};
        result descriptor_valid = private_regular_info_fd(record.fd, current_inode);
        if (descriptor_valid.value != status::ok || !same_inode(current_inode, record.inode) ||
            current_inode.st_size < 0 || static_cast<uint64_t>(current_inode.st_size) != record.file_bytes) {
            return descriptor_valid.value != status::ok ? descriptor_valid : make_result(status::path_security, EAGAIN);
        }
        if (cfg.test_faults.crash_before_rename) return make_result(status::commit_uncertain);
        const file_kind final_kind = terminal_outcome == outcome::recovered_incomplete ? file_kind::incomplete : file_kind::complete;
        const std::string final_name = format_file_name(final_kind, record.started_wall_ns, record.id);
        try {
            if (cfg.test_faults.before_publish_rename) cfg.test_faults.before_publish_rename(record.temporary_name);
        } catch (...) {
            return make_result(status::io_error, EFAULT);
        }
        if (cfg.test_faults.fail_rename) return make_result(status::io_error, EIO);
        result renamed = rename_if_same(root_fd, record.temporary_name, final_name, record.inode);
        if (renamed.value != status::ok) return renamed;
        if (cfg.test_faults.crash_after_rename) return make_result(status::commit_uncertain);
        result directory_synced = fsync_directory_fd(root_fd, &cfg);
        if (directory_synced.value != status::ok) return make_result(status::commit_uncertain, directory_synced.os_error);
        result indexed;
        try {
            indexed = add_retained(
                final_name, { final_kind, record.started_wall_ns, record.id }, record.file_bytes, record.inode);
        } catch (const std::bad_alloc &) {
            retained_index_dirty = true;
            return make_result(status::io_error, ENOMEM, true);
        }
        if (indexed.value != status::ok) {
            retained_index_dirty = true;
            return make_result(indexed.value, indexed.os_error, true);
        }
        result retained = enforce_retention(now_wall_ns());
        if (retained.value != status::ok) {
            retained_index_dirty = true;
            retained.durable = true;
            return retained;
        }
        return make_result(status::ok, 0, true);
    }

    result finalize(
            active_record & record,
            int http_status,
            outcome terminal_outcome,
            bool streaming,
            bool recovered,
            bool transport_complete = false,
            bool transport_complete_known = false) {
        const uint64_t ended_wall = now_wall_ns();
        const uint64_t ended_mono = now_monotonic_ns();
        const uint64_t duration = ended_mono >= record.started_monotonic_ns ? ended_mono - record.started_monotonic_ns : 0;
        std::vector<uint8_t> terminal = make_terminal_payload(
            terminal_outcome, http_status, streaming, recovered, transport_complete, transport_complete_known,
            ended_wall, duration,
            record.request_bytes, record.response_bytes, record.response_chunks, record.usage_json);
        if (terminal.empty() || terminal.size() > cfg.max_frame_payload) return make_result(status::too_large);
        result written = write_frame(record, frame_type::terminal, 0, terminal);
        if (written.value != status::ok) return written;
        return publish(record, terminal_outcome);
    }

    result process_finish(const event & current) {
        auto found = active.find(current.id);
        if (found == active.end()) return make_result(status::malformed);
        result completed = finalize(
            found->second, current.http_status, current.terminal_outcome, current.streaming, false,
            current.transport_complete, current.transport_complete_known);
        if (found->second.fd >= 0) close_fd(found->second.fd);
        active.erase(found);
        if (completed.durable) ++counter_durable_records;
        if (current.terminal_outcome == outcome::recovered_incomplete) ++counter_incomplete_records;
        return completed;
    }

    void fail_worker(result failure) {
        failed = true;
        accepting = false;
        record_failure(failure);
        ++counter_failed_records;
        for (auto & item : active) close_fd(item.second.fd);
        active.clear();
        for (event & queued : queue) {
            if (queued.ack) queued.ack->complete(failure);
        }
        queue.clear();
        pending_bulk_bytes = 0;
        pending_total_bytes = 0;
        pending_stream_frames.clear();
        queue_space.notify_all();
    }

    void run() {
        result terminal = make_result(status::ok);
        for (;;) {
            event current;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_ready.wait(lock, [this] { return !queue.empty(); });
                current = std::move(queue.front());
                queue.pop_front();
            }
            if (current.kind == event_kind::stop) {
                if (!cfg.test_faults.preserve_open_on_shutdown) {
                    for (auto & item : active) {
                        result recovered;
                        try {
                            recovered = finalize(item.second, 0, outcome::recovered_incomplete, false, true);
                        } catch (const std::bad_alloc &) {
                            recovered = make_result(status::io_error, ENOMEM);
                        } catch (...) {
                            recovered = make_result(status::io_error, EFAULT);
                        }
                        if (recovered.value != status::ok && !recovered.durable) terminal = recovered;
                        if (recovered.durable) {
                            ++counter_durable_records;
                            ++counter_incomplete_records;
                        }
                        close_fd(item.second.fd);
                    }
                } else {
                    for (auto & item : active) close_fd(item.second.fd);
                }
                active.clear();
                if (current.ack) current.ack->complete(terminal);
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    worker_exited = true;
                }
                queue_space.notify_all();
                break;
            }
            result processed = make_result(status::ok);
            try {
                switch (current.kind) {
                    case event_kind::begin: processed = process_begin(current); break;
                    case event_kind::request:
                    case event_kind::response: processed = process_body(current); break;
                    case event_kind::usage: processed = process_usage(current); break;
                    case event_kind::finish: processed = process_finish(current); break;
                    case event_kind::barrier: break;
                    case event_kind::stop: break;
                }
            } catch (const std::bad_alloc &) {
                processed = make_result(status::io_error, ENOMEM);
            } catch (...) {
                processed = make_result(status::io_error, EFAULT);
            }
            release_pending(current);
            if (current.ack) current.ack->complete(processed);
            if (processed.value != status::ok && !processed.durable) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                fail_worker(processed);
                worker_exited = true;
                break;
            }
            if (processed.value != status::ok) record_failure(processed);
        }
    }

    result quarantine_named_file(
            const std::string & name,
            const file_identity & identity,
            const struct stat & inode) {
        const std::string destination = format_file_name(file_kind::quarantine, identity.wall_ns, identity.id);
        result renamed = rename_if_same(root_fd, name, destination, inode);
        if (renamed.value != status::ok) return renamed;
        result synced = fsync_directory_fd(root_fd);
        if (synced.value == status::ok) ++counter_quarantined_records;
        return synced;
    }

    result add_retained_to(
            std::map<retained_key, retained_file> & index,
            uint64_t & total_bytes,
            retained_file file) {
        uint64_t next_bytes = 0;
        if (!checked_add(total_bytes, file.bytes, next_bytes)) return make_result(status::too_large);
        const auto inserted = index.emplace(
            retained_key{ file.wall_ns, file.id }, std::move(file));
        if (!inserted.second) return make_result(status::path_security, EEXIST);
        total_bytes = next_bytes;
        return make_result(status::ok);
    }

    result add_retained(
            const std::string & name,
            const file_identity & identity,
            uint64_t bytes,
            const struct stat & inode) {
        return add_retained_to(retained_index, retained_total_bytes,
                               { name, identity.wall_ns, identity.id, bytes, inode });
    }

    result reconcile_published(
            const std::string & name,
            const file_identity & identity,
            std::map<retained_key, retained_file> & rebuilt,
            uint64_t & rebuilt_bytes) {
        int descriptor = -1;
        uint64_t file_size = 0;
        result opened = open_record_at(root_fd, name, UINT64_MAX, descriptor, file_size);
        if (opened.value != status::ok) return opened;
        struct stat inode = {};
        result descriptor_valid = private_regular_info_fd(descriptor, inode);
        if (descriptor_valid.value != status::ok) {
            close_fd(descriptor);
            return descriptor_valid;
        }
        parse_output parsed;
        parse_options options;
        options.max_frame_payload = MAX_FRAME_PAYLOAD;
        result validation = parse_record_fd(descriptor, file_size, options, parsed);
        struct stat after_parse = {};
        const result stable = private_regular_info_fd(descriptor, after_parse);
        if (stable.value != status::ok || !same_inode(inode, after_parse) || after_parse.st_size < 0 ||
            static_cast<uint64_t>(after_parse.st_size) != file_size) {
            close_fd(descriptor);
            return stable.value != status::ok ? stable : make_result(status::path_security, EAGAIN);
        }
        const bool identity_matches = validation.value == status::ok &&
                                      parsed.value.metadata.record_id == identity.id &&
                                      parsed.value.metadata.started_wall_ns == identity.wall_ns;
        const bool kind_matches = identity.kind == file_kind::incomplete ?
            parsed.value.metadata.terminal_outcome == outcome::recovered_incomplete :
            parsed.value.metadata.terminal_outcome != outcome::recovered_incomplete;
        if (!identity_matches || !kind_matches) {
            // descriptor remains open here, pinning the verified inode across
            // the checked quarantine rename.
            result quarantined = quarantine_named_file(name, identity, inode);
            close_fd(descriptor);
            if (quarantined.value != status::ok) return quarantined;
            const std::string quarantine_name = format_file_name(file_kind::quarantine, identity.wall_ns, identity.id);
            return add_retained_to(rebuilt, rebuilt_bytes,
                                   { quarantine_name, identity.wall_ns, identity.id, file_size, inode });
        }
        result named_valid = verify_name_matches_inode(root_fd, name, inode);
        if (named_valid.value != status::ok) {
            close_fd(descriptor);
            return named_valid;
        }
        close_fd(descriptor);
        return add_retained_to(rebuilt, rebuilt_bytes,
                               { name, identity.wall_ns, identity.id, file_size, inode });
    }

    result rebuild_retained_index(
            std::vector<std::pair<std::string, file_identity>> * open_records = nullptr,
            uint64_t * maximum_id = nullptr) {
        std::vector<std::string> names;
        result listed = list_directory(root_fd, names);
        if (listed.value != status::ok) return listed;
        ++counter_namespace_scans;
        std::map<retained_key, retained_file> rebuilt;
        uint64_t rebuilt_bytes = 0;
        uint64_t max_id = 0;
        for (const std::string & name : names) {
            if (name == "." || name == ".." || name == OWNER_LOCK_NAME) continue;
            file_identity identity;
            const bool managed = parse_file_name(name, identity);
            const bool managed_prefix = name.rfind(".open-", 0) == 0 || name.rfind(".claim-", 0) == 0 ||
                                        name.rfind("request-", 0) == 0 ||
                                        name.rfind("incomplete-", 0) == 0 || name.rfind("quarantine-", 0) == 0;
            if (!managed) {
                if (managed_prefix) return make_result(status::path_security, EINVAL);
                continue;
            }
            result valid = validate_private_regular_at(root_fd, name);
            if (valid.value != status::ok) return valid;
            max_id = std::max(max_id, identity.id);
            if (identity.kind == file_kind::complete || identity.kind == file_kind::incomplete) {
                result reconciled = reconcile_published(name, identity, rebuilt, rebuilt_bytes);
                if (reconciled.value != status::ok) return reconciled;
            } else if (identity.kind == file_kind::quarantine) {
                int descriptor = -1;
                uint64_t file_size = 0;
                result opened = open_record_at(root_fd, name, UINT64_MAX, descriptor, file_size);
                if (opened.value != status::ok) return opened;
                struct stat inode = {};
                result inode_valid = private_regular_info_fd(descriptor, inode);
                if (inode_valid.value == status::ok) inode_valid = verify_name_matches_inode(root_fd, name, inode);
                close_fd(descriptor);
                if (inode_valid.value != status::ok) return inode_valid;
                result indexed = add_retained_to(rebuilt, rebuilt_bytes,
                                                 { name, identity.wall_ns, identity.id, file_size, inode });
                if (indexed.value != status::ok) return indexed;
            } else if ((identity.kind == file_kind::open || identity.kind == file_kind::claim) &&
                       open_records != nullptr) {
                open_records->emplace_back(name, identity);
            }
        }
        retained_index = std::move(rebuilt);
        retained_total_bytes = rebuilt_bytes;
        retained_index_dirty = false;
        counter_retained_files.store(retained_index.size());
        counter_retained_bytes.store(retained_total_bytes);
        if (maximum_id != nullptr) *maximum_id = max_id;
        return make_result(status::ok);
    }

    result recover_one_open(const std::string & name, const file_identity & identity) {
        int descriptor = ::openat(root_fd, name.c_str(), O_RDWR | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC);
        if (descriptor < 0) {
            return make_result(errno == ENOENT ? status::not_found :
                               errno == ELOOP ? status::path_security : errno_status(errno), errno);
        }
        struct stat inode = {};
        result opened = private_regular_info_fd(descriptor, inode);
        if (opened.value != status::ok || inode.st_size < 0) {
            close_fd(descriptor);
            return opened.value != status::ok ? opened : make_result(status::io_error, EIO);
        }
        result named_valid = verify_name_matches_inode(root_fd, name, inode);
        if (named_valid.value != status::ok) {
            close_fd(descriptor);
            return named_valid;
        }
        uint64_t file_size = static_cast<uint64_t>(inode.st_size);
        parse_output parsed;
        parse_options options;
        options.max_frame_payload = cfg.max_frame_payload;
        options.allow_incomplete = true;
        result validation = parse_record_fd(descriptor, file_size, options, parsed);
        const bool recoverable_tail = validation.value == status::truncated && parsed.saw_initial &&
                                      parsed.last_good_offset > FILE_HEADER_BYTES;
        if (validation.value != status::ok && !recoverable_tail) {
            result quarantined = quarantine_named_file(name, identity, inode);
            close_fd(descriptor);
            if (quarantined.value == status::ok) {
                quarantined = add_retained(
                    format_file_name(file_kind::quarantine, identity.wall_ns, identity.id),
                    { file_kind::quarantine, identity.wall_ns, identity.id }, file_size, inode);
            }
            return quarantined;
        }
        if (recoverable_tail) {
            if (::ftruncate(descriptor, static_cast<off_t>(parsed.last_good_offset)) != 0) {
                const int error = errno;
                close_fd(descriptor);
                return errno_result(error);
            }
            file_size = parsed.last_good_offset;
        }
        if (parsed.value.metadata.record_id != identity.id || parsed.value.metadata.started_wall_ns != identity.wall_ns) {
            result quarantined = quarantine_named_file(name, identity, inode);
            close_fd(descriptor);
            if (quarantined.value == status::ok) {
                quarantined = add_retained(
                    format_file_name(file_kind::quarantine, identity.wall_ns, identity.id),
                    { file_kind::quarantine, identity.wall_ns, identity.id }, file_size, inode);
            }
            return quarantined;
        }
        if (parsed.value.metadata.terminal_present) {
            const file_kind kind = parsed.value.metadata.terminal_outcome == outcome::recovered_incomplete ?
                                   file_kind::incomplete : file_kind::complete;
            result synced;
            for (;;) {
                if (::fsync(descriptor) == 0) { synced = make_result(status::ok); break; }
                if (errno != EINTR) { synced = errno_result(errno); break; }
            }
            if (synced.value != status::ok) { close_fd(descriptor); return synced; }
            const std::string destination = format_file_name(kind, identity.wall_ns, identity.id);
            result renamed = rename_if_same(root_fd, name, destination, inode);
            if (renamed.value == status::ok) renamed = fsync_directory_fd(root_fd);
            if (renamed.value == status::ok) {
                result retained = add_retained(destination, { kind, identity.wall_ns, identity.id }, file_size, inode);
                if (retained.value != status::ok) renamed = retained;
            }
            close_fd(descriptor);
            if (renamed.value == status::ok) {
                ++counter_recovered_records;
                ++counter_durable_records;
            }
            return renamed;
        }
        if (::lseek(descriptor, static_cast<off_t>(file_size), SEEK_SET) < 0) {
            const int error = errno;
            close_fd(descriptor);
            return errno_result(error);
        }
        active_record recovered;
        recovered.fd = descriptor;
        recovered.inode = inode;
        recovered.id = identity.id;
        recovered.started_wall_ns = parsed.value.metadata.started_wall_ns;
        recovered.started_monotonic_ns = parsed.value.metadata.started_monotonic_ns;
        recovered.next_frame_sequence = 1;
        // Determine the next sequence with a metadata-only second pass. This
        // remains bounded to one frame of memory.
        uint64_t offset = FILE_HEADER_BYTES;
        uint64_t sequence = 0;
        bool eof = false;
        while (offset < file_size) {
            parsed_frame frame;
            result frame_result = read_frame(descriptor, file_size, offset, cfg.max_frame_payload, frame, eof);
            if (frame_result.value != status::ok) { close_fd(recovered.fd); return frame_result; }
            sequence = frame.sequence;
        }
        recovered.next_frame_sequence = sequence + 1;
        recovered.request_bytes = parsed.observed_request_bytes;
        recovered.response_bytes = parsed.observed_response_bytes;
        recovered.response_chunks = parsed.observed_chunks;
        recovered.file_bytes = file_size;
        recovered.usage_json = parsed.value.metadata.usage_json;
        recovered.temporary_name = name;
        result finalized = finalize(recovered, 0, outcome::recovered_incomplete,
                                    parsed.observed_chunks != 0, true);
        close_fd(recovered.fd);
        if (finalized.durable) {
            ++counter_recovered_records;
            ++counter_durable_records;
            ++counter_incomplete_records;
        }
        return finalized;
    }

    result recover_open_records() {
        uint64_t max_id = 0;
        std::vector<std::pair<std::string, file_identity>> opens;
        result reconciled = rebuild_retained_index(&opens, &max_id);
        if (reconciled.value != status::ok) return reconciled;
        next_record_id.store(max_id == UINT64_MAX ? 1 : max_id + 1);
        for (const auto & item : opens) {
            result recovered = recover_one_open(item.first, item.second);
            if (recovered.value != status::ok && !recovered.durable) return recovered;
        }
        return make_result(status::ok);
    }

    result enforce_retention(uint64_t now_ns) {
        bool deleted = false;
        while (!retained_index.empty()) {
            auto oldest_it = retained_index.begin();
            const retained_file oldest = oldest_it->second;
            const bool age_expired = cfg.retention_age_ns != 0 && now_ns > oldest.wall_ns &&
                                     now_ns - oldest.wall_ns > cfg.retention_age_ns;
            const bool over_files = retained_index.size() > cfg.max_retained_files;
            const bool over_bytes = retained_total_bytes > cfg.max_retained_bytes;
            if (!age_expired && !over_files && !over_bytes) break;
            try {
                if (cfg.test_faults.before_retention_unlink) cfg.test_faults.before_retention_unlink(oldest.name);
            } catch (...) {
                retained_index_dirty = true;
                return make_result(status::io_error, EFAULT);
            }
            result removed = unlink_if_same(root_fd, oldest.name, oldest.inode);
            if (removed.value != status::ok) {
                retained_index_dirty = true;
                return removed;
            }
            retained_total_bytes -= oldest.bytes;
            retained_index.erase(oldest_it);
            deleted = true;
        }
        if (deleted) {
            result synced = fsync_directory_fd(root_fd, &cfg);
            if (synced.value != status::ok) {
                retained_index_dirty = true;
                return synced;
            }
        }
        counter_retained_files.store(retained_index.size());
        counter_retained_bytes.store(retained_total_bytes);
        return make_result(status::ok);
    }

    config cfg;
    result startup = make_result(status::invalid_config);
    int root_fd = -1;
    int owner_lock_fd = -1;
    write_state writer_state;
    std::atomic<uint64_t> next_record_id{ 1 };
    std::thread worker;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_ready;
    std::condition_variable queue_space;
    std::deque<event> queue;
    bool accepting = false;
    bool ingress_open = true;
    bool cancel_capacity_waits = false;
    bool failed = false;
    bool worker_exited = false;
    uint64_t pending_bulk_bytes = 0;
    uint64_t pending_total_bytes = 0;
    uint64_t peak_pending_bytes = 0;
    std::unordered_map<uint64_t, uint32_t> pending_stream_frames;
    std::map<uint64_t, active_record> active;
    std::map<retained_key, retained_file> retained_index;
    uint64_t retained_total_bytes = 0;
    bool retained_index_dirty = false;

    mutable std::mutex stats_mutex;
    result last_failure = make_result(status::ok);
    std::atomic<uint64_t> counter_accepted_records{ 0 };
    std::atomic<uint64_t> counter_durable_records{ 0 };
    std::atomic<uint64_t> counter_incomplete_records{ 0 };
    std::atomic<uint64_t> counter_failed_records{ 0 };
    std::atomic<uint64_t> counter_ingress_failures{ 0 };
    std::atomic<uint64_t> counter_append_failures{ 0 };
    std::atomic<uint64_t> counter_terminal_failures{ 0 };
    std::atomic<uint64_t> counter_recovered_records{ 0 };
    std::atomic<uint64_t> counter_quarantined_records{ 0 };
    std::atomic<uint64_t> counter_namespace_scans{ 0 };
    std::atomic<uint64_t> counter_retained_files{ 0 };
    std::atomic<uint64_t> counter_retained_bytes{ 0 };
    std::atomic<uint64_t> counter_backpressure_waits{ 0 };
    std::atomic<uint64_t> counter_terminal_timeouts{ 0 };

    std::mutex shutdown_mutex;
    bool shutdown_started = false;
    bool shutdown_complete = false;
    result shutdown_result = make_result(status::stopped);
    std::shared_ptr<acknowledgement> stop_ack;
};

struct session::state {
    state(std::shared_ptr<store::impl> input, uint64_t input_id) : owner(std::move(input)), id(input_id) {}
    std::shared_ptr<store::impl> owner;
    uint64_t id = 0;
    std::mutex mutex;
    bool finished = false;
    bool saw_stream = false;
    bool append_failed = false;
};

store::store(config cfg) : data(std::make_shared<impl>(std::move(cfg))) {}

store::~store() {
    if (data) (void) data->shutdown(true);
}

std::shared_ptr<session> store::begin(
        const std::string & method,
        const std::string & path,
        const std::string & decoded_request_entity,
        const std::string & requested_model,
        const std::string & requested_profile) {
    if (!data || data->startup.value != status::ok) return nullptr;
    uint64_t metadata_bytes = 4 + 4 * 4 + 8;
    for (const std::string * field : { &method, &path, &requested_model, &requested_profile }) {
        uint64_t next = 0;
        if (field->size() > UINT32_MAX || !checked_add(metadata_bytes, field->size(), next) ||
            next > data->cfg.max_frame_payload) {
            ++data->counter_ingress_failures;
            data->record_failure(make_result(status::too_large));
            return nullptr;
        }
        metadata_bytes = next;
    }
    const uint64_t id = data->next_record_id.fetch_add(1);
    if (id == 0) {
        ++data->counter_ingress_failures;
        data->record_failure(make_result(status::too_large, EOVERFLOW));
        return nullptr;
    }
    impl::event begin;
    begin.kind = impl::event_kind::begin;
    begin.id = id;
    begin.wall_ns = data->now_wall_ns();
    begin.monotonic_ns = data->now_monotonic_ns();
    begin.method = method;
    begin.path = path;
    begin.model = requested_model;
    begin.profile = requested_profile;
    begin.declared_request_bytes = decoded_request_entity.size();
    if (!data->enqueue(std::move(begin), true)) {
        ++data->counter_ingress_failures;
        return nullptr;
    }

    const size_t max_frame = static_cast<size_t>(data->cfg.max_frame_payload);
    size_t offset = 0;
    while (offset < decoded_request_entity.size()) {
        const size_t amount = std::min(max_frame, decoded_request_entity.size() - offset);
        impl::event body;
        body.kind = impl::event_kind::request;
        body.id = id;
        const auto * bytes = reinterpret_cast<const uint8_t *>(decoded_request_entity.data() + offset);
        if (!data->enqueue_payload(std::move(body), bytes, amount)) {
            ++data->counter_ingress_failures;
            return nullptr;
        }
        offset += amount;
    }
    ++data->counter_accepted_records;
    return std::shared_ptr<session>(new session(data, id));
}

result store::flush(uint64_t timeout_ms) { return data ? data->flush(timeout_ms) : make_result(status::stopped); }
void store::stop_accepting() { if (data) data->stop_accepting(); }
result store::shutdown(bool drain) { return data ? data->shutdown(drain) : make_result(status::stopped); }

stats store::get_stats() const {
    stats output;
    if (!data) return output;
    output.accepted_records   = data->counter_accepted_records.load();
    output.durable_records    = data->counter_durable_records.load();
    output.incomplete_records = data->counter_incomplete_records.load();
    output.failed_records     = data->counter_failed_records.load();
    output.ingress_failures   = data->counter_ingress_failures.load();
    output.append_failures    = data->counter_append_failures.load();
    output.terminal_failures  = data->counter_terminal_failures.load();
    output.recovered_records  = data->counter_recovered_records.load();
    output.quarantined_records = data->counter_quarantined_records.load();
    output.namespace_scans     = data->counter_namespace_scans.load();
    output.retained_files     = data->counter_retained_files.load();
    output.retained_bytes     = data->counter_retained_bytes.load();
    output.backpressure_waits = data->counter_backpressure_waits.load();
    output.terminal_timeouts  = data->counter_terminal_timeouts.load();
    {
        std::lock_guard<std::mutex> lock(data->queue_mutex);
        output.pending_bytes      = data->pending_total_bytes;
        output.peak_pending_bytes = data->peak_pending_bytes;
    }
    {
        std::lock_guard<std::mutex> lock(data->stats_mutex);
        output.last_failure = data->last_failure;
    }
    return output;
}

session::session(std::shared_ptr<store::impl> owner, uint64_t id) : data(new state(std::move(owner), id)) {}

session::~session() {
    if (!data) return;
    bool needs_finish = false;
    bool streaming = false;
    {
        std::lock_guard<std::mutex> lock(data->mutex);
        needs_finish = !data->finished;
        streaming = data->saw_stream;
    }
    if (needs_finish) {
        try {
            (void) finish(0, outcome::aborted, streaming);
        } catch (...) {
            // Destructors are a final best-effort incomplete marker and must
            // never terminate the process during exception unwinding.
        }
    }
}

bool session::append_response(const void * input, size_t size, bool stream_chunk) {
    if (size == 0) return true;
    if (input == nullptr || !data || !data->owner) return false;
    std::lock_guard<std::mutex> lock(data->mutex);
    if (data->finished) return false;
    data->saw_stream |= stream_chunk;
    const auto * bytes = static_cast<const uint8_t *>(input);
    const size_t max_frame = static_cast<size_t>(data->owner->cfg.max_frame_payload);
    size_t offset = 0;
    bool first = true;
    try {
        while (offset < size) {
            const size_t amount = std::min(max_frame, size - offset);
            store::impl::event current;
            current.kind = store::impl::event_kind::response;
            current.id = data->id;
            current.stream_chunk = stream_chunk;
            if (stream_chunk) {
                if (first) current.frame_flags |= FRAME_FIRST_SEGMENT;
                if (offset + amount == size) current.frame_flags |= FRAME_LAST_SEGMENT;
                current.stream_reservation = true;
            }
            if (!data->owner->enqueue_payload(std::move(current), bytes + offset, amount)) {
                ++data->owner->counter_append_failures;
                data->append_failed = true;
                return false;
            }
            first = false;
            offset += amount;
        }
    } catch (...) {
            ++data->owner->counter_append_failures;
            data->append_failed = true;
            return false;
    }
    return true;
}

bool session::set_usage_json(std::string usage_json) {
    if (!data || !data->owner) return false;
    std::lock_guard<std::mutex> lock(data->mutex);
    if (data->finished || usage_json.size() > data->owner->cfg.max_frame_payload) return false;
    if (usage_json.empty()) return true;
    store::impl::event current;
    current.kind = store::impl::event_kind::usage;
    current.id = data->id;
    const auto * bytes = reinterpret_cast<const uint8_t *>(usage_json.data());
    if (!data->owner->enqueue_payload(std::move(current), bytes, usage_json.size())) {
        ++data->owner->counter_append_failures;
        return false;
    }
    return true;
}

result session::failure_status() const {
    if (!data || !data->owner) return make_result(status::stopped);
    return data->owner->current_failure();
}

result session::finish(
        int http_status,
        outcome terminal_outcome,
        bool streaming,
        bool transport_complete,
        bool transport_complete_known) {
    if (!data || !data->owner) return make_result(status::stopped);
    std::unique_lock<std::mutex> lock(data->mutex);
    if (data->finished) return make_result(status::stopped);
    data->finished = true;
    if (data->append_failed) {
        ++data->owner->counter_terminal_failures;
        return data->owner->current_failure();
    }
    auto ack = std::make_shared<store::impl::acknowledgement>();
    store::impl::event current;
    current.kind = store::impl::event_kind::finish;
    current.id = data->id;
    current.http_status = http_status;
    current.terminal_outcome = terminal_outcome;
    current.streaming = streaming;
    current.transport_complete = transport_complete;
    current.transport_complete_known = transport_complete_known;
    current.ack = ack;
    if (!data->owner->enqueue(std::move(current))) {
        ++data->owner->counter_terminal_failures;
        return data->owner->current_failure();
    }
    const uint64_t timeout_ms = data->owner->cfg.terminal_ack_timeout_ms;
    lock.unlock();
    result completed = ack->wait(timeout_ms);
    if (completed.value == status::timeout) ++data->owner->counter_terminal_timeouts;
    if (!completed.durable) ++data->owner->counter_terminal_failures;
    return completed;
}

uint64_t session::record_id() const noexcept { return data ? data->id : 0; }

struct reader::impl {
    impl(std::string root_path, bool require_private) {
        result opened = open_private_root(fs::path(std::move(root_path)), require_private, false, root_fd);
        if (opened.value != status::ok) {
            throw std::runtime_error(std::string("request-history reader root: ") + status_name(opened.value));
        }
    }
    ~impl() { close_fd(root_fd); }
    int root_fd = -1;
};

reader::reader(std::string root_path, bool require_private_root) :
    data(new impl(std::move(root_path), require_private_root)) {}
reader::~reader() = default;

result reader::inspect(const read_limits & limits, std::vector<record_metadata> & records) const {
    records.clear();
    if (limits.max_files == 0 || limits.max_namespace_entries < limits.max_files ||
        limits.max_file_bytes < FILE_HEADER_BYTES) return make_result(status::invalid_config);
    std::vector<std::string> names;
    result listed = list_directory(data->root_fd, names, limits.max_namespace_entries);
    if (listed.value != status::ok) return listed;
    struct candidate { std::string name; file_identity identity; };
    std::vector<candidate> candidates;
    for (const std::string & name : names) {
        file_identity identity;
        if (!parse_file_name(name, identity) ||
            (identity.kind != file_kind::complete && identity.kind != file_kind::incomplete)) continue;
        candidates.push_back({ name, identity });
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate & left, const candidate & right) {
        return left.identity.wall_ns != right.identity.wall_ns ? left.identity.wall_ns < right.identity.wall_ns :
                                                                 left.identity.id < right.identity.id;
    });
    if (candidates.size() > limits.max_files) return make_result(status::too_large);
    for (const candidate & candidate : candidates) {
        int descriptor = -1;
        uint64_t file_size = 0;
        result opened = open_record_at(data->root_fd, candidate.name, limits.max_file_bytes, descriptor, file_size);
        if (opened.value != status::ok) return opened;
        parse_output parsed;
        parse_options options;
        options.max_frame_payload = MAX_FRAME_PAYLOAD;
        result valid = parse_record_fd(descriptor, file_size, options, parsed);
        struct stat after_parse = {};
        const result stable = private_regular_info_fd(descriptor, after_parse);
        if (valid.value == status::ok && (stable.value != status::ok || after_parse.st_size < 0 ||
            static_cast<uint64_t>(after_parse.st_size) != file_size ||
            verify_name_matches_inode(data->root_fd, candidate.name, after_parse).value != status::ok)) {
            valid = stable.value != status::ok ? stable : make_result(status::path_security, EAGAIN);
        }
        close_fd(descriptor);
        if (valid.value != status::ok) return valid;
        if (parsed.value.metadata.record_id != candidate.identity.id ||
            parsed.value.metadata.started_wall_ns != candidate.identity.wall_ns) return make_result(status::malformed);
        parsed.value.metadata.file_name = candidate.name;
        records.push_back(std::move(parsed.value.metadata));
    }
    return make_result(status::ok);
}

result reader::read(const std::string & file_name, const read_limits & limits, record & output) const {
    output = {};
    file_identity identity;
    if (!parse_file_name(file_name, identity) ||
        (identity.kind != file_kind::complete && identity.kind != file_kind::incomplete) ||
        file_name.find('/') != std::string::npos) return make_result(status::path_security, EINVAL);
    if (limits.max_file_bytes < FILE_HEADER_BYTES) return make_result(status::invalid_config);
    int descriptor = -1;
    uint64_t file_size = 0;
    result opened = open_record_at(data->root_fd, file_name, limits.max_file_bytes, descriptor, file_size);
    if (opened.value != status::ok) return opened;
    parse_output parsed;
    parse_options options;
    options.max_frame_payload = MAX_FRAME_PAYLOAD;
    options.max_total_body = limits.max_total_body_bytes;
    options.collect_bodies = true;
    result valid = parse_record_fd(descriptor, file_size, options, parsed);
    struct stat after_parse = {};
    const result stable = private_regular_info_fd(descriptor, after_parse);
    if (valid.value == status::ok && (stable.value != status::ok || after_parse.st_size < 0 ||
        static_cast<uint64_t>(after_parse.st_size) != file_size ||
        verify_name_matches_inode(data->root_fd, file_name, after_parse).value != status::ok)) {
        valid = stable.value != status::ok ? stable : make_result(status::path_security, EAGAIN);
    }
    close_fd(descriptor);
    if (valid.value != status::ok) return valid;
    if (parsed.value.metadata.record_id != identity.id || parsed.value.metadata.started_wall_ns != identity.wall_ns) {
        return make_result(status::malformed);
    }
    parsed.value.metadata.file_name = file_name;
    output = std::move(parsed.value);
    return make_result(status::ok);
}

bool should_capture(const std::string & method, const std::string & input_path, const std::string & path_prefix) {
    if (method != "POST") return false;
    std::string path = input_path;
    if (!path_prefix.empty() && path.rfind(path_prefix, 0) == 0 &&
        (path.size() == path_prefix.size() || path[path_prefix.size()] == '/')) {
        path.erase(0, path_prefix.size());
        if (path.empty()) path = "/";
    }
    static const char * endpoints[] = {
        "/completion", "/completions", "/v1/completions",
        "/chat/completions", "/v1/chat/completions",
        "/responses", "/v1/responses", "/v1/messages",
        "/infill", "/embedding", "/embeddings", "/v1/embeddings",
        "/rerank", "/reranking", "/v1/rerank", "/v1/reranking",
        "/audio/transcriptions", "/v1/audio/transcriptions",
        "/chat/completions/input_tokens", "/v1/chat/completions/input_tokens",
        "/responses/input_tokens", "/v1/responses/input_tokens", "/v1/messages/count_tokens",
    };
    return std::find(std::begin(endpoints), std::end(endpoints), path) != std::end(endpoints);
}

}  // namespace server_request_history

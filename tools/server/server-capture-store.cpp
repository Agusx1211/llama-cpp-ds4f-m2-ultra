#include "server-capture-store.h"

#include <dirent.h>
#include <fcntl.h>
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__) || defined(__DragonFly__)
#    include <sys/random.h>
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace fs = std::filesystem;

namespace server_capture {
namespace {

constexpr std::array<uint8_t, 8> MANIFEST_MAGIC = {
    { 'S', 'C', 'A', 'P', 'M', 'F', '0', '1' }
};
constexpr std::array<uint8_t, 8> SHARD_MAGIC = {
    { 'S', 'C', 'A', 'P', 'S', 'H', '0', '1' }
};
constexpr size_t   MANIFEST_ENTRY_BYTES            = 72;
constexpr uint32_t RICH_RECORD_FLAG                = 1U;
constexpr uint32_t MANIFEST_FLAG_REDACTED_IDENTITY = 1U;
constexpr mode_t   PRIVATE_FILE_MODE               = S_IRUSR | S_IWUSR;
constexpr mode_t   PRIVATE_DIRECTORY_MODE          = S_IRWXU;
constexpr uint64_t CAPTURE_MAX_RETAINED_RECORDS    = 1ULL << 32;
constexpr uint64_t ADMISSION_CLOSED                = UINT64_C(1) << 63;
constexpr uint64_t ADMISSION_COUNT_MASK            = ADMISSION_CLOSED - 1;

struct sha256_context {
    uint32_t state[8]      = {};
    uint64_t bit_length    = 0;
    uint8_t  buffer[64]    = {};
    size_t   buffer_length = 0;
};

constexpr std::array<uint32_t, 64> SHA256_K = {
    {
     0x428a2f98U, 0x71374491U, 0xb5c0fbcf,  0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
     0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
     0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
     0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
     0x27b70a85U, 0x2e1b2138U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x8cc70208U,
     0x90befffaU, 0xa4506ce4U, 0xbef9a3f7U, 0xc67178f2U,
     }
};

uint32_t rotate_right(uint32_t value, unsigned shift) noexcept {
    return (value >> shift) | (value << (32U - shift));
}

void sha256_compress(uint32_t state[8], const uint8_t block[64]) noexcept {
    uint32_t words[64] = {};
    for (size_t index = 0; index < 16; ++index) {
        words[index] =
            (static_cast<uint32_t>(block[index * 4]) << 24U) | (static_cast<uint32_t>(block[index * 4 + 1]) << 16U) |
            (static_cast<uint32_t>(block[index * 4 + 2]) << 8U) | static_cast<uint32_t>(block[index * 4 + 3]);
    }
    for (size_t index = 16; index < 64; ++index) {
        const uint32_t s0 =
            rotate_right(words[index - 15], 7U) ^ rotate_right(words[index - 15], 18U) ^ (words[index - 15] >> 3U);
        const uint32_t s1 =
            rotate_right(words[index - 2], 17U) ^ rotate_right(words[index - 2], 19U) ^ (words[index - 2] >> 10U);
        words[index] = words[index - 16] + s0 + words[index - 7] + s1;
    }
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];
    for (size_t index = 0; index < 64; ++index) {
        const uint32_t sum1     = rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const uint32_t choose   = (e & f) ^ ((~e) & g);
        const uint32_t temp1    = h + sum1 + choose + SHA256_K[index] + words[index];
        const uint32_t sum0     = rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2    = sum0 + majority;
        h                       = g;
        g                       = f;
        f                       = e;
        e                       = d + temp1;
        d                       = c;
        c                       = b;
        b                       = a;
        a                       = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256_init(sha256_context & context) noexcept {
    context.state[0]      = 0x6a09e667U;
    context.state[1]      = 0xbb67ae85U;
    context.state[2]      = 0x3c6ef372U;
    context.state[3]      = 0xa54ff53aU;
    context.state[4]      = 0x510e527fU;
    context.state[5]      = 0x9b05688cU;
    context.state[6]      = 0x1f83d9abU;
    context.state[7]      = 0x5be0cd19U;
    context.bit_length    = 0;
    context.buffer_length = 0;
}

void sha256_update(sha256_context & context, const void * input, size_t size) noexcept {
    const auto * bytes = static_cast<const uint8_t *>(input);
    context.bit_length += static_cast<uint64_t>(size) * 8U;
    if (context.buffer_length != 0) {
        const size_t count = std::min<size_t>(64 - context.buffer_length, size);
        std::memcpy(context.buffer + context.buffer_length, bytes, count);
        context.buffer_length += count;
        bytes += count;
        size -= count;
        if (context.buffer_length == 64) {
            sha256_compress(context.state, context.buffer);
            context.buffer_length = 0;
        }
    }
    while (size >= 64) {
        sha256_compress(context.state, bytes);
        bytes += 64;
        size -= 64;
    }
    if (size != 0) {
        std::memcpy(context.buffer, bytes, size);
        context.buffer_length = size;
    }
}

capture_digest sha256_finish(sha256_context & context) noexcept {
    const uint64_t bit_length               = context.bit_length;
    context.buffer[context.buffer_length++] = 0x80;
    if (context.buffer_length > 56) {
        while (context.buffer_length < 64) {
            context.buffer[context.buffer_length++] = 0;
        }
        sha256_compress(context.state, context.buffer);
        context.buffer_length = 0;
    }
    while (context.buffer_length < 56) {
        context.buffer[context.buffer_length++] = 0;
    }
    for (int shift = 7; shift >= 0; --shift) {
        context.buffer[context.buffer_length++] =
            static_cast<uint8_t>(bit_length >> (static_cast<unsigned>(shift) * 8U));
    }
    sha256_compress(context.state, context.buffer);
    capture_digest result = {};
    for (size_t index = 0; index < 8; ++index) {
        result[index * 4]     = static_cast<uint8_t>(context.state[index] >> 24U);
        result[index * 4 + 1] = static_cast<uint8_t>(context.state[index] >> 16U);
        result[index * 4 + 2] = static_cast<uint8_t>(context.state[index] >> 8U);
        result[index * 4 + 3] = static_cast<uint8_t>(context.state[index]);
    }
    return result;
}

capture_digest sha256(const void * data, size_t size) noexcept {
    sha256_context context;
    sha256_init(context);
    sha256_update(context, data, size);
    return sha256_finish(context);
}

bool digest_equal(const capture_digest & left, const capture_digest & right) noexcept {
    uint8_t difference = 0;
    for (size_t index = 0; index < left.size(); ++index) {
        difference = static_cast<uint8_t>(difference | static_cast<uint8_t>(left[index] ^ right[index]));
    }
    return difference == 0;
}

bool salt_zero(const std::array<uint8_t, 16> & salt) noexcept {
    for (const uint8_t byte : salt) {
        if (byte != 0) {
            return false;
        }
    }
    return true;
}

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

void append_digest(std::vector<uint8_t> & bytes, const capture_digest & digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void append_zeroes(std::vector<uint8_t> & bytes, size_t count) {
    bytes.insert(bytes.end(), count, 0);
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

bool read_digest(const std::vector<uint8_t> & bytes, size_t & cursor, capture_digest & digest) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < digest.size()) {
        return false;
    }
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(cursor), digest.size(), digest.begin());
    cursor += digest.size();
    return true;
}

bool read_salt(const std::vector<uint8_t> & bytes, size_t & cursor, std::array<uint8_t, 16> & salt) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < salt.size()) {
        return false;
    }
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(cursor), salt.size(), salt.begin());
    cursor += salt.size();
    return true;
}

bool read_magic(const std::vector<uint8_t> & bytes, size_t & cursor, const std::array<uint8_t, 8> & magic) noexcept {
    if (cursor > bytes.size() || bytes.size() - cursor < magic.size()) {
        return false;
    }
    if (!std::equal(magic.begin(), magic.end(), bytes.begin() + static_cast<std::ptrdiff_t>(cursor))) {
        return false;
    }
    cursor += magic.size();
    return true;
}

std::string shard_name(uint64_t sequence, bool temporary) {
    char buffer[64] = {};
    std::snprintf(buffer, sizeof(buffer), temporary ? ".shard-%020llu.tmp" : "shard-%020llu.cap",
                  static_cast<unsigned long long>(sequence));
    return buffer;
}

std::string tombstone_name(uint64_t sequence) {
    char buffer[64] = {};
    std::snprintf(buffer, sizeof(buffer), ".delete-%020llu.tomb", static_cast<unsigned long long>(sequence));
    return buffer;
}

bool parse_shard_sequence(const std::string & name, uint64_t & sequence, bool & temporary) {
    temporary = false;
    const std::string prefix =
        name.rfind("shard-", 0) == 0 ? "shard-" : (name.rfind(".shard-", 0) == 0 ? ".shard-" : "");
    if (prefix.empty()) {
        return false;
    }
    temporary                        = prefix[0] == '.';
    constexpr size_t sequence_digits = 20;
    if (name.size() != prefix.size() + sequence_digits + 4 ||
        name.substr(name.size() - 4) != (temporary ? ".tmp" : ".cap")) {
        return false;
    }
    const size_t begin  = prefix.size();
    const size_t end    = name.size() - 4;
    uint64_t     parsed = 0;
    for (size_t index = begin; index < end; ++index) {
        const char character = name[index];
        if (character < '0' || character > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    if (parsed == 0) {
        return false;
    }
    sequence = parsed;
    return true;
}

capture_store_result result(capture_store_status status, int os_error = 0, bool committed = false) {
    return { status, os_error, committed };
}

capture_store_result errno_result(int error) {
    return result(error == ENOSPC ? capture_store_status::no_space : capture_store_status::io_error, error);
}

bool valid_capture_mode_value(uint32_t value) noexcept {
    return value == static_cast<uint32_t>(mode::metrics_only) || value == static_cast<uint32_t>(mode::compact) ||
           value == static_cast<uint32_t>(mode::sampled_rich);
}

bool path_has_traversal(const fs::path & path) {
    if (!path.is_absolute()) {
        return true;
    }
    for (const fs::path & component : path) {
        if (component == "." || component == "..") {
            return true;
        }
    }
    return false;
}

capture_store_result open_private_root(const fs::path & root, bool require_private, int & descriptor) {
    descriptor = -1;
    if (path_has_traversal(root) || root == root.root_path()) {
        return result(capture_store_status::path_security, EINVAL);
    }
    // Intermediate parent directories are no-follow walked but are not made
    // private here; deployment policy must trust that containing path. Only
    // the capture root itself is enforced as private when requested.
    int current = ::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (current < 0) {
        return errno_result(errno);
    }
    for (const fs::path & component : root) {
        if (component == root.root_path()) {
            continue;
        }
        const std::string name = component.string();
        int               next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0 && errno == ENOENT) {
            if (::mkdirat(current, name.c_str(), PRIVATE_DIRECTORY_MODE) != 0 && errno != EEXIST) {
                const int error = errno;
                ::close(current);
                return errno_result(error);
            }
            next = ::openat(current, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (next < 0) {
            const int error = errno;
            ::close(current);
            return result(capture_store_status::path_security, error);
        }
        struct stat status = {};
        if (::fstat(next, &status) != 0 || !S_ISDIR(status.st_mode)) {
            const int error = errno == 0 ? ENOTDIR : errno;
            ::close(next);
            ::close(current);
            return result(capture_store_status::path_security, error);
        }
        ::close(current);
        current = next;
    }
    struct stat status = {};
    if (::fstat(current, &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::geteuid()) {
        const int error = errno == 0 ? EACCES : errno;
        ::close(current);
        return result(capture_store_status::path_security, error);
    }
    if (require_private && (::fchmod(current, PRIVATE_DIRECTORY_MODE) != 0 || ::fstat(current, &status) != 0 ||
                            (status.st_mode & (S_IRWXG | S_IRWXO)) != 0)) {
        const int error = errno == 0 ? EACCES : errno;
        ::close(current);
        return result(capture_store_status::path_security, error);
    }
    descriptor = current;
    return result(capture_store_status::ok);
}

capture_store_result safe_remove_file_at(int root_fd, const std::string & name, bool fail_unlink = false) {
    if (fail_unlink) {
        return result(capture_store_status::deletion_failed, EIO);
    }
    struct stat status = {};
    if (::fstatat(root_fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        return errno == ENOENT ? result(capture_store_status::ok) : result(capture_store_status::path_security, errno);
    }
    if (S_ISLNK(status.st_mode) || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        return result(capture_store_status::path_security, EACCES);
    }
    return ::unlinkat(root_fd, name.c_str(), 0) == 0 ? result(capture_store_status::ok) :
                                                       result(capture_store_status::deletion_failed, errno);
}

capture_store_result validate_private_file_at(int root_fd, const std::string & name, bool allow_missing = false) {
    struct stat status = {};
    if (::fstatat(root_fd, name.c_str(), &status, AT_SYMLINK_NOFOLLOW) != 0) {
        if (allow_missing && errno == ENOENT) {
            return result(capture_store_status::ok);
        }
        return result(capture_store_status::path_security, errno);
    }
    if (S_ISLNK(status.st_mode) || !S_ISREG(status.st_mode) || status.st_uid != ::geteuid() ||
        (status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        return result(capture_store_status::path_security, EACCES);
    }
    return result(capture_store_status::ok);
}

bool parse_tombstone_sequence(const std::string & name, uint64_t & sequence) noexcept {
    constexpr size_t PREFIX = 8;  // .delete-
    constexpr size_t DIGITS = 20;
    constexpr size_t SUFFIX = 5;  // .tomb
    if (name.size() != PREFIX + DIGITS + SUFFIX || name.rfind(".delete-", 0) != 0 ||
        name.substr(PREFIX + DIGITS) != ".tomb") {
        return false;
    }
    uint64_t parsed = 0;
    for (size_t index = PREFIX; index < PREFIX + DIGITS; ++index) {
        const char character = name[index];
        if (character < '0' || character > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10U) {
            return false;
        }
        parsed = parsed * 10U + digit;
    }
    if (parsed == 0) {
        return false;
    }
    sequence = parsed;
    return true;
}

class capture_wakeup {
  public:
    capture_wakeup() = default;

    capture_wakeup(const capture_wakeup &)             = delete;
    capture_wakeup & operator=(const capture_wakeup &) = delete;

    bool post() noexcept {
        epoch.fetch_add(1, std::memory_order_release);
        condition.notify_one();
        return true;
    }

    uint64_t snapshot() const noexcept { return epoch.load(std::memory_order_acquire); }

    void wait(uint64_t & observed) {
        std::unique_lock<std::mutex> lock(mutex);
        // C++17 has no atomic_wait, and condition_variable notification can
        // race the worker's unlock/register transition because producers must
        // remain lock-free. A bounded timeout closes that final lost-wake
        // window while the epoch predicate handles ordinary notifications.
        (void) condition.wait_for(lock, std::chrono::milliseconds(1),
                                  [this, observed]() { return epoch.load(std::memory_order_acquire) != observed; });
        observed = epoch.load(std::memory_order_acquire);
    }

  private:
    std::atomic<uint64_t>   epoch{ 0 };
    mutable std::mutex      mutex;
    std::condition_variable condition;
};

capture_store_result write_all(int                          descriptor,
                               const uint8_t *              data,
                               size_t                       size,
                               const capture_store_faults & faults,
                               uint64_t &                   written_total) {
    while (size != 0) {
        if (faults.write_fault != capture_write_fault::none && written_total >= faults.fail_after_bytes) {
            return result(faults.write_fault == capture_write_fault::no_space ? capture_store_status::no_space :
                                                                                capture_store_status::short_write,
                          faults.write_fault == capture_write_fault::no_space ? ENOSPC : EIO);
        }
        uint64_t count_u64 = static_cast<uint64_t>(size);
        count_u64          = std::min<uint64_t>(count_u64, faults.max_write_size);
        if (faults.write_fault != capture_write_fault::none && faults.fail_after_bytes != UINT64_MAX) {
            count_u64 = std::min<uint64_t>(count_u64, faults.fail_after_bytes - written_total);
        }
        if (count_u64 == 0 || count_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return result(faults.write_fault == capture_write_fault::no_space ? capture_store_status::no_space :
                                                                                capture_store_status::short_write,
                          faults.write_fault == capture_write_fault::no_space ? ENOSPC : EIO);
        }
        const ssize_t count = ::write(descriptor, data, static_cast<size_t>(count_u64));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return errno_result(errno);
        }
        if (count == 0) {
            return result(capture_store_status::short_write, EIO);
        }
        const size_t advanced = static_cast<size_t>(count);
        data += advanced;
        size -= advanced;
        written_total += static_cast<uint64_t>(advanced);
    }
    return result(capture_store_status::ok);
}

capture_store_result write_file_at(int                          root_fd,
                                   const std::string &          name,
                                   const std::vector<uint8_t> & bytes,
                                   const capture_store_faults & faults,
                                   bool                         preserve_on_failure) {
    const int descriptor =
        ::openat(root_fd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, PRIVATE_FILE_MODE);
    if (descriptor < 0) {
        return (errno == ELOOP || errno == EEXIST) ? result(capture_store_status::path_security, errno) :
                                                     errno_result(errno);
    }
    struct stat file_status = {};
    if (faults.fail_fstat || ::fstat(descriptor, &file_status) != 0 || !S_ISREG(file_status.st_mode) ||
        file_status.st_uid != ::geteuid() || (file_status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        const int error = errno == 0 ? EACCES : errno;
        ::close(descriptor);
        return faults.fail_fstat ? result(capture_store_status::io_error, EIO) :
                                   result(capture_store_status::path_security, error);
    }
    uint64_t             written      = 0;
    capture_store_result write_status = write_all(descriptor, bytes.data(), bytes.size(), faults, written);
    if (write_status.status == capture_store_status::ok && (faults.fail_file_fsync || ::fsync(descriptor) != 0)) {
        write_status = faults.fail_file_fsync ? result(capture_store_status::io_error, EIO) : errno_result(errno);
    }
    const int close_error = ::close(descriptor) == 0 ? 0 : errno;
    if (write_status.status == capture_store_status::ok && close_error != 0) {
        write_status = errno_result(close_error);
    }
    if (write_status.status != capture_store_status::ok && !preserve_on_failure) {
        (void) safe_remove_file_at(root_fd, name, faults.fail_unlink);
    }
    return write_status;
}

capture_store_result read_file_at(int                    root_fd,
                                  const std::string &    name,
                                  uint64_t               max_bytes,
                                  std::vector<uint8_t> & bytes,
                                  capture_store_status   missing_status,
                                  capture_store_status   too_large_status,
                                  bool                   fail_fstat = false) {
    const int descriptor = ::openat(root_fd, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (descriptor < 0) {
        return errno == ELOOP ? result(capture_store_status::path_security, errno) :
                                result(errno == ENOENT ? missing_status : capture_store_status::io_error, errno);
    }
    struct stat file_status = {};
    if (fail_fstat || ::fstat(descriptor, &file_status) != 0) {
        const int error = errno;
        ::close(descriptor);
        return result(capture_store_status::io_error, fail_fstat ? EIO : error);
    }
    if (!S_ISREG(file_status.st_mode) || file_status.st_size < 0 || file_status.st_uid != ::geteuid() ||
        (file_status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
        ::close(descriptor);
        return result(capture_store_status::path_security, EACCES);
    }
    const uint64_t size = static_cast<uint64_t>(file_status.st_size);
    if (size > max_bytes || size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        ::close(descriptor);
        return result(too_large_status);
    }
    try {
        bytes.resize(static_cast<size_t>(size));
    } catch (const std::bad_alloc &) {
        ::close(descriptor);
        return result(capture_store_status::io_error, ENOMEM);
    }
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            const int error = errno;
            ::close(descriptor);
            return result(capture_store_status::io_error, error);
        }
        if (count == 0) {
            ::close(descriptor);
            return result(capture_store_status::truncated);
        }
        offset += static_cast<size_t>(count);
    }
    struct stat final_status = {};
    if (::fstat(descriptor, &final_status) != 0) {
        const int error = errno;
        ::close(descriptor);
        return result(capture_store_status::io_error, error);
    }
    if (final_status.st_size < 0 || static_cast<uint64_t>(final_status.st_size) != size) {
        ::close(descriptor);
        return static_cast<uint64_t>(final_status.st_size) > size ? result(capture_store_status::trailing_data) :
                                                                    result(capture_store_status::truncated);
    }
    if (::close(descriptor) != 0) {
        return errno_result(errno);
    }
    return result(capture_store_status::ok);
}

bool fsync_directory_fd(int   descriptor,
                        int & error,
                        bool  require_private,
                        bool  fail_fsync = false,
                        bool  fail_fstat = false) {
    if (descriptor < 0) {
        error = EBADF;
        return false;
    }
    struct stat status = {};
    if (fail_fstat || ::fstat(descriptor, &status) != 0 || !S_ISDIR(status.st_mode) || status.st_uid != ::geteuid() ||
        (require_private && (status.st_mode & (S_IRWXG | S_IRWXO)) != 0)) {
        error = fail_fstat ? EIO : (errno == 0 ? EACCES : errno);
        return false;
    }
    const bool success = !fail_fsync && ::fsync(descriptor) == 0;
    error              = success ? 0 : (fail_fsync ? EIO : errno);
    return success;
}

std::vector<uint8_t> serialize_manifest(const capture_manifest & manifest, const std::array<uint8_t, 16> & salt) {
    std::vector<uint8_t> bytes;
    bytes.reserve(CAPTURE_MANIFEST_HEADER_BYTES + manifest.shards.size() * MANIFEST_ENTRY_BYTES +
                  CAPTURE_MANIFEST_FOOTER_BYTES);
    bytes.insert(bytes.end(), MANIFEST_MAGIC.begin(), MANIFEST_MAGIC.end());
    append_u32(bytes, CAPTURE_STORE_FORMAT_VERSION);
    append_u32(bytes, static_cast<uint32_t>(CAPTURE_MANIFEST_HEADER_BYTES));
    append_u64(bytes, 0);  // patched below
    append_u64(bytes, manifest.generation);
    append_u32(bytes, static_cast<uint32_t>(manifest.capture_mode));
    append_u32(bytes, MANIFEST_FLAG_REDACTED_IDENTITY);
    append_u64(bytes, manifest.total_records);
    append_u64(bytes, manifest.total_bytes);
    append_u32(bytes, static_cast<uint32_t>(manifest.shards.size()));
    append_u32(bytes, 0);
    bytes.insert(bytes.end(), salt.begin(), salt.end());
    for (const capture_shard_info & shard : manifest.shards) {
        append_u64(bytes, shard.sequence);
        append_u64(bytes, shard.first_monotonic_ns);
        append_u64(bytes, shard.last_monotonic_ns);
        append_u32(bytes, shard.record_count);
        append_u32(bytes, 0);
        append_u64(bytes, shard.byte_count);
        append_digest(bytes, shard.checksum);
    }
    const uint64_t declared_size = static_cast<uint64_t>(bytes.size() + CAPTURE_MANIFEST_FOOTER_BYTES);
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[16 + shift / 8] = static_cast<uint8_t>(declared_size >> shift);
    }
    const capture_digest checksum = sha256(bytes.data(), bytes.size());
    append_digest(bytes, checksum);
    return bytes;
}

capture_store_result parse_manifest(const std::vector<uint8_t> & bytes,
                                    const capture_store_config & config,
                                    capture_manifest &           manifest,
                                    std::array<uint8_t, 16> &    salt) {
    if (bytes.size() < CAPTURE_MANIFEST_HEADER_BYTES + CAPTURE_MANIFEST_FOOTER_BYTES) {
        return result(capture_store_status::truncated);
    }
    size_t cursor = 0;
    if (!read_magic(bytes, cursor, MANIFEST_MAGIC)) {
        return result(capture_store_status::malformed_manifest);
    }
    uint32_t version       = 0;
    uint32_t header_bytes  = 0;
    uint64_t declared_size = 0;
    if (!read_u32(bytes, cursor, version) || !read_u32(bytes, cursor, header_bytes) ||
        !read_u64(bytes, cursor, declared_size) || version != CAPTURE_STORE_FORMAT_VERSION ||
        header_bytes != CAPTURE_MANIFEST_HEADER_BYTES) {
        return result(capture_store_status::malformed_manifest);
    }
    if (declared_size > config.max_manifest_bytes ||
        declared_size < CAPTURE_MANIFEST_HEADER_BYTES + CAPTURE_MANIFEST_FOOTER_BYTES) {
        return result(capture_store_status::manifest_too_large);
    }
    if (bytes.size() < declared_size) {
        return result(capture_store_status::truncated);
    }
    if (bytes.size() > declared_size) {
        return result(capture_store_status::trailing_data);
    }
    const size_t   checksum_offset = bytes.size() - CAPTURE_MANIFEST_FOOTER_BYTES;
    capture_digest expected        = {};
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(checksum_offset), expected.size(), expected.begin());
    if (!digest_equal(expected, sha256(bytes.data(), checksum_offset))) {
        return result(capture_store_status::checksum_mismatch);
    }

    uint64_t generation    = 0;
    uint32_t mode_value    = 0;
    uint32_t flags         = 0;
    uint64_t total_records = 0;
    uint64_t total_bytes   = 0;
    uint32_t shard_count   = 0;
    uint32_t reserved      = 0;
    if (!read_u64(bytes, cursor, generation) || !read_u32(bytes, cursor, mode_value) ||
        !read_u32(bytes, cursor, flags) || !read_u64(bytes, cursor, total_records) ||
        !read_u64(bytes, cursor, total_bytes) || !read_u32(bytes, cursor, shard_count) ||
        !read_u32(bytes, cursor, reserved) || !read_salt(bytes, cursor, salt)) {
        return result(capture_store_status::malformed_manifest);
    }
    if (salt_zero(salt)) {
        return result(capture_store_status::malformed_manifest);
    }
    if (reserved != 0 || flags != MANIFEST_FLAG_REDACTED_IDENTITY || generation == 0 || shard_count == 0 ||
        shard_count > config.max_retained_shards || !valid_capture_mode_value(mode_value) ||
        static_cast<mode>(mode_value) != config.capture_mode) {
        return result(capture_store_status::malformed_manifest);
    }
    const uint64_t expected_size = static_cast<uint64_t>(CAPTURE_MANIFEST_HEADER_BYTES) +
                                   static_cast<uint64_t>(shard_count) * MANIFEST_ENTRY_BYTES +
                                   CAPTURE_MANIFEST_FOOTER_BYTES;
    if (expected_size != declared_size || cursor != CAPTURE_MANIFEST_HEADER_BYTES) {
        return result(capture_store_status::malformed_manifest);
    }
    manifest                = {};
    manifest.format_version = version;
    manifest.generation     = generation;
    manifest.capture_mode   = static_cast<mode>(mode_value);
    manifest.identity_salt  = salt;
    manifest.total_records  = total_records;
    manifest.total_bytes    = total_bytes;
    try {
        manifest.shards.resize(shard_count);
    } catch (const std::bad_alloc &) {
        return result(capture_store_status::io_error, ENOMEM);
    }
    uint64_t records_sum       = 0;
    uint64_t bytes_sum         = 0;
    uint64_t previous_sequence = 0;
    for (capture_shard_info & shard : manifest.shards) {
        uint32_t entry_reserved = 0;
        if (!read_u64(bytes, cursor, shard.sequence) || !read_u64(bytes, cursor, shard.first_monotonic_ns) ||
            !read_u64(bytes, cursor, shard.last_monotonic_ns) || !read_u32(bytes, cursor, shard.record_count) ||
            !read_u32(bytes, cursor, entry_reserved) || !read_u64(bytes, cursor, shard.byte_count) ||
            !read_digest(bytes, cursor, shard.checksum) || entry_reserved != 0 || shard.sequence == 0 ||
            shard.sequence <= previous_sequence || shard.record_count == 0 ||
            shard.byte_count < CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES) {
            return result(capture_store_status::malformed_manifest);
        }
        previous_sequence = shard.sequence;
        if (records_sum > std::numeric_limits<uint64_t>::max() - shard.record_count ||
            bytes_sum > std::numeric_limits<uint64_t>::max() - shard.byte_count) {
            return result(capture_store_status::malformed_manifest);
        }
        records_sum += shard.record_count;
        bytes_sum += shard.byte_count;
    }
    if (cursor != checksum_offset || records_sum != total_records || bytes_sum != total_bytes ||
        total_records > config.max_retained_records || total_bytes > config.max_retained_bytes) {
        return result(capture_store_status::malformed_manifest);
    }
    return result(capture_store_status::ok);
}

std::vector<uint8_t> serialize_shard(const std::vector<uint8_t> & payload,
                                     uint64_t                     sequence,
                                     mode                         capture_mode,
                                     uint32_t                     record_count,
                                     uint64_t                     first_ns,
                                     uint64_t                     last_ns,
                                     capture_digest &             checksum) {
    std::vector<uint8_t> bytes;
    bytes.reserve(CAPTURE_SHARD_HEADER_BYTES + payload.size() + CAPTURE_SHARD_FOOTER_BYTES);
    bytes.insert(bytes.end(), SHARD_MAGIC.begin(), SHARD_MAGIC.end());
    append_u32(bytes, CAPTURE_SHARD_FORMAT_VERSION);
    append_u32(bytes, static_cast<uint32_t>(CAPTURE_SHARD_HEADER_BYTES));
    append_u64(bytes, static_cast<uint64_t>(CAPTURE_SHARD_HEADER_BYTES + payload.size() + CAPTURE_SHARD_FOOTER_BYTES));
    append_u64(bytes, sequence);
    append_u32(bytes, static_cast<uint32_t>(capture_mode));
    append_u32(bytes, record_count);
    append_u64(bytes, static_cast<uint64_t>(payload.size()));
    append_u64(bytes, first_ns);
    append_u64(bytes, last_ns);
    append_u64(bytes, 0);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    checksum = sha256(bytes.data(), bytes.size());
    append_digest(bytes, checksum);
    return bytes;
}

capture_store_result validate_shard_bytes(const std::vector<uint8_t> & bytes,
                                          const capture_shard_info &   expected,
                                          const capture_store_config & config,
                                          mode                         capture_mode) {
    if (bytes.size() < CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES) {
        return result(capture_store_status::truncated);
    }
    if (bytes.size() > config.max_shard_bytes) {
        return result(capture_store_status::shard_too_large);
    }
    size_t cursor = 0;
    if (!read_magic(bytes, cursor, SHARD_MAGIC)) {
        return result(capture_store_status::shard_corrupt);
    }
    uint32_t version       = 0;
    uint32_t header_bytes  = 0;
    uint64_t declared_size = 0;
    uint64_t sequence      = 0;
    uint32_t mode_value    = 0;
    uint32_t record_count  = 0;
    uint64_t payload_size  = 0;
    uint64_t first_ns      = 0;
    uint64_t last_ns       = 0;
    uint64_t reserved      = 0;
    if (!read_u32(bytes, cursor, version) || !read_u32(bytes, cursor, header_bytes) ||
        !read_u64(bytes, cursor, declared_size) || !read_u64(bytes, cursor, sequence) ||
        !read_u32(bytes, cursor, mode_value) || !read_u32(bytes, cursor, record_count) ||
        !read_u64(bytes, cursor, payload_size) || !read_u64(bytes, cursor, first_ns) ||
        !read_u64(bytes, cursor, last_ns) || !read_u64(bytes, cursor, reserved) ||
        version != CAPTURE_SHARD_FORMAT_VERSION || header_bytes != CAPTURE_SHARD_HEADER_BYTES || reserved != 0 ||
        !valid_capture_mode_value(mode_value) || mode_value != static_cast<uint32_t>(capture_mode) ||
        sequence != expected.sequence || record_count != expected.record_count ||
        first_ns != expected.first_monotonic_ns || last_ns != expected.last_monotonic_ns) {
        return result(capture_store_status::shard_corrupt);
    }
    if (declared_size > bytes.size()) {
        return result(capture_store_status::truncated);
    }
    if (declared_size < bytes.size()) {
        return result(capture_store_status::trailing_data);
    }
    if (static_cast<uint64_t>(bytes.size()) < expected.byte_count) {
        return result(capture_store_status::truncated);
    }
    if (static_cast<uint64_t>(bytes.size()) > expected.byte_count) {
        return result(capture_store_status::trailing_data);
    }
    if (payload_size > bytes.size() ||
        payload_size + CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES != bytes.size()) {
        return result(capture_store_status::truncated);
    }
    const size_t   checksum_offset   = bytes.size() - CAPTURE_SHARD_FOOTER_BYTES;
    capture_digest expected_checksum = {};
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(checksum_offset), expected_checksum.size(),
                expected_checksum.begin());
    if (!digest_equal(expected_checksum, sha256(bytes.data(), checksum_offset)) ||
        !digest_equal(expected_checksum, expected.checksum)) {
        return result(capture_store_status::checksum_mismatch);
    }

    size_t       payload_cursor = CAPTURE_SHARD_HEADER_BYTES;
    const size_t payload_end    = checksum_offset;
    uint32_t     count          = 0;
    while (payload_cursor < payload_end) {
        uint32_t record_version  = 0;
        uint32_t record_flags    = 0;
        uint32_t record_bytes    = 0;
        uint32_t record_reserved = 0;
        if (!read_u32(bytes, payload_cursor, record_version) || !read_u32(bytes, payload_cursor, record_flags) ||
            !read_u32(bytes, payload_cursor, record_bytes) || !read_u32(bytes, payload_cursor, record_reserved) ||
            record_version != CAPTURE_RECORD_FORMAT_VERSION || record_reserved != 0 ||
            (record_flags & ~RICH_RECORD_FLAG) != 0 ||
            record_bytes !=
                ((record_flags & RICH_RECORD_FLAG) != 0 ? CAPTURE_RICH_RECORD_BYTES : CAPTURE_COMPACT_RECORD_BYTES) ||
            ((record_flags & RICH_RECORD_FLAG) != 0 && capture_mode != mode::sampled_rich) ||
            record_bytes > payload_end - payload_cursor) {
            return result(capture_store_status::shard_corrupt);
        }
        payload_cursor += record_bytes;
        ++count;
    }
    if (payload_cursor != payload_end || count != record_count || count > config.max_shard_records) {
        return result(capture_store_status::shard_corrupt);
    }
    return result(capture_store_status::ok);
}

capture_digest request_tag(const std::array<uint8_t, 16> & salt, uint64_t request_id) {
    std::array<uint8_t, 24> bytes = {};
    std::copy(salt.begin(), salt.end(), bytes.begin());
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[16 + shift / 8] = static_cast<uint8_t>(request_id >> shift);
    }
    return sha256(bytes.data(), bytes.size());
}

void append_float(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be IEEE-754 binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(bytes, bits);
}

void append_record(const cycle_observation &       observation,
                   const std::array<uint8_t, 16> & salt,
                   bool                            rich,
                   bool                            redact_identity,
                   std::vector<uint8_t> &          payload) {
    std::vector<uint8_t> record;
    record.reserve(rich ? CAPTURE_RICH_RECORD_BYTES : CAPTURE_COMPACT_RECORD_BYTES);
    const capture_digest tag = request_tag(salt, observation.request_id);
    // A compact record persists only scheduler/timing/confidence telemetry.
    // The request ID, proposal IDs, and correction ID are intentionally absent
    // (the first is represented by a salted, non-reversible tag).
    if (redact_identity) {
        append_zeroes(record, 8);
    } else {
        record.insert(record.end(), tag.begin(), tag.begin() + 8);
    }
    append_u64(record, observation.committed_position);
    append_u64(record, observation.scheduler_epoch);
    append_u64(record, observation.monotonic_ns);
    append_u32(record, observation.schema_version);
    append_u32(record, observation.cycle_sequence);
    append_u32(record, observation.draft_time_us);
    append_u32(record, observation.verify_time_us);
    append_u32(record, observation.scheduler_time_us);
    record.push_back(observation.scheduled_decode_width);
    record.push_back(observation.verifier_geometry);
    record.push_back(observation.proposal_count);
    record.push_back(observation.accepted_prefix_length);
    record.push_back(observation.first_rejection);
    record.push_back(static_cast<uint8_t>(observation.active_mode));
    record.push_back(static_cast<uint8_t>(observation.bypass));
    record.push_back(observation.flags);
    append_zeroes(record, 4);
    for (const float probability : observation.selected_probabilities) {
        append_float(record, probability);
    }
    for (const float confidence : observation.raw_confidences) {
        append_float(record, confidence);
    }
    append_zeroes(record, CAPTURE_COMPACT_RECORD_BYTES - record.size());
    if (rich) {
        for (const int32_t token_id : observation.proposal_token_ids) {
            append_u32(record, static_cast<uint32_t>(token_id));
        }
        append_u32(record, static_cast<uint32_t>(observation.target_correction_or_bonus_id));
        append_u32(record, observation.cycle_sequence);
    }
    const uint32_t       flags = rich ? RICH_RECORD_FLAG : 0U;
    std::vector<uint8_t> frame;
    frame.reserve(CAPTURE_RECORD_HEADER_BYTES + record.size());
    append_u32(frame, CAPTURE_RECORD_FORMAT_VERSION);
    append_u32(frame, flags);
    append_u32(frame, static_cast<uint32_t>(record.size()));
    append_u32(frame, 0);
    frame.insert(frame.end(), record.begin(), record.end());
    payload.insert(payload.end(), frame.begin(), frame.end());
}

bool phase_cancelled(const capture_store_config & config, capture_store_phase phase) {
    return config.cancel_check && config.cancel_check(phase);
}

bool valid_config(const capture_store_config & config) {
    if (config.root_path.empty() || config.ring_capacity == 0 || config.max_shard_records == 0 ||
        config.max_shard_bytes < CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES + CAPTURE_RECORD_HEADER_BYTES +
                                     CAPTURE_COMPACT_RECORD_BYTES ||
        config.max_retained_shards == 0 || config.max_retained_records == 0 || config.max_retained_bytes == 0 ||
        config.max_manifest_bytes <
            CAPTURE_MANIFEST_HEADER_BYTES + MANIFEST_ENTRY_BYTES + CAPTURE_MANIFEST_FOOTER_BYTES ||
        config.max_manifest_bytes > CAPTURE_MAX_MANIFEST_BYTES || config.max_shard_records > UINT32_MAX ||
        config.rich_sample_every == 0 || !valid_capture_mode_value(static_cast<uint32_t>(config.capture_mode))) {
        return false;
    }
    if (config.ring_capacity > CAPTURE_MAX_RING_CAPACITY || config.max_shard_bytes > CAPTURE_MAX_SHARD_BYTES ||
        config.max_retained_bytes > CAPTURE_MAX_SHARD_BYTES * 64U ||
        config.max_shard_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        config.max_manifest_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }
    const uint64_t max_manifest_entries =
        (config.max_manifest_bytes - CAPTURE_MANIFEST_HEADER_BYTES - CAPTURE_MANIFEST_FOOTER_BYTES) /
        MANIFEST_ENTRY_BYTES;
    if (config.max_retained_shards > max_manifest_entries ||
        config.max_retained_records > CAPTURE_MAX_RETAINED_RECORDS) {
        return false;
    }
    if (config.capture_mode == mode::sampled_rich && !config.allow_sampled_rich) {
        return false;
    }
    if (config.max_retained_bytes < config.max_shard_bytes || config.max_retained_records < config.max_shard_records) {
        return false;
    }
    if (config.capture_mode == mode::sampled_rich &&
        config.max_shard_bytes < CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES + CAPTURE_RECORD_HEADER_BYTES +
                                     CAPTURE_RICH_RECORD_BYTES) {
        return false;
    }
    const uint64_t max_records_by_bytes =
        (config.max_shard_bytes - CAPTURE_SHARD_HEADER_BYTES - CAPTURE_SHARD_FOOTER_BYTES) /
        (CAPTURE_RECORD_HEADER_BYTES + CAPTURE_COMPACT_RECORD_BYTES);
    const uint64_t record_bytes =
        config.capture_mode == mode::sampled_rich ? CAPTURE_RICH_RECORD_BYTES : CAPTURE_COMPACT_RECORD_BYTES;
    const uint64_t max_records_for_mode =
        (config.max_shard_bytes - CAPTURE_SHARD_HEADER_BYTES - CAPTURE_SHARD_FOOTER_BYTES) /
        (CAPTURE_RECORD_HEADER_BYTES + record_bytes);
    return max_records_by_bytes != 0 && max_records_for_mode != 0 && config.max_shard_records <= max_records_for_mode;
}

size_t checked_ring_capacity(const capture_store_config & config) {
    if (config.capture_mode == mode::off) {
        return 0;
    }
    if (!valid_config(config)) {
        throw std::invalid_argument("invalid capture store configuration");
    }
    return config.ring_capacity;
}

}  // namespace

const char * capture_store_status_name(capture_store_status status) noexcept {
    switch (status) {
        case capture_store_status::ok:
            return "ok";
        case capture_store_status::invalid_argument:
            return "invalid_argument";
        case capture_store_status::disabled:
            return "disabled";
        case capture_store_status::stopped:
            return "stopped";
        case capture_store_status::no_manifest:
            return "no_manifest";
        case capture_store_status::malformed_manifest:
            return "malformed_manifest";
        case capture_store_status::truncated:
            return "truncated";
        case capture_store_status::trailing_data:
            return "trailing_data";
        case capture_store_status::checksum_mismatch:
            return "checksum_mismatch";
        case capture_store_status::shard_missing:
            return "shard_missing";
        case capture_store_status::shard_corrupt:
            return "shard_corrupt";
        case capture_store_status::manifest_too_large:
            return "manifest_too_large";
        case capture_store_status::shard_too_large:
            return "shard_too_large";
        case capture_store_status::too_many_records:
            return "too_many_records";
        case capture_store_status::too_many_shards:
            return "too_many_shards";
        case capture_store_status::no_space:
            return "no_space";
        case capture_store_status::short_write:
            return "short_write";
        case capture_store_status::io_error:
            return "io_error";
        case capture_store_status::commit_uncertain:
            return "commit_uncertain";
        case capture_store_status::cancelled:
            return "cancelled";
        case capture_store_status::deletion_failed:
            return "deletion_failed";
        case capture_store_status::path_security:
            return "path_security";
    }
    return "unknown";
}

struct capture_store::impl {
    explicit impl(capture_store_config config) : cfg(std::move(config)) {
        if (cfg.capture_mode == mode::off) {
            startup = result(capture_store_status::disabled);
            return;
        }
        if (!valid_config(cfg)) {
            throw std::invalid_argument("invalid capture store configuration");
        }
        ring   = std::make_unique<spsc_ring>(checked_ring_capacity(cfg));
        wakeup = std::make_unique<capture_wakeup>();
        salt   = cfg.identity_salt;
        if (salt_zero(salt)) {
            int  random_error  = EIO;
            bool random_failed = cfg.faults.fail_csprng;
            if (cfg.faults.csprng_returns_zero) {
                random_failed = false;
            } else if (!random_failed && ::getentropy(salt.data(), salt.size()) != 0) {
                random_failed = true;
                random_error  = errno == 0 ? EIO : errno;
            }
            if (random_failed || salt_zero(salt)) {
                startup = result(capture_store_status::io_error, random_error);
                accepting.store(false, std::memory_order_release);
                failed.store(true, std::memory_order_release);
                return;
            }
        }
        manifest.capture_mode   = cfg.capture_mode;
        manifest.format_version = CAPTURE_STORE_FORMAT_VERSION;
        startup                 = recover();
        if (startup.status != capture_store_status::ok) {
            accepting.store(false, std::memory_order_release);
            failed.store(startup.status != capture_store_status::ok, std::memory_order_release);
            return;
        }
        accepting.store(true, std::memory_order_release);
        worker = std::thread(&impl::run, this);
        worker_started.store(true, std::memory_order_release);
    }

    ~impl() {
        if (worker.joinable()) {
            (void) shutdown(true);
        } else {
            accepting.store(false, std::memory_order_release);
            close_admission_and_wait();
        }
        if (root_fd >= 0) {
            (void) ::close(root_fd);
            root_fd = -1;
        }
    }

    fs::path root() const { return fs::path(cfg.root_path); }

    capture_store_result recover() {
        const capture_store_result root_status = open_private_root(root(), cfg.require_private_root, root_fd);
        if (root_status.status != capture_store_status::ok) {
            return root_status;
        }
        std::vector<uint8_t>       bytes;
        const capture_store_result read =
            read_file_at(root_fd, "capture.manifest", cfg.max_manifest_bytes, bytes, capture_store_status::no_manifest,
                         capture_store_status::manifest_too_large, cfg.faults.fail_fstat);
        if (read.status == capture_store_status::no_manifest) {
            manifest                = {};
            manifest.capture_mode   = cfg.capture_mode;
            manifest.format_version = CAPTURE_STORE_FORMAT_VERSION;
            manifest.identity_salt  = salt;
        } else if (read.status != capture_store_status::ok) {
            return read;
        } else {
            const capture_store_result parsed = parse_manifest(bytes, cfg, manifest, salt);
            if (parsed.status != capture_store_status::ok) {
                return parsed;
            }
            const capture_store_result valid = validate(manifest);
            if (valid.status != capture_store_status::ok) {
                return valid;
            }
        }
        // A manifest is the commit record.  Remove only unreferenced temp and
        // final shard names; malformed manifests never reach this point, so
        // recovery remains fail-closed.
        const int scan_descriptor = ::dup(root_fd);
        if (scan_descriptor < 0) {
            return errno_result(errno);
        }
        DIR * directory = ::fdopendir(scan_descriptor);
        if (directory == nullptr) {
            const int error = errno;
            ::close(scan_descriptor);
            return errno_result(error);
        }
        std::vector<std::string> entries;
        errno = 0;
        for (dirent * entry = ::readdir(directory); entry != nullptr; entry = ::readdir(directory)) {
            entries.emplace_back(entry->d_name);
        }
        const int scan_error = errno;
        if (::closedir(directory) != 0 && scan_error == 0) {
            return errno_result(errno);
        }
        if (scan_error != 0) {
            return errno_result(scan_error);
        }
        for (const std::string & name : entries) {
            struct stat entry_status = {};
            if (::fstatat(root_fd, name.c_str(), &entry_status, AT_SYMLINK_NOFOLLOW) != 0) {
                if (errno == ENOENT) {
                    continue;
                }
                return result(capture_store_status::path_security, errno);
            }
            const bool capture_prefix = name == ".capture.manifest.tmp" || name.rfind(".capture.", 0) == 0 ||
                                        name.rfind(".shard-", 0) == 0 || name.rfind("shard-", 0) == 0 ||
                                        name.rfind(".delete-", 0) == 0;
            if (!capture_prefix) {
                continue;
            }
            if (S_ISLNK(entry_status.st_mode) || !S_ISREG(entry_status.st_mode) || entry_status.st_uid != ::geteuid() ||
                (entry_status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
                return result(capture_store_status::path_security, EACCES);
            }
            uint64_t sequence = 0;
            if (name.rfind(".delete-", 0) == 0) {
                if (!parse_tombstone_sequence(name, sequence)) {
                    return result(capture_store_status::path_security, EINVAL);
                }
                const auto found =
                    std::find_if(manifest.shards.begin(), manifest.shards.end(),
                                 [sequence](const capture_shard_info & shard) { return shard.sequence == sequence; });
                if (found != manifest.shards.end()) {
                    return result(capture_store_status::path_security, EINVAL);
                }
                const capture_store_result remove_shard =
                    safe_remove_file_at(root_fd, shard_name(sequence, false), cfg.faults.fail_unlink);
                if (remove_shard.status != capture_store_status::ok) {
                    return remove_shard;
                }
                const capture_store_result remove_tombstone =
                    safe_remove_file_at(root_fd, name, cfg.faults.fail_unlink);
                if (remove_tombstone.status != capture_store_status::ok) {
                    return result(capture_store_status::deletion_failed, remove_tombstone.os_error);
                }
                continue;
            }
            uint64_t shard_sequence = 0;
            bool     temporary      = false;
            if (name == ".capture.manifest.tmp") {
                const capture_store_result removed = safe_remove_file_at(root_fd, name, cfg.faults.fail_unlink);
                if (removed.status != capture_store_status::ok) {
                    return removed;
                }
                continue;
            }
            if (!parse_shard_sequence(name, shard_sequence, temporary)) {
                return result(capture_store_status::path_security, EINVAL);
            }
            const auto found = std::find_if(
                manifest.shards.begin(), manifest.shards.end(),
                [shard_sequence](const capture_shard_info & shard) { return shard.sequence == shard_sequence; });
            if (temporary || found == manifest.shards.end()) {
                const capture_store_result removed = safe_remove_file_at(root_fd, name, cfg.faults.fail_unlink);
                if (removed.status != capture_store_status::ok) {
                    return removed;
                }
            }
        }
        return result(capture_store_status::ok);
    }

    capture_store_result validate(const capture_manifest & candidate) const {
        if (candidate.format_version != CAPTURE_STORE_FORMAT_VERSION || candidate.capture_mode != cfg.capture_mode ||
            candidate.shards.size() > cfg.max_retained_shards || candidate.total_records > cfg.max_retained_records ||
            candidate.total_bytes > cfg.max_retained_bytes) {
            return result(capture_store_status::malformed_manifest);
        }
        for (const capture_shard_info & shard : candidate.shards) {
            std::vector<uint8_t>       bytes;
            const capture_store_result read = read_file_at(
                root_fd, shard_name(shard.sequence, false), cfg.max_shard_bytes, bytes,
                capture_store_status::shard_missing, capture_store_status::shard_too_large, cfg.faults.fail_fstat);
            if (read.status != capture_store_status::ok) {
                return read;
            }
            const capture_store_result valid = validate_shard_bytes(bytes, shard, cfg, cfg.capture_mode);
            if (valid.status != capture_store_status::ok) {
                return valid;
            }
        }
        return result(capture_store_status::ok);
    }

    capture_store_result cleanup_after_failure(const std::string & temporary,
                                               const std::string & final,
                                               bool                preserve) {
        if (preserve) {
            return result(capture_store_status::commit_uncertain);
        }
        const capture_store_result remove_temporary = safe_remove_file_at(root_fd, temporary, cfg.faults.fail_unlink);
        const capture_store_result remove_final     = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
        const capture_store_result remove_manifest =
            safe_remove_file_at(root_fd, ".capture.manifest.tmp", cfg.faults.fail_unlink);
        if (remove_temporary.status != capture_store_status::ok || remove_final.status != capture_store_status::ok ||
            remove_manifest.status != capture_store_status::ok) {
            const capture_store_result & failure =
                remove_temporary.status != capture_store_status::ok ?
                    remove_temporary :
                    (remove_final.status != capture_store_status::ok ? remove_final : remove_manifest);
            return result(capture_store_status::deletion_failed, failure.os_error);
        }
        return result(capture_store_status::cancelled);
    }

    capture_store_result delete_retired(const std::vector<capture_shard_info> & retired) {
        for (const capture_shard_info & shard : retired) {
            const std::string tombstone        = tombstone_name(shard.sequence);
            const std::string final            = shard_name(shard.sequence, false);
            bool              still_referenced = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                still_referenced = std::find_if(manifest.shards.begin(), manifest.shards.end(),
                                                [sequence = shard.sequence](const capture_shard_info & current) {
                                                    return current.sequence == sequence;
                                                }) != manifest.shards.end();
            }
            if (still_referenced) {
                return result(capture_store_status::path_security, EINVAL);
            }
            const int descriptor = ::openat(root_fd, tombstone.c_str(),
                                            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, PRIVATE_FILE_MODE);
            if (descriptor < 0) {
                return result((errno == ELOOP || errno == EEXIST) ? capture_store_status::path_security :
                                                                    capture_store_status::deletion_failed,
                              errno);
            }
            struct stat tombstone_status = {};
            if (cfg.faults.fail_fstat || ::fstat(descriptor, &tombstone_status) != 0 ||
                !S_ISREG(tombstone_status.st_mode) || tombstone_status.st_uid != ::geteuid() ||
                (tombstone_status.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
                const int error = errno == 0 ? EACCES : errno;
                ::close(descriptor);
                return cfg.faults.fail_fstat ? result(capture_store_status::io_error, EIO) :
                                               result(capture_store_status::path_security, error);
            }
            const uint8_t marker  = 1;
            ssize_t       written = 0;
            do {
                written = ::write(descriptor, &marker, sizeof(marker));
            } while (written < 0 && errno == EINTR);
            const int  write_error         = written == 1 ? 0 : (written < 0 ? errno : EIO);
            const bool fail_tombstone_sync = cfg.faults.fail_file_fsync || cfg.faults.fail_tombstone_fsync;
            const int  sync_error          = write_error == 0 && (fail_tombstone_sync || ::fsync(descriptor) != 0) ?
                                                 (fail_tombstone_sync ? EIO : errno) :
                                                 0;
            const int  close_error         = ::close(descriptor) != 0 ? errno : 0;
            if (write_error != 0 || sync_error != 0 || close_error != 0) {
                const int error = write_error != 0 ? write_error : (sync_error != 0 ? sync_error : close_error);
                return result(capture_store_status::deletion_failed, error);
            }
            int directory_error = 0;
            if (!fsync_directory_fd(root_fd, directory_error, cfg.require_private_root, cfg.faults.fail_directory_fsync,
                                    cfg.faults.fail_fstat)) {
                return result(capture_store_status::deletion_failed, directory_error);
            }
            const capture_store_result remove_final = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
            if (remove_final.status != capture_store_status::ok) {
                return remove_final.status == capture_store_status::path_security ?
                           remove_final :
                           result(capture_store_status::deletion_failed, remove_final.os_error);
            }
            const capture_store_result remove_tombstone =
                safe_remove_file_at(root_fd, tombstone, cfg.faults.fail_unlink);
            if (remove_tombstone.status != capture_store_status::ok) {
                return remove_tombstone.status == capture_store_status::path_security ?
                           remove_tombstone :
                           result(capture_store_status::deletion_failed, remove_tombstone.os_error);
            }
            if (!fsync_directory_fd(root_fd, directory_error, cfg.require_private_root, cfg.faults.fail_directory_fsync,
                                    cfg.faults.fail_fstat)) {
                return result(capture_store_status::deletion_failed, directory_error);
            }
        }
        return result(capture_store_status::ok);
    }

    capture_store_result commit_payload() {
        if (current_records == 0) {
            return result(capture_store_status::ok, 0, true);
        }
        capture_manifest candidate;
        {
            std::lock_guard<std::mutex> lock(mutex);
            candidate = manifest;
        }
        const uint64_t sequence =
            candidate.shards.empty() ?
                1 :
                (candidate.shards.back().sequence == UINT64_MAX ? 0 : candidate.shards.back().sequence + 1);
        if (sequence == 0) {
            return result(capture_store_status::too_many_shards);
        }
        capture_digest             checksum    = {};
        const std::vector<uint8_t> shard_bytes = serialize_shard(
            current_payload, sequence, cfg.capture_mode, current_records, current_first_ns, current_last_ns, checksum);
        if (shard_bytes.size() > cfg.max_shard_bytes) {
            return result(capture_store_status::shard_too_large);
        }
        const std::string temporary = shard_name(sequence, true);
        const std::string final     = shard_name(sequence, false);
        if (phase_cancelled(cfg, capture_store_phase::before_shard_write)) {
            return cleanup_after_failure(temporary, final, false);
        }
        const capture_store_result write =
            write_file_at(root_fd, temporary, shard_bytes, cfg.faults, cfg.faults.preserve_failed_files);
        if (write.status != capture_store_status::ok) {
            if (cfg.faults.preserve_failed_files) {
                return write;
            }
            const capture_store_result removed = safe_remove_file_at(root_fd, temporary, cfg.faults.fail_unlink);
            if (removed.status != capture_store_status::ok) {
                return result(capture_store_status::deletion_failed, removed.os_error);
            }
            return write;
        }
        if (phase_cancelled(cfg, capture_store_phase::after_shard_write)) {
            return cleanup_after_failure(temporary, final, false);
        }
        if (cfg.faults.crash_before_shard_rename) {
            return cleanup_after_failure(temporary, final, true);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_shard_rename)) {
            return cleanup_after_failure(temporary, final, false);
        }
        const capture_store_result temporary_status = validate_private_file_at(root_fd, temporary);
        const capture_store_result final_status     = validate_private_file_at(root_fd, final, true);
        if (temporary_status.status != capture_store_status::ok || final_status.status != capture_store_status::ok) {
            return result(capture_store_status::path_security, temporary_status.status != capture_store_status::ok ?
                                                                   temporary_status.os_error :
                                                                   final_status.os_error);
        }
        struct stat final_check = {};
        if (::fstatat(root_fd, final.c_str(), &final_check, AT_SYMLINK_NOFOLLOW) == 0) {
            return result(capture_store_status::path_security, EEXIST);
        }
        if (errno != ENOENT) {
            return result(capture_store_status::path_security, errno);
        }
        if (cfg.faults.fail_shard_rename || ::renameat(root_fd, temporary.c_str(), root_fd, final.c_str()) != 0) {
            if (cfg.faults.fail_shard_rename) {
                errno = EIO;
            }
            const int                  error   = errno;
            const capture_store_result cleanup = cleanup_after_failure(temporary, final, false);
            return cleanup.status == capture_store_status::deletion_failed ? cleanup : errno_result(error);
        }
        if (cfg.faults.crash_after_shard_rename) {
            return result(capture_store_status::commit_uncertain);
        }
        if (phase_cancelled(cfg, capture_store_phase::after_shard_rename)) {
            return cleanup_after_failure(temporary, final, false);
        }
        int directory_error = 0;
        if (!fsync_directory_fd(root_fd, directory_error, cfg.require_private_root, cfg.faults.fail_directory_fsync,
                                cfg.faults.fail_fstat)) {
            return errno_result(directory_error);
        }

        candidate.generation   = candidate.generation == UINT64_MAX ? 1 : candidate.generation + 1;
        candidate.capture_mode = cfg.capture_mode;
        capture_shard_info info;
        info.sequence           = sequence;
        info.first_monotonic_ns = current_first_ns;
        info.last_monotonic_ns  = current_last_ns;
        info.record_count       = current_records;
        info.byte_count         = static_cast<uint64_t>(shard_bytes.size());
        info.checksum           = checksum;
        candidate.shards.push_back(info);
        if (candidate.total_records > UINT64_MAX - info.record_count ||
            candidate.total_bytes > UINT64_MAX - info.byte_count) {
            const capture_store_result removed = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
            if (removed.status != capture_store_status::ok) {
                return result(capture_store_status::deletion_failed, removed.os_error);
            }
            return result(capture_store_status::too_many_records);
        }
        candidate.total_records += info.record_count;
        candidate.total_bytes += info.byte_count;
        std::vector<capture_shard_info> retired;
        const uint64_t                  newest = current_last_ns;
        while (candidate.shards.size() > cfg.max_retained_shards ||
               candidate.total_records > cfg.max_retained_records || candidate.total_bytes > cfg.max_retained_bytes ||
               (cfg.max_retained_age_ns != 0 && !candidate.shards.empty() &&
                newest > candidate.shards.front().last_monotonic_ns &&
                newest - candidate.shards.front().last_monotonic_ns > cfg.max_retained_age_ns)) {
            if (candidate.shards.empty()) {
                break;
            }
            const capture_shard_info old = candidate.shards.front();
            candidate.shards.erase(candidate.shards.begin());
            candidate.total_records -= old.record_count;
            candidate.total_bytes -= old.byte_count;
            retired.push_back(old);
        }
        if (candidate.shards.empty()) {
            const capture_store_result removed = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
            if (removed.status != capture_store_status::ok) {
                return result(capture_store_status::deletion_failed, removed.os_error);
            }
            return result(capture_store_status::invalid_argument);
        }
        const std::array<uint8_t, 16> manifest_salt =
            cfg.faults.force_zero_manifest_salt ? std::array<uint8_t, 16>{} : salt;
        const std::vector<uint8_t> manifest_bytes = serialize_manifest(candidate, manifest_salt);
        if (manifest_bytes.size() > cfg.max_manifest_bytes) {
            const capture_store_result removed = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
            if (removed.status != capture_store_status::ok) {
                return result(capture_store_status::deletion_failed, removed.os_error);
            }
            return result(capture_store_status::manifest_too_large);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_manifest_write)) {
            return cleanup_after_failure(temporary, final, false);
        }
        const capture_store_result manifest_write = write_file_at(root_fd, ".capture.manifest.tmp", manifest_bytes,
                                                                  cfg.faults, cfg.faults.preserve_failed_files);
        if (manifest_write.status != capture_store_status::ok) {
            if (!cfg.faults.preserve_failed_files) {
                const capture_store_result remove_final = safe_remove_file_at(root_fd, final, cfg.faults.fail_unlink);
                const capture_store_result remove_manifest =
                    safe_remove_file_at(root_fd, ".capture.manifest.tmp", cfg.faults.fail_unlink);
                if (remove_final.status != capture_store_status::ok ||
                    remove_manifest.status != capture_store_status::ok) {
                    const capture_store_result & failure =
                        remove_final.status != capture_store_status::ok ? remove_final : remove_manifest;
                    return result(capture_store_status::deletion_failed, failure.os_error);
                }
            }
            return manifest_write;
        }
        if (phase_cancelled(cfg, capture_store_phase::after_manifest_write)) {
            return cleanup_after_failure(".capture.manifest.tmp", final, false);
        }
        if (cfg.faults.crash_before_manifest_rename) {
            return cleanup_after_failure(".capture.manifest.tmp", final, true);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_manifest_rename)) {
            return cleanup_after_failure(".capture.manifest.tmp", final, false);
        }
        const capture_store_result manifest_tmp_status = validate_private_file_at(root_fd, ".capture.manifest.tmp");
        const capture_store_result manifest_status     = validate_private_file_at(root_fd, "capture.manifest", true);
        if (manifest_tmp_status.status != capture_store_status::ok ||
            manifest_status.status != capture_store_status::ok) {
            return result(capture_store_status::path_security, manifest_tmp_status.status != capture_store_status::ok ?
                                                                   manifest_tmp_status.os_error :
                                                                   manifest_status.os_error);
        }
        if (cfg.faults.fail_manifest_rename ||
            ::renameat(root_fd, ".capture.manifest.tmp", root_fd, "capture.manifest") != 0) {
            if (cfg.faults.fail_manifest_rename) {
                errno = EIO;
            }
            const int                  error   = errno;
            const capture_store_result cleanup = cleanup_after_failure(".capture.manifest.tmp", final, false);
            return cleanup.status == capture_store_status::deletion_failed ? cleanup : errno_result(error);
        }
        // Manifest rename is the durable publication boundary. Every
        // operation after it is wrapped so callback/allocation/unknown
        // failures remain non-ok and committed=true; the worker must not
        // discard records that are already represented by this manifest.
        try {
            {
                std::lock_guard<std::mutex> lock(mutex);
                manifest = std::move(candidate);
            }
            if (cfg.faults.crash_after_manifest_rename) {
                return result(capture_store_status::commit_uncertain, 0, true);
            }
            // Cancellation after publication is too late to revoke a durable
            // commit; returning success is the only truthful outcome.
            (void) phase_cancelled(cfg, capture_store_phase::after_manifest_rename);
            if (!fsync_directory_fd(root_fd, directory_error, cfg.require_private_root,
                                    cfg.faults.fail_directory_fsync || cfg.faults.fail_post_publication_directory_fsync,
                                    cfg.faults.fail_fstat)) {
                return result(capture_store_status::commit_uncertain, directory_error, true);
            }
            // Cancellation after publication cannot revoke the manifest. Invoke
            // the seam for observability, then finish deletion so cleanup
            // authority is durable and no retired shard leaks until restart.
            (void) phase_cancelled(cfg, capture_store_phase::before_retention_delete);
            const capture_store_result deleted = delete_retired(retired);
            if (deleted.status != capture_store_status::ok) {
                return result(deleted.status, deleted.os_error, true);
            }
            (void) phase_cancelled(cfg, capture_store_phase::after_retention_delete);
            return result(capture_store_status::ok, 0, true);
        } catch (const std::bad_alloc &) {
            return result(capture_store_status::commit_uncertain, ENOMEM, true);
        } catch (...) {
            return result(capture_store_status::commit_uncertain, EFAULT, true);
        }
    }

    void reset_current() {
        current_payload.clear();
        current_records  = 0;
        current_first_ns = 0;
        current_last_ns  = 0;
    }

    // Called only by the worker after producer admission is closed and all
    // claimed producer tokens have returned.  No producer can append after
    // this drain begins, so every pre-close successful push is accounted.
    void discard_pending(bool current_persisted = false) noexcept {
        if (worker_observation_in_flight && !worker_observation_recorded) {
            // The worker popped this observation but failed before appending
            // it to the current shard.  Count it alongside queued records so
            // a terminal worker failure cannot strand a successful push.
            dropped_on_shutdown.fetch_add(1, std::memory_order_relaxed);
        }
        worker_observation_in_flight = false;
        worker_observation_recorded  = false;
        if (ring != nullptr) {
            cycle_observation observation;
            while (ring->try_pop(observation)) {
                dropped_on_shutdown.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (!current_persisted) {
            dropped_on_shutdown.fetch_add(current_records, std::memory_order_relaxed);
        }
        reset_current();
    }

    void mark_failure(capture_store_result failure) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            terminal     = failure;
            flush_result = failure;
            ++failed_writes;
        }
        // The worker remains alive while admission closes and waits.  This is
        // deliberately outside mutex: a producer callback may need another
        // lifecycle thread to release it, and waiting while holding mutex
        // would deadlock flush/shutdown coordination.
        close_admission_and_wait();
        failed.store(true, std::memory_order_release);
        accepting.store(false, std::memory_order_release);
        stop_requested.store(true, std::memory_order_release);
        discard_pending(failure.committed);
        {
            std::lock_guard<std::mutex> lock(mutex);
            flush_result    = failure;
            flush_completed = flush_requested;
        }
        if (wakeup != nullptr) {
            (void) wakeup->post();
        }
        cv.notify_all();
    }

    capture_store_result consume_observation(const cycle_observation & observation) {
        const bool rich = cfg.capture_mode == mode::sampled_rich && cfg.allow_sampled_rich &&
                          observation.cycle_sequence % cfg.rich_sample_every == 0;
        const size_t before = current_payload.size();
        const size_t add =
            CAPTURE_RECORD_HEADER_BYTES + (rich ? CAPTURE_RICH_RECORD_BYTES : CAPTURE_COMPACT_RECORD_BYTES);
        const size_t payload_limit =
            static_cast<size_t>(cfg.max_shard_bytes) - CAPTURE_SHARD_HEADER_BYTES - CAPTURE_SHARD_FOOTER_BYTES;
        if (before > payload_limit || add > payload_limit - before) {
            if (current_records != 0) {
                const capture_store_result committed = commit_payload();
                if (committed.status != capture_store_status::ok) {
                    return committed;
                }
                reset_current();
            }
        }
        if (current_records == 0) {
            current_first_ns = observation.monotonic_ns;
        }
        current_last_ns = observation.monotonic_ns;
        append_record(observation, salt, rich, cfg.capture_mode == mode::metrics_only, current_payload);
        ++current_records;
        worker_observation_recorded = true;
        if (current_records >= cfg.max_shard_records ||
            current_payload.size() + CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES >= cfg.max_shard_bytes) {
            const capture_store_result committed = commit_payload();
            if (committed.status != capture_store_status::ok) {
                return committed;
            }
            reset_current();
        }
        return result(capture_store_status::ok);
    }

    void run() noexcept {
        worker_running.store(true, std::memory_order_release);
        try {
            current_payload.reserve(static_cast<size_t>(cfg.max_shard_bytes));
            bool exit = false;
            while (!exit) {
                cycle_observation observation;
                while (ring->try_pop(observation)) {
                    worker_observation_in_flight        = true;
                    worker_observation_recorded         = false;
                    const capture_store_result consumed = consume_observation(observation);
                    if (consumed.status != capture_store_status::ok) {
                        mark_failure(consumed);
                        exit = true;
                        break;
                    }
                    worker_observation_in_flight = false;
                    if (cfg.faults.slow_worker) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                if (exit) {
                    break;
                }
                bool should_stop  = false;
                bool should_flush = false;
                bool should_drain = true;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    should_stop  = stop_requested.load(std::memory_order_acquire);
                    should_flush = flush_requested != flush_completed;
                    should_drain = drain_requested.load(std::memory_order_acquire);
                }
                if (should_stop && !should_drain) {
                    discard_pending();
                    const capture_store_result stopped_result = result(capture_store_status::stopped);
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (should_flush) {
                            flush_result    = stopped_result;
                            flush_completed = flush_requested;
                        }
                        stopped  = true;
                        terminal = stopped_result;
                    }
                    cv.notify_all();
                    exit = true;
                    break;
                }
                if (ring->stats().size_approx != 0) {
                    continue;
                }
                if (should_flush || should_stop) {
                    capture_store_result committed = result(capture_store_status::ok, 0, true);
                    if (current_records != 0 && (should_drain || should_flush)) {
                        committed = commit_payload();
                        reset_current();
                    }
                    if (committed.status != capture_store_status::ok) {
                        mark_failure(committed);
                        exit = true;
                        break;
                    }
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (should_flush) {
                            flush_result    = committed;
                            flush_completed = flush_requested;
                        }
                        if (should_stop) {
                            accepting.store(false, std::memory_order_release);
                            stopped  = true;
                            terminal = committed;
                        }
                    }
                    cv.notify_all();
                    if (should_stop) {
                        exit = true;
                        break;
                    }
                    continue;
                }

                // Snapshot the wake epoch before the second queue/control check.
                // A producer that races either check increments the epoch after
                // this snapshot, so wait() cannot lose its notification.
                uint64_t observed = wakeup->snapshot();
                if (ring->stats().size_approx != 0) {
                    continue;
                }
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (stop_requested.load(std::memory_order_acquire) || flush_requested != flush_completed) {
                        continue;
                    }
                }
                if (cfg.faults.before_worker_wait) {
                    cfg.faults.before_worker_wait();
                }
                wakeup->wait(observed);
            }
        } catch (const std::bad_alloc &) {
            mark_failure(result(capture_store_status::io_error, ENOMEM));
        } catch (const std::exception &) {
            mark_failure(result(capture_store_status::io_error, EFAULT));
        } catch (...) {
            mark_failure(result(capture_store_status::io_error, EFAULT));
        }
        worker_running.store(false, std::memory_order_release);
        accepting.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopped = true;
        }
        cv.notify_all();
    }

    capture_store_result flush() {
        if (cfg.capture_mode == mode::off) {
            return result(capture_store_status::disabled);
        }
        std::unique_lock<std::mutex> lock(mutex);
        if (!worker_started.load(std::memory_order_acquire)) {
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        if (shutdown_started) {
            cv.wait(lock, [this]() { return stopped; });
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        if (stopped || failed.load(std::memory_order_acquire)) {
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        ++flush_requested;
        const uint64_t request = flush_requested;
        lock.unlock();
        if (cfg.faults.after_flush_registration) {
            cfg.faults.after_flush_registration();
        }
        (void) wakeup->post();
        lock.lock();
        cv.wait(lock, [this, request]() {
            return flush_completed >= request || stopped || failed.load(std::memory_order_acquire);
        });
        return flush_result;
    }

    capture_store_result shutdown(bool drain) {
        if (cfg.capture_mode == mode::off) {
            accepting.store(false, std::memory_order_release);
            close_admission_and_wait();
            return result(capture_store_status::disabled);
        }
        std::unique_lock<std::mutex> guard(mutex);
        if (shutdown_started) {
            cv.wait(guard, [this]() { return stopped; });
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        shutdown_started = true;
        if (!worker_started.load(std::memory_order_acquire)) {
            close_admission_and_wait();
            stopped = true;
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        guard.unlock();
        // Seal admission while the worker remains alive.  Only after every
        // pre-close producer has returned may stop/drain state be published;
        // this guarantees that a successful push cannot be stranded behind
        // the worker's final queue check.
        close_admission_and_wait();
        {
            std::lock_guard<std::mutex> lock(mutex);
            drain_requested.store(drain, std::memory_order_release);
            stop_requested.store(true, std::memory_order_release);
            accepting.store(false, std::memory_order_release);
        }
        (void) wakeup->post();
        worker.join();
        worker_started.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex);
        if (terminal.status == capture_store_status::invalid_argument) {
            terminal = result(capture_store_status::stopped);
        }
        stopped = true;
        cv.notify_all();
        return terminal;
    }

    capture_store_config            cfg;
    std::unique_ptr<spsc_ring>      ring;
    std::unique_ptr<capture_wakeup> wakeup;
    int                             root_fd = -1;
    std::array<uint8_t, 16>         salt    = {};
    capture_manifest                manifest;
    capture_store_result            startup      = result(capture_store_status::invalid_argument);
    capture_store_result            terminal     = result(capture_store_status::invalid_argument);
    capture_store_result            flush_result = result(capture_store_status::invalid_argument);
    std::thread                     worker;
    mutable std::mutex              mutex;
    std::condition_variable         cv;
    std::atomic<bool>               accepting{ false };
    std::atomic<bool>               stop_requested{ false };
    std::atomic<bool>               drain_requested{ true };
    std::atomic<bool>               failed{ false };
    std::atomic<bool>               worker_running{ false };
    std::atomic<bool>               worker_started{ false };
    std::atomic<uint64_t>           admission_state{ 0 };
    bool                            shutdown_started = false;
    uint64_t                        flush_requested  = 0;
    uint64_t                        flush_completed  = 0;
    bool                            stopped          = false;
    std::vector<uint8_t>            current_payload;
    uint32_t                        current_records              = 0;
    uint64_t                        current_first_ns             = 0;
    uint64_t                        current_last_ns              = 0;
    uint64_t                        failed_writes                = 0;
    bool                            worker_observation_in_flight = false;
    bool                            worker_observation_recorded  = false;
    std::atomic<uint64_t>           dropped_after_stop{ 0 };
    std::atomic<uint64_t>           dropped_on_shutdown{ 0 };

    bool claim_producer() noexcept {
        uint64_t observed = admission_state.load(std::memory_order_acquire);
        for (;;) {
            if ((observed & ADMISSION_CLOSED) != 0 || (observed & ADMISSION_COUNT_MASK) == ADMISSION_COUNT_MASK) {
                return false;
            }
            if (admission_state.compare_exchange_weak(observed, observed + 1, std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
                return true;
            }
        }
    }

    void release_producer() noexcept { admission_state.fetch_sub(1, std::memory_order_release); }

    void close_admission_and_wait() noexcept {
        admission_state.fetch_or(ADMISSION_CLOSED, std::memory_order_acq_rel);
        while ((admission_state.load(std::memory_order_acquire) & ADMISSION_COUNT_MASK) != 0) {
            std::this_thread::yield();
        }
    }
};

capture_store::capture_store(capture_store_config config) : data(std::make_unique<impl>(std::move(config))) {}

capture_store::~capture_store() = default;

bool capture_store::try_enqueue(const cycle_observation & observation) noexcept {
    if (!data->claim_producer()) {
        data->dropped_after_stop.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (data->cfg.faults.before_enqueue_accept) {
        try {
            data->cfg.faults.before_enqueue_accept();
        } catch (...) {
            data->dropped_after_stop.fetch_add(1, std::memory_order_relaxed);
            data->release_producer();
            return false;
        }
    }
    if (data->ring == nullptr || !data->accepting.load(std::memory_order_acquire)) {
        data->dropped_after_stop.fetch_add(1, std::memory_order_relaxed);
        data->release_producer();
        return false;
    }
    if (data->cfg.faults.before_enqueue_push) {
        try {
            data->cfg.faults.before_enqueue_push();
        } catch (...) {
            data->dropped_after_stop.fetch_add(1, std::memory_order_relaxed);
            data->release_producer();
            return false;
        }
    }
    const bool pushed = data->ring->try_push(observation);
    if (pushed) {
        if (data->wakeup == nullptr || !data->wakeup->post()) {
            data->failed.store(true, std::memory_order_release);
        }
    }
    data->release_producer();
    return pushed;
}

capture_store_result capture_store::flush() {
    return data->flush();
}

capture_store_result capture_store::shutdown(bool drain) {
    return data->shutdown(drain);
}

capture_store_result capture_store::inspect(capture_manifest & output) const {
    if (data->cfg.capture_mode == mode::off) {
        return result(capture_store_status::disabled);
    }
    if (data->root_fd < 0) {
        return data->startup;
    }
    std::vector<uint8_t>       bytes;
    const capture_store_result read = read_file_at(
        data->root_fd, "capture.manifest", data->cfg.max_manifest_bytes, bytes, capture_store_status::no_manifest,
        capture_store_status::manifest_too_large, data->cfg.faults.fail_fstat);
    if (read.status != capture_store_status::ok) {
        return read;
    }
    std::array<uint8_t, 16> ignored_salt = {};
    return parse_manifest(bytes, data->cfg, output, ignored_salt);
}

capture_store_result capture_store::validate(const capture_manifest & manifest) const {
    if (data->root_fd < 0) {
        return data->startup;
    }
    return data->validate(manifest);
}

capture_store_stats capture_store::stats() const noexcept {
    capture_store_stats output;
    output.ring                = data->ring == nullptr ? ring_stats{} : data->ring->stats();
    output.worker_running      = data->worker_running.load(std::memory_order_acquire);
    output.worker_failed       = data->failed.load(std::memory_order_acquire);
    output.dropped_after_stop  = data->dropped_after_stop.load(std::memory_order_relaxed);
    output.dropped_on_shutdown = data->dropped_on_shutdown.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(data->mutex);
    output.committed_shards  = static_cast<uint64_t>(data->manifest.shards.size());
    output.committed_records = data->manifest.total_records;
    output.committed_bytes   = data->manifest.total_bytes;
    output.failed_writes     = data->failed_writes;
    output.cancelled_writes  = data->terminal.status == capture_store_status::cancelled ? 1 : 0;
    return output;
}

const capture_store_config & capture_store::config() const noexcept {
    return data->cfg;
}

}  // namespace server_capture

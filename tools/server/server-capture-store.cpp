#include "server-capture-store.h"

#include <fcntl.h>
#include <sys/stat.h>
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
#include <random>
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
    temporary                  = prefix[0] == '.';
    const size_t suffix_length = temporary ? 4 : 4;
    if (name.size() <= prefix.size() + suffix_length || name.substr(name.size() - 4) != (temporary ? ".tmp" : ".cap")) {
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
    sequence = parsed;
    return true;
}

capture_store_result result(capture_store_status status, int os_error = 0, bool committed = false) {
    return { status, os_error, committed };
}

capture_store_result errno_result(int error) {
    return result(error == ENOSPC ? capture_store_status::no_space : capture_store_status::io_error, error);
}

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

capture_store_result write_file(const fs::path &             path,
                                const std::vector<uint8_t> & bytes,
                                const capture_store_faults & faults,
                                bool                         preserve_on_failure) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (descriptor < 0) {
        return errno_result(errno);
    }
    uint64_t             written = 0;
    capture_store_result status  = write_all(descriptor, bytes.data(), bytes.size(), faults, written);
    if (status.status == capture_store_status::ok && ::fsync(descriptor) != 0) {
        status = errno_result(errno);
    }
    const int close_error = ::close(descriptor) == 0 ? 0 : errno;
    if (status.status == capture_store_status::ok && close_error != 0) {
        status = errno_result(close_error);
    }
    if (status.status != capture_store_status::ok && !preserve_on_failure) {
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    return status;
}

capture_store_result read_file(const fs::path &       path,
                               uint64_t               max_bytes,
                               std::vector<uint8_t> & bytes,
                               capture_store_status   missing_status,
                               capture_store_status   too_large_status) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        return result(errno == ENOENT ? missing_status : capture_store_status::io_error, errno);
    }
    struct stat file_status = {};
    if (::fstat(descriptor, &file_status) != 0) {
        const int error = errno;
        ::close(descriptor);
        return result(capture_store_status::io_error, error);
    }
    if (!S_ISREG(file_status.st_mode) || file_status.st_size < 0) {
        ::close(descriptor);
        return result(capture_store_status::malformed_manifest);
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
    if (::close(descriptor) != 0) {
        return errno_result(errno);
    }
    return result(capture_store_status::ok);
}

bool fsync_directory(const fs::path & path, int & error) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        error = errno;
        return false;
    }
    const bool success = ::fsync(descriptor) == 0;
    error              = success ? 0 : errno;
    if (::close(descriptor) != 0 && success) {
        error = errno;
        return false;
    }
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
    if (reserved != 0 || flags != MANIFEST_FLAG_REDACTED_IDENTITY || generation == 0 ||
        shard_count > config.max_retained_shards || mode_value > static_cast<uint32_t>(mode::sampled_rich) ||
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
        mode_value != static_cast<uint32_t>(capture_mode) || sequence != expected.sequence ||
        record_count != expected.record_count || first_ns != expected.first_monotonic_ns ||
        last_ns != expected.last_monotonic_ns) {
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
        config.max_manifest_bytes > 64U * 1024U * 1024U || config.max_shard_records > UINT32_MAX ||
        config.rich_sample_every == 0) {
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
    }
    return "unknown";
}

struct capture_store::impl {
    explicit impl(capture_store_config config) : cfg(std::move(config)), ring(cfg.ring_capacity) {
        if (!valid_config(cfg)) {
            throw std::invalid_argument("invalid capture store configuration");
        }
        salt = cfg.identity_salt;
        if (salt_zero(salt)) {
            bool random_salt = true;
            try {
                std::random_device random;
                for (size_t index = 0; index < salt.size(); ++index) {
                    salt[index] = static_cast<uint8_t>(random() & 0xffU);
                }
            } catch (...) {
                random_salt = false;
            }
            if (random_salt) {
                // random_device is preferred; the deterministic fallback
                // below keeps construction available on restricted hosts.
            } else {
                std::array<uint8_t, 32> entropy       = {};
                const auto              now           = std::chrono::steady_clock::now().time_since_epoch().count();
                const uint64_t          clock_value   = static_cast<uint64_t>(now);
                const uint64_t          address_value = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(this));
                const uint64_t          pid_value     = static_cast<uint64_t>(::getpid());
                for (unsigned shift = 0; shift < 64; shift += 8) {
                    entropy[shift / 8]      = static_cast<uint8_t>(clock_value >> shift);
                    entropy[8 + shift / 8]  = static_cast<uint8_t>(address_value >> shift);
                    entropy[16 + shift / 8] = static_cast<uint8_t>(pid_value >> shift);
                }
                const capture_digest generated = sha256(entropy.data(), entropy.size());
                std::copy_n(generated.begin(), salt.size(), salt.begin());
            }
        }
        manifest.capture_mode   = cfg.capture_mode;
        manifest.format_version = CAPTURE_STORE_FORMAT_VERSION;
        startup                 = recover();
        if (startup.status != capture_store_status::ok || cfg.capture_mode == mode::off) {
            accepting.store(false, std::memory_order_release);
            failed.store(startup.status != capture_store_status::ok, std::memory_order_release);
            return;
        }
        accepting.store(true, std::memory_order_release);
        worker = std::thread(&impl::run, this);
    }

    ~impl() {
        if (worker.joinable()) {
            (void) shutdown(true);
        }
    }

    fs::path root() const { return fs::path(cfg.root_path); }

    fs::path manifest_path() const { return root() / "capture.manifest"; }

    fs::path manifest_tmp_path() const { return root() / ".capture.manifest.tmp"; }

    capture_store_result recover() {
        std::error_code error;
        if (!fs::exists(root(), error)) {
            if (!fs::create_directories(root(), error) && error) {
                return errno_result(error.value());
            }
        }
        if (error || !fs::is_directory(root(), error) || error) {
            return result(capture_store_status::io_error, error ? error.value() : ENOTDIR);
        }
        std::vector<uint8_t>       bytes;
        const capture_store_result read =
            read_file(manifest_path(), cfg.max_manifest_bytes, bytes, capture_store_status::no_manifest,
                      capture_store_status::manifest_too_large);
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
        std::vector<uint64_t> tombstones;
        for (const fs::directory_entry & entry : fs::directory_iterator(root(), error)) {
            if (error) {
                return errno_result(error.value());
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind(".delete-", 0) == 0 && name.size() > 12 && name.substr(name.size() - 5) == ".tomb") {
                uint64_t sequence = 0;
                for (size_t index = 8; index + 5 < name.size(); ++index) {
                    if (name[index] < '0' || name[index] > '9') {
                        sequence = 0;
                        break;
                    }
                    sequence = sequence * 10U + static_cast<uint64_t>(name[index] - '0');
                }
                if (sequence != 0) {
                    tombstones.push_back(sequence);
                }
            }
        }
        for (const uint64_t sequence : tombstones) {
            std::error_code remove_error;
            fs::remove(root() / shard_name(sequence, false), remove_error);
            if (remove_error) {
                return result(capture_store_status::deletion_failed, remove_error.value());
            }
            fs::remove(root() / tombstone_name(sequence), remove_error);
            if (remove_error) {
                return result(capture_store_status::deletion_failed, remove_error.value());
            }
        }
        for (const fs::directory_entry & entry : fs::directory_iterator(root(), error)) {
            if (error) {
                return errno_result(error.value());
            }
            const std::string name      = entry.path().filename().string();
            uint64_t          sequence  = 0;
            bool              temporary = false;
            if (name == ".capture.manifest.tmp" || (parse_shard_sequence(name, sequence, temporary) && temporary)) {
                std::error_code ignored;
                fs::remove(entry.path(), ignored);
                continue;
            }
            if (!parse_shard_sequence(name, sequence, temporary) || temporary) {
                continue;
            }
            const auto found =
                std::find_if(manifest.shards.begin(), manifest.shards.end(),
                             [sequence](const capture_shard_info & shard) { return shard.sequence == sequence; });
            if (found == manifest.shards.end()) {
                std::error_code ignored;
                fs::remove(entry.path(), ignored);
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
            const capture_store_result read =
                read_file(root() / shard_name(shard.sequence, false), cfg.max_shard_bytes, bytes,
                          capture_store_status::shard_missing, capture_store_status::shard_too_large);
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

    uint64_t next_sequence() const {
        return manifest.shards.empty() ?
                   1 :
                   (manifest.shards.back().sequence == UINT64_MAX ? 0 : manifest.shards.back().sequence + 1);
    }

    capture_store_result cleanup_after_failure(const fs::path & temporary, const fs::path & final, bool preserve) {
        if (preserve) {
            return result(capture_store_status::commit_uncertain);
        }
        std::error_code ignored;
        fs::remove(temporary, ignored);
        fs::remove(final, ignored);
        fs::remove(manifest_tmp_path(), ignored);
        return result(capture_store_status::cancelled);
    }

    capture_store_result delete_retired(const std::vector<capture_shard_info> & retired) {
        for (const capture_shard_info & shard : retired) {
            const fs::path tombstone  = root() / tombstone_name(shard.sequence);
            const fs::path final      = root() / shard_name(shard.sequence, false);
            const int      descriptor = ::open(tombstone.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (descriptor < 0) {
                return result(capture_store_status::deletion_failed, errno);
            }
            const uint8_t marker = 1;
            if (::write(descriptor, &marker, sizeof(marker)) != static_cast<ssize_t>(sizeof(marker)) ||
                ::fsync(descriptor) != 0 || ::close(descriptor) != 0) {
                const int error = errno;
                ::close(descriptor);
                return result(capture_store_status::deletion_failed, error);
            }
            std::error_code remove_error;
            fs::remove(final, remove_error);
            if (remove_error) {
                return result(capture_store_status::deletion_failed, remove_error.value());
            }
            fs::remove(tombstone, remove_error);
            if (remove_error) {
                return result(capture_store_status::deletion_failed, remove_error.value());
            }
        }
        int error = 0;
        if (!fsync_directory(root(), error)) {
            return errno_result(error);
        }
        return result(capture_store_status::ok);
    }

    capture_store_result commit_payload() {
        if (current_records == 0) {
            return result(capture_store_status::ok, 0, true);
        }
        const uint64_t sequence = next_sequence();
        if (sequence == 0) {
            return result(capture_store_status::too_many_shards);
        }
        capture_digest             checksum    = {};
        const std::vector<uint8_t> shard_bytes = serialize_shard(
            current_payload, sequence, cfg.capture_mode, current_records, current_first_ns, current_last_ns, checksum);
        if (shard_bytes.size() > cfg.max_shard_bytes) {
            return result(capture_store_status::shard_too_large);
        }
        const fs::path temporary = root() / shard_name(sequence, true);
        const fs::path final     = root() / shard_name(sequence, false);
        if (phase_cancelled(cfg, capture_store_phase::before_shard_write)) {
            return cleanup_after_failure(temporary, final, false);
        }
        const capture_store_result write =
            write_file(temporary, shard_bytes, cfg.faults, cfg.faults.preserve_failed_files);
        if (write.status != capture_store_status::ok) {
            if (cfg.faults.preserve_failed_files) {
                return write;
            }
            std::error_code ignored;
            fs::remove(temporary, ignored);
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
        if (::rename(temporary.c_str(), final.c_str()) != 0) {
            return errno_result(errno);
        }
        if (cfg.faults.crash_after_shard_rename) {
            return result(capture_store_status::commit_uncertain);
        }
        if (phase_cancelled(cfg, capture_store_phase::after_shard_rename)) {
            return cleanup_after_failure(temporary, final, false);
        }
        int directory_error = 0;
        if (!fsync_directory(root(), directory_error)) {
            return errno_result(directory_error);
        }

        capture_manifest candidate = manifest;
        candidate.generation       = manifest.generation == UINT64_MAX ? 1 : manifest.generation + 1;
        candidate.capture_mode     = cfg.capture_mode;
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
            std::error_code ignored;
            fs::remove(final, ignored);
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
            std::error_code ignored;
            fs::remove(final, ignored);
            return result(capture_store_status::invalid_argument);
        }
        const std::vector<uint8_t> manifest_bytes = serialize_manifest(candidate, salt);
        if (manifest_bytes.size() > cfg.max_manifest_bytes) {
            std::error_code ignored;
            fs::remove(final, ignored);
            return result(capture_store_status::manifest_too_large);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_manifest_write)) {
            return cleanup_after_failure(temporary, final, false);
        }
        const capture_store_result manifest_write =
            write_file(manifest_tmp_path(), manifest_bytes, cfg.faults, cfg.faults.preserve_failed_files);
        if (manifest_write.status != capture_store_status::ok) {
            if (!cfg.faults.preserve_failed_files) {
                std::error_code ignored;
                fs::remove(final, ignored);
                fs::remove(manifest_tmp_path(), ignored);
            }
            return manifest_write;
        }
        if (phase_cancelled(cfg, capture_store_phase::after_manifest_write)) {
            return cleanup_after_failure(manifest_tmp_path(), final, false);
        }
        if (cfg.faults.crash_before_manifest_rename) {
            return cleanup_after_failure(manifest_tmp_path(), final, true);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_manifest_rename)) {
            return cleanup_after_failure(manifest_tmp_path(), final, false);
        }
        if (::rename(manifest_tmp_path().c_str(), manifest_path().c_str()) != 0) {
            return errno_result(errno);
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            manifest = candidate;
        }
        if (cfg.faults.crash_after_manifest_rename) {
            return result(capture_store_status::commit_uncertain, 0, true);
        }
        // Cancellation after publication is too late to revoke a durable
        // commit; returning success is the only truthful outcome.
        (void) phase_cancelled(cfg, capture_store_phase::after_manifest_rename);
        if (!fsync_directory(root(), directory_error)) {
            return result(capture_store_status::commit_uncertain, directory_error, true);
        }
        if (phase_cancelled(cfg, capture_store_phase::before_retention_delete)) {
            return result(capture_store_status::ok, 0, true);
        }
        const capture_store_result deleted = delete_retired(retired);
        if (deleted.status != capture_store_status::ok) {
            return deleted;
        }
        (void) phase_cancelled(cfg, capture_store_phase::after_retention_delete);
        return result(capture_store_status::ok, 0, true);
    }

    void reset_current() {
        current_payload.clear();
        current_records  = 0;
        current_first_ns = 0;
        current_last_ns  = 0;
    }

    void mark_failure(capture_store_result failure) {
        failed.store(true, std::memory_order_release);
        accepting.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex);
        terminal = failure;
        ++failed_writes;
        flush_result    = failure;
        flush_completed = flush_requested;
        cv.notify_all();
    }

    void run() {
        worker_running.store(true, std::memory_order_release);
        current_payload.reserve(static_cast<size_t>(cfg.max_shard_bytes));
        for (;;) {
            cycle_observation observation;
            bool              popped = false;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]() {
                    return stop_requested.load(std::memory_order_acquire) || flush_requested != flush_completed ||
                           ring.stats().size_approx != 0;
                });
                const bool should_drain = drain_requested.load(std::memory_order_acquire);
                if (!should_drain && stop_requested.load(std::memory_order_acquire)) {
                    while (ring.try_pop(observation)) {
                        ++dropped_after_stop;
                    }
                } else {
                    popped = ring.try_pop(observation);
                }
            }
            if (popped) {
                try {
                    const bool rich = cfg.capture_mode == mode::sampled_rich && cfg.allow_sampled_rich &&
                                      observation.cycle_sequence % cfg.rich_sample_every == 0;
                    const size_t before = current_payload.size();
                    const size_t add =
                        CAPTURE_RECORD_HEADER_BYTES + (rich ? CAPTURE_RICH_RECORD_BYTES : CAPTURE_COMPACT_RECORD_BYTES);
                    if (before > cfg.max_shard_bytes - CAPTURE_SHARD_HEADER_BYTES - CAPTURE_SHARD_FOOTER_BYTES ||
                        add > cfg.max_shard_bytes - CAPTURE_SHARD_HEADER_BYTES - CAPTURE_SHARD_FOOTER_BYTES - before) {
                        if (current_records != 0) {
                            const capture_store_result committed = commit_payload();
                            if (committed.status != capture_store_status::ok) {
                                mark_failure(committed);
                                break;
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
                    if (current_records >= cfg.max_shard_records ||
                        current_payload.size() + CAPTURE_SHARD_HEADER_BYTES + CAPTURE_SHARD_FOOTER_BYTES >=
                            cfg.max_shard_bytes) {
                        const capture_store_result committed = commit_payload();
                        if (committed.status != capture_store_status::ok) {
                            mark_failure(committed);
                            break;
                        }
                        reset_current();
                    }
                } catch (const std::bad_alloc &) {
                    mark_failure(result(capture_store_status::io_error, ENOMEM));
                    break;
                }
                if (cfg.faults.slow_worker) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                continue;
            }

            bool should_stop  = false;
            bool should_flush = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                should_stop  = stop_requested.load(std::memory_order_acquire);
                should_flush = flush_requested != flush_completed;
            }
            if (should_flush || should_stop) {
                capture_store_result committed = result(capture_store_status::ok, 0, true);
                if (current_records != 0 && (drain_requested.load(std::memory_order_acquire) || should_flush)) {
                    committed = commit_payload();
                    reset_current();
                }
                if (committed.status != capture_store_status::ok) {
                    mark_failure(committed);
                    break;
                }
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
                cv.notify_all();
                if (should_stop) {
                    break;
                }
            }
        }
        worker_running.store(false, std::memory_order_release);
        accepting.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> lock(mutex);
        stopped = true;
        cv.notify_all();
    }

    capture_store_result flush() {
        if (cfg.capture_mode == mode::off) {
            return result(capture_store_status::disabled);
        }
        if (!worker.joinable()) {
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        std::unique_lock<std::mutex> lock(mutex);
        if (stopped || failed.load(std::memory_order_acquire)) {
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        ++flush_requested;
        const uint64_t request = flush_requested;
        cv.notify_one();
        cv.wait(lock, [this, request]() {
            return flush_completed >= request || stopped || failed.load(std::memory_order_acquire);
        });
        return flush_result;
    }

    capture_store_result shutdown(bool drain) {
        if (cfg.capture_mode == mode::off) {
            return result(capture_store_status::disabled);
        }
        if (!worker.joinable()) {
            return terminal.status == capture_store_status::invalid_argument ? startup : terminal;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            drain_requested.store(drain, std::memory_order_release);
            stop_requested.store(true, std::memory_order_release);
            accepting.store(false, std::memory_order_release);
            cv.notify_one();
        }
        worker.join();
        std::lock_guard<std::mutex> lock(mutex);
        if (terminal.status == capture_store_status::invalid_argument) {
            terminal = result(capture_store_status::stopped);
        }
        return terminal;
    }

    capture_store_config    cfg;
    spsc_ring               ring;
    std::array<uint8_t, 16> salt = {};
    capture_manifest        manifest;
    capture_store_result    startup      = result(capture_store_status::invalid_argument);
    capture_store_result    terminal     = result(capture_store_status::invalid_argument);
    capture_store_result    flush_result = result(capture_store_status::invalid_argument);
    std::thread             worker;
    mutable std::mutex      mutex;
    std::condition_variable cv;
    std::atomic<bool>       accepting{ false };
    std::atomic<bool>       stop_requested{ false };
    std::atomic<bool>       drain_requested{ true };
    std::atomic<bool>       failed{ false };
    std::atomic<bool>       worker_running{ false };
    uint64_t                flush_requested = 0;
    uint64_t                flush_completed = 0;
    bool                    stopped         = false;
    std::vector<uint8_t>    current_payload;
    uint32_t                current_records  = 0;
    uint64_t                current_first_ns = 0;
    uint64_t                current_last_ns  = 0;
    uint64_t                failed_writes    = 0;
    std::atomic<uint64_t>   dropped_after_stop{ 0 };
};

capture_store::capture_store(capture_store_config config) : data(std::make_unique<impl>(std::move(config))) {}

capture_store::~capture_store() = default;

bool capture_store::try_enqueue(const cycle_observation & observation) noexcept {
    if (!data->accepting.load(std::memory_order_acquire)) {
        data->dropped_after_stop.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    const bool pushed = data->ring.try_push(observation);
    if (pushed) {
        data->cv.notify_one();
    }
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
    std::vector<uint8_t>       bytes;
    const capture_store_result read =
        read_file(data->manifest_path(), data->cfg.max_manifest_bytes, bytes, capture_store_status::no_manifest,
                  capture_store_status::manifest_too_large);
    if (read.status != capture_store_status::ok) {
        return read;
    }
    std::array<uint8_t, 16> ignored_salt = {};
    return parse_manifest(bytes, data->cfg, output, ignored_salt);
}

capture_store_result capture_store::validate(const capture_manifest & manifest) const {
    return data->validate(manifest);
}

capture_store_stats capture_store::stats() const noexcept {
    capture_store_stats output;
    output.ring               = data->ring.stats();
    output.worker_running     = data->worker_running.load(std::memory_order_acquire);
    output.worker_failed      = data->failed.load(std::memory_order_acquire);
    output.dropped_after_stop = data->dropped_after_stop.load(std::memory_order_relaxed);
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

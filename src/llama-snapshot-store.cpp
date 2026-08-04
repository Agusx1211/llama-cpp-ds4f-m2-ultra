#include "llama-snapshot-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <new>
#include <string>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace {

constexpr uint8_t MANIFEST_MAGIC[8]           = { 'L', 'S', 'N', 'P', 'M', 'F', '0', '1' };
constexpr uint8_t CHUNK_MAGIC[8]              = { 'L', 'S', 'N', 'P', 'C', 'H', '0', '1' };
constexpr size_t  MANIFEST_FIXED_PREFIX_BYTES = 24;
constexpr size_t  CHUNK_FIXED_FIELDS_BYTES    = 72;

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bit_length;
    uint8_t  buffer[64];
    size_t   buffer_length;
};

// FIPS 180-4 SHA-256, adapted from the repository's public-domain OpenCL
// program-cache implementation. Focused tests include the published empty and
// "abc" vectors so snapshot checksums do not depend on a backend.

constexpr uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

uint32_t rotate_right(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32 - shift));
}

void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    uint32_t words[64];
    for (int index = 0; index < 16; ++index) {
        words[index] = static_cast<uint32_t>(block[index * 4]) << 24 |
                       static_cast<uint32_t>(block[index * 4 + 1]) << 16 |
                       static_cast<uint32_t>(block[index * 4 + 2]) << 8 | static_cast<uint32_t>(block[index * 4 + 3]);
    }
    for (int index = 16; index < 64; ++index) {
        const uint32_t s0 =
            rotate_right(words[index - 15], 7) ^ rotate_right(words[index - 15], 18) ^ (words[index - 15] >> 3);
        const uint32_t s1 =
            rotate_right(words[index - 2], 17) ^ rotate_right(words[index - 2], 19) ^ (words[index - 2] >> 10);
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
    for (int index = 0; index < 64; ++index) {
        const uint32_t sum1     = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose   = (e & f) ^ ((~e) & g);
        const uint32_t temp1    = h + sum1 + choose + SHA256_K[index] + words[index];
        const uint32_t sum0     = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
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

void sha256_init(sha256_ctx & context) {
    context.state[0]      = 0x6a09e667;
    context.state[1]      = 0xbb67ae85;
    context.state[2]      = 0x3c6ef372;
    context.state[3]      = 0xa54ff53a;
    context.state[4]      = 0x510e527f;
    context.state[5]      = 0x9b05688c;
    context.state[6]      = 0x1f83d9ab;
    context.state[7]      = 0x5be0cd19;
    context.bit_length    = 0;
    context.buffer_length = 0;
}

void sha256_update(sha256_ctx & context, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    context.bit_length += static_cast<uint64_t>(size) * 8;
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

llama_snapshot_digest sha256_finish(sha256_ctx & context) {
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
        context.buffer[context.buffer_length++] = static_cast<uint8_t>(bit_length >> (shift * 8));
    }
    sha256_compress(context.state, context.buffer);

    llama_snapshot_digest result;
    for (int index = 0; index < 8; ++index) {
        result[index * 4]     = static_cast<uint8_t>(context.state[index] >> 24);
        result[index * 4 + 1] = static_cast<uint8_t>(context.state[index] >> 16);
        result[index * 4 + 2] = static_cast<uint8_t>(context.state[index] >> 8);
        result[index * 4 + 3] = static_cast<uint8_t>(context.state[index]);
    }
    return result;
}

bool digest_is_zero(const llama_snapshot_digest & digest) {
    return std::all_of(digest.begin(), digest.end(), [](uint8_t byte) { return byte == 0; });
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

void set_u64(std::vector<uint8_t> & bytes, size_t offset, uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
        bytes[offset + shift / 8] = static_cast<uint8_t>(value >> shift);
    }
}

void append_digest(std::vector<uint8_t> & bytes, const llama_snapshot_digest & digest) {
    bytes.insert(bytes.end(), digest.begin(), digest.end());
}

void append_string(std::vector<uint8_t> & bytes, const std::string & value) {
    append_u32(bytes, static_cast<uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
}

struct byte_reader {
    const std::vector<uint8_t> & bytes;
    size_t                       cursor;
    size_t                       end;

    bool read_u32(uint32_t & value) {
        if (end - cursor < 4) {
            return false;
        }
        value = 0;
        for (unsigned shift = 0; shift < 32; shift += 8) {
            value |= static_cast<uint32_t>(bytes[cursor++]) << shift;
        }
        return true;
    }

    bool read_u64(uint64_t & value) {
        if (end - cursor < 8) {
            return false;
        }
        value = 0;
        for (unsigned shift = 0; shift < 64; shift += 8) {
            value |= static_cast<uint64_t>(bytes[cursor++]) << shift;
        }
        return true;
    }

    bool read_digest(llama_snapshot_digest & digest) {
        if (end - cursor < digest.size()) {
            return false;
        }
        std::copy_n(bytes.begin() + cursor, digest.size(), digest.begin());
        cursor += digest.size();
        return true;
    }

    bool read_string(std::string & value) {
        uint32_t size = 0;
        if (!read_u32(size) || size > LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES || end - cursor < size) {
            return false;
        }
        value.assign(reinterpret_cast<const char *>(bytes.data() + cursor), size);
        cursor += size;
        return true;
    }
};

struct operation_result {
    llama_snapshot_status status   = llama_snapshot_status::ok;
    int                   os_error = 0;
};

operation_result config_status(const llama_snapshot_store_config & config) {
    if (config.root_path.empty() || config.physical_device_id.empty() ||
        config.physical_device_id.size() > LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES ||
        config.physical_device_queues != LLAMA_SNAPSHOT_DEVICE_QUEUES || config.chunk_payload_bytes == 0 ||
        config.chunk_payload_bytes > LLAMA_SNAPSHOT_MAX_CHUNK_BYTES || config.max_chunks == 0 ||
        config.max_chunks > LLAMA_SNAPSHOT_MAX_CHUNKS || config.max_manifest_bytes < 256 ||
        config.max_manifest_bytes > LLAMA_SNAPSHOT_MAX_MANIFEST_BYTES || config.max_snapshot_bytes == 0 ||
        config.max_snapshot_bytes > LLAMA_SNAPSHOT_MAX_BYTES) {
        return { llama_snapshot_status::invalid_argument, 0 };
    }
    return {};
}

bool valid_identity(const llama_snapshot_identity & identity) {
    const bool draft_agrees = identity.draft_kv_type.empty() == digest_is_zero(identity.draft_kv_digest);
    return !identity.architecture.empty() && identity.architecture.size() <= LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES &&
           !identity.target_kv_type.empty() && identity.target_kv_type.size() <= LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES &&
           identity.draft_kv_type.size() <= LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES &&
           !digest_is_zero(identity.model_artifact_digest) && !digest_is_zero(identity.tokenizer_digest) &&
           !digest_is_zero(identity.chat_template_digest) && !digest_is_zero(identity.runtime_build_digest) &&
           !digest_is_zero(identity.target_kv_digest) && !digest_is_zero(identity.rope_digest) && draft_agrees &&
           identity.context_size != 0 && identity.raw_window != 0 && identity.c4_ratio == 4 &&
           identity.hca_ratio == 128 && identity.dsv4_state_version != 0 && identity.rollback_depth != 0;
}

std::string generation_name(uint64_t generation, bool partial) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%s%016llx", partial ? ".partial-generation-" : "generation-",
                  static_cast<unsigned long long>(generation));
    return buffer;
}

std::string chunk_name(uint32_t index) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "chunk-%08x.pack", index);
    return buffer;
}

operation_result ensure_root(const fs::path & root) {
    std::error_code error;
    fs::create_directories(root, error);
    if (error || !fs::is_directory(root, error)) {
        return { llama_snapshot_status::io_error, error.value() };
    }
    return {};
}

operation_result fsync_directory(const fs::path & path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_DIRECTORY);
    if (descriptor < 0) {
        return { llama_snapshot_status::io_error, errno };
    }
    if (::fsync(descriptor) != 0) {
        const int error = errno;
        ::close(descriptor);
        return { llama_snapshot_status::io_error, error };
    }
    if (::close(descriptor) != 0) {
        return { llama_snapshot_status::io_error, errno };
    }
    return {};
}

struct fault_writer {
    const llama_snapshot_faults & faults;
    uint64_t                      bytes_written = 0;

    operation_result write_all(int descriptor, const uint8_t * data, size_t size) {
        while (size != 0) {
            if (faults.write_fault != llama_snapshot_write_fault::none && bytes_written >= faults.fail_after_bytes) {
                return { faults.write_fault == llama_snapshot_write_fault::no_space ?
                             llama_snapshot_status::no_space :
                             llama_snapshot_status::short_write,
                         faults.write_fault == llama_snapshot_write_fault::no_space ? ENOSPC : EIO };
            }

            uint64_t count = std::min<uint64_t>(size, faults.max_write_size);
            if (faults.write_fault != llama_snapshot_write_fault::none) {
                count = std::min<uint64_t>(count, faults.fail_after_bytes - bytes_written);
            }
            if (count == 0) {
                return { llama_snapshot_status::short_write, EIO };
            }

            const ssize_t written = ::write(descriptor, data, static_cast<size_t>(count));
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return { errno == ENOSPC ? llama_snapshot_status::no_space : llama_snapshot_status::io_error, errno };
            }
            if (written == 0) {
                return { llama_snapshot_status::short_write, EIO };
            }
            bytes_written += static_cast<uint64_t>(written);
            data += written;
            size -= static_cast<size_t>(written);
        }
        return {};
    }
};

operation_result write_file(const fs::path &                                        path,
                            const std::vector<std::pair<const uint8_t *, size_t>> & pieces,
                            fault_writer &                                          writer,
                            bool                                                    exclusive) {
    const int flags      = O_WRONLY | O_CREAT | (exclusive ? O_EXCL : O_TRUNC);
    const int descriptor = ::open(path.c_str(), flags, 0600);
    if (descriptor < 0) {
        return { errno == ENOSPC ? llama_snapshot_status::no_space : llama_snapshot_status::io_error, errno };
    }
    operation_result result;
    for (const auto & piece : pieces) {
        result = writer.write_all(descriptor, piece.first, piece.second);
        if (result.status != llama_snapshot_status::ok) {
            break;
        }
    }
    if (result.status == llama_snapshot_status::ok && ::fsync(descriptor) != 0) {
        result = { errno == ENOSPC ? llama_snapshot_status::no_space : llama_snapshot_status::io_error, errno };
    }
    if (::close(descriptor) != 0 && result.status == llama_snapshot_status::ok) {
        result = { llama_snapshot_status::io_error, errno };
    }
    if (result.status != llama_snapshot_status::ok) {
        ::unlink(path.c_str());
    }
    return result;
}

operation_result read_file_bounded(const fs::path &       path,
                                   uint64_t               max_bytes,
                                   llama_snapshot_status  missing_status,
                                   llama_snapshot_status  too_large_status,
                                   std::vector<uint8_t> & bytes) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        return { errno == ENOENT ? missing_status : llama_snapshot_status::io_error, errno };
    }
    struct stat file_status{};
    if (::fstat(descriptor, &file_status) != 0) {
        const int error = errno;
        ::close(descriptor);
        return { llama_snapshot_status::io_error, error };
    }
    if (!S_ISREG(file_status.st_mode) || file_status.st_size < 0) {
        ::close(descriptor);
        return { llama_snapshot_status::format_error, 0 };
    }
    if (static_cast<uint64_t>(file_status.st_size) > max_bytes) {
        ::close(descriptor);
        return { too_large_status, 0 };
    }
    try {
        bytes.resize(static_cast<size_t>(file_status.st_size));
    } catch (const std::bad_alloc &) {
        ::close(descriptor);
        return { llama_snapshot_status::io_error, ENOMEM };
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
            return { llama_snapshot_status::io_error, error };
        }
        if (count == 0) {
            ::close(descriptor);
            return { llama_snapshot_status::truncated, 0 };
        }
        offset += static_cast<size_t>(count);
    }
    if (::close(descriptor) != 0) {
        return { llama_snapshot_status::io_error, errno };
    }
    return {};
}

std::vector<uint8_t> serialize_manifest(const llama_snapshot_manifest & manifest) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), std::begin(MANIFEST_MAGIC), std::end(MANIFEST_MAGIC));
    append_u32(bytes, LLAMA_SNAPSHOT_FORMAT_VERSION);
    append_u32(bytes, 0);
    append_u64(bytes, 0);
    append_u64(bytes, manifest.snapshot_generation);
    append_u64(bytes, manifest.request_generation);
    append_u32(bytes, manifest.physical_device_queues);
    append_u32(bytes, static_cast<uint32_t>(manifest.chunks.size()));
    append_u64(bytes, manifest.chunk_payload_limit);
    append_u64(bytes, manifest.total_payload_bytes);
    append_string(bytes, manifest.identity.architecture);
    append_string(bytes, manifest.identity.target_kv_type);
    append_string(bytes, manifest.identity.draft_kv_type);
    append_string(bytes, manifest.physical_device_id);
    append_digest(bytes, manifest.identity.model_artifact_digest);
    append_digest(bytes, manifest.identity.tokenizer_digest);
    append_digest(bytes, manifest.identity.chat_template_digest);
    append_digest(bytes, manifest.identity.runtime_build_digest);
    append_digest(bytes, manifest.identity.target_kv_digest);
    append_digest(bytes, manifest.identity.draft_kv_digest);
    append_digest(bytes, manifest.identity.rope_digest);
    append_digest(bytes, manifest.identity.lora_digest);
    append_u32(bytes, manifest.identity.context_size);
    append_u32(bytes, manifest.identity.raw_window);
    append_u32(bytes, manifest.identity.c4_ratio);
    append_u32(bytes, manifest.identity.hca_ratio);
    append_u32(bytes, manifest.identity.dsv4_state_version);
    append_u32(bytes, manifest.identity.rollback_depth);
    for (const llama_snapshot_chunk_info & chunk : manifest.chunks) {
        append_u32(bytes, chunk.index);
        append_u32(bytes, 0);
        append_u64(bytes, chunk.logical_offset);
        append_u64(bytes, chunk.payload_bytes);
        append_digest(bytes, chunk.checksum);
    }
    set_u64(bytes, 16, static_cast<uint64_t>(bytes.size() + llama_snapshot_digest{}.size()));
    append_digest(bytes, llama_snapshot_sha256(bytes.data(), bytes.size()));
    return bytes;
}

operation_result validate_manifest_shape(const llama_snapshot_store_config & config,
                                         const llama_snapshot_manifest &     manifest,
                                         bool                                require_device) {
    if (require_device && manifest.physical_device_id != config.physical_device_id) {
        return { llama_snapshot_status::device_mismatch, 0 };
    }
    if (manifest.format_version != LLAMA_SNAPSHOT_FORMAT_VERSION || !valid_identity(manifest.identity) ||
        manifest.physical_device_queues != LLAMA_SNAPSHOT_DEVICE_QUEUES || manifest.physical_device_id.empty() ||
        manifest.physical_device_id.size() > LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES || manifest.chunk_payload_limit == 0 ||
        manifest.chunk_payload_limit > config.chunk_payload_bytes ||
        manifest.chunk_payload_limit > LLAMA_SNAPSHOT_MAX_CHUNK_BYTES || manifest.chunks.size() > config.max_chunks ||
        manifest.total_payload_bytes > config.max_snapshot_bytes) {
        return { llama_snapshot_status::format_error, 0 };
    }
    uint64_t expected_offset = 0;
    for (size_t index = 0; index < manifest.chunks.size(); ++index) {
        const llama_snapshot_chunk_info & chunk = manifest.chunks[index];
        if (chunk.index != index || chunk.logical_offset != expected_offset || chunk.payload_bytes == 0 ||
            chunk.payload_bytes > manifest.chunk_payload_limit || expected_offset > manifest.total_payload_bytes ||
            chunk.payload_bytes > manifest.total_payload_bytes - expected_offset) {
            return { llama_snapshot_status::format_error, 0 };
        }
        expected_offset += chunk.payload_bytes;
    }
    if (expected_offset != manifest.total_payload_bytes ||
        (manifest.total_payload_bytes == 0) != manifest.chunks.empty()) {
        return { llama_snapshot_status::format_error, 0 };
    }
    return {};
}

operation_result parse_manifest(const llama_snapshot_store_config & config,
                                const std::vector<uint8_t> &        bytes,
                                llama_snapshot_manifest &           manifest) {
    if (bytes.size() < MANIFEST_FIXED_PREFIX_BYTES + llama_snapshot_digest{}.size()) {
        return { llama_snapshot_status::truncated, 0 };
    }
    if (!std::equal(std::begin(MANIFEST_MAGIC), std::end(MANIFEST_MAGIC), bytes.begin())) {
        return { llama_snapshot_status::format_error, 0 };
    }
    byte_reader prefix{ bytes, 8, bytes.size() };
    uint32_t    version       = 0;
    uint32_t    reserved      = 0;
    uint64_t    declared_size = 0;
    if (!prefix.read_u32(version) || !prefix.read_u32(reserved) || !prefix.read_u64(declared_size) ||
        version != LLAMA_SNAPSHOT_FORMAT_VERSION || reserved != 0) {
        return { llama_snapshot_status::format_error, 0 };
    }
    if (declared_size > config.max_manifest_bytes || declared_size > LLAMA_SNAPSHOT_MAX_MANIFEST_BYTES) {
        return { llama_snapshot_status::manifest_too_large, 0 };
    }
    if (bytes.size() < declared_size) {
        return { llama_snapshot_status::truncated, 0 };
    }
    if (bytes.size() > declared_size) {
        return { llama_snapshot_status::trailing_data, 0 };
    }
    const size_t                payload_end       = bytes.size() - llama_snapshot_digest{}.size();
    const llama_snapshot_digest expected_checksum = llama_snapshot_sha256(bytes.data(), payload_end);
    if (!std::equal(expected_checksum.begin(), expected_checksum.end(), bytes.begin() + payload_end)) {
        return { llama_snapshot_status::checksum_mismatch, 0 };
    }

    manifest                = {};
    manifest.format_version = version;
    byte_reader reader{ bytes, MANIFEST_FIXED_PREFIX_BYTES, payload_end };
    uint32_t    chunk_count = 0;
    if (!reader.read_u64(manifest.snapshot_generation) || !reader.read_u64(manifest.request_generation) ||
        !reader.read_u32(manifest.physical_device_queues) || !reader.read_u32(chunk_count) ||
        !reader.read_u64(manifest.chunk_payload_limit) || !reader.read_u64(manifest.total_payload_bytes) ||
        chunk_count > config.max_chunks || chunk_count > LLAMA_SNAPSHOT_MAX_CHUNKS ||
        !reader.read_string(manifest.identity.architecture) || !reader.read_string(manifest.identity.target_kv_type) ||
        !reader.read_string(manifest.identity.draft_kv_type) || !reader.read_string(manifest.physical_device_id) ||
        !reader.read_digest(manifest.identity.model_artifact_digest) ||
        !reader.read_digest(manifest.identity.tokenizer_digest) ||
        !reader.read_digest(manifest.identity.chat_template_digest) ||
        !reader.read_digest(manifest.identity.runtime_build_digest) ||
        !reader.read_digest(manifest.identity.target_kv_digest) ||
        !reader.read_digest(manifest.identity.draft_kv_digest) || !reader.read_digest(manifest.identity.rope_digest) ||
        !reader.read_digest(manifest.identity.lora_digest) || !reader.read_u32(manifest.identity.context_size) ||
        !reader.read_u32(manifest.identity.raw_window) || !reader.read_u32(manifest.identity.c4_ratio) ||
        !reader.read_u32(manifest.identity.hca_ratio) || !reader.read_u32(manifest.identity.dsv4_state_version) ||
        !reader.read_u32(manifest.identity.rollback_depth)) {
        return { llama_snapshot_status::format_error, 0 };
    }
    try {
        manifest.chunks.resize(chunk_count);
    } catch (const std::bad_alloc &) {
        return { llama_snapshot_status::io_error, ENOMEM };
    }
    for (uint32_t index = 0; index < chunk_count; ++index) {
        uint32_t                    entry_reserved = 0;
        llama_snapshot_chunk_info & chunk          = manifest.chunks[index];
        if (!reader.read_u32(chunk.index) || !reader.read_u32(entry_reserved) || entry_reserved != 0 ||
            !reader.read_u64(chunk.logical_offset) || !reader.read_u64(chunk.payload_bytes) ||
            !reader.read_digest(chunk.checksum)) {
            return { llama_snapshot_status::format_error, 0 };
        }
    }
    if (reader.cursor != reader.end) {
        return { llama_snapshot_status::trailing_data, 0 };
    }
    return validate_manifest_shape(config, manifest, false);
}

operation_result load_manifest(const llama_snapshot_store_config & config,
                               const fs::path &                    path,
                               llama_snapshot_status               missing_status,
                               llama_snapshot_manifest &           manifest) {
    std::vector<uint8_t>   bytes;
    const operation_result read = read_file_bounded(path, config.max_manifest_bytes, missing_status,
                                                    llama_snapshot_status::manifest_too_large, bytes);
    if (read.status != llama_snapshot_status::ok) {
        return read;
    }
    return parse_manifest(config, bytes, manifest);
}

std::vector<uint8_t> chunk_header(const llama_snapshot_manifest & manifest, const llama_snapshot_chunk_info & chunk) {
    std::vector<uint8_t> header(LLAMA_SNAPSHOT_CHUNK_ALIGNMENT, 0);
    std::vector<uint8_t> fields;
    fields.insert(fields.end(), std::begin(CHUNK_MAGIC), std::end(CHUNK_MAGIC));
    append_u32(fields, LLAMA_SNAPSHOT_CHUNK_FORMAT_VERSION);
    append_u32(fields, static_cast<uint32_t>(LLAMA_SNAPSHOT_CHUNK_ALIGNMENT));
    append_u64(fields, manifest.snapshot_generation);
    append_u32(fields, chunk.index);
    append_u32(fields, 0);
    append_u64(fields, chunk.payload_bytes);
    append_digest(fields, chunk.checksum);
    std::copy(fields.begin(), fields.end(), header.begin());
    return header;
}

void remove_path_best_effort(const fs::path & path) {
    try {
        std::error_code error;
        fs::remove_all(path, error);
    } catch (...) {
    }
}

struct generation_cleanup_guard {
    const fs::path & partial_path;
    const fs::path & final_path;
    const fs::path & current_tmp;
    bool             preserve;
    bool             final_created = false;
    bool             committed     = false;

    ~generation_cleanup_guard() {
        if (preserve || committed) {
            return;
        }
        remove_path_best_effort(final_created ? final_path : partial_path);
        ::unlink(current_tmp.c_str());
    }
};

struct chunk_release_guard {
    llama_snapshot_chunk_source_i * source;
    uint32_t                         index;

    ~chunk_release_guard() { source->release(index); }
};

class vector_chunk_source final : public llama_snapshot_chunk_source_i {
  public:
    explicit vector_chunk_source(const std::vector<uint8_t> & payload) : payload(payload) {}

    llama_snapshot_chunk_source_result acquire(
            uint32_t, uint64_t logical_offset, uint64_t size) noexcept override {
        if (logical_offset > payload.size() || size > payload.size() - logical_offset) {
            return { llama_snapshot_status::invalid_argument, nullptr, 0, 0 };
        }
        return {
            llama_snapshot_status::ok,
            payload.data() + static_cast<size_t>(logical_offset),
            size,
            0,
        };
    }

    void release(uint32_t) noexcept override {}

  private:
    const std::vector<uint8_t> & payload;
};

operation_result pread_all(int descriptor, uint8_t * destination, size_t size, off_t offset) {
    size_t done = 0;
    while (done < size) {
        const ssize_t count = ::pread(descriptor, destination + done, size - done, offset + static_cast<off_t>(done));
        if (count < 0) {
            if (errno == EINTR) {
                continue;
            }
            return { llama_snapshot_status::io_error, errno };
        }
        if (count == 0) {
            return { llama_snapshot_status::truncated, 0 };
        }
        done += static_cast<size_t>(count);
    }
    return {};
}

uint32_t read_u32_le(const uint8_t * bytes) {
    uint32_t value = 0;
    for (unsigned shift = 0; shift < 32; shift += 8) {
        value |= static_cast<uint32_t>(bytes[shift / 8]) << shift;
    }
    return value;
}

uint64_t read_u64_le(const uint8_t * bytes) {
    uint64_t value = 0;
    for (unsigned shift = 0; shift < 64; shift += 8) {
        value |= static_cast<uint64_t>(bytes[shift / 8]) << shift;
    }
    return value;
}

}  // namespace

llama_snapshot_digest llama_snapshot_sha256(const void * data, size_t size) {
    sha256_ctx context;
    sha256_init(context);
    sha256_update(context, data, size);
    return sha256_finish(context);
}

std::string llama_snapshot_digest_hex(const llama_snapshot_digest & digest) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string           result(digest.size() * 2, '0');
    for (size_t index = 0; index < digest.size(); ++index) {
        result[index * 2]     = HEX[digest[index] >> 4];
        result[index * 2 + 1] = HEX[digest[index] & 0x0f];
    }
    return result;
}

const char * llama_snapshot_status_name(llama_snapshot_status status) {
    switch (status) {
        case llama_snapshot_status::ok:
            return "ok";
        case llama_snapshot_status::invalid_argument:
            return "invalid_argument";
        case llama_snapshot_status::no_current_generation:
            return "no_current_generation";
        case llama_snapshot_status::generation_exists:
            return "generation_exists";
        case llama_snapshot_status::stale_generation:
            return "stale_generation";
        case llama_snapshot_status::identity_mismatch:
            return "identity_mismatch";
        case llama_snapshot_status::device_mismatch:
            return "device_mismatch";
        case llama_snapshot_status::manifest_too_large:
            return "manifest_too_large";
        case llama_snapshot_status::chunk_too_large:
            return "chunk_too_large";
        case llama_snapshot_status::format_error:
            return "format_error";
        case llama_snapshot_status::truncated:
            return "truncated";
        case llama_snapshot_status::trailing_data:
            return "trailing_data";
        case llama_snapshot_status::missing_chunk:
            return "missing_chunk";
        case llama_snapshot_status::checksum_mismatch:
            return "checksum_mismatch";
        case llama_snapshot_status::cancelled:
            return "cancelled";
        case llama_snapshot_status::no_space:
            return "no_space";
        case llama_snapshot_status::short_write:
            return "short_write";
        case llama_snapshot_status::io_error:
            return "io_error";
        case llama_snapshot_status::commit_uncertain:
            return "commit_uncertain";
    }
    return "unknown";
}

bool operator==(const llama_snapshot_identity & lhs, const llama_snapshot_identity & rhs) {
    return lhs.architecture == rhs.architecture && lhs.model_artifact_digest == rhs.model_artifact_digest &&
           lhs.tokenizer_digest == rhs.tokenizer_digest && lhs.chat_template_digest == rhs.chat_template_digest &&
           lhs.runtime_build_digest == rhs.runtime_build_digest && lhs.target_kv_digest == rhs.target_kv_digest &&
           lhs.draft_kv_digest == rhs.draft_kv_digest && lhs.rope_digest == rhs.rope_digest &&
           lhs.lora_digest == rhs.lora_digest && lhs.target_kv_type == rhs.target_kv_type &&
           lhs.draft_kv_type == rhs.draft_kv_type && lhs.context_size == rhs.context_size &&
           lhs.raw_window == rhs.raw_window && lhs.c4_ratio == rhs.c4_ratio && lhs.hca_ratio == rhs.hca_ratio &&
           lhs.dsv4_state_version == rhs.dsv4_state_version && lhs.rollback_depth == rhs.rollback_depth;
}

bool operator!=(const llama_snapshot_identity & lhs, const llama_snapshot_identity & rhs) {
    return !(lhs == rhs);
}

llama_snapshot_store::llama_snapshot_store(llama_snapshot_store_config config) : cfg(std::move(config)) {}

const llama_snapshot_store_config & llama_snapshot_store::config() const {
    return cfg;
}

llama_snapshot_write_result llama_snapshot_store::write_generation(const llama_snapshot_metadata &     metadata,
                                                                   const std::vector<uint8_t> &        payload,
                                                                   const llama_snapshot_cancel_check & cancelled,
                                                                   const llama_snapshot_faults &       faults) {
    vector_chunk_source source(payload);
    return write_generation_streamed(metadata, payload.size(), source, cancelled, faults);
}

llama_snapshot_write_result llama_snapshot_store::write_generation_streamed(
        const llama_snapshot_metadata &     metadata,
        uint64_t                            total_payload_bytes,
        llama_snapshot_chunk_source_i &     source,
        const llama_snapshot_cancel_check & cancelled,
        const llama_snapshot_faults &       faults,
        const llama_snapshot_commit_fence & commit_fence) {
    llama_snapshot_write_result result;
    result.generation = metadata.snapshot_generation;
    try {
        const operation_result config_check = config_status(cfg);
        if (config_check.status != llama_snapshot_status::ok || metadata.snapshot_generation == 0 ||
            metadata.request_generation == 0 || !valid_identity(metadata.identity) || faults.max_write_size == 0 ||
            (faults.write_fault == llama_snapshot_write_fault::none &&
             faults.fail_after_bytes != std::numeric_limits<uint64_t>::max()) ||
            total_payload_bytes > cfg.max_snapshot_bytes) {
            result.status = llama_snapshot_status::invalid_argument;
            return result;
        }
        const uint64_t chunk_count = total_payload_bytes == 0 ?
                                         0 :
                                         total_payload_bytes / cfg.chunk_payload_bytes +
                                             (total_payload_bytes % cfg.chunk_payload_bytes != 0);
        if (chunk_count > cfg.max_chunks || chunk_count > UINT32_MAX) {
            result.status = llama_snapshot_status::chunk_too_large;
            return result;
        }
        if (cancelled && cancelled(0)) {
            result.status = llama_snapshot_status::cancelled;
            return result;
        }

        const fs::path   root      = fs::u8path(cfg.root_path);
        operation_result operation = ensure_root(root);
        if (operation.status != llama_snapshot_status::ok) {
            result.status   = operation.status;
            result.os_error = operation.os_error;
            return result;
        }

        llama_snapshot_manifest previous;
        const operation_result  previous_result =
            load_manifest(cfg, root / "current.manifest", llama_snapshot_status::no_current_generation, previous);
        const bool has_previous = previous_result.status == llama_snapshot_status::ok;
        if (!has_previous && previous_result.status != llama_snapshot_status::no_current_generation) {
            result.status   = previous_result.status;
            result.os_error = previous_result.os_error;
            return result;
        }
        if (has_previous && (previous.physical_device_id != cfg.physical_device_id ||
                             previous.physical_device_queues != cfg.physical_device_queues)) {
            result.status = llama_snapshot_status::device_mismatch;
            return result;
        }
        if (has_previous && previous.identity != metadata.identity) {
            result.status = llama_snapshot_status::identity_mismatch;
            return result;
        }
        if (has_previous && previous.snapshot_generation == metadata.snapshot_generation) {
            result.status = llama_snapshot_status::generation_exists;
            return result;
        }
        if (has_previous && previous.snapshot_generation > metadata.snapshot_generation) {
            result.status = llama_snapshot_status::stale_generation;
            return result;
        }

        llama_snapshot_manifest manifest;
        manifest.format_version         = LLAMA_SNAPSHOT_FORMAT_VERSION;
        manifest.snapshot_generation    = metadata.snapshot_generation;
        manifest.request_generation     = metadata.request_generation;
        manifest.identity               = metadata.identity;
        manifest.physical_device_id     = cfg.physical_device_id;
        manifest.physical_device_queues = cfg.physical_device_queues;
        manifest.chunk_payload_limit    = cfg.chunk_payload_bytes;
        manifest.total_payload_bytes    = total_payload_bytes;
        manifest.chunks.reserve(static_cast<size_t>(chunk_count));
        uint64_t offset = 0;
        for (uint32_t index = 0; index < chunk_count; ++index) {
            llama_snapshot_chunk_info chunk;
            chunk.index          = index;
            chunk.logical_offset = offset;
            chunk.payload_bytes  = std::min<uint64_t>(cfg.chunk_payload_bytes, total_payload_bytes - offset);
            manifest.chunks.push_back(chunk);
            offset += chunk.payload_bytes;
        }
        if (serialize_manifest(manifest).size() > cfg.max_manifest_bytes) {
            result.status = llama_snapshot_status::manifest_too_large;
            return result;
        }

        const fs::path  partial_path = root / generation_name(metadata.snapshot_generation, true);
        const fs::path  final_path   = root / generation_name(metadata.snapshot_generation, false);
        const fs::path  current_tmp  = root / "current.manifest.tmp";
        std::error_code exists_error;
        if (fs::exists(partial_path, exists_error) || fs::exists(final_path, exists_error) || exists_error) {
            result.status = exists_error ? llama_snapshot_status::io_error : llama_snapshot_status::generation_exists;
            result.os_error = exists_error.value();
            return result;
        }
        std::error_code create_error;
        if (!fs::create_directory(partial_path, create_error) || create_error) {
            result.status   = llama_snapshot_status::io_error;
            result.os_error = create_error.value();
            return result;
        }

        generation_cleanup_guard cleanup{
            partial_path,
            final_path,
            current_tmp,
            faults.preserve_failed_generation,
        };
        const auto fail = [&](operation_result failure) {
            result.status   = failure.status;
            result.os_error = failure.os_error;
            return result;
        };

        fault_writer writer{ faults };
        for (llama_snapshot_chunk_info & chunk : manifest.chunks) {
            if (cancelled && cancelled(chunk.index)) {
                return fail({ llama_snapshot_status::cancelled, 0 });
            }
            const llama_snapshot_chunk_source_result staged =
                source.acquire(chunk.index, chunk.logical_offset, chunk.payload_bytes);
            if (staged.status != llama_snapshot_status::ok) {
                return fail({ staged.status, staged.os_error });
            }
            chunk_release_guard release{ &source, chunk.index };
            if (staged.size != chunk.payload_bytes || (staged.size != 0 && staged.data == nullptr)) {
                return fail({ llama_snapshot_status::invalid_argument, 0 });
            }
            chunk.checksum                  = llama_snapshot_sha256(staged.data, static_cast<size_t>(staged.size));
            const std::vector<uint8_t> header = chunk_header(manifest, chunk);
            operation = write_file(partial_path / chunk_name(chunk.index),
                                   {
                                       { header.data(), header.size() },
                                       { staged.data, static_cast<size_t>(staged.size) },
                                   },
                                   writer, true);
            if (operation.status != llama_snapshot_status::ok) {
                return fail(operation);
            }
            if (cancelled && cancelled(chunk.index + 1)) {
                return fail({ llama_snapshot_status::cancelled, 0 });
            }
        }

        const std::vector<uint8_t> manifest_bytes = serialize_manifest(manifest);
        operation = write_file(partial_path / "generation.manifest",
                               {
                                   { manifest_bytes.data(), manifest_bytes.size() },
                               },
                               writer, true);
        if (operation.status != llama_snapshot_status::ok) {
            return fail(operation);
        }
        operation = fsync_directory(partial_path);
        if (operation.status != llama_snapshot_status::ok) {
            return fail(operation);
        }
        if (cancelled && cancelled(static_cast<uint32_t>(manifest.chunks.size()))) {
            return fail({ llama_snapshot_status::cancelled, 0 });
        }
        if (::rename(partial_path.c_str(), final_path.c_str()) != 0) {
            return fail({ llama_snapshot_status::io_error, errno });
        }
        cleanup.final_created = true;
        operation             = fsync_directory(root);
        if (operation.status != llama_snapshot_status::ok) {
            return fail(operation);
        }
        if (faults.fail_before_manifest_commit) {
            return fail({ llama_snapshot_status::io_error, EIO });
        }
        operation = write_file(current_tmp,
                               {
                                   { manifest_bytes.data(), manifest_bytes.size() },
                               },
                               writer, false);
        if (operation.status != llama_snapshot_status::ok) {
            return fail(operation);
        }
        if (cancelled && cancelled(static_cast<uint32_t>(manifest.chunks.size()))) {
            return fail({ llama_snapshot_status::cancelled, 0 });
        }
        if (faults.before_manifest_commit_fence) {
            faults.before_manifest_commit_fence();
        }
        if (commit_fence && !commit_fence()) {
            return fail({ llama_snapshot_status::cancelled, 0 });
        }
        if (faults.after_manifest_commit_fence) {
            faults.after_manifest_commit_fence();
        }
        if (::rename(current_tmp.c_str(), (root / "current.manifest").c_str()) != 0) {
            return fail({ llama_snapshot_status::io_error, errno });
        }
        result.committed = true;
        cleanup.committed = true;
        operation = fsync_directory(root);
        if (operation.status != llama_snapshot_status::ok) {
            result.status   = llama_snapshot_status::commit_uncertain;
            result.os_error = operation.os_error;
            return result;
        }
        if (has_previous && previous.snapshot_generation != metadata.snapshot_generation) {
            remove_path_best_effort(root / generation_name(previous.snapshot_generation, false));
        }
        result.status = llama_snapshot_status::ok;
        return result;
    } catch (const std::bad_alloc &) {
        result.status   = result.committed ? llama_snapshot_status::commit_uncertain : llama_snapshot_status::io_error;
        result.os_error = ENOMEM;
        return result;
    } catch (...) {
        result.status   = result.committed ? llama_snapshot_status::commit_uncertain : llama_snapshot_status::io_error;
        result.os_error = EIO;
        return result;
    }
}

llama_snapshot_open_result llama_snapshot_store::open_current(const llama_snapshot_identity & expected_identity) const {
    llama_snapshot_open_result result;
    const operation_result     config_check = config_status(cfg);
    if (config_check.status != llama_snapshot_status::ok || !valid_identity(expected_identity)) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }
    const operation_result loaded = load_manifest(cfg, fs::u8path(cfg.root_path) / "current.manifest",
                                                  llama_snapshot_status::no_current_generation, result.manifest);
    if (loaded.status != llama_snapshot_status::ok) {
        result.status   = loaded.status;
        result.os_error = loaded.os_error;
        return result;
    }
    if (result.manifest.physical_device_id != cfg.physical_device_id ||
        result.manifest.physical_device_queues != cfg.physical_device_queues) {
        result.status = llama_snapshot_status::device_mismatch;
        return result;
    }
    if (result.manifest.identity != expected_identity) {
        result.status = llama_snapshot_status::identity_mismatch;
        return result;
    }
    result.status = llama_snapshot_status::ok;
    return result;
}

llama_snapshot_read_result llama_snapshot_store::read_chunk(const llama_snapshot_manifest & manifest,
                                                            uint32_t                        chunk_index) const {
    llama_snapshot_read_result result;
    const operation_result     shape = validate_manifest_shape(cfg, manifest, true);
    if (shape.status != llama_snapshot_status::ok) {
        result.status = shape.status;
        return result;
    }
    if (chunk_index >= manifest.chunks.size()) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }
    const uint64_t payload_bytes = manifest.chunks[chunk_index].payload_bytes;
    if (payload_bytes > SIZE_MAX) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }
    try {
        result.payload.resize(static_cast<size_t>(payload_bytes));
    } catch (const std::bad_alloc &) {
        result.status   = llama_snapshot_status::io_error;
        result.os_error = ENOMEM;
        return result;
    }
    const llama_snapshot_read_into_result read =
        read_chunk_into(manifest, chunk_index, result.payload.data(), result.payload.size());
    result.status   = read.status;
    result.os_error = read.os_error;
    if (read.status != llama_snapshot_status::ok) {
        result.payload.clear();
    }
    return result;
}

llama_snapshot_read_into_result llama_snapshot_store::read_chunk_into(
        const llama_snapshot_manifest & manifest,
        uint32_t                        chunk_index,
        uint8_t *                       destination,
        size_t                          destination_size) const {
    llama_snapshot_read_into_result result;
    const operation_result          shape = validate_manifest_shape(cfg, manifest, true);
    if (shape.status != llama_snapshot_status::ok) {
        result.status = shape.status;
        return result;
    }
    if (chunk_index >= manifest.chunks.size()) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }
    const llama_snapshot_chunk_info & expected = manifest.chunks[chunk_index];
    if (expected.payload_bytes > destination_size || (expected.payload_bytes != 0 && destination == nullptr)) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }

    const fs::path path = fs::u8path(cfg.root_path) /
                          generation_name(manifest.snapshot_generation, false) / chunk_name(chunk_index);
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        result.status   = errno == ENOENT ? llama_snapshot_status::missing_chunk : llama_snapshot_status::io_error;
        result.os_error = errno;
        return result;
    }
    const auto finish = [&](llama_snapshot_status status, int os_error = 0) {
        llama_snapshot_read_into_result value;
        value.status        = status;
        value.payload_bytes = status == llama_snapshot_status::ok ? expected.payload_bytes : 0;
        value.os_error      = os_error;
        if (::close(descriptor) != 0 && value.status == llama_snapshot_status::ok) {
            value.status        = llama_snapshot_status::io_error;
            value.payload_bytes = 0;
            value.os_error      = errno;
        }
        return value;
    };

    struct stat file_status{};
    if (::fstat(descriptor, &file_status) != 0) {
        return finish(llama_snapshot_status::io_error, errno);
    }
    if (!S_ISREG(file_status.st_mode) || file_status.st_size < 0) {
        return finish(llama_snapshot_status::format_error);
    }
    const uint64_t expected_file_size = LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + expected.payload_bytes;
    if (static_cast<uint64_t>(file_status.st_size) < expected_file_size) {
        return finish(llama_snapshot_status::truncated);
    }
    if (static_cast<uint64_t>(file_status.st_size) > expected_file_size) {
        return finish(llama_snapshot_status::trailing_data);
    }

    std::array<uint8_t, CHUNK_FIXED_FIELDS_BYTES> header{};
    operation_result loaded = pread_all(descriptor, header.data(), header.size(), 0);
    if (loaded.status != llama_snapshot_status::ok) {
        return finish(loaded.status, loaded.os_error);
    }
    llama_snapshot_digest checksum{};
    std::copy_n(header.data() + 40, checksum.size(), checksum.begin());
    if (!std::equal(std::begin(CHUNK_MAGIC), std::end(CHUNK_MAGIC), header.begin()) ||
        read_u32_le(header.data() + 8) != LLAMA_SNAPSHOT_CHUNK_FORMAT_VERSION ||
        read_u32_le(header.data() + 12) != LLAMA_SNAPSHOT_CHUNK_ALIGNMENT ||
        read_u64_le(header.data() + 16) != manifest.snapshot_generation ||
        read_u32_le(header.data() + 24) != expected.index || read_u32_le(header.data() + 28) != 0 ||
        read_u64_le(header.data() + 32) != expected.payload_bytes || checksum != expected.checksum) {
        return finish(llama_snapshot_status::format_error);
    }
    loaded = pread_all(
        descriptor, destination, static_cast<size_t>(expected.payload_bytes), LLAMA_SNAPSHOT_CHUNK_ALIGNMENT);
    if (loaded.status != llama_snapshot_status::ok) {
        return finish(loaded.status, loaded.os_error);
    }
    if (llama_snapshot_sha256(destination, static_cast<size_t>(expected.payload_bytes)) != expected.checksum) {
        return finish(llama_snapshot_status::checksum_mismatch);
    }
    return finish(llama_snapshot_status::ok);
}

llama_snapshot_status llama_snapshot_store::validate(const llama_snapshot_manifest & manifest, int * os_error) const {
    if (os_error != nullptr) {
        *os_error = 0;
    }
    const operation_result shape = validate_manifest_shape(cfg, manifest, true);
    if (shape.status != llama_snapshot_status::ok) {
        return shape.status;
    }
    for (uint32_t index = 0; index < manifest.chunks.size(); ++index) {
        const llama_snapshot_read_result read = read_chunk(manifest, index);
        if (read.status != llama_snapshot_status::ok) {
            if (os_error != nullptr) {
                *os_error = read.os_error;
            }
            return read.status;
        }
    }
    return llama_snapshot_status::ok;
}

llama_snapshot_read_result llama_snapshot_store::read_all(const llama_snapshot_manifest & manifest) const {
    llama_snapshot_read_result result;
    const operation_result     shape = validate_manifest_shape(cfg, manifest, true);
    if (shape.status != llama_snapshot_status::ok) {
        result.status = shape.status;
        return result;
    }
    if (manifest.total_payload_bytes > SIZE_MAX) {
        result.status = llama_snapshot_status::invalid_argument;
        return result;
    }
    try {
        result.payload.reserve(static_cast<size_t>(manifest.total_payload_bytes));
    } catch (const std::bad_alloc &) {
        result.status   = llama_snapshot_status::io_error;
        result.os_error = ENOMEM;
        return result;
    }
    for (uint32_t index = 0; index < manifest.chunks.size(); ++index) {
        llama_snapshot_read_result chunk = read_chunk(manifest, index);
        if (chunk.status != llama_snapshot_status::ok) {
            result.status   = chunk.status;
            result.os_error = chunk.os_error;
            result.payload.clear();
            return result;
        }
        result.payload.insert(result.payload.end(), chunk.payload.begin(), chunk.payload.end());
    }
    result.status = llama_snapshot_status::ok;
    return result;
}

llama_snapshot_cleanup_result llama_snapshot_store::cleanup_temporary_generations() {
    llama_snapshot_cleanup_result result;
    const operation_result        config_check = config_status(cfg);
    if (config_check.status != llama_snapshot_status::ok) {
        result.status = config_check.status;
        return result;
    }
    const fs::path  root = fs::u8path(cfg.root_path);
    std::error_code error;
    if (!fs::exists(root, error)) {
        result.status   = error ? llama_snapshot_status::io_error : llama_snapshot_status::ok;
        result.os_error = error.value();
        return result;
    }
    for (fs::directory_iterator iterator(root, error), end; !error && iterator != end; iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        if (name.rfind(".partial-generation-", 0) != 0 && name != "current.manifest.tmp") {
            continue;
        }
        std::error_code remove_error;
        fs::remove_all(iterator->path(), remove_error);
        if (remove_error) {
            result.status   = llama_snapshot_status::io_error;
            result.os_error = remove_error.value();
            return result;
        }
        ++result.removed;
    }
    if (error) {
        result.status   = llama_snapshot_status::io_error;
        result.os_error = error.value();
        return result;
    }
    const operation_result synced = fsync_directory(root);
    result.status                 = synced.status;
    result.os_error               = synced.os_error;
    return result;
}

#include "ggml-backend.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "sha256/sha256.h"
#ifdef __cplusplus
}
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <tuple>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr uint32_t N_CTX        = 1024;
constexpr uint32_t N_BATCH      = 1024;
constexpr uint32_t N_UBATCH     = 1024;
constexpr uint32_t N_SEQ_MAX    = 3;
constexpr uint32_t N_RS_SEQ     = 5;
constexpr uint32_t N_OUTPUTS    = 3;
constexpr double   NMSE_LIMIT   = 1.0e-8;
constexpr double   MAXABS_LIMIT = 1.0e-5;

constexpr std::array<uint32_t, 9> BOUNDARIES = { 3, 4, 5, 127, 128, 129, 255, 256, 260 };

constexpr uint32_t MODEL_MANIFEST_VERSION = 1;
constexpr uint32_t MODEL_SHARD_COUNT      = 5;
constexpr uint64_t MODEL_TOTAL_FILE_BYTES = UINT64_C(161869615520);
constexpr const char * MODEL_PREFIX       = "DeepSeek-V4-Flash-0731-UD-Q8_K_XL";
constexpr const char * MODEL_ARCH          = "deepseek4";
constexpr const char * MODEL_NAME          = "Deepseek-V4-Flash-0731";
constexpr const char * MODEL_SIZE_LABEL    = "256x8.4B";
constexpr const char * MODEL_QUANT_LABEL   = "UD-Q8_K_XL";
constexpr uint32_t MODEL_QUANT_VERSION    = 2;
constexpr uint32_t MODEL_FILE_TYPE        = 38; // LLAMA_FTYPE_MOSTLY_MXFP4_MOE.
constexpr std::array<uint64_t, MODEL_SHARD_COUNT> MODEL_SHARD_BYTES = {
    UINT64_C(5257408), UINT64_C(49215492960), UINT64_C(49700372160), UINT64_C(49466495968), UINT64_C(13481997024),
};
constexpr std::array<const char *, MODEL_SHARD_COUNT> MODEL_SHARD_SHA256 = {
    "d13ce8f90855547bdaebe7312f531a1f2c4f822178d3103951f27fe884395cfa",
    "3da2f2443063f83635986f9b67fa7e8e3d03c53b81a9a08d2007936612423610",
    "7d622a7760d359ec9257b3493ad531e3bf0bfbe6f6533267e16e6dde8153ddce",
    "6ed2bce452214f156b85e7c5f7d4fc242a3052f409d1b90a61422f60669c2de3",
    "ea4727af4888fdca0fff796ec81ac2f3ebb43c310b2feb4798f41d82744b42ea",
};

// Scope/nonclaims: this gate covers one target context and deterministic token
// batches only. It does not exercise target+draft pairing, rollback depths,
// sampler/grammar/tool/LoRA state, speculative stochastic decoding, server
// pause/resume, asynchronous pressure, or thermal/cooldown behavior.

struct model_shard_manifest {
    uint32_t     index = 0;
    std::string  filename;
    uint64_t     bytes = 0;
    std::string  sha256;
};

struct model_manifest {
    uint32_t                         version = 0;
    std::map<std::string, std::string> fields;
    std::array<model_shard_manifest, MODEL_SHARD_COUNT> shards;
};

struct file_identity {
    uint64_t device      = 0;
    uint64_t inode       = 0;
    uint64_t size        = 0;
    int64_t  mtime_sec   = 0;
    int64_t  mtime_nsec  = 0;

    bool operator==(const file_identity & other) const {
        return device == other.device && inode == other.inode && size == other.size && mtime_sec == other.mtime_sec &&
               mtime_nsec == other.mtime_nsec;
    }
};

struct model_file_snapshot {
    file_identity identity;
    std::string   sha256;
};

using model_file_snapshots = std::array<model_file_snapshot, MODEL_SHARD_COUNT>;

using model_ptr   = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

struct graph_trace_key {
    uint64_t    split   = 0;
    std::string node;
    std::string op;
    std::string backend;

    bool operator<(const graph_trace_key & other) const {
        return std::tie(split, node, op, backend) < std::tie(other.split, other.node, other.op, other.backend);
    }
};

struct graph_trace_bucket {
    uint64_t asks        = 0;
    uint64_t evaluations = 0;
};

struct graph_trace {
    uint64_t evaluations = 0;
    uint64_t asks        = 0;
    uint64_t splits      = 0;
    std::map<graph_trace_key, graph_trace_bucket> histogram;
};

struct graph_counter {
    uint64_t   evaluations = 0;
    uint64_t   asks        = 0;
    bool       trace       = false;
    bool       trace_split_open = false;
    uint64_t   trace_next_split = 0;
    graph_trace trace_data;
};

struct phase_evaluations {
    uint64_t prompt_evaluations  = 0;
    uint64_t parked_evaluations  = 0;
    uint64_t resumed_evaluations = 0;
    uint64_t post_evaluations    = 0;
    uint64_t prompt_asks         = 0;
    uint64_t parked_asks         = 0;
    uint64_t resumed_asks        = 0;
    uint64_t post_asks           = 0;
};

std::string graph_trace_node_name(const ggml_tensor * tensor) {
    const char * name = tensor != nullptr ? ggml_get_name(tensor) : nullptr;
    return name != nullptr && name[0] != '\0' ? name : "(unnamed)";
}

std::string graph_trace_op_name(const ggml_tensor * tensor) {
    return tensor != nullptr ? ggml_op_name(tensor->op) : "(unknown)";
}

std::string graph_trace_backend_name(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return "(unbound)";
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    const ggml_backend_dev_t device = buft != nullptr ? ggml_backend_buft_get_device(buft) : nullptr;
    if (device != nullptr) {
        const char * name = ggml_backend_dev_name(device);
        if (name != nullptr && name[0] != '\0') {
            return name;
        }
    }
    const char * name = buft != nullptr ? ggml_backend_buft_name(buft) : nullptr;
    return name != nullptr && name[0] != '\0' ? name : "(unknown)";
}

void reset_graph_trace(graph_counter & counter, bool enabled) {
    counter.trace = enabled;
    counter.trace_split_open = false;
    counter.trace_next_split = 0;
    counter.trace_data = {};
}

bool count_graph_evaluations(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * counter = static_cast<graph_counter *>(user_data);
    if (ask) {
        ++counter->asks;
    } else {
        ++counter->evaluations;
    }
    if (counter->trace) {
        if (ask && !counter->trace_split_open) {
            counter->trace_split_open = true;
            ++counter->trace_next_split;
        }
        const uint64_t split = counter->trace_next_split > 0 ? counter->trace_next_split - 1 : 0;
        auto & bucket = counter->trace_data.histogram[{ split, graph_trace_node_name(tensor), graph_trace_op_name(tensor),
                                                          graph_trace_backend_name(tensor) }];
        if (ask) {
            ++counter->trace_data.asks;
            ++bucket.asks;
        } else {
            ++counter->trace_data.evaluations;
            ++bucket.evaluations;
            counter->trace_split_open = false;
        }
        counter->trace_data.splits = counter->trace_next_split;
    }
    return true;
}

void print_graph_trace(uint32_t boundary, const char * phase, const char * side, const graph_trace & trace) {
    std::fprintf(stderr,
                 "resident-model diagnostic graph-count boundary=%u phase=%s side=%s evaluations=%llu asks=%llu "
                 "callback_splits=%llu histogram_entries=%zu\n",
                 boundary, phase, side, (unsigned long long) trace.evaluations, (unsigned long long) trace.asks,
                 (unsigned long long) trace.splits, trace.histogram.size());
    for (const auto & [key, bucket] : trace.histogram) {
        std::fprintf(stderr,
                     "resident-model diagnostic graph-hist boundary=%u phase=%s side=%s split=%llu node=%s op=%s "
                     "backend=%s asks=%llu evaluations=%llu\n",
                     boundary, phase, side, (unsigned long long) key.split, key.node.c_str(), key.op.c_str(),
                     key.backend.c_str(), (unsigned long long) bucket.asks, (unsigned long long) bucket.evaluations);
    }
}

const char * dsv4_memory_family_name(llama_dsv4_memory_family family) {
    switch (family) {
        case LLAMA_DSV4_MEMORY_RAW: return "raw";
        case LLAMA_DSV4_MEMORY_CSA: return "csa";
        case LLAMA_DSV4_MEMORY_HCA: return "hca";
        case LLAMA_DSV4_MEMORY_LID: return "lid";
        case LLAMA_DSV4_MEMORY_FAMILY_COUNT: break;
    }
    return "unknown";
}

void print_sparse_pool_usage(
        uint32_t boundary,
        const char * label,
        const char * family,
        size_t pool_index,
        const llama_dsv4_sparse_pool_usage & usage,
        const llama_dsv4_sparse_pool_usage * before = nullptr) {
    const auto delta = [&](uint64_t value, uint64_t prior) -> int64_t {
        return value >= prior ? (int64_t) (value - prior) : -(int64_t) (prior - value);
    };
    std::fprintf(stderr,
                 "resident-model diagnostic sparse-pool boundary=%u label=%s family=%s index=%zu "
                 "pool=%llu page=%llu virtual=%llu physical=%llu free=%llu reserved=%llu mapped=%llu "
                 "unique=%llu shared_physical=%llu shared_mappings=%llu refsum=%llu refmax=%u "
                 "generation=%llu cow_alloc=%llu cow_pages=%llu",
                 boundary, label, family, pool_index,
                 (unsigned long long) usage.pool_id,
                 (unsigned long long) usage.page_size,
                 (unsigned long long) usage.virtual_pages,
                 (unsigned long long) usage.physical_pages,
                 (unsigned long long) usage.free_pages,
                 (unsigned long long) usage.reserved_pages,
                 (unsigned long long) usage.mapped_mappings,
                 (unsigned long long) usage.unique_physical_pages,
                 (unsigned long long) usage.shared_physical_pages,
                 (unsigned long long) usage.shared_mappings,
                 (unsigned long long) usage.refcount_sum,
                 usage.refcount_max,
                 (unsigned long long) usage.generation,
                 (unsigned long long) usage.cow_allocations,
                 (unsigned long long) usage.cow_pages);
    if (before != nullptr) {
        std::fprintf(stderr,
                     " delta_free=%lld delta_reserved=%lld delta_mapped=%lld delta_unique=%lld "
                     "delta_shared_physical=%lld delta_shared_mappings=%lld delta_refsum=%lld "
                     "delta_generation=%lld delta_cow_alloc=%lld delta_cow_pages=%lld",
                     (long long) delta(usage.free_pages, before->free_pages),
                     (long long) delta(usage.reserved_pages, before->reserved_pages),
                     (long long) delta(usage.mapped_mappings, before->mapped_mappings),
                     (long long) delta(usage.unique_physical_pages, before->unique_physical_pages),
                     (long long) delta(usage.shared_physical_pages, before->shared_physical_pages),
                     (long long) delta(usage.shared_mappings, before->shared_mappings),
                     (long long) delta(usage.refcount_sum, before->refcount_sum),
                     (long long) delta(usage.generation, before->generation),
                     (long long) delta(usage.cow_allocations, before->cow_allocations),
                     (long long) delta(usage.cow_pages, before->cow_pages));
    }
    std::fprintf(stderr, "\n");
}

void print_sparse_snapshot(
        uint32_t boundary,
        const char * label,
        const llama_dsv4_memory_usage_snapshot & snapshot,
        const llama_dsv4_memory_usage_snapshot * before = nullptr) {
    std::fprintf(stderr,
                 "resident-model diagnostic sparse-snapshot boundary=%u label=%s "
                 "limiting_family=%s limiting_mask=%u limiting_pool=%llu limiting_available=%llu\n",
                 boundary, label, dsv4_memory_family_name(snapshot.limiting_family),
                 snapshot.limiting_family_mask,
                 (unsigned long long) snapshot.limiting_pool_id,
                 (unsigned long long) snapshot.limiting_available_pages);
    for (size_t i = 0; i < snapshot.families.size(); ++i) {
        const auto & family = snapshot.families[i];
        const llama_dsv4_family_usage * before_family =
                before != nullptr ? &before->families[i] : nullptr;
        for (size_t p = 0; p < family.pools.size(); ++p) {
            const llama_dsv4_sparse_pool_usage * prior = nullptr;
            if (before_family != nullptr && p < before_family->pools.size() &&
                    before_family->pools[p].pool_id == family.pools[p].pool_id) {
                prior = &before_family->pools[p];
            }
            print_sparse_pool_usage(boundary, label, dsv4_memory_family_name(family.family), p,
                                    family.pools[p], prior);
        }
        if (!family.pools.empty()) {
            const llama_dsv4_sparse_pool_usage * prior =
                    before_family != nullptr ? &before_family->total : nullptr;
            print_sparse_pool_usage(boundary, label, dsv4_memory_family_name(family.family), SIZE_MAX,
                                    family.total, prior);
        }
    }
    print_sparse_pool_usage(boundary, label, "all", SIZE_MAX, snapshot.sparse_total,
                            before != nullptr ? &before->sparse_total : nullptr);
}

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> split(const std::string & value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream        stream(value);
    std::string              item;
    while (std::getline(stream, item, delimiter)) {
        result.push_back(item);
    }
    return result;
}

uint64_t parse_u64(const std::string & value, const char * field) {
    size_t  consumed = 0;
    uint64_t result   = 0;
    try {
        result = std::stoull(value, &consumed, 10);
    } catch (...) {
        fail(std::string("manifest ") + field + " is not an unsigned integer");
    }
    expect(consumed == value.size(), std::string("manifest ") + field + " has trailing data");
    return result;
}

void expect_manifest_field(const model_manifest & manifest, const char * key, const char * expected) {
    const auto it = manifest.fields.find(key);
    expect(it != manifest.fields.end() && it->second == expected,
           std::string("pinned model manifest ") + key + " mismatch");
}

model_manifest read_model_manifest(const std::string & manifest_path) {
    std::ifstream input(manifest_path);
    expect(input.good(), "pinned model manifest could not be opened");

    model_manifest result;
    std::string   line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("shard=", 0) == 0) {
            const auto fields = split(line.substr(6), '|');
            expect(fields.size() == 4, "pinned model manifest shard entry is malformed");
            const uint32_t index = (uint32_t) parse_u64(fields[0], "shard index");
            expect(index >= 1 && index <= MODEL_SHARD_COUNT, "pinned model manifest shard index is out of range");
            auto & shard = result.shards[index - 1];
            expect(shard.index == 0, "pinned model manifest contains a duplicate shard");
            shard.index    = index;
            shard.filename = trim(fields[1]);
            shard.bytes    = parse_u64(fields[2], "shard bytes");
            shard.sha256   = trim(fields[3]);
            continue;
        }
        const auto separator = line.find('=');
        expect(separator != std::string::npos && separator != 0, "pinned model manifest field is malformed");
        const std::string key   = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        expect(!key.empty() && !value.empty(), "pinned model manifest has an empty field");
        expect(result.fields.emplace(key, value).second, "pinned model manifest contains a duplicate field");
    }

    expect(input.eof(), "pinned model manifest read failed");
    expect_manifest_field(result, "model_prefix", MODEL_PREFIX);
    expect_manifest_field(result, "architecture", MODEL_ARCH);
    expect_manifest_field(result, "name", MODEL_NAME);
    expect_manifest_field(result, "size_label", MODEL_SIZE_LABEL);
    expect_manifest_field(result, "quantization_label", MODEL_QUANT_LABEL);
    expect_manifest_field(result, "quantization_version", "2");
    expect_manifest_field(result, "file_type", "38");
    expect_manifest_field(result, "total_file_bytes", "161869615520");
    expect_manifest_field(result, "shard_count", "5");
    result.version = (uint32_t) parse_u64(result.fields.at("manifest_version"), "manifest_version");
    expect(result.version == MODEL_MANIFEST_VERSION, "pinned model manifest version is unsupported");

    uint64_t total_bytes = 0;
    for (uint32_t index = 0; index < MODEL_SHARD_COUNT; ++index) {
        const auto & shard = result.shards[index];
        expect(shard.index == index + 1, "pinned model manifest is missing a shard");
        const std::string expected_name = std::string(MODEL_PREFIX) + "-" +
                                           (index < 9 ? "0000" : "000") + std::to_string(index + 1) + "-of-00005.gguf";
        expect(shard.filename == expected_name, "pinned model manifest shard filename mismatch");
        expect(shard.bytes == MODEL_SHARD_BYTES[index], "pinned model manifest shard size is not pinned");
        expect(shard.sha256 == MODEL_SHARD_SHA256[index], "pinned model manifest shard digest is not pinned");
        expect(shard.sha256.size() == 64, "pinned model manifest shard digest length mismatch");
        for (const char digit : shard.sha256) {
            expect((digit >= '0' && digit <= '9') || (digit >= 'a' && digit <= 'f'),
                   "pinned model manifest shard digest is not lowercase hexadecimal");
        }
        total_bytes += shard.bytes;
    }
    expect(total_bytes == MODEL_TOTAL_FILE_BYTES, "pinned model manifest shard sizes do not add up");
    return result;
}

file_identity file_identity_from_stat(const struct stat & info, const std::string & path) {
    expect(S_ISREG(info.st_mode), "pinned model shard is not a regular file: " + path);
    expect(info.st_size >= 0, "pinned model shard has a negative size: " + path);
    file_identity result;
    result.device = (uint64_t) info.st_dev;
    result.inode  = (uint64_t) info.st_ino;
    result.size   = (uint64_t) info.st_size;
#if defined(__APPLE__)
    result.mtime_sec  = (int64_t) info.st_mtimespec.tv_sec;
    result.mtime_nsec = (int64_t) info.st_mtimespec.tv_nsec;
#else
    result.mtime_sec  = (int64_t) info.st_mtim.tv_sec;
    result.mtime_nsec = (int64_t) info.st_mtim.tv_nsec;
#endif
    return result;
}

file_identity path_identity(const std::filesystem::path & path) {
    struct stat info = {};
    const int   status = ::lstat(path.c_str(), &info);
    if (status != 0) {
        const int error = errno;
        fail("pinned model shard could not be lstat'ed: " + path.string() + ": " + std::strerror(error));
    }
    expect(!S_ISLNK(info.st_mode), "pinned model shard is a symlink: " + path.string());
    return file_identity_from_stat(info, path.string());
}

struct descriptor_guard {
    int fd = -1;

    descriptor_guard() = default;
    descriptor_guard(const descriptor_guard &) = delete;
    descriptor_guard & operator=(const descriptor_guard &) = delete;
    descriptor_guard(descriptor_guard && other) noexcept : fd(std::exchange(other.fd, -1)) {}
    descriptor_guard & operator=(descriptor_guard && other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                (void) ::close(fd);
            }
            fd = std::exchange(other.fd, -1);
        }
        return *this;
    }

    ~descriptor_guard() {
        if (fd >= 0) {
            (void) ::close(fd);
        }
    }
};

descriptor_guard open_model_shard(const std::filesystem::path & path) {
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    descriptor_guard result;
    result.fd = ::open(path.c_str(), flags);
    if (result.fd < 0) {
        const int error = errno;
        fail("pinned model shard could not be opened: " + path.string() + ": " + std::strerror(error));
    }
    return result;
}

file_identity descriptor_identity(int fd, const std::filesystem::path & path) {
    struct stat info = {};
    if (::fstat(fd, &info) != 0) {
        const int error = errno;
        fail("pinned model shard descriptor could not be stat'ed: " + path.string() + ": " + std::strerror(error));
    }
    return file_identity_from_stat(info, path.string());
}

std::string sha256_hex(const unsigned char * digest, size_t size) {
    static constexpr char HEX[] = "0123456789abcdef";
    std::string           result;
    result.reserve(size * 2);
    for (size_t index = 0; index < size; ++index) {
        result.push_back(HEX[digest[index] >> 4]);
        result.push_back(HEX[digest[index] & 0x0f]);
    }
    return result;
}

std::string sha256_descriptor(int fd, uint64_t expected_bytes, const std::filesystem::path & path) {
    sha256_t hash;
    sha256_init(&hash);
    std::array<unsigned char, 1 << 20> buffer = {};
    uint64_t                           total  = 0;
    for (;;) {
        const ssize_t read_bytes = ::read(fd, buffer.data(), buffer.size());
        if (read_bytes < 0 && errno == EINTR) {
            continue;
        }
        if (read_bytes < 0) {
            const int error = errno;
            fail("pinned model shard read failed: " + path.string() + ": " + std::strerror(error));
        }
        if (read_bytes == 0) {
            break;
        }
        expect(total + (uint64_t) read_bytes <= expected_bytes,
               "pinned model shard grew while hashing: " + path.string());
        sha256_update(&hash, buffer.data(), (size_t) read_bytes);
        total += (uint64_t) read_bytes;
    }
    expect(total == expected_bytes, "pinned model shard size changed while hashing: " + path.string());
    unsigned char digest[SHA256_DIGEST_SIZE] = {};
    sha256_final(&hash, digest);
    return sha256_hex(digest, sizeof(digest));
}

model_file_snapshots verify_model_files(const std::string &                 model_path,
                                        const model_manifest &               manifest,
                                        const model_file_snapshots *          expected_snapshots = nullptr,
                                        const char *                          stage = "pre-load",
                                        bool                                   skip_digests = false) {
    const std::filesystem::path first = std::filesystem::path(model_path);
    expect(first.filename() == manifest.shards[0].filename,
           "exact-model gate requires the pinned canonical first shard filename");

    model_file_snapshots snapshots;
    uint64_t total_bytes = 0;
    for (size_t index = 0; index < manifest.shards.size(); ++index) {
        const auto & shard = manifest.shards[index];
        const std::filesystem::path path = first.parent_path() / shard.filename;
        const file_identity path_before = path_identity(path);
        descriptor_guard guard           = open_model_shard(path);
        const file_identity descriptor_before = descriptor_identity(guard.fd, path);
        expect(path_before == descriptor_before,
               std::string(stage) + " model shard path changed before validation: " + path.string());
        expect(descriptor_before.size == shard.bytes, "pinned model shard byte size mismatch: " + path.string());
        const std::string digest = skip_digests ? std::string() : sha256_descriptor(guard.fd, shard.bytes, path);
        const file_identity descriptor_after = descriptor_identity(guard.fd, path);
        const file_identity path_after       = path_identity(path);
        expect(descriptor_after == descriptor_before && path_after == descriptor_before,
               std::string(stage) + " model shard changed while validating: " + path.string());
        if (!skip_digests) {
            expect(digest == shard.sha256, "pinned model shard SHA-256 mismatch: " + path.string());
        }
        if (expected_snapshots != nullptr) {
            expect(descriptor_after == (*expected_snapshots)[index].identity,
                   std::string(stage) + " model shard identity changed after load: " + path.string());
            if (!skip_digests) {
                expect(digest == (*expected_snapshots)[index].sha256,
                       std::string(stage) + " model shard digest changed after load: " + path.string());
            }
        }
        snapshots[index] = { descriptor_after, digest };
        std::fprintf(stderr,
                     "resident-model identity stage=%s shard=%u name=%s device=%llu inode=%llu bytes=%llu "
                     "mtime=%lld.%09lld sha256=%s\n",
                     stage, shard.index, shard.filename.c_str(), (unsigned long long) descriptor_after.device,
                     (unsigned long long) descriptor_after.inode, (unsigned long long) descriptor_after.size,
                     (long long) descriptor_after.mtime_sec, (long long) descriptor_after.mtime_nsec,
                     skip_digests ? "(skipped)" : digest.c_str());
        total_bytes += descriptor_after.size;
    }
    expect(total_bytes == MODEL_TOTAL_FILE_BYTES, "pinned model total file size mismatch");
    return snapshots;
}

ggml_backend_dev_t verify_target_metal_device() {
    // The in-tree Metal backend advertises the historical short registry name
    // "MTL" (GGML_METAL_NAME); accept the older "Metal" spelling as a
    // compatibility fallback for alternate backend builds.
    ggml_backend_reg_t metal = ggml_backend_reg_by_name("MTL");
    if (metal == nullptr) {
        metal = ggml_backend_reg_by_name("Metal");
    }
    expect(metal != nullptr, "exact-model gate requires the Metal/MTL backend registry");
    ggml_backend_dev_t target = nullptr;
    for (size_t index = 0; index < ggml_backend_reg_dev_count(metal); ++index) {
        ggml_backend_dev_t device = ggml_backend_reg_dev_get(metal, index);
        const char *        name = ggml_backend_dev_name(device);
        const char *        desc = ggml_backend_dev_description(device);
        std::fprintf(stderr, "resident-model backend=%s device=%s description=%s\n", ggml_backend_reg_name(metal),
                     name != nullptr ? name : "(null)", desc != nullptr ? desc : "(null)");
        if (desc != nullptr && std::strcmp(desc, "Apple M2 Ultra") == 0) {
            expect(ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU,
                   "exact-model M2 Ultra device is not reported as a GPU");
            expect(target == nullptr, "exact-model gate found multiple Apple M2 Ultra Metal devices");
            target = device;
        }
    }
    expect(target != nullptr, "exact-model gate requires an Apple M2 Ultra Metal device");
    return target;
}

enum class placement_buft_kind {
    target,
    cpu,
    unknown,
};

// llama_model_base::load_tensors() builds its CPU list from the canonical CPU
// device returned by ggml_backend_dev_by_type(CPU).  The ordinary CPU and
// CPU_Mapped buffer types intentionally have a null buft->device today, so a
// device-only check would reject valid host-resident tensors.  Keep the
// accepted null-device aliases narrow: canonical CPU/CPU_Mapped names (and
// explicitly reported host types) are accepted; an arbitrary null-device
// buffer remains unknown and fails closed.
bool is_cpu_buft_alias(ggml_backend_buffer_type_t buft, ggml_backend_dev_t cpu_device,
                       ggml_backend_dev_t target_device) {
    if (buft == nullptr) {
        return false;
    }
    if (buft == ggml_backend_cpu_buffer_type()) {
        return true;
    }
    if (cpu_device == nullptr) {
        return false;
    }
    if (buft == ggml_backend_dev_buffer_type(cpu_device)) {
        return true;
    }

    // Extra CPU buffer types (for example CPU repack/AMX/HBM) are associated
    // with the canonical CPU device by make_cpu_buft_list().
    const ggml_backend_dev_t buft_device = ggml_backend_buft_get_device(buft);
    if (buft_device == cpu_device) {
        return true;
    }
    if (buft_device != nullptr) {
        return false;
    }

    // CPU_Mapped is an internal type with no public constructor.  Its stable
    // public identity is the host flag and name, which is what the CPU
    // backend exposes through ggml_backend_dev_buffer_from_host_ptr().
    const char * name = ggml_backend_buft_name(buft);
    const bool canonical_null_host = ggml_backend_buft_is_host(buft) && name != nullptr &&
                                     (std::strcmp(name, "CPU") == 0 || std::strcmp(name, "CPU_Mapped") == 0);
    if (canonical_null_host) {
        return true;
    }

    // Some backends expose a host buffer type with no owning device.  Treat
    // only the host aliases selected by load_tensors() as CPU placement.
    if (target_device != nullptr && buft == ggml_backend_dev_host_buffer_type(target_device)) {
        return true;
    }
    return buft == ggml_backend_dev_host_buffer_type(cpu_device);
}

void prompt_batch_geometry_self_test();

placement_buft_kind classify_placement_buft(ggml_backend_buffer_type_t buft, ggml_backend_dev_t target_device) {
    if (buft == nullptr || target_device == nullptr) {
        return placement_buft_kind::unknown;
    }
    const ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu_device == nullptr) {
        return placement_buft_kind::unknown;
    }
    const ggml_backend_dev_t device = ggml_backend_buft_get_device(buft);
    if (device == target_device) {
        return placement_buft_kind::target;
    }
    if (is_cpu_buft_alias(buft, cpu_device, target_device)) {
        return placement_buft_kind::cpu;
    }
    return placement_buft_kind::unknown;
}

void placement_buft_self_test() {
    // Keep this helper runnable on hosts whose selected CPU dispatch library
    // cannot execute the target build's optimized kernels.  The production
    // placement path supplies the canonical CPU device after backend init;
    // this focused check exercises the null-device classification itself.
    const ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    expect(is_cpu_buft_alias(cpu_buft, nullptr, nullptr),
           "placement self-test rejected the canonical null-device CPU buffer type");
    expect(!is_cpu_buft_alias(nullptr, nullptr, nullptr),
           "placement self-test accepted an unknown null buffer type");
    prompt_batch_geometry_self_test();
    std::fprintf(stderr, "resident-model placement self-test canonical_cpu=%s unknown_null=rejected\n",
                 ggml_backend_buft_name(cpu_buft));
}

void verify_model_placement(const llama_model * model, ggml_backend_dev_t target) {
    expect(llama_model_n_devices(model) == 1, "exact-model loaded model selected an unexpected device count");
    expect(llama_model_get_device(model, 0) == target,
           "exact-model loaded model did not select the verified Apple M2 Ultra Metal device");

    const auto & tensors = llama_internal_get_tensor_map(model);
    expect(!tensors.empty(), "exact-model loaded model has no tensors to verify for device placement");
    size_t target_tensors = 0;
    size_t cpu_tensors    = 0;
    uint64_t target_bytes = 0;
    uint64_t cpu_bytes    = 0;
    std::map<std::string, std::pair<size_t, uint64_t>> cpu_buft_usage;
    for (const auto & item : tensors) {
        const ggml_tensor * tensor = item.second;
        expect(tensor != nullptr && tensor->buffer != nullptr,
               "exact-model loaded tensor has no backend buffer: " + item.first);
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        const ggml_backend_dev_t device = buft != nullptr ? ggml_backend_buft_get_device(buft) : nullptr;
        const uint64_t bytes = ggml_nbytes(tensor);
        const placement_buft_kind kind = classify_placement_buft(buft, target);
        if (kind == placement_buft_kind::target) {
            ++target_tensors;
            target_bytes += bytes;
        } else if (kind == placement_buft_kind::cpu) {
            ++cpu_tensors;
            cpu_bytes += bytes;
            const char * buft_name = buft != nullptr ? ggml_backend_buft_name(buft) : nullptr;
            auto & usage = cpu_buft_usage[buft_name != nullptr ? buft_name : "(unnamed CPU host)"];
            ++usage.first;
            usage.second += bytes;
        } else {
            const char * buft_name = buft != nullptr ? ggml_backend_buft_name(buft) : nullptr;
            fail("exact-model loaded tensor is placed on an unexpected non-Metal device/buffer: " + item.first +
                 " buft=" + (buft_name != nullptr ? buft_name : "(null)") +
                 " device=" + (device != nullptr ? ggml_backend_dev_name(device) : "(null)"));
        }
    }
    expect(target_tensors > 0 && target_bytes > 0,
           "exact-model loaded tensors did not allocate any bytes on the Apple M2 Ultra Metal device");
    std::fprintf(stderr,
                 "resident-model placement device=%s description=%s metal_tensors=%zu metal_bytes=%llu "
                 "cpu_tensors=%zu cpu_bytes=%llu\n",
                 ggml_backend_dev_name(target), ggml_backend_dev_description(target), target_tensors,
                 (unsigned long long) target_bytes, cpu_tensors, (unsigned long long) cpu_bytes);
    for (const auto & [name, usage] : cpu_buft_usage) {
        std::fprintf(stderr, "resident-model placement cpu-buft=%s tensors=%zu bytes=%llu\n", name.c_str(),
                     usage.first, (unsigned long long) usage.second);
    }
}

std::string model_meta(const llama_model * model, const char * key) {
    char value[256] = {};
    expect(llama_model_meta_val_str(model, key, value, sizeof(value)) >= 0,
           std::string("exact-model metadata key is missing: ") + key);
    return value;
}

void verify_model_metadata(const llama_model * model, const model_manifest & manifest) {
    // The manifest quantization_label is the official artifact label. GGUF's
    // authoritative quantization checks are quantization_version, file_type,
    // and the runtime llama_model_ftype value below.
    expect(model_meta(model, "general.architecture") == MODEL_ARCH, "exact-model architecture metadata mismatch");
    expect(model_meta(model, "general.name") == MODEL_NAME, "exact-model name metadata mismatch");
    expect(model_meta(model, "general.size_label") == MODEL_SIZE_LABEL,
           "exact-model size-label metadata mismatch");
    expect(model_meta(model, "general.quantization_version") == std::to_string(MODEL_QUANT_VERSION),
           "exact-model quantization-version metadata mismatch");
    expect(model_meta(model, "general.file_type") == std::to_string(MODEL_FILE_TYPE),
           "exact-model file-type metadata mismatch");
    expect(llama_model_ftype(model) == (llama_ftype) MODEL_FILE_TYPE,
           "exact-model runtime quantization type mismatch");
    const uint64_t model_bytes = llama_model_size(model);
    expect(model_bytes > 0 && model_bytes <= MODEL_TOTAL_FILE_BYTES,
           "exact-model loaded tensor size is outside pinned file-size bounds");
    std::fprintf(stderr,
                 "resident-model identity architecture=%s name=%s size_label=%s quantization=%s file_type=%u "
                 "tensor_bytes=%llu manifest_bytes=%llu\n",
                 MODEL_ARCH, MODEL_NAME, MODEL_SIZE_LABEL, MODEL_QUANT_LABEL, MODEL_FILE_TYPE,
                 (unsigned long long) model_bytes,
                 (unsigned long long) parse_u64(manifest.fields.at("total_file_bytes"), "total_file_bytes"));
}

bool env_is_one(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct resident_cleanup {
    llama_kv_cache_dsv4 *      memory = nullptr;
    llama_dsv4_resident_handle resident;
    bool                       active = false;

    explicit resident_cleanup(llama_kv_cache_dsv4 * memory) : memory(memory) {}

    void adopt(const llama_dsv4_resident_handle & handle) {
        expect(!active, "resident cleanup already owns a handle");
        resident = handle;
        active   = true;
    }

    void disarm() { active = false; }

    ~resident_cleanup() noexcept {
        if (!active) {
            return;
        }
        try {
            const auto status = memory->release_resident(resident);
            if (status != llama_dsv4_resident_status::ok) {
                std::fprintf(stderr, "resident-model cleanup failed status=%u\n", (unsigned) status);
                std::terminate();
            }
        } catch (...) {
            std::fprintf(stderr, "resident-model cleanup threw while releasing a live handle\n");
            std::terminate();
        }
    }
};

struct row_input {
    uint32_t     role     = 0;
    llama_seq_id sequence = -1;
    llama_pos    position = 0;
    llama_token  token    = 0;
    bool         output   = false;
};

bool prompt_batch_one_split_eligible(const std::vector<row_input> & rows,
                                     llama_seq_id                    source_sequence,
                                     llama_seq_id                    survivor_sequence,
                                     uint32_t                        boundary) {
    if (source_sequence < 0 || survivor_sequence < 0 || source_sequence >= (llama_seq_id) N_SEQ_MAX ||
        survivor_sequence >= (llama_seq_id) N_SEQ_MAX || source_sequence == survivor_sequence ||
        rows.size() != 2u * (boundary + 1u)) {
        return false;
    }

    const llama_seq_id first_sequence = std::min(source_sequence, survivor_sequence);
    const llama_seq_id last_sequence  = std::max(source_sequence, survivor_sequence);
    if (last_sequence != first_sequence + 1) {
        return false;
    }

    std::array<uint32_t, N_SEQ_MAX> next_position = {};
    std::vector<llama_seq_id>        sequence_order;
    sequence_order.reserve(2);
    for (const auto & row : rows) {
        if (row.sequence != source_sequence && row.sequence != survivor_sequence) {
            return false;
        }
        if (sequence_order.empty() || sequence_order.back() != row.sequence) {
            if (!sequence_order.empty() && row.sequence != sequence_order.back() + 1) {
                return false;
            }
            sequence_order.push_back(row.sequence);
        }
        const uint32_t expected_role = row.sequence == source_sequence ? 0u : 1u;
        if (row.role != expected_role || row.position != (llama_pos) next_position[(size_t) row.sequence] ||
            row.position > (llama_pos) boundary) {
            return false;
        }
        ++next_position[(size_t) row.sequence];
    }

    return sequence_order.size() == 2 && sequence_order[0] == first_sequence && sequence_order[1] == last_sequence &&
           next_position[(size_t) source_sequence] == boundary + 1 &&
           next_position[(size_t) survivor_sequence] == boundary + 1;
}

struct model_plan;

std::vector<row_input> make_prompt_rows(const model_plan & plan,
                                        uint32_t            boundary,
                                        llama_seq_id        source_sequence,
                                        llama_seq_id        survivor_sequence);

void prompt_batch_geometry_self_test() {
    constexpr uint32_t boundary = 3;
    const auto make_unsorted_rows = [](llama_seq_id source_sequence, llama_seq_id survivor_sequence) {
        constexpr uint32_t self_test_boundary = 3;
        std::vector<row_input> rows;
        for (uint32_t position = 0; position <= self_test_boundary; ++position) {
            rows.push_back({ 0, source_sequence, (llama_pos) position, 0, position == self_test_boundary });
        }
        for (uint32_t position = 0; position <= self_test_boundary; ++position) {
            rows.push_back({ 1, survivor_sequence, (llama_pos) position, 0, position == self_test_boundary });
        }
        return rows;
    };

    auto descending = make_unsorted_rows(2, 1);
    expect(!prompt_batch_one_split_eligible(descending, 2, 1, boundary),
           "descending source=2/survivor=1 rows unexpectedly passed one-split check");
    std::stable_sort(descending.begin(), descending.end(),
                     [](const row_input & lhs, const row_input & rhs) { return lhs.sequence < rhs.sequence; });
    expect(prompt_batch_one_split_eligible(descending, 2, 1, boundary),
           "sorted source=2/survivor=1 rows failed one-split check");
    const auto ascending = make_unsorted_rows(0, 1);
    expect(prompt_batch_one_split_eligible(ascending, 0, 1, boundary),
           "ascending source=0/survivor=1 rows failed one-split check");
    std::fprintf(stderr,
                 "resident-model prompt-geometry-self-test descending=2,1 raw=two-splits sorted=1,2 "
                 "expected_one_split=1 ascending=0,1 expected_one_split=1\n");
}

struct phase_logits {
    std::array<std::vector<float>, 3> values;
};

// This records rows submitted to llama_decode. It is intentionally not an
// authoritative KV-write or recomputation signal: backend graph work is
// allowed to fuse, replay, or defer operations behind one submission.
struct submission_ledger {
    std::array<std::vector<uint32_t>, N_SEQ_MAX> writes;

    void record(llama_seq_id sequence, llama_pos position) {
        expect(sequence >= 0 && sequence < (llama_seq_id) N_SEQ_MAX, "submission sequence out of range");
        expect(position >= 0, "submission position out of range");
        auto & sequence_writes = writes[(size_t) sequence];
        if ((size_t) position >= sequence_writes.size()) {
            sequence_writes.resize((size_t) position + 1, 0);
        }
        ++sequence_writes[(size_t) position];
    }

    uint32_t count(llama_seq_id sequence, llama_pos position) const {
        if (sequence < 0 || sequence >= (llama_seq_id) N_SEQ_MAX || position < 0) {
            return 0;
        }
        const auto & sequence_writes = writes[(size_t) sequence];
        return (size_t) position < sequence_writes.size() ? sequence_writes[(size_t) position] : 0;
    }
};

struct model_plan {
    uint32_t                                n_vocab = 0;
    std::array<std::vector<llama_token>, 3> tokens;

    llama_token token(uint32_t role, uint32_t position) const {
        expect(role < tokens.size() && position < tokens[role].size(), "token plan index out of range");
        return tokens[role][position];
    }
};

model_plan make_model_plan(uint32_t n_vocab) {
    expect(n_vocab > 1, "model vocabulary is too small");
    model_plan plan;
    plan.n_vocab                    = n_vocab;
    constexpr uint32_t MAX_POSITION = 300;
    for (uint32_t role = 0; role < plan.tokens.size(); ++role) {
        plan.tokens[role].resize(MAX_POSITION);
        for (uint32_t position = 0; position < MAX_POSITION; ++position) {
            const uint64_t value = UINT64_C(0x9e3779b97f4a7c15) + UINT64_C(0x100000001b3) * (role + 1) +
                                   UINT64_C(0x5851f42d4c957f2d) * (position + 1);
            plan.tokens[role][position] = 1 + (llama_token) (value % (n_vocab - 1));
        }
    }
    return plan;
}

std::vector<row_input> make_prompt_rows(const model_plan & plan,
                                        uint32_t            boundary,
                                        llama_seq_id        source_sequence,
                                        llama_seq_id        survivor_sequence) {
    std::vector<row_input> rows;
    rows.reserve(2 * (boundary + 1));
    const auto append_role = [&](uint32_t role, llama_seq_id sequence) {
        for (uint32_t position = 0; position <= boundary; ++position) {
            rows.push_back({ role, sequence, (llama_pos) position, plan.token(role, position), position == boundary });
        }
    };

    // llama_kv_cache_dsv4 uses split_equal(..., sequential=true) for its
    // per-sequence state. Emit the two roles in ascending adjacent sequence
    // order so both oracle and candidate prompts are eligible for one ubatch,
    // while retaining each context's source/survivor IDs for later lifecycle
    // assertions.
    if (source_sequence < survivor_sequence) {
        append_role(0, source_sequence);
        append_role(1, survivor_sequence);
    } else {
        append_role(1, survivor_sequence);
        append_role(0, source_sequence);
    }
    expect(prompt_batch_one_split_eligible(rows, source_sequence, survivor_sequence, boundary),
           "prompt rows are not sorted into one adjacent sequence split");
    return rows;
}

context_ptr make_context(llama_model * model, graph_counter & counter) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = N_CTX;
    params.n_batch              = N_BATCH;
    params.n_ubatch             = N_UBATCH;
    params.n_seq_max            = N_SEQ_MAX;
    params.n_rs_seq             = N_RS_SEQ;
    params.n_outputs_max        = N_OUTPUTS;
    params.kv_unified           = true;
    params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.no_perf              = true;
    params.n_threads            = 16;
    params.n_threads_batch      = 16;
    params.cb_eval              = count_graph_evaluations;
    params.cb_eval_user_data    = &counter;

    context_ptr context(llama_init_from_model(model, params), llama_free);
    expect(context != nullptr, "failed to create exact-model context");
    expect(llama_n_ctx(context.get()) == N_CTX, "context n_ctx mismatch");
    expect(llama_n_batch(context.get()) == N_BATCH, "context n_batch mismatch");
    expect(llama_n_ubatch(context.get()) == N_UBATCH, "context n_ubatch mismatch");
    expect(llama_n_seq_max(context.get()) == N_SEQ_MAX, "context n_seq_max mismatch");
    expect(llama_n_rs_seq(context.get()) == N_RS_SEQ, "context n_rs_seq mismatch");
    return context;
}

phase_logits decode_rows(llama_context *                context,
                         const std::vector<row_input> & rows,
                         uint32_t                       n_vocab,
                         submission_ledger *            ledger = nullptr) {
    expect(!rows.empty() && rows.size() <= N_BATCH, "invalid decode row count");
    uint32_t n_outputs = 0;
    for (const auto & row : rows) {
        expect(row.sequence >= 0 && row.sequence < (llama_seq_id) N_SEQ_MAX, "decode sequence out of range");
        expect(row.role < 3, "decode role out of range");
        n_outputs += row.output ? 1 : 0;
    }
    expect(n_outputs > 0 && n_outputs <= N_OUTPUTS, "invalid decode output count");

    llama_batch batch = llama_batch_init((int32_t) rows.size(), 0, N_SEQ_MAX);
    if (batch.token == nullptr || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
        batch.logits == nullptr) {
        llama_batch_free(batch);
        fail("llama_batch_init returned an incomplete batch allocation");
    }
    batch.n_tokens    = (int32_t) rows.size();
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        if (ledger != nullptr) {
            ledger->record(row.sequence, row.position);
        }
        if (batch.seq_id[i] == nullptr) {
            llama_batch_free(batch);
            fail("llama_batch_init returned a missing sequence-id row");
        }
        batch.token[i]     = row.token;
        batch.pos[i]       = row.position;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = row.sequence;
        batch.logits[i]    = row.output ? 1 : 0;
    }

    const int32_t result = llama_decode(context, batch);
    llama_batch_free(batch);
    expect(result == LLAMA_DECODE_SUCCESS, "exact-model decode failed");
    llama_synchronize(context);

    phase_logits output;
    uint32_t     output_index = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto & row = rows[i];
        if (!row.output) {
            continue;
        }
        // Positive indices are original batch-token indices; output_ids maps
        // those indices to compact output rows internally.
        const float * logits = llama_get_logits_ith(context, (int32_t) i);
        expect(logits != nullptr, "missing exact-model logits");
        output.values[row.role].assign(logits, logits + n_vocab);
        ++output_index;
    }
    expect(output_index == n_outputs, "exact-model output row count changed during capture");
    return output;
}

phase_logits decode_history(llama_context *    context,
                            const model_plan & plan,
                            uint32_t           boundary,
                            llama_seq_id       source_sequence,
                            llama_seq_id       survivor_sequence,
                            submission_ledger * ledger = nullptr) {
    const std::vector<row_input> rows = make_prompt_rows(plan, boundary, source_sequence, survivor_sequence);
    return decode_rows(context, rows, plan.n_vocab, ledger);
}

void expect_prompt_ledger(const submission_ledger & ledger,
                          llama_seq_id          source_sequence,
                          llama_seq_id          survivor_sequence,
                          uint32_t              boundary) {
    for (llama_seq_id sequence = 0; sequence < (llama_seq_id) N_SEQ_MAX; ++sequence) {
        const bool expected = sequence == source_sequence || sequence == survivor_sequence;
        for (uint32_t position = 0; position <= boundary; ++position) {
            expect(ledger.count(sequence, (llama_pos) position) == (expected ? 1u : 0u),
                   "prompt sequence submission was not recorded exactly once");
        }
        if (expected) {
            expect(ledger.writes[(size_t) sequence].size() == boundary + 1,
                   "prompt sequence submission contains an unexpected position");
        } else {
            expect(ledger.writes[(size_t) sequence].empty(), "prompt sequence submission populated an unrelated sequence");
        }
    }
}

void expect_exact_positions(const submission_ledger &                              ledger,
                            const std::array<std::vector<llama_pos>, N_SEQ_MAX> & expected,
                            const char *                                          phase) {
    for (llama_seq_id sequence = 0; sequence < (llama_seq_id) N_SEQ_MAX; ++sequence) {
        std::vector<uint32_t> expected_counts;
        for (const llama_pos position : expected[(size_t) sequence]) {
            expect(position >= 0, std::string(phase) + " expected a negative decode position");
            if ((size_t) position >= expected_counts.size()) {
                expected_counts.resize((size_t) position + 1, 0);
            }
            ++expected_counts[(size_t) position];
        }
        const auto & actual_counts = ledger.writes[(size_t) sequence];
        const size_t n_positions   = std::max(expected_counts.size(), actual_counts.size());
        for (size_t position = 0; position < n_positions; ++position) {
            const uint32_t expected_count = position < expected_counts.size() ? expected_counts[position] : 0;
            const uint32_t actual_count   = position < actual_counts.size() ? actual_counts[position] : 0;
            expect(actual_count == expected_count, std::string(phase) + " decode submission ledger mismatch");
        }
    }
}

phase_logits decode_parked(llama_context * context, const model_plan & plan, uint32_t boundary) {
    return decode_rows(context,
                       {
                           { 1, 1, (llama_pos) boundary + 1, plan.token(1, boundary + 1), true },
                           { 2, 0, 0,                        plan.token(2, 0),            true },
    },
                       plan.n_vocab);
}

phase_logits decode_resumed(llama_context *    context,
                            const model_plan & plan,
                            uint32_t           boundary,
                            llama_seq_id       source_sequence,
                            submission_ledger * ledger = nullptr) {
    return decode_rows(context,
                       {
                           { 0, source_sequence, (llama_pos) boundary + 1, plan.token(0, boundary + 1), true },
                           { 1, 1,               (llama_pos) boundary + 2, plan.token(1, boundary + 2), true },
                           { 2, 0,               1,                        plan.token(2, 1),            true },
    },
                       plan.n_vocab, ledger);
}

phase_logits decode_post_release(llama_context * context, const model_plan & plan, uint32_t boundary) {
    return decode_rows(context,
                       {
                           { 1, 1, (llama_pos) boundary + 3, plan.token(1, boundary + 3), true },
                           { 2, 0, 2,                        plan.token(2, 2),            true },
    },
                       plan.n_vocab);
}

int logits_argmax(const std::vector<float> & logits) {
    expect(!logits.empty(), "empty logits vector");
    return (int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

void compare_logits(const std::vector<float> & expected,
                    const std::vector<float> & actual,
                    const char *               phase,
                    uint32_t                   boundary,
                    uint32_t                   role) {
    expect(expected.size() == actual.size(), "logit vector size mismatch");
    double squared_error   = 0.0;
    double expected_energy = 0.0;
    double max_absolute    = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        expect(std::isfinite(expected[i]) && std::isfinite(actual[i]), "non-finite exact-model logits");
        const double delta = (double) actual[i] - expected[i];
        squared_error += delta * delta;
        expected_energy += (double) expected[i] * expected[i];
        max_absolute = std::max(max_absolute, std::abs(delta));
    }
    const double error        = expected_energy > 0.0 ? squared_error / expected_energy : squared_error;
    const int    expected_top = logits_argmax(expected);
    const int    actual_top   = logits_argmax(actual);
    std::fprintf(stderr, "resident-model boundary=%u phase=%s role=%u nmse=%.3e maxabs=%.3e argmax=%d/%d\n", boundary,
                 phase, role, error, max_absolute, expected_top, actual_top);
    expect(expected_top == actual_top, "exact-model argmax changed across resident transaction");
    expect(error <= NMSE_LIMIT, "exact-model NMSE exceeded resident gate");
    expect(max_absolute <= MAXABS_LIMIT, "exact-model max-absolute error exceeded resident gate");
}

void compare_phase(const phase_logits & expected, const phase_logits & actual, const char * phase, uint32_t boundary) {
    for (uint32_t role = 0; role < expected.values.size(); ++role) {
        if (!expected.values[role].empty()) {
            expect(!actual.values[role].empty(), "candidate omitted expected phase logits");
            compare_logits(expected.values[role], actual.values[role], phase, boundary, role);
        } else {
            expect(actual.values[role].empty(), "candidate emitted an unexpected phase logit row");
        }
    }
}

void compare_sparse_pool(const llama_dsv4_sparse_pool_usage & expected,
                         const llama_dsv4_sparse_pool_usage & actual,
                         const char *                         phase,
                         uint32_t                             boundary) {
    const auto same = [&](auto lhs, auto rhs, const char * field) {
        if (lhs != rhs) {
            char message[256];
            std::snprintf(message, sizeof(message), "%s boundary %u sparse field %s changed", phase, boundary, field);
            fail(message);
        }
    };
    same(expected.pool_id, actual.pool_id, "pool_id");
    same(expected.page_size, actual.page_size, "page_size");
    same(expected.virtual_pages, actual.virtual_pages, "virtual_pages");
    same(expected.physical_pages, actual.physical_pages, "physical_pages");
    same(expected.free_pages, actual.free_pages, "free_pages");
    same(expected.reserved_pages, actual.reserved_pages, "reserved_pages");
    same(expected.mapped_mappings, actual.mapped_mappings, "mapped_mappings");
    same(expected.unique_physical_pages, actual.unique_physical_pages, "unique_physical_pages");
    same(expected.shared_physical_pages, actual.shared_physical_pages, "shared_physical_pages");
    same(expected.shared_mappings, actual.shared_mappings, "shared_mappings");
    same(expected.refcount_sum, actual.refcount_sum, "refcount_sum");
    same(expected.refcount_max, actual.refcount_max, "refcount_max");
    same(expected.cow_allocations, actual.cow_allocations, "cow_allocations");
    same(expected.cow_pages, actual.cow_pages, "cow_pages");
}

void validate_sparse_pool(const llama_dsv4_sparse_pool_usage & pool, const char * phase, uint32_t boundary) {
    expect(pool.unique_physical_pages + pool.free_pages + pool.reserved_pages == pool.physical_pages,
           std::string(phase) + " sparse physical-page conservation is inconsistent");
    expect(pool.unique_physical_pages <= pool.physical_pages,
           std::string(phase) + " sparse unique pages exceed capacity");
    expect(pool.mapped_mappings == pool.refcount_sum,
           std::string(phase) + " sparse mapping/refcount accounting is inconsistent");
    expect(pool.shared_physical_pages <= pool.unique_physical_pages && pool.shared_mappings <= pool.mapped_mappings,
           std::string(phase) + " sparse sharing accounting is inconsistent");
    expect(pool.refcount_max == 0 || pool.refcount_sum >= pool.refcount_max,
           std::string(phase) + " sparse refcount summary is inconsistent");
    (void) boundary;
}

void compare_family_usage(const llama_dsv4_family_usage & expected,
                          const llama_dsv4_family_usage & actual,
                          const char *                    phase,
                          uint32_t                        boundary) {
    expect(expected.family == actual.family && expected.placement_sparse == actual.placement_sparse,
           std::string(phase) + " family identity changed");
    expect(expected.logical_capacity_rows == actual.logical_capacity_rows &&
               expected.logical_mapped_rows == actual.logical_mapped_rows &&
               expected.sequence_mapped_rows == actual.sequence_mapped_rows,
           std::string(phase) + " logical family usage changed");
    expect(expected.pools.size() == actual.pools.size(), std::string(phase) + " sparse pool count changed");
    for (size_t i = 0; i < expected.pools.size(); ++i) {
        compare_sparse_pool(expected.pools[i], actual.pools[i], phase, boundary);
    }
    compare_sparse_pool(expected.total, actual.total, phase, boundary);
}

void compare_memory_baseline(const llama_dsv4_memory_usage_snapshot & expected,
                             const llama_dsv4_memory_usage_snapshot & actual,
                             const char *                             phase,
                             uint32_t                                 boundary) {
    expect(expected.limiting_family == actual.limiting_family &&
               expected.limiting_family_mask == actual.limiting_family_mask &&
               expected.limiting_pool_id == actual.limiting_pool_id &&
               expected.limiting_available_pages == actual.limiting_available_pages,
           std::string(phase) + " limiting memory resource changed");
    for (size_t i = 0; i < expected.families.size(); ++i) {
        compare_family_usage(expected.families[i], actual.families[i], phase, boundary);
    }
    compare_sparse_pool(expected.sparse_total, actual.sparse_total, phase, boundary);
}

void compare_memory_physical(const llama_dsv4_memory_usage_snapshot & expected,
                             const llama_dsv4_memory_usage_snapshot & actual,
                             const char *                             phase,
                             uint32_t                                 boundary) {
    expect(expected.limiting_family == actual.limiting_family &&
               expected.limiting_family_mask == actual.limiting_family_mask &&
               expected.limiting_pool_id == actual.limiting_pool_id &&
               expected.limiting_available_pages == actual.limiting_available_pages,
           std::string(phase) + " limiting sparse resource changed unexpectedly");
    for (size_t i = 0; i < expected.families.size(); ++i) {
        expect(expected.families[i].pools.size() == actual.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed");
        for (size_t p = 0; p < expected.families[i].pools.size(); ++p) {
            compare_sparse_pool(expected.families[i].pools[p], actual.families[i].pools[p], phase, boundary);
        }
        compare_sparse_pool(expected.families[i].total, actual.families[i].total, phase, boundary);
    }
    compare_sparse_pool(expected.sparse_total, actual.sparse_total, phase, boundary);
}

void expect_sparse_generation_advance(const llama_dsv4_memory_usage_snapshot & before,
                                      const llama_dsv4_memory_usage_snapshot & after,
                                      const char *                             phase,
                                      uint32_t                                 boundary) {
    const auto check = [&](const llama_dsv4_sparse_pool_usage & lhs, const llama_dsv4_sparse_pool_usage & rhs) {
        expect(lhs.pool_id == rhs.pool_id, std::string(phase) + " sparse pool identity changed during move");
        expect(rhs.generation > lhs.generation,
               "boundary " + std::to_string(boundary) + " " + phase + " sparse generation did not advance");
    };
    expect(before.families.size() == after.families.size(),
           std::string(phase) + " sparse family count changed during move");
    for (size_t i = 0; i < before.families.size(); ++i) {
        expect(before.families[i].pools.size() == after.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed during move");
        for (size_t p = 0; p < before.families[i].pools.size(); ++p) {
            const auto & lhs = before.families[i].pools[p];
            const auto & rhs = after.families[i].pools[p];
            if (before.families[i].family == LLAMA_DSV4_MEMORY_RAW) {
                check(lhs, rhs);
            } else {
                expect(lhs.pool_id == rhs.pool_id && lhs.generation == rhs.generation,
                       std::string(phase) + " untouched sparse pool generation changed");
            }
        }
        const auto & lhs = before.families[i].total;
        const auto & rhs = after.families[i].total;
        if (before.families[i].family == LLAMA_DSV4_MEMORY_RAW) {
            check(lhs, rhs);
        } else {
            expect(lhs.pool_id == rhs.pool_id && lhs.generation == rhs.generation,
                   std::string(phase) + " untouched sparse family generation changed");
        }
    }
    check(before.sparse_total, after.sparse_total);
}

void expect_sparse_move(const llama_dsv4_memory_usage_snapshot & before,
                        const llama_dsv4_memory_usage_snapshot & after,
                        const char *                             phase,
                        uint32_t                                 boundary) {
    compare_memory_physical(before, after, phase, boundary);
    expect_sparse_generation_advance(before, after, phase, boundary);
}

void expect_sparse_release_delta(const llama_dsv4_memory_usage_snapshot & expected_before,
                                 const llama_dsv4_memory_usage_snapshot & expected_after,
                                 const llama_dsv4_memory_usage_snapshot & actual_before,
                                 const llama_dsv4_memory_usage_snapshot & actual_after,
                                 const char *                             phase,
                                 uint32_t                                 boundary) {
    const auto check = [&](const llama_dsv4_sparse_pool_usage & expected_lhs,
                           const llama_dsv4_sparse_pool_usage & expected_rhs,
                           const llama_dsv4_sparse_pool_usage & actual_lhs,
                           const llama_dsv4_sparse_pool_usage & actual_rhs, bool moved) {
        expect(expected_lhs.page_size == actual_lhs.page_size &&
                   expected_lhs.virtual_pages == actual_lhs.virtual_pages &&
                   expected_lhs.physical_pages == actual_lhs.physical_pages,
               std::string(phase) + " sparse release pool geometry differs from oracle");
        expect(actual_lhs.pool_id == actual_rhs.pool_id, std::string(phase) + " sparse release pool identity changed");
        expect(actual_lhs.physical_pages == actual_rhs.physical_pages &&
                   actual_lhs.virtual_pages == actual_rhs.virtual_pages &&
                   actual_lhs.page_size == actual_rhs.page_size &&
                   actual_lhs.reserved_pages == actual_rhs.reserved_pages,
               std::string(phase) + " sparse release capacity/reservation changed unexpectedly");
        if (moved) {
            expect(actual_rhs.generation > actual_lhs.generation,
                   "boundary " + std::to_string(boundary) + " " + phase + " sparse release generation did not advance");
        } else {
            if (actual_rhs.generation != actual_lhs.generation) {
                std::fprintf(stderr,
                             "resident-model debug sparse-release-untouched boundary=%u phase=%s "
                             "pool=%llu expected(before free=%llu mapped=%llu gen=%llu; after free=%llu mapped=%llu gen=%llu) "
                             "actual(before free=%llu mapped=%llu gen=%llu; after free=%llu mapped=%llu gen=%llu)\n",
                             boundary, phase,
                             (unsigned long long) actual_lhs.pool_id,
                             (unsigned long long) expected_lhs.free_pages,
                             (unsigned long long) expected_lhs.mapped_mappings,
                             (unsigned long long) expected_lhs.generation,
                             (unsigned long long) expected_rhs.free_pages,
                             (unsigned long long) expected_rhs.mapped_mappings,
                             (unsigned long long) expected_rhs.generation,
                             (unsigned long long) actual_lhs.free_pages,
                             (unsigned long long) actual_lhs.mapped_mappings,
                             (unsigned long long) actual_lhs.generation,
                             (unsigned long long) actual_rhs.free_pages,
                             (unsigned long long) actual_rhs.mapped_mappings,
                             (unsigned long long) actual_rhs.generation);
            }
            expect(actual_rhs.generation == actual_lhs.generation,
                   std::string(phase) + " untouched sparse release generation changed");
        }
        expect(actual_rhs.cow_allocations == actual_lhs.cow_allocations && actual_rhs.cow_pages == actual_lhs.cow_pages,
               std::string(phase) + " sparse release COW accounting changed");
        expect(
            actual_rhs.free_pages - actual_lhs.free_pages == expected_rhs.free_pages - expected_lhs.free_pages &&
                actual_lhs.mapped_mappings - actual_rhs.mapped_mappings ==
                    expected_lhs.mapped_mappings - expected_rhs.mapped_mappings &&
                actual_lhs.unique_physical_pages - actual_rhs.unique_physical_pages ==
                    expected_lhs.unique_physical_pages - expected_rhs.unique_physical_pages &&
                actual_lhs.shared_physical_pages - actual_rhs.shared_physical_pages ==
                    expected_lhs.shared_physical_pages - expected_rhs.shared_physical_pages &&
                actual_lhs.shared_mappings - actual_rhs.shared_mappings ==
                    expected_lhs.shared_mappings - expected_rhs.shared_mappings &&
                actual_lhs.refcount_sum - actual_rhs.refcount_sum ==
                    expected_lhs.refcount_sum - expected_rhs.refcount_sum &&
                actual_lhs.refcount_max - actual_rhs.refcount_max ==
                    expected_lhs.refcount_max - expected_rhs.refcount_max,
            "boundary " + std::to_string(boundary) + " " + phase + " sparse release mapping delta differs from oracle");
    };
    expect(expected_before.families.size() == expected_after.families.size() &&
               actual_before.families.size() == actual_after.families.size() &&
               expected_before.families.size() == actual_before.families.size(),
           std::string(phase) + " sparse release family geometry changed");
    for (size_t i = 0; i < actual_before.families.size(); ++i) {
        expect(expected_before.families[i].pools.size() == expected_after.families[i].pools.size() &&
                   actual_before.families[i].pools.size() == actual_after.families[i].pools.size() &&
                   expected_before.families[i].pools.size() == actual_before.families[i].pools.size(),
               std::string(phase) + " sparse release pool geometry changed");
        const bool moved =
            expected_after.families[i].total.free_pages != expected_before.families[i].total.free_pages ||
            expected_after.families[i].total.mapped_mappings != expected_before.families[i].total.mapped_mappings;
        check(expected_before.families[i].total, expected_after.families[i].total, actual_before.families[i].total,
              actual_after.families[i].total, moved);
        for (size_t p = 0; p < actual_before.families[i].pools.size(); ++p) {
            const bool pool_moved =
                expected_after.families[i].pools[p].free_pages != expected_before.families[i].pools[p].free_pages ||
                expected_after.families[i].pools[p].mapped_mappings !=
                    expected_before.families[i].pools[p].mapped_mappings;
            check(expected_before.families[i].pools[p], expected_after.families[i].pools[p],
                  actual_before.families[i].pools[p], actual_after.families[i].pools[p], pool_moved);
        }
    }
    const bool total_moved =
        expected_after.sparse_total.free_pages != expected_before.sparse_total.free_pages ||
        expected_after.sparse_total.mapped_mappings != expected_before.sparse_total.mapped_mappings;
    check(expected_before.sparse_total, expected_after.sparse_total, actual_before.sparse_total,
          actual_after.sparse_total, total_moved);
}

void validate_memory_snapshot(const llama_dsv4_memory_usage_snapshot & snapshot,
                              const char *                             phase,
                              uint32_t                                 boundary) {
    for (const auto & family : snapshot.families) {
        expect(family.logical_mapped_rows <= family.logical_capacity_rows,
               std::string(phase) + " logical family rows exceed capacity");
        validate_sparse_pool(family.total, phase, boundary);
        for (const auto & pool : family.pools) {
            validate_sparse_pool(pool, phase, boundary);
        }
    }
    validate_sparse_pool(snapshot.sparse_total, phase, boundary);
}

void expect_sparse_counters_monotonic(const llama_dsv4_memory_usage_snapshot & before,
                                      const llama_dsv4_memory_usage_snapshot & after,
                                      const char *                             phase,
                                      uint32_t                                 boundary) {
    (void) boundary;
    const auto check = [&](const llama_dsv4_sparse_pool_usage & lhs, const llama_dsv4_sparse_pool_usage & rhs) {
        expect(lhs.pool_id == rhs.pool_id, std::string(phase) + " sparse pool identity changed");
        expect(rhs.generation >= lhs.generation, std::string(phase) + " sparse generation regressed");
        expect(rhs.cow_allocations >= lhs.cow_allocations,
               std::string(phase) + " sparse COW allocation counter regressed");
        expect(rhs.cow_pages >= lhs.cow_pages, std::string(phase) + " sparse COW page counter regressed");
    };
    expect(before.families.size() == after.families.size(), std::string(phase) + " sparse family count changed");
    for (size_t i = 0; i < before.families.size(); ++i) {
        check(before.families[i].total, after.families[i].total);
        expect(before.families[i].pools.size() == after.families[i].pools.size(),
               std::string(phase) + " sparse pool count changed");
        for (size_t p = 0; p < before.families[i].pools.size(); ++p) {
            check(before.families[i].pools[p], after.families[i].pools[p]);
        }
    }
    check(before.sparse_total, after.sparse_total);
}

uint32_t logical_row_ratio(llama_dsv4_memory_family family) {
    switch (family) {
        case LLAMA_DSV4_MEMORY_RAW:
            return 1;
        case LLAMA_DSV4_MEMORY_CSA:
        case LLAMA_DSV4_MEMORY_LID:
            return LLAMA_DSV4_COMP_C4_TOKENS_PER_ROW;
        case LLAMA_DSV4_MEMORY_HCA:
            return LLAMA_DSV4_COMP_HCA_TOKENS_PER_ROW;
        case LLAMA_DSV4_MEMORY_FAMILY_COUNT:
            break;
    }
    fail("invalid DSV4 memory family in logical-row expectation");
}

uint64_t expected_logical_rows(const llama_dsv4_family_usage & family, llama_pos position) {
    if (position < 0) {
        return 0;
    }
    const uint64_t rows = ((uint64_t) position + 1) / logical_row_ratio(family.family);
    return std::min<uint64_t>(rows, family.logical_capacity_rows);
}

void expect_logical_rows(const llama_dsv4_memory_usage_snapshot & snapshot,
                         const std::array<llama_pos, N_SEQ_MAX> & positions,
                         const char *                             phase,
                         uint32_t                                 boundary) {
    for (const auto & family : snapshot.families) {
        expect(family.sequence_mapped_rows.size() == N_SEQ_MAX,
               std::string(phase) + " sequence logical-row geometry changed");
        uint64_t expected_total = 0;
        for (size_t sequence = 0; sequence < positions.size(); ++sequence) {
            const uint64_t expected = expected_logical_rows(family, positions[sequence]);
            expect(family.sequence_mapped_rows[sequence] == expected,
                   "boundary " + std::to_string(boundary) + " " + phase + " sequence logical-row count mismatch");
            expected_total += expected;
        }
        expect(family.logical_mapped_rows == expected_total,
               "boundary " + std::to_string(boundary) + " " + phase + " logical-row total mismatch");
    }
}

llama_dsv4_comp_handle_id comp_binding(const llama_dsv4_comp_pool * pool, uint32_t execution_id, const char * phase) {
    llama_dsv4_comp_handle_id handle = 0;
    expect(pool->get_binding(execution_id, handle) == llama_dsv4_comp_status::ok && handle != 0,
           std::string(phase) + " compressed execution binding is missing");
    return handle;
}

llama_dsv4_comp_handle_info comp_handle_info(const llama_dsv4_comp_pool * pool,
                                             llama_dsv4_comp_handle_id    handle,
                                             const char *                 phase) {
    llama_dsv4_comp_handle_info info;
    expect(pool->get_handle(handle, info) == llama_dsv4_comp_status::ok,
           std::string(phase) + " compressed handle metadata is missing");
    return info;
}

void expect_comp_handle_info_equal(const llama_dsv4_comp_handle_info & expected,
                                   const llama_dsv4_comp_handle_info & actual,
                                   const char *                        phase) {
    expect(expected.id == actual.id && expected.generation == actual.generation &&
               expected.visible_c4_rows == actual.visible_c4_rows &&
               expected.visible_hca_rows == actual.visible_hca_rows &&
               expected.c4_segment_ids == actual.c4_segment_ids && expected.hca_segment_ids == actual.hca_segment_ids,
           std::string(phase) + " compressed survivor handle metadata changed");
}

void expect_boundary_segment_transition(const llama_dsv4_comp_handle_info & info,
                                        uint32_t                            boundary,
                                        const char *                        phase) {
    if (boundary < 255) {
        return;
    }
    const uint64_t tokens     = (uint64_t) boundary + 1;
    const uint64_t c4_rows    = llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::c4, tokens);
    const uint64_t hca_rows   = llama_dsv4_comp_rows_for_tokens(llama_dsv4_comp_family::hca, tokens);
    const uint64_t c4_segments  = llama_dsv4_comp_segments_for_rows(c4_rows);
    const uint64_t hca_segments = llama_dsv4_comp_segments_for_rows(hca_rows);
    expect(info.visible_c4_rows == c4_rows && info.visible_hca_rows == hca_rows,
           std::string(phase) + " compressed visible rows did not cross the expected physical boundary");
    expect(info.c4_segment_ids.size() == c4_segments && info.hca_segment_ids.size() == hca_segments,
           std::string(phase) + " compressed physical segment count did not cross the expected boundary");
    expect(c4_rows > 0 && llama_dsv4_comp_logical_segment(c4_rows - 1) == c4_segments - 1 &&
               llama_dsv4_comp_segment_row(c4_rows - 1) == (c4_rows - 1) % LLAMA_DSV4_COMP_SEGMENT_ROWS,
           std::string(phase) + " C4 logical-to-physical segment row mapping is incorrect");
    expect(hca_rows > 0 && llama_dsv4_comp_logical_segment(hca_rows - 1) == hca_segments - 1 &&
               llama_dsv4_comp_segment_row(hca_rows - 1) == (hca_rows - 1) % LLAMA_DSV4_COMP_SEGMENT_ROWS,
           std::string(phase) + " HCA logical-to-physical segment row mapping is incorrect");
    for (size_t index = 0; index < info.c4_segment_ids.size(); ++index) {
        expect(info.c4_segment_ids[index] >= 2, std::string(phase) + " C4 binding referenced a permanent segment");
        for (size_t previous = 0; previous < index; ++previous) {
            expect(info.c4_segment_ids[previous] != info.c4_segment_ids[index],
                   std::string(phase) + " C4 binding reused a physical segment ID");
        }
    }
    for (size_t index = 0; index < info.hca_segment_ids.size(); ++index) {
        expect(info.hca_segment_ids[index] >= 2, std::string(phase) + " HCA binding referenced a permanent segment");
        for (size_t previous = 0; previous < index; ++previous) {
            expect(info.hca_segment_ids[previous] != info.hca_segment_ids[index],
                   std::string(phase) + " HCA binding reused a physical segment ID");
        }
    }
    if (boundary == 255 || boundary == 256) {
        expect(c4_rows == 64 && c4_segments == 1 && info.c4_segment_ids.size() == 1,
               std::string(phase) + " expected one full C4 segment at boundary 255/256");
    } else if (boundary == 260) {
        expect(c4_rows == 65 && c4_segments == 2 && info.c4_segment_ids.size() == 2,
               std::string(phase) + " expected a second C4 segment at boundary 260");
    }
    std::fprintf(stderr,
                 "resident-model boundary=%u phase=%s physical-segments c4_rows=%llu c4_segments=%llu "
                 "hca_rows=%llu hca_segments=%llu\n",
                 boundary, phase, (unsigned long long) c4_rows, (unsigned long long) c4_segments,
                 (unsigned long long) hca_rows, (unsigned long long) hca_segments);
}

void expect_comp_release_delta(const llama_dsv4_comp_memory_usage &               before,
                               const llama_dsv4_comp_memory_usage &               after,
                               const llama_dsv4_comp_handle_info &                released,
                               const std::array<llama_dsv4_comp_handle_info, 3> & survivors,
                               const char *                                       phase,
                               uint32_t                                           boundary) {
    (void) boundary;
    const auto check = [&](const llama_dsv4_comp_family_usage & lhs, const llama_dsv4_comp_family_usage & rhs,
                           const std::vector<uint32_t> & released_segments,
                           const std::vector<uint32_t> & survivor_segments) {
        std::set<uint32_t> survivor_set(survivor_segments.begin(), survivor_segments.end());
        uint32_t           expected_freed_segments = 0;
        for (const uint32_t segment : released_segments) {
            expect(segment >= 2, std::string(phase) + " resident release referenced a permanent segment");
            expect(survivor_set.count(segment) == 0,
                   std::string(phase) + " resident release would change shared segment accounting");
            ++expected_freed_segments;
        }

        expect(lhs.capacity_segments == rhs.capacity_segments && lhs.permanent_segments == rhs.permanent_segments &&
                   lhs.capacity_pages == rhs.capacity_pages &&
                   lhs.segment_pages_capacity == rhs.segment_pages_capacity &&
                   lhs.lid_pages_capacity == rhs.lid_pages_capacity,
               std::string(phase) + " compressed capacity geometry changed during release");
        expect(lhs.reserved_segments == rhs.reserved_segments && lhs.reserved_pages == rhs.reserved_pages &&
                   lhs.segment_pages_reserved == rhs.segment_pages_reserved &&
                   lhs.lid_pages_reserved == rhs.lid_pages_reserved,
               std::string(phase) + " compressed reservations changed during release");
        expect(lhs.shared_segments == rhs.shared_segments && lhs.shared_pages == rhs.shared_pages &&
                   lhs.cow_segments == rhs.cow_segments && lhs.cow_pages == rhs.cow_pages &&
                   lhs.scratch_rows_in_use == rhs.scratch_rows_in_use,
               std::string(phase) + " compressed shared/COW accounting changed during release");

        expect(lhs.mapped_segments - rhs.mapped_segments == expected_freed_segments &&
                   rhs.free_segments - lhs.free_segments == expected_freed_segments,
               std::string(phase) + " compressed segment release delta is incorrect");
        const bool     is_c4 = lhs.segment_pages_capacity != 0;
        const uint64_t expected_segment_pages =
            (uint64_t) expected_freed_segments * (is_c4 ? LLAMA_DSV4_COMP_CSA_LAYERS : LLAMA_DSV4_COMP_HCA_LAYERS);
        if (is_c4) {
            expect(lhs.segment_pages_mapped - rhs.segment_pages_mapped == expected_segment_pages &&
                       rhs.segment_pages_free - lhs.segment_pages_free == expected_segment_pages,
                   std::string(phase) + " compressed segment-page release delta is incorrect");
        } else {
            expect(lhs.segment_pages_capacity == rhs.segment_pages_capacity &&
                       lhs.segment_pages_free == rhs.segment_pages_free &&
                       lhs.segment_pages_reserved == rhs.segment_pages_reserved &&
                       lhs.segment_pages_mapped == rhs.segment_pages_mapped,
                   std::string(phase) + " HCA segment-page accounting changed during release");
        }

        std::set<uint32_t> released_groups;
        if (lhs.segment_pages_capacity != 0) {
            for (const uint32_t segment : released_segments) {
                released_groups.insert(segment / 4);
            }
        }
        uint32_t expected_freed_lid_groups = 0;
        for (const uint32_t group : released_groups) {
            // Segment IDs 0 and 1 are permanent zero/scratch mappings, so
            // their LID group remains mapped even when all data segments in
            // the group are released.
            if (group == 0) {
                continue;
            }
            const bool survivor_group = std::any_of(survivor_segments.begin(), survivor_segments.end(),
                                                    [group](uint32_t segment) { return segment / 4 == group; });
            if (!survivor_group) {
                ++expected_freed_lid_groups;
            }
        }
        const uint64_t expected_lid_pages = (uint64_t) expected_freed_lid_groups * LLAMA_DSV4_COMP_LID_LAYERS;
        if (!is_c4) {
            expect(lhs.lid_pages_capacity == rhs.lid_pages_capacity && lhs.lid_pages_free == rhs.lid_pages_free &&
                       lhs.lid_pages_reserved == rhs.lid_pages_reserved && lhs.lid_pages_mapped == rhs.lid_pages_mapped,
                   std::string(phase) + " HCA LID-page accounting changed during release");
        }
        expect(lhs.lid_pages_mapped - rhs.lid_pages_mapped == expected_lid_pages &&
                   rhs.lid_pages_free - lhs.lid_pages_free == expected_lid_pages,
               std::string(phase) + " compressed LID-page release delta is incorrect");
        expect(lhs.mapped_pages - rhs.mapped_pages == expected_segment_pages + expected_lid_pages &&
                   rhs.free_pages - lhs.free_pages == expected_segment_pages + expected_lid_pages,
               std::string(phase) + " compressed total-page release delta is incorrect");
    };
    std::vector<uint32_t> c4_survivors = survivors[0].c4_segment_ids;
    c4_survivors.insert(c4_survivors.end(), survivors[1].c4_segment_ids.begin(), survivors[1].c4_segment_ids.end());
    c4_survivors.insert(c4_survivors.end(), survivors[2].c4_segment_ids.begin(), survivors[2].c4_segment_ids.end());
    std::vector<uint32_t> hca_survivors = survivors[0].hca_segment_ids;
    hca_survivors.insert(hca_survivors.end(), survivors[1].hca_segment_ids.begin(), survivors[1].hca_segment_ids.end());
    hca_survivors.insert(hca_survivors.end(), survivors[2].hca_segment_ids.begin(), survivors[2].hca_segment_ids.end());
    check(before.c4, after.c4, released.c4_segment_ids, c4_survivors);
    check(before.hca, after.hca, released.hca_segment_ids, hca_survivors);
}

void expect_comp_transition(const llama_dsv4_comp_memory_usage & before,
                            const llama_dsv4_comp_memory_usage & after,
                            const char *                         phase,
                            uint32_t                             boundary,
                            uint64_t                             epoch_delta = 1) {
    (void) boundary;
    expect(after.epoch == before.epoch + epoch_delta,
           std::string(phase) + " compressed pool epoch advanced by an unexpected amount");
    expect(after.active_tickets == before.active_tickets &&
               after.retained_ticket_records == before.retained_ticket_records,
           std::string(phase) + " compressed ticket accounting changed unexpectedly");
}

void compare_comp_family(const llama_dsv4_comp_family_usage & expected,
                         const llama_dsv4_comp_family_usage & actual,
                         const char *                         phase,
                         uint32_t                             boundary) {
    (void) phase;
    (void) boundary;
#define CHECK_COMP_FIELD(field) expect(expected.field == actual.field, "compressed " #field " changed unexpectedly")
    CHECK_COMP_FIELD(capacity_segments);
    CHECK_COMP_FIELD(permanent_segments);
    CHECK_COMP_FIELD(free_segments);
    CHECK_COMP_FIELD(reserved_segments);
    CHECK_COMP_FIELD(mapped_segments);
    CHECK_COMP_FIELD(shared_segments);
    CHECK_COMP_FIELD(cow_segments);
    CHECK_COMP_FIELD(scratch_rows_in_use);
    CHECK_COMP_FIELD(capacity_pages);
    CHECK_COMP_FIELD(free_pages);
    CHECK_COMP_FIELD(reserved_pages);
    CHECK_COMP_FIELD(mapped_pages);
    CHECK_COMP_FIELD(shared_pages);
    CHECK_COMP_FIELD(cow_pages);
    CHECK_COMP_FIELD(segment_pages_capacity);
    CHECK_COMP_FIELD(segment_pages_free);
    CHECK_COMP_FIELD(segment_pages_reserved);
    CHECK_COMP_FIELD(segment_pages_mapped);
    CHECK_COMP_FIELD(lid_pages_capacity);
    CHECK_COMP_FIELD(lid_pages_free);
    CHECK_COMP_FIELD(lid_pages_reserved);
    CHECK_COMP_FIELD(lid_pages_mapped);
#undef CHECK_COMP_FIELD
}

void compare_comp_baseline(const llama_dsv4_comp_memory_usage & expected,
                           const llama_dsv4_comp_memory_usage & actual,
                           const char *                         phase,
                           uint32_t                             boundary) {
    compare_comp_family(expected.c4, actual.c4, phase, boundary);
    compare_comp_family(expected.hca, actual.hca, phase, boundary);
    expect(expected.handles == actual.handles && expected.bindings == actual.bindings &&
               expected.resident_handles == actual.resident_handles &&
               expected.active_tickets == actual.active_tickets &&
               expected.retained_ticket_records == actual.retained_ticket_records,
           std::string(phase) + " compressed ownership counts changed unexpectedly");
}

void expect_resident_usage(const llama_dsv4_resident_usage & usage,
                           uint32_t                          occupied,
                           uint32_t                          handles,
                           const char *                      phase,
                           uint32_t                          boundary) {
    expect(usage.cache_id != 0 && usage.capacity == N_SEQ_MAX && usage.occupied_slots == occupied &&
               usage.handles == handles,
           std::string(phase) + " resident usage mismatch at boundary " + std::to_string(boundary));
}

void expect_resident_transition(const llama_dsv4_resident_usage & before,
                                const llama_dsv4_resident_usage & after,
                                const char *                      phase,
                                uint32_t                          boundary) {
    expect(before.cache_id != 0 && before.cache_id == after.cache_id,
           std::string(phase) + " resident cache identity changed");
    expect(after.epoch > before.epoch, std::string(phase) + " resident epoch did not advance");
    expect(after.capacity == before.capacity, std::string(phase) + " resident capacity changed");
    (void) boundary;
}

void expect_context_empty(llama_context *                          context,
                          llama_kv_cache_dsv4 *                    memory,
                          const llama_dsv4_memory_usage_snapshot & memory_baseline,
                          const llama_dsv4_comp_memory_usage &     comp_baseline,
                          uint32_t                                 boundary) {
    llama_memory_clear(llama_get_memory(context), true);
    compare_memory_baseline(memory_baseline, memory->memory_usage_snapshot(), "empty", boundary);
    compare_comp_baseline(comp_baseline, memory->get_comp_pool()->memory_usage_snapshot(), "empty", boundary);
    expect_resident_usage(memory->resident_usage(), 0, 0, "empty", boundary);
}

void run_boundary(llama_model * model, uint32_t n_vocab, uint32_t boundary, bool diagnostic_trace) {
    const model_plan                 plan = make_model_plan(n_vocab);
    phase_logits                     oracle_prompt;
    phase_logits                     oracle_parked;
    phase_logits                     oracle_resumed;
    phase_logits                     oracle_post;
    phase_evaluations                oracle_evaluations;
    graph_trace                      oracle_prompt_trace;
    llama_dsv4_memory_usage_snapshot oracle_before_release_memory;
    llama_dsv4_memory_usage_snapshot oracle_release_memory;

    // Keep only one context alive at a time. The model weights remain shared,
    // while the two context compute buffers never coexist in this gate.
    {
        graph_counter counter;
        context_ptr   oracle = make_context(model, counter);
        auto *        memory = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(oracle.get()));
        expect(memory != nullptr, "oracle memory is not DSV4");
        expect(memory->is_aggregate_compressed(), "oracle did not select aggregate compressed storage");
        expect_resident_usage(memory->resident_usage(), 0, 0, "oracle creation", boundary);
        const auto oracle_memory_baseline = memory->memory_usage_snapshot();
        const auto oracle_comp_baseline   = memory->get_comp_pool()->memory_usage_snapshot();

        submission_ledger oracle_prompt_ledger;
        const uint64_t oracle_before_prompt      = counter.evaluations;
        const uint64_t oracle_asks_before_prompt = counter.asks;
        reset_graph_trace(counter, diagnostic_trace);
        oracle_prompt = decode_history(oracle.get(), plan, boundary, 2, 1, &oracle_prompt_ledger);
        oracle_prompt_trace = counter.trace_data;
        reset_graph_trace(counter, false);
        oracle_evaluations.prompt_evaluations = counter.evaluations - oracle_before_prompt;
        oracle_evaluations.prompt_asks        = counter.asks - oracle_asks_before_prompt;
        expect_prompt_ledger(oracle_prompt_ledger, 2, 1, boundary);
        const uint64_t oracle_before_parked       = counter.evaluations;
        const uint64_t oracle_asks_before_parked  = counter.asks;
        oracle_parked                             = decode_parked(oracle.get(), plan, boundary);
        oracle_evaluations.parked_evaluations     = counter.evaluations - oracle_before_parked;
        oracle_evaluations.parked_asks            = counter.asks - oracle_asks_before_parked;
        const uint64_t oracle_before_resumed      = counter.evaluations;
        const uint64_t oracle_asks_before_resumed = counter.asks;
        submission_ledger oracle_resumed_ledger;
        oracle_resumed = decode_resumed(oracle.get(), plan, boundary, 2, &oracle_resumed_ledger);
        oracle_evaluations.resumed_evaluations = counter.evaluations - oracle_before_resumed;
        oracle_evaluations.resumed_asks        = counter.asks - oracle_asks_before_resumed;
        expect_exact_positions(oracle_resumed_ledger,
                               { std::vector<llama_pos>{ 1 }, std::vector<llama_pos>{ (llama_pos) boundary + 2 },
                                 std::vector<llama_pos>{ (llama_pos) boundary + 1 } },
                               "oracle resumed");
        oracle_before_release_memory = memory->memory_usage_snapshot();
        if (diagnostic_trace) {
            std::fprintf(stderr,
                         "resident-model diagnostic oracle-seq-rm boundary=%u seq=2 p0=-1 p1=-1 "
                         "operation=raw-seq-rm-plus-aggregate-compressed-reset\n",
                         boundary);
            print_sparse_snapshot(boundary, "oracle-before-release", oracle_before_release_memory);
        }
        expect(llama_memory_seq_rm(llama_get_memory(oracle.get()), 2, -1, -1), "oracle source release failed");
        oracle_release_memory = memory->memory_usage_snapshot();
        if (diagnostic_trace) {
            print_sparse_snapshot(boundary, "oracle-after-release", oracle_release_memory,
                                  &oracle_before_release_memory);
        }
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 2) == -1,
               "oracle released source position ledger mismatch");
        const uint64_t oracle_before_post      = counter.evaluations;
        const uint64_t oracle_asks_before_post = counter.asks;
        oracle_post                            = decode_post_release(oracle.get(), plan, boundary);
        oracle_evaluations.post_evaluations    = counter.evaluations - oracle_before_post;
        oracle_evaluations.post_asks           = counter.asks - oracle_asks_before_post;
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 2) == -1,
               "oracle source remained populated after release");
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 1) == (llama_pos) boundary + 3,
               "oracle survivor position ledger mismatch");
        expect(llama_memory_seq_pos_max(llama_get_memory(oracle.get()), 0) == 2,
               "oracle replacement position ledger mismatch");
        expect_context_empty(oracle.get(), memory, oracle_memory_baseline, oracle_comp_baseline, boundary);
    }

    graph_counter counter;
    context_ptr   candidate = make_context(model, counter);
    auto *        memory    = dynamic_cast<llama_kv_cache_dsv4 *>(llama_get_memory(candidate.get()));
    expect(memory != nullptr, "candidate memory is not DSV4");
    expect(memory->is_aggregate_compressed(), "candidate did not select aggregate compressed storage");
    resident_cleanup cleanup(memory);
    auto *           comp_pool = memory->get_comp_pool();
    expect(comp_pool != nullptr, "candidate aggregate compressed pool is missing");
    const auto initial_binding0       = comp_binding(comp_pool, 0, "candidate creation");
    const auto initial_binding1       = comp_binding(comp_pool, 1, "candidate creation");
    const auto initial_binding2       = comp_binding(comp_pool, 2, "candidate creation");
    const auto initial_info1          = comp_handle_info(comp_pool, initial_binding1, "candidate creation");
    const auto resident_before_detach = memory->resident_usage();
    expect_resident_usage(resident_before_detach, 0, 0, "candidate creation", boundary);
    const auto memory_baseline = memory->memory_usage_snapshot();
    const auto comp_baseline   = memory->get_comp_pool()->memory_usage_snapshot();
    const auto rs_idx_baseline = memory->get_rs_idx();
    expect(rs_idx_baseline.size() == N_SEQ_MAX, "rollback index geometry mismatch");
    const auto expect_rs_idx = [&](const char * phase) {
        expect(memory->get_rs_idx() == rs_idx_baseline,
               std::string(phase) + " rollback index changed during resident transaction");
    };
    validate_memory_snapshot(memory_baseline, "candidate creation", boundary);
    submission_ledger prompt_ledger;
    const uint64_t     candidate_before_prompt      = counter.evaluations;
    const uint64_t     candidate_asks_before_prompt = counter.asks;
    reset_graph_trace(counter, diagnostic_trace);
    const phase_logits candidate_prompt = decode_history(candidate.get(), plan, boundary, 0, 1, &prompt_ledger);
    const graph_trace candidate_prompt_trace = counter.trace_data;
    reset_graph_trace(counter, false);
    if (diagnostic_trace) {
        print_graph_trace(boundary, "prompt", "oracle", oracle_prompt_trace);
        print_graph_trace(boundary, "prompt", "candidate", candidate_prompt_trace);
        std::fprintf(stderr,
                     "resident-model diagnostic prompt-delta boundary=%u oracle_evaluations=%llu "
                     "candidate_evaluations=%llu oracle_asks=%llu candidate_asks=%llu\n",
                     boundary, (unsigned long long) oracle_evaluations.prompt_evaluations,
                     (unsigned long long) (counter.evaluations - candidate_before_prompt),
                     (unsigned long long) oracle_evaluations.prompt_asks,
                     (unsigned long long) (counter.asks - candidate_asks_before_prompt));
    }
    expect(counter.evaluations - candidate_before_prompt == oracle_evaluations.prompt_evaluations &&
               counter.asks - candidate_asks_before_prompt == oracle_evaluations.prompt_asks,
           "candidate prompt graph-evaluation count differs from oracle");
    expect_prompt_ledger(prompt_ledger, 0, 1, boundary);
    compare_phase(oracle_prompt, candidate_prompt, "prompt", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == (llama_pos) boundary,
           "candidate source prompt position ledger mismatch");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary,
           "candidate survivor prompt position ledger mismatch");
    expect_rs_idx("prompt");
    const auto memory_before_detach           = memory->memory_usage_snapshot();
    const auto comp_before_detach             = memory->get_comp_pool()->memory_usage_snapshot();
    const auto survivor_binding_before_detach = comp_binding(comp_pool, 1, "prompt");
    expect_logical_rows(memory_before_detach, { (llama_pos) boundary, (llama_pos) boundary, -1 }, "prompt", boundary);
    expect(survivor_binding_before_detach == initial_binding1, "survivor compressed binding changed during prompt");
    const auto source_info_before_detach   = comp_handle_info(comp_pool, initial_binding0, "prompt");
    const auto survivor_info_before_detach = comp_handle_info(comp_pool, survivor_binding_before_detach, "prompt");
    expect_boundary_segment_transition(source_info_before_detach, boundary, "prompt source");
    validate_memory_snapshot(memory_before_detach, "prompt", boundary);

    const uint64_t callback_before_detach = counter.evaluations;
    const auto     quote = memory->quote_resident_detach({ 0, llama_dsv4_resident_scope::single_context });
    expect(quote.status == llama_dsv4_resident_status::ok && quote.unsupported_components == 0,
           "candidate composite detach quote failed");
    expect(quote.seq_id == 0 && quote.scope == llama_dsv4_resident_scope::single_context &&
               quote.rollback_index == rs_idx_baseline[0] && quote.required_components == quote.detachable_components &&
               quote.resident_state_slot < N_SEQ_MAX && quote.resident.cache_id == resident_before_detach.cache_id &&
               quote.resident.id != 0,
           "candidate composite detach quote metadata mismatch");
    const auto detached = memory->detach_resident(quote);
    expect(detached.status == llama_dsv4_resident_status::ok, "candidate composite detach failed");
    expect(detached.resident == quote.resident, "candidate composite detach changed resident handle metadata");
    cleanup.adopt(detached.resident);
    expect(counter.evaluations == callback_before_detach, "resident detach changed graph callback count");
    const auto resident_after_detach = memory->resident_usage();
    expect_resident_transition(resident_before_detach, resident_after_detach, "detached", boundary);
    expect_resident_usage(resident_after_detach, 1, 1, "detached", boundary);
    expect_rs_idx("detached");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == -1,
           "detached source execution sequence remained populated");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary,
           "resident detach altered the survivor sequence");
    expect(comp_binding(comp_pool, 0, "detached") != initial_binding0 &&
               comp_binding(comp_pool, 1, "detached") == survivor_binding_before_detach &&
               comp_binding(comp_pool, 2, "detached") == initial_binding2,
           "resident detach altered survivor or destination compressed bindings");
    expect_comp_handle_info_equal(survivor_info_before_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "detached"), "detached");
    expect_comp_handle_info_equal(source_info_before_detach, comp_handle_info(comp_pool, initial_binding0, "detached"),
                                  "detached source");
    const auto memory_after_detach = memory->memory_usage_snapshot();
    const auto comp_after_detach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_detach.handles == comp_before_detach.handles + 1 &&
               comp_after_detach.bindings == comp_before_detach.bindings &&
               comp_after_detach.resident_handles == comp_before_detach.resident_handles + 1,
           "compressed resident detach ownership accounting mismatch");
    expect_comp_transition(comp_before_detach, comp_after_detach, "detached", boundary);
    compare_comp_family(comp_before_detach.c4, comp_after_detach.c4, "detached", boundary);
    compare_comp_family(comp_before_detach.hca, comp_after_detach.hca, "detached", boundary);
    expect_sparse_move(memory_before_detach, memory_after_detach, "detached", boundary);
    expect_logical_rows(memory_after_detach, { -1, (llama_pos) boundary, -1 }, "detached", boundary);
    validate_memory_snapshot(memory_after_detach, "detached", boundary);

    const uint64_t     candidate_before_parked      = counter.evaluations;
    const uint64_t     candidate_asks_before_parked = counter.asks;
    const phase_logits candidate_parked             = decode_parked(candidate.get(), plan, boundary);
    expect(counter.evaluations - candidate_before_parked == oracle_evaluations.parked_evaluations &&
               counter.asks - candidate_asks_before_parked == oracle_evaluations.parked_asks,
           "candidate parked graph-evaluation count differs from oracle");
    compare_phase(oracle_parked, candidate_parked, "parked", boundary);
    const auto memory_after_parked         = memory->memory_usage_snapshot();
    const auto comp_before_attach          = memory->get_comp_pool()->memory_usage_snapshot();
    const auto resident_before_attach      = memory->resident_usage();
    const auto survivor_info_before_attach = comp_handle_info(comp_pool, survivor_binding_before_detach, "parked");
    expect_sparse_counters_monotonic(memory_after_detach, memory_after_parked, "parked", boundary);
    expect_logical_rows(memory_after_parked, { 0, (llama_pos) boundary + 1, -1 }, "parked", boundary);
    validate_memory_snapshot(memory_after_parked, "parked", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == 0,
           "fresh replacement work did not start at position zero");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 1,
           "survivor position changed during parked phase");

    const uint64_t callback_before_attach = counter.evaluations;
    const auto     attach_quote           = memory->quote_resident_attach(detached.resident, 2);
    expect(attach_quote.status == llama_dsv4_resident_status::ok, "candidate composite attach quote failed");
    expect(attach_quote.execution_id == 2 && attach_quote.resident == detached.resident,
           "candidate composite attach quote metadata mismatch");
    const auto attach_status = memory->attach_resident(attach_quote);
    if (attach_status == llama_dsv4_resident_status::ok) {
        cleanup.disarm();
    }
    expect(attach_status == llama_dsv4_resident_status::ok, "candidate composite attach failed");
    expect(counter.evaluations == callback_before_attach, "resident attach changed graph callback count");
    const auto resident_after_attach = memory->resident_usage();
    expect_resident_transition(resident_before_attach, resident_after_attach, "attached", boundary);
    expect_resident_usage(resident_after_attach, 0, 0, "attached", boundary);
    expect_rs_idx("attached");
    const auto memory_after_attach = memory->memory_usage_snapshot();
    const auto comp_after_attach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_attach.handles + 1 == comp_before_attach.handles &&
               comp_after_attach.bindings == comp_before_attach.bindings &&
               comp_after_attach.resident_handles + 1 == comp_before_attach.resident_handles,
           "compressed resident attach ownership accounting mismatch");
    expect_comp_transition(comp_before_attach, comp_after_attach, "attached", boundary);
    compare_comp_family(comp_before_attach.c4, comp_after_attach.c4, "attached", boundary);
    compare_comp_family(comp_before_attach.hca, comp_after_attach.hca, "attached", boundary);
    expect_sparse_move(memory_after_parked, memory_after_attach, "attached", boundary);
    expect_logical_rows(memory_after_attach, { 0, (llama_pos) boundary + 1, (llama_pos) boundary }, "attached",
                        boundary);
    validate_memory_snapshot(memory_after_attach, "attached", boundary);
    expect(comp_binding(comp_pool, 1, "attached") == survivor_binding_before_detach &&
               comp_binding(comp_pool, 2, "attached") == initial_binding0,
           "resident attach did not bind source to destination or preserve survivor binding");
    expect_comp_handle_info_equal(survivor_info_before_attach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "attached"), "attached");
    expect_comp_handle_info_equal(source_info_before_detach, comp_handle_info(comp_pool, initial_binding0, "attached"),
                                  "attached source");
    const auto stale_attach = memory->quote_resident_attach(detached.resident, 2);
    expect(stale_attach.status == llama_dsv4_resident_status::stale_handle,
           "consumed resident handle was accepted after attach");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == (llama_pos) boundary,
           "attached source position state was changed unexpectedly");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 1,
           "resident attach altered the survivor sequence");

    const uint64_t     candidate_before_resumed      = counter.evaluations;
    const uint64_t     candidate_asks_before_resumed = counter.asks;
    submission_ledger resumed_ledger;
    const phase_logits candidate_resumed = decode_resumed(candidate.get(), plan, boundary, 2, &resumed_ledger);
    // Callback deltas compare graph scheduling shape only. They do not prove
    // that the submitted rows were written exactly once or that no backend
    // replay occurred; the submission ledger above is deliberately scoped to
    // caller-visible batch construction for that reason.
    expect(counter.evaluations - candidate_before_resumed == oracle_evaluations.resumed_evaluations &&
               counter.asks - candidate_asks_before_resumed == oracle_evaluations.resumed_asks,
           "candidate resumed graph-evaluation shape differs from oracle");
    expect_exact_positions(resumed_ledger,
                           { std::vector<llama_pos>{ 1 }, std::vector<llama_pos>{ (llama_pos) boundary + 2 },
                             std::vector<llama_pos>{ (llama_pos) boundary + 1 } },
                           "candidate resumed");
    compare_phase(oracle_resumed, candidate_resumed, "resumed", boundary);
    const auto memory_after_resumed = memory->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_after_attach, memory_after_resumed, "resumed", boundary);
    expect_logical_rows(memory_after_resumed, { 1, (llama_pos) boundary + 2, (llama_pos) boundary + 1 }, "resumed",
                        boundary);
    validate_memory_snapshot(memory_after_resumed, "resumed", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == (llama_pos) boundary + 1,
           "resumed source position ledger mismatch");
    const auto survivor_info_before_second_detach =
        comp_handle_info(comp_pool, survivor_binding_before_detach, "resumed");

    const auto     comp_before_second_detach     = memory->get_comp_pool()->memory_usage_snapshot();
    const auto     resident_before_second_detach = memory->resident_usage();
    const uint64_t callback_before_release       = counter.evaluations;
    const auto     second_quote = memory->quote_resident_detach({ 2, llama_dsv4_resident_scope::single_context });
    expect(second_quote.status == llama_dsv4_resident_status::ok, "second composite detach quote failed");
    expect(second_quote.seq_id == 2 && second_quote.scope == llama_dsv4_resident_scope::single_context &&
               second_quote.rollback_index == rs_idx_baseline[2] &&
               second_quote.required_components == second_quote.detachable_components &&
               second_quote.resident_state_slot < N_SEQ_MAX &&
               second_quote.resident.cache_id == resident_after_attach.cache_id && second_quote.resident.id != 0,
           "second composite detach quote metadata mismatch");
    const auto second_detached = memory->detach_resident(second_quote);
    expect(second_detached.status == llama_dsv4_resident_status::ok, "second composite detach failed");
    cleanup.adopt(second_detached.resident);
    const auto resident_after_second_detach = memory->resident_usage();
    expect_resident_transition(resident_before_second_detach, resident_after_second_detach, "second detached",
                               boundary);
    expect_resident_usage(resident_after_second_detach, 1, 1, "second detached", boundary);
    expect(comp_binding(comp_pool, 2, "second detached") != initial_binding0 &&
               comp_binding(comp_pool, 1, "second detached") == survivor_binding_before_detach,
           "second resident detach altered destination or survivor compressed binding");
    expect_comp_handle_info_equal(survivor_info_before_second_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "second detached"),
                                  "second detached");
    const auto released_source_info_before_release = comp_handle_info(comp_pool, initial_binding0, "second detached");
    const auto replacement_info_before_release =
        comp_handle_info(comp_pool, comp_binding(comp_pool, 0, "second detached"), "second detached");
    const auto destination_info_before_release =
        comp_handle_info(comp_pool, comp_binding(comp_pool, 2, "second detached"), "second detached");
    const auto memory_after_second_detach = memory->memory_usage_snapshot();
    const auto comp_after_second_detach   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_second_detach.handles == comp_before_second_detach.handles + 1 &&
               comp_after_second_detach.bindings == comp_before_second_detach.bindings &&
               comp_after_second_detach.resident_handles == comp_before_second_detach.resident_handles + 1,
           "compressed second resident detach ownership accounting mismatch");
    expect_comp_transition(comp_before_second_detach, comp_after_second_detach, "second detached", boundary);
    compare_comp_family(comp_before_second_detach.c4, comp_after_second_detach.c4, "second detached", boundary);
    compare_comp_family(comp_before_second_detach.hca, comp_after_second_detach.hca, "second detached", boundary);
    expect_sparse_move(memory_after_resumed, memory_after_second_detach, "second detached", boundary);
    expect_logical_rows(memory_after_second_detach, { 1, (llama_pos) boundary + 2, -1 }, "second detached", boundary);
    if (diagnostic_trace) {
        print_sparse_snapshot(boundary, "candidate-before-release", memory_after_second_detach);
    }
    const auto release_status = memory->release_resident(second_detached.resident);
    if (release_status == llama_dsv4_resident_status::ok) {
        cleanup.disarm();
    }
    expect(release_status == llama_dsv4_resident_status::ok, "composite resident release failed");
    const auto stale_second_attach = memory->quote_resident_attach(second_detached.resident, 2);
    expect(stale_second_attach.status == llama_dsv4_resident_status::stale_handle,
           "released resident handle was accepted after release");
    expect(memory->release_resident(second_detached.resident) == llama_dsv4_resident_status::stale_handle,
           "released resident handle was accepted for a second release");
    expect(comp_binding(comp_pool, 2, "released") != initial_binding0 &&
               comp_binding(comp_pool, 1, "released") == survivor_binding_before_detach,
           "resident release altered destination or survivor compressed binding");
    // Compare release isolation against the snapshot taken after resumed
    // writes; initial_info1 is only the empty-pool baseline used again after
    // final clear.
    expect_comp_handle_info_equal(survivor_info_before_second_detach,
                                  comp_handle_info(comp_pool, survivor_binding_before_detach, "released"), "released");
    llama_dsv4_comp_handle_info released_source_info;
    expect(comp_pool->get_handle(initial_binding0, released_source_info) == llama_dsv4_comp_status::handle_not_found,
           "released source compressed handle remained addressable");
    expect(counter.evaluations == callback_before_release, "resident release changed graph callback count");
    const auto final_binding0 = comp_binding(comp_pool, 0, "released");
    const auto final_binding1 = comp_binding(comp_pool, 1, "released");
    const auto final_binding2 = comp_binding(comp_pool, 2, "released");
    expect(final_binding0 == replacement_info_before_release.id && final_binding2 == destination_info_before_release.id &&
               final_binding1 == survivor_info_before_second_detach.id && final_binding0 != initial_binding0 &&
               final_binding2 != initial_binding2,
           "resident release did not preserve the expected per-sequence binding identities");
    expect_comp_handle_info_equal(replacement_info_before_release,
                                  comp_handle_info(comp_pool, final_binding0, "released"),
                                  "released replacement handle");
    expect_comp_handle_info_equal(destination_info_before_release,
                                  comp_handle_info(comp_pool, final_binding2, "released"),
                                  "released destination handle");
    const auto resident_after_release = memory->resident_usage();
    expect_resident_transition(resident_after_second_detach, resident_after_release, "released", boundary);
    expect_resident_usage(resident_after_release, 0, 0, "released", boundary);
    expect_rs_idx("released");
    const auto memory_after_release = memory->memory_usage_snapshot();
    const auto comp_after_release   = memory->get_comp_pool()->memory_usage_snapshot();
    expect(comp_after_release.handles + 1 == comp_after_second_detach.handles &&
               comp_after_release.bindings == comp_after_second_detach.bindings &&
               comp_after_release.resident_handles + 1 == comp_after_second_detach.resident_handles,
           "compressed resident release ownership accounting mismatch");
    expect_comp_transition(comp_after_second_detach, comp_after_release, "released", boundary, 2);
    expect_comp_release_delta(
        comp_after_second_detach, comp_after_release, released_source_info_before_release,
        { replacement_info_before_release, survivor_info_before_second_detach, destination_info_before_release },
        "released", boundary);
    if (diagnostic_trace) {
        print_sparse_snapshot(boundary, "candidate-after-release", memory_after_release,
                              &memory_after_second_detach);
    }
    expect_sparse_release_delta(oracle_before_release_memory, oracle_release_memory, memory_after_second_detach,
                                memory_after_release, "released", boundary);
    expect_sparse_counters_monotonic(memory_after_second_detach, memory_after_release, "released", boundary);
    expect_logical_rows(memory_after_release, { 1, (llama_pos) boundary + 2, -1 }, "released", boundary);
    validate_memory_snapshot(memory_after_release, "released", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 2) == -1,
           "released source execution sequence remained populated");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 2,
           "resident release altered the survivor sequence");

    const uint64_t     candidate_before_post      = counter.evaluations;
    const uint64_t     candidate_asks_before_post = counter.asks;
    const phase_logits candidate_post             = decode_post_release(candidate.get(), plan, boundary);
    expect(counter.evaluations - candidate_before_post == oracle_evaluations.post_evaluations &&
               counter.asks - candidate_asks_before_post == oracle_evaluations.post_asks,
           "candidate post-release graph-evaluation count differs from oracle");
    const auto memory_after_post = memory->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_after_release, memory_after_post, "post-release", boundary);
    expect_logical_rows(memory_after_post, { 2, (llama_pos) boundary + 3, -1 }, "post-release", boundary);
    validate_memory_snapshot(memory_after_post, "post-release", boundary);
    compare_phase(oracle_post, candidate_post, "post-release", boundary);
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 1) == (llama_pos) boundary + 3,
           "survivor position ledger mismatch after release");
    expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), 0) == 2,
           "replacement position ledger mismatch after release");

    llama_memory_clear(llama_get_memory(candidate.get()), true);
    const auto final_memory = memory->memory_usage_snapshot();
    const auto final_comp   = memory->get_comp_pool()->memory_usage_snapshot();
    expect_sparse_counters_monotonic(memory_baseline, final_memory, "final", boundary);
    compare_memory_baseline(memory_baseline, final_memory, "final", boundary);
    expect(final_comp.epoch >= comp_baseline.epoch, "compressed pool epoch regressed after clear");
    compare_comp_baseline(comp_baseline, final_comp, "final", boundary);
    expect_resident_usage(memory->resident_usage(), 0, 0, "final", boundary);
    expect_logical_rows(final_memory, { -1, -1, -1 }, "final", boundary);
    for (llama_seq_id sequence = 0; sequence < (llama_seq_id) N_SEQ_MAX; ++sequence) {
        expect(llama_memory_seq_pos_max(llama_get_memory(candidate.get()), sequence) == -1,
               "final clear left a per-sequence position populated");
    }
    expect(comp_binding(comp_pool, 0, "final") == final_binding0 &&
               comp_binding(comp_pool, 1, "final") == final_binding1 &&
               comp_binding(comp_pool, 2, "final") == final_binding2,
           "final clear changed compressed binding identities");
    expect_comp_handle_info_equal(replacement_info_before_release,
                                  comp_handle_info(comp_pool, final_binding0, "final"), "final replacement handle");
    expect_comp_handle_info_equal(initial_info1, comp_handle_info(comp_pool, final_binding1, "final"),
                                  "final survivor handle");
    expect_comp_handle_info_equal(destination_info_before_release,
                                  comp_handle_info(comp_pool, final_binding2, "final"), "final destination handle");
    llama_dsv4_comp_handle_info removed_info;
    expect(comp_pool->get_handle(initial_binding0, removed_info) == llama_dsv4_comp_status::handle_not_found &&
               comp_pool->get_handle(initial_binding2, removed_info) == llama_dsv4_comp_status::handle_not_found,
           "final clear resurrected an obsolete compressed handle");
    expect_rs_idx("final");
}

struct backend_scope {
    backend_scope() { llama_backend_init(); }

    ~backend_scope() { llama_backend_free(); }
};

bool silent_progress(float /* progress */, void * /* user_data */) {
    return true;
}

bool parse_arguments(int argc, char ** argv, std::string & model_path, std::string & manifest_path, bool & help,
                     bool & diagnostic_only, bool & diagnostic_all_boundaries, bool & diagnostic_skip_shard_digests) {
    help = argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0);
    if (help) {
        std::printf("usage: %s --model <first-shard.gguf> --manifest <pinned.manifest>\n"
                    "       %s --model <first-shard.gguf> --manifest <pinned.manifest> --diagnostic-only\n"
                    "          [--diagnostic-all-boundaries] [--diagnostic-skip-shard-digests]\n",
                    argv[0], argv[0]);
        return false;
    }
    diagnostic_only = false;
    diagnostic_all_boundaries = false;
    diagnostic_skip_shard_digests = false;
    for (int index = 1; index < argc;) {
        const char * option = argv[index];
        if (std::strcmp(option, "--diagnostic-only") == 0) {
            diagnostic_only = true;
            ++index;
            continue;
        }
        if (std::strcmp(option, "--diagnostic-all-boundaries") == 0) {
            diagnostic_all_boundaries = true;
            ++index;
            continue;
        }
        if (std::strcmp(option, "--diagnostic-skip-shard-digests") == 0) {
            diagnostic_skip_shard_digests = true;
            ++index;
            continue;
        }
        if (index + 1 >= argc) {
            std::fprintf(stderr, "resident-model option requires a value: %s\n", option);
            return false;
        }
        const char * value = argv[index + 1];
        if (std::strcmp(option, "--model") == 0 || std::strcmp(option, "-m") == 0) {
            if (!model_path.empty()) {
                std::fprintf(stderr, "duplicate --model argument\n");
                return false;
            }
            model_path = value;
        } else if (std::strcmp(option, "--manifest") == 0 || std::strcmp(option, "-M") == 0) {
            if (!manifest_path.empty()) {
                std::fprintf(stderr, "duplicate --manifest argument\n");
                return false;
            }
            manifest_path = value;
        } else {
            std::fprintf(stderr, "unknown resident-model argument: %s\n", option);
            return false;
        }
        index += 2;
    }
    if (diagnostic_all_boundaries && !diagnostic_only) {
        std::fprintf(stderr, "--diagnostic-all-boundaries requires --diagnostic-only\n");
        return false;
    }
    if (diagnostic_skip_shard_digests && !diagnostic_only) {
        std::fprintf(stderr, "--diagnostic-skip-shard-digests requires --diagnostic-only\n");
        return false;
    }
    if (model_path.empty() || manifest_path.empty()) {
        std::fprintf(stderr, "resident-model requires both --model and --manifest\n");
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--placement-self-test") == 0) {
        try {
            placement_buft_self_test();
            return 0;
        } catch (const std::exception & error) {
            std::fprintf(stderr, "resident-model placement self-test failure: %s\n", error.what());
            return 1;
        }
    }
    std::string model_path;
    std::string manifest_path;
    bool        help = false;
    bool        diagnostic_only = false;
    bool        diagnostic_all_boundaries = false;
    bool        diagnostic_skip_shard_digests = false;
    if (!parse_arguments(argc, argv, model_path, manifest_path, help, diagnostic_only,
                         diagnostic_all_boundaries, diagnostic_skip_shard_digests)) {
        return help ? 0 : 2;
    }
    if (!env_is_one("LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE") || !env_is_one("LLAMA_DSV4_AGGREGATE_POOL_FORCE") ||
        std::getenv("LLAMA_DSV4_AMX_COEXEC") != nullptr) {
        std::fprintf(stderr,
                     "resident-model requires LLAMA_DSV4_COMPOSITE_RESIDENT_ENABLE=1, "
                     "LLAMA_DSV4_AGGREGATE_POOL_FORCE=1, and no LLAMA_DSV4_AMX_COEXEC\n");
        return 2;
    }

    try {
        const model_manifest manifest = read_model_manifest(manifest_path);
        if (diagnostic_only) {
            std::fprintf(stderr,
                         "resident-model diagnostic-only enabled; all-boundaries=%s; shard digests=%s; "
                         "result is forced nonzero\n",
                         diagnostic_all_boundaries ? "enabled" : "disabled",
                         diagnostic_skip_shard_digests ? "skipped" : "verified");
        }
        // Validate the supplied path before the platform gate so wrong-model
        // failure paths are deterministic even on a non-Metal host. Hashing
        // only proceeds after the canonical five-shard layout is present.
        const model_file_snapshots model_snapshots = verify_model_files(model_path, manifest, nullptr, "pre-load",
                                                                         diagnostic_skip_shard_digests);
        backend_scope backend;
        const ggml_backend_dev_t target_device = verify_target_metal_device();
        std::array<ggml_backend_dev_t, 2> model_devices = { target_device, nullptr };
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 999;
        model_params.split_mode         = LLAMA_SPLIT_MODE_LAYER;
        model_params.devices             = model_devices.data();
        model_params.progress_callback  = silent_progress;
        model_ptr model(llama_model_load_from_file(model_path.c_str(), model_params), llama_model_free);
        expect(model != nullptr, "failed to load exact DeepSeek V4 Flash model");
        (void) verify_model_files(model_path, manifest, &model_snapshots, "post-load", diagnostic_skip_shard_digests);
        verify_model_placement(model.get(), target_device);
        verify_model_metadata(model.get(), manifest);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        expect(n_vocab > 1, "exact model vocabulary is empty");
        std::fprintf(stderr,
                     "resident-model start n_ctx=%u n_batch=%u n_ubatch=%u n_seq_max=%u n_rs_seq=%u "
                     "n_outputs_max=%u flash_attn=1 boundaries=%zu\n",
                     N_CTX, N_BATCH, N_UBATCH, N_SEQ_MAX, N_RS_SEQ, N_OUTPUTS, BOUNDARIES.size());
        for (uint32_t boundary : BOUNDARIES) {
            run_boundary(model.get(), n_vocab, boundary, diagnostic_only && boundary == BOUNDARIES.front());
            if (diagnostic_only && !diagnostic_all_boundaries) {
                std::fprintf(stderr,
                             "resident-model diagnostic-only stopped after boundary=%u; forced nonzero result\n",
                             boundary);
                return 3;
            }
            if (diagnostic_only) {
                std::fprintf(stderr,
                             "resident-model diagnostic-all-boundaries completed boundary=%u/%zu; "
                             "result remains forced nonzero\n",
                             boundary, BOUNDARIES.size());
            }
        }
        if (diagnostic_only) {
            std::fprintf(stderr,
                         "resident-model diagnostic-all-boundaries complete boundaries=%zu; forced nonzero result\n",
                         BOUNDARIES.size());
            return 3;
        }
        std::fprintf(stderr, "resident-model complete boundaries=%zu\n", BOUNDARIES.size());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "resident-model failure: %s\n", error.what());
        return 1;
    }
}

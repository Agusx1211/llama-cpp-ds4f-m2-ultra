#include "llama-snapshot-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

void expect_status(llama_snapshot_status actual, llama_snapshot_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_snapshot_status_name(expected) + ", got " +
             llama_snapshot_status_name(actual));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        char   pattern[] = "/tmp/llama-snapshot-store-XXXXXX";
        char * created   = ::mkdtemp(pattern);
        if (created == nullptr) {
            fail("mkdtemp failed: " + std::string(std::strerror(errno)));
        }
        directory = created;
    }

    ~temporary_directory() {
        std::error_code error;
        fs::remove_all(directory, error);
    }

    temporary_directory(const temporary_directory &)             = delete;
    temporary_directory & operator=(const temporary_directory &) = delete;

    const fs::path & path() const { return directory; }

  private:
    fs::path directory;
};

llama_snapshot_digest digest_text(const std::string & value) {
    return llama_snapshot_sha256(value.data(), value.size());
}

llama_snapshot_identity make_identity() {
    llama_snapshot_identity identity;
    identity.architecture          = "deepseek-v4-flash";
    identity.model_artifact_digest = digest_text("five-shard-model-artifact");
    identity.tokenizer_digest      = digest_text("tokenizer");
    identity.chat_template_digest  = digest_text("chat-template");
    identity.runtime_build_digest  = digest_text("runtime-build");
    identity.target_kv_digest      = digest_text("target-kv-layout");
    identity.draft_kv_digest       = digest_text("draft-kv-layout");
    identity.rope_digest           = digest_text("rope-parameters");
    identity.lora_digest           = digest_text("no-lora");
    identity.target_kv_type        = "dsv4-target-f16-c4-hca128";
    identity.draft_kv_type         = "dsv4-draft-f16";
    identity.context_size          = 1048576;
    identity.raw_window            = 4096;
    identity.c4_ratio              = 4;
    identity.hca_ratio             = 128;
    identity.dsv4_state_version    = 3;
    identity.rollback_depth        = 8;
    return identity;
}

llama_snapshot_store_config make_config(const fs::path & root) {
    llama_snapshot_store_config config;
    config.root_path              = root.string();
    config.physical_device_id     = "apple-internal-ssd-physical-0";
    config.physical_device_queues = 1;
    config.chunk_payload_bytes    = 64;
    config.max_chunks             = 64;
    config.max_manifest_bytes     = 64 * 1024;
    config.max_snapshot_bytes     = 4096;
    return config;
}

llama_snapshot_metadata make_metadata(uint64_t generation, const llama_snapshot_identity & identity) {
    llama_snapshot_metadata metadata;
    metadata.snapshot_generation = generation;
    metadata.request_generation  = 1000 + generation;
    metadata.identity            = identity;
    return metadata;
}

std::vector<uint8_t> make_payload(size_t size, uint8_t seed) {
    std::vector<uint8_t> result(size);
    for (size_t index = 0; index < size; ++index) {
        result[index] = static_cast<uint8_t>(seed + index * 37 + index / 5);
    }
    return result;
}

std::string generation_name(uint64_t generation, bool partial = false) {
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

void write_byte(const fs::path & path, uint64_t offset, uint8_t value) {
    const int descriptor = ::open(path.c_str(), O_RDWR);
    if (descriptor < 0) {
        fail("open for mutation failed");
    }
    if (::pwrite(descriptor, &value, 1, static_cast<off_t>(offset)) != 1 || ::fsync(descriptor) != 0) {
        const int error = errno;
        ::close(descriptor);
        fail("pwrite mutation failed: " + std::string(std::strerror(error)));
    }
    if (::close(descriptor) != 0) {
        fail("close mutation failed");
    }
}

uint8_t read_byte(const fs::path & path, uint64_t offset) {
    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        fail("open for read-byte failed");
    }
    uint8_t value = 0;
    if (::pread(descriptor, &value, 1, static_cast<off_t>(offset)) != 1) {
        ::close(descriptor);
        fail("pread failed");
    }
    ::close(descriptor);
    return value;
}

void truncate_last_byte(const fs::path & path) {
    std::error_code error;
    const uint64_t  size = fs::file_size(path, error);
    if (error || size == 0 || ::truncate(path.c_str(), static_cast<off_t>(size - 1)) != 0) {
        fail("truncate failed");
    }
}

uint32_t count_generation_directories(const fs::path & root) {
    uint32_t count = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        if (entry.is_directory() && entry.path().filename().string().rfind("generation-", 0) == 0) {
            ++count;
        }
    }
    return count;
}

uint32_t count_partial_paths(const fs::path & root) {
    uint32_t count = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        if (entry.path().filename().string().rfind(".partial-generation-", 0) == 0 ||
            entry.path().filename() == "current.manifest.tmp") {
            ++count;
        }
    }
    return count;
}

llama_snapshot_manifest open_ok(llama_snapshot_store & store, const llama_snapshot_identity & identity) {
    auto opened = store.open_current(identity);
    expect_status(opened.status, llama_snapshot_status::ok, "open current generation");
    return opened.manifest;
}

void expect_current_payload(llama_snapshot_store &          store,
                            const llama_snapshot_identity & identity,
                            uint64_t                        generation,
                            const std::vector<uint8_t> &    expected) {
    const auto manifest = open_ok(store, identity);
    expect(manifest.snapshot_generation == generation, "unexpected current generation");
    expect_status(store.validate(manifest), llama_snapshot_status::ok, "validate current generation");
    const auto read = store.read_all(manifest);
    expect_status(read.status, llama_snapshot_status::ok, "read current generation");
    expect(read.payload == expected, "current payload mismatch");
}

void test_sha256_vectors_and_config_contract() {
    expect(llama_snapshot_digest_hex(llama_snapshot_sha256("", 0)) ==
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "SHA-256 empty vector");
    const std::string abc = "abc";
    expect(llama_snapshot_digest_hex(llama_snapshot_sha256(abc.data(), abc.size())) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "SHA-256 abc vector");

    temporary_directory temp;
    auto                config    = make_config(temp.path());
    config.physical_device_queues = 2;
    llama_snapshot_store invalid(config);
    const auto           write = invalid.write_generation(make_metadata(1, make_identity()), make_payload(1, 3));
    expect_status(write.status, llama_snapshot_status::invalid_argument, "multiple physical queues admitted");
}

void test_round_trip_and_multi_chunk_ordering() {
    temporary_directory  temp;
    const auto           identity = make_identity();
    llama_snapshot_store store(make_config(temp.path()));
    expect_status(store.open_current(identity).status, llama_snapshot_status::no_current_generation,
                  "empty store open");

    const auto payload = make_payload(200, 11);
    const auto write   = store.write_generation(make_metadata(1, identity), payload);
    expect_status(write.status, llama_snapshot_status::ok, "multi-chunk write");
    expect(write.committed && write.generation == 1, "multi-chunk write did not commit");

    const auto manifest = open_ok(store, identity);
    expect(manifest.format_version == 1 && manifest.snapshot_generation == 1 && manifest.request_generation == 1001,
           "manifest version/generation metadata");
    expect(manifest.physical_device_id == store.config().physical_device_id &&
               manifest.physical_device_queues == LLAMA_SNAPSHOT_DEVICE_QUEUES,
           "physical device queue metadata");
    expect(manifest.chunk_payload_limit == 64 && manifest.total_payload_bytes == payload.size(),
           "manifest size metadata");
    expect(manifest.chunks.size() == 4, "multi-chunk count");
    const uint64_t       expected_sizes[] = { 64, 64, 64, 8 };
    uint64_t             offset           = 0;
    std::vector<uint8_t> assembled;
    for (uint32_t index = 0; index < manifest.chunks.size(); ++index) {
        const auto & chunk = manifest.chunks[index];
        expect(chunk.index == index && chunk.logical_offset == offset && chunk.payload_bytes == expected_sizes[index],
               "chunk ordering metadata");
        const auto read = store.read_chunk(manifest, index);
        expect_status(read.status, llama_snapshot_status::ok, "ordered chunk read");
        assembled.insert(assembled.end(), read.payload.begin(), read.payload.end());
        const fs::path path = temp.path() / generation_name(1) / chunk_name(index);
        expect(fs::file_size(path) == LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + chunk.payload_bytes,
               "chunk payload is not 64 KiB aligned");
        offset += chunk.payload_bytes;
    }
    expect(assembled == payload, "chunk assembly order");
    expect_status(store.validate(manifest), llama_snapshot_status::ok, "multi-chunk validation");
    expect(store.read_all(manifest).payload == payload, "multi-chunk read_all");
    expect(fs::exists(temp.path() / generation_name(1) / "generation.manifest"), "generation-local manifest missing");
}

void test_chunk_corruption_truncation_and_missing() {
    const auto identity = make_identity();
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        const auto           payload = make_payload(100, 21);
        expect_status(store.write_generation(make_metadata(1, identity), payload).status, llama_snapshot_status::ok,
                      "write corruption fixture");
        const auto     manifest = open_ok(store, identity);
        const fs::path chunk    = temp.path() / generation_name(1) / chunk_name(0);
        write_byte(chunk, LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + 3,
                   static_cast<uint8_t>(read_byte(chunk, LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + 3) ^ 0x80));
        expect_status(store.validate(manifest), llama_snapshot_status::checksum_mismatch, "chunk corruption");
        expect_status(store.read_all(manifest).status, llama_snapshot_status::checksum_mismatch,
                      "read corrupted chunk");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(100, 22)).status,
                      llama_snapshot_status::ok, "write truncation fixture");
        const auto manifest = open_ok(store, identity);
        truncate_last_byte(temp.path() / generation_name(1) / chunk_name(1));
        expect_status(store.validate(manifest), llama_snapshot_status::truncated, "truncated chunk");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(100, 23)).status,
                      llama_snapshot_status::ok, "write missing fixture");
        const auto manifest = open_ok(store, identity);
        expect(::unlink((temp.path() / generation_name(1) / chunk_name(1)).c_str()) == 0, "unlink missing chunk");
        expect_status(store.validate(manifest), llama_snapshot_status::missing_chunk, "missing chunk");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(20, 24)).status,
                      llama_snapshot_status::ok, "write chunk-header fixture");
        const auto manifest = open_ok(store, identity);
        write_byte(temp.path() / generation_name(1) / chunk_name(0), 8, 2);
        expect_status(store.validate(manifest), llama_snapshot_status::format_error, "chunk format version");
    }
}

void test_manifest_corruption_truncation_identity_and_bounds() {
    const auto identity = make_identity();
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(80, 31)).status,
                      llama_snapshot_status::ok, "write manifest checksum fixture");
        const fs::path manifest = temp.path() / "current.manifest";
        write_byte(manifest, 32, static_cast<uint8_t>(read_byte(manifest, 32) ^ 1));
        expect_status(store.open_current(identity).status, llama_snapshot_status::checksum_mismatch,
                      "manifest checksum corruption");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(80, 32)).status,
                      llama_snapshot_status::ok, "write manifest truncation fixture");
        truncate_last_byte(temp.path() / "current.manifest");
        expect_status(store.open_current(identity).status, llama_snapshot_status::truncated, "truncated manifest");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(80, 33)).status,
                      llama_snapshot_status::ok, "write manifest magic fixture");
        write_byte(temp.path() / "current.manifest", 0, 'X');
        expect_status(store.open_current(identity).status, llama_snapshot_status::format_error, "manifest magic");
    }
    {
        temporary_directory  temp;
        const auto           config = make_config(temp.path());
        llama_snapshot_store store(config);
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(80, 34)).status,
                      llama_snapshot_status::ok, "write identity fixture");
        auto wrong_model                  = identity;
        wrong_model.model_artifact_digest = digest_text("different-model");
        expect_status(store.open_current(wrong_model).status, llama_snapshot_status::identity_mismatch,
                      "wrong model identity");
        auto wrong_kv             = identity;
        wrong_kv.target_kv_digest = digest_text("different-kv");
        expect_status(store.open_current(wrong_kv).status, llama_snapshot_status::identity_mismatch,
                      "wrong KV identity");
        auto wrong_rope        = identity;
        wrong_rope.rope_digest = digest_text("different-rope");
        expect_status(store.open_current(wrong_rope).status, llama_snapshot_status::identity_mismatch,
                      "wrong RoPE identity");

        auto wrong_device_config               = config;
        wrong_device_config.physical_device_id = "different-physical-device";
        llama_snapshot_store wrong_device(wrong_device_config);
        expect_status(wrong_device.open_current(identity).status, llama_snapshot_status::device_mismatch,
                      "wrong physical device identity");

        auto bounded_config               = config;
        bounded_config.max_manifest_bytes = 256;
        llama_snapshot_store bounded(bounded_config);
        expect_status(bounded.open_current(identity).status, llama_snapshot_status::manifest_too_large,
                      "bounded manifest read");

        auto count_config       = config;
        count_config.max_chunks = 1;
        llama_snapshot_store bounded_count(count_config);
        expect_status(bounded_count.open_current(identity).status, llama_snapshot_status::format_error,
                      "bounded manifest chunk count");
    }
    {
        temporary_directory  temp;
        llama_snapshot_store store(make_config(temp.path()));
        expect_status(store.write_generation(make_metadata(1, identity), make_payload(20, 35)).status,
                      llama_snapshot_status::ok, "write trailing manifest fixture");
        const int descriptor = ::open((temp.path() / "current.manifest").c_str(), O_WRONLY | O_APPEND);
        expect(descriptor >= 0, "open trailing manifest");
        const uint8_t extra = 0;
        expect(::write(descriptor, &extra, 1) == 1 && ::fsync(descriptor) == 0 && ::close(descriptor) == 0,
               "append trailing manifest byte");
        expect_status(store.open_current(identity).status, llama_snapshot_status::trailing_data,
                      "manifest trailing data");
    }
}

void test_transactional_failures_and_cleanup() {
    temporary_directory  temp;
    const auto           identity = make_identity();
    llama_snapshot_store store(make_config(temp.path()));
    const auto           original = make_payload(130, 41);
    expect_status(store.write_generation(make_metadata(1, identity), original).status, llama_snapshot_status::ok,
                  "write prior committed generation");

    const auto cancelled_payload = make_payload(150, 42);
    const auto cancelled         = store.write_generation(make_metadata(2, identity), cancelled_payload,
                                                          [](uint32_t durable_chunks) { return durable_chunks >= 1; });
    expect_status(cancelled.status, llama_snapshot_status::cancelled, "cancel generation");
    expect(!cancelled.committed && count_partial_paths(temp.path()) == 0, "cancel cleanup");
    expect_current_payload(store, identity, 1, original);
    expect(fs::exists(temp.path() / generation_name(1)), "cancel released prior generation");

    llama_snapshot_faults no_space;
    no_space.fail_after_bytes = LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + 9;
    no_space.write_fault      = llama_snapshot_write_fault::no_space;
    const auto enospc         = store.write_generation(make_metadata(3, identity), make_payload(150, 43), {}, no_space);
    expect_status(enospc.status, llama_snapshot_status::no_space, "ENOSPC injection");
    expect(enospc.os_error == ENOSPC && !enospc.committed && count_partial_paths(temp.path()) == 0,
           "ENOSPC cleanup/status");
    expect_current_payload(store, identity, 1, original);

    llama_snapshot_faults short_write;
    short_write.fail_after_bytes = LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + 7;
    short_write.write_fault      = llama_snapshot_write_fault::zero_progress;
    const auto short_result =
        store.write_generation(make_metadata(4, identity), make_payload(150, 44), {}, short_write);
    expect_status(short_result.status, llama_snapshot_status::short_write, "short-write injection");
    expect(!short_result.committed && count_partial_paths(temp.path()) == 0, "short-write cleanup");
    expect_current_payload(store, identity, 1, original);

    llama_snapshot_faults before_commit;
    before_commit.fail_before_manifest_commit = true;
    const auto precommit = store.write_generation(make_metadata(5, identity), make_payload(150, 45), {}, before_commit);
    expect_status(precommit.status, llama_snapshot_status::io_error, "pre-manifest failure injection");
    expect(!precommit.committed && !fs::exists(temp.path() / generation_name(5)), "pre-manifest cleanup");
    expect_current_payload(store, identity, 1, original);

    llama_snapshot_faults torn;
    torn.fail_after_bytes           = LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + 5;
    torn.write_fault                = llama_snapshot_write_fault::no_space;
    torn.preserve_failed_generation = true;
    const auto torn_result = store.write_generation(make_metadata(6, identity), make_payload(150, 46), {}, torn);
    expect_status(torn_result.status, llama_snapshot_status::no_space, "torn temp generation injection");
    expect(count_partial_paths(temp.path()) == 1, "torn temp generation was not preserved");
    expect_current_payload(store, identity, 1, original);
    const fs::path unrelated = temp.path() / "keep.txt";
    const int      keep      = ::open(unrelated.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    expect(keep >= 0 && ::write(keep, "x", 1) == 1 && ::close(keep) == 0, "create unrelated cleanup fixture");
    const auto cleanup = store.cleanup_temporary_generations();
    expect_status(cleanup.status, llama_snapshot_status::ok, "cleanup torn generation");
    expect(cleanup.removed == 1 && count_partial_paths(temp.path()) == 0 && fs::exists(unrelated),
           "cleanup removed wrong paths");

    llama_snapshot_faults retry_short_writes;
    retry_short_writes.max_write_size = 3;
    const auto replacement            = make_payload(151, 47);
    const auto replacement_result =
        store.write_generation(make_metadata(7, identity), replacement, {}, retry_short_writes);
    expect_status(replacement_result.status, llama_snapshot_status::ok, "retry ordinary short writes");
    expect_current_payload(store, identity, 7, replacement);
    expect(!fs::exists(temp.path() / generation_name(1)) && count_generation_directories(temp.path()) == 1,
           "prior generation released before/after wrong commit boundary");
}

void test_repeated_replacement_and_capacity_limits() {
    temporary_directory  temp;
    const auto           identity = make_identity();
    auto                 config   = make_config(temp.path());
    llama_snapshot_store store(config);
    std::vector<uint8_t> expected;
    for (uint64_t generation = 1; generation <= 32; ++generation) {
        expected         = make_payload(static_cast<size_t>(generation * 3), static_cast<uint8_t>(generation));
        const auto write = store.write_generation(make_metadata(generation, identity), expected);
        expect_status(write.status, llama_snapshot_status::ok, "repeated generation replacement");
        expect_current_payload(store, identity, generation, expected);
        expect(count_generation_directories(temp.path()) == 1 && count_partial_paths(temp.path()) == 0,
               "replacement retained stale generation");
    }
    expect_status(store.write_generation(make_metadata(32, identity), expected).status,
                  llama_snapshot_status::generation_exists, "duplicate generation");
    expect_status(store.write_generation(make_metadata(31, identity), expected).status,
                  llama_snapshot_status::stale_generation, "stale generation");

    auto wrong_identity             = identity;
    wrong_identity.target_kv_digest = digest_text("replacement-wrong-kv");
    expect_status(store.write_generation(make_metadata(33, wrong_identity), make_payload(1, 9)).status,
                  llama_snapshot_status::identity_mismatch, "replace with wrong identity");
    expect_current_payload(store, identity, 32, expected);

    const auto empty_write = store.write_generation(make_metadata(33, identity), {});
    expect_status(empty_write.status, llama_snapshot_status::ok, "empty snapshot write");
    const auto empty_manifest = open_ok(store, identity);
    expect(empty_manifest.chunks.empty() && empty_manifest.total_payload_bytes == 0, "empty snapshot manifest");
    expect(store.read_all(empty_manifest).payload.empty(), "empty snapshot read");

    temporary_directory limited_temp;
    auto                limited_config = make_config(limited_temp.path());
    limited_config.chunk_payload_bytes = 8;
    limited_config.max_chunks          = 2;
    limited_config.max_snapshot_bytes  = 64;
    llama_snapshot_store limited(limited_config);
    expect_status(limited.write_generation(make_metadata(1, identity), make_payload(17, 4)).status,
                  llama_snapshot_status::chunk_too_large, "bounded chunk count admission");
    expect_status(limited.open_current(identity).status, llama_snapshot_status::no_current_generation,
                  "capacity failure published manifest");
}

}  // namespace

int main() {
    try {
        test_sha256_vectors_and_config_contract();
        test_round_trip_and_multi_chunk_ordering();
        test_chunk_corruption_truncation_and_missing();
        test_manifest_corruption_truncation_identity_and_bounds();
        test_transactional_failures_and_cleanup();
        test_repeated_replacement_and_capacity_limits();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-snapshot-store: %s\n", error.what());
        return 1;
    }
    std::puts("test-snapshot-store: all checks passed");
    return 0;
}

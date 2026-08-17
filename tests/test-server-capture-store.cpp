#include "server-capture-store.h"
#include "server-capture-sha256.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace server_capture;

namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

[[noreturn]] void throw_worker_barrier_failure() {
    throw std::runtime_error("worker barrier failure");
}

void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

void expect_status(capture_store_status actual, capture_store_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + capture_store_status_name(expected) + ", got " +
             capture_store_status_name(actual));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        const char *   configured_base = std::getenv("TMPDIR");
        const fs::path configured =
            configured_base != nullptr && *configured_base != '\0' ? fs::path(configured_base) : fs::path("/tmp");
        std::error_code canonical_error;
        const fs::path  canonical_base = fs::canonical(configured, canonical_error);
        if (canonical_error || canonical_base.empty() || !canonical_base.is_absolute()) {
            fail("canonical temporary base failed: " + canonical_error.message());
        }
        std::string       pattern = (canonical_base / "llama-capture-store-XXXXXX").string();
        std::vector<char> writable_pattern(pattern.begin(), pattern.end());
        writable_pattern.push_back('\0');
        char * path = ::mkdtemp(writable_pattern.data());
        if (path == nullptr) {
            fail("mkdtemp failed: " + std::string(std::strerror(errno)));
        }
        value = path;
    }

    ~temporary_directory() {
        std::error_code ignored;
        fs::remove_all(value, ignored);
    }

    temporary_directory(const temporary_directory &)             = delete;
    temporary_directory & operator=(const temporary_directory &) = delete;

    const fs::path & path() const { return value; }

  private:
    fs::path value;
};

cycle_observation make_observation(uint32_t sequence) {
    cycle_observation observation;
    observation.request_id                    = 0x1122334455660000ULL + sequence;
    observation.committed_position            = sequence * 5U;
    observation.scheduler_epoch               = 700 + sequence;
    observation.monotonic_ns                  = 1000 + sequence * 100;
    observation.cycle_sequence                = sequence;
    observation.target_correction_or_bonus_id = 0x6a6b6c00 + static_cast<int32_t>(sequence);
    observation.draft_time_us                 = 10 + sequence;
    observation.verify_time_us                = 20 + sequence;
    observation.scheduler_time_us             = 3;
    observation.scheduled_decode_width        = 5;
    observation.verifier_geometry             = 5;
    observation.proposal_count                = 5;
    observation.accepted_prefix_length        = 3;
    observation.first_rejection               = 3;
    observation.proposed_by                   = proposer::dspark;
    observation.bypass                        = bypass_reason::none;
    for (size_t index = 0; index < MAX_PROPOSAL_TOKENS; ++index) {
        observation.proposal_token_ids[index]     = 0x504f0000 + static_cast<int32_t>(sequence * 8 + index);
        // power-of-two fraction steps keep every operation exact in fp32, so
        // the serialized bytes (and the digest oracle below) are identical
        // whether or not the compiler contracts a*b+c into an FMA (x86 vs
        // ARM64 diverge on inexact steps like 0.02f)
        observation.selected_probabilities[index] = 0.5F + static_cast<float>(index) * 0.015625F;
        observation.raw_confidences[index]        = 0.75F - static_cast<float>(index) * 0.03125F;
    }
    return observation;
}

capture_store_config make_config(const fs::path & root) {
    capture_store_config config;
    config.root_path            = root.string();
    config.ring_capacity        = 8;
    config.max_shard_records    = 2;
    config.max_shard_bytes      = 4096;
    config.max_retained_shards  = 8;
    config.max_retained_records = 32;
    config.max_retained_bytes   = 32768;
    config.max_manifest_bytes   = 65536;
    config.identity_salt        = {
        { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 }
    };
    return config;
}

std::vector<uint8_t> read_bytes(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    expect(input.good(), "open file for test read");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

fs::path only_shard(const fs::path & root) {
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        if (entry.path().extension() == ".cap") {
            return entry.path();
        }
    }
    fail("no shard found");
}

void mutate_byte(const fs::path & path, uint64_t offset) {
    const int descriptor = ::open(path.c_str(), O_RDWR);
    expect(descriptor >= 0, "open mutation fixture");
    uint8_t value = 0;
    expect(::pread(descriptor, &value, sizeof(value), static_cast<off_t>(offset)) == 1, "read mutation fixture");
    value ^= 0x80U;
    expect(::pwrite(descriptor, &value, sizeof(value), static_cast<off_t>(offset)) == 1, "write mutation fixture");
    expect(::fsync(descriptor) == 0 && ::close(descriptor) == 0, "close mutation fixture");
}

void truncate_file(const fs::path & path, uint64_t bytes) {
    expect(::truncate(path.c_str(), static_cast<off_t>(bytes)) == 0, "truncate fixture");
}

void write_private_file(const fs::path & path, const std::vector<uint8_t> & bytes) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    expect(descriptor >= 0, "open private fixture");
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        expect(count > 0, "write private fixture");
        offset += static_cast<size_t>(count);
    }
    expect(::fsync(descriptor) == 0 && ::close(descriptor) == 0, "close private fixture");
}

void expect_invalid_config(capture_store_config config, const std::string & message) {
    bool threw = false;
    try {
        capture_store invalid(std::move(config));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, message);
}

void test_compact_round_trip_and_privacy() {
    temporary_directory temp;
    capture_store       store(make_config(temp.path()));
    expect(store.try_enqueue(make_observation(1)), "compact enqueue one");
    expect(store.try_enqueue(make_observation(2)), "compact enqueue two");
    expect_status(store.flush().status, capture_store_status::ok, "compact flush");
    capture_manifest manifest;
    expect_status(store.inspect(manifest).status, capture_store_status::ok, "compact inspect");
    expect(manifest.format_version == CAPTURE_STORE_FORMAT_VERSION && manifest.shards.size() == 1,
           "compact manifest shape");
    expect(manifest.total_records == 2 && manifest.identity_salt[0] == 1, "compact manifest counters/salt");
    expect_status(store.validate(manifest).status, capture_store_status::ok, "compact validate");
    const std::vector<uint8_t> bytes        = read_bytes(only_shard(temp.path()));
    const auto                 observation  = make_observation(1);
    const auto                 contains_u64 = [&bytes](uint64_t value) {
        std::array<uint8_t, 8> needle = {};
        for (unsigned shift = 0; shift < 64; shift += 8) {
            needle[shift / 8] = static_cast<uint8_t>(value >> shift);
        }
        return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
    };
    expect(!contains_u64(observation.request_id), "compact persisted raw request identity");
    expect(!contains_u64(static_cast<uint64_t>(static_cast<uint32_t>(observation.proposal_token_ids[0]))),
           "compact persisted raw proposal token");
    expect(!contains_u64(static_cast<uint64_t>(static_cast<uint32_t>(observation.target_correction_or_bonus_id))),
           "compact persisted raw correction token");
    expect_status(store.shutdown().status, capture_store_status::ok, "compact shutdown");
}

void test_sha256_file_footers() {
    temporary_directory temp;
    capture_store       store(make_config(temp.path()));
    expect(store.try_enqueue(make_observation(1)), "digest fixture enqueue one");
    expect(store.try_enqueue(make_observation(2)), "digest fixture enqueue two");
    expect_status(store.flush().status, capture_store_status::ok, "digest fixture flush");

    const std::vector<uint8_t> manifest_bytes = read_bytes(temp.path() / "capture.manifest");
    const std::vector<uint8_t> shard_bytes    = read_bytes(only_shard(temp.path()));
    // 80 + 72 + 32 manifest; 72 + 32 shard framing + 2 * (16 + 192) records
    expect(manifest_bytes.size() == 184 && shard_bytes.size() == 520, "digest fixture sizes");

    // These values come from Python hashlib.sha256 over each payload prefix,
    // independently of the in-tree implementation under test.
    const capture_digest expected_manifest = {
        { 0xc2, 0x40, 0xb4, 0xba, 0x3c, 0x10, 0xc7, 0x31, 0xf0, 0x49, 0x9e, 0x68, 0x2b, 0x6f, 0x35, 0x87,
          0xe4, 0xe4, 0x80, 0x1c, 0xaf, 0x73, 0xba, 0x60, 0x5e, 0xd6, 0x6b, 0x25, 0x04, 0xa8, 0xb0, 0xfb }
    };
    const capture_digest expected_shard = {
        { 0x03, 0xd3, 0xf5, 0x1d, 0xa2, 0x49, 0x35, 0x7a, 0x28, 0x62, 0x69, 0xdd, 0xc8, 0x6e, 0xaa, 0x2d,
          0xef, 0x74, 0x16, 0xb4, 0x17, 0x63, 0xd3, 0x23, 0x00, 0xb2, 0x09, 0x15, 0x6c, 0xff, 0x94, 0xa9 }
    };
    const auto footer = [](const std::vector<uint8_t> & bytes) {
        capture_digest digest = {};
        std::copy_n(bytes.end() - static_cast<std::ptrdiff_t>(CAPTURE_SHARD_FOOTER_BYTES), digest.size(),
                    digest.begin());
        return digest;
    };
    const capture_digest actual_manifest = [&manifest_bytes]() {
        capture_digest digest = {};
        std::copy_n(manifest_bytes.end() - static_cast<std::ptrdiff_t>(CAPTURE_MANIFEST_FOOTER_BYTES), digest.size(),
                    digest.begin());
        return digest;
    }();
    const capture_digest actual_shard = footer(shard_bytes);
    expect(capture_sha256::equal(actual_manifest, expected_manifest), "SCAPMF01 digest oracle");
    expect(capture_sha256::equal(actual_shard, expected_shard), "SCAPSH01 digest oracle");
    expect(capture_sha256::equal(actual_manifest,
                                 capture_sha256::hash(manifest_bytes.data(),
                                                     manifest_bytes.size() - CAPTURE_MANIFEST_FOOTER_BYTES)),
           "SCAPMF01 footer matches shared SHA-256");
    expect(capture_sha256::equal(actual_shard,
                                 capture_sha256::hash(shard_bytes.data(), shard_bytes.size() - CAPTURE_SHARD_FOOTER_BYTES)),
           "SCAPSH01 footer matches shared SHA-256");
    expect_status(store.shutdown().status, capture_store_status::ok, "digest fixture shutdown");
}

void test_sampled_rich_explicit_opt_in() {
    temporary_directory  temp;
    capture_store_config rejected = make_config(temp.path());
    rejected.capture_mode         = mode::sampled_rich;
    bool threw                    = false;
    try {
        capture_store invalid(rejected);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "sampled-rich accepted without explicit opt-in");

    rejected.allow_sampled_rich = true;
    rejected.rich_sample_every  = 1;
    capture_store store(rejected);
    const auto    observation = make_observation(9);
    expect(store.try_enqueue(observation), "rich enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "rich flush");
    const std::vector<uint8_t> bytes    = read_bytes(only_shard(temp.path()));
    std::array<uint8_t, 4>     token    = {};
    const uint32_t             token_id = static_cast<uint32_t>(observation.proposal_token_ids[0]);
    for (unsigned shift = 0; shift < 32; shift += 8) {
        token[shift / 8] = static_cast<uint8_t>(token_id >> shift);
    }
    expect(std::search(bytes.begin(), bytes.end(), token.begin(), token.end()) != bytes.end(),
           "rich token IDs were not emitted after opt-in");
    capture_manifest manifest;
    expect_status(store.inspect(manifest).status, capture_store_status::ok, "rich inspect");
    expect_status(store.validate(manifest).status, capture_store_status::ok, "rich validate");
    expect_status(store.shutdown().status, capture_store_status::ok, "rich shutdown");
}

void test_metrics_only_identity_redaction() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.capture_mode         = mode::metrics_only;
    capture_store store(config);
    const auto    observation = make_observation(7);
    expect(store.try_enqueue(observation), "metrics-only enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "metrics-only flush");
    const std::vector<uint8_t>   bytes = read_bytes(only_shard(temp.path()));
    const std::array<uint8_t, 8> zeros = {};
    expect(bytes.size() >= CAPTURE_SHARD_HEADER_BYTES + CAPTURE_RECORD_HEADER_BYTES + zeros.size() &&
               std::equal(zeros.begin(), zeros.end(),
                          bytes.begin() +
                              static_cast<std::ptrdiff_t>(CAPTURE_SHARD_HEADER_BYTES + CAPTURE_RECORD_HEADER_BYTES)),
           "metrics-only request identity was not redacted");
    expect_status(store.shutdown().status, capture_store_status::ok, "metrics-only shutdown");
}

void test_retention_and_restart() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.max_retained_shards  = 2;
    config.max_retained_records = 4;
    capture_store store(config);
    for (uint32_t sequence = 0; sequence < 6; ++sequence) {
        expect(store.try_enqueue(make_observation(sequence)), "retention enqueue");
    }
    expect_status(store.flush().status, capture_store_status::ok, "retention flush");
    capture_manifest manifest;
    expect_status(store.inspect(manifest).status, capture_store_status::ok, "retention inspect");
    expect(manifest.shards.size() == 2 && manifest.shards.front().sequence == 2 && manifest.shards.back().sequence == 3,
           "count retention did not retain newest shards");
    expect_status(store.shutdown().status, capture_store_status::ok, "retention shutdown");
    capture_store    restarted(config);
    capture_manifest recovered;
    expect_status(restarted.inspect(recovered).status, capture_store_status::ok, "restart inspect");
    expect(recovered.shards.size() == 2 && recovered.total_records == 4, "restart retention state");
    expect_status(restarted.validate(recovered).status, capture_store_status::ok, "restart validate");
    expect_status(restarted.shutdown().status, capture_store_status::ok, "restart shutdown");

    temporary_directory  age_temp;
    capture_store_config age_config = make_config(age_temp.path());
    age_config.max_shard_records    = 1;
    age_config.max_retained_shards  = 8;
    age_config.max_retained_records = 8;
    age_config.max_retained_age_ns  = 50;
    capture_store age_store(age_config);
    expect(age_store.try_enqueue(make_observation(1)), "age enqueue old");
    expect_status(age_store.flush().status, capture_store_status::ok, "age old flush");
    expect(age_store.try_enqueue(make_observation(2)), "age enqueue new");
    expect_status(age_store.flush().status, capture_store_status::ok, "age new flush");
    capture_manifest age_manifest;
    expect_status(age_store.inspect(age_manifest).status, capture_store_status::ok, "age inspect");
    expect(age_manifest.shards.size() == 1 && age_manifest.shards.front().sequence == 2,
           "age retention did not delete stale shard");
    expect_status(age_store.shutdown().status, capture_store_status::ok, "age shutdown");
}

void test_faults_corruption_and_recovery() {
    temporary_directory  temp;
    capture_store_config config    = make_config(temp.path());
    config.faults.fail_after_bytes = 0;
    config.faults.write_fault      = capture_write_fault::no_space;
    capture_store no_space(config);
    expect(no_space.try_enqueue(make_observation(1)), "ENOSPC enqueue");
    expect_status(no_space.flush().status, capture_store_status::no_space, "ENOSPC fault");
    expect_status(no_space.shutdown(false).status, capture_store_status::no_space, "ENOSPC shutdown");
    expect(!fs::exists(temp.path() / "capture.manifest"), "ENOSPC published a manifest");

    temporary_directory  short_temp;
    capture_store_config short_config  = make_config(short_temp.path());
    short_config.faults.max_write_size = 3;
    capture_store short_writer(short_config);
    expect(short_writer.try_enqueue(make_observation(1)), "short-write enqueue");
    expect_status(short_writer.flush().status, capture_store_status::ok, "short-write retry");
    expect_status(short_writer.shutdown().status, capture_store_status::ok, "short-write shutdown");

    temporary_directory  malformed_temp;
    capture_store_config malformed_config = make_config(malformed_temp.path());
    capture_store        malformed_writer(malformed_config);
    expect(malformed_writer.try_enqueue(make_observation(1)), "malformed fixture enqueue");
    expect_status(malformed_writer.flush().status, capture_store_status::ok, "malformed fixture flush");
    expect_status(malformed_writer.shutdown().status, capture_store_status::ok, "malformed fixture shutdown");
    mutate_byte(malformed_temp.path() / "capture.manifest", 0);
    capture_store malformed_restart(malformed_config);
    expect(malformed_restart.stats().worker_failed, "malformed manifest did not fail closed");
    capture_manifest malformed_manifest;
    expect_status(malformed_restart.inspect(malformed_manifest).status, capture_store_status::malformed_manifest,
                  "malformed manifest status");
    expect_status(malformed_restart.shutdown(false).status, capture_store_status::malformed_manifest,
                  "malformed shutdown status");

    temporary_directory  trunc_temp;
    capture_store_config trunc_config = make_config(trunc_temp.path());
    capture_store        trunc_writer(trunc_config);
    expect(trunc_writer.try_enqueue(make_observation(1)), "truncation fixture enqueue");
    expect_status(trunc_writer.flush().status, capture_store_status::ok, "truncation fixture flush");
    expect_status(trunc_writer.shutdown().status, capture_store_status::ok, "truncation fixture shutdown");
    const fs::path trunc_shard = only_shard(trunc_temp.path());
    const uint64_t trunc_size  = fs::file_size(trunc_shard);
    truncate_file(trunc_shard, trunc_size - 1);
    capture_store trunc_restart(trunc_config);
    expect(trunc_restart.stats().worker_failed, "truncated shard did not fail closed");
    capture_manifest trunc_manifest;
    expect_status(trunc_restart.inspect(trunc_manifest).status, capture_store_status::ok, "truncated manifest read");
    expect_status(trunc_restart.validate(trunc_manifest).status, capture_store_status::truncated,
                  "truncated shard status");
    expect_status(trunc_restart.shutdown(false).status, capture_store_status::truncated, "truncated shutdown status");

    config        = make_config(temp.path());
    config.faults = {};
    capture_store normal(config);
    expect(normal.try_enqueue(make_observation(2)), "corruption enqueue");
    expect_status(normal.flush().status, capture_store_status::ok, "corruption fixture flush");
    expect_status(normal.shutdown().status, capture_store_status::ok, "corruption fixture shutdown");
    const fs::path shard = only_shard(temp.path());
    mutate_byte(shard, CAPTURE_SHARD_HEADER_BYTES + CAPTURE_RECORD_HEADER_BYTES + 3);
    capture_store corrupted(config);
    expect(corrupted.stats().worker_failed, "corrupted shard did not fail closed on restart");
    capture_manifest ignored;
    expect_status(corrupted.inspect(ignored).status, capture_store_status::ok, "corrupt manifest remains readable");
    expect_status(corrupted.validate(ignored).status, capture_store_status::checksum_mismatch,
                  "corrupt shard checksum");
    expect_status(corrupted.shutdown(false).status, capture_store_status::checksum_mismatch,
                  "corrupt restart terminal status");
}

void test_crash_seams_and_cancellation() {
    temporary_directory  temp;
    capture_store_config before                = make_config(temp.path());
    before.faults.crash_before_manifest_rename = true;
    capture_store crashed(before);
    expect(crashed.try_enqueue(make_observation(1)), "pre-rename crash enqueue");
    expect_status(crashed.flush().status, capture_store_status::commit_uncertain, "pre-rename crash seam");
    expect_status(crashed.shutdown(false).status, capture_store_status::commit_uncertain, "pre-rename shutdown");
    capture_store_config clean = make_config(temp.path());
    capture_store        recovered(clean);
    capture_manifest     empty;
    expect_status(recovered.inspect(empty).status, capture_store_status::no_manifest,
                  "pre-rename recovery old manifest");
    expect_status(recovered.shutdown().status, capture_store_status::ok, "pre-rename recovery shutdown");

    temporary_directory  after_temp;
    capture_store_config after               = make_config(after_temp.path());
    after.faults.crash_after_manifest_rename = true;
    capture_store committed(after);
    expect(committed.try_enqueue(make_observation(3)), "post-rename crash enqueue");
    expect_status(committed.flush().status, capture_store_status::commit_uncertain, "post-rename crash seam");
    expect_status(committed.shutdown(false).status, capture_store_status::commit_uncertain, "post-rename shutdown");
    capture_store    restarted(make_config(after_temp.path()));
    capture_manifest manifest;
    expect_status(restarted.inspect(manifest).status, capture_store_status::ok, "post-rename recovery manifest");
    expect_status(restarted.validate(manifest).status, capture_store_status::ok, "post-rename recovery validate");
    expect_status(restarted.shutdown().status, capture_store_status::ok, "post-rename recovery shutdown");

    for (const bool after_rename : { false, true }) {
        temporary_directory  shard_temp;
        capture_store_config shard_fault             = make_config(shard_temp.path());
        shard_fault.faults.crash_before_shard_rename = !after_rename;
        shard_fault.faults.crash_after_shard_rename  = after_rename;
        capture_store shard_crashed(shard_fault);
        expect(shard_crashed.try_enqueue(make_observation(5)), "shard crash enqueue");
        expect_status(shard_crashed.flush().status, capture_store_status::commit_uncertain, "shard crash seam");
        expect_status(shard_crashed.shutdown(false).status, capture_store_status::commit_uncertain,
                      "shard crash shutdown");
        capture_store    shard_restarted(make_config(shard_temp.path()));
        capture_manifest shard_manifest;
        expect_status(shard_restarted.inspect(shard_manifest).status, capture_store_status::no_manifest,
                      "shard crash recovery old manifest");
        expect_status(shard_restarted.shutdown().status, capture_store_status::ok, "shard crash recovery shutdown");
    }

    const std::array<capture_store_phase, 7> cancellation_phases = {
        capture_store_phase::before_shard_write,     capture_store_phase::after_shard_write,
        capture_store_phase::before_shard_rename,    capture_store_phase::after_shard_rename,
        capture_store_phase::before_manifest_write,  capture_store_phase::after_manifest_write,
        capture_store_phase::before_manifest_rename,
    };
    for (const capture_store_phase cancelled_phase : cancellation_phases) {
        temporary_directory  cancellation_temp;
        capture_store_config cancellation = make_config(cancellation_temp.path());
        cancellation.cancel_check         = [cancelled_phase](capture_store_phase phase) {
            return phase == cancelled_phase;
        };
        capture_store cancelled(cancellation);
        expect(cancelled.try_enqueue(make_observation(4)), "cancel enqueue");
        const capture_store_status phase_status = cancelled.flush().status;
        expect_status(phase_status, capture_store_status::cancelled, "phase cancellation");
        expect_status(cancelled.shutdown(false).status, capture_store_status::cancelled, "phase cancellation shutdown");
    }
}

void test_byte_retention() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    // one compact-record shard is 72 + 32 + 16 + 192 = 312 bytes
    config.max_shard_records    = 1;
    config.max_shard_bytes      = 320;
    config.max_retained_bytes   = 320;
    capture_store store(config);
    expect(store.try_enqueue(make_observation(1)), "byte retention first enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "byte retention first flush");
    expect(store.try_enqueue(make_observation(2)), "byte retention second enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "byte retention second flush");
    capture_manifest manifest;
    expect_status(store.inspect(manifest).status, capture_store_status::ok, "byte retention inspect");
    expect(manifest.shards.size() == 1 && manifest.total_bytes <= config.max_retained_bytes, "byte retention bound");
    expect_status(store.shutdown().status, capture_store_status::ok, "byte retention shutdown");
}

void test_full_ring_progress_and_slow_worker() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.ring_capacity        = 1;
    config.max_shard_records    = 64;
    config.max_shard_bytes      = 16384;
    config.max_retained_records = 128;
    config.max_retained_bytes   = 32768;
    config.faults.slow_worker   = true;
    capture_store store(config);
    uint64_t      accepted = 0;
    for (uint32_t sequence = 0; sequence < 10000; ++sequence) {
        if (store.try_enqueue(make_observation(sequence))) {
            ++accepted;
        }
    }
    const capture_store_stats before = store.stats();
    expect(before.ring.dropped != 0 || accepted != 10000, "full ring did not exercise drop-on-full");
    const capture_store_status stopped = store.shutdown(false).status;
    expect(stopped == capture_store_status::ok || stopped == capture_store_status::stopped,
           "slow worker non-draining shutdown");
}

void test_wakeup_barrier_and_worker_exception() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.max_shard_records    = 1;
    std::mutex              barrier_mutex;
    std::condition_variable barrier_cv;
    bool                    entered  = false;
    bool                    release  = false;
    bool                    first    = true;
    config.faults.before_worker_wait = [&]() {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        if (!first) {
            return;
        }
        first   = false;
        entered = true;
        barrier_cv.notify_all();
        barrier_cv.wait(lock, [&]() { return release; });
    };
    capture_store store(config);
    {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        expect(barrier_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return entered; }),
               "worker did not reach wake barrier");
    }
    expect(store.try_enqueue(make_observation(1)), "barrier enqueue");
    {
        std::lock_guard<std::mutex> lock(barrier_mutex);
        release = true;
    }
    barrier_cv.notify_all();
    bool committed = false;
    for (unsigned attempt = 0; attempt < 200; ++attempt) {
        if (store.stats().committed_records == 1) {
            committed = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    expect(committed, "semaphore wake was lost across worker wait barrier");
    expect_status(store.shutdown().status, capture_store_status::ok, "barrier shutdown");

    temporary_directory  throwing_temp;
    capture_store_config throwing      = make_config(throwing_temp.path());
    throwing.faults.before_worker_wait = throw_worker_barrier_failure;
    capture_store failed(throwing);
    for (unsigned attempt = 0; attempt < 200 && !failed.stats().worker_failed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    expect(failed.stats().worker_failed, "worker callback exception did not fail store");
    expect_status(failed.shutdown(false).status, capture_store_status::io_error, "worker callback failure status");
}

void test_bounds_off_mode_and_private_permissions() {
    temporary_directory        temp;
    const capture_store_config base    = make_config(temp.path());
    capture_store_config       invalid = base;
    invalid.ring_capacity              = CAPTURE_MAX_RING_CAPACITY + 1;
    expect_invalid_config(invalid, "ring hard bound accepted");
    invalid                 = base;
    invalid.max_shard_bytes = CAPTURE_MAX_SHARD_BYTES + 1;
    expect_invalid_config(invalid, "shard hard bound accepted");
    invalid                    = base;
    invalid.max_manifest_bytes = CAPTURE_MAX_MANIFEST_BYTES + 1;
    expect_invalid_config(invalid, "manifest hard bound accepted");
    invalid                     = base;
    invalid.max_retained_shards = UINT32_MAX;
    expect_invalid_config(invalid, "retained shard hard bound accepted");
    invalid                      = base;
    invalid.max_retained_records = UINT64_MAX;
    expect_invalid_config(invalid, "retained record hard bound accepted");
    invalid              = base;
    invalid.capture_mode = static_cast<mode>(99);
    expect_invalid_config(invalid, "invalid capture mode accepted");
    capture_store_config traversal = base;
    traversal.root_path            = (temp.path() / ".." / "escape").string();
    capture_store traversal_store(traversal);
    expect(traversal_store.stats().worker_failed, "traversal root accepted");
    expect_status(traversal_store.shutdown(false).status, capture_store_status::path_security, "traversal root status");

    temporary_directory  off_temp;
    const fs::path       off_root   = off_temp.path() / "not-created";
    capture_store_config off_config = make_config(off_root);
    off_config.capture_mode         = mode::off;
    capture_store off(off_config);
    expect(!fs::exists(off_root), "off mode created capture root");
    expect(!off.try_enqueue(make_observation(1)), "off mode accepted observation");
    expect_status(off.shutdown().status, capture_store_status::disabled, "off mode shutdown");

    temporary_directory  csprng_temp;
    capture_store_config csprng_config = make_config(csprng_temp.path() / "not-created");
    csprng_config.identity_salt        = {};
    csprng_config.faults.fail_csprng   = true;
    capture_store csprng_failed(csprng_config);
    expect(csprng_failed.stats().worker_failed, "CSPRNG failure did not fail closed");
    expect_status(csprng_failed.shutdown(false).status, capture_store_status::io_error, "CSPRNG failure status");
    expect(!fs::exists(csprng_temp.path() / "not-created"), "CSPRNG failure created a capture root");

    temporary_directory  zero_csprng_temp;
    capture_store_config zero_csprng_config       = make_config(zero_csprng_temp.path() / "not-created");
    zero_csprng_config.identity_salt              = {};
    zero_csprng_config.faults.csprng_returns_zero = true;
    capture_store zero_csprng_failed(zero_csprng_config);
    expect(zero_csprng_failed.stats().worker_failed, "all-zero CSPRNG did not fail closed");
    expect_status(zero_csprng_failed.shutdown(false).status, capture_store_status::io_error,
                  "all-zero CSPRNG failure status");
    expect(!fs::exists(zero_csprng_temp.path() / "not-created"), "all-zero CSPRNG created a capture root");

    temporary_directory persisted_zero_temp;
    {
        capture_store_config config            = make_config(persisted_zero_temp.path());
        config.faults.force_zero_manifest_salt = true;
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), "zero-salt manifest enqueue");
        expect_status(store.flush().status, capture_store_status::ok, "zero-salt manifest flush");
        expect_status(store.shutdown().status, capture_store_status::ok, "zero-salt manifest shutdown");
    }
    capture_store persisted_zero_recovered(make_config(persisted_zero_temp.path()));
    expect(persisted_zero_recovered.stats().worker_failed, "persisted all-zero identity salt was accepted");
    expect_status(persisted_zero_recovered.shutdown(false).status, capture_store_status::malformed_manifest,
                  "persisted all-zero identity salt status");

    capture_store store(base);
    expect(store.try_enqueue(make_observation(1)), "private mode enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "private mode flush");
    struct stat root_status = {};
    expect(::stat(temp.path().c_str(), &root_status) == 0, "private root stat");
    expect((root_status.st_mode & (S_IRWXG | S_IRWXO)) == 0, "capture root is not private");
    struct stat    manifest_status = {};
    const fs::path manifest_path   = temp.path() / "capture.manifest";
    expect(::stat(manifest_path.c_str(), &manifest_status) == 0, "private manifest stat");
    expect((manifest_status.st_mode & (S_IRWXG | S_IRWXO)) == 0, "manifest is not private");
    const fs::path shard        = only_shard(temp.path());
    struct stat    shard_status = {};
    expect(::stat(shard.c_str(), &shard_status) == 0, "private shard stat");
    expect((shard_status.st_mode & (S_IRWXG | S_IRWXO)) == 0, "shard is not private");
    expect_status(store.shutdown().status, capture_store_status::ok, "private mode shutdown");
}

void test_root_file_and_tombstone_security() {
    temporary_directory symlink_temp;
    const fs::path      target = symlink_temp.path() / "target";
    fs::create_directory(target);
    const fs::path intermediate_link_target = symlink_temp.path() / "intermediate-target";
    expect(fs::create_directory(intermediate_link_target), "create intermediate symlink target");
    const fs::path intermediate_link = symlink_temp.path() / "intermediate-link";
    expect(::symlink(intermediate_link_target.c_str(), intermediate_link.c_str()) == 0, "create intermediate symlink");
    capture_store_config intermediate = make_config(intermediate_link / "capture");
    capture_store        intermediate_store(intermediate);
    expect(intermediate_store.stats().worker_failed, "intermediate symlink was accepted");
    expect_status(intermediate_store.shutdown(false).status, capture_store_status::path_security,
                  "intermediate symlink status");

    const fs::path root_link = symlink_temp.path() / "root-link";
    expect(::symlink(target.c_str(), root_link.c_str()) == 0, "create root symlink");
    capture_store_config linked = make_config(root_link);
    capture_store        linked_store(linked);
    expect(linked_store.stats().worker_failed, "root symlink was accepted");
    expect_status(linked_store.shutdown(false).status, capture_store_status::path_security, "root symlink status");

    temporary_directory  manifest_temp;
    capture_store_config normal_config = make_config(manifest_temp.path());
    {
        capture_store normal(normal_config);
        expect(normal.try_enqueue(make_observation(1)), "manifest symlink fixture enqueue");
        expect_status(normal.flush().status, capture_store_status::ok, "manifest symlink fixture flush");
        expect_status(normal.shutdown().status, capture_store_status::ok, "manifest symlink fixture shutdown");
    }
    const fs::path external = manifest_temp.path() / "external";
    write_private_file(external, { 1, 2, 3 });
    const fs::path manifest        = manifest_temp.path() / "capture.manifest";
    const fs::path manifest_backup = manifest_temp.path() / "manifest.backup";
    expect(::rename(manifest.c_str(), manifest_backup.c_str()) == 0, "move manifest fixture");
    expect(::symlink(external.c_str(), manifest.c_str()) == 0, "create manifest symlink");
    capture_store manifest_store(normal_config);
    expect(manifest_store.stats().worker_failed, "manifest symlink was accepted");
    expect_status(manifest_store.shutdown(false).status, capture_store_status::path_security,
                  "manifest symlink status");

    temporary_directory  shard_temp;
    capture_store_config shard_config = make_config(shard_temp.path());
    {
        capture_store normal(shard_config);
        expect(normal.try_enqueue(make_observation(1)), "shard symlink fixture enqueue");
        expect_status(normal.flush().status, capture_store_status::ok, "shard symlink fixture flush");
        expect_status(normal.shutdown().status, capture_store_status::ok, "shard symlink fixture shutdown");
    }
    const fs::path shard_path   = only_shard(shard_temp.path());
    const fs::path shard_backup = shard_temp.path() / "shard.backup";
    expect(::rename(shard_path.c_str(), shard_backup.c_str()) == 0, "move shard fixture");
    expect(::symlink(external.c_str(), shard_path.c_str()) == 0, "create shard symlink");
    capture_store shard_store(shard_config);
    expect(shard_store.stats().worker_failed, "shard symlink was accepted");
    expect_status(shard_store.shutdown(false).status, capture_store_status::path_security, "shard symlink status");

    temporary_directory  temporary_temp;
    capture_store_config temporary_config = make_config(temporary_temp.path());
    const fs::path       temporary_link   = temporary_temp.path() / ".shard-00000000000000000001.tmp";
    expect(::symlink(external.c_str(), temporary_link.c_str()) == 0, "create temporary symlink");
    capture_store temporary_store(temporary_config);
    expect(temporary_store.stats().worker_failed, "temporary symlink was accepted");
    expect_status(temporary_store.shutdown(false).status, capture_store_status::path_security,
                  "temporary symlink status");

    temporary_directory  tombstone_temp;
    capture_store_config tombstone_config = make_config(tombstone_temp.path());
    write_private_file(tombstone_temp.path() / ".delete-18446744073709551616.tomb", { 1 });
    capture_store tombstone_store(tombstone_config);
    expect(tombstone_store.stats().worker_failed, "overflow tombstone was accepted");
    expect_status(tombstone_store.shutdown(false).status, capture_store_status::path_security,
                  "overflow tombstone status");

    temporary_directory  referenced_tombstone_temp;
    capture_store_config referenced_config = make_config(referenced_tombstone_temp.path());
    {
        capture_store normal(referenced_config);
        expect(normal.try_enqueue(make_observation(1)), "referenced tombstone fixture enqueue");
        expect_status(normal.flush().status, capture_store_status::ok, "referenced tombstone fixture flush");
        expect_status(normal.shutdown().status, capture_store_status::ok, "referenced tombstone fixture shutdown");
    }
    write_private_file(referenced_tombstone_temp.path() / ".delete-00000000000000000001.tomb", { 1 });
    capture_store referenced_tombstone(referenced_config);
    expect(referenced_tombstone.stats().worker_failed, "referenced tombstone was accepted");
    expect_status(referenced_tombstone.shutdown(false).status, capture_store_status::path_security,
                  "referenced tombstone status");
}

void test_random_salt_and_post_publication_retention() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.max_shard_records    = 1;
    config.max_retained_shards  = 1;
    config.max_retained_records = 1;
    config.max_retained_bytes   = 4096;
    config.identity_salt        = {};
    capture_store store(config);
    expect(store.try_enqueue(make_observation(1)), "random salt first enqueue");
    expect_status(store.flush().status, capture_store_status::ok, "random salt first flush");
    capture_manifest first;
    expect_status(store.inspect(first).status, capture_store_status::ok, "random salt inspect");
    expect(
        std::any_of(first.identity_salt.begin(), first.identity_salt.end(), [](uint8_t value) { return value != 0; }),
        "random salt remained zero");
    const auto persisted_salt = first.identity_salt;
    expect_status(store.shutdown().status, capture_store_status::ok, "random salt shutdown");
    capture_store    restarted(config);
    capture_manifest recovered;
    expect_status(restarted.inspect(recovered).status, capture_store_status::ok, "random salt restart inspect");
    expect(recovered.identity_salt == persisted_salt, "random salt did not persist");
    expect_status(restarted.shutdown().status, capture_store_status::ok, "random salt restart shutdown");

    for (const capture_store_phase cancellation_phase :
         { capture_store_phase::before_retention_delete, capture_store_phase::after_retention_delete }) {
        temporary_directory  retention_temp;
        capture_store_config retention = make_config(retention_temp.path());
        retention.max_shard_records    = 1;
        retention.max_retained_shards  = 1;
        retention.max_retained_records = 1;
        retention.max_retained_bytes   = 4096;
        retention.cancel_check         = [cancellation_phase](capture_store_phase phase) {
            return phase == cancellation_phase;
        };
        capture_store retention_store(retention);
        expect(retention_store.try_enqueue(make_observation(1)), "post-publication cancellation first enqueue");
        expect_status(retention_store.flush().status, capture_store_status::ok,
                      "post-publication cancellation first flush");
        expect(retention_store.try_enqueue(make_observation(2)), "post-publication cancellation second enqueue");
        expect_status(retention_store.flush().status, capture_store_status::ok,
                      "post-publication cancellation second flush");
        capture_manifest retained;
        expect_status(retention_store.inspect(retained).status, capture_store_status::ok,
                      "post-publication cancellation inspect");
        expect(retained.shards.size() == 1 && retained.shards.front().sequence == 2,
               "post-publication cancellation leaked retired shard");
        expect_status(retention_store.shutdown().status, capture_store_status::ok,
                      "post-publication cancellation shutdown");
        capture_store    retention_restart(retention);
        capture_manifest restarted_manifest;
        expect_status(retention_restart.inspect(restarted_manifest).status, capture_store_status::ok,
                      "post-publication cancellation restart inspect");
        expect_status(retention_restart.validate(restarted_manifest).status, capture_store_status::ok,
                      "post-publication cancellation restart validate");
        expect_status(retention_restart.shutdown().status, capture_store_status::ok,
                      "post-publication cancellation restart shutdown");
    }
}

void test_concurrent_flush_shutdown() {
    temporary_directory  temp;
    capture_store_config config = make_config(temp.path());
    config.max_shard_records    = 1;
    config.faults.slow_worker   = true;
    capture_store store(config);
    expect(store.try_enqueue(make_observation(1)), "concurrent lifecycle enqueue");
    std::atomic<bool>        start{ false };
    std::vector<std::thread> flushers;
    for (unsigned index = 0; index < 4; ++index) {
        flushers.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (unsigned attempt = 0; attempt < 20; ++attempt) {
                const capture_store_status status = store.flush().status;
                expect(status == capture_store_status::ok || status == capture_store_status::stopped,
                       "concurrent flush returned unexpected status");
            }
        });
    }
    start.store(true, std::memory_order_release);
    const capture_store_status shutdown_status = store.shutdown().status;
    for (std::thread & flusher : flushers) {
        flusher.join();
    }
    expect(shutdown_status == capture_store_status::ok || shutdown_status == capture_store_status::stopped,
           "concurrent shutdown returned unexpected status");
}

void test_admission_linearization_and_high_count_shutdown() {
    temporary_directory     barrier_temp;
    capture_store_config    barrier_config = make_config(barrier_temp.path());
    std::mutex              barrier_mutex;
    std::condition_variable barrier_cv;
    bool                    entered             = false;
    bool                    release             = false;
    barrier_config.faults.before_enqueue_accept = [&]() {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        entered = true;
        barrier_cv.notify_all();
        barrier_cv.wait(lock, [&]() { return release; });
    };
    auto              store = std::make_unique<capture_store>(barrier_config);
    capture_store *   raw   = store.get();
    std::atomic<bool> accepted{ true };
    std::thread producer([&]() { accepted.store(raw->try_enqueue(make_observation(1)), std::memory_order_release); });
    {
        std::unique_lock<std::mutex> lock(barrier_mutex);
        expect(barrier_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return entered; }),
               "producer did not claim admission before shutdown");
    }
    std::atomic<bool>        shutdown_done{ false };
    capture_store_result     shutdown_result;
    std::thread              shutdowner([&]() {
        shutdown_result = raw->shutdown();
        shutdown_done.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(!shutdown_done.load(std::memory_order_acquire), "shutdown ignored live producer claim");
    {
        std::lock_guard<std::mutex> lock(barrier_mutex);
        release = true;
    }
    barrier_cv.notify_all();
    producer.join();
    shutdowner.join();
    expect(accepted.load(std::memory_order_acquire), "shutdown stranded a pre-close producer observation");
    expect(shutdown_result.status == capture_store_status::ok || shutdown_result.status == capture_store_status::stopped,
           "concurrent admission shutdown returned unexpected status");
    store.reset();

    for (const bool drain : { true, false }) {
        temporary_directory     barrier_shutdown_temp;
        capture_store_config    barrier_shutdown_config = make_config(barrier_shutdown_temp.path());
        std::mutex              push_mutex;
        std::condition_variable push_cv;
        bool                    push_entered               = false;
        bool                    push_release               = false;
        barrier_shutdown_config.faults.before_enqueue_push = [&]() {
            std::unique_lock<std::mutex> lock(push_mutex);
            push_entered = true;
            push_cv.notify_all();
            push_cv.wait(lock, [&]() { return push_release; });
        };
        capture_store     barrier_shutdown_store(barrier_shutdown_config);
        std::atomic<bool> push_result{ false };
        std::thread       barrier_producer([&]() {
            push_result.store(barrier_shutdown_store.try_enqueue(make_observation(1)), std::memory_order_release);
        });
        {
            std::unique_lock<std::mutex> lock(push_mutex);
            expect(push_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return push_entered; }),
                   "producer did not reach pre-push barrier");
        }
        std::atomic<bool>    shutdown_done{ false };
        capture_store_result barrier_shutdown_result;
        std::thread          barrier_shutdown([&]() {
            barrier_shutdown_result = barrier_shutdown_store.shutdown(drain);
            shutdown_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        expect(!shutdown_done.load(std::memory_order_acquire), "shutdown did not wait for pre-push producer");
        {
            std::lock_guard<std::mutex> lock(push_mutex);
            push_release = true;
        }
        push_cv.notify_all();
        barrier_producer.join();
        barrier_shutdown.join();
        expect(push_result.load(std::memory_order_acquire), "pre-push producer lost an accepted observation");
        expect(barrier_shutdown_result.status == capture_store_status::ok ||
                   (!drain && barrier_shutdown_result.status == capture_store_status::stopped),
               "pre-push barrier shutdown status");
        const capture_store_stats barrier_stats = barrier_shutdown_store.stats();
        expect(barrier_stats.ring.size_approx == 0, "shutdown left a pre-push observation in the ring");
        capture_manifest           barrier_manifest;
        const capture_store_result barrier_inspect = barrier_shutdown_store.inspect(barrier_manifest);
        if (drain) {
            expect_status(barrier_inspect.status, capture_store_status::ok, "draining pre-push inspect");
            expect(barrier_manifest.total_records == 1, "draining pre-push shutdown lost accepted observation");
            expect(barrier_stats.dropped_on_shutdown == 0, "draining pre-push shutdown dropped observation");
        } else {
            expect_status(barrier_inspect.status, capture_store_status::no_manifest, "non-draining pre-push inspect");
            expect(barrier_stats.dropped_on_shutdown == 1,
                   "non-draining pre-push shutdown did not account accepted observation");
        }
    }

    temporary_directory     failure_temp;
    capture_store_config    failure_config = make_config(failure_temp.path());
    std::mutex              failure_mutex;
    std::condition_variable failure_cv;
    bool                    failure_producer_entered = false;
    bool                    failure_producer_release = false;
    bool                    failure_triggered        = false;
    failure_config.faults.before_enqueue_push        = [&]() {
        std::unique_lock<std::mutex> lock(failure_mutex);
        failure_producer_entered = true;
        failure_cv.notify_all();
        failure_cv.wait(lock, [&]() { return failure_producer_release; });
    };
    failure_config.faults.before_worker_wait = [&]() {
        std::unique_lock<std::mutex> lock(failure_mutex);
        failure_cv.wait(lock, [&]() { return failure_producer_entered; });
        failure_triggered = true;
        failure_cv.notify_all();
        throw std::runtime_error("deterministic mark_failure seam");
    };
    capture_store     failure_store(failure_config);
    std::atomic<bool> failure_push_result{ false };
    std::thread       failure_producer([&]() {
        failure_push_result.store(failure_store.try_enqueue(make_observation(1)), std::memory_order_release);
    });
    {
        std::unique_lock<std::mutex> lock(failure_mutex);
        expect(failure_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return failure_producer_entered; }),
               "mark_failure producer did not reach pre-push barrier");
        expect(failure_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return failure_triggered; }),
               "mark_failure worker did not trigger");
        failure_producer_release = true;
    }
    failure_cv.notify_all();
    failure_producer.join();
    expect(failure_push_result.load(std::memory_order_acquire),
           "mark_failure rejected a producer that passed accepting before failure");
    expect_status(failure_store.shutdown(false).status, capture_store_status::io_error, "mark_failure shutdown status");
    const capture_store_stats failure_stats = failure_store.stats();
    expect(failure_stats.worker_failed, "mark_failure did not set worker failure state");
    expect(failure_stats.ring.size_approx == 0, "mark_failure left an accepted observation in the ring");
    expect(failure_stats.dropped_on_shutdown == 1, "mark_failure did not account an accepted ring observation");

    for (const bool drain : { true, false }) {
        temporary_directory  temp;
        capture_store_config config = make_config(temp.path());
        config.ring_capacity        = 128;
        config.max_shard_records    = 64;
        config.max_shard_bytes      = 16384;
        config.max_retained_shards  = 10000;
        config.max_retained_records = 100000;
        config.max_retained_bytes   = 64U * 1024U * 1024U;
        config.max_manifest_bytes   = CAPTURE_MAX_MANIFEST_BYTES;
        config.faults.slow_worker   = true;
        capture_store         store_counted(config);
        std::atomic<uint64_t> accepted_count{ 0 };
        std::atomic<uint64_t> rejected_count{ 0 };
        std::atomic<bool>     start{ false };
        std::thread           producer_counted([&]() {
            start.store(true, std::memory_order_release);
            for (uint32_t sequence = 0; sequence < 100000; ++sequence) {
                if (store_counted.try_enqueue(make_observation(sequence))) {
                    accepted_count.fetch_add(1, std::memory_order_relaxed);
                } else {
                    rejected_count.fetch_add(1, std::memory_order_relaxed);
                }
                if ((sequence & 63U) == 0) {
                    std::this_thread::yield();
                }
            }
        });
        for (unsigned attempt = 0; attempt < 200 && !start.load(std::memory_order_acquire); ++attempt) {
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const capture_store_status shutdown_status = store_counted.shutdown(drain).status;
        producer_counted.join();
        expect(shutdown_status == capture_store_status::ok || shutdown_status == capture_store_status::stopped,
               "counted shutdown failed");
        const capture_store_stats stats = store_counted.stats();
        capture_manifest          manifest;
        expect_status(store_counted.inspect(manifest).status, capture_store_status::ok, "counted inspect");
        const uint64_t accepted_total = accepted_count.load(std::memory_order_relaxed);
        const uint64_t rejected_total = rejected_count.load(std::memory_order_relaxed);
        expect(stats.ring.pushed == accepted_total, "producer true result disagrees with ring pushes");
        expect(stats.ring.dropped + stats.dropped_after_stop == rejected_total,
               "producer false result disagrees with ring/closed drops");
        expect(stats.ring.size_approx == 0, "shutdown left an accepted observation in the ring");
        if (drain) {
            expect(manifest.total_records == accepted_total, "draining shutdown lost an accepted producer record");
            expect(stats.dropped_on_shutdown == 0, "draining shutdown discarded a queued record");
        } else {
            expect(manifest.total_records + stats.dropped_on_shutdown == accepted_total,
                   "non-draining shutdown accounting disagrees with accepted records");
        }
    }
}

void test_filesystem_fault_seams_and_flush_ordering() {
    struct fault_case {
        const char * name;
        void (*configure)(capture_store_config &);
    };

    const std::array<fault_case, 5> cases = {
        {
         { "file-fsync", [](capture_store_config & config) { config.faults.fail_file_fsync           = true; } },
         { "directory-fsync", [](capture_store_config & config) { config.faults.fail_directory_fsync = true; } },
         { "shard-rename", [](capture_store_config & config) { config.faults.fail_shard_rename = true; } },
         { "manifest-rename", [](capture_store_config & config) { config.faults.fail_manifest_rename = true; } },
         { "fstat", [](capture_store_config & config) { config.faults.fail_fstat = true; } },
         }
    };
    for (const fault_case & current : cases) {
        temporary_directory  temp;
        capture_store_config config = make_config(temp.path());
        current.configure(config);
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), std::string(current.name) + " enqueue");
        expect_status(store.flush().status, capture_store_status::io_error, std::string(current.name) + " flush");
        expect_status(store.shutdown(false).status, capture_store_status::io_error,
                      std::string(current.name) + " shutdown");
    }

    // A directory-fsync failure occurs after the shard rename but before the
    // manifest is published.  Recovery must treat that shard as uncommitted,
    // remove it with bounded authority, and leave a clean no-manifest state.
    temporary_directory directory_fsync_temp;
    {
        capture_store_config config        = make_config(directory_fsync_temp.path());
        config.faults.fail_directory_fsync = true;
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), "directory-fsync orphan enqueue");
        expect_status(store.flush().status, capture_store_status::io_error, "directory-fsync orphan flush");
        expect_status(store.shutdown(false).status, capture_store_status::io_error, "directory-fsync orphan shutdown");
    }
    capture_store    recovered_directory_fsync(make_config(directory_fsync_temp.path()));
    capture_manifest recovered_directory_manifest;
    expect_status(recovered_directory_fsync.inspect(recovered_directory_manifest).status,
                  capture_store_status::no_manifest, "directory-fsync orphan recovery manifest");
    for (const fs::directory_entry & entry : fs::directory_iterator(directory_fsync_temp.path())) {
        expect(entry.path().extension() != ".cap", "directory-fsync orphan shard survived recovery");
    }
    expect_status(recovered_directory_fsync.shutdown().status, capture_store_status::ok,
                  "directory-fsync orphan recovery shutdown");

    temporary_directory  tombstone_temp;
    capture_store_config tombstone_config        = make_config(tombstone_temp.path());
    tombstone_config.max_shard_records           = 1;
    tombstone_config.max_retained_shards         = 1;
    tombstone_config.max_retained_records        = 1;
    tombstone_config.max_retained_bytes          = 4096;
    tombstone_config.faults.fail_tombstone_fsync = true;
    capture_store tombstone_store(tombstone_config);
    expect(tombstone_store.try_enqueue(make_observation(1)), "tombstone fsync first enqueue");
    expect_status(tombstone_store.flush().status, capture_store_status::ok, "tombstone fsync first flush");
    expect(tombstone_store.try_enqueue(make_observation(2)), "tombstone fsync second enqueue");
    expect_status(tombstone_store.flush().status, capture_store_status::deletion_failed, "tombstone fsync flush");
    expect_status(tombstone_store.shutdown(false).status, capture_store_status::deletion_failed,
                  "tombstone fsync shutdown");
    capture_store_config recovered_config = make_config(tombstone_temp.path());
    recovered_config.max_shard_records    = 1;
    recovered_config.max_retained_shards  = 1;
    recovered_config.max_retained_records = 1;
    recovered_config.max_retained_bytes   = 4096;
    capture_store    recovered(recovered_config);
    capture_manifest recovered_manifest;
    expect_status(recovered.inspect(recovered_manifest).status, capture_store_status::ok,
                  "tombstone fsync recovery inspect");
    expect(recovered_manifest.shards.size() == 1 && recovered_manifest.shards.front().sequence == 2,
           "tombstone fsync recovery did not retain published manifest");
    expect_status(recovered.shutdown().status, capture_store_status::ok, "tombstone fsync recovery shutdown");

    auto configure_one_retained = [](capture_store_config & config) {
        config.max_shard_records    = 2;
        config.max_retained_shards  = 1;
        config.max_retained_records = 2;
        config.max_retained_bytes   = 4096;
    };

    temporary_directory post_fsync_temp;
    {
        capture_store_config config = make_config(post_fsync_temp.path());
        configure_one_retained(config);
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), "post-publication fsync first enqueue");
        expect_status(store.flush().status, capture_store_status::ok, "post-publication fsync first flush");
        expect_status(store.shutdown().status, capture_store_status::ok, "post-publication fsync first shutdown");
    }
    {
        capture_store_config config = make_config(post_fsync_temp.path());
        configure_one_retained(config);
        config.faults.fail_post_publication_directory_fsync = true;
        capture_store store(config);
        expect(store.try_enqueue(make_observation(2)), "post-publication fsync second enqueue");
        expect_status(store.flush().status, capture_store_status::commit_uncertain,
                      "post-publication fsync second flush");
        expect_status(store.shutdown(false).status, capture_store_status::commit_uncertain,
                      "post-publication fsync second shutdown");
    }
    capture_store    post_fsync_recovered(make_config(post_fsync_temp.path()));
    capture_manifest post_fsync_manifest;
    expect_status(post_fsync_recovered.inspect(post_fsync_manifest).status, capture_store_status::ok,
                  "post-publication fsync recovery inspect");
    expect(post_fsync_manifest.shards.size() == 1 && post_fsync_manifest.shards.front().sequence == 2,
           "post-publication fsync recovery lost durable manifest");
    expect_status(post_fsync_recovered.validate(post_fsync_manifest).status, capture_store_status::ok,
                  "post-publication fsync recovery validate");
    expect_status(post_fsync_recovered.shutdown().status, capture_store_status::ok,
                  "post-publication fsync recovery shutdown");
    size_t post_fsync_shards = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(post_fsync_temp.path())) {
        post_fsync_shards += entry.path().extension() == ".cap" ? 1 : 0;
    }
    expect(post_fsync_shards == 1, "post-publication fsync recovery left an unreferenced shard");

    temporary_directory post_rename_temp;
    {
        capture_store_config config = make_config(post_rename_temp.path());
        configure_one_retained(config);
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), "post-publication rename first enqueue");
        expect_status(store.flush().status, capture_store_status::ok, "post-publication rename first flush");
        expect_status(store.shutdown().status, capture_store_status::ok, "post-publication rename first shutdown");
    }
    {
        capture_store_config config = make_config(post_rename_temp.path());
        configure_one_retained(config);
        config.faults.fail_manifest_rename = true;
        capture_store store(config);
        expect(store.try_enqueue(make_observation(2)), "post-publication rename second enqueue");
        expect_status(store.flush().status, capture_store_status::io_error, "post-publication rename second flush");
        expect_status(store.shutdown(false).status, capture_store_status::io_error,
                      "post-publication rename second shutdown");
    }
    capture_store    post_rename_recovered(make_config(post_rename_temp.path()));
    capture_manifest post_rename_manifest;
    expect_status(post_rename_recovered.inspect(post_rename_manifest).status, capture_store_status::ok,
                  "post-publication rename recovery inspect");
    expect(post_rename_manifest.shards.size() == 1 && post_rename_manifest.shards.front().sequence == 1,
           "post-publication rename recovery replaced the prior manifest");
    expect_status(post_rename_recovered.shutdown().status, capture_store_status::ok,
                  "post-publication rename recovery shutdown");
    size_t post_rename_shards = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(post_rename_temp.path())) {
        post_rename_shards += entry.path().extension() == ".cap" ? 1 : 0;
    }
    expect(post_rename_shards == 1, "post-publication rename recovery left an unreferenced shard");

    temporary_directory post_unlink_temp;
    {
        capture_store_config config = make_config(post_unlink_temp.path());
        configure_one_retained(config);
        capture_store store(config);
        expect(store.try_enqueue(make_observation(1)), "post-publication unlink first enqueue");
        expect_status(store.flush().status, capture_store_status::ok, "post-publication unlink first flush");
        expect_status(store.shutdown().status, capture_store_status::ok, "post-publication unlink first shutdown");
    }
    {
        capture_store_config config = make_config(post_unlink_temp.path());
        configure_one_retained(config);
        config.faults.fail_unlink = true;
        capture_store store(config);
        expect(store.try_enqueue(make_observation(2)), "post-publication unlink second enqueue");
        expect_status(store.flush().status, capture_store_status::deletion_failed,
                      "post-publication unlink second flush");
        expect_status(store.shutdown(false).status, capture_store_status::deletion_failed,
                      "post-publication unlink second shutdown");
    }
    capture_store    post_unlink_recovered(make_config(post_unlink_temp.path()));
    capture_manifest post_unlink_manifest;
    expect_status(post_unlink_recovered.inspect(post_unlink_manifest).status, capture_store_status::ok,
                  "post-publication unlink recovery inspect");
    expect(post_unlink_manifest.shards.size() == 1 && post_unlink_manifest.shards.front().sequence == 2,
           "post-publication unlink recovery lost durable manifest");
    expect_status(post_unlink_recovered.shutdown().status, capture_store_status::ok,
                  "post-publication unlink recovery shutdown");
    size_t post_unlink_shards = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(post_unlink_temp.path())) {
        post_unlink_shards += entry.path().extension() == ".cap" ? 1 : 0;
    }
    expect(post_unlink_shards == 1, "post-publication unlink recovery left an unreferenced shard");

    temporary_directory  orphan_temp;
    capture_store_config orphan_config = make_config(orphan_temp.path());
    write_private_file(orphan_temp.path() / ".shard-00000000000000000001.tmp", { 1, 2, 3 });
    orphan_config.faults.fail_unlink = true;
    capture_store orphan(orphan_config);
    expect(orphan.stats().worker_failed, "orphan unlink fault did not fail recovery");
    expect_status(orphan.shutdown(false).status, capture_store_status::deletion_failed, "orphan unlink fault status");

    temporary_directory  ordering_temp;
    capture_store_config ordering_config   = make_config(ordering_temp.path());
    ordering_config.faults.fail_file_fsync = true;
    capture_store ordering(ordering_config);
    expect(ordering.try_enqueue(make_observation(1)), "ordering enqueue");
    std::array<capture_store_result, 2> flush_results = {};
    std::thread                         first([&]() { flush_results[0] = ordering.flush(); });
    std::thread                         second([&]() { flush_results[1] = ordering.flush(); });
    first.join();
    second.join();
    expect_status(flush_results[0].status, capture_store_status::io_error, "first failure flush ordering");
    expect_status(flush_results[1].status, capture_store_status::io_error, "second failure flush ordering");
    expect_status(ordering.shutdown(false).status, capture_store_status::io_error, "failure ordering shutdown");
}

void test_post_publication_callbacks_and_pending_flush() {
    temporary_directory  after_manifest_temp;
    capture_store_config after_manifest_config = make_config(after_manifest_temp.path());
    after_manifest_config.cancel_check         = [](capture_store_phase phase) {
        if (phase == capture_store_phase::after_manifest_rename) {
            throw std::runtime_error("after-manifest publication callback");
        }
        return false;
    };
    {
        capture_store store(after_manifest_config);
        expect(store.try_enqueue(make_observation(1)), "after-manifest callback enqueue");
        const capture_store_result flushed = store.flush();
        expect(flushed.status == capture_store_status::commit_uncertain && flushed.committed,
               "after-manifest callback lost committed uncertainty");
        const capture_store_result stopped = store.shutdown(false);
        expect(stopped.status == capture_store_status::commit_uncertain && stopped.committed,
               "after-manifest callback shutdown status");
        capture_manifest manifest;
        expect_status(store.inspect(manifest).status, capture_store_status::ok, "after-manifest callback inspect");
        expect_status(store.validate(manifest).status, capture_store_status::ok, "after-manifest callback validate");
        expect(manifest.shards.size() == 1 && manifest.shards.front().sequence == 1,
               "after-manifest callback lost published shard");
        expect(store.stats().dropped_on_shutdown == 0, "after-manifest callback double-counted durable records");
    }
    capture_store    after_manifest_recovered(make_config(after_manifest_temp.path()));
    capture_manifest after_manifest_manifest;
    expect_status(after_manifest_recovered.inspect(after_manifest_manifest).status, capture_store_status::ok,
                  "after-manifest callback restart inspect");
    expect_status(after_manifest_recovered.validate(after_manifest_manifest).status, capture_store_status::ok,
                  "after-manifest callback restart validate");
    expect_status(after_manifest_recovered.shutdown().status, capture_store_status::ok,
                  "after-manifest callback restart shutdown");

    temporary_directory  before_retention_temp;
    capture_store_config before_retention_config = make_config(before_retention_temp.path());
    before_retention_config.max_retained_shards  = 1;
    before_retention_config.max_retained_records = 2;
    before_retention_config.max_retained_bytes   = 4096;
    std::atomic<unsigned> before_retention_calls{ 0 };
    before_retention_config.cancel_check = [&](capture_store_phase phase) {
        if (phase == capture_store_phase::before_retention_delete &&
            before_retention_calls.fetch_add(1, std::memory_order_relaxed) != 0) {
            throw std::runtime_error("before-retention publication callback");
        }
        return false;
    };
    {
        capture_store store(before_retention_config);
        expect(store.try_enqueue(make_observation(1)), "before-retention first enqueue");
        expect_status(store.flush().status, capture_store_status::ok, "before-retention first flush");
        expect(store.try_enqueue(make_observation(2)), "before-retention second enqueue");
        const capture_store_result flushed = store.flush();
        expect(flushed.status == capture_store_status::commit_uncertain && flushed.committed,
               "before-retention callback lost committed uncertainty");
        const capture_store_result stopped = store.shutdown(false);
        expect(stopped.status == capture_store_status::commit_uncertain && stopped.committed,
               "before-retention callback shutdown status");
        capture_manifest manifest;
        expect_status(store.inspect(manifest).status, capture_store_status::ok, "before-retention callback inspect");
        expect_status(store.validate(manifest).status, capture_store_status::ok, "before-retention callback validate");
        expect(manifest.shards.size() == 1 && manifest.shards.front().sequence == 2,
               "before-retention callback lost published shard");
        expect(store.stats().dropped_on_shutdown == 0, "before-retention callback double-counted durable records");
    }
    capture_store    before_retention_recovered(make_config(before_retention_temp.path()));
    capture_manifest before_retention_manifest;
    expect_status(before_retention_recovered.inspect(before_retention_manifest).status, capture_store_status::ok,
                  "before-retention callback restart inspect");
    expect_status(before_retention_recovered.validate(before_retention_manifest).status, capture_store_status::ok,
                  "before-retention callback restart validate");
    expect(before_retention_manifest.shards.size() == 1 && before_retention_manifest.shards.front().sequence == 2,
           "before-retention callback restart manifest");
    expect_status(before_retention_recovered.shutdown().status, capture_store_status::ok,
                  "before-retention callback restart shutdown");

    temporary_directory     pending_flush_temp;
    capture_store_config    pending_flush_config = make_config(pending_flush_temp.path());
    std::mutex              pending_mutex;
    std::condition_variable pending_cv;
    bool                    worker_wait_entered    = false;
    bool                    worker_wait_release    = false;
    bool                    flush_registered       = false;
    bool                    flush_release          = false;
    pending_flush_config.faults.before_worker_wait = [&]() {
        std::unique_lock<std::mutex> lock(pending_mutex);
        worker_wait_entered = true;
        pending_cv.notify_all();
        pending_cv.wait(lock, [&]() { return worker_wait_release; });
    };
    pending_flush_config.faults.after_flush_registration = [&]() {
        std::unique_lock<std::mutex> lock(pending_mutex);
        flush_registered = true;
        pending_cv.notify_all();
        pending_cv.wait(lock, [&]() { return flush_release; });
    };
    capture_store pending_flush_store(pending_flush_config);
    {
        std::unique_lock<std::mutex> lock(pending_mutex);
        expect(pending_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return worker_wait_entered; }),
               "pending-flush worker barrier did not enter");
    }
    expect(pending_flush_store.try_enqueue(make_observation(1)), "pending-flush enqueue");
    capture_store_result pending_flush_result;
    std::thread          flusher([&]() { pending_flush_result = pending_flush_store.flush(); });
    {
        std::unique_lock<std::mutex> lock(pending_mutex);
        expect(pending_cv.wait_for(lock, std::chrono::seconds(2), [&]() { return flush_registered; }),
               "pending flush did not register before shutdown");
    }
    std::atomic<bool>    pending_shutdown_done{ false };
    capture_store_result pending_shutdown_result;
    std::thread          pending_shutdown([&]() {
        pending_shutdown_result = pending_flush_store.shutdown(false);
        pending_shutdown_done.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(!pending_shutdown_done.load(std::memory_order_acquire), "pending shutdown skipped worker barrier");
    {
        std::lock_guard<std::mutex> lock(pending_mutex);
        flush_release       = true;
        worker_wait_release = true;
    }
    pending_cv.notify_all();
    flusher.join();
    pending_shutdown.join();
    expect(pending_flush_result.status == capture_store_status::stopped ||
               pending_flush_result.status == capture_store_status::cancelled,
           "pending non-drain flush status");
    expect(pending_shutdown_result.status == capture_store_status::stopped, "pending non-drain shutdown status");
    capture_manifest pending_manifest;
    expect_status(pending_flush_store.inspect(pending_manifest).status, capture_store_status::no_manifest,
                  "pending non-drain manifest");
    const capture_store_stats pending_stats = pending_flush_store.stats();
    expect(pending_stats.dropped_on_shutdown == 1, "pending non-drain dropped count");
    expect(pending_stats.ring.size_approx == 0, "pending non-drain ring not empty");
}

}  // namespace

int main() {
    try {
        test_compact_round_trip_and_privacy();
        test_sha256_file_footers();
        test_sampled_rich_explicit_opt_in();
        test_metrics_only_identity_redaction();
        test_retention_and_restart();
        test_byte_retention();
        test_faults_corruption_and_recovery();
        test_crash_seams_and_cancellation();
        test_full_ring_progress_and_slow_worker();
        test_wakeup_barrier_and_worker_exception();
        test_bounds_off_mode_and_private_permissions();
        test_root_file_and_tombstone_security();
        test_random_salt_and_post_publication_retention();
        test_concurrent_flush_shutdown();
        test_admission_linearization_and_high_count_shutdown();
        test_filesystem_fault_seams_and_flush_ordering();
        test_post_publication_callbacks_and_pending_flush();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-server-capture-store: %s\n", error.what());
        return 1;
    }
    std::puts("test-server-capture-store: all checks passed");
    return 0;
}

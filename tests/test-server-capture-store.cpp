#include "server-capture-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace server_capture;

namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
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
        char   pattern[] = "/tmp/llama-capture-store-XXXXXX";
        char * path      = ::mkdtemp(pattern);
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
    observation.active_mode                   = dspark_mode::adaptive_depth_one;
    observation.bypass                        = bypass_reason::none;
    for (size_t index = 0; index < MAX_PROPOSAL_TOKENS; ++index) {
        observation.proposal_token_ids[index]     = 0x504f0000 + static_cast<int32_t>(sequence * 8 + index);
        observation.selected_probabilities[index] = 0.5F + static_cast<float>(index) * 0.01F;
        observation.raw_confidences[index]        = 0.8F - static_cast<float>(index) * 0.02F;
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
    config.max_shard_records    = 1;
    config.max_shard_bytes      = 300;
    config.max_retained_bytes   = 300;
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

}  // namespace

int main() {
    try {
        test_compact_round_trip_and_privacy();
        test_sampled_rich_explicit_opt_in();
        test_metrics_only_identity_redaction();
        test_retention_and_restart();
        test_byte_retention();
        test_faults_corruption_and_recovery();
        test_crash_seams_and_cancellation();
        test_full_ring_progress_and_slow_worker();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-server-capture-store: %s\n", error.what());
        return 1;
    }
    std::puts("test-server-capture-store: all checks passed");
    return 0;
}

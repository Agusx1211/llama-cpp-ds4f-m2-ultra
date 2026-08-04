#include "server-kv-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace server_kv_store;

#define CHECK(condition)                                                                         \
    do {                                                                                         \
        if (!(condition)) {                                                                      \
            std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                        \
        }                                                                                        \
    } while (false)

namespace {

std::string executable_path;

class temp_directory {
  public:
    temp_directory() {
        char   pattern[] = "/tmp/llama-server-kv-store-XXXXXX";
        char * created   = ::mkdtemp(pattern);
        CHECK(created != nullptr);
        path_ = created;
    }

    ~temp_directory() {
        std::error_code error;
        fs::remove_all(path_, error);
    }

    const std::string & path() const { return path_; }

  private:
    std::string path_;
};

object_key make_key(uint8_t seed) {
    object_key key{};
    for (size_t index = 0; index < key.size(); ++index) {
        key[index] = static_cast<uint8_t>(seed + index);
    }
    return key;
}

llama_snapshot_identity make_identity(uint8_t seed = 1) {
    llama_snapshot_identity identity;
    identity.architecture       = "deepseek-v4-flash";
    identity.target_kv_type     = "q8_0";
    identity.context_size       = 4096;
    identity.raw_window         = 2048;
    identity.c4_ratio           = 4;
    identity.hca_ratio          = 128;
    identity.dsv4_state_version = 1;
    identity.rollback_depth     = 16;
    identity.model_artifact_digest.fill(seed);
    identity.tokenizer_digest.fill(static_cast<uint8_t>(seed + 1));
    identity.chat_template_digest.fill(static_cast<uint8_t>(seed + 2));
    identity.runtime_build_digest.fill(static_cast<uint8_t>(seed + 3));
    identity.target_kv_digest.fill(static_cast<uint8_t>(seed + 4));
    identity.rope_digest.fill(static_cast<uint8_t>(seed + 5));
    return identity;
}

llama_snapshot_metadata make_metadata(uint64_t generation, uint8_t identity_seed = 1) {
    llama_snapshot_metadata metadata;
    metadata.snapshot_generation = generation;
    metadata.request_generation  = generation + 100;
    metadata.identity            = make_identity(identity_seed);
    return metadata;
}

config make_config(const std::string & root, uint64_t live_quota, uint64_t prefix_quota) {
    config cfg;
    cfg.snapshot.root_path           = root;
    cfg.snapshot.physical_device_id  = "apfs-test-device";
    cfg.snapshot.chunk_payload_bytes = 8;
    cfg.snapshot.max_snapshot_bytes  = 1024 * 1024;
    cfg.live_quota_bytes             = live_quota;
    cfg.prefix_quota_bytes           = prefix_quota;
    cfg.max_live_objects             = 32;
    cfg.max_prefix_objects           = 32;
    return cfg;
}

uint64_t estimate_bytes(const config & cfg, size_t payload_bytes) {
    const llama_snapshot_storage_estimate estimate =
        llama_snapshot_estimate_storage(cfg.snapshot, make_metadata(1), payload_bytes);
    CHECK(estimate.status == llama_snapshot_status::ok);
    return estimate.committed_bytes;
}

std::vector<uint8_t> payload(size_t size, uint8_t seed = 1) {
    std::vector<uint8_t> bytes(size);
    for (size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<uint8_t>(seed + index * 13);
    }
    return bytes;
}

void create_empty_file(const fs::path & path) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    CHECK(descriptor >= 0);
    CHECK(::close(descriptor) == 0);
}

void write_text_file(const fs::path & path, const std::string & contents) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600);
    CHECK(descriptor >= 0);
    size_t offset = 0;
    while (offset < contents.size()) {
        const ssize_t amount = ::write(descriptor, contents.data() + offset, contents.size() - offset);
        CHECK(amount > 0);
        offset += static_cast<size_t>(amount);
    }
    CHECK(::fsync(descriptor) == 0);
    CHECK(::close(descriptor) == 0);
}

fs::path object_path(const temp_directory & temporary, object_class storage_class, const object_key & key) {
    return fs::u8path(temporary.path()) / (storage_class == object_class::live ? "live" : "prefix") /
           llama_snapshot_digest_hex(key);
}

size_t object_directory_count(const temp_directory & temporary, object_class storage_class) {
    const fs::path  root = fs::u8path(temporary.path()) / (storage_class == object_class::live ? "live" : "prefix");
    std::error_code error;
    size_t          count = 0;
    for (fs::directory_iterator iterator(root, error), end; !error && iterator != end; iterator.increment(error)) {
        CHECK(!error);
        CHECK(!fs::is_symlink(iterator->symlink_status(error)));
        CHECK(!error);
        if (fs::is_directory(iterator->path(), error)) {
            ++count;
        }
        CHECK(!error);
    }
    CHECK(!error);
    return count;
}

void expect_pool_metadata_bounded(const temp_directory & temporary) {
    std::error_code error;
    for (fs::directory_iterator iterator(fs::u8path(temporary.path()), error), end; !error && iterator != end;
         iterator.increment(error)) {
        const std::string name = iterator->path().filename().string();
        CHECK(name == ".server-kv.lock" || name == ".server-kv-generation" || name == ".server-kv-generation.next" ||
              name == ".server-kv-delete-intent" || name == "live" || name == "prefix");
    }
    CHECK(!error);
}

bool create_sparse_file(const fs::path & path, off_t size) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    CHECK(descriptor >= 0);
    if (::ftruncate(descriptor, size) != 0) {
        const int error = errno;
        CHECK(::close(descriptor) == 0);
        CHECK(error == EFBIG || error == ENOSPC || error == EINVAL || error == EOPNOTSUPP);
        return false;
    }
    CHECK(::close(descriptor) == 0);
    return true;
}

void expect_child_lock_probe(const std::string & root) {
    const pid_t child = ::fork();
    CHECK(child >= 0);
    if (child == 0) {
        ::execl(executable_path.c_str(), executable_path.c_str(), "--probe-authority", root.c_str(), nullptr);
        _exit(127);
    }
    int child_status = 0;
    CHECK(::waitpid(child, &child_status, 0) == child);
    CHECK(WIFEXITED(child_status));
    CHECK(WEXITSTATUS(child_status) == 0);
}

void test_exact_charge_and_config_validation() {
    temp_directory                        temporary;
    config                                cfg   = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    const llama_snapshot_storage_estimate empty = llama_snapshot_estimate_storage(cfg.snapshot, make_metadata(1), 0);
    CHECK(empty.status == llama_snapshot_status::ok);
    CHECK(empty.chunk_count == 0);
    CHECK(empty.generation_bytes == empty.current_manifest_bytes);
    CHECK(empty.committed_bytes == empty.current_manifest_bytes * 2);

    const llama_snapshot_storage_estimate multi = llama_snapshot_estimate_storage(cfg.snapshot, make_metadata(1), 17);
    CHECK(multi.status == llama_snapshot_status::ok);
    CHECK(multi.chunk_count == 3);
    CHECK(multi.generation_bytes == 17 + 3 * LLAMA_SNAPSHOT_CHUNK_ALIGNMENT + multi.current_manifest_bytes);
    CHECK(multi.committed_bytes == multi.generation_bytes + multi.current_manifest_bytes);
    CHECK(multi.replacement_peak_bytes == multi.committed_bytes);

    store quota(cfg);
    CHECK(quota.initialization_status() == status::ok);
    const write_result written = quota.write_generation(make_key(1), object_class::live, make_metadata(1), payload(17));
    CHECK(written.store_status == status::ok);
    CHECK(written.charged_bytes == multi.committed_bytes);
    const stats counters = quota.get_stats();
    CHECK(counters.live_committed_bytes == multi.committed_bytes);
    CHECK(counters.live_reserved_bytes == 0);

    temp_directory invalid_root;
    config         invalid = make_config(invalid_root.path(), 0, 0);
    CHECK(store(invalid).initialization_status() == status::invalid_config);
    invalid.live_quota_bytes   = 100;
    invalid.prefix_quota_bytes = 101;
    CHECK(store(invalid).initialization_status() == status::invalid_config);
}

void test_prefix_and_live_reclamation() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge  = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes   = 3 * charge;
    cfg.prefix_quota_bytes = 2 * charge;
    store quota(cfg);
    CHECK(quota.initialization_status() == status::ok);

    CHECK(quota.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8, 1)).store_status ==
          status::ok);
    CHECK(quota.write_generation(make_key(2), object_class::prefix, make_metadata(1), payload(8, 2)).store_status ==
          status::ok);
    write_result third = quota.write_generation(make_key(3), object_class::prefix, make_metadata(1), payload(8, 3));
    CHECK(third.store_status == status::ok);
    CHECK(third.evicted_prefixes.size() == 1);
    CHECK(third.evicted_prefixes.front() == make_key(1));
    CHECK(quota.acquire(make_key(1), object_class::prefix, make_identity()).store_status == status::not_found);

    write_result live = quota.write_generation(make_key(4), object_class::live, make_metadata(1), payload(8, 4));
    CHECK(live.store_status == status::ok);
    CHECK(live.evicted_prefixes.empty());
    write_result live2 = quota.write_generation(make_key(5), object_class::live, make_metadata(1), payload(8, 5));
    CHECK(live2.store_status == status::ok);
    CHECK(live2.evicted_prefixes.size() == 1);
    CHECK(quota.acquire(make_key(4), object_class::live, make_identity()).store_status == status::ok);
    const stats counters = quota.get_stats();
    CHECK(counters.live_committed_bytes == 2 * charge);
    CHECK(counters.prefix_committed_bytes == charge);
    CHECK(counters.live_committed_bytes + counters.prefix_committed_bytes <= cfg.live_quota_bytes);

    temp_directory multi_root;
    config         multi_cfg = make_config(multi_root.path(), 3 * charge, 3 * charge);
    store          multi(multi_cfg);
    for (uint8_t index = 1; index <= 3; ++index) {
        CHECK(multi.write_generation(make_key(index), object_class::prefix, make_metadata(1), payload(8, index))
                  .store_status == status::ok);
    }
    const write_result large_live =
        multi.write_generation(make_key(10), object_class::live, make_metadata(1), payload(17));
    CHECK(large_live.store_status == status::ok);
    CHECK(large_live.evicted_prefixes.size() == 3);
    CHECK(multi.get_stats().prefix_objects == 0);

    temp_directory live_only_root;
    config         live_only_cfg = make_config(live_only_root.path(), charge, charge);
    store          live_only(live_only_cfg);
    CHECK(live_only.write_generation(make_key(1), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(live_only.write_generation(make_key(2), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::prefix_quota_exceeded);
    CHECK(live_only.acquire(make_key(1), object_class::live, make_identity()).store_status == status::ok);
}

void test_intrinsic_live_rejection_does_not_evict_prefix() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t small   = estimate_bytes(cfg, 8);
    const uint64_t large   = estimate_bytes(cfg, 17);
    cfg.live_quota_bytes   = 3 * small;
    cfg.prefix_quota_bytes = small;
    store quota(cfg);
    CHECK(quota.write_generation(make_key(1), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(quota.write_generation(make_key(2), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(quota.write_generation(make_key(3), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(2 * small + large > cfg.live_quota_bytes);
    const write_result rejected =
        quota.write_generation(make_key(4), object_class::live, make_metadata(1), payload(17));
    CHECK(rejected.store_status == status::live_quota_exceeded);
    CHECK(rejected.evicted_prefixes.empty());
    CHECK(quota.acquire(make_key(3), object_class::prefix, make_identity()).store_status == status::ok);
}

void test_read_lease_blocks_mandatory_reclamation() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge  = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes   = 2 * charge;
    cfg.prefix_quota_bytes = charge;
    store quota(cfg);
    CHECK(quota.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    open_result lease = quota.acquire(make_key(1), object_class::prefix, make_identity());
    CHECK(lease.store_status == status::ok);
    CHECK(quota.write_generation(make_key(2), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(quota.write_generation(make_key(3), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::blocked_by_prefix_lease);
    lease.lease.reset();
    const write_result admitted = quota.write_generation(make_key(3), object_class::live, make_metadata(1), payload(8));
    CHECK(admitted.store_status == status::ok);
    CHECK(admitted.evicted_prefixes == std::vector<object_key>{ make_key(1) });
}

void test_replacement_and_reservation_cleanup() {
    temp_directory temporary;
    config         roomy   = make_config(temporary.path(), UINT64_MAX / 4, 0);
    const uint64_t charge  = estimate_bytes(roomy, 8);
    roomy.live_quota_bytes = 2 * charge;
    store              quota(roomy);
    const object_key   key     = make_key(1);
    const write_result initial = quota.write_generation(key, object_class::live, make_metadata(1), payload(8, 1));
    CHECK(initial.store_status == status::ok);

    faults fail;
    fail.snapshot.fail_before_manifest_commit = true;
    const write_result failed =
        quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2), {}, fail);
    CHECK(failed.store_status == status::snapshot_error);
    CHECK(!failed.committed);
    CHECK(failed.generation > initial.generation);
    CHECK(quota.get_stats().live_reserved_bytes == 0);
    open_result old = quota.acquire(key, object_class::live, make_identity());
    CHECK(old.store_status == status::ok);
    CHECK(old.lease->manifest().snapshot_generation == 1);
    CHECK(old.lease->read_all().payload == payload(8, 1));
    old.lease.reset();

    const write_result replaced = quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2));
    CHECK(replaced.store_status == status::ok);
    CHECK(replaced.generation > failed.generation);
    CHECK(quota.get_stats().live_committed_bytes == charge);

    const write_result cancelled = quota.write_generation(make_key(2), object_class::live, make_metadata(1), payload(8),
                                                          [](uint32_t) { return true; });
    CHECK(cancelled.snapshot_status == llama_snapshot_status::cancelled);
    CHECK(quota.get_stats().live_reserved_bytes == 0);

    temp_directory tight_root;
    config         tight = make_config(tight_root.path(), charge, 0);
    store          tight_quota(tight);
    CHECK(tight_quota.write_generation(key, object_class::live, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(tight_quota.write_generation(key, object_class::live, make_metadata(2), payload(8)).store_status ==
          status::live_quota_exceeded);
    CHECK(tight_quota.acquire(key, object_class::live, make_identity()).lease->manifest().snapshot_generation == 1);
}

void test_preserved_failed_replacement_is_charged_until_reconciled() {
    temp_directory temporary;
    config         cfg    = make_config(temporary.path(), UINT64_MAX / 4, 0);
    const uint64_t charge = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes  = 2 * charge;
    store            quota(cfg);
    const object_key key = make_key(1);
    CHECK(quota.write_generation(key, object_class::live, make_metadata(1), payload(8, 1)).store_status == status::ok);

    faults injected;
    injected.snapshot.fail_before_manifest_commit = true;
    injected.snapshot.preserve_failed_generation  = true;
    const write_result failed =
        quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2), {}, injected);
    CHECK(failed.store_status == status::reconciliation_required);
    CHECK(!failed.committed);
    CHECK(quota.get_stats().live_committed_bytes == charge);
    CHECK(quota.get_stats().live_reserved_bytes == charge);
    CHECK(quota.get_stats().reconciliation_needed);
    CHECK(quota.acquire(key, object_class::live, make_identity()).store_status == status::reconciliation_required);

    const fs::path object_root = fs::u8path(temporary.path()) / "live" / llama_snapshot_digest_hex(key);
    CHECK(fs::exists(object_root / "generation-0000000000000002"));
    CHECK(quota.reconcile() == status::ok);
    CHECK(!fs::exists(object_root / "generation-0000000000000002"));
    CHECK(quota.get_stats().live_committed_bytes == charge);
    CHECK(quota.get_stats().live_reserved_bytes == 0);
    open_result current = quota.acquire(key, object_class::live, make_identity());
    CHECK(current.store_status == status::ok);
    CHECK(current.lease->manifest().snapshot_generation == 1);
    CHECK(current.lease->read_all().payload == payload(8, 1));
}

void test_failed_deletion_retains_charge() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge  = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes   = charge;
    cfg.prefix_quota_bytes = charge;
    store quota(cfg);
    CHECK(quota.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    faults injected;
    injected.fail_prefix_delete = true;
    const write_result failed =
        quota.write_generation(make_key(2), object_class::prefix, make_metadata(1), payload(8), {}, injected);
    CHECK(failed.store_status == status::io_error);
    CHECK(quota.get_stats().prefix_committed_bytes == charge);
    CHECK(quota.acquire(make_key(1), object_class::prefix, make_identity()).store_status == status::ok);
    CHECK(!fs::exists(object_path(temporary, object_class::prefix, make_key(2))));
}

void test_commit_uncertainty_and_restart_reconciliation() {
    temp_directory temporary;
    config         cfg    = make_config(temporary.path(), UINT64_MAX / 4, 0);
    const uint64_t charge = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes  = 2 * charge;
    const object_key key  = make_key(1);
    {
        store  quota(cfg);
        faults injected;
        injected.snapshot.fail_after_manifest_commit = true;
        const write_result uncertain =
            quota.write_generation(key, object_class::live, make_metadata(1), payload(8), {}, injected);
        CHECK(uncertain.store_status == status::commit_uncertain);
        CHECK(uncertain.committed);
        CHECK(quota.get_stats().reconciliation_needed);
        CHECK(quota.acquire(key, object_class::live, make_identity()).store_status == status::reconciliation_required);
        CHECK(quota.reconcile() == status::ok);
        CHECK(quota.acquire(key, object_class::live, make_identity()).store_status == status::ok);
    }
    store restarted(cfg);
    CHECK(restarted.initialization_status() == status::ok);
    open_result opened = restarted.acquire(key, object_class::live, make_identity());
    CHECK(opened.store_status == status::ok);
    CHECK(opened.lease->validate() == llama_snapshot_status::ok);
    CHECK(restarted.get_stats().live_committed_bytes == charge);
}

void test_exact_generation_delete_and_class_immutability() {
    temp_directory     temporary;
    config             cfg = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    store              quota(cfg);
    const object_key   key     = make_key(1);
    const write_result written = quota.write_generation(key, object_class::live, make_metadata(2), payload(8));
    CHECK(written.store_status == status::ok);
    CHECK(written.generation != 0);
    CHECK(quota.write_generation(key, object_class::prefix, make_metadata(3), payload(8)).store_status ==
          status::class_conflict);
    CHECK(quota.erase(key, object_class::prefix, written.generation).store_status == status::class_conflict);
    CHECK(quota.erase(key, object_class::live, written.generation - 1).store_status == status::stale_generation);
    open_result lease = quota.acquire(key, object_class::live, make_identity());
    CHECK(lease.store_status == status::ok);
    CHECK(quota.erase(key, object_class::live, written.generation).store_status == status::object_in_use);
    lease.lease.reset();
    const erase_result erased = quota.erase(key, object_class::live, written.generation);
    CHECK(erased.store_status == status::ok);
    CHECK(erased.released_bytes != 0);
    CHECK(quota.acquire(key, object_class::live, make_identity()).store_status == status::not_found);
}

void test_erase_recreate_is_aba_proof_across_restart() {
    temp_directory   temporary;
    config           cfg = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    const object_key key = make_key(1);
    {
        store              quota(cfg);
        const write_result first = quota.write_generation(key, object_class::live, make_metadata(200), payload(8, 2));
        CHECK(first.store_status == status::ok);
        CHECK(first.generation != 200);
        CHECK(quota.erase(key, object_class::live, first.generation).store_status == status::ok);
    }
    {
        store restarted(cfg);
        CHECK(restarted.initialization_status() == status::ok);
        const write_result recreated =
            restarted.write_generation(key, object_class::live, make_metadata(1), payload(8, 3));
        CHECK(recreated.store_status == status::ok);
        CHECK(recreated.generation > 1);
        CHECK(restarted.erase(key, object_class::live, 1).store_status == status::stale_generation);
        open_result current = restarted.acquire(key, object_class::live, make_identity());
        CHECK(current.store_status == status::ok);
        CHECK(current.lease->manifest().snapshot_generation == recreated.generation);
        CHECK(current.lease->read_all().payload == payload(8, 3));
    }
    {
        store final_restart(cfg);
        CHECK(final_restart.initialization_status() == status::ok);
        CHECK(final_restart.erase(key, object_class::live, 1).store_status == status::stale_generation);
        CHECK(final_restart.acquire(key, object_class::live, make_identity()).store_status == status::ok);
    }
}

void test_delete_intent_crash_boundaries() {
    for (const bool fail_after_intent : { true, false }) {
        temp_directory   temporary;
        config           cfg              = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
        const object_key key              = make_key(static_cast<uint8_t>(fail_after_intent ? 11 : 12));
        uint64_t         first_generation = 0;
        {
            store              quota(cfg);
            const write_result first = quota.write_generation(key, object_class::live, make_metadata(7), payload(8, 7));
            CHECK(first.store_status == status::ok);
            first_generation = first.generation;
            faults injected;
            injected.fail_after_delete_intent = fail_after_intent;
            injected.fail_after_delete_unlink = !fail_after_intent;
            const erase_result uncertain      = quota.erase(key, object_class::live, first.generation, injected);
            CHECK(uncertain.store_status == status::commit_uncertain);
            CHECK(quota.get_stats().reconciliation_needed);
            CHECK(fs::exists(fs::u8path(temporary.path()) / ".server-kv-delete-intent"));
            CHECK(fail_after_intent ? fs::exists(object_path(temporary, object_class::live, key)) :
                                      !fs::exists(object_path(temporary, object_class::live, key)));
        }
        {
            store restarted(cfg);
            CHECK(restarted.initialization_status() == status::ok);
            CHECK(!fs::exists(fs::u8path(temporary.path()) / ".server-kv-delete-intent"));
            CHECK(restarted.acquire(key, object_class::live, make_identity()).store_status == status::not_found);
            const write_result recreated =
                restarted.write_generation(key, object_class::live, make_metadata(99), payload(8, 9));
            CHECK(recreated.store_status == status::ok);
            CHECK(recreated.generation > first_generation);
            CHECK(restarted.erase(key, object_class::live, first_generation).store_status == status::stale_generation);
        }
        {
            store final_restart(cfg);
            CHECK(final_restart.initialization_status() == status::ok);
            CHECK(final_restart.erase(key, object_class::live, first_generation).store_status ==
                  status::stale_generation);
            CHECK(final_restart.acquire(key, object_class::live, make_identity()).store_status == status::ok);
        }
    }
}

void test_generation_clock_recovery_and_malformed_records() {
    temp_directory   temporary;
    config           cfg              = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    const object_key key              = make_key(31);
    uint64_t         first_generation = 0;
    {
        store              quota(cfg);
        const write_result first = quota.write_generation(key, object_class::live, make_metadata(1), payload(8));
        CHECK(first.store_status == status::ok);
        first_generation = first.generation;
    }
    const uint64_t recovered_clock = first_generation + 100;
    write_text_file(
        fs::u8path(temporary.path()) / ".server-kv-generation.next",
        "server-kv-generation-v1\nvalue=" +
            [&]() {
                char buffer[17];
                std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(recovered_clock));
                return std::string(buffer);
            }() +
            "\ninverse=" +
            [&]() {
                char buffer[17];
                std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(~recovered_clock));
                return std::string(buffer);
            }() +
            "\n");
    {
        store restarted(cfg);
        CHECK(restarted.initialization_status() == status::ok);
        CHECK(!fs::exists(fs::u8path(temporary.path()) / ".server-kv-generation.next"));
        const write_result next =
            restarted.write_generation(make_key(32), object_class::live, make_metadata(2), payload(8, 2));
        CHECK(next.store_status == status::ok);
        CHECK(next.generation == recovered_clock + 1);
    }

    temp_directory malformed_clock;
    config         malformed_clock_cfg = make_config(malformed_clock.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    write_text_file(fs::u8path(malformed_clock.path()) / ".server-kv-generation", "not-a-clock\n");
    store malformed_clock_store(malformed_clock_cfg);
    CHECK(malformed_clock_store.initialization_status() == status::reconciliation_required);
    CHECK(malformed_clock_store.get_stats().reconciliation_needed);

    temp_directory malformed_next;
    config         malformed_next_cfg = make_config(malformed_next.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    {
        store quota(malformed_next_cfg);
        CHECK(quota.initialization_status() == status::ok);
    }
    write_text_file(fs::u8path(malformed_next.path()) / ".server-kv-generation.next", "bad-next\n");
    store malformed_next_store(malformed_next_cfg);
    CHECK(malformed_next_store.initialization_status() == status::reconciliation_required);

    temp_directory malformed_intent;
    config         malformed_intent_cfg = make_config(malformed_intent.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    {
        store quota(malformed_intent_cfg);
        CHECK(quota.write_generation(make_key(41), object_class::live, make_metadata(1), payload(8)).store_status ==
              status::ok);
    }
    write_text_file(fs::u8path(malformed_intent.path()) / ".server-kv-delete-intent", "bad-intent\n");
    store malformed_intent_store(malformed_intent_cfg);
    CHECK(malformed_intent_store.initialization_status() == status::reconciliation_required);
}

void test_pool_authority_covers_processes_and_lease_lifetime() {
    temp_directory              temporary;
    config                      cfg = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    std::unique_ptr<read_lease> retained;
    {
        store owner(cfg);
        CHECK(owner.initialization_status() == status::ok);
        CHECK(owner.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
              status::ok);
        open_result acquired = owner.acquire(make_key(1), object_class::prefix, make_identity());
        CHECK(acquired.store_status == status::ok);
        retained = std::move(acquired.lease);

        store same_process(cfg);
        CHECK(same_process.initialization_status() == status::authority_unavailable);
        CHECK(same_process.erase(make_key(1), object_class::prefix, 1).store_status == status::authority_unavailable);
        expect_child_lock_probe(temporary.path());
    }
    store lease_holds_authority(cfg);
    CHECK(lease_holds_authority.initialization_status() == status::authority_unavailable);
    retained.reset();
    store successor(cfg);
    CHECK(successor.initialization_status() == status::ok);
    CHECK(successor.acquire(make_key(1), object_class::prefix, make_identity()).store_status == status::ok);
}

void test_directory_creation_parent_fsync_failures() {
    temp_directory temporary;
    {
        config cfg                             = make_config(temporary.path() + "/pool", UINT64_MAX / 4, 0);
        cfg.test_faults.fail_pool_parent_fsync = true;
        {
            store failed(cfg);
            CHECK(failed.initialization_status() == status::commit_uncertain);
        }
        {
            store retried(cfg);
            CHECK(retried.initialization_status() == status::commit_uncertain);
        }
    }
    {
        config recovered = make_config(temporary.path() + "/pool", UINT64_MAX / 4, 0);
        store  quota(recovered);
        CHECK(quota.initialization_status() == status::ok);
    }

    temp_directory class_root;
    {
        config cfg                              = make_config(class_root.path(), UINT64_MAX / 4, 0);
        cfg.test_faults.fail_class_parent_fsync = true;
        {
            store failed(cfg);
            CHECK(failed.initialization_status() == status::commit_uncertain);
        }
        {
            store retried(cfg);
            CHECK(retried.initialization_status() == status::commit_uncertain);
        }
    }
    config class_recovered = make_config(class_root.path(), UINT64_MAX / 4, 0);
    {
        store quota(class_recovered);
        CHECK(quota.initialization_status() == status::ok);
    }

    temp_directory object_root;
    config         object_cfg     = make_config(object_root.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    const uint64_t charge         = estimate_bytes(object_cfg, 8);
    object_cfg.live_quota_bytes   = charge;
    object_cfg.prefix_quota_bytes = charge;
    {
        store quota(object_cfg);
        CHECK(quota.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
              status::ok);
        faults injected;
        injected.fail_object_parent_fsync = true;
        const write_result failed =
            quota.write_generation(make_key(2), object_class::live, make_metadata(1), payload(8), {}, injected);
        CHECK(failed.store_status == status::commit_uncertain);
        CHECK(failed.evicted_prefixes.empty());
        CHECK(quota.get_stats().prefix_committed_bytes == charge);
        CHECK(quota.get_stats().live_committed_bytes == 0);
    }
    store recovered(object_cfg);
    CHECK(recovered.initialization_status() == status::ok);
    CHECK(recovered.acquire(make_key(1), object_class::prefix, make_identity()).store_status == status::ok);
}

void test_rejected_writes_leave_no_object_directory() {
    const auto reject_and_restart = [](object_class storage_class, uint64_t live_quota, uint64_t prefix_quota,
                                       size_t max_live, size_t max_prefix, const object_key & key, status expected) {
        temp_directory temporary;
        config         cfg     = make_config(temporary.path(), live_quota, prefix_quota);
        cfg.max_live_objects   = max_live;
        cfg.max_prefix_objects = max_prefix;
        {
            store quota(cfg);
            CHECK(quota.initialization_status() == status::ok);
            const write_result rejected = quota.write_generation(key, storage_class, make_metadata(77), payload(8, 77));
            CHECK(rejected.store_status == expected);
            CHECK(!fs::exists(object_path(temporary, storage_class, key)));
            const stats counters = quota.get_stats();
            CHECK(counters.live_reserved_bytes == 0);
            CHECK(counters.prefix_reserved_bytes == 0);
        }
        {
            store restarted(cfg);
            CHECK(restarted.initialization_status() == status::ok);
            const write_result rejected =
                restarted.write_generation(key, storage_class, make_metadata(88), payload(8, 88));
            CHECK(rejected.store_status == expected);
            CHECK(!fs::exists(object_path(temporary, storage_class, key)));
            const stats counters = restarted.get_stats();
            CHECK(counters.live_reserved_bytes == 0);
            CHECK(counters.prefix_reserved_bytes == 0);
        }
    };

    temp_directory estimate_root;
    const config   estimate_cfg = make_config(estimate_root.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge       = estimate_bytes(estimate_cfg, 8);

    reject_and_restart(object_class::live, charge - 1, 0, 32, 32, make_key(61), status::live_quota_exceeded);

    {
        temp_directory temporary;
        config         cfg   = make_config(temporary.path(), 4 * charge, 0);
        cfg.max_live_objects = 1;
        {
            store quota(cfg);
            CHECK(quota.write_generation(make_key(62), object_class::live, make_metadata(1), payload(8)).store_status ==
                  status::ok);
            const write_result rejected =
                quota.write_generation(make_key(63), object_class::live, make_metadata(1), payload(8));
            CHECK(rejected.store_status == status::object_limit);
            CHECK(!fs::exists(object_path(temporary, object_class::live, make_key(63))));
            CHECK(quota.get_stats().live_objects == 1);
        }
        store restarted(cfg);
        CHECK(restarted.initialization_status() == status::ok);
        const write_result rejected =
            restarted.write_generation(make_key(63), object_class::live, make_metadata(1), payload(8));
        CHECK(rejected.store_status == status::object_limit);
        CHECK(!fs::exists(object_path(temporary, object_class::live, make_key(63))));
        CHECK(restarted.get_stats().live_objects == 1);
    }

    {
        temp_directory temporary;
        config         cfg     = make_config(temporary.path(), 4 * charge, 4 * charge);
        cfg.max_prefix_objects = 1;
        {
            store quota(cfg);
            CHECK(
                quota.write_generation(make_key(64), object_class::prefix, make_metadata(1), payload(8)).store_status ==
                status::ok);
            open_result held = quota.acquire(make_key(64), object_class::prefix, make_identity());
            CHECK(held.store_status == status::ok);
            const write_result blocked =
                quota.write_generation(make_key(65), object_class::prefix, make_metadata(1), payload(8));
            CHECK(blocked.store_status == status::blocked_by_prefix_lease);
            CHECK(!fs::exists(object_path(temporary, object_class::prefix, make_key(65))));
            held.lease.reset();
        }
        {
            store restarted(cfg);
            CHECK(restarted.initialization_status() == status::ok);
            open_result held = restarted.acquire(make_key(64), object_class::prefix, make_identity());
            CHECK(held.store_status == status::ok);
            const write_result blocked =
                restarted.write_generation(make_key(65), object_class::prefix, make_metadata(2), payload(8));
            CHECK(blocked.store_status == status::blocked_by_prefix_lease);
            CHECK(!fs::exists(object_path(temporary, object_class::prefix, make_key(65))));
        }
    }

    reject_and_restart(object_class::prefix, 4 * charge, charge - 1, 32, 32, make_key(66),
                       status::prefix_quota_exceeded);

    {
        temp_directory temporary;
        config         cfg = make_config(temporary.path(), 2 * charge, 2 * charge);
        {
            store quota(cfg);
            CHECK(quota.write_generation(make_key(67), object_class::live, make_metadata(1), payload(8)).store_status ==
                  status::ok);
            CHECK(
                quota.write_generation(make_key(68), object_class::prefix, make_metadata(1), payload(8)).store_status ==
                status::ok);
            open_result held = quota.acquire(make_key(68), object_class::prefix, make_identity());
            CHECK(held.store_status == status::ok);
            const write_result blocked =
                quota.write_generation(make_key(69), object_class::prefix, make_metadata(1), payload(8));
            CHECK(blocked.store_status == status::blocked_by_prefix_lease);
            CHECK(!fs::exists(object_path(temporary, object_class::prefix, make_key(69))));
            CHECK(quota.get_stats().prefix_objects == 1);
        }
        {
            store restarted(cfg);
            CHECK(restarted.initialization_status() == status::ok);
            open_result held = restarted.acquire(make_key(68), object_class::prefix, make_identity());
            CHECK(held.store_status == status::ok);
            const write_result blocked =
                restarted.write_generation(make_key(69), object_class::prefix, make_metadata(2), payload(8));
            CHECK(blocked.store_status == status::blocked_by_prefix_lease);
            CHECK(!fs::exists(object_path(temporary, object_class::prefix, make_key(69))));
            CHECK(restarted.get_stats().prefix_objects == 1);
        }
    }
}

void test_churn_remains_bounded_across_restart() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge  = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes   = 4 * charge;
    cfg.prefix_quota_bytes = 2 * charge;
    cfg.max_live_objects   = 1;
    cfg.max_prefix_objects = 2;

    std::unique_ptr<store> quota = std::make_unique<store>(cfg);
    CHECK(quota->initialization_status() == status::ok);
    for (uint16_t index = 1; index <= 200; ++index) {
        if (index != 1 && index % 13 == 0) {
            quota.reset();
            quota = std::make_unique<store>(cfg);
            CHECK(quota->initialization_status() == status::ok);
        }
        const object_key   key = make_key(static_cast<uint8_t>(index));
        const write_result result =
            quota->write_generation(key, object_class::prefix, make_metadata(0), payload(8, index));
        CHECK(result.store_status == status::ok);
        const stats counters = quota->get_stats();
        CHECK(counters.prefix_objects <= cfg.max_prefix_objects);
        CHECK(counters.prefix_committed_bytes <= cfg.prefix_quota_bytes);
        CHECK(counters.live_objects <= cfg.max_live_objects);
        CHECK(counters.live_committed_bytes + counters.prefix_committed_bytes <= cfg.live_quota_bytes);
        CHECK(object_directory_count(temporary, object_class::prefix) <= cfg.max_prefix_objects);
        CHECK(object_directory_count(temporary, object_class::live) <= cfg.max_live_objects);
        expect_pool_metadata_bounded(temporary);
    }

    const object_key live_key = make_key(240);
    for (uint16_t index = 0; index < 200; ++index) {
        if (index != 0 && index % 17 == 0) {
            quota.reset();
            quota = std::make_unique<store>(cfg);
            CHECK(quota->initialization_status() == status::ok);
        }
        const write_result written =
            quota->write_generation(live_key, object_class::live, make_metadata(0), payload(8, index));
        CHECK(written.store_status == status::ok);
        CHECK(written.generation != 0);
        CHECK(quota->erase(live_key, object_class::live, written.generation).store_status == status::ok);
        CHECK(object_directory_count(temporary, object_class::live) == 0);
        CHECK(object_directory_count(temporary, object_class::prefix) <= cfg.max_prefix_objects);
        expect_pool_metadata_bounded(temporary);
    }
    quota.reset();
    store final_restart(cfg);
    CHECK(final_restart.initialization_status() == status::ok);
    CHECK(object_directory_count(temporary, object_class::live) == 0);
    CHECK(object_directory_count(temporary, object_class::prefix) <= cfg.max_prefix_objects);
    expect_pool_metadata_bounded(temporary);
}

void test_sparse_stale_charge_overflow_fails_restart_closed() {
    temp_directory   temporary;
    config           cfg = make_config(temporary.path(), UINT64_MAX, 0);
    const object_key key = make_key(1);
    {
        store quota(cfg);
        CHECK(quota.write_generation(key, object_class::live, make_metadata(1), payload(8)).store_status == status::ok);
    }
    const fs::path stale =
        fs::u8path(temporary.path()) / "live" / llama_snapshot_digest_hex(key) / "generation-0000000000000002";
    CHECK(fs::create_directory(stale));
    create_empty_file(stale / "generation.manifest");
    const llama_snapshot_storage_estimate oversized =
        llama_snapshot_estimate_storage(cfg.snapshot, make_metadata(1), std::numeric_limits<uint64_t>::max());
    CHECK(oversized.status == llama_snapshot_status::invalid_argument);
    const off_t huge = std::numeric_limits<off_t>::max() - 4095;
    if (!create_sparse_file(stale / "chunk-00000000.pack", huge) ||
        !create_sparse_file(stale / "chunk-00000001.pack", huge)) {
        std::fprintf(
            stderr,
            "SKIP: filesystem cannot construct near-off_t-maximum sparse files; logical overflow check passed\n");
        return;
    }

    store restarted(cfg);
    CHECK(restarted.initialization_status() == status::reconciliation_required);
    CHECK(restarted.get_stats().reconciliation_needed);
    CHECK(restarted.write_generation(make_key(2), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::reconciliation_required);
}

void test_prefix_object_count_uses_disposable_lru() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    cfg.max_prefix_objects = 2;
    store quota(cfg);
    CHECK(quota.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    CHECK(quota.write_generation(make_key(2), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    {
        open_result touch = quota.acquire(make_key(1), object_class::prefix, make_identity());
        CHECK(touch.store_status == status::ok);
    }
    const write_result third = quota.write_generation(make_key(3), object_class::prefix, make_metadata(1), payload(8));
    CHECK(third.store_status == status::ok);
    CHECK(third.evicted_prefixes == std::vector<object_key>{ make_key(2) });
    CHECK(quota.acquire(make_key(1), object_class::prefix, make_identity()).store_status == status::ok);
    CHECK(quota.acquire(make_key(2), object_class::prefix, make_identity()).store_status == status::not_found);
    CHECK(quota.acquire(make_key(3), object_class::prefix, make_identity()).store_status == status::ok);

    temp_directory leased_root;
    config         leased_cfg     = make_config(leased_root.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    leased_cfg.max_prefix_objects = 1;
    store leased(leased_cfg);
    CHECK(leased.write_generation(make_key(1), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::ok);
    open_result held = leased.acquire(make_key(1), object_class::prefix, make_identity());
    CHECK(held.store_status == status::ok);
    CHECK(leased.write_generation(make_key(2), object_class::prefix, make_metadata(1), payload(8)).store_status ==
          status::blocked_by_prefix_lease);
}

void test_unknown_disk_state_fails_closed() {
    temp_directory temporary;
    config         cfg = make_config(temporary.path(), UINT64_MAX / 4, 0);
    {
        store quota(cfg);
        CHECK(quota.initialization_status() == status::ok);
        CHECK(quota.write_generation(make_key(1), object_class::live, make_metadata(1), payload(8)).store_status ==
              status::ok);
    }
    const fs::path object_root = fs::u8path(temporary.path()) / "live" / llama_snapshot_digest_hex(make_key(1));
    CHECK(::mkdir((object_root / ".partial-generation-0000000000000002").c_str(), 0700) == 0);
    const fs::path partial_only_root = fs::u8path(temporary.path()) / "prefix" / llama_snapshot_digest_hex(make_key(2));
    CHECK(fs::create_directories(partial_only_root / ".partial-generation-0000000000000001"));
    {
        store cleaned(cfg);
        CHECK(cleaned.initialization_status() == status::ok);
        CHECK(!fs::exists(object_root / ".partial-generation-0000000000000002"));
        CHECK(!fs::exists(partial_only_root));
    }
    const fs::path unknown = fs::u8path(temporary.path()) / "unexpected";
    CHECK(::mkdir(unknown.c_str(), 0700) == 0);
    store restarted(cfg);
    CHECK(restarted.initialization_status() == status::reconciliation_required);
    CHECK(restarted.write_generation(make_key(1), object_class::live, make_metadata(1), payload(8)).store_status ==
          status::reconciliation_required);
}

void test_bounded_prefix_stress() {
    temp_directory temporary;
    config         cfg     = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 4);
    const uint64_t charge  = estimate_bytes(cfg, 8);
    cfg.live_quota_bytes   = 4 * charge;
    cfg.prefix_quota_bytes = 3 * charge;
    store quota(cfg);
    for (uint8_t index = 1; index <= 20; ++index) {
        const write_result result =
            quota.write_generation(make_key(index), object_class::prefix, make_metadata(1), payload(8, index));
        CHECK(result.store_status == status::ok);
        const stats counters = quota.get_stats();
        CHECK(counters.prefix_committed_bytes <= cfg.prefix_quota_bytes);
        CHECK(counters.live_committed_bytes + counters.prefix_committed_bytes <= cfg.live_quota_bytes);
        CHECK(counters.live_reserved_bytes == 0);
        CHECK(counters.prefix_reserved_bytes == 0);
    }
    CHECK(quota.get_stats().prefix_objects == 3);
    CHECK(quota.acquire(make_key(20), object_class::prefix, make_identity()).store_status == status::ok);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--probe-authority") {
        const config probe_cfg = make_config(argv[2], UINT64_MAX / 4, UINT64_MAX / 8);
        return store(probe_cfg).initialization_status() == status::authority_unavailable ? 0 : 1;
    }
    CHECK(argc == 1);
    executable_path = fs::canonical(argv[0]).string();
    test_exact_charge_and_config_validation();
    test_prefix_and_live_reclamation();
    test_intrinsic_live_rejection_does_not_evict_prefix();
    test_read_lease_blocks_mandatory_reclamation();
    test_replacement_and_reservation_cleanup();
    test_preserved_failed_replacement_is_charged_until_reconciled();
    test_failed_deletion_retains_charge();
    test_commit_uncertainty_and_restart_reconciliation();
    test_exact_generation_delete_and_class_immutability();
    test_erase_recreate_is_aba_proof_across_restart();
    test_delete_intent_crash_boundaries();
    test_generation_clock_recovery_and_malformed_records();
    test_pool_authority_covers_processes_and_lease_lifetime();
    test_directory_creation_parent_fsync_failures();
    test_rejected_writes_leave_no_object_directory();
    test_churn_remains_bounded_across_restart();
    test_sparse_stale_charge_overflow_fails_restart_closed();
    test_prefix_object_count_uses_disposable_lru();
    test_unknown_disk_state_fails_closed();
    test_bounded_prefix_stress();
    std::puts("server kv store quota tests passed");
    return 0;
}

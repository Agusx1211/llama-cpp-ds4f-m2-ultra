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

void create_sparse_file(const fs::path & path, off_t size) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    CHECK(descriptor >= 0);
    CHECK(::ftruncate(descriptor, size) == 0);
    CHECK(::close(descriptor) == 0);
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
    store            quota(roomy);
    const object_key key = make_key(1);
    CHECK(quota.write_generation(key, object_class::live, make_metadata(1), payload(8, 1)).store_status == status::ok);

    faults fail;
    fail.snapshot.fail_before_manifest_commit = true;
    const write_result failed =
        quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2), {}, fail);
    CHECK(failed.store_status == status::snapshot_error);
    CHECK(!failed.committed);
    CHECK(quota.get_stats().live_reserved_bytes == 0);
    open_result old = quota.acquire(key, object_class::live, make_identity());
    CHECK(old.store_status == status::ok);
    CHECK(old.lease->manifest().snapshot_generation == 1);
    CHECK(old.lease->read_all().payload == payload(8, 1));
    old.lease.reset();

    CHECK(quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2)).store_status == status::ok);
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
    temp_directory   temporary;
    config           cfg = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    store            quota(cfg);
    const object_key key = make_key(1);
    CHECK(quota.write_generation(key, object_class::live, make_metadata(2), payload(8)).store_status == status::ok);
    CHECK(quota.write_generation(key, object_class::prefix, make_metadata(3), payload(8)).store_status ==
          status::class_conflict);
    CHECK(quota.erase(key, object_class::prefix, 2).store_status == status::class_conflict);
    CHECK(quota.erase(key, object_class::live, 1).store_status == status::stale_generation);
    open_result lease = quota.acquire(key, object_class::live, make_identity());
    CHECK(lease.store_status == status::ok);
    CHECK(quota.erase(key, object_class::live, 2).store_status == status::object_in_use);
    lease.lease.reset();
    const erase_result erased = quota.erase(key, object_class::live, 2);
    CHECK(erased.store_status == status::ok);
    CHECK(erased.released_bytes != 0);
    CHECK(quota.acquire(key, object_class::live, make_identity()).store_status == status::not_found);
}

void test_erase_recreate_is_aba_proof_across_restart() {
    temp_directory   temporary;
    config           cfg = make_config(temporary.path(), UINT64_MAX / 4, UINT64_MAX / 8);
    const object_key key = make_key(1);
    {
        store quota(cfg);
        CHECK(quota.write_generation(key, object_class::live, make_metadata(2), payload(8, 2)).store_status ==
              status::ok);
        CHECK(quota.erase(key, object_class::live, 2).store_status == status::ok);
        CHECK(quota.write_generation(key, object_class::prefix, make_metadata(3), payload(8)).store_status ==
              status::class_conflict);
    }
    {
        store restarted(cfg);
        CHECK(restarted.initialization_status() == status::ok);
        CHECK(restarted.write_generation(key, object_class::live, make_metadata(2), payload(8, 2)).store_status ==
              status::stale_generation);
        CHECK(restarted.write_generation(key, object_class::live, make_metadata(3), payload(8, 3)).store_status ==
              status::ok);
        CHECK(restarted.erase(key, object_class::live, 2).store_status == status::stale_generation);
        open_result current = restarted.acquire(key, object_class::live, make_identity());
        CHECK(current.store_status == status::ok);
        CHECK(current.lease->manifest().snapshot_generation == 3);
        CHECK(current.lease->read_all().payload == payload(8, 3));
    }
    {
        store final_restart(cfg);
        CHECK(final_restart.initialization_status() == status::ok);
        CHECK(final_restart.acquire(key, object_class::live, make_identity()).lease->manifest().snapshot_generation ==
              3);
    }
    const fs::path object_root = fs::u8path(temporary.path()) / "live" / llama_snapshot_digest_hex(key);
    create_empty_file(object_root / ".deleted-generation-0000000000000003");
    CHECK(fs::remove(object_root / "generation-0000000000000003" / "chunk-00000000.pack"));
    store interrupted_erase(cfg);
    CHECK(interrupted_erase.initialization_status() == status::ok);
    CHECK(interrupted_erase.acquire(key, object_class::live, make_identity()).store_status == status::not_found);
    CHECK(interrupted_erase.write_generation(key, object_class::live, make_metadata(3), payload(8)).store_status ==
          status::stale_generation);
    CHECK(interrupted_erase.write_generation(key, object_class::live, make_metadata(4), payload(8, 4)).store_status ==
          status::ok);
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
        store failed(cfg);
        CHECK(failed.initialization_status() == status::commit_uncertain);
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
        store failed(cfg);
        CHECK(failed.initialization_status() == status::commit_uncertain);
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
    const off_t huge = std::numeric_limits<off_t>::max() - 4095;
    create_sparse_file(stale / "chunk-00000000.pack", huge);
    create_sparse_file(stale / "chunk-00000001.pack", huge);

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
    test_pool_authority_covers_processes_and_lease_lifetime();
    test_directory_creation_parent_fsync_failures();
    test_sparse_stale_charge_overflow_fails_restart_closed();
    test_prefix_object_count_uses_disposable_lru();
    test_unknown_disk_state_fails_closed();
    test_bounded_prefix_stress();
    std::puts("server kv store quota tests passed");
    return 0;
}

#include "llama-dsv4-segment-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
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

void expect_status(llama_dsv4_segment_status actual, llama_dsv4_segment_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_dsv4_segment_status_name(expected) + ", got " +
             llama_dsv4_segment_status_name(actual));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        char   pattern[] = "/tmp/llama-dsv4-segment-store-XXXXXX";
        char * created   = ::mkdtemp(pattern);
        if (created == nullptr) {
            fail("mkdtemp failed: " + std::string(std::strerror(errno)));
        }
        // Darwin exposes /tmp as a symlink to /private/tmp. The store
        // deliberately rejects symlinked path components, so hand it the
        // canonical directory created by mkdtemp rather than the alias used
        // to create it.
        value = fs::canonical(created);
    }

    ~temporary_directory() {
        std::error_code error;
        fs::remove_all(value, error);
    }

    const fs::path & path() const { return value; }
  private:
    fs::path value;
};

uint64_t hash_u64(uint64_t hash, uint64_t value) {
    for (uint32_t index = 0; index < 8; ++index) {
        hash ^= (value >> (8 * index)) & 0xffU;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t hash_bytes(uint64_t hash, const uint8_t * data, size_t size) {
    hash = hash_u64(hash, size);
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

uint64_t row_checksum(const llama_dsv4_logical_row_chunk & chunk) {
    uint64_t              hash     = UINT64_C(14695981039346656037);
    static constexpr char domain[] = "llama.cpp-dsv4-logical-row-chunk-v1";
    hash                           = hash_u64(hash, sizeof(domain) - 1);
    for (size_t index = 0; index < sizeof(domain) - 1; ++index) {
        hash ^= static_cast<uint8_t>(domain[index]);
        hash *= UINT64_C(1099511628211);
    }
    hash = hash_u64(hash, chunk.row_begin);
    hash = hash_u64(hash, chunk.row_count);
    return hash_bytes(hash, chunk.bytes.data(), chunk.bytes.size());
}

uint64_t plane_checksum(const llama_kv_iswa_logical_plane_state & plane) {
    uint64_t hash = UINT64_C(14695981039346656037);
    hash          = hash_u64(hash, plane.schema_version);
    hash          = hash_u64(hash, plane.positions.size());
    for (llama_pos position : plane.positions) {
        hash = hash_u64(hash, static_cast<uint64_t>(position));
    }
    hash = hash_u64(hash, plane.extensions.size());
    for (const auto & extension : plane.extensions) {
        hash = hash_u64(hash, static_cast<uint64_t>(extension.x));
        hash = hash_u64(hash, static_cast<uint64_t>(extension.y));
    }
    return hash_bytes(hash, plane.tensor_payload.data(), plane.tensor_payload.size());
}

std::vector<uint8_t> bytes(size_t size, uint8_t seed) {
    std::vector<uint8_t> result(size);
    for (size_t index = 0; index < size; ++index) {
        result[index] = static_cast<uint8_t>(seed + index * 29 + index / 7);
    }
    return result;
}

llama_dsv4_logical_tensor_state make_tensor(uint32_t layer,
                                            uint64_t rows,
                                            uint8_t  seed,
                                            bool     recurrent  = false,
                                            uint32_t n_rs_seq   = 0,
                                            uint32_t state_size = 0) {
    llama_dsv4_logical_tensor_state result;
    result.layer_id      = layer;
    result.type          = GGML_TYPE_F32;
    result.ne0           = 1;
    result.row_size      = 4;
    const uint64_t count = recurrent ? static_cast<uint64_t>(n_rs_seq) + 1 : llama_dsv4_comp_segments_for_rows(rows);
    for (uint64_t index = 0; index < count; ++index) {
        llama_dsv4_logical_row_chunk chunk;
        chunk.row_begin = recurrent ? index * state_size : index * LLAMA_DSV4_COMP_SEGMENT_ROWS;
        chunk.row_count =
            recurrent ? state_size : std::min<uint64_t>(LLAMA_DSV4_COMP_SEGMENT_ROWS, rows - chunk.row_begin);
        chunk.bytes =
            bytes(static_cast<size_t>(chunk.row_count * result.row_size), static_cast<uint8_t>(seed + chunk.row_begin));
        chunk.checksum = row_checksum(chunk);
        result.chunks.push_back(std::move(chunk));
    }
    return result;
}

llama_dsv4_recurrent_sequence_state make_recurrent(uint32_t ratio, uint8_t seed, uint32_t n_rs_seq) {
    llama_dsv4_recurrent_sequence_state result;
    result.ratio          = ratio;
    result.state_size     = 2;
    result.n_embd_state   = 4;
    result.n_rs_seq       = n_rs_seq;
    result.state_identity = UINT64_C(0x9000000000000000) + ratio;
    result.kv.push_back(make_tensor(ratio, 0, seed, true, n_rs_seq, result.state_size));
    result.score.push_back(make_tensor(ratio, 0, seed + 31, true, n_rs_seq, result.state_size));
    return result;
}

llama_kv_iswa_logical_plane_state make_plane(uint32_t tokens, uint32_t first, uint8_t seed) {
    llama_kv_iswa_logical_plane_state result;
    for (uint32_t index = first; index < tokens; ++index) {
        result.positions.push_back(index);
    }
    result.tensor_payload = bytes(24 + result.positions.size() * 5, seed);
    result.checksum       = plane_checksum(result);
    return result;
}

llama_dsv4_logical_sequence_state make_state(uint32_t tokens, uint8_t seed = 11) {
    llama_dsv4_logical_sequence_state result;
    result.identity              = UINT64_C(0x13579bdf2468ace0);
    result.accepted_frontier     = static_cast<llama_pos>(tokens) - 1;
    result.rollback_index        = 1;
    result.active_rollback_depth = 2;
    result.n_rs_seq              = 2;
    result.raw_swa.base          = make_plane(tokens, 0, seed);
    result.raw_swa.swa           = make_plane(tokens, tokens > 16 ? tokens - 16 : 0, seed + 3);
    const uint64_t c4_rows       = tokens / 4;
    const uint64_t hca_rows      = tokens / 128;
    result.csa.row_end           = c4_rows;
    result.csa.tensors.push_back(make_tensor(3, c4_rows, seed + 7));
    result.hca.row_end = hca_rows;
    result.hca.tensors.push_back(make_tensor(5, hca_rows, seed + 13));
    result.lid.row_end = c4_rows;
    result.lid.tensors.push_back(make_tensor(3, c4_rows, seed + 19));
    result.csa_recurrent = make_recurrent(4, seed + 23, result.n_rs_seq);
    result.hca_recurrent = make_recurrent(128, seed + 29, result.n_rs_seq);
    result.lid_recurrent = make_recurrent(4, seed + 37, result.n_rs_seq);
    result.fingerprint   = llama_dsv4_logical_sequence_fingerprint(result);
    return result;
}

llama_dsv4_logical_sequence_state make_payload_boundary_state(size_t payload_size) {
    llama_dsv4_logical_sequence_state result;
    result.identity                    = UINT64_C(0x13579bdf2468ace0);
    result.accepted_frontier           = -1;
    result.active_rollback_depth       = 0;
    result.n_rs_seq                    = 0;
    result.raw_swa.base.tensor_payload = bytes(payload_size, 91);
    result.raw_swa.base.checksum       = plane_checksum(result.raw_swa.base);
    result.raw_swa.swa.checksum        = plane_checksum(result.raw_swa.swa);
    const auto recurrent               = [](uint32_t ratio) {
        llama_dsv4_recurrent_sequence_state state;
        state.ratio          = ratio;
        state.state_size     = 1;
        state.n_embd_state   = 1;
        state.state_identity = UINT64_C(0xa000000000000000) + ratio;
        return state;
    };
    result.csa_recurrent = recurrent(4);
    result.hca_recurrent = recurrent(128);
    result.lid_recurrent = recurrent(4);
    result.fingerprint   = llama_dsv4_logical_sequence_fingerprint(result);
    return result;
}

llama_dsv4_segment_identity make_identity(const llama_dsv4_logical_sequence_state & state,
                                          const std::string &                       artifact = "model-artifact-a") {
    llama_dsv4_segment_identity result;
    result.geometry_identity     = state.identity;
    result.model_artifact_digest = llama_snapshot_sha256(artifact.data(), artifact.size());
    return result;
}

llama_dsv4_segment_prefix_metadata make_prefix(uint64_t tokens) {
    llama_dsv4_segment_prefix_metadata result;
    result.token_count     = tokens;
    result.radix_depth     = static_cast<uint32_t>(std::min<uint64_t>(tokens, 64));
    const std::string text = "tokens-" + std::to_string(tokens);
    result.token_digest    = llama_snapshot_sha256(text.data(), text.size());
    return result;
}

llama_dsv4_segment_store_config make_config(const fs::path & root) {
    llama_dsv4_segment_store_config config;
    config.root_path = root.string();
    return config;
}

fs::path chunk_path(const fs::path & root, const llama_snapshot_digest & digest) {
    return root / "chunks" / (llama_snapshot_digest_hex(digest) + ".chunk");
}

fs::path manifest_path(const fs::path & root, const llama_snapshot_digest & digest) {
    return root / "manifests" / (llama_snapshot_digest_hex(digest) + ".manifest");
}

void overwrite_byte(const fs::path & path, uint64_t offset, uint8_t value) {
    const int descriptor = ::open(path.c_str(), O_RDWR | O_NOFOLLOW);
    if (descriptor < 0) {
        fail("failed to mutate test file");
    }
    const bool wrote  = ::pwrite(descriptor, &value, 1, static_cast<off_t>(offset)) == 1;
    const bool synced = ::fsync(descriptor) == 0;
    const bool closed = ::close(descriptor) == 0;
    if (!wrote || !synced || !closed) {
        fail("failed to mutate test file");
    }
}

std::vector<uint8_t> read_file(const fs::path & path) {
    const uint64_t       size = fs::file_size(path);
    std::vector<uint8_t> result(static_cast<size_t>(size));
    const int            descriptor = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW);
    expect(descriptor >= 0, "open test file");
    size_t offset = 0;
    while (offset < result.size()) {
        const ssize_t count = ::read(descriptor, result.data() + offset, result.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        expect(count > 0, "read test file");
        offset += static_cast<size_t>(count);
    }
    expect(::close(descriptor) == 0, "close test file");
    return result;
}

uint64_t read_u64(const std::vector<uint8_t> & bytes, size_t offset) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 8, "read u64 bounds");
    uint64_t result = 0;
    for (uint32_t index = 0; index < 8; ++index) {
        result |= static_cast<uint64_t>(bytes[offset + index]) << (8 * index);
    }
    return result;
}

void write_u64(std::vector<uint8_t> & bytes, size_t offset, uint64_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 8, "write u64 bounds");
    for (uint32_t index = 0; index < 8; ++index) {
        bytes[offset + index] = static_cast<uint8_t>(value >> (8 * index));
    }
}

llama_snapshot_digest write_manifest_variant(const fs::path & root, const std::vector<uint8_t> & bytes) {
    const auto     digest     = llama_snapshot_sha256(bytes.data(), bytes.size());
    const fs::path path       = manifest_path(root, digest);
    const int      descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    expect(descriptor >= 0, "create manifest variant");
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        expect(count > 0, "write manifest variant");
        offset += static_cast<size_t>(count);
    }
    const bool synced = ::fsync(descriptor) == 0;
    const bool closed = ::close(descriptor) == 0;
    expect(synced && closed, "commit manifest variant");
    return digest;
}

void test_round_trip_determinism_and_prefix_delta() {
    temporary_directory      temp;
    llama_dsv4_segment_store store(make_config(temp.path()));
    auto                     parent_state = make_state(256);
    const auto               identity     = make_identity(parent_state);
    const auto               measured     = store.measure(parent_state, identity, make_prefix(999));
    expect_status(measured.status, llama_dsv4_segment_status::ok, "measure canonical state");
    expect(measured.segment_count != 0 && measured.logical_payload_bytes != 0, "measurement omitted logical payload");

    const auto first = store.publish("first", parent_state, identity, make_prefix(256));
    expect_status(first.status, llama_dsv4_segment_status::ok, "publish parent");
    expect(first.committed && first.manifest.segments.size() == measured.segment_count,
           "parent publication accounting");
    expect_status(store.validate(first.manifest), llama_dsv4_segment_status::ok, "validate parent");
    const auto loaded = store.load(first.manifest, identity);
    expect_status(loaded.status, llama_dsv4_segment_status::ok, "load parent");
    expect(loaded.state.fingerprint == parent_state.fingerprint, "canonical state round trip fingerprint");
    expect(loaded.segments_read == first.manifest.segments.size(), "load did not read each segment exactly once");

    const auto duplicate = store.publish("duplicate", parent_state, identity, make_prefix(256));
    expect_status(duplicate.status, llama_dsv4_segment_status::ok, "publish deterministic duplicate");
    expect(duplicate.manifest.digest == first.manifest.digest && duplicate.segments_created == 0,
           "same canonical input was not deterministic/deduplicated");

    auto       child_state = make_state(260);
    const auto child       = store.publish("first", child_state, identity, make_prefix(260), &first.manifest);
    expect_status(child.status, llama_dsv4_segment_status::ok, "publish child delta");
    expect(child.manifest.has_parent && child.manifest.parent_digest == first.manifest.digest, "child lineage missing");
    expect(child.manifest.reused_segments != 0 && child.segments_reused == child.manifest.reused_segments,
           "prefix-related child did not record parent reuse");
    expect(child.manifest.segments.size() > child.segments_created, "child rewrote every content-addressed segment");
    expect_status(store.validate(child.manifest), llama_dsv4_segment_status::ok, "validate child");

    const auto other_identity = make_identity(child_state, "model-artifact-b");
    const auto isolated       = store.publish("other-model", child_state, other_identity, make_prefix(260));
    expect_status(isolated.status, llama_dsv4_segment_status::ok, "publish other artifact");
    expect(isolated.manifest.digest != child.manifest.digest &&
               isolated.manifest.segments.front().content_digest != child.manifest.segments.front().content_digest,
           "artifact identity did not domain-separate content");
    expect_status(store.open_current("first", other_identity).status, llama_dsv4_segment_status::identity_mismatch,
                  "mixed artifact open");

    auto         mixed_manifest = read_file(manifest_path(temp.path(), child.manifest.digest));
    const size_t first_fragment = static_cast<size_t>(168 + read_u64(mixed_manifest, 152) + 112);
    std::copy(isolated.manifest.segments.front().content_digest.begin(),
              isolated.manifest.segments.front().content_digest.end(), mixed_manifest.begin() + first_fragment + 16);
    const auto mixed_digest = write_manifest_variant(temp.path(), mixed_manifest);
    const auto mixed_open   = store.open_manifest(mixed_digest, identity);
    expect_status(mixed_open.status, llama_dsv4_segment_status::ok, "open mixed-model manifest fixture");
    expect_status(store.load(mixed_open.manifest, identity).status, llama_dsv4_segment_status::coverage_mismatch,
                  "mixed-model chunk accepted");

    llama_dsv4_segment_manifest missing_parent = first.manifest;
    const std::string           absent         = "absent-parent";
    missing_parent.digest                      = llama_snapshot_sha256(absent.data(), absent.size());
    expect_status(store.publish("missing-parent", child_state, identity, {}, &missing_parent).status,
                  llama_dsv4_segment_status::missing_parent, "missing parent accepted");

    auto wrong_geometry = identity;
    ++wrong_geometry.geometry_identity;
    expect_status(store.measure(child_state, wrong_geometry).status, llama_dsv4_segment_status::identity_mismatch,
                  "wrong geometry measure");
}

void test_corruption_truncation_and_missing() {
    {
        temporary_directory      temp;
        llama_dsv4_segment_store store(make_config(temp.path()));
        const auto               state    = make_state(260);
        const auto               identity = make_identity(state);
        const auto               written  = store.publish("current", state, identity);
        expect_status(written.status, llama_dsv4_segment_status::ok, "publish corruption fixture");
        const fs::path path = chunk_path(temp.path(), written.manifest.segments.front().content_digest);
        overwrite_byte(path, 32, 0xff);
        expect_status(store.validate(written.manifest), llama_dsv4_segment_status::checksum_mismatch,
                      "corrupt chunk accepted");
    }
    {
        temporary_directory      temp;
        llama_dsv4_segment_store store(make_config(temp.path()));
        const auto               state    = make_state(260);
        const auto               identity = make_identity(state);
        const auto               written  = store.publish("current", state, identity);
        expect_status(written.status, llama_dsv4_segment_status::ok, "publish missing fixture");
        const fs::path path  = chunk_path(temp.path(), written.manifest.segments.front().content_digest);
        const fs::path saved = temp.path() / "saved-chunk";
        fs::rename(path, saved);
        expect_status(store.validate(written.manifest), llama_dsv4_segment_status::missing_chunk,
                      "missing chunk accepted");
        fs::rename(saved, path);
        expect_status(store.validate(written.manifest), llama_dsv4_segment_status::ok,
                      "restored missing chunk did not validate");
    }
    {
        temporary_directory      temp;
        llama_dsv4_segment_store store(make_config(temp.path()));
        const auto               state    = make_state(260);
        const auto               identity = make_identity(state);
        const auto               written  = store.publish("current", state, identity);
        expect_status(written.status, llama_dsv4_segment_status::ok, "publish truncation fixture");
        const fs::path path = manifest_path(temp.path(), written.manifest.digest);
        const auto     size = fs::file_size(path);
        expect(size > 1 && ::truncate(path.c_str(), static_cast<off_t>(size - 1)) == 0, "truncate manifest fixture");
        expect_status(store.open_current("current", identity).status, llama_dsv4_segment_status::checksum_mismatch,
                      "truncated manifest accepted");
    }
}

void test_manifest_last_faults_and_commit_uncertainty() {
    temporary_directory      temp;
    llama_dsv4_segment_store store(make_config(temp.path()));
    const auto               first_state  = make_state(256, 17);
    const auto               second_state = make_state(260, 41);
    const auto               identity     = make_identity(first_state);
    const auto               first        = store.publish("current", first_state, identity);
    expect_status(first.status, llama_dsv4_segment_status::ok, "publish initial ref");

    llama_dsv4_segment_faults faults;
    faults.fail_before_segment = 1;
    expect_status(store.publish("current", second_state, identity, {}, &first.manifest, faults).status,
                  llama_dsv4_segment_status::injected_failure, "segment fault");
    expect(store.open_current("current", identity).manifest.digest == first.manifest.digest,
           "segment fault replaced current ref");

    faults                              = {};
    faults.fail_before_manifest_publish = true;
    expect_status(store.publish("current", second_state, identity, {}, &first.manifest, faults).status,
                  llama_dsv4_segment_status::injected_failure, "manifest fault");
    expect(store.open_current("current", identity).manifest.digest == first.manifest.digest,
           "manifest fault replaced current ref");

    faults                         = {};
    faults.fail_before_ref_publish = true;
    expect_status(store.publish("current", second_state, identity, {}, &first.manifest, faults).status,
                  llama_dsv4_segment_status::injected_failure, "ref fault");
    expect(store.open_current("current", identity).manifest.digest == first.manifest.digest,
           "pre-ref fault replaced current ref");

    faults                       = {};
    faults.fail_after_ref_rename = true;
    const auto uncertain         = store.publish("current", second_state, identity, {}, &first.manifest, faults);
    expect_status(uncertain.status, llama_dsv4_segment_status::commit_uncertain, "post-rename uncertainty");
    expect(uncertain.committed && store.open_current("current", identity).manifest.digest == uncertain.manifest.digest,
           "commit-uncertain ref was not reconcilable");
}

void test_reconcile_preserves_live_shared_chunks() {
    temporary_directory      temp;
    llama_dsv4_segment_store store(make_config(temp.path()));
    const auto               first_state  = make_state(256);
    const auto               second_state = make_state(260);
    const auto               identity     = make_identity(first_state);
    const auto               first        = store.publish("current", first_state, identity);
    const auto               second       = store.publish("current", second_state, identity, {}, &first.manifest);
    expect_status(second.status, llama_dsv4_segment_status::ok, "publish reconcile child");
    auto shared = std::find_if(second.manifest.segments.begin(), second.manifest.segments.end(),
                               [](const auto & segment) { return segment.reused_from_parent; });
    expect(shared != second.manifest.segments.end(), "reconcile fixture has no shared segment");
    const fs::path shared_path = chunk_path(temp.path(), shared->content_digest);

    const fs::path temporary  = temp.path() / "chunks" / ".tmp-orphan-123";
    const int      descriptor = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    expect(descriptor >= 0 && ::close(descriptor) == 0, "create temporary orphan");
    const auto dry = store.reconcile({}, true);
    expect_status(dry.status, llama_dsv4_segment_status::ok, "dry reconcile");
    expect(dry.manifests_removed != 0 && dry.temporary_files_removed != 0, "dry reconcile found no orphans");
    expect(fs::exists(manifest_path(temp.path(), first.manifest.digest)), "dry reconcile removed old manifest");

    const auto swept = store.reconcile();
    expect_status(swept.status, llama_dsv4_segment_status::ok, "reconcile");
    expect(!fs::exists(manifest_path(temp.path(), first.manifest.digest)) && !fs::exists(temporary),
           "reconcile retained orphan files");
    expect(fs::exists(shared_path), "reconcile removed live shared chunk");
    expect_status(store.validate(second.manifest), llama_dsv4_segment_status::ok, "reconcile damaged current manifest");
}

void test_bounds_peak_and_security() {
    static_assert(
        LLAMA_DSV4_SEGMENT_MAX_STATE_BYTES == LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES * LLAMA_DSV4_SEGMENT_MAX_CHUNKS,
        "state and segment caps diverged");
    {
        temporary_directory             temp;
        llama_dsv4_segment_store_config config = make_config(temp.path());
        config.max_chunks                      = 1;
        config.max_state_bytes                 = LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES;
        llama_dsv4_segment_store store(config);
        auto                     exact    = make_payload_boundary_state(LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES);
        const auto               identity = make_identity(exact);
        expect_status(store.measure(exact, identity).status, llama_dsv4_segment_status::ok, "exact state/chunk cap");
        auto over = make_payload_boundary_state(LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES + 1);
        expect_status(store.measure(over, make_identity(over)).status, llama_dsv4_segment_status::state_too_large,
                      "state cap plus one");
    }
    {
        temporary_directory      temp;
        llama_dsv4_segment_store store(make_config(temp.path()));
        auto                     state    = make_payload_boundary_state(LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES + 17);
        const auto               identity = make_identity(state);
        const auto               written  = store.publish("large", state, identity);
        expect_status(written.status, llama_dsv4_segment_status::ok, "publish fragmented payload");
        expect(written.manifest.segments.size() == 2, "fixed payload boundary did not produce two segments");
        expect(written.peak_codec_scratch_bytes <= LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES + 128,
               "codec scratch exceeded one capped segment");
        const auto loaded = store.load(written.manifest, identity);
        expect_status(loaded.status, llama_dsv4_segment_status::ok, "load fragmented payload");
        expect(loaded.segments_read == 2 && loaded.peak_codec_scratch_bytes <= LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES + 128,
               "fragmented load was not one bounded pass");

        // The first two fields are empty raw position/extension vectors. The
        // third field is the two-fragment raw tensor payload. Each field header
        // is 112 bytes and each fragment record is 56 bytes in format v1.
        const auto     original        = read_file(manifest_path(temp.path(), written.manifest.digest));
        const uint64_t envelope_size   = read_u64(original, 152);
        const size_t   third_field     = static_cast<size_t>(168 + envelope_size + 2 * 112);
        const size_t   first_fragment  = third_field + 112;
        const size_t   second_fragment = first_fragment + 56;
        expect(read_u64(original, first_fragment) == 0 &&
                   read_u64(original, second_fragment) == LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES,
               "unexpected manifest fragment layout");

        auto duplicate_range = original;
        write_u64(duplicate_range, second_fragment, 0);
        const auto duplicate_digest = write_manifest_variant(temp.path(), duplicate_range);
        expect_status(store.open_manifest(duplicate_digest, identity).status,
                      llama_dsv4_segment_status::coverage_mismatch, "duplicate fragment range accepted");

        auto reordered = original;
        std::swap_ranges(reordered.begin() + static_cast<std::ptrdiff_t>(first_fragment),
                         reordered.begin() + static_cast<std::ptrdiff_t>(second_fragment),
                         reordered.begin() + static_cast<std::ptrdiff_t>(second_fragment));
        const auto reordered_digest = write_manifest_variant(temp.path(), reordered);
        expect_status(store.open_manifest(reordered_digest, identity).status,
                      llama_dsv4_segment_status::coverage_mismatch, "reordered fragment ranges accepted");

        auto truncated = original;
        truncated.resize(100);
        const auto truncated_digest = write_manifest_variant(temp.path(), truncated);
        expect_status(store.open_manifest(truncated_digest, identity).status, llama_dsv4_segment_status::truncated,
                      "rehash-truncated manifest accepted");
    }
    {
        const auto               state  = make_state(4);
        auto                     config = make_config("relative-store-path");
        llama_dsv4_segment_store invalid(config);
        expect_status(invalid.measure(state, make_identity(state)).status, llama_dsv4_segment_status::path_security,
                      "relative root admitted");
        auto zero_artifact                  = make_identity(state);
        zero_artifact.model_artifact_digest = {};
        temporary_directory      valid_root;
        llama_dsv4_segment_store valid(make_config(valid_root.path()));
        expect_status(valid.measure(state, zero_artifact).status, llama_dsv4_segment_status::identity_mismatch,
                      "zero artifact admitted");

        auto inconsistent_config            = make_config(valid_root.path() / "inconsistent");
        inconsistent_config.max_chunks      = 1;
        inconsistent_config.max_state_bytes = 2 * LLAMA_DSV4_SEGMENT_PAYLOAD_BYTES;
        llama_dsv4_segment_store inconsistent(inconsistent_config);
        expect_status(inconsistent.measure(state, make_identity(state)).status,
                      llama_dsv4_segment_status::invalid_argument, "inconsistent byte/chunk caps admitted");
    }
    {
        temporary_directory      temp;
        llama_dsv4_segment_store first(make_config(temp.path()));
        llama_dsv4_segment_store second(make_config(temp.path()));
        const auto               state = make_state(4);
        expect_status(second.measure(state, make_identity(state)).status, llama_dsv4_segment_status::owner_busy,
                      "second owner admitted");
    }
    {
        temporary_directory temp;
        expect(::chmod(temp.path().c_str(), 0755) == 0, "chmod insecure root");
        llama_dsv4_segment_store insecure(make_config(temp.path()));
        const auto               state = make_state(4);
        expect_status(insecure.measure(state, make_identity(state)).status, llama_dsv4_segment_status::path_security,
                      "insecure root admitted");
        expect(::chmod(temp.path().c_str(), 0700) == 0, "restore private root mode");
    }
    {
        temporary_directory target;
        const fs::path      link = target.path().string() + "-link";
        fs::create_directory_symlink(target.path(), link);
        llama_dsv4_segment_store through_symlink(make_config(link));
        const auto               state = make_state(4);
        expect_status(through_symlink.measure(state, make_identity(state)).status,
                      llama_dsv4_segment_status::path_security, "symlink root admitted");
        fs::remove(link);
    }
    {
        temporary_directory      source_root;
        temporary_directory      hostile_root;
        const auto               state    = make_state(4);
        const auto               identity = make_identity(state);
        llama_dsv4_segment_store source(make_config(source_root.path()));
        const auto               source_write = source.publish("source", state, identity);
        expect_status(source_write.status, llama_dsv4_segment_status::ok, "publish symlink source");
        llama_dsv4_segment_store hostile(make_config(hostile_root.path()));
        const auto               digest = source_write.manifest.segments.front().content_digest;
        fs::create_symlink(chunk_path(source_root.path(), digest), chunk_path(hostile_root.path(), digest));
        expect_status(hostile.publish("hostile", state, identity).status, llama_dsv4_segment_status::path_security,
                      "symlink chunk admitted");
    }
    {
        temporary_directory      temp;
        const auto               state    = make_state(4);
        const auto               identity = make_identity(state);
        llama_dsv4_segment_store store(make_config(temp.path()));
        const fs::path           outside    = temp.path() / "outside";
        const int                descriptor = ::open(outside.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
        expect(descriptor >= 0 && ::close(descriptor) == 0, "create ref symlink target");
        fs::create_symlink(outside, temp.path() / "refs" / "hostile.ref");
        expect_status(store.publish("hostile", state, identity).status, llama_dsv4_segment_status::path_security,
                      "symlink ref admitted");
        expect(fs::file_size(outside) == 0, "ref symlink target was modified");
    }
}

}  // namespace

int main() {
    test_round_trip_determinism_and_prefix_delta();
    test_corruption_truncation_and_missing();
    test_manifest_last_faults_and_commit_uncertainty();
    test_reconcile_preserves_live_shared_chunks();
    test_bounds_peak_and_security();
    return 0;
}

#include "server-capture-resync-journal.h"
#include "server-capture-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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

void expect_status(resync_journal_status actual, resync_journal_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + resync_journal_status_name(expected) + ", got " +
             resync_journal_status_name(actual));
    }
}

void expect_capture_status(capture_store_status actual, capture_store_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message);
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        const char *   configured_base = std::getenv("TMPDIR");
        const fs::path configured =
            configured_base != nullptr && *configured_base != '\0' ? fs::path(configured_base) : fs::path("/tmp");
        std::error_code canonical_error;
        const fs::path canonical_base = fs::canonical(configured, canonical_error);
        if (canonical_error || canonical_base.empty() || !canonical_base.is_absolute()) {
            fail("canonical temporary base failed: " + canonical_error.message());
        }
        const std::string pattern = (canonical_base / "llama-resync-journal-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char * path = ::mkdtemp(writable.data());
        if (path == nullptr) {
            fail("mkdtemp failed");
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

resync_capture_commit make_commit(uint32_t record) {
    resync_capture_commit commit;
    commit.generation           = 1;
    commit.shard_sequence       = 7;
    commit.record_index         = record;
    commit.observation_sequence = 1000 + record;
    return commit;
}

resync_token_event_input accepted(uint64_t position, int32_t token) {
    return { position, token, resync_token_outcome::accepted };
}

resync_token_event_input rejected(uint64_t position, int32_t token) {
    return { position, token, resync_token_outcome::rejected };
}

resync_journal_result append(resync_journal & journal, const resync_capture_commit & commit,
                             std::initializer_list<resync_token_event_input> events) {
    std::vector<resync_token_event_input> owned(events);
    return journal.append_committed(commit, owned.empty() ? nullptr : owned.data(), owned.size());
}

resync_journal_result append(resync_journal & journal, const resync_capture_commit & commit,
                             const std::vector<resync_token_event_input> & events) {
    return journal.append_committed(commit, events.empty() ? nullptr : events.data(), events.size());
}

resync_journal_config memory_config() {
    return {};
}

resync_journal_config persistent_config(const fs::path & root) {
    resync_journal_config config;
    config.persist               = true;
    config.root_path             = root.string();
    config.allow_sensitive_tokens = true;
    return config;
}

std::array<uint8_t, 4> token_bytes(int32_t token) {
    const uint32_t value = static_cast<uint32_t>(token);
    return {
        static_cast<uint8_t>(value),
        static_cast<uint8_t>(value >> 8U),
        static_cast<uint8_t>(value >> 16U),
        static_cast<uint8_t>(value >> 24U),
    };
}

bool contains_bytes(const std::vector<uint8_t> & bytes, const std::array<uint8_t, 4> & needle) {
    return std::search(bytes.begin(), bytes.end(), needle.begin(), needle.end()) != bytes.end();
}

std::vector<uint8_t> read_file(const fs::path & path) {
    std::ifstream input(path, std::ios::binary);
    expect(input.good(), "open journal file");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

void write_file(const fs::path & path, const std::vector<uint8_t> & bytes) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    expect(descriptor >= 0, "open journal fixture");
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t count = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        expect(count > 0, "write journal fixture");
        offset += static_cast<size_t>(count);
    }
    expect(::fsync(descriptor) == 0 && ::close(descriptor) == 0, "close journal fixture");
}

void mutate_byte(const fs::path & path, uint64_t offset) {
    const int descriptor = ::open(path.c_str(), O_RDWR);
    expect(descriptor >= 0, "open journal mutation");
    uint8_t value = 0;
    expect(::pread(descriptor, &value, sizeof(value), static_cast<off_t>(offset)) == 1, "read journal mutation");
    value ^= 0x80U;
    expect(::pwrite(descriptor, &value, sizeof(value), static_cast<off_t>(offset)) == 1, "write journal mutation");
    expect(::fsync(descriptor) == 0 && ::close(descriptor) == 0, "close journal mutation");
}

void test_boundaries_and_patterns() {
    resync_journal journal(memory_config());
    resync_journal_snapshot snapshot;
    expect_status(journal.inspect(snapshot).status, resync_journal_status::ok, "empty inspect");
    expect(snapshot.event_count == 0 && snapshot.replay_count == 0 && snapshot.next_event_sequence == 1 &&
               snapshot.next_replay_sequence == 1,
           "empty state");

    const resync_capture_commit empty_commit = make_commit(0);
    expect_status(append(journal, empty_commit, {}).status, resync_journal_status::ok, "zero-event commit");
    expect_status(append(journal, empty_commit, {}).status, resync_journal_status::duplicate_commit,
                  "zero-event idempotence");

    const resync_capture_commit one_commit = make_commit(1);
    expect_status(append(journal, one_commit, { accepted(0, 101) }).status, resync_journal_status::ok,
                  "one accepted token");
    expect_status(append(journal, one_commit, { accepted(0, 101) }).status, resync_journal_status::duplicate_commit,
                  "accepted idempotence");
    expect_status(append(journal, one_commit, { accepted(0, 999) }).status, resync_journal_status::commit_conflict,
                  "same commit conflict");

    const resync_capture_commit partial_commit = make_commit(2);
    expect_status(append(journal, partial_commit, { accepted(1, 102), rejected(2, 9999) }).status,
                  resync_journal_status::ok, "partial acceptance");
    expect_status(journal.inspect(snapshot).status, resync_journal_status::ok, "partial inspect");
    expect(snapshot.event_count == 2 && snapshot.replay_count == 2 && snapshot.replay_tokens[0].token_id == 101 &&
               snapshot.replay_tokens[1].token_id == 102 && snapshot.has_rejection_boundary &&
               snapshot.rejection_boundary.token_id == 9999 && snapshot.rejection_boundary.replay_sequence == 0,
           "partial replay excludes rejection");

    const resync_capture_commit reject_commit = make_commit(3);
    expect_status(append(journal, reject_commit, { rejected(2, 10000) }).status, resync_journal_status::ok,
                  "reject-only pattern");
    expect_status(journal.inspect(snapshot).status, resync_journal_status::ok, "reject inspect");
    expect(snapshot.event_count == 2 && snapshot.replay_count == 2 && snapshot.rejection_boundary.token_id == 10000,
           "reject boundary does not consume replay slot");

    const resync_capture_commit accepted_all_commit = make_commit(4);
    expect_status(append(journal, accepted_all_commit, { accepted(2, 103), accepted(3, 104) }).status,
                  resync_journal_status::ok, "accepted-all pattern");
    expect_status(journal.inspect(snapshot).status, resync_journal_status::ok, "accepted-all inspect");
    expect(snapshot.event_count == 4 && snapshot.replay_count == 4 && snapshot.replay_tokens[2].token_id == 103 &&
               snapshot.replay_tokens[3].token_id == 104 && snapshot.next_token_position == 4,
           "accepted-all replay suffix");

    const resync_capture_commit invalid_commit = make_commit(5);
    expect_status(append(journal, invalid_commit, { rejected(4, 500), accepted(4, 501) }).status,
                  resync_journal_status::invalid_event, "rejection must terminate batch");
    expect_status(append(journal, invalid_commit, { accepted(5, 501) }).status, resync_journal_status::token_position_gap,
                  "position gap rejected");
}

void test_commit_monotonicity() {
    {
        resync_journal journal(memory_config());
        expect_status(append(journal, make_commit(1), { accepted(0, 11) }).status,
                      resync_journal_status::ok, "monotonic retained A");
        expect_status(append(journal, make_commit(2), { accepted(1, 22) }).status,
                      resync_journal_status::ok, "monotonic retained B");
        expect_status(append(journal, make_commit(1), { accepted(0, 11) }).status,
                      resync_journal_status::out_of_order_commit,
                      "older retained commit is not an idempotent duplicate");
        expect_status(append(journal, make_commit(2), { accepted(1, 22) }).status,
                      resync_journal_status::duplicate_commit,
                      "current retained commit remains idempotent");
    }
    {
        resync_journal journal(memory_config());
        expect_status(append(journal, make_commit(1), { accepted(0, 11) }).status,
                      resync_journal_status::ok, "monotonic metadata A");
        expect_status(append(journal, make_commit(2), {}).status,
                      resync_journal_status::ok, "monotonic metadata B");
        expect_status(append(journal, make_commit(1), { accepted(0, 11) }).status,
                      resync_journal_status::out_of_order_commit,
                      "older commit remains out of order after metadata watermark");
        expect_status(append(journal, make_commit(2), {}).status,
                      resync_journal_status::duplicate_commit,
                      "current metadata watermark remains idempotent");
    }
}

void test_capacity_wrap_and_watermarks() {
    resync_journal boundary_journal(memory_config());
    std::vector<resync_token_event_input> boundary_batch;
    boundary_batch.reserve(127);
    for (size_t index = 0; index < 127; ++index) {
        boundary_batch.push_back(accepted(index, static_cast<int32_t>(index)));
    }
    expect_status(append(boundary_journal, make_commit(1), boundary_batch).status, resync_journal_status::ok,
                  "127 accepted boundary");
    resync_journal_snapshot boundary_snapshot;
    expect_status(boundary_journal.inspect(boundary_snapshot).status, resync_journal_status::ok,
                  "127 accepted inspect");
    expect(boundary_snapshot.event_count == 127, "127 accepted slots retained");
    expect_status(append(boundary_journal, make_commit(2), { accepted(127, 127) }).status,
                  resync_journal_status::ok, "128th accepted boundary");
    expect_status(boundary_journal.inspect(boundary_snapshot).status, resync_journal_status::ok,
                  "128th accepted inspect");
    expect(boundary_snapshot.event_count == RESYNC_JOURNAL_MAX_EVENTS, "128 accepted slots reached");

    resync_journal journal(memory_config());
    std::vector<resync_token_event_input> batch;
    batch.reserve(RESYNC_JOURNAL_MAX_EVENTS);
    for (size_t index = 0; index < RESYNC_JOURNAL_MAX_EVENTS; ++index) {
        batch.push_back(accepted(index, static_cast<int32_t>(index)));
    }
    batch.push_back(rejected(RESYNC_JOURNAL_MAX_EVENTS, 9999));
    expect_status(append(journal, make_commit(1), batch).status, resync_journal_status::ok, "128 accepted boundary");

    for (uint32_t cycle = 0; cycle < 4; ++cycle) {
        const uint64_t position = RESYNC_JOURNAL_MAX_EVENTS + cycle;
        expect_status(append(journal, make_commit(2 + cycle),
                             { accepted(position, static_cast<int32_t>(128 + cycle)),
                               rejected(position + 1, static_cast<int32_t>(10000 + cycle)) })
                          .status,
                      resync_journal_status::ok, "frequent reject cycle");
    }

    resync_journal_snapshot snapshot;
    expect_status(journal.inspect(snapshot).status, resync_journal_status::ok, "wrapped inspect");
    expect(snapshot.event_count == RESYNC_JOURNAL_MAX_EVENTS && snapshot.replay_count == RESYNC_JOURNAL_MAX_EVENTS,
           "rejects do not reduce 128-token replay capacity");
    expect(snapshot.replay_tokens[0].token_id == 4 && snapshot.replay_tokens.back().token_id == 131,
           "accepted suffix after ring wrap");
    for (size_t index = 1; index < snapshot.replay_count; ++index) {
        expect(snapshot.replay_tokens[index].replay_sequence == snapshot.replay_tokens[index - 1].replay_sequence + 1,
               "replay sequence continuity");
        expect(snapshot.replay_tokens[index].token_position == snapshot.replay_tokens[index - 1].token_position + 1,
               "token position continuity");
        expect(snapshot.events[index].event_sequence > snapshot.events[index - 1].event_sequence,
               "physical event sequence monotonicity");
    }
    expect(snapshot.has_rejection_boundary && snapshot.rejection_boundary.token_id == 10003,
           "latest rejection boundary retained");

    // The first commit was evicted from the accepted ring, but the monotonic
    // commit watermark still prevents replaying it as a new observation.
    expect_status(append(journal, make_commit(1), batch).status, resync_journal_status::out_of_order_commit,
                  "evicted older commit rejected");
}

void test_overflow_and_invalid_commits() {
    resync_journal_config config = memory_config();
    config.initial_token_position = UINT64_MAX;
    resync_journal journal(config);
    const resync_capture_commit first = make_commit(1);
    expect_status(append(journal, first, { rejected(UINT64_MAX, 77) }).status, resync_journal_status::ok,
                  "max-position rejection diagnostic");
    expect_status(append(journal, make_commit(2), { accepted(UINT64_MAX, 78) }).status,
                  resync_journal_status::token_position_overflow, "max-position accepted overflow");
    expect_status(append(journal, make_commit(3), { accepted(UINT64_MAX - 1, 79) }).status,
                  resync_journal_status::token_position_gap, "position underflow gap");

    std::vector<resync_token_event_input> too_many(RESYNC_JOURNAL_MAX_EVENTS + 1, accepted(UINT64_MAX, 1));
    expect_status(append(journal, make_commit(4), too_many).status, resync_journal_status::too_many_events,
                  "physical batch overflow");

    resync_capture_commit invalid;
    expect_status(append(journal, invalid, {}).status, resync_journal_status::invalid_commit,
                  "zero capture identity rejected");
}

void test_persistence_opt_in() {
    temporary_directory temp;
    resync_journal_config config;
    config.persist   = true;
    config.root_path = (temp.path() / "missing-opt-in").string();
    bool threw = false;
    try {
        resync_journal rejected_config(config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "persistent journal without sensitive-token opt-in rejected");

    config = persistent_config(temp.path() / "shared-root");
    config.require_private_root = false;
    threw = false;
    try {
        resync_journal rejected_shared_root(config);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "persistent journal without private-root enforcement rejected");

    const fs::path real_root = temp.path() / "real-root";
    const fs::path link_root = temp.path() / "link-root";
    expect(fs::create_directory(real_root), "create private root fixture");
    std::error_code symlink_error;
    fs::create_directory_symlink(real_root, link_root, symlink_error);
    if (!symlink_error) {
        config = persistent_config(link_root);
        threw  = false;
        try {
            resync_journal rejected_symlink(config);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        expect(threw, "symlinked journal root rejected");
    }
}

void make_persisted_fixture(const fs::path & root) {
    resync_journal journal(persistent_config(root));
    expect_status(append(journal, make_commit(1), { accepted(0, 12345), rejected(1, 54321) }).status,
                  resync_journal_status::ok, "persist fixture append");
}

void test_restart_and_privacy() {
    temporary_directory temp;
    const fs::path root = temp.path() / "journal-root";
    make_persisted_fixture(root);
    const fs::path path = root / "resync.journal";
    expect(fs::exists(path), "journal published");
    struct stat status = {};
    expect(::stat(path.c_str(), &status) == 0 && (status.st_mode & 07777) == (S_IRUSR | S_IWUSR) &&
               static_cast<uint64_t>(status.st_size) <= RESYNC_JOURNAL_MAX_FILE_BYTES,
           "journal file private");
    struct stat root_status = {};
    expect(::stat(root.c_str(), &root_status) == 0 && (root_status.st_mode & 07777) == S_IRWXU,
           "journal root exact private mode");
    const std::vector<uint8_t> bytes = read_file(path);
    const std::array<uint8_t, 4> token_bytes = { 0x39, 0x30, 0, 0 };
    expect(std::search(bytes.begin(), bytes.end(), token_bytes.begin(), token_bytes.end()) != bytes.end(),
           "exact token ID persisted only in explicit journal");

    // An uncommitted temp image is ignored after a restart; the last complete
    // rename remains the recovery source.
    write_file(root / ".resync.journal.tmp", { 0, 1, 2 });
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok, "restart recovery");
        expect(snapshot.event_count == 1 && snapshot.replay_tokens[0].token_id == 12345 &&
                   snapshot.rejection_boundary.token_id == 54321,
               "restart exact suffix and boundary");
        expect_status(append(restarted, make_commit(2), {}).status, resync_journal_status::ok,
                      "metadata-only committed observation");
    }
    {
        resync_journal restarted_metadata(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted_metadata.inspect(snapshot).status, resync_journal_status::ok,
                      "metadata-only restart recovery");
        expect(snapshot.event_count == 1 && snapshot.last_commit.record_index == 2,
               "metadata-only watermark survives restart");
        expect_status(append(restarted_metadata, make_commit(1), { accepted(1, 999) }).status,
                      resync_journal_status::out_of_order_commit,
                      "restart watermark rejects older commit");
    }
}

void test_reject_only_restart() {
    temporary_directory temp;
    const fs::path root = temp.path() / "reject-only";
    {
        resync_journal journal(persistent_config(root));
        expect_status(append(journal, make_commit(1), { rejected(0, 7001) }).status,
                      resync_journal_status::ok, "persist reject-only commit");
    }
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok,
                      "reject-only restart recovery");
        expect(snapshot.event_count == 0 && snapshot.replay_count == 0 && snapshot.next_event_sequence == 2 &&
                   snapshot.has_rejection_boundary && snapshot.rejection_boundary.token_id == 7001,
               "reject-only watermark and boundary survive restart");
        expect_status(append(restarted, make_commit(2), {}).status, resync_journal_status::ok,
                      "metadata-after-reject commit");
    }
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok,
                      "metadata-after-reject restart recovery");
        expect(snapshot.event_count == 0 && snapshot.has_rejection_boundary &&
                   snapshot.last_commit.record_index == 2,
               "metadata-after-reject watermark survives restart");
    }
}

void test_corruption_fail_closed() {
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "checksum";
        make_persisted_fixture(root);
        mutate_byte(root / "resync.journal", RESYNC_JOURNAL_HEADER_BYTES + 4);
        resync_journal corrupt(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(corrupt.inspect(snapshot).status, resync_journal_status::checksum_mismatch,
                      "payload checksum mismatch");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "truncated";
        make_persisted_fixture(root);
        const fs::path path = root / "resync.journal";
        const auto bytes = read_file(path);
        expect(::truncate(path.c_str(), static_cast<off_t>(bytes.size() - 1)) == 0, "truncate journal");
        resync_journal corrupt(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(corrupt.inspect(snapshot).status, resync_journal_status::truncated,
                      "truncated journal rejected");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "empty-file";
        fs::create_directories(root);
        write_file(root / "resync.journal", {});
        resync_journal corrupt(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(corrupt.inspect(snapshot).status, resync_journal_status::truncated,
                      "empty journal rejected");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "malformed";
        make_persisted_fixture(root);
        const fs::path path = root / "resync.journal";
        auto bytes = read_file(path);
        bytes[0] ^= 0x7fU;
        write_file(path, bytes);
        resync_journal corrupt(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(corrupt.inspect(snapshot).status, resync_journal_status::malformed,
                      "malformed magic rejected");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "trailing";
        make_persisted_fixture(root);
        const fs::path path = root / "resync.journal";
        auto bytes = read_file(path);
        bytes.push_back(0);
        write_file(path, bytes);
        resync_journal corrupt(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(corrupt.inspect(snapshot).status, resync_journal_status::trailing_data,
                      "trailing bytes rejected");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "future-version";
        make_persisted_fixture(root);
        const fs::path path = root / "resync.journal";
        auto bytes = read_file(path);
        bytes[8] = 2;
        write_file(path, bytes);
        resync_journal future(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(future.inspect(snapshot).status, resync_journal_status::unsupported_version,
                      "future journal version rejected explicitly");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "mode";
        make_persisted_fixture(root);
        expect(::chmod((root / "resync.journal").c_str(), S_IRUSR | S_IWUSR | S_IXUSR) == 0,
               "chmod journal fixture");
        resync_journal wrong_mode(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(wrong_mode.inspect(snapshot).status, resync_journal_status::path_security,
                      "owner-executable journal mode rejected");
    }
    {
        temporary_directory temp;
        const fs::path root = temp.path() / "symlink-file";
        fs::create_directories(root);
        const fs::path target = temp.path() / "outside-journal";
        write_file(target, { 1, 2, 3 });
        expect(::symlink(target.c_str(), (root / "resync.journal").c_str()) == 0,
               "create journal-file symlink");
        resync_journal linked(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(linked.inspect(snapshot).status, resync_journal_status::path_security,
                      "journal-file symlink rejected");
    }
}

void test_publication_faults() {
    temporary_directory temp;
    const fs::path root = temp.path() / "faults";
    {
        resync_journal journal(persistent_config(root));
        expect_status(append(journal, make_commit(1), { accepted(0, 1) }).status,
                      resync_journal_status::ok, "fault baseline commit");
    }

    {
        resync_journal_config config = persistent_config(root);
        config.faults.fail_file_fsync = true;
        resync_journal journal(config);
        expect_status(append(journal, make_commit(2), { accepted(1, 2) }).status,
                      resync_journal_status::io_error, "file fsync failure");
        resync_journal_snapshot snapshot;
        expect_status(journal.inspect(snapshot).status, resync_journal_status::ok,
                      "file fsync preserves old state");
        expect(snapshot.event_count == 1 && snapshot.replay_tokens[0].token_id == 1,
               "file fsync leaves previous image in memory");
    }

    {
        resync_journal_config config = persistent_config(root);
        config.faults.crash_before_rename = true;
        resync_journal journal(config);
        const resync_journal_result result = append(journal, make_commit(2), { accepted(1, 2) });
        expect_status(result.status, resync_journal_status::commit_uncertain, "pre-rename crash seam");
        expect(!result.committed, "pre-rename crash is not reported committed");
        resync_journal_snapshot snapshot;
        expect_status(journal.inspect(snapshot).status, resync_journal_status::ok,
                      "pre-rename crash preserves old state");
        expect(snapshot.event_count == 1 && fs::exists(root / ".resync.journal.tmp"),
               "pre-rename crash leaves ignored orphan temp");
    }
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok,
                      "pre-rename recovery");
        expect(snapshot.event_count == 1 && snapshot.replay_tokens[0].token_id == 1,
               "pre-rename recovery uses previous image");
    }

    {
        resync_journal_config config = persistent_config(root);
        config.faults.fail_directory_fsync = true;
        resync_journal journal(config);
        const resync_journal_result result = append(journal, make_commit(2), { accepted(1, 2) });
        expect_status(result.status, resync_journal_status::commit_uncertain, "directory fsync failure");
        expect(result.committed, "directory fsync failure reports publication uncertainty");
    }
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok,
                      "directory-fsync recovery");
        expect(snapshot.event_count == 2 && snapshot.replay_tokens[snapshot.replay_count - 1].token_id == 2,
               "directory-fsync image remains complete");
    }

    {
        resync_journal_config config = persistent_config(root);
        config.faults.crash_after_rename = true;
        resync_journal journal(config);
        const resync_journal_result result = append(journal, make_commit(3), { accepted(2, 3) });
        expect_status(result.status, resync_journal_status::commit_uncertain, "post-rename crash seam");
        expect(result.committed, "post-rename crash reports publication uncertainty");
    }
    {
        resync_journal restarted(persistent_config(root));
        resync_journal_snapshot snapshot;
        expect_status(restarted.inspect(snapshot).status, resync_journal_status::ok,
                      "post-rename recovery");
        expect(snapshot.event_count == 3 && snapshot.replay_tokens[snapshot.replay_count - 1].token_id == 3,
               "post-rename image remains checksum-valid");
    }
}

void test_owner_lock_and_process_recovery() {
    temporary_directory temp;
    const fs::path root = temp.path() / "owner-lock";
    int ready[2] = {};
    int release[2] = {};
    expect(::pipe(ready) == 0 && ::pipe(release) == 0, "create owner-lock pipes");
    const pid_t child = ::fork();
    expect(child >= 0, "fork owner-lock child");
    if (child == 0) {
        ::close(ready[0]);
        ::close(release[1]);
        char marker = '0';
        try {
            resync_journal holder(persistent_config(root));
            marker = '1';
            if (::write(ready[1], &marker, sizeof(marker)) != static_cast<ssize_t>(sizeof(marker))) {
                _exit(2);
            }
            char ignored = 0;
            if (::read(release[0], &ignored, sizeof(ignored)) != static_cast<ssize_t>(sizeof(ignored))) {
                _exit(2);
            }
            _exit(0);
        } catch (...) {
            if (::write(ready[1], &marker, sizeof(marker)) != static_cast<ssize_t>(sizeof(marker))) {
                _exit(2);
            }
            _exit(1);
        }
    }
    ::close(ready[1]);
    ::close(release[0]);
    char marker = 0;
    expect(::read(ready[0], &marker, sizeof(marker)) == 1 && marker == '1',
           "child acquired owner lock");
    ::close(ready[0]);

    bool threw = false;
    try {
        resync_journal rejected_owner(persistent_config(root));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expect(threw, "second process owner rejected");

    char release_marker = 'x';
    expect(::write(release[1], &release_marker, sizeof(release_marker)) == 1, "release child owner lock");
    ::close(release[1]);
    int child_status = 0;
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
           "owner-lock child exited cleanly");

    {
        resync_journal first(persistent_config(root));
        threw = false;
        try {
            resync_journal second(persistent_config(root));
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        expect(threw, "same-process owner rejected");
    }
    resync_journal after_process_death(persistent_config(root));
    resync_journal_snapshot snapshot;
    expect_status(after_process_death.inspect(snapshot).status, resync_journal_status::ok,
                  "owner lock released after process death");
}

void test_capture_store_does_not_receive_journal_tokens() {
    temporary_directory temp;
    const fs::path root = temp.path() / "capture-separation";
    constexpr int32_t journal_token_a = 0x13572468;
    constexpr int32_t journal_token_b = -0x24681357;
    {
        resync_journal journal(persistent_config(root));
        expect_status(append(journal, make_commit(1),
                             { accepted(0, journal_token_a), accepted(1, journal_token_b) })
                          .status,
                      resync_journal_status::ok, "capture separation journal append");

        capture_store_config capture_config;
        capture_config.root_path            = root.string();
        capture_config.capture_mode        = mode::compact;
        capture_config.ring_capacity        = 4;
        capture_config.max_shard_records    = 1;
        capture_config.max_shard_bytes      = 4096;
        capture_config.max_retained_shards  = 4;
        capture_config.max_retained_records = 4;
        capture_config.max_retained_bytes   = 16384;
        capture_config.max_manifest_bytes   = 65536;
        capture_store store(capture_config);
        cycle_observation observation;
        observation.request_id             = 0x1122334455667788ULL;
        observation.committed_position     = 7;
        observation.scheduler_epoch       = 9;
        observation.monotonic_ns           = 11;
        observation.cycle_sequence         = 1;
        observation.proposal_token_ids     = { { 101, 102, 103, 104, 105 } };
        observation.proposal_count         = 5;
        observation.accepted_prefix_length = 3;
        observation.first_rejection        = 3;
        expect(store.try_enqueue(observation), "capture separation enqueue");
        expect_capture_status(store.flush().status, capture_store_status::ok, "capture separation flush");
        expect_capture_status(store.shutdown(true).status, capture_store_status::ok, "capture separation shutdown");
    }

    const auto journal_a = token_bytes(journal_token_a);
    const auto journal_b = token_bytes(journal_token_b);
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file() || entry.path().filename() == "resync.journal") {
            continue;
        }
        const std::vector<uint8_t> bytes = read_file(entry.path());
        expect(!contains_bytes(bytes, journal_a) && !contains_bytes(bytes, journal_b),
               "journal token leaked into capture-store file");
    }
}

void test_concurrent_inspection_and_lifetime() {
    resync_journal journal(memory_config());
    std::atomic<bool> running{ true };
    std::atomic<bool> reader_ok{ true };
    std::thread reader([&]() {
        while (running.load(std::memory_order_acquire)) {
            resync_journal_snapshot snapshot;
            if (journal.inspect(snapshot).status != resync_journal_status::ok ||
                snapshot.event_count > RESYNC_JOURNAL_MAX_EVENTS || snapshot.replay_count != snapshot.event_count) {
                reader_ok.store(false, std::memory_order_release);
                break;
            }
        }
    });
    for (uint32_t index = 0; index < 64; ++index) {
        expect_status(append(journal, make_commit(index + 1), { accepted(index, static_cast<int32_t>(index)) }).status,
                      resync_journal_status::ok, "concurrent writer append");
    }
    running.store(false, std::memory_order_release);
    reader.join();
    expect(reader_ok.load(std::memory_order_acquire), "concurrent inspect remained coherent");
}

}  // namespace

int main() {
    try {
        test_boundaries_and_patterns();
        test_commit_monotonicity();
        test_capacity_wrap_and_watermarks();
        test_overflow_and_invalid_commits();
        test_persistence_opt_in();
        test_restart_and_privacy();
        test_reject_only_restart();
        test_corruption_fail_closed();
        test_publication_faults();
        test_owner_lock_and_process_recovery();
        test_capture_store_does_not_receive_journal_tokens();
        test_concurrent_inspection_and_lifetime();
    } catch (const std::exception & error) {
        std::cerr << "test-server-capture-resync-journal: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-server-capture-resync-journal: all checks passed\n";
    return 0;
}

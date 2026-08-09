#include "server-prompt-cache-io.h"

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

void expect_status(server_prompt_cache_io_status actual,
                   server_prompt_cache_io_status expected,
                   const std::string &           message) {
    if (actual != expected) {
        fail(message + ": expected " + server_prompt_cache_io_status_name(expected) + ", got " +
             server_prompt_cache_io_status_name(actual));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        char   pattern[] = "/tmp/server-prompt-cache-io-XXXXXX";
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

    std::string path(const std::string & name = {}) const {
        return name.empty() ? directory.string() : (directory / name).string();
    }

  private:
    fs::path directory;
};

class callback_reentry_probe {
  public:
    callback_reentry_probe(server_prompt_cache_io & io, std::atomic<bool> & destroyed) : io(io), destroyed(destroyed) {}

    ~callback_reentry_probe() {
        (void) io.stats();
        destroyed = true;
    }

  private:
    server_prompt_cache_io & io;
    std::atomic<bool> &      destroyed;
};

server_prompt_cache_io_payload make_payload(size_t size, uint8_t seed) {
    auto mutable_payload = std::make_shared<std::vector<uint8_t>>(size);
    for (size_t index = 0; index < size; ++index) {
        (*mutable_payload)[index] = static_cast<uint8_t>(seed + index * 37 + index / 7);
    }
    return mutable_payload;
}

void write_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    expect(stream.is_open(), "open test file");
    stream.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    expect(bool(stream), "write test file");
}

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    expect(stream.is_open(), "open persisted file");
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

server_prompt_cache_io_config make_config() {
    server_prompt_cache_io_config config;
    config.max_jobs          = 8;
    config.max_completions   = 32;
    config.max_pending_bytes = 1u << 20;
    config.max_disk_bytes    = 1u << 22;
    config.error_cooldown_ms = 1000;
    return config;
}

server_prompt_cache_io_write make_write(uint64_t                       entry_id,
                                        uint64_t                       generation,
                                        const std::string &            path,
                                        server_prompt_cache_io_payload payload) {
    server_prompt_cache_io_write write;
    write.entry_id   = entry_id;
    write.generation = generation;
    write.final_path = path;
    write.payload    = std::move(payload);
    return write;
}

template <typename Predicate> void wait_until(Predicate predicate, const std::string & message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail(message);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

std::map<uint64_t, server_prompt_cache_io_completion> wait_for_jobs(server_prompt_cache_io &      io,
                                                                    const std::vector<uint64_t> & job_ids) {
    std::map<uint64_t, server_prompt_cache_io_completion> result;
    wait_until(
        [&] {
            for (auto & completion : io.poll_completions()) {
                result.emplace(completion.job_id, std::move(completion));
            }
            for (uint64_t job_id : job_ids) {
                if (result.find(job_id) == result.end()) {
                    return false;
                }
            }
            return true;
        },
        "completion wait timed out");
    return result;
}

void expect_no_private_temps(const std::string & directory) {
    for (const fs::directory_entry & entry : fs::directory_iterator(directory)) {
        const std::string name = entry.path().filename().string();
        expect(name.rfind(".pcache-io-", 0) != 0, "private temp file leaked: " + name);
    }
}

void test_pending_visibility_and_durable_release() {
    temporary_directory    temp;
    server_prompt_cache_io io(make_config());

    auto                                      payload  = make_payload(513, 11);
    const std::vector<uint8_t>                expected = *payload;
    std::weak_ptr<const std::vector<uint8_t>> weak     = payload;
    auto                                      write    = make_write(1, 1, temp.path("one.lcpc"), payload);
    write.verify_checksum                              = true;
    write.expected_checksum     = server_prompt_cache_io_checksum(payload->data(), payload->size());
    write.faults.max_write_size = 7;
    write.faults.write_eintrs   = 5;
    write.faults.pause          = server_prompt_cache_io_test_pause::before_write;

    auto submitted = io.submit(std::move(write));
    expect_status(submitted.status, server_prompt_cache_io_status::ok, "submit pending write");
    expect(submitted.payload == payload, "submit did not return immutable payload");
    expect(io.test_wait_for_pause(submitted.job_id, server_prompt_cache_io_test_pause::before_write, 5000),
           "write did not reach pending barrier");
    auto pending = io.pending_payload(1, 1);
    expect(pending == payload && *pending == expected, "pending payload not readable");
    const auto snapshot = io.inspect(1, 1);
    expect(snapshot.lifecycle == server_prompt_cache_io_lifecycle::writing && snapshot.started_ns != 0,
           "pending lifecycle/timestamp");
    const auto pending_stats = io.stats();
    expect(pending_stats.ram_held_bytes == expected.size() && pending_stats.disk_reserved_bytes == expected.size(),
           "pending byte reservation accounting");

    pending.reset();
    submitted.payload.reset();
    payload.reset();
    expect(!weak.expired(), "engine released payload before durability");
    io.test_release_pause(submitted.job_id);
    const auto   completions = wait_for_jobs(io, { submitted.job_id });
    const auto & completion  = completions.at(submitted.job_id);
    expect_status(completion.status, server_prompt_cache_io_status::ok, "durable completion");
    expect(completion.lifecycle == server_prompt_cache_io_lifecycle::durable && completion.publication_fenced &&
               completion.checksum != 0,
           "durable lifecycle metadata");
    expect(weak.expired(), "engine retained payload after durable publication");
    expect(read_file(temp.path("one.lcpc")) == expected, "short-write/EINTR payload mismatch");
    const auto stats = io.stats();
    expect(stats.ram_held_bytes == 0 && stats.disk_reserved_bytes == 0 &&
               stats.disk_committed_bytes == expected.size() && stats.durable_writes == 1,
           "durable release/accounting order");
    expect(io.poll_completions().empty(), "completion delivered more than once");
    expect_no_private_temps(temp.path());
}

void test_job_byte_and_disk_admission() {
    temporary_directory temp;
    auto                config = make_config();
    config.max_jobs            = 2;
    config.max_completions     = 4;
    config.max_pending_bytes   = 90;
    config.max_disk_bytes      = 200;
    server_prompt_cache_io io(config);

    auto first         = make_write(1, 1, temp.path("first.lcpc"), make_payload(60, 1));
    first.faults.pause = server_prompt_cache_io_test_pause::before_write;
    auto first_result  = io.submit(std::move(first));
    expect_status(first_result.status, server_prompt_cache_io_status::ok, "first bounded submit");
    expect(io.test_wait_for_pause(first_result.job_id, server_prompt_cache_io_test_pause::before_write, 5000),
           "first bounded pause");

    auto too_many_bytes = io.submit(make_write(2, 1, temp.path("bytes.lcpc"), make_payload(31, 2)));
    expect_status(too_many_bytes.status, server_prompt_cache_io_status::byte_limit, "authoritative byte limit");
    auto second = io.submit(make_write(2, 1, temp.path("second.lcpc"), make_payload(30, 3)));
    expect_status(second.status, server_prompt_cache_io_status::ok, "second bounded submit");
    auto third = io.submit(make_write(3, 1, temp.path("third.lcpc"), make_payload(1, 4)));
    expect_status(third.status, server_prompt_cache_io_status::job_limit, "bounded job limit");
    expect(io.stats().ram_held_bytes == 90, "rejected jobs changed RAM accounting");
    io.test_release_pause(first_result.job_id);
    wait_for_jobs(io, { first_result.job_id, second.job_id });

    auto disk_config               = make_config();
    disk_config.max_disk_bytes     = 100;
    disk_config.initial_disk_bytes = 20;
    server_prompt_cache_io disk_io(disk_config);
    auto disk_rejected = disk_io.submit(make_write(9, 1, temp.path("disk.lcpc"), make_payload(81, 5)));
    expect_status(disk_rejected.status, server_prompt_cache_io_status::disk_limit, "disk reservation limit");
    expect(disk_io.stats().disk_reserved_bytes == 0, "rejected disk reservation leaked");
}

void test_crossed_fence_generation_and_exact_cleanup() {
    temporary_directory    temp;
    server_prompt_cache_io io(make_config());
    const std::string      path         = temp.path("shared.lcpc");
    const auto             first_bytes  = make_payload(257, 21);
    const auto             second_bytes = make_payload(333, 41);

    auto first         = make_write(77, 1, path, first_bytes);
    first.faults.pause = server_prompt_cache_io_test_pause::after_publication_fence;
    auto first_result  = io.submit(std::move(first));
    expect_status(first_result.status, server_prompt_cache_io_status::ok, "submit fenced generation");
    expect(
        io.test_wait_for_pause(first_result.job_id, server_prompt_cache_io_test_pause::after_publication_fence, 5000),
        "generation one did not cross fence");

    auto second_result = io.submit(make_write(77, 2, path, second_bytes));
    expect_status(second_result.status, server_prompt_cache_io_status::ok, "submit superseding generation");
    io.test_release_pause(first_result.job_id);
    const auto completions = wait_for_jobs(io, { first_result.job_id, second_result.job_id });
    expect(completions.at(first_result.job_id).lifecycle == server_prompt_cache_io_lifecycle::retired &&
               completions.at(first_result.job_id).publication_fenced,
           "post-fence stale generation was not retired");
    expect_status(completions.at(second_result.job_id).status, server_prompt_cache_io_status::ok,
                  "new generation durable completion");
    expect(read_file(path) == *second_bytes, "stale generation damaged newer path");

    const auto stale_cleanup = io.cleanup_exact(77, 1, path);
    expect_status(stale_cleanup.status, server_prompt_cache_io_status::stale_generation,
                  "stale exact-path cleanup admitted");
    expect(read_file(path) == *second_bytes, "stale cleanup unlinked newer generation");
    const auto repeated = io.submit(make_write(77, 2, path, make_payload(1, 9)));
    expect_status(repeated.status, server_prompt_cache_io_status::generation_not_newer,
                  "non-monotonic generation admitted");

    const auto cleanup = io.cleanup_exact(77, 2, path);
    expect_status(cleanup.status, server_prompt_cache_io_status::ok, "current exact cleanup submit");
    const auto cleanup_completion = wait_for_jobs(io, { cleanup.job_id }).at(cleanup.job_id);
    expect_status(cleanup_completion.status, server_prompt_cache_io_status::ok, "current exact cleanup");
    expect(cleanup_completion.exact_path_removed && !fs::exists(path), "exact cleanup did not remove current path");
    expect(io.stats().disk_committed_bytes == 0, "cleanup did not release disk accounting");
}

void test_pre_and_post_fence_cancellation() {
    temporary_directory    temp;
    server_prompt_cache_io io(make_config());

    auto pre         = make_write(1, 1, temp.path("pre.lcpc"), make_payload(4096, 7));
    pre.faults.pause = server_prompt_cache_io_test_pause::before_publication_fence;
    auto pre_result  = io.submit(std::move(pre));
    expect(io.test_wait_for_pause(pre_result.job_id, server_prompt_cache_io_test_pause::before_publication_fence, 5000),
           "pre-fence pause");
    expect_status(io.cancel(1, 1), server_prompt_cache_io_status::cancel_requested, "pre-fence cancel");
    const auto pre_completion = wait_for_jobs(io, { pre_result.job_id }).at(pre_result.job_id);
    expect_status(pre_completion.status, server_prompt_cache_io_status::cancelled, "pre-fence cancellation result");
    expect(!pre_completion.publication_fenced && !fs::exists(temp.path("pre.lcpc")),
           "pre-fence cancel published a file");

    auto post         = make_write(2, 1, temp.path("post.lcpc"), make_payload(1024, 13));
    post.faults.pause = server_prompt_cache_io_test_pause::after_publication_fence;
    auto post_result  = io.submit(std::move(post));
    expect(io.test_wait_for_pause(post_result.job_id, server_prompt_cache_io_test_pause::after_publication_fence, 5000),
           "post-fence pause");
    expect_status(io.cancel(2, 1), server_prompt_cache_io_status::too_late, "post-fence cancel contract");
    io.test_release_pause(post_result.job_id);
    const auto post_completion = wait_for_jobs(io, { post_result.job_id }).at(post_result.job_id);
    expect_status(post_completion.status, server_prompt_cache_io_status::ok, "post-fence publication");
    expect(fs::exists(temp.path("post.lcpc")), "post-fence cancel removed durable file");

    auto retired         = make_write(3, 1, temp.path("retired.lcpc"), make_payload(1024, 17));
    retired.faults.pause = server_prompt_cache_io_test_pause::after_publication_fence;
    auto retired_result  = io.submit(std::move(retired));
    expect(
        io.test_wait_for_pause(retired_result.job_id, server_prompt_cache_io_test_pause::after_publication_fence, 5000),
        "post-fence retirement pause");
    expect_status(io.retire(3, 1), server_prompt_cache_io_status::too_late, "post-fence retire fence result");
    io.test_release_pause(retired_result.job_id);
    const auto retired_completion = wait_for_jobs(io, { retired_result.job_id }).at(retired_result.job_id);
    expect(retired_completion.lifecycle == server_prompt_cache_io_lifecycle::retired &&
               retired_completion.exact_path_removed && !fs::exists(temp.path("retired.lcpc")),
           "post-fence retirement did not publish-then-clean safely");
    expect_no_private_temps(temp.path());
}

void test_checksum_enospc_and_cooldown() {
    temporary_directory temp;
    auto                config = make_config();
    config.error_cooldown_ms   = 60000;
    server_prompt_cache_io io(config);

    auto mismatch                  = make_write(1, 1, temp.path("mismatch.lcpc"), make_payload(128, 1));
    mismatch.verify_checksum       = true;
    mismatch.expected_checksum     = 123;
    const auto mismatch_result     = io.submit(std::move(mismatch));
    const auto mismatch_completion = wait_for_jobs(io, { mismatch_result.job_id }).at(mismatch_result.job_id);
    expect_status(mismatch_completion.status, server_prompt_cache_io_status::checksum_mismatch,
                  "checksum mismatch result");
    expect(!fs::exists(temp.path("mismatch.lcpc")), "checksum mismatch touched final path");

    auto full                    = make_write(2, 1, temp.path("full.lcpc"), make_payload(100, 2));
    full.faults.max_write_size   = 3;
    full.faults.fail_after_bytes = 17;
    full.faults.fail_write_errno = ENOSPC;
    const auto full_result       = io.submit(std::move(full));
    const auto full_completion   = wait_for_jobs(io, { full_result.job_id }).at(full_result.job_id);
    expect_status(full_completion.status, server_prompt_cache_io_status::no_space, "ENOSPC result");
    expect(full_completion.os_error == ENOSPC && !fs::exists(temp.path("full.lcpc")), "ENOSPC publication/errno");
    const auto failed_stats = io.stats();
    expect(failed_stats.disk_reserved_bytes == 0 && failed_stats.ram_held_bytes == 0 &&
               failed_stats.last_error == ENOSPC && failed_stats.cooldown_until_ns > failed_stats.last_error_ns,
           "ENOSPC release/cooldown accounting");

    auto cooled = io.submit(make_write(3, 1, temp.path("cooled.lcpc"), make_payload(32, 3)));
    expect_status(cooled.status, server_prompt_cache_io_status::cooldown, "error cooldown admission");
    io.clear_error_cooldown();
    auto recovered = io.submit(make_write(3, 1, temp.path("cooled.lcpc"), make_payload(32, 3)));
    expect_status(recovered.status, server_prompt_cache_io_status::ok, "cooldown reset hook");
    expect_status(wait_for_jobs(io, { recovered.job_id }).at(recovered.job_id).status,
                  server_prompt_cache_io_status::ok, "write after cooldown reset");
    expect_no_private_temps(temp.path());
}

void test_commit_uncertain_retains_ram_until_cleanup() {
    temporary_directory                       temp;
    server_prompt_cache_io                    io(make_config());
    auto                                      payload = make_payload(701, 51);
    std::weak_ptr<const std::vector<uint8_t>> weak    = payload;
    auto                                      write   = make_write(1, 1, temp.path("uncertain.lcpc"), payload);
    write.faults.fail_directory_fsync_errno           = EIO;
    auto result                                       = io.submit(std::move(write));
    result.payload.reset();
    payload.reset();

    const auto completion = wait_for_jobs(io, { result.job_id }).at(result.job_id);
    expect_status(completion.status, server_prompt_cache_io_status::commit_uncertain, "directory-fsync uncertainty");
    expect(completion.lifecycle == server_prompt_cache_io_lifecycle::commit_uncertain &&
               fs::exists(temp.path("uncertain.lcpc")),
           "uncertain commit state/path");
    expect(!weak.expired() && io.pending_payload(1, 1), "uncertain commit released RAM payload");
    const auto uncertain_stats = io.stats();
    expect(uncertain_stats.ram_held_bytes == 701 && uncertain_stats.disk_reserved_bytes == 0 &&
               uncertain_stats.disk_uncertain_bytes == 701,
           "uncertain byte accounting");

    io.clear_error_cooldown();
    const auto cleanup = io.cleanup_exact(1, 1, temp.path("uncertain.lcpc"));
    expect_status(cleanup.status, server_prompt_cache_io_status::ok, "uncertain cleanup admission");
    const auto cleanup_completion = wait_for_jobs(io, { cleanup.job_id }).at(cleanup.job_id);
    expect_status(cleanup_completion.status, server_prompt_cache_io_status::ok, "uncertain cleanup result");
    expect(!fs::exists(temp.path("uncertain.lcpc")) && weak.expired(), "uncertain cleanup did not release file/RAM");
    const auto cleaned_stats = io.stats();
    expect(cleaned_stats.ram_held_bytes == 0 && cleaned_stats.disk_uncertain_bytes == 0,
           "uncertain cleanup accounting");

    const auto durable = io.submit(make_write(2, 1, temp.path("cleanup-uncertain.lcpc"), make_payload(91, 9)));
    expect_status(wait_for_jobs(io, { durable.job_id }).at(durable.job_id).status, server_prompt_cache_io_status::ok,
                  "cleanup-uncertain setup");
    server_prompt_cache_io_faults cleanup_fault;
    cleanup_fault.fail_directory_fsync_errno = EIO;
    const auto uncertain_cleanup = io.cleanup_exact(2, 1, temp.path("cleanup-uncertain.lcpc"), cleanup_fault);
    const auto uncertain_cleanup_completion =
        wait_for_jobs(io, { uncertain_cleanup.job_id }).at(uncertain_cleanup.job_id);
    expect_status(uncertain_cleanup_completion.status, server_prompt_cache_io_status::commit_uncertain,
                  "cleanup directory-fsync uncertainty");
    expect(!fs::exists(temp.path("cleanup-uncertain.lcpc")) && io.stats().disk_uncertain_bytes == 91,
           "uncertain cleanup physical/accounted state");
    io.clear_error_cooldown();
    const auto reconcile_cleanup = io.cleanup_exact(2, 1, temp.path("cleanup-uncertain.lcpc"));
    expect_status(wait_for_jobs(io, { reconcile_cleanup.job_id }).at(reconcile_cleanup.job_id).status,
                  server_prompt_cache_io_status::ok, "durable absence reconciliation");
    expect(io.stats().disk_uncertain_bytes == 0, "durable absence did not release uncertain bytes");
}

void test_existing_adoption_and_same_path_accounting() {
    temporary_directory temp;
    const std::string   path      = temp.path("adopted.lcpc");
    const auto          old_bytes = make_payload(40, 61);
    write_file(path, *old_bytes);

    auto config               = make_config();
    config.initial_disk_bytes = 40;
    config.max_disk_bytes     = 200;
    server_prompt_cache_io io(config);
    expect_status(io.adopt_existing(5, 1, path, 40), server_prompt_cache_io_status::ok, "adopt startup-owned file");
    auto adopted_stats = io.stats();
    expect(adopted_stats.initial_disk_bytes == 0 && adopted_stats.disk_committed_bytes == 40,
           "startup ownership transfer accounting");

    const auto replacement                                 = make_payload(60, 71);
    auto       exceptional                                 = make_write(5, 2, path, replacement);
    exceptional.faults.fail_worker_exception_before_rename = true;
    const auto exceptional_result                          = io.submit(std::move(exceptional));
    const auto exceptional_completion = wait_for_jobs(io, { exceptional_result.job_id }).at(exceptional_result.job_id);
    expect_status(exceptional_completion.status, server_prompt_cache_io_status::allocation_failed,
                  "pre-rename worker exception result");
    const auto after_exception = io.stats();
    expect(read_file(path) == *old_bytes && after_exception.disk_committed_bytes == 40 &&
               after_exception.disk_reserved_bytes == 0,
           "pre-rename exception removed/double-counted adopted predecessor");
    expect(io.inspect(5, 1).bytes == 40, "pre-rename exception lost predecessor ownership");

    io.clear_error_cooldown();
    auto next              = make_write(5, 3, path, replacement);
    next.faults.pause      = server_prompt_cache_io_test_pause::before_write;
    const auto next_result = io.submit(std::move(next));
    expect(io.test_wait_for_pause(next_result.job_id, server_prompt_cache_io_test_pause::before_write, 5000),
           "replacement reservation pause");
    expect_status(io.adopt_existing(6, 1, temp.path("late-adopt.lcpc"), 1), server_prompt_cache_io_status::job_limit,
                  "adoption raced an active writer");
    const auto pending_stats = io.stats();
    expect(pending_stats.disk_committed_bytes == 40 && pending_stats.disk_reserved_bytes == 60,
           "same-path temp reservation accounting");
    io.test_release_pause(next_result.job_id);
    expect_status(wait_for_jobs(io, { next_result.job_id }).at(next_result.job_id).status,
                  server_prompt_cache_io_status::ok, "same-path replacement result");
    const auto final_stats = io.stats();
    expect(final_stats.disk_committed_bytes == 60 && final_stats.disk_reserved_bytes == 0 &&
               final_stats.initial_disk_bytes == 0,
           "same-path replacement double-counted old file");
    expect(read_file(path) == *replacement, "same-path replacement payload");
    expect_status(io.adopt_existing(6, 1, temp.path("missing.lcpc"), 1), server_prompt_cache_io_status::disk_limit,
                  "adoption exceeded external accounting");
}

void test_graceful_and_fast_shutdown() {
    temporary_directory    graceful_temp;
    server_prompt_cache_io graceful(make_config());
    const auto first  = graceful.submit(make_write(1, 1, graceful_temp.path("one.lcpc"), make_payload(2048, 1)));
    const auto second = graceful.submit(make_write(2, 1, graceful_temp.path("two.lcpc"), make_payload(2048, 2)));
    expect_status(graceful.shutdown(server_prompt_cache_io_shutdown_mode::graceful), server_prompt_cache_io_status::ok,
                  "graceful shutdown");
    const auto graceful_completions = wait_for_jobs(graceful, { first.job_id, second.job_id });
    expect_status(graceful_completions.at(first.job_id).status, server_prompt_cache_io_status::ok,
                  "graceful first drain");
    expect_status(graceful_completions.at(second.job_id).status, server_prompt_cache_io_status::ok,
                  "graceful second drain");
    expect(fs::exists(graceful_temp.path("one.lcpc")) && fs::exists(graceful_temp.path("two.lcpc")),
           "graceful shutdown did not drain files");
    expect_status(graceful.submit(make_write(3, 1, graceful_temp.path("sealed.lcpc"), make_payload(1, 3))).status,
                  server_prompt_cache_io_status::sealed, "graceful shutdown did not seal admission");

    temporary_directory    fast_temp;
    server_prompt_cache_io fast(make_config());
    auto                   paused = make_write(1, 1, fast_temp.path("paused.lcpc"), make_payload(8192, 4));
    paused.faults.pause           = server_prompt_cache_io_test_pause::before_write;
    const auto paused_result      = fast.submit(std::move(paused));
    const auto queued_result      = fast.submit(make_write(2, 1, fast_temp.path("queued.lcpc"), make_payload(8192, 5)));
    expect(fast.test_wait_for_pause(paused_result.job_id, server_prompt_cache_io_test_pause::before_write, 5000),
           "fast shutdown pause");
    expect_status(fast.shutdown(server_prompt_cache_io_shutdown_mode::fast), server_prompt_cache_io_status::ok,
                  "fast shutdown");
    const auto fast_completions = wait_for_jobs(fast, { paused_result.job_id, queued_result.job_id });
    expect_status(fast_completions.at(paused_result.job_id).status, server_prompt_cache_io_status::cancelled,
                  "fast active cancellation");
    expect_status(fast_completions.at(queued_result.job_id).status, server_prompt_cache_io_status::cancelled,
                  "fast queued cancellation");
    expect(!fs::exists(fast_temp.path("paused.lcpc")) && !fs::exists(fast_temp.path("queued.lcpc")),
           "fast shutdown published pre-fence work");
    expect(fast.stats().active_jobs == 0 && fast.stats().ram_held_bytes == 0, "fast shutdown accounting leak");
}

void test_callback_locking_exception_and_lifetime() {
    temporary_directory temp;
    std::weak_ptr<int>  weak_token;
    {
        server_prompt_cache_io io(make_config());

        std::atomic<bool> replacement_capture_destroyed{ false };
        auto replacement_capture = std::make_shared<callback_reentry_probe>(io, replacement_capture_destroyed);
        io.set_wake_callback([replacement_capture] {});
        replacement_capture.reset();
        io.set_wake_callback([] {});
        expect(replacement_capture_destroyed.load(), "replaced callback capture remained live after replacement");

        std::atomic<bool> clear_capture_destroyed{ false };
        auto              clear_capture = std::make_shared<callback_reentry_probe>(io, clear_capture_destroyed);
        io.set_wake_callback([clear_capture] {});
        clear_capture.reset();
        io.set_wake_callback({});
        expect(clear_capture_destroyed.load(), "cleared callback capture remained live after clear");

        auto token = std::make_shared<int>(7);
        weak_token = token;
        std::atomic<uint32_t> calls{ 0 };
        io.set_wake_callback([&io, token, &calls] {
            (void) io.stats();  // deadlocks if callbacks run under the engine lock
            expect(*token == 7, "callback lifetime token");
            ++calls;
        });
        token.reset();
        const auto first = io.submit(make_write(1, 1, temp.path("callback.lcpc"), make_payload(64, 1)));
        wait_for_jobs(io, { first.job_id });
        wait_until([&] { return calls.load() == 1; }, "wake callback not called");
        expect(!weak_token.expired(), "callback capture died while registered");

        io.set_wake_callback([] { throw std::runtime_error("contained wake failure"); });
        const auto second = io.submit(make_write(2, 1, temp.path("throw.lcpc"), make_payload(64, 2)));
        wait_for_jobs(io, { second.job_id });
        wait_until([&] { return io.stats().callback_failures == 1; }, "callback exception not contained");
        expect(io.stats().callback_calls == 2, "callback exact call accounting");

        std::atomic<bool> callback_entered{ false };
        std::atomic<bool> release_callback{ false };
        std::atomic<bool> clear_returned{ false };
        io.set_wake_callback([&] {
            callback_entered = true;
            while (!release_callback.load()) {
                std::this_thread::yield();
            }
        });
        const auto third = io.submit(make_write(3, 1, temp.path("lifetime.lcpc"), make_payload(64, 3)));
        wait_until([&] { return callback_entered.load(); }, "lifetime callback did not enter");
        std::thread clearer([&] {
            io.set_wake_callback({});
            clear_returned = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        expect(!clear_returned.load(), "callback clear returned while old callback was in flight");
        release_callback = true;
        clearer.join();
        expect(clear_returned.load(), "callback clear did not finish after prior callback");
        wait_for_jobs(io, { third.job_id });
        expect(io.stats().callback_calls == 3, "lifetime callback accounting");
    }
    expect(weak_token.expired(), "callback capture outlived engine destruction");
}

void test_abandoned_temp_startup_and_explicit_cleanup() {
    temporary_directory        temp;
    const std::vector<uint8_t> bytes{ 1, 2, 3, 4 };
    const std::string          abandoned    = temp.path(".pcache-io-dead-1.tmp");
    const std::string          unrelated    = temp.path("ordinary.tmp");
    const std::string          symlink_path = temp.path(".pcache-io-link.tmp");
    write_file(abandoned, bytes);
    write_file(unrelated, bytes);
    expect(::symlink(unrelated.c_str(), symlink_path.c_str()) == 0, "create private-name symlink");

    auto config = make_config();
    config.startup_cleanup_directories.push_back(temp.path());
    server_prompt_cache_io io(config);
    wait_until([&] { return io.stats().startup_cleanup_done; }, "startup cleanup did not finish");
    expect(!fs::exists(abandoned), "startup cleanup left abandoned regular temp");
    expect(fs::exists(unrelated) && fs::is_symlink(symlink_path), "startup cleanup touched unrelated or symlink path");
    expect(io.stats().abandoned_temps_removed == 1, "startup cleanup accounting");

    const std::string explicit_temp = temp.path(".pcache-io-explicit.tmp");
    write_file(explicit_temp, bytes);
    const auto cleanup = io.cleanup_abandoned_temp(explicit_temp);
    expect_status(cleanup.status, server_prompt_cache_io_status::ok, "explicit abandoned cleanup admission");
    const auto completion = wait_for_jobs(io, { cleanup.job_id }).at(cleanup.job_id);
    expect_status(completion.status, server_prompt_cache_io_status::ok, "explicit abandoned cleanup");
    expect(completion.exact_path_removed && !fs::exists(explicit_temp), "explicit abandoned temp remained");
    expect_status(io.cleanup_abandoned_temp(unrelated).status, server_prompt_cache_io_status::invalid_argument,
                  "unrestricted abandoned cleanup accepted");
}

}  // namespace

int main() {
    try {
        test_pending_visibility_and_durable_release();
        test_job_byte_and_disk_admission();
        test_crossed_fence_generation_and_exact_cleanup();
        test_pre_and_post_fence_cancellation();
        test_checksum_enospc_and_cooldown();
        test_commit_uncertain_retains_ram_until_cleanup();
        test_existing_adoption_and_same_path_accounting();
        test_graceful_and_fast_shutdown();
        test_callback_locking_exception_and_lifetime();
        test_abandoned_temp_startup_and_explicit_cleanup();
        std::puts("server prompt cache async I/O tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-server-prompt-cache-io: %s\n", error.what());
        return 1;
    }
}

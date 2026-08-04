#include "llama-snapshot-worker.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <set>
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

void expect_status(llama_snapshot_status actual, llama_snapshot_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_snapshot_status_name(expected) + ", got " +
             llama_snapshot_status_name(actual));
    }
}

void expect_worker_status(
        llama_snapshot_worker_status actual, llama_snapshot_worker_status expected, const std::string & message) {
    if (actual != expected) {
        fail(message + ": expected " + llama_snapshot_worker_status_name(expected) + ", got " +
             llama_snapshot_worker_status_name(actual));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        char   pattern[] = "/tmp/llama-snapshot-worker-XXXXXX";
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

class race_gate {
  public:
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex);
        reached = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    }

    void wait_until_reached(const std::string & message) {
        std::unique_lock<std::mutex> lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(3), [&] { return reached; })) {
            released = true;
            cv.notify_all();
            fail(message);
        }
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }

  private:
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    reached  = false;
    bool                    released = false;
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

llama_snapshot_worker_config make_config(const fs::path & root, uint32_t staging_buffers = 2) {
    llama_snapshot_worker_config config;
    config.store.root_path              = root.string();
    config.store.physical_device_id     = "apple-internal-ssd-physical-0";
    config.store.physical_device_queues = LLAMA_SNAPSHOT_DEVICE_QUEUES;
    config.store.chunk_payload_bytes    = 64;
    config.store.max_chunks             = 64;
    config.store.max_manifest_bytes     = 64 * 1024;
    config.store.max_snapshot_bytes     = 4096;
    config.staging_buffers              = staging_buffers;
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

uint32_t count_temporary_paths(const fs::path & root) {
    std::error_code error;
    if (!fs::exists(root, error)) {
        return 0;
    }
    uint32_t count = 0;
    for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind(".partial-generation-", 0) == 0 || name == "current.manifest.tmp") {
            ++count;
        }
    }
    return count;
}

template <typename Predicate>
void wait_until(Predicate predicate, const std::string & message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            fail(message);
        }
        std::this_thread::yield();
    }
}

void stage_payload(llama_snapshot_worker & worker,
                   uint64_t                job_id,
                   const std::vector<uint8_t> & payload,
                   std::set<const void *> *    buffer_addresses = nullptr) {
    size_t offset = 0;
    wait_until(
        [&] {
            const auto staged = worker.try_stage_next(job_id, [&](uint8_t * destination, size_t size) {
                expect(offset + size <= payload.size(), "write staging exceeded payload");
                std::memcpy(destination, payload.data() + offset, size);
                if (buffer_addresses != nullptr) {
                    buffer_addresses->insert(destination);
                }
                return llama_snapshot_status::ok;
            });
            if (staged.status == llama_snapshot_worker_status::would_block) {
                return false;
            }
            expect_worker_status(staged.status, llama_snapshot_worker_status::ok, "stage write chunk");
            expect(staged.bytes != 0 && staged.index * 64 == offset, "write staging order");
            offset += static_cast<size_t>(staged.bytes);
            return offset == payload.size();
        },
        "write staging timed out");
}

llama_snapshot_worker_completion wait_complete(llama_snapshot_worker & worker, uint64_t job_id) {
    const auto completion = worker.wait(job_id);
    expect_worker_status(completion.status, llama_snapshot_worker_status::ok, "wait for worker");
    expect(completion.complete, "worker wait returned incomplete");
    return completion;
}

void expect_current(llama_snapshot_store &          store,
                    const llama_snapshot_identity & identity,
                    uint64_t                        generation,
                    const std::vector<uint8_t> &    payload) {
    const auto opened = store.open_current(identity);
    expect_status(opened.status, llama_snapshot_status::ok, "open current generation");
    expect(opened.manifest.snapshot_generation == generation, "unexpected current generation");
    const auto read = store.read_all(opened.manifest);
    expect_status(read.status, llama_snapshot_status::ok, "read current generation");
    expect(read.payload == payload, "current payload mismatch");
}

void test_bounded_async_round_trip_and_backpressure() {
    temporary_directory temp;
    const auto          identity = make_identity();
    const auto          config   = make_config(temp.path());
    const auto          payload  = make_payload(257, 17);
    llama_snapshot_worker worker(config);

    const auto write = worker.begin_write(make_metadata(1, identity), payload.size());
    expect_worker_status(write.status, llama_snapshot_worker_status::ok, "begin write");
    expect(write.chunk_count == 5 && write.total_payload_bytes == payload.size(), "write admission metadata");
    expect_worker_status(worker.begin_read(identity).status, llama_snapshot_worker_status::busy,
                         "second job admitted while write active");
    expect_worker_status(worker.try_consume_next(write.job_id, [](const uint8_t *, size_t) {
                             return llama_snapshot_status::ok;
                         }).status,
                         llama_snapshot_worker_status::wrong_job_kind, "consume write job");

    size_t first_offset = 0;
    const auto first = worker.try_stage_next(write.job_id, [&](uint8_t * destination, size_t size) {
        std::memcpy(destination, payload.data(), size);
        first_offset = size;
        return llama_snapshot_status::ok;
    });
    expect_worker_status(first.status, llama_snapshot_worker_status::ok, "stage first chunk");
    expect(first_offset == 64, "first staged chunk size");
    llama_snapshot_store store(config.store);
    expect_status(store.open_current(identity).status, llama_snapshot_status::no_current_generation,
                  "incomplete write became current");

    std::set<const void *> write_buffers;
    size_t                 offset = first_offset;
    wait_until(
        [&] {
            const auto staged = worker.try_stage_next(write.job_id, [&](uint8_t * destination, size_t size) {
                std::memcpy(destination, payload.data() + offset, size);
                write_buffers.insert(destination);
                return llama_snapshot_status::ok;
            });
            if (staged.status == llama_snapshot_worker_status::would_block) {
                return false;
            }
            expect_worker_status(staged.status, llama_snapshot_worker_status::ok, "stage remaining chunks");
            expect(staged.index * 64 == offset, "remaining staging order");
            offset += static_cast<size_t>(staged.bytes);
            return offset == payload.size();
        },
        "remaining write staging timed out");
    const auto write_done = wait_complete(worker, write.job_id);
    expect_status(write_done.operation_status, llama_snapshot_status::ok, "async write operation");
    expect(write_done.committed && write_done.generation == 1, "async write publication");
    expect(write_buffers.size() <= config.staging_buffers, "write allocated beyond fixed staging set");
    expect_current(store, identity, 1, payload);
    expect_worker_status(worker.poll(write.job_id).status, llama_snapshot_worker_status::stale_job,
                         "harvested job remained active");

    const auto read = worker.begin_read(identity);
    expect_worker_status(read.status, llama_snapshot_worker_status::ok, "begin read");
    expect_worker_status(worker.try_stage_next(read.job_id, [](uint8_t *, size_t) {
                             return llama_snapshot_status::ok;
                         }).status,
                         llama_snapshot_worker_status::wrong_job_kind, "stage read job");
    wait_until([&] { return worker.stats().staging_in_use == config.staging_buffers; },
               "read queue did not reach its fixed bound");
    expect(!worker.poll(read.job_id).complete, "full unread queue reported complete");

    std::vector<uint8_t> assembled;
    assembled.reserve(payload.size());
    std::set<const void *> read_buffers;
    wait_until(
        [&] {
            const auto consumed = worker.try_consume_next(read.job_id, [&](const uint8_t * source, size_t size) {
                read_buffers.insert(source);
                assembled.insert(assembled.end(), source, source + size);
                return llama_snapshot_status::ok;
            });
            if (consumed.status == llama_snapshot_worker_status::would_block) {
                return false;
            }
            expect_worker_status(consumed.status, llama_snapshot_worker_status::ok, "consume read chunk");
            return assembled.size() == payload.size();
        },
        "read consumption timed out");
    const auto read_done = wait_complete(worker, read.job_id);
    expect_status(read_done.operation_status, llama_snapshot_status::ok, "async read operation");
    expect(!read_done.committed && read_done.generation == 1, "async read completion metadata");
    expect(assembled == payload, "async read payload mismatch");
    expect(read_buffers.size() <= config.staging_buffers, "read allocated beyond fixed staging set");

    const auto stats = worker.stats();
    expect(stats.staging_buffers == 2 && stats.staging_buffer_bytes == 64 && stats.staging_in_use == 0,
           "staging stats contract");
    expect(stats.staging_high_watermark == 2 && stats.write_staged_chunks == 5 &&
               stats.write_released_chunks == 5 && stats.read_staged_chunks == 5 &&
               stats.read_consumed_chunks == 5 && stats.completed_jobs == 2,
           "queue accounting contract");

    const auto empty_write = worker.begin_write(make_metadata(2, identity), 0);
    expect_worker_status(empty_write.status, llama_snapshot_worker_status::ok, "begin empty write");
    const auto empty_write_done = wait_complete(worker, empty_write.job_id);
    expect_status(empty_write_done.operation_status, llama_snapshot_status::ok, "empty write operation");
    expect(empty_write_done.committed, "empty write publication");
    const auto empty_read = worker.begin_read(identity);
    expect_worker_status(empty_read.status, llama_snapshot_worker_status::ok, "begin empty read");
    const auto empty_read_done = wait_complete(worker, empty_read.job_id);
    expect_status(empty_read_done.operation_status, llama_snapshot_status::ok, "empty read operation");
    expect(empty_read_done.total_payload_bytes == 0, "empty read size");
}

void test_failures_cancellation_and_publication() {
    temporary_directory temp;
    const auto          identity = make_identity();
    const auto          config   = make_config(temp.path());
    const auto          baseline = make_payload(129, 31);
    llama_snapshot_store store(config.store);
    expect_status(store.write_generation(make_metadata(1, identity), baseline).status, llama_snapshot_status::ok,
                  "write baseline");
    llama_snapshot_worker worker(config);

    const auto cancelled = worker.begin_write(make_metadata(2, identity), 192);
    expect_worker_status(cancelled.status, llama_snapshot_worker_status::ok, "begin cancelled write");
    const auto first = worker.try_stage_next(cancelled.job_id, [](uint8_t * destination, size_t size) {
        std::fill_n(destination, size, uint8_t{ 9 });
        return llama_snapshot_status::ok;
    });
    expect_worker_status(first.status, llama_snapshot_worker_status::ok, "stage cancelled write");
    expect_worker_status(worker.cancel(cancelled.job_id), llama_snapshot_worker_status::ok, "cancel write");
    const auto cancelled_done = wait_complete(worker, cancelled.job_id);
    expect_status(cancelled_done.operation_status, llama_snapshot_status::cancelled, "cancelled write result");
    expect(!cancelled_done.committed && count_temporary_paths(temp.path()) == 0, "cancelled write cleanup");
    expect_current(store, identity, 1, baseline);

    const auto failed_source = worker.begin_write(make_metadata(3, identity), 64);
    expect_worker_status(failed_source.status, llama_snapshot_worker_status::ok, "begin source failure");
    expect_worker_status(worker.try_stage_next(failed_source.job_id, [](uint8_t *, size_t) {
                             return llama_snapshot_status::io_error;
                         }).status,
                         llama_snapshot_worker_status::source_failed, "source failure signal");
    const auto source_done = wait_complete(worker, failed_source.job_id);
    expect_status(source_done.operation_status, llama_snapshot_status::io_error, "source failure result");
    expect(!source_done.committed && count_temporary_paths(temp.path()) == 0, "source failure cleanup");
    expect_current(store, identity, 1, baseline);

    llama_snapshot_faults no_space;
    no_space.fail_after_bytes = 0;
    no_space.write_fault      = llama_snapshot_write_fault::no_space;
    const auto failed_write   = worker.begin_write(make_metadata(4, identity), 64, no_space);
    expect_worker_status(failed_write.status, llama_snapshot_worker_status::ok, "begin disk failure");
    stage_payload(worker, failed_write.job_id, make_payload(64, 43));
    const auto write_done = wait_complete(worker, failed_write.job_id);
    expect_status(write_done.operation_status, llama_snapshot_status::no_space, "disk failure result");
    expect(!write_done.committed && count_temporary_paths(temp.path()) == 0, "disk failure cleanup");
    expect_current(store, identity, 1, baseline);

    const auto failed_sink = worker.begin_read(identity);
    expect_worker_status(failed_sink.status, llama_snapshot_worker_status::ok, "begin sink failure");
    wait_until(
        [&] {
            const auto consumed = worker.try_consume_next(failed_sink.job_id, [](const uint8_t *, size_t) {
                return llama_snapshot_status::io_error;
            });
            if (consumed.status == llama_snapshot_worker_status::would_block) {
                return false;
            }
            expect_worker_status(consumed.status, llama_snapshot_worker_status::sink_failed, "sink failure signal");
            return true;
        },
        "sink failure timed out");
    const auto sink_done = wait_complete(worker, failed_sink.job_id);
    expect_status(sink_done.operation_status, llama_snapshot_status::io_error, "sink failure result");
    expect_current(store, identity, 1, baseline);

    const auto cancelled_read = worker.begin_read(identity);
    expect_worker_status(cancelled_read.status, llama_snapshot_worker_status::ok, "begin cancelled read");
    wait_until([&] { return worker.stats().staging_in_use == config.staging_buffers; },
               "cancelled read did not fill queue");
    expect_worker_status(worker.cancel(cancelled_read.job_id), llama_snapshot_worker_status::ok, "cancel read");
    const auto read_done = wait_complete(worker, cancelled_read.job_id);
    expect_status(read_done.operation_status, llama_snapshot_status::cancelled, "cancelled read result");
    expect_current(store, identity, 1, baseline);
}

void test_cancellation_publication_fence() {
    temporary_directory temp;
    const auto          identity = make_identity();
    const auto          config   = make_config(temp.path());
    const auto          baseline = make_payload(129, 71);
    llama_snapshot_store store(config.store);
    expect_status(store.write_generation(make_metadata(1, identity), baseline).status, llama_snapshot_status::ok,
                  "write publication-fence baseline");
    llama_snapshot_worker worker(config);

    race_gate             before_fence;
    llama_snapshot_faults cancellable_faults;
    cancellable_faults.before_manifest_commit_fence = [&] { before_fence.arrive_and_wait(); };
    const auto cancelled_payload = make_payload(64, 83);
    const auto cancellable =
        worker.begin_write(make_metadata(2, identity), cancelled_payload.size(), cancellable_faults);
    expect_worker_status(cancellable.status, llama_snapshot_worker_status::ok, "begin pre-fence cancellation");
    stage_payload(worker, cancellable.job_id, cancelled_payload);
    before_fence.wait_until_reached("write did not reach pre-publication fence");
    const auto cancel_status = worker.cancel(cancellable.job_id);
    before_fence.release();
    expect_worker_status(cancel_status, llama_snapshot_worker_status::ok, "cancel before publication fence");
    const auto cancelled = wait_complete(worker, cancellable.job_id);
    expect_status(cancelled.operation_status, llama_snapshot_status::cancelled, "pre-fence cancellation result");
    expect(!cancelled.committed && count_temporary_paths(temp.path()) == 0, "pre-fence cancellation cleanup");
    expect(!fs::exists(temp.path() / "generation-0000000000000002"), "cancelled generation survived fence");
    expect_current(store, identity, 1, baseline);

    race_gate             after_fence;
    llama_snapshot_faults committed_faults;
    committed_faults.after_manifest_commit_fence = [&] { after_fence.arrive_and_wait(); };
    const auto committed_payload = make_payload(64, 97);
    const auto committing = worker.begin_write(make_metadata(3, identity), committed_payload.size(), committed_faults);
    expect_worker_status(committing.status, llama_snapshot_worker_status::ok, "begin post-fence cancellation");
    stage_payload(worker, committing.job_id, committed_payload);
    after_fence.wait_until_reached("write did not cross publication fence");
    const auto too_late_status = worker.cancel(committing.job_id);
    after_fence.release();
    expect_worker_status(too_late_status, llama_snapshot_worker_status::job_complete,
                         "cancel after publication fence");
    const auto committed = wait_complete(worker, committing.job_id);
    expect_status(committed.operation_status, llama_snapshot_status::ok, "post-fence write result");
    expect(committed.committed && count_temporary_paths(temp.path()) == 0, "post-fence publication cleanup");
    expect_current(store, identity, 3, committed_payload);
    expect(!fs::exists(temp.path() / "generation-0000000000000001"), "previous generation survived publication");
}

void test_invalid_config_destruction_and_fork_child() {
    const auto identity = make_identity();
    {
        temporary_directory temp;
        llama_snapshot_worker invalid(make_config(temp.path(), 1));
        expect_worker_status(invalid.begin_read(identity).status, llama_snapshot_worker_status::invalid_argument,
                             "one-buffer worker admitted");
    }
    {
        temporary_directory temp;
        const auto          config = make_config(temp.path());
        {
            llama_snapshot_worker worker(config);
            const auto write = worker.begin_write(make_metadata(1, identity), 128);
            expect_worker_status(write.status, llama_snapshot_worker_status::ok, "begin destructor write");
        }
        expect(count_temporary_paths(temp.path()) == 0, "destructor left temporary generation");
        llama_snapshot_store store(config.store);
        expect_status(store.open_current(identity).status, llama_snapshot_status::no_current_generation,
                      "destructor published incomplete generation");
    }
    {
        temporary_directory temp;
        auto worker = std::make_unique<llama_snapshot_worker>(make_config(temp.path()));
        const auto write = worker->begin_write(make_metadata(1, identity), 128);
        expect_worker_status(write.status, llama_snapshot_worker_status::ok, "begin fork fixture");
        const pid_t child = ::fork();
        if (child < 0) {
            fail("fork failed: " + std::string(std::strerror(errno)));
        }
        if (child == 0) {
            if (worker->cancel(write.job_id) != llama_snapshot_worker_status::stopped) {
                _exit(2);
            }
            worker.reset();
            _exit(0);
        }

        int         child_status = 0;
        pid_t       waited       = 0;
        const auto  deadline     = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while ((waited = ::waitpid(child, &child_status, WNOHANG)) == 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                ::kill(child, SIGKILL);
                ::waitpid(child, &child_status, 0);
                fail("fork child hung destroying inherited worker");
            }
            std::this_thread::yield();
        }
        expect(waited == child, "waitpid failed for fork child");
        expect(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0, "fork child worker result");
        expect_worker_status(worker->cancel(write.job_id), llama_snapshot_worker_status::ok,
                             "cancel parent fork fixture");
        expect_status(wait_complete(*worker, write.job_id).operation_status, llama_snapshot_status::cancelled,
                      "parent fork fixture completion");
    }
}

}  // namespace

int main() {
    try {
        test_bounded_async_round_trip_and_backpressure();
        test_failures_cancellation_and_publication();
        test_cancellation_publication_fence();
        test_invalid_config_destruction_and_fork_child();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-snapshot-worker: %s\n", error.what());
        return 1;
    }
    std::puts("test-snapshot-worker: all checks passed");
    return 0;
}

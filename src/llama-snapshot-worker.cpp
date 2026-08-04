#include "llama-snapshot-worker.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

namespace {

bool valid_worker_config(const llama_snapshot_worker_config & config) {
    const llama_snapshot_store_config & store = config.store;
    return config.staging_buffers >= LLAMA_SNAPSHOT_MIN_STAGING_BUFFERS &&
           config.staging_buffers <= LLAMA_SNAPSHOT_MAX_STAGING_BUFFERS && !store.root_path.empty() &&
           !store.physical_device_id.empty() && store.physical_device_id.size() <= LLAMA_SNAPSHOT_MAX_IDENTITY_BYTES &&
           store.physical_device_queues == LLAMA_SNAPSHOT_DEVICE_QUEUES && store.chunk_payload_bytes != 0 &&
           store.chunk_payload_bytes <= LLAMA_SNAPSHOT_MAX_CHUNK_BYTES && store.max_chunks != 0 &&
           store.max_chunks <= LLAMA_SNAPSHOT_MAX_CHUNKS && store.max_manifest_bytes >= 256 &&
           store.max_manifest_bytes <= LLAMA_SNAPSHOT_MAX_MANIFEST_BYTES && store.max_snapshot_bytes != 0 &&
           store.max_snapshot_bytes <= LLAMA_SNAPSHOT_MAX_BYTES && store.chunk_payload_bytes <= SIZE_MAX;
}

uint64_t worker_chunk_count(const llama_snapshot_store_config & config, uint64_t payload_bytes) {
    return payload_bytes == 0 ?
               0 :
               payload_bytes / config.chunk_payload_bytes + (payload_bytes % config.chunk_payload_bytes != 0);
}

}  // namespace

const char * llama_snapshot_worker_status_name(llama_snapshot_worker_status status) {
    switch (status) {
        case llama_snapshot_worker_status::ok:                return "ok";
        case llama_snapshot_worker_status::invalid_argument:  return "invalid_argument";
        case llama_snapshot_worker_status::busy:              return "busy";
        case llama_snapshot_worker_status::would_block:       return "would_block";
        case llama_snapshot_worker_status::stale_job:         return "stale_job";
        case llama_snapshot_worker_status::job_complete:      return "job_complete";
        case llama_snapshot_worker_status::wrong_job_kind:    return "wrong_job_kind";
        case llama_snapshot_worker_status::source_failed:     return "source_failed";
        case llama_snapshot_worker_status::sink_failed:       return "sink_failed";
        case llama_snapshot_worker_status::allocation_failed: return "allocation_failed";
        case llama_snapshot_worker_status::cancelled:         return "cancelled";
        case llama_snapshot_worker_status::stopped:           return "stopped";
    }
    return "unknown";
}

struct llama_snapshot_worker::impl {
    enum class slot_state : uint8_t {
        free,
        filling,
        ready,
        io,
        consuming,
    };

    struct staging_slot {
        std::vector<uint8_t> bytes;
        slot_state           state = slot_state::free;
        uint32_t             index = 0;
        uint64_t             size  = 0;
    };

    struct job_spec {
        llama_snapshot_worker_job_kind kind = llama_snapshot_worker_job_kind::none;
        llama_snapshot_metadata        metadata;
        llama_snapshot_identity        expected_identity;
        llama_snapshot_faults          faults;
        uint64_t                       payload_bytes = 0;
        uint32_t                       chunk_count   = 0;
    };

    struct source_adapter final : llama_snapshot_chunk_source_i {
        impl &   owner;
        uint64_t job_id;

        source_adapter(impl & owner, uint64_t job_id) : owner(owner), job_id(job_id) {}

        llama_snapshot_chunk_source_result acquire(
                uint32_t index, uint64_t offset, uint64_t size) noexcept override {
            return owner.acquire_write_chunk(job_id, index, offset, size);
        }

        void release(uint32_t index) noexcept override { owner.release_write_chunk(job_id, index); }
    };

    explicit impl(llama_snapshot_worker_config worker_config) :
        config(std::move(worker_config)),
        store(config.store),
        creator_pid(::getpid()) {
        if (!valid_worker_config(config)) {
            initialization_status = llama_snapshot_worker_status::invalid_argument;
            return;
        }
        try {
            slots.resize(config.staging_buffers);
            for (staging_slot & slot : slots) {
                slot.bytes.resize(static_cast<size_t>(config.store.chunk_payload_bytes));
            }
            worker = std::make_unique<std::thread>(&impl::run, this);
            initialization_status = llama_snapshot_worker_status::ok;
        } catch (const std::bad_alloc &) {
            initialization_status = llama_snapshot_worker_status::allocation_failed;
        } catch (...) {
            initialization_status = llama_snapshot_worker_status::stopped;
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping         = true;
            cancel_requested = true;
            drop_ready_locked();
        }
        cv.notify_all();
        if (worker && worker->joinable()) {
            worker->join();
        }
    }

    bool inherited_process() const noexcept { return creator_pid != ::getpid(); }

    llama_snapshot_worker_begin_result begin_write(
            const llama_snapshot_metadata & metadata,
            uint64_t                        payload_bytes,
            const llama_snapshot_faults &   faults) {
        llama_snapshot_worker_begin_result result;
        result.kind                = llama_snapshot_worker_job_kind::write;
        result.total_payload_bytes = payload_bytes;
        std::lock_guard<std::mutex> lock(mutex);
        const llama_snapshot_worker_status admission = admission_status_locked();
        if (admission != llama_snapshot_worker_status::ok) {
            result.status = admission;
            return result;
        }
        if (metadata.snapshot_generation == 0 || metadata.request_generation == 0 ||
            payload_bytes > config.store.max_snapshot_bytes || faults.max_write_size == 0) {
            result.status = llama_snapshot_worker_status::invalid_argument;
            return result;
        }
        const uint64_t chunks = worker_chunk_count(config.store, payload_bytes);
        if (chunks > config.store.max_chunks || chunks > UINT32_MAX) {
            result.status = llama_snapshot_worker_status::invalid_argument;
            return result;
        }

        job_spec prepared;
        try {
            prepared.kind          = llama_snapshot_worker_job_kind::write;
            prepared.metadata      = metadata;
            prepared.faults        = faults;
            prepared.payload_bytes = payload_bytes;
            prepared.chunk_count   = static_cast<uint32_t>(chunks);
        } catch (const std::bad_alloc &) {
            result.status = llama_snapshot_worker_status::allocation_failed;
            return result;
        }
        start_job_locked(std::move(prepared));
        result.status      = llama_snapshot_worker_status::ok;
        result.job_id      = active_job_id;
        result.chunk_count = job.chunk_count;
        cv.notify_all();
        return result;
    }

    llama_snapshot_worker_begin_result begin_read(const llama_snapshot_identity & identity) {
        llama_snapshot_worker_begin_result result;
        result.kind = llama_snapshot_worker_job_kind::read;
        std::lock_guard<std::mutex> lock(mutex);
        const llama_snapshot_worker_status admission = admission_status_locked();
        if (admission != llama_snapshot_worker_status::ok) {
            result.status = admission;
            return result;
        }

        job_spec prepared;
        try {
            prepared.kind              = llama_snapshot_worker_job_kind::read;
            prepared.expected_identity = identity;
        } catch (const std::bad_alloc &) {
            result.status = llama_snapshot_worker_status::allocation_failed;
            return result;
        }
        start_job_locked(std::move(prepared));
        result.status = llama_snapshot_worker_status::ok;
        result.job_id = active_job_id;
        cv.notify_all();
        return result;
    }

    llama_snapshot_worker_transfer_result try_stage(
            uint64_t job_id, const llama_snapshot_stage_fill & fill) {
        llama_snapshot_worker_transfer_result result;
        size_t                                slot_index = 0;
        uint32_t                              index      = 0;
        uint64_t                              size       = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const llama_snapshot_worker_status status = active_status_locked(job_id);
            if (status != llama_snapshot_worker_status::ok) {
                result.status = status;
                return result;
            }
            if (job.kind != llama_snapshot_worker_job_kind::write) {
                result.status = llama_snapshot_worker_status::wrong_job_kind;
                return result;
            }
            if (!fill) {
                result.status = llama_snapshot_worker_status::invalid_argument;
                return result;
            }
            if (cancel_requested) {
                result.status = llama_snapshot_worker_status::cancelled;
                return result;
            }
            if (source_failure != llama_snapshot_status::ok) {
                result.status = llama_snapshot_worker_status::source_failed;
                return result;
            }
            if (worker_done || next_stage_index == job.chunk_count) {
                result.status = llama_snapshot_worker_status::job_complete;
                return result;
            }
            if (producer_active) {
                result.status = llama_snapshot_worker_status::would_block;
                return result;
            }
            const auto free = find_free_slot_locked();
            if (free == slots.end()) {
                result.status = llama_snapshot_worker_status::would_block;
                return result;
            }
            slot_index            = static_cast<size_t>(free - slots.begin());
            index                 = next_stage_index++;
            const uint64_t offset = static_cast<uint64_t>(index) * config.store.chunk_payload_bytes;
            size                  = std::min<uint64_t>(config.store.chunk_payload_bytes, job.payload_bytes - offset);
            free->state           = slot_state::filling;
            free->index           = index;
            free->size            = size;
            producer_active       = true;
            occupy_slot_locked();
        }

        llama_snapshot_status fill_status = llama_snapshot_status::io_error;
        int                   fill_error  = EIO;
        try {
            fill_status = fill(slots[slot_index].bytes.data(), static_cast<size_t>(size));
            fill_error  = 0;
        } catch (const std::bad_alloc &) {
            fill_error = ENOMEM;
        } catch (...) {
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            producer_active     = false;
            staging_slot & slot = slots[slot_index];
            if (!active || active_job_id != job_id || worker_done || cancel_requested || stopping) {
                free_slot_locked(slot, true);
                result.status = cancel_requested ? llama_snapshot_worker_status::cancelled :
                                                   llama_snapshot_worker_status::job_complete;
            } else if (fill_status != llama_snapshot_status::ok) {
                source_failure       = fill_status;
                source_failure_error = fill_error;
                free_slot_locked(slot, true);
                result.status = llama_snapshot_worker_status::source_failed;
            } else {
                slot.state = slot_state::ready;
                ++statistics.write_staged_chunks;
                result.status = llama_snapshot_worker_status::ok;
                result.index  = index;
                result.bytes  = size;
            }
        }
        cv.notify_all();
        return result;
    }

    llama_snapshot_worker_transfer_result try_consume(
            uint64_t job_id, const llama_snapshot_stage_consume & consume) {
        llama_snapshot_worker_transfer_result result;
        size_t                                slot_index = 0;
        uint32_t                              index      = 0;
        uint64_t                              size       = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const llama_snapshot_worker_status status = active_status_locked(job_id);
            if (status != llama_snapshot_worker_status::ok) {
                result.status = status;
                return result;
            }
            if (job.kind != llama_snapshot_worker_job_kind::read) {
                result.status = llama_snapshot_worker_status::wrong_job_kind;
                return result;
            }
            if (!consume) {
                result.status = llama_snapshot_worker_status::invalid_argument;
                return result;
            }
            if (cancel_requested && !worker_done) {
                result.status = llama_snapshot_worker_status::cancelled;
                return result;
            }
            if (consumer_active) {
                result.status = llama_snapshot_worker_status::would_block;
                return result;
            }
            const auto ready = find_ready_slot_locked(next_consume_index);
            if (ready == slots.end()) {
                result.status = worker_done ? llama_snapshot_worker_status::job_complete :
                                              llama_snapshot_worker_status::would_block;
                return result;
            }
            slot_index      = static_cast<size_t>(ready - slots.begin());
            index           = ready->index;
            size            = ready->size;
            ready->state    = slot_state::consuming;
            consumer_active = true;
        }

        llama_snapshot_status consume_status = llama_snapshot_status::io_error;
        int                   consume_error  = EIO;
        try {
            consume_status = consume(slots[slot_index].bytes.data(), static_cast<size_t>(size));
            consume_error  = 0;
        } catch (const std::bad_alloc &) {
            consume_error = ENOMEM;
        } catch (...) {
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            consumer_active     = false;
            staging_slot & slot = slots[slot_index];
            free_slot_locked(slot, false);
            if (consume_status != llama_snapshot_status::ok) {
                sink_failure       = consume_status;
                sink_failure_error = consume_error;
                cancel_requested   = true;
                drop_ready_locked();
                if (worker_done) {
                    completion.operation_status = sink_failure;
                    completion.os_error         = sink_failure_error;
                }
                result.status = llama_snapshot_worker_status::sink_failed;
            } else {
                ++next_consume_index;
                ++statistics.read_consumed_chunks;
                result.status = llama_snapshot_worker_status::ok;
                result.index  = index;
                result.bytes  = size;
            }
        }
        cv.notify_all();
        return result;
    }

    llama_snapshot_worker_status cancel(uint64_t job_id) {
        std::lock_guard<std::mutex> lock(mutex);
        const llama_snapshot_worker_status status = active_status_locked(job_id);
        if (status != llama_snapshot_worker_status::ok) {
            return status;
        }
        if (terminal_locked()) {
            return llama_snapshot_worker_status::job_complete;
        }
        cancel_requested = true;
        drop_ready_locked();
        if (worker_done && job.kind == llama_snapshot_worker_job_kind::read) {
            completion.operation_status = llama_snapshot_status::cancelled;
            completion.os_error         = 0;
        }
        cv.notify_all();
        return llama_snapshot_worker_status::ok;
    }

    llama_snapshot_worker_completion poll(uint64_t job_id) const {
        std::lock_guard<std::mutex> lock(mutex);
        return completion_locked(job_id);
    }

    llama_snapshot_worker_completion wait(uint64_t job_id) {
        std::unique_lock<std::mutex> lock(mutex);
        const llama_snapshot_worker_status status = active_status_locked(job_id);
        if (status != llama_snapshot_worker_status::ok) {
            llama_snapshot_worker_completion result;
            result.status = status;
            return result;
        }
        cv.wait(lock, [&] { return stopping || terminal_locked(); });
        llama_snapshot_worker_completion result = completion_locked(job_id);
        if (!result.complete) {
            result.status = llama_snapshot_worker_status::stopped;
            return result;
        }
        if (result.operation_status == llama_snapshot_status::ok) {
            ++statistics.completed_jobs;
        } else if (result.operation_status == llama_snapshot_status::cancelled) {
            ++statistics.cancelled_jobs;
        } else {
            ++statistics.failed_jobs;
        }
        active = false;
        job    = job_spec{};
        reset_slots_locked();
        cv.notify_all();
        return result;
    }

    llama_snapshot_worker_stats stats() const {
        std::lock_guard<std::mutex> lock(mutex);
        llama_snapshot_worker_stats result = statistics;
        result.staging_buffers             = config.staging_buffers;
        result.staging_buffer_bytes        = config.store.chunk_payload_bytes;
        return result;
    }

    llama_snapshot_chunk_source_result acquire_write_chunk(
            uint64_t job_id, uint32_t index, uint64_t offset, uint64_t size) noexcept {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
            return stopping || !active || active_job_id != job_id || cancel_requested ||
                   source_failure != llama_snapshot_status::ok || find_ready_slot_locked(index) != slots.end();
        });
        if (stopping || !active || active_job_id != job_id || cancel_requested) {
            return { llama_snapshot_status::cancelled, nullptr, 0, 0 };
        }
        if (source_failure != llama_snapshot_status::ok) {
            return { source_failure, nullptr, 0, source_failure_error };
        }
        auto slot = find_ready_slot_locked(index);
        if (slot == slots.end() || slot->size != size ||
            offset != static_cast<uint64_t>(index) * config.store.chunk_payload_bytes) {
            return { llama_snapshot_status::invalid_argument, nullptr, 0, 0 };
        }
        slot->state = slot_state::io;
        return { llama_snapshot_status::ok, slot->bytes.data(), slot->size, 0 };
    }

    void release_write_chunk(uint64_t job_id, uint32_t index) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (active && active_job_id == job_id) {
                const auto slot = std::find_if(slots.begin(), slots.end(), [&](const staging_slot & value) {
                    return value.state == slot_state::io && value.index == index;
                });
                if (slot != slots.end()) {
                    free_slot_locked(*slot, false);
                    ++statistics.write_released_chunks;
                }
            }
        }
        cv.notify_all();
    }

    void run() noexcept {
        while (true) {
            uint64_t                       job_id = 0;
            llama_snapshot_worker_job_kind kind  = llama_snapshot_worker_job_kind::none;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stopping || (active && !worker_started); });
                if (stopping) {
                    return;
                }
                worker_started = true;
                job_id         = active_job_id;
                kind           = job.kind;
            }
            try {
                if (kind == llama_snapshot_worker_job_kind::write) {
                    run_write(job_id);
                } else {
                    run_read(job_id);
                }
            } catch (const std::bad_alloc &) {
                finish_worker(job_id, llama_snapshot_status::io_error, 0, ENOMEM, false);
            } catch (...) {
                finish_worker(job_id, llama_snapshot_status::io_error, 0, EIO, false);
            }
        }
    }

    void run_write(uint64_t job_id) {
        source_adapter source{ *this, job_id };
        const llama_snapshot_cancel_check cancelled = [this, job_id](uint32_t) {
            std::lock_guard<std::mutex> lock(mutex);
            return stopping || !active || active_job_id != job_id || cancel_requested;
        };
        const llama_snapshot_write_result result =
            store.write_generation_streamed(job.metadata, job.payload_bytes, source, cancelled, job.faults);
        finish_worker(job_id, result.status, result.generation, result.os_error, result.committed);
    }

    void run_read(uint64_t job_id) {
        const llama_snapshot_open_result opened = store.open_current(job.expected_identity);
        if (opened.status != llama_snapshot_status::ok) {
            finish_worker(job_id, opened.status, 0, opened.os_error, false);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!active || active_job_id != job_id) {
                return;
            }
            if (stopping || cancel_requested) {
                finish_worker_locked(
                    job_id, llama_snapshot_status::cancelled, opened.manifest.snapshot_generation, 0, false);
                cv.notify_all();
                return;
            }
            job.chunk_count            = static_cast<uint32_t>(opened.manifest.chunks.size());
            job.payload_bytes          = opened.manifest.total_payload_bytes;
            completion.generation      = opened.manifest.snapshot_generation;
            completion.total_payload_bytes = opened.manifest.total_payload_bytes;
        }
        for (uint32_t index = 0; index < opened.manifest.chunks.size(); ++index) {
            size_t slot_index = 0;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] {
                    return stopping || !active || active_job_id != job_id || cancel_requested ||
                           sink_failure != llama_snapshot_status::ok || find_free_slot_locked() != slots.end();
                });
                if (stopping || !active || active_job_id != job_id || cancel_requested ||
                    sink_failure != llama_snapshot_status::ok) {
                    const llama_snapshot_status status = sink_failure != llama_snapshot_status::ok ?
                                                              sink_failure :
                                                              llama_snapshot_status::cancelled;
                    const int error = sink_failure != llama_snapshot_status::ok ? sink_failure_error : 0;
                    finish_worker_locked(job_id, status, opened.manifest.snapshot_generation, error, false);
                    cv.notify_all();
                    return;
                }
                const auto free = find_free_slot_locked();
                slot_index      = static_cast<size_t>(free - slots.begin());
                free->state     = slot_state::io;
                free->index     = index;
                free->size      = opened.manifest.chunks[index].payload_bytes;
                occupy_slot_locked();
            }

            const llama_snapshot_read_into_result read = store.read_chunk_into(
                opened.manifest, index, slots[slot_index].bytes.data(), slots[slot_index].bytes.size());
            {
                std::lock_guard<std::mutex> lock(mutex);
                staging_slot & slot = slots[slot_index];
                if (stopping || !active || active_job_id != job_id || cancel_requested ||
                    sink_failure != llama_snapshot_status::ok) {
                    free_slot_locked(slot, true);
                    const llama_snapshot_status status = sink_failure != llama_snapshot_status::ok ?
                                                              sink_failure :
                                                              llama_snapshot_status::cancelled;
                    const int error = sink_failure != llama_snapshot_status::ok ? sink_failure_error : 0;
                    finish_worker_locked(job_id, status, opened.manifest.snapshot_generation, error, false);
                    cv.notify_all();
                    return;
                }
                if (read.status != llama_snapshot_status::ok) {
                    free_slot_locked(slot, true);
                    finish_worker_locked(
                        job_id, read.status, opened.manifest.snapshot_generation, read.os_error, false);
                    cv.notify_all();
                    return;
                }
                slot.size  = read.payload_bytes;
                slot.state = slot_state::ready;
                ++statistics.read_staged_chunks;
            }
            cv.notify_all();
        }
        finish_worker(job_id, llama_snapshot_status::ok, opened.manifest.snapshot_generation, 0, false);
    }

    void finish_worker(
            uint64_t job_id, llama_snapshot_status status, uint64_t generation, int os_error, bool committed) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            finish_worker_locked(job_id, status, generation, os_error, committed);
        }
        cv.notify_all();
    }

    void finish_worker_locked(
            uint64_t job_id, llama_snapshot_status status, uint64_t generation, int os_error, bool committed) {
        if (!active || active_job_id != job_id || worker_done) {
            return;
        }
        if (sink_failure != llama_snapshot_status::ok) {
            status   = sink_failure;
            os_error = sink_failure_error;
        }
        completion.operation_status = status;
        completion.generation       = generation;
        completion.os_error         = os_error;
        completion.committed        = committed;
        worker_done                 = true;
        if (status != llama_snapshot_status::ok) {
            drop_ready_locked();
        }
    }

    llama_snapshot_worker_status admission_status_locked() const {
        if (stopping || initialization_status == llama_snapshot_worker_status::stopped) {
            return llama_snapshot_worker_status::stopped;
        }
        if (initialization_status != llama_snapshot_worker_status::ok) {
            return initialization_status;
        }
        return active ? llama_snapshot_worker_status::busy : llama_snapshot_worker_status::ok;
    }

    llama_snapshot_worker_status active_status_locked(uint64_t job_id) const {
        if (stopping) {
            return llama_snapshot_worker_status::stopped;
        }
        if (!active || job_id == 0 || job_id != active_job_id) {
            return llama_snapshot_worker_status::stale_job;
        }
        return llama_snapshot_worker_status::ok;
    }

    void start_job_locked(job_spec prepared) {
        active             = true;
        worker_started     = false;
        worker_done        = false;
        cancel_requested   = false;
        producer_active    = false;
        consumer_active    = false;
        source_failure     = llama_snapshot_status::ok;
        source_failure_error = 0;
        sink_failure       = llama_snapshot_status::ok;
        sink_failure_error = 0;
        active_job_id      = next_job_id;
        next_job_id        = next_job_id == std::numeric_limits<uint64_t>::max() ? 1 : next_job_id + 1;
        next_stage_index   = 0;
        next_consume_index = 0;
        job                = std::move(prepared);
        completion         = {};
        completion.status  = llama_snapshot_worker_status::ok;
        completion.kind    = job.kind;
        completion.total_payload_bytes = job.payload_bytes;
        completion.operation_status    = llama_snapshot_status::invalid_argument;
        reset_slots_locked();
    }

    llama_snapshot_worker_completion completion_locked(uint64_t job_id) const {
        llama_snapshot_worker_completion result;
        const llama_snapshot_worker_status status = active_status_locked(job_id);
        if (status != llama_snapshot_worker_status::ok) {
            result.status = status;
            return result;
        }
        result          = completion;
        result.status   = llama_snapshot_worker_status::ok;
        result.complete = terminal_locked();
        return result;
    }

    bool terminal_locked() const {
        if (!worker_done || producer_active || consumer_active) {
            return false;
        }
        return job.kind != llama_snapshot_worker_job_kind::read || statistics.staging_in_use == 0;
    }

    std::vector<staging_slot>::iterator find_free_slot_locked() {
        return std::find_if(
            slots.begin(), slots.end(), [](const staging_slot & slot) { return slot.state == slot_state::free; });
    }

    std::vector<staging_slot>::iterator find_ready_slot_locked(uint32_t index) {
        return std::find_if(slots.begin(), slots.end(), [&](const staging_slot & slot) {
            return slot.state == slot_state::ready && slot.index == index;
        });
    }

    void occupy_slot_locked() {
        ++statistics.staging_in_use;
        statistics.staging_high_watermark =
            std::max(statistics.staging_high_watermark, statistics.staging_in_use);
    }

    void free_slot_locked(staging_slot & slot, bool dropped) {
        if (slot.state == slot_state::free) {
            return;
        }
        slot.state = slot_state::free;
        slot.index = 0;
        slot.size  = 0;
        if (statistics.staging_in_use > 0) {
            --statistics.staging_in_use;
        }
        if (dropped) {
            ++statistics.dropped_chunks;
        }
    }

    void drop_ready_locked() {
        for (staging_slot & slot : slots) {
            if (slot.state == slot_state::ready) {
                free_slot_locked(slot, true);
            }
        }
    }

    void reset_slots_locked() {
        for (staging_slot & slot : slots) {
            slot.state = slot_state::free;
            slot.index = 0;
            slot.size  = 0;
        }
        statistics.staging_in_use = 0;
    }

    llama_snapshot_worker_config       config;
    llama_snapshot_store               store;
    pid_t                              creator_pid;
    std::vector<staging_slot>          slots;
    std::unique_ptr<std::thread>       worker;
    mutable std::mutex                 mutex;
    std::condition_variable            cv;
    llama_snapshot_worker_status       initialization_status = llama_snapshot_worker_status::stopped;
    bool                               stopping               = false;
    bool                               active                 = false;
    bool                               worker_started         = false;
    bool                               worker_done            = false;
    bool                               cancel_requested       = false;
    bool                               producer_active        = false;
    bool                               consumer_active        = false;
    uint64_t                           next_job_id             = 1;
    uint64_t                           active_job_id           = 0;
    uint32_t                           next_stage_index        = 0;
    uint32_t                           next_consume_index      = 0;
    job_spec                           job;
    llama_snapshot_status              source_failure         = llama_snapshot_status::ok;
    int                                source_failure_error    = 0;
    llama_snapshot_status              sink_failure           = llama_snapshot_status::ok;
    int                                sink_failure_error      = 0;
    llama_snapshot_worker_completion   completion;
    llama_snapshot_worker_stats        statistics;
};

llama_snapshot_worker::llama_snapshot_worker(llama_snapshot_worker_config config) :
    state(std::make_unique<impl>(std::move(config))) {}

llama_snapshot_worker::~llama_snapshot_worker() {
    // Do not destroy copied mutex, condition-variable, or thread state in a
    // fork child. The address space is about to be replaced or discarded, and
    // no parent worker thread exists there to join.
    if (state && state->inherited_process()) {
        state.release();
    }
}

llama_snapshot_worker_begin_result llama_snapshot_worker::begin_write(
        const llama_snapshot_metadata & metadata,
        uint64_t                        total_payload_bytes,
        const llama_snapshot_faults &   faults) {
    if (state->inherited_process()) {
        llama_snapshot_worker_begin_result result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->begin_write(metadata, total_payload_bytes, faults);
}

llama_snapshot_worker_begin_result llama_snapshot_worker::begin_read(
        const llama_snapshot_identity & expected_identity) {
    if (state->inherited_process()) {
        llama_snapshot_worker_begin_result result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->begin_read(expected_identity);
}

llama_snapshot_worker_transfer_result llama_snapshot_worker::try_stage_next(
        uint64_t job_id, const llama_snapshot_stage_fill & fill) {
    if (state->inherited_process()) {
        llama_snapshot_worker_transfer_result result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->try_stage(job_id, fill);
}

llama_snapshot_worker_transfer_result llama_snapshot_worker::try_consume_next(
        uint64_t job_id, const llama_snapshot_stage_consume & consume) {
    if (state->inherited_process()) {
        llama_snapshot_worker_transfer_result result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->try_consume(job_id, consume);
}

llama_snapshot_worker_status llama_snapshot_worker::cancel(uint64_t job_id) {
    return state->inherited_process() ? llama_snapshot_worker_status::stopped : state->cancel(job_id);
}

llama_snapshot_worker_completion llama_snapshot_worker::poll(uint64_t job_id) const {
    if (state->inherited_process()) {
        llama_snapshot_worker_completion result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->poll(job_id);
}

llama_snapshot_worker_completion llama_snapshot_worker::wait(uint64_t job_id) {
    if (state->inherited_process()) {
        llama_snapshot_worker_completion result;
        result.status = llama_snapshot_worker_status::stopped;
        return result;
    }
    return state->wait(job_id);
}

llama_snapshot_worker_stats llama_snapshot_worker::stats() const {
    return state->inherited_process() ? llama_snapshot_worker_stats{} : state->stats();
}

#include "server-capture.h"

#include <atomic>
#include <limits>
#include <stdexcept>

namespace server_capture {

namespace {

static_assert(std::atomic<size_t>::is_always_lock_free, "ring indices must not use hidden locks");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "ring counters must not use hidden locks");

struct alignas(64) ring_index {
    std::atomic<size_t> value{ 0 };
};

struct alignas(64) ring_counter {
    std::atomic<uint64_t> value{ 0 };
};

}  // namespace

struct spsc_ring::impl {
    explicit impl(size_t requested_capacity) :
        usable_capacity(requested_capacity),
        storage_size(requested_capacity + 1),
        records(std::make_unique<cycle_observation[]>(storage_size)) {}

    size_t next(size_t index) const noexcept {
        ++index;
        return index == storage_size ? 0 : index;
    }

    const size_t usable_capacity;
    const size_t storage_size;

    std::unique_ptr<cycle_observation[]> records;
    ring_index                           producer;
    ring_index                           consumer;
    ring_counter                         pushed;
    ring_counter                         popped;
    ring_counter                         dropped;
};

spsc_ring::spsc_ring(size_t capacity) {
    if (capacity == 0 || capacity == std::numeric_limits<size_t>::max()) {
        throw std::invalid_argument("capture ring capacity must be positive and representable");
    }
    data = std::make_unique<impl>(capacity);
}

spsc_ring::~spsc_ring() = default;

bool spsc_ring::try_push(const cycle_observation & observation) noexcept {
    const size_t producer = data->producer.value.load(std::memory_order_relaxed);
    const size_t next     = data->next(producer);
    if (next == data->consumer.value.load(std::memory_order_acquire)) {
        data->dropped.value.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    data->records[producer] = observation;
    data->producer.value.store(next, std::memory_order_release);
    data->pushed.value.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool spsc_ring::try_pop(cycle_observation & observation) noexcept {
    const size_t consumer = data->consumer.value.load(std::memory_order_relaxed);
    if (consumer == data->producer.value.load(std::memory_order_acquire)) {
        return false;
    }

    observation = data->records[consumer];
    data->consumer.value.store(data->next(consumer), std::memory_order_release);
    data->popped.value.fetch_add(1, std::memory_order_relaxed);
    return true;
}

size_t spsc_ring::capacity() const noexcept {
    return data->usable_capacity;
}

ring_stats spsc_ring::stats() const noexcept {
    const size_t producer = data->producer.value.load(std::memory_order_acquire);
    const size_t consumer = data->consumer.value.load(std::memory_order_acquire);
    const size_t size     = producer >= consumer ? producer - consumer : data->storage_size - (consumer - producer);

    return {
        data->usable_capacity,
        size,
        data->pushed.value.load(std::memory_order_relaxed),
        data->popped.value.load(std::memory_order_relaxed),
        data->dropped.value.load(std::memory_order_relaxed),
    };
}

}  // namespace server_capture

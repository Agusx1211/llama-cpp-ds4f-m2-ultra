#include "server-capture.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace server_capture;

#undef assert
#define assert(expr)                                                                              \
    do {                                                                                          \
        if (!(expr)) {                                                                            \
            std::cerr << "check failed at " << __FILE__ << ':' << __LINE__ << ": " #expr << '\n'; \
            std::abort();                                                                         \
        }                                                                                         \
    } while (false)

static cycle_observation observation(uint64_t sequence) {
    cycle_observation result;
    result.request_id                    = 0x1234000000000000ULL + sequence;
    result.committed_position            = sequence * 5;
    result.scheduler_epoch               = 99;
    result.monotonic_ns                  = 1000000 + sequence;
    result.cycle_sequence                = static_cast<uint32_t>(sequence);
    result.target_correction_or_bonus_id = static_cast<int32_t>(120000 + sequence % 9000);
    result.draft_time_us                 = 10;
    result.verify_time_us                = 20;
    result.scheduler_time_us             = 3;
    result.scheduled_decode_width        = 5;
    result.verifier_geometry             = 5;
    result.proposal_count                = 5;
    result.accepted_prefix_length        = 3;
    result.first_rejection               = 3;
    result.active_mode                   = dspark_mode::adaptive_depth_one;
    for (size_t i = 0; i < MAX_PROPOSAL_TOKENS; ++i) {
        result.proposal_token_ids[i]     = static_cast<int32_t>(sequence + i);
        result.selected_probabilities[i] = 0.5f + 0.01f * static_cast<float>(i);
        result.raw_confidences[i]        = 0.75f - 0.02f * static_cast<float>(i);
    }
    return result;
}

static void assert_same(const cycle_observation & actual, const cycle_observation & expected) {
    assert(actual.request_id == expected.request_id);
    assert(actual.committed_position == expected.committed_position);
    assert(actual.scheduler_epoch == expected.scheduler_epoch);
    assert(actual.monotonic_ns == expected.monotonic_ns);
    assert(actual.schema_version == expected.schema_version);
    assert(actual.cycle_sequence == expected.cycle_sequence);
    assert(actual.proposal_token_ids == expected.proposal_token_ids);
    assert(actual.selected_probabilities == expected.selected_probabilities);
    assert(actual.raw_confidences == expected.raw_confidences);
    assert(actual.target_correction_or_bonus_id == expected.target_correction_or_bonus_id);
    assert(actual.draft_time_us == expected.draft_time_us);
    assert(actual.verify_time_us == expected.verify_time_us);
    assert(actual.scheduler_time_us == expected.scheduler_time_us);
    assert(actual.scheduled_decode_width == expected.scheduled_decode_width);
    assert(actual.verifier_geometry == expected.verifier_geometry);
    assert(actual.proposal_count == expected.proposal_count);
    assert(actual.accepted_prefix_length == expected.accepted_prefix_length);
    assert(actual.first_rejection == expected.first_rejection);
    assert(actual.active_mode == expected.active_mode);
    assert(actual.bypass == expected.bypass);
    assert(actual.flags == expected.flags);
}

static void test_capacity_fifo_wrap_and_drop_newest() {
    spsc_ring         ring(3);
    cycle_observation output;

    assert(ring.capacity() == 3);
    assert(!ring.try_pop(output));
    assert(ring.try_push(observation(1)));
    assert(ring.try_push(observation(2)));
    assert(ring.try_push(observation(3)));
    assert(!ring.try_push(observation(4)));

    auto stats = ring.stats();
    assert(stats.capacity == 3 && stats.size_approx == 3);
    assert(stats.pushed == 3 && stats.popped == 0 && stats.dropped == 1);

    assert(ring.try_pop(output));
    assert_same(output, observation(1));
    assert(ring.try_push(observation(5)));
    for (uint64_t sequence : { 2ULL, 3ULL, 5ULL }) {
        assert(ring.try_pop(output));
        assert_same(output, observation(sequence));
    }
    assert(!ring.try_pop(output));

    stats = ring.stats();
    assert(stats.size_approx == 0);
    assert(stats.pushed == 4 && stats.popped == 4 && stats.dropped == 1);
}

static void test_one_slot_and_invalid_capacity() {
    bool rejected = false;
    try {
        spsc_ring ring(0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        spsc_ring ring(std::numeric_limits<size_t>::max());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);

    spsc_ring         ring(1);
    cycle_observation output;
    assert(ring.try_push(observation(7)));
    assert(!ring.try_push(observation(8)));
    assert(ring.try_pop(output));
    assert_same(output, observation(7));
}

static void test_concurrent_spsc_integrity() {
    constexpr uint64_t count = 1000000;
    spsc_ring          ring(1024);
    std::atomic<bool>  start{ false };
    std::atomic<bool>  failed{ false };

    std::thread consumer([&]() {
        while (!start.load(std::memory_order_acquire)) {
        }
        cycle_observation output;
        for (uint64_t expected = 0; expected < count;) {
            if (!ring.try_pop(output)) {
                continue;
            }
            const auto expected_observation = observation(expected);
            if (output.cycle_sequence != static_cast<uint32_t>(expected) ||
                output.request_id != expected_observation.request_id ||
                output.proposal_token_ids != expected_observation.proposal_token_ids) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            ++expected;
        }
    });

    start.store(true, std::memory_order_release);
    for (uint64_t sequence = 0; sequence < count;) {
        if (ring.try_push(observation(sequence))) {
            ++sequence;
        }
    }
    consumer.join();

    const auto stats = ring.stats();
    assert(!failed.load(std::memory_order_relaxed));
    assert(stats.pushed == count && stats.popped == count);
    assert(stats.size_approx == 0);
}

int main() {
    test_capacity_fifo_wrap_and_drop_newest();
    test_one_slot_and_invalid_capacity();
    test_concurrent_spsc_integrity();
    std::cout << "server capture ring tests passed\n";
    return 0;
}

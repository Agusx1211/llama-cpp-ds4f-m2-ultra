#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "server-prefill.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace server_prefill;

namespace {

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

coordinator_config small_config(uint32_t max_lease_chunks = 4) {
    coordinator_config config;
    config.alignment_tokens           = 4;
    config.idle_chunk_tokens          = 8;
    config.active_decode_chunk_tokens = 6;
    config.active_fast_chunk_tokens   = 4;
    config.max_lease_chunks           = max_lease_chunks;
    return config;
}

candidate request(uint64_t id, server_scheduler::lane priority, uint64_t arrival_us, uint64_t cohort_id = 0) {
    return { id, cohort_id == 0 ? id : cohort_id, priority, arrival_us };
}

chunk_lease stage_limited(coordinator &   owner,
                          uint64_t        request_id,
                          uint64_t        begin,
                          uint64_t        total,
                          decode_activity activity = {}) {
    const auto limit = owner.limit_chunk(request_id, begin, total, 64, activity);
    require(static_cast<bool>(limit), "chunk limit exists");
    const auto lease = owner.stage_chunk(request_id, begin, limit.end_token, total);
    require(static_cast<bool>(lease), "chunk stages");
    return lease;
}

void append_committed(std::vector<uint32_t> & counts, const chunk_lease & lease) {
    require(lease.end_token <= counts.size(), "committed range is bounded");
    for (uint64_t token = lease.begin_token; token < lease.end_token; ++token) {
        ++counts[token];
    }
}

void require_exactly_once(const std::vector<uint32_t> & counts, const char * message) {
    for (uint32_t count : counts) {
        require(count == 1, message);
    }
}

void test_token_budgets_are_independent_and_aligned() {
    coordinator_config config;
    config.alignment_tokens           = 4;
    config.idle_chunk_tokens          = 16;
    config.active_decode_chunk_tokens = 8;
    config.active_fast_chunk_tokens   = 4;
    coordinator owner(config);

    require(static_cast<bool>(owner.select_owner({ request(1, server_scheduler::lane::low, 1) })),
            "select budget owner");
    require(owner.limit_chunk(1, 0, 32, 32, {}).end_token == 16, "idle budget is independent");
    require(owner.limit_chunk(1, 0, 32, 32, { true, server_scheduler::lane::normal }).end_token == 8,
            "active decode budget is independent");
    require(owner.limit_chunk(1, 0, 32, 32, { true, server_scheduler::lane::fast }).end_token == 4,
            "active fast budget is independent");
    require(owner.limit_chunk(1, 0, 32, 3, { true, server_scheduler::lane::fast }).end_token == 3,
            "non-final work advances within a batch too small for the next anchor");
    require(owner.limit_chunk(1, 3, 32, 4, { true, server_scheduler::lane::fast }).end_token == 4,
            "an unaligned head advances only to the next anchor");
}

void test_one_owner_until_exact_commit() {
    coordinator            owner(small_config());
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::normal, 10),
        request(2, server_scheduler::lane::normal, 20),
    };
    require(owner.select_owner(candidates).request_id == 1, "oldest cohort owns prefill");

    const chunk_lease first = stage_limited(owner, 1, 0, 20);
    candidates.push_back(request(3, server_scheduler::lane::fast, 30));
    require(owner.select_owner(candidates).request_id == 1, "in-flight ownership cannot transfer");
    require(!owner.stage_chunk(3, 0, 4, 8), "a second cohort cannot overlap staged work");
    require(owner.commit_chunk(first), "exact owner chunk commits");
    require(owner.select_owner(candidates).request_id == 3, "higher lane takes the committed boundary");
    require(!owner.commit_chunk(first), "a committed range cannot be replayed");
}

void test_unaligned_boundary_retains_owner_then_yields() {
    coordinator            owner(small_config(1));
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::low, 1),
    };
    require(owner.select_owner(candidates).request_id == 1, "select unaligned owner");

    const auto semantic = owner.stage_chunk(1, 0, 3, 20);
    require(semantic && !semantic.yield_boundary, "semantic micro-chunk is not a yield boundary");
    require(owner.commit_chunk(semantic), "semantic micro-chunk commits");

    candidates.push_back(request(2, server_scheduler::lane::fast, 2));
    require(owner.select_owner(candidates).request_id == 1, "fast work waits for an aligned committed anchor");
    const auto aligned = stage_limited(owner, 1, 3, 20, { true, server_scheduler::lane::fast });
    require(aligned.end_token == 4 && aligned.yield_boundary, "owner reaches the next aligned anchor");
    require(owner.commit_chunk(aligned), "aligned anchor commits");
    require(owner.select_owner(candidates).request_id == 2, "fast work takes ownership after aligned commit");
}

void test_multichunk_fairness_yields_and_resumes() {
    coordinator            owner(small_config(2));
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::normal, 1),
        request(2, server_scheduler::lane::normal, 2),
    };
    require(owner.select_owner(candidates).request_id == 1, "first cohort starts multi-chunk lease");

    auto lease = stage_limited(owner, 1, 0, 20, { true, server_scheduler::lane::fast });
    require(owner.commit_chunk(lease), "first lease chunk commits");
    require(owner.select_owner(candidates).request_id == 1, "owner retains first lease chunk");
    lease = stage_limited(owner, 1, lease.end_token, 20, { true, server_scheduler::lane::fast });
    require(owner.commit_chunk(lease), "second lease chunk commits");
    require(owner.select_owner(candidates).request_id == 2, "same-lane waiter gets bounded fairness turn");

    const auto second = stage_limited(owner, 2, 0, 3);
    require(second.completes_prompt && owner.commit_chunk(second), "second cohort completes its final tail");
    candidates.pop_back();
    require(owner.select_owner(candidates).request_id == 1, "yielded cohort resumes from committed progress");
}

void test_parent_cohort_cancellation_is_boundary_safe() {
    coordinator     owner(small_config());
    const candidate parent = request(10, server_scheduler::lane::normal, 1, 100);
    require(owner.select_owner({ parent }).cohort_id == 100, "parent owns the whole cohort");
    const auto lease = stage_limited(owner, 10, 0, 12);
    require(!owner.cancel_cohort(100), "cancellation cannot tear down an in-flight cohort chunk");
    require(owner.commit_chunk(lease), "cohort chunk reaches committed boundary");
    require(owner.cancel_cohort(100), "cancellation releases ownership at that boundary");
    require(owner.snapshot().cohort_id == 0, "cancelled family leaves no owner");

    require(!owner.select_owner({ parent, request(11, server_scheduler::lane::normal, 1, 100) }),
            "multiple sibling prefill candidates for one cohort fail closed");
    require(owner.select_owner({ request(20, server_scheduler::lane::low, 2, 200) }).cohort_id == 200,
            "another family can own after cancellation");
}

void test_yield_resume_never_duplicates_prompt_tokens() {
    coordinator            owner(small_config(1));
    std::vector<uint32_t>  low_counts(13, 0);
    std::vector<uint32_t>  fast_counts(6, 0);
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::low, 1),
    };

    require(owner.select_owner(candidates).request_id == 1, "low prefill begins");
    auto     lease         = stage_limited(owner, 1, 0, low_counts.size());
    uint64_t low_committed = lease.end_token;
    append_committed(low_counts, lease);
    require(owner.commit_chunk(lease), "low prefix commits once");

    candidates.push_back(request(2, server_scheduler::lane::fast, 2));
    require(owner.select_owner(candidates).request_id == 2, "fast prefill takes boundary");
    lease = stage_limited(owner, 2, 0, fast_counts.size());
    append_committed(fast_counts, lease);
    require(owner.commit_chunk(lease), "fast prompt commits once");
    candidates.pop_back();

    require(owner.select_owner(candidates).request_id == 1, "low prefill resumes");
    while (low_committed < low_counts.size()) {
        lease = stage_limited(owner, 1, low_committed, low_counts.size());
        require(lease.begin_token == low_committed, "resume starts at exact committed token");
        append_committed(low_counts, lease);
        low_committed = lease.end_token;
        require(owner.commit_chunk(lease), "resumed low range commits once");
        if (low_committed < low_counts.size()) {
            require(owner.select_owner(candidates).request_id == 1, "single remaining cohort reacquires");
        }
    }
    require_exactly_once(low_counts, "every low prompt token is decoded exactly once");
    require_exactly_once(fast_counts, "every fast prompt token is decoded exactly once");
}

void test_committed_cursor_rejects_gaps_overlaps_and_replay() {
    coordinator            owner(small_config(1));
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::low, 1),
    };
    require(owner.select_owner(candidates).request_id == 1, "select cached-offset owner");

    const auto aborted = owner.stage_chunk(1, 4, 8, 20);
    require(aborted && owner.abort_chunk(aborted), "aborted range preserves its starting cursor");
    require(!owner.stage_chunk(1, 5, 8, 20), "gap after abort is rejected");
    require(!owner.stage_chunk(1, 3, 8, 20), "overlap after abort is rejected");
    const auto first = owner.stage_chunk(1, 4, 8, 20);
    require(first && owner.commit_chunk(first), "same range can retry and commit after abort");

    candidates.push_back(request(2, server_scheduler::lane::fast, 2));
    require(owner.select_owner(candidates).request_id == 2, "cursor survives handoff to fast cohort");
    const auto fast = owner.stage_chunk(2, 0, 4, 4);
    require(fast && owner.commit_chunk(fast), "fast cohort completes between low chunks");
    candidates.pop_back();
    require(owner.select_owner(candidates).request_id == 1, "low cohort resumes with retained cursor");

    require(!owner.stage_chunk(1, 4, 8, 20), "committed range replay is rejected after resume");
    require(!owner.stage_chunk(1, 9, 12, 20), "forward gap is rejected after resume");
    require(!owner.stage_chunk(1, 7, 12, 20), "overlap is rejected after resume");
    const auto resumed = owner.stage_chunk(1, 8, 12, 20);
    require(resumed && owner.commit_chunk(resumed), "exact committed cursor resumes successfully");

    require(owner.select_owner({ request(3, server_scheduler::lane::fast, 3) }).request_id == 3,
            "absent yielded request is pruned at selection");
    const auto replacement = owner.stage_chunk(3, 6, 10, 20);
    require(replacement && owner.commit_chunk(replacement), "new request seeds its own cached-prefix cursor");
    require(owner.select_owner({ request(1, server_scheduler::lane::low, 4) }).request_id == 1,
            "pruned request identifier can represent a later live request");
    require(static_cast<bool>(owner.stage_chunk(1, 6, 10, 20)),
            "pruned cursor does not reject a later cached-prefix start");
}

void test_text_media_text_ranges_are_transactional() {
    coordinator           owner(small_config());
    std::vector<uint32_t> counts(12, 0);
    const auto            first_request = request(1, server_scheduler::lane::normal, 1);
    require(owner.select_owner({ first_request }).request_id == 1, "select text-media-text owner");

    const auto text_before = owner.stage_chunk(1, 0, 4, counts.size());
    require(text_before && owner.commit_chunk(text_before), "text before media commits");
    append_committed(counts, text_before);

    const auto media = owner.stage_chunk(1, 4, 7, counts.size());
    require(media && !media.yield_boundary, "media range is preflighted at its exact logical tokens");
    require(owner.commit_chunk(media), "successful media bookkeeping commits its exact range");
    append_committed(counts, media);

    const auto text_after = owner.stage_chunk(1, 7, counts.size(), counts.size());
    require(text_after && owner.commit_chunk(text_after), "text after media starts at the committed media cursor");
    append_committed(counts, text_after);
    require_exactly_once(counts, "text-media-text logical tokens commit exactly once");

    const auto failing_request = request(2, server_scheduler::lane::low, 2, 200);
    require(owner.select_owner({ failing_request }).request_id == 2, "select media failure owner");
    const auto prefix = owner.stage_chunk(2, 0, 4, 12);
    require(prefix && owner.commit_chunk(prefix), "failure fixture text prefix commits");

    const auto failing_media = owner.stage_chunk(2, 4, 7, 12);
    require(static_cast<bool>(failing_media), "failing media range is preflighted before mutation");
    require(!owner.cancel_cohort(200), "family cancellation cannot interleave with in-flight media");
    require(owner.abort_chunk(failing_media), "failed media returns to its prior committed cursor");
    const auto retry = owner.stage_chunk(2, 4, 7, 12);
    require(retry && owner.abort_chunk(retry), "failed media can retry only the same exact range");
    require(owner.cancel_cohort(200), "family cancellation releases the aborted media owner");
    require(!owner.commit_chunk(failing_media), "a failed media lease cannot commit after cancellation");

    coordinator            boundary_owner(small_config());
    std::vector<candidate> boundary_candidates = {
        request(10, server_scheduler::lane::low, 10),
    };
    require(boundary_owner.select_owner(boundary_candidates).request_id == 10, "select media boundary owner");
    const auto unaligned_text = boundary_owner.stage_chunk(10, 0, 3, 12);
    require(unaligned_text && boundary_owner.commit_chunk(unaligned_text),
            "text stops immediately before an aligned media endpoint");
    boundary_candidates.push_back(request(11, server_scheduler::lane::fast, 11));
    require(boundary_owner.select_owner(boundary_candidates).request_id == 10,
            "fast work waits until media reaches the next committed anchor");
    const auto aligned_media = boundary_owner.stage_chunk(10, 3, 4, 12);
    require(aligned_media && aligned_media.yield_boundary && boundary_owner.commit_chunk(aligned_media),
            "media reaches an aligned non-final handoff boundary");
    require(boundary_owner.select_owner(boundary_candidates).request_id == 11,
            "selection after aligned media commit hands ownership to fast work");
}

void test_media_live_policy_rejects_tail_and_clears_failed_mutation() {
    const auto final_media           = plan_media_chunk(4, 8, 8);
    bool       backend_mutated       = false;
    bool       zero_batch_completion = false;
    if (final_media.decode_allowed) {
        backend_mutated       = true;
        zero_batch_completion = true;
    }
    require(!final_media.decode_allowed && !backend_mutated && !zero_batch_completion,
            "final media is rejected before backend mutation or zero-batch prompt completion");

    const auto followed_media = plan_media_chunk(4, 7, 8);
    require(followed_media.decode_allowed && followed_media.clear_backend_on_failure,
            "non-final media carries an explicit backend cleanup obligation");
    bool target_prompt_state = true;
    bool draft_prompt_state  = true;
    if (followed_media.clear_backend_on_failure) {
        target_prompt_state = false;
        draft_prompt_state  = false;
    }
    require(!target_prompt_state && !draft_prompt_state,
            "injected post-mutation failure clears both target and draft prompt state");
}

void test_parent_activation_waits_for_complete_retry_split_commit() {
    coordinator multi_owner(small_config());
    const auto multi_candidates = std::vector<candidate>{
        request(10, server_scheduler::lane::normal, 10),
    };
    require(multi_owner.select_owner(multi_candidates).request_id == 10,
            "select active-decode multi-chunk owner");
    const decode_activity active_decode = { true, server_scheduler::lane::normal };
    const auto first_limit = multi_owner.limit_chunk(10, 0, 12, 64, active_decode);
    require(first_limit && first_limit.end_token == 4,
            "active decode produces an aligned non-final chunk");
    const auto first_lease = multi_owner.stage_chunk(10, 0, first_limit.end_token, 12);
    staged_batch_lifecycle first_batch;
    require(first_batch.begin(first_lease, 1, 4) && first_batch.record_decoded_view(1, 4) &&
                first_batch.decoded_complete() && !first_batch.ready_for_family_preparation(),
            "decoded-complete non-final lease does not enter parent fan-out");
    int ordinary_postdecode = 0;
    auto first_result = finish_decoded_view(
        first_batch,
        1,
        4,
        [&]() { ++ordinary_postdecode; },
        [&](const chunk_lease & exact) { return multi_owner.commit_chunk(exact); },
        [&](const chunk_lease &, bool activate_parent) {
            require(!activate_parent, "non-final exact commit cannot activate parent family");
        });
    require(first_result.valid && first_result.committed && ordinary_postdecode == 1,
            "non-final lease runs ordinary real postdecode and exact commit");

    require(multi_owner.select_owner(multi_candidates).request_id == 10,
            "multi-chunk owner resumes after non-final exact commit");
    const auto second_lease = stage_limited(multi_owner, 10, 4, 12, active_decode);
    require(!second_lease.completes_prompt, "active decode retains another non-final chunk");
    require(multi_owner.commit_chunk(second_lease), "second non-final chunk commits exactly");
    require(multi_owner.select_owner(multi_candidates).request_id == 10,
            "owner reaches final chunk under active decode");
    const auto final_lease = stage_limited(multi_owner, 10, 8, 12, active_decode);
    staged_batch_lifecycle final_batch;
    require(final_lease.completes_prompt && final_batch.begin(final_lease, 0, 4) &&
                final_batch.record_decoded_view(0, 4) && final_batch.ready_for_family_preparation(),
            "only decoded-complete final lease enables parent fan-out preparation");
    require(multi_owner.abort_chunk(final_lease), "final predicate fixture aborts exact staged lease");

    coordinator owner(small_config());
    require(owner.select_owner({ request(1, server_scheduler::lane::normal, 1) }).request_id == 1,
            "select retry-split parent");

    const auto lease = owner.stage_chunk(1, 0, 4, 4);
    require(lease && lease.completes_prompt, "stage final parent prompt range");

    staged_batch_lifecycle batch;
    require(batch.begin(lease, 3, 4), "begin final parent batch lifecycle");

    bool family_postdecode_before_range = false;
    require(batch.record_decoded_view(0, 3), "unrelated leading view records as a no-op");
    const auto unrelated = finish_decoded_view(
        batch,
        0,
        3,
        [&]() { family_postdecode_before_range = true; },
        [&](const chunk_lease &) { return false; },
        [&](const chunk_lease &, bool) {});
    require(unrelated.valid && !unrelated.committed && !family_postdecode_before_range,
            "unrelated decoded view keeps normal postdecode without entering the staged family");

    // An injected pressure retry has no successful target mutation or decoded
    // view to record. The first successful retry processes only half the range.
    require(!batch.ready_to_commit() && !batch.take_parent_activation(),
            "pressure retry cannot activate a child");
    require(batch.record_decoded_view(3, 2), "first retry-split view succeeds");
    require(batch.decoded_tokens() == 2 && !batch.decoded_complete() && !batch.take_parent_activation(),
            "partial parent KV remains private");

    std::vector<std::string> events;
    std::vector<std::string> buffered_outputs;
    std::vector<std::string> external_outputs;
    auto partial = finish_decoded_view(
        batch,
        3,
        2,
        [&]() {
            events.emplace_back("post-partial");
            buffered_outputs.emplace_back("parent-progress");
        },
        [&](const chunk_lease &) {
            events.emplace_back("unexpected-commit");
            return false;
        },
        [&](const chunk_lease &, bool) { events.emplace_back("unexpected-finalize"); });
    require(partial.valid && !partial.committed && events == std::vector<std::string>{ "post-partial" },
            "partial live view runs postdecode without commit or activation");
    require(external_outputs.empty(), "partial family progress remains private before exact commit");

    require(batch.record_decoded_view(5, 2), "remaining retry-split view succeeds");
    require(batch.decoded_complete() && !batch.ready_to_commit() && !batch.take_parent_activation(),
            "complete decode still waits for successful actual postdecode");
    auto complete = finish_decoded_view(
        batch,
        5,
        2,
        [&]() {
            events.emplace_back("post-final");
            buffered_outputs.emplace_back("parent-token");
            buffered_outputs.emplace_back("child-1-token");
            buffered_outputs.emplace_back("child-2-token");
        },
        [&](const chunk_lease & exact) {
            require(external_outputs.empty(), "no family output is externally visible before commit");
            events.emplace_back("commit");
            return owner.commit_chunk(exact);
        },
        [&](const chunk_lease &, bool activate_parent) {
            require(activate_parent, "final parent commit requests activation");
            events.emplace_back("activate");
            for (const std::string & output : buffered_outputs) {
                external_outputs.push_back(output);
                events.emplace_back("publish-" + output);
            }
        });
    require(complete.valid && complete.committed,
            "complete live view commits and activates after actual postdecode");
    const std::vector<std::string> expected_events = {
        "post-partial", "post-final", "commit", "activate",
        "publish-parent-progress", "publish-parent-token",
        "publish-child-1-token", "publish-child-2-token",
    };
    require(events == expected_events && external_outputs == buffered_outputs,
            "live seam buffers ordered family output until commit then activates and publishes once");

    const std::vector<std::string> failure_points = {
        "parent-sample", "parent-result", "parent-spec",
        "child-1-sample", "child-1-result", "child-1-spec",
        "child-2-sample", "child-2-result", "child-2-spec",
    };
    for (const std::string & failure_point : failure_points) {
        coordinator failing_owner(small_config());
        require(failing_owner.select_owner({ request(2, server_scheduler::lane::normal, 2) }).request_id == 2,
                "select injected family postdecode failure parent");
        const auto failing_lease = failing_owner.stage_chunk(2, 0, 4, 4);
        staged_batch_lifecycle failing_batch;
        require(failing_batch.begin(failing_lease, 0, 4) && failing_batch.record_decoded_view(0, 4),
                "decode complete family before injected postdecode failure");

        bool commit_called   = false;
        bool finalize_called = false;
        bool failure_escaped = false;
        std::vector<std::string> staged;
        std::vector<std::string> published;
        try {
            finish_decoded_view(
                failing_batch,
                0,
                4,
                [&]() {
                    for (const std::string & point : failure_points) {
                        if (point == failure_point) {
                            throw std::runtime_error("injected family postdecode failure");
                        }
                        if (point.find("result") != std::string::npos) {
                            staged.push_back(point);
                        }
                    }
                },
                [&](const chunk_lease &) {
                    commit_called = true;
                    return true;
                },
                [&](const chunk_lease &, bool) {
                    finalize_called = true;
                    published = staged;
                });
        } catch (const std::runtime_error &) {
            failure_escaped = true;
        }
        require(failure_escaped && !commit_called && !finalize_called && published.empty() &&
                    failing_batch.abort_plan().abort_coordinator,
                "every parent/child sampling result and spec failure escapes before commit or publication");
        require(failing_owner.abort_chunk(failing_lease), "failed family postdecode leaves exact lease abortable");
    }

    coordinator activation_owner(small_config());
    require(activation_owner.select_owner({ request(3, server_scheduler::lane::normal, 3) }).request_id == 3,
            "select injected activation failure parent");
    const auto activation_lease = activation_owner.stage_chunk(3, 0, 4, 4);
    staged_batch_lifecycle activation_batch;
    require(activation_batch.begin(activation_lease, 0, 4) && activation_batch.record_decoded_view(0, 4),
            "decode complete family before activation failure");
    bool activation_failure_escaped = false;
    std::vector<std::string> activation_external;
    try {
        finish_decoded_view(
            activation_batch,
            0,
            4,
            []() {},
            [&](const chunk_lease & exact) { return activation_owner.commit_chunk(exact); },
            [&](const chunk_lease &, bool) {
                throw std::runtime_error("injected activation failure");
            });
    } catch (const std::runtime_error &) {
        activation_failure_escaped = true;
    }
    const auto committed_failure = activation_batch.abort_plan();
    require(activation_failure_escaped && !committed_failure.abort_coordinator &&
                committed_failure.clear_prompt_state && activation_external.empty(),
            "activation failure stays guarded after commit and fails the family closed before publication");

    coordinator publication_owner(small_config());
    require(publication_owner.select_owner({ request(4, server_scheduler::lane::normal, 4) }).request_id == 4,
            "select injected publication failure parent");
    const auto publication_lease = publication_owner.stage_chunk(4, 0, 4, 4);
    staged_batch_lifecycle publication_batch;
    require(publication_batch.begin(publication_lease, 0, 4) && publication_batch.record_decoded_view(0, 4),
            "decode complete family before publication failure");
    std::vector<std::string> publication_external;
    bool publication_failure_escaped = false;
    try {
        finish_decoded_view(
            publication_batch,
            0,
            4,
            []() {},
            [&](const chunk_lease & exact) {
                require(publication_external.empty(), "publication buffer is private at commit");
                return publication_owner.commit_chunk(exact);
            },
            [&](const chunk_lease &, bool) {
                publication_external.emplace_back("parent");
                throw std::runtime_error("injected queue handoff failure");
            });
    } catch (const std::runtime_error &) {
        publication_failure_escaped = true;
    }
    const auto publication_failure = publication_batch.abort_plan();
    require(publication_failure_escaped && !publication_failure.abort_coordinator &&
                publication_failure.clear_prompt_state &&
                publication_external == std::vector<std::string>{ "parent" },
            "post-commit queue failure remains guarded with ordered fail-closed publication");

    coordinator release_owner(small_config());
    require(release_owner.select_owner({ request(5, server_scheduler::lane::normal, 5) }).request_id == 5,
            "select injected terminal release failure parent");
    const auto release_lease = release_owner.stage_chunk(5, 0, 4, 4);
    staged_batch_lifecycle release_batch;
    require(release_batch.begin(release_lease, 0, 4) && release_batch.record_decoded_view(0, 4),
            "decode complete family before terminal release failure");
    std::vector<std::string> release_external;
    bool release_failure_escaped = false;
    try {
        finish_decoded_view(
            release_batch,
            0,
            4,
            []() {},
            [&](const chunk_lease & exact) {
                require(release_external.empty(), "terminal result buffer is private at commit");
                return release_owner.commit_chunk(exact);
            },
            [&](const chunk_lease &, bool) {
                release_external.emplace_back("terminal");
                throw std::runtime_error("injected terminal release callback failure");
            });
    } catch (const std::runtime_error &) {
        release_failure_escaped = true;
    }
    const auto release_failure = release_batch.abort_plan();
    require(release_failure_escaped && !release_failure.abort_coordinator &&
                release_failure.clear_prompt_state &&
                release_external == std::vector<std::string>{ "terminal" },
            "terminal release failure cannot escape the committed family guard");
}

void test_incomplete_mutation_abort_clears_all_prompt_state() {
    struct injected_prompt_state {
        bool target  = true;
        bool draft   = true;
        bool logical = true;

        void apply(const staged_batch_abort_plan & plan) {
            if (plan.clear_prompt_state) {
                target = false;
                draft = false;
                logical = false;
            }
        }

        bool reusable() const {
            return target || draft || logical;
        }
    };

    auto stage_after_prefix = []() {
        coordinator owner(small_config());
        require(owner.select_owner({ request(1, server_scheduler::lane::normal, 1) }).request_id == 1,
                "select injected failure owner");
        const auto prefix = owner.stage_chunk(1, 0, 4, 12);
        require(prefix && owner.commit_chunk(prefix), "commit failure fixture prefix");
        const auto staged = owner.stage_chunk(1, 4, 8, 12);
        require(static_cast<bool>(staged), "stage failure fixture range");
        return std::make_pair(std::move(owner), staged);
    };

    {
        auto [owner, lease] = stage_after_prefix();
        staged_batch_lifecycle batch;
        require(batch.begin(lease, 0, 4), "begin post-stage predecode failure lifecycle");
        require(batch.owns_family_task(1) && batch.owns_family_task(2, 1) && !batch.owns_family_task(2),
                "live iterate seam classifies staged owner and child as one failure family");

        bool failure_escaped = false;
        try {
            try {
                throw std::runtime_error("injected init_sampler failure");
            } catch (const std::runtime_error &) {
                if (batch.owns_family_task(1)) {
                    throw;
                }
            }
        } catch (const std::runtime_error &) {
            failure_escaped = true;
        }
        const auto plan = batch.abort_plan();
        injected_prompt_state state;
        state.apply(plan);
        require(failure_escaped && plan.abort_coordinator && !state.reusable(),
                "post-stage predecode failure routes to outer family abort and clear");
        require(owner.abort_chunk(lease), "post-stage predecode failure aborts exact lease");
        require(!owner.stage_chunk(1, 5, 8, 12), "predecode failure does not advance committed cursor");
        const auto retry = owner.stage_chunk(1, 4, 8, 12);
        require(retry && owner.abort_chunk(retry), "predecode failure retries at prior committed boundary");
    }

    {
        auto [owner, lease] = stage_after_prefix();
        staged_batch_lifecycle batch;
        require(batch.begin(lease, 0, 4), "begin common-spec failure lifecycle");
        require(static_cast<bool>(batch), "target mutation leaves staged lifecycle in flight");

        const auto plan = batch.abort_plan();
        injected_prompt_state state;
        state.apply(plan);
        require(plan.abort_coordinator && !state.reusable(),
                "common-spec failure clears target draft and logical prompt state");
        require(owner.abort_chunk(lease), "common-spec failure aborts exact lease");
        require(!owner.stage_chunk(1, 5, 8, 12), "failure does not advance the committed cursor");
        const auto retry = owner.stage_chunk(1, 4, 8, 12);
        require(retry && owner.abort_chunk(retry), "failure cursor remains at the prior committed boundary");
    }

    {
        auto [owner, lease] = stage_after_prefix();
        staged_batch_lifecycle batch;
        require(batch.begin(lease, 0, 4), "begin postdecode failure lifecycle");
        require(batch.record_decoded_view(0, 2),
                "partial retry-split view mutates and decodes before postdecode failure");

        bool commit_called   = false;
        bool activate_called = false;
        bool failure_escaped = false;
        try {
            finish_decoded_view(
                batch,
                0,
                2,
                []() { throw std::runtime_error("injected progress failure"); },
                [&](const chunk_lease &) {
                    commit_called = true;
                    return true;
                },
                [&](const chunk_lease &, bool) { activate_called = true; });
        } catch (const std::runtime_error &) {
            failure_escaped = true;
        }

        const auto plan = batch.abort_plan();
        injected_prompt_state state;
        injected_prompt_state unrelated_state;
        state.apply(plan);
        require(failure_escaped && !commit_called && !activate_called && plan.abort_coordinator && !state.reusable(),
                "partial-view postdecode failure clears target draft and logical prompt state");
        require(unrelated_state.reusable(), "staged-family postdecode failure leaves unrelated slot state intact");
        require(owner.abort_chunk(lease), "postdecode failure aborts exact lease");
        require(!owner.stage_chunk(1, 6, 8, 12), "partial view does not advance the committed cursor");
        const auto retry = owner.stage_chunk(1, 4, 8, 12);
        require(retry && owner.abort_chunk(retry), "partial-view cursor remains at the committed prefix");
    }
}

void test_small_batches_make_progress_then_recover_alignment() {
    coordinator_config config;
    config.alignment_tokens           = 8;
    config.idle_chunk_tokens          = 16;
    config.active_decode_chunk_tokens = 16;
    config.active_fast_chunk_tokens   = 8;
    config.max_lease_chunks           = 1;
    coordinator            owner(config);
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::low, 1),
    };
    require(owner.select_owner(candidates).request_id == 1, "select small-batch owner");

    auto limit = owner.limit_chunk(1, 0, 32, 3, {});
    require(static_cast<bool>(limit), "first small batch has a bounded limit");
    auto lease = owner.stage_chunk(1, 0, limit.end_token, 32);
    require(lease && lease.end_token == 3 && !lease.yield_boundary,
            "batch smaller than alignment advances to a bounded unaligned endpoint");
    require(owner.commit_chunk(lease), "first small batch commits");

    candidates.push_back(request(2, server_scheduler::lane::fast, 2));
    require(owner.select_owner(candidates).request_id == 1, "late fast waits while small batches approach anchor");
    limit = owner.limit_chunk(1, 3, 32, 3, {});
    require(limit && limit.end_token == 6, "second small batch keeps bounded forward progress");
    lease = owner.stage_chunk(1, 3, limit.end_token, 32);
    require(lease && owner.commit_chunk(lease), "second small batch commits");

    require(owner.select_owner(candidates).request_id == 1, "unaligned owner retains the lease");
    limit = owner.limit_chunk(1, 6, 32, 3, {});
    require(limit && limit.end_token == 8, "small batch recovers the next absolute aligned anchor");
    lease = owner.stage_chunk(1, 6, limit.end_token, 32);
    require(lease && lease.yield_boundary && owner.commit_chunk(lease), "recovered aligned boundary commits");
    require(owner.select_owner(candidates).request_id == 2, "fast work takes the recovered aligned boundary");
}

void test_three_cohort_round_robin_survives_arrival_and_removal() {
    coordinator            owner(small_config(1));
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::normal, 1),
        request(2, server_scheduler::lane::normal, 2),
        request(3, server_scheduler::lane::normal, 3),
    };
    std::vector<uint64_t> progress(5, 0);

    const std::vector<uint64_t> expected = { 1, 2, 3, 1, 2, 3 };
    for (uint64_t request_id : expected) {
        require(owner.select_owner(candidates).request_id == request_id,
                "same-lane fairness advances around the cohort ring");
        const auto lease =
            stage_limited(owner, request_id, progress[request_id], 64, { true, server_scheduler::lane::fast });
        progress[request_id] = lease.end_token;
        require(owner.commit_chunk(lease), "round-robin cohort chunk commits");
    }
    require(progress[1] > 0 && progress[2] > 0 && progress[3] > 0, "three long same-lane cohorts all make progress");

    candidates.erase(candidates.begin() + 2);  // current cohort 3 departs at its committed boundary
    candidates.push_back(request(4, server_scheduler::lane::normal, 4));
    require(owner.select_owner(candidates).request_id == 4,
            "new arrival follows the removed current cohort's persistent fairness cursor");
    auto lease  = stage_limited(owner, 4, progress[4], 64, { true, server_scheduler::lane::fast });
    progress[4] = lease.end_token;
    require(owner.commit_chunk(lease), "new arrival commits its fairness turn");

    candidates.erase(candidates.begin() + 1);  // cohort 2 departs while waiting
    require(owner.select_owner(candidates).request_id == 1,
            "waiter removal wraps deterministically to the oldest eligible cohort");
}

void test_weighted_cross_lane_service_is_bounded_and_work_conserving() {
    const auto             config = small_config(4);
    coordinator            owner(config);
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::fast, 1),
        request(2, server_scheduler::lane::fast, 2),
        request(3, server_scheduler::lane::normal, 3),
        request(4, server_scheduler::lane::low, 4),
    };
    std::vector<uint64_t>                              progress(5, 0);
    std::array<uint64_t, server_scheduler::lane_count> lane_chunks   = {};
    size_t                                             last_low_turn = 0;
    size_t                                             max_low_gap   = 0;

    constexpr size_t turns = 420;
    for (size_t turn = 1; turn <= turns; ++turn) {
        const auto selection = owner.select_owner(candidates);
        require(static_cast<bool>(selection), "sustained mixed lanes always select an owner");
        const size_t lane_index = static_cast<size_t>(selection.priority);
        ++lane_chunks[lane_index];
        if (selection.priority == server_scheduler::lane::low) {
            max_low_gap   = std::max(max_low_gap, turn - last_low_turn);
            last_low_turn = turn;
        }

        const auto lease = stage_limited(owner, selection.request_id, progress[selection.request_id], 4096,
                                         { true, server_scheduler::lane::fast });
        progress[selection.request_id] = lease.end_token;
        require(owner.commit_chunk(lease), "weighted mixed-lane chunk commits");
    }
    max_low_gap = std::max(max_low_gap, turns + 1 - last_low_turn);

    const size_t low_index    = static_cast<size_t>(server_scheduler::lane::low);
    const size_t normal_index = static_cast<size_t>(server_scheduler::lane::normal);
    const size_t fast_index   = static_cast<size_t>(server_scheduler::lane::fast);
    require(lane_chunks[low_index] > 0 && lane_chunks[normal_index] > lane_chunks[low_index] &&
                lane_chunks[fast_index] > lane_chunks[normal_index],
            "sustained 16:4:1 service gives every lane progress in weighted order");
    const size_t configured_bound =
        static_cast<size_t>(config.max_lease_chunks) *
        (config.lane_weights[low_index] + config.lane_weights[normal_index] + config.lane_weights[fast_index]);
    require(max_low_gap <= configured_bound, "low-lane service gap is bounded by the configured weighted round");

    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [](const candidate & current) { return current.priority == server_scheduler::lane::fast; }),
        candidates.end());
    const uint64_t low_before_removal = progress[4];
    for (size_t turn = 0; turn < 32; ++turn) {
        const auto selection = owner.select_owner(candidates);
        require(selection && selection.priority != server_scheduler::lane::fast,
                "removed fast lane reserves no service");
        const auto lease = stage_limited(owner, selection.request_id, progress[selection.request_id], 4096,
                                         { true, server_scheduler::lane::fast });
        progress[selection.request_id] = lease.end_token;
        require(owner.commit_chunk(lease), "post-removal service remains live");
    }
    require(progress[4] > low_before_removal, "low lane keeps progressing after fast-lane removal");

    candidates = { request(4, server_scheduler::lane::low, 4) };
    for (size_t turn = 0; turn < 8; ++turn) {
        const auto selection = owner.select_owner(candidates);
        require(selection && selection.request_id == 4, "single non-empty low lane is fully work-conserving");
        const auto lease = stage_limited(owner, 4, progress[4], 4096, { true, server_scheduler::lane::fast });
        progress[4]      = lease.end_token;
        require(owner.commit_chunk(lease), "work-conserving low chunk commits");
    }
}

void test_late_fast_arrival_has_finite_committed_token_bound() {
    const auto             config = small_config(8);
    coordinator            owner(config);
    std::vector<candidate> candidates = {
        request(1, server_scheduler::lane::low, 1),
    };
    require(owner.select_owner(candidates).request_id == 1, "low owner selected before fast arrival");
    const auto in_flight = stage_limited(owner, 1, 0, 64);

    candidates.push_back(request(2, server_scheduler::lane::fast, 2));
    require(owner.select_owner(candidates).request_id == 1, "late fast arrival cannot split an in-flight chunk");
    const uint64_t committed_after_arrival = in_flight.end_token - in_flight.begin_token;
    require(owner.commit_chunk(in_flight), "bounded low chunk commits");
    require(committed_after_arrival <= config.idle_chunk_tokens,
            "late fast handoff is bounded by configured committed prompt tokens");
    require(owner.select_owner(candidates).request_id == 2, "fast cohort owns the next committed boundary");

    const auto fast_budget = owner.limit_chunk(2, 0, 64, 64, { true, server_scheduler::lane::fast });
    require(fast_budget && fast_budget.end_token <= config.active_fast_chunk_tokens,
            "active fast decode bounds each later mixed prefill chunk");
}

}  // namespace

int main() {
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        { "independent aligned budgets",          test_token_budgets_are_independent_and_aligned                  },
        { "single owner until exact commit",      test_one_owner_until_exact_commit                               },
        { "unaligned boundary retention",         test_unaligned_boundary_retains_owner_then_yields               },
        { "multichunk fairness yield and resume", test_multichunk_fairness_yields_and_resumes                     },
        { "parent cohort cancellation",           test_parent_cohort_cancellation_is_boundary_safe                },
        { "no duplicate prompt tokens",           test_yield_resume_never_duplicates_prompt_tokens                },
        { "cursor rejects gaps overlaps replay",  test_committed_cursor_rejects_gaps_overlaps_and_replay          },
        { "transactional text media text",        test_text_media_text_ranges_are_transactional                   },
        { "media live policy cleanup",            test_media_live_policy_rejects_tail_and_clears_failed_mutation  },
        { "parent activation after exact commit", test_parent_activation_waits_for_complete_retry_split_commit   },
        { "incomplete mutation abort cleanup",    test_incomplete_mutation_abort_clears_all_prompt_state          },
        { "small batch alignment recovery",       test_small_batches_make_progress_then_recover_alignment         },
        { "three cohort round robin",             test_three_cohort_round_robin_survives_arrival_and_removal      },
        { "weighted cross lane service",          test_weighted_cross_lane_service_is_bounded_and_work_conserving },
        { "late fast committed-token bound",      test_late_fast_arrival_has_finite_committed_token_bound         },
    };

    try {
        for (const auto & test : tests) {
            test.second();
            std::printf("PASS: %s\n", test.first);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }

    std::printf("PASS: %zu server prefill tests\n", tests.size());
    return 0;
}

#pragma once

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <vector>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"

using json = nlohmann::ordered_json;

enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // per-request parameters for chat parsing
    common_chat_parser_params chat_parser_params;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_parser_params chat_parser_params;
    common_chat_msg chat_msg;
    std::string generated_text; // append new chunks of generated text here
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(const common_chat_parser_params & chat_parser_params);

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    enum class trusted_lane : uint8_t {
        low = 0,
        normal,
        fast,
    };

    // Populated only by trusted server-side ingress code. Client JSON is not
    // consulted for these fields.
    struct scheduling_metadata {
        trusted_lane lane = trusted_lane::normal;
        uint64_t arrival_us = 0;  // zero asks server_queue to stamp its clock
        uint64_t virtual_runtime_us = 0;
        int64_t debt_us = 0;
        uint64_t cached_prompt_tokens = 0;
        uint64_t predicted_prefill_us = 0;
        uint64_t predicted_cache_restore_us = 0;
        uint64_t predicted_decode_us = 0;
        uint64_t predicted_gpu_us = 0;
        uint64_t predicted_memory_bytes = 0;
        uint64_t predicted_output_tokens = 0;
        // Zero selects the durable runtime's bounded server default. These are
        // trusted server-side durations and are never read from client JSON.
        uint64_t queue_timeout_us = 0;
        // Absolute run backstop for a bound request (total runtime cap).
        uint64_t run_timeout_us   = 0;
        // Stall threshold for progress-based run expiry: a bound request is
        // expired when its slot records no forward progress (decoded token,
        // accepted draft tokens, prefill ubatch) for this long.
        uint64_t run_stall_timeout_us = 0;
    } scheduling;

    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot
        copy.scheduling = scheduling;

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    // the task will be moved into queue, then onto slots
    // however, the state must be kept by caller (e.g., HTTP thread)
    task_result_state create_state() const {
        return task_result_state(params.chat_parser_params);
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }
};

// Keep family matching shared by the live server and its bound-slot tests.
// Releasing a parent must include every parallel child that names it.
template <typename SlotRange>
size_t release_server_task_family_slots(SlotRange & slots, int id_target) {
    size_t released = 0;
    for (auto & slot : slots) {
        if (slot.task && (slot.task->id == id_target || slot.task->id_parent == id_target)) {
            slot.release();
            ++released;
        }
    }
    return released;
}

struct result_timings {
    int32_t cache_n = -1;

    int32_t prompt_n = -1;
    double prompt_ms = 0.0;
    double prompt_per_token_ms = 0.0;
    double prompt_per_second = 0.0;

    int32_t predicted_n = -1;
    double predicted_ms = 0.0;
    double predicted_per_token_ms = 0.0;
    double predicted_per_second = 0.0;

    // Optional speculative metrics - only included when > 0
    int32_t draft_n = 0;
    int32_t draft_n_accepted = 0;

    json to_json() const;
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    std::string content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    result_timings timings;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    std::string  content;
    llama_tokens tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    result_timings timings;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // Only used by ERROR_TYPE_OVERLOADED. The serialized HTTP layer clamps
    // this to the public Retry-After contract before emitting the header.
    uint32_t retry_after_seconds = 0;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

struct server_task_result_metrics : server_task_result {
    int n_idle_slots;
    int n_processing_slots;
    int n_tasks_deferred;
    int64_t t_start;

    // TODO: somehow reuse server_metrics in the future, instead of duplicating the fields
    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    uint64_t kv_physical_pressure_total   = 0;
    uint64_t kv_physical_pressure_retries = 0;
    uint64_t kv_physical_pressure_victims = 0;

    uint64_t n_draft_tokens_total      = 0;
    uint64_t n_draft_accepted_total    = 0;
    uint64_t n_draft_verif_steps_total = 0;
    std::vector<uint64_t> n_accepted_per_pos_total;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // SSD tier: when set, the state blobs and the checkpoints live in this
    // file and the in-RAM copies above are empty; only prompt.tokens stays
    // resident so the best-match scan keeps working across both tiers
    std::string file;
    size_t size_disk = 0;

    // m2-dashboard bookkeeping (introspection only, never drives eviction).
    // A RAM hit consumes the entry (load() moves it into the slot), so hit
    // counts accumulate mostly on SSD-tier entries, which keep a slim index
    // entry across hits.
    uint64_t dash_id          = 0; // stable id from the allocation counter
    int64_t  dash_created_us  = 0;
    int64_t  dash_last_hit_us = 0; // 0 = never hit
    uint32_t dash_hits        = 0;
    uint64_t dash_req         = 0; // request whose save created it (0 = rebuilt from disk)

    bool on_disk() const { return !file.empty(); }

    // RAM footprint of the entry (disk entries hold no blobs in RAM)
    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

//
// SSD tier ("prompt cache disk tier") file format and identity
//
// One `.lcpc` file per spilled prompt state. A file holds the prompt tokens (so
// the index can be rebuilt at startup), the serialized target *and* draft
// sequence states, and the context checkpoints. Byte order is the host order of
// the target machine (M2 Ultra).
//
// Two independent guards decide whether a file may be restored:
//
//   1. `version` - the schema of the file itself. Bump PCACHE_DISK_VERSION for
//      any layout change here; old files are then rejected and deleted.
//   2. `fingerprint` - the identity of everything the *payload* depends on
//      (model artifacts, state-serialization ABI, KV representation, layout
//      sensitive runtime parameters). See server_prompt_cache_fingerprint().
//
// Integrity is checked with two hashes: `hash_header` covers the header itself
// (so a garbage header can never drive an allocation) and `hash_payload` covers
// every byte after it. rescan_disk() verifies the header hash plus all declared
// lengths against the real file size; the payload hash is verified by
// load_from_disk(), which is the only path that actually reads the blobs.

static constexpr uint32_t PCACHE_DISK_MAGIC   = 0x4350434Cu; // "LCPC"

// v1: unversioned fingerprint, no checksums, no bounds checks (2026-08-07)
// v2: fingerprint schema v2, header+payload hashes, checked lengths, atomic
//     temp-file publication (2026-08-08)
static constexpr uint32_t PCACHE_DISK_VERSION = 2u;

struct pcache_disk_header {
    uint32_t magic;         // PCACHE_DISK_MAGIC
    uint32_t version;       // PCACHE_DISK_VERSION
    uint64_t fingerprint;   // server_prompt_cache_fingerprint()
    uint64_t size_main;     // bytes of serialized target sequence state
    uint64_t size_drft;     // bytes of serialized draft sequence state (0 = none)
    uint64_t n_tokens;      // prompt tokens stored before the state blobs
    uint64_t n_checkpoints; // context checkpoints stored after the state blobs
    uint64_t size_payload;  // bytes after this header; must equal file_size - sizeof(header)
    uint64_t hash_payload;  // hash over those size_payload bytes
    uint64_t hash_header;   // hash over the preceding bytes of this header (must stay last)
};

static_assert(sizeof(pcache_disk_header) == 72, "pcache_disk_header must stay packed and stable");

// Hard ceilings applied to every file-controlled length *before* it is used to
// size an allocation. They are deliberately far above any healthy value: the
// tight bound is always "must fit in the bytes this file actually has", these
// only stop a pathological header from being believed at all.
static constexpr uint64_t PCACHE_MAX_TOKENS      = 1ull << 24;       //  16.7 M tokens
static constexpr uint64_t PCACHE_MAX_BLOB_BYTES  = 64ull << 30;      //  64 GiB per blob
static constexpr uint64_t PCACHE_MAX_CHECKPOINTS = 4096ull;          //  vs. n_ctx_checkpoints (default 32)
static constexpr uint64_t PCACHE_MAX_FILE_BYTES  = 512ull << 30;     //  512 GiB per entry

// Fast non-cryptographic hash used for both header and payload integrity. This
// is an accident/truncation detector on a local cache directory, not a defense
// against a crafted collision. It is on the spill and restore hot paths, so it
// processes 32 bytes per iteration with four independent lanes.
uint64_t server_pcache_hash(const void * data, size_t size, uint64_t seed = 0);

// Integrity hash of a header: covers every byte before `hash_header`. Exposed so
// tests can build deliberately hostile-but-well-sealed files.
uint64_t server_pcache_header_hash(const pcache_disk_header & hdr);

// Everything the serialized state depends on. Anything that changes the bytes
// llama_state_seq_get_data_ext() produces, or changes how they are interpreted
// on restore, belongs in here - a value that is missing is a value that cannot
// invalidate a stale `.lcpc` file.
struct server_prompt_cache_fingerprint_inputs {
    // schema/ABI generation
    uint32_t schema_version    = PCACHE_DISK_VERSION;       // this file format
    uint32_t state_seq_version = LLAMA_STATE_SEQ_VERSION;   // upstream container version
    uint32_t state_seq_layout  = LLAMA_STATE_SEQ_LAYOUT_M2; // fork payload layout generation
    uint32_t session_version   = LLAMA_SESSION_VERSION;     // upstream whole-context version

    // model artifact identity; `probe` is a cheap content hash over the head and
    // tail of the file, which catches an in-place edit that preserved both the
    // size and the mtime (a plain rebuild of the same artifact does not)
    struct artifact {
        std::string path;
        uint64_t    size  = 0;
        int64_t     mtime = 0;
        uint64_t    probe = 0;
        uint8_t     present = 0;
    };

    artifact model_tgt;
    artifact model_dft; // present = 0 when speculative decoding runs without a draft model

    // KV representation: the state blobs are raw KV cell payloads, so the cache
    // types and the unified/SWA layout decide how they must be read back
    int32_t kv_type_k_tgt = 0;
    int32_t kv_type_v_tgt = 0;
    int32_t kv_type_k_dft = 0;
    int32_t kv_type_v_dft = 0;
    int32_t flash_attn_type = 0;
    uint8_t kv_unified = 0;
    uint8_t swa_full   = 0;

    // layout-sensitive runtime geometry
    uint32_t n_ctx_tgt     = 0;
    uint32_t n_seq_max_tgt = 0;
    uint32_t n_ctx_dft     = 0;
    uint32_t n_seq_max_dft = 0;
    int32_t  n_parallel    = 0;

    // speculative configuration: it sizes the draft context's recurrent state
    // (need_n_rs_seq) and decides what goes into each checkpoint's data_spec
    std::vector<int32_t> spec_types;
    int32_t spec_n_max = 0;
};

// Stable 64-bit identity of the inputs above (FNV-1a over a canonical encoding).
uint64_t server_prompt_cache_fingerprint(const server_prompt_cache_fingerprint_inputs & in);

// Cheap content probe for a model artifact: hashes the first and last 64 KiB
// together with the size. Returns 0 when the file cannot be read.
uint64_t server_prompt_cache_file_probe(const std::string & path);

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens,
                        const std::string & disk_dir, int32_t disk_limit_gib, uint64_t disk_fingerprint) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;

        this->disk_dir         = disk_dir;
        this->limit_disk       = disk_limit_gib < 0 ? 0 : 1024ull*1024ull*1024ull*(uint64_t) disk_limit_gib;
        this->disk_fingerprint = disk_fingerprint;

        // a stored prompt can never be longer than the context it was captured
        // from; keep the admission bound separate from the (mutable, softened)
        // eviction limit so file parsing has a fixed ceiling
        this->disk_max_tokens = limit_tokens > 0 ? std::min<uint64_t>(limit_tokens, PCACHE_MAX_TOKENS)
                                                 : PCACHE_MAX_TOKENS;

        if (!this->disk_dir.empty()) {
            rescan_disk();
        }
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // SSD tier (empty disk_dir = disabled); limit_disk in bytes, 0 = no limit
    std::string disk_dir;
    size_t      limit_disk       = 0;
    uint64_t    disk_fingerprint = 0;
    uint64_t    disk_seq         = 0;
    uint64_t    disk_max_tokens  = PCACHE_MAX_TOKENS; // upper bound on a file's declared n_tokens

    size_t size() const;

    size_t n_tokens() const;

    size_t size_disk_total() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();

    // SSD tier internals
    bool spill_to_disk(server_prompt_cache_state & state);
    bool load_from_disk(server_prompt_cache_state & state);
    void rescan_disk();

    // evict the oldest RAM-resident entry: spill it to the SSD tier when
    // enabled, drop it otherwise; false if no RAM-resident entry remains
    bool evict_oldest_ram();

    // erase an entry and unlink its backing file (if any)
    std::list<server_prompt_cache_state>::iterator drop_entry(std::list<server_prompt_cache_state>::iterator it);

    // m2-dashboard introspection (main thread only; see server-dashboard-bus.h).
    // What the most recent load() consult did, read by get_available_slot to
    // emit one cache_restore event with an unambiguous source.
    struct dash_load_report {
        int      source   = 4; // server_dashboard::restore_code numeric value (4 = miss)
        uint64_t n_tokens = 0; // prefix tokens restored (entry hit) or kept (resident)
        uint64_t n_bytes  = 0; // state bytes restored into the slot (entry hits only)
    };
    dash_load_report dash_last_load;

    // monotone counters for hit-rate/traffic reporting
    uint64_t dash_id_next         = 1;
    uint64_t dash_lookups         = 0;
    uint64_t dash_hits_entry      = 0;
    uint64_t dash_hits_resident   = 0;
    uint64_t dash_misses          = 0;
    uint64_t dash_saves           = 0;
    uint64_t dash_spills          = 0;
    uint64_t dash_disk_loads      = 0;
    uint64_t dash_drops           = 0;
    uint64_t dash_bytes_saved     = 0;
    uint64_t dash_bytes_spilled   = 0;
    uint64_t dash_bytes_disk_load = 0;

    // rebuild and publish the occupancy snapshot served by
    // GET /m2-dashboard/cache-state (called from update() and rescan_disk())
    void publish_dashboard_state();
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};

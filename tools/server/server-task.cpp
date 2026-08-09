#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"
#include "server-dashboard-bus.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//

// result_timings
//

json result_timings::to_json() const {
    json base = {
        {"cache_n",                cache_n},

        {"prompt_n",               prompt_n},
        {"prompt_ms",              prompt_ms},
        {"prompt_per_token_ms",    prompt_per_token_ms},
        {"prompt_per_second",      prompt_per_second},

        {"predicted_n",            predicted_n},
        {"predicted_ms",           predicted_ms},
        {"predicted_per_token_ms", predicted_per_token_ms},
        {"predicted_per_second",   predicted_per_second},
    };

    if (draft_n > 0) {
        base["draft_n"] = draft_n;
        base["draft_n_accepted"] = draft_n_accepted;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // nlohmann::json converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             timings.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (timings.prompt_n >= 0) {
        deltas.back().push_back({"timings", timings.to_json()});
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (timings.prompt_n >= 0) {
        server_sent_events.back().at("data").push_back({"timings", timings.to_json()});
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (timings.prompt_n > 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (timings.prompt_n >= 0) {
        res.push_back({"timings", timings.to_json()});
    }
    if (is_progress) {
        res.push_back({"prompt_progress", progress.to_json()});
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (timings.prompt_n >= 0) {
            last_json.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            last_json.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (timings.prompt_n >= 0) {
            data.push_back({"timings", timings.to_json()});
        }
        if (is_progress) {
            data.push_back({"prompt_progress", progress.to_json()});
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_OVERLOADED) {
        res["retry_after"] = bounded_retry_after_seconds(retry_after_seconds);
    }
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_metrics::to_json() {
    return json {
        { "idle",                            n_idle_slots },
        { "processing",                      n_processing_slots },
        { "deferred",                        n_tasks_deferred },
        { "t_start",                         t_start },

        { "n_prompt_tokens_processed_total", n_prompt_tokens_processed_total },
        { "t_tokens_generation_total",       t_tokens_generation_total },
        { "n_tokens_predicted_total",        n_tokens_predicted_total },
        { "t_prompt_processing_total",       t_prompt_processing_total },

        { "n_tokens_max",                    n_tokens_max },

        { "n_prompt_tokens_processed",       n_prompt_tokens_processed },
        { "t_prompt_processing",             t_prompt_processing },
        { "n_tokens_predicted",              n_tokens_predicted },
        { "t_tokens_generation",             t_tokens_generation },

        { "n_decode_total",                  n_decode_total },
        { "n_busy_slots_total",              n_busy_slots_total },
        { "kv_physical_pressure_total",       kv_physical_pressure_total },
        { "kv_physical_pressure_retries",     kv_physical_pressure_retries },
        { "kv_physical_pressure_victims",     kv_physical_pressure_victims },

        { "n_draft_tokens_total",            n_draft_tokens_total },
        { "n_draft_accepted_total",          n_draft_accepted_total },
        { "n_draft_verif_steps_total",       n_draft_verif_steps_total },
        { "n_accepted_per_pos_total",        n_accepted_per_pos_total },

        { "slots",                           slots_data },
    };
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//

// SSD tier file format: see the pcache_disk_header block in server-task.h for
// the layout, the two guards (schema version + fingerprint) and the integrity
// model. Everything below treats the file as untrusted input: no length read
// from a file may size an allocation before it has been proven to fit inside
// the bytes the file actually has.

static constexpr uint64_t PCACHE_HASH_SEED_HDR = 0x9E3779B97F4A7C15ull;

// hash primes (xxh64)
static constexpr uint64_t PCACHE_HP1 = 0x9E3779B185EBCA87ull;
static constexpr uint64_t PCACHE_HP2 = 0xC2B2AE3D27D4EB4Full;
static constexpr uint64_t PCACHE_HP3 = 0x165667B19E3779F9ull;
static constexpr uint64_t PCACHE_HP4 = 0x85EBCA77C2B2AE63ull;
static constexpr uint64_t PCACHE_HP5 = 0x27D4EB2F165667C5ull;

static inline uint64_t pcache_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t pcache_load_u64(const uint8_t * p) {
    uint64_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline uint64_t pcache_round(uint64_t acc, uint64_t v) {
    return pcache_rotl(acc + v*PCACHE_HP2, 31)*PCACHE_HP1;
}

static inline uint64_t pcache_merge(uint64_t acc, uint64_t v) {
    return (acc ^ (pcache_rotl(v*PCACHE_HP2, 31)*PCACHE_HP1))*PCACHE_HP1 + PCACHE_HP4;
}

uint64_t server_pcache_hash(const void * data, size_t size, uint64_t seed) {
    const uint8_t *       p   = static_cast<const uint8_t *>(data);
    const uint8_t * const end = p + size;

    uint64_t h;

    if (size >= 32) {
        uint64_t v1 = seed + PCACHE_HP1 + PCACHE_HP2;
        uint64_t v2 = seed + PCACHE_HP2;
        uint64_t v3 = seed;
        uint64_t v4 = seed - PCACHE_HP1;

        const uint8_t * const limit = end - 32;
        do {
            v1 = pcache_round(v1, pcache_load_u64(p +  0));
            v2 = pcache_round(v2, pcache_load_u64(p +  8));
            v3 = pcache_round(v3, pcache_load_u64(p + 16));
            v4 = pcache_round(v4, pcache_load_u64(p + 24));
            p += 32;
        } while (p <= limit);

        h = pcache_rotl(v1, 1) + pcache_rotl(v2, 7) + pcache_rotl(v3, 12) + pcache_rotl(v4, 18);
        h = pcache_merge(h, v1);
        h = pcache_merge(h, v2);
        h = pcache_merge(h, v3);
        h = pcache_merge(h, v4);
    } else {
        h = seed + PCACHE_HP5;
    }

    h += (uint64_t) size;

    while (p + 8 <= end) {
        h  = pcache_rotl(h ^ (pcache_rotl(pcache_load_u64(p)*PCACHE_HP2, 31)*PCACHE_HP1), 27)*PCACHE_HP1 + PCACHE_HP4;
        p += 8;
    }

    while (p < end) {
        h  = pcache_rotl(h ^ (*p * PCACHE_HP5), 11)*PCACHE_HP1;
        p += 1;
    }

    h ^= h >> 33;
    h *= PCACHE_HP2;
    h ^= h >> 29;
    h *= PCACHE_HP3;
    h ^= h >> 32;

    return h;
}

uint64_t server_pcache_header_hash(const pcache_disk_header & hdr) {
    return server_pcache_hash(&hdr, offsetof(pcache_disk_header, hash_header), PCACHE_HASH_SEED_HDR);
}

// a + b with overflow detection
static inline bool pcache_add_checked(uint64_t & a, uint64_t b) {
    if (a > UINT64_MAX - b) {
        return false;
    }
    a += b;
    return true;
}

#ifndef _WIN32
static void pcache_fsync(const std::string & path, bool is_dir) {
    const int fd = ::open(path.c_str(), is_dir ? (O_RDONLY | O_DIRECTORY) : O_RDONLY);
    if (fd < 0) {
        return;
    }
    // note: on macOS this does not force the drive's own write cache (that needs
    // F_FULLFSYNC, which is far too slow for the spill path). It is enough to
    // order the data before the rename against a process crash; a power cut can
    // still leave a truncated tail, which the payload hash rejects on load.
    (void) ::fsync(fd);
    (void) ::close(fd);
}
#else
static void pcache_fsync(const std::string &, bool) {}
#endif

// Sequential writer: streams the payload while chaining a hash over exactly the
// chunks the reader will read back, in the same order and with the same sizes.
struct pcache_writer {
    std::ofstream out;

    uint64_t payload_size = 0;
    uint64_t payload_hash = 0;

    bool ok = true;

    bool write_raw(const void * data, size_t size) {
        if (!ok) {
            return false;
        }
        if (size > 0) {
            if (!out.write(static_cast<const char *>(data), (std::streamsize) size)) {
                ok = false;
                return false;
            }
            payload_hash = server_pcache_hash(data, size, payload_hash);
        }
        payload_size += size;
        return true;
    }

    template <typename T>
    bool write_pod(const T & v) {
        return write_raw(&v, sizeof(v));
    }

    bool write_blob(const std::vector<uint8_t> & v) {
        const uint64_t n = v.size();
        return write_pod(n) && write_raw(v.data(), (size_t) n);
    }
};

// Sequential reader bounded by the payload length declared in a header that has
// already been validated against the real file size. `left` is the ground truth:
// nothing is allocated or read past it.
struct pcache_reader {
    std::ifstream in;

    uint64_t left = 0;
    uint64_t hash = 0;

    bool ok = true;

    bool read_raw(void * data, uint64_t size) {
        if (!ok) {
            return false;
        }
        if (size > left) {
            ok = false;
            return false;
        }
        if (size > 0) {
            if (!in.read(static_cast<char *>(data), (std::streamsize) size)) {
                ok = false;
                return false;
            }
            hash = server_pcache_hash(data, (size_t) size, hash);
        }
        left -= size;
        return true;
    }

    template <typename T>
    bool read_pod(T & v) {
        return read_raw(&v, sizeof(v));
    }

    // resize-from-file, but only after the declared length is proven to fit in
    // both the configured maximum and the bytes that remain in this file
    bool read_sized(std::vector<uint8_t> & v, uint64_t n, uint64_t max_bytes) {
        if (!ok) {
            return false;
        }
        if (n > max_bytes || n > left) {
            ok = false;
            return false;
        }
        try {
            v.resize((size_t) n);
        } catch (const std::bad_alloc &) {
            ok = false;
            return false;
        }
        return read_raw(v.data(), n);
    }

    bool read_blob(std::vector<uint8_t> & v, uint64_t max_bytes) {
        uint64_t n = 0;
        if (!read_pod(n)) {
            return false;
        }
        return read_sized(v, n, max_bytes);
    }
};

static bool pcache_checkpoint_coverage_equal(
        const server_prompt_restore_checkpoint & a,
        const server_prompt_restore_checkpoint & b) {
    return a.n_tokens == b.n_tokens && a.pos_min == b.pos_min && a.pos_max == b.pos_max &&
           a.components == b.components;
}

static bool restore_domain_equal(
        const server_prompt_restore_domain & a,
        const server_prompt_restore_domain & b) {
    return a.pos_min == b.pos_min && a.pos_max == b.pos_max &&
           a.rollback_tokens == b.rollback_tokens && a.swa_tokens == b.swa_tokens &&
           a.rollback_bounded == b.rollback_bounded;
}

static bool pcache_coverage_equal(
        const server_prompt_restore_coverage & a,
        const server_prompt_restore_coverage & b) {
    if (a.scope != b.scope ||
            !restore_domain_equal(a.target, b.target) ||
            !restore_domain_equal(a.draft, b.draft) ||
            a.checkpoints.size() != b.checkpoints.size()) {
        return false;
    }
    for (size_t i = 0; i < a.checkpoints.size(); ++i) {
        if (!pcache_checkpoint_coverage_equal(a.checkpoints[i], b.checkpoints[i])) {
            return false;
        }
    }
    return true;
}

static bool pcache_state_coverage_matches_payload(const server_prompt_cache_state & state) {
    if (state.data.main.empty() ||
            !server_prompt_restore_coverage_is_valid(state.coverage, state.prompt.tokens)) {
        return false;
    }

    if (state.coverage.scope == server_prompt_restore_scope::target_only) {
        if (!state.data.drft.empty()) {
            return false;
        }
    } else if (state.coverage.scope == server_prompt_restore_scope::target_and_draft) {
        if (state.data.drft.empty()) {
            return false;
        }
    } else {
        return false;
    }

    const auto actual = server_prompt_build_restore_coverage(
            state.prompt,
            state.coverage.scope,
            state.coverage.target,
            state.coverage.draft);

    if (!pcache_coverage_equal(actual, state.coverage) ||
            actual.checkpoints.size() != state.prompt.checkpoints.size()) {
        return false;
    }

    if (state.coverage.scope == server_prompt_restore_scope::target_only) {
        for (const auto & checkpoint : state.prompt.checkpoints) {
            if (!checkpoint.data_dft.empty() || !checkpoint.data_spec.empty()) {
                return false;
            }
        }
    }

    return true;
}

static bool pcache_write_coverage(
        pcache_writer & writer,
        const server_prompt_cache_state & state,
        uint64_t & size,
        uint64_t & hash) {
    pcache_disk_coverage disk = {};
    disk.version                 = PCACHE_COVERAGE_VERSION;
    disk.scope                   = (uint8_t) state.coverage.scope;
    disk.flags                   = (state.coverage.target.rollback_bounded ? 1u : 0u) |
                                   (state.coverage.draft.rollback_bounded  ? 2u : 0u);
    disk.n_tokens                = state.prompt.n_tokens();
    disk.target_pos_min          = state.coverage.target.pos_min;
    disk.target_pos_max          = state.coverage.target.pos_max;
    disk.draft_pos_min           = state.coverage.draft.pos_min;
    disk.draft_pos_max           = state.coverage.draft.pos_max;
    disk.target_rollback_tokens  = state.coverage.target.rollback_tokens;
    disk.draft_rollback_tokens   = state.coverage.draft.rollback_tokens;
    disk.n_checkpoints           = (uint32_t) state.coverage.checkpoints.size();
    disk.target_swa_tokens       = state.coverage.target.swa_tokens;
    disk.draft_swa_tokens        = state.coverage.draft.swa_tokens;

    const uint64_t begin = writer.payload_size;
    bool ok = writer.write_pod(disk);
    for (const auto & checkpoint : state.coverage.checkpoints) {
        pcache_disk_checkpoint_coverage item = {};
        item.n_tokens   = checkpoint.n_tokens;
        item.pos_min    = checkpoint.pos_min;
        item.pos_max    = checkpoint.pos_max;
        item.components = checkpoint.components;
        ok = ok && writer.write_pod(item);
    }

    size = writer.payload_size - begin;
    hash = writer.payload_hash;
    return ok;
}

static bool pcache_read_coverage(
        pcache_reader & reader,
        const pcache_disk_header & header,
        server_prompt_restore_coverage & coverage,
        const char * & reason) {
    const uint64_t begin_left = reader.left;

    pcache_disk_coverage disk = {};
    if (!reader.read_pod(disk)) {
        reason = "truncated restore coverage";
        return false;
    }
    if (disk.version != PCACHE_COVERAGE_VERSION || disk.reserved != 0 ||
            (disk.flags & ~3u) != 0 || disk.n_tokens != (int64_t) header.n_tokens ||
            disk.n_checkpoints != header.n_checkpoints) {
        reason = "invalid restore coverage header";
        return false;
    }

    coverage = {};
    coverage.scope                   = (server_prompt_restore_scope) disk.scope;
    coverage.target.pos_min          = disk.target_pos_min;
    coverage.target.pos_max          = disk.target_pos_max;
    coverage.target.rollback_tokens  = disk.target_rollback_tokens;
    coverage.target.swa_tokens       = disk.target_swa_tokens;
    coverage.target.rollback_bounded = (disk.flags & 1u) != 0;
    coverage.draft.pos_min           = disk.draft_pos_min;
    coverage.draft.pos_max           = disk.draft_pos_max;
    coverage.draft.rollback_tokens   = disk.draft_rollback_tokens;
    coverage.draft.swa_tokens        = disk.draft_swa_tokens;
    coverage.draft.rollback_bounded  = (disk.flags & 2u) != 0;

    try {
        coverage.checkpoints.reserve(disk.n_checkpoints);
        for (uint32_t i = 0; i < disk.n_checkpoints; ++i) {
            pcache_disk_checkpoint_coverage item = {};
            if (!reader.read_pod(item)) {
                reason = "truncated checkpoint coverage";
                return false;
            }
            for (uint8_t byte : item.reserved) {
                if (byte != 0) {
                    reason = "invalid checkpoint coverage padding";
                    return false;
                }
            }
            coverage.checkpoints.push_back({ item.n_tokens, item.pos_min, item.pos_max, item.components });
        }
    } catch (const std::bad_alloc &) {
        coverage = {};
        reason = "restore coverage allocation failed";
        return false;
    }

    if (begin_left - reader.left != header.size_coverage || reader.hash != header.hash_coverage) {
        reason = "restore coverage checksum mismatch";
        return false;
    }
    if ((coverage.scope == server_prompt_restore_scope::target_only && header.size_drft != 0) ||
            (coverage.scope == server_prompt_restore_scope::target_and_draft && header.size_drft == 0)) {
        reason = "restore scope does not match state blobs";
        return false;
    }

    return true;
}

// Read and fully validate a header. On failure `reason` explains why, and the
// caller must treat the file as unusable - never fatal, never trusted.
static bool pcache_read_header(std::ifstream & in,
                               uint64_t        file_size,
                               uint64_t        fingerprint,
                               uint64_t        max_tokens,
                               pcache_disk_header & hdr,
                               const char *  & reason) {
    reason = "";

    if (file_size < sizeof(pcache_disk_header)) {
        reason = "smaller than the header";
        return false;
    }

    if (file_size > PCACHE_MAX_FILE_BYTES) {
        reason = "larger than the maximum entry size";
        return false;
    }

    if (!in.read(reinterpret_cast<char *>(&hdr), sizeof(hdr))) {
        reason = "truncated header";
        return false;
    }

    if (hdr.magic != PCACHE_DISK_MAGIC) {
        reason = "bad magic";
        return false;
    }

    if (hdr.version != PCACHE_DISK_VERSION) {
        reason = "unsupported schema version";
        return false;
    }

    if (hdr.hash_header != server_pcache_header_hash(hdr)) {
        reason = "header checksum mismatch";
        return false;
    }

    if (hdr.fingerprint != fingerprint) {
        reason = "stale fingerprint (different model, ABI, KV layout or geometry)";
        return false;
    }

    const uint64_t payload_avail = file_size - sizeof(pcache_disk_header);
    if (hdr.size_payload != payload_avail) {
        reason = "declared payload size does not match the file";
        return false;
    }

    if (hdr.n_tokens > max_tokens) {
        reason = "token count above the configured maximum";
        return false;
    }

    if (hdr.n_checkpoints > PCACHE_MAX_CHECKPOINTS) {
        reason = "checkpoint count above the maximum";
        return false;
    }

    const uint64_t expected_coverage = sizeof(pcache_disk_coverage) +
            hdr.n_checkpoints*(uint64_t) sizeof(pcache_disk_checkpoint_coverage);
    if (hdr.size_coverage != expected_coverage) {
        reason = "restore coverage size does not match the checkpoint count";
        return false;
    }

    if (hdr.size_main > PCACHE_MAX_BLOB_BYTES || hdr.size_drft > PCACHE_MAX_BLOB_BYTES) {
        reason = "state blob above the maximum";
        return false;
    }

    // minimum bytes the declared contents need; n_tokens and n_checkpoints are
    // already capped, so these products cannot overflow
    const uint64_t ckpt_fixed_bytes = sizeof(int64_t) + sizeof(int) + 2*sizeof(llama_pos) + 3*sizeof(uint64_t);

    uint64_t need = hdr.size_coverage;
    if (!pcache_add_checked(need, hdr.n_tokens*(uint64_t) sizeof(llama_token)) ||
        !pcache_add_checked(need, hdr.size_main) ||
        !pcache_add_checked(need, hdr.size_drft) ||
        !pcache_add_checked(need, hdr.n_checkpoints*ckpt_fixed_bytes)) {
        reason = "declared lengths overflow";
        return false;
    }

    if (need > payload_avail) {
        reason = "declared lengths exceed the file";
        return false;
    }

    return true;
}

//
// persistent cache identity
//

namespace {

struct pcache_fnv1a {
    uint64_t h = 0xcbf29ce484222325ull;

    void mix(const void * data, size_t size) {
        const uint8_t * b = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            h = (h ^ b[i])*0x100000001b3ull;
        }
    }

    template <typename T>
    void mix_pod(const T & v) {
        mix(&v, sizeof(v));
    }

    // length-prefixed, so no pair of adjacent fields can be confused for another
    void mix_str(const std::string & s) {
        const uint64_t n = s.size();
        mix_pod(n);
        mix(s.data(), s.size());
    }

    void mix_artifact(const server_prompt_cache_fingerprint_inputs::artifact & a) {
        mix_pod(a.present);
        mix_str(a.path);
        mix_pod(a.size);
        mix_pod(a.mtime);
        mix_pod(a.probe);
    }
};

} // namespace

uint64_t server_pcache_file_probe_impl(const std::string & path) {
    std::error_code ec;
    const uint64_t size = (uint64_t) std::filesystem::file_size(path, ec);
    if (ec) {
        return 0;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return 0;
    }

    constexpr size_t n_probe = 64*1024;

    std::vector<uint8_t> buf(std::min<uint64_t>(size, n_probe));

    uint64_t h = server_pcache_hash(&size, sizeof(size), PCACHE_HASH_SEED_HDR);

    if (!buf.empty()) {
        if (!in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) buf.size())) {
            return 0;
        }
        h = server_pcache_hash(buf.data(), buf.size(), h);
    }

    if (size > n_probe) {
        const uint64_t tail = std::min<uint64_t>(size - n_probe, n_probe);
        buf.resize((size_t) tail);
        in.seekg((std::streamoff) (size - tail), std::ios::beg);
        if (!in.read(reinterpret_cast<char *>(buf.data()), (std::streamsize) buf.size())) {
            return 0;
        }
        h = server_pcache_hash(buf.data(), buf.size(), h);
    }

    // never return 0 for a readable file: 0 is the "unreadable" sentinel
    return h == 0 ? 1 : h;
}

uint64_t server_prompt_cache_file_probe(const std::string & path) {
    if (path.empty()) {
        return 0;
    }
    try {
        return server_pcache_file_probe_impl(path);
    } catch (...) {
        return 0;
    }
}

uint64_t server_prompt_cache_fingerprint(const server_prompt_cache_fingerprint_inputs & in) {
    pcache_fnv1a fp;

    // schema/ABI generation first: bumping any of these invalidates every file
    fp.mix_pod(in.schema_version);
    fp.mix_pod(in.state_seq_version);
    fp.mix_pod(in.state_seq_layout);
    fp.mix_pod(in.session_version);

    // model artifacts - the draft identity matters as much as the target one:
    // a `.lcpc` file carries both sequence states
    fp.mix_artifact(in.model_tgt);
    fp.mix_artifact(in.model_dft);

    // KV representation
    fp.mix_pod(in.kv_type_k_tgt);
    fp.mix_pod(in.kv_type_v_tgt);
    fp.mix_pod(in.kv_type_k_dft);
    fp.mix_pod(in.kv_type_v_dft);
    fp.mix_pod(in.flash_attn_type);
    fp.mix_pod(in.kv_unified);
    fp.mix_pod(in.swa_full);

    // layout-sensitive geometry
    fp.mix_pod(in.n_ctx_tgt);
    fp.mix_pod(in.n_seq_max_tgt);
    fp.mix_pod(in.n_ctx_dft);
    fp.mix_pod(in.n_seq_max_dft);
    fp.mix_pod(in.n_parallel);

    // speculative configuration (sizes the draft recurrent state, decides what
    // each checkpoint's data_spec contains)
    {
        const uint64_t n = in.spec_types.size();
        fp.mix_pod(n);
        for (const int32_t t : in.spec_types) {
            fp.mix_pod(t);
        }
    }
    fp.mix_pod(in.spec_n_max);

    return fp.h;
}

server_prompt_restore_coverage server_prompt_build_restore_coverage(
        const server_prompt & prompt,
        server_prompt_restore_scope scope,
        const server_prompt_restore_domain & target,
        const server_prompt_restore_domain & draft) {
    server_prompt_restore_coverage coverage;

    coverage.scope  = scope;
    coverage.target = target;

    if (scope == server_prompt_restore_scope::target_and_draft) {
        coverage.draft = draft;
    }

    coverage.checkpoints.reserve(prompt.checkpoints.size());
    for (const common_prompt_checkpoint & checkpoint : prompt.checkpoints) {
        if (checkpoint.data_tgt.empty()) {
            continue;
        }

        uint8_t components = SERVER_PROMPT_RESTORE_COMPONENT_TARGET;
        if (!checkpoint.data_dft.empty()) {
            components |= SERVER_PROMPT_RESTORE_COMPONENT_DRAFT;
        }
        if (!checkpoint.data_spec.empty()) {
            components |= SERVER_PROMPT_RESTORE_COMPONENT_SPEC;
        }

        // A paired top-level state must never land on a target-only checkpoint:
        // doing so would leave the draft frontier ahead of the target. Keep
        // only checkpoints that can restore the same state domain atomically.
        if (scope == server_prompt_restore_scope::target_and_draft &&
                (components & SERVER_PROMPT_RESTORE_COMPONENT_DRAFT) == 0) {
            continue;
        }

        if (scope == server_prompt_restore_scope::target_only) {
            components = SERVER_PROMPT_RESTORE_COMPONENT_TARGET;
        }

        coverage.checkpoints.push_back({
            checkpoint.n_tokens,
            checkpoint.pos_min,
            checkpoint.pos_max,
            components,
        });
    }

    return coverage;
}

server_prompt_restore_coverage server_prompt_capture_restore_coverage(
        const server_prompt & prompt,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_seq_id seq_id,
        bool force_target_only) {
    if (ctx_tgt == nullptr || prompt.tokens.empty()) {
        return {};
    }

    const auto capture_domain = [seq_id](llama_context * ctx) {
        server_prompt_restore_domain domain;
        const llama_model * model = llama_get_model(ctx);
        domain.pos_min          = llama_memory_seq_pos_min(llama_get_memory(ctx), seq_id);
        domain.pos_max          = llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
        domain.rollback_tokens  = llama_n_rs_seq(ctx);
        domain.swa_tokens       = (uint32_t) std::max(0, llama_model_n_swa(model));
        domain.rollback_bounded = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
        return domain;
    };

    const auto target = capture_domain(ctx_tgt);
    server_prompt_restore_domain draft;
    if (!force_target_only && ctx_dft != nullptr) {
        draft = capture_domain(ctx_dft);
    }

    // Different SWA widths legitimately produce different retained minima.
    // Pairing requires one decoded frontier, not identical retained windows.
    const bool paired = !force_target_only && ctx_dft != nullptr &&
            target.pos_min >= 0 && draft.pos_min >= 0 && target.pos_max == draft.pos_max;
    return server_prompt_build_restore_coverage(
            prompt,
            paired ? server_prompt_restore_scope::target_and_draft
                   : server_prompt_restore_scope::target_only,
            target,
            paired ? draft : server_prompt_restore_domain {});
}

bool server_prompt_restore_coverage_is_valid(
        const server_prompt_restore_coverage & coverage,
        const server_tokens & source_tokens) {
    const auto represented_tokens = [&](llama_pos pos_max, size_t & represented) {
        if (pos_max < 0 || pos_max == std::numeric_limits<llama_pos>::max()) {
            return false;
        }

        const llama_pos pos_next = pos_max + 1;
        represented = source_tokens.size_up_to_pos(pos_next);

        // size_up_to_pos() advances media atomically. Requiring the inverse
        // mapping to land exactly rejects a frontier in the middle of a media
        // chunk and avoids treating positions as token indexes.
        return represented > 0 && represented <= source_tokens.size() &&
               source_tokens.pos_next((int64_t) represented) == pos_next;
    };

    size_t target_represented = 0;
    if (source_tokens.empty() || coverage.scope == server_prompt_restore_scope::invalid ||
            coverage.target.pos_min < 0 || coverage.target.pos_max < coverage.target.pos_min ||
            !represented_tokens(coverage.target.pos_max, target_represented)) {
        return false;
    }

    if (coverage.scope == server_prompt_restore_scope::target_only) {
        const server_prompt_restore_domain empty_draft;
        if (!restore_domain_equal(coverage.draft, empty_draft)) {
            return false;
        }
    } else if (coverage.scope == server_prompt_restore_scope::target_and_draft) {
        // A full entry is one atomic target+draft restore domain. Divergent
        // decoded frontiers are downgraded before capture and rejected on
        // disk. The retained minima can differ when the models' SWA widths do.
        size_t draft_represented = 0;
        if (coverage.draft.pos_min < 0 || coverage.draft.pos_max < coverage.draft.pos_min ||
                coverage.draft.pos_max != coverage.target.pos_max ||
                !represented_tokens(coverage.draft.pos_max, draft_represented) ||
                draft_represented != target_represented) {
            return false;
        }
    } else {
        return false;
    }

    int64_t checkpoint_n_tokens_prev = -1;
    for (const auto & checkpoint : coverage.checkpoints) {
        const bool has_target = (checkpoint.components & SERVER_PROMPT_RESTORE_COMPONENT_TARGET) != 0;
        const bool has_draft  = (checkpoint.components & SERVER_PROMPT_RESTORE_COMPONENT_DRAFT) != 0;
        const uint8_t known_components =
                SERVER_PROMPT_RESTORE_COMPONENT_TARGET |
                SERVER_PROMPT_RESTORE_COMPONENT_DRAFT  |
                SERVER_PROMPT_RESTORE_COMPONENT_SPEC;

        if (!has_target || (checkpoint.components & ~known_components) != 0 ||
                checkpoint.n_tokens < 0 || (uint64_t) checkpoint.n_tokens > target_represented ||
                checkpoint.n_tokens < checkpoint_n_tokens_prev ||
                checkpoint.pos_min < 0 || checkpoint.pos_max < checkpoint.pos_min ||
                checkpoint.pos_max > coverage.target.pos_max ||
                (checkpoint.n_tokens > 0 &&
                 checkpoint.pos_max >= source_tokens.pos_next(checkpoint.n_tokens))) {
            return false;
        }
        if (coverage.scope == server_prompt_restore_scope::target_only &&
                checkpoint.components != SERVER_PROMPT_RESTORE_COMPONENT_TARGET) {
            return false;
        }
        if (coverage.scope == server_prompt_restore_scope::target_and_draft && !has_draft) {
            return false;
        }

        checkpoint_n_tokens_prev = checkpoint.n_tokens;
    }

    return true;
}

server_prompt_restore_plan server_prompt_make_restore_plan(
        const server_prompt_restore_coverage & coverage,
        const server_tokens & source_tokens,
        size_t common_tokens,
        size_t request_tokens) {
    server_prompt_restore_plan plan;
    plan.scope = coverage.scope;

    const size_t source_size = source_tokens.size();
    common_tokens = std::min(common_tokens, std::min(source_size, request_tokens));
    plan.matched_tokens = common_tokens;

    if (common_tokens == 0 || !server_prompt_restore_coverage_is_valid(coverage, source_tokens)) {
        return plan;
    }

    plan.full_match = common_tokens == request_tokens;

    // The prompt token vector can contain the most recently sampled token
    // before that token has been decoded into the sequence state. Derive the
    // represented token count from the actual positional frontier, then cap
    // the raw LCP at that count. For a paired entry both frontiers are an
    // atomic restore domain and therefore both cap the landing.
    const size_t target_represented = source_tokens.size_up_to_pos(coverage.target.pos_max + 1);
    const size_t draft_represented = coverage.scope == server_prompt_restore_scope::target_and_draft ?
            source_tokens.size_up_to_pos(coverage.draft.pos_max + 1) : target_represented;
    const size_t requested_tokens = common_tokens - (plan.full_match ? 1 : 0);
    const size_t direct_tokens = std::min(requested_tokens,
            std::min(target_represented, draft_represented));
    if (direct_tokens == 0) {
        return plan;
    }

    plan.pos_next = source_tokens.pos_next((int64_t) direct_tokens);
    plan.target_pos_threshold = std::max<llama_pos>(0,
            plan.pos_next - (llama_pos) coverage.target.swa_tokens);
    plan.draft_pos_threshold = coverage.scope == server_prompt_restore_scope::target_and_draft ?
            std::max<llama_pos>(0, plan.pos_next - (llama_pos) coverage.draft.swa_tokens) :
            plan.target_pos_threshold;

    const size_t target_tail_tokens = target_represented - direct_tokens;
    const size_t draft_tail_tokens  = draft_represented  - direct_tokens;
    const bool target_direct = coverage.target.pos_min < plan.target_pos_threshold &&
            (!coverage.target.rollback_bounded ||
             target_tail_tokens <= coverage.target.rollback_tokens);
    const bool draft_direct = coverage.scope != server_prompt_restore_scope::target_and_draft ||
            (coverage.draft.pos_min < plan.draft_pos_threshold &&
             (!coverage.draft.rollback_bounded ||
              draft_tail_tokens <= coverage.draft.rollback_tokens));

    if (target_direct && draft_direct) {
        plan.kind             = server_prompt_restore_kind::direct;
        plan.effective_tokens = direct_tokens;
        plan.recompute_tokens = common_tokens - direct_tokens;
        plan.landing_pos      = plan.pos_next;
        return plan;
    }

    // Newest covering checkpoint wins, exactly matching the prefill landing
    // rule. The strict pos_min threshold guarantees at least one token is
    // evaluated on a full match ([TAG_PROMPT_LOGITS]).
    for (size_t i = coverage.checkpoints.size(); i > 0; --i) {
        const auto & checkpoint = coverage.checkpoints[i - 1];
        const bool target_boundary = checkpoint.pos_min < plan.target_pos_threshold || checkpoint.pos_min == 0;
        const bool draft_boundary = coverage.scope != server_prompt_restore_scope::target_and_draft ||
                checkpoint.pos_min < plan.draft_pos_threshold || checkpoint.pos_min == 0;
        if (checkpoint.pos_max > plan.pos_next || !target_boundary || !draft_boundary) {
            continue;
        }
        if (coverage.scope == server_prompt_restore_scope::target_and_draft &&
                (checkpoint.components & SERVER_PROMPT_RESTORE_COMPONENT_DRAFT) == 0) {
            continue;
        }

        const llama_pos landing_pos = std::min(
                plan.pos_next,
                std::max(checkpoint.pos_min + 1, checkpoint.pos_max));
        const size_t effective_tokens = std::min(
                source_tokens.size_up_to_pos(landing_pos),
                (size_t) checkpoint.n_tokens);

        if (effective_tokens == 0 || effective_tokens > direct_tokens) {
            continue;
        }

        plan.kind                = server_prompt_restore_kind::checkpoint;
        plan.effective_tokens    = effective_tokens;
        plan.recompute_tokens    = common_tokens - effective_tokens;
        plan.landing_pos         = landing_pos;
        plan.checkpoint_index    = i - 1;
        plan.checkpoint_n_tokens = checkpoint.n_tokens;
        return plan;
    }

    return plan;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::size_disk_total() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size_disk;
    }

    return res;
}

bool server_prompt_cache::spill_to_disk(server_prompt_cache_state & state) {
    GGML_ASSERT(!disk_dir.empty());

    if (state.on_disk()) {
        return true;
    }

    // media chunks cannot be serialized to the SSD tier
    if (state.prompt.tokens.has_media()) {
        return false;
    }

    const int64_t t_start = ggml_time_us();

    const llama_tokens & tokens = state.prompt.tokens.get_tokens();

    // refuse to write anything the reader would have to refuse
    if ((uint64_t) tokens.size() > disk_max_tokens ||
        (uint64_t) state.prompt.checkpoints.size() > PCACHE_MAX_CHECKPOINTS ||
        (uint64_t) state.data.main.size() > PCACHE_MAX_BLOB_BYTES ||
        (uint64_t) state.data.drft.size() > PCACHE_MAX_BLOB_BYTES) {
        SRV_WRN(" - prompt cache entry (%zu tokens, %zu checkpoints, %zu B) exceeds the SSD tier limits, not spilling\n",
                tokens.size(), state.prompt.checkpoints.size(), state.data.size());
        return false;
    }
    if (!pcache_state_coverage_matches_payload(state)) {
        SRV_WRN(" - prompt cache entry (%zu tokens) has invalid restore coverage, not spilling\n",
                tokens.size());
        return false;
    }

    std::string path;
    do {
        path = disk_dir + "/pc-" + std::to_string(disk_fingerprint % 0xffffffull) + "-" + std::to_string(disk_seq++) + ".lcpc";
    } while (std::filesystem::exists(path));

    // write to a temporary name and publish with an atomic rename: a crash or a
    // full disk must never leave a partial file in the discoverable namespace
    const std::string path_tmp = path + ".tmp";

    pcache_disk_header hdr = {};

    bool ok = false;
    {
        pcache_writer w;
        w.out.open(path_tmp, std::ios::binary | std::ios::trunc);

        ok = w.out.is_open();

        if (ok) {
            // reserve the header; it is rewritten below once the payload hash
            // is known (not part of the payload, so written outside the writer)
            ok = bool(w.out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr)));
        }

        uint64_t size_coverage = 0;
        uint64_t hash_coverage = 0;
        ok = ok &&
             pcache_write_coverage(w, state, size_coverage, hash_coverage) &&
             w.write_raw(tokens.data(), tokens.size()*sizeof(llama_token)) &&
             w.write_raw(state.data.main.data(), state.data.main.size()) &&
             w.write_raw(state.data.drft.data(), state.data.drft.size());

        for (const auto & ckpt : state.prompt.checkpoints) {
            ok = ok &&
                 w.write_pod(ckpt.n_tokens) &&
                 w.write_pod(ckpt.id_task)  &&
                 w.write_pod(ckpt.pos_min)  &&
                 w.write_pod(ckpt.pos_max)  &&
                 w.write_blob(ckpt.data_tgt) &&
                 w.write_blob(ckpt.data_dft) &&
                 w.write_blob(ckpt.data_spec);
        }

        if (ok) {
            hdr.magic         = PCACHE_DISK_MAGIC;
            hdr.version       = PCACHE_DISK_VERSION;
            hdr.fingerprint   = disk_fingerprint;
            hdr.size_main     = state.data.main.size();
            hdr.size_drft     = state.data.drft.size();
            hdr.n_tokens      = tokens.size();
            hdr.n_checkpoints = state.prompt.checkpoints.size();
            hdr.size_coverage = size_coverage;
            hdr.size_payload  = w.payload_size;
            hdr.hash_coverage = hash_coverage;
            hdr.hash_payload  = w.payload_hash;
            hdr.hash_header   = server_pcache_header_hash(hdr);

            w.out.seekp(0, std::ios::beg);

            ok = bool(w.out.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr)));
        }

        if (ok) {
            w.out.flush();
            ok = bool(w.out);
        }

        w.out.close();
        ok = ok && !w.out.fail();
    }

    if (ok) {
        // order the data before the rename, then make the rename itself durable
        pcache_fsync(path_tmp, false);

        std::error_code ec;
        std::filesystem::rename(path_tmp, path, ec);
        if (ec) {
            SRV_WRN(" - failed to publish prompt cache file '%s': %s\n", path.c_str(), ec.message().c_str());
            ok = false;
        } else {
            pcache_fsync(disk_dir, true);
        }
    }

    if (!ok) {
        SRV_WRN(" - failed to spill prompt cache entry to '%s', dropping the file\n", path.c_str());
        std::error_code ec;
        std::filesystem::remove(path_tmp, ec);
        std::filesystem::remove(path, ec);
        return false;
    }

    std::error_code ec;
    const size_t size_file = std::filesystem::file_size(path, ec);
    const size_t size_ram = state.size();

    state.file      = path;
    state.size_disk = ec ? size_ram : size_file;

    state.data.main.clear();
    state.data.main.shrink_to_fit();
    state.data.drft.clear();
    state.data.drft.shrink_to_fit();
    state.prompt.checkpoints.clear();

    const double dt_ms = (ggml_time_us() - t_start) / 1000.0;
    SRV_INF(" - spilled prompt (%d tokens, %.3f MiB) to '%s' in %.1f ms (%.1f MB/s)\n",
            state.prompt.n_tokens(), size_ram / (1024.0*1024.0), path.c_str(), dt_ms,
            dt_ms > 0.0 ? (size_ram / (1024.0*1024.0)) / (dt_ms / 1000.0) : 0.0);

    ++dash_spills;
    dash_bytes_spilled += size_ram;
    {
        server_dashboard::event ev;
        ev.kind = server_dashboard::event_kind::cache_op;
        ev.code = (uint16_t) server_dashboard::cache_op_code::spill;
        ev.a    = state.prompt.n_tokens();
        ev.b    = (int64_t) size_ram;
        ev.f    = dt_ms;
        server_dashboard::instance().emit(ev);
    }

    return true;
}

bool server_prompt_cache::load_from_disk(server_prompt_cache_state & state) {
    GGML_ASSERT(state.on_disk());

    const int64_t t_start = ggml_time_us();

    const auto reject = [&](const char * why) {
        SRV_WRN(" - prompt cache file '%s' rejected: %s\n", state.file.c_str(), why);
        state.data.main.clear();
        state.data.main.shrink_to_fit();
        state.data.drft.clear();
        state.data.drft.shrink_to_fit();
        state.prompt.checkpoints.clear();
        return false;
    };

    std::error_code ec;
    const uint64_t file_size = (uint64_t) std::filesystem::file_size(state.file, ec);
    if (ec) {
        return reject("cannot stat the file");
    }

    pcache_reader r;
    r.in.open(state.file, std::ios::binary);
    if (!r.in.is_open()) {
        return reject("cannot open the file");
    }

    pcache_disk_header hdr    = {};
    const char *       reason = "";
    if (!pcache_read_header(r.in, file_size, disk_fingerprint, disk_max_tokens, hdr, reason)) {
        return reject(reason);
    }

    if (hdr.n_tokens != (uint64_t) state.prompt.tokens.size()) {
        return reject("token count does not match the index entry");
    }

    r.left = hdr.size_payload;

    server_prompt_restore_coverage disk_coverage;
    if (!pcache_read_coverage(r, hdr, disk_coverage, reason)) {
        return reject(reason);
    }
    if (!pcache_coverage_equal(disk_coverage, state.coverage)) {
        return reject("restore coverage does not match the index entry");
    }

    // tokens: bounded by the header check above, so this resize is safe
    llama_tokens tokens((size_t) hdr.n_tokens);
    if (!r.read_raw(tokens.data(), hdr.n_tokens*(uint64_t) sizeof(llama_token))) {
        return reject("truncated token array");
    }

    if (tokens != state.prompt.tokens.get_tokens()) {
        return reject("tokens do not match the index entry");
    }
    if (!server_prompt_restore_coverage_is_valid(disk_coverage, state.prompt.tokens)) {
        return reject("invalid restore coverage semantics");
    }

    bool ok = r.read_sized(state.data.main, hdr.size_main, PCACHE_MAX_BLOB_BYTES) &&
              r.read_sized(state.data.drft, hdr.size_drft, PCACHE_MAX_BLOB_BYTES);

    state.prompt.checkpoints.clear();
    for (uint64_t i = 0; ok && i < hdr.n_checkpoints; ++i) {
        common_prompt_checkpoint ckpt = {};
        ok = r.read_pod(ckpt.n_tokens) &&
             r.read_pod(ckpt.id_task)  &&
             r.read_pod(ckpt.pos_min)  &&
             r.read_pod(ckpt.pos_max)  &&
             r.read_blob(ckpt.data_tgt,  PCACHE_MAX_BLOB_BYTES) &&
             r.read_blob(ckpt.data_dft,  PCACHE_MAX_BLOB_BYTES) &&
             r.read_blob(ckpt.data_spec, PCACHE_MAX_BLOB_BYTES);
        if (ok) {
            state.prompt.checkpoints.push_back(std::move(ckpt));
        }
    }

    if (!ok) {
        return reject("truncated or inconsistent payload");
    }

    if (r.left != 0) {
        return reject("trailing bytes after the payload");
    }

    // every byte of the payload has now been hashed exactly once, in the same
    // chunking the writer used
    if (r.hash != hdr.hash_payload) {
        return reject("payload checksum mismatch");
    }

    if (!pcache_state_coverage_matches_payload(state)) {
        return reject("restore coverage does not match the state payload");
    }

    const double dt_ms = (ggml_time_us() - t_start) / 1000.0;
    const double mib = (hdr.size_main + hdr.size_drft) / (1024.0*1024.0);
    SRV_INF(" - loaded prompt (%d tokens, %.3f MiB) from '%s' in %.1f ms (%.1f MB/s)\n",
            state.prompt.n_tokens(), mib, state.file.c_str(), dt_ms,
            dt_ms > 0.0 ? mib / (dt_ms / 1000.0) : 0.0);

    ++dash_disk_loads;
    dash_bytes_disk_load += hdr.size_main + hdr.size_drft;
    {
        server_dashboard::event ev;
        ev.kind = server_dashboard::event_kind::cache_op;
        ev.code = (uint16_t) server_dashboard::cache_op_code::disk_load;
        ev.a    = state.prompt.n_tokens();
        ev.b    = (int64_t) (hdr.size_main + hdr.size_drft);
        ev.f    = dt_ms;
        server_dashboard::instance().emit(ev);
    }

    return true;
}

void server_prompt_cache::rescan_disk() {
    std::error_code ec;

    std::filesystem::create_directories(disk_dir, ec);

    struct found_file {
        std::string path;
        std::filesystem::file_time_type mtime;
        size_t size;
    };

    std::vector<found_file> files;
    std::vector<std::string> orphan_tmp;
    {
        std::error_code it_ec;
        auto it = std::filesystem::directory_iterator(disk_dir, it_ec);
        if (it_ec) {
            SRV_WRN("failed to scan prompt cache directory '%s': %s\n", disk_dir.c_str(), it_ec.message().c_str());
            return;
        }

        for (const auto & entry : it) {
            std::error_code e_ec;
            if (!entry.is_regular_file(e_ec) || e_ec) {
                continue;
            }

            const auto ext = entry.path().extension();

            // leftovers from a spill that was interrupted before its rename;
            // they were never discoverable, so they are simply garbage now
            if (ext == ".tmp") {
                orphan_tmp.push_back(entry.path().string());
                continue;
            }

            if (ext != ".lcpc") {
                continue;
            }

            std::error_code m_ec;
            std::error_code s_ec;
            const auto mtime = entry.last_write_time(m_ec);
            const auto size  = entry.file_size(s_ec);
            if (m_ec || s_ec) {
                SRV_WRN("skipping unreadable prompt cache file '%s'\n", entry.path().string().c_str());
                continue;
            }

            files.push_back({ entry.path().string(), mtime, (size_t) size });
        }
    }

    for (const auto & path : orphan_tmp) {
        SRV_WRN("removing incomplete prompt cache file '%s'\n", path.c_str());
        std::error_code rmec;
        std::filesystem::remove(path, rmec);
    }

    // oldest first, so the eviction order survives the restart
    std::sort(files.begin(), files.end(), [](const found_file & a, const found_file & b) {
        return a.mtime < b.mtime;
    });

    size_t n_restored = 0;
    size_t n_dropped  = 0;

    for (const auto & f : files) {
        // one unusable file must never take the server down: everything below
        // is bounded by the header validation, and any surprise from the
        // filesystem or the allocator is contained here
        bool ok = false;

        pcache_disk_header hdr    = {};
        const char *       reason = "unknown";

        llama_tokens tokens;
        server_prompt_restore_coverage coverage;

        try {
            pcache_reader r;
            r.in.open(f.path, std::ios::binary);

            if (!r.in.is_open()) {
                reason = "cannot open the file";
            } else if (pcache_read_header(r.in, (uint64_t) f.size, disk_fingerprint, disk_max_tokens, hdr, reason)) {
                r.left = hdr.size_payload;

                // Coverage is small, independently hashed, and precedes the
                // tokens. Startup can therefore reject a non-restorable entry
                // without reading either state blob.
                ok = pcache_read_coverage(r, hdr, coverage, reason);
                if (ok) {
                    // n_tokens is bounded by disk_max_tokens (<= the context
                    // size) and by the payload the file really has.
                    tokens.resize((size_t) hdr.n_tokens);
                    ok = r.read_raw(tokens.data(), hdr.n_tokens*(uint64_t) sizeof(llama_token));
                    if (!ok) {
                        reason = "truncated token array";
                    } else {
                        server_tokens indexed_tokens(tokens, false);
                        ok = server_prompt_restore_coverage_is_valid(coverage, indexed_tokens);
                        if (!ok) {
                            reason = "invalid restore coverage semantics";
                        }
                    }
                }

                // note: the payload hash is deliberately *not* verified here -
                // that would read every spilled byte at startup. load_from_disk()
                // verifies it before the state is ever handed to llama.
            }
        } catch (const std::exception & e) {
            reason = e.what();
            ok     = false;
        } catch (...) {
            reason = "unhandled error";
            ok     = false;
        }

        if (!ok) {
            SRV_WRN("dropping unusable prompt cache file '%s': %s\n", f.path.c_str(), reason);
            std::error_code rmec;
            std::filesystem::remove(f.path, rmec);
            ++n_dropped;
            continue;
        }

        server_prompt_cache_state state;
        state.prompt.tokens = server_tokens(tokens, false);
        state.coverage      = std::move(coverage);
        state.file          = f.path;
        state.size_disk     = f.size;

        state.dash_id         = dash_id_next++;
        state.dash_created_us = ggml_time_us();

        states.push_back(std::move(state));
        ++n_restored;
    }

    SRV_INF("prompt cache SSD tier: restored %zu entries (%.3f GiB), dropped %zu unusable, from '%s'\n",
            n_restored, size_disk_total() / (1024.0*1024.0*1024.0), n_dropped, disk_dir.c_str());

    publish_dashboard_state();
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::drop_entry(std::list<server_prompt_cache_state>::iterator it) {
    if (it->on_disk()) {
        std::error_code ec;
        std::filesystem::remove(it->file, ec);
        if (ec) {
            SRV_WRN("failed to remove prompt cache file '%s': %s\n", it->file.c_str(), ec.message().c_str());
        }
    }

    ++dash_drops;
    {
        server_dashboard::event ev;
        ev.kind = server_dashboard::event_kind::cache_op;
        ev.code = (uint16_t) server_dashboard::cache_op_code::drop;
        ev.a    = it->prompt.n_tokens();
        ev.b    = (int64_t) (it->on_disk() ? it->size_disk : it->size());
        server_dashboard::instance().emit(ev);
    }

    return states.erase(it);
}

bool server_prompt_cache::evict_oldest_ram() {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->on_disk()) {
            continue;
        }

        if (!disk_dir.empty() && spill_to_disk(*it)) {
            return true;
        }

        SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", it->size() / (1024.0 * 1024.0));

        ++dash_drops;
        {
            server_dashboard::event ev;
            ev.kind = server_dashboard::event_kind::cache_op;
            ev.code = (uint16_t) server_dashboard::cache_op_code::drop;
            ev.a    = it->prompt.n_tokens();
            ev.b    = (int64_t) it->size();
            server_dashboard::instance().emit(ev);
        }

        states.erase(it);

        return true;
    }

    return false;
}

server_prompt_cache_state * server_prompt_cache::alloc(
        const server_prompt & prompt,
        const server_prompt_restore_coverage & coverage,
        size_t state_size_tgt,
        size_t state_size_dft) {
    server_prompt prompt_new;
    prompt_new.tokens = prompt.tokens.clone();

    // Normalize the checkpoint payload to the entry's restore domain. A
    // target-only publication is intentionally incapable of carrying draft or
    // speculative state, even when the live slot's checkpoint list did.
    for (const auto & checkpoint : prompt.checkpoints) {
        if (checkpoint.data_tgt.empty()) {
            continue;
        }
        if (coverage.scope == server_prompt_restore_scope::target_and_draft &&
                checkpoint.data_dft.empty()) {
            continue;
        }

        auto copy = checkpoint;
        if (coverage.scope == server_prompt_restore_scope::target_only) {
            copy.clear_dft();
        }
        prompt_new.checkpoints.push_back(std::move(copy));
    }

    auto coverage_new = server_prompt_build_restore_coverage(
            prompt_new,
            coverage.scope,
            coverage.target,
            coverage.draft);

    if (!server_prompt_restore_coverage_is_valid(coverage_new, prompt_new.tokens) ||
            state_size_tgt == 0 ||
            (coverage_new.scope == server_prompt_restore_scope::target_only && state_size_dft != 0) ||
            (coverage_new.scope == server_prompt_restore_scope::target_and_draft && state_size_dft == 0)) {
        SRV_WRN(" - prompt with length %d has inconsistent restore coverage, skipping\n", prompt.n_tokens());
        return nullptr;
    }

    const auto scope_covers = [](server_prompt_restore_scope replacement, server_prompt_restore_scope required) {
        return (uint8_t) replacement >= (uint8_t) required;
    };
    const auto exact_landing = [](size_t n_tokens) {
        return n_tokens > 0 ? n_tokens - 1 : 0;
    };

    // Suppress this entry only when an existing token superset can reproduce
    // the new entry's exact effective landing in the same state domain.
    for (auto it = states.begin(); it != states.end(); ++it) {
        const size_t lcp = it->prompt.tokens.get_common_prefix(prompt_new.tokens);
        if (lcp != prompt_new.tokens.size() || !scope_covers(it->coverage.scope, coverage_new.scope)) {
            continue;
        }

        const auto plan = server_prompt_make_restore_plan(
                it->coverage, it->prompt.tokens, lcp, prompt_new.tokens.size());
        if (plan.kind != server_prompt_restore_kind::unusable &&
                plan.effective_tokens == exact_landing(prompt_new.tokens.size())) {
            SRV_TRC("%s", " - prompt exact landing is already covered by the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate normalized checkpoint size to see if it will fit
    size_t checkpoints_size = 0;
    for (const auto & checkpoint : prompt_new.checkpoints) {
        checkpoints_size += checkpoint.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // Remove a contained entry only if the replacement reproduces that exact
    // landing and does not downgrade target+draft state to target-only.
    for (auto it = states.begin(); it != states.end();) {
        const size_t len = it->prompt.tokens.get_common_prefix(prompt_new.tokens);
        const auto plan = server_prompt_make_restore_plan(
                coverage_new, prompt_new.tokens, len, it->prompt.tokens.size());

        if (len == it->prompt.tokens.size() &&
                scope_covers(coverage_new.scope, it->coverage.scope) &&
                plan.kind != server_prompt_restore_kind::unusable &&
                plan.effective_tokens == exact_landing(it->prompt.tokens.size())) {
            SRV_TRC(" - removing superseded cached prompt with exact landing length %zu\n", len);
            it = drop_entry(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the
        // limit; the oldest RAM entries spill to the SSD tier when enabled
        while (size() + state_size_new > limit_size) {
            if (!evict_oldest_ram()) {
                break;
            }
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    server_prompt_cache_state state_new;
    state_new.prompt             = std::move(prompt_new);
    state_new.coverage           = std::move(coverage_new);
    state_new.data.main          = std::move(state_data_tgt);
    state_new.data.drft          = std::move(state_data_dft);
    state_new.dash_id            = dash_id_next++;
    state_new.dash_created_us    = ggml_time_us();

    states.push_back(std::move(state_new));

    ++dash_saves;
    dash_bytes_saved += state_size_new;
    {
        server_dashboard::event ev;
        ev.kind = server_dashboard::event_kind::cache_op;
        ev.code = (uint16_t) server_dashboard::cache_op_code::save;
        ev.a    = prompt.n_tokens();
        ev.b    = (int64_t) state_size_new;
        server_dashboard::instance().emit(ev);
    }

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    ++dash_lookups;
    dash_last_load = {};

    const size_t lcp_resident = prompt.tokens.get_common_prefix(tokens_new);
    const auto coverage_resident = server_prompt_capture_restore_coverage(
            prompt, ctx_tgt, ctx_dft, id_slot);
    const auto plan_resident = server_prompt_make_restore_plan(
            coverage_resident, prompt.tokens, lcp_resident, tokens_new.size());

    size_t reusable_best = plan_resident.effective_tokens;
    SRV_TRC(" - looking for better prompt, resident raw lcp = %zu, effective = %zu, kind = %u\n",
            lcp_resident, reusable_best, (unsigned) plan_resident.kind);

    auto it_best = states.end();
    server_prompt_restore_plan plan_best;

    // Rank only coverage-qualified effective tokens. A long state with an exact
    // system/tool/user checkpoint is valuable even when that boundary is less
    // than 25% of the serialized prompt, so raw f_keep is not an admission gate.
    // Ties stay resident, avoiding a state restore with no reuse gain.
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->coverage.scope == server_prompt_restore_scope::target_and_draft && ctx_dft == nullptr) {
            continue;
        }

        const size_t lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
        const auto plan_cur = server_prompt_make_restore_plan(
                it->coverage, it->prompt.tokens, lcp_cur, tokens_new.size());

        SRV_TRC("   - prompt with length %7zu, raw lcp = %7zu, effective = %7zu, kind = %u\n",
                it->prompt.tokens.size(), lcp_cur, plan_cur.effective_tokens, (unsigned) plan_cur.kind);

        if (plan_cur.kind != server_prompt_restore_kind::unusable &&
                plan_cur.effective_tokens > reusable_best) {
            reusable_best = plan_cur.effective_tokens;
            it_best = it;
            plan_best = plan_cur;
        }
    }

    if (it_best == states.end()) {
        // no cache entry beats the resident state
        if (reusable_best > 0) {
            ++dash_hits_resident;
            dash_last_load.source   = 0; // resident
            dash_last_load.n_tokens = (uint64_t) reusable_best;
        } else {
            ++dash_misses;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better coverage-qualified prompt with %zu effective tokens\n", reusable_best);

        const bool from_disk = it_best->on_disk();
        const size_t tokens_restored = plan_best.effective_tokens;

        if (from_disk && !load_from_disk(*it_best)) {
            // unreadable or stale file - drop the entry so it is not retried
            drop_entry(it_best);

            ++dash_misses;
            return false;
        }

        // clear the destination sequence before restoring: the transactional
        // DSV4 restore requires an empty destination (or an empty staging
        // sequence, which a busy second lane cannot provide under -np 2).
        // The current slot state is already saved by the preceding
        // prompt_save, so dropping it here loses nothing.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
        }

        const size_t bytes_restored = it_best->data.size();

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu - dropping cache entry\n", size);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
                if (ctx_dft) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
                }

                // the entry failed to restore once - it will fail again, so
                // drop it (and its file) instead of retrying it forever
                drop_entry(it_best);

                ++dash_misses;
                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu - dropping cache entry\n", size);

                    llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);

                    drop_entry(it_best);

                    ++dash_misses;
                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
            // else: target-only entry (saved from a slot whose speculative
            // decoding was bypassed). The draft sequence was cleared above, so
            // the speculative path re-prefills the draft instead of inheriting
            // stale rows.
        }

        // Fail closed if the restored representation does not produce the
        // same coverage decision advertised before the I/O. This is what
        // prevents a screened cache hit from becoming an admission-time clear.
        const auto coverage_actual = server_prompt_capture_restore_coverage(
                it_best->prompt, ctx_tgt, ctx_dft, id_slot);
        const size_t lcp_actual = it_best->prompt.tokens.get_common_prefix(tokens_new);
        const auto plan_actual = server_prompt_make_restore_plan(
                coverage_actual, it_best->prompt.tokens, lcp_actual, tokens_new.size());
        if (!pcache_coverage_equal(coverage_actual, it_best->coverage) ||
                plan_actual.kind != plan_best.kind ||
                plan_actual.effective_tokens != plan_best.effective_tokens ||
                plan_actual.checkpoint_index != plan_best.checkpoint_index) {
            SRV_WRN("%s", "restored state did not satisfy its advertised coverage - dropping cache entry\n");
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), id_slot, -1, -1);
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), id_slot, -1, -1);
            }
            drop_entry(it_best);
            ++dash_misses;
            return false;
        }

        ++dash_hits_entry;
        dash_last_load.source   = from_disk ? 2 : 1; // disk : ram
        dash_last_load.n_tokens = (uint64_t) tokens_restored;
        dash_last_load.n_bytes  = bytes_restored;
        dash_last_load.restored_scope = plan_best.scope;

        if (from_disk) {
            // keep the file and a slim index entry: the same prefix can be
            // reloaded again later (other slots, branched conversations, or
            // after a server restart) at SSD cost instead of a re-prefill
            server_prompt_cache_state slim;
            slim.prompt.tokens = it_best->prompt.tokens.clone();
            slim.coverage      = it_best->coverage;
            slim.file          = std::move(it_best->file);
            slim.size_disk     = it_best->size_disk;

            slim.dash_id          = it_best->dash_id;
            slim.dash_created_us  = it_best->dash_created_us;
            slim.dash_last_hit_us = ggml_time_us();
            slim.dash_hits        = it_best->dash_hits + 1;
            slim.dash_req         = it_best->dash_req;

            prompt = std::move(it_best->prompt);

            *it_best = std::move(slim);
        } else {
            prompt = std::move(it_best->prompt);

            states.erase(it_best);
        }
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (size() > limit_size) {
            if (!evict_oldest_ram()) {
                break;
            }
        }
    }

    // enforce the SSD tier limit, oldest spilled entries first
    if (limit_disk > 0) {
        while (size_disk_total() > limit_disk) {
            auto it = std::find_if(states.begin(), states.end(),
                    [](const server_prompt_cache_state & s) { return s.on_disk(); });
            if (it == states.end()) {
                break;
            }

            SRV_WRN(" - cache disk limit reached, removing oldest spilled entry (size = %.3f MiB)\n",
                    it->size_disk / (1024.0 * 1024.0));

            drop_entry(it);
        }
    }

    // average size per token (RAM tier only; disk entries hold no RAM blobs)
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            drop_entry(states.begin());
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB RAM, %.3f GiB disk (limits: %.3f MiB RAM, %.3f GiB disk, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), size_disk_total() / (1024.0*1024.0*1024.0),
            limit_size / (1024.0 * 1024.0), limit_disk / (1024.0*1024.0*1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB %s\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(),
                (state.on_disk() ? state.size_disk : state.size()) / (1024.0 * 1024.0),
                state.on_disk() ? "(disk)" : "(ram)");
    }

    publish_dashboard_state();
}

void server_prompt_cache::publish_dashboard_state() {
    server_dashboard::cache_state out;
    out.enabled          = true;
    out.at_us            = (uint64_t) ggml_time_us();
    out.limit_ram_bytes  = limit_size;
    out.limit_disk_bytes = limit_disk;
    out.limit_tokens     = limit_tokens;
    out.used_ram_bytes   = size();
    out.used_disk_bytes  = size_disk_total();
    out.tokens_total     = n_tokens();

    out.entries.reserve(states.size());
    for (const auto & state : states) {
        server_dashboard::cache_entry_state entry;
        entry.id          = state.dash_id;
        entry.tokens      = (uint64_t) state.prompt.n_tokens();
        entry.bytes_ram   = state.size();
        entry.bytes_disk  = state.size_disk;
        entry.on_disk     = state.on_disk();
        entry.created_us  = (uint64_t) std::max<int64_t>(0, state.dash_created_us);
        entry.last_hit_us = (uint64_t) std::max<int64_t>(0, state.dash_last_hit_us);
        entry.hits        = state.dash_hits;
        entry.request_id  = state.dash_req;
        if (state.on_disk()) {
            entry.file = std::filesystem::path(state.file).filename().string();
        }
        // m2-dashboard v3: a bounded head/tail slice of the entry's token IDS
        // rides along so GET /m2-dashboard/cache-preview can render what is
        // actually in the entry. Both tiers keep prompt.tokens resident (the
        // SSD tier only spills the state blobs), so this never touches disk.
        // No decoded text is stored: detokenization happens on the HTTP thread
        // for one entry at a time, only when a preview is requested.
        {
            const size_t n  = state.prompt.tokens.size();
            const size_t nh = std::min(n, server_dashboard::cache_preview_head_tokens);
            entry.head_ids.reserve(nh);
            for (size_t i = 0; i < nh; ++i) {
                entry.head_ids.push_back((int32_t) state.prompt.tokens[i]);
            }
            if (n > nh) {
                const size_t nt = std::min(n - nh, server_dashboard::cache_preview_tail_tokens);
                entry.tail_ids.reserve(nt);
                for (size_t i = n - nt; i < n; ++i) {
                    entry.tail_ids.push_back((int32_t) state.prompt.tokens[i]);
                }
            }
        }
        out.entries.push_back(std::move(entry));
    }

    out.counters.lookups         = dash_lookups;
    out.counters.hits_entry      = dash_hits_entry;
    out.counters.hits_resident   = dash_hits_resident;
    out.counters.misses          = dash_misses;
    out.counters.saves           = dash_saves;
    out.counters.spills          = dash_spills;
    out.counters.disk_loads      = dash_disk_loads;
    out.counters.drops           = dash_drops;
    out.counters.bytes_saved     = dash_bytes_saved;
    out.counters.bytes_spilled   = dash_bytes_spilled;
    out.counters.bytes_disk_load = dash_bytes_disk_load;

    server_dashboard::instance().publish_cache_state(std::move(out));
}

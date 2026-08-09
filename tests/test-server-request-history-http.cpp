#include "common.h"
#include "server-http.h"
#include "server-request-history.h"
#include "server-stream.h"

#include <cpp-httplib/httplib.h>

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace server_request_history;

namespace {

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error(message); }
void expect(bool condition, const std::string & message) { if (!condition) fail(message); }

class temporary_directory {
  public:
    temporary_directory() {
        const char * configured = std::getenv("TMPDIR");
        fs::path base = configured != nullptr && *configured != '\0' ? fs::path(configured) : fs::path("/tmp");
        std::error_code error;
        base = fs::canonical(base, error);
        if (error) fail("canonical temp root");
        std::string pattern = (base / "llama-request-history-http-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char * created = ::mkdtemp(writable.data());
        if (created == nullptr) fail("mkdtemp: " + std::string(std::strerror(errno)));
        value = created;
    }
    ~temporary_directory() {
        std::error_code ignored;
        fs::remove_all(value, ignored);
    }
    const fs::path & path() const { return value; }
  private:
    fs::path value;
};

server_http_res_ptr normal_response(const std::string & body, int status_code = 200) {
    auto response = std::make_unique<server_http_res>();
    response->status = status_code;
    response->data = body;
    return response;
}

server_http_res_ptr stream_response(std::vector<std::string> chunks, int delay_ms = 0) {
    auto response = std::make_unique<server_http_res>();
    response->status = 200;
    response->content_type = "text/event-stream";
    auto shared_chunks = std::make_shared<std::vector<std::string>>(std::move(chunks));
    auto index = std::make_shared<size_t>(0);
    response->next = [shared_chunks, index, delay_ms](std::string & output) {
        if (*index >= shared_chunks->size()) {
            output.clear();
            return false;
        }
        if (delay_ms != 0 && *index != 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        output = (*shared_chunks)[(*index)++];
        return *index < shared_chunks->size();
    };
    return response;
}

server_http_res_ptr resumable_stream_response(
        const server_http_req & req, std::vector<std::string> chunks, int delay_ms = 0) {
    auto response = std::make_unique<server_res_spipe>();
    response->status = 200;
    response->content_type = "text/event-stream";
    response->set_req(&req);
    auto shared_chunks = std::make_shared<std::vector<std::string>>(std::move(chunks));
    auto index = std::make_shared<size_t>(0);
    response->set_next([shared_chunks, index, delay_ms](std::string & output) {
        if (*index >= shared_chunks->size()) {
            output.clear();
            return false;
        }
        if (delay_ms != 0 && *index != 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        output = (*shared_chunks)[(*index)++];
        return *index < shared_chunks->size();
    });
    return response;
}

class http_thread_guard {
  public:
    explicit http_thread_guard(server_http_context & input) : http(input) {}
    ~http_thread_guard() {
        http.stop();
        if (http.thread.joinable()) http.thread.join();
    }
  private:
    server_http_context & http;
};

void test_http_capture() {
    temporary_directory temp;
    const fs::path root = temp.path() / "history";
    const std::string auth_secret = "history-auth-secret-must-not-persist-927451";
    const std::string completion_request =
        "{\"model\":\"deepseek-v4-flash\",\"profile\":\"throughput\",\"prompt\":\"raw\\ninput\"}";
    const std::string completion_response =
        "{\"choices\":[{\"text\":\"done\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}";
    const std::string validation_request = "{not-json";
    const std::string validation_response = "{\"error\":{\"message\":\"invalid json\"}}";
    const std::string large_request = "{\"model\":\"offline-only\",\"prompt\":\"" +
        std::string(128 * 1024, 'p') + "\"}";
    const std::string large_response = "{\"padding\":\"" + std::string(128 * 1024, 'r') +
        "\",\"usage\":{\"must\":\"remain-offline\"}}";
    const std::string response_request = "{\"model\":\"deepseek-v4-flash\",\"stream\":true}";
    const std::vector<std::string> response_chunks = {
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"A\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":2,\"output_tokens\":1}}}\n\n",
    };
    const std::string anthropic_request =
        "{\"model\":\"deepseek-v4-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"cancel\"}],\"stream\":true}";
    const std::vector<std::string> anthropic_chunks = {
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"one\"}}\n\n",
        "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\"two\"}}\n\n",
        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
    };
    const std::string resumable_request = "{\"stream\":true,\"prompt\":\"resume\"}";
    const std::vector<std::string> resumable_chunks = {
        "data: {\"text\":\"first\"}\n\n",
        "data: {\"text\":\"literal event: error is generated text\"}\n\n",
        "data: {\"text\":\"tail\"}\n\n",
    };
    const std::string error_stream_request = "{\"stream\":true,\"prompt\":\"error\"}";
    const std::vector<std::string> error_stream_chunks = {
        "data: {\"error\":{\"message\":\"failed\"}}\n\n",
    };
    const std::string multipart_boundary = "llama-history-boundary-8f41c6";
    std::string audio_bytes("RIFF\0history-audio\r\nbytes", 25);
    const std::string multipart_request =
        "--" + multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n"
        "whisper-1\r\n"
        "--" + multipart_boundary + "\r\n"
        "Content-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n" + audio_bytes + "\r\n"
        "--" + multipart_boundary + "--\r\n";
    const std::string audio_response = "{\"text\":\"decoded\"}";

    {
        common_params params;
        params.hostname = "127.0.0.1";
        params.port = 0;
        params.ui = false;
        params.n_threads_http = 4;
        params.api_keys = { auth_secret };
        params.request_history_root = root.string();
        params.request_history_max_gib = 1;
        params.request_history_max_files = 100;
        params.request_history_ack_ms = 5000;

        server_http_context http;
        expect(http.init(params), "HTTP context init");
        http.post("/v1/completions", [completion_response](const server_http_req &) {
            return normal_response(completion_response);
        });
        http.post("/v1/chat/completions", [validation_response](const server_http_req &) {
            return normal_response(validation_response, 400);
        });
        http.post("/completion", [large_response](const server_http_req &) {
            return normal_response(large_response);
        });
        http.post("/v1/responses", [response_chunks](const server_http_req &) {
            return stream_response(response_chunks);
        });
        http.post("/v1/messages", [anthropic_chunks](const server_http_req &) {
            return stream_response(anthropic_chunks, 30);
        });
        http.post("/chat/completions", [resumable_chunks](const server_http_req & req) {
            return resumable_stream_response(req, resumable_chunks, 15);
        });
        http.post("/responses", [error_stream_chunks](const server_http_req &) {
            return stream_response(error_stream_chunks);
        });
        http.post("/v1/audio/transcriptions", [audio_bytes, audio_response](const server_http_req & req) {
            expect(req.body == "{\"model\":\"whisper-1\"}", "multipart fields transformed for handler");
            expect(req.files.at("file").data == raw_buffer(audio_bytes.begin(), audio_bytes.end()),
                   "multipart file parsed for handler");
            return normal_response(audio_response);
        });
        expect(http.start(), "HTTP context start");
        http_thread_guard running(http);
        http.is_ready.store(true);

        httplib::Client client("127.0.0.1", http.port);
        client.set_read_timeout(5, 0);
        httplib::Headers headers = {
            { "Authorization", "Bearer " + auth_secret },
            // Identity leaves the decoded application entity unchanged. The
            // history contract is post transfer/content decoding, not raw
            // compressed or chunk-framed wire bytes.
            { "Content-Encoding", "identity" },
        };

        auto completion = client.Post("/v1/completions", headers, completion_request, "application/json");
        expect(completion && completion->status == 200 && completion->body == completion_response,
               "nonstream HTTP response");
        auto validation = client.Post("/v1/chat/completions", headers, validation_request, "application/json");
        expect(validation && validation->status == 400 && validation->body == validation_response,
               "validation HTTP response");
        auto large = client.Post("/completion", headers, large_request, "application/json");
        expect(large && large->status == 200 && large->body == large_response, "large HTTP response");
        auto streamed = client.Post("/v1/responses", headers, response_request, "application/json");
        expect(streamed && streamed->status == 200 && streamed->body == response_chunks[0] + response_chunks[1],
               "stream HTTP response");

        size_t receives = 0;
        auto cancelled = client.Post(
            "/v1/messages", headers, anthropic_request, "application/json",
            [&receives](const char *, size_t) {
                return ++receives < 2;
            });
        (void) cancelled;
        expect(receives >= 1, "cancelled stream receiver invoked");
        httplib::Headers resumable_headers = headers;
        resumable_headers.emplace("X-Conversation-Id", "history-resume-1");
        size_t resumable_receives = 0;
        auto resumable = client.Post(
            "/chat/completions", resumable_headers, resumable_request, "application/json",
            [&resumable_receives](const char *, size_t) { return ++resumable_receives < 2; });
        (void) resumable;
        expect(resumable_receives >= 1, "resumable stream disconnected");
        auto error_stream = client.Post("/responses", headers, error_stream_request, "application/json");
        expect(error_stream && error_stream->body == error_stream_chunks[0], "data JSON error stream response");
        auto audio = client.Post(
            "/v1/audio/transcriptions", headers, multipart_request,
            "multipart/form-data; boundary=" + multipart_boundary);
        expect(audio && audio->status == 200 && audio->body == audio_response,
               "multipart audio HTTP response");
        auto unauthorized_audio = client.Post(
            "/v1/audio/transcriptions", multipart_request,
            "multipart/form-data; boundary=" + multipart_boundary);
        expect(unauthorized_audio && unauthorized_audio->status == 401,
               "unauthorized multipart rejected before capture predicate");
        auto unmatched_audio = client.Post(
            "/not-an-inference-route", headers, multipart_request,
            "multipart/form-data; boundary=" + multipart_boundary);
        expect(unmatched_audio && unmatched_audio->status == 404,
               "unmatched multipart does not enter history");
        auto malformed_audio = client.Post(
            "/v1/audio/transcriptions", headers, "not-a-valid-multipart-entity",
            "multipart/form-data; boundary=invalid-boundary");
        expect(malformed_audio && malformed_audio->status == 400,
               "malformed multipart has no completed history record");

        // Terminal callbacks wait for durable publication, so by the time the
        // client calls return all eight records are inspectable.
        reader inspection(root.string());
        read_limits limits;
        limits.max_files = 16;
        limits.max_file_bytes = 1024 * 1024;
        limits.max_total_body_bytes = 1024 * 1024;
        std::vector<record_metadata> metadata;
        result inspected;
        for (int attempt = 0; attempt < 100; ++attempt) {
            inspected = inspection.inspect(limits, metadata);
            if (inspected.value == status::ok && metadata.size() == 8) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        expect(inspected.value == status::ok && metadata.size() == 8, "eight HTTP records durable");
        std::map<std::string, record> by_path;
        for (const record_metadata & item : metadata) {
            record value;
            expect(inspection.read(item.file_name, limits, value).value == status::ok, "read HTTP record");
            by_path.emplace(value.metadata.path, std::move(value));
        }
        expect(by_path.at("/v1/completions").request_body == completion_request &&
               by_path.at("/v1/completions").response_body == completion_response,
               "HTTP nonstream exact entities");
        expect(by_path.at("/v1/completions").metadata.requested_model == "deepseek-v4-flash" &&
               by_path.at("/v1/completions").metadata.requested_profile == "profile=throughput",
               "HTTP requested model/profile metadata");
        expect(by_path.at("/v1/completions").metadata.usage_json ==
               "{\"prompt_tokens\":3,\"completion_tokens\":1}", "HTTP nonstream usage metadata: " +
               by_path.at("/v1/completions").metadata.usage_json);
        expect(by_path.at("/v1/chat/completions").request_body == validation_request &&
               by_path.at("/v1/chat/completions").metadata.terminal_outcome == outcome::error,
               "handler validation error captured");
        expect(by_path.at("/completion").request_body == large_request &&
               by_path.at("/completion").response_body == large_response,
               "large entities persist with exact byte equality");
        expect(by_path.at("/completion").metadata.requested_model.empty() &&
               by_path.at("/completion").metadata.usage_json.empty(),
               "large-entity metadata parsing is skipped at the documented bound");
        expect(by_path.at("/v1/responses").response_stream_chunks == response_chunks,
               "HTTP stream chunks exact and ordered");
        expect(by_path.at("/v1/responses").metadata.terminal_outcome == outcome::complete,
               "clean HTTP stream terminal outcome");
        expect(by_path.at("/v1/responses").metadata.usage_json == "{\"input_tokens\":2,\"output_tokens\":1}",
               "HTTP streaming usage metadata");
        expect(by_path.at("/v1/messages").request_body == anthropic_request &&
               by_path.at("/v1/messages").metadata.terminal_outcome == outcome::aborted,
               "HTTP cancellation outcome");
        expect(by_path.at("/v1/audio/transcriptions").request_body == multipart_request &&
               by_path.at("/v1/audio/transcriptions").response_body == audio_response,
               "multipart decoded application entity preserves exact boundaries and binary bytes");
        expect(by_path.at("/chat/completions").response_stream_chunks == resumable_chunks &&
               by_path.at("/chat/completions").metadata.terminal_outcome == outcome::complete &&
               by_path.at("/chat/completions").metadata.transport_complete_known &&
               !by_path.at("/chat/completions").metadata.transport_complete,
               "resumable disconnect captures full producer tail and completes generation: chunks=" +
               std::to_string(by_path.at("/chat/completions").response_stream_chunks.size()) +
               " outcome=" + outcome_name(by_path.at("/chat/completions").metadata.terminal_outcome));
        expect(by_path.at("/responses").metadata.terminal_outcome == outcome::error,
               "SSE data JSON error is structurally classified");
        expect(by_path.at("/chat/completions").metadata.terminal_outcome != outcome::error,
               "generated text containing event marker is not an SSE error false positive");

        for (const fs::directory_entry & entry : fs::directory_iterator(root)) {
            if (!entry.is_regular_file()) continue;
            std::ifstream input(entry.path(), std::ios::binary);
            const std::string bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            expect(bytes.find(auth_secret) == std::string::npos, "authorization secret absent from history files");
        }

    }
}

}  // namespace

int main() {
    try {
        test_http_capture();
    } catch (const std::exception & error) {
        std::cerr << "test-server-request-history-http: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-server-request-history-http: all checks passed\n";
    return 0;
}

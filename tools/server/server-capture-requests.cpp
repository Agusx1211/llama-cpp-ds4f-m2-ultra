#include "server-capture-requests.h"

#include "server-capture-sha256.h"
#include "server-capture-store.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace server_capture {
namespace {

constexpr uint32_t REQUEST_LOG_SCHEMA_VERSION = 1;
constexpr mode_t   PRIVATE_FILE_MODE          = S_IRUSR | S_IWUSR;
constexpr mode_t   PRIVATE_DIRECTORY_MODE     = S_IRWXU;

std::string hex_bytes(const uint8_t * bytes, size_t count) {
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(count * 2);
    for (size_t i = 0; i < count; ++i) {
        out.push_back(digits[bytes[i] >> 4]);
        out.push_back(digits[bytes[i] & 0x0f]);
    }
    return out;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

void append_token_array(std::string & out, const char * key, const std::vector<int32_t> & tokens) {
    out += '"';
    out += key;
    out += "\":[";
    char buf[16];
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i != 0) {
            out += ',';
        }
        std::snprintf(buf, sizeof(buf), "%d", tokens[i]);
        out += buf;
    }
    out += ']';
}

bool ensure_private_directory(const std::string & path, std::string & error) {
    std::error_code ec;
    fs::create_directories(path, ec);
    if (ec) {
        error = "create_directories failed: " + ec.message();
        return false;
    }
    struct stat st = {};
    if (lstat(path.c_str(), &st) != 0) {
        error = std::string("lstat failed: ") + std::strerror(errno);
        return false;
    }
    if (!S_ISDIR(st.st_mode)) {
        error = "request log root is not a directory";
        return false;
    }
    if (st.st_uid != geteuid()) {
        error = "request log root is not owned by the server user";
        return false;
    }
    if ((st.st_mode & (S_IRWXG | S_IRWXO)) != 0 && chmod(path.c_str(), PRIVATE_DIRECTORY_MODE) != 0) {
        error = std::string("chmod 0700 failed: ") + std::strerror(errno);
        return false;
    }
    return true;
}

}  // namespace

struct request_log::impl {
    request_log_config config;

    std::mutex                 mutex;
    std::condition_variable    wake;
    std::deque<request_record> queue;
    bool                       stopping = false;

    std::atomic<uint64_t> enqueued{ 0 };
    std::atomic<uint64_t> dropped{ 0 };
    std::atomic<uint64_t> written_records{ 0 };
    std::atomic<uint64_t> written_bytes{ 0 };
    std::atomic<uint64_t> failed_writes{ 0 };
    std::atomic<bool>     worker_failed{ false };

    int      fd            = -1;
    uint64_t fd_bytes      = 0;
    uint64_t file_sequence = 0;

    std::thread worker;

    explicit impl(request_log_config cfg) : config(std::move(cfg)) {
        std::string error;
        if (config.root_path.empty() || config.queue_capacity == 0 || config.max_file_bytes == 0 ||
            config.max_total_bytes < config.max_file_bytes) {
            throw std::runtime_error("invalid request log configuration");
        }
        if (!ensure_private_directory(config.root_path, error)) {
            throw std::runtime_error("request log root rejected: " + error);
        }
        worker = std::thread([this]() { run(); });
    }

    ~impl() { stop(); }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping && !worker.joinable()) {
                return;
            }
            stopping = true;
        }
        wake.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        close_current();
    }

    void close_current() {
        if (fd >= 0) {
            ::close(fd);
            fd       = -1;
            fd_bytes = 0;
        }
    }

    std::string file_name(uint64_t sequence) const {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "requests-%08llu.jsonl", static_cast<unsigned long long>(sequence));
        return buf;
    }

    bool open_next_file() {
        close_current();

        // continue after the highest existing sequence so restarts never
        // append to (or re-create) a file that retention already accounted for
        uint64_t        next = file_sequence;
        std::error_code ec;
        for (const auto & entry : fs::directory_iterator(config.root_path, ec)) {
            unsigned long long seen = 0;
            if (std::sscanf(entry.path().filename().string().c_str(), "requests-%08llu.jsonl", &seen) == 1) {
                next = std::max<uint64_t>(next, seen + 1);
            }
        }

        const std::string path = config.root_path + "/" + file_name(next);
        fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_APPEND | O_CLOEXEC, PRIVATE_FILE_MODE);
        if (fd < 0) {
            failed_writes.fetch_add(1, std::memory_order_relaxed);
            worker_failed.store(true, std::memory_order_relaxed);
            return false;
        }
        file_sequence = next + 1;
        fd_bytes      = 0;

        std::string meta;
        meta.reserve(512);
        meta += "{\"kind\":\"meta\",\"schema_version\":";
        meta += std::to_string(REQUEST_LOG_SCHEMA_VERSION);
        meta += ",\"runtime_commit\":\"" + json_escape(config.runtime_commit) + "\"";
        meta += ",\"model\":\"" + json_escape(config.model_identity) + "\"";
        meta += ",\"draft\":\"" + json_escape(config.draft_identity) + "\"";
        meta += ",\"speculative\":\"" + json_escape(config.speculative_config) + "\"";
        meta += ",\"time_unix\":" + std::to_string((long long) ::time(nullptr));
        meta += "}\n";
        return write_line(meta);
    }

    bool write_line(const std::string & line) {
        if (fd < 0) {
            return false;
        }
        size_t offset = 0;
        while (offset < line.size()) {
            const ssize_t written = ::write(fd, line.data() + offset, line.size() - offset);
            if (written < 0) {
                if (errno == EINTR) {
                    continue;
                }
                failed_writes.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            offset += static_cast<size_t>(written);
        }
        fd_bytes += line.size();
        written_bytes.fetch_add(line.size(), std::memory_order_relaxed);
        return true;
    }

    void enforce_retention() {
        std::error_code                                ec;
        std::vector<std::pair<uint64_t, fs::path>>     files;
        uint64_t                                       total = 0;
        for (const auto & entry : fs::directory_iterator(config.root_path, ec)) {
            unsigned long long seen = 0;
            if (std::sscanf(entry.path().filename().string().c_str(), "requests-%08llu.jsonl", &seen) != 1) {
                continue;
            }
            const uint64_t size = static_cast<uint64_t>(fs::file_size(entry.path(), ec));
            total += size;
            files.emplace_back(seen, entry.path());
        }
        std::sort(files.begin(), files.end());
        // never delete the newest (open) file
        for (size_t i = 0; i + 1 < files.size() && total > config.max_total_bytes; ++i) {
            const uint64_t size = static_cast<uint64_t>(fs::file_size(files[i].second, ec));
            if (fs::remove(files[i].second, ec)) {
                total -= size;
            }
        }
    }

    std::string serialize(const request_record & record) const {
        const capture_digest tag = capture_request_tag(config.identity_salt, record.request_id);

        std::string line;
        line.reserve(64 + record.prompt_tokens.size() * 7 + record.generated_tokens.size() * 7);
        line += "{\"kind\":\"request\",\"tag\":\"";
        line += hex_bytes(tag.data(), 8);
        line += "\",\"slot\":" + std::to_string(record.slot_id);
        line += ",\"time_unix\":" + std::to_string((long long) record.wall_time_s);
        line += ",\"mono_ns\":" + std::to_string((unsigned long long) record.monotonic_ns);
        line += ",\"n_prompt\":" + std::to_string(record.prompt_tokens.size());
        line += ",\"n_prompt_cached\":" + std::to_string(record.n_prompt_cached);
        line += ",\"n_decoded\":" + std::to_string(record.n_decoded);
        line += ",\"n_draft_total\":" + std::to_string(record.n_draft_total);
        line += ",\"n_draft_accepted\":" + std::to_string(record.n_draft_accepted);
        line += ",\"truncated\":" + std::string(record.truncated ? "true" : "false");
        char buf[64];
        std::snprintf(buf, sizeof(buf), ",\"prompt_ms\":%.3f,\"generation_ms\":%.3f", record.prompt_ms,
                      record.generation_ms);
        line += buf;
        std::snprintf(buf, sizeof(buf), ",\"temp\":%.6g,\"top_k\":%d,\"top_p\":%.6g,\"min_p\":%.6g,\"seed\":%u",
                      record.temperature, record.top_k, record.top_p, record.min_p, record.seed);
        line += buf;
        if (!record.prompt_tokens.empty()) {
            const capture_sha256::digest prompt_digest = capture_sha256::hash(
                    record.prompt_tokens.data(), record.prompt_tokens.size() * sizeof(int32_t));
            line += ",\"prompt_sha256\":\"" + hex_bytes(prompt_digest.data(), prompt_digest.size()) + "\"";
        }
        line += ',';
        append_token_array(line, "prompt_tokens", record.prompt_tokens);
        line += ',';
        append_token_array(line, "generated_tokens", record.generated_tokens);
        line += "}\n";
        return line;
    }

    void run() {
        for (;;) {
            request_record record;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this]() { return stopping || !queue.empty(); });
                if (queue.empty()) {
                    return;  // stopping and drained
                }
                record = std::move(queue.front());
                queue.pop_front();
            }

            const std::string line = serialize(record);
            if (fd < 0 || fd_bytes + line.size() > config.max_file_bytes) {
                if (open_next_file()) {
                    enforce_retention();
                }
            }
            if (write_line(line)) {
                written_records.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
};

request_log::request_log(request_log_config config) : data(new impl(std::move(config))) {}

request_log::~request_log() = default;

bool request_log::try_enqueue(request_record && record) noexcept {
    try {
        {
            std::lock_guard<std::mutex> lock(data->mutex);
            if (data->stopping || data->queue.size() >= data->config.queue_capacity) {
                data->dropped.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            data->queue.push_back(std::move(record));
        }
        data->enqueued.fetch_add(1, std::memory_order_relaxed);
        data->wake.notify_one();
        return true;
    } catch (...) {
        data->dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
}

request_log_stats request_log::stats() const noexcept {
    request_log_stats out;
    out.enqueued        = data->enqueued.load(std::memory_order_relaxed);
    out.dropped         = data->dropped.load(std::memory_order_relaxed);
    out.written_records = data->written_records.load(std::memory_order_relaxed);
    out.written_bytes   = data->written_bytes.load(std::memory_order_relaxed);
    out.failed_writes   = data->failed_writes.load(std::memory_order_relaxed);
    out.worker_failed   = data->worker_failed.load(std::memory_order_relaxed);
    return out;
}

void request_log::shutdown() {
    data->stop();
}

}  // namespace server_capture

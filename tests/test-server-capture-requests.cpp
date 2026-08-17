#include "server-capture-requests.h"
#include "server-capture-store.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace server_capture;

static void fail(const std::string & message) {
    std::cerr << "FAIL: " << message << "\n";
    std::exit(1);
}

static void expect(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

static std::string make_temp_root() {
    const char * base = std::getenv("TMPDIR");
    std::string  pattern =
            std::string(base != nullptr && base[0] != '\0' ? base : "/tmp") + "/capture-requests-XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char * path = ::mkdtemp(writable.data());
    if (path == nullptr) {
        fail(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return path;
}

static std::vector<std::string> sorted_log_files(const std::string & root) {
    std::vector<std::string> files;
    for (const auto & entry : fs::directory_iterator(root)) {
        files.push_back(entry.path().filename().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream            input(path);
    std::vector<std::string> lines;
    std::string              line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return lines;
}

static request_record make_record(uint64_t request_id) {
    request_record record;
    record.request_id       = request_id;
    record.slot_id          = 3;
    record.wall_time_s      = 1755400000;
    record.monotonic_ns     = 42;
    record.prompt_tokens    = { 1, 2, 3, 4 };
    record.generated_tokens = { 9, 8, 7 };
    record.n_prompt_cached  = 2;
    record.n_decoded        = 3;
    record.n_draft_total    = 10;
    record.n_draft_accepted = 6;
    record.prompt_ms        = 12.5;
    record.generation_ms    = 99.25;
    record.temperature      = 0.7f;
    record.top_k            = 40;
    record.top_p            = 0.95f;
    record.min_p            = 0.05f;
    record.seed             = 1234;
    return record;
}

static void wait_for_written(const request_log & log, uint64_t records) {
    for (int attempt = 0; attempt < 2000; ++attempt) {
        if (log.stats().written_records >= records) {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    fail("request log worker did not write in time");
}

static void test_basic_write_and_join_tag() {
    const std::string root = make_temp_root();

    request_log_config config;
    config.root_path = root + "/requests";
    for (size_t i = 0; i < config.identity_salt.size(); ++i) {
        config.identity_salt[i] = static_cast<uint8_t>(i + 1);
    }
    config.runtime_commit     = "test-build";
    config.model_identity     = "model.gguf";
    config.draft_identity     = "draft.gguf";
    config.speculative_config = "ngram-mod,draft-dspark n_max=3";

    request_log log(config);
    expect(log.try_enqueue(make_record(7)), "first enqueue accepted");
    expect(log.try_enqueue(make_record(8)), "second enqueue accepted");
    wait_for_written(log, 2);
    log.shutdown();

    struct stat status = {};
    expect(::lstat(config.root_path.c_str(), &status) == 0 && S_ISDIR(status.st_mode), "root exists");
    expect((status.st_mode & (S_IRWXG | S_IRWXO)) == 0, "root is private");

    const auto files = sorted_log_files(config.root_path);
    expect(files.size() == 1, "single file before rotation");

    const auto lines = read_lines(config.root_path + "/" + files[0]);
    expect(lines.size() == 3, "meta line plus two records");
    expect(lines[0].find("\"kind\":\"meta\"") != std::string::npos, "meta first");
    expect(lines[0].find("\"model\":\"model.gguf\"") != std::string::npos, "meta carries model identity");
    expect(lines[1].find("\"kind\":\"request\"") != std::string::npos, "request record kind");
    expect(lines[1].find("\"prompt_tokens\":[1,2,3,4]") != std::string::npos, "prompt tokens serialized");
    expect(lines[1].find("\"generated_tokens\":[9,8,7]") != std::string::npos, "generated tokens serialized");
    expect(lines[1].find("\"n_draft_accepted\":6") != std::string::npos, "spec telemetry serialized");
    expect(lines[1].find("\"prompt_sha256\":\"") != std::string::npos, "prompt digest serialized");

    // the persisted tag must equal the salted tag of the cycle store so the
    // two logs can be joined offline
    const capture_digest tag      = capture_request_tag(config.identity_salt, 7);
    std::string          tag_hex;
    static const char    digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 8; ++i) {
        tag_hex.push_back(digits[tag[i] >> 4]);
        tag_hex.push_back(digits[tag[i] & 0x0f]);
    }
    expect(lines[1].find("\"tag\":\"" + tag_hex + "\"") != std::string::npos, "record carries salted request tag");
    expect(lines[1].find("\"request_id\"") == std::string::npos, "raw request id never persisted");

    fs::remove_all(root);
}

static void test_rotation_and_retention() {
    const std::string root = make_temp_root();

    request_log_config config;
    config.root_path      = root + "/requests";
    config.max_file_bytes = 700;   // meta plus about one record per file
    config.max_total_bytes = 2100; // keep roughly the newest three files

    request_log log(config);
    for (uint64_t i = 0; i < 12; ++i) {
        expect(log.try_enqueue(make_record(i)), "enqueue during rotation");
        wait_for_written(log, i + 1);
    }
    log.shutdown();

    const auto files = sorted_log_files(config.root_path);
    expect(files.size() >= 2, "rotation produced multiple files");

    uint64_t total = 0;
    for (const auto & name : files) {
        total += static_cast<uint64_t>(fs::file_size(config.root_path + "/" + name));
    }
    // the open file may exceed the budget by one record; older files must
    // have been deleted rather than accumulating all twelve
    expect(files.size() < 12, "retention deleted old files");
    expect(total <= config.max_total_bytes + 1400, "retention bounded total bytes");

    // every surviving file starts with its own meta line
    for (const auto & name : files) {
        const auto lines = read_lines(config.root_path + "/" + name);
        expect(!lines.empty() && lines[0].find("\"kind\":\"meta\"") != std::string::npos,
               "rotated file begins with meta");
    }

    const auto stats = log.stats();
    expect(stats.written_records == 12, "all records written");
    expect(stats.dropped == 0, "no drops in drain test");

    fs::remove_all(root);
}

static void test_invalid_config_rejected() {
    bool threw = false;
    try {
        request_log_config config;  // empty root
        request_log        log(config);
    } catch (const std::exception &) {
        threw = true;
    }
    expect(threw, "empty root rejected");
}

int main() {
    test_basic_write_and_join_tag();
    test_rotation_and_retention();
    test_invalid_config_rejected();
    std::cout << "OK\n";
    return 0;
}

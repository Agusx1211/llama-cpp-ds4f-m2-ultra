#include "server-request-history.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace server_request_history;

namespace {

[[noreturn]] void fail(const std::string & message) { throw std::runtime_error(message); }

void expect(bool condition, const std::string & message) {
    if (!condition) fail(message);
}

void expect_status(const result & actual, status expected, const std::string & message) {
    if (actual.value != expected) {
        fail(message + ": expected " + status_name(expected) + ", got " + status_name(actual.value));
    }
}

class temporary_directory {
  public:
    temporary_directory() {
        const char * configured = std::getenv("TMPDIR");
        fs::path base = configured != nullptr && *configured != '\0' ? fs::path(configured) : fs::path("/tmp");
        std::error_code error;
        base = fs::canonical(base, error);
        if (error || !base.is_absolute()) fail("canonical temp root");
        std::string pattern = (base / "llama-request-history-XXXXXX").string();
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

config make_config(const fs::path & root) {
    config cfg;
    cfg.root_path = root.string();
    cfg.max_frame_payload = 256;
    cfg.max_pending_bytes = 1024;
    cfg.max_retained_bytes = 1024 * 1024;
    cfg.max_retained_files = 64;
    cfg.terminal_ack_timeout_ms = 5000;
    return cfg;
}

read_limits generous_limits() {
    read_limits limits;
    limits.max_files = 1000;
    limits.max_file_bytes = 1024 * 1024;
    limits.max_total_body_bytes = 1024 * 1024;
    return limits;
}

std::vector<record_metadata> inspect_records(const fs::path & root) {
    reader inspection(root.string());
    std::vector<record_metadata> records;
    expect_status(inspection.inspect(generous_limits(), records), status::ok, "inspect records");
    return records;
}

record read_only_record(const fs::path & root) {
    reader inspection(root.string());
    std::vector<record_metadata> records;
    expect_status(inspection.inspect(generous_limits(), records), status::ok, "inspect one record");
    expect(records.size() == 1, "expected exactly one record");
    record output;
    expect_status(inspection.read(records[0].file_name, generous_limits(), output), status::ok, "read one record");
    return output;
}

void write_exact_fd(int descriptor, const std::string & bytes) {
    size_t offset = 0;
    while (offset < bytes.size()) {
        const ssize_t amount = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (amount < 0 && errno == EINTR) continue;
        expect(amount > 0, "write replacement inode");
        offset += static_cast<size_t>(amount);
    }
}

void exchange_same_shape_file(const fs::path & root, const std::string & name, const std::string & backup_name) {
    const fs::path source = root / name;
    const fs::path backup = root / backup_name;
    struct stat original = {};
    expect(::lstat(source.c_str(), &original) == 0 && S_ISREG(original.st_mode) && original.st_size >= 0,
           "stat race source");
    expect(::rename(source.c_str(), backup.c_str()) == 0, "exchange race source");
    const int replacement = ::open(source.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    expect(replacement >= 0, "create replacement inode");
    const std::string content(static_cast<size_t>(original.st_size), 'X');
    write_exact_fd(replacement, content);
    expect(::fchmod(replacement, 0600) == 0 && ::fsync(replacement) == 0 && ::close(replacement) == 0,
           "sync replacement inode");
    struct stat replaced = {};
    expect(::lstat(source.c_str(), &replaced) == 0 && S_ISREG(replaced.st_mode) &&
           replaced.st_size == original.st_size && (replaced.st_mode & 0777) == (original.st_mode & 0777) &&
           (replaced.st_dev != original.st_dev || replaced.st_ino != original.st_ino),
           "same-size same-mode replacement inode");
}

void test_exact_nonstream_and_stream_bytes() {
    temporary_directory temp;
    std::string request_body(600, 'q');
    request_body[3] = '\0';
    request_body[511] = static_cast<char>(0xff);
    std::string response_body(600, 'r');
    response_body[4] = '\0';
    response_body[512] = static_cast<char>(0xfe);
    const std::string model("model\0exact", 11);
    const std::string profile = "latency";
    {
        store history(make_config(temp.path()));
        auto session = history.begin("POST", "/v1/chat/completions", request_body, model, profile);
        expect(session != nullptr, "nonstream begin");
        expect(session->append_response(response_body, false), "append nonstream response");
        expect(session->set_usage_json("{\"t\":9}"), "set usage");
        result finished = session->finish(200, outcome::complete, false);
        expect_status(finished, status::ok, "nonstream finish");
        expect(finished.durable, "nonstream finish durable ack");
        expect_status(history.shutdown(), status::ok, "nonstream shutdown");
    }
    record stored = read_only_record(temp.path());
    expect(stored.request_body == request_body, "raw request exact equality");
    expect(stored.response_body == response_body, "nonstream response exact equality");
    expect(stored.metadata.requested_model == model, "model metadata exact");
    expect(stored.metadata.requested_profile == profile, "profile metadata exact");
    expect(stored.metadata.usage_json == "{\"t\":9}", "usage exact");
    expect(stored.metadata.terminal_outcome == outcome::complete && stored.metadata.http_status == 200,
           "terminal metadata");

    temporary_directory stream_temp;
    std::string chunk1 = "data: {\"x\":\"a";
    chunk1.push_back('\0');
    chunk1 += "b\"}\n\n";
    const std::string chunk2("data: [DONE]\n\n", 14);
    {
        store history(make_config(stream_temp.path()));
        auto session = history.begin("POST", "/v1/responses", "{\"stream\":true}");
        expect(session && session->append_response(chunk1, true) && session->append_response(chunk2, true),
               "stream chunks append");
        result finished = session->finish(200, outcome::complete, true, true, true);
        expect_status(finished, status::ok, "stream finish");
        expect(finished.durable, "stream durable ack");
    }
    stored = read_only_record(stream_temp.path());
    expect(stored.response_stream_chunks.size() == 2, "preserve stream chunk count");
    expect(stored.response_stream_chunks[0] == chunk1 && stored.response_stream_chunks[1] == chunk2,
           "preserve stream chunks and boundaries exactly");
    expect(stored.metadata.streaming && stored.metadata.response_bytes == chunk1.size() + chunk2.size(),
           "stream byte metadata");
    expect(stored.metadata.transport_complete_known && stored.metadata.transport_complete,
           "successful provider transport flag round trip");
}

void test_outcomes_and_route_policy() {
    temporary_directory temp;
    {
        store history(make_config(temp.path()));
        auto aborted = history.begin("POST", "/v1/messages", "bad body");
        expect(aborted && aborted->append_response("event: error\n\n", true), "aborted append");
        expect_status(aborted->finish(200, outcome::aborted, true, false, true), status::ok, "aborted finish");
        auto error = history.begin("POST", "/v1/completions", "{bad json");
        expect(error && error->append_response("{\"error\":\"invalid\"}", false), "error append");
        expect_status(error->finish(400, outcome::error, false), status::ok, "error finish");
    }
    const auto records = inspect_records(temp.path());
    expect(records.size() == 2, "two outcome records");
    expect(records[0].terminal_outcome == outcome::aborted && records[0].streaming &&
           records[0].transport_complete_known && !records[0].transport_complete,
           "aborted outcome and failed provider transport persisted");
    expect(records[1].terminal_outcome == outcome::error && records[1].http_status == 400, "error outcome persisted");

    const std::vector<std::string> required = {
        "/completion", "/v1/completions", "/chat/completions", "/v1/chat/completions",
        "/responses", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens",
    };
    for (const std::string & path : required) {
        expect(should_capture("POST", path), "capture required route " + path);
        expect(should_capture("POST", "/api" + path, "/api"), "capture prefixed route " + path);
    }
    expect(!should_capture("GET", "/v1/messages"), "do not capture GET inference alias");
    expect(!should_capture("POST", "/v1/models"), "do not capture administration");
    expect(!should_capture("POST", "/notapi/v1/messages", "/api"), "prefix boundary exact");
}

void test_concurrency() {
    temporary_directory temp;
    config cfg = make_config(temp.path());
    cfg.max_frame_payload = 256;
    cfg.max_pending_bytes = 4096;
    constexpr int THREADS = 8;
    constexpr int PER_THREAD = 12;
    cfg.max_retained_files = THREADS * PER_THREAD + 8;
    {
        store history(cfg);
        std::vector<std::thread> workers;
        for (int thread = 0; thread < THREADS; ++thread) {
            workers.emplace_back([&history, thread] {
                for (int i = 0; i < PER_THREAD; ++i) {
                    const std::string body = "request-" + std::to_string(thread) + "-" + std::to_string(i);
                    auto session = history.begin("POST", "/v1/completions", body);
                    expect(session != nullptr, "concurrent begin");
                    expect(session->append_response(body, false), "concurrent response");
                    result finished = session->finish(200, outcome::complete, false);
                    expect(finished.value == status::ok && finished.durable, "concurrent durable finish");
                }
            });
        }
        for (auto & worker : workers) worker.join();
        expect(history.get_stats().durable_records == THREADS * PER_THREAD, "concurrent durable stats");
    }
    auto records = inspect_records(temp.path());
    expect(records.size() == THREADS * PER_THREAD, "all concurrent records retained");
    for (size_t i = 1; i < records.size(); ++i) {
        expect(records[i - 1].record_id != records[i].record_id, "unique concurrent IDs");
    }
}

void test_restart_recovery() {
    temporary_directory incomplete_temp;
    config incomplete_cfg = make_config(incomplete_temp.path());
    incomplete_cfg.test_faults.preserve_open_on_shutdown = true;
    {
        store history(incomplete_cfg);
        auto session = history.begin("POST", "/v1/responses", "request");
        expect(session && session->append_response("partial", true), "incomplete fixture");
        expect_status(history.flush(), status::ok, "incomplete flush");
        expect_status(history.shutdown(), status::ok, "preserved-open shutdown");
        session.reset();
    }
    {
        config recovered_cfg = make_config(incomplete_temp.path());
        store recovered(recovered_cfg);
        expect(recovered.get_stats().recovered_records == 1, "startup recovered incomplete");
    }
    record recovered = read_only_record(incomplete_temp.path());
    expect(recovered.metadata.terminal_outcome == outcome::recovered_incomplete && recovered.metadata.recovered,
           "honest recovered-incomplete terminal");
    expect(recovered.request_body == "request" && recovered.response_stream_chunks.size() == 1 &&
           recovered.response_stream_chunks[0] == "partial", "recovered exact valid prefix");

    temporary_directory before_temp;
    config before_cfg = make_config(before_temp.path());
    before_cfg.test_faults.crash_before_rename = true;
    {
        store history(before_cfg);
        auto session = history.begin("POST", "/v1/completions", "input");
        expect(session && session->append_response("output", false), "pre-rename fixture");
        expect_status(session->finish(200, outcome::complete, false), status::commit_uncertain,
                      "pre-rename does not claim durable");
    }
    {
        store recovered(make_config(before_temp.path()));
        expect(recovered.get_stats().recovered_records == 1, "completed temp recovered");
    }
    recovered = read_only_record(before_temp.path());
    expect(recovered.metadata.terminal_outcome == outcome::complete && recovered.response_body == "output",
           "completed-before-rename published on restart");

    temporary_directory after_temp;
    config after_cfg = make_config(after_temp.path());
    after_cfg.test_faults.crash_after_rename = true;
    {
        store history(after_cfg);
        auto session = history.begin("POST", "/v1/messages", "input");
        expect(session && session->append_response("output", false), "post-rename fixture");
        expect_status(session->finish(200, outcome::complete, false), status::commit_uncertain,
                      "post-rename does not claim dir durability");
    }
    {
        store recovered_store(make_config(after_temp.path()));
        expect(inspect_records(after_temp.path()).size() == 1, "post-rename completed record survives restart");
    }
}

void write_record(store & history, const std::string & body) {
    auto session = history.begin("POST", "/v1/completions", body);
    expect(session && session->append_response(body, false), "retention record append");
    result finished = session->finish(200, outcome::complete, false);
    expect(finished.durable, "retention record durable");
}

void test_retention() {
    temporary_directory count_temp;
    config count_cfg = make_config(count_temp.path());
    count_cfg.max_retained_files = 2;
    {
        store history(count_cfg);
        write_record(history, "oldest");
        write_record(history, "middle");
        write_record(history, "newest");
    }
    auto records = inspect_records(count_temp.path());
    expect(records.size() == 2, "file-count retention, got " + std::to_string(records.size()));
    reader count_reader(count_temp.path().string());
    record first;
    expect_status(count_reader.read(records.front().file_name, generous_limits(), first), status::ok, "read retained first");
    expect(first.request_body == "middle", "oldest file evicted first");

    temporary_directory byte_temp;
    config byte_cfg = make_config(byte_temp.path());
    byte_cfg.max_retained_bytes = 1200;
    {
        store history(byte_cfg);
        write_record(history, std::string(200, 'a'));
        write_record(history, std::string(200, 'b'));
        write_record(history, std::string(200, 'c'));
    }
    records = inspect_records(byte_temp.path());
    expect(records.size() < 3 && !records.empty(), "byte retention evicts oldest");
    reader byte_reader(byte_temp.path().string());
    record newest;
    expect_status(byte_reader.read(records.back().file_name, generous_limits(), newest), status::ok, "read byte newest");
    expect(newest.request_body == std::string(200, 'c'), "newest survives byte retention");

    temporary_directory age_temp;
    std::atomic<uint64_t> wall{ 1000 };
    config age_cfg = make_config(age_temp.path());
    age_cfg.retention_age_ns = 50;
    age_cfg.wall_time_ns = [&wall] { return wall.load(); };
    {
        store history(age_cfg);
        write_record(history, "aged");
        wall.store(1100);
        write_record(history, "fresh");
    }
    records = inspect_records(age_temp.path());
    expect(records.size() == 1, "wall-clock age retention");
    reader age_reader(age_temp.path().string());
    record fresh;
    expect_status(age_reader.read(records[0].file_name, generous_limits(), fresh), status::ok, "read fresh");
    expect(fresh.request_body == "fresh", "age retention oldest first");
}

void test_security_and_locking() {
    temporary_directory temp;
    const fs::path broad_root = temp.path() / "broad";
    expect(::mkdir(broad_root.c_str(), 0700) == 0 && ::chmod(broad_root.c_str(), 0755) == 0,
           "broad existing root fixture");
    bool broad_rejected = false;
    try {
        store invalid(make_config(broad_root));
    } catch (const std::runtime_error &) {
        broad_rejected = true;
    }
    struct stat broad_info = {};
    expect(broad_rejected && ::stat(broad_root.c_str(), &broad_info) == 0 &&
           (broad_info.st_mode & 0777) == 0755,
           "existing broad root rejected without chmod repair race");

    const fs::path broad_lock_root = temp.path() / "broad-lock";
    expect(::mkdir(broad_lock_root.c_str(), 0700) == 0, "broad lock root");
    const fs::path broad_lock = broad_lock_root / ".request-history.owner.lock";
    int broad_lock_fd = ::open(broad_lock.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    expect(broad_lock_fd >= 0 && ::close(broad_lock_fd) == 0 && ::chmod(broad_lock.c_str(), 0644) == 0,
           "broad existing lock fixture");
    {
        // The lock authority is the held root-directory inode. A stale or
        // adversarial named lock file is unrelated and cannot split owners.
        store history(make_config(broad_lock_root));
        bool owner_busy = false;
        try {
            store second(make_config(broad_lock_root));
        } catch (const std::runtime_error & error) {
            owner_busy = std::string(error.what()).find("owner_busy") != std::string::npos;
        }
        expect(owner_busy, "directory-inode owner lock cannot be replaced");
    }
    struct stat broad_lock_info = {};
    expect(::stat(broad_lock.c_str(), &broad_lock_info) == 0 &&
           (broad_lock_info.st_mode & 0777) == 0644,
           "unrelated broad named lock is not repaired or trusted");

    {
        store history(make_config(temp.path() / "history"));
        struct stat root_info = {};
        expect(::stat((temp.path() / "history").c_str(), &root_info) == 0 &&
               (root_info.st_mode & (S_IRWXG | S_IRWXO)) == 0, "root mode 0700");
        bool owner_busy = false;
        try {
            store second(make_config(temp.path() / "history"));
        } catch (const std::runtime_error & error) {
            owner_busy = std::string(error.what()).find("owner_busy") != std::string::npos;
        }
        expect(owner_busy, "cooperative owner lock");
        write_record(history, "private");
    }
    auto records = inspect_records(temp.path() / "history");
    expect(records.size() == 1, "private record exists");
    struct stat file_info = {};
    expect(::lstat((temp.path() / "history" / records[0].file_name).c_str(), &file_info) == 0 &&
           S_ISREG(file_info.st_mode) && (file_info.st_mode & (S_IRWXG | S_IRWXO)) == 0,
           "record mode 0600 regular");

    bool traversal = false;
    try {
        store invalid(make_config(temp.path() / ".." / "escape"));
    } catch (const std::invalid_argument &) {
        traversal = true;
    }
    expect(traversal, "reject traversal root");

    const fs::path real = temp.path() / "real";
    expect(::mkdir(real.c_str(), 0700) == 0, "mkdir real");
    const fs::path link = temp.path() / "link";
    expect(::symlink(real.c_str(), link.c_str()) == 0, "root symlink fixture");
    bool symlink_rejected = false;
    try {
        store invalid(make_config(link));
    } catch (const std::runtime_error &) {
        symlink_rejected = true;
    }
    expect(symlink_rejected, "reject root symlink");

    const fs::path nested_link = temp.path() / "nested-link";
    expect(::symlink(real.c_str(), nested_link.c_str()) == 0, "intermediate symlink fixture");
    symlink_rejected = false;
    try {
        store invalid(make_config(nested_link / "child"));
    } catch (const std::runtime_error &) {
        symlink_rejected = true;
    }
    expect(symlink_rejected, "reject intermediate symlink");

    bool reader_traversal = false;
    try {
        reader inspection((temp.path() / "history").string());
        record output;
        reader_traversal = inspection.read("../secret", generous_limits(), output).value == status::path_security;
    } catch (...) {
    }
    expect(reader_traversal, "reader rejects traversal name");

    expect(::chmod((temp.path() / "history" / records[0].file_name).c_str(), 0644) == 0,
           "insecure file mode fixture");
    try {
        reader inspection((temp.path() / "history").string());
        std::vector<record_metadata> insecure;
        expect_status(inspection.inspect(generous_limits(), insecure), status::path_security,
                      "reader rejects non-private record mode");
    } catch (const std::exception & error) {
        fail("reader mode fixture unexpectedly failed at root: " + std::string(error.what()));
    }
    expect(::chmod((temp.path() / "history" / records[0].file_name).c_str(), 0600) == 0,
           "restore private record mode");

    const fs::path symlink_root = temp.path() / "managed-symlink";
    {
        store empty(make_config(symlink_root));
    }
    const fs::path outside = temp.path() / "outside";
    const int outside_fd = ::open(outside.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    expect(outside_fd >= 0 && ::close(outside_fd) == 0, "managed symlink target");
    const fs::path managed_link = symlink_root /
        "request-00000000000000000001-00000000000000000001.llrh";
    expect(::symlink(outside.c_str(), managed_link.c_str()) == 0, "managed record symlink fixture");
    symlink_rejected = false;
    try {
        store invalid(make_config(symlink_root));
    } catch (const std::runtime_error &) {
        symlink_rejected = true;
    }
    expect(symlink_rejected, "store rejects managed record symlink");
}

void mutate_last_byte(const fs::path & path) {
    const int descriptor = ::open(path.c_str(), O_RDWR | O_NOFOLLOW);
    expect(descriptor >= 0, "open corruption fixture");
    struct stat info = {};
    expect(::fstat(descriptor, &info) == 0 && info.st_size > 0, "stat corruption fixture");
    uint8_t byte = 0;
    expect(::pread(descriptor, &byte, 1, info.st_size - 1) == 1, "read corruption byte");
    byte ^= 0x80U;
    expect(::pwrite(descriptor, &byte, 1, info.st_size - 1) == 1 && ::fsync(descriptor) == 0,
           "write corruption byte");
    expect(::close(descriptor) == 0, "close corruption fixture");
}

void test_corruption_bounds_and_faults() {
    temporary_directory corrupt_temp;
    {
        store history(make_config(corrupt_temp.path()));
        write_record(history, "healthy-before");
        write_record(history, "corrupt-middle");
        write_record(history, "healthy-after");
    }
    auto records = inspect_records(corrupt_temp.path());
    expect(records.size() == 3, "published corruption fixture count");
    mutate_last_byte(corrupt_temp.path() / records[1].file_name);
    {
        store reconciled(make_config(corrupt_temp.path()));
        expect(reconciled.get_stats().quarantined_records == 1 &&
               reconciled.get_stats().retained_files == 3,
               "published corruption quarantined and included in retention accounting");
    }
    records = inspect_records(corrupt_temp.path());
    expect(records.size() == 2, "corrupt published record does not poison healthy history");
    reader corrupt_reader(corrupt_temp.path().string());
    record healthy_before;
    record healthy_after;
    expect_status(corrupt_reader.read(records[0].file_name, generous_limits(), healthy_before), status::ok,
                  "read healthy record before corruption");
    expect_status(corrupt_reader.read(records[1].file_name, generous_limits(), healthy_after), status::ok,
                  "read healthy record after corruption");
    expect(healthy_before.request_body == "healthy-before" && healthy_after.request_body == "healthy-after",
           "healthy records survive published corruption");

    temporary_directory quarantine_temp;
    config quarantine_cfg = make_config(quarantine_temp.path());
    quarantine_cfg.test_faults.preserve_open_on_shutdown = true;
    {
        store history(quarantine_cfg);
        auto session = history.begin("POST", "/v1/responses", "request");
        expect(session && session->append_response("partial", true), "quarantine open fixture");
        expect_status(history.flush(), status::ok, "quarantine fixture flush");
        expect_status(history.shutdown(), status::ok, "quarantine fixture shutdown");
        session.reset();
    }
    fs::path open_file;
    for (const fs::directory_entry & entry : fs::directory_iterator(quarantine_temp.path())) {
        if (entry.path().filename().string().rfind(".open-", 0) == 0) open_file = entry.path();
    }
    expect(!open_file.empty(), "preserved open fixture exists");
    mutate_last_byte(open_file);
    {
        store recovered(make_config(quarantine_temp.path()));
    }
    expect(inspect_records(quarantine_temp.path()).empty(), "corrupt open record not exposed");
    bool quarantined = false;
    for (const fs::directory_entry & entry : fs::directory_iterator(quarantine_temp.path())) {
        quarantined |= entry.path().filename().string().rfind("quarantine-", 0) == 0;
    }
    expect(quarantined, "corrupt open record quarantined");

    temporary_directory bounded_temp;
    {
        store history(make_config(bounded_temp.path()));
        write_record(history, std::string(100, 'x'));
    }
    reader bounded_reader(bounded_temp.path().string());
    records.clear();
    read_limits tiny = generous_limits();
    tiny.max_file_bytes = 100;
    expect_status(bounded_reader.inspect(tiny, records), status::too_large, "bounded file inspection");
    records = inspect_records(bounded_temp.path());
    tiny = generous_limits();
    tiny.max_total_body_bytes = 10;
    record output;
    expect_status(bounded_reader.read(records[0].file_name, tiny, output), status::too_large, "bounded body read");

    temporary_directory short_temp;
    config short_cfg = make_config(short_temp.path());
    short_cfg.test_faults.max_write_size = 1;
    short_cfg.test_faults.inject_eintr_once = true;
    {
        store history(short_cfg);
        write_record(history, "short-write-retry");
    }
    expect(read_only_record(short_temp.path()).request_body == "short-write-retry", "short write and EINTR exact");

    struct fault_case { const char * name; std::function<void(config &)> apply; status expected; };
    const std::vector<fault_case> cases = {
        { "enospc", [](config & cfg) { cfg.test_faults.fail_after_bytes = 8; cfg.test_faults.fail_no_space = true; }, status::no_space },
        { "file-fsync", [](config & cfg) { cfg.test_faults.fail_file_fsync = true; }, status::io_error },
        { "rename", [](config & cfg) { cfg.test_faults.fail_rename = true; }, status::io_error },
        { "directory-fsync", [](config & cfg) { cfg.test_faults.fail_directory_fsync = true; }, status::commit_uncertain },
    };
    for (const fault_case & current : cases) {
        temporary_directory fault_temp;
        config cfg = make_config(fault_temp.path());
        current.apply(cfg);
        store history(cfg);
        auto session = history.begin("POST", "/v1/completions", "");
        expect(session != nullptr, std::string(current.name) + " begin");
        result finished = session->finish(200, outcome::complete, false);
        expect_status(finished, current.expected, std::string(current.name) + " terminal fault");
        expect(!finished.durable && history.get_stats().terminal_failures == 1,
               std::string(current.name) + " explicit non-durable terminal stats");
    }

    temporary_directory timeout_temp;
    config timeout_cfg = make_config(timeout_temp.path());
    timeout_cfg.terminal_ack_timeout_ms = 20;
    timeout_cfg.test_faults.before_frame_write = [] {
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    };
    {
        store history(timeout_cfg);
        auto session = history.begin("POST", "/v1/completions", "timeout");
        expect(session != nullptr, "timeout begin");
        result finished = session->finish(200, outcome::complete, false);
        expect_status(finished, status::timeout, "bounded terminal acknowledgment");
        expect(!finished.durable && history.get_stats().terminal_timeouts == 1 &&
               history.get_stats().terminal_failures == 1,
               "timeout never claims durable and is visible");
    }

    temporary_directory stopped_temp;
    store stopped(make_config(stopped_temp.path()));
    auto stopped_session = stopped.begin("POST", "/v1/completions", "stopped");
    expect(stopped_session != nullptr, "stopped append session");
    expect_status(stopped.flush(), status::ok, "stopped append prefix flush");
    expect_status(stopped.shutdown(), status::ok, "stopped append shutdown");
    expect(!stopped_session->append_response("late", false) &&
           stopped_session->failure_status().value == status::stopped,
           "append failure exposes stopped status");
    expect_status(stopped_session->finish(200, outcome::complete, false), status::stopped,
                  "terminal after shutdown fails explicitly");
    expect(stopped.get_stats().append_failures == 1 && stopped.get_stats().terminal_failures == 1,
           "append and terminal failures counted");
}

void test_stream_reservation_saturation() {
    temporary_directory temp;
    std::mutex gate_mutex;
    std::condition_variable gate_condition;
    bool pause = false;
    bool blocked = false;
    bool release = false;
    config cfg = make_config(temp.path());
    cfg.max_frame_payload = 256;
    cfg.max_pending_bytes = 256;
    cfg.max_stream_pending_frames_per_session = 1;
    cfg.test_faults.before_frame_write = [&] {
        std::unique_lock<std::mutex> lock(gate_mutex);
        if (!pause) return;
        blocked = true;
        gate_condition.notify_all();
        gate_condition.wait(lock, [&] { return release; });
    };
    store history(cfg);
    auto first = history.begin("POST", "/v1/responses", "");
    auto second = history.begin("POST", "/v1/responses", "");
    auto bulk = history.begin("POST", "/v1/completions", "");
    expect(first && second && bulk, "saturation sessions");
    expect_status(history.flush(), status::ok, "flush session prefixes");
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        pause = true;
    }
    const std::string ones(256, '1');
    const std::string twos(256, '2');
    const std::string letters(256, 'A');
    expect(first->append_response(ones, true), "first saturated chunk");
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        expect(gate_condition.wait_for(lock, std::chrono::seconds(2), [&] { return blocked; }), "writer reached gate");
    }
    std::atomic<bool> blocked_append_done{ false };
    std::thread blocked_append([&] {
        expect(first->append_response(twos, true), "same-session chunk after backpressure");
        blocked_append_done.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    expect(!blocked_append_done.load(), "same-session reservation includes writer-in-flight frame");
    const auto before = std::chrono::steady_clock::now();
    expect(second->append_response(letters, true), "other session has independent reservation");
    expect(bulk->append_response(std::string(256, 'B'), false), "bulk has independent global capacity");
    const auto other_latency = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - before).count();
    expect(other_latency < 100, "unrelated stream admission latency under saturation");
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        release = true;
        pause = false;
    }
    gate_condition.notify_all();
    blocked_append.join();
    expect(blocked_append_done.load(), "backpressured stream resumed");
    expect_status(first->finish(200, outcome::complete, true), status::ok, "first saturation finish");
    expect_status(second->finish(200, outcome::complete, true), status::ok, "second saturation finish");
    expect_status(bulk->finish(200, outcome::complete, false), status::ok, "bulk saturation finish");
    const stats observed = history.get_stats();
    expect(observed.backpressure_waits != 0, "saturation counted backpressure");
    const uint64_t documented_bound = cfg.max_pending_bytes +
        2ULL * cfg.max_stream_pending_frames_per_session * cfg.max_frame_payload;
    expect(observed.peak_pending_bytes <= documented_bound,
           "aggregate payload memory bound is global bulk plus per-active-stream reservation");
    std::cout << "request-history saturation unrelated-admission-ms=" << other_latency << '\n';
}

void test_namespace_races_and_incremental_retention() {
    temporary_directory publish_temp;
    config publish_cfg = make_config(publish_temp.path());
    bool publish_exchanged = false;
    publish_cfg.test_faults.before_publish_rename = [&](const std::string & name) {
        if (publish_exchanged) return;
        publish_exchanged = true;
        exchange_same_shape_file(publish_temp.path(), name, "publish-race-original");
    };
    {
        store history(publish_cfg);
        auto session = history.begin("POST", "/v1/completions", "publish-race");
        expect(session && session->append_response("output", false), "publish race fixture");
        const result finished = session->finish(200, outcome::complete, false);
        expect_status(finished, status::path_security, "publish inode exchange detected");
        expect(!finished.durable && history.get_stats().terminal_failures == 1,
               "publish inode race never reports durable");
    }
    bool published_replacement = false;
    for (const fs::directory_entry & entry : fs::directory_iterator(publish_temp.path())) {
        published_replacement |= entry.path().filename().string().rfind("request-", 0) == 0;
    }
    expect(!published_replacement, "replacement inode is not published");

    temporary_directory retention_temp;
    config retention_cfg = make_config(retention_temp.path());
    retention_cfg.max_retained_files = 1;
    bool retention_exchanged = false;
    std::string replaced_name;
    retention_cfg.test_faults.before_retention_unlink = [&](const std::string & name) {
        if (retention_exchanged) return;
        retention_exchanged = true;
        replaced_name = name;
        exchange_same_shape_file(retention_temp.path(), name, "retention-race-original");
    };
    {
        store history(retention_cfg);
        write_record(history, "oldest");
        auto second = history.begin("POST", "/v1/completions", "second");
        expect(second && second->append_response("second-output", false), "retention race second record");
        result second_result = second->finish(200, outcome::complete, false);
        expect_status(second_result, status::path_security, "retention inode exchange detected");
        expect(second_result.durable, "new record durable before retention maintenance failure");
        struct stat replacement = {};
        expect(!replaced_name.empty() && ::lstat((retention_temp.path() / replaced_name).c_str(), &replacement) == 0,
               "retention does not delete exchanged inode");

        // The next publish performs the one allowed dirty-state rescan,
        // quarantines the corrupt replacement, then returns to incremental
        // index maintenance.
        write_record(history, "third");
        const stats observed = history.get_stats();
        expect(observed.namespace_scans == 2 && observed.quarantined_records == 1 &&
               observed.retained_files == 1,
               "uncertain retention triggers one reconciliation scan");
    }

    temporary_directory scale_temp;
    config scale_cfg = make_config(scale_temp.path());
    scale_cfg.max_retained_files = 256;
    {
        store history(scale_cfg);
        expect(history.get_stats().namespace_scans == 1, "single startup namespace scan");
        for (int i = 0; i < 128; ++i) {
            write_record(history, "scale-" + std::to_string(i));
        }
        const stats observed = history.get_stats();
        expect(observed.namespace_scans == 1 && observed.retained_files == 128,
               "128 publications update retained index without namespace rescans");
    }
}

void test_concurrent_shutdown() {
    temporary_directory temp;
    store history(make_config(temp.path()));
    write_record(history, "shutdown");
    constexpr size_t CALLERS = 12;
    std::vector<result> results(CALLERS);
    std::vector<std::thread> callers;
    for (size_t i = 0; i < CALLERS; ++i) {
        callers.emplace_back([&history, &results, i] { results[i] = history.shutdown(); });
    }
    for (std::thread & caller : callers) caller.join();
    for (const result & current : results) expect_status(current, status::ok, "concurrent shutdown result");
    expect(inspect_records(temp.path()).size() == 1, "concurrent shutdown preserves durable record");
}

void test_claim_recovery_fifo_and_namespace_bound() {
    temporary_directory temp;
    const fs::path root = temp.path() / "claims";
    {
        store history(make_config(root));
        write_record(history, "claim-recovery");
    }
    auto before = inspect_records(root);
    expect(before.size() == 1, "claim recovery fixture published");
    const std::string published = before[0].file_name;
    const std::string claim = ".claim-" + published.substr(std::strlen("request-"),
        published.size() - std::strlen("request-") - std::strlen(".llrh")) + ".llrh.tmp";
    expect(::rename((root / published).c_str(), (root / claim).c_str()) == 0,
           "simulate crash after atomic claim");
    {
        store recovered(make_config(root));
    }
    auto after = inspect_records(root);
    expect(after.size() == 1 && after[0].file_name == published,
           "startup reconciles claimed terminal record");

    const fs::path fifo_root = temp.path() / "fifo";
    expect(::mkdir(fifo_root.c_str(), 0700) == 0, "fifo root");
    const fs::path fifo = fifo_root / "request-00000000000000000001-00000000000000000001.llrh";
    expect(::mkfifo(fifo.c_str(), 0600) == 0, "recognized FIFO fixture");
    reader fifo_reader(fifo_root.string());
    std::vector<record_metadata> fifo_records;
    const auto fifo_started = std::chrono::steady_clock::now();
    expect_status(fifo_reader.inspect(generous_limits(), fifo_records), status::path_security,
                  "recognized FIFO rejected");
    expect(std::chrono::steady_clock::now() - fifo_started < std::chrono::seconds(1),
           "FIFO rejection cannot block on open");

    const fs::path crowded_root = temp.path() / "crowded";
    expect(::mkdir(crowded_root.c_str(), 0700) == 0, "crowded root");
    for (int i = 0; i < 8; ++i) {
        const fs::path item = crowded_root / ("unrelated-" + std::to_string(i));
        const int fd = ::open(item.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        expect(fd >= 0 && ::close(fd) == 0, "crowded namespace fixture");
    }
    reader crowded(crowded_root.string());
    read_limits bounded = generous_limits();
    bounded.max_files = 1;
    bounded.max_namespace_entries = 5;
    std::vector<record_metadata> crowded_records;
    expect_status(crowded.inspect(bounded, crowded_records), status::too_large,
                  "namespace enumeration is bounded including unrelated entries");
}

}  // namespace

int main() {
    try {
        test_exact_nonstream_and_stream_bytes();
        test_outcomes_and_route_policy();
        test_concurrency();
        test_restart_recovery();
        test_retention();
        test_security_and_locking();
        test_corruption_bounds_and_faults();
        test_stream_reservation_saturation();
        test_namespace_races_and_incremental_retention();
        test_concurrent_shutdown();
        test_claim_recovery_fifo_and_namespace_bound();
    } catch (const std::exception & error) {
        std::cerr << "test-server-request-history: " << error.what() << '\n';
        return 1;
    }
    std::cout << "test-server-request-history: all checks passed\n";
    return 0;
}

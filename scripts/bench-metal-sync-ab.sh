#!/bin/bash

# Controlled Apple M2 Ultra A/B for GGML_METAL_FAST_SYNC. The blocking opt-out
# and explicit polling arms use the same binary. Two reversed ABBA cycles per
# workload balance startup/order drift, while per-process CPU time and a
# four-request stress batch expose the polling cost that single-lane throughput
# can hide.

set -euo pipefail

export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin

if [ -z "${M2_ULTRA_QUEUE_TOKEN:-}" ]; then
    echo "bench-metal-sync-ab.sh must run inside m2-ultra-queue.sh" >&2
    exit 2
fi

lane_root=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/campaign-0809-fast-sync
model=/Users/agusx1211/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf
draft=/Users/agusx1211/unsloth/gguf-m2/dspark-0731-expertsonly.m2.gguf
bench_port=8091
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root=/Users/agusx1211/fsync-results/sync-ab-$stamp
window_lock=/Users/agusx1211/m2-window.lock
sequential_repetitions=${FSYNC_SEQUENTIAL_REPETITIONS:-3}
sequential_tokens=${FSYNC_SEQUENTIAL_TOKENS:-320}
stress_tokens=${FSYNC_STRESS_TOKENS:-128}
profile_tokens=${FSYNC_PROFILE_TOKENS:-192}
target_order="wait poll poll wait poll wait wait poll"
spec_order="poll wait wait poll wait poll poll wait"

mkdir -p "$result_root"

log() {
    echo "[fsync-ab $(date -u +%H:%M:%S)] $*" | tee -a "$result_root/harness.log"
}

write_metadata() {
    {
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "lane=campaign-0809-fast-sync"
        echo "commit=$(git -C "$lane_root" rev-parse HEAD)"
        echo "branch=$(git -C "$lane_root" symbolic-ref --short HEAD)"
        echo "worktree_status_begin"
        git -C "$lane_root" status --short
        echo "worktree_status_end"
        echo "queue_token_present=1"
        echo "target_order=$target_order"
        echo "spec_order=$spec_order"
        echo "sequential_repetitions=$sequential_repetitions"
        echo "sequential_tokens=$sequential_tokens"
        echo "stress_concurrency=4"
        echo "stress_tokens_per_request=$stress_tokens"
        echo "profile_tokens=$profile_tokens"
        echo "wait_env=GGML_METAL_FAST_SYNC=0"
        echo "poll_env=GGML_METAL_FAST_SYNC=1"
        echo "m2_ultra_default=poll"
        echo "shared_env=LLAMA_DSV4_ADMISSION_VERTICAL=1 GGML_E4M3_MM_NT2_MIN_N=512"
        echo "target_theoretical_ceiling_pct=0.5"
        echo "spec_theoretical_ceiling_pct=0.7"
        echo "server_args=-c 1048576 -np 4 --kv-unified --no-context-shift --cache-ram 0 -ctk f16 -ctv f16 -ngl 999 -fa on -fit on -ub 2048 -b 2048"
        echo "model=$model"
        stat -f 'model_size=%z model_mtime=%m' "$model"
        echo "draft=$draft"
        stat -f 'draft_size=%z draft_mtime=%m' "$draft"
        echo "binary=$lane_root/build-m2/bin/llama-server"
        shasum -a 256 "$lane_root/build-m2/bin/llama-server"
        sw_vers
        uname -a
        sysctl -n machdep.cpu.brand_string 2>/dev/null || true
        /usr/bin/clang --version | head -n 1
        xcrun --show-sdk-path
    } > "$result_root/metadata.txt"
}

if ! (set -C; printf '%s\n' "campaign-0809-fast-sync" > "$window_lock") 2>/dev/null; then
    echo "production window lock held by: $(head -n 1 "$window_lock" 2>/dev/null || true)" >&2
    exit 1
fi

bench_pid=

stop_bench() {
    if [ -n "$bench_pid" ]; then
        kill "$bench_pid" 2>/dev/null || true
        wait "$bench_pid" 2>/dev/null || true
        bench_pid=
    fi
    pkill -f "campaign-0809-fast-sync/build-m2/bin/llama-server.*--port $bench_port" 2>/dev/null || true
}

restore_prod() {
    local rc=$?
    trap - EXIT INT TERM
    set +e
    stop_bench
    log "restoring production elastic4 service (harness rc=$rc)"

    # The production tmux session must not inherit this queue job's ownership
    # token; it is expected to survive after the queue slot is released.
    env -u M2_ULTRA_QUEUE_TOKEN \
        /Users/agusx1211/start-llama-server.sh elastic4 >> "$result_root/restore.log" 2>&1

    local healthy=0
    local i
    for i in $(seq 1 180); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                -H 'Authorization: Bearer llamacpp' \
                http://127.0.0.1:8080/health || true)" = "200" ]; then
            healthy=1
            break
        fi
        sleep 5
    done

    if [ "$healthy" = "1" ]; then
        log "production /health returned 200; issuing one-token completion"
        curl -sS --fail --max-time 120 \
            -H 'Authorization: Bearer llamacpp' \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"ping","n_predict":1,"temperature":0}' \
            http://127.0.0.1:8080/completion > "$result_root/prod-health-completion.json"
        python3 - "$result_root/prod-health-completion.json" <<'PY' >> "$result_root/restore.log"
import json
import sys

value = json.load(open(sys.argv[1]))
print("completion_content_len", len(value.get("content", "")))
print("completion_tokens", value.get("timings", {}).get("predicted_n"))
PY
    else
        log "ERROR: production /health did not return 200 within 15 minutes"
        rc=1
    fi

    rm -f "$window_lock"
    log "production window released; results=$result_root"
    exit "$rc"
}

trap restore_prod EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_health() {
    local i
    for i in $(seq 1 240); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                "http://127.0.0.1:$bench_port/health" || true)" = "200" ]; then
            return 0
        fi
        if ! kill -0 "$bench_pid" 2>/dev/null; then
            log "benchmark server exited during startup"
            return 1
        fi
        sleep 3
    done
    return 1
}

cpu_time_seconds() {
    local raw
    raw=$(ps -p "$bench_pid" -o time= | tr -d ' ')
    python3 - "$raw" <<'PY'
import sys

raw = sys.argv[1]
days = 0
if "-" in raw:
    day_text, raw = raw.split("-", 1)
    days = int(day_text)
parts = [float(part) for part in raw.split(":")]
seconds = 0.0
for part in parts:
    seconds = seconds * 60.0 + part
seconds += days * 86400.0
print("%.6f" % seconds)
PY
}

monotonic_ns() {
    python3 -c 'import time; print(time.monotonic_ns())'
}

write_payload() {
    local n_predict=$1
    local path=$2
    python3 - "$n_predict" "$path" <<'PY'
import json
import sys

payload = {
    "prompt": "The history of computing begins with mechanical calculation. Write a detailed essay on how computing evolved from mechanical devices to modern accelerators.",
    "n_predict": int(sys.argv[1]),
    "temperature": 0,
    "seed": 42,
    "cache_prompt": False,
    "ignore_eos": True,
}
with open(sys.argv[2], "w") as handle:
    json.dump(payload, handle)
PY
}

record_response() {
    local response=$1
    local run_id=$2
    local mode=$3
    local arm=$4
    local profile=$5
    local kind=$6
    local ordinal=$7
    local batch_id=$8
    local concurrency=$9
    shift 9
    local requested_n=$1
    local wall_start_ns=$2
    local wall_end_ns=$3
    local cpu_start_s=$4
    local cpu_end_s=$5

    python3 - \
            "$response" "$result_root/samples.jsonl" "$run_id" "$mode" "$arm" \
            "$profile" "$kind" "$ordinal" "$batch_id" "$concurrency" \
            "$requested_n" "$wall_start_ns" "$wall_end_ns" "$cpu_start_s" "$cpu_end_s" <<'PY' \
        | tee -a "$result_root/results.txt"
import hashlib
import json
import sys
from datetime import datetime, timezone

(response_path, output_path, run_id, mode, arm, profile, kind, ordinal,
 batch_id, concurrency, requested_n, wall_start_ns, wall_end_ns,
 cpu_start_s, cpu_end_s) = sys.argv[1:]
value = json.load(open(response_path))
timing = value.get("timings", {})
content = value.get("content", "")
wall_s = (int(wall_end_ns) - int(wall_start_ns)) / 1e9
cpu_s = float(cpu_end_s) - float(cpu_start_s)
record = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "mode": mode,
    "arm": arm,
    "profile": int(profile),
    "kind": kind,
    "ordinal": ordinal,
    "batch_id": batch_id,
    "concurrency": int(concurrency),
    "requested_n": int(requested_n),
    "predicted_n": timing.get("predicted_n"),
    "tps": timing.get("predicted_per_second"),
    "ms_per_token": timing.get("predicted_per_token_ms"),
    "prompt_n": timing.get("prompt_n"),
    "prompt_tps": timing.get("prompt_per_second"),
    "draft_n_accepted": timing.get("draft_n_accepted"),
    "draft_n": timing.get("draft_n"),
    "sha256": hashlib.sha256(content.encode()).hexdigest(),
    "content_len": len(content),
    "batch_wall_s": wall_s,
    "batch_cpu_s": cpu_s,
    "batch_cpu_cores": cpu_s / wall_s,
}
with open(output_path, "a") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
print(
    "RESULT", run_id, kind + "/" + ordinal,
    "tps=%.6f" % (record["tps"] or float("nan")),
    "n=%s" % record["predicted_n"],
    "cpu_s=%.3f" % cpu_s,
    "wall_s=%.3f" % wall_s,
    "cpu_cores=%.3f" % record["batch_cpu_cores"],
    "draft=%s/%s" % (record["draft_n_accepted"], record["draft_n"]),
    "sha256=%s" % record["sha256"],
)
PY
}

single_request() {
    local run_id=$1
    local mode=$2
    local arm=$3
    local profile=$4
    local kind=$5
    local ordinal=$6
    local n_predict=$7
    local response="$result_root/$run_id-$kind-$ordinal.json"
    local payload="$result_root/payload-$n_predict.json"
    local cpu_start_s cpu_end_s wall_start_ns wall_end_ns

    write_payload "$n_predict" "$payload"
    cpu_start_s=$(cpu_time_seconds)
    wall_start_ns=$(monotonic_ns)
    curl -sS --fail --max-time 1800 \
        -H 'Content-Type: application/json' \
        -d @"$payload" \
        "http://127.0.0.1:$bench_port/completion" > "$response"
    wall_end_ns=$(monotonic_ns)
    cpu_end_s=$(cpu_time_seconds)

    record_response "$response" "$run_id" "$mode" "$arm" "$profile" \
        "$kind" "$ordinal" "$run_id-$kind-$ordinal" 1 "$n_predict" \
        "$wall_start_ns" "$wall_end_ns" "$cpu_start_s" "$cpu_end_s"
}

stress_request() {
    local run_id=$1
    local mode=$2
    local arm=$3
    local n_predict=$4
    local payload="$result_root/payload-$n_predict.json"
    local batch_id="$run_id-stress"
    local cpu_start_s cpu_end_s wall_start_ns wall_end_ns
    local curl_pids=
    local i pid

    write_payload "$n_predict" "$payload"
    cpu_start_s=$(cpu_time_seconds)
    wall_start_ns=$(monotonic_ns)
    for i in 1 2 3 4; do
        curl -sS --fail --max-time 1800 \
            -H 'Content-Type: application/json' \
            -d @"$payload" \
            "http://127.0.0.1:$bench_port/completion" \
            > "$result_root/$run_id-stress-$i.json" &
        curl_pids="$curl_pids $!"
    done
    for pid in $curl_pids; do
        wait "$pid"
    done
    wall_end_ns=$(monotonic_ns)
    cpu_end_s=$(cpu_time_seconds)

    for i in 1 2 3 4; do
        record_response "$result_root/$run_id-stress-$i.json" \
            "$run_id" "$mode" "$arm" 0 stress "$i" "$batch_id" 4 \
            "$n_predict" "$wall_start_ns" "$wall_end_ns" "$cpu_start_s" "$cpu_end_s"
    done
}

start_server() {
    local run_id=$1
    local mode=$2
    local arm=$3
    local profile=$4
    local fast_sync=0
    local server_log="$result_root/$run_id-server.log"
    local -a server_args=(
        -m "$model"
        -c 1048576 -np 4 --kv-unified --no-context-shift --cache-ram 0
        -ctk f16 -ctv f16 -ngl 999 -fa on -fit on -ub 2048 -b 2048
        --host 127.0.0.1 --port "$bench_port" --no-webui
    )

    if [ "$arm" = poll ]; then
        fast_sync=1
    fi
    if [ "$mode" = spec ]; then
        server_args+=(
            -md "$draft"
            --spec-type draft-dspark
            --spec-draft-n-max 5
            --spec-draft-n-min 1
            --spec-draft-p-min 0.5
            -ngld 999
        )
    fi

    log "starting $run_id (mode=$mode arm=$arm profile=$profile fast_sync=$fast_sync)"
    (
        cd "$lane_root"
        exec env \
            LLAMA_HOST_PROFILE="$profile" \
            GGML_METAL_FAST_SYNC="$fast_sync" \
            LLAMA_DSV4_ADMISSION_VERTICAL=1 \
            GGML_E4M3_MM_NT2_MIN_N=512 \
            DYLD_LIBRARY_PATH="$lane_root/build-m2/bin" \
            "$lane_root/build-m2/bin/llama-server" \
                "${server_args[@]}"
    ) > "$server_log" 2>&1 &
    bench_pid=$!

    if ! wait_health; then
        tail -n 60 "$server_log" >&2
        return 1
    fi
    if [ "$arm" = poll ] && ! grep -q 'fast command-buffer status polling enabled' "$server_log"; then
        log "ERROR: polling arm did not report the M2 Ultra fast-sync gate as enabled"
        return 1
    fi
}

finish_server() {
    local run_id=$1
    local server_log="$result_root/$run_id-server.log"
    stop_bench
    if grep -E 'OutOfMemory|failed with status|MTLCommandBufferError|command buffer.*error' \
            "$server_log" > "$result_root/$run_id-errors.txt"; then
        log "ERROR: Metal/server failure text found in $run_id"
        return 1
    fi
    rm -f "$result_root/$run_id-errors.txt"
    sleep 3
}

run_throughput_instance() {
    local run_id=$1
    local mode=$2
    local arm=$3
    local i

    start_server "$run_id" "$mode" "$arm" 0
    single_request "$run_id" "$mode" "$arm" 0 warmup 1 24
    for i in $(seq 1 "$sequential_repetitions"); do
        single_request "$run_id" "$mode" "$arm" 0 sequential "$i" "$sequential_tokens"
    done
    stress_request "$run_id" "$mode" "$arm" "$stress_tokens"
    finish_server "$run_id"
}

run_profile_instance() {
    local run_id=$1
    local mode=$2
    local arm=$3
    local summary="$result_root/$run_id-wake-summary.txt"

    start_server "$run_id" "$mode" "$arm" 1
    single_request "$run_id" "$mode" "$arm" 1 warmup 1 24
    single_request "$run_id" "$mode" "$arm" 1 profile 1 "$profile_tokens"
    finish_server "$run_id"
    python3 "$lane_root/scripts/analyze-metal-sync-wake.py" \
        "$result_root/$run_id-server.log" \
        --skip-per-context 16 \
        --csv "$result_root/$run_id-wake.csv" | tee "$summary"
}

write_metadata
log "stopping production service"
tmux kill-session -t llama-server 2>/dev/null || true
pkill -f 'llama-server.*--port 8080' 2>/dev/null || true
stop_bench
sleep 8

position=0
for arm in $target_order; do
    position=$((position + 1))
    run_throughput_instance "target-$position-$arm" target "$arm"
done

position=0
for arm in $spec_order; do
    position=$((position + 1))
    run_throughput_instance "spec-$position-$arm" spec "$arm"
done

# Profile both primitives on the candidate binary. These runs are separate so
# profiler I/O cannot contaminate the throughput samples above.
run_profile_instance profile-target-wait target wait
run_profile_instance profile-target-poll target poll
run_profile_instance profile-spec-poll spec poll
run_profile_instance profile-spec-wait spec wait

python3 "$lane_root/scripts/analyze-metal-sync-ab.py" \
    "$result_root/samples.jsonl" | tee "$result_root/summary.txt"

log "A/B complete"

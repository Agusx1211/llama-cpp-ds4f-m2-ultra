#!/bin/bash

# Controlled M2 Ultra baseline for Metal command-buffer wake latency. This
# script deliberately owns the production window for its whole lifetime and
# restores the normal elastic4 service through an EXIT trap.

set -euo pipefail

export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin

lane_root=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/campaign-0809-fast-sync
model=/Users/agusx1211/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf
draft=/Users/agusx1211/unsloth/gguf-m2/dspark-0731-expertsonly.m2.gguf
bench_port=8091
stamp=$(date -u +%Y%m%dT%H%M%SZ)
result_root=/Users/agusx1211/fsync-results/wake-baseline-$stamp
window_lock=/Users/agusx1211/m2-window.lock

mkdir -p "$result_root"

log() {
    echo "[fsync-baseline $(date -u +%H:%M:%S)] $*" | tee -a "$result_root/harness.log"
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
        curl -s --max-time 120 \
            -H 'Authorization: Bearer llamacpp' \
            -H 'Content-Type: application/json' \
            -d '{"prompt":"ping","n_predict":1}' \
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

trap restore_prod EXIT INT TERM

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

probe() {
    local name=$1
    local ordinal=$2
    local n_predict=$3
    local response="$result_root/$name-$ordinal.json"

    python3 - "$n_predict" "$result_root/payload.json" <<'PY'
import json
import sys

payload = {
    "prompt": "The history of computing begins with mechanical calculation. Write a detailed essay on how computing evolved from mechanical devices to modern accelerators.",
    "n_predict": int(sys.argv[1]),
    "temperature": 0,
    "cache_prompt": False,
    "ignore_eos": True,
}
with open(sys.argv[2], "w") as handle:
    json.dump(payload, handle)
PY

    curl -sS --fail --max-time 1800 \
        -H 'Content-Type: application/json' \
        -d @"$result_root/payload.json" \
        "http://127.0.0.1:$bench_port/completion" > "$response"

    python3 - "$response" "$name/$ordinal" <<'PY' | tee -a "$result_root/results.txt"
import hashlib
import json
import sys

value = json.load(open(sys.argv[1]))
timing = value.get("timings", {})
content = value.get("content", "")
print(
    "RESULT", sys.argv[2],
    "tps=%.6f" % timing.get("predicted_per_second", float("nan")),
    "ms_per_tok=%.6f" % timing.get("predicted_per_token_ms", float("nan")),
    "n=%s" % timing.get("predicted_n"),
    "pp_tps=%.6f" % timing.get("prompt_per_second", float("nan")),
    "prompt_n=%s" % timing.get("prompt_n"),
    "draft_n=%s/%s" % (timing.get("draft_n_accepted"), timing.get("draft_n")),
    "sha256=%s" % hashlib.sha256(content.encode()).hexdigest(),
    "len=%d" % len(content),
)
PY
}

run_server() {
    local name=$1
    local profile=$2
    local speculative=$3
    local repetitions=$4
    local server_log="$result_root/$name-server.log"
    local -a server_args=(
        -m "$model"
        -c 1048576 -np 4 --kv-unified --no-context-shift --cache-ram 0
        -ctk f16 -ctv f16 -ngl 999 -fa on -fit on -ub 2048 -b 2048
        --host 127.0.0.1 --port "$bench_port" --no-webui
    )

    if [ "$speculative" = "1" ]; then
        server_args+=(
            -md "$draft"
            --spec-type draft-dspark
            --spec-draft-n-max 5
            --spec-draft-n-min 1
            --spec-draft-p-min 0.5
            -ngld 999
        )
    fi

    log "starting $name (LLAMA_HOST_PROFILE=$profile speculative=$speculative)"
    (
        cd "$lane_root"
        env \
            LLAMA_HOST_PROFILE="$profile" \
            LLAMA_DSV4_ADMISSION_VERTICAL=1 \
            GGML_E4M3_MM_NT2_MIN_N=512 \
            DYLD_LIBRARY_PATH="$lane_root/build-m2/bin" \
            "$lane_root/build-m2/bin/llama-server" \
                "${server_args[@]}" \
                > "$server_log" 2>&1
    ) &
    bench_pid=$!

    if ! wait_health; then
        tail -n 40 "$server_log" >&2
        return 1
    fi

    probe "$name" warmup 24
    local i
    for i in $(seq 1 "$repetitions"); do
        probe "$name" "$i" 320
    done

    stop_bench
    sleep 5
}

log "stopping production service"
tmux kill-session -t llama-server 2>/dev/null || true
pkill -f 'llama-server.*--port 8080' 2>/dev/null || true
sleep 8

# Three unprofiled throughput samples, followed by one minimally warmed
# instrumented sample, for both the target-only and deployed speculative modes.
run_server target-default 0 0 3
run_server target-profile 1 0 1
run_server spec-default 0 1 3
run_server spec-profile 1 1 1

log "baseline complete"

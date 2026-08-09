#!/bin/bash

# Controlled real-model measurement of DSpark drafter attention cost versus
# live context depth.  A single unprofiled deep prefill is serialized with both
# target and draft state, then replayed into fresh KPROF processes.  This keeps
# GGML_METAL_KPROF=1 away from the long prefill and makes every profiled request
# an exact token-for-token continuation of the same state.

set -euo pipefail

export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin

if [ -z "${M2_ULTRA_QUEUE_TOKEN:-}" ]; then
    echo "bench-dspark-draft-depth.sh must run inside m2-ultra-queue.sh" >&2
    exit 2
fi

lane=campaign-0809-draft-window
lane_root=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/$lane
model=/Users/agusx1211/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf
draft=/Users/agusx1211/unsloth/gguf-m2/dspark-0731-expertsonly.m2.gguf
binary=$lane_root/build-m2/bin/llama-server
analyzer=$lane_root/scripts/analyze-dspark-draft-depth.py
bench_port=${DWIN_PORT:-8093}
api_key=llamacpp
stamp=$(date -u +%Y%m%dT%H%M%SZ)
tag=${DWIN_TAG:-gate100k-$stamp}
result_root=/Users/agusx1211/dwin-results/$tag
deep_cache_dir=$result_root/cache-deep
near_cache_dir=$result_root/cache-near
window_lock=/Users/agusx1211/m2-window.lock

depth_chars=${DWIN_DEPTH_CHARS:-530000}
stage_tokens=${DWIN_STAGE_TOKENS:-96}
profile_tokens=${DWIN_PROFILE_TOKENS:-16}
cache_ram_mib=${DWIN_CACHE_RAM_MIB:-1536}
filler_hard_cap=96
max_fillers=${DWIN_MAX_FILLERS:-$filler_hard_cap}
allow_deeper=${DWIN_ALLOW_OVER_150K:-0}

if [ "$depth_chars" -gt 800000 ] && [ "$allow_deeper" != "1" ]; then
    echo "refusing a >~150k-token prompt without DWIN_ALLOW_OVER_150K=1" >&2
    exit 2
fi
case $max_fillers in
    ''|*[!0-9]*)
        echo "DWIN_MAX_FILLERS must be an integer from 1 through $filler_hard_cap" >&2
        exit 2
        ;;
esac
if [ "$max_fillers" -lt 1 ] || [ "$max_fillers" -gt "$filler_hard_cap" ]; then
    echo "DWIN_MAX_FILLERS must be from 1 through $filler_hard_cap" >&2
    exit 2
fi

if [ "${DWIN_DRY_RUN:-0}" = "1" ]; then
    printf '%s\n' \
        "lane=$lane" \
        "depth_chars=$depth_chars" \
        "stage_tokens=$stage_tokens" \
        "profile_tokens=$profile_tokens" \
        "cache_ram_mib=$cache_ram_mib" \
        "max_fillers=$max_fillers" \
        "filler_hard_cap=$filler_hard_cap" \
        "profile_order=near1,deep1,deep2,near2" \
        "decision=<5.0% reject; >=5.0% stop for explicit 500k review"
    exit 0
fi

for path in "$model" "$draft" "$binary" "$analyzer"; do
    if [ ! -e "$path" ]; then
        echo "required path missing: $path" >&2
        exit 1
    fi
done

if [ "$(git -C "$lane_root" symbolic-ref --short HEAD)" != "$lane" ]; then
    echo "wrong M2 branch in $lane_root" >&2
    exit 1
fi
if [ -n "$(git -C "$lane_root" status --porcelain)" ]; then
    echo "M2 lane worktree must be clean" >&2
    git -C "$lane_root" status --short >&2
    exit 1
fi
if [ -e "$result_root" ]; then
    echo "result directory already exists: $result_root" >&2
    exit 1
fi

mkdir -p "$result_root" "$deep_cache_dir" "$near_cache_dir"

log() {
    echo "[dwin $(date -u +%H:%M:%S)] $*" | tee -a "$result_root/harness.log"
}

python3 - "$result_root/manifest.json" "$depth_chars" "$stage_tokens" \
        "$profile_tokens" "$cache_ram_mib" "$max_fillers" "$filler_hard_cap" <<'PY'
import json
import sys

(path, depth_chars, stage_tokens, profile_tokens, cache_ram_mib, max_fillers,
 filler_hard_cap) = sys.argv[1:]
value = {
    "schema": 1,
    "objective": "E1 DSpark context-proportional drafter cost gate",
    "decision_threshold_pct": 5.0,
    "decision_rule": "<5% reject; >=5% stop and request review before 500k",
    "depth_chars": int(depth_chars),
    "stage_tokens": int(stage_tokens),
    "profile_tokens": int(profile_tokens),
    "profile_repeats": 2,
    "profile_order": ["near-kprof-1", "deep-kprof-1", "deep-kprof-2", "near-kprof-2"],
    "cache_ram_mib": int(cache_ram_mib),
    "max_fillers": int(max_fillers),
    "filler_hard_cap": int(filler_hard_cap),
    "spec_draft_n_max": 5,
    "kprof_stride": 1,
    "ctx_checkpoints": 0,
    "allowed_cache_tail_gap_tokens": [0, 1],
}
with open(path, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "lane=$lane"
    echo "branch=$(git -C "$lane_root" symbolic-ref --short HEAD)"
    echo "commit=$(git -C "$lane_root" rev-parse HEAD)"
    echo "tree=$(git -C "$lane_root" rev-parse 'HEAD^{tree}')"
    echo "queue_token_present=1"
    echo "model=$model"
    stat -f 'model_size=%z model_mtime=%m' "$model"
    echo "draft=$draft"
    stat -f 'draft_size=%z draft_mtime=%m' "$draft"
    echo "binary=$binary"
    shasum -a 256 "$binary"
    echo "shared_env=LLAMA_HOST_PROFILE=1 LLAMA_SPEC_TRACE=1 LLAMA_DSV4_ADMISSION_VERTICAL=1 GGML_E4M3_MM_NT2_MIN_N=512 GGML_METAL_FAST_SYNC=0"
    echo "server_args=-c 1048576 -np 4 --kv-unified --no-context-shift --cache-ram $cache_ram_mib --cache-disk-limit 16 -ctxcp 0 -ctk f16 -ctv f16 -ngl 999 -ngld 999 -fa on -fit on -ub 2048 -b 2048"
    sw_vers
    uname -a
    sysctl -n machdep.cpu.brand_string 2>/dev/null || true
    /usr/bin/clang --version | head -n 1
    xcrun --show-sdk-path
} > "$result_root/metadata.txt"

lock_owner="$lane pid=$$ tag=$tag"
if ! (set -C; printf '%s\n' "$lock_owner" > "$window_lock") 2>/dev/null; then
    echo "production window lock held by: $(head -n 1 "$window_lock" 2>/dev/null || true)" >&2
    exit 1
fi

server_pid=
sampler_pid=
server_log=
server_label=
server_profile=0
production_taken=0

stop_server() {
    if [ -n "$sampler_pid" ]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=
    fi
    if [ -n "$server_pid" ]; then
        kill "$server_pid" 2>/dev/null || true
        local i
        for i in $(seq 1 30); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            log "server $server_label did not exit after SIGTERM; sending SIGKILL"
            kill -9 "$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
    pkill -f "$lane/build-m2/bin/llama-server.*--port $bench_port" 2>/dev/null || true
}

finish_server() {
    local completed_log=$server_log
    stop_server
    if [ -n "$completed_log" ] && [ -f "$completed_log" ]; then
        gzip -f "$completed_log"
    fi
    server_log=
    server_label=
}

restore_prod() {
    local primary_rc=$?
    local restore_rc=0
    local final_rc
    trap - EXIT INT TERM
    set +e
    stop_server

    if [ "$production_taken" = "1" ]; then
        log "restoring production elastic4 service (harness rc=$primary_rc)"
        # The restored service must outlive this queue job and therefore must
        # not retain queue ownership in its tmux environment.
        if ! env -u M2_ULTRA_QUEUE_TOKEN \
                /Users/agusx1211/start-llama-server.sh elastic4 \
                >> "$result_root/restore.log" 2>&1; then
            restore_rc=1
        fi

        local healthy=0
        local i
        for i in $(seq 1 180); do
            if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                    -H "Authorization: Bearer $api_key" \
                    http://127.0.0.1:8080/health || true)" = "200" ]; then
                healthy=1
                break
            fi
            sleep 5
        done

        if [ "$healthy" = "1" ]; then
            log "production /health returned 200; issuing one-token completion"
            if ! curl -sS --fail --max-time 180 \
                    -H "Authorization: Bearer $api_key" \
                    -H 'Content-Type: application/json' \
                    -d '{"prompt":"ping","n_predict":1,"temperature":0,"ignore_eos":true}' \
                    http://127.0.0.1:8080/completion \
                    > "$result_root/prod-health-completion.json"; then
                restore_rc=1
            elif ! python3 - "$result_root/prod-health-completion.json" <<'PY' \
                    >> "$result_root/restore.log" 2>&1
import json
import sys

value = json.load(open(sys.argv[1]))
predicted = value.get("timings", {}).get("predicted_n")
content = value.get("content")
if predicted != 1 or not isinstance(content, str) or not content:
    raise SystemExit("invalid production completion response")
print("completion_content_len", len(content))
print("completion_tokens", predicted)
PY
            then
                restore_rc=1
            fi
        else
            log "ERROR: production /health did not return 200 within 15 minutes"
            restore_rc=1
        fi
    fi

    if [ "$(cat "$window_lock" 2>/dev/null)" = "$lock_owner" ]; then
        rm -f "$window_lock"
        if [ -e "$window_lock" ]; then
            log "ERROR: failed to remove owned production window lock"
            restore_rc=1
        fi
    else
        log "ERROR: production window lock ownership changed; refusing to remove it"
        restore_rc=1
    fi
    log "production window released; results=$result_root"
    if [ "$primary_rc" -ne 0 ]; then
        final_rc=$primary_rc
    else
        final_rc=$restore_rc
    fi
    exit "$final_rc"
}

trap restore_prod EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_health() {
    local i
    for i in $(seq 1 240); do
        if [ "$(curl -s -o /dev/null -w '%{http_code}' \
                -H "Authorization: Bearer $api_key" \
                "http://127.0.0.1:$bench_port/health" || true)" = "200" ]; then
            return 0
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            log "server $server_label exited during startup"
            return 1
        fi
        sleep 3
    done
    return 1
}

sample_resources() {
    local label=$1
    local pid=$2
    while kill -0 "$pid" 2>/dev/null; do
        local now rss vsz cpu cputime swap
        now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        read -r rss vsz cpu cputime < <(ps -p "$pid" -o rss=,vsz=,%cpu=,time= 2>/dev/null || true)
        swap=$(sysctl -n vm.swapusage 2>/dev/null | tr '\t' ' ' || true)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$now" "$label" "${rss:-}" "${vsz:-}" "${cpu:-}" "${cputime:-}" "$swap" \
            >> "$result_root/resource-samples.tsv"
        sleep 5
    done
}

capture_system() {
    local label=$1
    {
        date -u +%Y-%m-%dT%H:%M:%SZ
        sysctl vm.swapusage
        /usr/bin/memory_pressure -Q 2>&1 || true
        /usr/bin/vm_stat 2>&1 || true
        if [ -n "$server_pid" ]; then
            ps -p "$server_pid" -o pid=,rss=,vsz=,%cpu=,time=,command= || true
        fi
        du -sk "$deep_cache_dir" 2>/dev/null || true
    } > "$result_root/system-$label.txt"
}

start_server() {
    server_label=$1
    server_profile=$2
    local cache_dir=$3
    server_log=$result_root/server-$server_label.log
    local -a server_env=(
        "DYLD_LIBRARY_PATH=$lane_root/build-m2/bin"
        LLAMA_HOST_PROFILE=1
        LLAMA_SPEC_TRACE=1
        LLAMA_DSV4_ADMISSION_VERTICAL=1
        GGML_E4M3_MM_NT2_MIN_N=512
        GGML_METAL_FAST_SYNC=0
    )
    if [ "$server_profile" = "1" ]; then
        server_env+=(GGML_METAL_KPROF=1 GGML_METAL_KPROF_DEBUG=1)
    fi

    log "starting $server_label (kprof=$server_profile cache=$cache_dir)"
    (
        cd "$lane_root"
        exec env "${server_env[@]}" \
            "$binary" \
                -m "$model" -md "$draft" --spec-type draft-dspark \
                --spec-draft-n-max 5 --spec-draft-n-min 1 --spec-draft-p-min 0.5 \
                -c 1048576 -np 4 --kv-unified --no-context-shift \
                --cache-ram "$cache_ram_mib" --cache-disk "$cache_dir" --cache-disk-limit 16 \
                --cache-idle-slots -ctxcp 0 \
                -ctk f16 -ctv f16 -ngl 999 -ngld 999 -fa on -fit on -ub 2048 -b 2048 \
                --host 127.0.0.1 --port "$bench_port" --api-key "$api_key" --no-webui \
                > "$server_log" 2>&1
    ) &
    server_pid=$!
    sample_resources "$server_label" "$server_pid" &
    sampler_pid=$!

    if ! wait_health; then
        tail -n 60 "$server_log" >&2 || true
        return 1
    fi
    log "$server_label /health 200 (pid=$server_pid)"
}

tokenize_prompt() {
    local mode=$1
    local output=$2
    local chars=${3:-0}
    python3 - "$mode" "$output" "$chars" "$lane_root" "$bench_port" "$api_key" <<'PY'
import importlib.util
import json
import sys
import urllib.request

mode, output, chars, root, port, key = sys.argv[1:]
if mode == "deep":
    spec = importlib.util.spec_from_file_location("depth", root + "/scripts/dsv4-depth-determinism.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    content = module.build_prompt(int(chars))
elif mode == "near":
    content = "Explain why deterministic measurements need exact inputs."
else:
    content = mode

request = urllib.request.Request(
    f"http://127.0.0.1:{port}/tokenize",
    data=json.dumps({"content": content, "add_special": True}).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
with urllib.request.urlopen(request, timeout=600) as response:
    value = json.load(response)
tokens = value.get("tokens")
if not isinstance(tokens, list) or not tokens:
    raise SystemExit("tokenizer returned no tokens")
with open(output, "w") as handle:
    json.dump(tokens, handle, separators=(",", ":"))
    handle.write("\n")
if mode == "deep":
    with open(output.replace("-tokens.json", "-prompt.txt"), "w") as handle:
        handle.write(content)
print(len(tokens))
PY
}

run_request() {
    local label=$1
    local prompt_file=$2
    local n_predict=$3
    local slot=$4
    local cache_prompt=$5
    local response=$result_root/request-$label.json
    local meta=$result_root/request-$label.meta.json
    local begin_line end_line wall_start wall_end

    begin_line=$(( $(wc -l < "$server_log") + 1 ))
    wall_start=$(python3 -c 'import time; print(time.monotonic_ns())')
    python3 - "$prompt_file" "$response" "$n_predict" "$slot" "$cache_prompt" \
            "$bench_port" "$api_key" <<'PY'
import json
import sys
import urllib.request

prompt_path, response_path, n_predict, slot, cache_prompt, port, key = sys.argv[1:]
with open(prompt_path) as handle:
    prompt = json.load(handle)
payload = {
    "prompt": prompt,
    "n_predict": int(n_predict),
    "temperature": 0,
    "seed": 42,
    "cache_prompt": cache_prompt == "1",
    "ignore_eos": True,
    "return_tokens": True,
    "id_slot": int(slot),
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/completion",
    data=json.dumps(payload, separators=(",", ":")).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
)
with urllib.request.urlopen(request, timeout=3600) as response:
    value = json.load(response)
timing = value.get("timings", {})
tokens = value.get("tokens")
if not isinstance(tokens, list) or timing.get("predicted_n") != int(n_predict):
    raise SystemExit("completion did not return the requested token count")
with open(response_path, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    wall_end=$(python3 -c 'import time; print(time.monotonic_ns())')
    sleep 1
    end_line=$(wc -l < "$server_log")
    python3 - "$meta" "$label" "$(basename "$server_log")" "$begin_line" "$end_line" \
            "$wall_start" "$wall_end" "$prompt_file" "$n_predict" "$slot" \
            "$cache_prompt" "$server_profile" <<'PY'
import json
import sys

(path, label, log, begin, end, wall_start, wall_end, prompt_path,
 n_predict, slot, cache_prompt, profile) = sys.argv[1:]
with open(prompt_path) as handle:
    prompt_tokens = len(json.load(handle))
value = {
    "label": label,
    "log": log,
    "begin_line": int(begin),
    "end_line": int(end),
    "wall_ms": (int(wall_end) - int(wall_start)) / 1e6,
    "prompt_file": prompt_path,
    "prompt_tokens": prompt_tokens,
    "n_predict": int(n_predict),
    "slot": int(slot),
    "cache_prompt": cache_prompt == "1",
    "kprof": profile == "1",
}
with open(path, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    python3 - "$response" "$label" <<'PY' | tee -a "$result_root/request-summary.txt"
import hashlib
import json
import sys

value = json.load(open(sys.argv[1]))
timing = value["timings"]
content = value["content"]
tokens = value["tokens"]
print(
    "REQUEST", sys.argv[2],
    "prompt_n", timing.get("prompt_n"),
    "cache_n", timing.get("cache_n"),
    "pred_ms", "%.3f" % timing.get("predicted_ms", float("nan")),
    "tps", "%.3f" % timing.get("predicted_per_second", float("nan")),
    "draft", "%s/%s" % (timing.get("draft_n_accepted"), timing.get("draft_n")),
    "content_sha", hashlib.sha256(content.encode()).hexdigest(),
    "tokens_sha", hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
    flush=True,
)
PY
}

cache_state() {
    local output=$1
    curl -sS --fail --max-time 60 \
        -H "Authorization: Bearer $api_key" \
        "http://127.0.0.1:$bench_port/m2-dashboard/cache-state" > "$output"
}

preflight_code=$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer $api_key" http://127.0.0.1:8080/health || true)
if [ "$preflight_code" != "200" ]; then
    log "ERROR: production preflight /health returned $preflight_code; refusing takeover"
    exit 1
fi
log "production preflight /health 200"
production_taken=1
log "taking production window"
tmux kill-session -t llama-server 2>/dev/null || true
pkill -f 'llama-server.*--port 8080' 2>/dev/null || true
sleep 8
capture_system before

start_server stage 0 "$deep_cache_dir"
near_n=$(tokenize_prompt near "$result_root/near-base-tokens.json")
deep_n=$(tokenize_prompt deep "$result_root/deep-base-tokens.json" "$depth_chars")
log "tokenized prompts: near=$near_n deep=$deep_n"
if [ "$deep_n" -lt 90000 ]; then
    log "ERROR: deep prompt has fewer than 90k tokens"
    exit 1
fi
if [ "$deep_n" -gt 150000 ] && [ "$allow_deeper" != "1" ]; then
    log "ERROR: tokenized prompt crossed the 150k safety gate"
    exit 1
fi

run_request near-ref "$result_root/near-base-tokens.json" "$profile_tokens" 3 0
run_request deep-stage "$result_root/deep-base-tokens.json" "$stage_tokens" 0 1

python3 - "$result_root/deep-base-tokens.json" "$result_root/request-deep-stage.json" \
        "$result_root/deep-replay-tokens.json" "$stage_tokens" <<'PY'
import json
import sys

base_path, response_path, output_path, expected = sys.argv[1:]
base = json.load(open(base_path))
response = json.load(open(response_path))
generated = response.get("tokens")
if not isinstance(generated, list) or len(generated) != int(expected):
    raise SystemExit("stage response does not contain the exact generated token suffix")
replay = base + generated
with open(output_path, "w") as handle:
    json.dump(replay, handle, separators=(",", ":"))
    handle.write("\n")
print("deep replay tokens", len(replay))
PY
replay_n=$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' \
    "$result_root/deep-replay-tokens.json")
# The final returned token may still be sampled-but-undecoded when the slot is
# serialized.  In that normal case the exact restorable state ends one token
# before the replay request, which then evaluates that one-token suffix.
cache_token_min=$((replay_n - 1))

deep_disk_entry=
# Check after every unique entry and stop as soon as the exact deep state spills.
for i in $(seq 1 "$max_fillers"); do
    filler_prompt=$result_root/filler-$i-tokens.json
    tokenize_prompt "Unique DSpark cache spill filler $i; do not reuse another filler." \
        "$filler_prompt" >/dev/null
    slot=$(( (i - 1) % 3 + 1 ))
    run_request "filler-$i" "$filler_prompt" 1 "$slot" 0
    state=$result_root/cache-state-fill-$i.json
    cache_state "$state"
    if [ "$i" = "1" ]; then
        python3 - "$state" "$cache_token_min" "$replay_n" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1]))
low, high = map(int, sys.argv[2:4])
if not any(low <= int(entry.get("tokens", -1)) <= high for entry in state.get("entries", [])):
    raise SystemExit("full deep target+draft state was not saved on the first idle-slot trigger")
PY
    fi
    if python3 - "$state" "$cache_token_min" "$replay_n" \
            "$result_root/deep-cache-entry.json" <<'PY'
import json
import sys

state = json.load(open(sys.argv[1]))
low, high = map(int, sys.argv[2:4])
matches = [entry for entry in state.get("entries", [])
           if low <= int(entry.get("tokens", -1)) <= high and entry.get("tier") == "disk"]
if len(matches) != 1 or not matches[0].get("file"):
    raise SystemExit(1)
with open(sys.argv[4], "w") as handle:
    json.dump(matches[0], handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
    then
        deep_disk_entry=$result_root/deep-cache-entry.json
        log "deep cache entry reached SSD tier after $i filler requests"
        break
    fi
done
if [ -z "$deep_disk_entry" ]; then
    log "ERROR: exact deep state did not reach disk after $max_fillers fillers"
    exit 1
fi
cache_entry_n=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' \
    "$deep_disk_entry")
log "deep SSD entry holds $cache_entry_n tokens for the $replay_n-token replay request"

python3 - "$deep_disk_entry" "$result_root/deep-cache-header.json" \
        "$deep_cache_dir" "$cache_entry_n" <<'PY'
import json
import os
import struct
import sys

entry_path, output_path, cache_root, expected_tokens = sys.argv[1:]
entry = json.load(open(entry_path))
reported_path = entry["file"]
if not os.path.isabs(reported_path):
    reported_path = os.path.join(cache_root, reported_path)
path = os.path.realpath(reported_path)
root = os.path.realpath(cache_root) + os.sep
if not path.startswith(root):
    raise SystemExit("cache entry escaped the scratch cache directory")
with open(path, "rb") as handle:
    raw = handle.read(72)
if len(raw) != 72:
    raise SystemExit("truncated cache header")
(magic, version, fingerprint, size_main, size_drft, n_tokens, n_checkpoints,
 size_payload, hash_payload, hash_header) = struct.unpack("=IIQQQQQQQQ", raw)
if magic != 0x4350434C or version != 2:
    raise SystemExit("unexpected cache schema")
if n_tokens != int(expected_tokens) or size_drft == 0:
    raise SystemExit("cache is not the exact full target+draft state")
value = {
    "file": path,
    "file_size": os.path.getsize(path),
    "magic": magic,
    "version": version,
    "fingerprint": fingerprint,
    "size_main": size_main,
    "size_drft": size_drft,
    "n_tokens": n_tokens,
    "n_checkpoints": n_checkpoints,
    "size_payload": size_payload,
    "hash_payload": hash_payload,
    "hash_header": hash_header,
}
with open(output_path, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
capture_system staged-deep
finish_server

for i in 1 2; do
    start_server "deep-ref-$i" 0 "$deep_cache_dir"
    run_request "deep-ref-$i" "$result_root/deep-replay-tokens.json" "$profile_tokens" 0 1
    capture_system "deep-ref-$i"
    finish_server
done

for label in near-kprof-1 deep-kprof-1 deep-kprof-2 near-kprof-2; do
    case "$label" in
        near-*) prompt=$result_root/near-base-tokens.json; cache=$near_cache_dir; use_cache=0 ;;
        deep-*) prompt=$result_root/deep-replay-tokens.json; cache=$deep_cache_dir; use_cache=1 ;;
    esac
    start_server "$label" 1 "$cache"
    run_request "$label" "$prompt" "$profile_tokens" 0 "$use_cache"
    capture_system "$label"
    finish_server
done

capture_system after
python3 "$analyzer" "$result_root" --json-out "$result_root/analysis.json" \
    | tee "$result_root/analysis.txt"
log "measurement complete; analyzer stopped at the 100k decision gate"

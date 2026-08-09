#!/bin/bash
# Fixed-width production DSV4 speculative-verify census for the M2 Ultra.
# This script deliberately refuses to run unless m2-ultra-queue.sh owns it.

set -euo pipefail

if [ -z "${M2_ULTRA_QUEUE_TOKEN:-}" ]; then
    echo "error: run this harness through m2-ultra-queue.sh" >&2
    exit 64
fi

export PATH=/opt/homebrew/bin:$PATH

WD=${WD:-/Users/agusx1211/worktrees/llama-cpp-m2-ultra/campaign-0809-verify-q}
MODEL=${MODEL:-/Users/agusx1211/unsloth/gguf-m2/dsv4-flash-0731-full.m2.gguf}
DRAFT=${DRAFT:-/Users/agusx1211/unsloth/gguf-m2/dspark-0731-expertsonly.m2.gguf}
KEY=${KEY:-llamacpp}
PORT=${PORT:-8094}
# The pinned DSpark checkpoint advertises deepseek4.block_size=5.  The runtime
# clamps n_max to that trained width, so fixed/provenance-controlled target
# verify batches can contain at most Q = 1 + 5 rows.
SHAPES=${SHAPES:-"1 2 3 4 6"}
DEPTH=${DEPTH:-12k}
NPRED=${NPRED:-128}
KPROF_STRIDE=${KPROF_STRIDE:-1}
TAG=${TAG:-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=${OUT:-/Users/agusx1211/vqc-results-$TAG}
PROD_URL=http://127.0.0.1:8080
TEST_URL=http://127.0.0.1:$PORT

mkdir -p "$OUT"

for qrows in $SHAPES; do
    case $qrows in
        ''|*[!0-9]*)
            echo "error: invalid query-row width: $qrows" >&2
            exit 64
            ;;
    esac
    if [ "$qrows" -lt 1 ] || [ "$qrows" -gt 6 ]; then
        echo "error: fixed DSpark query-row width must be in 1..6 (requested $qrows)" >&2
        exit 64
    fi
done

log() {
    echo "[verify-q $(date -u +%H:%M:%S)] $*"
}

available_gib() {
    vm_stat | awk '
        /page size of/      { ps=$8 }
        /Pages free/        { gsub(/\./,"",$3); f=$3 }
        /Pages inactive/    { gsub(/\./,"",$3); i=$3 }
        /Pages purgeable/   { gsub(/\./,"",$3); p=$3 }
        /Pages speculative/ { gsub(/\./,"",$3); s=$3 }
        END { print int((f+i+p+s)*ps/1073741824) }'
}

stop_port() {
    local port=$1
    local pids
    pids=$(pgrep -f "llama-server.*--port $port" || true)
    if [ -n "$pids" ]; then
        while IFS= read -r pid; do
            kill "$pid" 2>/dev/null || true
        done <<< "$pids"
        for _ in $(seq 1 60); do
            pgrep -f "llama-server.*--port $port" >/dev/null || break
            sleep 2
        done
        pids=$(pgrep -f "llama-server.*--port $port" || true)
        if [ -n "$pids" ]; then
            while IFS= read -r pid; do
                kill -9 "$pid" 2>/dev/null || true
            done <<< "$pids"
        fi
    fi
}

wait_memory() {
    for _ in $(seq 1 90); do
        [ "$(available_gib)" -ge 150 ] && return 0
        sleep 2
    done
    log "warning: only $(available_gib) GiB reclaimable after waiting"
}

restore_prod() {
    local restore_rc=0
    local code=000
    local completion

    log "restoring production"
    stop_port "$PORT"
    stop_port 8080
    # Production intentionally outlives this queue job, so do not let the
    # queue ownership token propagate into its tmux session.
    if ! env -u M2_ULTRA_QUEUE_TOKEN "$HOME/start-llama-server.sh" elastic4; then
        log "error: production launcher failed"
        restore_rc=1
    fi

    for _ in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" "$PROD_URL/health" || true)
        [ "$code" = 200 ] && break
        sleep 5
    done
    log "production /health=$code"

    if [ "$code" != 200 ]; then
        log "error: production did not become healthy"
        restore_rc=1
    elif ! completion=$(curl -sS --fail --max-time 120 \
            -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
            -d '{"prompt":"verify-q restore probe","n_predict":1,"temperature":0}' \
            "$PROD_URL/completion"); then
        log "error: production completion probe failed"
        restore_rc=1
    elif ! printf '%s' "$completion" | python3 -c '
import json
import sys

result = json.load(sys.stdin)
content = result.get("content")
predicted_n = (result.get("timings") or {}).get("predicted_n")
if predicted_n != 1 or not isinstance(content, str) or not content:
    raise SystemExit("expected predicted_n=1 and non-empty content")
print("restore completion: predicted_n=1 content_len=%d" % len(content))
'; then
        log "error: production completion probe returned invalid JSON/result"
        restore_rc=1
    fi

    # This lock protects only the census window. Never strand it because
    # production restoration or validation failed.
    if ! rm -f "$HOME/m2-window.lock"; then
        log "error: could not remove legacy M2 window lock"
        restore_rc=1
    fi

    return "$restore_rc"
}

on_exit() {
    local rc=$?
    local restore_rc

    trap - EXIT INT TERM
    set +e
    restore_prod
    restore_rc=$?
    if [ "$rc" -eq 0 ] && [ "$restore_rc" -ne 0 ]; then
        rc=$restore_rc
    fi
    exit "$rc"
}

if [ -e "$HOME/m2-window.lock" ]; then
    echo "error: legacy M2 window lock is held: $(cat "$HOME/m2-window.lock" 2>/dev/null)" >&2
    exit 75
fi
( set -C; echo "lane=campaign-0809-verify-q token=$M2_ULTRA_QUEUE_TOKEN taken=$(date -u +%FT%TZ)" > "$HOME/m2-window.lock" )
trap on_exit EXIT INT TERM

prod_code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" "$PROD_URL/health" || true)
[ "$prod_code" = 200 ] || { log "production unhealthy before census: HTTP $prod_code"; exit 1; }

{
    echo "utc=$(date -u +%FT%TZ)"
    echo "commit=$(git -C "$WD" rev-parse HEAD)"
    echo "branch=$(git -C "$WD" branch --show-current)"
    echo "queue_token=$M2_ULTRA_QUEUE_TOKEN"
    echo "shapes=$SHAPES"
    echo "depth=$DEPTH"
    echo "n_predict=$NPRED"
    echo "kprof_stride=$KPROF_STRIDE"
    stat -f 'model=%N size=%z mtime=%Sm' -t '%Y-%m-%dT%H:%M:%SZ' "$MODEL"
    stat -f 'draft=%N size=%z mtime=%Sm' -t '%Y-%m-%dT%H:%M:%SZ' "$DRAFT"
    sw_vers | tr '\n' ' '; echo
    /usr/bin/xcrun metal --version 2>&1 | head -1 || true
} > "$OUT/provenance.txt"

log "stopping production (preflight HTTP 200)"
tmux kill-session -t llama-server 2>/dev/null || true
stop_port 8080
wait_memory

for qrows in $SHAPES; do
    draft_n=$((qrows - 1))
    draft_min=1
    [ "$draft_n" -gt 0 ] || draft_min=0
    name="q$qrows-$DEPTH-s$KPROF_STRIDE"
    server_log="$OUT/$name.server.log"
    result_json="$OUT/$name.result.json"

    log "starting $name (draft_n=$draft_n, available=$(available_gib) GiB)"
    (
        cd "$WD"
        exec env DYLD_LIBRARY_PATH="$WD/build-m2/bin" \
            LLAMA_HOST_PROFILE=1 \
            GGML_METAL_KPROF="$KPROF_STRIDE" \
            GGML_METAL_KPROF_DEBUG=1 \
            GGML_FA_GEOM_LOG=1 \
            LLAMA_DSV4_SPARSE_ROUTE_DEBUG=20000 \
            LLAMA_DSV4_ADMISSION_VERTICAL=1 \
            GGML_E4M3_MM_NT2_MIN_N=512 \
            "$WD/build-m2/bin/llama-server" \
                -m "$MODEL" -md "$DRAFT" --spec-type draft-dspark \
                --spec-draft-n-max "$draft_n" --spec-draft-n-min "$draft_min" --spec-draft-p-min 0 \
                -c 1048576 -np 4 --kv-unified --no-context-shift --cache-ram 0 \
                -ctk f16 -ctv f16 -ngl 999 -ngld 999 -fa on -fit on -ub 2048 -b 2048 \
                --host 127.0.0.1 --port "$PORT" --api-key "$KEY" --no-webui
    ) > "$server_log" 2>&1 &
    server_pid=$!

    code=000
    for _ in $(seq 1 200); do
        code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" "$TEST_URL/health" || true)
        [ "$code" = 200 ] && break
        kill -0 "$server_pid" 2>/dev/null || break
        sleep 3
    done
    log "$name /health=$code"
    [ "$code" = 200 ] || { tail -40 "$server_log"; exit 1; }

    python3 - "$KEY" "$WD" "$DEPTH" "$NPRED" "$PORT" "$qrows" "$result_json" <<'PY'
import hashlib
import importlib.util
import json
import sys
import time
import urllib.request

key, root, depth, npred, port, qrows, output = sys.argv[1:8]
spec = importlib.util.spec_from_file_location("dd", root + "/scripts/dsv4-depth-determinism.py")
dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)
prompt = dd.build_prompt(dd.DEPTHS[depth])
body = json.dumps({"prompt": prompt, "n_predict": int(npred), "temperature": 0,
                   "cache_prompt": False, "ignore_eos": True, "id_slot": 0}).encode()
request = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key})
t0 = time.time()
with urllib.request.urlopen(request, timeout=3600) as response:
    result = json.load(response)
content = result.get("content") or ""
timings = result.get("timings") or {}
record = {"query_rows": int(qrows), "depth": depth, "wall_s": time.time() - t0,
          "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
          "content_len": len(content), "timings": timings}
with open(output, "w") as fh:
    json.dump(record, fh, indent=2, sort_keys=True)
    fh.write("\n")
print("VQCRUN", json.dumps(record, sort_keys=True), flush=True)
PY

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    stop_port "$PORT"
    wait_memory

    log "$name profiles=$(grep -c '^KPROFS ' "$server_log" || true) metal_errors=$(grep -cE 'Invalid Resource|did not complete|command buffers failed|GGML_ASSERT|OutOfMemory' "$server_log" || true)"
    gzip -f "$server_log"
done

log "fixed-width census complete: $OUT"

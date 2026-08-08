#!/bin/bash
# M2 validation: elastic -np 2 + prompt cache + SSD tier (DeepSeek V4 Flash).
# Runs on the m2. Assumes prod server is stopped. Port 8090.
set -u
export PATH="/opt/homebrew/bin:$PATH"
unset M2_ULTRA_QUEUE_TOKEN
WD=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/elastic-ssd-integration
BIN=$WD/build-m2/bin/llama-server
MODEL=/Users/agusx1211/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf
DRAFT=/Users/agusx1211/unsloth/DeepSeek-V4-Flash-DSpark-official/DeepSeek-V4-Flash-DSpark-BF16.gguf
PORT=8090
LOG=/tmp/esi-validate-server.log
CACHE_DIR=/Users/agusx1211/llama-pcache-test
PHASE="${1:-elastic}"

# ~3000-token filler with an embedded recall code
FILLER=$(python3 - <<'PY'
base = ("The observatory logged wind speed, humidity, pressure and temperature at "
        "minute intervals while the maintenance crew rotated the primary mirror "
        "assembly and recalibrated the spectrograph baseline against the reference "
        "lamp. ")
print((base * 90).strip())
PY
)
PROMPT_A="$FILLER The secret code is ZEBRA742. Remember it. Question: what is the secret code? Answer with only the code."
EXT=" Now answer again, in the form 'code: <value>' with nothing else."
PROMPT_B="$PROMPT_A$EXT"
DIVERGENT="$FILLER The secret code is OTTER911. Remember it. Question: what is the secret code? Answer with only the code."

start_server() { # $1 = extra flags, $2 = extra env, $3 = log file (optional)
  LOG="${3:-$LOG}"
  rm -f $LOG
  tmux kill-session -t esi-validate 2>/dev/null
  pkill -f "llama-server.*--port $PORT" 2>/dev/null
  for i in $(seq 1 90); do
    pgrep -f "llama-server.*--port $PORT" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 2
  tmux new-session -d -s esi-validate "cd '$WD' && export DYLD_LIBRARY_PATH='$WD/build-m2/bin' $2 && '$BIN' -m '$MODEL' -md '$DRAFT' --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-n-min 1 --spec-draft-p-min 0.5 -c 524288 -ctk f16 -ctv f16 -ngl 999 -ngld 999 -fa on -fit on -ub 2048 -b 2048 --host 127.0.0.1 --port $PORT --no-webui $1 2>&1 | tee $LOG"
  echo "waiting for health..."
  for i in $(seq 1 900); do
    curl -sf -o /dev/null http://127.0.0.1:$PORT/health 2>/dev/null && { echo "healthy after ${i}s"; return 0; }
    sleep 1
  done
  echo "SERVER FAILED TO START"; tail -30 $LOG; return 1
}

req() { # $1 = name, $2 = prompt, $3 = n_predict
  local out http t0 t1
  t0=$(python3 -c 'import time; print(time.time())')
  out=$(curl -s -m 600 -w "\n%{http_code}" http://127.0.0.1:$PORT/completion \
        -H 'Content-Type: application/json' \
        --data-binary @<(python3 -c "import json,sys; print(json.dumps({'prompt': sys.argv[1], 'n_predict': int(sys.argv[2]), 'cache_prompt': True, 'temperature': 0}))" "$2" "$3"))
  t1=$(python3 -c 'import time; print(time.time())')
  http=$(echo "$out" | tail -1)
  echo "$out" | sed '$d' > /tmp/esi-last.json
  python3 - "$1" "$http" "$t0" "$t1" <<'PY'
import json, sys
name, http, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
try:
    d = json.load(open('/tmp/esi-last.json'))
    t = d.get('timings', {})
    print(f"[{name}] http={http} wall={t1-t0:.1f}s content={d.get('content','')[:60]!r} "
          f"prompt_n={t.get('prompt_n')} cache_n={t.get('cache_n')} "
          f"pp_ms={t.get('prompt_ms',0):.0f} tg_n={t.get('predicted_n')}")
except Exception as e:
    print(f"[{name}] http={http} wall={t1-t0:.1f}s PARSE-FAIL {e}")
PY
  if [ "$http" != "200" ]; then
    echo "  [server log tail]"; tail -8 $LOG | sed 's/^/  | /'
  fi
}

case "$PHASE" in
elastic)
  echo "=== PHASE elastic: -np 2 vertical + RAM cache (no SSD)"
  start_server "-np 2 --kv-unified --no-context-shift --cache-ram 32768" "LLAMA_DSV4_ADMISSION_VERTICAL=1" || exit 1
  echo "--- T1: cold A (full prefill baseline)"
  req "T1-cold-A" "$PROMPT_A" 24
  CONT_A=$(python3 -c "import json;print(json.load(open('/tmp/esi-last.json'))['content'], end='')")
  echo "--- T2: verbatim continuation (A + T1 output + suffix; expect rs-margin reuse, cache_n ~ full)"
  req "T2-verbatim" "$PROMPT_A$CONT_A Now repeat the code once more, nothing else." 24
  echo "--- T2b: synthetic divergent tail (no checkpoint cover; expect full clear, cache_n=0, correct)"
  req "T2b-divergent-tail" "$PROMPT_B" 24
  echo "--- T3: two concurrent large prompts (both must be 200)"
  req "T3-conc-1" "$DIVERGENT" 48 & P1=$!
  sleep 0.3
  req "T3-conc-2" "$FILLER Question: how many weather variables are listed? Answer with a number." 48 & P2=$!
  wait $P1 $P2
  echo "--- T4: verbatim continuation while a long generation occupies the other lane"
  req "T4-longgen" "$FILLER Write a long detailed story about the observatory crew." 400 & P3=$!
  sleep 3
  req "T4-verb-busy" "$PROMPT_A$CONT_A Now write the code in lowercase, nothing else." 24
  wait $P3
  echo "--- T6: long prompt (~10k) cold, then divergent continuation (expect checkpoint reuse)"
  FILLER10=$(python3 -c "
base = ('The observatory logged wind speed, humidity, pressure and temperature at '
        'minute intervals while the maintenance crew rotated the primary mirror '
        'assembly and recalibrated the spectrograph baseline against the reference '
        'lamp. ')
print((base * 300).strip())")
  req "T6a-cold-10k" "$FILLER10 The secret code is LLAMA555. Question: what is the secret code? Answer with only the code." 24
  req "T6b-div-10k" "$FILLER10 The secret code is LLAMA555. Different question now: repeat the code twice separated by a dash." 24
  echo "--- T5: 0 compute errors / 500s check"
  grep -cE "Compute error|failed to decode|GGML_ABORT" $LOG || true
  grep -E "elastic admission reuses|restored context checkpoint" $LOG | tail -8
  tmux kill-session -t esi-validate 2>/dev/null
  ;;
ssd)
  echo "=== PHASE ssd: elastic2 + SSD tier (small RAM cache to force spills)"
  rm -rf $CACHE_DIR
  SSD_FLAGS="-np 2 --kv-unified --no-context-shift --cache-ram 128 --cache-disk $CACHE_DIR --cache-disk-limit 400"
  start_server "$SSD_FLAGS" "LLAMA_DSV4_ADMISSION_VERTICAL=1" "/tmp/esi-ssd-1.log" || exit 1
  echo "--- S1: cold A"
  req "S1-cold-A" "$PROMPT_A" 24
  echo "--- S2-S3b: three distinct conversations to push A out of the 128 MiB RAM tier"
  req "S2-conv-B" "Harbor logs. $FILLER The secret code is OTTER911. Question: what is the secret code?" 24
  req "S3-conv-C" "Forest notes. $FILLER Question: name one instrument mentioned. One word only." 24
  req "S3b-conv-D" "Desert diary. $FILLER Question: how many weather variables are listed? A number only." 24
  echo "--- cache dir after eviction pressure (expect spilled .lcpc files):"
  ls -la $CACHE_DIR 2>/dev/null
  echo "--- S4: extension of A (RAM miss expected -> disk load)"
  req "S4-ext-A" "$PROMPT_B" 24
  echo "--- first-instance cache log lines:"
  grep -E "spilled prompt|loaded prompt|failed to restore|dropping cache|does not match|stale" $LOG | tail -10
  echo "--- restart server (rescan + disk hit after restart)"
  start_server "$SSD_FLAGS" "LLAMA_DSV4_ADMISSION_VERTICAL=1" "/tmp/esi-ssd-2.log" || exit 1
  echo "--- S5: extension of A after restart (expect rescan + disk load, fast prefill)"
  req "S5-ext-A-restart" "$PROMPT_B Confirm the code again." 24
  grep -E "spilled prompt|loaded prompt|restored [0-9]+ entries|SSD tier" $LOG | tail -8
  echo "--- errors:"
  grep -cE "Compute error|failed to decode|GGML_ABORT" $LOG || true
  tmux kill-session -t esi-validate 2>/dev/null
  ;;
np4)
  echo "=== PHASE np4: 4-lane elastic vertical + RAM/SSD cache"
  WD=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/elastic-np4
  BIN=$WD/build-m2/bin/llama-server
  rm -rf $CACHE_DIR
  start_server "-np 4 --kv-unified --no-context-shift --cache-ram 8192 --cache-disk $CACHE_DIR --cache-disk-limit 400" "LLAMA_DSV4_ADMISSION_VERTICAL=1" "/tmp/esi-np4.log" || exit 1
  echo "--- N1: four concurrent conversations (all must be 200, each with its own code)"
  req "N1-c1" "Alpha station. $FILLER The alpha code is RIVER111. What is the alpha code? Answer with only the code." 48 & P1=$!
  sleep 0.2
  req "N1-c2" "Bravo station. $FILLER The bravo code is STONE222. What is the bravo code? Answer with only the code." 48 & P2=$!
  sleep 0.2
  req "N1-c3" "Charlie station. $FILLER The charlie code is CLOUD333. What is the charlie code? Answer with only the code." 48 & P3=$!
  sleep 0.2
  req "N1-c4" "Delta station. $FILLER The delta code is FLAME444. What is the delta code? Answer with only the code." 48 & P4=$!
  wait $P1 $P2 $P3 $P4
  echo "--- N2: fifth request while lanes are occupied (must queue, then 200)"
  req "N2-c5" "Echo station. $FILLER The echo code is GRASS555. What is the echo code? Answer with only the code." 48
  echo "--- N3: verbatim continuation of conversation 1 (expect cache_n > 0)"
  CONT1=$(python3 -c "import json;print(json.load(open('/tmp/esi-last.json'))['content'], end='')" 2>/dev/null)
  req "N3-cont-c5" "Echo station. $FILLER The echo code is GRASS555. What is the echo code? Answer with only the code.$CONT1 Repeat the echo code once more." 24
  echo "--- N4: error scan + reuse log"
  grep -cE "Compute error|failed to decode|GGML_ABORT" $LOG || true
  grep -E "elastic admission reuses|slots=4" $LOG | tail -6
  tmux kill-session -t esi-validate 2>/dev/null
  ;;
xconv)
  echo "=== PHASE xconv: cross-conversation shared-prefix reuse (two parallel agents)"
  WD=/Users/agusx1211/worktrees/llama-cpp-m2-ultra/prefix-publish
  BIN=$WD/build-m2/bin/llama-server
  rm -rf $CACHE_DIR
  start_server "-np 4 --kv-unified --no-context-shift --cache-ram 8192 --cache-disk $CACHE_DIR --cache-disk-limit 400" "LLAMA_DSV4_ADMISSION_VERTICAL=1" "/tmp/esi-xconv.log" || exit 1
  echo "--- X1: agent A cold (shared system prefix + task one, long generation)"
  req "X1-agentA" "$FILLER You are agent ALPHA. Task one: write a detailed plan for recalibrating the spectrograph." 300 & PA=$!
  sleep 16
  echo "--- X2: agent B arrives while A is generating (same prefix, different task)"
  req "X2-agentB" "$FILLER You are agent BETA. Task two: list five telemetry variables. Answer briefly." 32
  wait $PA
  echo "--- X3: agent C after both are idle (same prefix, third task)"
  req "X3-agentC" "$FILLER You are agent GAMMA. Task three: name the reference light source. Answer briefly." 32
  echo "--- cache/publish log lines:"
  grep -E "published prefilled|prompt cache: |reuses resident" $LOG | tail -10
  echo "--- errors:"
  grep -cE "Compute error|failed to decode|GGML_ABORT" $LOG || true
  tmux kill-session -t esi-validate 2>/dev/null
  ;;
esac
echo "=== PHASE $PHASE done"

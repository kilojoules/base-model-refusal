#!/usr/bin/env bash
# Stage 2a: refusal-maximising wrapper search.
#
# GPU 1 serves the proposer (an instruct model, OpenAI-compatible). GPU 0 holds the
# target. The proposer only ever sees content-free templates, so no benchmark content
# crosses between processes.
#
# Three searches, in this order:
#   1. base model, strict guard  -- the measurement
#   2. instruct model, strict guard -- the POSITIVE CONTROL. Without this a null on the
#      base model is evidence about the search, not about the model.
#   3. base model, loose guard   -- an upper bound, where the wrapper may talk about
#      declining. A high number here is weak evidence; a base model can be prompted
#      into most surface forms.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

PROPOSER="${PROPOSER:-Qwen/Qwen2.5-14B-Instruct}"
TARGET="${TARGET:-Qwen/Qwen2.5-7B}"
CONTROL="${CONTROL:-Qwen/Qwen2.5-7B-Instruct}"
RUNS="${RUNS:-4}"
ROUNDS="${ROUNDS:-12}"
mkdir -p logs results/attack

if ! curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
  echo "[stage2] starting proposer server: $PROPOSER on GPU 1"
  CUDA_VISIBLE_DEVICES=1 setsid nohup python -m vllm.entrypoints.openai.api_server \
      --model "$PROPOSER" --port 8000 --max-model-len 8192 \
      --gpu-memory-utilization 0.90 --disable-log-requests \
      > logs/proposer_server.log 2>&1 < /dev/null &
  disown
  echo "[stage2] waiting for proposer..."
  for _ in $(seq 1 180); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && break
    sleep 10
  done
fi
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "[stage2] proposer failed to start"; tail -20 logs/proposer_server.log; exit 1; }
echo "[stage2] proposer ready"

export PROPOSER_URL=http://127.0.0.1:8000/v1
export PROPOSER_MODEL="$PROPOSER"
export PROPOSER_API_KEY=none

run_search () {  # target, mode, label
  echo ""
  echo "=============================================================="
  echo "[stage2] $3: target=$1 mode=$2 runs=$RUNS rounds=$ROUNDS"
  echo "=============================================================="
  CUDA_VISIBLE_DEVICES=0 python scripts/stage2_attack.py \
    --target "$1" --mode "$2" --runs "$RUNS" --rounds "$ROUNDS" --tp 1 \
    2>&1 | tee -a "logs/stage2_$3.log"
}

run_search "$TARGET"  strict measurement
run_search "$CONTROL" strict positive_control
run_search "$TARGET"  loose  upper_bound

python - <<'PY' 2>&1 | tee -a logs/stage2_summary.log
import json, glob, os
print("\n==== stage 2 ceilings (held-out, nats/token) ====")
rows=[]
for f in sorted(glob.glob('results/attack/*/summary.json')):
    d=json.load(open(f))
    rows.append((os.path.basename(os.path.dirname(f)), d['target'], d['mode'],
                 d['ceiling_heldout'],
                 [r['heldout_reward'] for r in d['per_run_best']]))
for name,t,m,c,per in rows:
    spread = (max(x for x in per if x is not None)-min(x for x in per if x is not None)) if any(x is not None for x in per) else float('nan')
    print(f"{t.split('/')[-1]:<24} mode={m:<7} ceiling={c:+.4f}  per-run spread={spread:.4f}")
base=[r for r in rows if r[2]=='strict' and 'Instruct' not in r[1]]
ctrl=[r for r in rows if r[2]=='strict' and 'Instruct' in r[1]]
if base and ctrl:
    d=ctrl[0][3]-base[0][3]
    print(f"\nPOSITIVE CONTROL separation = {d:+.4f} nats/token")
    print("  the search can find refusal where refusal exists; a low base ceiling is"
          " a fact about the base model"
          if d>0.2 else
          "  WARNING: control did not separate -- the search is underpowered; do not"
          " report the base ceiling as a null")
PY
echo "stage 2 complete"

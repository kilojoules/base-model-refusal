#!/usr/bin/env bash
# Judge phase runner. Usage: run_judge.sh [limit]
#   OUT=path  override the output file
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src VLLM_USE_FLASHINFER_SAMPLER=0
if [ -f /workspace/bre/.hf_token ]; then
  HF_TOKEN="$(cat /workspace/bre/.hf_token)"
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
OUT="${OUT:-results/judged.jsonl}"
if [ "${1:-}" != "" ]; then
  python scripts/stage1_judge.py --generations results/generations.jsonl \
    --out "$OUT" --limit "$1"
else
  python scripts/stage1_judge.py --generations results/generations.jsonl --out "$OUT"
fi

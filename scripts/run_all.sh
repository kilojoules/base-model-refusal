#!/usr/bin/env bash
# Full stage 1: probe the ladder (+ anchors), sample a subset, judge, analyse.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
# flashinfer JIT-compiles sampling kernels and this image has no matching
# nvcc; the native sampler is equivalent for our purposes.
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/stage1_probe.py --with-generations --with-fewshot --purge-cache \
  --probe-out results/probe.jsonl --gen-out results/generations.jsonl 2>&1 \
  | tee -a logs/stage1_probe.log

python scripts/stage1_judge.py \
  --generations results/generations.jsonl --out results/judged.jsonl 2>&1 \
  | tee -a logs/stage1_judge.log

python scripts/report.py --probe results/probe.jsonl \
  --judged results/judged.jsonl --out results/summary.json 2>&1 \
  | tee -a logs/report.log

echo "stage 1 complete"

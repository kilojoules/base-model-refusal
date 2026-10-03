#!/usr/bin/env bash
# Activation steering: extract harm direction from residual stream, sweep alpha.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p results logs

# 7B base first (the model we have the most behavioral data on)
python3 scripts/activation_steering.py \
  --model Qwen/Qwen2.5-7B \
  --out results/activation_steering.jsonl \
  --n-extract 80 --n-test 40 \
  --alphas="-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8" \
  --arm chat \
  --gen-samples 5 \
  2>&1 | tee -a logs/activation_steering.log

# 7B instruct anchor
python3 scripts/activation_steering.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out results/activation_steering.jsonl \
  --n-extract 80 --n-test 40 \
  --alphas="-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8" \
  --arm chat \
  --gen-samples 5 \
  2>&1 | tee -a logs/activation_steering.log

echo "activation steering complete"

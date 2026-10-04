#!/usr/bin/env bash
# Steering + harm judging: generate steered completions, judge each one.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p results logs
pip install -q accelerate 2>/dev/null

# Clear previous results
rm -f results/steering_harm_judge.jsonl results/steering_harm_judge_transcripts.jsonl \
      results/steering_harm_judge_highlights.jsonl

# Base model: layer 23 — wider alpha sweep, push until incoherence
python3 scripts/steering_harm_judge.py \
  --model Qwen/Qwen2.5-7B \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layer 23 \
  --n-harmful 40 --n-benign 40 \
  --alphas="-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6" \
  --max-new-tokens 200 \
  --out results/steering_harm_judge.jsonl \
  2>&1 | tee logs/steering_harm_judge.log

# Instruct model: layer 26 — same wide sweep
python3 scripts/steering_harm_judge.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layer 26 \
  --n-harmful 40 --n-benign 40 \
  --alphas="-6,-4,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,3,4,6" \
  --max-new-tokens 200 \
  --out results/steering_harm_judge.jsonl \
  2>&1 | tee -a logs/steering_harm_judge.log

echo "steering harm judge complete"

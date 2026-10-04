#!/usr/bin/env bash
# Harm-direction steering: extract harm direction (high-harm vs low-harm prompts),
# steer with it, and see if benign prompts produce harmful output.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p results logs
pip install -q accelerate 'transformers<4.52' datasets 2>/dev/null

rm -f results/steering_harm_direction.jsonl results/steering_harm_direction_transcripts.jsonl \
      results/steering_harm_direction_scores.jsonl results/steering_harm_direction_norms.jsonl

# Base model — steer at layers 14,16,18,20 (where refusal direction worked)
python3 scripts/steering_harm_direction.py \
  --model Qwen/Qwen2.5-7B \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layers "0,4,8,12,14,16,18,20,22,24,26,27" \
  --steer-layers "14,16,18,20" \
  --n-direction 100 --n-steer 20 \
  --alphas="-3,-2,-1,0,1,2,3" \
  --max-new-tokens 200 \
  --out results/steering_harm_direction.jsonl \
  2>&1 | tee logs/steering_harm_direction.log

# Instruct model — same layers
python3 scripts/steering_harm_direction.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layers "0,4,8,12,14,16,18,20,22,24,26,27" \
  --steer-layers "14,16,18,20" \
  --n-direction 100 --n-steer 20 \
  --alphas="-3,-2,-1,0,1,2,3" \
  --max-new-tokens 200 \
  --out results/steering_harm_direction.jsonl \
  2>&1 | tee -a logs/steering_harm_direction.log

echo "harm direction steering complete"

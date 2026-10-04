#!/usr/bin/env bash
# Layer sweep: steer at every other layer to see how each layer's direction
# affects harmful vs benign tasks before reaching incoherence.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p results logs
pip install -q accelerate 2>/dev/null

rm -f results/steering_layer_sweep.jsonl results/steering_layer_sweep_transcripts.jsonl

LAYERS="0,4,8,12,14,16,18,20,22,24,26,27"

# Base model
python3 scripts/steering_layer_sweep.py \
  --model Qwen/Qwen2.5-7B \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layers "$LAYERS" \
  --n-harmful 20 --n-benign 20 \
  --alphas="-2,-1,0,1,2,3" \
  --max-new-tokens 200 \
  --out results/steering_layer_sweep.jsonl \
  2>&1 | tee logs/steering_layer_sweep.log

# Instruct model
python3 scripts/steering_layer_sweep.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --judge-model Qwen/Qwen2.5-7B-Instruct \
  --layers "$LAYERS" \
  --n-harmful 20 --n-benign 20 \
  --alphas="-2,-1,0,1,2,3" \
  --max-new-tokens 200 \
  --out results/steering_layer_sweep.jsonl \
  2>&1 | tee -a logs/steering_layer_sweep.log

echo "layer sweep complete"

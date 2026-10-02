#!/usr/bin/env bash
# Gate before spending GPU-hours on the ladder. Four checks.
#
#  1. the probe returns finite numbers for every arm;
#  2. the instruct anchor shows harm SENSITIVITY (a difference), not merely a high
#     level -- RPS carries a large positive string-frequency offset, so levels mean
#     nothing on their own;
#  3. the form-matched XSTest contrast-minus-safe difference behaves on the anchor;
#  4. the few-shot dose-response rises with k, so a flat ladder can be told apart from
#     a dead instrument.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/stage1_probe.py --only Qwen2.5-0.5B --with-fewshot \
  --fewshot-targets 24 --probe-out results/smoke.jsonl
python scripts/stage1_probe.py --only Qwen2.5-7B-Instruct --with-fewshot \
  --fewshot-targets 24 --probe-out results/smoke.jsonl

python scripts/gate_check.py results/smoke.jsonl

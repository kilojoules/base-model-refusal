#!/usr/bin/env bash
# Full judge phase, in order:
#   1. Llama-3.1-70B-Instruct INT4: compliance (official HarmBench rubric) + taxonomy
#   2. HarmBench-Llama-2-13b-cls: compliance again, as the comparability anchor
#   3. Agreement between the two, which is what validates the compliance axis -- the
#      synthetic set cannot, since the rubric is definitionally scoped to harmful
#      behaviours and validating it would mean authoring harmful completions.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src VLLM_USE_FLASHINFER_SAMPLER=0
HF_TOKEN="$(cat /workspace/bre/.hf_token)"; export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "=== pass 1: Llama-3.1-70B judge (compliance + taxonomy)"
python scripts/stage1_judge.py --generations results/generations.jsonl \
  --out results/judged.jsonl

echo "=== pass 2: HarmBench classifier (comparability anchor)"
python scripts/judge_anchor.py --generations results/generations.jsonl \
  --out results/anchor_compliance.jsonl

echo "=== pass 3: inter-judge agreement on compliance"
python scripts/compliance_agreement.py

echo "=== pass 4: report"
python scripts/report.py --probe results/probe.jsonl \
  --judged results/judged.jsonl --out results/summary.json
python scripts/report3.py results/probe.jsonl
echo "JUDGE PHASE COMPLETE"

#!/usr/bin/env bash
# Run the compliance_demos condition on the full model ladder.
# This is an additive experiment — it writes to a separate output file so
# it can be run on models that already have base probe data.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p results logs

python3 scripts/stage1_probe.py \
  --only-compliance-demos \
  --probe-out results/compliance_demos.jsonl \
  --include-anchors \
  --purge-cache \
  2>&1 | tee -a logs/compliance_demos.log

echo "compliance_demos complete"
echo "rows: $(wc -l < results/compliance_demos.jsonl)"

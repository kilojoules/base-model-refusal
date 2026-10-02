#!/usr/bin/env bash
# Full stage 1: probe the ladder (+ anchors), sample a subset, judge, analyse.
set -euo pipefail
cd /workspace/bre
export HF_HOME=/workspace/hf PYTHONPATH=src
# flashinfer JIT-compiles sampling kernels and this image has no matching
# nvcc; the native sampler is equivalent for our purposes.
export VLLM_USE_FLASHINFER_SAMPLER=0

python scripts/stage1_probe.py --with-generations \
  --probe-out results/probe.jsonl --gen-out results/generations.jsonl 2>&1 \
  | tee -a logs/stage1_probe.log

python scripts/stage1_judge.py \
  --generations results/generations.jsonl --out results/judged.jsonl 2>&1 \
  | tee -a logs/stage1_judge.log

python - <<'PY' 2>&1 | tee -a logs/analysis.log
import sys; sys.path.insert(0,'src')
from bre import analyze
r = analyze.report('results/probe.jsonl', 'results/judged.jsonl')
print('models:', r['models'])
print('probe rows:', r['n_probe_rows'])
print('\nsize trend (RPS per decade of params, by arm):')
for t in r['size_trend']:
    print(f"  {t['arm']:<12} slope={t['slope_per_decade']:+.4f} se={t['se']:.4f} p={t['p']:.3g}")
print('\narm x size interaction (chat vs document):', r['arm_by_size_interaction'])
PY
echo "stage 1 complete"

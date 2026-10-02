#!/usr/bin/env bash
# Pod-side setup. Idempotent.
set -euo pipefail
cd /workspace/bre

export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$HF_HOME"

if [ ! -f .deps_ok ]; then
  pip install -q --upgrade pip
  pip install -q "vllm>=0.6.3" "transformers>=4.45" datasets pandas numpy scipy pyyaml \
      statsmodels hf_transfer
  touch .deps_ok
fi

python -c "import torch,vllm; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),
'gpus',torch.cuda.device_count()); print('vllm',vllm.__version__)"

if [ ! -f data/prompts.jsonl ]; then
  PYTHONPATH=src python -m bre.data
fi
python - <<'PY'
import json
rows=[json.loads(l) for l in open('data/prompts.jsonl')]
from collections import Counter
print('prompt counts by split:', dict(Counter(r['split'] for r in rows)))
PY
echo "bootstrap ok"

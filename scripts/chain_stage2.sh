#!/usr/bin/env bash
# Wait for the judge phase, then run stage 2 so the pod is not billing idle between them.
# Stage 2 is the refusal-maximising attacker search, which was part of the original plan.
set -uo pipefail
cd /workspace/bre
LOG=logs/judge_full.log

echo "[chain] waiting for judge phase to finish"
while true; do
  if grep -q "JUDGE PHASE COMPLETE" "$LOG" 2>/dev/null; then
    echo "[chain] judge complete, starting stage 2"; break
  fi
  if ! pgrep -f run_judge_full >/dev/null; then
    sleep 20
    if ! pgrep -f run_judge_full >/dev/null \
       && ! grep -q "JUDGE PHASE COMPLETE" "$LOG" 2>/dev/null; then
      echo "[chain] judge phase exited without completing; NOT starting stage 2"
      exit 1
    fi
  fi
  sleep 60
done

# Let the judge's GPU memory actually release before claiming it.
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 10
done
echo "[chain] GPUs free, launching stage 2"
exec bash scripts/run_stage2.sh

#!/usr/bin/env bash
# "GPUs must remain hot or be destroyed."
#
# Waits for stage 2 to finish (or die), pulls everything, verifies the pull, then
# terminates the pod. Terminating destroys the volume, so the verification gate is not
# optional: if the sync looks wrong, this refuses to terminate and says so.
set -uo pipefail
cd "$(dirname "$0")/.."
POD_ID="${POD_ID:-dy2nkznmlsxfa2}"

probe () {
  ssh -o ConnectTimeout=25 -o BatchMode=yes bre-pod '
    if grep -q "stage 2 complete" /workspace/bre/logs/stage2_*.log \
         /workspace/bre/logs/chain_stage2.log 2>/dev/null; then echo DONE
    elif pgrep -f "run_stage2|stage2_attack|run_judge_full|chain_stage2" >/dev/null; then echo BUSY
    else echo IDLE; fi' 2>/dev/null
}

misses=0
while true; do
  case "$(probe)" in
    DONE) echo "[teardown] stage 2 reported complete"; break ;;
    BUSY) misses=0 ;;
    IDLE) misses=$((misses+1))
          echo "[teardown] nothing running ($misses/3)"
          [ "$misses" -ge 3 ] && { echo "[teardown] GPUs cold"; break; } ;;
    *)    echo "[teardown] ssh unreachable, retrying" ;;
  esac
  sleep 120
done

echo "[teardown] syncing results before terminating"
bash scripts/sync_results.sh

# Verification gate. Terminating destroys the volume; refuse if the pull looks wrong.
ok=1
for f in results/probe.jsonl results/generations.jsonl; do
  [ -s "$f" ] || { echo "[teardown] MISSING/EMPTY $f"; ok=0; }
done
lines=$(wc -l < results/probe.jsonl 2>/dev/null || echo 0)
[ "$lines" -ge 34200 ] || { echo "[teardown] probe.jsonl only $lines rows, expected >=34200"; ok=0; }

if [ "$ok" -ne 1 ]; then
  echo "[teardown] REFUSING to terminate: results not safely local. Pod left running."
  exit 1
fi
echo "[teardown] results verified local ($lines probe rows)"
python3 scripts/rp.py terminate "$POD_ID" && echo "[teardown] pod $POD_ID terminated"
python3 scripts/rp.py balance

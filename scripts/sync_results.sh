#!/usr/bin/env bash
# Pull everything off the pod. Run before any teardown: terminating destroys the volume.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results logs
for f in probe.jsonl generations.jsonl judged.jsonl judged.meta.json \
         anchor_compliance.jsonl compliance_agreement.json summary.json \
         summary_probe_only.json judge_validation.json judge_precision_check.json \
         judge_crosscheck.json smoke.jsonl; do
  scp -q "bre-pod:/workspace/bre/results/$f" results/ 2>/dev/null && echo "  pulled $f"
done
# Attacker output is a directory tree.
rm -rf results/attack.tmp
if ssh -o ConnectTimeout=20 bre-pod 'test -d /workspace/bre/results/attack' 2>/dev/null; then
  scp -qr bre-pod:/workspace/bre/results/attack results/attack.tmp 2>/dev/null \
    && rm -rf results/attack && mv results/attack.tmp results/attack \
    && echo "  pulled attack/"
fi
scp -qr bre-pod:/workspace/bre/logs logs/pod 2>/dev/null && echo "  pulled logs/"
echo "local results:"
du -sh results 2>/dev/null
ls -la results/ | awk '{print $5, $9}'

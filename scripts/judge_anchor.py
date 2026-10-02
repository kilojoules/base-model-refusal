#!/usr/bin/env python3
"""Comparability anchor: score compliance with the original HarmBench classifier.

The 13B fine-tuned classifier is what published attack-success numbers are measured with,
so running it alongside keeps this study's compliance figures comparable to the
literature. It is NOT used as the refusal judge: its negative class pools refusal with
incoherence, which for base models is most of it.

Writes a separate file keyed the same way as judged.jsonl so the two can be joined.
"""
from __future__ import annotations

import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, judge

ANCHOR = "cais/HarmBench-Llama-2-13b-cls"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--generations", default="results/generations.jsonl")
    ap.add_argument("--out", default="results/anchor_compliance.jsonl")
    ap.add_argument("--model", default=ANCHOR)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    yaml.safe_load(open(args.config))
    behaviors = data.read("data/prompts.jsonl")
    rows = [json.loads(l) for l in open(args.generations) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[anchor] {len(rows)} completions through {args.model}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.model)
    # Llama-2 derivative: 2048 positions, and vLLM refuses to exceed it.
    llm = LLM(model=args.model, tensor_parallel_size=1, dtype="bfloat16",
              max_model_len=2048, gpu_memory_utilization=0.90)
    t0 = time.time()
    # chat_model=False: this model expects the original Llama-2-wrapped rubric verbatim.
    verdicts, n_bad = judge.judge_compliance(llm, tok, rows, behaviors, chat_model=False)
    yes = verdicts.count("yes")
    print(f"[anchor] done in {time.time()-t0:.0f}s: yes={yes} "
          f"({yes/max(1,len(rows)):.1%}) unparseable={n_bad}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r, v in zip(rows, verdicts):
            fh.write(json.dumps({
                "behavior_id": r["behavior_id"], "arm": r["arm"], "model": r["model"],
                "sample": r["sample"], "split": r["split"],
                "anchor_compliance": v,
            }) + "\n")
    print(f"[anchor] wrote {len(rows)} rows -> {out}", flush=True)


if __name__ == "__main__":
    main()

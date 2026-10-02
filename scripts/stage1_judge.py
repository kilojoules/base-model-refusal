#!/usr/bin/env python3
"""Judge phase: compliance (official HarmBench classifier) then refusal taxonomy.

Run after stage1_probe.py --with-generations. Loads one judge at a time.
"""
from __future__ import annotations

import argparse, gc, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--generations", default="results/generations.jsonl")
    ap.add_argument("--out", default="results/judged.jsonl")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    behaviors = data.read("data/prompts.jsonl")
    rows = [json.loads(l) for l in open(args.generations) if l.strip()]
    print(f"[judge] {len(rows)} completions", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM
    import torch

    # Pass 1: compliance, official HarmBench classifier and official rubric.
    cspec = cfg["judge"]["compliance"]
    tok = AutoTokenizer.from_pretrained(cspec["model"])
    llm = LLM(model=cspec["model"], tensor_parallel_size=cspec["tp"],
              dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90)
    compliance = judge.judge_compliance(llm, tok, rows, behaviors)
    del llm; gc.collect(); torch.cuda.empty_cache()
    print(f"[judge] compliance done; yes={compliance.count('yes')}", flush=True)

    # Pass 2: refusal taxonomy, separate model, six-way label.
    tspec = cfg["judge"]["taxonomy"]
    tok2 = AutoTokenizer.from_pretrained(tspec["model"])
    llm2 = LLM(model=tspec["model"], tensor_parallel_size=tspec["tp"],
               max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
               trust_remote_code=True)
    taxonomy = judge.judge_taxonomy(llm2, tok2, rows, behaviors)
    del llm2; gc.collect(); torch.cuda.empty_cache()
    from collections import Counter
    print("[judge] taxonomy mix:", dict(Counter(taxonomy)), flush=True)

    n = judge.write(rows, compliance, taxonomy, args.out)
    print(f"[judge] wrote {n} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

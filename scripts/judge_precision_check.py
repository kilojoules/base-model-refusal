#!/usr/bin/env python3
"""Measure what INT4 quantization costs the judge.

The compliance pass runs on an INT4 Llama-3.1-70B because bf16 70B leaves only ~11GB of
KV cache on 2x80GB, capping concurrency near 16 sequences. That is a throughput decision,
and it needs a number attached rather than an assurance.

Re-judges a random subsample with the bf16 checkpoint and reports agreement with the INT4
verdicts. bf16 is loaded with a small max_num_seqs because of the cache limit, so keep the
subsample small.
"""
from __future__ import annotations

import argparse, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, judge

BF16 = "meta-llama/Llama-3.1-70B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--judged", default="results/judged.jsonl")
    ap.add_argument("--generations", default="results/generations.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bf16-model", default=BF16)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--out", default="results/judge_precision_check.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    behaviors = data.read("data/prompts.jsonl")
    gens = [json.loads(l) for l in open(args.generations) if l.strip()]
    judged = [json.loads(l) for l in open(args.judged) if l.strip()]

    # Align on (behavior_id, arm, model, sample); the judged file drops completion text.
    key = lambda r: (r["behavior_id"], r["arm"], r["model"], r["sample"])
    int4 = {key(r): r.get("compliance") for r in judged}
    pool = [g for g in gens if key(g) in int4]
    random.Random(cfg["seed"]).shuffle(pool)
    sub = pool[: args.n]
    print(f"[precision] re-judging {len(sub)} rows at bf16", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.bf16_model)
    llm = LLM(model=args.bf16_model, tensor_parallel_size=2, dtype="bfloat16",
              max_model_len=cfg["judge"]["compliance"].get("max_model_len", 4096),
              gpu_memory_utilization=0.95, max_num_seqs=args.max_num_seqs,
              trust_remote_code=True)
    ref, n_bad = judge.judge_compliance(llm, tok, sub, behaviors, chat_model=True)

    both = [(int4[key(g)], r) for g, r in zip(sub, ref)
            if int4[key(g)] is not None and r is not None]
    agree = sum(a == b for a, b in both)
    i_yes = sum(a == "yes" for a, _ in both)
    b_yes = sum(b == "yes" for _, b in both)
    res = {
        "n_compared": len(both), "agreement": agree / max(1, len(both)),
        "int4_yes_rate": i_yes / max(1, len(both)),
        "bf16_yes_rate": b_yes / max(1, len(both)),
        "bf16_unparseable": n_bad,
        "int4_model": cfg["judge"]["compliance"]["model"], "bf16_model": args.bf16_model,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nAgreement above ~0.95 with similar yes-rates means the INT4 judge is not")
    print("introducing a bias that matters at the resolution of these results.")


if __name__ == "__main__":
    main()

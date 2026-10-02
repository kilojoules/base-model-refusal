#!/usr/bin/env python3
"""Cross-check the refusal taxonomy with a judge from a different model family.

Both stage-1 passes share one Llama judge, which is fast but means the compliance and
stance axes are not independent. A disagreement between families on the stance labels is
the thing that would undermine the taxonomy conclusions, so measure it on a subsample
with a Qwen judge.
"""
from __future__ import annotations

import argparse, json, pathlib, random, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, judge

ALT = "Qwen/Qwen2.5-32B-Instruct-AWQ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--judged", default="results/judged.jsonl")
    ap.add_argument("--generations", default="results/generations.jsonl")
    ap.add_argument("--alt-model", default=ALT)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="results/judge_crosscheck.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    behaviors = data.read("data/prompts.jsonl")
    gens = [json.loads(l) for l in open(args.generations) if l.strip()]
    judged = [json.loads(l) for l in open(args.judged) if l.strip()]
    key = lambda r: (r["behavior_id"], r["arm"], r["model"], r["sample"])
    primary = {key(r): r.get("taxonomy") for r in judged}
    pool = [g for g in gens if key(g) in primary]
    random.Random(cfg["seed"]).shuffle(pool)
    sub = pool[: args.n]
    print(f"[crosscheck] re-labelling {len(sub)} rows with {args.alt_model}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(args.alt_model)
    llm = LLM(model=args.alt_model, tensor_parallel_size=2, max_model_len=4096,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    alt = judge.judge_taxonomy(llm, tok, sub, behaviors)

    pairs = [(primary[key(g)], a) for g, a in zip(sub, alt)
             if primary[key(g)] and a]
    exact = sum(a == b for a, b in pairs)
    # The binary question the results actually rest on.
    refus = lambda l: l in ("EXPLICIT_REFUSAL", "SOFT_DEFLECTION")
    bin_agree = sum(refus(a) == refus(b) for a, b in pairs)
    res = {
        "n_compared": len(pairs),
        "exact_label_agreement": exact / max(1, len(pairs)),
        "is_refusal_agreement": bin_agree / max(1, len(pairs)),
        "primary_mix": {k: v for k, v in Counter(a for a, _ in pairs).items()},
        "alt_mix": {k: v for k, v in Counter(b for _, b in pairs).items()},
        "alt_model": args.alt_model,
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nExact six-way agreement will be lower than binary refusal agreement; the")
    print("binary number is the one the conclusions depend on.")


if __name__ == "__main__":
    main()

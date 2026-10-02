#!/usr/bin/env python3
"""Judge phase: compliance (official HarmBench rubric) and refusal taxonomy.

Both passes run on one 70B instruct judge. The compliance pass uses the official
HarmBench rubric verbatim, with only its Llama-2 chat wrapper swapped for this model's
template. The taxonomy pass uses the six-way stance prompt.

The two axes stay separate in the output. Refusal is never derived by negating
compliance: for a base model the non-compliant class is mostly incoherence.
"""
from __future__ import annotations

import argparse, gc, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--generations", default="results/generations.jsonl")
    ap.add_argument("--out", default="results/judged.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N completions (for a quick check)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cspec, tspec = cfg["judge"]["compliance"], cfg["judge"]["taxonomy"]
    behaviors = data.read("data/prompts.jsonl")
    rows = [json.loads(l) for l in open(args.generations) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[judge] {len(rows)} completions", flush=True)
    print(f"[judge] judge model: {cspec.get('display_name', cspec['model'])}", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM
    import torch

    shared = cspec["model"] == tspec["model"]
    tok = AutoTokenizer.from_pretrained(cspec["model"])
    t0 = time.time()
    llm = LLM(model=cspec["model"], tensor_parallel_size=cspec["tp"],
              max_model_len=cspec.get("max_model_len", args.max_model_len),
              gpu_memory_utilization=0.92, trust_remote_code=True)
    print(f"[judge] engine up in {time.time()-t0:.0f}s"
          f"{' (shared for both passes)' if shared else ''}", flush=True)

    t0 = time.time()
    compliance, n_bad = judge.judge_compliance(
        llm, tok, rows, behaviors, chat_model=cspec.get("chat_model", False))
    yes = compliance.count("yes")
    print(f"[judge] compliance done in {time.time()-t0:.0f}s: "
          f"yes={yes} ({yes/max(1,len(rows)):.1%}) unparseable={n_bad}", flush=True)
    if n_bad > 0.02 * len(rows):
        print(f"[judge] WARNING: {n_bad} unparseable verdicts "
              f"({n_bad/len(rows):.1%}); treat the compliance column with care",
              flush=True)

    if not shared:
        del llm; gc.collect(); torch.cuda.empty_cache()
        tok = AutoTokenizer.from_pretrained(tspec["model"])
        llm = LLM(model=tspec["model"], tensor_parallel_size=tspec["tp"],
                  max_model_len=tspec.get("max_model_len", args.max_model_len),
                  gpu_memory_utilization=0.92, trust_remote_code=True)

    t0 = time.time()
    taxonomy = judge.judge_taxonomy(llm, tok, rows, behaviors)
    from collections import Counter
    mix = Counter(taxonomy)
    print(f"[judge] taxonomy done in {time.time()-t0:.0f}s", flush=True)
    for k, v in mix.most_common():
        print(f"    {str(k):<24}{v:>7} ({v/max(1,len(rows)):.1%})", flush=True)

    del llm; gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass

    n = judge.write(rows, compliance, taxonomy, args.out)
    meta = {"judge_model": cspec["model"],
            "judge_display": cspec.get("display_name", cspec["model"]),
            "n_rows": n, "compliance_yes": yes, "compliance_unparseable": n_bad,
            "taxonomy_mix": {str(k): v for k, v in mix.items()}}
    with open(pathlib.Path(args.out).with_suffix(".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[judge] wrote {n} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

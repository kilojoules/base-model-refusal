#!/usr/bin/env python3
"""Sensitivity check: does the leading-space tokenization artefact move the result?

The refusal and compliant openers are written with a leading space. That is natural
after "Assistant:" but not after a blank line, so the `document` arm pays a
tokenization penalty the `chat` arm does not. Both opener sets pay it equally, and a
constant per-arm offset cancels in a difference of differences across models -- but
"largely cancels" is an assumption, so measure it.

Re-probes two models with and without the whitespace fix and reports how far the
per-arm levels and the chat-minus-document gap move.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, formats, probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-7B-Instruct"])
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--out", default="results/sensitivity_ws.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    beh = data.read("data/prompts.jsonl")
    harmful = [b for b in beh if b["split"] == "harmful"][: args.limit]

    from transformers import AutoTokenizer
    from vllm import LLM
    import torch, gc

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        llm = LLM(model=name, tensor_parallel_size=1, dtype="bfloat16",
                  max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
                  trust_remote_code=True)
        for ws in (False, True):
            recs = [{"behavior_id": b["id"], "arm": arm, "split": "harmful",
                     "category": b["category"], "pair_type": None,
                     "prompt": formats.render(b, arm, tokenizer=tok)}
                    for b in harmful for arm in cfg["arms"]]
            probe.run(llm, tok, recs, out, batch_size=cfg["probe"]["batch_size"],
                      max_len=args.max_model_len, ws_aware=ws,
                      meta={"model": name, "ws_aware": ws, "params_b": 0.0})
        del llm; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

    import statistics as st
    from collections import defaultdict
    rows = [json.loads(l) for l in open(out) if l.strip()]
    g = defaultdict(list)
    for r in rows:
        if r["rps"] == r["rps"]:
            g[(r["model"], r["ws_aware"], r["arm"])].append(r["rps"])
    print(f"\n{'model':<24}{'arm':<12}{'ws=off':>9}{'ws=on':>9}{'shift':>9}")
    for model in sorted({k[0] for k in g}):
        for arm in cfg["arms"]:
            a = g.get((model, False, arm)); b = g.get((model, True, arm))
            if a and b:
                print(f"{model.split('/')[-1]:<24}{arm:<12}{st.mean(a):>9.3f}"
                      f"{st.mean(b):>9.3f}{st.mean(b)-st.mean(a):>9.3f}")
    print(f"\n{'model':<24}{'chat-minus-document gap':<26}{'ws=off':>9}{'ws=on':>9}{'shift':>9}")
    for model in sorted({k[0] for k in g}):
        vals = {}
        for ws in (False, True):
            c = g.get((model, ws, "chat")); d = g.get((model, ws, "document"))
            vals[ws] = st.mean(c) - st.mean(d) if c and d else float("nan")
        print(f"{model.split('/')[-1]:<24}{'':<26}{vals[False]:>9.3f}{vals[True]:>9.3f}"
              f"{vals[True]-vals[False]:>9.3f}")
    print("\nIf the gap shift is small relative to the between-model spread in the gap,")
    print("the artefact does not affect the conclusion and the ladder stands as run.")


if __name__ == "__main__":
    main()

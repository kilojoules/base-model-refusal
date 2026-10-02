#!/usr/bin/env python3
"""Stage 2a: search for refusal-maximising wrappers against one target base model.

Only the target needs a GPU. Both the proposer and the request-preservation check see
nothing but content-free templates, so they can run against any OpenAI-compatible
endpoint -- a local vLLM server or a hosted API -- and no benchmark content leaves the
machine holding the target.

Run the strict mode first; it is the scientifically interesting one. Loose mode is an
upper bound only.
"""
from __future__ import annotations

import argparse, json, os, pathlib, sys, urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data
from bre.attacker import reward as R, search, seeds


class ApiRequestCheck:
    """Request-preservation guard via an OpenAI-compatible endpoint."""

    def __init__(self, base_url, model, api_key):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key

    def __call__(self, wrappers):
        out = []
        for w in wrappers:
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "user",
                              "content": R.REQUEST_CHECK_PROMPT.format(wrapper=w)}],
                "temperature": 0.0, "max_tokens": 4,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as fh:
                    txt = json.load(fh)["choices"][0]["message"]["content"]
                out.append(txt.strip().lower().startswith("yes"))
            except Exception:
                out.append(True)  # fail open on transport errors; static guards still apply
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--target", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mode", choices=["strict", "loose"], default="strict")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--runs", type=int, default=4,
                    help="independent searches, each from a different genre seed set")
    ap.add_argument("--proposer-url", default=os.environ.get(
        "PROPOSER_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--proposer-model", default=os.environ.get(
        "PROPOSER_MODEL", "Qwen/Qwen2.5-14B-Instruct"))
    ap.add_argument("--proposer-key", default=os.environ.get("PROPOSER_API_KEY", "none"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    acfg = cfg["attacker"]
    target = args.target or acfg["target"]
    rounds = args.rounds or acfg["rounds"]
    outdir = args.outdir or f"results/attack/{target.split('/')[-1]}_{args.mode}"

    behaviors = data.read("data/prompts.jsonl")

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
    llm = LLM(model=target, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.90,
              trust_remote_code=True, enforce_eager=args.eager)

    proposer = search.Proposer(args.proposer_url, args.proposer_model, args.proposer_key,
                                mode=args.mode)
    checker = ApiRequestCheck(args.proposer_url, args.proposer_model, args.proposer_key)

    # Adapt the API checker into the signature search.run expects.
    class _JudgeShim:
        pass
    shim = _JudgeShim()
    R.check_requests_preserved = staticmethod(  # type: ignore[assignment]
        lambda jl, jt, ws: checker(ws))

    # Independent runs from different genre seed sets. One hill-climb can plateau in a
    # local optimum; the reported ceiling is the max over runs on held-out behaviours.
    seed_sets = seeds.seed_sets(args.runs, seed=cfg["seed"])
    all_runs = []
    for i, sset in enumerate(seed_sets):
        search.SEED_WRAPPERS = sset
        rd = f"{outdir}/run{i}"
        print(f"\n[attack] === run {i+1}/{args.runs}: {len(sset)} seeds ===", flush=True)
        final = search.run(
            llm, tok, behaviors, proposer, outdir=rd, rounds=rounds,
            k=acfg["proposals_per_round"], board_size=acfg["leaderboard_size"],
            search_n=acfg["search_behaviors"], heldout_n=acfg["heldout_behaviors"],
            max_chars=acfg["max_wrapper_chars"], mode=args.mode,
            seed=cfg["seed"] + i, judge_llm=shim, judge_tok=None,
            max_len=args.max_model_len,
        )
        for e in final:
            e["run"] = i
        all_runs.extend(final)
        if final:
            print(f"[attack] run {i} best heldout={final[0]['heldout_reward']:+.4f}",
                  flush=True)

    all_runs.sort(key=lambda e: -e["heldout_reward"])
    summary = {
        "target": target, "mode": args.mode, "runs": args.runs,
        "per_run_best": [
            {"run": i,
             "heldout_reward": max((e["heldout_reward"] for e in all_runs if e["run"] == i),
                                   default=None)}
            for i in range(args.runs)
        ],
        "ceiling_heldout": all_runs[0]["heldout_reward"] if all_runs else None,
        "board": all_runs[:20],
    }
    with open(f"{outdir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n[attack] target={target} mode={args.mode} runs={args.runs}")
    print(f"[attack] per-run best held-out: "
          + ", ".join(f"{r['run']}:{r['heldout_reward']:+.4f}"
                      if r["heldout_reward"] is not None else f"{r['run']}:n/a"
                      for r in summary["per_run_best"]))
    print(f"[attack] CEILING (max over runs, held-out) = {summary['ceiling_heldout']}")
    print("[attack] top wrappers:")
    for e in all_runs[:5]:
        w = e["wrapper"].replace("\n", "\\n")[:90]
        print(f"  run={e['run']} search={e['reward']:+.4f} "
              f"heldout={e['heldout_reward']:+.4f} "
              f"frac_pos={e.get('heldout_frac_positive',0):.2f}  {w!r}")
    print(f"[attack] wrote {outdir}/summary.json")


if __name__ == "__main__":
    main()

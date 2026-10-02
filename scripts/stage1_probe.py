#!/usr/bin/env python3
"""Stage 1: probe (and optionally sample) every model on the ladder.

One vLLM engine per model, loaded once, used for all arms and both splits, then torn
down before the next model. Resumable: a model already present in the output file is
skipped.
"""
from __future__ import annotations

import argparse, gc, json, os, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import yaml
from bre import data, formats, probe, generate, fewshot


def done_models(path):
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path) as fh:
        for l in fh:
            if l.strip():
                try: seen.add(json.loads(l)["model"])
                except Exception: pass
    return seen


def build_records(behaviors, arms, tokenizer):
    recs = []
    for b in behaviors:
        for arm in arms:
            recs.append({
                "behavior_id": b["id"], "arm": arm, "split": b["split"],
                "category": b["category"],
                "pair_type": b.get("pair_type"),
                "prompt": formats.render(b, arm, tokenizer=tokenizer),
            })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--probe-out", default="results/probe.jsonl")
    ap.add_argument("--gen-out", default="results/generations.jsonl")
    ap.add_argument("--with-generations", action="store_true")
    ap.add_argument("--with-fewshot", action="store_true",
                    help="also run the few-shot elicitation dose-response curve")
    ap.add_argument("--with-compliance-demos", action="store_true",
                    help="run the compliance_demos steering condition")
    ap.add_argument("--only-compliance-demos", action="store_true",
                    help="run ONLY the compliance_demos condition (skip base arms)")
    ap.add_argument("--fewshot-targets", type=int, default=120)
    ap.add_argument("--only", default=None, help="substring filter on model name")
    ap.add_argument("--include-anchors", action="store_true", default=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--purge-cache", action="store_true",
                    help="delete each model's weights after probing it; the run is\n                          resumable so weights are never needed twice")
    ap.add_argument("--eager", action="store_true",
                    help="disable CUDA graph capture (slower, more robust)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    behaviors = data.read("data/prompts.jsonl")
    harmful = [b for b in behaviors if b["split"] == "harmful"]
    print(f"[data] {len(harmful)} harmful, {len(behaviors)-len(harmful)} benign",
          flush=True)

    models = list(cfg["ladder"]["models"])
    if args.include_anchors:
        models += list(cfg["ladder"].get("anchors", []))
    if args.only:
        models = [m for m in models if args.only in m["name"]]

    already = done_models(args.probe_out)
    from transformers import AutoTokenizer
    from vllm import LLM
    import torch

    for spec in models:
        name = spec["name"]
        if name in already:
            print(f"[skip] {name} already in {args.probe_out}", flush=True)
            continue
        t0 = time.time()
        print(f"[load] {name} tp={spec['tp']}", flush=True)
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        llm = LLM(
            model=name, tensor_parallel_size=spec["tp"],
            dtype="bfloat16", max_model_len=args.max_model_len,
            gpu_memory_utilization=0.90, trust_remote_code=True,
            enforce_eager=args.eager,
        )
        meta = {
            "model": name, "params_b": spec["params_b"],
            "kind": spec.get("kind", "base"),
            "native_chat_template": formats.has_native_template(tok),
        }
        if args.only_compliance_demos:
            recs = fewshot.build_compliance_records(
                behaviors, n_targets=args.fewshot_targets, seed=cfg["seed"])
        else:
            recs = build_records(behaviors, cfg["arms"], tok)
            if args.with_fewshot:
                recs += fewshot.build_records(
                    behaviors, n_targets=args.fewshot_targets, seed=cfg["seed"])
            if args.with_compliance_demos:
                recs += fewshot.build_compliance_records(
                    behaviors, n_targets=args.fewshot_targets, seed=cfg["seed"])
        n = probe.run(llm, tok, recs, args.probe_out,
                      batch_size=cfg["probe"]["batch_size"],
                      max_len=args.max_model_len, meta=meta)
        print(f"[probe] {name}: {n} prompts in {time.time()-t0:.0f}s", flush=True)

        if args.with_generations:
            gcfg = cfg["generate"]
            k = gcfg["subsample_per_arm"]
            sub = []
            for arm in cfg["arms"]:
                arm_recs = [r for r in recs if r["arm"] == arm]
                hs = [r for r in arm_recs if r["split"] == "harmful"][:k]
                bs = [r for r in arm_recs if r["split"] == "benign"][: k // 2]
                ms = [r for r in arm_recs if r["split"] == "harmful_matched"][: k // 2]
                sub += hs + bs + ms
            m = generate.run(llm, sub, args.gen_out,
                             n_samples=gcfg["n_samples"],
                             temperature=gcfg["temperature"], top_p=gcfg["top_p"],
                             max_tokens=gcfg["max_tokens"], seed=cfg["seed"],
                             meta=meta)
            print(f"[gen] {name}: {m} completions", flush=True)

        del llm; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

        if args.purge_cache:
            # Removing the models--* directory is not enough on its own: newer
            # huggingface_hub versions keep weights in a shared, sharded blob store and
            # the model directory holds only links into it. Drop the directory, then reap
            # the blobs it was the last referrer to. Nothing is downloading at this point
            # in the loop, so no age guard is needed here.
            import shutil
            import subprocess
            slug = "models--" + name.replace("/", "--")
            cache = pathlib.Path(os.environ.get("HF_HOME", "~/.cache/huggingface"))
            for root in (cache / "hub", cache):
                d = root / slug
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[purge] removed {d}", flush=True)
                    break
            reaper = pathlib.Path(__file__).with_name("reap_orphan_blobs.py")
            if reaper.exists():
                try:
                    out = subprocess.run(
                        [sys.executable, str(reaper), "--min-age-min", "0"],
                        capture_output=True, text=True, timeout=600)
                    for line in out.stdout.strip().splitlines():
                        print(line, flush=True)
                except Exception as e:
                    print(f"[purge] reaper failed: {e}", flush=True)
        try:
            import shutil as _sh
            free = _sh.disk_usage("/workspace").free / 1e9
            print(f"[disk] {free:.0f}GB free", flush=True)
        except Exception:
            pass
        print(f"[done] {name} total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

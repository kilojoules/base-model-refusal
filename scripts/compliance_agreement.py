#!/usr/bin/env python3
"""Agreement between the Llama-70B judge and the official HarmBench classifier.

This is the compliance axis's validation. The synthetic validation set cannot do the job:
the HarmBench rubric asks whether a generation is an instance of a *harmful* behaviour, so
a benign answer to a benign prompt is definitionally non-compliant, and building genuine
positives would mean authoring harmful completions.

Agreement on real harmful generations between two independent judges -- one a general 70B
instruct model, one the 13B model the published numbers come from -- is obtainable without
anyone writing harmful content, and is the right evidence here.
"""
from __future__ import annotations

import argparse, json, pathlib
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="results/judged.jsonl")
    ap.add_argument("--anchor", default="results/anchor_compliance.jsonl")
    ap.add_argument("--out", default="results/compliance_agreement.json")
    args = ap.parse_args()

    key = lambda r: (r["behavior_id"], r["arm"], r["model"], r["sample"])
    primary = {key(json.loads(l)): json.loads(l).get("compliance")
               for l in open(args.judged) if l.strip()}
    anchor = {key(json.loads(l)): json.loads(l).get("anchor_compliance")
              for l in open(args.anchor) if l.strip()}

    shared = [k for k in primary if k in anchor
              and primary[k] is not None and anchor[k] is not None]
    agree = sum(primary[k] == anchor[k] for k in shared)
    # Restricted to HarmBench prompts, where the rubric is actually in scope.
    harm = [k for k in shared if not k[0].startswith("xstest_")]
    agree_h = sum(primary[k] == anchor[k] for k in harm)

    cm = Counter((primary[k], anchor[k]) for k in harm)
    res = {
        "n_compared_all": len(shared),
        "agreement_all": agree / max(1, len(shared)),
        "n_compared_harmbench": len(harm),
        "agreement_harmbench": agree_h / max(1, len(harm)),
        "llama_yes_rate_harmbench": sum(primary[k] == "yes" for k in harm) / max(1, len(harm)),
        "anchor_yes_rate_harmbench": sum(anchor[k] == "yes" for k in harm) / max(1, len(harm)),
        "confusion_harmbench": {f"llama={a},anchor={b}": n for (a, b), n in cm.items()},
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\nBoth yes-rates should be low for base models and similar to each other.")
    print("A large gap means the two judges disagree about what counts as compliance,")
    print("and the compliance column should then be reported with that caveat attached.")


if __name__ == "__main__":
    main()

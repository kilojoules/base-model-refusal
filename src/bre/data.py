"""Prompt-set construction.

Harmful behaviours come from HarmBench; the benign control comes from the safe split
of XSTest. Behaviour text is never printed to stdout by this module -- it is written to
disk and referenced by id everywhere else.
"""
from __future__ import annotations

import csv
import io
import json
import pathlib
import urllib.request

HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
    "data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


def load_harmful(functional_categories=("standard",), limit=None):
    """HarmBench behaviours as {id, text, category, context}."""
    with urllib.request.urlopen(HARMBENCH_CSV) as fh:
        raw = fh.read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    out = []
    for r in rows:
        if functional_categories and r["FunctionalCategory"] not in functional_categories:
            continue
        out.append(
            {
                "id": r["BehaviorID"],
                "text": r["Behavior"].strip(),
                "category": r["SemanticCategory"],
                "context": (r.get("ContextString") or "").strip() or None,
                "split": "harmful",
            }
        )
    out.sort(key=lambda d: d["id"])
    return out[:limit] if limit else out


def load_benign(limit=250):
    """XSTest safe split: benign prompts with superficially alarming surface form.

    This is the false-refusal floor. Without it, a non-answer rate cannot be
    attributed to harm-sensitivity.
    """
    from datasets import load_dataset

    ds = load_dataset("natolambert/xstest-v2-copy", "default")
    frames = []
    for name in ds:
        frames.extend(dict(r, _split=name) for r in ds[name])
    safe = [r for r in frames if "contrast" not in str(r.get("type", "")).lower()]
    # XSTest marks genuinely-unsafe contrast prompts; keep only the safe ones.
    safe = [r for r in safe if str(r.get("label", "safe")).lower() in ("safe", "")]
    out = []
    for i, r in enumerate(safe):
        text = (r.get("prompt") or r.get("question") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": f"xstest_safe_{i:04d}",
                "text": text,
                "category": str(r.get("type", "unknown")),
                "context": None,
                "split": "benign",
            }
        )
    out.sort(key=lambda d: d["id"])
    return out[:limit] if limit else out


def build(outdir="data", harmful_kw=None, benign_limit=250):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    harmful = load_harmful(**(harmful_kw or {}))
    benign = load_benign(limit=benign_limit)
    path = outdir / "prompts.jsonl"
    with path.open("w") as fh:
        for rec in harmful + benign:
            fh.write(json.dumps(rec) + "\n")
    # Counts only -- deliberately no content echoed.
    return {"harmful": len(harmful), "benign": len(benign), "path": str(path)}


def read(path="data/prompts.jsonl"):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))

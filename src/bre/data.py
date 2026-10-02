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


def load_xstest(limit_safe=250, limit_contrast=200):
    """XSTest, split into its safe prompts and their unsafe contrast twins.

    XSTest is built as matched pairs: every `contrast_X` type is a genuinely unsafe
    prompt whose surface form mirrors the safe `X` prompts. That matching is why this
    set matters here.

    HarmBench behaviours are imperatives ("write a tutorial that ..."), while XSTest
    safe prompts are mostly questions. Comparing those two directly conflates harm with
    grammatical form, so a HarmBench-minus-XSTest-safe difference is not a clean
    harm-sensitivity estimate. The within-XSTest contrast-minus-safe difference is,
    because form is held approximately fixed by construction.

    Returns (safe, contrast). `pair_type` lets the two be matched type by type.
    """
    from datasets import load_dataset

    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    safe, contrast = [], []
    for r in ds:
        typ = str(r["type"])
        text = (r.get("prompt") or "").strip()
        if not text:
            continue
        is_contrast = typ.startswith("contrast_") or typ == "contrast_discr"
        rec = {
            "id": f"xstest_{typ}_{r['id']}",
            "text": text,
            "category": typ,
            "pair_type": typ.replace("contrast_", "", 1) if is_contrast else typ,
            "context": None,
            "split": "harmful_matched" if is_contrast else "benign",
        }
        (contrast if is_contrast else safe).append(rec)
    safe.sort(key=lambda d: d["id"])
    contrast.sort(key=lambda d: d["id"])
    return safe[:limit_safe], contrast[:limit_contrast]


def load_benign(limit=250):
    """Backwards-compatible alias for the safe split alone."""
    return load_xstest(limit_safe=limit)[0]


def build(outdir="data", harmful_kw=None, benign_limit=250, contrast_limit=200):
    """Write all three prompt sets.

    harmful         - HarmBench standard behaviours; severity-matched to the literature.
    benign          - XSTest safe; the false-refusal floor.
    harmful_matched - XSTest unsafe contrast twins; form-matched to `benign`, which
                      makes contrast-minus-benign the clean harm-sensitivity estimate.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    harmful = load_harmful(**(harmful_kw or {}))
    benign, contrast = load_xstest(benign_limit, contrast_limit)
    path = outdir / "prompts.jsonl"
    with path.open("w") as fh:
        for rec in harmful + benign + contrast:
            fh.write(json.dumps(rec) + "\n")
    # Counts only -- deliberately no content echoed.
    return {"harmful": len(harmful), "benign": len(benign),
            "harmful_matched": len(contrast), "path": str(path)}


def read(path="data/prompts.jsonl"):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))

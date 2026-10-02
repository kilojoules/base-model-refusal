"""Secondary instrument: sample continuations, trim, persist.

Secondary because it is noisy, costly, and confounded by generation competence. Its job
is to confirm that the RPS probe predicts text a human would call refusal -- not to
carry the scaling claim.
"""
from __future__ import annotations

import json
import pathlib

from . import truncate


def run(llm, records, outpath, n_samples=5, temperature=0.9, top_p=0.95,
        max_tokens=256, seed=0, meta=None):
    from vllm import SamplingParams

    outpath = pathlib.Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    recs = list(records)
    params = SamplingParams(
        n=n_samples, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, seed=seed,
    )
    outs = llm.generate([r["prompt"] for r in recs], params)
    written = 0
    with outpath.open("a") as fh:
        for rec, out in zip(recs, outs):
            for k, comp in enumerate(out.outputs):
                t = truncate.trim(comp.text)
                fh.write(json.dumps({
                    "behavior_id": rec["behavior_id"],
                    "arm": rec["arm"],
                    "split": rec["split"],
                    "category": rec.get("category"),
                    "sample": k,
                    "completion": t["text"],
                    "raw_len": len(comp.text),
                    "trimmed_chars": t["trimmed_chars"],
                    "trim_reason": t["reason"],
                    "degenerate": t["degenerate"],
                    "finish_reason": comp.finish_reason,
                    **(meta or {}),
                }) + "\n")
                written += 1
    return written

#!/usr/bin/env python3
"""Final report: the contrasts, not the levels.

Every table here is a difference. Raw RPS carries a large positive string-frequency
offset (see DESIGN.md section 3), so a level on its own says nothing.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from bre import analyze, fewshot


def tbl(df, cols, floats=3):
    if df is None or len(df) == 0:
        return "  (no data)"
    d = df[[c for c in cols if c in df.columns]].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: f"{v:+.{floats}f}" if pd.notna(v) else "—")
    return d.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="results/probe.jsonl")
    ap.add_argument("--judged", default="results/judged.jsonl")
    ap.add_argument("--out", default="results/summary.json")
    args = ap.parse_args()

    p = analyze.load(args.probe)
    p = p[p.rps.notna()]
    fixed = p[~p.arm.astype(str).str.startswith("fewshot_")]

    print("=" * 78)
    print("BASE-MODEL REFUSAL EXAMINATION — RESULTS")
    print("=" * 78)
    print(f"\nmodels probed: {p.model.nunique()}   probe rows: {len(p)}")

    print("\n" + "-" * 78)
    print("0. OFFSET DIAGNOSTIC — mean RPS on BENIGN prompts")
    print("   This is the zero point. It is far above zero and that is string-frequency")
    print("   bias, not refusal. Read every later table as a difference against it.")
    print("-" * 78)
    off = analyze.rps_offset_check(fixed)
    print(tbl(off, ["model", "params_b", "arm", "benign_offset", "std", "count"]))

    print("\n" + "-" * 78)
    print("1. HARM SENSITIVITY, FORM-MATCHED  (XSTest contrast − XSTest safe)")
    print("   The clean estimate: surface form held fixed by construction.")
    print("   Positive means the model prefers refusal more for the unsafe twin.")
    print("-" * 78)
    m = analyze.matched_harm_sensitivity(fixed)
    print(tbl(m, ["model", "params_b", "arm", "value", "lo", "hi", "n"]))

    print("\n" + "-" * 78)
    print("2. HARM SENSITIVITY, UNMATCHED  (HarmBench − XSTest safe)")
    print("   Severity-matched to the literature but FORM-CONFOUNDED: HarmBench is")
    print("   imperative, XSTest safe is interrogative. Compare against table 1.")
    print("-" * 78)
    u = analyze.harm_sensitivity(fixed)
    print(tbl(u, ["model", "params_b", "arm", "benign", "harmful", "delta"]))

    print("\n" + "-" * 78)
    print("3. IMITATION GAP  (chat arm − document arm, HarmBench prompts)")
    print("   The headline. A gap that grows with size means larger models imitate an")
    print("   assistant better. A gap near zero with high sensitivity in both arms")
    print("   would instead suggest something genre-independent.")
    print("-" * 78)
    try:
        g = analyze.imitation_gap(fixed)
        print(tbl(g, ["model", "params_b", "value", "lo", "hi", "n"]))
    except Exception as e:
        print(f"  unavailable: {e}")

    print("\n" + "-" * 78)
    print("4. SIZE TREND — OLS of RPS on log10(params), clustered by behaviour")
    print("   Seven sizes supports a sign and a monotonicity claim, not a power law.")
    print("   No exponent is quoted.")
    print("-" * 78)
    try:
        st = analyze.size_trend(fixed)
        print(tbl(st, ["arm", "slope_per_decade", "se", "p", "n"], floats=4))
    except Exception as e:
        print(f"  unavailable: {e}")

    print("\n5. ARM × SIZE INTERACTION (does the imitation gap itself scale?)")
    try:
        print("   ", analyze.interaction_test(fixed))
    except Exception as e:
        print(f"   unavailable: {e}")

    print("\n" + "-" * 78)
    print("6. FEW-SHOT ELICITATION — RPS by number of in-context declines")
    print("   harm-tracking gap = harmful-demo minus benign-demo. If that gap is ~0 the")
    print("   model is doing format induction, not tracking harm.")
    print("-" * 78)
    fs = fewshot.summarize(args.probe)
    if fs.get("levels"):
        lv = pd.DataFrame(fs["levels"]).sort_values(["params_b", "condition"])
        print(tbl(lv, ["model", "params_b", "condition", "k0", "k1", "k2", "k4", "k8",
                       "k_to_positive"]))
    if fs.get("harm_tracking"):
        ht = pd.DataFrame(fs["harm_tracking"]).sort_values(["params_b", "k"])
        print("\n   harm-tracking gap by k:")
        print(tbl(ht, ["model", "params_b", "k", "harmful_demos", "benign_demos",
                       "harm_tracking_gap"]))

    judged = pathlib.Path(args.judged)
    if judged.exists():
        j = analyze.load(judged)
        print("\n" + "-" * 78)
        print("7. JUDGED GENERATIONS — taxonomy mix")
        print("   Read the INCOHERENT_OR_OFFTOPIC column for the smallest models. That")
        print("   column is why refusal is never derived by negating compliance.")
        print("-" * 78)
        t = analyze.taxonomy_mix(j)
        print(t.to_string(index=False))

    res = analyze.report(args.probe, str(judged) if judged.exists() else None, args.out)
    res["fewshot"] = fs
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

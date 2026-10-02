#!/usr/bin/env python3
"""Headline tables with pair types canonicalised, plus the sign test."""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from bre import analyze

probe = sys.argv[1] if len(sys.argv) > 1 else "results/probe.jsonl"
p = analyze.load(probe); p = p[p.rps.notna()]
fixed = analyze.normalize_pair_type(p[~p.arm.astype(str).str.startswith("fewshot_")])
pd.set_option("display.width", 200)
fmt = lambda v: f"{v:+.3f}"

s = analyze.sensitivity_long(fixed)
print(f"matched clusters after canonicalisation: {s.pair_type.nunique()}")

print("\n" + "-"*78)
print("A. HARM SENSITIVITY BY ARM — base ladder")
print("-"*78)
piv = (s[~s.model.str.contains("Instruct")]
       .pivot_table(index="params_b", columns="arm", values="delta")
       .reset_index().sort_values("params_b"))
print(piv.to_string(index=False, float_format=fmt))

print("\n" + "-"*78)
print("B. IMITATION GAP — chat sensitivity minus document sensitivity")
print("-"*78)
print(analyze.imitation_gap_dod(fixed).sort_values("params_b")
      .to_string(index=False, float_format=fmt))

print("\n" + "-"*78)
print("C. SIZE TREND of harm sensitivity — base ladder")
print("-"*78)
print(analyze.size_trend_sensitivity(fixed).to_string(index=False, float_format=fmt))

print("\nD. FRAME x SIZE INTERACTION:", analyze.interaction_sensitivity(fixed))

st = analyze.sign_test_chat_vs_document(fixed)
print("\n" + "-"*78)
print("E. SIGN TEST — chat above document, across the base ladder")
print("   No clustering assumptions: each model is one independent observation.")
print("-"*78)
print(f"   chat > document in {st['chat_above_document']}/{st['models']} models, "
      f"one-sided p = {st['sign_test_p']:.4f}")
print(pd.DataFrame(st["per_model"]).to_string(index=False, float_format=fmt))

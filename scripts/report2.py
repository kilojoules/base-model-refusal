#!/usr/bin/env python3
"""Corrected headline tables: sensitivity-based, offsets cancelled."""
from __future__ import annotations
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from bre import analyze

probe = sys.argv[1] if len(sys.argv) > 1 else "results/probe.jsonl"
p = analyze.load(probe)
p = p[p.rps.notna()]
fixed = p[~p.arm.astype(str).str.startswith("fewshot_")]
pd.set_option("display.width", 200)

def show(title, note, frame):
    print("\n" + "-" * 78); print(title)
    for l in note.splitlines(): print("   " + l)
    print("-" * 78)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:+.3f}")
          if len(frame) else "   (no data)")

s = analyze.sensitivity_long(fixed)
piv = (s[~s.model.str.contains("Instruct")]
       .pivot_table(index=["params_b"], columns="arm", values="delta")
       .reset_index().sort_values("params_b"))
show("A. HARM SENSITIVITY BY ARM, base ladder only",
     "unsafe twin minus form-matched safe twin, nats/token.\n"
     "The document column is the imitation control.",
     piv)

show("B. IMITATION GAP — corrected (difference of differences)",
     "chat-arm sensitivity minus document-arm sensitivity, bootstrapped over\n"
     "XSTest pair types. Positive means harm sensitivity needs the assistant frame.",
     analyze.imitation_gap_dod(fixed))

show("C. SIZE TREND of harm sensitivity, base ladder only",
     "OLS slope per decade of parameters, clustered by XSTest pair type.",
     analyze.size_trend_sensitivity(fixed))

print("\nD. FRAME x SIZE INTERACTION on sensitivity (base ladder only)")
print("   ", analyze.interaction_sensitivity(fixed))

show("E. FEW-SHOT: format induction vs harm tracking",
     "induction = RPS(k=8) minus RPS(k=0), averaged over demo conditions.\n"
     "tracking  = harmful-demo minus benign-demo RPS at k=8.\n"
     "tracking_share = tracking / induction.",
     analyze.fewshot_harm_tracking_vs_induction(probe))

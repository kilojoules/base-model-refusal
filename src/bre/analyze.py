"""Analysis: behaviour-clustered intervals and the size trend.

Behaviours are the resampling unit. The same behaviour recurs across arms, samples and
models and those observations are correlated, so bootstrapping over rows would give
intervals that are far too narrow. Resample behaviour ids with replacement instead.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def load(path):
    with open(path) as fh:
        return pd.DataFrame([json.loads(l) for l in fh if l.strip()])


def cluster_bootstrap(df, value, by, cluster="behavior_id", n=2000, seed=0, agg="mean"):
    """Resample clusters with replacement; return mean and percentile CI per group."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster].unique()
    idx = {c: g.index.values for c, g in df.groupby(cluster)}
    keys = list(df.groupby(by).groups.keys())
    draws = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        s = df.loc[rows]
        g = s.groupby(by)[value].agg(agg)
        for k in keys:
            draws[k].append(g.get(k, np.nan))
    out = []
    for k in keys:
        a = np.array(draws[k], dtype=float)
        a = a[~np.isnan(a)]
        point = df.groupby(by)[value].agg(agg).get(k, np.nan)
        out.append({
            **(dict(zip(by, k)) if isinstance(k, tuple) else {by[0] if isinstance(by, list) else by: k}),
            "value": float(point),
            "lo": float(np.percentile(a, 2.5)) if len(a) else math.nan,
            "hi": float(np.percentile(a, 97.5)) if len(a) else math.nan,
            "n": int((df.groupby(by).size()).get(k, 0)),
        })
    return pd.DataFrame(out)


def imitation_gap(df, value="rps"):
    """chat-minus-document, per model, on the harmful split. The headline number.

    A gap that grows with size means larger models imitate an assistant better. A gap
    near zero with a high level in both arms would instead suggest something
    genre-independent.
    """
    h = df[df.split == "harmful"]
    piv = h.pivot_table(index=["model", "params_b", "behavior_id"],
                        columns="arm", values=value, aggfunc="mean").reset_index()
    if not {"chat", "document"}.issubset(piv.columns):
        raise ValueError("need both chat and document arms")
    piv["gap"] = piv["chat"] - piv["document"]
    return cluster_bootstrap(piv, "gap", ["model", "params_b"])


def harm_sensitivity(df, value="rps"):
    """harmful-minus-benign within arm and model: the floor-corrected refusal signal.

    If this is ~0 while raw levels are high, the instrument is measuring refusal style
    rather than harm-sensitivity, and nothing should be reported until that is fixed.
    """
    g = df.groupby(["model", "params_b", "arm", "split"])[value].mean().reset_index()
    piv = g.pivot_table(index=["model", "params_b", "arm"],
                        columns="split", values=value).reset_index()
    piv["delta"] = piv.get("harmful") - piv.get("benign")
    return piv


def size_trend(df, value="rps", split="harmful"):
    """OLS of the metric on log10(params) within each arm, clustered by behaviour.

    Seven sizes supports a sign and a monotonicity claim. It does not support a power
    law, and no exponent is quoted.
    """
    import statsmodels.formula.api as smf

    d = df[df.split == split].copy()
    d["log_params"] = np.log10(d["params_b"])
    rows = []
    for arm, g in d.groupby("arm"):
        if g["log_params"].nunique() < 3:
            continue
        m = smf.ols(f"{value} ~ log_params", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g["behavior_id"]}
        )
        rows.append({
            "arm": arm,
            "slope_per_decade": float(m.params["log_params"]),
            "se": float(m.bse["log_params"]),
            "p": float(m.pvalues["log_params"]),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows)


def interaction_test(df, value="rps"):
    """Does the arm effect itself scale? Tests arm x log(size) on harmful prompts."""
    import statsmodels.formula.api as smf

    d = df[(df.split == "harmful") & (df.arm.isin(["chat", "document"]))].copy()
    d["log_params"] = np.log10(d["params_b"])
    d["is_chat"] = (d.arm == "chat").astype(int)
    m = smf.ols(f"{value} ~ log_params * is_chat", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["behavior_id"]}
    )
    k = "log_params:is_chat"
    return {"interaction": float(m.params[k]), "se": float(m.bse[k]),
            "p": float(m.pvalues[k]), "n": int(len(d))}


def taxonomy_mix(df):
    """Label composition per model and arm -- shows the incoherence share directly.

    This table is what makes the 'not-compliant equals refused' error visible: read the
    INCOHERENT_OR_OFFTOPIC column for the smallest models.
    """
    t = df[df.taxonomy.notna()]
    return (t.groupby(["model", "params_b", "arm", "split"])["taxonomy"]
             .value_counts(normalize=True).unstack(fill_value=0.0).reset_index())


def report(probe_path, judged_path=None, out="results/summary.json"):
    import pathlib

    p = load(probe_path)
    res = {
        "n_probe_rows": int(len(p)),
        "models": sorted(p.model.unique().tolist()),
        "rps_by_model_arm_split": cluster_bootstrap(
            p, "rps", ["model", "params_b", "arm", "split"]
        ).to_dict("records"),
        "refusal_fraction": cluster_bootstrap(
            p.assign(pos=(p.rps > 0).astype(float)), "pos",
            ["model", "params_b", "arm", "split"]
        ).to_dict("records"),
        "imitation_gap": imitation_gap(p).to_dict("records"),
        "harm_sensitivity": harm_sensitivity(p).to_dict("records"),
        "size_trend": size_trend(p).to_dict("records"),
    }
    try:
        res["arm_by_size_interaction"] = interaction_test(p)
    except Exception as e:
        res["arm_by_size_interaction"] = {"error": str(e)}
    if judged_path:
        j = load(judged_path)
        res["n_judged_rows"] = int(len(j))
        res["taxonomy_mix"] = taxonomy_mix(j).to_dict("records")
        res["compliance_rate"] = cluster_bootstrap(
            j.assign(c=(j.compliance == "yes").astype(float)), "c",
            ["model", "params_b", "arm", "split"]
        ).to_dict("records")
        res["refusal_rate_judged"] = cluster_bootstrap(
            j.assign(r=j.is_refusal.astype(float)), "r",
            ["model", "params_b", "arm", "split"]
        ).to_dict("records")
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    return res

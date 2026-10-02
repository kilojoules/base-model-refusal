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
        "harm_sensitivity_unmatched": harm_sensitivity(p).to_dict("records"),
        "harm_sensitivity_matched": matched_harm_sensitivity(p).to_dict("records"),
        "rps_offset_check": rps_offset_check(p).to_dict("records"),
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

def matched_harm_sensitivity(df, value="rps"):
    """Form-matched harm sensitivity: XSTest contrast minus XSTest safe, paired by type.

    This is the clean estimate. The HarmBench-minus-benign difference conflates harm
    with grammatical form, because HarmBench behaviours are imperatives and XSTest safe
    prompts are mostly questions. XSTest's contrast twins hold form approximately fixed
    by construction, so their difference isolates harm.
    """
    d = df[df.split.isin(["benign", "harmful_matched"])].copy()
    if d.empty or "pair_type" not in d.columns:
        return pd.DataFrame()
    g = (d.groupby(["model", "params_b", "arm", "pair_type", "split"])[value]
          .mean().unstack("split").reset_index())
    if not {"benign", "harmful_matched"}.issubset(g.columns):
        return pd.DataFrame()
    g["delta"] = g["harmful_matched"] - g["benign"]
    return cluster_bootstrap(g.rename(columns={"pair_type": "behavior_id"}),
                             "delta", ["model", "params_b", "arm"])


def rps_offset_check(df, value="rps"):
    """How much of the metric is a fixed string-frequency offset rather than signal?

    The refusal openers are formulaic, high-frequency English; the compliant openers are
    not. So RPS carries a large positive offset present even on benign prompts and even
    in models with no safety training. Levels are therefore uninterpretable and only
    differences are reported. This table makes the size of that offset explicit.
    """
    b = df[df.split == "benign"]
    return (b.groupby(["model", "params_b", "arm"])[value]
             .agg(["mean", "median", "std", "count"]).reset_index()
             .rename(columns={"mean": "benign_offset"}))


# ---------------------------------------------------------------------------
# Sensitivity-based versions of the headline tables.
#
# The first implementations of imitation_gap, size_trend and interaction_test
# differenced raw RPS *levels* on harmful prompts. That is contaminated by the
# per-arm offset: the chat frame sits roughly 0.3 to 0.6 nats above the document
# frame on benign prompts alone, before any harm enters. A chat-minus-document
# level difference therefore mostly measures the frame, not refusal.
#
# The quantity wanted is a difference of differences: how much more does the
# model prefer refusal for an unsafe prompt than for its form-matched safe twin,
# and how much does that shrink when the assistant frame is removed. Offsets
# cancel in the inner difference, and frame effects cancel in the outer one.
# ---------------------------------------------------------------------------

def sensitivity_long(df, value="rps"):
    """Per model, arm and XSTest pair type: unsafe-twin minus safe-twin RPS."""
    d = df[df.split.isin(["benign", "harmful_matched"])].copy()
    if d.empty or "pair_type" not in d.columns:
        return pd.DataFrame()
    g = (d.groupby(["model", "params_b", "arm", "pair_type", "split"])[value]
          .mean().unstack("split").reset_index())
    if not {"benign", "harmful_matched"}.issubset(g.columns):
        return pd.DataFrame()
    g["delta"] = g["harmful_matched"] - g["benign"]
    return g.dropna(subset=["delta"])


def imitation_gap_dod(df, value="rps"):
    """Difference of differences: chat-arm sensitivity minus document-arm sensitivity.

    This is the corrected headline. Positive means the model's harm sensitivity depends
    on the assistant frame, i.e. it is imitating an assistant rather than expressing a
    frame-independent disposition.
    """
    s = sensitivity_long(df, value)
    if s.empty:
        return pd.DataFrame()
    piv = s.pivot_table(index=["model", "params_b", "pair_type"],
                        columns="arm", values="delta").reset_index()
    if not {"chat", "document"}.issubset(piv.columns):
        return pd.DataFrame()
    piv["gap"] = piv["chat"] - piv["document"]
    return cluster_bootstrap(piv.rename(columns={"pair_type": "behavior_id"}),
                             "gap", ["model", "params_b"])


def size_trend_sensitivity(df, value="rps", base_only=True):
    """OLS of harm sensitivity on log10(params), within arm, clustered by pair type.

    Instruct checkpoints are excluded by default: they are anchors, not points on the
    pretraining-scale ladder, and including them would let two post-trained models drive
    the slope.
    """
    import statsmodels.formula.api as smf

    s = sensitivity_long(df, value)
    if s.empty:
        return pd.DataFrame()
    if base_only:
        s = s[~s.model.str.contains("Instruct")]
    s["log_params"] = np.log10(s["params_b"])
    rows = []
    for arm, g in s.groupby("arm"):
        if g["log_params"].nunique() < 3:
            continue
        m = smf.ols("delta ~ log_params", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": g["pair_type"]})
        rows.append({"arm": arm,
                     "slope_per_decade": float(m.params["log_params"]),
                     "se": float(m.bse["log_params"]),
                     "p": float(m.pvalues["log_params"]),
                     "n_obs": int(len(g)),
                     "n_clusters": int(g["pair_type"].nunique())})
    return pd.DataFrame(rows)


def interaction_sensitivity(df, value="rps", base_only=True):
    """Does the frame-dependence of harm sensitivity itself grow with scale?"""
    import statsmodels.formula.api as smf

    s = sensitivity_long(df, value)
    if s.empty:
        return {"error": "no matched data"}
    if base_only:
        s = s[~s.model.str.contains("Instruct")]
    s = s[s.arm.isin(["chat", "document"])].copy()
    s["log_params"] = np.log10(s["params_b"])
    s["is_chat"] = (s.arm == "chat").astype(int)
    m = smf.ols("delta ~ log_params * is_chat", data=s).fit(
        cov_type="cluster", cov_kwds={"groups": s["pair_type"]})
    k = "log_params:is_chat"
    return {"interaction": float(m.params[k]), "se": float(m.bse[k]),
            "p": float(m.pvalues[k]), "n_obs": int(len(s)),
            "n_clusters": int(s["pair_type"].nunique())}


def fewshot_harm_tracking_vs_induction(path):
    """Compare the two few-shot effects at the largest k.

    induction = RPS(k=max) - RPS(k=0), averaged over both demo conditions. This is how
    much refusal preference bare format evidence buys.
    tracking  = RPS(harmful demos) - RPS(benign demos) at k=max. This is how much the
    *content* of the demonstrations adds on top.

    If tracking is a small fraction of induction, the model is mostly learning "this
    document declines things" rather than "harmful requests get declined."
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame([r for r in rows if str(r["arm"]).startswith("fewshot_")])
    if df.empty:
        return pd.DataFrame()
    df["k"] = df["arm"].str.extract(r"_k(\d+)$").astype(int)
    df["condition"] = df["arm"].str.extract(r"fewshot_(\w+?_demos)_k")
    df = df[df.rps.notna()]
    kmax = int(df["k"].max())
    out = []
    for (m, p), g in df.groupby(["model", "params_b"]):
        k0 = g[g.k == 0]["rps"].mean()
        kmx = g[g.k == kmax]["rps"].mean()
        hd = g[(g.k == kmax) & (g.condition == "harmful_demos")]["rps"].mean()
        bd = g[(g.k == kmax) & (g.condition == "benign_demos")]["rps"].mean()
        induction, tracking = kmx - k0, hd - bd
        out.append({"model": m, "params_b": p, "k_max": kmax,
                    "induction": induction, "tracking": tracking,
                    "tracking_share": (tracking / induction) if induction else np.nan})
    return pd.DataFrame(out).sort_values("params_b")


# XSTest type names do not pair up by prefix alone. Two safe families map onto a single
# contrast family, so a naive "strip contrast_" leaves them unpaired and the matched
# analysis silently loses them -- it ran on 6 pairs instead of 8.
_PAIR_ALIASES = {
    "nons_group_real_discr": "discr",
    "real_group_nons_discr": "discr",
    "privacy_public": "privacy",
    "privacy_fictional": "privacy",
}


def normalize_pair_type(df):
    """Canonicalise pair_type so every safe family matches its contrast family."""
    d = df.copy()
    if "pair_type" not in d.columns:
        return d
    d["pair_type"] = (d["pair_type"].astype(str)
                      .str.replace(r"^contrast_", "", regex=True)
                      .replace(_PAIR_ALIASES))
    return d


def sign_test_chat_vs_document(df, value="rps", base_only=True):
    """Across models, is chat-arm sensitivity above document-arm sensitivity?

    With only a handful of XSTest pair types, cluster-robust p-values are not
    trustworthy. Each model is an independent fit, so a sign test across the ladder needs
    no clustering assumptions at all: under the null that the frame does not matter, the
    sign of (chat − document) is a fair coin for each model.
    """
    from scipy import stats

    s = sensitivity_long(normalize_pair_type(df), value)
    if s.empty:
        return {}
    if base_only:
        s = s[~s.model.str.contains("Instruct")]
    per = (s.groupby(["model", "params_b", "arm"])["delta"].mean()
            .unstack("arm").reset_index())
    if not {"chat", "document"}.issubset(per.columns):
        return {}
    per["gap"] = per["chat"] - per["document"]
    n = len(per)
    k = int((per["gap"] > 0).sum())
    p = float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)
    return {"models": n, "chat_above_document": k, "sign_test_p": p,
            "per_model": per[["model", "params_b", "chat", "document", "gap"]]
                         .sort_values("params_b").to_dict("records")}

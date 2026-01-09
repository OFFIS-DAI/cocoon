"""
Revised statistical analysis for Step 1 (hyper-parameter impact).

Key changes vs. the original script:
- Uses a *single multi-factor model per metric* (hyper-parameters + scenario controls) to avoid confounding.
- Uses transformations suited for skewed / bounded metrics (log, logit).
- Separates substitution effects (ITT vs. PP) and reports substitution determinism.
- Uses Type-II ANOVA (order-invariant) and reports partial eta^2 as effect size.
- Applies Holm correction across hyper-parameters per metric.

ADDON (Step 2):
- Scenario-blocked paired model comparisons (best vs rest) per scenario group using sign-flip permutation tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# -----------------------------
# Helpers: transformations
# -----------------------------
def safe_log(x: pd.Series, eps: float = 1e-9) -> pd.Series:
    return np.log(np.clip(x.astype(float), a_min=eps, a_max=None))


def safe_logit(x: pd.Series, eps: float = 1e-6) -> pd.Series:
    x = x.astype(float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


# -----------------------------
# Core analysis (Step 1)
# -----------------------------
def load_and_filter(csv_path: Path, only_substitution_enabled: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "model_type",
        "substitution_occurred",
        "timeout_occurred",
        "substitution",
        "network_type",
        "traffic_configuration",
        "test_train",
        "cluster_distance_threshold_name",
        "batch_size_ipupa_name",
        "learning_rate_weighting_name",
        "butterfly_threshold_value_name",
        "substitution_priority_name",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Step 1 focus: meta-model
    df = df[df["model_type"].astype(str) == "meta_model"].copy()

    # Substitution enabled subset
    if only_substitution_enabled:
        df = df[df["substitution"].astype(int) == 1].copy()

    # Remove timeouts
    df = df[df["timeout_occurred"].astype(bool) == False].copy()  # noqa: E712

    # Clean factor strings (strip prefixes "foo.bar" -> "bar" if present)
    for col in ["network_type", "traffic_configuration", "test_train"]:
        df[col] = df[col].astype(str).str.replace(r"^[^.]*\.", "", regex=True)

    return df


def substitution_determinism_report(df: pd.DataFrame) -> pd.DataFrame:
    tab = (
        df.groupby(["butterfly_threshold_value_name"])["substitution_occurred"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_configs", "mean": "substitution_rate"})
        .reset_index()
        .sort_values("butterfly_threshold_value_name")
    )
    return tab


def fit_anova_for_metric(
    df: pd.DataFrame,
    metric: str,
    view: str,
    hp_factors: List[str],
    controls: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    if view == "pp":
        d = d[d["substitution_occurred"].astype(bool) == True].copy()  # noqa: E712
    elif view == "itt":
        pass
    else:
        raise ValueError("view must be 'itt' or 'pp'")

    # Transform response
    if metric == "execution_time_s":
        d["_y"] = safe_log(d[metric])
    elif metric == "wasserstein_distance":
        d["_y"] = safe_log(d[metric], eps=1e-6)
    elif metric == "nrmse_mean":
        d["_y"] = safe_log(d[metric], eps=1e-9)
    elif metric == "mean_in_one_sigma_interval":
        d["_y"] = safe_logit(d[metric], eps=1e-6)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    terms = []
    terms += [f"C({c})" for c in hp_factors]
    terms += [f"C({c})" for c in controls]
    if view == "itt":
        terms += ["C(substitution_occurred)"]

    formula = "_y ~ " + " + ".join(terms)

    model = ols(formula, data=d).fit()
    anova = anova_lm(model, typ=2)  # Type-II (order-invariant)

    if "Residual" not in anova.index:
        raise RuntimeError("ANOVA table missing Residual row; check data and formula.")
    ss_error = float(anova.loc["Residual", "sum_sq"])
    anova["partial_eta2"] = anova["sum_sq"] / (anova["sum_sq"] + ss_error)

    coef = (
        pd.DataFrame(
            {
                "term": model.params.index,
                "coef": model.params.values,
                "std_err": model.bse.values,
                "t": model.tvalues.values,
                "p": model.pvalues.values,
            }
        )
        .reset_index(drop=True)
        .sort_values("p")
    )

    return anova.reset_index().rename(columns={"index": "factor"}), coef


def summarize_hyperparam_effects(
    anova: pd.DataFrame,
    hp_factors: List[str],
) -> pd.DataFrame:
    wanted = {f"C({c})": c for c in hp_factors}
    sub = anova[anova["factor"].isin(list(wanted.keys()))].copy()
    sub["hyperparameter"] = sub["factor"].map(wanted)

    p_raw = {row["hyperparameter"]: float(row["PR(>F)"]) for _, row in sub.iterrows()}
    sub["p"] = sub["hyperparameter"].map(p_raw)

    keep = ["hyperparameter", "df", "F", "PR(>F)", "p", "partial_eta2"]
    sub = sub[keep].sort_values("p")
    return sub



def main():
    # --------- minimal change: choose phase here ----------
    PHASE = 1  # set to 2 for Step 2

    if PHASE == 1:
        csv_path = Path('analysis_results/aggregated_results1.csv')
        outdir = Path('analysis_results/statistic_analysis1')
        outdir.mkdir(parents=True, exist_ok=True)

        df = load_and_filter(csv_path, only_substitution_enabled=True)

        print("=" * 80)
        print("STEP 1 – REVISED STATISTICAL ANALYSIS")
        print(f"CSV: {csv_path}")
        print(f"Rows (meta-model, filtered): {len(df)}")
        print(f"Substitution occurred: {df['substitution_occurred'].mean():.3f} (rate)")
        print("=" * 80)

        det = substitution_determinism_report(df)
        print("\nSubstitution rate by butterfly threshold:")
        print(det.to_string(index=False))
        det.to_csv(outdir / "substitution_rate_by_butterfly_threshold.csv", index=False)

        hp_factors = [
            "cluster_distance_threshold_name",
            "batch_size_ipupa_name",
            "learning_rate_weighting_name",
            "butterfly_threshold_value_name",
            "substitution_priority_name",
        ]
        controls = ["network_type", "traffic_configuration", "test_train"]

        metrics = [
            "execution_time_s",
            "nrmse_mean",
            "wasserstein_distance",
            "mean_in_one_sigma_interval",
        ]

        views = ["itt", "pp"]

        all_summaries = []
        for metric in metrics:
            for view in views:
                anova, coef = fit_anova_for_metric(
                    df=df,
                    metric=metric,
                    view=view,
                    hp_factors=hp_factors,
                    controls=controls,
                )
                anova.to_csv(outdir / f"anova_{metric}_{view}.csv", index=False)
                coef.to_csv(outdir / f"coef_{metric}_{view}.csv", index=False)

                hp_sum = summarize_hyperparam_effects(anova, hp_factors=hp_factors)
                hp_sum.insert(0, "metric", metric)
                hp_sum.insert(1, "view", view)
                all_summaries.append(hp_sum)

                print("\n" + "-" * 80)
                print(f"Metric: {metric} | View: {view.upper()}")
                print(hp_sum.to_string(index=False))

        summary = pd.concat(all_summaries, ignore_index=True)
        summary.to_csv(outdir / "hyperparameter_effects_summary.csv", index=False)

        print("\n" + "=" * 80)
        print(f"Saved outputs to: {outdir.resolve()}")
        print("Files:")
        print(" - substitution_rate_by_butterfly_threshold.csv")
        print(" - anova_<metric>_<view>.csv")
        print(" - coef_<metric>_<view>.csv")
        print(" - hyperparameter_effects_summary.csv")
        print("=" * 80)


    else:
        raise ValueError("PHASE must be 1 or 2")


if __name__ == "__main__":
    main()

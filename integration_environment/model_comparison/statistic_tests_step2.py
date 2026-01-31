# statistic_tests_step2.py

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import logit

CSV_PATH = "analysis_results/aggregated_results2.csv"
OUT_PATH = "analysis_results/statistic_analysis2/step2_modeltype_significance.csv"

SCENARIO_COLS = [
    "network_type",
    "traffic_configuration",
    "num_devices",
    "scenario_duration",
    "payload_size",
    "test_train_name",
]

EPS = 1e-12  # to avoid log(0)


def partial_eta2_from_models(full_model, reduced_model) -> float:
    """
    Partial eta^2 for model_type using SSE difference:
      SS_effect = SSE_reduced - SSE_full
      eta_p^2   = SS_effect / (SS_effect + SSE_full)
              = (SSE_reduced - SSE_full) / SSE_reduced
    """
    sse_full = float(np.sum(full_model.resid ** 2))
    sse_red = float(np.sum(reduced_model.resid ** 2))

    if not np.isfinite(sse_full) or not np.isfinite(sse_red) or sse_red <= 0:
        return np.nan

    ss_effect = sse_red - sse_full
    if ss_effect < 0:  # numerical noise
        ss_effect = 0.0

    return float(ss_effect / sse_red)


def run_modeltype_test(df_metric: pd.DataFrame, metric_col: str, controls_rhs: str):
    """
    Robust test for whether adding C(model_type) improves fit over controls only.
    Uses nested-model F-test to avoid Type-II ANOVA constraint-matrix failures.
    Returns (F, p, partial_eta2). NaNs if not testable.
    """
    if df_metric["model_type"].nunique(dropna=True) < 2:
        return np.nan, np.nan, np.nan

    full = smf.ols(f"{metric_col} ~ C(model_type) + {controls_rhs}", data=df_metric).fit()
    reduced = smf.ols(f"{metric_col} ~ {controls_rhs}", data=df_metric).fit()

    f_stat, p_val, df_diff = full.compare_f_test(reduced)

    # If df_diff == 0, model_type didn't add estimable parameters (perfect confounding)
    if df_diff == 0:
        return np.nan, np.nan, np.nan

    eta2 = partial_eta2_from_models(full, reduced)
    return float(f_stat), float(p_val), float(eta2)


def main():
    df = pd.read_csv(CSV_PATH)

    # Exclude timeouts
    df = df[df["timeout_occurred"] == False].copy()  # noqa: E712

    # Fill missing covariates so baselines aren't dropped by patsy
    for c in ["test_train_name", "payload_size", "scenario_duration", "traffic_configuration", "network_type"]:
        if c in df.columns:
            df[c] = df[c].fillna("baseline").astype(str)

    # Ensure model_type is present and patsy-friendly (object/str)
    df["model_type"] = df["model_type"].fillna("unknown").astype(str)

    # Ensure num_devices is numeric (but we treat it categorically in the model)
    df["num_devices"] = df["num_devices"].fillna("unknown").astype(str)

    # Transformations (robust against zeros)
    df["log_et"] = np.log(pd.to_numeric(df["execution_time_s"], errors="coerce").clip(lower=EPS))
    df["log_nrmse"] = np.log(pd.to_numeric(df["nrmse_mean"], errors="coerce").clip(lower=EPS))
    df["log_wasserstein"] = np.log(pd.to_numeric(df["wasserstein_distance"], errors="coerce").clip(lower=EPS))
    cov = pd.to_numeric(df["mean_in_one_sigma_interval"], errors="coerce").clip(1e-6, 1 - 1e-6)
    df["logit_coverage"] = logit(cov)

    # Controls (blocking factors)
    controls_rhs = (
        "C(network_type)"
        " + C(traffic_configuration)"
        " + C(num_devices)"
        " + C(scenario_duration)"
        " + C(payload_size)"
        " + C(test_train_name)"
    )

    results = []
    for metric_col, metric_name in [
        ("log_et", "execution_time_s (log)"),
        ("log_nrmse", "nrmse_mean (log)"),
        ("log_wasserstein", "wasserstein_distance (log)"),
        ("logit_coverage", "mean_in_one_sigma_interval (logit)"),
    ]:
        # Per-metric clean-up: drop rows with non-finite transformed values
        df_metric = df[np.isfinite(df[metric_col])].copy()
        df_metric = df_metric.dropna(subset=["model_type"] + SCENARIO_COLS)

        F, p, eta2 = run_modeltype_test(df_metric, metric_col, controls_rhs)

        results.append(
            {
                "metric": metric_name,
                "n_rows_used": int(len(df_metric)),
                "n_model_types_used": int(df_metric["model_type"].nunique()),
                "F_model_type": F,
                "p_value_model_type": p,
                "partial_eta2_model_type": eta2,
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(OUT_PATH, index=False)
    print(out)


if __name__ == "__main__":
    main()

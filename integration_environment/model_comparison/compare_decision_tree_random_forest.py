import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

DT_PATH = "analysis_results/aggregated_results1.csv"
RF_PATH = "analysis_results/aggregated_results3.csv"

from typing import Dict

METRIC_LABELS: Dict[str, str] = {
    "nrmse_mean": r"$\mathrm{NRMSE}$",
    "mean_in_one_sigma_interval": r"$C_{\pm\sigma}$",
    "wasserstein_distance": r"$W$",
    "execution_time_s": r"$ET$"
}


METRICS_CONT = [
    "nrmse_mean",
    "wasserstein_distance",
    "mean_in_one_sigma_interval",
    "execution_time_s",
    "score",
]
METRICS_BOOL = [
    "substitution_occurred",
    "timeout_occurred",
]


def robust_summary(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, q25=np.nan, q75=np.nan, min=np.nan, max=np.nan)
    return dict(
        n=int(s.shape[0]),
        mean=float(s.mean()),
        std=float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        median=float(s.median()),
        q25=float(s.quantile(0.25)),
        q75=float(s.quantile(0.75)),
        min=float(s.min()),
        max=float(s.max()),
    )


def bool_rate(s: pd.Series) -> dict:
    # akzeptiert bool, 0/1, "True"/"False"
    s2 = s.copy()
    if s2.dtype != bool:
        s2 = s2.map(lambda x: str(x).strip().lower() in {"true", "1", "yes"})
    s2 = s2.dropna()
    if len(s2) == 0:
        return dict(n=0, true_rate=np.nan)
    return dict(n=int(len(s2)), true_rate=float(s2.mean()))


def summarize_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for m in METRICS_CONT:
        if m not in df.columns:
            continue
        r = robust_summary(df[m])
        r.update(metric=m, model=label, kind="continuous")
        rows.append(r)
    for m in METRICS_BOOL:
        if m not in df.columns:
            continue
        r = bool_rate(df[m])
        r.update(metric=m, model=label, kind="boolean")
        rows.append(r)
    return pd.DataFrame(rows)

def boxplot_metrics_combined_seaborn(
    decision_tree_df: pd.DataFrame,
    random_forest_df: pd.DataFrame,
    metrics: list,
    metric_labels: dict,
    output_path: str = "analysis_results/plots_phase3/boxplots_combined_metrics.pdf"
):
    """
    Creates one figure with multiple boxplots (facets),
    comparing Decision Tree and Random Forest for several metrics.
    Uses seaborn, serif font, and LaTeX-style metric labels.
    """

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sns.set_theme(style="whitegrid", font='serif')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",  # Computer Modern (serif)
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
    })

    # Long-format dataframe
    dt_long = decision_tree_df[metrics].copy()
    dt_long["model"] = "Decision Tree"

    rf_long = random_forest_df[metrics].copy()
    rf_long["model"] = "Random Forest"

    df_long = pd.concat([dt_long, rf_long], ignore_index=True)
    df_long = df_long.melt(
        id_vars="model",
        var_name="metric",
        value_name="value"
    ).dropna(subset=["value"])

    # Faceted boxplots
    g = sns.catplot(
        data=df_long,
        x="model",
        y="value",
        col="metric",
        kind="box",
        col_wrap=4,
        sharey=False,
        height=3.5,
        aspect=0.8,
        showfliers=True,
        palette='YlOrBr'
    )

    # Replace facet titles and y-labels with symbolic names
    for ax in g.axes.flatten():
        metric = ax.get_title().replace("metric = ", "")
        label = metric_labels.get(metric, metric)

        ax.set_title(label)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main():
    dt = pd.read_csv(DT_PATH)
    rf = pd.read_csv(RF_PATH)

    # globale Summaries
    summary = pd.concat([
        summarize_df(dt, "decision_tree"),
        summarize_df(rf, "random_forest")
    ], ignore_index=True)

    # hübscher ausgeben
    print("\n=== GLOBAL SUMMARY (per model) ===")
    # Kontinuierliche Metriken
    cont = summary[summary["kind"] == "continuous"].copy()
    if not cont.empty:
        print(cont[["model", "metric", "n", "mean", "std", "median", "q25", "q75", "min", "max"]]
              .sort_values(["metric", "model"])
              .to_string(index=False))

    # Bool Metriken
    boo = summary[summary["kind"] == "boolean"].copy()
    if not boo.empty:
        print("\n=== RATES (per model) ===")
        print(boo[["model", "metric", "n", "true_rate"]]
              .sort_values(["metric", "model"])
              .to_string(index=False))

    boxplot_metrics_combined_seaborn(
        dt,
        rf,
        metrics=[
            "nrmse_mean",
            "wasserstein_distance",
            "mean_in_one_sigma_interval",
            "execution_time_s",
        ],
        metric_labels=METRIC_LABELS
    )


if __name__ == "__main__":
    main()

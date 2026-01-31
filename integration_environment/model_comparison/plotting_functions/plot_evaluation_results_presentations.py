# execution_times_plot_seaborn_simple.py
#
# Generates seaborn barplots of:
#   - Execution time by traffic_configuration and model_type (with broken y-axis)
#   - NMSE by traffic_configuration and model_type
#   - Mean in one sigma interval by traffic_configuration and model_type
#
# For meta_model: keeps only the best-scoring test-train configuration per traffic config.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from integration_environment.scenario_configuration import TrafficConfig

sns.set()

# =========================
# CONFIG
# =========================
# File paths
csv_path = Path("../analysis_results/aggregated_results2.csv")  # Input CSV

out_dir = Path("../analysis_results/plots_phase2")
out_dir.mkdir(parents=True, exist_ok=True)

out_png_time = out_dir / "execution_times_by_traffic_and_model.png"
out_csv_time = out_dir / "execution_times_aggregated.csv"

out_png_nmse = out_dir / "nmse_by_traffic_and_model.png"
out_csv_nmse = out_dir / "nmse_aggregated.csv"

out_png_sigma = out_dir / "mean_in_one_sigma_by_traffic_and_model.png"
out_csv_sigma = out_dir / "mean_in_one_sigma_aggregated.csv"

# Plot look & feel
sns.set_theme(style="whitegrid", context="talk", palette="pastel")

# Renaming for x-axis categories
rename_map = {
    "TrafficConfig.cbr_broadcast_1_mps": "CBR Broadcast, 1mps",
    "TrafficConfig.cbr_broadcast_1_mpm": "CBR Broadcast, 1mpm",
    "TrafficConfig.poisson_broadcast_1_mps_1": "Poisson Broadcast, 1mps",
    "TrafficConfig.poisson_broadcast_1_mpm_1": "Poisson Broadcast, 1mpm",
    "TrafficConfig.central_dsb_1mpm_5s_50p": "Complex (Central DSB), 1mpm",
    "TrafficConfig.central_dsb_5mpm_30s_75": "Complex (Central DSB), 5mpm",
}

# Desired plotting order (groups similar configs by rate & complexity)
traffic_order_raw = [
    "TrafficConfig.cbr_broadcast_1_mps",
    "TrafficConfig.cbr_broadcast_1_mpm",
    "TrafficConfig.poisson_broadcast_1_mps_1",
    "TrafficConfig.poisson_broadcast_1_mpm_1",
    "TrafficConfig.central_dsb_1mpm_5s_50p",
    "TrafficConfig.central_dsb_5mpm_30s_75",
]

# Corresponding labels (after rename)
traffic_order_labels = [rename_map[t] for t in traffic_order_raw]


# Optional y-limits for the added figures
# Set to None to auto-scale
ylims_nmse = None  # e.g., (0, 0.5)
ylims_sigma = (0, 1.2)  # typically a fraction; adjust if your data differs


# =========================
# Helpers
# =========================
def pick_col(df, candidates, required=True):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    if required:
        raise KeyError(f"Couldn't find {candidates}. Available: {df.columns}")
    return None


def is_meta_model(val: str) -> bool:
    v = str(val).lower().replace("-", "_").replace(" ", "_")
    return v in {"meta_model", "metamodel", "meta"} or "meta" in v


def plot_single_axis_bar(
        df_plot, x_col, y_col, hue_col, out_png, title, ylabel,
        rename_map=None, ylims=None, order=None
):
    """One-axis seaborn barplot with a single legend placed outside."""
    dfp = df_plot.copy()
    if rename_map:
        dfp[x_col] = dfp[x_col].replace(rename_map)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=dfp,
        x=x_col,
        y=y_col,
        hue=hue_col,
        estimator="mean",
        errorbar=("ci", 95),
        dodge=True,
        edgecolor="black",
        order=order,          # <--- NEW
    )
    ax.set_title(title)
    ax.set_xlabel("Traffic Configuration")
    ax.set_ylabel(ylabel)
    if ylims is not None:
        ax.set_ylim(*ylims)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Model Type", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =========================
# Load & prepare
# =========================
df = pd.read_csv(csv_path)

traffic_col = pick_col(df, ["traffic_configuration", "traffic_config", "traffic"])
model_col = pick_col(df, ["model_type"])
time_col = pick_col(df, ["execution_time_s", "execution_time", "runtime"])
score_col = pick_col(df, ["score"], required=False)

# New metric columns
nmse_col = pick_col(df, ["nrmse_mean"])
sigma_col = pick_col(df, ["mean_in_one_sigma_interval"])

df[model_col] = df[model_col].astype(str).str.strip()


# =========================
# NEU: Auswahl der besten Meta-Model-Konfiguration je Metrik
# =========================

def get_df_with_extra_substitution_column(df, model_col=model_col):
    """Wählt für das Meta-Model je Traffic-Konfiguration die beste Zeile bzgl. metric_col.
       better in {'min','max'}."""
    df_local = df.copy()
    meta_mask = df_local[model_col].apply(is_meta_model)
    if not meta_mask.any():
        return df_local

    meta_df = df_local[meta_mask].copy()
    meta_df = meta_df[meta_df['substitution_occurred'] == True]
    meta_df = meta_df[meta_df['test_train_name'] != 'traffic_load_split']

    non_meta = df_local[~meta_mask]
    return pd.concat([non_meta, meta_df], ignore_index=True)


# Erzeuge metrikspezifische DFs:
df_time = get_df_with_extra_substitution_column(df)
df_nmse = get_df_with_extra_substitution_column(df)
df_sigma = get_df_with_extra_substitution_column(df)

# =========================
# Aggregationen für CSVs (metrikspezifisch)
# =========================
agg_time = (
    df_time.groupby([traffic_col, model_col], as_index=False)[time_col]
    .mean().rename(columns={time_col: f"{time_col}_mean"})
    .sort_values([traffic_col, model_col])
)
agg_time.to_csv(out_csv_time, index=False)

agg_nmse = (
    df_nmse.groupby([traffic_col, model_col], as_index=False)[nmse_col]
    .mean().rename(columns={nmse_col: f"{nmse_col}_mean"})
    .sort_values([traffic_col, model_col])
)
agg_nmse.to_csv(out_csv_nmse, index=False)

agg_sigma = (
    df_sigma.groupby([traffic_col, model_col], as_index=False)[sigma_col]
    .mean().rename(columns={sigma_col: f"{sigma_col}_mean"})
    .sort_values([traffic_col, model_col])
)
agg_sigma.to_csv(out_csv_sigma, index=False)

# =========================
# PLOTS (metrikspezifische DFs verwenden)
# =========================
# --- 1) Execution time mit "broken y-axis"
df_time_plot = df_time.copy()
df_time_plot[traffic_col] = df_time_plot[traffic_col].replace(rename_map)

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(14, 10),
    gridspec_kw={'height_ratios': [1.5, 2]}
)

sns.barplot(
    data=df_time_plot, x=traffic_col, y=time_col, hue=model_col,
    estimator="mean", errorbar=("ci", 95), dodge=True, edgecolor="black",
    order=traffic_order_labels, ax=ax1
)
sns.barplot(
    data=df_time_plot, x=traffic_col, y=time_col, hue=model_col,
    estimator="mean", errorbar=("ci", 95), dodge=True, edgecolor="black",
    order=traffic_order_labels, ax=ax2
)

ax1.set_ylim(10, df_time_plot[time_col].max() + 20)
ax2.set_ylim(0, 10)

ax1.set_title("Execution Time by Traffic Configuration and Model Type")
ax2.set_xlabel("Traffic Configuration")
ax1.set_ylabel("")
ax2.set_ylabel("Execution Time (seconds)")

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=20, ha="right")
ax1.tick_params(labelbottom=False)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend_.remove()
ax2.legend(handles, labels, title="Model Type", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(out_png_time, dpi=200)
plt.close()

# --- 2) NMSE
plot_single_axis_bar(
    df_plot=df_nmse,
    x_col=traffic_col,
    y_col=nmse_col,
    hue_col=model_col,
    out_png=out_png_nmse,
    title="Normalized RMSE by Traffic Configuration and Model Type",
    ylabel="N-RMSE",
    rename_map=rename_map,
    ylims=ylims_nmse,
    order=traffic_order_labels,   # <--- NEW
)

# --- 3) Mean in one sigma interval
plot_single_axis_bar(
    df_plot=df_sigma,
    x_col=traffic_col,
    y_col=sigma_col,
    hue_col=model_col,
    out_png=out_png_sigma,
    title="Mean in One Sigma Interval by Traffic Configuration and Model Type",
    ylabel="Mean in One Sigma Interval",
    rename_map=rename_map,
    ylims=ylims_sigma,
    order=traffic_order_labels,   # <--- NEW
)

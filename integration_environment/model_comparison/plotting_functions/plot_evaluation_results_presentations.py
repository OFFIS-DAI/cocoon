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
    "TrafficConfig.cbr_broadcast_1_mps": "CBR Broadcast",
    "TrafficConfig.poisson_broadcast_1_mps_1": "Poisson Broadcast",
    "TrafficConfig.central_dsb_1mpm_5s_50p": "Complex (Central DSB)",
}

# Optional y-limits for the added figures
# Set to None to auto-scale
ylims_nmse = None  # e.g., (0, 0.5)
ylims_sigma = (0, 2.0)  # typically a fraction; adjust if your data differs


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
        df_plot, x_col, y_col, hue_col, out_png, title, ylabel, rename_map=None, ylims=None
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
model_col = pick_col(df, ["model_type", "model", "estimator"])
time_col = pick_col(df, ["execution_time_s", "execution_time", "runtime"])
score_col = pick_col(df, ["score", "accuracy", "auc", "r2"], required=False)

# New metric columns
nmse_col = pick_col(df, ["nrmse_mean", "nmse", "normalized_rmse", "nrmse"])
sigma_col = pick_col(df, ["mean_in_one_sigma_interval", "mean_in_1_sigma", "one_sigma_mean"])

df[model_col] = df[model_col].astype(str).str.strip()


# =========================
# NEU: Auswahl der besten Meta-Model-Konfiguration je Metrik
# =========================

def pick_best_meta_by_metric(df, metric_col, better: str, traffic_col=traffic_col, model_col=model_col):
    """Wählt für das Meta-Model je Traffic-Konfiguration die beste Zeile bzgl. metric_col.
       better in {'min','max'}."""
    df_local = df.copy()
    meta_mask = df_local[model_col].apply(is_meta_model)
    if not meta_mask.any():
        return df_local

    meta_df = df_local[meta_mask].copy()
    meta_df = meta_df[meta_df['substitution_occurred'] == True]
    if better == "min":
        best_idx = meta_df.groupby(traffic_col)[metric_col].idxmin()
    elif better == "max":
        best_idx = meta_df.groupby(traffic_col)[metric_col].idxmax()
    else:
        raise ValueError("better must be 'min' or 'max'")

    best_meta = meta_df.loc[best_idx]
    non_meta = df_local[~meta_mask]
    return pd.concat([non_meta, best_meta], ignore_index=True)


# Erzeuge metrikspezifische DFs:
df_time = pick_best_meta_by_metric(df, time_col, "min")
df_nmse = pick_best_meta_by_metric(df, nmse_col, "min")
df_sigma = pick_best_meta_by_metric(df, sigma_col, "max")

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
    estimator="mean", errorbar=("ci", 95), dodge=True, edgecolor="black", ax=ax1
)
sns.barplot(
    data=df_time_plot, x=traffic_col, y=time_col, hue=model_col,
    estimator="mean", errorbar=("ci", 95), dodge=True, edgecolor="black", ax=ax2
)

ax1.set_ylim(5, df_time_plot[time_col].max() + 20)
ax2.set_ylim(0, 5)

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
    ylims=ylims_nmse
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
    ylims=ylims_sigma
)

# =========================
# New: devices/scalability
# =========================

# Detect devices column
devices_col = pick_col(
    df,
    ["num_devices"],
    required=True
)

# Extract raw string values
df[devices_col] = df[devices_col].astype(str)

# Mapping from enum-like names to integers
device_map = {
    "NumDevices.five": 5,
    "NumDevices.fifty": 50,
    "NumDevices.hundred": 100,
    # add more if present in your data
}

# Apply mapping
df[devices_col] = df[devices_col].map(device_map)


# Helper: allow grouping by multiple keys for meta selection
def pick_best_meta_by_metric_grouped(df, metric_col, better: str,
                                     group_cols, model_col=model_col):
    """Pick best meta-model row per group (e.g., per traffic & num_devices) wrt metric_col."""
    df_local = df.copy()
    meta_mask = df_local[model_col].apply(is_meta_model)
    if not meta_mask.any():
        return df_local

    meta_df = df_local[meta_mask].copy()
    # keep only substituted meta rows (your original constraint)
    if "substitution_occurred" in meta_df.columns:
        meta_df = meta_df[meta_df["substitution_occurred"] == True]

    if better == "min":
        best_idx = meta_df.groupby(group_cols, dropna=False)[metric_col].idxmin()
    elif better == "max":
        best_idx = meta_df.groupby(group_cols, dropna=False)[metric_col].idxmax()
    else:
        raise ValueError("better must be 'min' or 'max'")

    best_meta = meta_df.loc[best_idx]
    non_meta = df_local[~meta_mask]
    return pd.concat([non_meta, best_meta], ignore_index=True)


# ---------- metric-specific DF for time with devices ----------
group_time_devices = [traffic_col, devices_col]
df_time_devices = pick_best_meta_by_metric_grouped(df, time_col, "min", group_time_devices, model_col=model_col)

# Aggregate mean time per (traffic, model, num_devices) for plotting/CSV
agg_time_devices = (
    df_time_devices.groupby([traffic_col, model_col, devices_col], as_index=False)[time_col]
    .mean()
    .rename(columns={time_col: f"{time_col}_mean"})
    .sort_values([traffic_col, model_col, devices_col])
)

out_png_time_devices = out_dir / "execution_time_vs_devices.png"
out_csv_time_devices = out_dir / "execution_time_vs_devices_aggregated.csv"
agg_time_devices.to_csv(out_csv_time_devices, index=False)

# Plot: execution time vs number of devices (ONE plot across all traffic configs)
df_plot = agg_time_devices.replace({traffic_col: rename_map})

plt.figure(figsize=(12, 7))
ax = sns.lineplot(
    data=df_plot,
    x=devices_col,
    y=f"{time_col}_mean",
    hue=model_col,          # color = model
    style=traffic_col,      # line style = traffic config
    markers=True,
    dashes=False,
    estimator=None          # already aggregated means
)

ax.set_xlabel("Number of Devices")
ax.set_ylabel("Execution Time (seconds)")
ax.legend(title="Model / Traffic")
plt.tight_layout()
plt.savefig(out_png_time_devices, dpi=200)
plt.close()

# =========================
# Meta-model: compare test–train configs
# =========================

# Prefer human-readable test–train name if present
tt_name_col = None
try:
    tt_name_col = pick_col(df, ["test_train_name"], required=False)
except KeyError:
    tt_name_col = None
tt_col_fallback = pick_col(df, ["test_train"], required=True)
testtrain_col = tt_name_col or tt_col_fallback  # use names if available

# =========================
# Meta-model: score by test–train (bar chart)
# =========================

# Keep only meta-model rows (and your constraint: substituted)
df_meta = df[df[model_col].apply(is_meta_model)].copy()
if "substitution_occurred" in df_meta.columns:
    df_meta = df_meta[df_meta["substitution_occurred"] == True]

# Use readable test–train names if available
tt_col = pick_col(df_meta, ["test_train_name", "test_train"], required=True)
df_meta[tt_col] = df_meta[tt_col].astype(str)

# Choose metric: prefer 'score'; fallback to NRMSE if score missing/NaN
metric_col = None
metric_label = None
if score_col and df_meta[score_col].notna().any():
    metric_col = score_col
    metric_label = "Score"
else:
    metric_col = nmse_col
    metric_label = "N-RMSE"

# (Optional) CSV aggregation for reproducibility
out_csv_meta_score = out_dir / "meta_scores_by_testtrain.csv"
(
    df_meta.groupby([traffic_col, tt_col], as_index=False)[metric_col]
           .mean()
           .rename(columns={metric_col: f"{metric_col}_mean"})
           .sort_values([traffic_col, tt_col])
).to_csv(out_csv_meta_score, index=False)

sns.set_theme(style="whitegrid", context="talk", palette="Set2")

# Plot: one bar chart across all traffic configs; hue = test–train
out_png_meta_score = out_dir / "meta_scores_by_testtrain.png"
plot_single_axis_bar(
    df_plot=df_meta,
    x_col=traffic_col,
    y_col=metric_col,
    hue_col=tt_col,
    out_png=out_png_meta_score,
    title=("Meta-Model: Score by Traffic Configuration and Test–Train"
           if metric_label == "Score"
           else "Meta-Model: N-RMSE by Traffic Configuration and Test–Train"),
    ylabel=metric_label,
    rename_map=rename_map,
    ylims=None  # adjust if you want fixed limits
)


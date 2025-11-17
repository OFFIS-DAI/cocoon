# plot_step2_dissertation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def summarize_step2(file_path: str) -> None:
    """
    Step-2 overview + plots:
    - Basic info about the CSV
    - Normalization of model_type
    - Plots for selected model types:
        * influence of traffic model, num_devices and scenario_duration
          on accuracy metrics (NRMSE, one-sigma coverage, Wasserstein distance)
    """

    outdir = Path("../analysis_results/plots_phase2")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 5) Plot settings ---
    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    sns.set_theme(style="whitegrid", font="serif")

    model_labels = {
        "meta_model": "Meta-Model",
        "channel": "Channel",
        "static_graph": "Static Graph"
    }

    # --- 1) Read CSV ---
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # --- 1a) Basic info ---
    print("=== Step 2 Overview ===")
    print(f"CSV file: {file_path.resolve()}")
    print(f"Rows (runs): {len(df)}")

    # --- 1b) Normalized model_type column (if present) ---
    if "model_type" in df.columns:
        df["model_type_norm"] = (
            df["model_type"].astype(str)
            .str.strip().str.lower()
            .replace({"meta model": "meta_model", "meta-model": "meta_model"})
        )
    else:
        df["model_type_norm"] = "unknown"
        print("\n[WARN] No 'model_type' column found. Using 'unknown' as model_type_norm.")

    # --- 2) Light overview on scenarios (if available) ---
    if "scenario_id" in df.columns:
        n_scenarios = df["scenario_id"].nunique()
        print(f"Unique scenarios: {n_scenarios}")
    else:
        print("Column 'scenario_id' not found (no scenario count shown).")

    print("\nModel type counts (all):")
    print(df["model_type_norm"].value_counts(dropna=False))

    # --- 3) Prepare variables needed for influence plots ---

    # Check required columns
    required_cols = [
        "traffic_configuration",
        "num_devices",
        "scenario_duration",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for plots: {missing}")

    # 3a) Clean traffic model naming
    df["traffic_model"] = (
        df["traffic_configuration"]
        .astype(str)
        .str.replace("TrafficConfig.", "", regex=False)
        .str.replace("evaluation_", "", regex=False)
        .str.strip()
    )

    # 3b) Parse num_devices → numeric
    dev_map = {
        "five": 5,
        "ten": 10,
        "twenty": 20,
        "fifty": 50,
    }
    df["num_devices_num"] = (
        df["num_devices"]
        .astype(str)
        .str.replace("NumDevices.", "", regex=False)
        .map(dev_map)
    )

    # Subset for plotting: drop rows where the respective x-variable is missing
    df_devices = df.dropna(subset=["num_devices_num"]).copy()

    # --- EXCLUDE model types: ideal + detailed (no subplots for them) ---
    exclude_models = ["ideal", "detailed"]
    df_devices = df_devices[~df_devices["model_type_norm"].isin(exclude_models)]

    # --- 4) Metrics to visualize ---
    metrics = [
        "nrmse_mean",
        "mean_in_one_sigma_interval",
        "wasserstein_distance",
    ]
    metrics_present = [m for m in metrics if m in df.columns]
    if not metrics_present:
        raise ValueError(
            "None of the expected metric columns "
            "['nrmse_mean', 'mean_in_one_sigma_interval', 'wasserstein_distance'] "
            "are present in the CSV."
        )

    metric_labels = {
        "nrmse_mean": "$NRMSE$",
        "mean_in_one_sigma_interval": "$C_{\pm\sigma}$",
        "wasserstein_distance": "$W$",
    }

    outdir = Path("../analysis_results/plots_phase2")
    outdir.mkdir(parents=True, exist_ok=True)

    # --- 4a) Overall accuracy metrics across all scenarios (per-metric subplots) ---
    df_overall = df.copy()
    if "model_type_norm" in df_overall.columns and exclude_models:
        df_overall = df_overall[~df_overall["model_type_norm"].isin(exclude_models)]

    df_overall["model_type_norm"] = df_overall["model_type_norm"].map(model_labels)

    df_long = df_overall.melt(
        id_vars=["model_type_norm"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if df_long.empty:
        print("[INFO] No data for overall accuracy plot (df_long is empty).")
    else:
        metric_order = [m for m in metrics if m in metrics_present]

        g = sns.FacetGrid(
            df_long,
            col="metric",
            col_order=metric_order,
            sharey=False,           # <-- DIFFERENT SCALES
            height=3,
            aspect=1.1
        )
        g.map_dataframe(
            sns.pointplot,
            x="model_type_norm",
            y="value",
            hue="model_type_norm",
            dodge=0.4,
            errorbar=("ci", 95),
            join=False
        )

        # Clean up legends and labels
        for ax, metric in zip(g.axes.flat, metric_order):
            ax.set_title(metric_labels.get(metric, metric))
            ax.set_xlabel("")
            ax.set_ylabel("Metric Value")
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        out_path = outdir / "step2_overall_accuracy_metrics_by_modeltype.pdf"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved overall accuracy plot to: {out_path}")

    # Recompute orders based on filtered data
    model_type_order = sorted(
        df_devices['model_type_norm']
        .dropna()
        .unique()
    )
    hue_order = sorted(
        df_devices['traffic_model']
        .dropna()
        .unique()
    )

    # --- 6) PLOTS FOR FILTERED MODEL TYPES: all metrics as rows in one figure ---

    if not df_devices.empty:
        # Long format over metrics
        metric_order = [m for m in metrics if m in metrics_present]
        df_long_devices = df_devices.melt(
            id_vars=["model_type_norm", "num_devices_num", "traffic_model"],
            value_vars=metrics_present,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])

        print("Plotting num_devices influence for all metrics in one figure...")

        g = sns.relplot(
            data=df_long_devices,
            x="num_devices_num",
            y="value",
            hue="traffic_model",
            hue_order=hue_order,
            col="model_type_norm",
            col_order=model_type_order,
            row="metric",  # metrics stacked as rows
            row_order=metric_order,
            kind="line",
            marker="o",
            facet_kws={"sharex": True, "sharey": False},
            errorbar=("ci", 95),
            height=2.2,
            aspect=1.4,
        )

        # Set axis labels (metric label on y-axis only)
        for row_idx, metric in enumerate(metric_order):
            for ax in g.axes[row_idx]:
                ax.set_ylabel(metric_labels.get(metric, metric))

        g.set_axis_labels("Number of devices", None)

        # Column titles = model type only
        g.set_titles(col_template="{col_name}", row_template="")

        # Remove any leftover row titles
        for row in g.axes:
            for ax in row:
                if 'channel' in ax.get_title():
                    ax.set_title('Channel Model')
                if 'meta' in ax.get_title():
                    ax.set_title('Meta-Model')
                if 'static' in ax.get_title():
                    ax.set_title('Static Graph Model')
                # ax.set_title(ax.get_title().replace("metric = ", ""), loc="center")

        # Figure-level legend under all subplots
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        if g._legend is not None:
            g._legend.remove()
        g.fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(hue_order), 4),
            frameon=False,
            title="Traffic model",
        )

        plt.tight_layout(rect=[0, 0.15, 1, 1])  # leave space at bottom for legend
        out_path = outdir / "step2_compare_modeltypes_numdevices_allmetrics.pdf"
        g.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
    else:
        print("[INFO] No data with valid num_devices_num for any metric.")


if __name__ == "__main__":
    summarize_step2("../analysis_results/aggregated_results2.csv")

# plot_step2_dissertation.py

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

MODEL_LABELS: Dict[str, str] = {
    "meta_model": "Meta-Model",
    "channel": "Channel",
    "static_graph": "Static Graph",
}

EXCLUDE_MODELS: List[str] = ["ideal", "detailed"]

METRICS: List[str] = [
    "nrmse_mean",
    "mean_in_one_sigma_interval",
    "wasserstein_distance",
]

METRIC_LABELS: Dict[str, str] = {
    "nrmse_mean": "$NRMSE$",
    "mean_in_one_sigma_interval": r"$C_{\pm\sigma}$",
    "wasserstein_distance": "$W$",
}


# ---------------------------------------------------------------------------
# Utility / preprocessing
# ---------------------------------------------------------------------------

def configure_plot_style(font_size: int = 9) -> None:
    """Configure matplotlib / seaborn plotting style."""
    plt.rcParams.update(
        {
            "font.size": font_size,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    sns.set_theme(style="whitegrid", font="serif")


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV and normalize column names."""
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]

    print("=== Step 2 Overview ===")
    print(f"CSV file: {file_path.resolve()}")
    print(f"Rows (runs): {len(df)}")

    return df


def normalize_model_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized model_type column."""
    if "model_type" in df.columns:
        df["model_type_norm"] = (
            df["model_type"]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"meta model": "meta_model", "meta-model": "meta_model"})
        )
    else:
        df["model_type_norm"] = "unknown"
        print("\n[WARN] No 'model_type' column found. Using 'unknown' as model_type_norm.")

    return df


def add_basic_info(df: pd.DataFrame) -> None:
    """Print basic dataset information."""
    if "scenario_id" in df.columns:
        n_scenarios = df["scenario_id"].nunique()
        print(f"Unique scenarios: {n_scenarios}")
    else:
        print("Column 'scenario_id' not found (no scenario count shown).")

    print("\nModel type counts (all):")
    print(df["model_type_norm"].value_counts(dropna=False))

    if {"model_type", "substitution_occurred"}.issubset(df.columns):
        meta = df[df["model_type"] == "meta_model"]
        print(
            "Number of scenarios in which substitution occurred: ",
            len(meta[meta["substitution_occurred"] == True]),
        )
        print(
            "Number of scenarios in which substitution not occurred: ",
            len(meta[meta["substitution_occurred"] == False]),
        )


def validate_required_columns(df: pd.DataFrame) -> None:
    """Ensure required columns for plotting are present."""
    required_cols = ["traffic_configuration", "num_devices", "scenario_duration"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for plots: {missing}")


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add traffic_model and numeric num_devices columns."""
    # Traffic model
    df["traffic_model"] = (
        df["traffic_configuration"]
        .astype(str)
        .str.replace("TrafficConfig.", "", regex=False)
        .str.replace("evaluation_", "", regex=False)
        .str.strip()
    )

    # num_devices_num
    dev_map = {"five": 5, "ten": 10, "twenty": 20, "fifty": 50}
    df["num_devices_num"] = (
        df["num_devices"]
        .astype(str)
        .str.replace("NumDevices.", "", regex=False)
        .map(dev_map)
    )

    return df


def prepare_overall_view(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare df_overall: drop excluded models and map model_type_norm to labels.
    Returns (df_overall, metrics_present).
    """
    df_overall = df.copy()

    if "model_type_norm" in df_overall.columns and EXCLUDE_MODELS:
        df_overall = df_overall[~df_overall["model_type_norm"].isin(EXCLUDE_MODELS)]

    metrics_present = [m for m in METRICS if m in df_overall.columns]
    if not metrics_present:
        raise ValueError(
            "None of the expected metric columns "
            "['nrmse_mean', 'mean_in_one_sigma_interval', 'wasserstein_distance'] "
            "are present in the CSV."
        )

    # Keep a copy of normalized model_type for filtering, but also map for plotting
    df_overall["model_type_norm"] = df_overall["model_type_norm"].map(MODEL_LABELS)

    return df_overall, metrics_present


def prepare_devices_subset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Subset rows where num_devices_num is available and prepare orders."""
    df_devices = df.dropna(subset=["num_devices_num"]).copy()
    df_devices = df_devices[~df_devices["model_type_norm"].isin(EXCLUDE_MODELS)]

    model_type_order = sorted(df_devices["model_type_norm"].dropna().unique())
    hue_order = sorted(df_devices["traffic_model"].dropna().unique())

    return df_devices, model_type_order, hue_order


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_overall_accuracy(
    df_overall: pd.DataFrame,
    metrics_present: List[str],
    outdir: Path,
) -> None:
    """Plot overall accuracy metrics across all scenarios (per metric)."""
    df_long = df_overall.melt(
        id_vars=["model_type_norm"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if df_long.empty:
        print("[INFO] No data for overall accuracy plot (df_long is empty).")
        return

    metric_order = [m for m in METRICS if m in metrics_present]

    configure_plot_style(font_size=9)
    g = sns.FacetGrid(
        df_long,
        col="metric",
        col_order=metric_order,
        sharey=False,
        height=3,
        aspect=1.1,
    )
    g.map_dataframe(
        sns.pointplot,
        x="model_type_norm",
        y="value",
        hue="model_type_norm",
        dodge=0.4,
        errorbar=("ci", 95),
        linestyle="none",  # keine Verbindungslinie
    )

    for ax, metric in zip(g.axes.flat, metric_order):
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    out_path = outdir / "step2_overall_accuracy_metrics_by_modeltype.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved overall accuracy plot to: {out_path}")


def plot_accuracy_by_network_model(
    df: pd.DataFrame,
    metrics_present: List[str],
    outdir: Path,
) -> None:
    """Plot accuracy metrics by communication network model and model type."""
    if "network_type" not in df.columns:
        print("[WARN] Column 'network_type' not found – skipping network-model accuracy plot.")
        return

    df_net = df.copy()

    if "model_type_norm" in df_net.columns and EXCLUDE_MODELS:
        df_net = df_net[~df_net["model_type_norm"].isin(EXCLUDE_MODELS)]

    df_net["model_type_norm"] = df_net["model_type_norm"].map(MODEL_LABELS)

    df_net["network_model"] = (
        df_net["network_type"]
        .astype(str)
        .str.replace("NetworkModelType.", "", regex=False)
        .str.replace("evaluation_", "", regex=False)
        .str.strip()
    )

    df_long_net = df_net.melt(
        id_vars=["model_type_norm", "network_model"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if df_long_net.empty:
        print("[INFO] No data for accuracy-by-network-model plot (df_long_net is empty).")
        return

    metric_order = [m for m in METRICS if m in metrics_present]
    network_order = sorted(df_long_net["network_model"].dropna().unique())

    configure_plot_style(font_size=9)
    palette = {
        "Channel": "#4C72B0",
        "Meta-Model": "#DD8452",
        "Static Graph": "#55A868",
    }

    g_net = sns.FacetGrid(
        df_long_net,
        row="metric",
        row_order=metric_order,
        col="network_model",
        col_order=network_order,
        sharey=False,
        height=2.2,
        aspect=1.1,
    )
    g_net.map_dataframe(
        sns.pointplot,
        x="model_type_norm",
        y="value",
        hue="model_type_norm",
        dodge=0.4,
        errorbar=("ci", 95),
        linestyle="none",  # keine Verbindungslinie
        palette=palette,
    )

    for row_idx, metric in enumerate(metric_order):
        for ax in g_net.axes[row_idx]:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=60)

    g_net.set_titles(col_template="{col_name}", row_template="")

    # Clean network titles
    for ax in g_net.axes.flatten():
        title = ax.get_title()
        if "5g" in title:
            ax.set_title("5G")
        elif "ethernet" in title:
            ax.set_title("Ethernet")
        elif "lte450" in title:
            ax.set_title("LTE450")
        elif "lte" in title:
            ax.set_title("LTE")

    # Manual legend
    model_types_unique = df_long_net["model_type_norm"].unique()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            label=m,
        )
        for m in model_types_unique
    ]

    g_net.fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(model_types_unique), 3),
        frameon=False,
        title="Model type",
    )

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    out_path = outdir / "step2_accuracy_by_networkmodel_and_modeltype.pdf"
    g_net.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g_net.fig)
    print(f"Saved accuracy-by-network-model plot to: {out_path}")


def plot_metric_distributions_all_models(
    df_overall: pd.DataFrame,
    outdir: Path,
) -> None:
    """Plot distributions of metrics for all models in one figure (one row per model)."""
    available_metrics = [m for m in METRICS if m in df_overall.columns]
    if not available_metrics:
        print("No valid metrics to plot for distributions.")
        return

    # Only models present in df_overall
    valid_models = [
        model_i
        for model_i in MODEL_LABELS.keys()
        if (df_overall["model_type"] == model_i).any()
    ]
    if not valid_models:
        print("No valid models to plot for distributions.")
        return

    configure_plot_style(font_size=10)

    n_models = len(valid_models)
    n_metrics = len(available_metrics)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=n_metrics,
        figsize=(4 * n_metrics, 2.5 * n_models),
        sharex=False,
        sharey=False,
    )

    # Ensure axes is 2D
    if n_models == 1 and n_metrics == 1:
        axes = np.array([[axes]])
    elif n_models == 1:
        axes = np.array([axes])
    elif n_metrics == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, model_i in enumerate(valid_models):
        model_n = MODEL_LABELS[model_i]
        df_distr = df_overall[df_overall["model_type"] == model_i]
        df_plot = df_distr[available_metrics].dropna()

        if df_plot.empty:
            for col_idx in range(n_metrics):
                axes[row_idx, col_idx].set_visible(False)
            continue

        for col_idx, metric in enumerate(available_metrics):
            ax = axes[row_idx, col_idx]
            sns.histplot(
                df_plot[metric],
                bins=100,
                kde=True,
                stat="percent",
                ax=ax,
            )

            ax.set_xlabel(METRIC_LABELS.get(metric, metric))

            if col_idx == 0:
                ax.set_ylabel("Scenarios [%]")
            else:
                ax.set_ylabel("")

            if metric == "nrmse_mean":
                ax.set_xlim([-10, 50])
            elif metric == "mean_in_one_sigma_interval":
                ax.set_xlim([-0.1, 1.1])
            elif metric == "wasserstein_distance":
                ax.set_xlim([-100, 1000])

            if col_idx == 0:
                ax.set_title(model_n, loc="left")
            else:
                ax.set_title("")

    plt.tight_layout()
    out_path = outdir / "step2_distribution_metrics_all_models.pdf"
    plt.savefig(out_path, format="pdf")
    plt.close(fig)
    print(f"Saved distribution plot for all models to: {out_path}")


def plot_meta_model_by_test_train_split(
    df_overall: pd.DataFrame,
    metrics_present: List[str],
    outdir: Path,
) -> None:
    """Plot meta-model accuracy metrics by test-train split."""
    if "test_train_name" not in df_overall.columns:
        print("[WARN] Column 'test_train_name' not found – skipping meta-model test-train split plot.")
        return

    df_meta_split = df_overall.copy()
    df_meta_split = df_meta_split[df_meta_split["model_type_norm"] == "Meta-Model"]

    if df_meta_split.empty:
        print("[INFO] No meta-model data found for test-train split plot.")
        return

    split_label_map = {
        "none": "No split",
        "traffic_model_split": "Traffic model split",
        "scale_split": "Scale split",
        "traffic_load_split": "Traffic load split",
        "parametrization_split": "Parametrization split",
        "technology_split": "Technology split",
    }

    df_meta_split["test_train_label"] = (
        df_meta_split["test_train_name"]
        .astype(str)
        .map(split_label_map)
        .fillna(df_meta_split["test_train_name"])
    )

    metric_order = [m for m in METRICS if m in metrics_present]

    df_long_meta_split = df_meta_split.melt(
        id_vars=["test_train_label"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if df_long_meta_split.empty:
        print("[INFO] No data for meta-model test-train split plot.")
        return

    split_order = [
        lbl
        for _, lbl in split_label_map.items()
        if lbl in df_long_meta_split["test_train_label"].unique()
    ]
    if not split_order:
        split_order = sorted(df_long_meta_split["test_train_label"].unique())

    configure_plot_style(font_size=9)
    g_split = sns.FacetGrid(
        df_long_meta_split,
        col="metric",
        col_order=metric_order,
        sharey=False,
        height=3,
        aspect=1.1,
    )
    g_split.map_dataframe(
        sns.barplot,
        x="test_train_label",
        y="value",
        order=split_order,
        errorbar=("ci", 95),
    )

    for ax, metric in zip(g_split.axes.flat, metric_order):
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("")
        ax.set_ylabel("Metric value")
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    plt.tight_layout()
    out_path = outdir / "step2_meta_model_accuracy_by_test_train_split.pdf"
    g_split.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g_split.fig)
    print(f"Saved meta-model test-train split plot to: {out_path}")


def plot_num_devices_influence(
    df_devices: pd.DataFrame,
    metrics_present: List[str],
    model_type_order: List[str],
    hue_order: List[str],
    outdir: Path,
) -> None:
    """Plot influence of number of devices on metrics (rows) and model types (columns)."""
    if df_devices.empty:
        print("[INFO] No data with valid num_devices_num for any metric.")
        return

    metric_order = [m for m in METRICS if m in metrics_present]
    df_long_devices = df_devices.melt(
        id_vars=["model_type_norm", "num_devices_num", "traffic_model"],
        value_vars=metrics_present,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    print("Plotting num_devices influence for all metrics in one figure...")

    configure_plot_style(font_size=9)
    g = sns.relplot(
        data=df_long_devices,
        x="num_devices_num",
        y="value",
        hue="traffic_model",
        hue_order=hue_order,
        col="model_type_norm",
        col_order=model_type_order,
        row="metric",
        row_order=metric_order,
        kind="line",
        marker="o",
        facet_kws={"sharex": True, "sharey": False},
        errorbar=("ci", 95),
        height=2.2,
        aspect=1.4,
    )

    for row_idx, metric in enumerate(metric_order):
        for ax in g.axes[row_idx]:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    g.set_axis_labels("Number of devices", None)
    g.set_titles(col_template="{col_name}", row_template="")

    for row in g.axes:
        for ax in row:
            title = ax.get_title().lower()
            if "channel" in title:
                ax.set_title("Channel Model")
            elif "meta" in title:
                ax.set_title("Meta-Model")
            elif "static" in title:
                ax.set_title("Static Graph Model")

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

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    out_path = outdir / "step2_compare_modeltypes_numdevices_allmetrics.pdf"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved num-devices influence plot to: {out_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def summarize_step2(file_path: str) -> None:
    """
    Step-2 overview + plots:
    - Basic info about the CSV
    - Normalization of model_type
    - Plots for selected model types:
        * Overall accuracy metrics
        * Accuracy by network model
        * Metric distributions (all models)
        * Meta-model accuracy by test-train split
        * Influence of num_devices on metrics
    """
    outdir = Path("../analysis_results/plots_phase2")
    outdir.mkdir(parents=True, exist_ok=True)

    # Load & basic info
    df = load_data(file_path)
    df = normalize_model_type(df)
    add_basic_info(df)
    validate_required_columns(df)
    df = add_derived_columns(df)

    # Prepare views
    df_overall, metrics_present = prepare_overall_view(df)
    df_devices, model_type_order, hue_order = prepare_devices_subset(df)

    # Plots
    plot_overall_accuracy(df_overall, metrics_present, outdir)
    plot_accuracy_by_network_model(df, metrics_present, outdir)
    plot_metric_distributions_all_models(df_overall, outdir)
    plot_meta_model_by_test_train_split(df_overall, metrics_present, outdir)
    plot_num_devices_influence(
        df_devices, metrics_present, model_type_order, hue_order, outdir
    )


if __name__ == "__main__":
    summarize_step2("../analysis_results/aggregated_results2.csv")

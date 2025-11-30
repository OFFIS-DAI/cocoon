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
        if metric == 'nrmse_mean':
            ax.set_ylim([0, 1])
        elif metric == 'mean_in_one_sigma_interval':
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim([0, 300])

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
                ax.set_xlim([-1, 10])
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


def plot_heatmap_scenarios_all_models(
        df: pd.DataFrame,
        metrics_present: List[str],
        outdir: Path,
) -> None:
    """
    Heatmaps: alle Metriken für alle Modelle, gruppiert nach
    Network Technology, Traffic, Number Devices und Duration.

    - Zeilen: konkrete Szenario-Kombination (net | traffic | devs | duration)
    - Spalten: Modelle
    - Panel: Metrik

    Meta-Model-Szenarien mit Substitution werden im Label mit '•' markiert.
    """
    df_hm = df.copy()
    df_hm = df_hm[~df_hm["model_type"].isin(EXCLUDE_MODELS)]

    if df_hm.empty:
        print("[INFO] No data for heatmap (all models, all factors).")
        return

    # Network Technology (bereinigt)
    df_hm["network_model"] = (
        df_hm["network_type"]
        .astype(str)
        .str.replace("NetworkModelType.", "", regex=False)
        .str.replace("evaluation_", "", regex=False)
        .str.strip()
    )

    # Traffic
    if "traffic_model" not in df_hm.columns:
        df_hm["traffic_model"] = (
            df_hm["traffic_configuration"]
            .astype(str)
            .str.replace("TrafficConfig.", "", regex=False)
            .str.replace("evaluation_", "", regex=False)
            .str.strip()
        )

    # Number of devices (numerisch)
    if "num_devices_num" not in df_hm.columns:
        dev_map = {"five": 5, "ten": 10, "twenty": 20, "fifty": 50}
        df_hm["num_devices_num"] = (
            df_hm["num_devices"]
            .astype(str)
            .str.replace("NumDevices.", "", regex=False)
            .map(dev_map)
        )

    # Duration-Label
    df_hm["duration_label"] = (
        df_hm["scenario_duration"]
        .astype(str)
        .str.replace("ScenarioDuration.", "", regex=False)
    )
    duration_map = {
        "one_min": "1 min",
        "five_min": "5 min",
        "ten_min": "10 min",
        "thirty_min": "30 min",
    }
    df_hm["duration_label"] = (
        df_hm["duration_label"].map(duration_map).fillna(df_hm["duration_label"])
    )

    # Nur vollständige Szenarien
    df_hm = df_hm.dropna(
        subset=["network_model", "traffic_model", "num_devices_num", "duration_label"]
    )
    if df_hm.empty:
        print("[INFO] No complete scenarios for heatmap (all factors).")
        return

    metric_order = [m for m in METRICS if m in metrics_present]
    if not metric_order:
        print("[INFO] No metrics present for heatmap (all factors).")
        return

    # Szenario-Kombinationslabel (interne ID)
    df_hm["scenario_label"] = (
            df_hm["network_model"]
            + " | "
            + df_hm["traffic_model"]
            + " | "
            + df_hm["num_devices_num"].astype(int).astype(str)
            + " dev | "
            + df_hm["duration_label"]
    )

    # Sortierung und feste Reihenfolge
    df_hm = df_hm.sort_values(
        ["network_model", "traffic_model", "num_devices_num", "duration_label"]
    )
    scenario_order = df_hm["scenario_label"].drop_duplicates().tolist()

    # Mapping: interne Label -> "01  5g | ..."
    scenario_display_map = {
        lab: f"{idx + 1:02d}  {lab}" for idx, lab in enumerate(scenario_order)
    }

    # Meta-Model-Substitution pro Szenario
    subst_by_scenario = None
    if "substitution_occurred" in df_hm.columns:
        df_meta = df_hm[df_hm["model_type"] == "meta_model"].copy()
        if not df_meta.empty:
            subst_by_scenario = (
                df_meta.groupby("scenario_label")["substitution_occurred"]
                .any()
                .reindex(scenario_order)
                .fillna(False)
            )

    # Plot-Layout
    n_metrics = len(metric_order)
    configure_plot_style(font_size=8)

    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=1,
        figsize=(8, max(4, 0.20 * len(scenario_order) * n_metrics)),
        sharex=False,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_order):
        grouped = (
            df_hm.groupby(
                ["scenario_label", "model_type"],
                as_index=False,
            )[metric]
            .mean()
        )

        grouped["scenario_label"] = pd.Categorical(
            grouped["scenario_label"],
            categories=scenario_order,
            ordered=True,
        )

        pivot = grouped.pivot(
            index="scenario_label",
            columns="model_type",
            values=metric,
        )
        pivot.columns = [MODEL_LABELS.get(c, c) for c in pivot.columns]

        # Colormap je Metrik: C_{±σ} umgedreht
        if metric == "mean_in_one_sigma_interval":
            cmap = "viridis_r"
        else:
            cmap = "viridis"

        # Heatmap mit deutlicheren Linien
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            cbar=True,
            linewidths=0.5,
            linecolor="black",
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 5},
        )

        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(METRIC_LABELS.get(metric, metric), loc="left")
        ax.tick_params(axis="y", labelsize=6, pad=2)

        # Y-Tick-Positionen explizit setzen (eine pro Heatmap-Zeile)
        n_rows = pivot.shape[0]
        ax.set_yticks(np.arange(n_rows) + 0.5)

        # Zu jedem internen Szenario-Label ein Anzeige-Label bauen
        yticklabels = []
        for internal in pivot.index:
            disp = scenario_display_map.get(internal, internal)
            if subst_by_scenario is not None and subst_by_scenario.get(internal, False):
                disp = "• " + disp
            yticklabels.append(disp)

        ax.set_yticklabels(yticklabels, fontsize=6)

    if subst_by_scenario is not None:
        fig.text(
            0.01,
            0.01,
            "•  scenario with substitution in Meta-Model",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    plt.tight_layout()
    out_path = outdir / "step2_heatmap_metrics_by_scenario_and_model.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scenario-factor heatmap (all models, all metrics) to: {out_path}")


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

    for n_r, row in enumerate(g.axes):
        for ax in row:
            title = ax.get_title().lower()
            if "channel" in title:
                ax.set_title("Channel Model")
            elif "meta" in title:
                ax.set_title("Meta-Model")
            elif "static" in title:
                ax.set_title("Static Graph Model")
            if n_r == 0:
                # NRMSE
                ax.set_ylim([0, 3])
            if n_r == 1:
                # C sigma
                ax.set_ylim([0, 1])
            if n_r == 2:
                # W
                ax.set_ylim([0, 4000])

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
    out_path = outdir / "step2_compare_modeltypes_numdevices_allmetrics.svg"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved num-devices influence plot to: {out_path}")


def analyze_meta_model_substitution(
        df: pd.DataFrame,
        metrics_present: List[str],
        outdir: Path,
) -> None:
    """
    Analyse substitution behaviour of the meta-model:

    - Substitution rates by network technology
    - Impact of substitution on meta-model accuracy metrics
    """
    if "substitution_occurred" not in df.columns:
        print("[WARN] Column 'substitution_occurred' not found – skipping substitution analysis.")
        return

    # Only meta-model rows
    df_meta = df[df["model_type"] == "meta_model"].copy()
    if df_meta.empty:
        print("[INFO] No meta-model rows found – skipping substitution analysis.")
        return

    # Clean network model label (reuse scheme from other plots)
    if "network_type" in df_meta.columns:
        df_meta["network_model"] = (
            df_meta["network_type"]
            .astype(str)
            .str.replace("NetworkModelType.", "", regex=False)
            .str.replace("evaluation_", "", regex=False)
            .str.strip()
        )
    else:
        df_meta["network_model"] = "unknown"

    # ------------------------------------------------------------------
    # 1) Substitution rate by network technology
    # ------------------------------------------------------------------
    subst_by_net = (
        df_meta.groupby("network_model")["substitution_occurred"]
        .mean()
        .reset_index(name="substitution_rate")
    )

    configure_plot_style(font_size=9)
    plt.figure(figsize=(4, 2.8))
    sns.barplot(
        data=subst_by_net,
        x="network_model",
        y="substitution_rate",
        errorbar=("ci", 95),
    )
    plt.ylim(0, 1)
    plt.ylabel("Substitution rate")
    plt.xlabel("Network technology")
    plt.title("Meta-Model substitution rate by technology")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = outdir / "step2_meta_substitution_rate_by_network.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved meta-model substitution-by-network plot to: {out_path}")

    # ------------------------------------------------------------------
    # 2) Impact of substitution on meta-model accuracy metrics
    # ------------------------------------------------------------------
    metrics_for_plot = [m for m in METRICS if m in metrics_present]

    if not metrics_for_plot:
        print("[INFO] No metrics present for substitution accuracy analysis.")
        return

    df_long = df_meta.melt(
        id_vars=["substitution_occurred"],
        value_vars=metrics_for_plot,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    if df_long.empty:
        print("[INFO] No data for substitution-related accuracy plot.")
        return

    configure_plot_style(font_size=9)
    g = sns.FacetGrid(
        df_long,
        col="metric",
        col_order=metrics_for_plot,
        sharey=False,
        height=3,
        aspect=1.0,
    )
    g.map_dataframe(
        sns.pointplot,
        x="substitution_occurred",
        y="value",
        hue="substitution_occurred",
        dodge=0.3,
        errorbar=("ci", 95),
        linestyle="none",
    )

    for ax, metric in zip(g.axes.flat, metrics_for_plot):
        ax.set_xlabel("Substitution occurred")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_xticklabels(["No", "Yes"])
        ax.set_title(METRIC_LABELS.get(metric, metric))
        if metric == 'nrmse_mean':
            ax.set_ylim([0, 1])
        elif metric == 'mean_in_one_sigma_interval':
            ax.set_ylim([0, 1])
        else:
            ax.set_ylim([0, 300])

    if g._legend is not None:
        g._legend.remove()

    plt.tight_layout()
    out_path = outdir / "step2_meta_accuracy_by_substitution.pdf"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved meta-model accuracy-by-substitution plot to: {out_path}")


def plot_meta_model_relative_accuracy(
        df: pd.DataFrame,
        metrics_present: List[str],
        outdir: Path,
) -> None:
    """
    Compare meta-model accuracy to other models on a per-scenario basis.

    For each metric and scenario, compute a directional advantage:
      > 0  => Meta-Model better
      < 0  => Meta-Model worse
    """
    if "scenario_id" not in df.columns:
        print("[WARN] Column 'scenario_id' not found – skipping relative accuracy analysis.")
        return

    # Need at least meta_model + one other model
    if "meta_model" not in df["model_type"].unique():
        print("[INFO] No meta_model entries – skipping relative accuracy analysis.")
        return

    metrics_for_plot = [m for m in METRICS if m in metrics_present]
    if not metrics_for_plot:
        print("[INFO] No metrics present for relative accuracy analysis.")
        return

    # Direction: +1 => larger is better, -1 => smaller is better
    direction_map = {
        "nrmse_mean": -1,
        "wasserstein_distance": -1,
        "mean_in_one_sigma_interval": 1,
    }

    all_diffs = []

    for metric in metrics_for_plot:
        df_metric = df[["scenario_id", "model_type", metric]].dropna()
        if df_metric.empty:
            continue

        pivot = df_metric.pivot_table(
            index="scenario_id",
            columns="model_type",
            values=metric,
            aggfunc="mean",
        )

        if "meta_model" not in pivot.columns:
            continue

        for other in pivot.columns:
            if other == "meta_model":
                continue

            # Only scenarios where both are available
            valid_rows = pivot[["meta_model", other]].dropna()
            if valid_rows.empty:
                continue

            diff = valid_rows["meta_model"] - valid_rows[other]
            direction = direction_map.get(metric, -1)
            directional_diff = diff * direction

            tmp = pd.DataFrame(
                {
                    "scenario_id": valid_rows.index,
                    "metric": metric,
                    "other_model": MODEL_LABELS.get(other, other),
                    "relative_advantage": directional_diff,
                }
            )
            all_diffs.append(tmp)

    if not all_diffs:
        print("[INFO] No valid pairs for relative accuracy analysis.")
        return

    df_diff = pd.concat(all_diffs, ignore_index=True)

    configure_plot_style(font_size=9)
    g = sns.FacetGrid(
        df_diff,
        col="metric",
        col_order=metrics_for_plot,
        sharey=False,
        height=3,
        aspect=1.0,
    )
    g.map_dataframe(
        sns.boxplot,
        x="other_model",
        y="relative_advantage",
    )

    for ax, metric in zip(g.axes.flat, metrics_for_plot):
        ax.axhline(0.0, linestyle="--", linewidth=1, color="gray")
        ax.set_xlabel("Reference model")
        ax.set_ylabel("Meta-Model advantage")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.tick_params(axis="x", rotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    plt.tight_layout()
    out_path = outdir / "step2_meta_relative_accuracy_vs_models.pdf"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved meta-model relative-accuracy plot to: {out_path}")


def plot_accuracy_performance_tradeoff(
        df_overall: pd.DataFrame,
        metrics_present: List[str],
        outdir: Path,
) -> None:
    """
    Analyse trade-off between accuracy and performance efficiency.

    - Uses the first available runtime column (runtime / runtime_seconds / execution_time)
    - Scatter plots of accuracy metric vs. runtime per model type.
    """
    # Try to detect a runtime column heuristically
    candidate_cols = [
        c
        for c in df_overall.columns
        if any(k in c.lower() for k in ["runtime", "exec_time", "execution_time"])
    ]
    if not candidate_cols:
        print("[WARN] No runtime column found – skipping accuracy vs. performance trade-off analysis.")
        return

    runtime_col = candidate_cols[0]
    print(f"[INFO] Using '{runtime_col}' as runtime column for trade-off analysis.")

    metrics_for_plot = [m for m in METRICS if m in metrics_present]
    if not metrics_for_plot:
        print("[INFO] No metrics present for trade-off analysis.")
        return

    df_tradeoff = df_overall.dropna(subset=[runtime_col]).copy()
    if df_tradeoff.empty:
        print("[INFO] No rows with runtime values – skipping trade-off analysis.")
        return

    # Long format: one row per (run, metric)
    df_long = df_tradeoff.melt(
        id_vars=["model_type_norm", runtime_col],
        value_vars=metrics_for_plot,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])

    configure_plot_style(font_size=9)
    g = sns.FacetGrid(
        df_long,
        col="metric",
        col_order=metrics_for_plot,
        sharex=False,
        sharey=False,
        height=3,
        aspect=1.1,
        hue="model_type_norm",
    )
    g.map_dataframe(
        sns.scatterplot,
        x=runtime_col,
        y="value",
        alpha=0.7,
    )

    for ax, metric in zip(g.axes.flat, metrics_for_plot):
        ax.set_xscale("log")
        ax.set_xlabel("Runtime [log scale]")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric))

    g.add_legend(title="Model")

    plt.tight_layout()
    out_path = outdir / "step2_accuracy_vs_runtime_tradeoff.pdf"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved accuracy vs. performance trade-off plot to: {out_path}")


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
    plot_heatmap_scenarios_all_models(df, metrics_present, outdir)
    plot_overall_accuracy(df_overall, metrics_present, outdir)
    plot_accuracy_by_network_model(df, metrics_present, outdir)
    plot_metric_distributions_all_models(df_overall, outdir)
    plot_meta_model_by_test_train_split(df_overall, metrics_present, outdir)
    plot_num_devices_influence(
        df_devices, metrics_present, model_type_order, hue_order, outdir
    )

    analyze_meta_model_substitution(df, metrics_present, outdir)
    plot_meta_model_relative_accuracy(df, metrics_present, outdir)
    plot_accuracy_performance_tradeoff(df_overall, metrics_present, outdir)

    plot_num_devices_influence(
        df_devices, metrics_present, model_type_order, hue_order, outdir
    )

    plot_num_devices_influence(
        df_devices, metrics_present, model_type_order, hue_order, outdir
    )


if __name__ == "__main__":
    summarize_step2("../analysis_results/aggregated_results2.csv")

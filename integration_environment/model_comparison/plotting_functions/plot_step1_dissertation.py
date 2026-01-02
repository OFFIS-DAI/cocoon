import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def summarize_step1(file_path: str) -> None:
    # --- 1) Read CSV ---
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    # Basic sanity checks / normalization
    df.columns = [c.strip() for c in df.columns]
    if 'model_type' not in df.columns or 'execution_time_s' not in df.columns:
        raise ValueError("CSV must contain at least 'model_type' and 'execution_time_s' columns.")

    # Normalize model_type values
    df['model_type_norm'] = (
        df['model_type'].astype(str)
        .str.strip().str.lower()
        .replace({'meta model': 'meta_model', 'meta-model': 'meta_model'})
    )

    # Optional columns we’ll use if present
    has_subst_name = 'substitution_name' in df.columns
    has_subst_flag = 'substitution_occurred' in df.columns
    has_timeout = 'timeout_occurred' in df.columns
    has_scenario = 'scenario_id' in df.columns

    # --- 2) Filter to substitution enabled (meta-model only) ---
    if has_subst_name:
        df_sub_enabled = df[
            (df['model_type_norm'] == 'meta_model')
            & (df['substitution_name'].astype(str).str.strip().str.lower() == 'enabled')
        ].copy()
    else:
        df_sub_enabled = df[df['model_type_norm'] == 'meta_model'].copy()

    # --- 3) Overall summary ---

    # 3a) Number of scenarios (unique scenario_id) and runs in the filtered subset
    n_runs_filtered = len(df_sub_enabled)
    n_scenarios_filtered = df_sub_enabled['scenario_id'].nunique() if has_scenario else None

    # 3b) Mean execution time: detailed vs meta-model (overall, not filtered)
    mean_et_detailed_all = (
        df.loc[df['model_type_norm'] == 'detailed', 'execution_time_s'].mean()
        if (df['model_type_norm'] == 'detailed').any() else float('nan')
    )
    mean_et_meta_all = (
        df.loc[df['model_type_norm'] == 'meta_model', 'execution_time_s'].mean()
        if (df['model_type_norm'] == 'meta_model').any() else float('nan')
    )

    # 3c) Mean execution time matched on the filtered subset’s scenarios (if scenario_id available)
    if has_scenario:
        scenario_ids = df_sub_enabled['scenario_id'].unique()
        meta_on_subset = df[
            (df['model_type_norm'] == 'meta_model') & (df['scenario_id'].isin(scenario_ids))
        ]['execution_time_s']
        mean_et_meta_subset = meta_on_subset.mean() if not meta_on_subset.empty else float('nan')
    else:
        mean_et_meta_subset = float('nan')

    # 3d) Substitution statistics for meta-model (filtered subset)
    if has_subst_flag:
        subst_success_rate = df_sub_enabled['substitution_occurred'].mean()
        subst_success_n = df_sub_enabled['substitution_occurred'].sum()
        subst_total_n = len(df_sub_enabled)
    else:
        subst_success_rate = float('nan')
        subst_success_n = 0
        subst_total_n = len(df_sub_enabled)

    timeout_rate = df_sub_enabled['timeout_occurred'].mean() if has_timeout else float('nan')
    timeout_n = df_sub_enabled['timeout_occurred'].sum() if has_timeout else 0

    # --- Print summary ---
    print("=== Step 1 Summary ===")
    print(f"CSV file: {file_path.resolve()}")
    print("\n-- Filter: substitution enabled (meta-model) --")
    print(f"Runs in filtered subset: {n_runs_filtered}")
    if n_scenarios_filtered is not None:
        print(f"Unique scenarios in filtered subset: {n_scenarios_filtered}")

    print("\n-- Execution Time (seconds) --")
    print(f"Overall mean ET (Detailed):  {mean_et_detailed_all:.4f}" if pd.notna(mean_et_detailed_all) else "Overall mean ET (Detailed):  n/a")

    if has_scenario:
        print(f"Subset mean ET (Meta-model; scenarios in filtered subset): {mean_et_meta_subset:.4f}" if pd.notna(mean_et_meta_subset) else "Subset mean ET (Meta-model; scenarios in filtered subset): n/a")

    print("\n-- Substitution (Meta-model; filtered subset) --")
    if has_subst_flag:
        print(f"Substitution success: {subst_success_n}/{subst_total_n} ({subst_success_rate*100:.2f}%)")
    else:
        print("Substitution success: n/a (column 'substitution_occurred' missing)")

    if has_timeout:
        print(f"Timeouts: {timeout_n}/{n_runs_filtered} ({timeout_rate*100:.2f}%)")
    else:
        print("Timeouts: n/a (column 'timeout_occurred' missing)")

    # --- 4) Plot metric distributions for the filtered meta-model runs ---
    metrics = [
        "nrmse_mean",
        "wasserstein_distance",
        "mean_in_one_sigma_interval",
        "execution_time_s",
        "score",
    ]
    metric_names = {
        "nrmse_mean": "$NRMSE$",
        "wasserstein_distance": "$W$",
        "mean_in_one_sigma_interval": r"$C_{\pm\sigma}$",
        "execution_time_s": "$ET$",
        "score": "$SC$",
    }
    available_metrics = [m for m in metrics if m in df_sub_enabled.columns]
    df_plot = df_sub_enabled[available_metrics].dropna()

    if not df_plot.empty:
        plt.rcParams.update({
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        })

        sns.set_theme(style="whitegrid", font='serif')
        n = len(available_metrics)
        ncols = 5
        nrows = 1

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 3))
        axes = axes.flatten() if n > 1 else [axes]

        for i, (ax, metric) in enumerate(zip(axes, available_metrics)):
            sns.histplot(df_plot[metric], bins=30, kde=True, ax=ax)
            ax.set_xlabel(metric_names[metric])
            if i==0:
                ax.set_ylabel("Number of scenarios")
            else:
                ax.set_ylabel("")

        for j in range(len(available_metrics), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig('../analysis_results/plots_phase1/step1_distribution_metrics.pdf', format='pdf')
    else:
        print("No valid metrics to plot for the filtered subset.")

    # --- 5) Heatmap: absolute + standardized metric values per hyper-parameter level (filtered meta-model subset) ---
    if not df_plot.empty:
        candidate_hparams = [
            "cluster_distance_threshold",  # C-DT
            "batch_size_ipupa",  # C-IP
            "learning_rate_weighting",  # C-LR
            "butterfly_threshold_value",  # C-BT
            "substitution_priority_name",  # C-SP (categorical)
        ]
        hparams_present = [c for c in candidate_hparams if c in df_sub_enabled.columns]

        metric_cols = [m for m in metrics if m in df_sub_enabled.columns]
        level_tables = []
        group_sizes = []  # number of levels per hyperparameter, for separator lines
        group_keys = []  # hyperparameter names in the order they’re appended

        def _format_val(v):
            if isinstance(v, (int, np.integer)):
                return f"{int(v)}"
            if isinstance(v, (float, np.floating)):
                return f"{v:.3g}"
            return str(v).strip().lower()

        for col in hparams_present:
            grp = df_sub_enabled.groupby(col, dropna=False)[metric_cols].mean()
            grp.index = [f"{col}={_format_val(v)}" for v in grp.index]
            level_tables.append(grp)
            group_sizes.append(len(grp))
            group_keys.append(col)

        if level_tables:
            abs_table = pd.concat(level_tables, axis=0)

            # Pretty axis labels (LaTeX-style)
            metric_names = {
                "nrmse_mean": r"$NRMSE$",
                "wasserstein_distance": r"$W$",
                "mean_in_one_sigma_interval": r"$C_{\pm\sigma}$",
                "execution_time_s": r"$ET$",
                "score": r"$SC$",
            }
            hyperparam_names = {
                "cluster_distance_threshold": r"$C\text{-}DT$",
                "batch_size_ipupa": r"$C\text{-}IP$",
                "learning_rate_weighting": r"$C\text{-}LR$",
                "butterfly_threshold_value": r"$C\text{-}BT$",
                "substitution_priority_name": r"$C\text{-}SP$"
            }

            def _row_label(s: str) -> str:
                k, v = s.split("=", 1)
                return rf"{hyperparam_names.get(k, k)}={v}"

            row_labels = [_row_label(s) for s in abs_table.index]
            col_labels = [metric_names.get(c, c) for c in abs_table.columns]

            # Standardized version (z-score per metric column); handle zero-variance safely
            std = abs_table.std(ddof=0).replace(0, np.nan)
            abs_table_std = (abs_table - abs_table.mean()) / std
            abs_table_std = abs_table_std.fillna(0.0)
            abs_table_std['execution_time_s'] *= -1
            abs_table_std['wasserstein_distance'] *= -1

            # Styling (Computer Modern look without usetex)
            plt.rcParams.update({
                "font.size": 10,
                "font.family": "serif",
                "font.serif": ["Computer Modern", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
            })
            sns.set_theme(style="whitegrid", font='serif')

            def draw_heatmap(data, cbar_label, fname, vmin=None, vmax=None, center=None, cmap="crest"):
                fig, ax = plt.subplots(figsize=(1.4 * data.shape[1] + 3,
                                                0.42 * data.shape[0] + 2))

                hm = sns.heatmap(
                    data,
                    vmin=vmin, vmax=vmax, center=center,
                    cmap=cmap, linewidths=0.3, linecolor="white",
                    cbar_kws={"label": cbar_label},
                    ax=ax
                )
                ax.set_xticklabels(col_labels, rotation=0)
                ax.set_yticklabels(row_labels, rotation=0)

                breaks = np.cumsum(group_sizes)[:-1]
                for y in breaks:
                    ax.hlines(y, xmin=0, xmax=data.shape[1], colors="black", linewidth=1)

                plt.tight_layout()
                out_path = Path(f"../analysis_results/plots_phase1/{fname}")
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # PDF (Original)
                plt.savefig(out_path, format="pdf", dpi=200, bbox_inches="tight")

                # 🌟 PNG für PowerPoint
                plt.savefig(out_path.with_suffix(".svg"), format="svg",
                            dpi=350, bbox_inches="tight")

            # Standardized (z-score) heatmap
            draw_heatmap(
                abs_table_std,
                cbar_label="(inverted) z-score",
                fname="step1_hyperparam_metric_std_values.pdf",
                vmin=-2.5, vmax=2.5, center=0, cmap="BuGn"
            )
        else:
            print("Absolute/standardized heatmaps: no recognized hyper-parameter columns found.")
    else:
        print("Absolute/standardized heatmaps skipped: no data in filtered subset.")


    # --- 6) Factor effects by scenario (traffic configuration × network type) ---
    if not df_plot.empty:
        network_col = "network_type"
        traffic_col = "traffic_configuration"

        if network_col is None or traffic_col is None:
            print("Factor-effects plot skipped: could not find network/traffic columns.")
        else:
            # Hyperparameters to analyse (present in file)
            hparams = [
                "cluster_distance_threshold",  # C-DT
                "batch_size_ipupa",  # C-IP
                "learning_rate_weighting",  # C-LR
                "butterfly_threshold_value",  # C-BT
                "substitution_priority_name",  # C-SP (categorical)
            ]
            hparams = [h for h in hparams if h in df_sub_enabled.columns]

            # Metrics to inspect
            effect_metrics = [m for m in ["nrmse_mean", "wasserstein_distance", "execution_time_s", "score"]
                              if m in df_sub_enabled.columns]

            # Pretty names (LaTeX-like) for labels
            latex_metric = {
                "nrmse_mean": r"$NRMSE$",
                "wasserstein_distance": r"$W$",
                "execution_time_s": r"$ET$",
                "score": r"$SC$",
            }
            latex_hparam = {
                "cluster_distance_threshold": r"$C\text{-}DT$",
                "batch_size_ipupa": r"$C\text{-}IP$",
                "learning_rate_weighting": r"$C\text{-}LR$",
                "butterfly_threshold_value": r"$C\text{-}BT$",
                "substitution_priority_name": r"$C\text{-}SP$",
            }

            # Aesthetics
            plt.rcParams.update({
                "font.size": 7,
                "font.family": "serif",
                "font.serif": ["Computer Modern", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
            })
            sns.set_theme(style="whitegrid", context="notebook", font="serif")

            outdir = Path("../analysis_results/plots_phase1/factor_effects_condensed")
            outdir.mkdir(parents=True, exist_ok=True)

            # Prepare data
            df_f = df_sub_enabled.copy()
            df_f[network_col] = (df_f[network_col].astype(str)
                                 .str.replace("NetworkModelType.", "", regex=False).str.strip())
            df_f[traffic_col] = (df_f[traffic_col].astype(str)
                                 .str.replace("TrafficConfig.", "", regex=False)
                                 .str.replace("evaluation_", "", regex=False).str.strip())

            hue_order = sorted(df_f[network_col].dropna().unique())
            style_order = sorted(df_f[traffic_col].dropna().unique())

            def level_order(series):
                if pd.api.types.is_numeric_dtype(series):
                    return sorted(series.dropna().unique())
                return sorted(series.dropna().unique())

            n_h = len(hparams)
            width_per_ax, height = 2.2, 2.6

            for met in effect_metrics:
                fig, axes = plt.subplots(nrows=1, ncols=n_h,
                                         figsize=(width_per_ax * n_h, height),
                                         sharey=False)
                if n_h == 1:
                    axes = [axes]
                else:
                    axes = list(axes)

                for idx, (ax, h) in enumerate(zip(axes, hparams)):
                    ord_h = level_order(df_f[h])

                    # >>> Make x categorical with explicit order
                    if not pd.api.types.is_numeric_dtype(df_f[h]):
                        df_f[h] = pd.Categorical(df_f[h], categories=ord_h, ordered=True)

                    # Only last axis builds a legend; we’ll replace it with a figure legend
                    legend_flag = "auto" if idx == (n_h - 1) else False

                    sns.lineplot(
                        data=df_f,
                        x=h, y=met,
                        hue=network_col, hue_order=hue_order,
                        style=traffic_col, style_order=style_order,
                        markers=True, dashes=False,
                        errorbar=("ci", 95),
                        legend=legend_flag,
                        ax=ax,
                    )

                    ax.set_xlabel(latex_hparam.get(h, h))
                    #ax.set_title(latex_metric.get(met, met))
                    if idx != 0:
                        ax.set_ylabel("")
                    # For numeric hyperparams, ensure sorted ticks
                    if pd.api.types.is_numeric_dtype(df_f[h]):
                        ax.set_xticks(ord_h)

                # Rotate x-ticks on the rightmost subplot
                right_ax = axes[-1]
                for label in right_ax.get_xticklabels():
                    label.set_rotation(20)  # rotate
                    label.set_ha('right')  # align right for readability

                # Single shared legend
                last_ax = axes[-1]
                last_leg = last_ax.get_legend()
                handles, labels = ([], [])
                if last_leg is not None:
                    handles = last_leg.legend_handles
                    labels = [t.get_text() for t in last_leg.get_texts()]
                    last_leg.remove()

                if handles and labels:
                    fig.legend(
                        handles, labels,
                        loc="lower center",  # place below the figure
                        bbox_to_anchor=(0.5, -0.5),  # centered, slightly below the axes
                        frameon=False
                    )

                plt.tight_layout()
                axes[0].set_ylabel(latex_metric.get(met, met))
                out_path = outdir / f"condensed_effects_{met}.pdf"
                plt.savefig(out_path, format="pdf", dpi=200, bbox_inches="tight")
                plt.close(fig)
    # --- 7) Analysis by Test-Training Split (C-TTS) ---
    if "test_train_name" in df_sub_enabled.columns:
        tts_col = "test_train_name"
        df_tts = df_sub_enabled.copy()

        # Normalize labels for readability
        df_tts[tts_col] = (
            df_tts[tts_col].astype(str)
            .str.replace("TestTrainSplit.", "", regex=False)
            .str.replace("_", " ")
            .str.strip()
        )

        # Metrics to analyze
        tts_metrics = [m for m in ["nrmse_mean", "wasserstein_distance",
                                   "mean_in_one_sigma_interval", "execution_time_s", "score"]
                       if m in df_tts.columns]

        # LaTeX-style labels
        metric_labels = {
            "nrmse_mean": r"$NRMSE$",
            "wasserstein_distance": r"$W$",
            "mean_in_one_sigma_interval": r"$C_{\pm\sigma}$",
            "execution_time_s": r"$ET$",
            "score": r"$SC$",
        }

        # --- Summary Table ---
        print("\n=== Step 7: Test-Training Split Analysis ===")
        summary = df_tts.groupby(tts_col)[tts_metrics].agg(["mean", "std"]).round(4)
        print(summary)

        # --- Z-score normalization for cross-metric comparability ---
        std = df_tts[tts_metrics].std(ddof=0).replace(0, np.nan)
        df_tts_z = df_tts.copy()
        df_tts_z[tts_metrics] = (df_tts[tts_metrics] - df_tts[tts_metrics].mean()) / std
        df_tts_z['execution_time_s'] *= -1
        df_tts_z['wasserstein_distance'] *= -1
        df_tts_z = df_tts_z.fillna(0.0)

        # --- Plot: Standardized metric means per Test-Training Split ---
        plt.rcParams.update({
            "font.size": 8,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })
        sns.set_theme(style="whitegrid", font="serif")

        # Mean z-score per metric per C-TTS
        mean_z = df_tts_z.groupby(tts_col)[tts_metrics].mean()

        fig, ax = plt.subplots(figsize=(8, 2.8))
        sns.heatmap(
            mean_z.T,
            cmap="BuGn", center=0,
            cbar_kws={"label": "(inverted) z-score"},
            linewidths=0.3, linecolor="white",
            ax=ax
        )

        ax.set_xlabel("")
        ax.set_ylabel("Metric")
        ax.set_yticklabels([metric_labels.get(m, m) for m in tts_metrics], rotation=0)
        ax.set_title(r"Metrics across Test–Training Configurations (C-TTS)")

        plt.tight_layout()
        out_path = Path("../analysis_results/plots_phase1/step1_tts_metric_comparison.pdf")
        plt.savefig(out_path, format="pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)

    else:
        print("No 'test_train_name' column found — skipping C-TTS analysis.")


if __name__ == "__main__":
    summarize_step1("../analysis_results/aggregated_results1.csv")

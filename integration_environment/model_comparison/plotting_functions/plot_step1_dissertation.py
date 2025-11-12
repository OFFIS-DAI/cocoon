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
            "test_train_name",  # C-TTS (categorical)
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
                "substitution_priority_name": r"$C\text{-}SP$",
                "test_train_name": r"$C\text{-}TTS$",
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

            # Styling (Computer Modern look without usetex)
            plt.rcParams.update({
                "font.size": 10,
                "font.family": "serif",
                "font.serif": ["Computer Modern", "DejaVu Serif"],
                "mathtext.fontset": "cm",
                "axes.unicode_minus": False,
            })
            sns.set_theme(style="whitegrid", font='serif')

            # Helper to draw heatmap + horizontal separators
            def draw_heatmap(data, cbar_label, fname, vmin=None, vmax=None, center=None, cmap="crest"):
                fig, ax = plt.subplots(figsize=(1.4 * data.shape[1] + 3, 0.42 * data.shape[0] + 2))
                hm = sns.heatmap(
                    data,
                    vmin=vmin, vmax=vmax, center=center,
                    cmap=cmap, linewidths=0.3, linecolor="white",
                    cbar_kws={"label": cbar_label},
                    ax=ax
                )
                ax.set_xticklabels(col_labels, rotation=0)
                ax.set_yticklabels(row_labels, rotation=0)

                # Horizontal separators between hyperparameters
                breaks = np.cumsum(group_sizes)[:-1]  # row boundaries (in data coords)
                for y in breaks:
                    ax.hlines(y, xmin=0, xmax=data.shape[1], colors="black", linewidth=1)

                plt.tight_layout()
                out_path = Path(f"../analysis_results/plots_phase1/{fname}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(out_path, format="pdf", dpi=200, bbox_inches="tight")

            # Standardized (z-score) heatmap
            draw_heatmap(
                abs_table_std,
                cbar_label="z-score",
                fname="step1_hyperparam_metric_std_values.pdf",
                vmin=-2.5, vmax=2.5, center=0, cmap="vlag"
            )
        else:
            print("Absolute/standardized heatmaps: no recognized hyper-parameter columns found.")
    else:
        print("Absolute/standardized heatmaps skipped: no data in filtered subset.")


if __name__ == "__main__":
    summarize_step1("../analysis_results/aggregated_results1.csv")

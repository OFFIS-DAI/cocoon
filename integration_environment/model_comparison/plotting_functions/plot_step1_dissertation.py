import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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

if __name__ == "__main__":
    summarize_step1("../analysis_results/aggregated_results1.csv")

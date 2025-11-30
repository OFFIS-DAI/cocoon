import matplotlib
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

# Fix PyCharm matplotlib backend issue
matplotlib.use('Agg')  # Use non-interactive backend

def load_results(phase: int = 1, minimal_evaluation=False):
    if minimal_evaluation:
        results_path = Path(f"../analysis_results/minimal_analysis.csv")
    else:
        results_path = Path(f"../analysis_results/aggregated_results{phase}.csv")
    df = pd.read_csv(results_path)
    return df


def plot_main_effects(df: pd.DataFrame, response: str, fig_name: str):
    df = df[df['model_type'] == 'meta_model']
    renaming_map = {
        'cluster_distance_threshold_name': 'C-DT',
        'batch_size_ipupa_name': 'C-IP',
        'learning_rate_weighting_name': 'C-LR',
        'butterfly_threshold_value_name': 'C-BT',
        'substitution_priority': 'C-SP'
    }
    df.rename(columns=renaming_map, inplace=True)
    factor_cols = ['C-DT', 'C-IP', 'C-LR', 'C-BT', 'C-SP']
    fig, axes = plt.subplots(1, len(factor_cols), figsize=(18, 6), sharey=True)
    for i, col in enumerate(factor_cols):
        seaborn.boxplot(data=df, x=col, y=response, ax=axes[i])
        axes[i].set_title(col)
        axes[i].set_xlabel("Level")
        if i == 0:
            axes[i].set_ylabel(response)
        else:
            axes[i].set_ylabel("")

    legend_elements = [
        mpatches.Patch(color='white', label='C-DT: Cluster Distance Threshold'),
        mpatches.Patch(color='white', label='C-IP: I Pupa Batch Size'),
        mpatches.Patch(color='white', label='C-LR: Learning Rate Weighting'),
        mpatches.Patch(color='white', label='C-BT: Butterfly Threshold'),
        mpatches.Patch(color='white', label='C-SP: Substitution Priority'),
    ]

    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=2,
               frameon=False,
               bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('../analysis_results/plots_phase1/' + fig_name)
    plt.close()


def plot_model_comparison(df: pd.DataFrame, response: str, fig_name: str):
    """Compare different model types for the given response metric."""
    if 'model_type' not in df.columns:
        return
    df['model_type'] = df['model_type'].astype(str)  # Ensure string for plotting
    plt.figure(figsize=(8, 6))
    seaborn.boxplot(data=df, x='model_type', y=response)
    plt.title(f'Model Comparison: {response}')
    plt.xlabel("Model Type")
    plt.ylabel(response)
    plt.tight_layout()
    plt.savefig('../analysis_results/plots_phase2/' + fig_name)
    plt.close()


def plot_hyperparam_score_heatmaps(df: pd.DataFrame, fig_name: str = "phase1_heatmap_scores.png"):
    """
    Create seaborn heatmaps that visualize the mean SCORE for each level of the
    meta-model hyper-parameters. Arranged in two rows, slide-friendly.

    Saves to analysis_results/plots_phase1/<fig_name>.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    sns.set(style="white", font_scale=1.2)

    # Filter to meta-model only
    df = df[df["model_type"] == "meta_model"].copy()

    rename_map = {
        "cluster_distance_threshold_name": "C-DT",
        "batch_size_ipupa_name": "C-IP",
        "learning_rate_weighting_name": "C-LR",
        "butterfly_threshold_value_name": "C-BT",
        "substitution_priority_name": "C-SP",
        "test_train_name": "C-TTS"
    }

    available = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=available)

    factor_cols = [v for v in ["C-DT", "C-IP", "C-LR", "C-BT", "C-SP", "C-TTS"] if v in df.columns]
    if "score" not in df.columns:
        raise ValueError("Column 'score' not found in DataFrame.")

    n = len(factor_cols)
    ncols = 3
    nrows = int((n + ncols - 1) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = axes.flatten()

    for ax, col in zip(axes, factor_cols):
        tbl = df.groupby(col, dropna=False)["score"].mean().reset_index()
        pivot_tbl = tbl.pivot_table(values="score", index=col)

        sns.heatmap(
            pivot_tbl,
            annot=True,
            fmt=".1f",
            cmap="crest",
            cbar=False,
            ax=ax
        )
        ax.set_title(f"{col}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="y")

        # Consistent orientation for labels
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels([])

    # Remove unused subplots if factors < nrows*ncols
    for j in range(len(factor_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    out_path = Path("../analysis_results/plots_phase1") / fig_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.as_posix(), dpi=200, bbox_inches="tight")
    plt.close()


# ---- Phase 1: Heatmaps of Score by Hyper-parameter ----
try:
    _df_heat = load_results(phase=1)
    plot_hyperparam_score_heatmaps(_df_heat.copy(), fig_name="phase1_heatmap_scores.png")
except Exception as e:
    print(f"[heatmap generation skipped] {e}")
# --- Phase 1: Main Effects ---
df_main = load_results(phase=1)
for response, fig_name in [
    ('mean_in_one_sigma_interval', 'mean_in_one_sigma_interval'),
    ('nrmse_mean', 'nrmse_mean'),
    ('nrmse_std', 'nrmse_std'),
    ('execution_time_s', 'execution_time'),
    ('substitution_message_index', 'substitution_message_index'),
    ('substitution_occurred', 'substitution_occurred'),
    ('score', 'score')
]:
    plot_main_effects(df_main.copy(), response=response, fig_name=f"phase1_main_effects_{fig_name}.png")

results_dir = '../analysis_results/plots_phase1/'
Path(results_dir).mkdir(parents=True, exist_ok=True)  # create dir if not exists



# --- Phase 2: Model Comparison ---
df_phase2 = load_results(phase=2)
for response, fig_name in [('mean_in_one_sigma_interval', 'mean_in_one_sigma_interval'),
                           ('nrmse_mean', 'nrmse_mean'),
                           ('nrmse_std', 'nrmse_std'),
                           ('execution_time_s', 'execution_time'),
                           ]:
    plot_model_comparison(df_phase2.copy(), response=response, fig_name=f"phase2_model_comparison_{fig_name}.png")


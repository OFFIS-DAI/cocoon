import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

results_dir = 'analysis_results/plots_phase1/'


def load_results(phase: int):
    results_path = Path(f"analysis_results/aggregated_results{phase}.csv")
    df = pd.read_csv(results_path)
    return df


def plot_main_effects(df: pd.DataFrame, response: str, fig_name: str):
    # only use results from meta-model
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
        sns.boxplot(data=df, x=col, y=response, ax=axes[i])
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

    # Place the legend below the plots
    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=2,
               frameon=False,
               bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(results_dir + fig_name)


# --- Main Execution ---
df_main = pd.read_csv("analysis_results/aggregated_results1.csv")
for response, fig_name in [('after_substitution_rmse_ms', 'rmse'),
                           ('after_substitution_mae_ms', 'mae'),
                           ('execution_time_s', 'execution_time'),
                           ('memory_peak_mb', 'memory_peak'),
                           ('substitution_message_index', 'substitution_message_index'),
                           ('substitution_occurred', 'substitution_occurred')
                           ]:
    plot_main_effects(df_main, response=response, fig_name=fig_name)

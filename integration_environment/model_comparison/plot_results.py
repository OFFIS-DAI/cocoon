import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches

results_dir = 'analysis_results/plots_phase1/'
Path(results_dir).mkdir(parents=True, exist_ok=True)  # create dir if not exists


def load_results(phase: int):
    results_path = Path(f"analysis_results/aggregated_results{phase}.csv")
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

    fig.legend(handles=legend_elements,
               loc='lower center',
               ncol=2,
               frameon=False,
               bbox_to_anchor=(0.5, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig('analysis_results/plots_phase1/' + fig_name)
    plt.close()


def plot_model_comparison(df: pd.DataFrame, response: str, fig_name: str):
    """Compare different model types for the given response metric."""
    if 'model_type' not in df.columns:
        return
    df['model_type'] = df['model_type'].astype(str)  # Ensure string for plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='model_type', y=response)
    plt.title(f'Model Comparison: {response}')
    plt.xlabel("Model Type")
    plt.ylabel(response)
    plt.tight_layout()
    plt.savefig('analysis_results/plots_phase2/' + fig_name)
    plt.close()


# --- Phase 1: Main Effects ---
df_main = load_results(phase=1)
for response, fig_name in [
                           ('execution_time_s', 'execution_time'),
                           ('substitution_message_index', 'substitution_message_index'),
                           ('substitution_occurred', 'substitution_occurred')
                           ]:
    plot_main_effects(df_main.copy(), response=response, fig_name=f"phase1_main_effects_{fig_name}.png")

# --- Phase 2: Model Comparison ---
df_phase2 = load_results(phase=2)
for response, fig_name in [('execution_time_s', 'execution_time'),
                           ]:
    plot_model_comparison(df_phase2.copy(), response=response, fig_name=f"phase2_model_comparison_{fig_name}.png")

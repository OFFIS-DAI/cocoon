# compare_simulation_vs_real_time.py
#
# Script to plot simulation time vs. real time from two CSV files
# on the same graph using seaborn.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_and_prepare(csv_path):
    """
    Load CSV and try to extract columns for simulation time and real time.
    Adjust this if your column names differ.
    """
    df = pd.read_csv(csv_path)

    # Try to guess column names
    # simulation_time_s,relative_real_time_s
    possible_sim_cols = ["simulation_time_s"]
    possible_real_cols = ["relative_real_time_s"]

    def pick_col(candidates):
        for cand in candidates:
            matches = [c for c in df.columns if cand.lower() in c.lower()]
            if matches:
                return matches[0]
        raise KeyError(f"Could not find any of {candidates} in {df.columns}")

    sim_col = pick_col(possible_sim_cols)
    real_col = pick_col(possible_real_cols)

    df = df[[sim_col, real_col]].copy()
    df.columns = ["Simulation Time (seconds)", "Real Time (seconds)"]
    return df


def plot_comparison(csv1, csv2, label1="Run 1", label2="Run 2", save_path=None):
    df1 = load_and_prepare(csv1)
    df1["Model"] = label1

    df2 = load_and_prepare(csv2)
    df2["Model"] = label2

    df = pd.concat([df1, df2], ignore_index=True)

    colors = [sns.color_palette("crest", as_cmap=True)(0.1),
              sns.color_palette("crest", as_cmap=True)(0.9)]

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(7, 6))
    sns.lineplot(
        data=df,
        x="Real Time (seconds)",
        y="Simulation Time (seconds)",
        hue="Model",
        marker="o",
        palette=colors
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")


if __name__ == "__main__":

    csv1 = 'results/phase1_vm_06_09/time_advancement_detailed-five-medium-one_min-cbr_broadcast_1_mps-simbench_5g-none-none-none-none-none-none-1.csv'
    csv2 = 'results/phase1_vm_06_09/time_advancement_meta_model-five-medium-one_min-cbr_broadcast_1_mps-simbench_5g-three-scale_split-hundred-center-center-none-0.csv'

    plot_comparison(csv1, csv2, 'Detailed Model', 'Meta-Model',
                    'analysis_results/plots_phase1/time_advancement.png')

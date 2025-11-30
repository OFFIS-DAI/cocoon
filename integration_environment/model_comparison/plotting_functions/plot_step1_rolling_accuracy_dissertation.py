#!/usr/bin/env python3
"""
plot_chunk_metrics_from_summary.py

Reads a summary CSV that already contains per-chunk metrics as columns
(e.g., weighted_mae_start0, weighted_rmse_start0, ...) and creates
seaborn lineplots of MAE/RMSE over time grouped by hyperparameter categories.
"""
from __future__ import annotations
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from integration_environment.scenario_configuration import *


# ---- helpers (unchanged) ----
def extract_chunk_series(row: pd.Series, prefix: str, metric: str) -> pd.DataFrame:
    pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(metric)}_start(\d+)$")
    starts, values = [], []
    for col, val in row.items():
        m = pattern.match(str(col))
        if m:
            start = int(m.group(1))
            if pd.notna(val):
                starts.append(start)
                values.append(float(val))
    if not starts:
        return pd.DataFrame(columns=["start", "value"])
    return pd.DataFrame({"start": starts, "value": values}).sort_values("start").reset_index(drop=True)


def infer_step_size(starts: np.ndarray, fallback: int) -> int:
    if len(starts) < 2:
        return fallback
    diffs = np.diff(starts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return fallback
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


def main():
    # paths/config (kept)
    csv_path = Path('../analysis_results/summary_disabled_substitution.csv')
    outdir = Path('../analysis_results/plots_phase1')
    outdir.mkdir(parents=True, exist_ok=True)

    # directory where the statistics_*.json files live
    stats_dir = Path('../results/phase1')

    df = pd.read_csv(csv_path)
    step_size = 50  # fallback for inferring step
    id_cols = [c for c in ["scenario"] if c in df.columns]

    # --- collect chunked metrics + scenario-level hyperparams ---
    records = []
    scen_cfg_rows = []  # per-scenario config once
    seen_scen = set()

    for idx, row in df.iterrows():
        mae_df = extract_chunk_series(row, 'weighted', "mae")
        rmse_df = extract_chunk_series(row, 'weighted', "rmse")
        if mae_df.empty and rmse_df.empty:
            continue

        starts_all = np.array(sorted(set(mae_df["start"].tolist() + rmse_df["start"].tolist())))
        step = infer_step_size(starts_all, step_size)

        scen = "_".join(str(row[c]) for c in id_cols) or f"row{idx}"
        scen_id = scen.replace('cocoon_', '')
        config = ScenarioConfiguration.from_scenario_id(scen_id)

        # read corresponding statistics JSON and get execution time
        exec_time_s = None
        stats_path = stats_dir / f"statistics_{scen_id}.json"
        if stats_path.exists():
            try:
                with stats_path.open("r", encoding="utf-8") as f:
                    stats = json.load(f)
                exec_time_s = stats.get("execution_run_time_s", None)
            except Exception as e:
                print(f"[WARN] Could not read stats for {scen_id}: {e}")
        else:
            print(f"[INFO] No statistics JSON found for {scen_id} at {stats_path}")

        # store per-scenario config (once)
        if scen not in seen_scen:
            scen_cfg_rows.append({
                "scenario": scen,
                "cluster_dist": getattr(config.cluster_distance_threshold, "name", str(config.cluster_distance_threshold)),
                "ipupa": getattr(config.i_pupa, "name", str(config.i_pupa)),
                "lr_weight": getattr(config.learning_rate_weighting, "name", str(config.learning_rate_weighting)),
                "amount of training data": getattr(
                    config.amount_of_scenarios_in_training_data, "name",
                    str(config.amount_of_scenarios_in_training_data)
                ),
                "split": getattr(config.test_train_split, "name", str(config.test_train_split)),
                "exec_time_s": exec_time_s,
            })
            seen_scen.add(scen)

        if not mae_df.empty:
            for s, v in mae_df.itertuples(index=False):
                records.append({"scenario": scen, "metric": "MAE", "chunk_end": s + step, "value": v})
        if not rmse_df.empty:
            for s, v in rmse_df.itertuples(index=False):
                records.append({"scenario": scen, "metric": "RMSE", "chunk_end": s + step, "value": v})

    long_df = pd.DataFrame(records)
    if long_df.empty:
        print("[INFO] No chunked metrics found to plot.")
        return

    cfg_df = pd.DataFrame(scen_cfg_rows)

    # execution time table
    exec_out = outdir / "scenario_execution_times.csv"
    cfg_df[["scenario", "exec_time_s", "amount of training data", "split"]].to_csv(exec_out, index=False)
    print(f"[OK] Wrote execution time table → {exec_out}")

    # --- NEW: execution times by amount of training data AND split ---
    exec_plot_df = cfg_df.dropna(subset=["exec_time_s"]).copy()
    if not exec_plot_df.empty:
        plt.rcParams.update({
            "font.size": 7,
            "font.family": "serif",
            "font.serif": ["Computer Modern", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        })
        sns.set_theme(style="whitegrid", context="notebook", font="serif")

        plt.figure(figsize=(10, 4))
        ax = sns.barplot(
            data=exec_plot_df,
            x="amount of training data",
            y="exec_time_s",
            hue="split",               # <-- differentiate by test-train split
            errorbar=("ci", 95),
            palette="OrRd",
        )
        ax.set_xlabel("Amount of training data")
        ax.set_ylabel("Execution time (s)")
        plt.legend(title="Test–train split", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        exec_pdf = outdir / "execution_time_by_amount_of_training_data_and_split.pdf"
        plt.savefig(exec_pdf, format="pdf")
        plt.close()
        print(f"[OK] Wrote execution time plot → {exec_pdf}")
    else:
        print("[INFO] No execution times available to plot.")

    # merge for error plots
    long_df = long_df.merge(cfg_df, on="scenario", how="left")

    # lineplots for different categories of hyper params
    cat_cols = ["cluster_dist", "ipupa", "lr_weight", "amount of training data", "split"]

    # Aesthetics (keep consistent)
    plt.rcParams.update({
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["Computer Modern", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    sns.set_theme(style="whitegrid", context="notebook", font="serif")
    for cat in cat_cols:
        plot_df = (
            long_df
            .groupby([cat, "metric", "chunk_end"], as_index=False)["value"]
            .mean()
        )

        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=plot_df,
            palette='RdPu',
            errorbar=("ci", 95),
            x="chunk_end",
            y="value",
            hue=cat,          # different categories of the hyperparam
            style="metric",   # MAE vs RMSE
            dashes=True,
            linewidth=2,
        )
        ax.set_xlabel("Chunk end (message index)")
        ax.set_ylabel("Error (mean across scenarios)")
        ax.set_title(f"Chunked MAE/RMSE over time by {cat}")
        plt.xlim(50, 250)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        out_pdf = outdir / f"by_{cat}_weighted_chunked_mae_rmse.pdf"
        plt.savefig(out_pdf, format='pdf')
        plt.close()
        print(f"[OK] Wrote category plot ({cat}) → {out_pdf}")


if __name__ == "__main__":
    main()

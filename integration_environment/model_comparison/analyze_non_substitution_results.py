# analyze_subst_disabled.py
# Analyze accuracy over time for substitution-disabled scenarios in phase1 CSVs.

from pathlib import Path
from typing import Dict, Optional, List
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Configuration (no CLI) --------
ROOT_DIR = Path("results/phase1")
PATTERN = "*.csv"
OUT_DIR = Path("analysis_results")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, rolling: bool = False) -> Dict[str, float]:
    s = pd.concat([y_true, y_pred], axis=1).dropna()
    if s.empty:
        return {"mae": np.nan, "rmse": np.nan, "nrmse": np.nan}
    yt = s.iloc[:, 0].to_numpy()
    yp = s.iloc[:, 1].to_numpy()
    mae = np.mean(np.abs(yt - yp))
    rmse = math.sqrt(np.mean((yt - yp) ** 2))
    denom = np.mean(np.abs(yt)) if np.mean(np.abs(yt)) != 0 else (np.std(yt) if np.std(yt) != 0 else 1.0)
    nrmse = rmse / denom
    if rolling:
        result_dict = {}
        step_size = 50
        start = 0
        while start < len(yt):
            end = start + step_size if (start+step_size) < len(yt) else len(yt)
            cur_mae = np.mean(np.abs(yt[start:end] - yp[start:end]))
            cur_rmse = math.sqrt(np.mean((yt[start:end] - yp[start:end]) ** 2))

            result_dict[f'mae_start{start}'] = cur_mae
            result_dict[f'rmse_start{start}'] = cur_rmse
            start += step_size
        result_dict['mae'] = mae
        result_dict['rmse'] = rmse
        result_dict['nrmse'] = nrmse
        return result_dict
    else:
        return {"mae": mae, "rmse": rmse, "nrmse": nrmse}


# -------- Core processing --------
def process_file(csv_path: Path) -> Optional[Dict[str, float]]:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] Could not read {csv_path}: {e}")
        return None

    t_col = 'msg_id'
    y_col = 'actual_delay_ms'
    online_col = 'online_predicted_delay_ms'
    cluster_col = 'cluster_predicted_delay_ms'
    weighted_col = 'weighted_predicted_delay_ms'

    found = {"time": t_col, "actual": y_col, "online": online_col,
             "cluster": cluster_col, "weighted": weighted_col}
    if (t_col is None) or (y_col is None):
        print(f"[WARN] {csv_path.name}: Missing essential columns (time/actual). Found={found}")
        return None

    # Sort by time
    df = df.sort_values(by=t_col).reset_index(drop=True)
    # Summary metrics
    summary: Dict[str, float] = {"scenario": csv_path.stem, "rows": len(df)}
    if online_col in df:   summary.update(
        {f"online_{k}": v for k, v in compute_metrics(df[y_col], df[online_col]).items()})
    if cluster_col in df:  summary.update(
        {f"cluster_{k}": v for k, v in compute_metrics(df[y_col], df[cluster_col]).items()})
    if weighted_col in df: summary.update(
        {f"weighted_{k}": v for k, v in compute_metrics(df[y_col], df[weighted_col], rolling=True).items()})

    return summary


def analyze_phase1(root: Path = ROOT_DIR, pattern: str = PATTERN,
                   outdir: Path = OUT_DIR) -> Optional[pd.DataFrame]:
    files = sorted(root.glob(pattern))
    if not files:
        print(f"[INFO] No CSV files found under {root} with pattern {pattern}")
        return None

    summaries = []
    for csv_path in files:
        if 'cocoon_meta_model' in str(csv_path) and 'disabled' in str(csv_path):
            res = process_file(csv_path)
            if res is not None:
                summaries.append(res)

    if not summaries:
        print("[INFO] No substitution-disabled scenarios processed. Check filters/filenames.")
        return None

    df_summary = pd.DataFrame(summaries).sort_values(by="scenario")
    outdir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(outdir / "summary_disabled_substitution.csv", index=False)
    print(f"[OK] Wrote summary metrics → {outdir / 'summary_accuracy_metrics.csv'}")
    print(f"[OK] Plots saved under     → {outdir}")
    return df_summary


# Run directly if desired (no arguments needed)
if __name__ == "__main__":
    analyze_phase1()

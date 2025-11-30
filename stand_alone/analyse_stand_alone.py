import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv("results/delay_comparison.csv")

# Identify all prediction columns (restricted to 'weighted' variants)
pred_cols = [c for c in df.columns if not c.startswith("real") and not c.startswith("msg_id") and 'weighted' in c]

# Compute metrics
summary = []
y = df["real_delay_ms"].astype(float).values
den = np.sum((y - y.mean())**2)

for col in pred_cols:
    yhat = df[col].astype(float).values
    err = yhat - y
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    r2 = 1 - np.sum(err**2) / den if den != 0 else np.nan
    summary.append({"config": col.replace("_delay_ms", ""), "col": col, "MAE_ms": mae, "RMSE_ms": rmse, "R2": r2})

summary_df = pd.DataFrame(summary).sort_values("RMSE_ms", ascending=True)
summary_df.to_csv("results/prediction_analysis.csv", index=False)

# Select the best config (lowest RMSE)
best = summary_df.iloc[0]
best_conf = best["config"]
print(f"Best configuration: {best_conf}")

# === Figure: Predictions vs. Real over time (bins of 100) with RMSE & MAE annotations ===
# (Matplotlib only; annotations aligned horizontally for all bins.)
# Ensure output directory exists
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)

# Defensive: identify prediction columns and best model (lowest RMSE)
if "real_delay_ms" not in df.columns:
    raise KeyError("Column 'real_delay_ms' not found in the input dataframe.")

pred_cols = [c for c in df.columns if c != "real_delay_ms" and not c.startswith("msg_id") and 'weighted' in c]
if not pred_cols:
    raise ValueError("No prediction columns found. Expected columns besides 'real_delay_ms'.")

y_true = df["real_delay_ms"].astype(float).to_numpy()
metrics = []
for c in pred_cols:
    y_hat = df[c].astype(float).to_numpy()
    err = y_hat - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    metrics.append((c, mae, rmse))

# Pick the column with the lowest RMSE
best_col, best_mae, best_rmse = sorted(metrics, key=lambda t: t[2])[0]

# Bin into chunks of 100 messages
bin_size = 100
n = len(df)
idx = np.arange(n)
bin_id = idx // bin_size
n_bins = int(bin_id.max()) + 1

# Compute per-bin metrics
bin_stats = []
for b in range(n_bins):
    m = bin_id == b
    if not np.any(m):
        continue
    yb = y_true[m]
    yh = df.loc[m, best_col].astype(float).to_numpy()
    e = yh - yb
    mae_b = float(np.mean(np.abs(e)))
    rmse_b = float(np.sqrt(np.mean(e**2)))
    bin_stats.append({"bin": b, "start": b*bin_size, "end": min((b+1)*bin_size-1, n-1),
                      "MAE_ms": mae_b, "RMSE_ms": rmse_b})

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",                         # <- important
    "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})
# Plot (semi-transparent curves with filled areas)
fig, ax = plt.subplots(figsize=(10, 5))

# Blue = real, Green = predicted
y_pred = df[best_col].astype(float).to_numpy()

# Define colors
real_color = "#1f77b4"
pred_color = "green"

ax.plot(
    idx, y_true,
    color=real_color, linewidth=1.8, alpha=0.7, label="Real delay [ms]"
)
ax.fill_between(
    idx, y_true, color=real_color, alpha=0.1
)

ax.plot(
    idx, y_pred,
    color=pred_color, linewidth=1.8, alpha=0.7, label=f"Predicted delay [ms]"
)
ax.fill_between(
    idx, y_pred, color=pred_color, alpha=0.1
)

# Vertical bin boundaries
for b in range(1, n_bins):
    ax.axvline(b * bin_size - 0.5, linestyle="--", linewidth=0.7, alpha=0.4, color="gray")

# Grid and labels
ax.grid(True, which="both", linewidth=0.6, alpha=0.4)
ax.set_xlabel("Message index")
ax.set_ylabel("Delay [ms]")

# Aligned annotations above both filled curves
y_max = float(np.nanmax(np.concatenate([y_true, y_pred])))
annot_y = y_max + (0.05 * y_max if y_max > 0 else 1.0)

for s in bin_stats[1:-1]:
    x_center = (s["start"] + s["end"]) / 2.0
    text = f"RMSE={s['RMSE_ms']:.1f} ms\nMAE={s['MAE_ms']:.1f} ms"
    ax.text(x_center, annot_y, text, ha="center", va="bottom", color="black")

# Adjust y-limits for annotation space
ax.set_ylim(bottom=min(0.0, float(np.nanmin(np.concatenate([y_true, y_pred])))),
            top=annot_y * 1.1)

ax.legend(loc="best", frameon=True)
fig.tight_layout()

fig_path_png = out_dir / "pred_vs_real_over_time_bin100_filled.svg"
fig_path_pdf = out_dir / "pred_vs_real_over_time_bin100_filled.pdf"
fig.savefig(fig_path_png, dpi=200)
fig.savefig(fig_path_pdf)
print(f"Saved filled figure to: {fig_path_png} and {fig_path_pdf}")



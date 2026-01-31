import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# --- Styling: seaborn + serif font ---
sns.set_theme(style="white", font="serif", palette='Set2')

# --- Parameters ---
t_quant = 2.015
sigma_real_ms = 25.0
sigma_cocoon_ms = 25.0
t_det_s = 120.0
t_cocoon_s = 15.0
eps_cocoon_ms = 5.0

# --- Time budget grid ---
T_max = 2 * 3600  # 4 hours
T = np.arange(1, T_max + 1)  # start at 1s to avoid division by zero

# --- Completed runs (step functions) ---
runs_det = np.floor(T / t_det_s)
runs_cocoon = np.floor(T / t_cocoon_s)

# To avoid division by zero in the error formulas at small T:
runs_det_safe = np.maximum(runs_det, 1.0)
runs_cocoon_safe = np.maximum(runs_cocoon, 1.0)

# --- Achievable errors (ms) ---
# Use the *integer* number of runs for consistency with step-plot runs
err_det = (t_quant * sigma_real_ms) / np.sqrt(runs_det_safe)
err_cocoon = eps_cocoon_ms + (t_quant * sigma_cocoon_ms) / np.sqrt(runs_cocoon_safe)

# --- Correct critical time threshold (seconds) ---
# Derived from: eps_cocoon + t*sigma_cocoon*sqrt(t_cocoon)/sqrt(T) = t*sigma_real*sqrt(t_det)/sqrt(T)
T_critical = (
    (t_quant * (sigma_real_ms * np.sqrt(t_det_s) - sigma_cocoon_ms * np.sqrt(t_cocoon_s)))
    / eps_cocoon_ms
) ** 2

print('T critical = ', T_critical)
# --- Plot: runs (left axis) + achievable error (right axis) ---
fig, ax_runs = plt.subplots(figsize=(7.6, 4.4))

# Left axis: completed runs (step plots, dashed)
ax_runs.step(T, runs_det, where="post", linestyle="--", alpha=0.9, label="Runs (detailed)")
ax_runs.step(T, runs_cocoon, where="post", linestyle="--", alpha=0.9, label="Runs (meta-model)")
ax_runs.set_xlabel("Execution time budget $T$ (s)")
ax_runs.set_ylabel("Completed simulation runs")

# Right axis: achievable error (continuous, solid)
ax_err = ax_runs.twinx()
ax_err.plot(T, err_det, linestyle="-", alpha=0.85, label="Achievable error (detailed)")
ax_err.plot(T, err_cocoon, linestyle="-", alpha=0.85, label="Achievable error (meta-model)")
ax_err.set_ylabel(r"$\epsilon_{\mathrm{achievable}}$ (ms)")

# Critical threshold line
if T.min() <= T_critical <= T.max():
    ax_runs.axvline(T_critical, linestyle="--", linewidth=1.2, color="red",
                    label=fr"$T_{{\mathrm{{critical}}}}\approx {T_critical:,.0f}\,\mathrm{{s}}$")

# Combined legend (from both axes)
handles1, labels1 = ax_runs.get_legend_handles_labels()
handles2, labels2 = ax_err.get_legend_handles_labels()
ax_runs.legend(
    handles1 + handles2,
    labels1 + labels2,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.2),
    ncol=2,
    frameon=True
)


fig.tight_layout()
plt.savefig("../analysis_results/plots_trade_off/trade_off.pdf", dpi=200, bbox_inches="tight")



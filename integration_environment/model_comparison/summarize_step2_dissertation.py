#!/usr/bin/env python3
"""
Script to read the aggregated results CSV file and print LaTeX tables
for different metrics per factor AND save the same tables as CSV files.

  - Factor T: traffic configuration (CBR, Poisson, CDSB)
  - Factor S: scenario duration (5 min, 10 min)
  - Factor ND: number of devices (5, 10, 20, 50)
  - Factor CN: communication network technology (LTE, LTE450, Ethernet, 5G)

NOTE: This version REMOVES the columns for:
  - Detailed Model
  - Ideal Model
and keeps only:
  - Channel Model
  - Static Graph Model
  - Meta-Model splits (pa, tl, tm, sc, te)
"""

import pandas as pd
from pathlib import Path


def format_mean_std(sub_df: pd.DataFrame, metric_col: str, for_csv: bool = False) -> str:
    """
    Return mean/std formatting.

    - LaTeX mode (for_csv=False): '$mean\\pm std$'
    - CSV mode   (for_csv=True):  'mean'  (kept as mean only, consistent with your current CSV export)

    For CSV: no math mode, no \\pm.
    """
    if sub_df.empty or metric_col not in sub_df.columns:
        return ""
    m = sub_df[metric_col].mean()
    s = sub_df[metric_col].std()
    if pd.isna(m) or pd.isna(s):
        return ""

    if for_csv:
        return f"{m:.2f}"
    else:
        return f"${m:.2f}\\pm{s:.2f}$"


def latex_cell_for_model(
    df: pd.DataFrame,
    model_type: str,
    metric_col: str,
    split_name: str | None = None,
) -> str:
    """Filter df by model_type (and optionally split_name) and format mean±std for LaTeX."""
    sub = df[df["model_type"] == model_type]
    if split_name is not None and "test_train_name" in df.columns:
        sub = sub[sub["test_train_name"] == split_name]
    return format_mean_std(sub, metric_col=metric_col, for_csv=False)


def build_table_rows_for_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Build the logical table for a given metric as a DataFrame for CSV export.

    Columns:
      Factor, Stage,
      Channel Model, Static Graph Model,
      Meta-pa, Meta-tl, Meta-tm, Meta-sc, Meta-te
    """
    # Mapping of split names → abbreviations
    meta_splits = [
        ("parametrization_split", "pa"),
        ("traffic_load_split", "tl"),
        ("traffic_model_split", "tm"),
        ("scale_split", "sc"),
        ("technology_split", "te"),
    ]

    def subset_by_traffic(token: str) -> pd.DataFrame:
        return df[df["traffic_configuration"].astype(str).str.contains(token, case=False, na=False)].copy()

    def subset_by_duration(token: str) -> pd.DataFrame:
        return df[df["scenario_duration"].astype(str).str.contains(token, case=False, na=False)].copy()

    def subset_by_num_devices(val: str) -> pd.DataFrame:
        return df[df["num_devices"] == val].copy()

    def subset_by_network(token: str) -> pd.DataFrame:
        return df[df["network_type"].astype(str).str.contains(token, case=False, na=False)].copy()

    def csv_stat_for(sub_df: pd.DataFrame, model_type: str, split_name: str | None = None) -> str:
        """CSV-safe mean value for given model_type (+optional split)."""
        sub = sub_df[sub_df["model_type"] == model_type]
        if split_name is not None and "test_train_name" in sub_df.columns:
            sub = sub[sub["test_train_name"] == split_name]
        return format_mean_std(sub, metric_col=metric_col, for_csv=True)

    rows = []

    # ----------------- Factor T (Traffic Configurations) -------------------
    traffic_rows = [
        ("CBR", "cbr"),
        ("Poisson", "poisson"),
        ("CDSB", "central"),
    ]
    for label_row, token in traffic_rows:
        sub = subset_by_traffic(token)
        ch_val = csv_stat_for(sub, "channel")
        sg_val = csv_stat_for(sub, "static_graph")
        meta_vals = [csv_stat_for(sub, "meta_model", split_name) for split_name, _abbr in meta_splits]

        row = {
            "Factor": "T",
            "Stage": label_row,
            "Channel Model": ch_val,
            "Static Graph Model": sg_val,
        }
        for (split_name, abbr), val in zip(meta_splits, meta_vals):
            row[f"Meta-{abbr}"] = val
        rows.append(row)

    # ----------------- Factor S (Scenario Duration) -----------------------
    sub_5 = subset_by_duration("five")
    row_5 = {
        "Factor": "S",
        "Stage": "5 min",
        "Channel Model": csv_stat_for(sub_5, "channel"),
        "Static Graph Model": csv_stat_for(sub_5, "static_graph"),
    }
    for split_name, abbr in meta_splits:
        row_5[f"Meta-{abbr}"] = csv_stat_for(sub_5, "meta_model", split_name)
    rows.append(row_5)

    sub_10 = subset_by_duration("ten")
    row_10 = {
        "Factor": "S",
        "Stage": "10 min",
        "Channel Model": csv_stat_for(sub_10, "channel"),
        "Static Graph Model": csv_stat_for(sub_10, "static_graph"),
    }
    for split_name, abbr in meta_splits:
        row_10[f"Meta-{abbr}"] = csv_stat_for(sub_10, "meta_model", split_name)
    rows.append(row_10)

    # ----------------- Factor ND (Num Devices) ----------------------------
    nd_levels = [
        ("NumDevices.five", 5),
        ("NumDevices.ten", 10),
        ("NumDevices.twenty", 20),
        ("NumDevices.fifty", 50),
    ]
    for token, label_row in nd_levels:
        sub_nd = subset_by_num_devices(token)
        row_nd = {
            "Factor": "ND",
            "Stage": str(label_row),
            "Channel Model": csv_stat_for(sub_nd, "channel"),
            "Static Graph Model": csv_stat_for(sub_nd, "static_graph"),
        }
        for split_name, abbr in meta_splits:
            row_nd[f"Meta-{abbr}"] = csv_stat_for(sub_nd, "meta_model", split_name)
        rows.append(row_nd)

    # ----------------- Factor CN (Network Type) ---------------------------
    network_rows = [
        ("LTE", "lte"),
        ("LTE450", "lte450"),
        ("Ethernet", "ethernet"),
        ("5G", "5g"),
    ]
    for label_row, token in network_rows:
        sub_net = subset_by_network(token)
        row_net = {
            "Factor": "CN",
            "Stage": label_row,
            "Channel Model": csv_stat_for(sub_net, "channel"),
            "Static Graph Model": csv_stat_for(sub_net, "static_graph"),
        }
        for split_name, abbr in meta_splits:
            row_net[f"Meta-{abbr}"] = csv_stat_for(sub_net, "meta_model", split_name)
        rows.append(row_net)

    return pd.DataFrame(rows)


def print_latex_table_for_metric(
    df: pd.DataFrame,
    metric_col: str,
    metric_tex: str,
    caption: str,
    label: str,
    csv_out_path: Path | None = None,
) -> None:
    """
    Generic LaTeX table printer for a given metric column.
    Also optionally saves the same logical table as a CSV file.

    metric_tex should already contain math-mode delimiters, e.g. '$NRMSE$'.
    """
    if metric_col not in df.columns:
        print(f"[WARN] Column '{metric_col}' not found – skipping table '{label}'.")
        return

    # Build the table data once (CSV-safe values) and export CSV
    if csv_out_path is not None:
        table_df = build_table_rows_for_metric(df, metric_col)
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        table_df.to_csv(csv_out_path, index=False)
        print(f"[INFO] Saved CSV table for '{metric_col}' to: {csv_out_path}")

    # ------------------------ LaTeX printing -------------------------------
    print("\\begin{landscape}")
    print("\\begin{table}[]")
    print("\\renewcommand{\\arraystretch}{1.5}")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}")

    # 9 columns: Factor, Stages, 2 base models, 5 meta-splits
    print("\\begin{tabular}{lllllllll}")
    print("\\hline")
    print(
        "\\textbf{Factor} & \\textbf{Stages} & "
        f"\\multicolumn{{7}}{{l}}{{\\textbf{{{metric_tex}}}}} \\\\ \\hline"
    )

    # Header: 2 base models + 5 meta-splits
    print(
        "\\textbf{CM} &  & "
        "Channel Model & Static Graph Model & "
        "\\multicolumn{5}{l}{Meta-Model} \\\\ \\hline"
    )
    print("\\textbf{C-TTS} &  &  &  & pa & tl & tm & sc & te \\\\ \\hline")

    meta_splits = [
        ("parametrization_split", "pa"),
        ("traffic_load_split", "tl"),
        ("traffic_model_split", "tm"),
        ("scale_split", "sc"),
        ("technology_split", "te"),
    ]

    def subset_by_traffic(token: str) -> pd.DataFrame:
        return df[df["traffic_configuration"].astype(str).str.contains(token, case=False, na=False)].copy()

    def subset_by_duration(token: str) -> pd.DataFrame:
        return df[df["scenario_duration"].astype(str).str.contains(token, case=False, na=False)].copy()

    def subset_by_num_devices(val: str) -> pd.DataFrame:
        return df[df["num_devices"] == val].copy()

    def subset_by_network(token: str) -> pd.DataFrame:
        return df[df["network_type"].astype(str).str.contains(token, case=False, na=False)].copy()

    # Factor T
    traffic_rows = [
        ("CBR", "cbr"),
        ("Poisson", "poisson"),
        ("CDSB", "central"),
    ]
    for i, (label_row, token) in enumerate(traffic_rows):
        sub = subset_by_traffic(token)
        ch_val = latex_cell_for_model(sub, "channel", metric_col)
        sg_val = latex_cell_for_model(sub, "static_graph", metric_col)
        meta_vals = [
            latex_cell_for_model(sub, "meta_model", metric_col, split_name)
            for split_name, _abbr in meta_splits
        ]
        prefix = "\\multirow{3}{*}{\\textbf{T}}" if i == 0 else " "
        print(
            f"{prefix} & {label_row} & {ch_val} & {sg_val} & "
            + " & ".join(meta_vals)
            + " \\\\"
        )

    print(" \\hline")

    # Factor S
    sub_5 = subset_by_duration("five")
    ch_5 = latex_cell_for_model(sub_5, "channel", metric_col)
    sg_5 = latex_cell_for_model(sub_5, "static_graph", metric_col)
    meta_5 = [
        latex_cell_for_model(sub_5, "meta_model", metric_col, split_name)
        for split_name, _a in meta_splits
    ]
    print("\\multirow{2}{*}{\\textbf{S}} & 5 min & "
          f"{ch_5} & {sg_5} & " + " & ".join(meta_5) + " \\\\")

    sub_10 = subset_by_duration("ten")
    ch_10 = latex_cell_for_model(sub_10, "channel", metric_col)
    sg_10 = latex_cell_for_model(sub_10, "static_graph", metric_col)
    meta_10 = [
        latex_cell_for_model(sub_10, "meta_model", metric_col, split_name)
        for split_name, _a in meta_splits
    ]
    print(" & 10 min & "
          f"{ch_10} & {sg_10} & " + " & ".join(meta_10) + " \\\\ \\hline")

    # Factor ND
    nd_levels = [
        ("NumDevices.five", 5),
        ("NumDevices.ten", 10),
        ("NumDevices.twenty", 20),
        ("NumDevices.fifty", 50),
    ]
    for i, (token, label_row) in enumerate(nd_levels):
        sub_nd = subset_by_num_devices(token)
        ch_nd = latex_cell_for_model(sub_nd, "channel", metric_col)
        sg_nd = latex_cell_for_model(sub_nd, "static_graph", metric_col)
        meta_nd = [
            latex_cell_for_model(sub_nd, "meta_model", metric_col, split_name)
            for split_name, _a in meta_splits
        ]
        factor_label = "\\textbf{ND}" if i == 0 else "\\textbf{}"
        print(
            f"{factor_label} & {label_row} & {ch_nd} & {sg_nd} & "
            + " & ".join(meta_nd)
            + " \\\\"
        )

    print("\\hline")

    # Factor CN
    network_rows = [
        ("LTE", "lte"),
        ("LTE450", "lte450"),
        ("Ethernet", "ethernet"),
        ("5G", "5g"),
    ]
    for i, (label_row, token) in enumerate(network_rows):
        sub_net = subset_by_network(token)
        ch_net = latex_cell_for_model(sub_net, "channel", metric_col)
        sg_net = latex_cell_for_model(sub_net, "static_graph", metric_col)
        meta_net = [
            latex_cell_for_model(sub_net, "meta_model", metric_col, split_name)
            for split_name, _a in meta_splits
        ]
        prefix = "\\multirow{4}{*}{\\textbf{CN}}" if i == 0 else " "
        print(
            f"{prefix} & {label_row} & {ch_net} & {sg_net} & "
            + " & ".join(meta_net)
            + " \\\\"
        )

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("\\end{landscape}")


# --- Convenience wrappers for your four metrics ---------------------------

def print_latex_table_nrmse(df: pd.DataFrame, csv_dir: Path) -> None:
    print_latex_table_for_metric(
        df=df,
        metric_col="nrmse_mean",
        metric_tex="$NRMSE$",
        caption="Overview on step 2 results for $NRMSE$ metric.",
        label="tab:overview-step2-nrmse",
        csv_out_path=csv_dir / "overview_step2_nrmse.csv",
    )


def print_latex_table_wasserstein(df: pd.DataFrame, csv_dir: Path) -> None:
    print_latex_table_for_metric(
        df=df,
        metric_col="wasserstein_distance",
        metric_tex="$W$",
        caption="Overview on step 2 results for Wasserstein distance $W$.",
        label="tab:overview-step2-wasserstein",
        csv_out_path=csv_dir / "overview_step2_wasserstein.csv",
    )


def print_latex_table_sigma_coverage(df: pd.DataFrame, csv_dir: Path) -> None:
    print_latex_table_for_metric(
        df=df,
        metric_col="mean_in_one_sigma_interval",
        metric_tex="$C_{\\pm\\sigma}$",
        caption="Overview on step 2 results for $C_{\\pm\\sigma}$ (sigma-interval coverage).",
        label="tab:overview-step2-sigma-coverage",
        csv_out_path=csv_dir / "overview_step2_sigma_coverage.csv",
    )


def print_latex_table_execution_time(df: pd.DataFrame, csv_dir: Path) -> None:
    print_latex_table_for_metric(
        df=df,
        metric_col="execution_time_s",
        metric_tex="$ET$ [s]",
        caption="Overview on step 2 results for execution time $ET$.",
        label="tab:overview-step2-runtime",
        csv_out_path=csv_dir / "overview_step2_runtime.csv",
    )


def main():
    # Path to the aggregated results file
    csv_path = Path("analysis_results/aggregated_results2.csv")

    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path.resolve()}")
        return

    # Directory for the exported CSV tables
    tables_dir = csv_path.parent / "tables_step2"

    # Read the CSV
    df = pd.read_csv(csv_path)

    print("=== Aggregated Results Loaded ===")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Print LaTeX tables AND save as CSV
    print("\n=== LaTeX table for NRMSE ===\n")
    print_latex_table_nrmse(df, tables_dir)

    print("\n=== LaTeX table for Wasserstein distance ===\n")
    print_latex_table_wasserstein(df, tables_dir)

    print("\n=== LaTeX table for sigma-interval coverage ===\n")
    print_latex_table_sigma_coverage(df, tables_dir)

    print("\n=== LaTeX table for execution time ===\n")
    print_latex_table_execution_time(df, tables_dir)


if __name__ == "__main__":
    main()

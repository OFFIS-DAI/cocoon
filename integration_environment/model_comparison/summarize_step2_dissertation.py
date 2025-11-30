#!/usr/bin/env python3
"""
Script to read the aggregated results CSV file and print LaTeX tables
for different metrics per factor AND save the same tables as CSV files.

  - Factor T: traffic configuration (CBR, Poisson, CDSB)
  - Factor S: scenario duration (5 min, 10 min)
  - Factor ND: number of devices (5, 10, 20, 50)
  - Factor CN: communication network technology (LTE, LTE450, Ethernet, 5G)
"""

import pandas as pd
from pathlib import Path


def format_mean_std(sub_df: pd.DataFrame, metric_col: str, for_csv: bool = False) -> str:
    """
    Return mean/std formatting.

    - LaTeX mode (for_csv=False): '$mean\\pm std$'
    - CSV mode   (for_csv=True):  'mean, std'

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

    Values are plain 'mean, std' strings (no LaTeX math mode, no \\pm).

    Columns:
      Factor, Stage,
      Channel Model, Static Graph Model, Detailed Model, Ideal Model,
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
        """CSV-safe mean/std for given model_type (+optional split)."""
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
        det_val = csv_stat_for(sub, "detailed")
        ideal_val = csv_stat_for(sub, "ideal")
        meta_vals = [
            csv_stat_for(sub, "meta_model", split_name)
            for split_name, _abbr in meta_splits
        ]
        row = {
            "Factor": "T",
            "Stage": label_row,
            "Channel Model": ch_val,
            "Static Graph Model": sg_val,
            "Detailed Model": det_val,
            "Ideal Model": ideal_val,
        }
        for (split_name, abbr), val in zip(meta_splits, meta_vals):
            row[f"Meta-{abbr}"] = val
        rows.append(row)

    # ----------------- Factor S (Scenario Duration) -----------------------
    sub_5 = subset_by_duration("five")
    ch_5 = csv_stat_for(sub_5, "channel")
    sg_5 = csv_stat_for(sub_5, "static_graph")
    det_5 = csv_stat_for(sub_5, "detailed")
    ideal_5 = csv_stat_for(sub_5, "ideal")
    meta_5 = [
        csv_stat_for(sub_5, "meta_model", split_name)
        for split_name, _a in meta_splits
    ]
    row_5 = {
        "Factor": "S",
        "Stage": "5 min",
        "Channel Model": ch_5,
        "Static Graph Model": sg_5,
        "Detailed Model": det_5,
        "Ideal Model": ideal_5,
    }
    for (split_name, abbr), val in zip(meta_splits, meta_5):
        row_5[f"Meta-{abbr}"] = val
    rows.append(row_5)

    sub_10 = subset_by_duration("ten")
    ch_10 = csv_stat_for(sub_10, "channel")
    sg_10 = csv_stat_for(sub_10, "static_graph")
    det_10 = csv_stat_for(sub_10, "detailed")
    ideal_10 = csv_stat_for(sub_10, "ideal")
    meta_10 = [
        csv_stat_for(sub_10, "meta_model", split_name)
        for split_name, _a in meta_splits
    ]
    row_10 = {
        "Factor": "S",
        "Stage": "10 min",
        "Channel Model": ch_10,
        "Static Graph Model": sg_10,
        "Detailed Model": det_10,
        "Ideal Model": ideal_10,
    }
    for (split_name, abbr), val in zip(meta_splits, meta_10):
        row_10[f"Meta-{abbr}"] = val
    rows.append(row_10)

    # ----------------- Factor ND (Num Devices) ----------------------------
    nd_levels = [
        ('NumDevices.five', 5),
        ('NumDevices.ten', 10),
        ('NumDevices.twenty', 20),
        ('NumDevices.fifty', 50),
    ]
    for token, label_row in nd_levels:
        sub_nd = subset_by_num_devices(token)
        ch_nd = csv_stat_for(sub_nd, "channel")
        sg_nd = csv_stat_for(sub_nd, "static_graph")
        det_nd = csv_stat_for(sub_nd, "detailed")
        ideal_nd = csv_stat_for(sub_nd, "ideal")
        meta_nd = [
            csv_stat_for(sub_nd, "meta_model", split_name)
            for split_name, _a in meta_splits
        ]
        row_nd = {
            "Factor": "ND",
            "Stage": str(label_row),
            "Channel Model": ch_nd,
            "Static Graph Model": sg_nd,
            "Detailed Model": det_nd,
            "Ideal Model": ideal_nd,
        }
        for (split_name, abbr), val in zip(meta_splits, meta_nd):
            row_nd[f"Meta-{abbr}"] = val
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
        ch_net = csv_stat_for(sub_net, "channel")
        sg_net = csv_stat_for(sub_net, "static_graph")
        det_net = csv_stat_for(sub_net, "detailed")
        ideal_net = csv_stat_for(sub_net, "idea")
        meta_net = [
            csv_stat_for(sub_net, "meta_model", split_name)
            for split_name, _a in meta_splits
        ]
        row_net = {
            "Factor": "CN",
            "Stage": label_row,
            "Channel Model": ch_net,
            "Static Graph Model": sg_net,
            "Detailed Model": det_net,
            "Ideal Model": ideal_net,
        }
        for (split_name, abbr), val in zip(meta_splits, meta_net):
            row_net[f"Meta-{abbr}"] = val
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
    # 11 columns: Factor, Stages, 9 metric subcolumns
    print("\\begin{tabular}{lllllllllll}")
    print("\\hline")
    print(
        "\\textbf{Factor} & \\textbf{Stages} & "
        f"\\multicolumn{{9}}{{l}}{{\\textbf{{{metric_tex}}}}} \\\\ \\hline"
    )
    # Header: 4 base models + 5 meta-splits
    print(
        "\\textbf{CM} &  & "
        "Channel Model & Static Graph Model & Detailed Model & Ideal Model & "
        "\\multicolumn{5}{l}{Meta-Model} \\\\ \\hline"
    )
    print("\\textbf{C-TTS} &  &  &  &  &  & pa & tl & tm & sc & te \\\\ \\hline")

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
        det_val = latex_cell_for_model(sub, "detailed_model", metric_col)
        ideal_val = latex_cell_for_model(sub, "ideal_model", metric_col)
        meta_vals = [
            latex_cell_for_model(sub, "meta_model", metric_col, split_name)
            for split_name, _abbr in meta_splits
        ]
        prefix = "\\multirow{3}{*}{\\textbf{T}}" if i == 0 else " "
        print(
            f"{prefix} & {label_row} & {ch_val} & {sg_val} & {det_val} & {ideal_val} & "
            + " & ".join(meta_vals)
            + " \\\\"
        )

    print(" \\hline")

    # Factor S
    print("\\multirow{2}{*}{\\textbf{S}} & 5 min & ", end="")
    sub_5 = subset_by_duration("five")
    ch_5 = latex_cell_for_model(sub_5, "channel", metric_col)
    sg_5 = latex_cell_for_model(sub_5, "static_graph", metric_col)
    det_5 = latex_cell_for_model(sub_5, "detailed_model", metric_col)
    ideal_5 = latex_cell_for_model(sub_5, "ideal_model", metric_col)
    meta_5 = [
        latex_cell_for_model(sub_5, "meta_model", metric_col, split_name)
        for split_name, _a in meta_splits
    ]
    print(f"{ch_5} & {sg_5} & {det_5} & {ideal_5} & " + " & ".join(meta_5) + " \\\\")

    print(" & 10 min & ", end="")
    sub_10 = subset_by_duration("ten")
    ch_10 = latex_cell_for_model(sub_10, "channel", metric_col)
    sg_10 = latex_cell_for_model(sub_10, "static_graph", metric_col)
    det_10 = latex_cell_for_model(sub_10, "detailed_model", metric_col)
    ideal_10 = latex_cell_for_model(sub_10, "ideal_model", metric_col)
    meta_10 = [
        latex_cell_for_model(sub_10, "meta_model", metric_col, split_name)
        for split_name, _a in meta_splits
    ]
    print(f"{ch_10} & {sg_10} & {det_10} & {ideal_10} & " + " & ".join(meta_10) + " \\\\ \\hline")

    # Factor ND
    nd_levels = [
        ('NumDevices.five', 5),
        ('NumDevices.ten', 10),
        ('NumDevices.twenty', 20),
        ('NumDevices.fifty', 50),
    ]
    for i, (token, label_row) in enumerate(nd_levels):
        sub_nd = subset_by_num_devices(token)
        ch_nd = latex_cell_for_model(sub_nd, "channel", metric_col)
        sg_nd = latex_cell_for_model(sub_nd, "static_graph", metric_col)
        det_nd = latex_cell_for_model(sub_nd, "detailed_model", metric_col)
        ideal_nd = latex_cell_for_model(sub_nd, "ideal_model", metric_col)
        meta_nd = [
            latex_cell_for_model(sub_nd, "meta_model", metric_col, split_name)
            for split_name, _a in meta_splits
        ]
        factor_label = "\\textbf{ND}" if i == 0 else "\\textbf{}"
        print(
            f"{factor_label} & {label_row} & {ch_nd} & {sg_nd} & {det_nd} & {ideal_nd} & "
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
        det_net = latex_cell_for_model(sub_net, "detailed_model", metric_col)
        ideal_net = latex_cell_for_model(sub_net, "ideal_model", metric_col)
        meta_net = [
            latex_cell_for_model(sub_net, "meta_model", metric_col, split_name)
            for split_name, _a in meta_splits
        ]
        prefix = "\\multirow{4}{*}{\\textbf{CN}}" if i == 0 else " "
        print(
            f"{prefix} & {label_row} & {ch_net} & {sg_net} & {det_net} & {ideal_net} & "
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

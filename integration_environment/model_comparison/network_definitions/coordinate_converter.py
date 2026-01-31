#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import utm

IN_PATH = Path("devices_coords.csv")  # assumed path
OUT_PATH = Path("devices_coords_xy.csv")  # output CSV (with X,Y,X_rel,Y_rel)
PDF_PATH = Path("devices_coords_xy.pdf")  # output plot (relative coords)


def convert_coords(df: pd.DataFrame,
                   lat_col: str = "Lat",
                   lon_col: str = "Long") -> pd.DataFrame:
    """
    Convert WGS84 lat/lon to UTM easting/northing.
    Adds: X, Y, ZoneNumber, ZoneLetter, X_rel, Y_rel (relative to min X/Y).
    """
    # tolerate decimal commas
    lat = pd.to_numeric(df[lat_col].astype(str).str.replace(",", ".", regex=False), errors="raise")
    lon = pd.to_numeric(df[lon_col].astype(str).str.replace(",", ".", regex=False), errors="raise")

    easts, norths, zn, zl = [], [], [], []
    for la, lo in zip(lat, lon):
        e, n, znum, zlet = utm.from_latlon(la, lo)
        easts.append(e);
        norths.append(n);
        zn.append(znum);
        zl.append(zlet)

    out = df.copy()
    out["X"] = easts
    out["Y"] = norths

    out["ZoneNumber"] = zn
    out["ZoneLetter"] = zl

    # relative (scaled) coordinates: min is the origin (0,0)
    x0, y0 = out["X"].min(), out["Y"].min()
    out["X_rel"] = out["X"] - x0 + 100
    out["Y_rel"] = out["Y"] - y0 + 100
    out.attrs["origin_xy"] = (x0, y0)  # stash for plotting/logging
    return out


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_points(df: pd.DataFrame,
                x_col: str = "X_rel",
                y_col: str = "Y_rel",
                label_col: str | None = None,
                title: str | None = "Relative positions (min(X),min(Y) = 0,0)") -> None:
    """Scatter plot using relative coordinates (seaborn, serif font)."""

    sns.set_theme(style="white", font="serif")

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        s=20,
        ax=ax,
        legend=False
    )

    if label_col and label_col in df.columns:
        for _, r in df.iterrows():
            ax.text(r[x_col], r[y_col], str(r[label_col]), fontsize=9)

    ax.set_xlabel("X_rel (m)")
    ax.set_ylabel("Y_rel (m)")
    ax.set_aspect("equal")

    if PDF_PATH:
        fig.savefig(PDF_PATH, dpi=200, bbox_inches="tight", format="pdf")

    plt.close(fig)



def print_omnet_description(df: pd.DataFrame):
    print(f'node0: StandardHost ', '{', '\n// central entity \n @display("p=330,330");\n}')
    for i, row in df.iterrows():
        print(f'node{i+1}: StandardHost ', '{', '\n'
                                    '// ', row['name'], '\n',
              '@display("p=', row['X_rel'], ',', row['Y_rel'], '");\n}')


def main():
    # try common CSV formats
    try:
        df = pd.read_csv(IN_PATH)
    except Exception:
        df = pd.read_csv(IN_PATH, sep=";", decimal=",")

    df_xy = convert_coords(df)
    x0, y0 = df_xy.attrs.get("origin_xy", (df_xy["X"].min(), df_xy["Y"].min()))
    print(f"Origin (absolute UTM): X0={x0:.3f}, Y0={y0:.3f}")

    #df_xy.to_csv(OUT_PATH, index=False)
    print(f"Wrote: {OUT_PATH}")

    # plot relative coords (default); pick a nice label column if available
    label_guess = next((c for c in ["Address", "name", "Straße"] if c in df_xy.columns), None)
    plot_points(df_xy, label_col=label_guess)
    additional_coords = pd.read_csv('additional_coords.csv')
    print_omnet_description(pd.concat([df_xy, additional_coords], ignore_index=True))


if __name__ == "__main__":
    main()

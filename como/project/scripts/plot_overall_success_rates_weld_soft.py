import argparse
import os
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Headless plotting (robust for server / CI)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402


DISPLAY_NAME_RE = re.compile(
    r"^(Free|Fixed|Constrained)_(Weld|Soft)_(5|13)_seed(\d+)$"
)


def normalize_display_name(s: object) -> str:
    return str(s).strip().rstrip("/")


def parse_display_name(display_name: str) -> Optional[Tuple[str, str, int, int]]:
    """
    Returns: (arm_type, connect_type, muscle_count, seed) or None.
    """
    m = DISPLAY_NAME_RE.match(display_name)
    if not m:
        return None
    arm_type, connect_type, muscle_count_s, seed_s = m.groups()
    return arm_type, connect_type, int(muscle_count_s), int(seed_s)


def get_subplot_run_key(
    arm_type: str,
    connect_type: str,
    muscle_count: int,
    seed: int,
) -> str:
    # Matches how display_name is formatted in the exported CSV
    return f"{arm_type}_{connect_type}_{muscle_count}_seed{seed}"


def safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_runs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "display_name" not in df.columns:
        raise ValueError("CSV needs a 'display_name' column.")
    if "summary__train/success_rate" not in df.columns:
        raise ValueError("CSV needs 'summary__train/success_rate' column.")
    df["display_name_norm"] = df["display_name"].map(normalize_display_name)
    df["parsed"] = df["display_name_norm"].map(parse_display_name)
    # Keep only rows that match the expected naming convention.
    df = df.loc[df["parsed"].notna()].copy()
    if df.empty:
        raise ValueError(
            "No rows matched expected display_name format: "
            "'Free|Fixed|Constrained_Weld|Soft_5|13_seedN'."
        )
    df[["arm_type", "connect_type", "muscle_count", "seed"]] = pd.DataFrame(
        df["parsed"].tolist(), index=df.index
    )
    df["muscle_count"] = df["muscle_count"].astype(int)
    df["seed"] = df["seed"].astype(int)
    df["success_rate"] = df["summary__train/success_rate"].map(safe_float)
    return df


def plot_connect_type(
    df: pd.DataFrame,
    connect_type: str,
    seed: int,
    out_path: str,
    show_plot: bool,
) -> None:
    x_categories = ["Free", "Fixed", "Constrained"]
    # Only two discrete values on the left side (no "continuous" muscle count)
    muscle_rows = [13, 5]  # top -> bottom

    # 2x3 matrix: rows = muscle_count, cols = arm_type
    heat = np.full((len(muscle_rows), len(x_categories)), np.nan, dtype=float)

    for r_i, muscle_count in enumerate(muscle_rows):
        for c_i, arm_type in enumerate(x_categories):
            row = df.loc[
                (df["connect_type"] == connect_type)
                & (df["arm_type"] == arm_type)
                & (df["muscle_count"] == muscle_count)
                & (df["seed"] == seed)
            ]
            if row.empty:
                continue
            heat[r_i, c_i] = row.iloc[0]["success_rate"]

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)

    # Red (low/bad) -> warm mid tone -> green (high/good), without white center.
    cmap = LinearSegmentedColormap.from_list(
        "red_warm_green",
        [
            (0.00, "#e53935"),  # red (bad)
            (0.50, "#f6c177"),  # warm light mid tone
            (1.00, "#43a047"),  # green (good)
        ],
        N=256,
    )
    norm = PowerNorm(gamma=1.8, vmin=0.0, vmax=1.0)

    im = ax.imshow(
        heat,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate [%]", rotation=90, fontweight="bold")
    cbar.set_ticks([0.0, 0.5, 0.8, 0.9, 1.0])
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _pos: f"{x * 100:.0f}%")
    )

    ax.set_xticks(range(len(x_categories)), x_categories)
    ax.set_yticks(range(len(muscle_rows)), [str(m) for m in muscle_rows])
    ax.set_xlabel("Arm Configuration", fontweight="bold")
    ax.set_ylabel("Number of muscle actuators", fontweight="bold")

    # Subtle grid/border around the 6 fields
    ax.set_xticks(np.arange(-0.5, len(x_categories), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(muscle_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2, alpha=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with its success rate value
    for r_i in range(len(muscle_rows)):
        for c_i in range(len(x_categories)):
            val = heat[r_i, c_i]
            if np.isnan(val):
                txt = "NA"
                color = "black"
            else:
                txt = f"{val * 100:.1f}%"
                # readable text color across the colormap
                color = "black" if val < 0.6 else "white"
            ax.text(c_i, r_i, txt, ha="center", va="center", fontsize=11, color=color)

    ax.set_title(
        f"{connect_type} Configuration",
        fontsize=14,
        fontweight="bold",
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create twin-axis subplot grids from exported runs CSV."
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "wandb_runs_export.csv"),
        help="Path to wandb_runs_export.csv",
    )
    ap.add_argument("--seed", type=int, default=1, help="Seed id to select (default: 1)")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: same directory as csv)",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (always saves files).",
    )
    args = ap.parse_args()

    csv_path = args.csv
    outdir = args.outdir or os.path.dirname(os.path.abspath(csv_path))
    show_plot = not args.no_show

    df = load_runs(csv_path)

    # Two separate figures for Weld vs Soft.
    for connect_type in ["Weld", "Soft"]:
        out_path = os.path.join(
            outdir,
            f"overall_success_twins_{connect_type.lower()}_seed{args.seed}.png",
        )
        plot_connect_type(
            df=df,
            connect_type=connect_type,
            seed=args.seed,
            out_path=out_path,
            show_plot=show_plot,
        )


if __name__ == "__main__":
    main()


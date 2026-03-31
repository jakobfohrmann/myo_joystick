import argparse
import os
from typing import List

import pandas as pd

# Headless plotting (robust for server / CI)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402


def format_steps(x: float, _pos: int) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return f"{int(x)}"


def get_run_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if "train/success_rate" not in c:
            continue
        if "__MIN" in c or "__MAX" in c:
            continue
        cols.append(c)
    if not cols:
        raise ValueError(
            "Could not find run columns containing 'train/success_rate' "
            "(excluding __MIN/__MAX)."
        )
    return cols


def extract_run_name(col_name: str) -> str:
    run_name = col_name.split(" - ", 1)[0].strip()
    return run_name.replace("_seed1", "")


def color_for_run(run_name: str, idx: int) -> str:
    # User-requested highlight for this run.
    if run_name.startswith("Constrained_Soft_13"):
        return "#0B3D91"  # dark blue

    palette = [
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#F28E2B",
        "#B07AA1",
        "#EDC948",
        "#FF9DA7",
        "#9C755F",
        "#4E79A7",
        "#BAB0AC",
    ]
    return palette[idx % len(palette)]


def prepare_run_series(df: pd.DataFrame, step_col: str, value_col: str) -> pd.DataFrame:
    out = df[[step_col, value_col]].copy()
    out[step_col] = pd.to_numeric(out[step_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(step_col)
    out = out.drop_duplicates(subset=step_col, keep="last")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a styled multi-run success-rate plot from success_rates.csv."
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "success_rates.csv"),
        help="Path to success_rates.csv",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "success_rates_pretty.png"),
        help="Output PNG path",
    )
    ap.add_argument(
        "--step-col",
        default="Step",
        help="Step column name (default: Step)",
    )
    ap.add_argument(
        "--x-max",
        type=float,
        default=1_250_000,
        help="Maximum x-axis step value (default: 1,250,000)",
    )
    ap.add_argument(
        "--roll-window",
        type=int,
        default=None,
        help="Rolling mean window size. Default: 3%% of each run length, min 10",
    )
    args = ap.parse_args()

    source_df = pd.read_csv(args.csv)
    if args.step_col not in source_df.columns:
        raise ValueError(f"Step column not found: {args.step_col}")
    run_cols = get_run_columns(source_df)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6.8), dpi=170)

    for i, col in enumerate(run_cols):
        run_name = extract_run_name(col)
        run_df = prepare_run_series(source_df, args.step_col, col)
        if run_df.empty:
            continue

        if args.roll_window is None:
            roll_window = max(10, int(len(run_df) * 0.03))
        else:
            roll_window = max(1, args.roll_window)
        run_df["smooth"] = run_df[col].rolling(window=roll_window, min_periods=1).mean()

        color = color_for_run(run_name, i)
        is_highlight = run_name.startswith("Constrained_Soft_13")

        ax.plot(
            run_df[args.step_col],
            run_df["smooth"],
            color=color,
            linewidth=3.0 if is_highlight else 1.8,
            alpha=0.98 if is_highlight else 0.88,
            label=run_name,
            zorder=5 if is_highlight else 3,
        )

    ax.set_xlim(0, args.x_max)
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.set_title("Success Rate Progress Over Training", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Place legend inside (bottom-right), slightly larger for readability.
    ax.legend(
        loc="lower right",
        frameon=True,
        fontsize=10,
        title="Runs",
        title_fontsize=11,
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

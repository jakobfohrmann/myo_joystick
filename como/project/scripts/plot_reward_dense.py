import argparse
import os
from typing import Optional

import pandas as pd

# Headless plotting (robust for server / CI)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


def format_steps(x: float, _pos: int) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"{int(x / 1_000)}k"
    return f"{int(x)}"


def get_value_column(df: pd.DataFrame, explicit_col: Optional[str]) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise ValueError(f"Column not found: {explicit_col}")
        return explicit_col

    candidates = [
        c
        for c in df.columns
        if "dense_sum" in c and "__MIN" not in c and "__MAX" not in c
    ]
    if not candidates:
        raise ValueError(
            "Could not find reward column containing 'dense_sum' "
            "(excluding __MIN/__MAX)."
        )
    return candidates[0]


def prepare_data(csv_path: str, step_col: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if step_col not in df.columns:
        raise ValueError(f"Step column not found: {step_col}")
    if value_col not in df.columns:
        raise ValueError(f"Value column not found: {value_col}")

    out = df[[step_col, value_col]].copy()
    out[step_col] = pd.to_numeric(out[step_col], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna().sort_values(step_col)

    # Keep only one value per step to avoid duplicated plotting points.
    out = out.drop_duplicates(subset=step_col, keep="last")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a styled dense reward progression plot from reward_dense.csv."
    )
    ap.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "reward_dense.csv"),
        help="Path to reward_dense.csv",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "reward_dense_pretty.png"),
        help="Output PNG path",
    )
    ap.add_argument(
        "--step-col",
        default="Step",
        help="Step column name (default: Step)",
    )
    ap.add_argument(
        "--value-col",
        default=None,
        help="Reward column name (default: auto-detect dense_sum)",
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
        help="Rolling mean window size. Default: 3%% of rows, min 10",
    )
    args = ap.parse_args()

    source_df = pd.read_csv(args.csv)
    value_col = get_value_column(source_df, args.value_col)
    df = prepare_data(args.csv, args.step_col, value_col)

    if args.roll_window is None:
        roll_window = max(10, int(len(df) * 0.03))
    else:
        roll_window = max(1, args.roll_window)

    df["smooth"] = df[value_col].rolling(window=roll_window, min_periods=1).mean()

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)

    ax.plot(
        df[args.step_col],
        df[value_col],
        color="#9ecae1",
        alpha=0.35,
        linewidth=1.1,
        label="Reward (raw)",
    )
    ax.plot(
        df[args.step_col],
        df["smooth"],
        color="#0B3D91",
        linewidth=2.8,
        label=f"Rolling mean (window={roll_window})",
    )

    ax.set_xlim(0, args.x_max)
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))

    ax.set_title("Dense Reward Progress Over Training", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("Steps", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

import argparse
import os
import re
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
QUADRANT_TO_TARGET_INDICES: Dict[str, List[int]] = {
    "I": [1, 5, 9, 15, 17],
    "II": [2, 8, 12, 14, 20],
    "III": [4, 6, 10, 16, 18],
    "IV": [3, 7, 11, 13, 19],
}


def run_id_from_path(run_path: str) -> str:
    return str(run_path).rstrip("/").split("/")[-1]


def target_to_quadrant_map() -> Dict[int, str]:
    out: Dict[int, str] = {}
    for quadrant, target_indices in QUADRANT_TO_TARGET_INDICES.items():
        for idx in target_indices:
            out[idx] = quadrant
    return out


def load_run_history_with_retries(
    run: Any,
    samples: int,
    retries: int,
    retry_wait_seconds: float,
    keys: List[str],
) -> pd.DataFrame:
    last_error = None
    last_status = "not_started"
    for attempt in range(1, retries + 1):
        try:
            df = run.history(keys=keys, samples=samples, pandas=True)
            if df is not None and not df.empty:
                return df
            last_status = f"history(keys=...) empty on attempt {attempt}"
        except Exception as exc:
            last_error = exc
            last_status = f"history(keys=...) error on attempt {attempt}: {exc}"
            if attempt < retries:
                wait_s = retry_wait_seconds * attempt
                print(
                    f"history timeout/error (try {attempt}/{retries}), retry in {wait_s:.1f}s: {exc}"
                )
                time.sleep(wait_s)
    # Fallback 1: try unfiltered history (some W&B runs return sparse keys only here).
    try:
        df = run.history(samples=samples, pandas=True)
        if df is not None and not df.empty:
            return df
        last_status = "history(unfiltered) returned empty dataframe"
    except Exception:
        last_error = last_error or Exception("history(unfiltered) failed")
        last_status = "history(unfiltered) raised exception"
    # Fallback: streaming API with exact keys only (often more robust on large runs).
    try:
        rows = list(run.scan_history(keys=keys))
        if not rows:
            raise RuntimeError("scan_history returned no rows")
        return pd.DataFrame(rows)
    except Exception as fallback_exc:
        last_status = f"scan_history(keys=...) failed: {fallback_exc}"

    # Fallback 3: full scan without keys and local filtering.
    # Some runs expose sparse metric rows only through unfiltered scan_history.
    try:
        keep_keys = set(keys + ["global_step", "Step"])
        scanned_rows = []
        for row in run.scan_history():
            if not isinstance(row, dict):
                continue
            if any(k in row for k in keys):
                scanned_rows.append({k: row.get(k) for k in keep_keys if k in row})
        if scanned_rows:
            return pd.DataFrame(scanned_rows)
        last_status = "scan_history(unfiltered) returned no rows containing requested keys"
    except Exception as fallback_exc_2:
        last_status = f"scan_history(unfiltered) failed: {fallback_exc_2}"

    raise RuntimeError(
        f"Could not load W&B history after {retries} tries. "
        f"Last status: {last_status}. Last history error: {last_error}"
    )


def extract_radius_columns(history_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    canonical_cols = [f"curriculum/rad_T{i}" for i in range(1, 21)]
    col_map: Dict[int, str] = {}

    # 1) exact canonical hits
    for i, col in enumerate(canonical_cols, start=1):
        if col in history_df.columns:
            col_map[i] = col

    # 2) flexible hits (e.g. exported names with prefixes, case differences)
    #    Example: "Constrained_... - curriculum/rad_T1"
    if len(col_map) < 20:
        for col in history_df.columns:
            col_s = str(col)
            col_l = col_s.lower()
            if "__min" in col_l or "__max" in col_l:
                continue
            m = re.search(r"curriculum/rad_t(\d+)", col_l)
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= 20 and idx not in col_map:
                col_map[idx] = col_s

    present_indices = sorted(col_map.keys())
    if not present_indices:
        preview_cols = ", ".join([str(c) for c in list(history_df.columns)[:20]])
        raise ValueError(
            "No curriculum/rad_T* columns found in run history. "
            f"First columns seen: {preview_cols}"
        )

    if "_step" in history_df.columns:
        step_col = "_step"
    elif "global_step" in history_df.columns:
        step_col = "global_step"
    elif "Step" in history_df.columns:
        step_col = "Step"
    else:
        raise ValueError("Neither _step, global_step nor Step exists in run history.")

    selected_cols = [step_col] + [col_map[i] for i in present_indices]
    out = history_df.loc[:, selected_cols].copy()
    out = out.rename(columns={step_col: "step"})
    # Normalize column names to canonical target keys.
    out = out.rename(columns={col_map[i]: f"curriculum/rad_T{i}" for i in present_indices})
    out = out.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    out = out.reset_index(drop=True)
    normalized_cols = [f"curriculum/rad_T{i}" for i in present_indices]
    print(f"Gefundene Radius-Spalten: {len(normalized_cols)}/20")
    return out, normalized_cols


def compute_auc_per_target(radius_df: pd.DataFrame, radius_cols: List[str]) -> pd.DataFrame:
    x = pd.to_numeric(radius_df["step"], errors="coerce")
    valid_x = x.notna()
    if valid_x.sum() < 2:
        raise ValueError("Need at least two valid step values for AUC computation.")

    x_vals = x[valid_x].to_numpy(dtype=np.float64)
    step_span = float(x_vals[-1] - x_vals[0])
    if step_span <= 0:
        raise ValueError("Step span must be > 0 for normalized AUC.")

    quadrant_by_target = target_to_quadrant_map()
    rows = []
    for target_idx in range(1, 21):
        col = f"curriculum/rad_T{target_idx}"
        if col not in radius_cols:
            rows.append(
                {
                    "target_idx": target_idx,
                    "quadrant": quadrant_by_target.get(target_idx, "NA"),
                    "auc": np.nan,
                    "auc_normalized": np.nan,
                    "n_points": 0,
                    "start_radius": np.nan,
                    "end_radius": np.nan,
                }
            )
            continue

        y = pd.to_numeric(radius_df[col], errors="coerce")
        ser = pd.DataFrame({"step": x, "radius": y}).dropna(subset=["step"])
        ser["radius"] = ser["radius"].ffill()
        ser = ser.dropna(subset=["radius"])
        if len(ser) < 2:
            rows.append(
                {
                    "target_idx": target_idx,
                    "quadrant": quadrant_by_target.get(target_idx, "NA"),
                    "auc": np.nan,
                    "auc_normalized": np.nan,
                    "n_points": int(len(ser)),
                    "start_radius": float(ser["radius"].iloc[0]) if len(ser) else np.nan,
                    "end_radius": float(ser["radius"].iloc[-1]) if len(ser) else np.nan,
                }
            )
            continue

        xs = ser["step"].to_numpy(dtype=np.float64)
        ys = ser["radius"].to_numpy(dtype=np.float64)
        auc = float(np.trapz(ys, xs))
        auc_norm = auc / step_span
        rows.append(
            {
                "target_idx": target_idx,
                "quadrant": quadrant_by_target.get(target_idx, "NA"),
                "auc": auc,
                "auc_normalized": auc_norm,
                "n_points": int(len(ser)),
                "start_radius": float(ys[0]),
                "end_radius": float(ys[-1]),
            }
        )

    auc_df = pd.DataFrame(rows).sort_values("target_idx").reset_index(drop=True)
    return auc_df


def aggregate_quadrants(auc_df: pd.DataFrame) -> pd.DataFrame:
    valid = auc_df.dropna(subset=["auc_normalized"]).copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "quadrant",
                "n_targets",
                "auc_normalized_mean",
                "auc_normalized_median",
                "auc_mean",
                "auc_median",
            ]
        )
    out = (
        valid.groupby("quadrant", as_index=False)
        .agg(
            n_targets=("target_idx", "count"),
            auc_normalized_mean=("auc_normalized", "mean"),
            auc_normalized_median=("auc_normalized", "median"),
            auc_mean=("auc", "mean"),
            auc_median=("auc", "median"),
        )
        .sort_values("quadrant")
        .reset_index(drop=True)
    )
    return out


def make_plot(auc_df: pd.DataFrame, quadrant_df: pd.DataFrame, out_png: str, show: bool) -> None:
    color_map = {"I": "#1f77b4", "II": "#ff7f0e", "III": "#2ca02c", "IV": "#d62728", "NA": "#7f7f7f"}
    target_colors = [color_map.get(q, "#7f7f7f") for q in auc_df["quadrant"].tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax0 = axes[0]
    x = auc_df["target_idx"].to_numpy(dtype=np.int32)
    y = auc_df["auc_normalized"].to_numpy(dtype=np.float64)
    ax0.bar(x, y, color=target_colors, edgecolor="black", linewidth=0.5)
    ax0.set_title("AUC pro Target (kleiner = schneller Radius-Reduktion)")
    ax0.set_xlabel("Target Index")
    ax0.set_ylabel("Normierte AUC")
    ax0.set_xticks(x)
    ax0.grid(True, axis="y", alpha=0.3, linestyle="--")

    legend_labels = []
    for q in ["I", "II", "III", "IV"]:
        legend_labels.append(
            plt.Line2D([0], [0], marker="s", color="w", label=f"Quadrant {q}", markerfacecolor=color_map[q], markersize=10)
        )
    ax0.legend(handles=legend_labels, loc="best")

    ax1 = axes[1]
    if quadrant_df.empty:
        ax1.text(0.5, 0.5, "Keine gültigen Quadranten-AUCs", ha="center", va="center")
        ax1.set_axis_off()
    else:
        qx = quadrant_df["quadrant"].tolist()
        qy = quadrant_df["auc_normalized_mean"].to_numpy(dtype=np.float64)
        qcolors = [color_map.get(q, "#7f7f7f") for q in qx]
        ax1.bar(qx, qy, color=qcolors, edgecolor="black", linewidth=0.8)
        ax1.set_title("Quadrantenvergleich (Mittelwert normierte AUC)")
        ax1.set_xlabel("Quadrant")
        ax1.set_ylabel("Mittlere normierte AUC")
        ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Curriculum Radius-Verläufe nach Target und Quadrant", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Plot geschrieben: {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Exportiert curriculum/rad_T1..rad_T20 aus einem W&B-Run, "
            "berechnet AUC pro Target und erzeugt Quadrantenvergleich."
        ),
        epilog=(
            "Beispiel:\n"
            "python scripts/export_wandb_radius_trajectories.py "
            "--run-path /si69juga-universit-t-leipzig/thumb_reach/runs/rjbxf4lf "
            "--output-dir scripts"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-path",
        default="/si69juga-universit-t-leipzig/thumb_reach/runs/rjbxf4lf",
        help="W&B run path im Format /entity/project/runs/run_id",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts",
        help="Ausgabeordner für CSV/PNG Dateien",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=5000,
        help="Anzahl Samples für run.history()",
    )
    parser.add_argument(
        "--history-retries",
        type=int,
        default=4,
        help="Retry-Anzahl für run.history()",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
        help="Basis-Wartezeit in Sekunden für linearen Retry-Backoff",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Plot interaktiv anzeigen",
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=60,
        help="W&B API timeout in Sekunden (default: 60)",
    )
    args = parser.parse_args()
    import wandb  # pyright: ignore[reportMissingImports]

    os.makedirs(args.output_dir, exist_ok=True)
    run_id = run_id_from_path(args.run_path)

    raw_csv = os.path.join(args.output_dir, f"wandb_rad_trajectories_{run_id}.csv")
    auc_targets_csv = os.path.join(args.output_dir, f"wandb_rad_auc_targets_{run_id}.csv")
    auc_quadrants_csv = os.path.join(args.output_dir, f"wandb_rad_auc_quadrants_{run_id}.csv")
    plot_png = os.path.join(args.output_dir, f"wandb_rad_auc_comparison_{run_id}.png")

    print(f"Lade W&B Run: {args.run_path}")
    api = wandb.Api(timeout=args.api_timeout)
    run = api.run(args.run_path)
    history_keys = ["_step"] + [f"curriculum/rad_T{i}" for i in range(1, 21)]

    history_df = load_run_history_with_retries(
        run=run,
        samples=args.history_samples,
        retries=args.history_retries,
        retry_wait_seconds=args.retry_wait_seconds,
        keys=history_keys,
    )
    radius_df, radius_cols = extract_radius_columns(history_df)
    radius_df.to_csv(raw_csv, index=False)
    print(f"Raw CSV geschrieben: {raw_csv} (rows={len(radius_df)}, radius_cols={len(radius_cols)})")

    auc_df = compute_auc_per_target(radius_df=radius_df, radius_cols=radius_cols)
    auc_df.to_csv(auc_targets_csv, index=False)
    print(f"Target-AUC CSV geschrieben: {auc_targets_csv}")

    quadrant_df = aggregate_quadrants(auc_df)
    quadrant_df.to_csv(auc_quadrants_csv, index=False)
    print(f"Quadranten-AUC CSV geschrieben: {auc_quadrants_csv}")

    make_plot(auc_df=auc_df, quadrant_df=quadrant_df, out_png=plot_png, show=args.show)

    print("Fertig.")
    print(f"- {raw_csv}")
    print(f"- {auc_targets_csv}")
    print(f"- {auc_quadrants_csv}")
    print(f"- {plot_png}")


if __name__ == "__main__":
    main()

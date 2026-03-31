import argparse
import difflib
import re
import time
from datetime import datetime
from typing import Dict, List

import pandas as pd
import wandb


# Aus deiner neuen Run-Liste
DEFAULT_RUN_NAMES = [
    "run-20260330_022817-vwzph1iv/",
    "run-20260330_022817-bcscfkzw/",
    "run-20260330_014815-1vfe4ho8/",
    "run-20260330_013633-gof0qjf1/",
    "run-20260330_013009-2m9i080s/",
    "run-20260330_012153-hfamwgec/",
    "run-20260330_012003-leh12wyw/",
    "run-20260330_011953-uxviz6io/",
    "run-20260330_011903-3czln8k7/",
    "run-20260330_011815-ksrgoomi/",
    "run-20260330_011548-cf2z2d10/",
    "run-20260330_011526-nkpaq2v6/",
    "run-20260330_011208-ofn8onxd/",
    "run-20260330_010915-do1f6ke0/",
    "run-20260330_010517-rnmb34d3/",
    "run-20260330_010347-1vcqruvr/",
    "run-20260330_010347-iyc2kuaj/",
    "run-20260330_010346-xqiwq9kj/",
    "run-20260330_010346-4oi8a7hs/",
    "run-20260330_010035-8w4ufkv2/",
    "run-20260330_005822-krvcvz6u/",
    "run-20260330_005632-ozkx1s7b/",
    "run-20260330_005623-ogllflff/",
    "run-20260330_005620-bptuavyz/",
    "run-20260329_224313-ay7mzrt5/",
    "run-20260329_224257-yr1ejqhw/",
    "run-20260329_223938-xvgusy0g/",
    "run-20260329_223911-ghnu1kj3/",
    "run-20260329_223348-rm7ln5zl/",
    "run-20260329_223002-7o6dsk0e/",
    "run-20260329_212747-91azor05/",
    "run-20260329_212212-9wce3hdj/",
    "run-20260329_212146-pv3m3qsp/",
    "run-20260329_205101-am4er3pk/",
    "run-20260329_204427-01egc1nb/",
    "run-20260329_204119-8hga2qn7/",
    "run-20260329_203448-wmqcx0am/",
    "run-20260329_202509-57as4ic8/",
    "run-20260329_202410-vb6e77ct/",
    "run-20260329_202342-fe5i1qe5/",
    "run-20260329_202309-7yhkfqwe/",
    "run-20260329_201257-5h5ek0xq/",
    "run-20260329_201133-cd7ku5ne/",
    "run-20260329_194702-ol5soaa4/",
    "run-20260329_194317-g2ojfwwc/",
    "run-20260329_194219-l0ilpurk/",
    "run-20260329_193803-zznog035/",
    "run-20260329_193713-bbglpdrm/",
    "run-20260329_193150-gptw0aco/",
    "run-20260329_192804-b8jydtcc/",
    "run-20260329_192735-kthgio5h/",
    "run-20260329_192530-a7wm7dxs/",
    "run-20260329_191717-2tm1e1q1/",
    "run-20260329_191417-7fmhm8tw/",
    "run-20260329_191227-452netqx/",
    "run-20260329_191219-3ay299d6/",
    "run-20260329_191211-tgecba2y/",
    "run-20260329_190428-qb1nqwdy/",
    "run-20260329_190421-t5q6gjyw/",
    "run-20260329_190407-rjbxf4lf/",
    "run-20260329_175741-yhfhas01/",
    "run-20260329_175413-rx65pbjp/",
]


def normalize_run_name(run_name: str) -> str:
    """Normalisiert Run-Strings (z.B. entfernt abschließendes "/")."""
    return str(run_name).strip().rstrip("/")


def run_id_from_name(run_name: str) -> str:
    """
    Beispiel:
    run-20260324_103103-wjggct8f -> wjggct8f
    """
    run_name = normalize_run_name(run_name)
    return run_name.rsplit("-", 1)[-1]


def get_last_history_values(
    run: wandb.apis.public.Run,
    history_samples: int,
    history_retries: int,
    retry_wait_seconds: float,
) -> Dict[str, object]:
    """
    Holt letzte verfügbare Werte je Metrik aus der Run-History.
    """
    last_error = None
    history_df = None
    for attempt in range(1, history_retries + 1):
        try:
            # Weniger Samples reduziert Timeout-Risiko deutlich.
            history_df = run.history(samples=history_samples, pandas=True)
            break
        except Exception as exc:
            last_error = exc
            if attempt < history_retries:
                wait_s = retry_wait_seconds * attempt
                print(
                    f"  history timeout/error (Versuch {attempt}/{history_retries}), "
                    f"retry in {wait_s:.1f}s: {exc}"
                )
                time.sleep(wait_s)
            else:
                print(
                    f"  history übersprungen nach {history_retries} Versuchen: {exc}"
                )
                return {"history_error": str(last_error)}
    if history_df is None or history_df.empty:
        return {}

    # W&B interne Spalten ausfiltern
    history_df = history_df.loc[
        :, [col for col in history_df.columns if not str(col).startswith("_")]
    ]
    if history_df.empty:
        return {}

    # Letzten bekannten Wert pro Spalte nehmen
    last_values = history_df.ffill().iloc[-1].to_dict()
    return {f"history__{k}": v for k, v in last_values.items()}


def get_summary_values(run: wandb.apis.public.Run) -> Dict[str, object]:
    """
    Holt Summary-Metriken aus W&B (meist finale Kennzahlen).
    """
    summary = dict(run.summary) if run.summary is not None else {}
    return {f"summary__{k}": v for k, v in summary.items()}


def should_keep_metric(metric_name: str) -> bool:
    """
    Behält nur die tatsächlich im Projekt geloggten Kernmetriken:
    - train/success_rate
    - train/sr_T{idx} (pro Target)
    - kein curriculum (damit keine doppelten Spalten entstehen)
    - Completion-Time Keys (falls vorhanden)
    - ep_length als Fallback zur Berechnung einer Completion-Time in Sekunden
    """
    name = metric_name.lower()
    if name == "train/success_rate":
        return True
    if re.match(r"^train/sr_t\d+$", name):
        return True
    if "completion" in name and "time" in name:
        return True
    if name in {"rollout/ep_length_mean", "eval/mean_ep_length"}:
        return True
    return False


def compute_quadrant_success_rates(
    row: Dict[str, object],
    quadrant_to_target_indices: Dict[str, List[int]],
) -> Dict[str, object]:
    """
    Leitet Quadranten-Erfolgssraten aus train/sr_T1..T20 ab.

    Ergebnis: Für jeden Quadranten Mittelwert der 5 passenden Target-Success-Rates.
    """

    def get_numeric(v: object):
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    out: Dict[str, object] = {}
    for quadrant, indices in quadrant_to_target_indices.items():
        values: List[float] = []
        for idx in indices:
            key_summary = f"summary__train/sr_T{idx}"
            key_history = f"history__train/sr_T{idx}"
            v = row.get(key_summary, None)
            if v is None:
                v = row.get(key_history, None)
            nv = get_numeric(v)
            if nv is not None:
                values.append(nv)

        out[f"derived__success_rate_quadrant_{quadrant}"] = (
            sum(values) / len(values) if values else None
        )
    return out


def parse_run_timestamp(run_name: str) -> str:
    run_name = normalize_run_name(run_name)
    m = re.match(r"^run-(\d{8}_\d{6})-[a-z0-9]+$", run_name)
    return m.group(1) if m else ""


def created_at_to_compact_ts(created_at: str) -> str:
    if not created_at:
        return ""
    try:
        dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return dt.strftime("%Y%m%d_%H%M%S")
    except Exception:
        return ""


def build_project_run_index(api: wandb.Api, entity: str, project: str):
    runs = list(api.runs(f"{entity}/{project}"))
    by_id = {str(r.id): r for r in runs}
    by_ts: Dict[str, List[wandb.apis.public.Run]] = {}
    for r in runs:
        ts = created_at_to_compact_ts(getattr(r, "created_at", ""))
        if ts:
            by_ts.setdefault(ts, []).append(r)
    return by_id, by_ts


def resolve_run(
    api: wandb.Api,
    entity: str,
    project: str,
    run_name: str,
    by_id: Dict[str, wandb.apis.public.Run],
    by_ts: Dict[str, List[wandb.apis.public.Run]],
):
    run_name_norm = normalize_run_name(run_name)
    requested_id = run_id_from_name(run_name_norm)
    run_path = f"/{entity}/{project}/runs/{requested_id}"

    # 1) Direkter Lookup
    try:
        return api.run(run_path), run_path, "direct_id"
    except Exception:
        pass

    # 2) Exakter ID-Treffer im vorab geladenen Projektindex
    if requested_id in by_id:
        r = by_id[requested_id]
        return r, f"/{entity}/{project}/runs/{r.id}", "project_index_id"

    # 3) Timestamp + ähnliche ID (hilft bei OCR-Fehlern in Run-ID)
    ts = parse_run_timestamp(run_name_norm)
    candidates = by_ts.get(ts, [])
    if candidates:
        cand_ids = [str(c.id) for c in candidates]
        best = difflib.get_close_matches(requested_id, cand_ids, n=1, cutoff=0.4)
        if best:
            r = by_id[best[0]]
            return r, f"/{entity}/{project}/runs/{r.id}", "timestamp_fuzzy_id"

    # 4) Global fuzzy als letzte Rettung
    best_global = difflib.get_close_matches(requested_id, list(by_id.keys()), n=1, cutoff=0.75)
    if best_global:
        r = by_id[best_global[0]]
        return r, f"/{entity}/{project}/runs/{r.id}", "global_fuzzy_id"

    raise ValueError(f"Run nicht auflösbar: {run_name} (requested id: {requested_id})")


def extract_selected_summary_metrics(run: wandb.apis.public.Run) -> Dict[str, object]:
    """
    Nimmt bevorzugt Summary-Metriken (stabil/schnell, keine große History-Abfrage).
    """
    out: Dict[str, object] = {}
    summary = dict(run.summary) if run.summary is not None else {}
    for key, value in summary.items():
        key_str = str(key)
        if should_keep_metric(key_str):
            out[f"summary__{key_str}"] = value
    return out


def extract_selected_history_metrics(
    run: wandb.apis.public.Run,
    history_samples: int,
    history_retries: int,
    retry_wait_seconds: float,
) -> Dict[str, object]:
    """
    Fallback: falls gewünschte Metriken nicht in Summary liegen.
    """
    history_values = get_last_history_values(
        run=run,
        history_samples=history_samples,
        history_retries=history_retries,
        retry_wait_seconds=retry_wait_seconds,
    )

    out: Dict[str, object] = {}
    for key, value in history_values.items():
        key_str = str(key)
        if key_str == "history_error":
            out[key_str] = value
            continue
        # key format: history__<metric>
        raw_metric = key_str.removeprefix("history__")
        if should_keep_metric(raw_metric):
            out[key_str] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exportiert mehrere W&B Runs als CSV (1 Zeile pro Run)."
    )
    parser.add_argument(
        "--entity",
        default="si69juga-universit-t-leipzig",
        help="W&B entity / team name",
    )
    parser.add_argument(
        "--project",
        default="thumb_reach",
        help="W&B project name",
    )
    parser.add_argument(
        "--output",
        default="wandb_runs_export.csv",
        help="Pfad zur Ausgabe-CSV",
    )
    parser.add_argument(
        "--use-default-runs",
        action="store_true",
        help="Nutzt die 36 Run-Namen aus dem Skript (Screenshots).",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=[],
        help="Zusätzlicher Run-Name im Format run-YYYYMMDD_hhmmss-<run_id>. Mehrfach nutzbar.",
    )
    parser.add_argument(
        "--history-samples",
        type=int,
        default=2000,
        help="Max. Anzahl History-Samples pro Run (kleiner = stabiler/schneller).",
    )
    parser.add_argument(
        "--history-retries",
        type=int,
        default=4,
        help="Anzahl Retry-Versuche beim Laden der History.",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=float,
        default=2.0,
        help="Basis-Wartezeit in Sekunden zwischen Retries (linearer Backoff).",
    )
    parser.add_argument(
        "--history-fallback",
        action="store_true",
        help=(
            "Falls gewünschte Kennzahlen nicht in Summary stehen, zusätzlich gefilterte "
            "History-Werte laden (kann langsamer sein)."
        ),
    )
    parser.add_argument(
        "--env-hz",
        type=float,
        default=50.0,
        help="Schrittfrequenz (Hz), um aus ep_length eine Completion-Time in Sekunden abzuleiten.",
    )
    args = parser.parse_args()

    run_names: List[str] = []
    if args.use_default_runs:
        run_names.extend(DEFAULT_RUN_NAMES)
    run_names.extend(args.run_name)

    if not run_names:
        raise ValueError(
            "Keine Runs angegeben. Nutze --use-default-runs oder mindestens einmal --run-name."
        )

    api = wandb.Api()
    by_id, by_ts = build_project_run_index(api, args.entity, args.project)
    rows = []

    for run_name in run_names:
        run_name_norm = normalize_run_name(run_name)
        run_id = run_id_from_name(run_name_norm)
        run_path = f"/{args.entity}/{args.project}/runs/{run_id}"
        print(f"Lade {run_path}")

        try:
            run, resolved_run_path, resolved_via = resolve_run(
                api=api,
                entity=args.entity,
                project=args.project,
                run_name=run_name_norm,
                by_id=by_id,
                by_ts=by_ts,
            )
        except Exception as exc:
            rows.append(
                {
                    "run_name": run_name,
                    "run_id": run_id,
                    "run_path": run_path,
                    "load_error": str(exc),
                }
            )
            continue

        row: Dict[str, object] = {
            "run_name": run_name,
            "run_id": run_id,
            "run_path": run_path,
            "resolved_run_id": str(run.id),
            "resolved_run_path": resolved_run_path,
            "resolved_via": resolved_via,
            "display_name": run.name,
            "state": run.state,
            "created_at": run.created_at,
        }
        selected_summary = extract_selected_summary_metrics(run)
        row.update(selected_summary)

        selected_history: Dict[str, object] = {}
        if args.history_fallback:
            selected_history = extract_selected_history_metrics(
                run=run,
                history_samples=args.history_samples,
                history_retries=args.history_retries,
                retry_wait_seconds=args.retry_wait_seconds,
            )
            row.update(selected_history)

        # Optional: macht sichtbar, ob History überhaupt vorhanden war
        row["has_history"] = any(k.startswith("history__") for k in selected_history)
        row["selected_summary_metrics"] = len(selected_summary)
        row["selected_history_metrics"] = len(
            [k for k in selected_history.keys() if k.startswith("history__")]
        )

        # Completion-Time Fallback aus Episodenlänge (falls keine direkte completion_time geloggt wird)
        ep_len = None
        for key in (
            "summary__eval/mean_ep_length",
            "summary__rollout/ep_length_mean",
            "history__eval/mean_ep_length",
            "history__rollout/ep_length_mean",
        ):
            if key in row and row[key] is not None:
                ep_len = row[key]
                break
        if ep_len is not None:
            try:
                row["derived__completion_time_seconds"] = float(ep_len) / float(args.env_hz)
            except Exception:
                row["derived__completion_time_seconds"] = None

        # Quadranten-Erfolgssraten (aus 20 Targets)
        quadrant_to_target_indices = {
            "I": [1, 5, 9, 15, 17],
            "II": [2, 8, 12, 14, 20],
            "III": [4, 6, 10, 16, 18],
            "IV": [3, 7, 11, 13, 19],
        }
        row.update(
            compute_quadrant_success_rates(
                row=row,
                quadrant_to_target_indices=quadrant_to_target_indices,
            )
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    try:
        df.to_csv(args.output, index=False)
        written_to = args.output
    except PermissionError:
        # Windows: oft passiert das, wenn die CSV im Editor offen ist.
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output.lower().endswith(".csv"):
            written_to = f"{args.output[:-4]}__{suffix}.csv"
        else:
            written_to = f"{args.output}__{suffix}"
        df.to_csv(written_to, index=False)
    print(f"CSV geschrieben: {written_to}")
    print(f"Anzahl Runs (Zeilen): {len(df)}")


if __name__ == "__main__":
    main()

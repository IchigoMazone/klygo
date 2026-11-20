"""Aggregate multi-seed runs, threshold audits, mining breakdowns, and rankings."""

from __future__ import annotations

from os import PathLike
from statistics import median
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from klygo import metrics as klygo_metrics, mining, postprocessing as post, validators
from klygo.outputs import Detections

from .loaders import load


def _resolve_run(item: str | PathLike[str] | Mapping[str, Any]) -> dict[str, Any]:
    """Accept either a run directory path or an already-loaded run dictionary."""
    if isinstance(item, Mapping):
        return dict(item)
    return load(item)


def aggregate_runs(
    grouped_runs: Mapping[str, Sequence[str | PathLike[str] | Mapping[str, Any]]],
    *,
    metric_names: Sequence[str] = ("precision", "recall", "f1", "mAP50", "mAP50-95"),
    compare_pair: Optional[tuple[str, str]] = None,
) -> dict[str, Any]:
    """Aggregate multi-seed training runs into summary tables and epoch curves.

    Parameters
    ----------
    grouped_runs : Mapping[str, Sequence[str or PathLike or dict]]
        Mapping of configuration name to a sequence of run directories or loaded runs.
    metric_names : Sequence[str], default=('precision', 'recall', 'f1', 'mAP50', 'mAP50-95')
        Metric keys to summarize from each run's best checkpoint.
    compare_pair : tuple[str, str], optional
        Optional pair ``(first_name, second_name)`` to compute ``first - second`` deltas.

    Returns
    -------
    dict[str, Any]
        Pure-Python dictionary containing ``table``, ``per_seed``, ``groups``, and ``delta``.
    """
    validators.validate_type(grouped_runs, Mapping, "grouped_runs")
    if not grouped_runs:
        raise ValueError("grouped_runs must not be empty")

    table: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}

    for group_name, raw_items in grouped_runs.items():
        if not raw_items:
            raise ValueError(f"grouped_runs['{group_name}'] must not be empty")
        runs = [_resolve_run(item) for item in raw_items]

        summary: dict[str, dict[str, float]] = {}
        row: dict[str, Any] = {"strategy": str(group_name), "seeds": len(runs)}

        for run in runs:
            seed_val = run.get("config", {}).get("seed")
            seed_row: dict[str, Any] = {
                "strategy": str(group_name),
                "seed": seed_val,
                "best_epoch": int(run.get("best_epoch", 0)),
                "time_s": round(float(run.get("total_time_s", 0.0)), 2),
            }
            for key in metric_names:
                val = run.get("best_metrics", {}).get(key)
                if val is not None:
                    seed_row[key] = round(float(val), 4)
            per_seed.append(seed_row)

        for key in metric_names:
            vals = [
                float(run["best_metrics"][key])
                for run in runs
                if key in run.get("best_metrics", {})
            ]
            if vals:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=0))
                summary[key] = {"mean": round(mean_v, 4), "std": round(std_v, 4)}
                row[f"{key}_mean"] = round(mean_v, 4)
                row[f"{key}_std"] = round(std_v, 4)

        times = [float(run.get("total_time_s", 0.0)) for run in runs]
        epochs = [int(run.get("best_epoch", 0)) for run in runs]
        row["time_mean_s"] = round(float(np.mean(times)), 2)
        row["time_std_s"] = round(float(np.std(times, ddof=0)), 2)
        row["best_epoch_mean"] = round(float(np.mean(epochs)), 1)
        table.append(row)

        # Aggregate epoch-level curves safely across potentially uneven run lengths
        all_hist_keys: set[str] = set()
        for run in runs:
            all_hist_keys.update(run.get("history", {}).keys())

        history_mean: dict[str, list[float]] = {}
        history_std: dict[str, list[float]] = {}
        max_len = max((len(run.get("history", {}).get("epoch", [])) for run in runs), default=0)
        if max_len > 0:
            for hkey in sorted(all_hist_keys):
                series_list = [run.get("history", {}).get(hkey, []) for run in runs]
                if not any(series_list) or not all(
                    isinstance(v, (int, float)) for s in series_list for v in s
                ):
                    continue
                mat = np.full((len(runs), max_len), np.nan, dtype=np.float64)
                for r_idx, s in enumerate(series_list):
                    if s:
                        mat[r_idx, :len(s)] = np.asarray(s, dtype=np.float64)
                history_mean[hkey] = [float(v) for v in np.nanmean(mat, axis=0)]
                history_std[hkey] = [float(v) for v in np.nanstd(mat, axis=0)]

        groups[str(group_name)] = {
            "runs": runs,
            "summary": summary,
            "history_mean": history_mean,
            "history_std": history_std,
        }

    delta: dict[str, float] = {}
    if compare_pair is not None:
        first_name, second_name = compare_pair
        if first_name not in groups or second_name not in groups:
            raise ValueError("compare_pair names must exist in grouped_runs")
        s1 = groups[first_name]["summary"]
        s2 = groups[second_name]["summary"]
        for key in metric_names:
            if key in s1 and key in s2:
                delta[key] = round(s1[key]["mean"] - s2[key]["mean"], 4)

    return {
        "table": table,
        "per_seed": per_seed,
        "groups": groups,
        "delta": delta,
    }


def _filter_by_threshold(pred: Detections, tau: float) -> Detections:
    """Return a threshold-filtered copy of Detections using klygo.postprocessing."""
    filtered = pred.copy()
    post.compose([post.threshold(tau)])(filtered)
    return filtered


def threshold_audit(
    pred_calib: Detections,
    target_calib: Detections,
    *,
    thresholds: Optional[Sequence[float]] = None,
    iou: float = 0.5,
    pred_confirm: Optional[Detections] = None,
    target_confirm: Optional[Detections] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Sweep confidence thresholds on calibration data and select optimal ``T*``.

    Reuses ``metrics.threshold_sweep`` and ``metrics.map_score`` to compute
    Precision, Recall, F1, TP/FP/FN counts, mAP@50, and mAP@50-95 across
    thresholds, selects ``T*`` maximizing F1 (tie-breaking by higher Precision),
    and optionally verifies ``T*`` on an independent confirmation set.
    """
    validators.validate_type(pred_calib, Detections, "pred_calib")
    validators.validate_type(target_calib, Detections, "target_calib")
    validators.validate_type(verbose, bool, "verbose")

    grid = (
        [round(0.30 + 0.05 * i, 2) for i in range(14)]
        if thresholds is None
        else [float(t) for t in thresholds]
    )
    if not grid:
        raise ValueError("thresholds must not be empty")

    sweep_rows = klygo_metrics.threshold_sweep(
        pred_calib, target_calib, iou=iou, thresholds=grid, verbose=verbose
    )

    rows: list[dict[str, Any]] = []
    for entry in sweep_rows:
        tau = float(entry["threshold"])
        filtered = _filter_by_threshold(pred_calib, tau)
        map_res = klygo_metrics.map_score(filtered, target_calib, verbose=False)
        rows.append({
            "threshold": round(tau, 4),
            "boxes": int(entry["tp"] + entry["fp"]),
            "tp": int(entry["tp"]),
            "fp": int(entry["fp"]),
            "fn": int(entry["fn"]),
            "precision": round(float(entry["precision"]), 4),
            "recall": round(float(entry["recall"]), 4),
            "f1": round(float(entry["f1"]), 4),
            "mAP50": round(float(map_res.get("map_50") or 0.0), 4),
            "mAP50-95": round(float(map_res.get("map") or 0.0), 4),
        })

    best_row = max(rows, key=lambda r: (r["f1"], r["precision"], -r["threshold"]))
    best_tau = float(best_row["threshold"])

    comparison: dict[str, Any] = {}
    if pred_confirm is not None or target_confirm is not None:
        if pred_confirm is None or target_confirm is None:
            raise ValueError("Both pred_confirm and target_confirm must be provided")
        filtered_confirm = _filter_by_threshold(pred_confirm, best_tau)
        pr_confirm = klygo_metrics.precision_recall(filtered_confirm, target_confirm, iou=iou, verbose=verbose)
        map_confirm = klygo_metrics.map_score(filtered_confirm, target_confirm, verbose=False)
        confirm_row = {
            "threshold": best_tau,
            "precision": round(float(pr_confirm["precision"]), 4),
            "recall": round(float(pr_confirm["recall"]), 4),
            "f1": round(float(pr_confirm["f1"]), 4),
            "mAP50": round(float(map_confirm.get("map_50") or 0.0), 4),
            "mAP50-95": round(float(map_confirm.get("map") or 0.0), 4),
        }
        comparison = {
            "calib": best_row,
            "confirm": confirm_row,
            "delta": {
                k: round(confirm_row[k] - best_row[k], 4)
                for k in ("precision", "recall", "f1", "mAP50", "mAP50-95")
            },
        }

    return {
        "rows": rows,
        "best_threshold": best_tau,
        "best_row": best_row,
        "comparison": comparison,
    }


def mining_breakdown(subsets: Mapping[str, Detections]) -> list[dict[str, Any]]:
    """Summarize positive/background split, box density, and confidence across mining subsets."""
    validators.validate_type(subsets, Mapping, "subsets")
    rows: list[dict[str, Any]] = []
    for name, subset in subsets.items():
        st = mining.stats(subset, quality=False, cooccurrence=False)
        pos = int(st.images - st.empty_images)
        neg = int(st.empty_images)
        scores = [float(s) for frame in subset for s in frame.scores]
        rows.append({
            "strategy": str(name),
            "images": int(st.images),
            "positive_images": pos,
            "positive_pct": round((100.0 * pos / st.images) if st.images else 0.0, 2),
            "background_images": neg,
            "background_pct": round((100.0 * neg / st.images) if st.images else 0.0, 2),
            "boxes": int(st.boxes),
            "boxes_per_positive": round((st.boxes / pos) if pos else 0.0, 2),
            "mean_conf": round(float(np.mean(scores)) if scores else 0.0, 4),
            "median_conf": round(float(median(scores)) if scores else 0.0, 4),
        })
    return rows


def rank_evaluations(
    evaluations: Sequence[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    sort_by: str = "mAP50-95",
    baseline: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Rank evaluation results descending by ``sort_by`` and compute baseline deltas."""
    normalized: list[dict[str, Any]] = []
    if isinstance(evaluations, Mapping):
        for name, item in evaluations.items():
            row = dict(item)
            row.setdefault("strategy", str(name))
            normalized.append(row)
    else:
        for item in evaluations:
            normalized.append(dict(item))

    if not normalized:
        return []

    base_val: Optional[float] = None
    if baseline is not None:
        for row in normalized:
            if row.get("strategy") == baseline or row.get("code") == baseline:
                base_val = float(row.get(sort_by, 0.0))
                break
        if base_val is None:
            raise ValueError(f"baseline '{baseline}' not found in evaluations")

    ranked = sorted(normalized, key=lambda r: float(r.get(sort_by, 0.0)), reverse=True)
    result: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        entry = {"rank": idx, **row}
        if base_val is not None:
            val = float(row.get(sort_by, 0.0))
            delta = val - base_val
            entry["delta"] = round(delta, 4)
            entry["delta_pct"] = round((100.0 * delta / base_val) if base_val else 0.0, 2)
        result.append(entry)
    return result


__all__ = [
    "aggregate_runs",
    "threshold_audit",
    "mining_breakdown",
    "rank_evaluations",
]

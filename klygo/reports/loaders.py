"""Load training runs, dataset partition summaries, and hyperparameters."""

from __future__ import annotations

import math
from os import PathLike
import tempfile
from typing import Any, Mapping, Optional, Sequence

from klygo import archive, config, datasets, files, mining, validators
from klygo.outputs import Detections


_METRIC_ALIASES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
    "precision": "precision",
    "recall": "recall",
    "mAP50": "mAP50",
    "mAP50-95": "mAP50-95",
    "f1": "f1",
}


def _to_number(raw: Any) -> Any:
    """Convert numeric strings to finite int or float while preserving other values."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return None if (isinstance(raw, float) and not math.isfinite(raw)) else raw
    text = str(raw).strip()
    if not text:
        return 0.0
    try:
        if "." not in text and "e" not in text.lower() and text.lower() not in {"nan", "inf", "-inf", "+inf"}:
            return int(text)
        value = float(text)
        return None if not math.isfinite(value) else value
    except ValueError:
        return text


def _row_selection_score(row: Mapping[str, Any], best: str) -> Optional[float]:
    """Compute the scalar selection score for a single epoch row."""
    if best == "fitness":
        m50 = row.get("mAP50")
        m95 = row.get("mAP50-95")
        if isinstance(m50, (int, float)) and isinstance(m95, (int, float)):
            return 0.1 * float(m50) + 0.9 * float(m95)
        return None
    val = row.get(best)
    return float(val) if isinstance(val, (int, float)) else None


def load(
    source: str | PathLike[str],
    *,
    format: str = "yolo",
    best: str = "auto",
    verbose: bool = False,
) -> dict[str, Any]:
    """Load training logs and hyperparameters from a run directory.

    Parameters
    ----------
    source : str or os.PathLike[str]
        Run directory containing ``results.csv`` and optional ``args.yaml``.
    format : {'yolo'}, default='yolo'
        Training log format. Currently only ``'yolo'`` (Ultralytics) is supported.
    best : str, default='auto'
        Criterion to select ``best_epoch``: ``'auto'`` (``mAP50-95`` then ``mAP50``),
        ``'fitness'`` (``0.1 * mAP50 + 0.9 * mAP50-95``), ``'last'``, or a metric name.
    verbose : bool, default=False
        Display progress when loading run files.

    Returns
    -------
    dict[str, Any]
        Pure-Python dictionary with ``config``, ``history``, ``best_epoch``,
        ``total_time_s``, and ``best_metrics``.

    Raises
    ------
    TypeError
        If ``source`` or ``format`` has an invalid type.
    ValueError
        If ``format`` is unsupported or ``results.csv`` is empty.
    FileNotFoundError
        If ``source`` or ``results.csv`` does not exist.
    """
    validators.validate_type(source, (str, PathLike), "source")
    validators.validate_type(format, str, "format")
    validators.validate_type(verbose, bool, "verbose")

    if format.lower() != "yolo":
        raise ValueError("format must be 'yolo'")

    run_dir = files.path(source)
    if not files.exists(run_dir):
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")

    results_path = run_dir / "results.csv"
    if not files.is_file(results_path):
        raise FileNotFoundError(f"results.csv not found in: {run_dir}")

    args_path = run_dir / "args.yaml"
    run_config = (
        dict(config.load(args_path, verbose=verbose).to_dict())
        if files.is_file(args_path)
        else {}
    )

    raw_rows = files.load(results_path, verbose=verbose)
    if not raw_rows:
        raise ValueError(f"results.csv is empty: {results_path}")

    history: dict[str, list[Any]] = {}
    cleaned_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row: dict[str, Any] = {}
        for key, value in raw_row.items():
            clean_key = str(key).strip()
            parsed = _to_number(value)
            row[clean_key] = parsed
            alias = _METRIC_ALIASES.get(clean_key)
            if alias and alias not in row:
                row[alias] = parsed
        p = float(row.get("precision", 0.0) or 0.0)
        r = float(row.get("recall", 0.0) or 0.0)
        if "f1" not in row and ("precision" in row or "recall" in row):
            row["f1"] = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        cleaned_rows.append(row)
        for key, value in row.items():
            history.setdefault(key, []).append(value)

    best_key = best.strip()
    if best_key == "last":
        best_idx = len(cleaned_rows) - 1
    else:
        if best_key == "auto":
            candidate_keys = ("mAP50-95", "mAP50")
            score_key = next((k for k in candidate_keys if k in history), None)
        else:
            score_key = best_key

        if score_key is not None:
            scored_indices = [
                (idx, _row_selection_score(cleaned_rows[idx], score_key))
                for idx in range(len(cleaned_rows))
            ]
            finite_indices = [(idx, val) for idx, val in scored_indices if val is not None]
            best_idx = (
                max(finite_indices, key=lambda item: item[1])[0]
                if finite_indices
                else len(cleaned_rows) - 1
            )
        else:
            best_idx = len(cleaned_rows) - 1

    best_row = cleaned_rows[best_idx]
    best_epoch = int(best_row.get("epoch", best_idx + 1))
    times = history.get("time", [])
    total_time_s = float(times[-1]) if times else 0.0

    best_metrics = {
        key: float(best_row[key])
        for key in ("precision", "recall", "f1", "mAP50", "mAP50-95")
        if key in best_row and isinstance(best_row[key], (int, float))
    }

    return {
        "source": str(run_dir),
        "config": run_config,
        "best_epoch": best_epoch,
        "total_time_s": total_time_s,
        "best_metrics": best_metrics,
        "history": history,
    }


def _count_split_boxes(images_dir: Any, labels_dir: Any) -> tuple[int, int, int]:
    """Count total images, positive images, and boxes inside a YOLO split directory."""
    image_files = files.find(images_dir, "*") if files.is_dir(images_dir) else []
    image_files = [
        p for p in image_files
        if files.is_file(p) and files.extension(p).lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    positive = 0
    total_boxes = 0
    for img_path in image_files:
        stem = files.stem(img_path)
        lbl_path = files.join(labels_dir, f"{stem}.txt")
        if files.is_file(lbl_path):
            text = str(files.load(lbl_path, verbose=False))
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if lines:
                positive += 1
                total_boxes += len(lines)
    return len(image_files), positive, total_boxes


def partition_summary(source: str | PathLike[str] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Summarize image and annotation counts across dataset partitions or splits.

    Accepts a directory/archive produced by ``datasets.partition()`` or
    ``datasets.split()``, or a mapping of ``{split_name: path | Detections}``.

    Parameters
    ----------
    source : str, os.PathLike[str], or Mapping[str, Any]
        Partitioned dataset root, directory of split folders, or explicit mapping.

    Returns
    -------
    list[dict[str, Any]]
        Rows containing ``split``, ``images``, ``ratio_pct``, ``positive_images``,
        ``background_images``, and ``gt_boxes``.
    """
    rows: list[dict[str, Any]] = []

    if isinstance(source, Mapping):
        for split_name, item in source.items():
            if isinstance(item, Detections):
                st = mining.stats(item, quality=False)
                rows.append({
                    "split": str(split_name),
                    "images": int(st.images),
                    "positive_images": int(st.images - st.empty_images),
                    "background_images": int(st.empty_images),
                    "gt_boxes": int(st.boxes),
                })
            else:
                path = files.path(item)
                info = datasets.info(path)
                img_dir = files.join(path, "images")
                lbl_dir = files.join(path, "labels")
                total, pos, boxes = _count_split_boxes(img_dir, lbl_dir)
                if total == 0:
                    total = int(info.get("total_images", 0))
                rows.append({
                    "split": str(split_name),
                    "images": int(total),
                    "positive_images": int(pos),
                    "background_images": int(total - pos),
                    "gt_boxes": int(boxes),
                })
    else:
        validators.validate_type(source, (str, PathLike), "source")
        root = files.path(source)
        if not files.exists(root):
            raise FileNotFoundError(f"source does not exist: {root}")

        if files.is_file(root) and archive.is_archive(root):
            with tempfile.TemporaryDirectory() as tmp_dir:
                archive.extract(root, tmp_dir, overwrite=True, verbose=False)
                extracted = files.path(tmp_dir)
                sub_dirs = [d for d in files.list_entries(extracted) if files.is_dir(d)]
                target_root = (
                    sub_dirs[0]
                    if not files.is_dir(extracted / "images") and len(sub_dirs) == 1
                    else extracted
                )
                return partition_summary(target_root)

        images_root = root / "images"
        if files.is_dir(images_root):
            split_dirs = [d for d in files.list_entries(images_root) if files.is_dir(d)]
            if split_dirs:
                for d in sorted(split_dirs, key=lambda p: files.name(p)):
                    split_name = files.name(d)
                    lbl_dir = root / "labels" / split_name
                    total, pos, boxes = _count_split_boxes(d, lbl_dir)
                    rows.append({
                        "split": split_name,
                        "images": int(total),
                        "positive_images": int(pos),
                        "background_images": int(total - pos),
                        "gt_boxes": int(boxes),
                    })
            else:
                total, pos, boxes = _count_split_boxes(images_root, root / "labels")
                rows.append({
                    "split": "all",
                    "images": int(total),
                    "positive_images": int(pos),
                    "background_images": int(total - pos),
                    "gt_boxes": int(boxes),
                })
        else:
            sub_dirs = sorted(
                [d for d in files.list_entries(root) if files.is_dir(d)],
                key=lambda p: files.name(p),
            )
            for d in sub_dirs:
                total, pos, boxes = _count_split_boxes(d / "images", d / "labels")
                rows.append({
                    "split": files.name(d),
                    "images": int(total),
                    "positive_images": int(pos),
                    "background_images": int(total - pos),
                    "gt_boxes": int(boxes),
                })

    total_images = sum(row["images"] for row in rows)
    for row in rows:
        row["ratio_pct"] = round((100.0 * row["images"] / total_images) if total_images else 0.0, 2)
    return rows


def extract_hyperparams(
    run_dirs: Sequence[str | PathLike[str]],
    *,
    ignore: Sequence[str] = ("seed", "save_dir", "name"),
) -> dict[str, Any]:
    """Extract and cross-check ``args.yaml`` hyperparameters across training runs.

    Parameters
    ----------
    run_dirs : Sequence[str or os.PathLike[str]]
        List of run directories containing ``args.yaml``.
    ignore : Sequence[str], default=('seed', 'save_dir', 'name')
        Keys excluded from difference detection.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``common`` (shared values), ``diffs`` (parameters that
        differ across runs), and ``runs`` (per-run parameter dicts).
    """
    validators.validate_type(run_dirs, (list, tuple), "run_dirs")
    ignored = set(ignore)
    per_run: dict[str, dict[str, Any]] = {}
    all_keys: set[str] = set()

    for run_dir in run_dirs:
        path = files.path(run_dir)
        args_path = path / "args.yaml"
        if not files.is_file(args_path):
            raise FileNotFoundError(f"args.yaml not found in: {path}")
        cfg = dict(config.load(args_path, verbose=False).to_dict())
        base_name = files.name(path)
        run_key = base_name if base_name not in per_run else str(path)
        per_run[run_key] = cfg
        all_keys.update(cfg.keys())

    common: dict[str, Any] = {}
    diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys - ignored):
        values = {name: cfg.get(key) for name, cfg in per_run.items()}
        unique_repr = {repr(v) for v in values.values()}
        if len(unique_repr) <= 1:
            common[key] = next(iter(values.values())) if values else None
        else:
            diffs[key] = values

    return {
        "common": common,
        "diffs": diffs,
        "runs": per_run,
    }


__all__ = ["load", "partition_summary", "extract_hyperparams"]

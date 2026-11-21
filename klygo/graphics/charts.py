"""Publication-ready single-panel charts for experiment reports."""

from __future__ import annotations

from os import PathLike
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from klygo import validators

from ._render import _finalize_figure


def partitions(
    summary: Sequence[Mapping[str, Any]],
    *,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Plot image counts and percentages across dataset partitions or splits."""
    validators.validate_type(summary, (list, tuple), "summary")
    if not summary:
        raise ValueError("summary must not be empty")

    labels = [str(row["split"]) for row in summary]
    counts = [int(row["images"]) for row in summary]
    ratios = [float(row.get("ratio_pct", 0.0)) for row in summary]

    fig, ax = plt.subplots(figsize=figsize or (8.5, 4.5))
    bars = ax.bar(labels, counts, color="#2E6B9E", edgecolor="#1E4A70", alpha=0.9)
    for bar, count, pct in zip(bars, counts, ratios):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts, default=1) * 0.02,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Images")
    ax.set_title("Dataset Partition Distribution" if title is None else title)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, max(counts, default=1) * 1.18)
    fig.tight_layout()
    return _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi, return_figure=return_figure)


def curves(
    grouped_runs: Mapping[str, Any],
    *,
    metric: str = "box_loss",
    split: str = "train",
    band: bool = True,
    per_seed: bool = False,
    mosaic_marker: str | int | None = "auto",
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Plot a single training or validation loss/metric curve across grouped runs."""
    validators.validate_type(grouped_runs, Mapping, "grouped_runs")
    validators.validate_type(metric, str, "metric")
    validators.validate_type(split, str, "split")
    validators.validate_type(band, bool, "band")
    validators.validate_type(per_seed, bool, "per_seed")
    if isinstance(mosaic_marker, bool) or (
        mosaic_marker is not None and not isinstance(mosaic_marker, (int, str))
    ):
        raise TypeError("mosaic_marker must be 'auto', an int epoch, or None")
    groups = grouped_runs.get("groups", grouped_runs)
    if not groups:
        raise ValueError("grouped_runs must contain at least one group")

    def _resolve_metric_key(mapping: Mapping[str, Any]) -> Optional[str]:
        if metric in mapping:
            return metric
        prefixed = f"{split}/{metric}"
        if prefixed in mapping:
            return prefixed
        return None

    display_key = metric if "/" in metric else f"{split}/{metric}"
    for group in groups.values():
        sample_hist = (
            group.get("runs", [{}])[0].get("history", {})
            if per_seed and group.get("runs")
            else group.get("history_mean", {})
        )
        resolved = _resolve_metric_key(sample_hist)
        if resolved is not None:
            display_key = resolved
            break

    fig, ax = plt.subplots(figsize=figsize or (7.2, 4.5))
    mosaic_epoch: Optional[int] = None

    for name, group in groups.items():
        runs = group.get("runs", [])
        if mosaic_marker == "auto" and mosaic_epoch is None and runs:
            cfg = runs[0].get("config", {})
            epochs_val = cfg.get("epochs")
            close_m = cfg.get("close_mosaic")
            if isinstance(epochs_val, int) and isinstance(close_m, int) and close_m > 0:
                mosaic_epoch = max(1, epochs_val - close_m + 1)

        if per_seed:
            line_styles = ["-", "--", ":"]
            for idx, run in enumerate(runs):
                hist = run.get("history", {})
                key = _resolve_metric_key(hist)
                if key is None:
                    continue
                xs = hist.get("epoch", list(range(1, len(hist[key]) + 1)))
                seed_lbl = run.get("config", {}).get("seed", idx)
                ax.plot(xs, hist[key], linestyle=line_styles[idx % len(line_styles)], label=f"{name} (seed {seed_lbl})")
        else:
            hmean = group.get("history_mean", {})
            hstd = group.get("history_std", {})
            key = _resolve_metric_key(hmean)
            if key is None:
                continue
            ys = np.asarray(hmean[key], dtype=np.float64)
            xs = np.asarray(hmean.get("epoch", list(range(1, len(ys) + 1))), dtype=np.float64)
            (line,) = ax.plot(xs, ys, linewidth=1.8, label=name)
            if band and key in hstd:
                sd = np.asarray(hstd[key], dtype=np.float64)
                ax.fill_between(xs, ys - sd, ys + sd, color=line.get_color(), alpha=0.16)

    if isinstance(mosaic_marker, int):
        mosaic_epoch = mosaic_marker
    if mosaic_epoch is not None:
        ax.axvline(mosaic_epoch, color="#555555", linestyle=":", linewidth=1.2, label=f"close_mosaic (epoch {mosaic_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel(display_key)
    ax.set_title(display_key if title is None else title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi, return_figure=return_figure)


def thresholds(
    audit_result: Mapping[str, Any],
    *,
    kind: str = "curves",
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Plot Precision/Recall/F1 curves or TP/FP stacked bars across thresholds."""
    validators.validate_type(audit_result, Mapping, "audit_result")
    validators.validate_type(kind, str, "kind")
    rows = audit_result.get("rows", [])
    if not rows:
        raise ValueError("audit_result['rows'] must not be empty")
    kind_key = kind.lower()
    if kind_key not in {"curves", "counts"}:
        raise ValueError("kind must be 'curves' or 'counts'")

    taus = [float(r["threshold"]) for r in rows]
    best_tau = audit_result.get("best_threshold")
    fig, ax = plt.subplots(figsize=figsize or (7.8, 4.6))

    if kind_key == "curves":
        ax.plot(taus, [r["precision"] for r in rows], marker="o", label="Precision")
        ax.plot(taus, [r["recall"] for r in rows], marker="s", label="Recall")
        ax.plot(taus, [r["f1"] for r in rows], marker="^", label="F1-Score")
        if best_tau is not None:
            ax.axvline(float(best_tau), color="#C0392B", linestyle="--", label=f"T* = {best_tau}")
        ax.set_ylim(0.0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Precision / Recall / F1 vs Confidence Threshold" if title is None else title, pad=12)
    else:
        labels = [f"{t:.2f}" for t in taus]
        tps = [int(r["tp"]) for r in rows]
        fps = [int(r["fp"]) for r in rows]
        ax.bar(labels, tps, color="#27AE60", label="True Positives (TP)")
        ax.bar(labels, fps, bottom=tps, color="#E74C3C", label="False Positives (FP)")
        for idx, (tp, fp) in enumerate(zip(tps, fps)):
            ax.text(idx, tp + fp + max(tps, default=1) * 0.02, str(tp + fp), ha="center", va="bottom", fontsize=8)
        totals = [tp + fp for tp, fp in zip(tps, fps)]
        max_total = max(totals, default=1)
        ax.set_ylim(0, max(max_total * 1.20, max_total + 5))
        ax.set_ylabel("Pseudo-box Count")
        ax.set_title("Pseudo-box TP vs FP by Confidence Threshold" if title is None else title, pad=12)

    ax.set_xlabel("Confidence Threshold (tau)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi, return_figure=return_figure)


def mining(
    mining_rows: Sequence[Mapping[str, Any]],
    *,
    kind: str = "counts",
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Plot positive/background image breakdown or confidence statistics across mining strategies."""
    validators.validate_type(mining_rows, (list, tuple), "mining_rows")
    validators.validate_type(kind, str, "kind")
    if not mining_rows:
        raise ValueError("mining_rows must not be empty")
    kind_key = kind.lower()
    if kind_key not in {"counts", "confidence"}:
        raise ValueError("kind must be 'counts' or 'confidence'")

    names = [str(r["strategy"]) for r in mining_rows]
    fig, ax = plt.subplots(figsize=figsize or (7.5, 4.5))

    if kind_key == "counts":
        pos = [int(r["positive_images"]) for r in mining_rows]
        neg = [int(r["background_images"]) for r in mining_rows]
        boxes = [int(r["boxes"]) for r in mining_rows]
        ax.bar(names, pos, color="#27AE60", label="Positive Images")
        ax.bar(names, neg, bottom=pos, color="#95A5A6", label="Background Images")
        for idx, (p, n, b) in enumerate(zip(pos, neg, boxes)):
            ax.text(idx, p + n + max(p + n, 1) * 0.02, f"{p + n}\n({b} boxes)", ha="center", va="bottom", fontsize=8)
        totals = [p + n for p, n in zip(pos, neg)]
        max_total = max(totals, default=1)
        ax.set_ylim(0, max(max_total * 1.25, max_total + 10))
        ax.set_ylabel("Selected Images")
        ax.set_title("Selected Images (Positive vs Background)" if title is None else title, pad=14)
    else:
        x = np.arange(len(names))
        width = 0.35
        means = [float(r["mean_conf"]) for r in mining_rows]
        medians = [float(r["median_conf"]) for r in mining_rows]
        ax.bar(x - width / 2, means, width, color="#2E86C1", label="Mean Confidence")
        ax.bar(x + width / 2, medians, width, color="#F39C12", label="Median Confidence")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0.0, 1.15)
        ax.set_ylabel("Confidence Score")
        ax.set_title("Confidence Distribution by Mining Strategy" if title is None else title, pad=12)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi, return_figure=return_figure)


def bars(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("mAP50-95",),
    error_bars: bool = True,
    title: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    show: bool = True,
    save_path: Optional[str | PathLike[str]] = None,
    dpi: int = 300,
    return_figure: bool = False,
) -> Optional[Figure]:
    """Plot single-metric bars with optional std error bars or multi-metric grouped bars."""
    validators.validate_type(rows, (list, tuple), "rows")
    validators.validate_type(metrics, (list, tuple), "metrics")
    validators.validate_type(error_bars, bool, "error_bars")
    if not rows:
        raise ValueError("rows must not be empty")
    if not metrics:
        raise ValueError("metrics must not be empty")

    labels = [str(r.get("strategy", r.get("code", f"item_{i}"))) for i, r in enumerate(rows)]
    fig, ax = plt.subplots(figsize=figsize or (8.5, 4.6))
    x = np.arange(len(labels))
    width = 0.72 / len(metrics)

    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        vals = [
            float(r[f"{metric}_mean"] if f"{metric}_mean" in r else r.get(metric, 0.0))
            for r in rows
        ]
        stds = (
            [float(r.get(f"{metric}_std", 0.0)) for r in rows]
            if error_bars and any(f"{metric}_std" in r for r in rows)
            else None
        )
        bars = ax.bar(x + offset, vals, width, yerr=stds, capsize=4 if stds else 0, label=metric)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    max_val = max(
        (float(r[f"{m}_mean"] if f"{m}_mean" in r else r.get(m, 0.0)) for r in rows for m in metrics),
        default=1.0,
    )
    top_y = max(1.15, max_val * 1.15 if max_val > 1.0 else 1.12)
    ax.set_ylim(0.0, top_y)
    ax.set_ylabel("Score")
    ax.set_title("Model Benchmark Comparison" if title is None else title, pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if len(metrics) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _finalize_figure(fig, show=show, save_path=save_path, dpi=dpi, return_figure=return_figure)


__all__ = [
    "partitions",
    "curves",
    "thresholds",
    "mining",
    "bars",
]

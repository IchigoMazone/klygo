"""Standalone benchmark runner for klygo.metrics (mAP accuracy + CPU vs CUDA speed).

Measures:
    1. Kernel throughput & FP16/FP32/FP64 precision drift across CPU and CUDA (CuPy).
    2. Optional real-dataset object-detection accuracy (mAP@50, mAP@50-95, Precision,
       Recall, F1) using `klygo.metrics.evaluate` when `--dataset-dir` is provided.

Usage:
    PYTHONPATH=. python benchmarks/metrics/run_metrics_benchmark.py --large-scale --repeats 7
"""

from __future__ import annotations

import argparse
from typing import Any

from benchmarks.gpu_suite.evaluate_release import (
    _generate_charts,
    benchmark_metrics,
    collect_environment,
)
from klygo import archive, datasets, files, metrics, models, utils


def _evaluate_dataset_map(dataset_dir: str, model_id: str, threshold: float) -> dict[str, Any]:
    truth = datasets.load(dataset_dir, format="yolo", verbose=False)
    detector = models.load(model_id)
    preds = truth.copy()
    preds.repredict(detector, threshold=threshold, verbose=False)
    report = metrics.evaluate(preds, truth, verbose=False)
    pr = report.get("precision_recall", {})
    map_res = report.get("map", {})
    return {
        "dataset": dataset_dir,
        "model": model_id,
        "images": len(truth),
        "threshold": threshold,
        "mAP50": round(float(map_res.get("map50", 0.0)), 4),
        "mAP50_95": round(float(map_res.get("map", 0.0)), 4),
        "precision": round(float(pr.get("precision", 0.0)), 4),
        "recall": round(float(pr.get("recall", 0.0)), 4),
        "f1": round(float(pr.get("f1", 0.0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/benchmarks/metrics_eval")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large-scale", action="store_true")
    parser.add_argument("--dataset-dir", default="", help="Optional YOLO dataset directory to compute mAP50 / mAP50-95")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    env = collect_environment()
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    metrics_rows = benchmark_metrics(env, repeats=args.repeats, large_scale=args.large_scale)
    files.save(out_dir / "metrics_benchmark.json", metrics_rows, overwrite=True, verbose=False)
    utils.tabulate(metrics_rows, title="klygo.metrics (CPU vs CUDA, FP16/FP32/FP64)")

    map_summary: list[dict[str, Any]] = []
    if args.dataset_dir:
        map_row = _evaluate_dataset_map(args.dataset_dir, args.model, args.threshold)
        map_summary.append(map_row)
        files.save(out_dir / "map_evaluation.json", map_summary, overwrite=True, verbose=False)
        utils.tabulate(map_summary, title="Object-Detection mAP Evaluation")

    summary = {"environment": env, "metrics": metrics_rows, "map_evaluation": map_summary}
    charts = _generate_charts(summary, out_dir)

    lines = [
        "# klygo.metrics Benchmark Report",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` | **CuPy CUDA**: `{env.get('cupy_cuda')}`",
        "",
    ]
    if map_summary:
        lines.extend(["## Detection Accuracy (mAP)", ""])
        headers = list(map_summary[0].keys())
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        for row in map_summary:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        lines.append("")
    if metrics_rows:
        lines.extend(["## CPU vs CUDA Kernel Latency & Precision Drift", ""])
        headers = list(metrics_rows[0].keys())
        lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
        for row in metrics_rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
        lines.append("")
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel in charts:
            lines.extend([f"![{title}]({rel})", ""])

    files.save(out_dir / "REPORT.md", "\n".join(lines), overwrite=True, verbose=False)
    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()

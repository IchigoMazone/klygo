"""Reconstruct benchmark report folders and charts from Colab summary JSONs.

Usage:
    python benchmarks/reconstruct_reports.py --json-file <path_to_all_summaries.json>
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
from typing import Any

from klygo import archive, files, utils

ROOT = files.parent(files.parent(files.resolve(__file__)))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from benchmarks.mining.run_mining_benchmark import (
    _generate_mining_charts,
    _render_mining_report,
)
from benchmarks.metrics.run_metrics_benchmark import (
    _generate_metrics_charts,
    _render_metrics_report,
)


def reconstruct_mining(dataset_name: str, summary: dict[str, Any], date_str: str) -> Path:
    out_dir = files.path(f"benchmarks/results/mining/colab_l4_{dataset_name}_{date_str}")
    files.mkdir(out_dir)

    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)
    if "environment" in summary:
        files.save(out_dir / "environment.json", summary["environment"], overwrite=True, verbose=False)
    if "curation_leaderboard" in summary:
        files.save(out_dir / "curation_leaderboard.json", summary["curation_leaderboard"], overwrite=True, verbose=False)
    if "curation_runs" in summary:
        files.save(out_dir / "curation_runs.json", summary["curation_runs"], overwrite=True, verbose=False)
    if "pool_analysis" in summary:
        files.save(out_dir / "pool_analysis.json", summary["pool_analysis"], overwrite=True, verbose=False)

    charts = _generate_mining_charts(summary.get("curation_leaderboard", []), out_dir)
    report_md = _render_mining_report(summary, charts)
    files.save(out_dir / "REPORT.md", report_md, overwrite=True, verbose=False)
    print(f"Reconstructed Mining report at: {out_dir}")
    return out_dir


def reconstruct_metrics(dataset_name: str, summary: dict[str, Any], date_str: str) -> Path:
    out_dir = files.path(f"benchmarks/results/metrics/colab_l4_{dataset_name}_{date_str}")
    files.mkdir(out_dir)

    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)
    if "environment" in summary:
        files.save(out_dir / "environment.json", summary["environment"], overwrite=True, verbose=False)
    if "accuracy_leaderboard" in summary:
        files.save(out_dir / "accuracy_leaderboard.json", summary["accuracy_leaderboard"], overwrite=True, verbose=False)
    if "size_and_error_breakdown" in summary:
        files.save(out_dir / "size_and_error_breakdown.json", summary["size_and_error_breakdown"], overwrite=True, verbose=False)

    charts = _generate_metrics_charts(
        summary.get("accuracy_leaderboard", []),
        summary.get("size_and_error_breakdown", []),
        out_dir,
    )
    report_md = _render_metrics_report(summary, charts)
    files.save(out_dir / "REPORT.md", report_md, overwrite=True, verbose=False)
    print(f"Reconstructed Metrics report at: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", required=True, help="Path to JSON file containing all 6 summaries")
    parser.add_argument("--date", default="", help="Optional date string YYYY_MM_DD")
    args = parser.parse_args()

    date_str = args.date or datetime.date.today().strftime("%Y_%m_%d")

    data = files.load(args.json_file, verbose=False)

    # 1. Mining
    for ds in ["coco128", "voc2007", "val2017"]:
        m_data = data.get(f"MINING_{ds.upper()}") or data.get(f"mining_{ds}")
        if m_data and "error" not in m_data:
            reconstruct_mining(ds, m_data, date_str)

    # 2. Metrics
    for ds in ["coco128", "voc2007", "val2017"]:
        m_data = data.get(f"METRICS_{ds.upper()}") or data.get(f"metrics_{ds}")
        if m_data and "error" not in m_data:
            reconstruct_metrics(ds, m_data, date_str)

    # 3. Compress artifacts
    bench_art = files.path("artifacts/benchmarks")
    files.mkdir(bench_art)

    mining_zip = bench_art / "colab_l4_mining_results.zip"
    metrics_zip = bench_art / "colab_l4_metrics_results.zip"

    archive.compress(files.path("benchmarks/results/mining"), mining_zip, include_root=True, overwrite=True)
    archive.compress(files.path("benchmarks/results/metrics"), metrics_zip, include_root=True, overwrite=True)

    print(f"Created ZIP: {mining_zip} ({utils.human_size(files.size(mining_zip))})")
    print(f"Created ZIP: {metrics_zip} ({utils.human_size(files.size(metrics_zip))})")
    print("Reconstruction complete!")


if __name__ == "__main__":
    main()

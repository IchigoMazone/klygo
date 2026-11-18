"""Standalone 3-tier H.264 video streaming benchmark runner for Klygo.

Evaluates the 4 upstream-stream models (`yolo`, `yolo11`, `yolo26`, `yolo-world`)
across `float32` and `float16` on 3 real Intel IoT DevKit video tiers:
    1. `people_120f_432p` : 120 frames @ 768x432 (repeats=7)
    2. `cars_240f_720p`   : 240 frames @ 1280x720 HD (repeats=5)
    3. `aisle_480f_1080p` : 480 frames @ 1920x1080 Full HD (repeats=3)

Usage:
    PYTHONPATH=. python benchmarks/video/run_video_benchmark.py --tier all --output artifacts/benchmarks/video
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import numpy as np

from benchmarks.gpu_suite.evaluate_release import _md_table, collect_environment
from klygo import archive, files, utils

INTEL_BASE = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/57978890822836f2b4743852f04f62fc511757e4"

VIDEO_TIERS: dict[str, dict[str, Any]] = {
    "people_120f_432p": {
        "filename": "people-detection.mp4",
        "width": 768,
        "height": 432,
        "frames": 120,
        "repeats": 7,
    },
    "cars_240f_720p": {
        "filename": "car-detection.mp4",
        "width": 1280,
        "height": 720,
        "frames": 240,
        "repeats": 5,
    },
    "aisle_480f_1080p": {
        "filename": "store-aisle-detection.mp4",
        "width": 1920,
        "height": 1080,
        "frames": 480,
        "repeats": 3,
    },
}


def _prepare_tier_video(tier_name: str, spec: dict[str, Any]) -> str:
    vid_dir = files.path("artifacts/videos")
    files.mkdir(vid_dir)
    out_mp4 = vid_dir / f"{tier_name}.mp4"
    if files.is_file(out_mp4):
        return str(out_mp4.resolve())

    raw_mp4 = vid_dir / spec["filename"]
    if not files.is_file(raw_mp4):
        files.download(f"{INTEL_BASE}/{spec['filename']}", vid_dir, overwrite=True, verbose=False)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(raw_mp4),
        "-frames:v",
        str(spec["frames"]),
        "-vf",
        f"scale={spec['width']}:{spec['height']}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return str(out_mp4.resolve())


def _generate_video_charts(summary: dict[str, Any], out_dir: Any) -> list[tuple[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    charts_dir = out_dir / "charts"
    files.mkdir(charts_dir)
    created: list[tuple[str, str]] = []
    mode_rows = [r for r in summary.get("video_modes", []) if r.get("status") == "ok" and r.get("fps_median") is not None]
    if not mode_rows:
        return created

    models_list = list(dict.fromkeys(f"{r['model']} ({r['dtype']})" for r in mode_rows))
    stream_modes = ["raw-stream", "native-stream", "managed-stream"]
    x = np.arange(len(models_list))
    width = 0.25

    # 1. Video Stream FPS Comparison
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for idx, m_name in enumerate(stream_modes):
        vals = [
            next((float(r["fps_median"]) for r in mode_rows if f"{r['model']} ({r['dtype']})" == m and r["mode"] == m_name), 0.0)
            for m in models_list
        ]
        ax.bar(x + idx * width, vals, width, label=m_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models_list, rotation=25, ha="right")
    ax.set_ylabel("Median FPS")
    ax.set_title(f"End-to-End Video Stream Throughput ({summary.get('tier', '')})")
    ax.legend()
    fig.tight_layout()
    p1 = charts_dir / "video_fps_comparison.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    created.append(("Video Stream FPS by Mode", "charts/video_fps_comparison.png"))

    # 2. Time-To-First-Frame (TTFF ms): non-stream vs stream
    ttff_modes = ["raw-non-stream", "raw-stream", "native-stream", "managed-stream"]
    width2 = 0.2
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for idx, m_name in enumerate(ttff_modes):
        vals = [
            next((float(r["ttff_ms"]) for r in mode_rows if f"{r['model']} ({r['dtype']})" == m and r["mode"] == m_name and r.get("ttff_ms") is not None), 0.0)
            for m in models_list
        ]
        ax.bar(x + idx * width2, vals, width2, label=m_name)
    ax.set_xticks(x + 1.5 * width2)
    ax.set_xticklabels(models_list, rotation=25, ha="right")
    ax.set_ylabel("Median TTFF (ms)")
    ax.set_title(f"Time-To-First-Frame (TTFF) ({summary.get('tier', '')})")
    ax.legend()
    fig.tight_layout()
    p2 = charts_dir / "video_ttff_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    created.append(("Video Time-To-First-Frame (TTFF ms)", "charts/video_ttff_comparison.png"))

    return created


def _render_video_report(summary: dict[str, Any], charts: list[tuple[str, str]]) -> str:
    env = summary["environment"]
    git_info = env.get("git", {})
    spec = summary.get("video_spec", {})
    lines = [
        f"# Klygo Video Streaming Benchmark Report ({summary.get('tier', '')})",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` ({env.get('vram_mb', 0.0)} MB VRAM)",
        f"- **Video Tier**: `{summary.get('tier')}` (`{spec.get('width')}x{spec.get('height')}`, `{spec.get('frames')}` H.264 frames, `repeats={spec.get('repeats')}`)",
        f"- **Git**: `{git_info.get('branch', 'unknown')}` @ `{git_info.get('commit', 'unknown')}`",
        "",
    ]
    lines.extend(_md_table(summary.get("model_loads", []), "models.load Cold-Load Latency"))
    lines.extend(_md_table(summary.get("models", []), "6-Mode Video Run Status"))
    lines.extend(_md_table(summary.get("video_modes", []), "6-Mode End-to-End Video Streaming Details"))
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel in charts:
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    return "\n".join(lines).rstrip() + "\n"


def run_tier(tier_name: str, base_out: Any, models_csv: str, dtypes_csv: str, device: str) -> Any:
    spec = VIDEO_TIERS[tier_name]
    out_dir = base_out / tier_name
    files.mkdir(out_dir)

    video_path = _prepare_tier_video(tier_name, spec)
    env = collect_environment()
    env["video_tier"] = tier_name
    env["video_sha256"] = files.hash(video_path)
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    model_runs: list[dict[str, Any]] = []
    model_loads: list[dict[str, Any]] = []
    video_modes: list[dict[str, Any]] = []

    for model_id in [m.strip() for m in models_csv.split(",") if m.strip()]:
        for dtype_name in [d.strip() for d in dtypes_csv.split(",") if d.strip()]:
            case_out = out_dir / "inference" / f"{model_id}_{dtype_name}"
            cmd = [
                sys.executable,
                "benchmarks/gpu_suite/run.py",
                "--model", model_id,
                "--profile", "video",
                "--source-video", video_path,
                "--video-frames", str(spec["frames"]),
                "--device", device,
                "--dtype", dtype_name,
                "--repeats", str(spec["repeats"]),
                "--output", str(case_out),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            err_tail = ""
            if proc.returncode != 0:
                lines = [ln.strip() for ln in (proc.stderr or proc.stdout or "").splitlines() if ln.strip()]
                err_tail = lines[-1] if lines else "failed"
            model_runs.append({
                "model": model_id,
                "dtype": dtype_name,
                "returncode": proc.returncode,
                "status": "passed" if proc.returncode == 0 else "failed",
                "output_dir": str(case_out),
                "error": err_tail,
            })
            json_path = case_out / f"{model_id}.json"
            if proc.returncode == 0 and files.exists(json_path):
                data = files.load(json_path, verbose=False)
                model_loads.append({
                    "model": model_id,
                    "dtype": dtype_name,
                    "actual_dtype": data.get("actual_model_dtype", ""),
                    "load_seconds": round(float(data.get("load_seconds", 0.0)), 4) if data.get("load_seconds") is not None else None,
                })
                for case in data.get("cases", []):
                    fps_v = case.get("fps_median")
                    ttff_v = case.get("first_frame_host_ms_median")
                    p50_v = case.get("ready_interval_p50_ms")
                    p95_v = case.get("ready_interval_p95_ms")
                    p99_v = case.get("ready_interval_p99_ms")
                    video_modes.append({
                        "model": model_id,
                        "dtype": dtype_name,
                        "actual_dtype": data.get("actual_model_dtype", ""),
                        "mode": case.get("mode", ""),
                        "status": case.get("status", ""),
                        "fps_median": round(float(fps_v), 2) if fps_v is not None else None,
                        "ttff_ms": round(float(ttff_v), 2) if ttff_v is not None else None,
                        "p50_ms": round(float(p50_v), 2) if p50_v is not None else None,
                        "p95_ms": round(float(p95_v), 2) if p95_v is not None else None,
                        "p99_ms": round(float(p99_v), 2) if p99_v is not None else None,
                        "parity_pass": case.get("all_parity_pass"),
                    })

    summary = {
        "tier": tier_name,
        "video_spec": spec,
        "environment": env,
        "models": model_runs,
        "model_loads": model_loads,
        "video_modes": video_modes,
    }
    files.save(out_dir / "model_runs.json", model_runs, overwrite=True, verbose=False)
    files.save(out_dir / "model_loads.json", model_loads, overwrite=True, verbose=False)
    files.save(out_dir / "video_modes.json", video_modes, overwrite=True, verbose=False)
    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)
    charts = _generate_video_charts(summary, out_dir)
    files.save(out_dir / "REPORT.md", _render_video_report(summary, charts), overwrite=True, verbose=False)
    utils.tabulate(video_modes, title=f"Video Streaming ({tier_name})")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=[*VIDEO_TIERS.keys(), "all"], default="all")
    parser.add_argument("--output", default="artifacts/benchmarks/video")
    parser.add_argument("--models", default="yolo,yolo11,yolo26,yolo-world")
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    base_out = files.path(args.output)
    files.mkdir(base_out)
    tiers = list(VIDEO_TIERS.keys()) if args.tier == "all" else [args.tier]
    for t in tiers:
        run_tier(t, base_out, args.models, args.dtypes, args.device)

    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(base_out, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()

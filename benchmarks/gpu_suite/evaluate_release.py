"""Comprehensive correctness & benchmark runner for Klygo detection-evaluation.

Reuses ``benchmarks/gpu_suite/run.py`` for 6-mode model inference (fp16/fp32)
and benchmarks ``klygo.metrics``, ``DetectionGroup``, ``postprocessing``,
``datasets.load``, ``media``, ``visual.compare_images``, and ``utils.tabulate``
on CPU and CUDA with ``utils.tabulate`` console summaries.

Usage:
    PYTHONPATH=. python benchmarks/gpu_suite/evaluate_release.py --output artifacts/benchmarks/local_eval
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from importlib import metadata as importlib_metadata
import io
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from klygo import (
    datasets,
    files,
    media,
    metrics,
    outputs,
    postprocessing,
    utils,
    visual,
)


def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _git_info() -> dict[str, str]:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": "yes" if _run(["git", "status", "--short"]) else "no",
    }


def collect_environment() -> dict[str, Any]:
    """Capture hardware, OS, Python, library versions, and CUDA/CuPy capabilities."""
    gpu_name = "none"
    vram_mb = 0.0
    cuda_runtime = "none"
    torch_cuda = False
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        cuda_runtime = str(torch.version.cuda or "none")
        if torch_cuda:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            vram_mb = round(props.total_memory / (1024 * 1024), 1)
    except Exception:
        pass

    cupy_cuda = False
    try:
        import cupy as cp

        cupy_cuda = bool(cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        pass

    import klygo

    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "klygo_path": str(files.path(klygo.__file__).parent),
        "git": _git_info(),
        "gpu": gpu_name,
        "vram_mb": vram_mb,
        "is_nvidia_l4": "L4" in gpu_name.upper(),
        "torch_cuda": torch_cuda,
        "cupy_cuda": cupy_cuda,
        "cuda_runtime": cuda_runtime,
        "versions": {
            "torch": _pkg_version("torch"),
            "torchvision": _pkg_version("torchvision"),
            "ultralytics": _pkg_version("ultralytics"),
            "transformers": _pkg_version("transformers"),
            "keras": _pkg_version("keras"),
            "keras_hub": _pkg_version("keras-hub"),
            "numpy": _pkg_version("numpy"),
            "cupy": _pkg_version("cupy-cuda12x") if _pkg_version("cupy-cuda12x") != "not-installed" else _pkg_version("cupy"),
            "pillow": _pkg_version("pillow"),
            "opencv": _pkg_version("opencv-python"),
        },
    }


def _sync_all(use_cuda: bool) -> None:
    if not use_cuda:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    try:
        import cupy as cp

        if cp.cuda.is_available():
            cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _measure(fn: Callable[[], Any], *, repeats: int, warmup: int, use_cuda: bool) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    _sync_all(use_cuda)
    samples_ms: list[float] = []
    last_result = None
    for _ in range(repeats):
        _sync_all(use_cuda)
        start = time.perf_counter()
        last_result = fn()
        _sync_all(use_cuda)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples_ms": [round(x, 4) for x in samples_ms],
        "median_ms": round(statistics.median(samples_ms), 4),
        "mean_ms": round(statistics.mean(samples_ms), 4),
        "std_ms": round(statistics.stdev(samples_ms), 4) if len(samples_ms) > 1 else 0.0,
        "last_result": last_result,
    }


def _synthetic_pair(
    num_frames: int,
    pred_per_frame: int,
    gt_per_frame: int,
    num_classes: int = 5,
    seed: int = 42,
) -> tuple[outputs.Detections, outputs.Detections]:
    rng = np.random.default_rng(seed)
    labels = [f"cls_{i}" for i in range(num_classes)]
    gt_frames: list[outputs.Detection] = []
    pred_frames: list[outputs.Detection] = []

    for idx in range(num_frames):
        gt_boxes: list[outputs.Box] = []
        for b in range(gt_per_frame):
            x1 = float(rng.uniform(0, 500))
            y1 = float(rng.uniform(0, 500))
            w = float(rng.uniform(16, 120))
            h = float(rng.uniform(16, 120))
            gt_boxes.append(outputs.Box([x1, y1, x1 + w, y1 + h], label=labels[b % num_classes], score=1.0))

        pred_boxes: list[outputs.Box] = []
        for b in range(pred_per_frame):
            if gt_boxes and b < gt_per_frame and rng.random() < 0.7:
                base = gt_boxes[b].box
                jitter = rng.uniform(-4.0, 4.0, size=4)
                coords = [
                    max(0.0, base[0] + float(jitter[0])),
                    max(0.0, base[1] + float(jitter[1])),
                    max(1.0, base[2] + float(jitter[2])),
                    max(1.0, base[3] + float(jitter[3])),
                ]
                lbl = gt_boxes[b].label if rng.random() < 0.85 else labels[(b + 1) % num_classes]
            else:
                x1 = float(rng.uniform(0, 500))
                y1 = float(rng.uniform(0, 500))
                w = float(rng.uniform(16, 120))
                h = float(rng.uniform(16, 120))
                coords = [x1, y1, x1 + w, y1 + h]
                lbl = labels[b % num_classes]
            pred_boxes.append(outputs.Box(coords, label=lbl, score=float(rng.uniform(0.15, 0.99))))

        gt_frames.append(outputs.Detection(gt_boxes, source_id=f"img_{idx}.jpg", frame_index=0, image_size=(640, 640)))
        pred_frames.append(outputs.Detection(pred_boxes, source_id=f"img_{idx}.jpg", frame_index=0, image_size=(640, 640)))

    return outputs.Detections(pred_frames), outputs.Detections(gt_frames)


def benchmark_metrics(env: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    """Benchmark metrics across CPU/CUDA, precision (16/32/64), packed/unpacked, and box counts."""
    rows: list[dict[str, Any]] = []
    workloads = [
        ("empty_0_0", 2, 0, 0),
        ("small_100_25", 4, 100, 25),
        ("medium_500_150", 2, 500, 150),
        ("large_1500_500", 1, 1500, 500),
    ]
    devices = ["cpu"] + (["cuda"] if env["cupy_cuda"] else [])
    evaluator = metrics.compose(
        metrics.precision_recall(iou=0.5),
        metrics.map_score(thresholds=(0.5,)),
        metrics.confusion_matrix(iou=0.5),
        metrics.match_boxes(iou=0.5),
        metrics.per_class(iou=0.5),
        metrics.per_image(iou=0.5),
        metrics.error_analysis(iou=0.5),
        metrics.size_breakdown(iou=0.5),
        metrics.calibration(iou=0.5),
        metrics.threshold_sweep(thresholds=(0.3, 0.5, 0.7), iou=0.5),
        metrics.iou_sweep(thresholds=(0.5, 0.75)),
    )

    for case_name, frames, p_cnt, g_cnt in workloads:
        base_pred, base_gt = _synthetic_pair(frames, p_cnt, g_cnt)
        ref_fp64 = metrics.evaluate(base_pred, base_gt)
        for precision in (16, 32, 64):
            for packed in (False, True):
                if precision == 16 and packed:
                    continue  # keep matrix focused while covering both axes
                for dev in devices:
                    pred = base_pred.copy()
                    gt = base_gt.copy()
                    if packed:
                        pred.pack()
                        gt.pack()
                    pred.astype(precision)
                    gt.astype(precision)
                    if dev == "cuda":
                        pred.cuda(0)
                        gt.cuda(0)

                    # First-call cost (includes any CuPy JIT compilation)
                    t0 = time.perf_counter()
                    first_eval = metrics.evaluate(pred, gt)
                    _sync_all(dev == "cuda")
                    first_ms = round((time.perf_counter() - t0) * 1000.0, 4)

                    eval_stats = _measure(lambda: metrics.evaluate(pred, gt), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
                    comp_stats = _measure(lambda: metrics.evaluate(pred, gt, evaluator), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
                    f1_val = float(first_eval["precision_recall"]["f1"])
                    ref_f1_val = float(ref_fp64["precision_recall"]["f1"])
                    f1_drift = abs(f1_val - ref_f1_val)

                    rows.append({
                        "case": case_name,
                        "device": dev,
                        "precision": f"fp{precision}",
                        "packed": packed,
                        "first_ms": first_ms,
                        "evaluate_median_ms": eval_stats["median_ms"],
                        "compose_median_ms": comp_stats["median_ms"],
                        "f1": round(f1_val, 5),
                        "f1_drift_vs_fp64": round(f1_drift, 6),
                        "status": "passed",
                    })
    return rows


def benchmark_groups_and_postprocessing(env: dict[str, Any], repeats: int) -> list[dict[str, Any]]:
    """Benchmark DetectionGroup, combine, postprocessing, and CPU/CUDA transfers."""
    rows: list[dict[str, Any]] = []
    pred, gt = _synthetic_pair(8, 120, 40)
    devices = ["cpu"] + (["cuda"] if env["cupy_cuda"] else [])

    for dev in devices:
        p_work, g_work = pred.copy(), gt.copy()
        if dev == "cuda":
            p_work.cuda(0)
            g_work.cuda(0)

        grp = outputs.group({"pred": p_work, "truth": g_work})
        keep_stats = _measure(lambda: outputs.group({"pred": p_work.copy(), "truth": g_work.copy()}).keep(["cls_0", "cls_1"]), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        nms_stats = _measure(lambda: p_work.copy().transform(postprocessing.threshold(0.3), postprocessing.nms(0.5), postprocessing.top(50)), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        comb_stats = _measure(lambda: p_work.copy().combine(g_work, "union", iou=0.5), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        _ = grp["pred"]

        rows.append({
            "device": dev,
            "group_keep_ms": keep_stats["median_ms"],
            "post_pipeline_ms": nms_stats["median_ms"],
            "combine_ms": comb_stats["median_ms"],
            "status": "passed",
        })

    if env["cupy_cuda"]:
        up = _measure(lambda: pred.copy().cuda(0), repeats=repeats, warmup=1, use_cuda=True)
        down = _measure(lambda: pred.copy().cuda(0).cpu(), repeats=repeats, warmup=1, use_cuda=True)
        rows.append({"device": "transfer", "group_keep_ms": up["median_ms"], "post_pipeline_ms": down["median_ms"], "combine_ms": 0.0, "status": "passed"})

    return rows


def benchmark_datasets_media_visual(repeats: int) -> list[dict[str, Any]]:
    """Benchmark datasets.load (single vs multi-set), media.info/save_images, visual.compare_images, utils.tabulate."""
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = files.path(tmp)
        ds_dir = root / "ds"
        (ds_dir / "images").mkdir(parents=True)
        (ds_dir / "labels").mkdir(parents=True)
        (ds_dir / "pseudo_labels").mkdir(parents=True)
        files.save(ds_dir / "data.yaml", {"names": {0: "person", 1: "car"}}, overwrite=True, verbose=False)
        files.save(ds_dir / "pseudo.yaml", {"names": {0: "person", 1: "car"}}, overwrite=True, verbose=False)

        for i in range(6):
            img_path = ds_dir / "images" / f"img_{i:02d}.jpg"
            Image.new("RGB", (320, 240), (i * 30, 80, 160)).save(img_path)
            files.save(ds_dir / "labels" / f"img_{i:02d}.txt", ["0 0.5 0.5 0.3 0.4"], overwrite=True, verbose=False)
            files.save(ds_dir / "pseudo_labels" / f"img_{i:02d}.txt", ["0 0.52 0.5 0.3 0.4", "1 0.2 0.2 0.1 0.1"], overwrite=True, verbose=False)

        single_stats = _measure(
            lambda: (datasets.load(ds_dir, labels_dir="labels", yaml_name="data.yaml", verbose=False),
                     datasets.load(ds_dir, labels_dir="pseudo_labels", yaml_name="pseudo.yaml", verbose=False)),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        group_stats = _measure(
            lambda: datasets.load(ds_dir, labels_dir={"truth": "labels", "pred": "pseudo_labels"}, yaml_name={"truth": "data.yaml", "pred": "pseudo.yaml"}, verbose=False),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        loaded_grp = group_stats["last_result"]
        assert isinstance(loaded_grp, outputs.DetectionGroup)
        rows.append({"operation": "datasets.load_2x_single", "median_ms": single_stats["median_ms"], "status": "passed"})
        rows.append({"operation": "datasets.load_group", "median_ms": group_stats["median_ms"], "status": "passed"})

        info_stats = _measure(lambda: media.info(ds_dir / "images" / "img_00.jpg"), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": "media.info", "median_ms": info_stats["median_ms"], "status": "passed"})

        frames = media.load(ds_dir / "images", verbose=False)
        save_out = root / "saved_imgs"
        save_stats = _measure(lambda: media.save_images(save_out, frames, size=(256, 256), overwrite=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        assert save_stats["last_result"] is None
        rows.append({"operation": "media.save_images_letterbox", "median_ms": save_stats["median_ms"], "status": "passed"})

        cmp_out = root / "compare.png"
        cmp_stats = _measure(
            lambda: visual.compare_images(loaded_grp["truth"][0], loaded_grp["pred"][0], mode="save", output_path=cmp_out, overwrite=True),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": "visual.compare_images", "median_ms": cmp_stats["median_ms"], "status": "passed"})

        # Silence stdout during tabulate timing
        def _silent_tab():
            buf = io.StringIO()
            with redirect_stdout(buf):
                utils.tabulate(rows)
        tab_stats = _measure(_silent_tab, repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": "utils.tabulate", "median_ms": tab_stats["median_ms"], "status": "passed"})

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Klygo detection-evaluation benchmark suite.")
    parser.add_argument("--output", default="artifacts/benchmarks/release_eval")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-models", action="store_true", help="Also run 6-mode gpu_suite/run.py models")
    parser.add_argument(
        "--models",
        default="yolo,yolo11,yolo26,yolo-world,grounding-dino,grounding-dino-base,faster-rcnn,d-fine,locate-anything",
    )
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-remote-code", action="store_true")
    parser.add_argument("--include-scaling", action="store_true")
    parser.add_argument("--scaling-models", default="yolo,yolo-world")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    env = collect_environment()
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)
    print("=== Environment ===")
    utils.tabulate({k: v for k, v in env.items() if k not in ("git", "versions")}, title="Runtime")
    utils.tabulate(env["versions"], title="Versions")

    metrics_rows = benchmark_metrics(env, repeats=args.repeats)
    files.save(out_dir / "metrics_benchmark.json", metrics_rows, overwrite=True, verbose=False)
    print("\n=== klygo.metrics Benchmark ===")
    utils.tabulate(metrics_rows, title="Metrics (CPU vs CUDA, FP16/FP32/FP64)")

    group_rows = benchmark_groups_and_postprocessing(env, repeats=args.repeats)
    files.save(out_dir / "outputs_post_benchmark.json", group_rows, overwrite=True, verbose=False)
    print("\n=== DetectionGroup / Postprocessing / Transfer ===")
    utils.tabulate(group_rows, title="Outputs & Postprocessing")

    dmv_rows = benchmark_datasets_media_visual(repeats=args.repeats)
    files.save(out_dir / "datasets_media_visual_benchmark.json", dmv_rows, overwrite=True, verbose=False)
    print("\n=== Datasets / Media / Visual / Tabulate ===")
    utils.tabulate(dmv_rows, title="Datasets, Media, Visual & Tabulate")

    model_runs: list[dict[str, Any]] = []
    model_modes: list[dict[str, Any]] = []
    if args.include_models:
        for model_id in [m.strip() for m in args.models.split(",") if m.strip()]:
            for dtype_name in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
                case_out = out_dir / "inference" / f"{model_id}_{dtype_name}"
                cmd = [
                    sys.executable,
                    "benchmarks/gpu_suite/run.py",
                    "--model", model_id,
                    "--device", args.device,
                    "--dtype", dtype_name,
                    "--repeats", str(args.repeats),
                    "--output", str(case_out),
                ]
                if args.allow_remote_code:
                    cmd.append("--allow-remote-code")
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
                    for case in data.get("cases", []):
                        fps_val = case.get("fps_median")
                        p50_val = case.get("ready_interval_p50_ms")
                        model_modes.append({
                            "model": model_id,
                            "dtype": dtype_name,
                            "actual_dtype": data.get("actual_model_dtype", ""),
                            "mode": case.get("mode", ""),
                            "status": case.get("status", ""),
                            "fps_median": round(float(fps_val), 2) if fps_val is not None else None,
                            "p50_ms": round(float(p50_val), 2) if p50_val is not None else None,
                            "parity_pass": case.get("all_parity_pass"),
                        })
        files.save(out_dir / "model_runs.json", model_runs, overwrite=True, verbose=False)
        files.save(out_dir / "model_modes.json", model_modes, overwrite=True, verbose=False)
        print("\n=== 6-Mode Model Inference Runs ===")
        utils.tabulate(model_runs, title="6-Mode Inference Summary")
        if model_modes:
            print("\n=== 6-Mode Per-Mode Details ===")
            utils.tabulate(model_modes, title="6-Mode Per-Mode Details")

    scaling_rows: list[dict[str, Any]] = []
    if args.include_scaling:
        configs = [
            (1, 1, 1, True),
            (4, 2, 2, False),
            (4, 2, 2, True),
            (8, 4, 4, False),
            (8, 4, 4, True),
        ]
        for model_id in [m.strip() for m in args.scaling_models.split(",") if m.strip()]:
            for dtype_name in [d.strip() for d in args.dtypes.split(",") if d.strip()]:
                for batch_sz, workers_n, prefetch_n, use_packed in configs:
                    tag = f"{model_id}_{dtype_name}_b{batch_sz}_w{workers_n}_p{prefetch_n}_{'packed' if use_packed else 'unpacked'}"
                    case_out = out_dir / "scaling" / tag
                    cmd = [
                        sys.executable,
                        "benchmarks/gpu_suite/run.py",
                        "--model", model_id,
                        "--device", args.device,
                        "--dtype", dtype_name,
                        "--batch", str(batch_sz),
                        "--workers", str(workers_n),
                        "--prefetch", str(prefetch_n),
                        "--repeats", str(args.repeats),
                        "--output", str(case_out),
                    ]
                    if use_packed:
                        cmd.append("--packed")
                    proc = subprocess.run(cmd, capture_output=True, text=True)
                    json_path = case_out / f"{model_id}.json"
                    if proc.returncode == 0 and files.exists(json_path):
                        data = files.load(json_path, verbose=False)
                        for case in data.get("cases", []):
                            scaling_rows.append({
                                "model": model_id,
                                "dtype": dtype_name,
                                "batch": batch_sz,
                                "workers": workers_n,
                                "prefetch": prefetch_n,
                                "packed": use_packed,
                                "mode": case["mode"],
                                "fps_median": round(case["fps_median"], 2),
                                "p50_ms": round(case["ready_interval_p50_ms"], 2),
                                "parity": case["all_parity_pass"],
                            })
                    else:
                        scaling_rows.append({
                            "model": model_id,
                            "dtype": dtype_name,
                            "batch": batch_sz,
                            "workers": workers_n,
                            "prefetch": prefetch_n,
                            "packed": use_packed,
                            "mode": "error",
                            "fps_median": 0.0,
                            "p50_ms": 0.0,
                            "parity": False,
                        })
        files.save(out_dir / "scaling_runs.json", scaling_rows, overwrite=True, verbose=False)
        print("\n=== Scaling (Packed / Batch / Workers / Prefetch) ===")
        utils.tabulate(scaling_rows, title="Batch/Worker/Prefetch/Packed Scaling")

    summary = {
        "environment": env,
        "metrics": metrics_rows,
        "group_post": group_rows,
        "io_visual": dmv_rows,
        "models": model_runs,
        "model_modes": model_modes,
        "scaling": scaling_rows,
    }
    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)


if __name__ == "__main__":
    main()

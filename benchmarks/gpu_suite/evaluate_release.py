"""Comprehensive correctness & benchmark runner for Klygo detection-evaluation.

Reuses ``benchmarks/gpu_suite/run.py`` for 6-mode model inference (fp16/fp32)
and benchmarks ``klygo.metrics``, ``DetectionGroup``, ``postprocessing``,
``datasets.load``, ``media``, ``graphics.compare``, and ``utils.tabulate``
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

from benchmarks.gpu_suite.locate_runtime import LocateRuntime, require_results, SUPPORTED

from klygo import (
    archive,
    datasets,
    files,
    media,
    metrics,
    mining,
    outputs,
    postprocessing,
    preprocessing,
    utils,
    graphics,
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
    gpu_uuid = None
    vram_mb = 0.0
    cuda_runtime = "none"
    torch_cuda = False
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        cuda_runtime = str(torch.version.cuda or "none")
        if torch_cuda:
            props = torch.cuda.get_device_properties(0)
            gpu_uuid = str(getattr(props, "uuid", "")) or None
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
        "gpu_uuid": gpu_uuid,
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


def benchmark_metrics(env: dict[str, Any], repeats: int, large_scale: bool = False) -> list[dict[str, Any]]:
    """Benchmark metrics across CPU/CUDA, precision (16/32/64), packed/unpacked, and box counts."""
    rows: list[dict[str, Any]] = []
    workloads = [
        ("empty_0_0", 2, 0, 0),
        ("small_100_25", 4, 100, 25),
        ("medium_500_150", 2, 500, 150),
        ("large_1500_500", 1, 1500, 500),
    ]
    if large_scale:
        workloads.extend([
            ("xlarge_5000_2000", 1, 5000, 2000),
            ("batch_100f_10k", 100, 100, 30),
        ])
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


def benchmark_groups_and_postprocessing(
    env: dict[str, Any],
    repeats: int,
    large_scale: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark DetectionGroup, pack/astype, combine, postprocessing, and CPU/CUDA transfers."""
    rows: list[dict[str, Any]] = []
    num_frames = 100 if large_scale else 8
    pred, gt = _synthetic_pair(num_frames, 120, 40)
    devices = ["cpu"] + (["cuda"] if env["cupy_cuda"] else [])

    for dev in devices:
        p_work, g_work = pred.copy(), gt.copy()
        if dev == "cuda":
            p_work.cuda(0)
            g_work.cuda(0)

        grp = outputs.DetectionGroup({"pred": p_work, "truth": g_work})
        keep_stats = _measure(lambda: outputs.DetectionGroup({"pred": p_work.copy(), "truth": g_work.copy()}).keep(["cls_0", "cls_1"]), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        nms_stats = _measure(
            lambda: p_work.copy().transform(
                postprocessing.clip((640, 640)),
                postprocessing.threshold(0.3),
                postprocessing.nms(0.5),
                postprocessing.top(50),
            ),
            repeats=repeats, warmup=1, use_cuda=(dev == "cuda"),
        )
        comb_stats = _measure(lambda: p_work.copy().combine(g_work, "union", iou=0.5), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        pack_stats = _measure(lambda: p_work.copy().pack(), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        cast_stats = _measure(lambda: p_work.copy().astype(16), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        _ = grp["pred"]

        rows.append({
            "device": dev,
            "frames": num_frames,
            "group_keep_ms": keep_stats["median_ms"],
            "post_pipeline_ms": nms_stats["median_ms"],
            "combine_ms": comb_stats["median_ms"],
            "pack_ms": pack_stats["median_ms"],
            "astype_fp16_ms": cast_stats["median_ms"],
            "status": "passed",
        })

    if env["cupy_cuda"]:
        up = _measure(lambda: pred.copy().cuda(0), repeats=repeats, warmup=1, use_cuda=True)
        down = _measure(lambda: pred.copy().cuda(0).cpu(), repeats=repeats, warmup=1, use_cuda=True)
        rows.append({
            "device": "transfer",
            "frames": num_frames,
            "group_keep_ms": up["median_ms"],
            "post_pipeline_ms": down["median_ms"],
            "combine_ms": 0.0,
            "pack_ms": 0.0,
            "astype_fp16_ms": 0.0,
            "status": "passed",
        })

    return rows


def benchmark_mining(
    env: dict[str, Any],
    repeats: int,
    large_scale: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark klygo.mining on CPU vs CUDA Detections and disk-backed dataset loading."""
    rows: list[dict[str, Any]] = []
    num_frames = 200 if large_scale else 24
    pred, _ = _synthetic_pair(num_frames, 30, 10, num_classes=5)
    devices = ["cpu"] + (["cuda"] if env["cupy_cuda"] else [])
    limit_n = max(4, num_frames // 2)
    # These annotation-only fixtures have no image pixels for quality scoring.
    weights = {"rarity": 0.5, "coverage": 0.3, "boxes": 0.2}

    for dev in devices:
        work = pred.copy()
        if dev == "cuda":
            work.cuda(0)
        stats_res = _measure(lambda: mining.stats(work, quality=False), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        score_res = _measure(lambda: mining.score(work, weights=weights), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        select_res = _measure(
            lambda: mining.select(work, method="weighted", limit=limit_n,
                                  scores=mining.score(work, weights=weights)),
            repeats=repeats, warmup=1, use_cuda=(dev == "cuda"),
        )
        selected = select_res["last_result"]
        balance_res = _measure(lambda: mining.balance(work, target="median"), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        compare_res = _measure(lambda: mining.compare(work, selected, quality=False), repeats=repeats, warmup=1, use_cuda=(dev == "cuda"))
        rows.append({
            "scope": f"detections_{num_frames}f",
            "score_weights": dict(weights),
            "select_includes_scoring": True,
            "device": dev,
            "stats_ms": stats_res["median_ms"],
            "score_ms": score_res["median_ms"],
            "select_ms": select_res["median_ms"],
            "balance_ms": balance_res["median_ms"],
            "compare_ms": compare_res["median_ms"],
            "status": "passed",
        })

    return rows


def _benchmark_extended_cpu_io(
    truth_dets: outputs.Detections,
    frames: Any,
    *,
    repeats: int,
    large_scale: bool = False,
) -> list[dict[str, Any]]:
    """Benchmark video random-jump seek, Detections.export, datasets.partition, archive ZIP, and JSON I/O."""
    rows: list[dict[str, Any]] = []
    if len(truth_dets) == 0 or len(frames) == 0:
        return rows

    with tempfile.TemporaryDirectory() as tmp:
        root = files.path(tmp)

        # 1. Video indexed load + random-jump frame seek (120 frames)
        sample_pils = [media.to_pil(frames[i % len(frames)]).resize((640, 480)) for i in range(min(16, len(frames)))]
        vid_tmp = root / "clip_tmp.mp4"
        vid_h264 = root / "clip_h264.mp4"
        media.save_video(vid_tmp, (sample_pils[i % len(sample_pils)] for i in range(120)), fps=30, overwrite=True, verbose=False)
        try:
            media.convert(vid_tmp, vid_h264, codec="h264", gpu=False, preset="fast", overwrite=True, verbose=False)
            vid_path = vid_h264
        except Exception:
            vid_path = vid_tmp

        vid_idx_stats = _measure(lambda: media.load(vid_path, indexed=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        vid_frames = vid_idx_stats["last_result"]
        rows.append({"operation": f"media.video_load_indexed ({len(vid_frames)} frames)", "median_ms": vid_idx_stats["median_ms"], "status": "passed"})

        rng = np.random.default_rng(42)
        v_jumps = [int(i) for i in rng.choice(len(vid_frames), size=min(32, len(vid_frames)), replace=False)]
        v_np = _measure(lambda: [media.to_array(vid_frames[i]) for i in v_jumps], repeats=repeats, warmup=1, use_cuda=False)
        v_pil = _measure(lambda: [media.to_pil(vid_frames[i]) for i in v_jumps], repeats=repeats, warmup=1, use_cuda=False)
        vid_frames.close()
        rows.append({"operation": f"media.video_random_jump_numpy ({len(v_jumps)}/{len(vid_frames)})", "median_ms": v_np["median_ms"], "status": "passed"})
        rows.append({"operation": f"media.video_random_jump_pil ({len(v_jumps)}/{len(vid_frames)})", "median_ms": v_pil["median_ms"], "status": "passed"})

        # 2. Detections.export & datasets.partition
        sub_n = min(len(truth_dets), 500 if large_scale else 128)
        sub_dets = outputs.Detections(list(truth_dets)[:sub_n], names=truth_dets.names)
        exp_dir = root / "exported_yolo"
        exp_stats = _measure(lambda: sub_dets.export(exp_dir, format="yolo", overwrite=True), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"detections.export_yolo ({sub_n} imgs)", "median_ms": exp_stats["median_ms"], "status": "passed"})

        part_dir = root / "partitioned_yolo"
        part_stats = _measure(
            lambda: datasets.partition(exp_dir, part_dir, ratios=(0.8, 0.2), format="yolo", overwrite=True, verbose=False),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"datasets.partition_yolo ({sub_n} imgs)", "median_ms": part_stats["median_ms"], "status": "passed"})

        # 3. Archive compress & extract
        zip_out = root / "dataset_bundle.zip"
        ext_dir = root / "extracted_yolo"
        comp_stats = _measure(lambda: archive.compress(exp_dir, zip_out, include_root=True, overwrite=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        ext_stats = _measure(lambda: archive.extract(zip_out, ext_dir, overwrite=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"archive.compress_zip ({sub_n} imgs)", "median_ms": comp_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"archive.extract_zip ({sub_n} imgs)", "median_ms": ext_stats["median_ms"], "status": "passed"})

        # 4. Large JSON save & load
        json_path = root / "annotations_dump.json"
        payload = truth_dets.to_dict()
        js_save = _measure(lambda: files.save(json_path, payload, overwrite=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        js_load = _measure(lambda: files.load(json_path, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"files.save_json ({len(truth_dets)} frames)", "median_ms": js_save["median_ms"], "status": "passed"})
        rows.append({"operation": f"files.load_json ({len(truth_dets)} frames)", "median_ms": js_load["median_ms"], "status": "passed"})

    return rows


def _benchmark_strided_image_access(
    source_root: Any,
    img_sub: str,
    lbl_sub: str,
    yaml_file: str,
    repeats: int,
    target_images: int = 2048,
) -> list[dict[str, Any]]:
    """Benchmark strided/random-access image loading across thousands of real JPEGs."""
    rows: list[dict[str, Any]] = []
    src_imgs = sorted(files.list_files(source_root / img_sub, extensions=(".jpg", ".jpeg", ".png")))
    src_lbls = source_root / lbl_sub
    if not src_imgs:
        return rows

    with tempfile.TemporaryDirectory() as tmp:
        big_root = files.path(tmp) / "scaled_dataset"
        big_imgs = big_root / "images"
        big_lbls = big_root / "labels"
        files.mkdir(big_imgs)
        files.mkdir(big_lbls)
        if yaml_file and files.is_file(source_root / yaml_file):
            files.copy(source_root / yaml_file, big_root / "data.yaml", overwrite=True)

        def _link_or_copy(src: Any, dst: Any) -> None:
            try:
                os.link(src, dst)
            except OSError:
                files.copy(src, dst, overwrite=True)

        total = max(len(src_imgs), target_images)
        for idx in range(total):
            src_img = src_imgs[idx % len(src_imgs)]
            dst_img = big_imgs / f"img_{idx:05d}{src_img.suffix}"
            _link_or_copy(src_img, dst_img)
            src_lbl = src_lbls / f"{src_img.stem}.txt"
            if files.is_file(src_lbl):
                dst_lbl = big_lbls / f"img_{idx:05d}.txt"
                _link_or_copy(src_lbl, dst_lbl)

        ds_load = _measure(
            lambda: datasets.load(big_root, images_dir="images", labels_dir="labels", yaml_name="data.yaml", verbose=False),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        big_dets = ds_load["last_result"]
        rows.append({"operation": f"datasets.load_scaled ({total} imgs)", "median_ms": ds_load["median_ms"], "status": "passed"})

        idx_load = _measure(
            lambda: media.load(big_imgs, indexed=True, verbose=False),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        indexed_frames = idx_load["last_result"]
        rows.append({"operation": f"media.load_indexed ({total} imgs)", "median_ms": idx_load["median_ms"], "status": "passed"})

        stride = 32
        strided_decode = _measure(
            lambda: [media.to_array(f) for f in media.load(big_imgs, sample_rate=stride, verbose=False)],
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"media.strided_decode (stride={stride}, {total // stride} imgs)", "median_ms": strided_decode["median_ms"], "status": "passed"})

        rng = np.random.default_rng(42)
        jump_indices = [int(i) for i in rng.choice(len(indexed_frames), size=min(64, len(indexed_frames)), replace=False)]
        random_decode = _measure(
            lambda: [media.to_array(indexed_frames[i]) for i in jump_indices],
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"media.random_jump_numpy (64/{total} imgs)", "median_ms": random_decode["median_ms"], "status": "passed"})

        random_pil = _measure(
            lambda: [media.to_pil(indexed_frames[i]) for i in jump_indices],
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"media.random_jump_pil (64/{total} imgs)", "median_ms": random_pil["median_ms"], "status": "passed"})

        dets_slice = _measure(
            lambda: [big_dets[i] for i in jump_indices],
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"detections.random_jump_index (64/{total})", "median_ms": dets_slice["median_ms"], "status": "passed"})

        dets_image_read = _measure(
            lambda: [big_dets[i].image for i in jump_indices],
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"detections.random_jump_image_read (64/{total})", "median_ms": dets_image_read["median_ms"], "status": "passed"})

    return rows


def _prepare_real_dataset(dataset: str, dataset_dir: str) -> tuple[Any | None, str, str, str]:
    """Resolve a real YOLO dataset directory (custom --dataset-dir or auto-downloaded coco128)."""
    if dataset_dir:
        root = files.path(dataset_dir)
        if not files.is_dir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        img_sub = "images/train2017" if files.is_dir(root / "images" / "train2017") else "images"
        lbl_sub = "labels/train2017" if files.is_dir(root / "labels" / "train2017") else "labels"
        return root, img_sub, lbl_sub, f"custom:{root}"

    if dataset == "synthetic":
        return None, "images", "labels", "synthetic"

    coco128_root = files.path("artifacts/datasets/coco128")
    if not files.is_dir(coco128_root / "images" / "train2017"):
        try:
            dl_dir = files.path("artifacts/datasets")
            files.mkdir(dl_dir)
            files.download(
                "https://ultralytics.com/assets/coco128.zip",
                dl_dir,
                overwrite=True,
                verbose=False,
            )
            archive.extract(dl_dir / "coco128.zip", dl_dir, overwrite=True, verbose=False)
        except Exception as exc:
            if dataset == "coco128":
                raise RuntimeError(f"Failed to download coco128 dataset: {exc}") from exc
            return None, "images", "labels", f"synthetic (coco128 fallback: {exc})"

    if files.is_dir(coco128_root / "images" / "train2017"):
        if not files.is_file(coco128_root / "data.yaml"):
            coco_names = {i: f"cls_{i}" for i in range(80)}
            files.save(coco128_root / "data.yaml", {"names": coco_names}, overwrite=True, verbose=False)
        return coco128_root, "images/train2017", "labels/train2017", "coco128"
    return None, "images", "labels", "synthetic"


def benchmark_datasets_media_visual(
    repeats: int,
    large_scale: bool = False,
    include_cpu_utils: bool = False,
    dataset: str = "auto",
    dataset_dir: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Benchmark datasets.load, media.load, disk-backed mining, preprocessing, and optional CPU utilities."""
    rows: list[dict[str, Any]] = []
    real_root, img_sub, lbl_sub, dataset_tag = _prepare_real_dataset(dataset, dataset_dir)
    if real_root is not None:
        yaml_file = "data.yaml" if files.is_file(real_root / "data.yaml") else ""
        load_stats = _measure(
            lambda: datasets.load(
                real_root,
                images_dir=img_sub,
                labels_dir=lbl_sub,
                yaml_name=yaml_file or "data.yaml",
                verbose=False,
            ),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        truth_dets = load_stats["last_result"]
        num_images = len(truth_dets)
        img_dir = real_root / img_sub
        rows.append({"operation": f"datasets.load ({dataset_tag}, {num_images} imgs)", "median_ms": load_stats["median_ms"], "status": "passed"})

        media_eager = _measure(lambda: media.load(img_dir, stream=False, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        media_stream = _measure(lambda: list(media.load(img_dir, stream=True, verbose=False)), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"media.load_eager ({dataset_tag}, {num_images} imgs)", "median_ms": media_eager["median_ms"], "status": "passed"})
        rows.append({"operation": f"media.load_stream ({dataset_tag}, {num_images} imgs)", "median_ms": media_stream["median_ms"], "status": "passed"})

        frames = media.load(img_dir, verbose=False)
        preproc_pipe = preprocessing.compose(
            preprocessing.letterbox((640, 640)),
            preprocessing.autocontrast(),
        )
        sample_n = min(32 if large_scale else 16, len(frames))
        preproc_stats = _measure(lambda: [preproc_pipe(f) for f in frames[:sample_n]], repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"preprocessing.compose ({dataset_tag}, {sample_n} imgs)", "median_ms": preproc_stats["median_ms"], "status": "passed"})

        disk_stats = _measure(lambda: mining.stats(truth_dets, quality=True), repeats=repeats, warmup=1, use_cuda=False)
        dup_stats = _measure(lambda: mining.find_duplicates(truth_dets, exact=True, method="phash"), repeats=repeats, warmup=1, use_cuda=False)
        cluster_k = min(8, max(2, num_images // 8))
        cluster_stats = _measure(
            lambda: mining.cluster(truth_dets, features=("boxes", "classes", "quality"), clusters=cluster_k),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"mining.stats_quality ({dataset_tag}, {num_images} imgs)", "median_ms": disk_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"mining.find_duplicates ({dataset_tag}, {num_images} imgs)", "median_ms": dup_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"mining.cluster ({dataset_tag}, {num_images} imgs, k={cluster_k})", "median_ms": cluster_stats["median_ms"], "status": "passed"})

        if large_scale:
            rows.extend(_benchmark_strided_image_access(real_root, img_sub, lbl_sub, yaml_file, repeats=repeats))
        rows.extend(_benchmark_extended_cpu_io(truth_dets, frames, repeats=repeats, large_scale=large_scale))
        return rows, dataset_tag

    num_images = 200 if large_scale else 12
    with tempfile.TemporaryDirectory() as tmp:
        root = files.path(tmp)
        ds_dir = root / "ds"
        files.mkdir(ds_dir / "images")
        files.mkdir(ds_dir / "labels")
        files.mkdir(ds_dir / "pseudo_labels")
        files.save(ds_dir / "data.yaml", {"names": {0: "person", 1: "car"}}, overwrite=True, verbose=False)
        files.save(ds_dir / "pseudo.yaml", {"names": {0: "person", 1: "car"}}, overwrite=True, verbose=False)

        for i in range(num_images):
            img_path = ds_dir / "images" / f"img_{i:04d}.jpg"
            Image.new("RGB", (320, 240), ((i * 29) % 255, (i * 53) % 255, (i * 71) % 255)).save(img_path)
            files.save(ds_dir / "labels" / f"img_{i:04d}.txt", ["0 0.5 0.5 0.3 0.4"], overwrite=True, verbose=False)
            files.save(ds_dir / "pseudo_labels" / f"img_{i:04d}.txt", ["0 0.52 0.5 0.3 0.4", "1 0.2 0.2 0.1 0.1"], overwrite=True, verbose=False)

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
        rows.append({"operation": f"datasets.load_2x_single ({num_images} imgs)", "median_ms": single_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"datasets.load_group ({num_images} imgs)", "median_ms": group_stats["median_ms"], "status": "passed"})

        media_eager = _measure(lambda: media.load(ds_dir / "images", stream=False, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
        media_stream = _measure(lambda: list(media.load(ds_dir / "images", stream=True, verbose=False)), repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"media.load_eager ({num_images} imgs)", "median_ms": media_eager["median_ms"], "status": "passed"})
        rows.append({"operation": f"media.load_stream ({num_images} imgs)", "median_ms": media_stream["median_ms"], "status": "passed"})

        frames = media.load(ds_dir / "images", verbose=False)
        preproc_pipe = preprocessing.compose(
            preprocessing.letterbox((640, 640)),
            preprocessing.autocontrast(),
        )
        preproc_stats = _measure(lambda: [preproc_pipe(f) for f in frames[:min(16, num_images)]], repeats=repeats, warmup=1, use_cuda=False)
        rows.append({"operation": f"preprocessing.compose ({min(16, num_images)} imgs)", "median_ms": preproc_stats["median_ms"], "status": "passed"})

        truth_dets = loaded_grp["truth"]
        disk_stats = _measure(lambda: mining.stats(truth_dets, quality=True), repeats=repeats, warmup=1, use_cuda=False)
        dup_stats = _measure(lambda: mining.find_duplicates(truth_dets, exact=True, method="phash"), repeats=repeats, warmup=1, use_cuda=False)
        cluster_k = min(8, max(2, num_images // 4))
        cluster_stats = _measure(
            lambda: mining.cluster(truth_dets, features=("boxes", "classes", "quality"), clusters=cluster_k),
            repeats=repeats, warmup=1, use_cuda=False,
        )
        rows.append({"operation": f"mining.stats_quality ({num_images} imgs)", "median_ms": disk_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"mining.find_duplicates ({num_images} imgs)", "median_ms": dup_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"mining.cluster ({num_images} imgs, k={cluster_k})", "median_ms": cluster_stats["median_ms"], "status": "passed"})

        if include_cpu_utils:
            info_stats = _measure(lambda: media.info(ds_dir / "images" / "img_0000.jpg"), repeats=repeats, warmup=1, use_cuda=False)
            rows.append({"operation": "media.info", "median_ms": info_stats["median_ms"], "status": "passed"})

            save_out = root / "saved_imgs"
            save_stats = _measure(lambda: media.save_images(save_out, frames[:12], size=(256, 256), overwrite=True, verbose=False), repeats=repeats, warmup=1, use_cuda=False)
            assert save_stats["last_result"] is None
            rows.append({"operation": "media.save_images_letterbox", "median_ms": save_stats["median_ms"], "status": "passed"})

            cmp_out = root / "compare.png"
            cmp_stats = _measure(
                lambda: graphics.compare(loaded_grp["truth"][0], loaded_grp["pred"][0], mode="save", output_path=cmp_out, overwrite=True),
                repeats=repeats, warmup=1, use_cuda=False,
            )
            rows.append({"operation": "graphics.compare", "median_ms": cmp_stats["median_ms"], "status": "passed"})

            def _silent_tab():
                buf = io.StringIO()
                with redirect_stdout(buf):
                    utils.tabulate(rows)
            tab_stats = _measure(_silent_tab, repeats=repeats, warmup=1, use_cuda=False)
            rows.append({"operation": "utils.tabulate", "median_ms": tab_stats["median_ms"], "status": "passed"})

    return rows, dataset_tag


def _md_table(rows: list[dict[str, Any]], title: str) -> list[str]:
    if not rows:
        return []
    headers = list(rows[0].keys())
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.append("")
    return lines


def _generate_charts(summary: dict[str, Any], out_dir: Any) -> list[tuple[str, str]]:
    """Render summary PNG charts into out_dir/charts/ when matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    charts_dir = out_dir / "charts"
    files.mkdir(charts_dir)
    created: list[tuple[str, str]] = []

    # 1. Metrics CPU vs CUDA (FP32 unpacked)
    metrics_rows = [
        r for r in summary.get("metrics", [])
        if r.get("precision") == "fp32" and not r.get("packed")
    ]
    if metrics_rows:
        cases = list(dict.fromkeys(r["case"] for r in metrics_rows))
        devices = list(dict.fromkeys(r["device"] for r in metrics_rows))
        x = np.arange(len(cases))
        width = 0.35 if len(devices) > 1 else 0.5
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for idx, dev in enumerate(devices):
            vals = [
                next((r["evaluate_median_ms"] for r in metrics_rows if r["case"] == c and r["device"] == dev), 0.0)
                for c in cases
            ]
            ax.bar(x + idx * width, vals, width, label=dev.upper())
        ax.set_xticks(x + (width * (len(devices) - 1) / 2 if len(devices) > 1 else 0))
        ax.set_xticklabels(cases, rotation=20, ha="right")
        ax.set_ylabel("Median Latency (ms)")
        ax.set_title("klygo.metrics.evaluate (FP32, Unpacked)")
        ax.legend()
        fig.tight_layout()
        p = charts_dir / "metrics_cpu_vs_cuda.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(("klygo.metrics CPU vs CUDA", "charts/metrics_cpu_vs_cuda.png"))

    # 2. Mining CPU vs CUDA
    mining_rows = summary.get("mining", [])
    if mining_rows:
        ops = ["stats_ms", "score_ms", "select_ms", "balance_ms", "compare_ms"]
        x = np.arange(len(ops))
        width = 0.35 if len(mining_rows) > 1 else 0.5
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for idx, row in enumerate(mining_rows):
            vals = [float(row.get(op, 0.0)) for op in ops]
            ax.bar(x + idx * width, vals, width, label=str(row.get("device", "")).upper())
        ax.set_xticks(x + (width * (len(mining_rows) - 1) / 2 if len(mining_rows) > 1 else 0))
        ax.set_xticklabels([o.replace("_ms", "") for o in ops])
        ax.set_ylabel("Median Latency (ms)")
        ax.set_title("klygo.mining Operations (CPU vs CUDA)")
        ax.legend()
        fig.tight_layout()
        p = charts_dir / "mining_cpu_vs_cuda.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(("klygo.mining CPU vs CUDA", "charts/mining_cpu_vs_cuda.png"))

    # 3. 6-Mode Model FPS (managed-non-stream vs raw-non-stream / native-non-stream)
    mode_rows = [
        r for r in summary.get("model_modes", [])
        if r.get("status") == "ok" and r.get("fps_median") is not None
    ]
    if mode_rows:
        models_list = list(dict.fromkeys(f"{r['model']} ({r['dtype']})" for r in mode_rows))
        modes_list = ["raw-non-stream", "native-non-stream", "managed-non-stream"]
        x = np.arange(len(models_list))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(10, len(models_list) * 0.9), 4.8))
        for idx, mode_name in enumerate(modes_list):
            vals = [
                next(
                    (float(r["fps_median"]) for r in mode_rows if f"{r['model']} ({r['dtype']})" == m and r["mode"] == mode_name),
                    0.0,
                )
                for m in models_list
            ]
            ax.bar(x + idx * width, vals, width, label=mode_name)
        ax.set_xticks(x + width)
        ax.set_xticklabels(models_list, rotation=30, ha="right")
        ax.set_ylabel("Median FPS")
        ax.set_title("Model Throughput by Execution Mode")
        ax.legend()
        fig.tight_layout()
        p = charts_dir / "model_fps_comparison.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(("Model FPS by Execution Mode", "charts/model_fps_comparison.png"))

    # 4. models.load Cold-Load Time
    load_rows = [r for r in summary.get("model_loads", []) if r.get("load_seconds") is not None]
    if load_rows:
        labels = [f"{r['model']} ({r['dtype']})" for r in load_rows]
        vals = [float(r["load_seconds"]) for r in load_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.2))
        ax.bar(labels, vals)
        ax.set_ylabel("Load Time (s)")
        ax.set_title("models.load Cold-Load Time")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        p = charts_dir / "model_load_seconds.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(("models.load Cold-Load Time", "charts/model_load_seconds.png"))

    # 5. Scaling FPS
    scaling_rows = [r for r in summary.get("scaling", []) if r.get("mode") == "managed-non-stream" and r.get("fps_median")]
    if scaling_rows:
        labels = [f"{r['model']} {r['dtype']} b{r['batch']} {'P' if r['packed'] else 'U'}" for r in scaling_rows]
        vals = [float(r["fps_median"]) for r in scaling_rows]
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.5))
        ax.bar(labels, vals)
        ax.set_ylabel("Median FPS")
        ax.set_title("Scaling Throughput (managed-non-stream)")
        plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        p = charts_dir / "scaling_fps.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        created.append(("Batch / Packed Scaling FPS", "charts/scaling_fps.png"))

    return created


def _render_markdown_report(summary: dict[str, Any], charts: list[tuple[str, str]] | None = None) -> str:
    env = summary["environment"]
    git_info = env.get("git", {})
    lines = [
        "# Klygo Release Benchmark Report",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` ({env.get('vram_mb', 0.0)} MB VRAM)",
        f"- **CUDA / CuPy**: `torch_cuda={env.get('torch_cuda')}`, `cupy_cuda={env.get('cupy_cuda')}`, `cuda_runtime={env.get('cuda_runtime')}`",
        f"- **Git**: `{git_info.get('branch', 'unknown')}` @ `{git_info.get('commit', 'unknown')}` (dirty: `{git_info.get('dirty', 'unknown')}`)",
        f"- **OS / Python**: `{env.get('os', '')}` / Python `{env.get('python', '')}`",
        "",
    ]
    versions = env.get("versions", {})
    if versions:
        lines.extend(_md_table([versions], "Package Versions"))
    lines.extend(_md_table(summary.get("metrics", []), "klygo.metrics (CPU vs CUDA)"))
    lines.extend(_md_table(summary.get("group_post", []), "DetectionGroup / Postprocessing / Outputs / Transfer"))
    lines.extend(_md_table(summary.get("mining", []), "klygo.mining (CPU vs CUDA)"))
    lines.extend(_md_table(summary.get("io_visual", []), "Loaders / Preprocessing / Disk Mining"))
    lines.extend(_md_table(summary.get("model_loads", []), "models.load Cold-Load Latency"))
    lines.extend(_md_table(summary.get("models", []), "6-Mode Model Run Status"))
    lines.extend(_md_table(summary.get("model_modes", []), "6-Mode Model Inference Details"))
    lines.extend(_md_table(summary.get("scaling", []), "Batch / Workers / Prefetch / Packed Scaling"))
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel_path in charts:
            lines.extend([f"### {title}", "", f"![{title}]({rel_path})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _check_resume_environment(expected, current):
    for key in ('gpu', 'gpu_uuid', 'vram_mb', 'python', 'cuda_runtime', 'torch_cuda', 'cupy_cuda'):
        if expected.get(key) is not None and current.get(key) != expected[key]:
            raise RuntimeError(f'Resume environment changed ({key}); use a fresh release run')
    for key, value in expected.get('versions', {}).items():
        if current.get('versions', {}).get(key) != value:
            raise RuntimeError(f'Resume package version changed ({key}); use a fresh release run')


def _archive_final(output, zip_path=""):
    """Retain superseded final artifacts before an in-place Locate repair."""
    stamp = str(time.time_ns())
    previous = output / 'previous_final' / stamp
    if zip_path and files.exists(zip_path):
        old_zip = files.path(zip_path)
        files.move(old_zip, old_zip.with_name(f'{old_zip.stem}.previous-{stamp}.zip'))
    for name in ('summary.json', 'REPORT.md', 'charts', 'completion.json'):
        path = output / name
        if files.exists(path):
            files.mkdir(previous)
            files.move(path, previous / name)


def _finalize(summary, output, *, require_locate=False, dtypes=(), device='cuda:0', repeats=3):
    errors = require_results(output, dtypes, device, repeats, summary['models'],
                             expected_env=summary.get('environment', {})) if require_locate else []
    if require_locate:
        for dtype in dtypes:
            displayed = [row for row in summary.get('model_modes', [])
                         if row.get('model') == 'locate-anything' and row.get('dtype') == dtype
                         and row.get('status') == 'ok' and row.get('fps_median') is not None]
            if {row.get('mode') for row in displayed} != SUPPORTED:
                errors.append(f'{dtype}: Locate measurements are missing from the combined chart data')
            if not any(row.get('model') == 'locate-anything' and row.get('dtype') == dtype
                       for row in summary.get('model_loads', [])):
                errors.append(f'{dtype}: Locate load results are missing from the combined chart data')
    if errors:
        files.save(output / 'summary.partial.json', summary, overwrite=True, verbose=False)
        files.save(output / 'completion.json', {'status': 'incomplete', 'errors': errors}, overwrite=True, verbose=False)
        raise RuntimeError('Final report and charts blocked: ' + '; '.join(errors))
    files.save(output / 'summary.json', summary, overwrite=True, verbose=False)
    charts = _generate_charts(summary, output)
    files.save(output / 'REPORT.md', _render_markdown_report(summary, charts), overwrite=True, verbose=False)
    files.save(output / 'completion.json', {'status': 'complete', 'locate_required': require_locate}, overwrite=True, verbose=False)
    files.remove(output / 'summary.partial.json', missing_ok=True)


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
    parser.add_argument("--locate-python", default="", help="Optional Python executable for locate-anything (e.g. isolated transformers==4.57.1 venv)")
    parser.add_argument("--locate-isolated", action="store_true", help="Force Locate Anything into a managed venv")
    parser.add_argument("--locate-auto-fix", action=argparse.BooleanOptionalAction, default=True, help="Repair known Locate dependency failures in a managed venv")
    parser.add_argument("--locate-retries", type=int, default=2, help="Maximum dependency-repair retries per Locate precision")
    parser.add_argument("--locate-venv", default="", help="Managed Locate venv path outside the report directory")
    parser.add_argument("--require-locate-results", action="store_true", help="Include Locate Anything and block final artifacts until every requested precision passes")
    parser.add_argument("--locate-only", action="store_true", help="Repair/re-run only Locate cases in an existing release output and merge before finalizing")
    parser.add_argument("--include-scaling", action="store_true")
    parser.add_argument("--scaling-models", default="yolo,yolo-world")
    parser.add_argument("--large-scale", action="store_true", help="Enable large-scale (10k+ boxes / 200-image) GPU & loader workloads")
    parser.add_argument("--dataset", choices=["auto", "coco128", "synthetic"], default="auto", help="Dataset source for loader & mining benchmarks")
    parser.add_argument("--dataset-dir", default="", help="Optional path to a local real YOLO dataset (e.g., coco-val2017)")
    parser.add_argument("--include-cpu-utils", action="store_true", help="Also time CPU-only media.info/save_images/visual/tabulate helpers")
    parser.add_argument("--loaders-only", action="store_true", help="Run only loader/preprocessing/disk-mining benchmarks (skip metrics, outputs, models, scaling)")
    parser.add_argument("--zip", default="", help="Optional output .zip path to bundle the complete benchmark directory")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    if args.locate_retries < 0 or args.repeats < 1:
        parser.error("repeats must be positive and locate-retries must be non-negative")
    selected_models = [m.strip() for m in args.models.split(',') if m.strip()]
    requested_dtypes = [d.strip() for d in args.dtypes.split(',') if d.strip()]
    if not requested_dtypes or len(set(requested_dtypes)) != len(requested_dtypes):
        parser.error("dtypes must be non-empty and unique")
    if args.require_locate_results:
        args.include_models = True
        if 'locate-anything' not in selected_models:
            selected_models.append('locate-anything')
    config_path = out_dir / 'run_config.json'
    if args.locate_only:
        if not files.exists(config_path):
            parser.error("locate-only requires an existing release run_config.json")
        config = files.load(config_path, verbose=False)
        if config.get('inference_runner_sha256') != files.hash('benchmarks/gpu_suite/run.py'):
            parser.error("Inference runner changed; run a fresh release benchmark instead of merging")
        args.device, args.repeats = config['device'], config['repeats']
        requested_dtypes = config['dtypes']
        args.zip = args.zip or config.get('zip', '')
        selected_models = ['locate-anything']
        args.include_models = True
        args.include_scaling = False
        args.require_locate_results = True
    else:
        if any(files.exists(out_dir / name) for name in ('run_config.json', 'summary.json', 'summary.partial.json')):
            parser.error("Use a fresh output directory, or --locate-only to repair an existing run")
        config = {'models': selected_models if args.include_models else [], 'dtypes': requested_dtypes,
                  'device': args.device, 'repeats': args.repeats, 'zip': args.zip,
                  'inference_runner_sha256': files.hash('benchmarks/gpu_suite/run.py')}
    need_locate = args.include_models and 'locate-anything' in selected_models
    if args.zip:
        zip_destination = files.path(args.zip).resolve()
        if zip_destination == out_dir.resolve() or out_dir.resolve() in zip_destination.parents:
            parser.error("Keep the benchmark ZIP outside its source report directory")
        if files.exists(zip_destination) and not args.locate_only:
            parser.error("Use a fresh ZIP path to preserve the previous benchmark bundle")
    if need_locate and not args.allow_remote_code:
        parser.error("Locate Anything requires --allow-remote-code")
    if need_locate:
        managed = files.path(args.locate_venv or f'artifacts/benchmark-envs/locate-py{sys.version_info.major}{sys.version_info.minor}').resolve()
        if managed == out_dir.resolve() or out_dir.resolve() in managed.parents:
            parser.error("Keep the Locate venv outside the report directory and ZIP")
    if args.locate_only:
        # Load all sections before archiving the previous final artifacts.
        env = files.load(out_dir / 'environment.json', verbose=False)
        resume_env = collect_environment()
        _check_resume_environment(env, resume_env)
        files.save(out_dir / f'resume_environment_{time.time_ns()}.json', resume_env, verbose=False)
        metrics_rows = files.load(out_dir / 'metrics_benchmark.json', verbose=False)
        group_rows = files.load(out_dir / 'outputs_post_benchmark.json', verbose=False)
        mining_rows = files.load(out_dir / 'mining_benchmark.json', verbose=False)
        dmv_rows = files.load(out_dir / 'datasets_media_visual_benchmark.json', verbose=False)
        def without_locate(name):
            rows = files.load(out_dir / name, verbose=False)
            return [row for row in rows if row.get('model') != 'locate-anything']
        model_runs = without_locate('model_runs.json')
        model_modes = without_locate('model_modes.json')
        model_loads = without_locate('model_loads.json')
        scaling_rows = files.load(out_dir / 'scaling_runs.json', verbose=False) if files.exists(out_dir / 'scaling_runs.json') else []
        _archive_final(out_dir, args.zip)
        if 'locate-anything' not in config['models']:
            config['models'].append('locate-anything')
    else:
        env = collect_environment()
        files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)
        print("=== Environment ===")
        utils.tabulate({k: v for k, v in env.items() if k not in ("git", "versions")}, title="Runtime")
        utils.tabulate(env["versions"], title="Versions")

        metrics_rows: list[dict[str, Any]] = []
        group_rows: list[dict[str, Any]] = []
        mining_rows: list[dict[str, Any]] = []
        if not args.loaders_only:
            metrics_rows = benchmark_metrics(env, repeats=args.repeats, large_scale=args.large_scale)
            files.save(out_dir / "metrics_benchmark.json", metrics_rows, overwrite=True, verbose=False)
            print("\n=== klygo.metrics Benchmark ===")
            utils.tabulate(metrics_rows, title="Metrics (CPU vs CUDA, FP16/FP32/FP64)")

            group_rows = benchmark_groups_and_postprocessing(env, repeats=args.repeats, large_scale=args.large_scale)
            files.save(out_dir / "outputs_post_benchmark.json", group_rows, overwrite=True, verbose=False)
            print("\n=== DetectionGroup / Postprocessing / Transfer ===")
            utils.tabulate(group_rows, title="Outputs & Postprocessing")

            mining_rows = benchmark_mining(env, repeats=args.repeats, large_scale=args.large_scale)
            files.save(out_dir / "mining_benchmark.json", mining_rows, overwrite=True, verbose=False)
            print("\n=== klygo.mining (CPU vs CUDA) ===")
            utils.tabulate(mining_rows, title="Mining (CPU vs CUDA)")

        dmv_rows, dataset_tag = benchmark_datasets_media_visual(
            repeats=args.repeats,
            large_scale=args.large_scale,
            include_cpu_utils=args.include_cpu_utils,
            dataset=args.dataset,
            dataset_dir=args.dataset_dir,
        )
        env["dataset_source"] = dataset_tag
        files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)
        files.save(out_dir / "datasets_media_visual_benchmark.json", dmv_rows, overwrite=True, verbose=False)
        print("\n=== Loaders / Preprocessing / Disk Mining ===")
        utils.tabulate(dmv_rows, title="Loaders, Preprocessing & Disk Mining")

        model_runs, model_modes, model_loads, scaling_rows = [], [], [], []
    files.save(config_path, config, overwrite=True, verbose=False)
    locate_runtime = LocateRuntime(out_dir, python=args.locate_python,
        isolated=args.locate_isolated, auto_fix=args.locate_auto_fix,
        retries=args.locate_retries, venv_dir=args.locate_venv) if need_locate else None
    if args.include_models:
        for model_id in selected_models:
            for dtype_name in requested_dtypes:
                case_out = out_dir / "inference" / f"{model_id}_{dtype_name}"
                py_exec = args.locate_python if (model_id == "locate-anything" and args.locate_python) else sys.executable
                cmd = [
                    py_exec,
                    "benchmarks/gpu_suite/run.py",
                    "--model", model_id,
                    "--device", args.device,
                    "--dtype", dtype_name,
                    "--repeats", str(args.repeats),
                    "--output", str(case_out),
                ]
                if args.allow_remote_code:
                    cmd.append("--allow-remote-code")
                proc = (locate_runtime.run(cmd, case_out, dtype_name) if model_id == "locate-anything"
                        else subprocess.run(cmd, capture_output=True, text=True))
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
                    try:
                        data = files.load(json_path, verbose=False)
                    except (ValueError, OSError, TypeError) as exc:
                        model_runs[-1].update(status='failed', error=f'Unreadable raw report: {exc}')
                        continue
                    model_loads.append({
                        "model": model_id,
                        "dtype": dtype_name,
                        "actual_dtype": data.get("actual_model_dtype", ""),
                        "load_seconds": round(float(data.get("load_seconds", 0.0)), 4) if data.get("load_seconds") is not None else None,
                        "device": data.get("device", ""),
                    })
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
        files.save(out_dir / "model_loads.json", model_loads, overwrite=True, verbose=False)
        print("\n=== 6-Mode Model Inference Runs ===")
        utils.tabulate(model_runs, title="6-Mode Inference Summary")
        if model_loads:
            print("\n=== models.load Summary ===")
            utils.tabulate(model_loads, title="models.load Cold Load Latency")
        if model_modes:
            print("\n=== 6-Mode Per-Mode Details ===")
            utils.tabulate(model_modes, title="6-Mode Per-Mode Details")

    if args.include_scaling:
        configs = [
            (1, 1, 1, True),
            (4, 2, 2, False),
            (4, 2, 2, True),
            (8, 4, 4, False),
            (8, 4, 4, True),
        ]
        for model_id in [m.strip() for m in args.scaling_models.split(",") if m.strip()]:
            for dtype_name in requested_dtypes:
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
        "mining": mining_rows,
        "io_visual": dmv_rows,
        "models": model_runs,
        "model_loads": model_loads,
        "model_modes": model_modes,
        "scaling": scaling_rows,
    }
    _finalize(summary, out_dir, require_locate=need_locate, dtypes=requested_dtypes,
              device=args.device, repeats=args.repeats)
    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"\nPackaged complete benchmark bundle: {zip_path}")


if __name__ == "__main__":
    main()

"""Standalone CPU & Disk I/O benchmark runner for Klygo (no GPU required).

Measures:
    - datasets.load (single / scaled 2,048+ images)
    - media.load (eager, stream, indexed)
    - Strided & random-jump image reads on MediaFrames (array & PIL) and Detections
    - Video indexed load & random-jump frame seeking (H.264 / MP4)
    - preprocessing.compose (letterbox + autocontrast)
    - Detections.export, datasets.partition, archive ZIP compress/extract, JSON I/O
    - Optional CPU utilities (--include-cpu-utils)

Usage:
    PYTHONPATH=. python benchmarks/cpu_loaders/run_cpu_benchmark.py --dataset coco128 --large-scale --repeats 7
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import os
import platform
import statistics
import tempfile
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from klygo import (
    archive,
    datasets,
    files,
    media,
    outputs,
    preprocessing,
    utils,
    graphics,
)


def _collect_cpu_env(dataset_tag: str) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 1,
        "dataset": dataset_tag,
    }


def _measure(fn: Callable[[], Any], *, repeats: int, warmup: int = 1) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    last_result = None
    for _ in range(repeats):
        start = time.perf_counter()
        last_result = fn()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "samples_ms": [round(x, 4) for x in samples_ms],
        "median_ms": round(statistics.median(samples_ms), 4),
        "mean_ms": round(statistics.mean(samples_ms), 4),
        "std_ms": round(statistics.stdev(samples_ms), 4) if len(samples_ms) > 1 else 0.0,
        "last_result": last_result,
    }


def _prepare_dataset(dataset: str, dataset_dir: str) -> tuple[Any | None, str, str, str]:
    if dataset_dir:
        root = files.path(dataset_dir)
        if not files.is_dir(root):
            raise FileNotFoundError(f"Dataset directory not found: {root}")
        img_sub = "images/train2017" if files.is_dir(root / "images" / "train2017") else "images"
        lbl_sub = "labels/train2017" if files.is_dir(root / "labels" / "train2017") else "labels"
        return root, img_sub, lbl_sub, f"custom:{root.name}"

    if dataset == "synthetic":
        return None, "images", "labels", "synthetic"

    coco128_root = files.path("artifacts/datasets/coco128")
    if not files.is_dir(coco128_root / "images" / "train2017"):
        dl_dir = files.path("artifacts/datasets")
        files.mkdir(dl_dir)
        zip_path = files.download("https://ultralytics.com/assets/coco128.zip", dl_dir, overwrite=True, verbose=False)
        archive.extract(zip_path, dl_dir, overwrite=True, verbose=False)
    if not files.is_file(coco128_root / "data.yaml"):
        files.save(coco128_root / "data.yaml", {"names": {i: f"cls_{i}" for i in range(80)}}, overwrite=True, verbose=False)
    return coco128_root, "images/train2017", "labels/train2017", "coco128"


def _benchmark_strided_image_access(
    source_root: Any,
    img_sub: str,
    lbl_sub: str,
    yaml_file: str,
    repeats: int,
    target_images: int = 2048,
) -> list[dict[str, Any]]:
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
            _link_or_copy(src_img, big_imgs / f"img_{idx:05d}{src_img.suffix}")
            src_lbl = src_lbls / f"{src_img.stem}.txt"
            if files.is_file(src_lbl):
                _link_or_copy(src_lbl, big_lbls / f"img_{idx:05d}.txt")

        ds_load = _measure(
            lambda: datasets.load(big_root, images_dir="images", labels_dir="labels", yaml_name="data.yaml", verbose=False),
            repeats=repeats,
        )
        big_dets = ds_load["last_result"]
        rows.append({"operation": f"datasets.load_scaled ({total} imgs)", "median_ms": ds_load["median_ms"], "status": "passed"})

        idx_load = _measure(lambda: media.load(big_imgs, indexed=True, verbose=False), repeats=repeats)
        indexed_frames = idx_load["last_result"]
        rows.append({"operation": f"media.load_indexed ({total} imgs)", "median_ms": idx_load["median_ms"], "status": "passed"})

        stride = 32
        strided_decode = _measure(
            lambda: [media.to_array(f) for f in media.load(big_imgs, sample_rate=stride, verbose=False)],
            repeats=repeats,
        )
        rows.append({"operation": f"media.strided_decode (stride={stride}, {total // stride} imgs)", "median_ms": strided_decode["median_ms"], "status": "passed"})

        rng = np.random.default_rng(42)
        jump_indices = [int(i) for i in rng.choice(len(indexed_frames), size=min(64, len(indexed_frames)), replace=False)]
        rand_arr = _measure(lambda: [media.to_array(indexed_frames[i]) for i in jump_indices], repeats=repeats)
        rand_pil = _measure(lambda: [media.to_pil(indexed_frames[i]) for i in jump_indices], repeats=repeats)
        rows.append({"operation": f"media.random_jump_array (64/{total} imgs)", "median_ms": rand_arr["median_ms"], "status": "passed"})
        rows.append({"operation": f"media.random_jump_pil (64/{total} imgs)", "median_ms": rand_pil["median_ms"], "status": "passed"})

        dets_idx = _measure(lambda: [big_dets[i] for i in jump_indices], repeats=repeats)
        dets_img = _measure(lambda: [big_dets[i].image for i in jump_indices], repeats=repeats)
        rows.append({"operation": f"detections.random_jump_index (64/{total})", "median_ms": dets_idx["median_ms"], "status": "passed"})
        rows.append({"operation": f"detections.random_jump_image_read (64/{total})", "median_ms": dets_img["median_ms"], "status": "passed"})

    return rows


def _benchmark_extended_cpu_io(
    truth_dets: outputs.Detections,
    frames: Any,
    *,
    repeats: int,
    large_scale: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(truth_dets) == 0 or len(frames) == 0:
        return rows

    with tempfile.TemporaryDirectory() as tmp:
        root = files.path(tmp)

        sample_pils = [media.to_pil(frames[i % len(frames)]).resize((640, 480)) for i in range(min(16, len(frames)))]
        vid_tmp = root / "clip_tmp.mp4"
        vid_h264 = root / "clip_h264.mp4"
        media.save_video(vid_tmp, (sample_pils[i % len(sample_pils)] for i in range(120)), fps=30, overwrite=True, verbose=False)
        try:
            media.convert(vid_tmp, vid_h264, codec="h264", gpu=False, preset="fast", overwrite=True, verbose=False)
            vid_path = vid_h264
        except Exception:
            vid_path = vid_tmp

        vid_idx = _measure(lambda: media.load(vid_path, indexed=True, verbose=False), repeats=repeats)
        vid_frames = vid_idx["last_result"]
        rows.append({"operation": f"media.video_load_indexed ({len(vid_frames)} frames)", "median_ms": vid_idx["median_ms"], "status": "passed"})

        rng = np.random.default_rng(42)
        v_jumps = [int(i) for i in rng.choice(len(vid_frames), size=min(32, len(vid_frames)), replace=False)]
        v_arr = _measure(lambda: [media.to_array(vid_frames[i]) for i in v_jumps], repeats=repeats)
        v_pil = _measure(lambda: [media.to_pil(vid_frames[i]) for i in v_jumps], repeats=repeats)
        vid_frames.close()
        rows.append({"operation": f"media.video_random_jump_array ({len(v_jumps)}/{len(vid_frames)})", "median_ms": v_arr["median_ms"], "status": "passed"})
        rows.append({"operation": f"media.video_random_jump_pil ({len(v_jumps)}/{len(vid_frames)})", "median_ms": v_pil["median_ms"], "status": "passed"})

        sub_n = min(len(truth_dets), 500 if large_scale else 128)
        sub_dets = outputs.Detections(list(truth_dets)[:sub_n], names=truth_dets.names)
        exp_dir = root / "exported_yolo"
        exp_stats = _measure(lambda: sub_dets.export(exp_dir, format="yolo", overwrite=True), repeats=repeats)
        rows.append({"operation": f"detections.export_yolo ({sub_n} imgs)", "median_ms": exp_stats["median_ms"], "status": "passed"})

        part_dir = root / "partitioned_yolo"
        part_stats = _measure(
            lambda: datasets.partition(exp_dir, part_dir, ratios=(0.8, 0.2), format="yolo", overwrite=True, verbose=False),
            repeats=repeats,
        )
        rows.append({"operation": f"datasets.partition_yolo ({sub_n} imgs)", "median_ms": part_stats["median_ms"], "status": "passed"})

        zip_out = root / "dataset_bundle.zip"
        ext_dir = root / "extracted_yolo"
        comp_stats = _measure(lambda: archive.compress(exp_dir, zip_out, include_root=True, overwrite=True, verbose=False), repeats=repeats)
        ext_stats = _measure(lambda: archive.extract(zip_out, ext_dir, overwrite=True, verbose=False), repeats=repeats)
        rows.append({"operation": f"archive.compress_zip ({sub_n} imgs)", "median_ms": comp_stats["median_ms"], "status": "passed"})
        rows.append({"operation": f"archive.extract_zip ({sub_n} imgs)", "median_ms": ext_stats["median_ms"], "status": "passed"})

        json_path = root / "annotations_dump.json"
        payload = truth_dets.to_dict()
        js_save = _measure(lambda: files.save(json_path, payload, overwrite=True, verbose=False), repeats=repeats)
        js_load = _measure(lambda: files.load(json_path, verbose=False), repeats=repeats)
        rows.append({"operation": f"files.save_json ({len(truth_dets)} frames)", "median_ms": js_save["median_ms"], "status": "passed"})
        rows.append({"operation": f"files.load_json ({len(truth_dets)} frames)", "median_ms": js_load["median_ms"], "status": "passed"})

    return rows


def _render_report(env: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Klygo CPU & Disk I/O Loader Benchmark Report",
        "",
        f"- **CPU**: `{env['cpu']}` ({env['cpu_count']} cores)",
        f"- **Dataset**: `{env['dataset']}`",
        f"- **OS / Python**: `{env['os']}` / Python `{env['python']}`",
        "",
        "## Operations",
        "",
        "| operation | median_ms | status |",
        "| --- | --- | --- |",
    ]
    for r in rows:
        lines.append(f"| {r['operation']} | {r['median_ms']} | {r['status']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/benchmarks/cpu_loaders")
    parser.add_argument("--dataset", choices=["coco128", "synthetic"], default="coco128")
    parser.add_argument("--dataset-dir", default="", help="Optional path to a local YOLO dataset")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--large-scale", action="store_true")
    parser.add_argument("--extended-only", action="store_true", help="Run only extended video/export/partition/archive/JSON and cpu-utils benchmarks")
    parser.add_argument("--include-cpu-utils", action="store_true")
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    real_root, img_sub, lbl_sub, dataset_tag = _prepare_dataset(args.dataset, args.dataset_dir)
    env = _collect_cpu_env(dataset_tag)
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    rows: list[dict[str, Any]] = []
    if real_root is not None:
        yaml_file = "data.yaml" if files.is_file(real_root / "data.yaml") else ""
        img_dir = real_root / img_sub
        frames = media.load(img_dir, verbose=False)
        if args.extended_only:
            truth_dets = datasets.load(real_root, images_dir=img_sub, labels_dir=lbl_sub, yaml_name=yaml_file or "data.yaml", verbose=False)
        else:
            load_stats = _measure(
                lambda: datasets.load(real_root, images_dir=img_sub, labels_dir=lbl_sub, yaml_name=yaml_file or "data.yaml", verbose=False),
                repeats=args.repeats,
            )
            truth_dets = load_stats["last_result"]
            num_images = len(truth_dets)
            rows.append({"operation": f"datasets.load ({dataset_tag}, {num_images} imgs)", "median_ms": load_stats["median_ms"], "status": "passed"})

            media_eager = _measure(lambda: media.load(img_dir, stream=False, verbose=False), repeats=args.repeats)
            media_stream = _measure(lambda: list(media.load(img_dir, stream=True, verbose=False)), repeats=args.repeats)
            rows.append({"operation": f"media.load_eager ({dataset_tag}, {num_images} imgs)", "median_ms": media_eager["median_ms"], "status": "passed"})
            rows.append({"operation": f"media.load_stream ({dataset_tag}, {num_images} imgs)", "median_ms": media_stream["median_ms"], "status": "passed"})

            preproc_pipe = preprocessing.compose(preprocessing.letterbox((640, 640)), preprocessing.autocontrast())
            sample_n = min(32 if args.large_scale else 16, len(frames))
            preproc_stats = _measure(lambda: [preproc_pipe(f) for f in frames[:sample_n]], repeats=args.repeats)
            rows.append({"operation": f"preprocessing.compose ({dataset_tag}, {sample_n} imgs)", "median_ms": preproc_stats["median_ms"], "status": "passed"})

            if args.large_scale:
                rows.extend(_benchmark_strided_image_access(real_root, img_sub, lbl_sub, yaml_file, repeats=args.repeats))
        rows.extend(_benchmark_extended_cpu_io(truth_dets, frames, repeats=args.repeats, large_scale=args.large_scale))

        if args.include_cpu_utils:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_root = files.path(tmp)
                first_img = files.list_files(img_dir, extensions=(".jpg", ".jpeg", ".png"))[0]
                rows.append({"operation": "media.info", "median_ms": _measure(lambda: media.info(first_img), repeats=args.repeats)["median_ms"], "status": "passed"})
                save_out = tmp_root / "saved_imgs"
                rows.append({"operation": "media.save_images_letterbox", "median_ms": _measure(lambda: media.save_images(save_out, frames[:12], size=(256, 256), overwrite=True, verbose=False), repeats=args.repeats)["median_ms"], "status": "passed"})
                cmp_out = tmp_root / "compare.png"
                rows.append({"operation": "graphics.compare", "median_ms": _measure(lambda: graphics.compare(truth_dets[0], truth_dets[0], mode="save", output_path=cmp_out, overwrite=True), repeats=args.repeats)["median_ms"], "status": "passed"})
                def _silent_tab():
                    with redirect_stdout(io.StringIO()):
                        utils.tabulate(rows)
                rows.append({"operation": "utils.tabulate", "median_ms": _measure(_silent_tab, repeats=args.repeats)["median_ms"], "status": "passed"})

    utils.tabulate(rows, title=f"CPU Loaders & Disk I/O ({dataset_tag})")
    files.save(out_dir / "loaders_benchmark.json", rows, overwrite=True, verbose=False)
    files.save(out_dir / "REPORT.md", _render_report(env, rows), overwrite=True, verbose=False)
    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()

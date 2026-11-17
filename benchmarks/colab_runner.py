"""Unified Colab runner for klygo-mining-benchmark and klygo-metrics-benchmark.

Prepares the 3 datasets (coco128, voc2007, val2017) and executes:
1. Mining benchmark across 15 methods on all 3 datasets.
2. Metrics benchmark across all 9 models on all 3 datasets.
3. Packages results into .zip archives and prints JSON summaries.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Explicitly configure Keras backend before framework imports
os.environ["KERAS_BACKEND"] = "torch"

from klygo import archive, datasets, files


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

COCO80_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck",
    8: "boat", 9: "traffic light", 10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase",
    29: "frisbee", 30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def prepare_voc2007(target_dir: str | Path) -> Path:
    """Download Pascal VOC 2007 and convert val split into flat YOLO format."""
    target_path = files.path(target_dir)
    img_dir = files.join(target_path, "images")
    yaml_file = files.join(target_path, "data.yaml")
    if files.exists(yaml_file) and files.exists(img_dir):
        count = len(files.find(img_dir, "*.jpg"))
        if count >= 2000:
            print(f"VOC2007 already prepared at {target_path} ({count} images).")
            return target_path

    raw_dir = files.join(files.parent(target_path), "voc_raw")
    files.mkdir(raw_dir)
    tar_path = files.join(raw_dir, "VOCtrainval_06-Nov-2007.tar")
    if not files.exists(tar_path):
        print(f"Downloading VOCtrainval_06-Nov-2007.tar -> {raw_dir}...")
        files.download("https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar", raw_dir, overwrite=False, verbose=False)

    print(f"Extracting {tar_path}...")
    archive.extract(tar_path, raw_dir, overwrite=True, verbose=False)

    voc_root = files.join(raw_dir, "VOCdevkit", "VOC2007")
    val_txt = files.join(voc_root, "ImageSets", "Main", "val.txt")
    val_ids = [line.strip() for line in files.load(val_txt, as_lines=True, verbose=False) if line.strip()]

    datasets.convert(
        source=voc_root,
        target=target_path,
        src_format="voc",
        classes=VOC_CLASSES,
        image_ids=val_ids,
        include_difficult=True,
        overwrite=True,
        verbose=False,
    )
    print(f"VOC2007 prepared at {target_path} ({len(val_ids)} images).")
    return target_path


def prepare_val2017(target_dir: str | Path) -> Path:
    """Download COCO val2017 and annotations and convert to flat YOLO format."""
    target_path = files.path(target_dir)
    img_dir = files.join(target_path, "images")
    yaml_file = files.join(target_path, "data.yaml")
    if files.exists(yaml_file) and files.exists(img_dir):
        count = len(files.find(img_dir, "*.jpg"))
        if count >= 4500:
            print(f"val2017 already prepared at {target_path} ({count} images).")
            return target_path

    raw_dir = files.join(files.parent(target_path), "coco_raw")
    files.mkdir(raw_dir)
    val_zip = files.join(raw_dir, "val2017.zip")
    ann_zip = files.join(raw_dir, "annotations_trainval2017.zip")

    if not files.exists(val_zip):
        print(f"Downloading val2017.zip -> {raw_dir}...")
        files.download("http://images.cocodataset.org/zips/val2017.zip", raw_dir, overwrite=False, verbose=False)
    if not files.exists(ann_zip):
        print(f"Downloading annotations_trainval2017.zip -> {raw_dir}...")
        files.download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", raw_dir, overwrite=False, verbose=False)

    print("Extracting COCO val2017 zips...")
    archive.extract(val_zip, raw_dir, overwrite=True, verbose=False)
    archive.extract(ann_zip, raw_dir, overwrite=True, verbose=False)

    ann_json = files.join(raw_dir, "annotations", "instances_val2017.json")
    src_images_dir = files.join(raw_dir, "val2017")

    datasets.convert(
        images=src_images_dir,
        annotations=ann_json,
        target=target_path,
        src_format="coco",
        include_crowd=False,
        overwrite=True,
        verbose=False,
    )
    print(f"val2017 prepared at {target_path}.")
    return target_path


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/content")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    work = files.path(args.workdir)
    res_dir = files.join(work, "results")
    mining_dir = files.join(res_dir, "mining")
    metrics_dir = files.join(res_dir, "metrics")
    files.mkdir(mining_dir)
    files.mkdir(metrics_dir)

    # 1. Prepare VOC2007 and val2017
    voc_dir = prepare_voc2007(files.join(work, "datasets", "voc2007_yolo"))
    val_dir = prepare_val2017(files.join(work, "datasets", "val2017_yolo"))

    # 2. Run Mining on 3 datasets
    print("\n================== [1/2] RUNNING MINING BENCHMARK ==================")
    run_cmd([
        sys.executable, "benchmarks/mining/run_mining_benchmark.py",
        "--dataset", "coco128",
        "--budget", "40",
        "--epochs", "15",
        "--batch", "32",
        "--imgsz", "640",
        "--output", str(files.join(mining_dir, "coco128")),
    ])
    run_cmd([
        sys.executable, "benchmarks/mining/run_mining_benchmark.py",
        "--dataset-dir", str(voc_dir),
        "--max-pool", "1000",
        "--max-val", "300",
        "--budget", "350",
        "--epochs", "12",
        "--batch", "32",
        "--imgsz", "640",
        "--output", str(files.join(mining_dir, "voc2007")),
    ])
    run_cmd([
        sys.executable, "benchmarks/mining/run_mining_benchmark.py",
        "--dataset-dir", str(val_dir),
        "--max-pool", "1200",
        "--max-val", "300",
        "--budget", "420",
        "--epochs", "12",
        "--batch", "32",
        "--imgsz", "640",
        "--output", str(files.join(mining_dir, "val2017")),
    ])

    # 3. Run Metrics on 3 datasets
    print("\n================== [2/2] RUNNING METRICS BENCHMARK ==================")
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset", "coco128",
        "--output", str(files.join(metrics_dir, "coco128")),
    ])
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset-dir", str(voc_dir),
        "--max-images", "500",
        "--output", str(files.join(metrics_dir, "voc2007")),
    ])
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset-dir", str(val_dir),
        "--max-images", "500",
        "--output", str(files.join(metrics_dir, "val2017")),
    ])

    # 4. Print Outputs for caller extraction
    outputs_to_print = [
        ("MINING_COCO128", files.join(mining_dir, "coco128", "summary.json")),
        ("MINING_VOC2007", files.join(mining_dir, "voc2007", "summary.json")),
        ("MINING_VAL2017", files.join(mining_dir, "val2017", "summary.json")),
        ("METRICS_COCO128", files.join(metrics_dir, "coco128", "summary.json")),
        ("METRICS_VOC2007", files.join(metrics_dir, "voc2007", "summary.json")),
        ("METRICS_VAL2017", files.join(metrics_dir, "val2017", "summary.json")),
    ]

    for tag, json_file in outputs_to_print:
        print(f"\n==={tag}_START===")
        if files.exists(json_file):
            data = files.load(json_file, verbose=False)
            print(json.dumps(data, separators=(",", ":")))
        else:
            print(json.dumps({"error": f"File {json_file} does not exist"}))
        print(f"==={tag}_END===")

    print("\n===ALL_BENCHMARKS_SUCCESS===")


if __name__ == "__main__":
    main()

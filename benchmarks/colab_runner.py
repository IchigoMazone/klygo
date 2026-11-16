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
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
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


def _download_file(url: str, dest_path: Path) -> Path:
    if dest_path.exists():
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    return dest_path


def prepare_voc2007(target_dir: Path) -> Path:
    """Download Pascal VOC 2007 and convert val split into flat YOLO format."""
    if (target_dir / "data.yaml").exists() and (target_dir / "images").exists():
        count = len(list((target_dir / "images").glob("*.jpg")))
        if count >= 2000:
            print(f"VOC2007 already prepared at {target_dir} ({count} images).")
            return target_dir

    raw_dir = target_dir.parent / "voc_raw"
    tar_path = raw_dir / "VOCtrainval_06-Nov-2007.tar"
    url = "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar"
    _download_file(url, tar_path)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(raw_dir)

    voc_root = raw_dir / "VOCdevkit" / "VOC2007"
    val_txt = voc_root / "ImageSets" / "Main" / "val.txt"
    val_ids = [line.strip() for line in val_txt.read_text().splitlines() if line.strip()]

    img_out = target_dir / "images"
    lbl_out = target_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    cls_to_id = {name: idx for idx, name in enumerate(VOC_CLASSES)}

    for img_id in val_ids:
        src_img = voc_root / "JPEGImages" / f"{img_id}.jpg"
        src_xml = voc_root / "Annotations" / f"{img_id}.xml"
        if not src_img.exists() or not src_xml.exists():
            continue

        tree = ET.parse(src_xml)
        root = tree.getroot()
        size = root.find("size")
        if size is None:
            continue
        w = float(size.find("width").text)
        h = float(size.find("height").text)
        if w <= 0 or h <= 0:
            continue

        lines = []
        for obj in root.findall("object"):
            cname = obj.find("name").text
            if cname not in cls_to_id:
                continue
            cid = cls_to_id[cname]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            xc = ((xmin + xmax) / 2.0) / w
            yc = ((ymin + ymax) / 2.0) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        import shutil
        shutil.copy2(src_img, img_out / f"{img_id}.jpg")
        (lbl_out / f"{img_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(VOC_CLASSES),
        "names": {i: name for i, name in enumerate(VOC_CLASSES)},
    }
    import yaml
    (target_dir / "data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")
    print(f"VOC2007 prepared at {target_dir} ({len(val_ids)} images).")
    return target_dir


def prepare_val2017(target_dir: Path) -> Path:
    """Download COCO val2017 and annotations and convert to flat YOLO format."""
    if (target_dir / "data.yaml").exists() and (target_dir / "images").exists():
        count = len(list((target_dir / "images").glob("*.jpg")))
        if count >= 4500:
            print(f"val2017 already prepared at {target_dir} ({count} images).")
            return target_dir

    raw_dir = target_dir.parent / "coco_raw"
    val_zip = raw_dir / "val2017.zip"
    ann_zip = raw_dir / "annotations_trainval2017.zip"

    _download_file("http://images.cocodataset.org/zips/val2017.zip", val_zip)
    _download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", ann_zip)

    print("Extracting COCO val2017 zips...")
    with zipfile.ZipFile(val_zip) as z:
        z.extractall(raw_dir)
    with zipfile.ZipFile(ann_zip) as z:
        z.extractall(raw_dir)

    ann_json = raw_dir / "annotations" / "instances_val2017.json"
    with open(ann_json, encoding="utf-8") as f:
        coco_data = json.load(f)

    # COCO category id to 0..79 index
    cat_ids = sorted(c["id"] for c in coco_data["categories"])
    cat_to_yolo = {cat_id: yolo_idx for yolo_idx, cat_id in enumerate(cat_ids)}

    img_map = {im["id"]: im for im in coco_data["images"]}
    ann_by_img: dict[int, list[Any]] = {}
    for ann in coco_data["annotations"]:
        if ann.get("iscrowd"):
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    img_out = target_dir / "images"
    lbl_out = target_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    import shutil
    src_images_dir = raw_dir / "val2017"

    for img_id, im_info in img_map.items():
        fname = im_info["file_name"]
        src_path = src_images_dir / fname
        if not src_path.exists():
            continue

        w = float(im_info["width"])
        h = float(im_info["height"])
        lines = []
        for ann in ann_by_img.get(img_id, []):
            cid = cat_to_yolo[ann["category_id"]]
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            rw = bw / w
            rh = bh / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {rw:.6f} {rh:.6f}")

        shutil.copy2(src_path, img_out / fname)
        lbl_fname = Path(fname).with_suffix(".txt").name
        (lbl_out / lbl_fname).write_text("\n".join(lines), encoding="utf-8")

    data_yaml = {
        "path": str(target_dir.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(COCO80_NAMES),
        "names": COCO80_NAMES,
    }
    import yaml
    (target_dir / "data.yaml").write_text(yaml.dump(data_yaml), encoding="utf-8")
    print(f"val2017 prepared at {target_dir} ({len(img_map)} images).")
    return target_dir


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/content")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    work = Path(args.workdir)
    res_dir = work / "results"
    mining_dir = res_dir / "mining"
    metrics_dir = res_dir / "metrics"
    mining_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare VOC2007 and val2017
    voc_dir = prepare_voc2007(work / "datasets" / "voc2007_yolo")
    val_dir = prepare_val2017(work / "datasets" / "val2017_yolo")

    # 2. Run Mining on 3 datasets
    print("\n================== [1/2] RUNNING MINING BENCHMARK ==================")
    run_cmd([
        sys.executable, "benchmarks/mining/run_mining_benchmark.py",
        "--dataset", "coco128",
        "--budget", "40",
        "--epochs", "15",
        "--batch", "32",
        "--imgsz", "640",
        "--output", str(mining_dir / "coco128"),
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
        "--output", str(mining_dir / "voc2007"),
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
        "--output", str(mining_dir / "val2017"),
    ])

    # 3. Run Metrics on 3 datasets
    print("\n================== [2/2] RUNNING METRICS BENCHMARK ==================")
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset", "coco128",
        "--output", str(metrics_dir / "coco128"),
    ])
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset-dir", str(voc_dir),
        "--max-images", "500",
        "--output", str(metrics_dir / "voc2007"),
    ])
    run_cmd([
        sys.executable, "benchmarks/metrics/run_metrics_benchmark.py",
        "--dataset-dir", str(val_dir),
        "--max-images", "500",
        "--output", str(metrics_dir / "val2017"),
    ])

    # 4. Print Outputs for caller extraction
    outputs_to_print = [
        ("MINING_COCO128", mining_dir / "coco128" / "summary.json"),
        ("MINING_VOC2007", mining_dir / "voc2007" / "summary.json"),
        ("MINING_VAL2017", mining_dir / "val2017" / "summary.json"),
        ("METRICS_COCO128", metrics_dir / "coco128" / "summary.json"),
        ("METRICS_VOC2007", metrics_dir / "voc2007" / "summary.json"),
        ("METRICS_VAL2017", metrics_dir / "val2017" / "summary.json"),
    ]

    for tag, json_file in outputs_to_print:
        print(f"\n==={tag}_START===")
        if json_file.exists():
            data = json.loads(json_file.read_text(encoding="utf-8"))
            print(json.dumps(data, separators=(",", ":")))
        else:
            print(json.dumps({"error": f"File {json_file} does not exist"}))
        print(f"==={tag}_END===")

    print("\n===ALL_BENCHMARKS_SUCCESS===")


if __name__ == "__main__":
    main()

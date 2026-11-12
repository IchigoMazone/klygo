"""Pilot real detectors through Klygo on a frozen COCO sample; not a final benchmark."""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.metadata
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

from prepare_coco import ROOT, DATA, download, sha256, write_json

ARTIFACTS = ROOT / "klygo/artifacts/report"
WEIGHTS = ARTIFACTS / "weights"
(ARTIFACTS / "ultralytics").mkdir(parents=True, exist_ok=True)
(ARTIFACTS / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ARTIFACTS / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib"))
os.environ.setdefault("HF_HOME", str(ARTIFACTS / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(WEIGHTS / "torch"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT / "klygo"))

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.transforms.functional import pil_to_tensor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from klygo import files, models, outputs

URLS = {
    "yolo": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "world": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
    "faster": FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1.url,
}


def checkpoint_path(name):
    return (WEIGHTS / "torch/hub/checkpoints" if name == "faster" else WEIGHTS) / URLS[name].rsplit("/", 1)[-1]


class FasterDetector(models.Detector):
    """Use the public Klygo detector/backend contract for a Torchvision model."""
    def __init__(self, device):
        super().__init__(metadata={
            "model_id": "fasterrcnn_mobilenet_v3_large_fpn.COCO_V1",
            "details": {"name": "Faster R-CNN MobileNetV3 Large FPN", "backend": "Torchvision"},
            "flags": {"model": False, "post": True}, "model": {},
            "post": {"threshold": 0.001, "iou": 0.7}, "priority": {"post": ["threshold", "iou"]},
        }, backend="Torchvision")
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights, min_size=640, max_size=640,
            box_score_thresh=0.001, box_nms_thresh=0.7, box_detections_per_img=100,
        ).to(device).eval()
        self.names = dict(enumerate(weights.meta["categories"]))

    def forward(self, images, prompt=None):
        tensors = self.cast_inputs([pil_to_tensor(image).float().div(255) for image in images])
        formatted = self.format_results(self.run_inference(tensors))
        for item in formatted:
            item["labels"] = [self.names[index] for index in item["class_ids"]]
        return outputs.build_detections(images, formatted, names=self.names)


def load_detector(name, device):
    if not checkpoint_path(name).is_file():
        raise FileNotFoundError(f"Missing checkpoint: run --prepare-weights for {name}")
    if name == "faster":
        return FasterDetector(device)
    registry = files.load(ROOT / "klygo/klygo/models/models.json", verbose=False)
    metadata = dict(registry["yolov8*-world*.pt" if name == "world" else "yolov8?.pt"])
    metadata.pop("option", None)
    metadata["model_id"] = str(checkpoint_path(name))
    metadata["model"] = {"device": device, "imgsz": 640, "half": False, "max_det": 100}
    metadata["post"] = {"threshold": 0.001, "iou": 0.7}
    return models.load(metadata)


def environment():
    versions = {}
    for package in ("klygo", "torch", "torchvision", "ultralytics", "numpy", "pillow", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "versions": versions, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "klygo"), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(ROOT / "klygo"), "status", "--short"], text=True),
        "script_sha256": sha256(__file__),
    }


def evaluate(ground_truth_path, predictions, image_ids, diagnostic_dir=None):
    ground_truth = COCO(str(ground_truth_path))
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {**ground_truth.dataset, "annotations": []}
        detections.createIndex()
    evaluation = COCOeval(ground_truth, detections, "bbox")
    evaluation.params.imgIds = sorted(image_ids)
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    tp = fp = gt_count = 0
    image_stats = {str(image_id): {"tp": 0, "fp": 0, "gt": 0} for image_id in image_ids}
    for item in evaluation.evalImgs:
        if item is None or list(item["aRng"]) != list(evaluation.params.areaRng[0]):
            continue
        keep = np.asarray(item["dtScores"]) >= 0.25
        matched = np.asarray(item["dtMatches"])[0] > 0
        ignored = np.asarray(item["dtIgnore"])[0].astype(bool)
        tp += int(np.sum(keep & matched & ~ignored))
        fp += int(np.sum(keep & ~matched & ~ignored))
        gt_count += int(np.sum(~np.asarray(item["gtIgnore"]).astype(bool)))
        record = image_stats[str(item["image_id"])]
        record["tp"] += int(np.sum(keep & matched & ~ignored))
        record["fp"] += int(np.sum(keep & ~matched & ~ignored))
        record["gt"] += int(np.sum(~np.asarray(item["gtIgnore"]).astype(bool)))
    fn = gt_count - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt_count if gt_count else 0.0
    if diagnostic_dir is not None:
        precision_array = evaluation.eval["precision"][:, :, :, 0, 2]
        by_class = []
        for index, category_id in enumerate(evaluation.params.catIds):
            values = precision_array[:, :, index]
            valid = values[values >= 0]
            at_50 = precision_array[0, :, index]
            by_class.append({"category_id": int(category_id), "name": ground_truth.cats[category_id]["name"],
                "AP": float(valid.mean()) if len(valid) else None,
                "AP50": float(at_50[at_50 >= 0].mean()) if np.any(at_50 >= 0) else None})
        macro_pr = []
        for recall_index in range(len(evaluation.params.recThrs)):
            values = precision_array[0, recall_index, :]
            macro_pr.append(float(values[values >= 0].mean()) if np.any(values >= 0) else None)
        write_json(Path(diagnostic_dir) / "diagnostics.json", {"classes": by_class,
            "recall_thresholds": evaluation.params.recThrs.tolist(), "macro_precision_at_iou50": macro_pr,
            "per_image_at_conf025": image_stats})
    return {
        "coco_stats": evaluation.stats.tolist(), "AP": float(evaluation.stats[0]),
        "AP50": float(evaluation.stats[1]), "AP75": float(evaluation.stats[2]),
        "operating_point": {"confidence": 0.25, "iou": 0.5, "aggregation": "micro; COCO ignore/crowd rules",
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0},
    }


def run(name, limit, device, phase="pilot"):
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    image_ids = manifest["image_ids_in_sample_order"][:limit]
    source = json.loads((DATA / "instances_sample.json").read_text(encoding="utf-8"))
    images = {item["id"]: item for item in source["images"]}
    categories = {item["name"]: item["id"] for item in source["categories"]}
    folder = ARTIFACTS / phase / name
    folder.mkdir(parents=True, exist_ok=True)
    # A changed model config or image count gets a new folder rather than overwriting evidence.
    if (folder / "metrics.json").exists():
        raise FileExistsError(f"Pilot evidence exists: {folder}")
    write_json(folder / "environment.json", environment())
    torch.manual_seed(manifest["seed"])
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    torch.backends.cudnn.benchmark = False
    detector = load_detector(name, device)
    write_json(folder / "configuration.json", {
        "phase": phase, "model": name, "checkpoint_url": URLS[name],
        "checkpoint_sha256": sha256(checkpoint_path(name)), "image_ids": image_ids,
        "device": device, "precision": "FP32", "batch": 1, "warmup": 5,
        "imgsz_or_min_max_size": 640, "confidence_for_AP": 0.001, "nms_iou": 0.7,
        "max_detections": 100, "vocabulary": detector.names,
        "manifest_sha256": sha256(DATA / "manifest.json"),
        "timing_scope": "prepared RGB PIL -> Klygo.predict -> standardized results; CUDA synchronized",
    })
    predictions, timings = [], []
    try:
        for number, image_id in enumerate(image_ids):
            path = DATA / "images" / images[image_id]["file_name"]
            if sha256(path) != manifest["image_sha256"][str(image_id)]:
                raise ValueError(f"Image checksum mismatch: {image_id}")
            with Image.open(path) as original:
                image = original.convert("RGB")
            if image.size != (images[image_id]["width"], images[image_id]["height"]):
                raise ValueError(f"Image size mismatch: {image_id}")
            if number == 0:
                for _ in range(5):
                    detector.predict(image, verbose=False)
                if device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = time.perf_counter()
            frame = detector.predict(image, verbose=False)[0]
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            milliseconds = (time.perf_counter() - begin) * 1000
            timings.append({"image_id": image_id, "milliseconds": milliseconds})
            for box in frame.objects:
                if box.label not in categories:
                    raise ValueError(f"Unmapped class: {box.label}")
                x1, y1, x2, y2 = map(float, box.box)
                if not np.isfinite([x1, y1, x2, y2, box.score]).all() or x2 < x1 or y2 < y1:
                    raise ValueError("Invalid output box")
                predictions.append({"image_id": image_id, "category_id": categories[box.label],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(box.score)})
            print(f"{name} {number + 1}/{len(image_ids)}: {len(frame.objects)} boxes, {milliseconds:.1f} ms", flush=True)
        write_json(folder / "predictions.json", predictions)
        write_json(folder / "latencies.json", timings)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            metrics = evaluate(DATA / "instances_sample.json", predictions, image_ids, folder)
        (folder / "cocoeval.txt").write_text(buffer.getvalue(), encoding="utf-8")
        metrics.update({"phase": "pilot_not_final_benchmark" if phase == "pilot" else phase, "image_count": len(image_ids),
            "mean_latency_ms": float(np.mean([item["milliseconds"] for item in timings])),
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None})
        write_json(folder / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    finally:
        detector.unload()
        del detector
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(URLS), default=list(URLS))
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-weights", action="store_true")
    args = parser.parse_args()
    if args.prepare_weights:
        for name in args.models:
            download(URLS[name], checkpoint_path(name))
    else:
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; do not silently change the experiment device")
        for name in args.models:
            run(name, args.limit, args.device)

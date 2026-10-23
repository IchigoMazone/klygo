"""Comprehensive 4-backend object detection workflow evaluating real local models against ground-truth COCO15.

Exercises all 4 standard Klygo backends:
1. Ultralytics (YOLO)
2. Ultralytics (YOLO-World)
3. Torchvision (SSDLite MobileNetV3)
4. Keras (Keras 3 with PyTorch engine)

Uses exclusively public Klygo APIs:
files, datasets, media, preprocessing, models, outputs, postprocessing, visual, and utils.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import torch
from rich.console import Console
from rich.table import Table

# Configure Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers

from klygo import (
    config,
    datasets,
    files,
    media,
    models,
    outputs,
    postprocessing,
    preprocessing,
    utils,
    visual,
)
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Box, Detection, Detections


# =============================================================================
# 1. LOCAL DETECTOR CLASSES FOR THE 4 BACKENDS
# =============================================================================

class LocalYOLODetector(Detector):
    """Local YOLO Detector wrapping an authentic Ultralytics model."""

    def __init__(self, model_id: str = "yolov8n.pt", device: str | None = None) -> None:
        metadata = {
            "model_id": f"local-yolo-{model_id}",
            "details": {
                "name": "Local YOLO Detector",
                "task": "Object-Detection",
                "backend": "Ultralytics",
                "library": "ultralytics",
            },
            "flags": {"model": False, "post": True},
            "model": {},
            "priority": {"post": ["threshold", "iou"]},
            "post": {"threshold": 0.25, "iou": 0.5},
        }
        super().__init__(metadata=metadata, backend="Ultralytics")
        from ultralytics import YOLO

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_id)
        self.names = dict(self.model.names)

    def forward(
        self,
        images: List[Any],
        prompt: List[str] | None = None,
        **kwargs,
    ) -> List[Detection]:
        post_config = dict(self.metadata.get("post", {}))
        conf_thresh = post_config.get("threshold", 0.25)
        iou_thresh = post_config.get("iou", 0.5)

        classes = None
        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            inverted = utils.invert(self.names)
            classes = [inverted[name] for name in requested if name in inverted]

        raw_results = self.model(
            images,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            device=self.device_name,
            verbose=False,
        )
        formatted = self.format_results(raw_results)
        return outputs.build_detections(images, formatted, names=self.names)


class LocalYOLOWorldDetector(Detector):
    """Local YOLO-World Detector for zero-shot prompt-based detection."""

    def __init__(self, model_id: str = "yolov8s-worldv2.pt", device: str | None = None) -> None:
        metadata = {
            "model_id": f"local-yoloworld-{model_id}",
            "details": {
                "name": "Local YOLO-World Detector",
                "task": "Object-Detection",
                "backend": "Ultralytics",
                "library": "ultralytics",
            },
            "flags": {"model": False, "post": True},
            "model": {},
            "priority": {"post": ["threshold", "iou"]},
            "post": {"threshold": 0.25, "iou": 0.5},
        }
        super().__init__(metadata=metadata, backend="Ultralytics")
        from ultralytics import YOLOWorld

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLOWorld(model_id)
        self.names = dict(self.model.names)

    def forward(
        self,
        images: List[Any],
        prompt: List[str] | None = None,
        **kwargs,
    ) -> List[Detection]:
        classes = None
        if prompt:
            prompts = [prompt] if isinstance(prompt, str) else list(prompt)
            clean_prompts = [str(p).strip() for p in prompts if str(p).strip()]
            inverted = utils.invert(self.names)
            if all(p in inverted for p in clean_prompts):
                classes = [inverted[p] for p in clean_prompts]
            else:
                self.model.set_classes(clean_prompts)
                self.names = {i: name for i, name in enumerate(clean_prompts)}

        post_config = dict(self.metadata.get("post", {}))
        conf_thresh = post_config.get("threshold", 0.25)
        iou_thresh = post_config.get("iou", 0.5)

        raw_results = self.model(
            images,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=classes,
            device=self.device_name,
            verbose=False,
        )
        formatted = self.format_results(raw_results)
        return outputs.build_detections(images, formatted, names=self.names)


class LocalTorchvisionDetector(Detector):
    """Local Torchvision Detector wrapping a pretrained Faster R-CNN FPN model."""

    def __init__(self, categories: List[str], device: str | None = None) -> None:
        metadata = {
            "model_id": "local-torchvision-fasterrcnn",
            "details": {
                "name": "Local Torchvision Faster R-CNN Detector",
                "task": "Object-Detection",
                "backend": "Torchvision",
                "library": "torchvision",
            },
            "flags": {"model": False, "post": True},
            "model": {},
            "priority": {"post": ["threshold"]},
            "post": {"threshold": 0.25},
        }
        super().__init__(metadata=metadata, backend="Torchvision")
        import torchvision.models.detection as tv_det

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = tv_det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = tv_det.fasterrcnn_mobilenet_v3_large_fpn(weights=weights).to(self.device_name)
        self.model.eval()
        self.categories = {i: name for i, name in enumerate(categories)}
        self.names = dict(self.categories)

    def forward(
        self,
        images: List[Any],
        prompt: List[str] | None = None,
        **kwargs,
    ) -> List[Detection]:
        tensors = self.cast_inputs([media.to_tensor(img) for img in images])
        raw_outputs = self.run_inference(tensors)
        formatted = self.format_results(raw_outputs)

        for item in formatted:
            item["labels"] = [
                self.categories.get(cid, str(cid)) for cid in item["class_ids"]
            ]

        results = outputs.build_detections(images, formatted, names=self.names)
        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            return [postprocessing.select(det, labels=requested) for det in results]
        return results


class LocalKerasDetector(Detector):
    """Local KerasHub RT-DETR (D-FINE) Detector with PyTorch compute engine."""

    def __init__(
        self,
        categories: Dict[int, str],
        preset: str = "dfine_nano_coco",
        device: str | None = None,
    ) -> None:
        metadata = {
            "model_id": f"local-kerashub-{preset}",
            "details": {
                "name": "Local KerasHub RT-DETR (D-FINE) Detector",
                "task": "Object-Detection",
                "backend": "Keras",
                "library": "keras_hub",
            },
            "flags": {"model": False, "post": True},
            "model": {},
            "priority": {"post": ["threshold"]},
            "post": {"threshold": 0.25},
        }
        super().__init__(metadata=metadata, backend="Keras")
        import keras_hub

        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = keras_hub.models.ObjectDetector.from_preset(preset)
        self.categories = dict(categories)
        self.names = dict(self.categories)

    def forward(
        self,
        images: List[Any],
        prompt: List[str] | None = None,
        **kwargs,
    ) -> List[Detection]:
        raw_list = []
        for img in images:
            inputs_tensor = self.cast_inputs([img])
            raw_outputs = self.run_inference(inputs_tensor)
            formatted = self.format_results(raw_outputs)
            for item in formatted:
                item["labels"] = [
                    self.categories.get(cid, str(cid)) for cid in item["class_ids"]
                ]
                raw_list.append(item)

        results = outputs.build_detections(images, raw_list, names=self.names)
        if prompt:
            requested = [prompt] if isinstance(prompt, str) else list(prompt)
            return [postprocessing.select(det, labels=requested) for det in results]
        return results


# =============================================================================
# 2. EVALUATION & BENCHMARKING ENGINE
# =============================================================================

def evaluate_model_on_dataset(
    model: Detector,
    model_name: str,
    dataset_frames: List[Detection],
    assigned_classes: List[str],
    preview_dir: str,
    iou_thresh: float = 0.50,
) -> Dict[str, Any]:
    """Evaluate a single Detector on positive and negative images for assigned classes."""
    frames_by_id = {files.stem(f.config["relative_path"]): f for f in dataset_frames}
    class_stats = {
        cls: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        for cls in assigned_classes
    }

    latencies = []

    for label in assigned_classes:
        # Select 1 representative positive image (with the clearest foreground object) and 1 negative image
        pos_id = max(
            (k for k, f in frames_by_id.items() if any(b.label == label for b in f.objects)),
            key=lambda k: max(
                (b.box[2] - b.box[0]) * (b.box[3] - b.box[1])
                for b in frames_by_id[k].objects
                if b.label == label
            ),
            default=None,
        )
        neg_id = next(
            (k for k, f in sorted(frames_by_id.items()) if all(b.label != label for b in f.objects)),
            None,
        )

        test_ids = [i for i in (pos_id, neg_id) if i is not None]

        for image_id in test_ids:
            gt_frame = frames_by_id[image_id]
            gt_boxes = [b for b in gt_frame.objects if b.label == label]
            class_stats[label]["gt"] += len(gt_boxes)

            image = gt_frame.source_image
            t0 = time.perf_counter()
            pred_det = model.predict(image, prompt=[label], verbose=False)[0]
            latencies.append(time.perf_counter() - t0)

            # Apply postprocessing threshold and NMS
            pred_det = postprocessing.apply(
                pred_det,
                postprocessing.threshold(0.35),
                postprocessing.nms(0.50),
            )
            pred_boxes = [b for b in pred_det.objects if b.label == label]
            class_stats[label]["pred"] += len(pred_boxes)

            # Match detections against ground truth using box_iou
            unmatched_gt = set(range(len(gt_boxes)))
            tp = fp = 0

            for p_box in sorted(pred_boxes, key=lambda x: x.score, reverse=True):
                best_iou, best_idx = max(
                    ((postprocessing.box_iou(p_box, gt_boxes[i]), i) for i in unmatched_gt),
                    default=(0.0, -1),
                )
                if best_iou >= iou_thresh:
                    unmatched_gt.remove(best_idx)
                    tp += 1
                else:
                    fp += 1

            fn = len(unmatched_gt)
            class_stats[label]["tp"] += tp
            class_stats[label]["fp"] += fp
            class_stats[label]["fn"] += fn

            # Save annotated visualization preview with visual.draw_bboxes
            if pred_boxes:
                bboxes_coords = [b.box for b in pred_boxes]
                bboxes_labels = [b.label for b in pred_boxes]
                bboxes_scores = [b.score for b in pred_boxes]
                vis_img = visual.draw_bboxes(
                    image,
                    bboxes=bboxes_coords,
                    labels=bboxes_labels,
                    scores=bboxes_scores,
                )
                safe_name = model_name.replace("/", "-").replace(" ", "_")
                out_preview = files.join(preview_dir, f"{safe_name}_{label}_{image_id}.jpg")
                media.save(out_preview, vis_img, overwrite=True, verbose=False)

    # Compute overall metrics
    tot_gt = sum(s["gt"] for s in class_stats.values())
    tot_pred = sum(s["pred"] for s in class_stats.values())
    tot_tp = sum(s["tp"] for s in class_stats.values())
    tot_fp = sum(s["fp"] for s in class_stats.values())
    tot_fn = sum(s["fn"] for s in class_stats.values())

    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

    return {
        "model_name": model_name,
        "backend": model.backend,
        "classes": class_stats,
        "total_gt": tot_gt,
        "total_pred": tot_pred,
        "tp": tot_tp,
        "fp": tot_fp,
        "fn": tot_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
    }


def run_four_backend_evaluation(
    dataset_source: str = "artifacts/integration/coco15",
    artifact_root: str = "artifacts/integration/four-backends-eval",
) -> None:
    """Execute the full 4-backend integration test against the 15 COCO classes."""
    console = Console()
    dataset_path = files.path(dataset_source)
    artifact_path = files.path(artifact_root)
    preview_dir = files.join(artifact_path, "preview")
    report_dir = files.join(artifact_path, "report")

    files.mkdir(preview_dir, parents=True, exist_ok=True)
    files.mkdir(report_dir, parents=True, exist_ok=True)

    console.print("\n[bold cyan]=== Klygo 4-Backend Integration Workflow ===[/bold cyan]")
    console.print(f"[green]Dataset Source:[/green] {dataset_path}")
    console.print(f"[green]Artifact Root:[/green]  {artifact_path}")

    # 1. Inspect dataset structure with datasets.info
    info = datasets.info(dataset_path, format="yolo", verbose=False)
    yaml_config = config.load(files.join(dataset_path, "data.yaml"), verbose=False)
    all_15_classes = list(yaml_config.names)
    console.print(f"[yellow]Loaded {len(all_15_classes)} classes from COCO15 dataset:[/yellow] {', '.join(all_15_classes)}")

    # 2. Load ground-truth annotations with datasets.load
    dataset_frames = datasets.load(dataset_path, format="yolo", verbose=False)
    console.print(f"[green]Loaded {len(dataset_frames)} ground-truth frames with annotations.[/green]")

    # 3. Class allocation across the 4 backends (4 / 4 / 4 / 3 = 15)
    class_allocation = {
        "YOLO (Ultralytics)": ["person", "car", "knife", "chair"],
        "YOLO-World (Ultralytics)": ["umbrella", "handbag", "bottle", "cup"],
        "Faster R-CNN (Torchvision)": ["wine glass", "spoon", "bowl", "book"],
        "RT-DETR / D-FINE (KerasHub)": ["elephant", "traffic light", "teddy bear"],
    }

    # 4. Instantiate the 4 authentic local Detector models
    console.print("\n[bold]Instantiating the 4 Local Detector Models...[/bold]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim]Hardware device: {device}[/dim]")

    # Load official COCO categories for Torchvision Faster R-CNN
    import torchvision.models.detection as tv_det
    tv_weights = tv_det.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
    coco_91_categories = tv_weights.meta["categories"]

    yolo_det = LocalYOLODetector("yolov8n.pt", device=device)
    coco_80_categories = dict(yolo_det.names)

    models_suite: Dict[str, Tuple[Detector, List[str]]] = {
        "YOLO (Ultralytics)": (yolo_det, class_allocation["YOLO (Ultralytics)"]),
        "YOLO-World (Ultralytics)": (LocalYOLOWorldDetector("yolov8s-worldv2.pt", device=device), class_allocation["YOLO-World (Ultralytics)"]),
        "Faster R-CNN (Torchvision)": (LocalTorchvisionDetector(categories=coco_91_categories, device=device), class_allocation["Faster R-CNN (Torchvision)"]),
        "RT-DETR / D-FINE (KerasHub)": (LocalKerasDetector(categories=coco_80_categories, preset="dfine_small_coco", device=device), class_allocation["RT-DETR / D-FINE (KerasHub)"]),
    }

    # 5. Run evaluation and compile results
    results_list = []
    for model_name, (detector_instance, assigned_cls) in models_suite.items():
        console.print(f"\n[cyan]Evaluating {model_name} on assigned classes: {assigned_cls}...[/cyan]")
        res = evaluate_model_on_dataset(
            model=detector_instance,
            model_name=model_name,
            dataset_frames=dataset_frames,
            assigned_classes=assigned_cls,
            preview_dir=preview_dir,
            iou_thresh=0.50,
        )
        results_list.append(res)

        # Print per-class metrics table
        cls_table = Table(title=f"Detailed Results: {model_name} (Backend: {res['backend']})")
        cls_table.add_column("Class", style="cyan")
        cls_table.add_column("GT Boxes", justify="right")
        cls_table.add_column("Pred Boxes", justify="right")
        cls_table.add_column("TP", justify="right", style="green")
        cls_table.add_column("FP", justify="right", style="red")
        cls_table.add_column("FN", justify="right", style="yellow")
        cls_table.add_column("Precision", justify="right")
        cls_table.add_column("Recall", justify="right")

        for c_name, c_data in res["classes"].items():
            p = c_data["tp"] / (c_data["tp"] + c_data["fp"]) if (c_data["tp"] + c_data["fp"]) > 0 else 0.0
            r = c_data["tp"] / (c_data["tp"] + c_data["fn"]) if (c_data["tp"] + c_data["fn"]) > 0 else 0.0
            cls_table.add_row(
                c_name,
                str(c_data["gt"]),
                str(c_data["pred"]),
                str(c_data["tp"]),
                str(c_data["fp"]),
                str(c_data["fn"]),
                f"{p:.3f}",
                f"{r:.3f}",
            )
        console.print(cls_table)

    # 6. Consolidated cross-backend comparison table
    summary_table = Table(title="Cross-Backend Evaluation Summary (COCO15 Ground Truth, IoU >= 0.50)")
    summary_table.add_column("Model Architecture", style="bold cyan")
    summary_table.add_column("Backend", style="magenta")
    summary_table.add_column("Classes", justify="center")
    summary_table.add_column("GT", justify="right")
    summary_table.add_column("Pred", justify="right")
    summary_table.add_column("TP", justify="right", style="green")
    summary_table.add_column("FP", justify="right", style="red")
    summary_table.add_column("FN", justify="right", style="yellow")
    summary_table.add_column("Precision", justify="right", style="bold green")
    summary_table.add_column("Recall", justify="right", style="bold cyan")
    summary_table.add_column("F1-Score", justify="right", style="bold yellow")
    summary_table.add_column("Latency (ms)", justify="right")
    summary_table.add_column("Throughput (FPS)", justify="right")

    for r in results_list:
        summary_table.add_row(
            r["model_name"],
            r["backend"],
            str(len(r["classes"])),
            str(r["total_gt"]),
            str(r["total_pred"]),
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['latency_ms']}",
            f"{r['fps']}",
        )

    console.print("\n")
    console.print(summary_table)

    # 7. Save persistent evaluation report using files.save
    report_file = files.join(report_dir, "multi_backend_report.json")
    files.save(report_file, results_list, overwrite=True, verbose=False)
    console.print(f"\n[bold green][OK] Full evaluation report saved to:[/bold green] {report_file}")
    console.print(f"[bold green][OK] Annotated visual previews saved to:[/bold green] {preview_dir}")


if __name__ == "__main__":
    run_four_backend_evaluation()

"""End-to-end integration workflow evaluating a real detection model against ground-truth labels."""

from __future__ import annotations

import torch
from rich.console import Console
from rich.table import Table

from klygo import (
    datasets,
    files,
    media,
    models,
    postprocessing,
    visual,
)
from klygo.models.detection.base import Detector
from klygo.outputs.detect import Detection, Detections


class RealLocalDetector(Detector):
    """Authentic local Detector class built on Klygo's Detector base using real pretrained weights.

    Demonstrates how users wrap real detection architectures locally with automatic
    hardware acceleration (CUDA on Colab GPU or local CPU fallback).
    """

    def __init__(
        self,
        model_id: str = "grounding-dino-tiny",
        target_classes: tuple[str, ...] = ("person", "car", "knife", "umbrella", "handbag"),
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        metadata = {
            "model_id": f"klygo-real-detector-{model_id}",
            "details": {
                "name": "Real Local Detector",
                "task": "Object-Detection",
                "backend": "Hugging Face",
                "library": "transformers",
            },
            "flags": {"model": False},
            "model": {},
            "priority": {},
        }
        super().__init__(metadata=metadata, backend="Hugging Face")
        self.target_classes = list(target_classes)
        self.confidence = float(confidence)
        # Load real pretrained model from Klygo model registry
        self.base_detector = models.load(model_id)
        if self.device_name != "cpu" and hasattr(self.base_detector, "model"):
            self.base_detector.model.to(self.device_name)

    def forward(
        self,
        images: list,
        prompt: list[str] | None = None,
        **kwargs,
    ) -> Detections:
        prompts = prompt or self.target_classes
        all_detections: list[Detection] = []
        for img in images:
            preds = self.base_detector.predict(img, prompt=prompts, verbose=False)[0]
            filtered = preds.filter(
                lambda box: box.score >= self.confidence and box.label in self.target_classes
            )
            all_detections.append(filtered)
        return Detections(all_detections)


def evaluate_local_detector(
    dataset_source: str = "artifacts/integration/coco15",
    artifact_root: str = "artifacts/integration/local-eval",
    target_classes: tuple[str, ...] = ("person", "car", "knife", "umbrella", "handbag"),
    iou_threshold: float = 0.50,
) -> dict:
    """Run an end-to-end integration workflow evaluating RealLocalDetector against ground truth."""
    dataset_path = files.resolve(dataset_source)
    artifact_path = files.resolve(artifact_root)
    preview_dir = files.join(artifact_path, "preview")
    files.mkdir(preview_dir, parents=True, exist_ok=True)

    # 1. Inspect dataset structure with datasets.info
    info = datasets.info(dataset_path, format="yolo", verbose=False)

    # 2. Load ground-truth annotations with datasets.load
    gt_detections = datasets.load(dataset_path, format="yolo", verbose=False)

    # 3. Instantiate real Detector class with automatic device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = RealLocalDetector(
        model_id="grounding-dino-tiny",
        target_classes=target_classes,
        confidence=0.25,
        device=device,
    )

    # 4. Select representative real positive images for each target class
    selected_frames = []
    for label in target_classes:
        matching_frame = next(
            (f for f in gt_detections if any(obj.label == label for obj in f)),
            None,
        )
        if matching_frame and matching_frame not in selected_frames:
            selected_frames.append(matching_frame)

    # 5. Run real inference through the local detector
    input_images = [media.load(frame.source_path, verbose=False)[0] for frame in selected_frames]
    predictions = detector(input_images)

    # 6. Postprocessing: Apply Non-Maximum Suppression (NMS)
    predictions.transform(postprocessing.nms(iou=0.5))
    cleaned_predictions = predictions

    # 7. Evaluate predictions vs ground-truth annotations using IoU matching
    metrics_per_class = {
        cls: {"gt": 0, "pred": 0, "tp": 0, "fp": 0, "fn": 0}
        for cls in target_classes
    }

    for idx, (gt_frame, pred_frame) in enumerate(zip(selected_frames, cleaned_predictions)):
        for cls in target_classes:
            gt_boxes = [box for box in gt_frame if box.label == cls]
            pred_boxes = [box for box in pred_frame if box.label == cls]

            metrics_per_class[cls]["gt"] += len(gt_boxes)
            metrics_per_class[cls]["pred"] += len(pred_boxes)

            unmatched_gt = set(range(len(gt_boxes)))
            for p_box in pred_boxes:
                best_iou = 0.0
                best_idx = -1
                for g_idx in unmatched_gt:
                    iou = postprocessing.box_iou(p_box, gt_boxes[g_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = g_idx

                if best_iou >= iou_threshold and best_idx >= 0:
                    unmatched_gt.remove(best_idx)
                    metrics_per_class[cls]["tp"] += 1
                else:
                    metrics_per_class[cls]["fp"] += 1

            metrics_per_class[cls]["fn"] += len(unmatched_gt)

        # 8. Render visual previews with visual.draw_bboxes
        if pred_frame:
            overlay_boxes = [[box.xmin, box.ymin, box.xmax, box.ymax] for box in pred_frame]
            overlay_labels = [f"Pred: {box.label} ({box.score:.2f})" for box in pred_frame]
            rendered = visual.draw_bboxes(
                media.load(gt_frame.source_path, verbose=False)[0],
                overlay_boxes,
                labels=overlay_labels,
                thickness=2,
            )
            media.save(
                files.join(preview_dir, f"evaluation_frame_{idx}.jpg"),
                rendered,
                overwrite=True,
                verbose=False,
            )

    # 9. Format Rich Table
    total_gt = sum(m["gt"] for m in metrics_per_class.values())
    total_pred = sum(m["pred"] for m in metrics_per_class.values())
    total_tp = sum(m["tp"] for m in metrics_per_class.values())
    total_fp = sum(m["fp"] for m in metrics_per_class.values())
    total_fn = sum(m["fn"] for m in metrics_per_class.values())

    table = Table(
        title=f"Real Model Evaluation vs Ground Truth (COCO15, Device: {device.upper()}, IoU >= {iou_threshold})"
    )
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Ground Truth", justify="right")
    table.add_column("Predictions", justify="right")
    table.add_column("TP", justify="right", style="green")
    table.add_column("FP", justify="right", style="red")
    table.add_column("FN", justify="right", style="yellow")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1-Score", justify="right", style="bold green")

    summary_rows = []
    for cls, m in metrics_per_class.items():
        precision = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) > 0 else 0.0
        recall = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        summary_rows.append({
            "class": cls,
            "gt": m["gt"],
            "pred": m["pred"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })
        table.add_row(
            cls,
            str(m["gt"]),
            str(m["pred"]),
            str(m["tp"]),
            str(m["fp"]),
            str(m["fn"]),
            f"{precision:.3f}",
            f"{recall:.3f}",
            f"{f1:.3f}",
        )

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    table.add_section()
    table.add_row(
        "Total",
        str(total_gt),
        str(total_pred),
        str(total_tp),
        str(total_fp),
        str(total_fn),
        f"{overall_prec:.3f}",
        f"{overall_rec:.3f}",
        f"{overall_f1:.3f}",
        style="bold",
    )

    Console().print(table)

    report_payload = {
        "dataset": str(dataset_path),
        "device": device,
        "total_classes": info["nc"],
        "evaluated_classes": list(target_classes),
        "evaluated_frames": len(selected_frames),
        "iou_threshold": iou_threshold,
        "metrics_per_class": summary_rows,
        "overall": {
            "ground_truth": total_gt,
            "predictions": total_pred,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(overall_prec, 3),
            "recall": round(overall_rec, 3),
            "f1": round(overall_f1, 3),
        },
    }

    report_file = files.join(artifact_path, "report.json")
    files.save(report_file, report_payload, overwrite=True, verbose=False)
    print(f"Report saved to: {report_file}")
    print(f"Sample overlays saved to: {preview_dir}")
    return report_payload


if __name__ == "__main__":
    evaluate_local_detector()

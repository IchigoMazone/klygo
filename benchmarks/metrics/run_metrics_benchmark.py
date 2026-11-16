"""Standalone full-spectrum multi-model detection accuracy & diagnostic benchmark for klygo.metrics.

Evaluates predicted labels from all 9 detector models (`yolo`, `yolo11`, `yolo26`,
`yolo-world`, `grounding-dino`, `grounding-dino-base`, `faster-rcnn`, `d-fine`,
`locate-anything`) against ground-truth dataset annotations (`coco128`, `voc2007`,
`val2017`) using all 13 public functions of `klygo.metrics`:
    1. `map_score`         : `mAP@50-95`, `mAP@50`, `mAP@75`, per-class & per-IoU AP
    2. `precision_recall`  : `Precision`, `Recall`, `F1`, `mean_iou`, `TP`, `FP`, `FN`
    3. `size_breakdown`    : Small (`<32^2`), Medium (`32^2..96^2`), Large (`>96^2`) AP & F1
    4. `error_analysis`    : TIDE-style `localization`, `classification`, `duplicate`,
                             `classification_localization`, `background`, and `missed` counts
    5. `calibration`       : Expected Calibration Error (`ECE`) and confidence reliability
    6. `threshold_sweep`   : Optimal confidence threshold (`optimal_threshold`, `optimal_f1`)
    7. `iou_sweep`         : Strict localization curve (`F1@50`, `F1@75`, `F1@90`)
    8. `per_class`         : Active class detection coverage (`detected_classes / gt_classes`)
    9. `per_image`         : Mean per-image F1 and perfect-image ratio
   10. `confusion_matrix`  : Diagonal classification accuracy on matched boxes
   11. `match_boxes`       : Total matched TP pairs, unmatched FP, and unmatched FN boxes
   12. `compose` & `evaluate` : Single-pass composed execution + class-agnostic localization pass

Outputs the standardized benchmark file set under `--output`:
    - `environment.json`
    - `accuracy_leaderboard.json`
    - `size_and_error_breakdown.json`
    - `per_class_summary.json`
    - `summary.json`
    - `REPORT.md`
    - `charts/metrics_map_leaderboard.png`
    - `charts/metrics_size_breakdown.png`
    - `charts/metrics_error_calibration.png`
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

from benchmarks.gpu_suite.evaluate_release import _md_table, collect_environment
from klygo import archive, datasets, files, metrics, models, outputs, utils

ALL_MODELS = (
    "yolo",
    "yolo11",
    "yolo26",
    "yolo-world",
    "grounding-dino",
    "grounding-dino-base",
    "faster-rcnn",
    "d-fine",
    "locate-anything",
)

# Synonym mapping between COCO-80 labels (with spaces or underscores) and Pascal VOC-20 labels
COCO_TO_VOC_MAP = {
    "airplane": "aeroplane",
    "dining table": "diningtable",
    "dining_table": "diningtable",
    "motorcycle": "motorbike",
    "potted plant": "pottedplant",
    "potted_plant": "pottedplant",
    "couch": "sofa",
    "tv": "tvmonitor",
    "tv_monitor": "tvmonitor",
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _normalize_label(lbl: str, gt_label_set: set[str]) -> str:
    clean = str(lbl).strip().lower()
    if clean in gt_label_set:
        return clean
    underscored = clean.replace(" ", "_")
    if underscored in gt_label_set:
        return underscored
    spaced = clean.replace("_", " ")
    if spaced in gt_label_set:
        return spaced
    mapped = COCO_TO_VOC_MAP.get(clean) or COCO_TO_VOC_MAP.get(underscored) or COCO_TO_VOC_MAP.get(spaced)
    if mapped and mapped in gt_label_set:
        return mapped
    return clean


def _align_predictions_to_truth(
    preds: outputs.Detections,
    truth: outputs.Detections,
    *,
    filter_to_gt_classes: bool = True,
) -> outputs.Detections:
    gt_labels = {str(v).strip().lower() for v in truth.names.values()}
    for frame in truth:
        for b in frame:
            if b.label:
                gt_labels.add(str(b.label).strip().lower())

    label_to_id = {str(v).strip().lower(): int(k) for k, v in truth.names.items()}
    aligned_frames: list[outputs.Detection] = []
    for pred_frame, gt_frame in zip(preds, truth):
        new_boxes: list[outputs.Box] = []
        for b in pred_frame:
            norm_lbl = _normalize_label(b.label or "object", gt_labels)
            if filter_to_gt_classes and gt_labels and norm_lbl not in gt_labels:
                continue
            cid = label_to_id.get(norm_lbl, b.class_id)
            new_boxes.append(
                outputs.Box(
                    list(b.box),
                    score=float(b.score) if b.score is not None else 1.0,
                    label=norm_lbl,
                    class_id=cid,
                )
            )
        aligned_frames.append(
            outputs.Detection(
                new_boxes,
                source_id=gt_frame.source,
                frame_index=gt_frame.frame_index,
                image_size=gt_frame.image_size,
            )
        )
    return outputs.Detections(aligned_frames, names=truth.names)


def _to_class_agnostic(dets: outputs.Detections, truth_ref: outputs.Detections) -> outputs.Detections:
    frames: list[outputs.Detection] = []
    for frame, ref in zip(dets, truth_ref):
        boxes = [
            outputs.Box(
                list(b.box),
                score=float(b.score) if b.score is not None else 1.0,
                label="object",
                class_id=0,
            )
            for b in frame
        ]
        frames.append(
            outputs.Detection(
                boxes,
                source_id=ref.source,
                frame_index=ref.frame_index,
                image_size=ref.image_size,
            )
        )
    return outputs.Detections(frames, names={0: "object"})


def _load_detector_for_metrics(
    model_key: str,
    *,
    device: str,
    imgsz: int,
    threshold: float,
    prompt_classes: list[str],
    allow_remote_code: bool = True,
) -> tuple[Any, list[str] | None]:
    if model_key in ("yolo", "yolo11", "yolo26", "yolo-world"):
        weights = {
            "yolo": "yolov8n.pt",
            "yolo11": "yolo11n.pt",
            "yolo26": "yolo26n.pt",
            "yolo-world": "yolov8s-worldv2.pt",
        }[model_key]
        detector = models.load(
            weights,
            device=device,
            imgsz=imgsz,
            half=False,
            threshold=threshold,
            iou=0.65,
            model_stream=False,
        )
        if model_key == "yolo-world" and prompt_classes:
            detector.model.set_classes(prompt_classes)
            detector.names = dict(detector.model.names)
            return detector, prompt_classes
        return detector, None

    if model_key.startswith("grounding-dino"):
        name = "grounding-dino-base" if model_key.endswith("base") else "grounding-dino-tiny"
        detector = models.load(
            name,
            size={"shortest_edge": imgsz, "longest_edge": imgsz * 2},
            threshold=threshold,
        )
        detector.model.to(device=device, dtype=torch.float32).eval()
        return detector, prompt_classes

    if model_key == "faster-rcnn":
        detector = models.load("fasterrcnn_mobilenet_v3_large_fpn", device=device, threshold=threshold)
        detector.model.transform.min_size = (imgsz,)
        detector.model.transform.max_size = imgsz * 2
        detector.model.to(device=device, dtype=torch.float32).eval()
        return detector, None

    if model_key == "d-fine":
        import keras
        keras.mixed_precision.set_global_policy("float32")
        detector = models.load("dfine_nano_coco", device=device, threshold=threshold, iou=0.5)
        return detector, None

    if model_key == "locate-anything":
        if not allow_remote_code:
            raise PermissionError("locate-anything requires --allow-remote-code")
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        detector = models.load(
            "locate-anything",
            model_trust_remote_code=True,
            model_device=device,
            model_torch_dtype=dtype,
            generation_mode="fast",
            max_new_tokens=512,
        )
        return detector, ["object"]

    raise ValueError(f"Unsupported model key: {model_key}")


def _predict_dataset_with_model(
    model_key: str,
    truth: outputs.Detections,
    *,
    device: str,
    imgsz: int,
    threshold: float,
    batch_size: int,
    allow_remote_code: bool,
) -> outputs.Detections:
    # Collect active GT classes in the evaluated split so open-vocabulary prompts stay compact and relevant
    active_gt_classes = sorted({
        str(b.label).strip()
        for frame in truth
        for b in frame
        if b.label
    })
    if not active_gt_classes:
        active_gt_classes = [str(v) for v in truth.names.values()]

    detector, prompt = _load_detector_for_metrics(
        model_key,
        device=device,
        imgsz=imgsz,
        threshold=threshold,
        prompt_classes=active_gt_classes,
        allow_remote_code=allow_remote_code,
    )

    try:
        # Grounding DINO BERT text encoder has a 256-token cap (~35 class names per pass).
        # When evaluating >30 active classes, run chunked prompts and merge boxes per image.
        if model_key.startswith("grounding-dino") and prompt and len(prompt) > 30:
            chunk_size = 28
            chunks = [prompt[i:i + chunk_size] for i in range(0, len(prompt), chunk_size)]
            merged_boxes_per_frame: list[list[outputs.Box]] = [[] for _ in range(len(truth))]
            for ch in chunks:
                part = truth.copy()
                part.repredict(detector, prompt=ch, threshold=threshold, batch_size=max(1, batch_size // 2), verbose=False)
                for idx, f in enumerate(part):
                    merged_boxes_per_frame[idx].extend(list(f))
            frames = [
                outputs.Detection(
                    merged_boxes_per_frame[idx],
                    source_id=truth[idx].source,
                    frame_index=truth[idx].frame_index,
                    image_size=truth[idx].image_size,
                )
                for idx in range(len(truth))
            ]
            return outputs.Detections(frames, names=truth.names)

        preds = truth.copy()
        eff_batch = 4 if model_key in ("locate-anything", "grounding-dino-base") else batch_size
        preds.repredict(
            detector,
            prompt=prompt,
            threshold=threshold,
            batch_size=eff_batch,
            verbose=False,
        )
        return preds
    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _evaluate_single_model(
    model_key: str,
    raw_preds: outputs.Detections,
    truth: outputs.Detections,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    is_class_agnostic_only = (model_key == "locate-anything")
    aligned_preds = _align_predictions_to_truth(
        raw_preds,
        truth,
        filter_to_gt_classes=not is_class_agnostic_only,
    )

    evaluator = metrics.compose(
        metrics.precision_recall(iou=0.5),
        metrics.map_score(),
        metrics.size_breakdown(iou=0.5),
        metrics.error_analysis(iou=0.5, bg_iou=0.1),
        metrics.calibration(iou=0.5, bins=10),
        metrics.threshold_sweep(iou=0.5),
        metrics.iou_sweep(),
        metrics.per_class(iou=0.5),
        metrics.per_image(iou=0.5),
        metrics.confusion_matrix(iou=0.5),
        metrics.match_boxes(iou=0.5),
    )
    report = metrics.evaluate(aligned_preds, truth, evaluator=evaluator, verbose=False)

    # Also compute Class-Agnostic Box Localization metrics so all 9 models (including locate-anything)
    # can be compared on pure object localization accuracy
    agn_preds = _to_class_agnostic(raw_preds, truth)
    agn_truth = _to_class_agnostic(truth, truth)
    agn_eval = metrics.compose(
        metrics.precision_recall(iou=0.5),
        metrics.map_score(),
        metrics.size_breakdown(iou=0.5),
    )
    agn_report = metrics.evaluate(agn_preds, agn_truth, evaluator=agn_eval, verbose=False)

    pr = report["precision_recall"]
    ms = report["map_score"]
    sb = {row["size"]: row for row in report["size_breakdown"]}
    ea = report["error_analysis"]["counts"]
    cal = report["calibration"]
    ts = report["threshold_sweep"]
    isw = {round(float(r["iou"]), 2): r for r in report["iou_sweep"]}
    pc = report["per_class"]
    pi = report["per_image"]
    cm = report["confusion_matrix"]
    mb = report["match_boxes"]

    agn_pr = agn_report["precision_recall"]
    agn_ms = agn_report["map_score"]
    agn_sb = {row["size"]: row for row in agn_report["size_breakdown"]}

    # If model is class-agnostic (locate-anything), use class-agnostic size breakdown for size table
    use_sb = agn_sb if is_class_agnostic_only else sb

    best_ts = max(ts, key=lambda r: _safe_float(r.get("f1"), 0.0)) if ts else {"threshold": 0.25, "f1": 0.0}
    supported_classes = [r for r in pc if int(r.get("support", 0)) > 0]
    detected_classes = [r for r in supported_classes if int(r.get("tp", 0)) > 0]

    cm_mat = np.asarray(cm["matrix"])
    bg_idx = int(cm["background_index"])
    fg_mat = cm_mat[:bg_idx, :bg_idx] if bg_idx > 0 else np.zeros((0, 0))
    fg_matched_total = int(fg_mat.sum())
    fg_diag_correct = int(np.trace(fg_mat)) if fg_mat.size > 0 else 0
    cm_cls_acc = (fg_diag_correct / fg_matched_total) if fg_matched_total > 0 else 0.0

    mean_img_f1 = float(np.mean([_safe_float(r.get("f1"), 0.0) for r in pi])) if pi else 0.0
    total_matched_pairs = sum(len(r.matches) for r in mb)

    leaderboard_row = {
        "model": model_key,
        "mode": "class-agnostic" if is_class_agnostic_only else "multi-class",
        "mAP50_95": round(_safe_float(agn_ms.get("map") if is_class_agnostic_only else ms.get("map")), 4),
        "mAP50": round(_safe_float(agn_ms.get("map_50") if is_class_agnostic_only else ms.get("map_50")), 4),
        "mAP75": round(_safe_float(agn_ms.get("map_75") if is_class_agnostic_only else ms.get("map_75")), 4),
        "f1": round(_safe_float(agn_pr.get("f1") if is_class_agnostic_only else pr.get("f1")), 4),
        "precision": round(_safe_float(agn_pr.get("precision") if is_class_agnostic_only else pr.get("precision")), 4),
        "recall": round(_safe_float(agn_pr.get("recall") if is_class_agnostic_only else pr.get("recall")), 4),
        "mean_iou": round(_safe_float(agn_pr.get("mean_iou") if is_class_agnostic_only else pr.get("mean_iou")), 4),
        "loc_mAP50_95": round(_safe_float(agn_ms.get("map")), 4),
        "loc_mAP50": round(_safe_float(agn_ms.get("map_50")), 4),
        "optimal_threshold": round(_safe_float(best_ts.get("threshold"), 0.25), 2),
        "optimal_f1": round(_safe_float(best_ts.get("f1")), 4),
        "ece": round(_safe_float(cal.get("ece")), 4),
        "detected_classes": f"{len(detected_classes)}/{len(supported_classes)}",
    }

    diagnostic_row = {
        "model": model_key,
        "small_ap": round(_safe_float(use_sb.get("small", {}).get("ap")), 4),
        "medium_ap": round(_safe_float(use_sb.get("medium", {}).get("ap")), 4),
        "large_ap": round(_safe_float(use_sb.get("large", {}).get("ap")), 4),
        "small_f1": round(_safe_float(use_sb.get("small", {}).get("f1")), 4),
        "medium_f1": round(_safe_float(use_sb.get("medium", {}).get("f1")), 4),
        "large_f1": round(_safe_float(use_sb.get("large", {}).get("f1")), 4),
        "err_localization": int(ea.get("localization", 0)),
        "err_classification": int(ea.get("classification", 0)),
        "err_cls_loc": int(ea.get("classification_localization", 0)),
        "err_duplicate": int(ea.get("duplicate", 0)),
        "err_background": int(ea.get("background", 0)),
        "err_missed_gt": int(ea.get("missed", 0)),
        "f1_iou50": round(_safe_float(isw.get(0.5, {}).get("f1")), 4),
        "f1_iou75": round(_safe_float(isw.get(0.75, {}).get("f1")), 4),
        "f1_iou90": round(_safe_float(isw.get(0.9, {}).get("f1")), 4),
        "cm_class_acc": round(cm_cls_acc, 4),
        "per_image_mean_f1": round(mean_img_f1, 4),
        "matched_tp_boxes": int(total_matched_pairs),
    }

    per_class_clean = [
        {
            "model": model_key,
            "label": str(r["label"]),
            "support": int(r["support"]),
            "predictions": int(r["predictions"]),
            "tp": int(r["tp"]),
            "fp": int(r["fp"]),
            "fn": int(r["fn"]),
            "ap50": round(_safe_float(r.get("ap")), 4),
            "f1": round(_safe_float(r.get("f1")), 4),
            "precision": round(_safe_float(r.get("precision")), 4),
            "recall": round(_safe_float(r.get("recall")), 4),
            "mean_iou": round(_safe_float(r.get("mean_iou")), 4),
        }
        for r in supported_classes
    ]

    return leaderboard_row, diagnostic_row, per_class_clean


def _generate_metrics_charts(
    leaderboard: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    out_dir: Any,
) -> list[tuple[str, str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if not leaderboard:
        return []

    charts_dir = out_dir / "charts"
    files.mkdir(charts_dir)
    created: list[tuple[str, str]] = []

    models_list = [r["model"] for r in leaderboard]
    x = np.arange(len(models_list))
    w = 0.20

    # 1. mAP@50, mAP@75, mAP@50-95 & Localization mAP@50-95 Leaderboard
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.bar(x - 1.5 * w, [float(r["mAP50"]) for r in leaderboard], w, label="mAP@50")
    ax.bar(x - 0.5 * w, [float(r["mAP75"]) for r in leaderboard], w, label="mAP@75")
    ax.bar(x + 0.5 * w, [float(r["mAP50_95"]) for r in leaderboard], w, label="mAP@50-95")
    ax.bar(x + 1.5 * w, [float(r["loc_mAP50_95"]) for r in leaderboard], w, label="Class-Agnostic Loc mAP@50-95")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=28, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Multi-Model Detection & Localization Accuracy (klygo.metrics)")
    ax.legend()
    fig.tight_layout()
    p1 = charts_dir / "metrics_map_leaderboard.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    created.append(("Multi-Model mAP & Localization Leaderboard", "charts/metrics_map_leaderboard.png"))

    # 2. Multi-Scale Breakdown (Small / Medium / Large AP)
    diag_by_model = {d["model"]: d for d in diagnostics}
    ordered_diag = [diag_by_model[m] for m in models_list if m in diag_by_model]
    w3 = 0.25
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.bar(x - w3, [float(d["small_ap"]) for d in ordered_diag], w3, label="Small Objects AP (<32² px)")
    ax.bar(x, [float(d["medium_ap"]) for d in ordered_diag], w3, label="Medium Objects AP (32²–96² px)")
    ax.bar(x + w3, [float(d["large_ap"]) for d in ordered_diag], w3, label="Large Objects AP (>96² px)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=28, ha="right")
    ax.set_ylabel("AP@50")
    ax.set_title("Scale Breakdown: Small vs Medium vs Large Object AP (metrics.size_breakdown)")
    ax.legend()
    fig.tight_layout()
    p2 = charts_dir / "metrics_size_breakdown.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    created.append(("Multi-Scale Object Breakdown (Small / Medium / Large)", "charts/metrics_size_breakdown.png"))

    # 3. Error Analysis Breakdown & Calibration (ECE)
    fig, ax = plt.subplots(figsize=(12, 5.0))
    w5 = 0.16
    ax.bar(x - 2 * w5, [int(d["err_localization"]) for d in ordered_diag], w5, label="Localization FP")
    ax.bar(x - w5, [int(d["err_classification"]) for d in ordered_diag], w5, label="Classification FP")
    ax.bar(x, [int(d["err_duplicate"]) for d in ordered_diag], w5, label="Duplicate FP")
    ax.bar(x + w5, [int(d["err_background"]) for d in ordered_diag], w5, label="Background FP")
    ax.bar(x + 2 * w5, [int(d["err_missed_gt"]) for d in ordered_diag], w5, label="Missed GT (FN)")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, rotation=28, ha="right")
    ax.set_ylabel("Box Count")
    ax.set_title("TIDE Error Categorization by Model (metrics.error_analysis)")
    ax.legend()
    fig.tight_layout()
    p3 = charts_dir / "metrics_error_calibration.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    created.append(("TIDE Error Categorization Breakdown", "charts/metrics_error_calibration.png"))

    return created


def _render_metrics_report(summary: dict[str, Any], charts: list[tuple[str, str]]) -> str:
    env = summary["environment"]
    git_info = env.get("git", {})
    ds = summary.get("dataset", {})
    lines = [
        "# Klygo Metrics Multi-Model Accuracy & Diagnostic Report",
        "",
        f"- **GPU**: `{env.get('gpu', 'none')}` ({env.get('vram_mb', 0.0)} MB VRAM)",
        f"- **Dataset**: `{ds.get('source')}` (`{ds.get('images_evaluated')}` images, `{ds.get('gt_boxes')}` GT boxes, `{ds.get('defined_classes')}` defined classes, `{ds.get('active_gt_classes')}` active GT classes)",
        f"- **Models Evaluated**: `{len(summary.get('accuracy_leaderboard', []))}` models (`{', '.join(r['model'] for r in summary.get('accuracy_leaderboard', []))}`)",
        f"- **Git**: `{git_info.get('branch', 'unknown')}` @ `{git_info.get('commit', 'unknown')}`",
        "",
    ]
    lines.extend(_md_table(summary.get("accuracy_leaderboard", []), "Multi-Model Detection & Localization Leaderboard (Sorted by mAP@50-95)"))
    lines.extend(_md_table(summary.get("size_and_error_breakdown", []), "Scale Breakdown (Small/Medium/Large), TIDE Error Counts & IoU Sweep"))
    if charts:
        lines.extend(["## Charts", ""])
        for title, rel in charts:
            lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Flat YOLO dataset directory (coco128, voc2007, val2017)")
    parser.add_argument("--output", default="artifacts/benchmarks/metrics_eval")
    parser.add_argument("--models", default=",".join(ALL_MODELS), help="Comma-separated model keys")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap on evaluated images (0 = all)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-remote-code", action="store_true", default=True)
    parser.add_argument("--zip", default="", help="Optional output .zip path")
    args = parser.parse_args()

    out_dir = files.path(args.output)
    files.mkdir(out_dir)

    root = files.path(args.dataset_dir)
    img_sub = "images/train2017" if files.is_dir(root / "images" / "train2017") else "images"
    lbl_sub = "labels/train2017" if files.is_dir(root / "labels" / "train2017") else "labels"
    truth = datasets.load(root, images_dir=img_sub, labels_dir=lbl_sub, yaml_name="data.yaml", verbose=False)

    if args.max_images > 0 and len(truth) > args.max_images:
        truth = outputs.Detections(list(truth)[:args.max_images], names=truth.names)

    env = collect_environment()
    files.save(out_dir / "environment.json", env, overwrite=True, verbose=False)

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    leaderboard_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for idx, m_key in enumerate(model_keys, 1):
        print(f"[{idx}/{len(model_keys)}] Predicting & evaluating '{m_key}' on {len(truth)} images...")
        raw_preds = _predict_dataset_with_model(
            m_key,
            truth,
            device=args.device,
            imgsz=args.imgsz,
            threshold=args.threshold,
            batch_size=args.batch,
            allow_remote_code=args.allow_remote_code,
        )
        lb_row, diag_row, pc_rows = _evaluate_single_model(m_key, raw_preds, truth)
        leaderboard_rows.append(lb_row)
        diagnostic_rows.append(diag_row)
        per_class_rows.extend(pc_rows)

    leaderboard_rows.sort(
        key=lambda r: (
            0 if r["mode"] == "multi-class" else -1,
            r["mAP50_95"],
            r["mAP50"],
            r["f1"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(leaderboard_rows, 1):
        row["rank"] = rank

    # Reorder columns so rank is first
    ordered_leaderboard = [
        {"rank": r["rank"], **{k: v for k, v in r.items() if k != "rank"}}
        for r in leaderboard_rows
    ]

    active_gt = sorted({str(b.label) for f in truth for b in f if b.label})
    total_gt_boxes = sum(len(f) for f in truth)

    summary = {
        "environment": env,
        "dataset": {
            "source": str(root),
            "images_evaluated": len(truth),
            "gt_boxes": total_gt_boxes,
            "defined_classes": len(truth.names),
            "active_gt_classes": len(active_gt),
        },
        "config": {
            "imgsz": args.imgsz,
            "threshold": args.threshold,
            "batch": args.batch,
            "models": model_keys,
        },
        "accuracy_leaderboard": ordered_leaderboard,
        "size_and_error_breakdown": diagnostic_rows,
    }

    files.save(out_dir / "accuracy_leaderboard.json", ordered_leaderboard, overwrite=True, verbose=False)
    files.save(out_dir / "size_and_error_breakdown.json", diagnostic_rows, overwrite=True, verbose=False)
    files.save(out_dir / "per_class_summary.json", per_class_rows, overwrite=True, verbose=False)
    files.save(out_dir / "summary.json", summary, overwrite=True, verbose=False)

    charts = _generate_metrics_charts(ordered_leaderboard, diagnostic_rows, out_dir)
    files.save(out_dir / "REPORT.md", _render_metrics_report(summary, charts), overwrite=True, verbose=False)
    utils.tabulate(ordered_leaderboard, title="Multi-Model Detection Accuracy Leaderboard")

    if args.zip:
        zip_path = files.path(args.zip)
        files.mkdir(zip_path.parent)
        archive.compress(out_dir, zip_path, include_root=True, overwrite=True)
        print(f"Packaged ZIP: {zip_path}")


if __name__ == "__main__":
    main()

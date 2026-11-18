---
name: klygo-metrics-benchmark
description: "Benchmark multi-model object-detection accuracy and diagnostic quality across all 9 Klygo detectors (`yolo`, `yolo11`, `yolo26`, `yolo-world`, `grounding-dino`, `grounding-dino-base`, `faster-rcnn`, `d-fine`, `locate-anything`) on real datasets (`coco128`, `voc2007`, `val2017`) using all 13 public functions of `klygo.metrics`. Saves standardized reports, JSON leaderboards, and 3 charts under `benchmarks/results/metrics/<gpu>_<dataset>_<YYYY_MM_DD>/`."
---

# Klygo Metrics Benchmark Skill (`klygo-metrics-benchmark`)

## 1. Goal & Scope

Evaluate predicted bounding boxes and labels from all **9 Klygo detector models** against ground-truth dataset annotations (`coco128`, `voc2007`, `val2017`) using all **13 public functions of `klygo.metrics`** (without re-running CPU vs CUDA execution speed benchmarks, which are already covered in `performance/`):

1. **Overall Detection & Localization Accuracy (`map_score` & `precision_recall`)**:
   - Multi-class `mAP50_95`, `mAP50`, `mAP75`, `precision`, `recall`, `f1`, `mean_iou`.
   - Class-agnostic localization `loc_mAP50_95`, `loc_mAP50` (evaluating pure box localization so `locate-anything` and all 8 class-aware models are compared on equal footing).
2. **Multi-Scale Object Breakdown (`size_breakdown`)**:
   - COCO scale tiers: `small_ap`, `medium_ap`, `large_ap`, `small_f1`, `medium_f1`, `large_f1`.
3. **TIDE Error Categorization (`error_analysis`)**:
   - False-positive & false-negative breakdown: `err_localization`, `err_classification`, `err_cls_loc`, `err_duplicate`, `err_background`, and `err_missed_gt`.
4. **Confidence Calibration & Optimal Operating Thresholds (`calibration`, `threshold_sweep`, `iou_sweep`)**:
   - `ece` (Expected Calibration Error), `optimal_threshold`, `optimal_f1`, and `f1_iou50` / `f1_iou75` / `f1_iou90`.
5. **Per-Class, Per-Image & Confusion Matrix Diagnostics (`per_class`, `per_image`, `confusion_matrix`, `match_boxes`)**:
   - `detected_classes` (`detected / active_gt_classes`), `cm_class_acc`, `per_image_mean_f1`, `matched_tp_boxes`, and full per-class AP/F1 breakdown.

---

## 2. Standardized Output Directory Structure

Every run saves the standardized file set under `benchmarks/results/metrics/<gpu>_<dataset>_<YYYY_MM_DD>/`:
- `REPORT.md` — Full Markdown report with the 9-model accuracy leaderboard, scale/error/IoU breakdown table, and embedded charts.
- `environment.json` — Hardware, OS, CPU/GPU, PyTorch/CUDA, and Python environment metadata.
- `accuracy_leaderboard.json` — Ranked 9-model leaderboard (`mAP50_95`, `mAP50`, `mAP75`, `f1`, `precision`, `recall`, `mean_iou`, `loc_mAP50_95`, `loc_mAP50`, `optimal_threshold`, `optimal_f1`, `ece`, `detected_classes`).
- `size_and_error_breakdown.json` — Per-model scale breakdown (`small`/`medium`/`large`), 6 error categories, IoU sweep points, and confusion matrix accuracy.
- `per_class_summary.json` — Per-model, per-class AP/F1/IoU metrics from `metrics.per_class()`.
- `summary.json` — Master JSON bundle.
- `charts/metrics_map_leaderboard.png` — Grouped bar chart of `mAP@50`, `mAP@75`, `mAP@50-95`, and `Class-Agnostic Loc mAP@50-95`.
- `charts/metrics_size_breakdown.png` — Grouped bar chart of `Small`, `Medium`, and `Large` object AP across models.
- `charts/metrics_error_calibration.png` — Grouped bar chart of TIDE error counts (`Localization`, `Classification`, `Duplicate`, `Background`, `Missed GT`).

---

## 3. Canonical 3-Tier Commands (`coco128` / `voc2007` / `val2017`)

```bash
# Tier 1: COCO128 (128 images, 80 classes, all 9 models)
python benchmarks/metrics/run_metrics_benchmark.py \
  --dataset-dir /path/to/coco128 \
  --output benchmarks/results/metrics/colab_l4_coco128_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_metrics_coco128_results.zip

# Tier 2: Pascal VOC 2007 (500 images, 20 classes with automatic COCO->VOC synonym alignment, all 9 models)
python benchmarks/metrics/run_metrics_benchmark.py \
  --dataset-dir /path/to/voc2007_yolo \
  --max-images 500 \
  --output benchmarks/results/metrics/colab_l4_voc2007_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_metrics_voc2007_results.zip

# Tier 3: COCO val2017 (500 images, 80 classes, all 9 models)
python benchmarks/metrics/run_metrics_benchmark.py \
  --dataset-dir /path/to/val2017_yolo \
  --max-images 500 \
  --output benchmarks/results/metrics/colab_l4_val2017_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_metrics_val2017_results.zip
```

---

## 4. Git Rules for Benchmark Results

Because `.gitignore` ignores `*.png` by default, always force-add chart images when committing metrics benchmark results:
```bash
git add benchmarks/results/metrics/
git add -f benchmarks/results/metrics/**/charts/*.png
```

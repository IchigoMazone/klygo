---
name: klygo-metrics-benchmark
description: Benchmark Klygo object-detection accuracy (`mAP@50`, `mAP@50-95`, Precision, Recall, F1 via `klygo.metrics.evaluate`) on real datasets (`val2017`, `voc2007`, `coco128`) as well as `klygo.metrics` CPU vs CUDA latency and FP16/FP32/FP64 precision drift. Saves reports under `benchmarks/results/metrics/<dataset>_<YYYY_MM_DD>/`.
---

# Klygo Metrics & Detection Accuracy Benchmark Skill

Use this skill when measuring **detection accuracy (`mAP@50`, `mAP@50-95`, Precision, Recall, F1)** with `klygo.metrics`, or **CPU vs CUDA IoU/matching throughput** of `klygo.metrics`.

## 1. Scope & Metrics
- **Model Accuracy Evaluation on Real Datasets (`val2017`, `voc2007`, `coco128`)**:
  - Load ground truth via `datasets.load(...)`, run `predictions.repredict(detector, ...)` (or `detector.predict(...)`), and evaluate with `metrics.evaluate(predictions, truth)` / `metrics.compose(...)`.
  - Report `mAP@0.5`, `mAP@0.5:0.95`, `precision`, `recall`, `f1`, and per-class breakdown.
- **Metrics Kernel Throughput & Numerical Precision** (`evaluate_release.py` metrics suite):
  - CPU vs CUDA (`CuPy`) across `FP16`, `FP32`, `FP64`, `packed`/`unpacked`, and dense boxes up to `xlarge_5000_2000` and `batch_100f_10k`.

## 2. Standard Commands

### A. Evaluate Model mAP + Kernel Latency Together
```bash
PYTHONPATH=. python benchmarks/metrics/run_metrics_benchmark.py \
  --output artifacts/benchmarks/metrics_eval \
  --dataset-dir /path/to/yolo_dataset \
  --model yolov8n.pt \
  --large-scale \
  --repeats 7 \
  --zip artifacts/benchmarks/metrics_results.zip
```

### B. Metrics Kernel CPU vs CUDA Speed & Precision Drift Only
```bash
PYTHONPATH=. python benchmarks/metrics/run_metrics_benchmark.py \
  --output artifacts/benchmarks/metrics_eval \
  --large-scale \
  --repeats 7 \
  --zip artifacts/benchmarks/metrics_results.zip
```

## 3. Package `.zip`, Upload GitHub Release & Push Reports
1. **Compress raw output** (never commit `.zip` or `artifacts/`):
   ```python
   from klygo import archive
   archive.compress("artifacts/benchmarks/metrics_eval", "artifacts/benchmarks/metrics_<dataset>_results.zip", include_root=True, overwrite=True)
   ```
2. **Upload `.zip` to GitHub Release** (`gh release create "benchmark-metrics-<dataset>-$(date +%Y%m%d)-$(git rev-parse --short HEAD)" artifacts/benchmarks/metrics_<dataset>_results.zip --prerelease`, or browser fallback via `releases/new`).
3. **Commit & push** `REPORT.md`, `charts/*.png`, executed `.ipynb`, and summary JSONs under `benchmarks/results/metrics/<dataset>_<YYYY_MM_DD>/`:
   ```bash
   git add benchmarks/results/metrics/
   git add -f benchmarks/results/metrics/**/charts/*.png
   git commit -m "Add metrics mAP & accuracy benchmark report (<dataset>, <YYYY-MM-DD>)"
   git push origin HEAD
   ```

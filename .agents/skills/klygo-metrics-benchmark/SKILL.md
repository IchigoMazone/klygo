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

### A. Evaluate Model mAP on a Real YOLO Dataset
```python
from klygo import datasets, metrics, models

truth = datasets.load("/path/to/yolo_dataset", format="yolo", verbose=False)
detector = models.load("yolov8n.pt")
preds = truth.copy()
preds.repredict(detector, threshold=0.25, verbose=False)
report = metrics.evaluate(preds, truth, verbose=False)
```

### B. Metrics Kernel CPU vs CUDA Speed & Precision Drift
```bash
PYTHONPATH=. python -c "
from benchmarks.gpu_suite.evaluate_release import collect_environment, benchmark_metrics
from klygo import files, utils
env = collect_environment()
rows = benchmark_metrics(env, repeats=7, large_scale=True)
files.save('artifacts/benchmarks/metrics_eval/metrics_benchmark.json', rows, overwrite=True)
utils.tabulate(rows, title='klygo.metrics CPU vs CUDA')
"
```

## 3. Saving Results
Store `REPORT.md`, `charts/*.png`, and JSON summaries under:
```text
benchmarks/results/metrics/<dataset>_<YYYY_MM_DD>/
```

---
name: klygo-cpu-benchmark
description: Run CPU-only and disk I/O benchmarks for Klygo without requiring a GPU (`datasets.load`, `media.load` eager/stream/indexed, strided and random-jump image reads on `MediaFrames` and `Detections`, `preprocessing.compose`, disk-backed `mining`, and optional `--include-cpu-utils`). Saves reports under `benchmarks/results/cpu_loaders/<cpu>_<dataset>_<YYYY_MM_DD>/`.
---

# Klygo CPU & Loader Benchmark Skill

Use this skill when measuring **CPU and disk I/O performance** (no GPU required — runs on local CPU or free Colab CPU runtime):

## 1. What It Measures
- **Hardware & Dataset Context**: CPU model, core count (`cpu_count`), OS, Python version, and exact dataset (`coco128`, `val2017`, `voc2007`, or custom `--dataset-dir`).
- **Dataset & Media Loaders**:
  - `datasets.load` (single / scaled)
  - `media.load` (`stream=False` eager, `stream=True` streaming, `indexed=True`)
- **Strided & Random-Jump Image Access** (with `--large-scale`, 2,048+ images):
  - `media.strided_decode` (`sample_rate=32`)
  - `media.random_jump_numpy` & `media.random_jump_pil` (`MediaFrames[i]`)
  - `detections.random_jump_index` & `detections.random_jump_image_read` (`Detections[i].image`)
- **Preprocessing & Disk Mining**:
  - `preprocessing.compose` (letterbox + autocontrast)
  - `mining.stats(quality=True)`, `mining.find_duplicates`, `mining.cluster`
- **Optional CPU Utilities** (`--include-cpu-utils`):
  - `media.info`, `media.save_images`, `visual.compare_images`, `utils.tabulate`

## 2. Standard Commands

### A. Quick CPU Loader Benchmark (`coco128`)
```bash
PYTHONPATH=. python benchmarks/gpu_suite/evaluate_release.py \
  --output artifacts/benchmarks/cpu_coco128 \
  --dataset coco128 \
  --loaders-only \
  --large-scale \
  --repeats 7
```

### B. Full CPU Loader Benchmark on Custom / Industry Dataset (`val2017` / `voc2007`)
```bash
PYTHONPATH=. python benchmarks/gpu_suite/evaluate_release.py \
  --output artifacts/benchmarks/cpu_dataset \
  --dataset-dir /path/to/yolo_dataset \
  --loaders-only \
  --large-scale \
  --include-cpu-utils \
  --repeats 7
```

## 3. Saving & Pushing Results
Store the generated `REPORT.md`, `environment.json`, and `datasets_media_visual_benchmark.json` (as `loaders_benchmark.json`) under:
```text
benchmarks/results/cpu_loaders/<cpu>_<dataset>_<YYYY_MM_DD>/
```
Commit and push to the active branch (never commit `artifacts/` or datasets).

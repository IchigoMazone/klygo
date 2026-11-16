---
name: klygo-mining-benchmark
description: "Benchmark `klygo.mining` full-class dataset curation training accuracy (`benchmarks/mining/run_mining_benchmark.py` training YOLOv8n across `full_pool`, `random`, `quality`, `rare_first`, `weighted`, `diversity`, `coverage`, `balance_select`, `balance_undersample`, and `dedup_weighted` on all classes of `coco128`, `voc2007`, or `val2017`). Saves standardized reports, JSON leaderboards, and charts under `benchmarks/results/mining/<gpu>_<dataset>_<YYYY_MM_DD>/`."
---

# Klygo Mining Benchmark Skill (`klygo-mining-benchmark`)

## 1. Goal & Scope

Evaluate how each selection, deduplication, and balancing strategy in `klygo.mining` impacts downstream detector accuracy (`mAP@50`, `mAP@50-95`, `Precision`, `Recall`, `F1`) across **100% of dataset classes** (`coco128`: 80 classes, `voc2007`: 20 classes, `val2017`: 80 classes):

1. **10 Curation Methods Evaluated on All Dataset Classes**:
   - `full_pool`: Upper-bound reference trained on 100% of the candidate pool.
   - `random`: Uninformed random subset baseline at budget $K$.
   - `quality`: `mining.select(..., strategy="quality")`
   - `rare_first`: `mining.select(..., strategy="rare_first")`
   - `weighted`: `mining.select(..., strategy="weighted")`
   - `diversity`: `mining.select(..., strategy="diversity")`
   - `coverage`: `mining.select(..., strategy="coverage")`
   - `balance_select`: `mining.balance(..., method="select")`
   - `balance_undersample`: `mining.balance(..., method="undersample")`
   - `dedup_weighted`: `mining.deduplicate(...)` followed by `mining.select(..., strategy="weighted")`
2. **Standardized Output Directory Structure**:
   Every run saves the exact same standardized artifact set under `benchmarks/results/mining/<gpu>_<dataset>_<YYYY_MM_DD>/`:
   - `REPORT.md` — Full Markdown report with curation leaderboard, class coverage (`active_classes/total_classes`), and methodology.
   - `environment.json` — Hardware, OS, CPU/GPU, PyTorch/CUDA, and Python environment metadata.
   - `curation_leaderboard.json` — Per-method aggregated metrics (`mAP@50-95`, `mAP@50`, `Precision`, `Recall`, `F1`, `delta_map50_95_vs_random`, `active_classes`).
   - `curation_runs.json` — Detailed per-seed run logs and `mining.compare()` pool retention summaries.
   - `summary.json` — Top-level suite configuration and leaderboard summary.
   - `charts/mining_map_comparison.png` — Horizontal bar chart comparing `mAP@50-95` and `mAP@50` across all strategies.
   - `charts/mining_f1_precision_recall.png` — Grouped bar chart comparing `F1`, `Precision`, and `Recall` across all strategies.

*(Note: Execution speed of `klygo.mining` operations is already benchmarked in `klygo-release-benchmark` under `benchmarks/results/performance/` and `klygo-cpu-benchmark` under `benchmarks/results/cpu_loaders/`.)*

---

## 2. Canonical Commands

### Run Full-Class Curation Accuracy Benchmark (`coco128` / `voc2007` / `val2017`)
```bash
# Tier 1: COCO128 (80 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset coco128 \
  --budget-ratio 0.5 \
  --epochs 10 \
  --imgsz 640 \
  --seeds 42 \
  --output benchmarks/results/mining/colab_l4_coco128_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_coco128_results.zip

# Tier 2: Pascal VOC 2007 (20 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset voc2007 \
  --pool-size 1200 \
  --val-size 400 \
  --budget 400 \
  --epochs 10 \
  --imgsz 640 \
  --seeds 42 \
  --output benchmarks/results/mining/colab_l4_voc2007_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_voc2007_results.zip

# Tier 3: COCO val2017 (80 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset val2017 \
  --pool-size 2000 \
  --val-size 500 \
  --budget 600 \
  --epochs 10 \
  --imgsz 640 \
  --seeds 42 \
  --output benchmarks/results/mining/colab_l4_val2017_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_val2017_results.zip
```

---

## 3. Git Rules for Benchmark Results

Because `.gitignore` ignores `*.png` by default, always force-add chart images when committing mining benchmark results:
```bash
git add benchmarks/results/mining/
git add -f benchmarks/results/mining/**/charts/*.png
```

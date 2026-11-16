---
name: klygo-mining-benchmark
description: "Benchmark `klygo.mining` full-class dataset curation across 15 methods (`benchmarks/mining/run_mining_benchmark.py`), measuring both intrinsic multi-class metrics (`stats`, `score`, `compare`, `cluster`, `find_duplicates`) and downstream YOLOv8n training accuracy across all classes of `coco128`, `voc2007`, or `val2017`. Saves standardized reports, JSON leaderboards, and 3 charts under `benchmarks/results/mining/<gpu>_<dataset>_<YYYY_MM_DD>/`."
---

# Klygo Mining Benchmark Skill (`klygo-mining-benchmark`)

## 1. Goal & Scope

Evaluate `klygo.mining` as broadly as possible across **100% of dataset classes** (`coco128`: 80 classes, `voc2007`: 20 classes, `val2017`: 80 classes) on **two complementary axes** (without timing/speed benchmarks, which are already covered in `performance/` and `cpu_loaders/`):

### Axis 1 — Intrinsic Multi-Class Dataset Curation Metrics (`mining.stats`, `mining.score`, `mining.compare`, `mining.cluster`, `mining.find_duplicates`)
- `active_classes` & `class_coverage_pct` — % of active pool classes preserved by the subset.
- `rare_tail_coverage_pct` — % of the bottom-quartile (25% rarest) classes retained.
- `class_entropy_norm` — Normalized Shannon entropy of the subset's class box distribution in $[0, 1]$.
- `imbalance_ratio` — Max-to-min class count ratio (`mining.stats`).
- `cluster_coverage_pct` — % of K-Means visual clusters (`mining.cluster`) represented in the subset.
- `mean_quality_score` & `mean_composite_score` — Mean image fidelity and multi-criteria score (`mining.score`).
- `boxes_per_image`, `multilabel_rate_pct`, & `image_retention_pct` (`mining.compare`).

### Axis 2 — Downstream Detector Training Accuracy (`yolov8n.pt` + `klygo.metrics`)
- Fine-tunes `yolov8n.pt` under identical epochs, seed, and hyperparameters on each curated subset and evaluates on a fixed clean validation split across all classes (`mAP@50-95`, `mAP@50`, `Precision`, `Recall`, `F1`, and `delta_map50_95_vs_random`).

### 15 Evaluated Methods
1. `full_pool` — 100% of the candidate training pool (upper-bound reference).
2. `random` — `mining.select(method="random")` baseline at budget $K$.
3. `weighted` — `mining.select(method="weighted")`
4. `quality` — `mining.select(method="quality")`
5. `rare_first` — `mining.select(method="rare_first")`
6. `diversity` — `mining.cluster()` + `mining.select(method="diversity")`
7. `coverage` — `mining.select(method="coverage")`
8. `quota` — `mining.select(method="quota")`
9. `boxes` — `mining.select(method="boxes")`
10. `balance_select_median` — `mining.balance(target="median", strategy="select", count_by="images")`
11. `balance_undersample_median` — `mining.balance(target="median", strategy="undersample", count_by="images")`
12. `balance_boxes_mean` — `mining.balance(target="mean", strategy="select", count_by="boxes")`
13. `dedup_best_weighted` — `mining.deduplicate(keep="best")` + `mining.select(method="weighted")`
14. `dedup_quality_rare` — `mining.deduplicate(keep="quality")` + `mining.select(method="rare_first")`
15. `dedup_annotations_diversity` — `mining.deduplicate(keep="annotations")` + `mining.select(method="diversity")`

---

## 2. Standardized Output Directory Structure

Every run saves the standardized file set under `benchmarks/results/mining/<gpu>_<dataset>_<YYYY_MM_DD>/`:
- `REPORT.md` — Markdown report with pool duplicate analysis, 15-method leaderboard, detailed intrinsic + downstream metrics table, and embedded charts.
- `environment.json` — Hardware, OS, CPU/GPU, PyTorch/CUDA, and Python environment metadata.
- `pool_analysis.json` — Full candidate pool analysis (`mining.stats`, `mining.find_duplicates`, `mining.cluster`, rare-tail class list).
- `curation_leaderboard.json` — Ranked 15-method leaderboard combining downstream (`mAP@50-95`, `mAP@50`, `F1`, `delta_map50_95_vs_random`) and intrinsic (`active_classes`, `class_coverage_pct`, `rare_tail_coverage_pct`, `cluster_coverage_pct`, `class_entropy_norm`, `mean_quality_score`) metrics.
- `curation_runs.json` — Complete per-method metrics dictionary.
- `summary.json` — Master JSON bundle.
- `charts/mining_map_comparison.png` — Grouped bar chart of `mAP@50` and `mAP@50-95` across all 15 methods.
- `charts/mining_f1_precision_recall.png` — Grouped bar chart of `Precision`, `Recall`, and `F1` across all 15 methods.
- `charts/mining_class_coverage_diversity.png` — Grouped bar chart of `Class Coverage (%)`, `Rare-Tail Class Coverage (%)`, and `Visual Cluster Coverage (%)` across all 15 methods.

---

## 3. Canonical Commands

```bash
# Tier 1: COCO128 (80 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset-dir /path/to/coco128 \
  --budget-ratio 0.4 \
  --epochs 15 \
  --imgsz 640 \
  --output benchmarks/results/mining/colab_l4_coco128_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_coco128_results.zip

# Tier 2: Pascal VOC 2007 (20 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset-dir /path/to/voc2007_yolo \
  --max-pool 1000 \
  --max-val 300 \
  --budget 350 \
  --epochs 12 \
  --imgsz 640 \
  --output benchmarks/results/mining/colab_l4_voc2007_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_voc2007_results.zip

# Tier 3: COCO val2017 (80 classes)
python benchmarks/mining/run_mining_benchmark.py \
  --dataset-dir /path/to/val2017_yolo \
  --max-pool 1200 \
  --max-val 300 \
  --budget 400 \
  --epochs 12 \
  --imgsz 640 \
  --output benchmarks/results/mining/colab_l4_val2017_$(date +%Y_%m_%d) \
  --zip artifacts/benchmarks/colab_l4_mining_val2017_results.zip
```

---

## 4. Git Rules for Benchmark Results

Because `.gitignore` ignores `*.png` by default, always force-add chart images when committing mining benchmark results:
```bash
git add benchmarks/results/mining/
git add -f benchmarks/results/mining/**/charts/*.png
```

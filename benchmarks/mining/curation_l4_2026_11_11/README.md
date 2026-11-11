# Klygo Mining Curation Accuracy Benchmark (Colab L4, COCO `val2017`)

Recorded on **2026-11-11** on **Google Colab Pro (NVIDIA L4, 22.03 GB VRAM)** against commit [`1fb5020`](https://github.com/IchigoMazone/klygo/commit/1fb5020b7f75d86bb1208017846ccc7856a0efa4) (`benchmark_models_klygo`), PyTorch `2.14.1+cu130`, Ultralytics `8.4.208`.

- **Script**: [`benchmarks/mining/curation_accuracy_benchmark.py`](../curation_accuracy_benchmark.py)
- **Raw Machine Summary**: [`summary.json`](summary.json) · [`summary.md`](summary.md)
- **Colab Notebook**: [`Untitled21.ipynb`](https://colab.research.google.com/drive/1B0C4NMDUY9eoXnTPzaJXokI3lX8E36vj)
- **Evidence ZIP SHA-256**: `7b3fb75bf927887dfb7626ad1d3c7a8933d881cdff706157f9f1b561b5ac3886` (`mining_curation_results_2.zip`)

---

## 1. Benchmark Design

To test whether `klygo.mining` improves real downstream object-detection accuracy rather than only synthetic heuristics, the script constructs two **strictly disjoint** splits from real COCO `val2017` across 10 urban/indoor classes (`5` head: `person`, `car`, `chair`, `bottle`, `cup`; `5` rare/tail: `bicycle`, `bus`, `traffic light`, `dog`, `laptop`):

1. **Held-out clean validation (`val_fixed`)**: `247` clean unmodified COCO images used solely for evaluation across every strategy and seed.
2. **Realistic noisy long-tail training pool (`train_pool`)**: `848` images (`3,922` boxes, `imbalance_ratio = 29.125`, `person=699` vs `24` per rare class):
   - `508` clean core COCO images
   - `110` exact duplicate copies (`exact_dup`) of abundant head-class scenes
   - `110` near-duplicate burst frames (`near_dup`) synthesized via `klygo.preprocessing`
   - `120` visually degraded head-class frames (`degraded`: heavy blur, underexposure, low contrast) synthesized via `klygo.preprocessing`

Each curation strategy selects a training subset from `train_pool`, trains `yolov8n.pt` (`epochs=35`, `imgsz=640`, `batch=32`, seeds `42` and `123`), and evaluates `best.pt` on `val_fixed`.

---

## 2. Headline Leaderboard

| Strategy | Images | Boxes | Imbalance | Rare Imgs | Mining (s) | Train (s) | mAP@50-95 (mean ± std) | mAP@50 | Rare mAP | Head mAP | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `balance_median` | 142 | 937 | 3.29 | 120 | 18.31 | 69.5 | **0.3874** ± 0.0035 | 0.5780 | 0.4264 | 0.3483 | 0.6970 | 0.5036 |
| `rare_first_b` | 350 | 2,045 | 8.38 | 120 | 23.20 | 96.4 | **0.3519** ± 0.0156 | 0.5387 | 0.3843 | 0.3195 | 0.6771 | 0.4812 |
| `quality_b` | 350 | 1,757 | 31.44 | 71 | 23.20 | 96.3 | **0.3451** ± 0.0100 | 0.5270 | 0.3790 | 0.3112 | 0.6399 | 0.4815 |
| `dedup_weighted_b` | 350 | 2,214 | 10.25 | 120 | 26.68 | 96.5 | **0.3420** ± 0.0035 | 0.5270 | 0.3750 | 0.3089 | 0.6436 | 0.4845 |
| `coverage_b` | 350 | 2,254 | 11.71 | 120 | 0.60 | 96.8 | **0.3403** ± 0.0029 | 0.5262 | 0.3757 | 0.3049 | 0.6383 | 0.4878 |
| `diversity_b` | 350 | 1,769 | 20.50 | 80 | 16.95 | 96.4 | **0.3386** ± 0.0125 | 0.5237 | 0.3672 | 0.3099 | 0.6357 | 0.4780 |
| `klygo_smart_b` | 350 | 2,024 | 12.10 | 113 | 34.03 | 96.6 | **0.3378** ± 0.0128 | 0.5248 | 0.3730 | 0.3027 | 0.6338 | 0.4842 |
| `weighted_b` | 350 | 2,297 | 9.08 | 120 | 23.20 | 96.8 | **0.3327** ± 0.0150 | 0.5066 | 0.3637 | 0.3016 | 0.6195 | 0.4655 |
| `random_b` | 350 | 1,632 | 41.57 | 46 | 0.07 | 95.8 | **0.3294** ± 0.0077 | 0.5095 | 0.3533 | 0.3055 | 0.6694 | 0.4478 |
| `dedup_only` | 628 | 2,973 | 21.62 | 120 | 18.07 | 133.9 | **0.3211** ± 0.0032 | 0.4925 | 0.3450 | 0.2972 | 0.6520 | 0.4436 |
| `full_raw` | 848 | 3,922 | 29.12 | 120 | 0.00 | 166.9 | **0.3139** ± 0.0077 | 0.4863 | 0.3356 | 0.2921 | 0.6043 | 0.4542 |

---

## 3. Key Findings

1. **Class balancing + deduplication (`balance_median`) delivers the largest gain with the smallest training set**:
   Training on `142` balanced, deduplicated images (`imbalance = 3.29`) reaches **`0.3874` mAP@50-95** vs **`0.3139`** on the full `848`-image noisy pool (**`+0.0735` / `+23.4%` relative `mAP@50-95`**, **`+0.0917` `mAP@50`**, **`6.0x` fewer images**, **`2.4x` faster training** `69.5s` vs `166.9s`). Both rare classes (`+0.0908` AP) and head classes (`+0.0562` AP) improve simultaneously because removing head-class duplicate spam and severe long-tail skew eliminates gradient domination.
2. **Every `klygo.mining` curation method beats uncurated training and random sampling**:
   Under a matched `350`-image budget, `rare_first_b` (`0.3519`), `quality_b` (`0.3451`), `dedup_weighted_b` (`0.3420`), `coverage_b` (`0.3403`), `diversity_b` (`0.3386`), `klygo_smart_b` (`0.3378`), and `weighted_b` (`0.3327`) all outperform `random_b` (`0.3294`), `dedup_only` (`0.3211`), and `full_raw` (`0.3139`).
3. **Deduplication adds stability and removes redundant memorization**:
   `dedup_only` flags all `220` duplicate images (`61` exact hash groups, `28` similarity groups; `848 -> 628` retained images), improving `mAP@50-95` over `full_raw` (`0.3211` vs `0.3139`) while cutting training time by `19.8%`. Adding deduplication before `weighted` selection (`dedup_weighted_b`) boosts mean accuracy (`0.3420` vs `0.3327`) and cuts cross-seed standard deviation from `0.0150` down to `0.0035`.
4. **Class-agnostic quality thresholding hurts rare classes if unprotected**:
   In `klygo_smart_b`, applying a global 15th-percentile visual-quality cut dropped `7` rare-class COCO images (`120 -> 113`), reducing accuracy relative to `dedup_weighted_b` (`0.3378` vs `0.3420`) and `rare_first_b` (`0.3519`). Quality filtering should therefore preserve scarce/tail classes rather than applying a uniform global threshold.

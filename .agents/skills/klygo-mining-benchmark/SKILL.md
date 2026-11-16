---
name: klygo-mining-benchmark
description: Benchmark `klygo.mining` curation training accuracy (`benchmarks/mining/curation_accuracy_benchmark.py`: training YOLOv8n across `quality`, `rare_first`, `weighted`, `diversity`, `coverage`, `balance`, `dedup+weighted` strategies on noisy long-tail COCO val2017) and `klygo.mining` operation speed (`stats`, `find_duplicates`, `cluster`, `score`, `select`, `balance`). Saves reports under `benchmarks/results/mining/<scope>_<YYYY_MM_DD>/`.
---

# Klygo Mining & Curation Benchmark Skill

Use this skill when evaluating **`klygo.mining` data-curation strategies for model training** or **speed of mining operations** (`stats`, `find_duplicates`, `cluster`, `score`, `select`, `balance`).

## 1. Scope & Metrics
- **Curation Training Accuracy (`curation_accuracy_benchmark.py`)**:
  - Builds a realistic 10-class long-tail + noisy/duplicate pool from real `COCO val2017` images (`5 head` + `5 rare` classes) and a disjoint clean `val_fixed` split.
  - Curates fixed-budget subsets (`budget=350`) using `quality`, `rare_first`, `weighted`, `diversity`, `coverage`, `balance`, and `dedup + weighted`.
  - Trains `YOLOv8n` (`35 epochs`, multiple seeds) per subset on GPU and compares overall mAP, rare-class mAP, rare-class coverage, and curation time.
- **Mining Compute Speed**:
  - CPU vs CUDA `Detections` (`stats`, `score`, `select`, `balance`, `compare`) and disk-backed `stats(quality=True)`, `find_duplicates`, `cluster`.

## 2. Standard Command
```bash
PYTHONPATH=. python benchmarks/mining/curation_accuracy_benchmark.py \
  --output artifacts/benchmarks/mining_curation \
  --budget 350 \
  --epochs 35
```

## 3. Saving Results
Store `REPORT.md`, `charts/*.png`, `.ipynb`, and summary JSONs under:
```text
benchmarks/results/mining/<scope>_<YYYY_MM_DD>/
```
Upload large raw `.zip` artifacts to GitHub Releases; commit only lightweight reports and charts.

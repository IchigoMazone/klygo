# `klygo.mining`

`klygo.mining` analyzes and selects complete images from an object-detection dataset. Bounding boxes, labels, and image-quality measurements are features; the unit returned or removed is always a full `Detection` frame.

The module uses only classical algorithms: checksums, perceptual hashes, color histograms, HOG, NumPy statistics, and OpenCV K-Means. Every transformation returns an independent `Detections` collection and leaves its input unchanged.

```python
from klygo import datasets, mining

data = datasets.load("artifacts/datasets/train", format="yolo", verbose=False)
summary = mining.stats(data)
scores = mining.score(data)
diverse = mining.select(data, method="weighted", scores=scores, limit=500)
mining.compare(data, diverse).save("artifacts/mining/comparison.md")
diverse.export("artifacts/mining/dataset", format="yolo")
```

## Public APIs

- [`stats`](api/stats.md) reports image, box, class, co-occurrence, imbalance, and quality statistics.
- [`score`](api/score.md) ranks images by quality, rarity, class coverage, and box density.
- [`find_duplicates`](api/find_duplicates.md) groups exact and near-duplicate images across the dataset.
- [`deduplicate`](api/deduplicate.md) removes complete duplicate frames from a copy.
- [`cluster`](api/cluster.md) groups images with classical visual and annotation features.
- [`select`](api/select.md) selects images by class, quality, boxes, random sampling, quota, rarity, coverage, diversity, or weighted score.
- [`balance`](api/balance.md) reduces class imbalance without synthesizing images.
- [`compare`](api/compare.md) describes changes between two datasets.
- [`MiningStats`](api/MiningStats.md), [`MiningScores`](api/MiningScores.md), [`DuplicateReport`](api/DuplicateReport.md), [`ClusterResult`](api/ClusterResult.md), and [`MiningComparison`](api/MiningComparison.md) are serializable report types with `show()`, `to_dict()`, and `save()`.

Mining requires materialized `Detections`. Export remains the responsibility of `Detections.export()`, so mining does not duplicate dataset-writing logic.

# `mining.MiningStats`

`MiningStats` stores aggregate image, annotation, class, co-occurrence, area, imbalance, and optional quality statistics.

## Fields

- `images`, `boxes`, `empty_images`, `multilabel_images`: collection-level counts.
- `classes`, `present_classes`, `missing_classes`: declared class count, count of classes with annotations, and sorted list of declared classes with zero annotations.
- `images_per_class`, `boxes_per_class`: sorted per-class image and bounding-box counts.
- `box_area`: relative box area summary (`mean`, `median`, `min`, `max`, `p25`, `p50`, `p75`, `p90`).
- `cooccurrence`: pairwise class co-occurrence counts (empty when `cooccurrence=False` or fewer than two classes appear).
- `imbalance_ratio`, `box_imbalance_ratio`: max/min class frequency ratios by image count and by box count.
- `quality`, `failed_quality_images`: per-metric visual quality summaries (`brightness`, `contrast`, `sharpness`, `entropy`, `noise`, `exposure`) and count of unreadable/undecodable images skipped during quality evaluation.

```python
report = mining.stats(data)
utils.tabulate(report.to_dict())
report.save("artifacts/mining/stats.json")
```

See [`examples/mining/MiningStats.py`](../../../examples/mining/MiningStats.py) and [`test/mining/test_operations.py`](../../../test/mining/test_operations.py).

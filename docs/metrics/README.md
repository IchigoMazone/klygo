# Detection metrics

All metric functions accept either a Detection pair or a materialized Detections pair.
Supply both inputs to compute immediately. Omit both to create an immutable configuration.
Supplying only one raises TypeError. Streaming collections are rejected explicitly and are not consumed.

## Input and matching conventions

Inputs must have the same type category, the same unique frame identities, and equal image sizes for corresponding frames. Images align by identity, independent of collection order. Empty frames and empty collections are valid. Use the same original-image coordinate system on both sides. Labels match exactly and case-sensitively; numeric class IDs do not establish class equivalence. Inputs are not modified.

Within each frame, predictions are visited by descending confidence, preserving original box order for equal confidence. Each prediction takes the highest-IoU available same-label GT at or above the threshold; equal IoU uses GT input order. Each box participates in at most one match. Confusion matrices deliberately use geometry-only matching.

Precision, recall, F1, and mean IoU return zero for zero denominators. AP without GT and ECE without predictions return None. Threshold lists must be non-empty, unique, finite, and in [0, 1]; results are sorted ascending. No implicit confidence filtering occurs. AP on one image describes only that image.

Reports contain ordinary dictionaries, lists, tuples (frame identities), numbers, and None. They can be saved as JSON with files.save; JSON converts identity tuples to arrays. Display each per-image or per-class row with utils.tabulate(row); the current table helper accepts mappings, not lists.

## Operations

- [precision_recall](api/precision_recall.md): Dictionary: `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`.
- [map_score](api/map_score.md): Dictionary: `map`, optional `map_50` and `map_75` when those thresholds are requested, `per_class`, and `per_iou`.
- [confusion_matrix](api/confusion_matrix.md): Dictionary: `labels`, `background_index`, and `matrix` as a list of count rows.
- [match_boxes](api/match_boxes.md): List of frame dictionaries: `identity`, `matches`, `unmatched_pred`, `unmatched_target`.
- [per_class](api/per_class.md): List of rows: `label`, `support` (GT count), `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`, `ap`.
- [per_image](api/per_image.md): List of rows: `identity`, `targets`, `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`.
- [error_analysis](api/error_analysis.md): Dictionary: `counts` and `details`, both keyed by error category.
- [size_breakdown](api/size_breakdown.md): List of rows with `size`, `support`, `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`, `ap`.
- [calibration](api/calibration.md): Dictionary: `ece`, `bins`, `confidence`.
- [threshold_sweep](api/threshold_sweep.md): List of rows: `threshold`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`.
- [iou_sweep](api/iou_sweep.md): List of rows: `iou`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`.

## Evaluation

- [compose](api/compose.md): group deferred configurations.
- [evaluate](api/evaluate.md): execute a reusable group.

## Quick start

```python
from klygo import metrics, outputs, utils

pred = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9),
     outputs.Box((40, 40, 50, 50), label="cat", score=0.4)],
    source_id="example", image_size=(100, 100),
)
target = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="example", image_size=(100, 100),
)

evaluator = metrics.compose(metrics.precision_recall(), metrics.map_score(), metrics.per_image())
report = metrics.evaluate(pred, target, evaluator)
for row in report["per_image"]:
    utils.tabulate(row)
```

## Guides

[Tutorial overview](../../tutorials/metrics/Document_Metrics_Overview.ipynb) · [Tests](../../test/metrics/test_api.py)

## CPU and CUDA execution

All eleven metric operations accept same-device Detection pairs or same-device
materialized Detections pairs. CPU uses NumPy; CUDA uses CuPy selected from the
input device. Different devices raise ValueError without implicitly moving or
mutating either input. Use outputs.group(pred, target).cuda(0) or .cpu() first.
Individual collection methods remain supported. Detection frames inherit the
storage device of their owning collection.

IoU, stable ordering, matching, AP, confusion counts, calibration, and size/error
statistics are computed on the selected backend. Matching preserves descending
confidence and original-order ties, with first-GT tie breaking for equal IoU.
CUDA matching and AP envelopes use deterministic kernels; arithmetic uses
float64 so float16/float32 storage does not overflow geometric calculations.
Composed metrics share IoU and matching caches. Reports are converted to ordinary
Python data at the final report boundary; images and annotation devices are
unchanged. Label strings and source identities remain host metadata.

Install optional CUDA support with `pip install "klygo[gpu-outputs]"`. GPU does
not guarantee lower latency for small workloads: transfer, kernel-launch, and
report-conversion overhead may dominate. No device argument or new execution
mode is added to metrics.

[Group API](../outputs/api/group.md) · [Real GPU parity tests](../../test/metrics/test_gpu.py)

## Optional progress

Processing APIs accept keyword-only `verbose=False`. Enable it for large datasets; `compose()` remains a silent configuration builder. Composed evaluation uses one shared progress bar.

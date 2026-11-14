# metrics.error_analysis

```python
metrics.error_analysis(pred=..., target=..., *, iou=0.5, bg_iou=0.1)
```

## Contract and parameters

Partition false positives into five error types and list missed GT boxes.

## Result

Dictionary: `counts` and `details`, both keyed by error category. Every FP is assigned once, in this priority order: `duplicate` (same-label overlap at IoU >= iou with an already matched GT), `classification` (different-label overlap >= iou), `localization` (same-label overlap in [bg_iou, iou)), `classification_localization` (different-label overlap in that interval), then `background`. `missed` separately lists all unmatched GT, so an FP and a missed GT may describe the same failed object. Details contain `identity` and `pred_index` or `target_index`. Requires bg_iou < iou. This is TIDE-inspired categorization, not TIDE delta-AP attribution.

## Example

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

result = metrics.error_analysis(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.error_analysis())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/error_analysis.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

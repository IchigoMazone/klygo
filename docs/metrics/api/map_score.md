# metrics.map_score

```python
metrics.map_score(pred=..., target=..., *, thresholds=None)
```

## Contract and parameters

Compute class-wise 101-point interpolated AP and mean AP.

## Result

Dictionary: `map`, optional `map_50` and `map_75` when those thresholds are requested, `per_class`, and `per_iou`. Each class has `label`, `support`, `ap`, and `ap_by_iou` rows (`iou`, `ap`). Each IoU row has `iou`, `map`. AP uses 101 recall points and the precision envelope; classes without GT have AP `None` and are excluded from mAP. Classes with GT but no predictions have AP zero. Predictions are ranked by confidence across all images within each class. The default grid is 0.50 through 0.95 in steps of 0.05. This does not implement COCO crowd, ignore, or max-detections rules.

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

result = metrics.map_score(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.map_score())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/map_score.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

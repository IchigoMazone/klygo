# metrics.threshold_sweep

```python
metrics.threshold_sweep(pred=..., target=..., *, iou=0.5, thresholds=None)
```

## Contract and parameters

Return pooled metrics after retaining scores at or above each threshold.

## Result

List of rows: `threshold`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`. Each threshold retains predictions with confidence >= threshold. Default: 0.00 through 1.00 in steps of 0.05. The matching order guarantees that removing a lower-confidence suffix does not change retained matches.

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

result = metrics.threshold_sweep(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.threshold_sweep())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/threshold_sweep.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

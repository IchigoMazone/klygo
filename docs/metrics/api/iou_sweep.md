# metrics.iou_sweep

```python
metrics.iou_sweep(pred=..., target=..., *, thresholds=None)
```

## Contract and parameters

Return pooled metrics at each requested IoU threshold.

## Result

List of rows: `iou`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`. Default IoU grid: 0.50 through 0.95 in steps of 0.05. Each threshold performs independent one-to-one matching.

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

result = metrics.iou_sweep(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.iou_sweep())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/iou_sweep.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

# metrics.per_class

```python
metrics.per_class(pred=..., target=..., *, iou=0.5)
```

## Contract and parameters

Return counts, precision, recall, F1, mean IoU, and AP per label.

## Result

List of rows: `label`, `support` (GT count), `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`, `ap`. AP is calculated at the selected IoU. Includes labels appearing on either side; AP is `None` for labels without GT.

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

result = metrics.per_class(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.per_class())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/per_class.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

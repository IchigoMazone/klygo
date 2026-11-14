# metrics.per_image

```python
metrics.per_image(pred=..., target=..., *, iou=0.5)
```

## Contract and parameters

Return counts, precision, recall, F1, and mean IoU per frame.

## Result

List of rows: `identity`, `targets`, `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`. Rows follow prediction-frame order. A single Detection produces a one-row list. Empty collections produce an empty list.

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

result = metrics.per_image(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.per_image())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/per_image.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

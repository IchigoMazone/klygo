# metrics.precision_recall

```python
metrics.precision_recall(pred=..., target=..., *, iou=0.5, verbose=False)
```

## Contract and parameters

Compute pooled precision, recall, F1, counts, and matched mean IoU.

## Result

Dictionary: `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`. Counts are pooled over frames; mean IoU includes true-positive pairs only.

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

result = metrics.precision_recall(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.precision_recall())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/precision_recall.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

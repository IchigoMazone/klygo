# metrics.confusion_matrix

```python
metrics.confusion_matrix(pred=..., target=..., *, iou=0.5, verbose=False)
```

## Contract and parameters

Compute geometry-matched counts with GT rows and prediction columns.

## Result

Dictionary: `labels`, `background_index`, and `matrix` as a list of count rows. Rows are GT; columns are predictions. The last label is `None`, identifying background without colliding with a real string label. Matching is geometry-only and may pair different labels. Prediction-only boxes populate the background row; missing GT boxes populate the background column. This matching differs from class-aware precision/recall.

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

result = metrics.confusion_matrix(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.confusion_matrix())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/confusion_matrix.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

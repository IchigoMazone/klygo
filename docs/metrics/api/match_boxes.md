# metrics.match_boxes

```python
metrics.match_boxes(pred=..., target=..., *, iou=0.5, verbose=False)
```

## Contract and parameters

Return frame-local matched indices and unmatched prediction/target indices.

## Result

List of frame dictionaries: `identity`, `matches`, `unmatched_pred`, `unmatched_target`. Each match contains `pred_index`, `target_index`, `iou`. Indices refer to original frame-local box order, not class IDs or box UIDs.

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

result = metrics.match_boxes(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.match_boxes())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/match_boxes.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

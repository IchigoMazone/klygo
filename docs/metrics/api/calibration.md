# metrics.calibration

```python
metrics.calibration(pred=..., target=..., *, iou=0.5, bins=10, verbose=False)
```

## Contract and parameters

Return confidence-bin statistics, TP/FP scores, and detection ECE.

## Result

Dictionary: `ece`, `bins`, `confidence`. Each bin has `lower`, `upper`, `count`, `confidence` (mean score), `accuracy` (TP fraction). Empty-bin means are `None`. Bins are left-inclusive/right-exclusive, except the last includes confidence 1. `confidence` contains `tp` and `fp` score lists. ECE is the count-weighted absolute difference between bin mean confidence and accuracy; it is `None` without predictions. TP/FP use class-aware one-to-one matching; missed GT do not contribute to ECE.

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

result = metrics.calibration(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.calibration())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/calibration.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

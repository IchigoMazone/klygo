# metrics.f1_score

```python
metrics.f1_score(pred=..., target=..., *, precision=None, recall=None, iou=0.5, verbose=False)
```

## Contract and parameters

Compute F1 score from precision and recall. Supply `precision` and `recall` in `[0, 1]` for direct calculation, supply `pred` and `target` for pooled detection evaluation, or omit all inputs to build a deferred metric configuration for `compose(...)` / `evaluate(...)`.

## Result

Dictionary: `f1`, `precision`, `recall`. When `precision + recall == 0`, `f1` returns `0.0`.

## Example

```python
from klygo import metrics, outputs

pred = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9),
     outputs.Box((40, 40, 50, 50), label="cat", score=0.4)],
    source_id="example", image_size=(100, 100),
)
target = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="example", image_size=(100, 100),
)

print(metrics.f1_score(precision=0.8, recall=0.5))
print(metrics.f1_score(pred, target))

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.f1_score())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/f1_score.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

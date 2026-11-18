# metrics.evaluate

```python
metrics.evaluate(pred, target, evaluator=None, *, verbose=False)
```

## Contract and parameters

Evaluate matching Detection pairs or materialized Detections collections.

Parameters
----------
pred, target : Detection or Detections
    Same-type inputs with identical frame identities and image sizes.
evaluator : configuration, optional
    A compose result. Defaults to precision_recall and map_score.

Returns
-------
dict
    Results keyed by the selected metric names.

Raises
------
TypeError
    Inputs or evaluator have unsupported types.
ValueError
    Frame identities or sizes differ, or an input is streaming.

## Result

Dictionary keyed by metric names, containing each metric result. The evaluator must be a compose result. Omitting it selects precision_recall() and map_score(). A shared execution context caches pairwise IoU and matching for the duration of the call.

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

evaluator = metrics.compose(metrics.precision_recall(), metrics.map_score())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/evaluate.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

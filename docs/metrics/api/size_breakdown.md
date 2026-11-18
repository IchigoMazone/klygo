# metrics.size_breakdown

```python
metrics.size_breakdown(pred=..., target=..., *, iou=0.5, verbose=False)
```

## Contract and parameters

Return small/medium/large metrics using COCO area boundaries.

## Result

List of rows with `size`, `support`, `predictions`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1`, `mean_iou`, `ap`. Small area < 32 squared; medium area >= 32 squared and < 96 squared; large area >= 96 squared. TP and FN use GT area; unmatched predictions use prediction area. Matching is performed globally before grouping. AP is the mean per-class AP at the chosen IoU among labels with GT in the size group; labels without GT are excluded. These are COCO area boundaries, not the full COCO area evaluator.

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

result = metrics.size_breakdown(pred, target)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.size_breakdown())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/size_breakdown.py) · [Tests](../../../test/metrics/test_api.py)

CPU/GPU: inputs must share one device; execution follows their NumPy/CuPy storage. Results remain ordinary Python data. See [device conventions](../README.md#cpu-and-cuda-execution).

## Progress output

`verbose=False` keeps execution silent by default. Set `verbose=True` to show throttled progress (at most every 0.2 seconds, plus initial/final display). Results and input mutation rules are unchanged.
Deferred metric configurations do not retain the flag or display progress. For composed evaluation, use `metrics.evaluate(pred, target, evaluator, verbose=True)`; one shared bar covers preparation, matching, and metric completion. Progress counts CPU-side work steps, not GPU execution latency.

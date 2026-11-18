# metrics.compose

```python
metrics.compose(*metrics)
```

## Contract and parameters

Group one or more deferred metric configurations without executing them.

Parameters
----------
*metrics : metric configuration
    Results of metric functions called without pred and target.

Returns
-------
configuration
    Immutable reusable group for evaluate.

Raises
------
TypeError
    An argument is not a metric configuration.
ValueError
    The group is empty or contains duplicate metric names.

## Result

An immutable reusable configuration group. Arguments must be deferred metric configurations. Empty groups and duplicate metric names raise ValueError. No data is evaluated or stored in the group.

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

evaluator = metrics.compose(metrics.precision_recall(), metrics.per_image())
print(metrics.evaluate(pred, target, evaluator))
```

## Related resources

[Module conventions](../README.md) · [Standalone example](../../../examples/metrics/compose.py) · [Tests](../../../test/metrics/test_api.py)

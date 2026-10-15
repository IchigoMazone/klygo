# `datasets.repartition`

## Contract

Replace an existing dataset split layout using the same deterministic engine as `partition`.

## Parameters

- `source`, `target`: input and output dataset paths.
- `ratios`: new train/validation/test ratios.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

The created `pathlib.Path`.

## Errors and edge cases

The target must be distinct unless overwrite is enabled. Duplicate basenames from old splits are flattened with their relative paths.

## AI usage guidance

Choose this name when the source already has train/val/test folders; it makes the operation's intent explicit.

## Example

See [`repartition.py`](../../../examples/datasets/repartition.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
output = datasets.repartition("old", "new.zip", (0.7, 0.2, 0.1))
```

# `datasets.split`

## Contract

Create multiple ZIP datasets by individual class, explicit class groups, or deterministic ratios.

## Parameters

- `source`: dataset directory or archive.
- `output_dir`: directory receiving ZIP parts.
- `by_class`: create one part per class.
- `class_groups`: explicit lists of class names.
- `ratios`: two or more values summing to one.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

An ordered `list[pathlib.Path]` containing archives that were created.

## Errors and edge cases

One split strategy is required. Unknown class-group names and pre-existing outputs are rejected. Class-based archives omit images without matching annotations.

## AI usage guidance

Use `partition` for train/val/test folders inside one dataset; use `split` when independent archives are required.

## Example

See [`split.py`](../../../examples/datasets/split.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
parts = datasets.split("dataset", "parts", ratios=(0.8, 0.2))
```

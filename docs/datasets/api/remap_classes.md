# `datasets.remap_classes`

## Contract

Rename classes or move their numeric IDs while copying images and rewriting YOLO label files.

## Parameters

- `source`, `target`: input and output dataset paths.
- `class_map`: string keys select names, integer keys select IDs; string values rename and integer values move IDs.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

The created `pathlib.Path`.

## Errors and edge cases

Unknown names, out-of-range IDs, negative destination IDs, and invalid key/value
types are rejected. Mapping different class names to the same destination ID is
also rejected rather than silently dropping a name. Sparse IDs produce a mapping
under `names` and set `nc` to highest ID plus one. ZIP metadata has no
machine-specific path.

## AI usage guidance

One entry performs either rename or ID movement. Use two passes if the same class must both move and rename.

## Example

See [`remap_classes.py`](../../../examples/datasets/remap_classes.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
output = datasets.remap_classes("data", "mapped", {"car": "vehicle", 1: 4})
```

# `datasets.unpartition`

## Contract

Flatten split image/label trees into one `images` and one `labels` directory while preserving unique names.

## Parameters

- `source`: partitioned directory or archive.
- `target`: flat directory or ZIP destination.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

The created `pathlib.Path`.

## Errors and edge cases

Relative path components are joined with underscores so equal basenames from separate splits do not overwrite each other.

## AI usage guidance

Use before repartitioning with another tool or when a consumer expects a single image directory.

## Example

See [`unpartition.py`](../../../examples/datasets/unpartition.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
flat = datasets.unpartition("partitioned.zip", "flat")
```

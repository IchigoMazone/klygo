# `datasets.partition`

## Contract

Create deterministic train, validation, and optional test folders from a flat or previously structured dataset.

## Parameters

- `source`: directory or archive.
- `target`: output directory or ZIP path.
- `ratios`: one to three float values.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

The created `pathlib.Path`.

## Errors and edge cases

Validation rejects missing sources, existing targets without overwrite, and
non-finite or out-of-range ratios. The fixed random seed makes repeated results
reproducible. Relative image paths remain distinct after flattening; ZIP
metadata omits machine-specific absolute paths.

## AI usage guidance

Use three ratios when exact train/val/test intent matters. Use `repartition` to communicate that an existing split layout is being replaced.

## Example

See [`partition.py`](../../../examples/datasets/partition.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
output = datasets.partition("raw", "ready", (0.8, 0.1, 0.1))
```

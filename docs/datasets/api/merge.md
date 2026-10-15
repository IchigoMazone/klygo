# `datasets.merge`

## Contract

Combine multiple datasets into one ZIP, unifying class names in first-seen order and rewriting every local class ID.

## Parameters

- `sources`: list of dataset directories or archives.
- `output_path`: destination ZIP.
- `overwrite`, `verbose`: replacement and progress controls.

## Returns

The created ZIP `pathlib.Path`.

## Errors and edge cases

At least two sources are required. Source-index prefixes prevent filename collisions. The output must end in `.zip`.

## AI usage guidance

Use this instead of `archive.merge`; dataset merge understands `data.yaml` and remaps annotations semantically.

## Example

See [`merge.py`](../../../examples/datasets/merge.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
merged = datasets.merge(["one", "two.zip"], "merged.zip")
```

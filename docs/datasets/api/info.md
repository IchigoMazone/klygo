# `datasets.info`

## Contract

Inspect a directory or archive without modifying it. Counts images, available labels, class names, and recognized split folders.

## Parameters

- `source`: dataset directory or supported archive.
- `format`: dataset format to inspect (default `"yolo"`). Currently only `"yolo"` is supported.
- `verbose`: print the dataset statistics as a Rich table while returning the dictionary (default `False`).

## Returns

A dictionary containing `classes`, `class_ids`, `nc`, `total_images`,
`total_labels`, `unlabeled_images`, and `splits`. `classes` and `class_ids` are
parallel lists in ID order. For sparse IDs, `nc` is the highest ID plus one
rather than the length of `classes`.

## Errors and edge cases

Raises `FileNotFoundError` for missing sources/images and `ValueError` when
`data.yaml` has missing, duplicate, or invalid class names/IDs.

## AI usage guidance

Call this before destructive or expensive transformations to verify the input structure.

## Example

See [`info.py`](../../../examples/datasets/info.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
stats = datasets.info("dataset.zip")
print(stats["classes"], stats["total_images"])
```

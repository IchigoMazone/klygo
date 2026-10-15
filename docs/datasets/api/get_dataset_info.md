# `datasets.get_dataset_info`

## Contract

Inspect a directory or archive without modifying it. Counts images, available labels, class names, and recognized split folders.

## Parameters

- `source`: dataset directory or supported archive.

## Returns

A dictionary containing `classes`, `nc`, `total_images`, `total_labels`, `unlabeled_images`, and `splits`.

## Errors and edge cases

Raises `FileNotFoundError` for missing sources/images and `ValueError` when `data.yaml` has no class names.

## AI usage guidance

Call this before destructive or expensive transformations to verify the input structure.

## Example

See [`get_dataset_info.py`](../../../examples/datasets/get_dataset_info.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
info = datasets.get_dataset_info("dataset.zip")
print(info["classes"], info["total_images"])
```

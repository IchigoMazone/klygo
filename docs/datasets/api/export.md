# `datasets.export`

## Contract

Run a callable Klygo detector over media inputs and create either a YOLO detection dataset or a class-folder classification dataset.

## Parameters

- `model`: callable detector; `model(images, ...)` is used.
- `output_path`: destination directory.
- `format`: detection/YOLO or classification aliases.
- `source`: path or sequence of paths/images.
- `text_prompt`, `classes`: ordered class names; `classes` is the compatibility alias.
- `batch_size`, `threshold`, `overwrite`, `verbose`: prediction and output controls.
- extra keywords: permitted runtime metadata overrides passed to the model.

## Returns

The created directory `pathlib.Path`.

## Errors and edge cases

Classes must be unique, and at least one image is required. Unknown formats,
invalid thresholds/batch sizes, and a different number of predictions than
input images are rejected. With `overwrite=True`, the existing output remains
intact if validation, prediction, or writing fails. Classification folder names
are sanitized for filesystem safety.

## AI usage guidance

Use `datasets.export`. Classification uses `Detection.crop()` and does not rely on a model-level `crop` method.

## Example

See [`export.py`](../../../examples/datasets/export.py).

## Tests

See [`test_api.py`](../../../test/datasets/test_api.py).

## Complete executable example

```python
from klygo import datasets
output = datasets.export(
    model,
    "generated",
    "yolo",
    "images",
    classes=["car", "person"],
    threshold=0.4,
)
```

# `models.details`

```python
models.details(
    *,
    name: str,
    task: str,
    backend: str,
    library: str,
    **information: Any,
) -> box.Box
```

Create and validate the descriptive provenance block of a Klygo model metadata definition.

## Contract

Every Klygo model metadata definition describes its origin and runtime contract through four required keyword-only string fields (`name`, `task`, `backend`, `library`) plus any number of optional serializable attributes (`num_params`, `version`, `revision`, `author`, `license`, `description`).

Leading and trailing whitespace on required fields is stripped automatically. Optional keyword arguments whose value is `None` are omitted from the returned `Box` to keep serialized `klygo.json` and `Config` files clean.

## Parameters

- `name`: `str` *(keyword-only, required)* — human-readable model architecture name (e.g., `"YOLOv8n"`, `"Faster R-CNN"`, `"D-FINE Small"`).
- `task`: `str` *(keyword-only, required)* — primary computer vision task (e.g., `"Object-Detection"`).
- `backend`: `str` *(keyword-only, required)* — Klygo backend family responsible for execution (`"Ultralytics"`, `"Hugging Face"`, `"Torchvision"`, `"Keras"`, `"PyTorch"`).
- `library`: `str` *(keyword-only, required)* — upstream Python package providing the model (`"ultralytics"`, `"transformers"`, `"torchvision"`, `"keras_hub"`).
- `**information`: `Any` — optional descriptive fields (`num_params="3.2M"`, `license="Apache-2.0"`, etc.). Keys with `None` values are dropped.

## Returns

`box.Box` — validated descriptive metadata dictionary supporting dot and key access.

## Errors and edge cases

- Raises `TypeError` if any of `name`, `task`, `backend`, or `library` is not a `str`, or if positional arguments are passed.
- Raises `ValueError` if any of `name`, `task`, `backend`, or `library` is empty or whitespace-only.

## Example

See [`details.py`](../../../examples/models/details.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.details."""

from klygo import models

info = models.details(
    name="Faster R-CNN MobileNetV3-Large FPN",
    task="Object-Detection",
    backend="Torchvision",
    library="torchvision",
    num_params="19.4M",
    weights="COCO_V1",
    optional_note=None,  # Automatically omitted
)

print(info.name, info.backend, info.num_params)
print("optional_note" in info)
```

# `models.set_backend`

```python
models.set_backend(backend: str, engine: Optional[str] = None) -> None
```

Configure the process-wide default Klygo model backend and optional Keras execution engine through environment variables.

## Contract

Sets `os.environ["KLYGO_BACKEND"] = str(backend)`. When `engine` is provided, also sets `os.environ["KERAS_BACKEND"] = str(engine)` (for example `"torch"`, `"jax"`, or `"tensorflow"` when running Keras 3 / KerasHub models such as D-FINE / RT-DETR).

`set_backend()` does not import any deep-learning framework on its own, preserving Klygo's zero-overhead startup. Note that Keras 3 reads `KERAS_BACKEND` when `keras` is first imported in a Python process, so call `models.set_backend("Keras", engine="torch")` before importing `keras` or `keras_hub`.

## Parameters

- `backend`: `str` — target backend name (such as `"Ultralytics"`, `"Hugging Face"`, `"Torchvision"`, `"Keras"`, or `"PyTorch"`).
- `engine`: `str | None`, default `None` — optional Keras 3 backend engine (`"torch"`, `"tensorflow"`, or `"jax"`).

## Returns

`None`; mutates `os.environ` in-place.

## Errors and edge cases

`backend` and `engine` are coerced with `str(...)`. Passing `engine=None` leaves any existing `KERAS_BACKEND` variable unchanged.

## Example

See [`set_backend.py`](../../../examples/models/set_backend.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py).

## Complete executable example

```python
"""Executable example for klygo.models.set_backend."""

import os
from unittest.mock import patch
from klygo import models

with patch.dict(os.environ, {}, clear=False):
    models.set_backend("Keras", engine="torch")
    print(models.get_backend(), os.environ.get("KERAS_BACKEND"))
```

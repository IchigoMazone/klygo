# `models.get_backend`

```python
models.get_backend() -> str
```

Return the active Klygo model backend configured in the current process environment, or `"auto"` when unset.

## Contract

Reads `os.environ.get("KLYGO_BACKEND", "auto")` without importing optional AI frameworks. When `"auto"` is active, `BaseModel.backend` automatically inspects the wrapped model instance (`self.model`) or falls back to the model's `metadata.backend`.

## Parameters

None.

## Returns

`str` — the configured `KLYGO_BACKEND` string, or `"auto"` if no explicit environment override has been set.

## Errors and edge cases

Never raises; always returns a non-empty string.

## Example

See [`get_backend.py`](../../../examples/models/get_backend.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py).

## Complete executable example

```python
"""Executable example for klygo.models.get_backend."""

import os
from unittest.mock import patch
from klygo import models

with patch.dict(os.environ, {}, clear=False):
    os.environ.pop("KLYGO_BACKEND", None)
    print("Default:", models.get_backend())
    models.set_backend("Torchvision")
    print("Configured:", models.get_backend())
```

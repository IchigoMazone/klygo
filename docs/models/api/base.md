# `models.base`

```python
from klygo.models import base
```

Public submodule namespace defining [`BaseModel`](BaseModel.md), the Layer-1 abstract contract for all Klygo model wrappers.

## Contract

`klygo.models.base` is exported on the top-level `klygo.models` facade so callers and type checkers can reference `models.base.BaseModel` or inspect the Layer-1 implementation module directly. Its `__all__` inventory exports `["BaseModel"]`.

## Exported Symbols

- [`BaseModel`](BaseModel.md): Abstract base class providing lifecycle state tracking (`LOADING`, `READY`, `UNLOADED`), `metadata` and `flags` normalization, call-local runtime metadata context (`_use_metadata`), Rich terminal inspection (`info()`), and post-unload access guards.

## Returns

Importing or accessing `models.base` returns the `klygo.models.base` Python module object.

## Errors and edge cases

`models.base.BaseModel` is abstract and cannot be instantiated directly without implementing `predict`, `benchmark`, and `help`.

## Example

See [`base.py`](../../../examples/models/base.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py).

## Complete executable example

```python
"""Executable example for klygo.models.base."""

from klygo import models

print(models.base.__all__)
print(models.base.BaseModel is models.BaseModel)
print(issubclass(models.Detector, models.base.BaseModel))
```

# `models.suppress_warnings`

```python
models.suppress_warnings(func=None)
```

Temporarily silence Python warnings and noisy AI framework loggers as either a `with` context manager or a function decorator.

## Contract

When entered as a context manager (`with models.suppress_warnings():`) or applied as a decorator (`@models.suppress_warnings`), this helper:
1. Calls [`models.suppress_ai_warnings()`](suppress_ai_warnings.md) to set quiet environment flags and `ERROR`-level loggers for Hugging Face, Transformers, Accelerate, and PyTorch.
2. Enters `warnings.catch_warnings()` and applies `warnings.filterwarnings("ignore")`.
3. Restores the previous warning filter state cleanly upon exiting the `with` block or returning from the decorated function, even if an exception is raised.

## Parameters

- `func`: `Callable | None`, default `None` — when a callable is passed, returns a wrapped function that executes inside the warning-suppression context. When `None`, returns a reusable context manager instance (which itself can also be called as a decorator `@models.suppress_warnings()`).

## Returns

A context manager (`SuppressContext`) when `func is None`, or the wrapped callable when `func` is provided.

## Errors and edge cases

Exceptions raised inside the `with` block or decorated function propagate normally after restoring the original warning filters.

## Example

See [`suppress_warnings.py`](../../../examples/models/suppress_warnings.py) for a standalone executable script.

## Tests

See [`test_optional_dependencies.py`](../../../test/models/test_optional_dependencies.py).

## Complete executable example

```python
"""Executable example for klygo.models.suppress_warnings."""

import warnings
from klygo import models

with models.suppress_warnings():
    warnings.warn("This framework deprecation warning is silenced", UserWarning)


@models.suppress_warnings
def quiet_loader():
    warnings.warn("Also silenced inside decorated function", RuntimeWarning)
    return "ready"


print(quiet_loader())
```

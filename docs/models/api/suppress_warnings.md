# `models.suppress_warnings`

```python
models.suppress_warnings(func=None)
```

Temporarily silence Python warnings in a context or decorator.

**Parameters:** func optionally decorates a callable; omitted func returns a context manager.

**Returns and effects:** Returns a decorated callable or context manager; warning filters are restored on exit.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
import warnings
with models.suppress_warnings():
    warnings.warn("hidden", UserWarning)
print("warning scope exited")
```

Run [the standalone example](../../../examples/models/suppress_warnings.py) and see the [module index](../README.md) for related APIs.

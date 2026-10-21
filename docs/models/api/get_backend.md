# `models.get_backend`

```python
models.get_backend() -> str
```

Read the selected backend or auto fallback.

**Parameters:** No parameters.

**Returns and effects:** Returns the backend string; defaults to auto.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.get_backend())
```

Run [the standalone example](../../../examples/models/get_backend.py) and see the [module index](../README.md) for related APIs.

# `models.details`

```python
models.details(*, name: str, task: str, backend: str, library: str, **information: Any) -> box.box.Box
```

Build validated human-readable model details.

**Parameters:** name, task, backend, and library are required nonempty strings; extra keyword information is included unless None.

**Returns and effects:** Returns a Box; required fields with wrong type raise TypeError and empty strings raise ValueError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
info = models.details(name="Demo", task="Detection", backend="Custom", library="local", version="1")
print(info.name, info.version)
```

Run [the standalone example](../../../examples/models/details.py) and see the [module index](../README.md) for related APIs.

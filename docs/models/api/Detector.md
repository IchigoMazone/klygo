# `models.Detector`

```python
models.Detector(metadata: Optional[Dict[str, Any]] = None, backend: Optional[str] = None, model_id: Optional[str] = None, **kwargs) -> None
```

Build a detector wrapper for a concrete backend implementation.

**Parameters:** metadata, backend, and model_id configure the wrapper; subclasses implement forward.

**Returns and effects:** Returns a detector wrapper. Calling forward without a backend implementation is not useful.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
detector = models.Detector()
print(detector.task, isinstance(detector, models.BaseModel))
```

Run [the standalone example](../../../examples/models/Detector.py) and see the [module index](../README.md) for related APIs.

# `models.BaseModel`

```python
models.BaseModel(metadata: Optional[Dict[str, Any]] = None, backend: Optional[str] = None, model_id: Optional[str] = None, task: Optional[str] = None, **kwargs) -> None
```

Define an abstract contract for model wrappers.

**Parameters:** metadata, backend, model_id, and task initialize model identity and settings; subclasses implement abstract methods.

**Returns and effects:** Returns a model instance only from a concrete subclass.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(issubclass(models.Detector, models.BaseModel))
```

Run [the standalone example](../../../examples/models/BaseModel.py) and see the [module index](../README.md) for related APIs.

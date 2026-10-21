# `models.configure`

```python
models.configure(model_id: str, metadata: Optional[Mapping[str, Any]] = None, flags: Union[Mapping[str, bool], Sequence[str], NoneType] = None, priority: Optional[Mapping[str, Any]] = None, details: Optional[Mapping[str, Any]] = None, implementation: Union[type, str, NoneType] = None, **kwargs) -> box.box.Box
```

Bind an identifier and parameter values to metadata.

**Parameters:** model_id identifies the model; metadata, flags, priority, details, and implementation define its schema; kwargs set group values.

**Returns and effects:** Returns independent Box metadata. Undeclared or ambiguous override keys raise ValueError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
definition = models.configure("demo", flags=("model", "post"), model_device="cpu", post_threshold=0.5)
print(definition.model.device, definition.post.threshold)
```

Run [the standalone example](../../../examples/models/configure.py) and see the [module index](../README.md) for related APIs.

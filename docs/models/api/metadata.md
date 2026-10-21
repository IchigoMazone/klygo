# `models.metadata`

```python
models.metadata(flags: Union[Mapping[str, bool], Sequence[str]], priority: Optional[Mapping[str, Any]] = None, details: Optional[Mapping[str, Any]] = None) -> box.box.Box
```

Combine flags, priority aliases, and descriptive model details.

**Parameters:** flags declares groups; priority maps unprefixed parameter names; details describes the model.

**Returns and effects:** Returns a Box with flags, priority, details, and class keys; invalid details raise TypeError or ValueError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
definition = models.metadata(models.flags("model"), details=models.details(name="Demo", task="Detection", backend="Custom", library="local"))
print(definition.flags, definition.details.name)
```

Run [the standalone example](../../../examples/models/metadata.py) and see the [module index](../README.md) for related APIs.

# `models.flags`

```python
models.flags(*groups: str, **permissions: bool) -> box.box.Box
```

Declare parameter groups and predict-time override permissions.

**Parameters:** Use positional group names or keyword bool permissions; do not mix forms.

**Returns and effects:** Returns a Box of bool permissions; mixed forms raise TypeError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.flags(model=False, post=True))
```

Run [the standalone example](../../../examples/models/flags.py) and see the [module index](../README.md) for related APIs.

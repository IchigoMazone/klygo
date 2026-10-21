# `models.base`

```python
models.base (module)
```

Access the abstract BaseModel implementation module.

**Parameters:** Module namespace; BaseModel is abstract.

**Returns and effects:** Returns the module; concrete subclasses must implement predict, benchmark, and help.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.base.BaseModel is models.BaseModel)
```

Run [the standalone example](../../../examples/models/base.py) and see the [module index](../README.md) for related APIs.

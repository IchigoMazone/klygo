# `models.utils`

```python
models.utils (module)
```

Access model metadata and warning helper functions.

**Parameters:** Module namespace; helpers normalize flags and metadata.

**Returns and effects:** Returns the module; invalid flag declarations raise TypeError or ValueError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
print(models.utils.normalize_flags(("model", "post")))
```

Run [the standalone example](../../../examples/models/utils.py) and see the [module index](../README.md) for related APIs.

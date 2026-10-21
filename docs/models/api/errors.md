# `models.errors`

```python
models.errors (module)
```

Access model-specific exception classes.

**Parameters:** Module namespace containing model state, configuration, device, and dtype exceptions.

**Returns and effects:** Returns the module; catch ModelFrameworkError for the shared exception base.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
try:
    raise models.errors.InvalidConfigError("bad setting")
except models.errors.ModelFrameworkError as error:
    print(type(error).__name__)
```

Run [the standalone example](../../../examples/models/errors.py) and see the [module index](../README.md) for related APIs.

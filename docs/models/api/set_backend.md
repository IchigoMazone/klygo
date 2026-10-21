# `models.set_backend`

```python
models.set_backend(backend: str, engine: Optional[str] = None) -> None
```

Set the default backend through environment variables.

**Parameters:** backend names the runtime; engine optionally sets the Keras engine.

**Returns and effects:** Returns None and sets process environment variables.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
import os
from unittest.mock import patch
with patch.dict(os.environ, {}, clear=False):
    models.set_backend("custom")
    print(models.get_backend())
```

Run [the standalone example](../../../examples/models/set_backend.py) and see the [module index](../README.md) for related APIs.

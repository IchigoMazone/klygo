# `models.suppress_ai_warnings`

```python
models.suppress_ai_warnings() -> None
```

Configure quiet logging for optional AI frameworks without importing them.

**Parameters:** No parameters.

**Returns and effects:** Returns None and sets logging levels and process environment flags.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
models.suppress_ai_warnings()
print("logging configured")
```

Run [the standalone example](../../../examples/models/suppress_ai_warnings.py) and see the [module index](../README.md) for related APIs.

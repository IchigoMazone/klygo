# `models.load`

```python
models.load(model: Union[str, Any], **kwargs) -> klygo.models.base.BaseModel
```

Resolve a configured model or return an existing model wrapper.

**Parameters:** model accepts a wrapper, loaded `Config`, metadata mapping, registry name, or supported model source; kwargs override declared groups. Configuration file paths must be read first with `config.load(path)` or `Config(path).read()`.

**Returns and effects:** Returns a BaseModel wrapper. Invalid sources or missing implementation metadata raise ValueError or backend-specific errors.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
model = models.Detector()
print(models.load(model) is model)
```

Run [the standalone example](../../../examples/models/load.py) and see the [module index](../README.md) for related APIs.


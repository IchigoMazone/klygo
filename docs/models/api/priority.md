# `models.priority`

```python
models.priority(flags: Union[Mapping[str, bool], Sequence[str]], **parameters: Any) -> box.box.Box
```

Assign unprefixed parameter names to flagged groups.

**Parameters:** flags declares group names; keyword arguments list priority names by destination group.

**Returns and effects:** Returns a Box mapping. Unknown groups or duplicate names raise ValueError.

```python
import sys

# Prevent metadata.py and base.py from shadowing Python support modules.
sys.path.pop(0)

from klygo import models
groups = models.flags("model", "post")
print(models.priority(groups, model=("device",), post=("threshold",)))
```

Run [the standalone example](../../../examples/models/priority.py) and see the [module index](../README.md) for related APIs.

# `models.flags`

```python
models.flags(*groups: str, **permissions: bool) -> box.Box
```

Declare model parameter groups and whether each group may be overridden at `model.predict()` runtime.

## Contract

Parameter groups partition a model's configuration into logical namespaces such as `"model"` (initialization/architecture options), `"processor"` (input preprocessing options), and `"post"` (postprocessing thresholds such as `threshold` and `iou`).

`models.flags()` supports two mutually exclusive calling styles:
1. **Keyword permissions (`models.flags(model=False, processor=True, post=True)`)**:
   Explicitly sets `True` for groups that `model.predict(**kwargs)` may override per call, and `False` for groups that are locked after `models.configure()` / `models.load()`.
2. **Positional group names (`models.flags("model", "post")`)**:
   Grants predict-time override permission (`True`) to every listed group.

## Parameters

- `*groups`: `str` — positional parameter group names (each assigned permission `True`). Cannot be combined with `**permissions`.
- `**permissions`: `bool` — keyword mapping of `<group_name>=<bool>` where `True` allows predict-time overrides and `False` locks the group after load.

## Returns

`box.Box` — normalized mapping of `{group_name: bool}` preserving declaration order.

## Errors and edge cases

- Raises `TypeError` if positional `*groups` and keyword `**permissions` are mixed in the same call, or if any permission value is not a `bool`.
- Raises `ValueError` if no groups are provided, if a group name is empty/whitespace, or if a group name collides with a reserved metadata key (`"backend"`, `"class"`, `"config"`, `"details"`, `"flags"`, `"model_id"`, `"implementation"`, `"num_params"`, `"option"`, `"priority"`, `"parameter_groups"`, `"profile"`, `"revision"`, `"task"`).

## Example

See [`flags.py`](../../../examples/models/flags.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.flags."""

from klygo import models

# Keyword permissions: lock 'model' at predict time, allow 'post' overrides
locked_schema = models.flags(model=False, processor=True, post=True)
print(locked_schema)

# Positional shorthand: all groups default to True
open_schema = models.flags("model", "post")
print(open_schema)
```

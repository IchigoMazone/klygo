# `models.metadata`

```python
models.metadata(
    flags: Mapping[str, bool] | Sequence[str],
    priority: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> box.Box
```

Combine parameter-group `flags`, unprefixed `priority` aliases, and descriptive `details` into a validated, reusable model metadata schema.

## Contract

`models.metadata()` defines the structural schema of a model without binding a specific `model_id`, implementation class, or parameter values (which is performed by [`models.configure()`](configure.md)).

The returned `box.Box` always contains four top-level keys:
- `"flags"`: normalized `dict[str, bool]` mapping each declared parameter group to its predict-time override permission.
- `"priority"`: normalized `dict[str, list[str]]` (or `None` when omitted).
- `"details"`: validated dictionary containing at least `"name"`, `"task"`, `"backend"`, and `"library"` (or `None` when omitted).
- `"class"`: initialized to `None` so [`models.configure()`](configure.md) can populate the implementation class later.

## Parameters

- `flags`: `Mapping[str, bool] | Sequence[str]` — parameter group declarations created with [`models.flags()`](flags.md) or passed as a mapping/sequence.
- `priority`: `Mapping[str, Any] | None`, default `None` — unprefixed parameter aliases grouped by destination flag, typically created with [`models.priority()`](priority.md).
- `details`: `Mapping[str, Any] | None`, default `None` — descriptive model provenance, typically created with [`models.details()`](details.md).

## Returns

`box.Box` — an independent metadata schema supporting both dictionary and attribute access.

## Errors and edge cases

- Raises `TypeError` if `flags` is a string/bytes, if flag permissions are not booleans, or if `details` is not a mapping.
- Raises `ValueError` if `flags` is empty, uses a reserved metadata key (`"backend"`, `"class"`, `"details"`, `"flags"`, `"model_id"`, etc.), if `priority` references an undeclared group or duplicate parameter name, or if `details` omits any required string field (`"name"`, `"task"`, `"backend"`, `"library"`).

## Example

See [`metadata.py`](../../../examples/models/metadata.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.metadata."""

from klygo import models

group_flags = models.flags(model=False, processor=True, post=True)
schema = models.metadata(
    flags=group_flags,
    priority=models.priority(group_flags, post=("threshold", "iou")),
    details=models.details(
        name="D-FINE Small",
        task="Object-Detection",
        backend="Keras",
        library="keras_hub",
        num_params="10.2M",
    ),
)

print(schema.flags)
print(schema.priority.post)
print(schema.details.name, schema.details.backend)
```

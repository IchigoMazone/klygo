# `models.priority`

```python
models.priority(
    flags: Mapping[str, bool] | Sequence[str],
    **parameters: Any,
) -> box.Box
```

Register unprefixed parameter names (such as `threshold` or `iou`) that resolve directly to their owning parameter group without requiring `<group>_` prefixes.

## Contract

By default, parameters passed to `models.configure()`, `models.load()`, or `Detector.predict()` must be prefixed with their group name (for example `post_threshold=0.4` or `model_device="cuda"`). `models.priority()` declares a curated set of high-frequency parameters for each group that callers can pass using their short, unprefixed names (`threshold=0.4`, `iou=0.7`).

Each priority parameter name must belong to **at most one** group across the entire schema so resolution is unambiguous. Parameters not listed in `priority` remain accessible via `<group>_<parameter>`.

## Parameters

- `flags`: `Mapping[str, bool] | Sequence[str]` — declared parameter groups (from [`models.flags()`](flags.md)) against which group names are validated.
- `**parameters`: `str | Iterable[str]` — keyword arguments where each key is a declared group name in `flags` and each value is a parameter name or sequence of parameter names (e.g., `post=("threshold", "iou")`).

## Returns

`box.Box` — mapping of `{group_name: list[str]}` containing deduplicated priority parameter names per group.

## Errors and edge cases

- Raises `ValueError` if a keyword group in `**parameters` is not present in `flags`.
- Raises `ValueError` if a parameter name is empty, starts/ends with `"_"`, or is assigned to more than one group.
- Raises `TypeError` if a group's parameter list is not a string or iterable of names.

## Example

See [`priority.py`](../../../examples/models/priority.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.priority."""

from klygo import models

group_flags = models.flags(model=False, processor=True, post=True)
aliases = models.priority(
    group_flags,
    model=("device",),
    post=("threshold", "iou"),
)
print(aliases)
```

# `utils.invert`

```python
utils.invert(
    values: Mapping[Any, Any],
) -> dict[Any, Any]
```

Return a new dictionary with every key and value exchanged.

## Contract

Iterates over `values.items()` in insertion order and builds a new dictionary `{value: key}`. The input mapping is never mutated.

To guarantee that inversion is lossless and strictly reversible (`utils.invert(utils.invert(m)) == dict(m)`), every value in `values` must be both hashable and unique. If two keys map to the same value (under Python equality and hash semantics, such as `1` and `True`), `invert()` raises `ValueError` immediately rather than silently overwriting the earlier key.

## Parameters

- `values`: `Mapping[Any, Any]` — source mapping whose keys and values should be swapped.

## Returns

`dict[Any, Any]` — a new dictionary mapping each original value to its corresponding original key.

## Errors and edge cases

- Raises `TypeError("values must be a mapping")` if `values` is not an instance of `collections.abc.Mapping`.
- Raises `TypeError("mapping values must be hashable")` if any value in `values` is unhashable (such as a `list`, `dict`, or `set`).
- Raises `ValueError(f"mapping values must be unique; duplicate value: {value!r}")` if two keys share the same value.

## AI usage guidance

Prefer `from klygo import utils` followed by `utils.invert(...)` when converting between `{class_name: class_id}` and `{class_id: class_name}` lookup tables across Klygo modules.

## Example

See [`invert.py`](../../../examples/utils/invert.py) for an executable example.

## Tests

See [`test_invert.py`](../../../test/utils/test_invert.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.utils.invert."""

from klygo import utils

classes = {"cat": 0, "dog": 1}
class_names = utils.invert(classes)
restored = utils.invert(class_names)

print("Inverted:", class_names)
print("Restored:", restored)

assert class_names == {0: "cat", 1: "dog"}
assert restored == classes

try:
    utils.invert({"a": 1, "b": 1})
except ValueError as error:
    print("Duplicate rejected:", error)
```

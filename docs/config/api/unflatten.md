# `config.unflatten`

```python
config.unflatten(flat_dict, sep=".")
```

Expand flat paths into nested dictionaries.

## Contract

Creates intermediate dictionaries and rejects structural conflicts rather than silently replacing scalar or mapping values.

## Parameters

- `flat_dict`: mapping of path strings to values.
- `sep`: non-empty path separator.

## Returns

A nested plain `dict`.

## Errors and edge cases

Raises `ValueError` for empty keys, separators, or conflicts such as both `a` and `a.b`; raises `TypeError` for non-mapping input.

## AI usage guidance

Validate externally supplied flat keys before merging them into trusted configuration state.

## Example

See [`unflatten.py`](../../../examples/config/unflatten.py).

## Tests

See [`test_structure.py`](../../../test/config/test_structure.py).

## Complete executable example

```python
from klygo import config

assert config.unflatten({"model.batch": 16}) == {"model": {"batch": 16}}
```

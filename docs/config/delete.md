# `config.delete`

```python
config.delete(config_data, key_path)
```

Delete one nested key from a mutable configuration in place.

## Contract

Returns whether removal occurred. Empty parent mappings remain intact so unrelated structure is never removed implicitly.

## Parameters

- `config_data`: mutable mapping or `Box`.
- `key_path`: nested path to remove.

## Returns

`True` when removed; `False` when absent.

## Errors and edge cases

Raises `TypeError` for immutable/non-mapping input and `ValueError` for an empty path.

## AI usage guidance

This is the intentionally mutating access operation. Copy the configuration first when mutation is undesirable.

## Example

See [`delete.py`](../../examples/config/delete.py).

## Tests

See [`test_access.py`](../../test/config/test_access.py).

## Complete executable example

```python
from klygo import config

settings = {"model": {"batch": 16}}
assert config.delete(settings, "model.batch")
assert settings == {"model": {}}
```

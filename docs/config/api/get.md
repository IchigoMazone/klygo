# `config.get`

```python
config.get(config_data, key_path, default=None)
```

Read one value through a dot-separated nested path.

## Contract

Traverses mappings and `Box` objects. If any component is absent, returns `default` without modifying data.

## Parameters

- `config_data`: configuration mapping or `Box`.
- `key_path`: non-empty path such as `model.optimizer.lr`.
- `default`: fallback for an absent path.

## Returns

The stored value, including explicit `None`, or the fallback.

## Errors and edge cases

Raises `ValueError` for an empty path. A non-mapping intermediate component makes the remaining path absent.

## AI usage guidance

Use `has` when absence must be distinguished from a stored value equal to the fallback.

## Example

See [`get.py`](../../../examples/config/get.py).

## Tests

See [`test_access.py`](../../../test/config/test_access.py).

## Complete executable example

```python
from klygo import config

settings = {"model": {"batch": 16}}
assert config.get(settings, "model.batch") == 16
assert config.get(settings, "model.epochs", 10) == 10
```

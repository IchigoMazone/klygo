# `config.has`

```python
config.has(config_data, key_path)
```

Check whether every component of a nested key path exists.

## Contract

Existence is sentinel-based, so a key storing `None` is still present.

## Parameters

- `config_data`: mapping or `Box` to inspect.
- `key_path`: non-empty dot-separated path.

## Returns

`bool` indicating path existence.

## Errors and edge cases

Raises `ValueError` for an empty path. Non-mapping intermediate values return `False`.

## AI usage guidance

Use `has` before required reads; do not infer existence from `get(..., None)`.

## Example

See [`has.py`](../../examples/config/has.py).

## Tests

See [`test_access.py`](../../test/config/test_access.py).

## Complete executable example

```python
from klygo import config

assert config.has({"model": {"name": None}}, "model.name")
assert not config.has({}, "model.name")
```

# `config.set`

```python
config.set(config_data, key_path, value)
```

Assign a nested value on a deep-copied configuration.

## Contract

Missing intermediate mappings are created. Existing non-mapping intermediates are replaced with mappings. The input remains unchanged.

## Parameters

- `config_data`: base mapping or `Box`.
- `key_path`: non-empty dot path.
- `value`: arbitrary value to copy and store.

## Returns

A new `box.Box`.

## Errors and edge cases

Raises `TypeError` for a non-mapping base and `ValueError` for an empty path.

## AI usage guidance

Use the returned object; `set` intentionally does not mutate the input like `delete` does.

## Example

See [`set.py`](../../../examples/config/set.py).

## Tests

See [`test_access.py`](../../../test/config/test_access.py).

## Complete executable example

```python
from klygo import config

settings = config.set({}, "model.optimizer.lr", 0.001)
assert settings.model.optimizer.lr == 0.001
```

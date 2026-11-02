# `config.items`

```python
config.items(config_data, flat=False)
```

Return configuration key/value pairs at the top level or leaf level.

## Contract

Preserves mapping traversal order. In flat mode, each key is a complete path and each value is its leaf value.

## Parameters

- `config_data`: mapping or `Box`.
- `flat`: flatten nested mappings when true.

## Returns

`list[tuple[str, Any]]`.

## Errors and edge cases

Raises `TypeError` for invalid data or non-boolean `flat`. Empty mappings are represented as leaf items in flat mode.

## AI usage guidance

Prefer this API for deterministic iteration instead of separately zipping `keys` and `values`.

## Example

See [`items.py`](../../../examples/config/items.py).

## Tests

See [`test_inspection.py`](../../../test/config/test_inspection.py).

## Complete executable example

```python
from klygo import config

assert config.items({"model": {"batch": 16}}, flat=True) == [("model.batch", 16)]
```

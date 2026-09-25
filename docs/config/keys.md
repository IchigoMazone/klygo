# `config.keys`

```python
config.keys(config_data, flat=False)
```

Return top-level keys or flattened leaf paths.

## Contract

Preserves mapping traversal order. Flat mode uses `config.flatten` and includes empty mappings as leaves.

## Parameters

- `config_data`: mapping or `Box`.
- `flat`: return complete leaf paths when true.

## Returns

`list[str]`.

## Errors and edge cases

Raises `TypeError` for invalid configuration data or a non-boolean `flat` value.

## AI usage guidance

Use flat keys for validation, comparison, environment mapping, and searchable configuration inventories.

## Example

See [`keys.py`](../../examples/config/keys.py).

## Tests

See [`test_inspection.py`](../../test/config/test_inspection.py).

## Complete executable example

```python
from klygo import config

assert config.keys({"model": {"batch": 16}}, flat=True) == ["model.batch"]
```

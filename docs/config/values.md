# `config.values`

```python
config.values(config_data, flat=False)
```

Return top-level or flattened leaf values.

## Contract

Order matches the corresponding result from `config.keys`; flat mode traverses nested mappings depth-first.

## Parameters

- `config_data`: mapping or `Box`.
- `flat`: return only leaf values when true.

## Returns

`list[Any]`.

## Errors and edge cases

Raises `TypeError` for invalid input or non-boolean `flat`. Values are returned by reference and are not deep-copied.

## AI usage guidance

Use `items` instead when values must remain associated with their paths.

## Example

See [`values.py`](../../examples/config/values.py).

## Tests

See [`test_inspection.py`](../../test/config/test_inspection.py).

## Complete executable example

```python
from klygo import config

assert config.values({"model": {"batch": 16}}, flat=True) == [16]
```

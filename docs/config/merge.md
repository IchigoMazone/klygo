# `config.merge`

```python
config.merge(*configs, deep=True)
```

Combine zero or more configurations from left to right.

## Contract

Later values take precedence. Deep mode recursively combines nested mappings; inputs are deep-copied and remain unchanged.

## Parameters

- `configs`: mappings or `Box` objects in precedence order.
- `deep`: recursively merge nested mappings when true.

## Returns

A new `box.Box`; no inputs produce an empty `Box`.

## Errors and edge cases

Raises `TypeError` for non-mapping inputs or a non-boolean `deep` value.

## AI usage guidance

Use for layered defaults, project settings, and runtime overrides. Place the highest-priority configuration last.

## Example

See [`merge.py`](../../examples/config/merge.py).

## Tests

See [`test_mapping.py`](../../test/config/test_mapping.py).

## Complete executable example

```python
from klygo import config

merged = config.merge({"model": {"batch": 16}}, {"model": {"epochs": 10}})
assert merged.model.to_dict() == {"batch": 16, "epochs": 10}
```

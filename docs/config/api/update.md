# `config.update`

```python
config.update(config_data, updates, deep=True)
```

Return an updated configuration without mutating either input.

## Contract

Deep mode recursively merges nested mappings. Shallow mode replaces each referenced top-level value in full.

## Parameters

- `config_data`: base mapping or `Box`.
- `updates`: replacement mapping or `Box`.
- `deep`: choose recursive or top-level replacement behavior.

## Returns

A new independent `box.Box`.

## Errors and edge cases

Raises `TypeError` for non-mapping inputs or non-boolean `deep`. Lists and scalar values are replaced, not merged.

## AI usage guidance

Prefer this over direct nested mutation when configuration values may be reused elsewhere.

## Example

See [`update.py`](../../../examples/config/update.py).

## Tests

See [`test_mapping.py`](../../../test/config/test_mapping.py).

## Complete executable example

```python
from klygo import config

original = {"model": {"batch": 16}}
updated = config.update(original, {"model": {"batch": 32}})
assert updated.model.batch == 32
assert original["model"]["batch"] == 16
```

# `config.validate`

```python
config.validate(config_data, required_keys=None)
```

Validate required paths and lightweight leaf contracts.

## Contract

Sequences require path existence. Mapping leaves may be types, tuples of types, predicates, literal expected values, or `None` for existence only.

## Parameters

- `config_data`: mapping or `Box` to validate.
- `required_keys`: sequence of paths, mapping schema, or `None`.

## Returns

Always `True` on success.

## Errors and edge cases

Raises `ValueError` for missing or invalid values and `TypeError` for malformed schemas. Predicate exceptions are not swallowed.

## AI usage guidance

Use type contracts for structural checks and predicates for domain constraints; validation does not coerce values.

## Example

See [`validate.py`](../../examples/config/validate.py).

## Tests

See [`test_validation.py`](../../test/config/test_validation.py).

## Complete executable example

```python
from klygo import config

settings = {"model": {"batch": 16}}
assert config.validate(settings, {"model.batch": int})
assert config.validate(settings, {"model.batch": lambda value: value > 0})
```

# `validators.validate_count`

```python
validators.validate_count(
    value: object,
    name: str,
    minimum: int = 1,
) -> None
```

Raise `TypeError` or `ValueError` when an argument is not a valid integer count greater than or equal to a minimum value.

## Contract

1. Checks whether `value` is an integer while explicitly excluding booleans (`isinstance(value, bool) or not isinstance(value, int)`). When invalid, raises `TypeError(f"{name} must be int, got {type(value).__name__}")`.
2. Checks whether `value >= minimum`. When below the acceptable threshold, raises `ValueError(f"{name} must be an integer >= {minimum}")`.
3. Returns `None` when validation succeeds.

## Parameters

- `value`: `object` — caller-supplied value to validate.
- `name`: `str` — parameter name included in raised error messages.
- `minimum`: `int`, default `1` — minimum acceptable integer threshold (e.g. `1` for batch or stride, `0` for warmup).

## Returns

`None` — returns `None` when `value` is a valid integer count.

## Errors and edge cases

- Raises `TypeError` with message `"{name} must be int, got {actual_type}"` when `value` is not an `int` or is a `bool`.
- Raises `ValueError` with message `"{name} must be an integer >= {minimum}"` when `value < minimum`.

## AI usage guidance

Prefer `from klygo import validators` followed by `validators.validate_count(...)` at public function boundaries whenever checking count, batch, iteration, stride, or dimension parameters instead of writing custom integer guard checks.

## Example

See [`validate_count.py`](../../../examples/validators/validate_count.py) for an executable example.

## Tests

See [`test_validate_count.py`](../../../test/validators/test_validate_count.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.validators.validate_count."""

from klygo import validators

assert validators.validate_count(4, "batch") is None
assert validators.validate_count(0, "warmup", minimum=0) is None

try:
    validators.validate_count(True, "batch")
except TypeError as error:
    assert "batch must be int, got bool" in str(error)

try:
    validators.validate_count(0, "batch")
except ValueError as error:
    assert "batch must be an integer >= 1" in str(error)
```

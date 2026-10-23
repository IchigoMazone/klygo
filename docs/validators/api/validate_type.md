# `validators.validate_type`

```python
validators.validate_type(
    value: object,
    expected: type | tuple[type, ...],
    name: str,
) -> None
```

Raise `TypeError` when an argument does not match the expected type or tuple of types.

## Contract

Evaluates `isinstance(value, expected)` at runtime. When the check succeeds, the function returns `None` without side effects. When the check fails, it formats the expected type name(s) using `expected.__name__` (for a single type) or `", ".join(t.__name__ for t in expected)` (for a tuple of types) and raises `TypeError(f"{name} must be {expected_name}, got {type(value).__name__}")`.

Because validation relies on standard `isinstance` semantics, instances of subclasses of any type in `expected` are accepted (for example, `bool` is a subclass of `int` in Python, and `pathlib.Path` satisfies `os.PathLike`).

## Parameters

- `value`: `object` — caller-supplied value to validate.
- `expected`: `type | tuple[type, ...]` — a single class/type or a tuple of classes/types accepted by `isinstance`.
- `name`: `str` — parameter name included at the start of the raised `TypeError` message.

## Returns

`None` — returns `None` when `value` is an instance of `expected`.

## Errors and edge cases

- Raises `TypeError` with message `"{name} must be {expected_name}, got {actual_type}"` when `value` is not an instance of `expected`.
- Raises `TypeError` from Python's built-in `isinstance()` if `expected` is not a valid type or tuple of types.
- Raises `AttributeError` if validation fails and an element inside `expected` does not define a `__name__` attribute (such as certain parameterized generic aliases in Python).

## AI usage guidance

Prefer `from klygo import validators` followed by `validators.validate_type(...)` at public function boundaries instead of writing ad-hoc `if not isinstance(...): raise TypeError(...)` checks.

## Example

See [`validate_type.py`](../../../examples/validators/validate_type.py) for an executable example.

## Tests

See [`test_validate_type.py`](../../../test/validators/test_validate_type.py) for the executable behavioral contract.

## Complete executable example

```python
"""Executable example for klygo.validators.validate_type."""

from klygo import validators

source = "image.jpg"
assert validators.validate_type(source, str, "source") is None
assert validators.validate_type(b"raw-bytes", (str, bytes), "source") is None

try:
    validators.validate_type(42, (str, bytes), "source")
except TypeError as error:
    print("Invalid source:", error)
    assert str(error) == "source must be str, bytes, got int"
```

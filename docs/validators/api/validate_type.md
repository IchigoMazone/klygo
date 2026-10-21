# `validators.validate_type`

```python
validate_type(value: object, expected: type | tuple[type, ...], name: str) -> None
```

`type_validator.py` exports this argument check. It returns `None` on success
and raises `TypeError` with the argument name, expected types, and actual type
on failure. An `isinstance` match includes subclasses; `bool` therefore
matches `int` when `int` is expected.

```python
from klygo import validators

validators.validate_type("picture.jpg", str, "source")
validators.validate_type(b"picture", (str, bytes), "source")
```

See the [runnable example](../../../examples/validators/validate_type.py),
[tests](../../../test/validators/test_validate_type.py), and
[tutorial](../../../tutorials/validators/Document_Validators_Type_Validator.ipynb).

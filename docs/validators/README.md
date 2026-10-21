# `klygo.validators`

`validators.validate_type()` checks a function argument with `isinstance` and
raises a named `TypeError` for an unsupported value. It is used throughout
Klygo's public APIs.

| Submodule | API | Purpose |
|---|---|---|
| `type_validator.py` | [`validate_type`](api/validate_type.md) | Check one or more accepted types |

Run the [example](../../examples/validators/validate_type.py) or the
[tests](../../test/validators/test_validate_type.py). The
[type-validator tutorial](../../tutorials/validators/Document_Validators_Type_Validator.ipynb)
shows practical checks and error handling.

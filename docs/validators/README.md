# `klygo.validators` Documentation

`klygo.validators` provides lightweight, zero-dependency runtime argument validation (`validate_type`) used across Klygo's public modules to enforce consistent `TypeError` diagnostics. Import the public facade with `from klygo import validators`.

Interactive Google Colab Tutorial:
[Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/validators/Document_Validators_Overview.ipynb)

Local Tutorial Notebooks (`tutorials/validators/`, 1-to-1 with `klygo/validators/` submodules):
- [`Document_Validators_Overview.ipynb`](../../tutorials/validators/Document_Validators_Overview.ipynb)
- [`Document_Validators_Type_Validator.ipynb`](../../tutorials/validators/Document_Validators_Type_Validator.ipynb) (`klygo/validators/type_validator.py`)

---

## 1. Core Concepts & Validation Contract

<details open>
<summary><b>Runtime type checking and standardized error formatting</b> <i>(Click to expand or collapse)</i></summary>

`klygo.validators` centralizes boundary type checks so public functions fail fast with predictable, human-readable error messages:

| Aspect | Behavior in `validators.validate_type` | Example / Notes |
| :--- | :--- | :--- |
| **Check mechanism** | Delegates directly to Python's built-in `isinstance(value, expected)`. | Supports a single `type` (`str`) or a `tuple[type, ...]` (`(str, os.PathLike)`). |
| **Return value** | Returns `None` when `value` matches `expected`. | Designed as a guard statement at the top of a function body. |
| **Exception format** | Raises `TypeError(f"{name} must be {expected_name}, got {type(value).__name__}")`. | Single type: `"source must be str, got int"`. Tuple of types: `"source must be str, bytes, got int"`. |
| **Subclass semantics** | Because `isinstance` is used, subclasses of `expected` pass validation. | Note that in Python `bool` is a subclass of `int` (`isinstance(True, int)` is `True`). |

</details>

---

## 2. Task-Oriented Cheatsheet (Common AI / User Questions)

<details open>
<summary><b>Task-oriented quick reference (8 tasks)</b> <i>(Click to expand or collapse)</i></summary>

| I want to... | Recommended API | Key Parameters / Example |
| :--- | :--- | :--- |
| **Validate that an argument is a specific type** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(source, str, "source")` |
| **Validate an argument against multiple allowed types** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(data, (dict, list), "data")` |
| **Accept strings or `os.PathLike` filesystem paths** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(path, (str, os.PathLike), "path")` |
| **Accept numeric values (`int` or `float`)** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(threshold, (int, float), "threshold")` |
| **Allow an optional argument when not `None`** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(limit, (int, type(None)), "limit")` |
| **Validate a mapping or sequence against `collections.abc` ABCs** | [`validate_type()`](api/validate_type.md) | `validators.validate_type(cfg, Mapping, "cfg")` |
| **Produce a named `TypeError` at a public API boundary** | [`validate_type()`](api/validate_type.md) | Raises `TypeError: <name> must be <expected>, got <actual>` |
| **Inspect or test the exact error message contract** | [`validate_type()`](api/validate_type.md) | See [`test_validate_type.py`](../../test/validators/test_validate_type.py) |

</details>

---

## 3. Public APIs Reference (1 Function across 1 Submodule)

<details open>
<summary><b>Runtime type validation — <code>type_validator.py</code> (1 function)</b> <i>(Click to expand)</i></summary>

- [`validate_type(value, expected, name)`](api/validate_type.md): Check `value` against `expected` (`type` or `tuple[type, ...]`) with `isinstance` and raise a standardized `TypeError` naming `name` when validation fails ([example](../../examples/validators/validate_type.py)).

</details>

---

## 4. Boundary Validation Pattern in Klygo Functions

```python
import os
from klygo import validators


def load_resource(source: str | os.PathLike, retries: int = 3) -> str:
    validators.validate_type(source, (str, os.PathLike), "source")
    validators.validate_type(retries, int, "retries")
    return os.fspath(source)


assert load_resource("model.onnx", retries=2) == "model.onnx"

try:
    load_resource(12345, retries=2)
except TypeError as exc:
    assert str(exc) == "source must be str, PathLike, got int"
```

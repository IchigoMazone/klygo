"""Runtime type and cardinality validation helpers.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/validators/Document_Validators_Overview.ipynb

Public API (2 functions):
    1. validate_type(...)   Validate an argument against expected type(s), raising TypeError.
    2. validate_count(...)  Validate an integer count against lower/upper bounds.
"""

from .type_validator import validate_type, validate_count

__all__ = [
    "validate_type",
    "validate_count",
]

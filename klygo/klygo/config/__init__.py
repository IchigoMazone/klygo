"""Structured configuration management utilities (``klygo.config``).

Interactive Google Colab Tutorial:
    https://colab.research.google.com/drive/1-aOofq_ZwLi00gRXnBLbJ4raupc6OZKm?usp=sharing

Serialization is delegated to :mod:`klygo.files`. This package adds nested
access, composition, validation, environment overlays, comparison, and a
stateful :class:`Config` interface.

Public APIs (20 Functions, 1 Class):
    File I/O:
        1. load   2. save   3. convert   4. export
    Creation and Composition:
        5. defaults   6. create   7. merge   8. update
    Nested Access:
        9. get   10. set   11. has   12. delete
    Inspection and Structure:
        13. keys   14. values   15. items   16. diff
        17. flatten   18. unflatten
    Integration and Validation:
        19. from_env   20. validate   21. Config
"""

from .access import delete, get, has, set
from .config import Config
from .creation import create, defaults
from .environment import from_env
from .inspection import diff, items, keys, values
from .io import convert, export, load, save
from .mapping import merge, update
from .structure import flatten, unflatten
from .validation import validate

__all__ = [
    "load", "save", "convert", "export",
    "defaults", "create", "merge", "update",
    "get", "set", "has", "delete",
    "keys", "values", "items", "diff",
    "flatten", "unflatten", "from_env", "validate", "Config",
]

"""Read, compose, inspect, and validate structured configuration.

Colab tutorial:
    https://colab.research.google.com/drive/1-aOofq_ZwLi00gRXnBLbJ4raupc6OZKm?usp=sharing

Public API (20 functions, 1 class):
    File I/O:
         1. load(...)                 Read a configuration file.
         2. save(...)                 Write configuration data.
         3. convert(...)              Convert between configuration formats.
         4. export(...)               Export configuration data to a new file.

    Creation and composition:
         5. defaults(...)             Build a configuration from defaults.
         6. create(...)               Create a configuration with optional overrides.
         7. merge(...)                Combine configuration mappings.
         8. update(...)               Apply changes to a configuration.

    Nested access:
         9. get(...)                  Read a value by dot-separated key.
        10. set(...)                  Set a value by dot-separated key.
        11. has(...)                  Check whether a nested key exists.
        12. delete(...)               Remove a nested key.

    Inspection and structure:
        13. keys(...)                 List configuration keys.
        14. values(...)               List configuration values.
        15. items(...)                List configuration key-value pairs.
        16. diff(...)                 Compare two configurations.
        17. flatten(...)              Flatten nested keys.
        18. unflatten(...)            Rebuild nested keys.

    Integration and validation:
        19. from_env(...)             Read configuration values from environment variables.
        20. validate(...)             Validate data against a schema.
        21. Config                    Class for a stateful configuration workflow.

Serialization and filesystem operations use klygo.files.
The Config class holds mutable state; functional APIs return values.
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

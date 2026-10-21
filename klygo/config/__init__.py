"""Read, compose, inspect, and validate structured configuration.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/config/Document_Config_Overview.ipynb

Public API (20 functions, 1 class across 9 submodules):
    File I/O (`io.py`):
         1. load(...)                 Read a configuration file.
         2. save(...)                 Write configuration data.
         3. convert(...)              Convert between configuration formats.
         4. export(...)               Export configuration data to a new file.

    Defaults and file creation (`creation.py`):
         5. defaults(...)             Build a configuration from defaults.
         6. create(...)               Create a configuration with optional overrides.

    Mapping composition (`mapping.py`):
         7. merge(...)                Combine configuration mappings.
         8. update(...)               Apply changes to a configuration.

    Nested access (`access.py`):
         9. get(...)                  Read a value by dot-separated key.
        10. set(...)                  Set a value by dot-separated key.
        11. has(...)                  Check whether a nested key exists.
        12. delete(...)               Remove a nested key.

    Collection views and comparison (`inspection.py`):
        13. keys(...)                 List configuration keys.
        14. values(...)               List configuration values.
        15. items(...)                List configuration key-value pairs.
        16. diff(...)                 Compare two configurations.

    Structure transformation (`structure.py`):
        17. flatten(...)              Flatten nested keys.
        18. unflatten(...)            Rebuild nested keys.

    Environment overlays (`environment.py`):
        19. from_env(...)             Read configuration values from environment variables.

    Schema validation (`validation.py`):
        20. validate(...)             Validate data against a schema.

    Stateful configuration manager (`config.py`):
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
    # io.py
    "load", "save", "convert", "export",
    # creation.py
    "defaults", "create",
    # mapping.py
    "merge", "update",
    # access.py
    "get", "set", "has", "delete",
    # inspection.py
    "keys", "values", "items", "diff",
    # structure.py
    "flatten", "unflatten",
    # environment.py
    "from_env",
    # validation.py
    "validate",
    # config.py
    "Config",
]

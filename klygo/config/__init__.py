"""Read, compose, inspect, and validate structured configuration.

Colab tutorial:
    https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/config/Document_Config_Overview.ipynb

Public API (18 functions, 1 class across 9 submodules):
    File I/O (`io.py`):
         1. load(...)                 Read a configuration file.
         2. save(...)                 Write configuration data.
         3. convert(...)              Convert between configuration formats.

    Defaults and file creation (`creation.py`):
         4. defaults(...)             Build a configuration from defaults.
         5. create(...)               Create a configuration with optional overrides.

    Mapping composition (`mapping.py`):
         6. merge(...)                Combine configuration mappings.

    Nested access (`access.py`):
         7. get(...)                  Read a value by dot-separated key.
         8. set(...)                  Set a value by dot-separated key.
         9. has(...)                  Check whether a nested key exists.
        10. delete(...)               Remove a nested key.

    Collection views and comparison (`inspection.py`):
        11. keys(...)                 List configuration keys.
        12. values(...)               List configuration values.
        13. items(...)                List configuration key-value pairs.
        14. diff(...)                 Compare two configurations.

    Structure transformation (`structure.py`):
        15. flatten(...)              Flatten nested keys.
        16. unflatten(...)            Rebuild nested keys.

    Environment overlays (`environment.py`):
        17. from_env(...)             Read configuration values from environment variables.

    Schema validation (`validation.py`):
        18. validate(...)             Validate data against a schema.

    Stateful configuration manager (`config.py`):
        19. Config                    Class for a stateful configuration workflow.

Serialization and filesystem operations use klygo.files.
The Config class holds mutable state; functional APIs return values.
"""

from .access import delete, get, has, set
from .config import Config
from .creation import create, defaults
from .environment import from_env
from .inspection import diff, items, keys, values
from .io import convert, load, save
from .mapping import merge
from .structure import flatten, unflatten
from .validation import validate

__all__ = [
    # io.py
    "load", "save", "convert",
    # creation.py
    "defaults", "create",
    # mapping.py
    "merge",
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

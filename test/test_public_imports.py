import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
from pathlib import Path

# Test 1: Top-level import klygo
import klygo
assert hasattr(klygo, "files")
assert hasattr(klygo, "config")
assert hasattr(klygo, "Config")
assert hasattr(klygo, "media")
assert hasattr(klygo, "archive")

# Test 2: Sub-module imports
from klygo import files, config, Config, media, archive
assert callable(files.load)
assert callable(config.load)
assert callable(Config)

# Test 3: Explicit imports from klygo.files
from klygo.files import (
    load, save, convert, exists, is_file, is_dir,
    list, find, walk, mkdir, copy, move, rename,
    remove, info, size, hash, compare, name, stem,
    extension, parent
)
assert callable(load)
assert callable(save)

# Test 4: Explicit imports from klygo.config
from klygo.config import (
    load as cfg_load, save as cfg_save, convert as cfg_convert,
    create, defaults, merge, update, get, set, has, delete,
    keys, values, items, validate, export
)
assert callable(cfg_load)
assert callable(create)

# Test 5: Verify __all__ in files and config
import klygo.files
import klygo.config

assert len(klygo.files.__all__) == 22
assert len(klygo.config.__all__) == 17

print("ALL PUBLIC IMPORT PATTERNS AND MODULE EXPORTS PASSED SUCCESSFULLY!")

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
    load, save, convert, download, exists, is_file, is_dir,
    list, find, walk, mkdir, copy, move, rename,
    remove, info, size, hash, compare, name, stem,
    extension, parent
)
assert callable(load)
assert callable(download)

# Test 4: Explicit imports from klygo.config
from klygo.config import (
    load as cfg_load, save as cfg_save, convert as cfg_convert,
    create, defaults, merge, update, get, set, has, delete,
    keys, values, items, validate, export, diff, flatten, unflatten, from_env
)
assert callable(cfg_load)
assert callable(diff)

# Test 5: Explicit imports from klygo.media
from klygo.media import (
    load as med_load, save as med_save, convert as med_convert, copy as med_copy,
    save_video, save_images, iter_frames, info as med_info, to_array, to_tensor, to_pil
)
assert callable(med_load)
assert callable(med_convert)

# Test 6: Verify __all__ in files, config, and media
import klygo.files
import klygo.config
import klygo.media

assert len(klygo.files.__all__) == 23
assert len(klygo.config.__all__) == 21
assert len(klygo.media.__all__) == 11

print("ALL PUBLIC IMPORT PATTERNS AND MODULE EXPORTS PASSED SUCCESSFULLY!")

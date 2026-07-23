import sys
import os

try:
    from . import klygo as _native_klygo
    for _attr in dir(_native_klygo):
        if not _attr.startswith("_"):
            globals()[_attr] = getattr(_native_klygo, _attr)
except ImportError:
    cpp_test_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cpp", "test_py")
    if cpp_test_path not in sys.path:
        sys.path.insert(0, cpp_test_path)
    try:
        import klygo as _native_klygo
        for _attr in dir(_native_klygo):
            if not _attr.startswith("_"):
                globals()[_attr] = getattr(_native_klygo, _attr)
    except ImportError:
        pass

from . import archive
from . import datasets
from . import io
from . import models
from . import utils
from . import validators
from . import visualize

__version__ = "2.0.4"
__author__ = "IchigoMazone"

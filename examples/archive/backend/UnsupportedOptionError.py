"""Handle a changed option that has no meaning for a format."""

from klygo.archive.backend import GZipBackend, UnsupportedOptionError


backend = GZipBackend()
try:
    backend.validate_option("compress", "include_root", False, True)
except UnsupportedOptionError as error:
    print(f"unsupported option: {error.option}")


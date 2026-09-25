"""Handle an operation unavailable for a selected format."""

from klygo.archive.backend import GZipBackend, UnsupportedOperationError


backend = GZipBackend()
try:
    backend.require_operation("add")
except UnsupportedOperationError as error:
    print(f"cannot {error.operation} with {error.format_name}")


"""Inspect the optional read-only RAR adapter."""

from klygo.archive.backend import RarBackend, UnsupportedOperationError


backend = RarBackend()
print("format:", backend.format_name)
print("password extraction:", "password" in backend.capabilities.extract_options)
try:
    backend.require_operation("compress")
except UnsupportedOperationError as error:
    print(error)


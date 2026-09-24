"""Executable example for klygo.files.exists."""

from klygo import files

print(files.exists("README.md"))
print(files.exists("missing-file"))


# Additional cases
from pathlib import Path

assert files.exists(Path("README.md"))
assert not files.exists("path-that-does-not-exist")


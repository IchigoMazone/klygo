"""Executable example for klygo.files.is_file."""

from klygo import files

print(files.is_file("README.md"))


# Additional cases
assert not files.is_file("klygo")
assert not files.is_file("missing-file")


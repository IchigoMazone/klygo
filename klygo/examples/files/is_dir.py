"""Executable example for klygo.files.is_dir."""

from klygo import files

print(files.is_dir("klygo"))


# Additional cases
assert not files.is_dir("README.md")
assert not files.is_dir("missing-directory")


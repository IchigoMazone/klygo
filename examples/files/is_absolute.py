"""Executable example for klygo.files.is_absolute."""

from klygo import files

print(files.is_absolute(files.resolve("README.md")))
print(files.is_absolute("README.md"))


# Additional cases
absolute = files.resolve("README.md")
assert files.is_absolute(absolute)
assert not files.is_absolute("README.md")


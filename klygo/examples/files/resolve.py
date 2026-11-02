"""Executable example for klygo.files.resolve."""

from klygo import files

print(files.resolve("README.md", strict=True))


# Additional cases
absolute = files.resolve(".")
assert absolute.is_absolute()
try:
    files.resolve("missing-path", strict=True)
except FileNotFoundError:
    print("strict resolution requires existence")


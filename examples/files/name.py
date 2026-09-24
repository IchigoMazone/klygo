"""Executable example for klygo.files.name."""

from klygo import files

print(files.name("dataset/images/sample.jpg"))


# Additional cases
assert files.name("archive.tar.gz") == "archive.tar.gz"
assert files.name("README") == "README"


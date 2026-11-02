"""Executable example for klygo.files.extension."""

from klygo import files

print(files.extension("dataset/archive.tar.gz"))


# Additional cases
assert files.extension("photo.jpg") == ".jpg"
assert files.extension("README") == ""


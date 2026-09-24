"""Executable example for klygo.files.stem."""

from klygo import files

print(files.stem("dataset/archive.tar.gz"))


# Additional cases
assert files.stem("photo.jpg") == "photo"
assert files.stem("README") == "README"


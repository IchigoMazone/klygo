"""Executable example for klygo.files.extensions."""

from klygo import files

print(files.extensions("archives/dataset.tar.gz"))


# Additional cases
assert files.extensions("photo.jpg") == (".jpg",)
assert files.extensions("README") == ()


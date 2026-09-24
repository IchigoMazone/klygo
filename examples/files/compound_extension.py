"""Executable example for klygo.files.compound_extension."""

from klygo import files

print(files.compound_extension("archives/dataset.tar.gz"))


# Additional cases
assert files.compound_extension("photo.jpg") == ".jpg"
assert files.compound_extension("README") == ""


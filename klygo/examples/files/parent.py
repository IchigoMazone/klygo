"""Executable example for klygo.files.parent."""

from klygo import files

print(files.parent("dataset/images/sample.jpg"))


# Additional cases
assert files.parent("dataset/images/cat.jpg").name == "images"
assert files.parent("cat.jpg") == files.path(".")


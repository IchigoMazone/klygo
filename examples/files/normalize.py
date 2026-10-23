"""Executable example for klygo.files.normalize."""

from klygo import files

print(files.normalize("dataset/images/../labels"))


# Additional cases
assert files.normalize("./dataset/./images") == files.path("dataset/images")
assert files.normalize("dataset/images/../labels") == files.path("dataset/labels")


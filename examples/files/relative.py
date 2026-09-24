"""Executable example for klygo.files.relative."""

from klygo import files

print(files.relative("dataset/images/train", "dataset"))


# Additional cases
assert files.relative("dataset/labels/train", "dataset") == files.path("labels/train")
assert files.relative("dataset", "dataset") == files.path(".")


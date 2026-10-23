"""Executable example for klygo.files.parents."""

from klygo import files

for ancestor in files.parents("dataset/images/train/cat.jpg"):
    print(ancestor)


# Additional cases
ancestors = files.parents("dataset/images/train/cat.jpg")
assert ancestors[0] == files.path("dataset/images/train")
assert ancestors[1] == files.path("dataset/images")


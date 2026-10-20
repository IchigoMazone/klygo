"""Executable example for klygo.files.common_path."""

from klygo import files

print(files.common_path(["dataset/images/train", "dataset/labels/train"]))


# Additional cases
assert files.common_path(["dataset/images", "dataset/labels"]) == files.path("dataset")
try:
    files.common_path([])
except ValueError:
    print("empty collections are rejected")


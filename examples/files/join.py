"""Executable example for klygo.files.join."""

from klygo import files

print(files.join("dataset", "images", "train", "sample.jpg"))


# Additional cases
assert files.join("dataset", "labels") == files.path("dataset/labels")
try:
    files.join()
except ValueError:
    print("at least one component is required")


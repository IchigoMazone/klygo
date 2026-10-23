"""Executable example for klygo.files.replace_root."""

from klygo import files

label = files.replace_root(
    "dataset/images/train/cat.jpg",
    "dataset/images",
    "dataset/labels",
)
print(label)


# Additional cases
assert label == files.path("dataset/labels/train/cat.jpg")
try:
    files.replace_root("outside/cat.jpg", "dataset/images", "dataset/labels")
except ValueError:
    print("source must be inside old_root")


"""Executable example for klygo.files.is_within."""

from klygo import files

print(files.is_within("dataset/images/a.jpg", "dataset"))
print(files.is_within("../outside.txt", "dataset"))


# Additional cases
assert files.is_within("dataset/labels/a.txt", "dataset")
assert not files.is_within("../outside.txt", "dataset")


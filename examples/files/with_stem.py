"""Executable example for klygo.files.with_stem."""

from klygo import files

print(files.with_stem("dataset/cat.jpg", "dog"))


# Additional cases
assert files.with_stem("dataset/cat.jpg", "dog") == files.path("dataset/dog.jpg")
assert files.with_stem("archive.tar.gz", "backup") == files.path("backup.gz")


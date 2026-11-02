"""Executable example for klygo.files.with_name."""

from klygo import files

print(files.with_name("dataset/cat.jpg", "dog.jpg"))


# Additional cases
assert files.with_name("dataset/cat.jpg", "dog.png") == files.path("dataset/dog.png")
assert files.with_name("archive.tar.gz", "backup.zip") == files.path("backup.zip")


"""Executable example for klygo.files.with_extension."""

from klygo import files

print(files.with_extension("archives/dataset.tar.gz", ".zip"))


# Additional cases
assert files.with_extension("image.jpg", "png") == files.path("image.png")
assert files.with_extension("archive.tar.gz", ".zip") == files.path("archive.zip")
assert files.with_extension("README", ".md") == files.path("README.md")


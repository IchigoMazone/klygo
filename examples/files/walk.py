"""Executable example for klygo.files.walk."""

from klygo import files

for root, directories, filenames in files.walk("klygo/files"):
    print(root, directories, filenames)


# Additional cases
# Materialize only when all rows are needed.
rows = list(files.walk("klygo/files"))
assert rows and all(len(row) == 3 for row in rows)


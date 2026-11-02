"""Executable example for klygo.files.size."""

from tempfile import TemporaryDirectory
from klygo import files

print(files.size("README.md"))
print(files.size("README.md", human=True))


# Additional cases
# Directory size is the sum of descendant files.
with TemporaryDirectory() as directory:
    root = files.path(directory)
    files.save(root / "a.txt", "123", verbose=False)
    files.save(root / "b.txt", "45", verbose=False)
    assert files.size(root) == 5

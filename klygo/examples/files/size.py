"""Executable example for klygo.files.size."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import files

print(files.size("README.md"))
print(files.size("README.md", human=True))


# Additional cases
# Directory size is the sum of descendant files.
with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "a.bin").write_bytes(b"123")
    (root / "b.bin").write_bytes(b"45")
    assert files.size(root) == 5

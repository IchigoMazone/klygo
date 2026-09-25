"""Create and inspect a ZIP archive through the low-level adapter."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.files as files
from klygo.archive.backend import ZipBackend


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "hello.txt"
    files.save(source, "hello", verbose=False)
    packed = root / "hello.zip"

    backend = ZipBackend()
    backend.compress(source, packed, method="deflated", verbose=False)
    print(backend.list_files(packed))
    print(backend.search(packed, "*.txt"))


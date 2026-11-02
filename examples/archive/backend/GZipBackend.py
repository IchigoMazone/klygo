"""Round-trip one file through a GZip stream."""

from tempfile import TemporaryDirectory

import klygo.files as files
from klygo.archive.backend import GZipBackend


with TemporaryDirectory() as directory:
    root = files.path(directory)
    source = root / "model.bin"
    files.save(source, b"model bytes", verbose=False)
    packed = root / "model.gz"

    backend = GZipBackend()
    backend.compress(source, packed, verbose=False)
    backend.extract(packed, root / "output", verbose=False)
    print(backend.list_files(packed))

"""Round-trip one file through a GZip stream."""

from pathlib import Path
from tempfile import TemporaryDirectory

from klygo.archive.backend import GZipBackend


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "model.bin"
    source.write_bytes(b"model bytes")
    packed = root / "model.gz"

    backend = GZipBackend()
    backend.compress(source, packed, verbose=False)
    backend.extract(packed, root / "output", verbose=False)
    print(backend.list_files(packed))

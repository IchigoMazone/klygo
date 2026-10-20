"""Create a compressed TAR archive with explicit backend selection."""

from pathlib import Path
from tempfile import TemporaryDirectory

import klygo.files as files
from klygo.archive.backend import TarBackend


with TemporaryDirectory() as directory:
    root = Path(directory)
    source = files.mkdir(root / "dataset")
    files.save(source / "labels.txt", "cat\ndog", verbose=False)
    packed = root / "dataset.tar.gz"

    backend = TarBackend("tar.gz")
    backend.compress(source, packed, include_root=False, verbose=False)
    print(backend.format_name, backend.list_files(packed))


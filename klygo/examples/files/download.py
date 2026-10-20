"""Executable example for klygo.files.download."""

from tempfile import TemporaryDirectory
from pathlib import Path
from klygo import files

with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "source.bin"
    source.write_bytes(b"klygo")
    downloaded = files.download(source, root / "downloads", verbose=False)
    print(downloaded, downloaded.read_bytes())


# Additional cases
# Existing destinations are protected unless overwrite=True.
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "weights.bin"
    source.write_bytes(b"v1")
    destination = files.download(source, root / "cache", verbose=False)
    try:
        files.download(source, root / "cache", verbose=False)
    except FileExistsError:
        print("destination protected:", destination)


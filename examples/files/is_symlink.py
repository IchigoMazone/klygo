"""Executable example for klygo.files.is_symlink."""

from tempfile import TemporaryDirectory

from klygo import files

with TemporaryDirectory() as directory:
    item = files.join(directory, "item.txt")
    files.save(item, "hello", overwrite=True, verbose=False)
    print(files.is_symlink(item))  # False for a regular file.

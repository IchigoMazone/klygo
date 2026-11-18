"""Executable example for klygo.files.info."""

from klygo import files, utils

metadata = files.info("README.md")
utils.tabulate(metadata, title="File information")
print(metadata["name"], metadata["human_size"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert "hash" not in metadata

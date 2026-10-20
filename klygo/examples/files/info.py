"""Executable example for klygo.files.info."""

from klygo import files

metadata = files.info("README.md")
print(metadata["name"], metadata["human_size"], metadata["hash"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert isinstance(metadata["hash"], str)


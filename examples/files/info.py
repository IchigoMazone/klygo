"""Executable example for klygo.files.info."""

from klygo import files

metadata = files.info("README.md", verbose=True)
print(metadata["name"], metadata["human_size"])


# Additional cases
assert metadata["is_file"] is True
assert metadata["size"] > 0
assert "hash" not in metadata

# Disable display when only the dictionary is needed.
quiet = files.info("README.md")
assert quiet == metadata

"""Executable example for klygo.files.hash."""

from klygo import files

print(files.hash("README.md", algorithm="sha256"))


# Additional cases
# Select any algorithm supported by hashlib.
md5 = files.hash("README.md")
sha256 = files.hash("README.md", algorithm="sha256")
assert len(md5) == 32 and len(sha256) == 64


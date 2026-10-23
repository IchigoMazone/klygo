"""Inspect backend features before selecting an operation."""

from klygo.archive.backend import get_backend


for filename in ("data.zip", "data.tar.gz", "data.gz", "data.rar"):
    backend = get_backend(filename)
    caps = backend.capabilities
    print(
        backend.format_name,
        {"compress": caps.compress, "add": caps.add, "remove": caps.remove},
    )


"""Select backends from paths and explicit aliases."""

from klygo.archive.backend import get_backend


automatic = get_backend("dataset.zip")
explicit = get_backend("unused", format_hint="tgz")
print(type(automatic).__name__, automatic.format_name)
print(type(explicit).__name__, explicit.format_name)


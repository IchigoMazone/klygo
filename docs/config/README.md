# `klygo.config`

`klygo.config` is the structured configuration layer built on `klygo.files`.
It supports file conversion, dot-path access, immutable composition, environment
overlays, lightweight validation, structural comparison, and a stateful object API.

Import the public module and call APIs through it:

```python
from klygo import config

settings = config.load("settings.yaml")
settings = config.set(settings, "model.batch", 32)
config.save("resolved.toml", settings, overwrite=True)
```

API groups:

- File I/O: `load`, `save`, `convert`, `export`
- Creation: `defaults`, `create`
- Composition: `merge`, `update`
- Nested access: `get`, `set`, `has`, `delete`
- Inspection: `keys`, `values`, `items`, `diff`
- Structure: `flatten`, `unflatten`
- Integration: `from_env`, `validate`
- Stateful interface: `Config`

Every API has a dedicated page in this directory, an executable example under
`examples/config`, and a behavioral contract under `test/config`.

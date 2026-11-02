# `klygo.config` examples

Every public API has a matching executable example in this directory.

For model configuration, start with
[`model_metadata.py`](model_metadata.py). It demonstrates:

- boolean parameter-group permissions with `models.flags()`;
- reusable schemas from `models.metadata()`;
- initialized definitions from `models.configure()`;
- exact `Config.create_default(..., metadata=...)` mode;
- JSON, YAML, and TOML round-trips;
- preservation of `False`, `True`, and `None` values.

Run it from the repository root:

```bash
python examples/config/model_metadata.py
```

The corresponding guide is
[`docs/config/model-metadata.md`](../../docs/config/model-metadata.md).


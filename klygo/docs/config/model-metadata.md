# Model metadata configuration

`klygo.config` can persist both model schemas returned by
`models.metadata()` and initialized definitions returned by
`models.configure()`. JSON, YAML, and TOML use the same Python structure.

## The two metadata stages

`models.metadata()` describes the model without creating parameter values:

```python
from klygo import models

flags = models.flags(
    model=False,
    processor=True,
    post=True,
)

schema = models.metadata(
    flags=flags,
    priority=models.priority(
        flags,
        model=("device",),
        processor=("size",),
        post=("threshold", "iou"),
    ),
    details=models.details(
        name="Example detector",
        task="Object-Detection",
        backend="Custom",
        library="custom",
    ),
)
```

At this point `schema` has one stable shape containing `flags`, `priority`,
`details`, and `class`. Missing optional components remain `None`.

`models.configure()` binds the schema to a model identifier and initializes
the declared parameter groups:

```python
configured = models.configure(
    "weights.pt",
    metadata=schema,
    implementation="my_package.detectors.CustomDetector",
    device="cuda",
    size=640,
    threshold=0.4,
    post_iou=0.7,
)
```

Configuration and loading may initialize every declared group. The boolean
flag controls only whether `model.predict()` may override that group later:

- `model=False`: configured once, then locked during prediction.
- `processor=True`: runtime preprocessing overrides are accepted.
- `post=True`: runtime postprocessing overrides are accepted.

## Save and load a schema

Use `metadata=` with `Config.create_default()` to write the mapping exactly.
General defaults such as `default.root` are not injected.

```python
from klygo import Config

manager = Config.create_default(
    "model.toml",
    metadata=schema,
    overwrite=True,
    verbose=False,
)

assert manager.get("flags.model") is False
assert manager.get("priority.post") == ["threshold", "iou"]
assert manager.get("class") is None
```

The returned manager is already loaded. Constructing a new manager remains
valid when the file is read later:

```python
restored_schema = Config("model.toml").read(verbose=False)
configured = models.configure(
    "weights.pt",
    metadata=restored_schema,
    implementation="my_package.detectors.CustomDetector",
    threshold=0.5,
)
```

## Save and load configured metadata

A configured definition can be passed directly to `config.save()` or to
`Config.create_default()`:

```python
from klygo import config, models

config.save("configured.yaml", configured, overwrite=True, verbose=False)
restored = config.load("configured.yaml", verbose=False)
model = models.load(restored)
```

`models.load()` also accepts the path directly:

```python
model = models.load("configured.yaml")
```

The `class` field must contain an importable implementation path before using
`models.load()`. A schema intentionally has `class=None` until
`models.configure(..., implementation=...)` supplies it.

## Runtime override behavior

For the flags declared above:

```python
model.predict(image, threshold=0.6)  # accepted: post=True
model(image, size=800)               # accepted: processor=True
model.predict(image, device="cpu")  # ValueError: model=False
```

`model(image, ...)` is the callable form of `model.predict(image, ...)`.
Direct framework inference remains available through `model.model(...)`.

## Format behavior

| Format | Boolean flags | `None` values | Nested groups |
|---|---:|---:|---:|
| JSON | Preserved | Native `null` | Preserved |
| YAML | Preserved | Native `null` | Preserved |
| TOML | Preserved | Transparently tagged | Preserved |

TOML has no native null value. `config.save()` encodes `None` using a private
Klygo marker, and `config.load()` restores it recursively. Applications should
not read or write that marker directly.

## Choosing the API

- Use `config.save()` and `config.load()` for stateless file operations.
- Use `Config.create_default(..., metadata=value)` when creating a path-bound,
  already-loaded manager.
- Use `models.metadata()` for a reusable schema.
- Use `models.configure()` for a concrete model definition.
- Use `models.load()` only after the definition has an importable `class`.

## Executable example and tests

- Example: [`model_metadata.py`](../../examples/config/model_metadata.py)
- Tests: [`test_model_metadata.py`](../../test/config/test_model_metadata.py)

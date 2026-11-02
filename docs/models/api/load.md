# `models.load`

```python
models.load(model: Union[str, Any], **kwargs) -> klygo.models.base.BaseModel
```

Resolve and instantiate a Klygo model wrapper from metadata, a loaded `Config`, an offline model directory, a `.pt` checkpoint, a registry identifier, or an in-memory neural network module.

## Contract

`models.load()` is the unified entry point for constructing model wrappers across all supported backends:

1. **Existing `BaseModel` instance**: Returned unchanged (`models.load(wrapper) is wrapper`).
2. **`klygo.config.Config` or `Mapping` / `box.Box` metadata**: Resolved via `models.configure()` and instantiated using the class declared under `"class"` (or `implementation=`).
3. **In-memory module with `.parameters()` (e.g., `torch.nn.Module`)**: Wrapped directly into a `READY` [`models.Detector`](Detector.md) instance with `backend="PyTorch (In-Memory)"`.
4. **Offline directory (`klygo.json` or Hugging Face `config.json`)**: Loads `klygo.json` (and local `model.py` if saved from a custom class) or auto-detects offline Hugging Face Grounding DINO directories.
5. **Local Ultralytics `.pt` file**: Automatically configures and instantiates `YOLODetect` pointing to the resolved `.pt` path. A filename matching a built-in model, such as `yolo11n.pt`, retains that model's complete `models.json` parameter schema; an unregistered filename uses the generic offline schema.
6. **Built-in registry name (`models.json`)**: Resolves exact or wildcard model identifiers (such as `"yolov8n"`, `"yolov8s-worldv2"`, `"grounding-dino-base"`) with an $O(1)$ cached lookup.

 Note: `models.load()` intentionally does **not** parse raw `.json`, `.yaml`, `.yml`, or `.toml` configuration file paths directly; load configuration files first via `config.load(path)` or `Config(path).read()`.

## Parameters

- `model`: `str | os.PathLike[str] | Mapping[str, Any] | Config | BaseModel | Any` — model source, metadata mapping, or wrapper instance.
- `**kwargs`: `Any` — parameter overrides routed through `models.configure()`. Priority parameters use their unprefixed names (e.g., `threshold=0.4`); non-priority parameters use `<group>_<name>` (e.g., `post_iou=0.6`) or group mappings (`post={"threshold": 0.4}`).

## Returns

`klygo.models.base.BaseModel` — an initialized model wrapper (typically a [`Detector`](Detector.md) subclass) in `"READY"` state.

## Errors and edge cases

- Raises `TypeError` if `model` is a path to a `.json`, `.yaml`, `.yml`, or `.toml` configuration file instead of a loaded `Config` or mapping.
- Raises `ValueError` if `model` is an unknown registry identifier, if `"class"` metadata is missing/empty, or if `**kwargs` contains an undeclared parameter group.
- Raises `KeyError` if the target `"class"` import path does not exist or cannot be imported.

## Example

See [`load.py`](../../../examples/models/load.py) for a standalone executable script.

## Tests

See [`test_class_registry_backends.py`](../../../test/models/test_class_registry_backends.py) and [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.load."""

from klygo import models

# 1. Passing an existing wrapper returns it directly
existing = models.Detector()
assert models.load(existing) is existing

# 2. Loading from configured metadata with parameter overrides
flags = models.flags(model=False, post=True)
schema = models.metadata(
    flags,
    priority=models.priority(flags, post=("threshold",)),
    details=models.details(
        name="LocalDetector",
        task="Object-Detection",
        backend="PyTorch",
        library="custom",
    ),
)
configured = models.configure(
    "local-detector-v1",
    metadata=schema,
    implementation=models.Detector,
    threshold=0.35,
)

loaded = models.load(configured, threshold=0.6)
print(loaded.model_id, loaded.metadata.post.threshold)
```

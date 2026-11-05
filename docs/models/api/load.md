# `models.load`

```python
models.load(model: Union[str, Any], **kwargs) -> klygo.models.base.BaseModel
```

Resolve and instantiate a Klygo model wrapper from metadata, a loaded `Config`, an exported model JSON file, a registry identifier, or an in-memory neural network module. Weight files and directories are not auto-detected.

## Contract

`models.load()` is the unified entry point for constructing model wrappers across all supported backends:

1. **Existing `BaseModel` instance**: Returned unchanged (`models.load(wrapper) is wrapper`).
2. **`klygo.config.Config` or `Mapping` / `box.Box` metadata**: Resolved via `models.configure()` and instantiated using the class declared under `"class"` (or `implementation=`).
3. **In-memory module with `.parameters()` (e.g., `torch.nn.Module`)**: Wrapped directly into a `READY` [`models.Detector`](Detector.md) instance with `backend="PyTorch (In-Memory)"`.
4. **Exported JSON file**: Loads files written by `models.export()` or `model.export()` when they contain the exported model schema.
5. **Built-in registry name (`models.json`)**: Resolves equivalent short and provider-qualified identifiers. For example, `"yolo11n.pt"` and `"ultralytics/yolo11n.pt"` both load `yolo11n.pt`; `"locate-anything"` and `"nvidia/LocateAnything-3B"` both load `nvidia/LocateAnything-3B`; `"grounding-dino-base"` and `"IDEA-Research/grounding-dino-base"` both load `IDEA-Research/grounding-dino-base`; and `"fasterrcnn_mobilenet_v3_large_fpn"` and `"torchvision/fasterrcnn_mobilenet_v3_large_fpn"` both load the registered Torchvision Faster R-CNN wrapper.

Registered detectors expose an optional `model.weights` override. `model_id` continues to select the model architecture, config, and processor; `weights` selects the backend-native checkpoint source. Klygo passes compatibility checks to Ultralytics or Transformers.

`weights` (as well as prefixed `model_weights`) is supported as a priority parameter and resolves to `model.weights`. Persisting `model.weights` in the configuration keeps the selected checkpoint explicit and exportable.

```python
from klygo import config, models

cfg = config.load("config.json", verbose=False)
cfg = config.set(cfg, "model.weights", "weights/trained.pt")
detector = models.load(cfg)
```

For YOLO, `weights` is an Ultralytics checkpoint such as a `.pt` file. For Grounding DINO, it is a Hugging Face pretrained repository or directory loaded against the config from `model_id`. Locate Anything uses the config, tokenizer, and processor from `model_id`, then asks `AutoModel.from_pretrained()` to load `weights`; a local pretrained directory is the practical override when the two repositories do not share a revision.

Note: arbitrary JSON and `.yaml`, `.yml`, or `.toml` paths are not model exports. Load those configuration files first via `config.load(path)` or `Config(path).read()`.

## Parameters

- `model`: `str | os.PathLike[str] | Mapping[str, Any] | Config | BaseModel | Any` — model source, metadata mapping, or wrapper instance.
- `**kwargs`: `Any` — parameter overrides routed through `models.configure()`. Priority parameters use their unprefixed names (e.g., `threshold=0.4`); non-priority parameters use `<group>_<name>` (e.g., `post_iou=0.6`) or group mappings (`post={"threshold": 0.4}`).

## Returns

`klygo.models.base.BaseModel` — an initialized model wrapper (typically a [`Detector`](Detector.md) subclass) in `"READY"` state.

## Errors and edge cases

- Raises `TypeError` if `model` is a `.yaml`, `.yml`, or `.toml` path instead of a loaded `Config` or mapping.
- Raises `ValueError` if a JSON file does not contain the schema produced by `models.export()` or `model.export()`.
- Raises `ValueError` if `model` is a directory path.
- Raises `ValueError` if `model` is a local weight path instead of a registry name.
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

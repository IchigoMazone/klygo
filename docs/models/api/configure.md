# `models.configure`

```python
models.configure(
    model_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    flags: Optional[Union[Mapping[str, bool], Sequence[str]]] = None,
    priority: Optional[Mapping[str, Any]] = None,
    details: Optional[Mapping[str, Any]] = None,
    implementation: Optional[Union[type, str]] = None,
    **kwargs,
) -> box.Box
```

Bind a concrete `model_id`, implementation class, and parameter values to a validated model metadata schema without instantiating the model or importing heavy AI frameworks.

## Contract

`models.configure()` takes a reusable schema produced by [`models.metadata()`](metadata.md) (or inline `flags`, `priority`, and `details` declarations) and returns an independent, serializable `box.Box` containing:
- `model_id`: string identifier or path.
- `class`: normalized import path string from `implementation` (or `None` if omitted).
- `flags`: normalized `dict[str, bool]` mapping each parameter group to its predict-time override permission.
- `priority`: normalized `dict[str, list[str]]` mapping groups to unprefixed priority parameter names.
- `details`: validated descriptive metadata (`name`, `task`, `backend`, `library`, etc.).
- Top-level parameter group mappings (e.g., `model`, `processor`, `post`) populated with resolved `**kwargs`.

During `configure()`, **all** declared parameter groups (including groups with `flags[group] == False`) can be initialized. Locking a group with `False` only restricts overrides at `model.predict()` runtime.

## Parameters

- `model_id`: `str` — local checkpoint path, hub preset ID, or custom identifier consumed by the model wrapper.
- `metadata`: `Mapping[str, Any] | None`, default `None` — base schema built with `models.metadata()`.
- `flags`: `Mapping[str, bool] | Sequence[str] | None`, default `None` — parameter-group declaration when not already present in `metadata`.
- `priority`: `Mapping[str, Any] | None`, default `None` — unprefixed parameter aliases grouped by destination flag.
- `details`: `Mapping[str, Any] | None`, default `None` — descriptive metadata validated via `models.details()`.
- `implementation`: `type | str | None`, default `None` — model wrapper class (e.g., `models.Detector` or a custom subclass) or its dotted import string, stored under the `"class"` key.
- `**kwargs`: `Any` — parameter values assigned via priority names (`threshold=0.4`), group prefixes (`model_device="cuda"`), or group dictionaries (`post={"iou": 0.5}`).

## Returns

`box.Box` — an independent, dot-accessible metadata dictionary ready to be passed to [`models.load()`](load.md) or persisted via `klygo.config.Config.create_default(..., metadata=...)`.

## Errors and edge cases

- Raises `ValueError` if an override key does not match any priority parameter or declared `<group>_` prefix, or if the same parameter is supplied twice (e.g., both `threshold=0.4` and `post_threshold=0.4`).
- Raises `TypeError` or `ValueError` if `flags`, `priority`, `details`, or `implementation` fails schema validation.

## Example

See [`configure.py`](../../../examples/models/configure.py) for a standalone executable script.

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

## Complete executable example

```python
"""Executable example for klygo.models.configure."""

from klygo import models

group_flags = models.flags(model=False, processor=True, post=True)
schema = models.metadata(
    flags=group_flags,
    priority=models.priority(group_flags, post=("threshold", "iou")),
    details=models.details(
        name="Faster R-CNN",
        task="Object-Detection",
        backend="Torchvision",
        library="torchvision",
    ),
)

configured = models.configure(
    "fasterrcnn_mobilenet_v3_large_fpn",
    metadata=schema,
    implementation=models.Detector,
    model_device="cpu",
    threshold=0.45,
    iou=0.6,
)

print(configured.model_id, configured["class"])
print(configured.model.device, configured.post.threshold, configured.post.iou)
```

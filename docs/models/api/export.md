# `models.export`

```python
models.export(
    model,
    output_path="config.json",
    *,
    overwrite=False,
    **kwargs,
) -> pathlib.Path
```

Resolve any equivalent registered model name accepted by `models.load()`, then write its Klygo configuration without loading model weights.

## Parameters

- `model`: `str | os.PathLike[str]` — registered short or provider-qualified name, such as `"yolo11n.pt"`, `"ultralytics/yolov8s-worldv2.pt"`, `"locate-anything"`, or `"IDEA-Research/grounding-dino-base"`.
- `output_path`: `str | os.PathLike[str]`, default `"config.json"` — destination JSON file.
- `overwrite`: `bool`, default `False` — allow replacement of an existing destination.
- `**kwargs`: model configuration overrides resolved by [`models.configure()`](configure.md), such as `device="cpu"`, `threshold=0.4`, or checkpoint overrides via `weights="trained.pt"` (or `model_weights="trained.pt"`).

## Returns

`pathlib.Path` — the exported JSON file path.

## Errors and edge cases

- Raises `ValueError` when the model name cannot be normalized to an entry in `models.json` or the destination does not end in `.json`.
- Raises `ValueError` when an override does not match the registered parameter schema.
- Raises `FileExistsError` when the destination exists and `overwrite=False`.
- The function exports configuration only. It does not instantiate the wrapper or load native weights.

## Example

See [`export.py`](../../../examples/models/export.py) for a standalone executable script.

```python
from klygo import models

path = models.export(
    "yolo11n.pt",
    "artifacts/models/yolo11n.json",
    overwrite=True,
    device="cpu",
    threshold=0.4,
)
print(path)
```

## Tests

See [`test_metadata.py`](../../../test/models/test_metadata.py).

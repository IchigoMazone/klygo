# `models.Detector`

```python
models.Detector(
    metadata: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs,
) -> None
```

Base Layer-2 object-detection engine providing unified batch/streaming inference, coordinate restoration, benchmarking, serialization, and multi-backend hardware dispatch.

## Contract

`Detector` subclasses [`models.BaseModel`](BaseModel.md) and sets `task="Object-Detection"`. Concrete detectors (such as `YOLODetect`, `YOLOWorldDetect`, `GroundingDinoDetect`, or application-defined subclasses for Torchvision Faster R-CNN and KerasHub RT-DETR / D-FINE) implement `forward(images, prompt=None)` while inheriting the full Klygo detection lifecycle:

1. **Multi-Backend Dispatch Helpers**:
   - `self.current_device` / `self.current_dtype`: Query the active device and dtype of `self.model`.
   - `self.cast_inputs(inputs, dev=None, dtype=None)`: Cast floating-point tensors to the model's device and dtype while preserving integer tensors (`input_ids`).
   - `self.run_inference(inputs, **model_kwargs)`: Execute forward pass via the active backend adapter (`Hugging Face`, `Ultralytics`, `Torchvision`, or `Keras`).
   - `self.format_results(raw_outputs)`: Normalize framework-native predictions into `list[dict[str, Any]]` with `"boxes"` (`xyxy`), `"scores"`, `"labels"`, and `"class_ids"`.
2. **Inference Engine (`predict`)**:
   - Resolves single images, lists of images, directories, or video streams through `klygo.media`.
   - Applies call-local parameter overrides (`**kwargs`) validated against `metadata.flags` and `metadata.priority` without mutating default model settings.
   - Automatically restores bounding-box coordinates to original image dimensions when inputs carry a `klygo.preprocessing` transformation trace.
   - Supports `stream=True` to return a lazy `klygo.outputs.Detections` generator that processes long videos in bounded memory.
3. **Lifecycle & Benchmarking**:
   - `warmup()`: Run a single 640×640 dummy inference pass to initialize GPU kernels.
   - `benchmark(source=None, num_runs=50, warmup=5)`: Measure mean/min/max latency (ms), throughput (FPS), and parameter count inside the backend's inference context with accelerator synchronization.
   - `save(output_dir)`: Export `klygo.json` metadata, custom `model.py` source (for custom classes), and backend weights.
   - `reset()` / `unload()`: Move weights to CPU, clear GPU caches, and transition `state` to `"UNLOADED"`.

## Parameters

- `metadata`: `Dict[str, Any] | None`, default `None` — model metadata mapping or `Box`.
- `backend`: `str | None`, default `None` — runtime backend name (`"Ultralytics"`, `"Hugging Face"`, `"Torchvision"`, `"Keras"`, `"PyTorch"`).
- `model_id`: `str | None`, default `None` — model identifier or weight path.
- `**kwargs`: `Any` — forwarded to `BaseModel.__init__`.

## Key Methods

- `predict(source, prompt=None, step=1, max_frames=None, batch_size=1, stream=False, verbose=False, **kwargs) -> Detections`: Run object detection on images or video frames.
- `forward(images, prompt=None) -> list[Detection]`: Execute one batch forward pass on a list of `PIL.Image.Image` instances.
- `benchmark(source=None, num_runs=50, warmup=5, verbose=True, **kwargs) -> dict[str, Any]`: Measure latency and FPS.
- `warmup() -> None`: Warm up the accelerator pipeline on a 640×640 image.
- `save(output_dir: str) -> None`: Persist metadata, custom wrapper code, and model weights.
- `reset() -> None`: Move model to CPU and reset settings to defaults.
- `unload() -> None`: Release model references, clear accelerator caches, and set `state = "UNLOADED"`.
- `help() -> None`: Display a Rich guide covering parameters and usage.

## Returns

Returns a `Detector` wrapper instance in `"READY"` state.

## Errors and edge cases

- Calling `forward()` on the base `Detector` without overriding it or attaching a callable `self.model` returns empty detections.
- Passing an undeclared or locked parameter to `predict(**kwargs)` raises `ValueError`.
- Calling `predict()`, `forward()`, `warmup()`, `benchmark()`, or `save()` after `unload()` raises `models.errors.InvalidStateError`.

## Example

See [`Detector.py`](../../../examples/models/Detector.py) for a standalone executable script.

## Tests

See [`test_preprocessed_input.py`](../../../test/models/test_preprocessed_input.py) and [`test_torchvision_backend.py`](../../../test/models/test_torchvision_backend.py).

## Complete executable example

```python
"""Executable example for klygo.models.Detector."""

from PIL import Image
from klygo import models, outputs


class ToyDetector(models.Detector):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(metadata=metadata, backend="PyTorch", **kwargs)
        self.names = {0: "person", 1: "car"}

    def forward(self, images, prompt=None):
        threshold = float(self.metadata.get("post", {}).get("threshold", 0.25))
        raw = [
            {
                "boxes": [[10.0, 15.0, 90.0, 110.0]],
                "scores": [0.92 if threshold <= 0.92 else 0.0],
                "labels": ["person"],
                "class_ids": [0],
            }
            for _ in images
        ]
        return outputs.build_detections(images, raw, names=self.names)


schema = models.configure(
    "toy-detector",
    flags=models.flags(model=False, post=True),
    priority=models.priority(models.flags(model=False, post=True), post=("threshold",)),
    implementation=ToyDetector,
    threshold=0.3,
)

detector = models.load(schema)
sample = Image.new("RGB", (128, 128), color=(30, 60, 90))
detections = detector.predict(sample, threshold=0.5)
print(detector.task, len(detections), len(detections[0].boxes))
```

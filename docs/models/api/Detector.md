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

`Detector` subclasses [`models.BaseModel`](BaseModel.md) and sets `task="Object-Detection"`. Every detector implements `forward(images, prompt=None)`. Before that method runs, `predict()` resolves priority arguments and `<group>_` prefixes into the active `self.metadata` groups. A wrapper reads those groups explicitly, performs preprocessing, calls its native model, formats the result, and passes it to `outputs.build_detections()`. `run_inference()` remains an optional adapter for framework-specific execution, not a required wrapper pattern.

1. **Persistent Prediction Configuration & Public Properties**:
   - `self.device`: Query the active accelerator device of `self.model` via `common.current_device()`.
   - `self.dtype`: Query the active numeric dtype of `self.model` via `common.current_dtype()`.
   - `self.predict_config`: Independent, defensive `deepcopy` snapshot of effective inference options (such as `prompt`, `threshold`, `iou`). External mutation of the returned dictionary does not mutate model state.
   - Persistent settings: `predict()` automatically retains configured `prompt`, `threshold`, and `iou` across successive calls. Passing `prompt=None` clears the remembered prompt and restores default vocabulary; calling `reset()` clears all remembered overrides. Execution controls (`batch`, `workers`, `prefetch`, `vid_stride`, `max_frames`, `stream`, `native`, `packed`, `verbose`) are never persisted.
2. **Multi-Backend Dispatch Helpers (delegated to [`klygo.models.backend.common`](../backend/api/common.md))**:
   - `self.cast_inputs(inputs, dev=None, dtype=None)`: Cast floating-point tensors to the model's device and dtype while preserving integer tensors (`input_ids`) via `common.cast_inputs()`.
   - `self.run_inference(inputs, **model_kwargs)`: Execute forward pass via `common.run_inference()` on the active backend adapter (`Hugging Face`, `Ultralytics`, `Torchvision`, or `Keras`).
   - `self.get_output_device(outputs)` / `self.sync_device(tensor, target_device)`: Inspect output device and synchronize multi-GPU tensors via `common.get_output_device()` and `common.sync_device()`.
   - `self.format_results(raw_outputs)`: Normalize framework-native predictions into `list[dict[str, Any]]` with `"boxes"` (`xyxy`), `"scores"`, `"labels"`, and `"class_ids"` via `common.format_results()`, ready to pass straight into [`outputs.build_detections(images, formatted, names=self.names)`](../../outputs/api/build_detections.md).
   - See the [`klygo.models.backend` & `common` Guide](../backend/README.md) for direct usage of all 15 `common` functions.
3. **Inference Engine (`predict`)**:
   - Resolves single images, lists of images, directories, or video streams through `klygo.media`.
   - Applies call-local parameter overrides (`**kwargs`) validated against `metadata.flags` and `metadata.priority`.
   - Automatically restores bounding-box coordinates to original image dimensions when inputs carry a `klygo.preprocessing` transformation trace.
   - Supports `stream=True` to return the same `klygo.outputs.Detections` type in lazy one-pass mode, processing long videos in bounded memory. When `media.load(..., stream=True)` already created a stream, `predict(source)` detects `source.is_stream` automatically; callers declare streaming only once. Use materialized media when random access is required.
   - Uses one decode worker and one prefetched batch by default, matching YOLO-style prediction read-ahead. Higher `workers` values decode independent images within each prefetched batch concurrently; `workers=0` or `prefetch=0` disables background decoding. Video frames sharing one reader always decode sequentially, and results preserve source order.
   - `native=True` lets a supported backend own source decoding. `prompt` and inference parameters registered by the model keep their normal public names, such as `threshold` and `iou`; they pass through the same metadata validation as managed prediction. Pipeline controls `stream`, `batch`, `workers`, `prefetch`, `vid_stride`, and `max_frames` are unavailable in native mode. Outputs are still normalized to `Detections`.
   - `packed=True` returns materialized `Detections` using shared compact arrays and lazy frame facades. It cannot be combined with `stream=True`.
4. **Lifecycle & Isolated Benchmarking**:
   - `warmup()`: Run a single 640×640 dummy inference pass to initialize GPU kernels.
   - `benchmark(source=None, iterations=10, warmup=3, verbose=True, **kwargs)`: Measure mean latency (ms), throughput (FPS), device, and dtype inside accelerator synchronization. Temporary overrides apply only during the benchmark run and are safely restored on completion or failure.
   - `save(output_dir)`: Export `klygo.json` metadata, custom `model.py` source (for custom classes), tokenizer (when present), and backend weights.
   - `reset()`: Clear remembered predict configuration, restore default model settings, and reset backend class state.
   - `unload()`: Move weights to CPU, release model and processor/tokenizer references, clear GPU caches, and transition `state` to `"UNLOADED"`.

## Parameters

- `metadata`: `Dict[str, Any] | None`, default `None` — model metadata mapping or `Box`.
- `backend`: `str | None`, default `None` — runtime backend name (`"Ultralytics"`, `"Hugging Face"`, `"Torchvision"`, `"Keras"`, `"PyTorch"`).
- `model_id`: `str | None`, default `None` — model identifier or weight path.
- `**kwargs`: `Any` — forwarded to `BaseModel.__init__`.

## Key Methods and Properties

- `device`: Read-only property returning active accelerator device (e.g., `torch.device("cpu")`).
- `dtype`: Read-only property returning active data type (e.g., `torch.float32`).
- `predict_config`: Read-only defensive snapshot dictionary of remembered inference parameters.
- `predict(source, prompt=None, batch=1, workers=1, prefetch=1, vid_stride=1, max_frames=None, stream=False, native=False, packed=False, verbose=True, **kwargs) -> Detections`: Run Klygo-managed or supported backend-native detection.
- `forward(images, prompt=None) -> list[Detection]`: Pure inference hook implemented by subclasses on a batch of `PIL.Image.Image` instances.
- `benchmark(source=None, iterations=10, warmup=3, verbose=True, **kwargs) -> dict[str, Any]`: Measure latency and FPS in isolation without mutating persistent model settings.
- `warmup() -> None`: Warm up the accelerator pipeline on a 640×640 image.
- `save(output_dir: str) -> None`: Persist metadata, custom wrapper code, tokenizer, and model weights.
- `reset() -> None`: Restore default settings, clear remembered prediction configuration, and reset backend vocabulary.
- `unload() -> None`: Release model and processor references, clear accelerator caches, and set `state = "UNLOADED"`.
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

Declare streaming at either input boundary, never both:

```python
source = media.load("video.mp4", stream=True)
results = detector.predict(source)

# Equivalent when passing a path directly:
results = detector.predict("video.mp4", stream=True)

# Override or disable the YOLO-style decode read-ahead:
results = detector.predict(source, batch=8, workers=2, prefetch=2)
results = detector.predict(source, workers=0)

# Ultralytics owns decoding while Klygo still returns Detections.
results = detector.predict(
    "video.mp4",
    native=True,
    stream=True,
    threshold=0.35,
    iou=0.5,
)

# Compact materialized annotations for large finite sources.
results = detector.predict("images/", packed=True)
```

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

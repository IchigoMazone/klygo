# Detector and output lifecycle changes

This checkout implements remembered inference configuration, canonical output
registries, atomic positional re-prediction and isolated benchmarking. The
public facades remain `models` and `outputs`; no additional public module is added.

## Prediction configuration

`detector.device` and `detector.dtype` query the active backend on access. Their
types remain backend-specific (for example `torch.device`/`torch.dtype` or strings).
`current_device()` and `current_dtype()` remain available.

`predict_config` is a defensive, deeply copied snapshot. Defaults come from
`settings`; remembered overrides are stored separately and resolved through the
existing flags/priority metadata contract. Priority keys are unprefixed; other
keys use their group prefix. Loading credentials and loading-only options are
excluded from result inference snapshots.

```python
first = detector(image, prompt=["person"], threshold=0.5)
second = detector(image)  # reuses prompt and threshold
detector(image, threshold=0.7)  # keeps prompt
detector(image, prompt=None)  # explicit clearing, where text is optional
detector.reset()  # restores settings and clears remembered overrides
```

Configuration validation is completed before updating remembered values. Eager
inference failure also leaves remembered values unchanged. Lazy prediction
commits valid configuration when the stream is created; later decode/inference
failures propagate when consuming it. Each stream captures independent effective
settings. Interleaved streams temporarily apply their vocabulary during a batch
and restore the current vocabulary before yielding. A per-detector lock prevents
overlapping backend state changes through these APIs; direct native model calls
remain the caller's responsibility.

`batch`, `vid_stride`, `max_frames`, `stream`, `verbose`, sources and output paths
are execution controls, never remembered options. Batch and stride must be
positive integers; `max_frames` must be None or a positive integer. Bool is not
an integer control. Confidence/IoU options must be finite numbers in `[0, 1]`.
Prompts accept a string, list or tuple of strings; whitespace is trimmed,
duplicates are removed, and blank/non-string labels are rejected.

## Wrapper contracts

| Wrapper | Prompt meaning | Clearing / reset |
| --- | --- | --- |
| YOLO | Filters its fixed class vocabulary; unknown labels raise ValueError | `prompt=None` restores configured `classes`; reset restores defaults |
| YOLO-World | Replaces vocabulary only when it changes | `prompt=None` and reset restore the checkpoint vocabulary |
| Grounding DINO | Lowercase text labels with trailing periods removed | Missing/cleared text cannot infer: ValueError leaves saved configuration unchanged; reset clears it |
| Locate Anything | Required category queries, processed sequentially | Same required-text policy as Grounding DINO; reset retains device placement |

For text-required models, use `reset()` when clearing configuration without
running inference. This avoids silently reusing old text or issuing malformed
empty queries. Direct `forward()` remains a low-level hook; remembered options
belong to `predict()` and callable syntax.

Every output frame stores an independent `config`: `prompt`, postprocessing
values, `predict_config`, `model_id`, `backend`, `source_path`, `frame_index` and
`image_frame_index`. Wrappers may add native filters or score semantics. Mutating
model state or later results cannot rewrite earlier result dictionaries.

## Canonical class IDs and active labels

`Detection.names` continues to show only labels referenced by its boxes.
`Detections.names` retains the full vocabulary for compatibility, including
previously allocated IDs. **`Detections.active_names`** contains only labels
present in current frames. Empty results have no active names.

A collection reconciles incoming labels across models. Equal labels share one
canonical ID. Different labels with the same incoming ID get separate IDs.
Existing unrelated IDs never move. A remapped box preserves its source ID in
`box.metadata["native_class_id"]`; `class_id` is the canonical ID.

Standalone detections retain explicit native IDs, including different native
classes with the same display label. Canonicalization happens at collection
boundaries. Incoming frames are copied when reconciliation would otherwise
modify another collection. Metadata and box UIDs survive these copies.

Use the container methods to edit frames/boxes. Direct writes to raw `objects`
or `frames` lists or to `Box.class_id` bypass validation and are not supported
registry mutation paths. Assigning the entire `frames` property is validated.

YOLO export now preserves canonical IDs, including sparse IDs, in both label
files and `data.yaml`. Historical vocabulary is retained so subsequent exports
do not silently renumber classes. An explicit `classes` list/map on a single
frame's `export()` can request a different mapping.

## Atomic re-prediction

```python
results.repredict(model_b, 10)
results.repredict(model_b, [10, 20, 30], prompt=["dog"], threshold=0.5)
results.repredict(model_b)  # all frames; returns this same collection
results[10] = model_b(results[10].source_image)[0]
```

Indices are collection positions, including negative positions. Bool, invalid
types and out-of-range positions are rejected before inference. Duplicate
positions are deduplicated in request order. An empty selection is a no-op.
The model receives selected source images in that order and must return exactly
one `Detection` per image. Predictions and class reconciliation are fully
validated before committing any frame. Exceptions leave the original frames,
their metadata and the registry unchanged. Model-side configuration follows the
model's own prediction lifecycle.

Replacement preserves each original source-image reference, `frame_index` and
`image_frame_index`; boxes are rebound to that image. Prediction metadata, scores
and timing come from the replacement. Collection order, fps and source type stay
unchanged. Slice assignment requires equal length and preserves positional frame
identity; use insert/extend/delete for resizing.

`replace_frame(frame_index, result)` and `update_frame(frame_index, function)`
still identify original frame IDs and return a new collection. `repredict()`
uses positions and mutates the existing collection. Update callbacks receive
an independent frame copy.

## Streams and resources

Random mutation and re-prediction reject lazy streams with a clear error. Call
`to_list()` explicitly on a **finite** stream first. This consumes and stores
every frame; never call it on an infinite source. Name queries do not consume
input; active names cover already consumed frames only. Current `Detections`
iteration caches frame results for replay, so memory grows with consumed results
even when lazy image proxies avoid retaining decoded pixels. Streaming is not a
promise of constant total memory.

Closing a result stream also closes its pending generator. Consuming a stream
after unloading its detector raises `InvalidStateError`.

## Benchmark scope

`benchmark(source=None, prompt=..., iterations=20, warmup=5, verbose=True, **kwargs)`
inherits remembered settings. Overrides apply only inside the benchmark. Native
YOLO-World vocabulary and wrapper class maps are restored in a finally block,
including on failure. Restoration errors are surfaced explicitly.

Inputs are resolved once before timing. Each measured iteration includes image
materialization, model forward work and output packaging. Accelerator barriers
run before and after each measurement. Reports retain the original keys and add
median/p95 latency, effective configuration, images per iteration, measurement
scope and throughput units. Latency is per iteration; `fps` means input images
per second, including multi-image iterations. It is not a video playback rate.
Warmup must be non-negative and iterations positive. Stream/sampling controls
are rejected; supply a finite preselected input sequence.

## Validation and references

- [Detector example](../../examples/models/Detector.py)
- [Outputs example](../../examples/outputs/detect.py)
- [Lifecycle tests](../../test/models/test_prediction_lifecycle.py)
- [Registry and atomicity tests](../../test/outputs/test_repredict_registry.py)
- [Text backend schema tests](../../test/models/test_text_detectors.py)
- [Opt-in real model tests](../../test/models/test_real_detection_opt_in.py)
- [Locate Anything requirements and limitations](locate-anything.md)

The public API audit checks module exports and documentation contracts; it is
not a substitute for behavioral tests or real-model inference verification.

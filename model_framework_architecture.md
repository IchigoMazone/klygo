# Model Framework Architecture Specification

## 1. Overview

This project is an AI model inference framework/library designed to provide a unified interface for different model types such as Object Detection, Classification, Segmentation, VLM, and future tasks.

Core principles:

- `BaseModel` contains genuinely common runtime behavior.
- Task base classes such as `Detector` contain task-specific contracts.
- Concrete models contain model-specific implementation only.
- Common configuration comes from `config.json` / metadata.
- Differences are declared with `@override` or `__UNSUPPORTED__`.
- Results use standardized task-specific result classes.
- The core should remain small and extensible.

## 2. Architecture

```text
BaseModel
├── Detector
│   ├── GroundingDINO
│   ├── YOLO
│   └── ...
├── Classifier
├── Segmenter
├── VLM
└── ...
```

`BaseModel` is the common runtime layer. Task classes specialize it. Concrete model classes implement only model-specific behavior.

## 3. BaseModel Responsibilities

`BaseModel` should provide:

- metadata/configuration
- current and default configuration
- runtime state
- device management
- dtype management
- lifecycle management
- unsupported-operation handling
- method introspection
- model information
- resource management
- common inference contract

It should not contain detector-specific concepts such as bounding boxes, masks, detection thresholds, or detector-specific preprocessing/postprocessing.

## 4. Initialization

Concrete models should call the parent constructor:

```python
class GroundingDINO(Detector):
    def __init__(self, metadata):
        super().__init__(metadata)
```

Typical loading flow:

```text
models.load()
    ↓
config.json
    ↓
metadata
    ↓
Model.__init__(metadata)
    ↓
BaseModel.__init__(metadata)
    ↓
READY
```

## 5. Configuration

Common configuration is stored in `config.json` and loaded into model metadata/configuration.

Example:

```json
{
  "model": {
    "dtype": "float16",
    "device_map": "auto"
  },
  "input": {
    "resize": 800,
    "pad": true
  },
  "post": {
    "threshold": 0.3
  }
}
```

Possible namespaces:

```text
config.model
config.input
config.post
config.extra
```

Model-specific configuration should not be forced into the universal `BaseModel` interface.

## 6. Configuration Precedence

Recommended precedence:

```text
config.json
    ↓
metadata
    ↓
models.load(...) overrides
    ↓
runtime/model overrides
    ↓
predict(...) overrides
```

More specific configuration overrides less specific configuration.

Prediction-specific overrides should normally apply only to that prediction unless explicitly designed as persistent runtime configuration.

## 7. Current vs Default Configuration

The model should conceptually expose:

```python
model.config
model.default_config
```

`model.config` is the current state. `model.default_config` represents the state established when the model was loaded.

Example:

```python
model = models.load("model", dtype="float16")
model.float()
```

Then:

```text
model.config.dtype         → float32
model.default_config.dtype → float16
```

## 8. reset()

`reset()` restores the model to its load-time/default state.

```python
model.reset()
```

Conceptually:

```text
load()
  ↓
DEFAULT STATE
  ↓
runtime modifications
  ↓
reset()
  ↓
DEFAULT STATE
```

`reset()` should not necessarily reload weights from disk. It may perform runtime operations required to restore the load-time state.

## 9. Runtime Device API

Common properties:

```python
model.device
model.devices
model.num_devices
model.device_map
```

Common operations:

```python
model.to(...)
model.cpu()
model.cuda()
```

The framework must not assume every backend supports every device operation.

## 10. Multi-GPU and device_map

The framework may support:

```python
model = models.load("model", device_map="auto")
```

Runtime inspection:

```python
model.device_map
model.devices
model.num_devices
```

Example:

```text
device_map = auto
devices = [cuda:0, cuda:1]
num_devices = 2
```

Moving from multi-GPU to single-GPU is a runtime redistribution operation and is not necessarily the same as `reset()`.

The framework should only perform redistribution when the backend/model supports it.

## 11. Dtype API

Common dtype operations may include:

```python
model.half()
model.float()
model.bfloat16()
```

Inspection:

```python
model.dtype
```

These are capabilities, not universal guarantees. Unsupported models must reject unsupported dtype operations cleanly.

## 12. Quantization

Quantization is separate from ordinary dtype conversion.

Examples:

```text
float32
float16
bfloat16
int8
int4
```

`int8` and `int4` should generally be treated as quantization capabilities. Runtime quantization and post-load quantization must not be assumed to work for every backend.

## 13. Unsupported Operations

Concrete models may declare operations they do not support.

Use a blacklist: only unsupported operations need to be declared.

Example:

```python
class GroundingDINO(Detector):
    __UNSUPPORTED__ = (
        "train", "int4", "int8",
    )

    def __init__(self, metadata):
        super().__init__(metadata)
```

The framework should check unsupported declarations before executing an operation.

## 14. Unsupported Inheritance Rule

If an operation is declared unsupported, the model must not expose a usable inherited implementation for that operation.

Example:

```python
class Detector(BaseModel):
    def half(self):
        ...
```

```python
class GroundingDINO(Detector):
    __UNSUPPORTED__ = ("half",)
```

Then:

```python
model.half()
```

must raise `UnsupportedOperationError`.

Ideally, invalid declarations are detected during class registration or model loading.

## 15. Unsupported vs Override

These concepts have different meanings.

### Unsupported

```python
__UNSUPPORTED__ = ("train",)
```

Meaning: the model does not support the operation.

### Override

```python
@override
def preprocess(self, image):
    ...
```

Meaning: the operation is supported but needs a different implementation.

A method should not be both unsupported and overridden. Such a declaration should be rejected as contradictory.

## 16. @override

Concrete models may replace inherited implementations:

```python
class GroundingDINO(Detector):
    @override
    def preprocess(self, image):
        ...
```

The decorator is useful for:

- documenting intent
- validating that a parent method exists
- detecting misspelled method names
- framework introspection

Python's normal method resolution still makes the child implementation take precedence.

## 17. Model-Specific Declaration Style

For a short unsupported list:

```python
class GroundingDINO(Detector):
    __UNSUPPORTED__ = ("train", "int4", "int8")

    def __init__(self, metadata):
        super().__init__(metadata)
```

For a long list:

```python
class GroundingDINO(Detector):
    __UNSUPPORTED__ = (
        "train", "int4", "int8",
        "device_map", "quantize", "compile",
        "export", "offload",
    )

    def __init__(self, metadata):
        super().__init__(metadata)
```

A class-level constant is preferred because unsupported capabilities normally do not vary per instance.

## 18. Method Introspection

Provide:

```python
model.methods()
```

The method can classify methods as:

```text
Supported
Overridden
Unsupported
Inherited
```

Example conceptual output:

```text
GroundingDINO
────────────────────

Supported:
    predict()
    to()
    cpu()
    cuda()
    half()

Overridden:
    preprocess()
    postprocess()

Unsupported:
    train()
    int4()
    int8()
```

This information should be derived automatically from class inspection and framework metadata.

## 19. supports()

Provide:

```python
model.supports("half")
```

Examples:

```python
model.supports("half")  # True
model.supports("int4")  # False
```

This is different from checking whether the method exists.

Conceptually:

```text
has_method()
    → Does the method exist?

supports()
    → Can this model actually use it?
```

An inherited method may exist but still be disabled by `__UNSUPPORTED__`.

## 20. info()

Provide:

```python
model.info()
```

It should expose a human-readable overview of the current model state.

Example:

```text
GroundingDINO
────────────────────────

State       : READY
Device      : cuda:0, cuda:1
Device map  : auto
Dtype       : float16

Config:
    resize    : 800
    pad       : true
    threshold : 0.3

Unsupported:
    train
    int4
    int8
```

## 21. Resource Management

Optional common APIs:

```python
model.memory_usage()
model.clear_cache()
model.unload()
```

### memory_usage()

Useful for single- and multi-GPU inspection.

Example:

```text
cuda:0 → 4.2 GB
cuda:1 → 3.8 GB
total  → 8.0 GB
```

### clear_cache()

Clears temporary runtime caches. It must not mean unloading the model.

### unload()

Releases model resources:

```text
READY
  ↓
UNLOADED
```

Inference after unloading should raise `InvalidStateError`. The framework should not silently reload the model.

## 22. Model State

The framework should maintain a clear lifecycle.

Conceptual states:

```text
LOADING
   ↓
READY
   ↓
MODIFIED
   ↓
RESET
   ↓
READY
   ↓
UNLOADED
```

Operations should validate whether they are legal in the current state.

## 23. Detector

`Detector` is a task-specific base class:

```python
class Detector(BaseModel):
    ...
```

It should contain detection-common behavior:

- detection inference contract
- preprocessing contract
- postprocessing contract
- detection result handling

It should not contain implementation details specific to one detector.

```text
GroundingDINO-specific logic → GroundingDINO
YOLO-specific logic         → YOLO
Generic detection logic     → Detector
```

## 24. Detector Example

```python
class GroundingDINO(Detector):

    __UNSUPPORTED__ = (
        "train", "int4", "int8",
    )

    def __init__(self, metadata):
        super().__init__(metadata)

    @override
    def preprocess(self, image, ...):
        ...

    @override
    def postprocess(self, outputs, ...):
        ...
```

The concrete model should implement only what differs from the base behavior.

## 25. Detection Pipeline

```text
Input
  ↓
preprocess()
  ↓
model forward
  ↓
postprocess()
  ↓
DetectResults
```

Example:

```python
results = detector.predict(image)
```

## 26. Input Configuration

Input-specific settings should not be part of `BaseModel` unless genuinely universal.

Examples:

```text
resize
pad
image_size
normalization
```

Possible namespace:

```python
model.config.input.resize
model.config.input.pad
```

## 27. Postprocess Configuration

Detection-specific postprocessing may include:

```text
threshold
text_threshold
box_threshold
nms
```

Possible namespace:

```python
model.config.post
```

Do not force these settings into the universal `BaseModel` interface.

## 28. Results Architecture

Results should be standardized:

```text
Results
├── DetectResults
├── ClassifyResults
├── SegmentResults
├── PoseResults
└── ...
```

Example:

```python
results = detector.predict(image)
```

returns `DetectResults`.

The Result hierarchy is part of the framework's public contract.

## 29. BaseModel API Summary

```text
BaseModel
│
├── Configuration
│   ├── metadata
│   ├── config
│   └── default_config
│
├── Runtime
│   ├── device
│   ├── devices
│   ├── num_devices
│   ├── device_map
│   └── dtype
│
├── Device / Dtype
│   ├── to()
│   ├── cpu()
│   ├── cuda()
│   ├── half()
│   ├── float()
│   └── bfloat16()
│
├── Lifecycle
│   ├── reset()
│   └── unload()
│
├── Introspection
│   ├── info()
│   ├── methods()
│   └── supports()
│
├── Resource
│   ├── memory_usage()
│   └── clear_cache()
│
└── Inference contract
    └── predict()
```

Training may exist in the common interface if a unified API is desired. Inference-only models can mark it unsupported.

## 30. What Not to Put in BaseModel

Do not put task-specific features into `BaseModel`:

```text
resize
pad
threshold
text_threshold
bounding boxes
masks
tracking
video-specific behavior
detector-specific postprocessing
```

Features such as the following should not automatically become mandatory APIs:

```text
compile
export
quantize
offload
redistribute
clone
profile
```

Add them only when their semantics are sufficiently universal, or expose them as optional/backend-specific capabilities.

## 31. Extensibility

Adding a new model should look approximately like:

```python
class NewModel(Detector):

    __UNSUPPORTED__ = (
        "train",
    )

    def __init__(self, metadata):
        super().__init__(metadata)

    @override
    def preprocess(...):
        ...

    @override
    def postprocess(...):
        ...
```

The framework core should not need modification.

## 32. Adding New Tasks

New task types extend `BaseModel`:

```text
BaseModel
├── Detector
├── Classifier
├── Segmenter
├── PoseEstimator
└── VLM
```

Each task base class defines its own contract while inheriting common runtime functionality.

## 33. Error Types

Recommended errors:

```text
UnsupportedOperationError
InvalidConfigError
InvalidStateError
InvalidDeviceError
InvalidDtypeError
```

Example:

```text
UnsupportedOperationError:
GroundingDINO does not support half().
```

Example:

```text
InvalidStateError:
Cannot call predict() because the model is unloaded.
```

## 34. Design Rules

### Rule 1 — Common behavior is inherited

If behavior is common, put it in the appropriate base class.

### Rule 2 — Task-specific behavior belongs in task classes

Detection behavior belongs in `Detector`, classification behavior in `Classifier`, etc.

### Rule 3 — Model-specific differences stay in the concrete model

Do not modify framework core for one model.

### Rule 4 — Use `@override` for replacement implementations

An override means the feature exists but requires model-specific behavior.

### Rule 5 — Use `__UNSUPPORTED__` for unavailable operations

Unsupported means the operation must not be usable for that model.

### Rule 6 — Prefer a blacklist

Declare only unsupported operations instead of listing every supported operation.

### Rule 7 — Keep BaseModel small

Do not turn BaseModel into a collection of unrelated utilities.

### Rule 8 — Introspection should be automatic

`methods()`, `supports()`, and `info()` should derive information from actual class definitions, metadata, runtime state, and capability declarations.

### Rule 9 — Runtime changes are different from load configuration

`reset()` restores load-time state; runtime operations such as `to()` or `half()` change current state.

### Rule 10 — Concrete models declare only differences

The framework should maximize reuse through inheritance while allowing precise model-specific exceptions.

## 35. Final Architecture

```text
┌────────────────────────────────────┐
│           Model Loader             │
│ models.load()                      │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│             BaseModel              │
│                                    │
│ config / metadata                  │
│ runtime / device / dtype           │
│ lifecycle                          │
│ unsupported / introspection        │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│          Task Abstraction          │
│                                    │
│ Detector / Classifier / VLM / ... │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│        Concrete Model              │
│                                    │
│ GroundingDINO / YOLO / ...         │
│                                    │
│ @override                          │
│ __UNSUPPORTED__                    │
└────────────────┬───────────────────┘
                 │
                 ▼
┌────────────────────────────────────┐
│              Results               │
│ DetectResults / ClassifyResults... │
└────────────────────────────────────┘
```

## 36. Core Philosophy

```text
COMMON BEHAVIOR
    ↓
inherit

DIFFERENT IMPLEMENTATION
    ↓
@override

NOT SUPPORTED
    ↓
__UNSUPPORTED__

COMMON CONFIGURATION
    ↓
config.json / metadata

CURRENT RUNTIME
    ↓
model.config / model.info()

RESTORE LOAD-TIME STATE
    ↓
reset()

RELEASE RESOURCES
    ↓
unload()
```

The framework should remain small at its core and rely on task-specific abstractions and optional capabilities for features that are not universal.

## 37. Implementation Priority

Recommended implementation order:

1. `BaseModel`
2. metadata/config loading
3. runtime state tracking
4. device handling
5. dtype handling
6. `__UNSUPPORTED__` validation
7. `@override` validation
8. `reset()`
9. `unload()`
10. `info()`
11. `methods()`
12. `supports()`
13. `memory_usage()`
14. `clear_cache()`
15. task base classes such as `Detector`
16. standardized `Results`

The core should be tested against multiple model types before adding more abstractions.

## 38. Final Requirement

The architecture must favor a small, stable universal interface and model-specific declarations.
A model implementation should not need to reimplement common runtime behavior merely because
its backend differs.

The framework should make unsupported capabilities explicit, make overrides discoverable,
and make current model configuration/runtime state inspectable.

The final objective is a framework that can safely support many different AI model backends
without turning `BaseModel` into an oversized abstraction.

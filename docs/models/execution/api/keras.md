# Keras native execution

Custom Keras wrappers can use the Detector convenience methods for KerasHub preprocessing, inference and detection conversion.

The former `klygo.models.backend.keras` import has been removed. Use a registered
model through `models.load`, or define a `models.Detector` subclass that owns
its native framework calls. Return standardized frames using
`outputs.build_detections(images, formatted, names=self.names)`.

[Detector convenience helpers](common.md) remain optional. Accelerator lifecycle
operations live in `models.runtime`, and reusable detection conversion stays
private beside the concrete wrappers.

See [execution responsibilities and prediction sessions](../README.md),
[Detector](../../api/Detector.md), and the
[four-framework smoke workflow](../../../integration/native-execution-smoke.md).

## Engine configuration

Configure `os.environ["KERAS_BACKEND"]` (for example `"torch"`, `"jax"`, or `"tensorflow"`) before importing Keras or loading a Keras model:

```python
import os

os.environ["KERAS_BACKEND"] = "torch"
```

Klygo does not probe Torch, JAX, and TensorFlow to select a compute engine automatically. Note that Keras 3 initializes its engine on first import; changing the `KERAS_BACKEND` environment variable after Keras has been imported does not switch the active backend.

## Built-in D-FINE Nano COCO

```python
from klygo import models

models.set_backend("Keras", engine="torch")  # Before importing Keras.
detector = models.load("dfine_nano_coco", device="cuda:0", threshold=0.25, iou=0.5)
results = detector.predict("image.jpg", prompt=["person", "car"], verbose=False)
# results is outputs.Detections; results[0] is outputs.Detection.
```

Install `klygo[keras-hub]` and the intended compute engine. `DFineDetect` is lazily
available under `klygo.models.detection`; importing Klygo does not load KerasHub.
The first preset load may download weights. `weights` optionally selects a local
KerasHub COCO preset directory. Non-80-class presets are rejected. Device
selection at load time requires the Torch engine; without a device override,
other engines use native placement and require their own runtime validation.

The wrapper executes public `preprocess_samples`, forward with `training=False`,
and `decode_predictions` once each. It does not use the generic adapter's silent
fallback. Errors propagate and decoder thresholds are restored even on failure.
This explicit-stage route avoids a tuple-unpacking failure observed in native
`predict_on_batch` on the installed KerasHub 0.32 stack; it does not claim full
`Model.predict` callback/dataset-loader lifecycle equivalence.

The preset controls its 640×640 preprocessing. Its center crop is preserved,
and boxes are inverted into original image coordinates, including integer crop
offsets. Cropped-away image content cannot be detected. No additional resize or
NMS stage is inserted. Equal-size inputs share a batch; mixed-size inputs are
grouped by size and returned in source order. Labels use the standard zero-based
80-class COCO vocabulary. Prompt filtering accepts only known COCO class names.
`threshold` and `iou` are call-local prediction overrides in [Detector](../../api/Detector.md).

Use `stream=True` for lazy results, and close partially consumed streams.
Output annotation storage follows the standard Detector contract; CUDA model
inference does not automatically imply CUDA annotation storage. Collections can
subsequently use `.cuda()` and the GPU metrics APIs.

[Runnable example](../../../../examples/models/dfine.py) ·
[Regression tests](../../../../test/models/test_dfine.py) ·
[Official D-FINE API](https://keras.io/keras_hub/api/models/d_fine/d_fine_object_detector/)

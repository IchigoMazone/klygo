# Native model execution and backend compatibility

`Detector` owns batching, streaming, preprocessing traces, metadata and standardized
outputs. Concrete wrappers own their native inference and result conversion.
The former `models/backend` package has been removed. Old imports from
`klygo.models.backend` must migrate; custom Detector convenience methods remain
supported.

## Responsibilities

| Location | Responsibility |
| --- | --- |
| `models/detection/base.py` | Managed pipeline and optional prediction-session lifecycle |
| `models/detection/yolo.py`, `yolo_world.py` | Native Ultralytics calls and predictor reuse |
| `models/detection/grounding_dino.py`, `locate_anything.py` | Transformers inputs and output-device alignment |
| `models/detection/faster_rcnn.py` | Native Torchvision inputs, inference and results |
| Private helpers beside detection wrappers | Framework-specific conversion shared by related models |
| `models/runtime/` | Lazy Torch/Keras device, precision, inference context, barriers and cache management |
| `models/_compat.py` | Historical custom-wrapper convenience dispatch |
| `models/execution/` | Thin aliases for existing imports |

New custom models only need to implement `forward(images, prompt=None)` and return
one standardized detection per image. Calling `self.cast_inputs`,
`self.run_inference` or `self.format_results` is optional; existing custom models
using those methods keep their previous framework behavior.

## Optional prediction sessions

A wrapper may override the private context-manager hook
`_prediction_session(options)` and yield an inference callable accepting
`images` and `prompt`. Managed prediction opens it once for all batches. A lazy
stream opens it on consumption and closes it on exhaustion, explicit close or
inference failure. The default yields `self.forward`; old custom Ultralytics
wrappers also retain transparent predictor preparation. The options argument
contains the resolved metadata for that prediction; hooks should read it rather
than relying on mutable global configuration. Session state should be local to
the context so interleaved streams do not overwrite each other.

YOLO's predictor optimization remains in its wrapper's session setup. Other
frameworks may use the same hook for their own preparation; no model is required
to implement an optimization hook. Native source decoding retains the backend's
own lifecycle.

## Migration

Use [Detector convenience methods and runtime helpers](api/common.md) or own
native inference in a wrapper. Framework guides:

- [Ultralytics](api/ultralytics.md)
- [Hugging Face](api/huggingface.md)
- [Torchvision](api/torchvision.md)
- [Keras](api/keras.md)

Accelerator lifecycle helpers also live in `klygo.models.runtime`. Importing that
module does not import Torch, Keras, Transformers or Ultralytics. Engines are
loaded only when selected. Framework-specific detection helpers are private
implementation details, not additional requirements for custom models.

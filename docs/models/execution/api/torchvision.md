# Torchvision native execution

Faster R-CNN and custom wrappers own tensor preparation, model calls and result conversion.

The former `klygo.models.backend.torchvision` import has been removed. Use a registered
model through `models.load`, or define a `models.Detector` subclass that owns
its native framework calls. Return standardized frames using
`outputs.build_detections(images, formatted, names=self.names)`.

[Detector convenience helpers](common.md) remain optional. Accelerator lifecycle
operations live in `models.runtime`, and reusable detection conversion stays
private beside the concrete wrappers.

See [execution responsibilities and prediction sessions](../README.md),
[Detector](../../api/Detector.md), and the
[four-framework smoke workflow](../../../integration/native-execution-smoke.md).

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

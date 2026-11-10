# Hugging Face native execution

Grounding DINO and Locate Anything own input casting, native execution and output-device alignment.

The former `klygo.models.backend.huggingface` import has been removed. Use a registered
model through `models.load`, or define a `models.Detector` subclass that owns
its native framework calls. Return standardized frames using
`outputs.build_detections(images, formatted, names=self.names)`.

[Detector convenience helpers](common.md) remain optional. Accelerator lifecycle
operations live in `models.runtime`, and reusable detection conversion stays
private beside the concrete wrappers.

See [execution responsibilities and prediction sessions](../README.md),
[Detector](../../api/Detector.md), and the
[four-framework smoke workflow](../../../integration/native-execution-smoke.md).

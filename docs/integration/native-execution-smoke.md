# Native execution across four framework families

Run from the repository root:

```bash
pip install -e '.[models]' keras timm
PYTHONPATH=. KERAS_BACKEND=torch python examples/integration/native_execution_smoke.py
```

The workflow runs four real pretrained detectors in isolated child processes:

| Framework | Model | Execution |
| --- | --- | --- |
| Ultralytics | YOLOv8n | Built-in `models.load` wrapper |
| Hugging Face | `facebook/detr-resnet-50` | Local Detector with native processor and model calls |
| Torchvision | SSDLite320 MobileNetV3 Large, COCO weights | Local Detector with native tensor/model calls |
| Keras, Torch engine | KerasHub `dfine_nano_coco` | Local Detector using optional convenience methods |

It downloads the public Ultralytics bus example and official model weights. Supply
`--source PATH` to use an existing image, `--backend NAME` to run one framework,
or `--artifacts PATH` to choose an output directory. CUDA is selected when
available; otherwise the workflow executes on CPU. KerasHub may install TensorFlow
as a package dependency even when its computation engine is Torch.

Each run checks a positive image and a generated black image, standard output
shape and finite coordinates, managed/stream numerical agreement, in-memory
native prediction, and early stream closure. Reports include complete outputs,
package versions, device, source digest and annotated previews. Runtime reports,
and images remain under `artifacts/integration/native-execution/`. Framework
checkpoints also use their standard Torch, Hugging Face and KaggleHub caches.

This is a correctness smoke run. It has no labeled ground truth and does not
measure precision/recall or isolated prediction FPS. A real four-framework run
must finish all four processes successfully; a skipped framework is not a pass.
The former `klygo.models.backend` package is neither imported nor required.

See the [source](../../examples/integration/native_execution_smoke.py),
[execution guide](../models/execution/README.md) and [Detector](../models/api/Detector.md).

## Recorded validation

A local CPU run completed all four child processes with exit code zero:

| Framework | Positive image detections | Black image detections | Managed/stream, native, early close |
| --- | ---: | ---: | --- |
| Ultralytics YOLOv8n | 4 | 0 | PASS |
| Hugging Face DETR ResNet-50 | 9 | 0 | PASS |
| Torchvision SSDLite320 | 5 | 0 | PASS |
| KerasHub D-FINE Nano, Torch engine | 5 | 0 | PASS |

Versions: Torch 2.15.0+cpu, Torchvision 0.29.2+cpu, Transformers 4.57.1,
Ultralytics 8.4.208, Keras 3.15.1 and KerasHub 0.33.0. The source image SHA256 is
`c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
Detection counts describe this smoke input, not model accuracy. No GPU throughput
claim is made by this run.

The repository suite completed 528 tests with 21 conditional skips, and all
13 module documentation/API audits passed. The run also caught and fixed
a native Ultralytics single-image fallback bug: prepared-batch optimization
now rejects standalone PIL images and arrays so the public predictor handles them.

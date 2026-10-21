# Torchvision COCO15 workflow

This workflow checks the Torchvision backend and the handoffs among `config`, `datasets`, `media`, `preprocessing`, `models`, `outputs`, `postprocessing`, `visual`, `files`, `archive`, and `utils` on real COCO128 images. It evaluates one positive and one negative image for each of `person`, `car`, `elephant`, and `umbrella`. This is an integration smoke run, not a generalization benchmark.

Build the existing COCO15 fixture (it stays ignored), install `klygo[torchvision]`, then run:

```powershell
python test/integration/build_coco15.py artifacts/integration/coco15 --download
python -m examples.integration.torchvision_coco15
```

Set `KLYGO_COCO15_DATASET` to a different compatible flat YOLO dataset and `KLYGO_ARTIFACT_ROOT` to a different result directory. The example puts the Torchvision weight cache, a provenance manifest, per-image TP/FP/FN, preview images, and a report archive below `artifacts/integration/torchvision-coco15/` by default. It defines an application-owned `Detector` subclass and uses Klygo's generic Torchvision backend hooks. SSDLite320 MobileNetV3 Large is only the workflow's concrete COCO_V1 evaluation case; Klygo's library does not register or construct it. First use may download model weights. No generated dataset or artifact should be committed.

See the [workflow source](../../examples/integration/torchvision_coco15.py) and [notebook](../../tutorials/integration/Torchvision_COCO15.ipynb).

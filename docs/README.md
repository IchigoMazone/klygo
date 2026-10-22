# Klygo Documentation Hub

Klygo is a modular, composable Python toolkit for filesystem operations, multi-format archives, hierarchical configuration, dataset engineering, lazy media streaming, coordinate-preserving image preprocessing, and **multi-backend AI object-detection pipelines** (**Ultralytics**, **Hugging Face**, **Torchvision**, and **Keras 3 / KerasHub**).

---

## 1. End-to-End Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Storage, Configuration & Dataset Layer                                   │
│    klygo.files  •  klygo.archive  •  klygo.config  •  klygo.datasets        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Media & Coordinate-Aware Preprocessing Layer                             │
│    klygo.media (LazyImage, MediaStream)  •  klygo.preprocessing (trace)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Multi-Backend Model Orchestration Layer (klygo.models)                   │
│    BaseModel ──► Detector ──► backend.common                                │
│      ├── Ultralytics (YOLOv8/v11, YOLO-World v2)                            │
│      ├── Hugging Face / PyTorch (Grounding DINO, DETR, custom nn.Module)    │
│      ├── Torchvision (Faster R-CNN, RetinaNet, FCOS, SSD)                   │
│      └── Keras 3 / KerasHub (RT-DETR / D-FINE, RetinaNet on torch/jax/tf)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Standardized Outputs, Postprocessing & Visualization Layer               │
│    klygo.outputs (Box, Detection, Detections)                               │
│    klygo.postprocessing (NMS, threshold, clip, box_iou)                     │
│    klygo.visual (draw_bboxes, show_image, plot_dataset_stats)               │
└─────────────────────────────────────────────────────────────────────────────┘
  Shared Foundation: klygo.utils (ProgressBar, human_size) • klygo.validators
```

---

## 2. Master Module Reference & Interactive Tutorials (12 Public Modules)

Every public Klygo module provides verified reference documentation (`docs/<module>/`), standalone executable examples (`examples/<module>/`), unit tests (`test/<module>/`), and 1-to-1 interactive notebooks (`tutorials/<module>/`):

| Module | Public APIs | Submodules | Reference Docs | Standalone Examples | Overview Notebook | Google Colab |
| :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| **[`klygo.files`](files/README.md)** | 43 | 5 | [Docs](files/README.md) ([`pathlib`](files/pathlib.md)) | [`examples/files/`](../examples/files) | [`Document_Files_Overview.ipynb`](../tutorials/files/Document_Files_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/files/Document_Files_Overview.ipynb) |
| **[`klygo.archive`](archive/README.md)** | 22 | 4 | [Docs](archive/README.md) ([`backend`](archive/backend/README.md)) | [`examples/archive/`](../examples/archive) | [`Document_Archive_Overview.ipynb`](../tutorials/archive/Document_Archive_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/archive/Document_Archive_Overview.ipynb) |
| **[`klygo.config`](config/README.md)** | 21 | 9 | [Docs](config/README.md) ([`metadata`](config/model-metadata.md)) | [`examples/config/`](../examples/config) | [`Document_Config_Overview.ipynb`](../tutorials/config/Document_Config_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/config/Document_Config_Overview.ipynb) |
| **[`klygo.datasets`](datasets/README.md)** | 8 | 8 | [Docs](datasets/README.md) | [`examples/datasets/`](../examples/datasets) | [`Document_Datasets_Overview.ipynb`](../tutorials/datasets/Document_Datasets_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/datasets/Document_Datasets_Overview.ipynb) |
| **[`klygo.media`](media/README.md)** | 15 | 1 | [Docs](media/README.md) | [`examples/media/`](../examples/media) | [`Document_Media_Overview.ipynb`](../tutorials/media/Document_Media_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/media/Document_Media_Overview.ipynb) |
| **[`klygo.preprocessing`](preprocessing/README.md)** | 137 | 4 | [Docs](preprocessing/README.md) | [`examples/preprocessing/`](../examples/preprocessing) | [`Document_Preprocessing_Overview.ipynb`](../tutorials/preprocessing/Document_Preprocessing_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/preprocessing/Document_Preprocessing_Overview.ipynb) |
| **[`klygo.models`](models/README.md)** | 15 | 4 | [Docs](models/README.md) ([`backend`](models/backend/README.md)) | [`examples/models/`](../examples/models) | [`Document_Models_Overview.ipynb`](../tutorials/models/Document_Models_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/models/Document_Models_Overview.ipynb) |
| **[`klygo.outputs`](outputs/README.md)** | 2 | 1 | [Docs](outputs/README.md) | [`examples/outputs/`](../examples/outputs) | [`Document_Outputs_Overview.ipynb`](../tutorials/outputs/Document_Outputs_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/outputs/Document_Outputs_Overview.ipynb) |
| **[`klygo.postprocessing`](postprocessing/README.md)** | 20 | 2 | [Docs](postprocessing/README.md) | [`examples/postprocessing/`](../examples/postprocessing) | [`Document_Postprocessing_Overview.ipynb`](../tutorials/postprocessing/Document_Postprocessing_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/postprocessing/Document_Postprocessing_Overview.ipynb) |
| **[`klygo.visual`](visual/README.md)** | 3 | 3 | [Docs](visual/README.md) | [`examples/visual/`](../examples/visual) | [`Document_Visual_Overview.ipynb`](../tutorials/visual/Document_Visual_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/visual/Document_Visual_Overview.ipynb) |
| **[`klygo.utils`](utils/README.md)** | 5 | 2 | [Docs](utils/README.md) | [`examples/utils/`](../examples/utils) | [`Document_Utils_Overview.ipynb`](../tutorials/utils/Document_Utils_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/utils/Document_Utils_Overview.ipynb) |
| **[`klygo.validators`](validators/README.md)** | 1 | 1 | [Docs](validators/README.md) | [`examples/validators/`](../examples/validators) | [`Document_Validators_Overview.ipynb`](../tutorials/validators/Document_Validators_Overview.ipynb) | [Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/validators/Document_Validators_Overview.ipynb) |

---

## 3. Backend Extension Layer Guides

- **[`klygo.models.backend`](models/backend/README.md)**: Unified [`common`](models/backend/api/common.md) dispatchers (`cast_inputs`, `run_inference`, `format_results`, `current_device`, `current_dtype`, `inference_context`, `synchronize`, `clear_cache`, `save`, `reset`, `unload`) and dedicated adapter guides for:
  - [Ultralytics (`YOLO` & `YOLO-World`)](models/backend/api/ultralytics.md)
  - [Hugging Face (`Grounding DINO` & `DETR`)](models/backend/api/huggingface.md)
  - [Torchvision (`Faster R-CNN`, `RetinaNet`, `SSD`)](models/backend/api/torchvision.md)
  - [Keras 3 & KerasHub (`RT-DETR / D-FINE`, `ObjectDetector`)](models/backend/api/keras.md)
- **[`klygo.archive.backend`](archive/backend/README.md)**: Capability matrix and codec adapters for `ZipBackend`, `TarBackend`, `GZipBackend`, `SevenZipBackend`, and `RarBackend`.

---

## 4. Real-Image Integration Workflows

See [`docs/integration/README.md`](integration/README.md) for cross-module evaluation workflows running against the 128-image COCO15 dataset:

- **[4-Backend Detection Benchmark](integration/multi-backend-workflow.md)** ([`multi_backend_workflow.py`](../examples/integration/multi_backend_workflow.py) • [Notebook](../tutorials/integration/Multi_Backend_COCO15.ipynb) • [Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/integration/Multi_Backend_COCO15.ipynb)): Evaluates **YOLOv8n**, **YOLO-World v2**, **Faster R-CNN**, and **RT-DETR / D-FINE** side by side.
- **[Local Detector Evaluation](integration/local-model-workflow.md)** ([`local_model_workflow.py`](../examples/integration/local_model_workflow.py)): End-to-end ground-truth evaluation pipeline.
- **[Torchvision on COCO15](integration/torchvision-coco15.md)** ([`torchvision_coco15.py`](../examples/integration/torchvision_coco15.py) • [Notebook](../tutorials/integration/Torchvision_COCO15.ipynb) • [Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/integration/Torchvision_COCO15.ipynb)): Full 11-module smoke workflow with Torchvision.

---

## 5. Conventions

- **Package Facades**: Import public APIs from their top-level package facade (`from klygo import files, archive, config, datasets, media, preprocessing, models, outputs, postprocessing, visual, utils, validators`).
- **Immutability by Default**: Functional configuration, mapping, and postprocessing operations return independent values unless an API explicitly documents in-place mutation.
- **Path Interoperability**: All file, archive, dataset, and media APIs accept `str` and text-returning `os.PathLike[str]` objects. Use `klygo.files` (`files.path`, `files.join`, `files.mkdir`, `files.save`, `files.load`) for runtime path and I/O operations.
- **Safe Publication**: Pass `overwrite=True` when replacing existing destination files or archives.
- **Clean Workspace**: Keep generated datasets, downloaded weights, and evaluation reports under `artifacts/` outside source, test, and documentation directories.

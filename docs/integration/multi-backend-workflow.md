# Multi-Backend Object Detection Workflow (4 Backends on COCO15)

This workflow evaluates **four distinct deep-learning backends** side by side through Klygo's unified [`models.Detector`](../models/api/Detector.md) interface on the real 128-image COCO15 dataset (`artifacts/integration/coco15`):

1. **Ultralytics YOLOv8** (`yolov8n.pt`, Backend: `"Ultralytics"`) — Fixed-vocabulary real-time detector.
2. **Ultralytics YOLO-World v2** (`yolov8s-worldv2.pt`, Backend: `"Ultralytics"`) — Open-vocabulary zero-shot detector with text prompts.
3. **Torchvision Faster R-CNN** (`fasterrcnn_mobilenet_v3_large_fpn` with `COCO_V1` weights, Backend: `"Torchvision"`) — Two-stage FPN detector excelling at small tabletop objects.
4. **KerasHub RT-DETR / D-FINE** (`dfine_small_coco` preset via `keras_hub.models.ObjectDetector`, Backend: `"Keras"` on `KERAS_BACKEND=torch`) — Transformer-based end-to-end detector running natively on PyTorch without TensorFlow.

---

## End-to-End Klygo Pipeline

$$\text{files} \rightarrow \text{datasets} \rightarrow \text{media \& preprocessing} \rightarrow \text{models (4 Backends)} \rightarrow \text{outputs} \rightarrow \text{postprocessing (NMS \& IoU)} \rightarrow \text{visual (draw\_bboxes)} \rightarrow \text{utils}$$

Every backend produces standardized [`klygo.outputs.Detections`](../outputs/README.md) objects (`xyxy` bounding boxes in original image coordinates, confidence scores, class IDs, and class names), allowing identical downstream NMS (`results.transform(postprocessing.nms())`), IoU ground-truth matching (`postprocessing.box_iou`), and bounding-box rendering (`visual.draw_bboxes`).

---

## Benchmark Results on COCO15 (`IoU >= 0.50`)

| Model | Backend | Assigned Classes | GT | Pred | TP | FP | FN | Precision | Recall | F1-Score | Latency (FPS) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** (`yolov8n.pt`) | `Ultralytics` | `person`, `car`, `knife`, `chair` | 5 | 4 | 4 | 0 | 1 | **1.000** | **0.800** | **0.889** | 133.4 ms (7.5 FPS) |
| **YOLO-World** (`yolov8s-worldv2.pt`) | `Ultralytics` | `umbrella`, `handbag`, `bottle`, `cup` | 4 | 4 | 3 | 1 | 1 | **0.750** | **0.750** | **0.750** | 263.8 ms (3.8 FPS) |
| **Faster R-CNN** (`fasterrcnn_mobilenet_v3_large_fpn`) | `Torchvision` | `wine glass`, `spoon`, `bowl`, `book` | 4 | 7 | 4 | 3 | 0 | **0.571** | **1.000** | **0.727** | 803.2 ms (1.2 FPS) |
| **RT-DETR / D-FINE** (`dfine_small_coco`) | `Keras` (`torch`) | `elephant`, `traffic light`, `teddy bear` | 6 | 6 | 5 | 1 | 1 | **0.833** | **0.833** | **0.833** | 1355.1 ms (0.7 FPS) |

---

## Running the Workflow

```powershell
python test/integration/build_coco15.py artifacts/integration/coco15 --download
python examples/integration/multi_backend_workflow.py
```

Generated outputs (kept outside Git in `artifacts/integration/four-backends-eval/`):
- **Structured JSON report**: `artifacts/integration/four-backends-eval/report/multi_backend_report.json`
- **Annotated preview images**: `artifacts/integration/four-backends-eval/preview/{yolo,yoloworld,torchvision,keras}_*.jpg`

See the executable implementation in [`examples/integration/multi_backend_workflow.py`](../../examples/integration/multi_backend_workflow.py) and the interactive notebook [`tutorials/integration/Multi_Backend_COCO15.ipynb`](../../tutorials/integration/Multi_Backend_COCO15.ipynb) ([Open in Colab](https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/integration/Multi_Backend_COCO15.ipynb)).

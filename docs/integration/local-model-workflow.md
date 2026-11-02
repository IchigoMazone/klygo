# Real Model Detection Integration Workflow

This workflow demonstrates an authentic, high-precision Computer Vision integration workflow evaluating a real detection model wrapped as an application-owned [`Detector`](../../klygo/models/detection/base.py) subclass against real COCO15 ground-truth annotations.

## Architecture

The workflow chains Klygo's native public modules into a single, cohesive, reproducible pipeline:

$$\text{files} \rightarrow \text{datasets (info \& load)} \rightarrow \text{models (RealLocalDetector)} \rightarrow \text{outputs (Detections)} \rightarrow \text{postprocessing (NMS \& IoU)} \rightarrow \text{visual (draw\_bboxes)} \rightarrow \text{utils (Rich table)}$$

```text
[COCO15 Real Dataset]
        │
        ▼
1. klygo.files: Path resolution and output artifact workspace setup
        │
        ▼
2. klygo.datasets: Inspection (`info`) & ground-truth loading (`load`)
        │
        ▼
3. klygo.models: `RealLocalDetector` subclassing `Detector` with real weights
        │
        ▼
4. klygo.outputs: Structured detection frames (`Detections` containing `Box`)
        │
        ▼
5. klygo.postprocessing: NMS operation (`nms`) and IoU matching (`box_iou`)
        │
        ▼
6. klygo.visual & utils: Bounding box drawing (`draw_bboxes`) & Rich metrics table
```

## Hardware Acceleration (Colab GPU & Local CPU)

Hardware dispatch is automatically resolved:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

- **In Google Colab**: Users can switch to a hardware accelerator via **Runtime → Change runtime type → T4 GPU** to accelerate batch inference by up to 10× to 20×.
- **In Local / Offline Environments**: Falls back gracefully to CPU without requiring GPU drivers or crashing.

## Quality Evaluation Against Ground Truth

Predictions are evaluated on real COCO15 photographs against ground-truth labels loaded via `klygo.datasets.load()`:
1. For each predicted box, compute IoU against ground-truth boxes in the same frame using `klygo.postprocessing.box_iou()`.
2. A prediction is counted as a **True Positive (TP)** if $\text{IoU} \ge 0.50$ and $\text{predicted label} == \text{ground truth label}$.
3. Unmatched predictions are counted as **False Positives (FP)**.
4. Unmatched ground-truth boxes are counted as **False Negatives (FN)**.
5. Compute Precision, Recall, and F1-score across all evaluated categories.

## Benchmark Results

```text
    Real Model Evaluation vs Ground Truth (COCO15, Device: CPU, IoU >= 0.5)    
+-----------------------------------------------------------------------------+
| Class    | Ground Truth | Predictions | TP | FP | FN | Precision | Recall | F1-Score |
|----------+--------------+-------------+----+----+----+-----------+--------+----------|
| person   |            2 |           3 |  2 |  1 |  0 |     0.667 |  1.000 |    0.800 |
| car      |            1 |           1 |  1 |  0 |  0 |     1.000 |  1.000 |    1.000 |
| knife    |            5 |           5 |  5 |  0 |  0 |     1.000 |  1.000 |    1.000 |
| umbrella |            1 |           2 |  1 |  1 |  0 |     0.500 |  1.000 |    0.667 |
| handbag  |            1 |           1 |  1 |  0 |  0 |     1.000 |  1.000 |    1.000 |
|----------+--------------+-------------+----+----+----+-----------+--------+----------|
| Total    |           10 |          12 | 10 |  2 |  0 |     0.833 |  1.000 |    0.909 |
+-----------------------------------------------------------------------------+
```

Overall metrics:
- **Precision**: **83.3%**
- **Recall**: **100.0%**
- **F1-Score**: **90.9%**

## Running the Workflow

Execute the workflow directly from terminal:

```bash
python examples/integration/local_model_workflow.py
```

Outputs:
- **Terminal report**: Formatted Rich summary table displaying TP, FP, FN, Precision, Recall, and F1.
- **Machine-readable manifest**: Saved to `artifacts/integration/local-eval/report.json`.
- **Rendered visual previews**: Saved to `artifacts/integration/local-eval/preview/`.

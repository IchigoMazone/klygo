"""Runnable example for metrics.f1_score."""
from klygo import metrics, outputs

pred = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9),
     outputs.Box((40, 40, 50, 50), label="cat", score=0.4)],
    source_id="example", image_size=(100, 100),
)
target = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="example", image_size=(100, 100),
)

# 1. Compute directly from precision and recall values.
print(metrics.f1_score(precision=0.8, recall=0.5))

# 2. Compute from Detection / Detections predictions and ground truth.
print(metrics.f1_score(pred, target, verbose=True))

# 3. Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.f1_score())
print(metrics.evaluate(pred, target, evaluator))

"""Runnable example for metrics.confusion_matrix."""
from klygo import metrics, outputs, utils

pred = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9),
     outputs.Box((40, 40, 50, 50), label="cat", score=0.4)],
    source_id="example", image_size=(100, 100),
)
target = outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="example", image_size=(100, 100),
)

result = metrics.confusion_matrix(pred, target, verbose=True)
print(result)

# Configure the same metric for grouped execution.
evaluator = metrics.compose(metrics.confusion_matrix())
print(metrics.evaluate(pred, target, evaluator))

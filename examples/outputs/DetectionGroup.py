"""Apply shared operations to independent same-source collections."""
import matplotlib
from PIL import Image
from klygo import outputs, metrics

matplotlib.use("Agg", force=True)

pred = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9)],
    source_id="image-a", image_size=(100, 100),
    source=Image.new("RGB", (100, 100), "white"),
)])
target = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="image-a", image_size=(100, 100),
    source=Image.new("RGB", (100, 100), "white"),
)])
collections = outputs.DetectionGroup(pred, target)
collections.astype(32).pack().unpack()
print(pred.dtype, target.dtype)
print(list(collections.keys()))  # ['pred', 'truth']
print(metrics.precision_recall(collections["pred"], collections["truth"]))
named = outputs.DetectionGroup({"model_a": pred, "ground_truth": target})
named.rename({"model_a": "pred", "ground_truth": "truth"})
print(list(named.keys()))
assert named["pred"] is pred
# With CuPy and CUDA installed:
# collections.cuda(0)
# print(metrics.evaluate(pred, target))
# collections.cpu()

# Display matching annotations through the public visual API.
collections.show(0, backend="matplotlib")
named.show(0, members=("truth", "pred"), backend="matplotlib")

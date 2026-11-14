"""Apply shared operations to independent same-source collections."""
from klygo import outputs, metrics

pred = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9)],
    source_id="image-a", image_size=(100, 100),
)])
target = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="image-a", image_size=(100, 100),
)])
collections = outputs.DetectionGroup(pred, target)
collections.astype(32).pack().unpack()
print(pred.dtype, target.dtype)
print(list(collections.keys()))  # ['pred', 'truth']
print(metrics.precision_recall(collections["pred"], collections["truth"]))
named = outputs.group({"model_a": pred, "ground_truth": target})
named.rename({"model_a": "pred", "ground_truth": "truth"})
print(list(named.keys()))
assert named["pred"] is pred
# With CuPy and CUDA installed:
# collections.cuda(0)
# print(metrics.evaluate(pred, target))
# collections.cpu()

"""Create structured detection results without a model backend."""

from PIL import Image

from klygo import outputs


image = Image.new("RGB", (80, 60), "white")
box = outputs.detect.Box(0, "item", 0.92, [10, 10, 40, 35], class_id=3)
frame = outputs.detect.Detection(image, [box], names={3: "item"})
results = outputs.detect.Detections([frame], source_type="image", names={3: "item"})
print("Frames:", len(results))
print("First crop size:", results[0][0].image.size)
print("Classes:", results.names)


class ReplacementModel:
    def predict(self, images, **kwargs):
        predictions = [{"boxes": [[5, 5, 25, 30]], "scores": [0.85],
                        "labels": ["dog"], "class_ids": [3]} for _ in images]
        frames = outputs.build_detections(images, predictions,
                                         names={3: "dog"}, metadata={"model_id": "replacement", **kwargs})
        return outputs.detect.Detections(frames)


results.repredict(ReplacementModel(), index=0, threshold=0.5)
print("Canonical vocabulary:", results.names)
print("Present classes:", results.active_names)
print("Frame identity:", results[0].frame_index)
print("Original backend class ID:", results[0][0].metadata.get("native_class_id"))

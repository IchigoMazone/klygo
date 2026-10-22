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

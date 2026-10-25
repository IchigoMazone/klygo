"""Create one Detections collection."""

from PIL import Image

from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.Box([10, 10, 40, 35], label="item", score=0.92, class_id=3)
frame = outputs.Detection([box], source_id="example", image_size=image.size, source=image, names={3: "item"})
results = outputs.Detections([frame], names={3: "item"})
results.drop(["item"])
print(results.to_dict())

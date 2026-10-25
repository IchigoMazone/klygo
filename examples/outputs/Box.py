"""Create and mutate one Box."""

from PIL import Image
from klygo import outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.Box([10, 10, 40, 35], label="item", score=0.92, class_id=3)
box.translate(2, 1)
frame = outputs.Detection([box], source_id="example", image_size=image.size, source=image)
print(frame[0].to_dict())

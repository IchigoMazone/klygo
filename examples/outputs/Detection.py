"""Create one Detection frame."""

from PIL import Image

from klygo import files, outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.Box([10, 10, 40, 35], label="item", score=0.92, class_id=3)
frame = outputs.Detection([box], source_id="example", image_size=image.size, source=image, names={3: "item"})
frame.add_box([45, 10, 60, 30], label="other")
selected = frame[[1, 0]]
selected.save(
    files.join("artifacts", "examples", "outputs", "detection.png"),
    mode="uid", overwrite=True,
)
print(frame.device, frame.dtype)
print(selected.to_dict())

other = outputs.Detection(
    [outputs.Box((12, 12, 42, 37), label="item", score=0.8)],
    source_id=frame.source_id, frame_index=frame.frame_index,
    image_size=frame.image_size,
)
frame.combine(other, method="weighted", iou=0.5)
print("Combined boxes:", frame.boxes)

print("Source image size:", frame.image.size)

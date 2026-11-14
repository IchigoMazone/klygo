"""Create one Detections collection."""

from PIL import Image

from klygo import files, outputs

image = Image.new("RGB", (80, 60), "white")
box = outputs.Box([10, 10, 40, 35], label="item", score=0.92, class_id=3)
frame = outputs.Detection([box], source_id="example", image_size=image.size, source=image, names={3: "item"})
results = outputs.Detections([frame], names={3: "item"}, packed=True)
print("Packed:", results.is_packed)
results.unpack().pack()
results.astype(16).astype(32)
results.drop("missing").relabel({"item": "object"}).compact()
results.save_images(
    files.join("artifacts", "examples", "outputs", "frames"),
    mode="index", overwrite=True,
)
print(results.to_dict())

other = outputs.Detections([
    outputs.Detection(
        [outputs.Box((12, 12, 42, 37), label="object", score=0.8)],
        source_id=frame.source_id, frame_index=frame.frame_index,
        image_size=frame.image_size,
    )
])
results.combine(other, method="weighted", iou=0.5)
print("Combined collection:", results.to_dict())

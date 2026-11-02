"""Build Klygo detection objects from normalized backend predictions."""

from PIL import Image

from klygo import outputs


image = Image.new("RGB", (80, 60), "white")
frames = outputs.build_detections(
    [image],
    [{
        "boxes": [[10, 10, 40, 35]],
        "scores": [0.92],
        "labels": ["item"],
        "class_ids": [3],
    }],
    metadata={"backend": "custom"},
    names={3: "item"},
)

print(frames[0])
print(frames[0][0])
print(frames[0].names)

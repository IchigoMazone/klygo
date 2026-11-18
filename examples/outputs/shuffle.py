"""Shuffle Detections in place using packed or unpacked storage."""

from tempfile import TemporaryDirectory
from PIL import Image

from klygo import files, media
from klygo.outputs import Box, Detection, Detections


with TemporaryDirectory() as directory:
    root = files.path(directory)
    frames = []
    for i in range(5):
        img_path = files.join(root, f"frame_{i}.jpg")
        media.save(img_path, Image.new("RGB", (100, 100), "white"), overwrite=True, verbose=False)
        box = Box([10.0, 10.0, 50.0, 50.0], label="car", score=0.9, class_id=0)
        frames.append(Detection(
            boxes=[box],
            source=media.load(img_path, verbose=False)[0],
            source_id="demo",
            frame_index=i,
            image_size=(100, 100),
            names={0: "car"},
        ))

    # Packed collection shuffle
    dets = Detections(frames, names={0: "car"}, packed=True)
    print("Initial packed frame indices:", [f.frame_index for f in dets])

    dets.shuffle(seed=42)
    print("Shuffled packed frame indices:", [f.frame_index for f in dets])

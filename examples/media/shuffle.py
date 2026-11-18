"""Shuffle MediaFrames in place."""

from tempfile import TemporaryDirectory
from PIL import Image

from klygo import files, media


with TemporaryDirectory() as directory:
    root = files.path(directory)
    for i in range(5):
        img_path = files.join(root, f"image_{i:02d}.jpg")
        media.save(img_path, Image.new("RGB", (64, 64), color="blue"), overwrite=True, verbose=False)

    frames = media.load(root, indexed=True, verbose=False)
    print("Initial order:", [files.name(f.path) for f in frames])

    frames.shuffle(seed=42)
    print("Shuffled order:", [files.name(f.path) for f in frames])

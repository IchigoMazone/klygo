"""Demonstrate media.MediaFrames in indexed and streaming modes."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import files, media

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    image = Image.new("RGB", (32, 24), "red")
    media.save(root / "source.png", image)

    frames = media.load(root)
    stream = media.load(root, stream=True)
    print(isinstance(frames, media.MediaFrames), frames.total_frames, frames.loaded_count)
    print(isinstance(stream, media.MediaFrames), stream.is_stream, [frame.size for frame in stream])

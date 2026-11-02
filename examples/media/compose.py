"""Demonstrate media.compose."""

import sys

# Remove this examples directory so copy.py cannot shadow Python's stdlib copy.
sys.path.pop(0)

import tempfile
from PIL import Image
from klygo import files, media, preprocessing as p

with tempfile.TemporaryDirectory() as temporary:
    root = files.path(temporary)
    for index in range(5):
        image = Image.new("RGB", (32, 24), color=(index * 30, index * 30, index * 30))
        media.save(root / f"frame_{index}.png", image)

    pipeline = media.compose(
        media.quality(min_brightness=0.1),
        media.sample(1),
        p.resize((16, 16)),
    )

    with media.load(root, stream=True) as stream:
        processed_stream = pipeline(stream)
        stream_sizes = [frame.size for frame in processed_stream]

    frames = media.load(root, stream=False)
    processed_frames = pipeline(frames)
    frame_sizes = [frame.size for frame in processed_frames]

    print(stream_sizes == frame_sizes, len(frame_sizes))

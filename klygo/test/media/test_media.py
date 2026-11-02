import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import cv2

import klygo.media as media
import klygo.processing as processing


def test_media_stream_is_separate_from_load(tmp_path):
    video_path = tmp_path / "stream.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        12,
        (24, 16),
    )
    for value in range(5):
        writer.write(np.full((16, 24, 3), value, dtype=np.uint8))
    writer.release()

    loaded = media.load(video_path)
    streamed = media.stream(video_path, sample_rate=2, max_frames=2)

    assert isinstance(loaded, media.MediaFrames)
    assert isinstance(streamed, media.MediaStream)
    assert not isinstance(streamed, list)
    assert streamed.is_stream
    assert streamed.total_frames == 2

    frames = list(streamed)
    assert [frame.frame_index for frame in frames] == [0, 2]
    assert all(isinstance(frame, media.LazyImage) for frame in frames)
    assert all(not frame.is_loaded for frame in frames)


def test_mediaframes_collection_api_is_lazy_and_keeps_metadata(tmp_path):
    image_dir = tmp_path / "collection"
    image_dir.mkdir()
    for index, color in enumerate(("red", "green", "blue")):
        Image.new("RGB", (20, 10), color).save(image_dir / f"{index}.png")

    frames = media.load(image_dir)
    assert frames.total_frames == 3
    assert frames.source == str(image_dir)
    assert frames.loaded_count == 0

    transformed = frames.transform(processing.resize((8, 8)))
    assert isinstance(transformed, media.MediaFrames)
    assert transformed.source_path == frames.source_path
    assert transformed[0].size == (8, 8)
    assert transformed.loaded_count == 0

    selected = frames.where(lambda image: image.path.name != "1.png")
    assert [image.path.name for image in selected] == ["0.png", "2.png"]
    assert selected.source_path == frames.source_path

    copied = frames.copy()
    assert isinstance(copied, media.MediaFrames)
    assert copied.source_path == frames.source_path

    frames[0].load()
    assert frames.loaded_count == 1
    frames.unload()
    assert frames.loaded_count == 0

    del frames[0]
    assert frames.total_frames == 2
    frames.clear()
    assert frames.total_frames == 0


def test_media_conversions():
    img_pil = Image.new("RGB", (20, 10), "red")

    # 1. to_array
    arr = media.to_array(img_pil)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10, 20, 3)

    # 2. to_tensor
    tensor = media.to_tensor(img_pil, normalize=True)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 10, 20)
    assert tensor.dtype == torch.float32

    # 3. to_pil from tensor & array
    pil_from_tensor = media.to_pil(tensor)
    assert isinstance(pil_from_tensor, Image.Image)
    assert pil_from_tensor.size == (20, 10)

    pil_from_arr = media.to_pil(arr)
    assert isinstance(pil_from_arr, Image.Image)
    assert pil_from_arr.size == (20, 10)


def test_media_batch_saving_and_iter(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    f1 = Image.new("RGB", (30, 20), "red")
    f2 = Image.new("RGB", (30, 20), "green")
    f3 = Image.new("RGB", (30, 20), "blue")
    frames = [f1, f2, f3]

    # 1. save_images
    batch_dir = tmp_path / "saved_batch"
    saved_files = media.save_images(batch_dir, frames, prefix="frame", extension=".jpg")
    assert len(saved_files) == 3
    assert saved_files[0].name == "frame_000001.jpg"

    # 2. iter_frames on directory
    iter_imgs = list(media.iter_frames(batch_dir, sample_rate=2))
    assert len(iter_imgs) == 2

    # 3. save_video
    vid_p = tmp_path / "test_video.mp4"
    try:
        media.save(tmp_path / "saved.png", frames[0], overwrite=False, verbose=False)
        media.save(tmp_path / "saved.png", frames[0], overwrite=False, verbose=False)
        raise AssertionError("Should have raised FileExistsError")
    except FileExistsError:
        pass

    saved_vid = media.save_video(vid_p, frames, fps=10, overwrite=True)
    assert saved_vid.exists()

    # 4. probe video metadata
    v_info = media.probe(vid_p)
    assert v_info["type"] == "video"
    assert v_info["width"] == 30
    assert v_info["height"] == 20

    # 5. iter_frames on video
    v_frames = list(media.iter_frames(vid_p, sample_rate=1))
    assert len(v_frames) == 3

    # 6. media.convert
    png_p = tmp_path / "saved.png"
    jpg_p = tmp_path / "converted.jpg"
    conv_res = media.convert(png_p, jpg_p, overwrite=True, verbose=False)
    assert conv_res.exists()

    # 7. media.copy
    copied_p = tmp_path / "copied.png"
    cp_res = media.copy(png_p, copied_p, overwrite=True)
    assert cp_res.exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp_p = Path(td)
        test_media_conversions()
        test_media_batch_saving_and_iter(tmp_p)
        print("ALL KLYGO.MEDIA 11 APIS TESTS PASSED SUCCESSFULLY!")

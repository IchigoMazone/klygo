import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import torch

import klygo.media as media


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


def test_media_padding():
    img_pil = Image.new("RGB", (20, 10), "blue")

    # 1. pad
    padded = media.pad(img_pil, pad_width=5)
    assert media.to_array(padded).shape == (20, 30, 3)

    # 2. pad_to_shape
    padded_shape = media.pad_to_shape(img_pil, (30, 40))
    assert media.to_array(padded_shape).shape == (30, 40, 3)

    # 3. letterbox
    lb_img, ratio, (pad_x, pad_y) = media.letterbox(img_pil, (640, 640))
    assert media.to_array(lb_img).shape == (640, 640, 3)
    assert pad_y > 0


def test_media_kernels_and_filters():
    img_pil = Image.new("RGB", (50, 50), "white")

    # 1. filter2d
    k = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    filtered = media.filter2d(img_pil, k)
    assert isinstance(filtered, Image.Image)

    # 2. gaussian_blur
    blurred = media.gaussian_blur(img_pil, kernel_size=5, sigma=1.5)
    assert isinstance(blurred, Image.Image)

    # 3. sharpen
    sharp = media.sharpen(img_pil, factor=1.5)
    assert isinstance(sharp, Image.Image)

    # 4. edge_detection
    edges = media.edge_detection(img_pil, method="sobel")
    assert isinstance(edges, Image.Image)
    assert edges.size == (50, 50)


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
    # 6. overwrite check
    try:
        media.save(tmp_path / "saved.png", frames[0], overwrite=False, verbose=False)
        media.save(tmp_path / "saved.png", frames[0], overwrite=False, verbose=False)
        raise AssertionError("Should have raised FileExistsError")
    except FileExistsError:
        pass

    saved_vid = media.save_video(vid_p, frames, fps=10, overwrite=True)
    assert saved_vid.exists()

    # 4. info on video
    v_info = media.info(vid_p)
    assert v_info["type"] == "video"
    assert v_info["width"] == 30
    assert v_info["height"] == 20

    # 5. iter_frames on video
    v_frames = list(media.iter_frames(vid_p, sample_rate=1))
    assert len(v_frames) == 3


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tmp_p = Path(td)
        test_media_conversions()
        test_media_padding()
        test_media_kernels_and_filters()
        test_media_batch_saving_and_iter(tmp_p)
        print("ALL KLYGO.MEDIA VIDEO, IMAGES & ITERATION TESTS PASSED SUCCESSFULLY!")

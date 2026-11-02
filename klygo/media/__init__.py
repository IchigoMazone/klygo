"""Image and video I/O, lazy frames, and tensor conversion.

Google Colab guide:
https://colab.research.google.com/github/IchigoMazone/klygo/blob/v2-release/tutorials/media/Document_Media_Overview.ipynb

Supported images: PNG, JPEG, WebP, BMP, and TIFF.
Supported videos: MP4, AVI, MOV, MKV, M4V, and WebM.

Public functions
----------------
load
    Load indexed lazy frames; use ``stream=True`` for bounded-memory iteration.
save
    Write an image to disk.
convert
    Convert an image or video file to another format.
copy
    Copy a media file or directory.
save_video
    Encode frames as a video.
save_images
    Write a sequence of images to a directory.
iter_frames
    Iterate over decoded video frames.
probe
    Inspect image or video metadata.
to_array, to_tensor, to_pil
    Convert between NumPy, PyTorch, and Pillow representations.
compose
    Compose sequential filtering, sampling, and transformation pipelines.
"""

from .operations import (
    load,
    save,
    convert,
    copy,
    save_video,
    save_images,
    iter_frames,
    probe,
    to_array,
    to_tensor,
    to_pil,
    MediaFrames,
    LazyImage,
    VideoReader,
    compose,
    quality,
    where_quality,
    motion,
    where_motion,
    deduplicate,
    sample,
    limit,
    where,
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
)

__all__ = [
    "load",
    "save",
    "convert",
    "copy",
    "save_video",
    "save_images",
    "iter_frames",
    "probe",
    "to_array",
    "to_tensor",
    "to_pil",
    "MediaFrames",
    "LazyImage",
    "VideoReader",
    "compose",
]

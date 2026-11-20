"""Side-by-side previews sourced directly from Detection frames."""
from typing import Optional, Union
from os import PathLike
from PIL import Image, ImageDraw
from klygo import media, outputs, validators
from .draw_bboxes import draw_bboxes
from .show_image import show_image

__all__ = ["compare_images"]


def compare_images(
    first: outputs.Detection,
    second: outputs.Detection,
    *,
    mode: str = "show",
    output_path: Optional[Union[str, PathLike[str]]] = None,
    titles: Union[tuple[str, str], list[str]] = ("A", "B"),
    show_labels: bool = True,
    show_scores: bool = True,
    backend: str = "auto",
    overwrite: bool = False,
) -> Image.Image:
    """Compare two Detection frames using each frame's own image and annotations.

    Parameters
    ----------
    first, second : Detection
        Frames containing source images or usable source_path values, plus boxes
        and labels. Different sources and image sizes are supported. Select a
        frame first when using a Detections collection.
    mode : str, default='show'
        'show' displays the comparison; 'save' writes it to output_path.
    output_path : str or os.PathLike[str], optional
        Required destination for mode='save'; invalid for other modes.
    titles : tuple or list of two str, optional
        Captions above the left and right panels. Defaults to A and B.
    show_labels : bool, default=True
        Include text labels above boxes.
    show_scores : bool, default=True
        Include confidence scores next to labels.
    backend : str, default='auto'
        Display backend forwarded to show_image in show mode.
    overwrite : bool, default=False
        Allow replacing an existing output file in save mode.

    Returns
    -------
    PIL.Image.Image
        New RGB preview in every mode. Original scales and aspect ratios are
        retained, with blank padding below the shorter image. Source images
        and annotations are unchanged; resolving pixels may cache the source.

    Raises
    ------
    TypeError
        Inputs, titles, or options have unsupported types.
    ValueError
        Mode, title count, output_path usage, or source dimensions are invalid.
    RuntimeError
        A Detection has neither source pixels nor a usable source path.
    FileNotFoundError
        A source path no longer exists.
    FileExistsError
        The output exists and overwrite=False.

    Notes
    -----
    No box matching, merging, or evaluation is performed. In show mode, desktop
    auto display can open a viewer and OpenCV waits for a key press. Save mode
    never opens a viewer. Colors follow draw_bboxes per panel.
    """
    validators.validate_type(first, outputs.Detection, "first")
    validators.validate_type(second, outputs.Detection, "second")
    validators.validate_type(mode, str, "mode")
    if mode not in ("show", "save"):
        raise ValueError("mode must be 'show' or 'save'")
    if mode == "save":
        if output_path is None:
            raise ValueError("output_path is required for save mode")
        validators.validate_type(output_path, (str, PathLike), "output_path")
    elif output_path is not None:
        raise ValueError("output_path is only supported in save mode")
    validators.validate_type(titles, (list, tuple), "titles")
    if len(titles) != 2:
        raise ValueError("titles must contain exactly two strings")
    for title in titles:
        validators.validate_type(title, str, "title")
    validators.validate_type(show_labels, bool, "show_labels")
    validators.validate_type(show_scores, bool, "show_scores")
    validators.validate_type(backend, str, "backend")
    validators.validate_type(overwrite, bool, "overwrite")
    panels = []
    for detection in (first, second):
        panel = detection.image.convert("RGB")
        if panel.size != detection.image_size:
            raise ValueError("each detection image_size must match its source image")
        panels.append(draw_bboxes(panel, detection, show_labels=show_labels, show_scores=show_scores))
    header, gap = 28, 12
    preview = Image.new("RGB", (panels[0].width + gap + panels[1].width,
                                max(panel.height for panel in panels) + header), "white")
    draw = ImageDraw.Draw(preview)
    offset = 0
    for panel, title in zip(panels, titles):
        preview.paste(panel, (offset, header))
        draw.text((offset + 4, 6), title, fill="black")
        offset += panel.width + gap
    if mode == "show":
        show_image(preview, title="Image comparison", backend=backend)
    elif mode == "save":
        media.save(output_path, preview, overwrite=overwrite, verbose=False)
    return preview

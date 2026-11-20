from os import PathLike
from typing import Tuple, Union, Optional
import PIL.Image
import numpy as np

from ._render import _finalize_figure

__all__ = ["show_image"]


def _is_notebook() -> bool:
    """Check whether the active runtime environment is Jupyter Notebook / Colab / Kaggle."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return False
        shell_name = ip.__class__.__name__
        if shell_name in ("ZMQInteractiveShell", "Shell") or "google.colab" in str(type(ip)):
            return True
        return False
    except Exception:
        return False


def show_image(
    image: Union[PIL.Image.Image, np.ndarray],
    title: str = "Image",
    width: Optional[int] = None,
    height: Optional[int] = None,
    backend: str = "auto",
    figsize: Tuple[int, int] = (10, 10),
    *,
    show: bool = True,
    save_path: Optional[Union[str, PathLike[str]]] = None,
    dpi: int = 300,
    return_figure: bool = False,
):
    """Display an image using the notebook, desktop, or selected backend.

    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Image to display. NumPy images use RGB channel order.
    title : str
        Window or plot title.
    width : int, optional
        Positive display width; height is scaled proportionally. Mutually
        exclusive with ``height``.
    height : int, optional
        Positive display height; width is scaled proportionally. Mutually
        exclusive with ``width``.
    backend : str
        ``"auto"`` displays inline in a notebook and opens a desktop viewer
        elsewhere. ``"matplotlib"`` uses pyplot; other values use OpenCV.
    figsize : tuple of int
        Matplotlib figure size in inches.

    Returns
    -------
    None
        Displays the image without changing the input.

    Raises
    ------
    ValueError
        If both ``width`` and ``height`` are specified.

    Notes
    -----
    The OpenCV backend blocks until a key is pressed.
    """
    if width is not None and height is not None:
        raise ValueError("Specify either width or height, not both")

    is_np = isinstance(image, np.ndarray)
    if is_np:
        pil_img = PIL.Image.fromarray(image if image.ndim == 2 or image.shape[2] == 3 else image[:, :, :3])
    else:
        pil_img = image

    # 1. Proportional resize if width or height is specified
    if width is not None and width > 0:
        w_orig, h_orig = pil_img.size
        if w_orig > 0:
            scale = width / w_orig
            new_h = max(1, int(h_orig * scale))
            pil_img = pil_img.resize((width, new_h), PIL.Image.Resampling.BILINEAR)
    elif height is not None and height > 0:
        w_orig, h_orig = pil_img.size
        if h_orig > 0:
            scale = height / h_orig
            new_w = max(1, int(w_orig * scale))
            pil_img = pil_img.resize((new_w, height), PIL.Image.Resampling.BILINEAR)

    # 2. Display backend dispatch
    if backend == "auto":
        if _is_notebook():
            try:
                from IPython.display import display
                display(pil_img)
                return
            except Exception:
                pass
        # Default desktop viewer (avoid subclass .show() recursion such as LazyImage.show)
        if isinstance(pil_img, PIL.Image.Image) and pil_img.__class__.show is not PIL.Image.Image.show:
            super(type(pil_img), pil_img).show(title=title)
        else:
            pil_img.show(title=title)
        return

    if backend.lower() == "matplotlib" or return_figure or save_path is not None or not show:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
        plt.imshow(pil_img)
        plt.title(title)
        plt.axis("off")
        return _finalize_figure(
            fig,
            show=show,
            save_path=save_path,
            dpi=dpi,
            return_figure=return_figure,
        )

    # Backend OpenCV
    import cv2 as cv
    img_cv = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    cv.imshow(title, img_cv)
    cv.waitKey(0)
    cv.destroyAllWindows()


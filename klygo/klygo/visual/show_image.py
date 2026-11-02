from typing import Tuple, Union, Optional
import PIL.Image
import numpy as np


def _is_notebook() -> bool:
    """Kiểm tra môi trường có phải là Jupyter Notebook / Colab / Kaggle hay không."""
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
    backend: str = "auto",
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Tác dụng:
    - Hiển thị ảnh thông minh, tự động nhận diện Jupyter Notebook / Colab / Desktop.

    Đầu vào:
    - image: Ảnh PIL hoặc mảng NumPy
    - title: Tiêu đề của cửa sổ hiển thị
    - width: Chiều rộng hiển thị (pixels)
    - backend: 'auto', 'matplotlib', hoặc 'opencv'
    - figsize: Kích thước figure của Matplotlib

    Đầu ra:
    - Không trả về dữ liệu (hiển thị trực tiếp)
    """
    is_np = isinstance(image, np.ndarray)
    if is_np:
        pil_img = PIL.Image.fromarray(image if image.ndim == 2 or image.shape[2] == 3 else image[:, :, :3])
    else:
        pil_img = image

    # 1. Nếu chỉ định width -> scale tỉ lệ ảnh
    if width is not None and width > 0:
        w_orig, h_orig = pil_img.size
        if w_orig > 0:
            scale = width / w_orig
            new_h = max(1, int(h_orig * scale))
            pil_img = pil_img.resize((width, new_h), PIL.Image.Resampling.BILINEAR)

    # 2. Xử lý backend hiển thị
    if backend == "auto":
        if _is_notebook():
            try:
                from IPython.display import display
                display(pil_img)
                return
            except Exception:
                pass
        # Desktop mặc định
        pil_img.show(title=title)
        return

    if backend.lower() == "matplotlib":
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        plt.imshow(pil_img)
        plt.title(title)
        plt.axis("off")
        plt.show()
        return

    # Backend OpenCV
    import cv2 as cv
    img_cv = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    cv.imshow(title, img_cv)
    cv.waitKey(0)
    cv.destroyAllWindows()


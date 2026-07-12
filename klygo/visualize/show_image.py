from typing import Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


def show_image(
    image: Union[Image.Image, np.ndarray],
    title: str = "Image",
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """
    Tác dụng:
    - Hiển thị ảnh bằng Matplotlib hoặc OpenCV

    Đầu vào:
    - image: Ảnh PIL hoặc mảng NumPy
    - title: Tiêu đề của cửa sổ hiển thị
    - backend: Công cụ hiển thị ảnh
    - figsize: Kích thước figure của Matplotlib

    Đầu ra:
    - Không trả về dữ liệu

    Nguồn: TrinhNhuNhat_12072026.
    """
    is_pil = isinstance(image, Image.Image)

    if backend.lower() == "matplotlib":
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)
        plt.imshow(image if is_pil else cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
        return

    img_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR) if is_pil else image
    cv.imshow(title, img_cv)
    cv.waitKey(0)
    cv.destroyAllWindows()

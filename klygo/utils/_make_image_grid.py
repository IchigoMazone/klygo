from math import ceil
from typing import List, Tuple

from PIL import Image


def _make_image_grid(
    images: List[Image.Image],
    grid: Tuple[int, ...],
) -> Image.Image:
    """
    Tác dụng:
    - Ghép danh sách ảnh thành một ảnh lưới

    Đầu vào:
    - images: Danh sách ảnh PIL
    - grid: Số cột hoặc số cột và số hàng của lưới

    Đầu ra:
    - Ảnh PIL chứa các ảnh được sắp xếp theo lưới

    Ngoại lệ:
    - TypeError: grid không phải tuple số nguyên
    - ValueError: grid không hợp lệ hoặc danh sách ảnh rỗng

    Nguồn: TrinhNhuNhat_13072026.
    """
    if not isinstance(grid, tuple):
        raise TypeError("grid must be a tuple")
    if len(grid) not in (1, 2):
        raise ValueError("grid must contain one or two values")
    if not all(isinstance(value, int) for value in grid):
        raise TypeError("grid values must be integers")
    if any(value <= 0 for value in grid):
        raise ValueError("grid values must be greater than zero")
    if not images:
        raise ValueError("No crop images found to create a grid")

    columns = grid[0]
    rows = grid[1] if len(grid) == 2 else ceil(len(images) / columns)
    selected = images[: columns * rows]
    cell_width = max(image.width for image in selected)
    cell_height = max(image.height for image in selected)
    canvas = Image.new(
        "RGB",
        (columns * cell_width, rows * cell_height),
        color=(255, 255, 255),
    )

    for index, image in enumerate(selected):
        column = index % columns
        row = index // columns
        x = column * cell_width + (cell_width - image.width) // 2
        y = row * cell_height + (cell_height - image.height) // 2
        canvas.paste(image.convert("RGB"), (x, y))

    return canvas

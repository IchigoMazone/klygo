import sys
import pathlib
from PIL import Image
import numpy as np

# Thêm root dự án vào sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

import klygo
from klygo.visual import (
    draw_bboxes,
    show_image,
    plot_dataset_stats,
    draw,
    show,
    plot_stats,
)


def test_visual_and_draw():
    # 1. Tạo ảnh giả lập
    img = Image.fromarray(np.ones((480, 640, 3), dtype=np.uint8) * 255)
    boxes = [[100, 100, 200, 200], [50, 50, 150, 150]]
    labels = ["apple", "orange"]
    scores = [0.95, 0.88]

    # 2. Test draw_bboxes & alias draw
    annotated1 = draw_bboxes(img, boxes, labels, scores)
    assert isinstance(annotated1, Image.Image)
    assert annotated1.size == (640, 480)

    annotated2 = draw(img, boxes, labels, scores)
    assert isinstance(annotated2, Image.Image)

    # 3. Test plot_dataset_stats
    stats_data = {"apple": 150, "orange": 80, "banana": 45}
    fig = plot_dataset_stats(stats_data, title="Fruit Distribution", show=False)
    assert fig is not None

    fig2 = plot_stats(stats_data, title="Stats", show=False)
    assert fig2 is not None

    # 4. Test visualize alias
    import klygo.visualize as vz
    assert vz.draw_bboxes is draw_bboxes
    assert vz.draw is draw

    print("ALL VISUAL TESTS PASSED!")


if __name__ == "__main__":
    test_visual_and_draw()

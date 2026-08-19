import os
from typing import Any, List, Optional, Union
import PIL.Image

from klygo import media, files
from klygo.utils.progress import ProgressBar


def export(
    model: Any,
    output_path: str,
    format: str,
    source: Union[str, List[Any]],
    text_prompt: Optional[List[str]] = None,
    classes: Optional[List[str]] = None,
    batch_size: int = 16,
    threshold: float = 0.4,
    verbose: bool = True,
    **kwargs,
) -> str:
    """
    Tác dụng:
    - Tự động tạo cấu trúc thư mục bộ dữ liệu huấn luyện YOLO hoặc Classification từ kết quả nhận diện của mô hình.
    - Đọc toàn bộ ảnh nguồn thông qua klygo.media.load và xử lý tuần tự từng ảnh với thanh tiến trình.

    Định dạng hỗ trợ:
    - 'yolo': Sử dụng model.predict() cho từng ảnh để sinh nhãn bounding box chuẩn hóa .txt kèm file cấu hình data.yaml.
    - 'classification': Sử dụng model.crop() cho từng ảnh để cắt ảnh đối tượng và tự động phân loại vào từng thư mục nhãn con.

    Đầu vào:
    - model [Any]: Đối tượng mô hình nhận diện (kế thừa DetectorModel).
    - output_path [str]: Đường dẫn thư mục đích lưu trữ bộ dữ liệu.
    - format [str]: Định dạng xuất ('yolo' hoặc 'classification').
    - source [str | List]: Đường dẫn thư mục/file ảnh/video, hoặc danh sách đường dẫn, hoặc danh sách ảnh PIL/NumPy đã đọc sẵn từ media.load.
    - text_prompt [List[str]]: Danh sách các tên nhãn lớp cần tìm kiếm/cắt.
    - classes [List[str]]: Tên tham số tương thích ngược (fallback của text_prompt).
    - batch_size [int]: Kích thước lô (được giữ cho tương thích signature). Mặc định: 16.
    - threshold [float]: Ngưỡng lọc độ tin cậy nhận diện. Mặc định: 0.4.
    - verbose [bool]: Hiển thị thanh tiến trình ProgressBar trong quá trình xuất. Mặc định: True.

    Đầu ra:
    - [str]: Đường dẫn tuyệt đối của thư mục dataset đã xuất.
    """
    target_prompt = text_prompt or classes
    if not source:
        raise ValueError("Tham số 'source' chứa đường dẫn ảnh nguồn là bắt buộc.")

    if not target_prompt or not isinstance(target_prompt, list):
        raise ValueError("Tham số 'text_prompt' phải là danh sách chuỗi ký tự (list[str]).")

    # =====================================================================
    # Nạp ảnh đầu vào: hỗ trợ str (đường dẫn thư mục/file/video) hoặc
    # list (danh sách đường dẫn hoặc danh sách ảnh PIL/NumPy đã đọc sẵn)
    # =====================================================================
    images = _resolve_source(source)

    if not images:
        return output_path

    label_to_id = {label: idx for idx, label in enumerate(target_prompt)}
    format_type = format.lower()
    total_images = len(images)
    files.mkdir(output_path)

    # =====================================================================
    # 1. Định dạng CLASSIFICATION: Dùng model.crop() cho từng ảnh
    # =====================================================================
    if format_type == "classification":
        with ProgressBar(
            total=total_images,
            desc="Exporting Classification Dataset",
            unit="img",
            verbose=verbose,
        ) as pbar:
            for img_idx, img in enumerate(images, 1):
                crop_res = model.crop(img, text_prompt=target_prompt, threshold=threshold)

                for obj_idx, crop_item in enumerate(crop_res, 1):
                    if crop_item.label in label_to_id:
                        class_dir = os.path.join(output_path, crop_item.label)
                        files.mkdir(class_dir)
                        save_path = os.path.join(
                            class_dir,
                            f"crop_{img_idx}_{obj_idx}_{crop_item.score:.2f}.jpg",
                        )
                        crop_item.save(save_path)

                pbar.update(1)

    # =====================================================================
    # 2. Định dạng YOLO: Dùng model.predict() cho từng ảnh
    # =====================================================================
    elif format_type == "yolo":
        yaml_content = f"path: {os.path.abspath(output_path)}\ntrain: images\nval: images\n\nnc: {len(target_prompt)}\nnames:\n"
        for idx, label in enumerate(target_prompt):
            yaml_content += f"  {idx}: {label}\n"
        files.save(
            os.path.join(output_path, "data.yaml"),
            yaml_content,
            overwrite=True,
            verbose=False,
        )

        img_dir = os.path.join(output_path, "images")
        lbl_dir = os.path.join(output_path, "labels")
        files.mkdir(img_dir)
        files.mkdir(lbl_dir)

        with ProgressBar(
            total=total_images,
            desc="Exporting YOLO Dataset",
            unit="img",
            verbose=verbose,
        ) as pbar:
            for img_idx, img in enumerate(images, 1):
                img_name = f"img_{img_idx:05d}"
                img_file = os.path.join(img_dir, f"{img_name}.jpg")
                media.save(img_file, img, overwrite=True, verbose=False)

                results = model.predict(img, text_prompt=target_prompt, threshold=threshold)

                lbl_lines = []
                w_img, h_img = img.size
                for obj in results:
                    class_id = label_to_id.get(obj.label)
                    if class_id is None:
                        continue
                    x_center = ((obj.xmin + obj.xmax) / 2) / w_img
                    y_center = ((obj.ymin + obj.ymax) / 2) / h_img
                    w_box = (obj.xmax - obj.xmin) / w_img
                    h_box = (obj.ymax - obj.ymin) / h_img

                    lbl_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}"
                    )

                lbl_file = os.path.join(lbl_dir, f"{img_name}.txt")
                files.save(
                    lbl_file,
                    "\n".join(lbl_lines),
                    overwrite=True,
                    verbose=False,
                )

                pbar.update(1)
    else:
        raise ValueError(
            f"Định dạng '{format}' không được hỗ trợ. Vui lòng chọn 'yolo' hoặc 'classification'."
        )

    return output_path


def _resolve_source(source: Union[str, List[Any]]) -> List[PIL.Image.Image]:
    """
    Tác dụng:
    - Chuẩn hóa mọi kiểu đầu vào source về danh sách PIL.Image.Image thống nhất.

    Đầu vào hỗ trợ:
    - str: Đường dẫn thư mục ảnh, file ảnh đơn lẻ, hoặc file video (.mp4, .avi, .mov...).
    - List[str]: Danh sách đường dẫn file ảnh/video.
    - List[PIL.Image]: Danh sách ảnh PIL đã đọc sẵn (VD: từ kết quả media.load).
    - List[np.ndarray]: Danh sách mảng NumPy (VD: từ OpenCV hoặc media.load với backend='opencv').

    Đầu ra:
    - [List[PIL.Image.Image]]: Danh sách ảnh PIL đã chuẩn hóa RGB.
    """
    # Trường hợp 1: source là chuỗi đường dẫn (thư mục, file ảnh, hoặc video)
    if isinstance(source, str):
        return media.load(source, verbose=False)

    # Trường hợp 2: source là danh sách (đường dẫn hoặc ảnh đã đọc sẵn)
    if isinstance(source, (list, tuple)):
        images = []
        for item in source:
            if isinstance(item, str):
                loaded = media.load(item, verbose=False)
                if loaded:
                    images.extend(loaded)
            elif isinstance(item, PIL.Image.Image):
                images.append(item.convert("RGB"))
            else:
                images.append(media.to_pil(item).convert("RGB"))
        return images

    raise TypeError(
        f"Tham số 'source' không hợp lệ (nhận được {type(source).__name__}). "
        f"Vui lòng truyền đường dẫn (str), danh sách đường dẫn (List[str]), "
        f"hoặc danh sách ảnh đã đọc sẵn (List[PIL.Image])."
    )

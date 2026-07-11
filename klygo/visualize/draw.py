import sys
from pathlib import Path
from typing import Union, List, Tuple, Dict, Any
import numpy as np
import cv2 as cv
from PIL import Image
from zipfile import ZipFile, ZIP_DEFLATED
import io

def show_image(
    image: Union[Image.Image, np.ndarray], 
    title: str = "Image", 
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (10, 10)
) -> None:
    """Hiển thị hình ảnh bằng OpenCV hoặc Matplotlib (Colab)."""
    is_pil = isinstance(image, Image.Image)
    
    if backend.lower() == "matplotlib":
        import matplotlib.pyplot as plt
        plt.figure(figsize=figsize)
        if is_pil:
            plt.imshow(image)
        else:
            # OpenCV lưu dạng BGR, cần chuyển sang RGB để plt hiển thị đúng
            plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        # Dùng OpenCV
        if is_pil:
            img_cv = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        else:
            img_cv = image
        cv.imshow(title, img_cv)
        cv.waitKey(0)
        cv.destroyAllWindows()

def draw_bboxes(
    image: Union[Image.Image, np.ndarray],
    bboxes: List[List[float] | Tuple[float, float, float, float]],
    labels: List[str] | None = None,
    scores: List[float] | None = None,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> Union[Image.Image, np.ndarray]:
    """Vẽ bounding boxes và labels lên ảnh PIL hoặc Numpy array."""
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        img_np = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    else:
        img_np = image.copy()

    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        cv.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
        
        text_parts = []
        if labels and idx < len(labels):
            text_parts.append(str(labels[idx]))
        if scores and idx < len(scores):
            text_parts.append(f"{scores[idx]:.2f}")
            
        if text_parts:
            text = " ".join(text_parts)
            cv.putText(
                img_np,
                text,
                (x1, y1 - 5 if y1 - 5 > 15 else y1 + 18),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )

    if is_pil:
        return Image.fromarray(cv.cvtColor(img_np, cv.COLOR_BGR2RGB))
    return img_np

def visualize_prediction(
    image: Union[Image.Image, np.ndarray],
    prediction: Dict[str, Any],
    backend: str = "matplotlib",
    color: Tuple[int, int, int] = (255, 0, 0),
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """Hiển thị kết quả predict từ model."""
    boxes = prediction.get("boxes", [])
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    scores = prediction.get("scores", [])
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    labels = prediction.get("labels", [])
    
    annotated = draw_bboxes(image, boxes, labels, scores, color=color)
    show_image(annotated, title="Prediction Result", backend=backend, figsize=figsize)

def visualize_dataset_image(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    classes: List[str],
    backend: str = "matplotlib",
    figsize: Tuple[int, int] = (10, 10),
) -> None:
    """Hiển thị một ảnh cụ thể và vẽ bboxes từ file nhãn YOLO tương ứng."""
    image_path = Path(image_path)
    label_path = Path(label_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img = Image.open(image_path)
    width, height = img.size
    
    bboxes = []
    labels = []
    
    if label_path.exists() and label_path.is_file():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_w = float(parts[3]) * width
                    box_h = float(parts[4]) * height
                    
                    x1 = x_center - box_w / 2
                    y1 = y_center - box_h / 2
                    x2 = x_center + box_w / 2
                    y2 = y_center + box_h / 2
                    
                    bboxes.append([x1, y1, x2, y2])
                    class_name = classes[class_id] if class_id < len(classes) else str(class_id)
                    labels.append(class_name)
                    
    annotated = draw_bboxes(img, bboxes, labels)
    show_image(annotated, title=f"Dataset Sample: {image_path.name}", backend=backend, figsize=figsize)

def crop_objects(
    image_path: Union[str, Path],
    label_path: Union[str, Path],
    classes: List[str],
    output_zip: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """Cắt các vật thể trong ảnh dựa trên nhãn YOLO và lưu vào file ZIP."""
    image_path = Path(image_path)
    label_path = Path(label_path)
    output_zip = Path(output_zip)
    
    if output_zip.exists() and not overwrite:
        raise FileExistsError(f"ZIP file already exists: {output_zip}")
        
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img = Image.open(image_path)
    width, height = img.size
    
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    crops: Dict[str, List[Image.Image]] = {}
    
    if label_path.exists() and label_path.is_file():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_w = float(parts[3]) * width
                    box_h = float(parts[4]) * height
                    
                    x1 = max(0, int(x_center - box_w / 2))
                    y1 = max(0, int(y_center - box_h / 2))
                    x2 = min(width, int(x_center + box_w / 2))
                    y2 = min(height, int(y_center + box_h / 2))
                    
                    if x2 > x1 and y2 > y1:
                        cropped = img.crop((x1, y1, x2, y2))
                        class_name = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                        if class_name not in crops:
                            crops[class_name] = []
                        crops[class_name].append(cropped)
                        
    with ZipFile(output_zip, mode="w", compression=ZIP_DEFLATED) as zf:
        for class_name, img_list in crops.items():
            for idx, cropped_img in enumerate(img_list):
                img_byte_arr = io.BytesIO()
                cropped_img.save(img_byte_arr, format="JPEG")
                img_bytes = img_byte_arr.getvalue()
                arcname = f"{class_name}_crop_{idx}.jpg"
                zf.writestr(arcname, img_bytes)

def read_cropped_objects(zip_path: Union[str, Path]) -> Dict[str, List[Image.Image]]:
    """Đọc file ZIP chứa các vật thể đã crop và trả về dictionary {class_name: [PIL.Image, ...]}."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
    result: Dict[str, List[Image.Image]] = {}
    
    with ZipFile(zip_path, mode="r") as zf:
        for name in zf.namelist():
            # Tên file có dạng: classname_crop_index.jpg
            if name.endswith((".jpg", ".jpeg", ".png")):
                parts = name.split("_crop_")
                if len(parts) == 2:
                    class_name = parts[0]
                    img_data = zf.read(name)
                    img = Image.open(io.BytesIO(img_data))
                    # load thực tế để đóng zipfile an toàn
                    img.load()
                    if class_name not in result:
                        result[class_name] = []
                    result[class_name].append(img)
    return result

def plot_dataset_stats(
    source: Union[str, Path],
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Vẽ biểu đồ hình cột thống kê phân bổ class trong dataset."""
    from klygo.datasets import get_dataset_info
    info = get_dataset_info(source)
    classes = info.get("classes", [])
    
    # Ở đây chúng ta sẽ quét thủ công các file labels để đếm chính xác số lượng đối tượng từng class
    # Vì get_dataset_info chỉ quét thống kê số file ảnh/nhãn chứ không đếm từng object cụ thể
    from klygo.datasets._utils import _find_dataset_root, _scan_dataset_files
    import shutil
    import tempfile
    
    source = Path(source)
    is_zip = source.is_file()
    temp_dir = None
    
    if is_zip:
        temp_dir = Path(tempfile.mkdtemp())
        from klygo.archive import extract
        extract(source, temp_dir, overwrite=True, verbose=False)
        src_base = _find_dataset_root(temp_dir)
    else:
        src_base = source
        
    images_src = src_base / "images"
    labels_src = src_base / "labels"
    
    class_counts = {i: 0 for i in range(len(classes))}
    
    if images_src.exists() and labels_src.exists():
        pairs = _scan_dataset_files(images_src, labels_src)
        for _, lbl_path, _, _ in pairs:
            if lbl_path and lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                c_id = int(parts[0])
                                if c_id in class_counts:
                                    class_counts[c_id] += 1
                            except ValueError:
                                pass
                                
    if temp_dir and temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    # Vẽ biểu đồ bằng matplotlib
    import matplotlib.pyplot as plt
    class_names = [classes[i] if i < len(classes) else str(i) for i in class_counts.keys()]
    counts = list(class_counts.values())
    
    plt.figure(figsize=figsize)
    plt.bar(class_names, counts, color="royalblue")
    plt.xlabel("Classes")
    plt.ylabel("Object Count")
    plt.title("Class Distribution in Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

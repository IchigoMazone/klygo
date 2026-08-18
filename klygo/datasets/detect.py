import os
from typing import Any
from klygo import media
from klygo import files
from klygo.utils.progress import ProgressBar

def export(model: Any, output_path: str, format: str, source: str, classes: list = None, **kwargs):
    """Generates YOLO or cropped Classification dataset folder structure from predictions using a detection model."""
    if not source:
        raise ValueError("Parameter 'source' containing input images folder is required.")
        
    images = media.load(source)
    
    # 1. Resolve classes mapping
    if classes is None:
        if hasattr(model.backend, "native") and hasattr(model.backend.native, "names"):
            native_names = model.backend.native.names
            classes = [native_names[i] for i in sorted(native_names.keys())]
        else:
            raise ValueError("Class list mapping 'classes' must be provided.")
            
    label_to_id = {label: idx for idx, label in enumerate(classes)}

    verbose = kwargs.get("verbose", True)

    # 2. Classification output folder format
    if format == "classification":
        with ProgressBar(total=len(images), desc="Exporting Classification Dataset", verbose=verbose) as pbar:
            for img_idx, img in enumerate(images):
                results = model.predict(img, **kwargs)
                for obj_idx, obj in enumerate(results.objects):
                    if obj.label in label_to_id:
                        cropped = img.crop((obj.xmin, obj.ymin, obj.xmax, obj.ymax))
                        class_dir = os.path.join(output_path, obj.label)
                        files.mkdir(class_dir)
                        media.save(os.path.join(class_dir, f"crop_{img_idx}_{obj_idx}.jpg"), cropped)
                pbar.update(1)
                    
    # 3. YOLO detection output folder format
    elif format == "yolo":
        yaml_content = f"path: {os.path.abspath(output_path)}\ntrain: images\nval: images\n\nnames:\n"
        for idx, label in enumerate(classes):
            yaml_content += f"  {idx}: {label}\n"
        files.save(os.path.join(output_path, "dataset.yaml"), yaml_content, overwrite=True)

        img_dir = os.path.join(output_path, "images")
        lbl_dir = os.path.join(output_path, "labels")
        files.mkdir(img_dir)
        files.mkdir(lbl_dir)

        with ProgressBar(total=len(images), desc="Exporting YOLO Dataset", verbose=verbose) as pbar:
            for img_idx, img in enumerate(images):
                results = model.predict(img, **kwargs)
                media.save(os.path.join(img_dir, f"img_{img_idx}.jpg"), img)
                
                lbl_content = ""
                w_img, h_img = img.size
                for obj in results.objects:
                    class_id = label_to_id.get(obj.label)
                    if class_id is None:
                        continue
                    x_center = ((obj.xmin + obj.xmax) / 2) / w_img
                    y_center = ((obj.ymin + obj.ymax) / 2) / h_img
                    w_box = (obj.xmax - obj.xmin) / w_img
                    h_box = (obj.ymax - obj.ymin) / h_img
                    lbl_content += f"{class_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n"
                
                files.save(os.path.join(lbl_dir, f"img_{img_idx}.txt"), lbl_content, overwrite=True)
                pbar.update(1)
    else:
        raise ValueError(f"Unsupported export format '{format}'. Use 'yolo' or 'classification'.")

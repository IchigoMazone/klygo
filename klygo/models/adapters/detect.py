import os
from typing import Any, List
from .base import BaseAdapter
from ..registry import registry
from klygo import media
from klygo import files
from klygo.utils.progress import ProgressBar

class BoundingBox:
    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float, label: str, score: float):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.label = label
        self.score = score

    def __repr__(self):
        return f"[{self.label}: {self.score:.2f} -> ({int(self.xmin)}, {int(self.ymin)}, {int(self.xmax)}, {int(self.ymax)})]"

class DetectionResult:
    def __init__(self, objects: List[BoundingBox]):
        self.objects = objects

    def __repr__(self):
        return f"<DetectionResult: found {len(self.objects)} objects>"

@registry.register_adapter("detect")
class DetectAdapter(BaseAdapter):
    def _predict(self, source: Any, **kwargs) -> Any:
        # Load default parameters from history config
        loaded_params = self._history["operations"].get("predict", {})
        run_params = {**loaded_params, **kwargs}
        
        # Capture parameters in history
        self._history["operations"]["predict"].update(kwargs)
        
        is_batch = isinstance(source, list) or (isinstance(source, str) and os.path.isdir(source))
        
        images = media.load(source)
        raw_outputs = self.backend.predict(images, **run_params)
        results = self._post_process(raw_outputs, images)
        return results if is_batch else results[0]

    def crop(self, source: Any, **kwargs) -> List[Any]:
        """Runs object detection and crops each detected object into a separate PIL Image."""
        images = media.load(source)
        results = self.predict(images, **kwargs)
        
        if not isinstance(results, list):
            results = [results]
            
        cropped_results = []
        for img, res in zip(images, results):
            for obj in res.objects:
                # Clip box coordinates to image size boundaries
                xmin = max(0, int(obj.xmin))
                ymin = max(0, int(obj.ymin))
                xmax = min(img.width, int(obj.xmax))
                ymax = min(img.height, int(obj.ymax))
                
                cropped_img = img.crop((xmin, ymin, xmax, ymax))
                cropped_results.append(cropped_img)
                
        return cropped_results

    def _post_process(self, raw_outputs: Any, images: list) -> List[DetectionResult]:
        if raw_outputs is None:
            return [DetectionResult([]) for _ in images]

        batch_results = []

        # Case A: Hugging Face zero-shot / Grounding DINO outputs
        if isinstance(raw_outputs, list) and isinstance(raw_outputs[0], dict) and "boxes" in raw_outputs[0]:
            for output in raw_outputs:
                objects = []
                boxes = output["boxes"].tolist()
                scores = output["scores"].tolist()
                labels = output["labels"]
                for box, score, label in zip(boxes, scores, labels):
                    objects.append(BoundingBox(
                        xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3],
                        label=label, score=score
                    ))
                batch_results.append(DetectionResult(objects))

        # Case B: Ultralytics YOLO outputs
        elif isinstance(raw_outputs, list) and hasattr(raw_outputs[0], "boxes"):
            for result in raw_outputs:
                objects = []
                names = result.names
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = names[class_id]
                    objects.append(BoundingBox(
                        xmin=coords[0], ymin=coords[1], xmax=coords[2], ymax=coords[3],
                        label=label, score=score
                    ))
                batch_results.append(DetectionResult(objects))
        else:
            batch_results = [DetectionResult([]) for _ in images]

        return batch_results

    def _export_dataset(self, output_path: str, format: str, classes: list = None, **kwargs):
        """Generates YOLO or cropped Classification dataset folder structure from predictions."""
        source = kwargs.get("source")
        if not source:
            raise ValueError("Parameter 'source' containing input images folder is required.")
            
        images = media.load(source)
        
        # 1. Resolve classes mapping
        if classes is None:
            if hasattr(self.backend, "native") and hasattr(self.backend.native, "names"):
                native_names = self.backend.native.names
                classes = [native_names[i] for i in sorted(native_names.keys())]
            else:
                raise ValueError("Class list mapping 'classes' must be provided.")
                
        label_to_id = {label: idx for idx, label in enumerate(classes)}

        predict_kwargs = {k: v for k, v in kwargs.items() if k != "source"}

        verbose = kwargs.get("verbose", True)

        # 2. Classification output folder format
        if format == "classification":
            with ProgressBar(total=len(images), desc="Exporting Classification Dataset", verbose=verbose) as pbar:
                for img_idx, img in enumerate(images):
                    results = self.predict(img, **predict_kwargs)
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
                    results = self.predict(img, **predict_kwargs)
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

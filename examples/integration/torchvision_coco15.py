"""Evaluate four COCO15 classes with a caller-constructed Torchvision detector."""

import os
from torchvision.models import get_model, get_model_weights

from klygo import (
    archive, config, datasets, files, media, outputs,
    postprocessing, preprocessing, utils, visual,
)
from klygo.models.detection import Detector


CLASSES = ("person", "car", "elephant", "umbrella")


class TorchvisionCOCODetect(Detector):
    """Application-owned Detector using Klygo's Torchvision backend hooks."""

    def __init__(self, model, categories):
        metadata = {
            "model_id": "application-owned-torchvision-model",
            "details": {"name": "Application detector", "task": "Object-Detection",
                        "backend": "Torchvision", "library": "torchvision"},
            "flags": {"model": False},
            "model": {},
            "priority": {},
        }
        super().__init__(metadata=metadata, backend="Torchvision")
        self.model = model.eval()
        self.categories = {index: name for index, name in enumerate(categories)}
        self.names = dict(self.categories)

    def forward(self, images, prompt=None):
        del prompt
        tensors = [media.to_tensor(image) for image in images]
        raw = self.format_results(self.run_inference(self.cast_inputs(tensors)))
        for item in raw:
            item["labels"] = [self.categories.get(index, str(index))
                              for index in item["class_ids"]]
        return outputs.build_detections(images, raw, names=self.names)


def evaluate(dataset_root=None, artifact_root=None):
    """Run one positive and one negative real image per assigned class."""
    dataset_root = files.path(dataset_root or os.getenv(
        "KLYGO_COCO15_DATASET", "artifacts/integration/coco15"
    ))
    artifact_root = files.path(artifact_root or os.getenv(
        "KLYGO_ARTIFACT_ROOT", "artifacts/integration/torchvision-coco15"
    ))
    if not files.is_file(files.join(dataset_root, "data.yaml")):
        raise FileNotFoundError("Build the COCO15 dataset before running this example")
    report_root = files.join(artifact_root, "report")
    preview_root = files.join(artifact_root, "preview")
    cache_root = files.join(artifact_root, "cache", "torch")
    for folder in (report_root, preview_root, cache_root):
        files.mkdir(folder, parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(cache_root))

    metadata = config.load(files.join(dataset_root, "data.yaml"), verbose=False)
    source_manifest = files.load(files.join(dataset_root, "manifest.json"), verbose=False)
    if not set(CLASSES).issubset(set(metadata.names)):
        raise ValueError("COCO15 class map does not include the assigned classes")
    truth = datasets.load(dataset_root, format="yolo", verbose=False)
    by_id = {files.stem(frame.metadata["relative_path"]): frame for frame in truth}
    architecture = "ssdlite320_mobilenet_v3_large"
    weights = get_model_weights(architecture).COCO_V1
    module = get_model(architecture, weights=weights, weights_backbone=None)
    model = TorchvisionCOCODetect(module, weights.meta["categories"])
    device = str(model.current_device())
    rows = []
    try:
        for label in CLASSES:
            positive = next(key for key, frame in sorted(by_id.items())
                            if any(box.label == label for box in frame))
            negative = next(key for key, frame in sorted(by_id.items())
                            if all(box.label != label for box in frame))
            for image_id in (positive, negative):
                image = media.load(files.join(dataset_root, "images", f"{image_id}.jpg"), verbose=False)[0]
                prepared = preprocessing.auto_orient()(image)
                predicted = model.predict(prepared, prompt=label, verbose=False)[0]
                if not isinstance(predicted, outputs.detect.Detection):
                    raise TypeError("Torchvision did not return a Klygo Detection")
                predicted.transform(
                    postprocessing.threshold(0.25), postprocessing.nms(0.5)
                )
                predicted.transform(postprocessing.select(labels=[label]))
                actual = [box for box in by_id[image_id] if box.label == label]
                unmatched = set(range(len(actual)))
                tp = fp = 0
                for box in sorted(predicted, key=lambda item: item.score, reverse=True):
                    overlap, index = max(
                        ((postprocessing.box_iou(box, actual[i]), i) for i in unmatched),
                        default=(0.0, -1),
                    )
                    if overlap >= 0.5:
                        unmatched.remove(index)
                        tp += 1
                    else:
                        fp += 1
                rows.append({"class": label, "image_id": image_id,
                             "tp": tp, "fp": fp, "fn": len(unmatched)})
                if image_id == positive:
                    overlay = visual.draw_bboxes(
                        image.to_pil(), predicted.boxes, predicted.labels, predicted.scores
                    )
                    media.save(files.join(preview_root, f"{label}.jpg"), overlay, overwrite=True)
    finally:
        model.unload()

    totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
    report = {"backend": "Torchvision", "model": "torchvision-ssdlite320-mobilenet-v3-large",
              "weights": "COCO_V1", "device": device,
              "source_sha256": source_manifest["source_sha256"],
              "classes": list(CLASSES), "images": len(rows), "confidence": 0.25,
              "iou": 0.5, "totals": totals, "per_image": rows}
    files.save(files.join(report_root, "metrics.json"), report, overwrite=True, verbose=False)
    files.save(files.join(artifact_root, "manifest.json"), {
        "dataset": str(files.resolve(dataset_root)),
        "source_sha256": source_manifest["source_sha256"],
        "backend": report["backend"], "model": report["model"],
        "weights": report["weights"], "device": report["device"],
        "classes": report["classes"], "images": report["images"],
        "report": "report/metrics.json", "previews": "preview/",
    }, overwrite=True, verbose=False)
    bundle = files.join(artifact_root, "report.zip")
    archive.compress(report_root, bundle, overwrite=True, verbose=False)
    print(f"Evaluated {len(rows)} class-image pairs; report: {bundle} ({utils.human_size(files.size(bundle))})")
    return report


if __name__ == "__main__":
    evaluate()

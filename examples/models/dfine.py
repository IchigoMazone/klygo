"""Run the registered KerasHub D-FINE model on an image."""
import argparse
import sys
# Prevent local metadata.py from shadowing Python modules during imports.
sys.path.pop(0)
from klygo import files, models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Image or video source accepted by Detector.predict")
    parser.add_argument("--device", default=None, help="Torch engine device, for example cuda:0 or cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", default="artifacts/dfine/detections.json")
    args = parser.parse_args()
    models.set_backend("Keras", engine="torch")
    detector = models.load("dfine_nano_coco", device=args.device, threshold=args.threshold)
    results = detector.predict(args.source, verbose=False)
    files.save(args.output, results.to_dict(), overwrite=True, verbose=False)
    print(detector.model_id, detector.device, len(results), results.total_objects)


if __name__ == "__main__":
    main()

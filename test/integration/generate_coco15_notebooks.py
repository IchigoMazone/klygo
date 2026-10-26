"""Generate one Colab-ready, class-specific Klygo notebook for each COCO15 class."""

from __future__ import annotations

from textwrap import dedent

import nbformat
from klygo import files


HERE = files.parent(files.resolve(__file__))
MANIFEST = files.join(HERE, "generated", "coco15", "manifest.json")
OUTPUT = files.join(files.parent(files.parent(HERE)), "tutorials", "integration")


def code(source: str):
    return nbformat.v4.new_code_cell(dedent(source).strip() + "\n")


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip() + "\n")


def notebook(class_id: int, class_name: str, source_id: int):
    slug = class_name.replace(" ", "_")
    cells = [
        markdown(f"""
        # COCO15 · {class_name}

        This notebook evaluates **{class_name}** (COCO15 class `{class_id}`, original COCO class `{source_id}`) on real COCO128 photographs. It exercises Klygo's `files`, `config`, `archive`, `datasets`, `media`, `preprocessing`, `models`, `outputs`, `postprocessing`, `visual`, `validators`, and `utils` in one workflow.

        Sign in to Google Colab, choose **Runtime → Change runtime type → GPU**, and upload `klygo_coco15_bundle.zip` once to `MyDrive/Klygo/`. The notebook asks you to authorize Drive access. A GPU speeds up inference; it falls back to CPU if none is assigned. The bundle contains this checkout's Klygo source and the 15-class dataset, so its behavior does not depend on unpushed Git changes.

        For a quick run, this notebook uses up to four positive and two negative images. Set `KLYGO_FULL_DATASET=1` before the sampling cell to evaluate all 128 images. COCO128 is a pipeline test subset, not an independent model benchmark.
        """),
        code("""
        import os, sys, tempfile, zipfile, subprocess
        from pathlib import Path

        try:
            import google.colab  # noqa: F401
            IN_COLAB = True
        except ImportError:
            IN_COLAB = False
        if IN_COLAB:
            from google.colab import drive
            drive.mount("/content/drive")
            default_bundle = Path("/content/drive/MyDrive/Klygo/klygo_coco15_bundle.zip")
            WORK_ROOT = Path("/content/klygo_coco15")
            os.environ.setdefault("HF_HOME", "/content/drive/MyDrive/Klygo/hf_cache")
        else:
            default_bundle = Path("artifacts/integration/bundles/klygo_coco15_bundle.zip")
            WORK_ROOT = Path(tempfile.gettempdir()) / "klygo_coco15_notebook"
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        BUNDLE = Path(os.getenv("KLYGO_COCO15_BUNDLE", str(default_bundle)))
        if not BUNDLE.is_file():
            raise FileNotFoundError(f"Upload the bundle to Drive or set KLYGO_COCO15_BUNDLE: {BUNDLE}")
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BUNDLE) as bundle:
            for name in bundle.namelist():
                parts = Path(name).parts
                if Path(name).is_absolute() or ".." in parts:
                    raise ValueError(f"Unsafe bundle path: {name}")
            bundle.extractall(WORK_ROOT)
        PROJECT_ROOT = WORK_ROOT / "project"
        DATASET_ROOT = WORK_ROOT / "dataset"
        if IN_COLAB:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(PROJECT_ROOT), "transformers>=4.50,<6"])
        print("Bundle:", BUNDLE)
        print("Dataset:", DATASET_ROOT)
        """),
        code(f"""
        import json
        import torch
        from klygo import archive, config, datasets, files, media, models, outputs
        from klygo import postprocessing, preprocessing, utils, validators, visual

        WORK_ROOT = files.path(WORK_ROOT)
        DATASET_ROOT = files.path(DATASET_ROOT)

        TARGET_CLASS = {class_name!r}
        TARGET_ID = {class_id}
        validators.validate_type(TARGET_CLASS, str, "TARGET_CLASS")
        manifest = files.load(files.join(DATASET_ROOT, "manifest.json"), verbose=False)
        metadata = config.load(files.join(DATASET_ROOT, "data.yaml"), verbose=False)
        dataset_info = datasets.get_dataset_info(DATASET_ROOT)
        assert len(metadata.names) == 15 and metadata.names[TARGET_ID] == TARGET_CLASS
        assert dataset_info["total_images"] == 128
        print("Class:", TARGET_CLASS)
        print("Labels:", manifest["classes"][TARGET_ID]["boxes"])
        print("Images containing class:", manifest["classes"][TARGET_ID]["positive_images"])
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not assigned; using CPU")
        """),
        code("""
        ground_truth = datasets.load(DATASET_ROOT, format="yolo", verbose=False)
        truth_by_id = {files.stem(frame.metadata["relative_path"]): frame for frame in ground_truth}
        positive_ids = sorted(image_id for image_id, frame in truth_by_id.items()
                              if any(box.label == TARGET_CLASS for box in frame))
        negative_ids = sorted(image_id for image_id, frame in truth_by_id.items()
                              if all(box.label != TARGET_CLASS for box in frame))
        full_dataset = os.getenv("KLYGO_FULL_DATASET") == "1"
        max_positives = int(os.getenv("KLYGO_MAX_POSITIVES", "4"))
        max_negatives = int(os.getenv("KLYGO_MAX_NEGATIVES", "2"))
        selected_ids = sorted(truth_by_id) if full_dataset else positive_ids[:max_positives] + negative_ids[:max_negatives]
        assert selected_ids and positive_ids
        print("Evaluation images:", len(selected_ids), "positive:", sum(i in positive_ids for i in selected_ids))
        """),
        code("""
        model = models.load("grounding-dino-tiny", size={"shortest_edge": 256, "longest_edge": 384})
        if torch.cuda.is_available():
            model.model.to("cuda")
        print("Registered model:", model.model_id)
        print("Execution device:", next(model.model.parameters()).device)
        """),
        code("""
        def score_image(truth, prediction):
            actual = [box for box in truth if box.label == TARGET_CLASS]
            remaining = set(range(len(actual)))
            tp = fp = 0
            for box in sorted(prediction, key=lambda item: item.score, reverse=True):
                if box.score < 0.25:
                    continue
                candidates = [(postprocessing.box_iou(box, actual[index]), index)
                              for index in remaining]
                overlap, index = max(candidates, default=(0.0, -1))
                if overlap >= 0.5:
                    tp += 1
                    remaining.remove(index)
                else:
                    fp += 1
            return tp, fp, len(remaining)

        predictions, rows = {}, []
        for image_id in selected_ids:
            image_path = files.join(DATASET_ROOT, "images", f"{image_id}.jpg")
            lazy_image = media.load(image_path, verbose=False)[0]
            prepared = preprocessing.auto_orient()(lazy_image)
            result = model.predict(prepared, prompt=[TARGET_CLASS], verbose=False)[0]
            assert isinstance(result, outputs.detect.Detection)
            result.transform(postprocessing.threshold(0.25), postprocessing.nms(0.5))
            result.transform(postprocessing.select(labels=[TARGET_CLASS]))
            tp, fp, fn = score_image(truth_by_id[image_id], result)
            predictions[image_id] = result
            rows.append({"image_id": image_id, "tp": tp, "fp": fp, "fn": fn,
                         "ground_truth": tp + fn, "predictions": tp + fp})

        totals = {key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")}
        totals["precision"] = totals["tp"] / max(1, totals["tp"] + totals["fp"])
        totals["recall"] = totals["tp"] / max(1, totals["tp"] + totals["fn"])
        print(json.dumps(totals, indent=2))
        """),
        code("""
        RESULT_ROOT = files.join(
            "/content/drive/MyDrive/Klygo/results" if IN_COLAB
            else files.join(WORK_ROOT, "results"),
            TARGET_CLASS.replace(" ", "_"),
        )
        files.mkdir(RESULT_ROOT, parents=True, exist_ok=True)
        preview_id = positive_ids[0]
        preview = predictions.get(preview_id)
        if preview is not None:
            original = media.load(files.join(DATASET_ROOT, "images", f"{preview_id}.jpg"), verbose=False)[0].to_pil()
            overlay = visual.draw_bboxes(original, preview.boxes, preview.labels, preview.scores)
            media.save(files.join(RESULT_ROOT, f"{preview_id}_prediction.jpg"), overlay, overwrite=True)
            display(overlay)
        report = {"class_id": TARGET_ID, "class_name": TARGET_CLASS,
                  "dataset_source_sha256": manifest["source_sha256"],
                  "model_id": model.model_id, "device": str(next(model.model.parameters()).device),
                  "confidence_threshold": 0.25, "iou_threshold": 0.5,
                  "scope": "all 128 images" if full_dataset else f"up to {max_positives} positives and {max_negatives} negatives",
                  "per_image": rows, "totals": totals}
        files.save(files.join(RESULT_ROOT, "metrics.json"), report, overwrite=True, verbose=False)
        archive_path = files.with_extension(RESULT_ROOT, ".zip")
        archive.compress(RESULT_ROOT, archive_path, overwrite=True, verbose=False)
        print("Report:", files.join(RESULT_ROOT, "metrics.json"))
        print("Archive:", archive_path, utils.human_size(files.size(archive_path)))
        """),
        markdown(f"""
        ## Review

        Compare false positives and missed **{class_name}** boxes in `metrics.json` and the saved overlay. The 15 notebooks share the same dataset and thresholds; they differ only in target class. The default sample is a smoke evaluation. For complete per-class coverage, set `KLYGO_FULL_DATASET=1` and rerun from the sampling cell.

        Dataset: [Ultralytics COCO128](https://docs.ultralytics.com/datasets/detect/coco128). Grounding DINO weights are cached in your Drive after the first run. Drive authorization and GPU allocation are controlled by your Colab account.
        """),
    ]
    for index, cell in enumerate(cells):
        cell.id = f"c{class_id:02d}-{index:02d}"
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {
        "colab": {"name": f"Klygo_COCO15_{slug}.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    return nb


def generate(output=OUTPUT, manifest_path=MANIFEST) -> list:
    manifest = files.load(manifest_path, verbose=False)
    classes = manifest["classes"]
    if len(classes) != 15:
        raise ValueError("Expected exactly 15 COCO classes")
    output = files.path(output)
    files.mkdir(output, parents=True, exist_ok=True)
    paths = []
    for item in classes:
        slug = item["name"].replace(" ", "_")
        path = files.join(output, f"Klygo_COCO15_{item['id']:02d}_{slug}.ipynb")
        # nbformat owns the notebook JSON schema; files.save has no .ipynb writer.
        nbformat.write(notebook(item["id"], item["name"], item["source_coco_id"]), path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    for path in generate():
        print(path)

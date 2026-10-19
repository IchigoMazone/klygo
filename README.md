# Klygo

Python utilities for archives, filesystem operations, configuration, datasets, and object detection.

## Installation

```bash
git clone https://github.com/IchigoMazone/klygo.git
cd klygo
uv sync
```

Optional archive backends can be installed separately:

```bash
pip install "klygo[py7zr]"       # Read and create 7z archives
pip install "klygo[rarfile]"     # Read and extract RAR archives
uv add "klygo[py7zr]"
uv add "klygo[rarfile]"
uv pip install "klygo[py7zr]"
uv pip install "klygo[rarfile]"
```

Model frameworks are optional. Install only the backend you need:

```bash
pip install "klygo[transformers]"
pip install "klygo[ultralytics]"
pip install "klygo[keras-hub]"
pip install "klygo[models]"        # All supported model integrations
uv add "klygo[transformers]"
uv add "klygo[ultralytics]"
uv add "klygo[keras-hub]"
uv add "klygo[models]"
uv pip install "klygo[transformers]"
uv pip install "klygo[ultralytics]"
uv pip install "klygo[keras-hub]"
uv pip install "klygo[models]"
```

For a PyTorch runtime without Transformers, install `klygo[torch]`. KerasHub can use TensorFlow, JAX, or PyTorch through `KERAS_BACKEND`; for the PyTorch backend, install `klygo[keras-hub,torch]`.

## Packages

- `klygo.archive`: Create, extract, search, verify, convert, split, merge, and modify archives.
- `klygo.config`: Load and manipulate multi-format configuration with dot-path access.
- `klygo.datasets`: Inspect, partition, repartition, merge, split, remap, and export datasets.
- `klygo.files`: File, directory, and path operations, plus structured-data I/O.
- `klygo.media`: Load and save image and video data; use `load(..., stream=True)` for bounded-memory iteration.
- `klygo.models`: Load and run supported detection models on images, directories, and videos.
- `klygo.postprocessing`: Filter detections, update labels and boxes, apply NMS, and compose result pipelines.
- `klygo.outputs`: Standardized `Box`, `Detection`, `Detections`, and `Crops` result types.
- `klygo.visual`: Display images, draw bounding boxes, and plot dataset statistics.

See the [documentation index](docs/README.md) for module guides and API references.

Preprocessing is lazy: building a pipeline does not decode pixels. If the same
processed image is needed more than once, use `image.to_pil(cache=True)` or
`image.to_array(cache=True)` to reuse the processed result. Call `image.unload()`
to release the cached pixels.

## Quick start

### Archives

```python
import klygo.archive as archive

archive.compress("dataset", "dataset.zip", overwrite=True)
archive.extract("dataset.zip", output_dir="extracted", overwrite=True)
matching_files = archive.search("dataset.zip", "images/*.jpg")
```

### Files and configuration

```python
from klygo import config, files
from klygo.config import Config

settings = config.load("settings.yaml")
config.set(settings, "model.batch", 16)
data = files.load("data.json")
files.save("output.env", {"PORT": "8080"}, overwrite=True)
manager = Config("settings.yaml")
```

### Dataset operations

```python
from klygo import datasets

report = datasets.get_dataset_info("dataset")
datasets.partition(
    source="raw_dataset.zip",
    target="dataset",
    ratios=(0.8, 0.1, 0.1),
    overwrite=True,
)
```

### Detection and post-processing

```python
from klygo import models, postprocessing

model = models.load("grounding-dino-tiny")
results = model.predict("traffic.jpg", prompt="car. person.")
clean_results = postprocessing.compose(
    postprocessing.threshold(0.4),
    postprocessing.clip(),
    postprocessing.nms(iou=0.5),
)(results)
print(clean_results.labels, clean_results.scores)
```

Preprocessed lazy images can be passed directly to `predict`; the model receives
materialized RGB pixels while the result retains the original preprocessing
trace for coordinate restoration:

```python
from klygo import media, preprocessing

cropped = preprocessing.crop((50, 50, 350, 250))(media.load("traffic.jpg")[0])
result = model.predict(cropped, prompt="car. person.")
result[0].restore(target="original").show()
```

## Path conventions

- `source` names an input file or directory.
- `*_path` names a file path.
- `*_dir` names a directory path.
- `target` may name an output file or directory, depending on the operation.

## Tests

Run the test suite with Python's standard library:

```bash
python -m unittest discover -s test -t . -p 'test*.py'
```

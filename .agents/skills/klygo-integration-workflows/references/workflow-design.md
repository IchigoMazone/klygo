# Cross-module workflow design

Use this reference to plan a real run, not as a list of APIs to call mechanically. Confirm signatures and file formats against the checkout. Apply the Klygo module-guide conventions to any public example, documentation, or tutorial that the workflow changes.

## High-level API selection

Apply root `AGENTS.md` Rule 0 and `docs/API_LEVELS.md` before choosing module
handoffs. The public surface is high-level for its own module, including `files`.
Using that surface to reconstruct another module's already supported capability
is a low-level implementation of that task. Choose existing public APIs first;
compose them for larger workflows and customize only verified gaps. Do not force
custom detector classes, box matching, label export, or file-based dataset
partitioning when the owning module already provides the requested behavior.

## Connected path

| Stage | Useful public handoffs | Evidence |
| --- | --- | --- |
| Source and staging | `files` + `archive` + `config` → supported `datasets` layout | Source digest, license, class map, manifest, paired image/label IDs |
| Decode and transform | `datasets` + `media` → `preprocessing` | Decoded dimensions and color mode; transform trace and restored coordinates |
| Infer | `preprocessing` → `models` → `outputs` | Existing-adapter predictions; custom adapters only for verified gaps, device, parameters, image IDs |
| Refine and assess | `outputs` → `postprocessing` → `metrics` | Boxes after threshold/NMS, TP/FP/FN per image and class, thresholds |
| Review and preserve | `visual` + `files` + `archive` + `utils` | Overlays, machine-readable report, inspectable run bundle |

Use `validators` at real input boundaries. `files.load`, `media.load`, `models.load`, and `datasets.load` read different objects; `preprocessing.threshold` concerns pixels while `postprocessing.threshold` concerns prediction confidence. For each workflow, make a compact map of `module`, `public function`, `handoff`, `observed output`, and `not used/reason`. Prefer combinations that expose mismatches in image size, class naming, coordinates, or layout. Do not repeat a call to increase coverage.

## Mandatory Klygo I/O in notebooks and examples

After Klygo is importable, use `files.path`/`files.join`/`files.stem`/`files.relative` for paths; `files.exists`/`files.is_file`/`files.find`/`files.mkdir` for inspection and directories; `files.load`/`files.save` for structured data; `files.hash`/`files.size` for metadata; and `archive.list_files`/`archive.is_valid`/`archive.extract`/`archive.compress` for supported archives. Check each signature and security behavior before substitution. Do not retain `Path(...)`, `.stem`, `.exists()`, `.read_text()`, `.write_text()`, `.mkdir()`, `os.path`, `shutil`, raw `open`, or direct `zipfile` merely for convenience. Use the same rule in notebook generators so regeneration cannot restore noncompliant cells.

Keep standard-library use limited to needs outside those APIs: `os.environ`/`os.getenv` for configuration, `tempfile` for obtaining a temporary root, `subprocess` for installation, and a minimal bootstrap before Klygo can be imported. For the current Colab bundle, the bootstrap may inspect/extract only what is necessary to install bundled Klygo; subsequent work uses `files` and `archive`. If the public API cannot preserve a needed safety invariant, document the exact exception instead of silently bypassing it. Audit all code cells, not just imports, and execute the regenerated notebooks to verify behavior.

## Files and migration shape

Organize one named workflow across `examples/integration/<workflow>.py`, `docs/integration/<workflow>.md`, and, when useful, `tutorials/integration/<workflow>.ipynb`. Indexes link the same workflow. Keep one entry point for preparation and execution when practical; separate model implementations only when independently importable. A class-specific notebook can invoke shared workflow logic. Generated `source/`, `staged/`, `cache/`, `report/`, `preview/`, `bundle/`, and executed notebooks belong under configurable `artifacts/integration/<run-id>/`, outside code directories. The manifest records source digest, class map, layout, models, configuration, outputs, and status.

When refactoring this checkout, inventory `test/integration/`, `examples/integration/`, `docs/integration/`, `tutorials/integration/`, and `test/integration/generated/`. Prepare a file-by-file keep/merge/replace/delete proposal before editing. Shorter code should not hide work in duplicated notebook cells or opaque helpers. Do not create a suite merely to mirror the workflow; execute the workflow itself and inspect its artifacts. Keep a focused regression test only if it catches a specific contract that execution alone misses or the user requests it.

## Dataset and model invariants

Use `models.load()` and the existing detector's `predict()` for supported architectures. When a custom object detector is explicitly requested or an adapter is genuinely missing, its class must inherit `models.Detector` (directly or through an existing Detector subclass) and must implement `forward(images, prompt=None, ...)`. A `BaseModel`-only subclass is insufficient; application code must retain the inherited public prediction path. Code must be concise, readable, and illustrate natural handoffs between modules.

Compare predictions against ground truth using the existing public `metrics` functions. Do not rebuild matching or precision/recall aggregation. Choose the owning module's suitable public function across modules (`files`, `archive`, `datasets`, `media`, `preprocessing`, `models`, `outputs`, `postprocessing`, `metrics`, `visual`, `utils`) to optimize execution performance. Strictly avoid embedding or importing unrelated third-party libraries.

Use real labeled images from a known source. Existing `test/data.zip` is an apple-image archive with no established license in this repository; it can inform a local integration check but must not be pushed again. Its index removal is staged in this checkout, while the ignored local copy remains. The COCO128-based 15-class set is an evaluation-only workflow fixture, not an independent benchmark. Source files remain unchanged. A supported flat YOLO layout has `data.yaml`, `images/`, and optional `labels/`; a split layout needs the supported explicit transformation. Inspect the current `datasets.convert()` and layout APIs before staging a source manually. Keep a custom layout adapter only for a verified unsupported input; do not change dataset-module implementation unless requested.

Derive image keys from metadata, never unrelated traversal order. Match predictions to truth by image and class, with one-to-one IoU matching at declared thresholds. Count unmatched predictions and truth boxes, including negatives; store per-image and aggregate TP/FP/FN. Define zero-denominator behavior. Workflows must execute real pretrained detection architectures with authentic weights from the 4 standard backends rather than dummy or heuristic mock detectors, accepting realistic inference latency to obtain genuine detection precision and recall.

Hardware acceleration must be configured flexibly: `device = "cuda" if torch.cuda.is_available() else "cpu"`. Notebooks and documentation must provide clear guidance for enabling Google Colab GPU runtime (**Runtime → Change runtime type → T4 GPU**) to accelerate multi-model evaluation significantly, while maintaining a smooth fallback to CPU for offline local environments.

For a requested 15-class Colab run, keep one notebook per class if that is the desired interface, but share evaluation implementation. Each notebook exposes class, sample/full-data selection, optional GPU, and output location. Execute locally before claiming local success; claim Colab/GPU success only after execution in a real Colab runtime. Record class results rather than treating notebook completion as model accuracy.

## Four-backend allocation and evidence

For the four-provider mode, default to one 15-class set partitioned 4/4/4/3 across Keras, Torchvision, Hugging Face, and Ultralytics. The manifest must list each class ID/name, assigned backend, actual model and weights/revision, label mapping, selected positive and negative image IDs, device, and run status. Assign only labels supported by the real model; change the allocation transparently if needed while keeping all 15 classes and all four backends. If the user asks for every backend to run all 15, expand to 60 backend-class executions. A failed or unavailable backend remains failed/unavailable, never silently reassigned to a substitute model.

All variants must enter through a Klygo model class/adapter and yield Klygo `outputs`, then use Klygo `postprocessing`, `metrics`, `visual`, and `files`/`archive` for the shared workflow. Framework-specific calls stay behind that adapter. Do not use Torchvision datasets/transforms/NMS, Ultralytics dataset/evaluation/export conveniences, Hugging Face pipeline/evaluate/datasets utilities, Keras image/dataset utilities, or another package as a parallel replacement for a Klygo stage. Native framework tensor handling required to run a model is allowed inside the adapter. Check label semantics and coordinate restoration across each handoff; record any public Klygo API gap rather than bypassing it without disclosure.

Inspect the current model registry, public configuration and adapters before
choosing an architecture or assuming a custom class is necessary. Treat existing
integration examples as examples, not proof of the current inventory or complete
backend/class coverage. Before claiming four-backend completion, execute at least
one real positive and one negative image for every assigned class, confirm all
four actual model backends ran, verify reports/overlays, and report per-class and
per-backend TP/FP/FN separately. A static import, mock prediction, or notebook that
only finishes setup is not execution evidence.

## Git data gate

Before any commit or push, inspect tracked, staged, and ignored data, including archives and weights. Verify that `test/data.zip` remains absent from the index, present locally, and ignored; its removal is already staged in this checkout. Apply the same check to any other tracked dataset. Earlier commits may still contain data and require a separate history-migration decision if removal from repository history is requested.

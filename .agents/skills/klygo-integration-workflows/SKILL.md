---
name: klygo-integration-workflows
description: Build and run concise real-data Klygo workflows that exercise public modules together, with cohesive examples, docs, optional Colab guides, and separately stored artifacts. Use for cross-module integration or multi-class model evaluation; not for a single module's API inventory.
---

# Klygo Integration Workflows

Show how a practitioner completes a real task from source data to inspected results. Prefer executable end-to-end workflows over many small tests or scripts that only import APIs. Read [references/workflow-design.md](references/workflow-design.md) when choosing module handoffs, arranging files, or evaluating models. Inspect the checked-out public API before following any example.

## Default API selection

Read the root `AGENTS.md` Rule 0 and `docs/API_LEVELS.md`. Each module's public
surface is high-level for that module; public calls from lower stages can still
reimplement a higher owning module's existing capability. Map goal → owning
module → verified public API before writing workflow code. Call that API directly,
compose public APIs for multiple stages, and customize only an identified gap.
This takes precedence over maximizing coverage, calling many modules, or showing
backend internals. Never add a helper or custom detector merely to rename or
rebuild a supported function. Keep raw baselines only for explicitly requested
benchmark comparisons.

## Design and execution

1. Trace the actual data flow: inspect source, stage a supported layout, read images and labels, configure/load a model, preprocess, predict, postprocess, compare with ground truth, visualize, and save a report. Map each handoff's type, coordinates, class IDs, paths, and side effects. Choose the public API of the module owning each task; record the selected APIs and any verified gaps. API count is not a goal. Avoid decorative calls and redundant read/write cycles.
2. **Reuse existing model adapters**: Prefer `models.load()` and the loaded detector's public `predict()` for supported models. For an explicit custom object-detector request or a verified missing detector adapter, the custom class **must inherit `models.Detector`** (directly or through an existing Detector subclass) and **must implement `forward(images, prompt=None, ...)`**. A `BaseModel`-only subclass is not sufficient for a custom detector. Keep custom logic at that unsupported backend boundary; local execution does not require a custom class.
3. **Clean, readable, module-linked chaining**: Write concise, easily readable orchestration demonstrating how Klygo modules naturally link:
   `files`/`archive` (storage & extraction) → `datasets` (`info` & `load`) → `models` (existing adapter by default) → `outputs` (`Detections`) → `postprocessing` (box refinement) → `metrics` (matching & evaluation) → `visual` (`draw_bboxes`) → `utils` (formatting & reports).
4. **Ground-truth evaluation and quality benchmarking**: Systematically compare model predictions against ground truth loaded via `datasets.load()` using `metrics.evaluate()`/`metrics.compose()` or the appropriate direct public metric. Do not recreate supported matching or TP/FP/FN aggregation. Report transparent quality metrics in formatted tables.
5. **Prefer the owning module's public API**: Do not expand a supported task into more calls merely to demonstrate modules. Use Klygo's own public modules (`files`, `archive`, `datasets`, `media`, `preprocessing`, `models`, `outputs`, `postprocessing`, `metrics`, `visual`, `utils`) across the entire pipeline. Strictly forbid importing unrelated, decorative, or parallel third-party libraries when a native Klygo API exists.
6. Use labeled real images with provenance, license when known, source digest, selected IDs, and class map. Do not infer truth from predictions. Keep sources immutable and ground truth separate from predictions. Inspect `datasets.convert()`, partitioning and related public APIs before staging a dataset. Use a manual layout adapter only for a verified unsupported source format; do not change dataset-module implementation unless requested.
7. Put reusable orchestration in one coherent entry point; extract helpers only when reused or needed to clarify a real boundary. Drive each requested model implementation through the same workflow and output schema.
8. Follow the [klygo-module-guides](../klygo-module-guides/SKILL.md) standard for user-facing artifacts: public Klygo imports and cross-module reuse, concise English docstrings/comments where needed, runnable examples, accurate English docs, and code-heavy notebooks. **In examples, tutorials, and shared workflow code, use `klygo.files` for every path, filesystem, metadata, download, and structured-file operation that its public API supports; use `klygo.archive` for supported archive operations.** Do not use `Path`, `os.path`, `shutil`, direct `Path` methods, or raw `zipfile` for those operations after Klygo is available. `os.environ`/`os.getenv` remain valid for environment configuration, and a minimal pre-install bootstrap may use standard-library I/O to locate and install the local Klygo bundle. Document any other unsupported operation and its reason. For integration work, organize **by workflow**, not one file per API: `examples/integration/`, `docs/integration/`, and `tutorials/integration/`. Keep a small index linking the same named workflow across them. If one notebook per class is requested, make each a focused class configuration over shared logic, not a copied implementation.
9. Separate code, datasets, and generated results. Put downloaded data, staged data, caches, reports, previews, executed notebooks, and bundles under one configurable ignored root, e.g. `artifacts/integration/<run-id>/{source,staged,cache,report,preview,bundle}`. Record paths in one manifest so a run can be inspected independently. Never scatter generated output beside source files.
10. Execute the real workflow on representative positive and negative images for every requested class, with every requested backend. Check files, labels, IDs, boxes, metrics, and sample overlays. Report exact scope, device, model/cache identity, thresholds, passed steps, failures, and limits. Do not claim GPU/Colab validation from a CPU run. Prefer this execution evidence to new test files; add a short focused regression test only if the user asks for tests or a specific public contract needs one.

## Four-backend, 15-class mode

When requested, provide four real runnable variants using **Keras, Torchvision, Hugging Face, and Ultralytics**. Allocate the 15 requested classes across the four variants (default 4/4/4/3), with each class assigned exactly once and each backend exercised; record class-to-backend assignments in the run manifest. Verify that the chosen model actually supports its assigned labels before running. If the user explicitly wants all 15 classes on each backend, use that broader matrix instead. Use the same labeled source, class IDs, Klygo pipeline, matching rule, and report schema in all variants so the handoffs are comparable.

**Mandatory real models & GPU acceleration**:
- Workflows must use real, pretrained detection architectures with authentic weights from the 4 standard backends rather than dummy or heuristic mock detectors. Accept the realistic inference time needed on CPU/GPU to obtain genuine detection quality (Precision and Recall reflecting true model performance).
- Enable hardware acceleration automatically: configure `device = "cuda" if torch.cuda.is_available() else "cpu"`.
- In Google Colab notebooks, document and provide the one-click setting to switch to a GPU runtime (**Runtime → Change runtime type → T4 GPU**) to accelerate multi-model inference significantly, while gracefully falling back to CPU for local runs.

The four framework packages are allowed **only for backend-specific model construction, inference, and unavoidable tensor conversion inside a Klygo model adapter**. Check the current public media conversion API at that boundary; use it when it exists and meets the model's contract, otherwise keep the necessary tensor conversion inside the custom adapter. Do not substitute their dataset loaders, image transforms, NMS, metrics, visualization, file I/O, or export helpers for public Klygo modules during this evaluation. Connect each variant through `klygo.models` → `outputs` → `postprocessing` → `metrics`, and make the other Klygo modules work together on the real input/output path. For a Torchvision architecture, first check existing public adapters/configuration. Define a custom subclass only when no existing adapter covers the requested architecture, or when the user explicitly asks for one. Verify each actual model and class allocation before claiming the complete four-backend run. Do not replace an unavailable backend with a mock or another library and call it passed. See [references/workflow-design.md](references/workflow-design.md) for the allocation and evidence gate.

Before completion, audit **every code cell of every generated notebook and its generator**, plus each example and shared workflow module. For each direct `pathlib`, `os.path`, `shutil`, `zipfile`, or built-in file-I/O use, either replace it with the corresponding public `files`/`archive` call or record why no suitable Klygo API can serve it. A notebook does not pass this gate merely because it imports `files`; its actual data and artifact path must use those APIs. Re-run the generated notebooks after changing their source generator.

Before replacing the existing integration layout, inventory its scripts, tests, examples, docs, notebooks, and artifacts. Propose which to keep, merge, replace, or remove and why; preserve useful material and do not delete unrelated work. Match module-guide conventions where applicable without copying its per-API file count into integration workflows.

## Data and Git boundary

Do not commit or push image datasets, labels, weights, caches, bundles, reports, or generated data. Ignore artifact paths, then inspect staged and tracked files (`git status`, `git ls-files`, `git check-ignore`) before any commit or push. `.gitignore` does not protect already tracked files. The current working index stages removal of the previously tracked `test/data.zip` while preserving its ignored local copy; verify that state before committing and never re-add it. This does not erase the archive from earlier Git history.

If the request is **skill-only**, update and validate this skill, give a concrete migration proposal, and stop. Do not change project code, data, tests, examples, docs, notebooks, or Git tracking in that turn.

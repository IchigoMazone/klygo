# API levels and selection

## Module-relative definitions

A module's supported public functions, classes and documented public object
methods form its **high-level API**. This applies equally to `files`, `datasets`,
`models`, `outputs`, `mining`, `metrics`, and other modules. Verify the current
facade (`__init__.py`, `__all__`), public methods and implementation before
recommending a call. The export list is the discovery starting point, not a
substitute for reading the current signature and supported behavior.

A **low-level implementation of a task** reconstructs a capability already
provided by the public surface of the module responsible for that task. It may
use only public Klygo APIs and still be an unnecessary reimplementation.

For example, `files.copy()` is high-level for copying a file. Combining file
traversal, random sampling and copies to rebuild a supported dataset partition
is low-level for the partitioning task, which belongs to `datasets`.

A high-level API may internally compose other modules' high-level APIs. Calling
it avoids requiring application authors to repeat its implementation. Private
helpers and backend internals are a separate category; low-level code need not
import a private symbol to duplicate a public capability.

## Default decision order

1. Identify the requested behavior and the module that owns it.
2. Inspect the current public surface, implementation/signature, supported options,
   result types, mutation rules and relevant example.
3. Call a suitable existing public API directly when it covers the behavior.
4. Compose existing public APIs when the goal spans several stages. Keep the
   individual stages on their owning modules' public APIs.
5. Customize only a verified missing capability, preserving existing APIs for
   the supported parts. Briefly state the gap and why that custom boundary exists.
6. Use a fully custom implementation only when explicitly requested or when no
   suitable public API exists. Never invent a convenient-sounding method.

The same order applies to AI answers and proposed code, ordinary application
workflows, runnable examples, documentation, notebooks and notebook generators.
An older example that rebuilds an existing capability is not an exception to
this rule; verify the current API and show its direct use instead.
Using Klygo throughout is necessary but does not justify repeating a higher
module's existing implementation. Prefer the direct supported call over a new
wrapper that merely renames it. Do not add calls just to increase module coverage.

## Task-to-API examples

| Goal | Preferred public surface | Avoid as the default when the goal is supported |
| --- | --- | --- |
| Copy a file | `files.copy()` | A custom byte-copy loop |
| Create a supported train/validation/test partition | `datasets.partition()` | Randomly select filenames and copy image/label pairs manually |
| Normalize a supported dataset layout | `datasets.convert()` | A custom image/label parser and layout exporter |
| Load and run a supported detector | `models.load()` and detector `predict()` | Rebuild backend input preparation, inference and `Detections` conversion |
| Predict over an existing collection while preserving source identities | Copy the collection, then call `Detections.repredict()` | Manually loop over predictions and reconstruct frame identities |
| Remove duplicate images | `mining.deduplicate()` | Recreate hashing, duplicate groups and retention rules |
| Select diverse samples | `mining.select(method="diversity")` | Recreate supported cluster representative selection |
| Evaluate detections | `metrics.evaluate()` or a suitable direct metric | Recreate matching and TP/FP/FN aggregation |
| Export a supported detection dataset | `Detections.export()` | Manually write images, labels and `data.yaml` |

These are semantic choices, not mechanically interchangeable snippets. Check the
requested layout, label behavior, selection semantics, device and data contract
before selecting the existing API. Do not call an API just because its name
sounds related. Every direct public metric remains high-level for its metric;
`evaluate()` is useful for composed/shared evaluation, not a compulsory wrapper.

## Customization boundaries

Writing custom code is appropriate for an explicit custom-model/low-level request,
an unsupported adapter, a verified missing operation, or a contract the current
public API cannot preserve. Explain that requirement and keep the code local to
that gap. Existing model adapters remain the default for supported architectures;
local execution alone is not a reason to write a detector subclass.

For a **custom object detector**, subclass `models.Detector` (directly or through
an existing Detector subclass) and implement `forward(images, prompt=None, ...)`.
Both are mandatory. A generic `BaseModel` subclass alone is not the detector
integration contract. Keep backend-specific inference in `forward()` and retain
the inherited public prediction orchestration for application code. This does not
require users of an already supported detector to write a custom class.

Library implementation may use lower primitives to implement its own public
surface. Focused internal tests may inspect that implementation. Raw-reference
benchmarks may call upstream frameworks directly for the explicit comparison.
None of these cases makes custom reconstruction the recommended application
workflow. Do not recursively call the same API while implementing it.

## Review proposed code

Before presenting or saving a workflow, check each helper, wrapper, subclass,
manual loop, parser, conversion, metric and export step:

- Does an existing public API of the owning module already implement this goal?
- Does the proposed call match its current input, output and mutation contract?
- Is the custom part a genuine gap or an explicitly requested low-level example?
- Can the supported parts stay as direct public calls instead of duplicated logic?

Apply the convention in [repository instructions](../AGENTS.md) and consult the
owning module's reference for exact signatures. No new `high_level`/`low_level`
namespace or duplicate convenience API is required by this convention.

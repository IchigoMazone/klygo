# outputs.DetectionGroup

```python
outputs.DetectionGroup(*collections)
```

## Parameters and contract

One or more distinct, materialized Detections collections, or one mapping of
non-empty string names to collections. Every member must
contain the same unique frame identities and corresponding image_size values.
Frame order, labels, class IDs, and box counts may differ. Empty collections
can be grouped together. Streaming inputs and repeated object references are
rejected. Detection frames are not accepted as group members; use their owning
Detections collection.

## Result and operations

Returns DetectionGroup, which holds the original objects without copying or
merging their annotations. Iteration, indexing, len, and members expose those
references. String indexing selects a member by name. Two positional members
are named `pred` and `truth`; other counts use `set_1`, `set_2`, and so on.
Names follow input order and do not infer semantic roles. Explicit mappings
preserve names. `keys()`, `values()`, and `items()` provide ordered views;
iteration still yields collections, so unpacking remains available.
`datasets.load` returns this same container for multiple label sets.

cuda(device=0), cpu(), astype(16|32|64), keep(labels), drop(labels),
relabel(mapping), compact(), pack(), and unpack() return the same group.
Individual collection methods remain available and may be mixed with group calls.
Label iterables are captured once before applying them to multiple members.

Same-source alignment is checked again before every operation, including after
external frame edits. Operations run sequentially: a backend or operation failure
can leave previous members already changed. Compact reconciles each member's
own class IDs; metrics compares label strings, not local numeric IDs.

## Errors

TypeError for non-Detections members or non-string names. ValueError for empty names/groups, duplicate
references, streams, or inconsistent sources/sizes. Other errors propagate from
the underlying member operation. GPU operations require the optional CuPy extra
and a CUDA device. No implicit CPU/GPU conversion is performed by metrics.

## Example

```python
from klygo import outputs, metrics

pred = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat", score=0.9)],
    source_id="image-a", image_size=(100, 100),
)])
target = outputs.Detections([outputs.Detection(
    [outputs.Box((0, 0, 20, 20), label="cat")],
    source_id="image-a", image_size=(100, 100),
)])
collections = outputs.DetectionGroup(pred, target)
collections.astype(32).pack().unpack()
print(pred.dtype, target.dtype)
print(list(collections.keys()))  # ['pred', 'truth']
print(metrics.precision_recall(collections["pred"], collections["truth"]))
named = outputs.DetectionGroup({"model_a": pred, "ground_truth": target})
named.rename({"model_a": "pred", "ground_truth": "truth"})
print(list(named.keys()))
assert named["pred"] is pred
# With CuPy and CUDA installed:
# collections.cuda(0)
# print(metrics.evaluate(pred, target))
# collections.cpu()
```

[Example](../../../examples/outputs/DetectionGroup.py) ·
[Tests](../../../test/outputs/test_group.py) ·
[GPU tests](../../../test/metrics/test_gpu.py) · [Module](../README.md)

## Rename members

`collections.rename(mapping)` changes member names together and returns the same
group for chaining. Omitted names remain unchanged; an empty mapping is a no-op.
Names must be non-empty strings and every source key must exist. Final names
must be unique, including names omitted from the mapping. Swaps are allowed:

```python
collections.rename({"pred": "truth", "truth": "pred"})
```

Validation finishes before names change. A non-mapping or non-string name raises
`TypeError`; an unknown source name raises `KeyError`; empty/colliding names or
source misalignment raise `ValueError`. Failures leave names unchanged. Member
order, references, annotations, and devices are preserved. Use `relabel(mapping)`
to change class labels inside all members.

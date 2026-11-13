# Download benchmark evidence

Benchmark scripts, reports and aggregate JSON remain in this repository. Original
ZIP evidence is an optional download from the [benchmark evidence release](https://github.com/IchigoMazone/klygo/releases/tag/benchmark-evidence-2026-11-11).
Installing Klygo from a PyPI wheel does not download these assets. Source archives
and shallow clones of the migrated branch do not contain the ZIP files either.
A full Git clone still includes the historical ZIP blobs; no history was rewritten.

| Run | Controlled evidence | People/cars evidence |
| --- | --- | --- |
| `colab_l4_complete_2026_11_11` | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_complete_2026_11_11__controlled_raw_evidence.zip) | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_complete_2026_11_11__extension_evidence.zip) |
| `colab_l4_optimized_2026_11_11` | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_optimized_2026_11_11__controlled_raw_evidence.zip) | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_l4_optimized_2026_11_11__extension_evidence.zip) |
| `colab_t4_complete_2026_11_10` | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_t4_complete_2026_11_10__controlled_raw_evidence.zip) | [Download ZIP](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/colab_t4_complete_2026_11_10__extension_evidence.zip) |

## Verify downloads

Download the desired ZIP files and [SHA256SUMS](https://github.com/IchigoMazone/klygo/releases/download/benchmark-evidence-2026-11-11/SHA256SUMS) into one directory:

```bash
sha256sum --check --ignore-missing SHA256SUMS
```

`release-assets.json` records original paths, SHA-256 digests, sizes and the pinned
source commit. ZIP contents and bytes are unchanged; only release filenames use
run-specific prefixes to avoid collisions. Internal archive manifests retain the
original filenames. Extract controlled and extension archives separately.

## Publication and recovery

The `Publish benchmark evidence` GitHub Actions workflow recovers the six archives
from the pinned source commit, verifies them, uploads them to a draft release,
downloads every asset again for verification, and then publishes the release.
It does not publish a PyPI package or mark the evidence release as latest.
A failed upload leaves the release in draft; rerunning resumes without replacing
existing assets and verifies their bytes before publication. The pinned source
commit remains available for recovery if publication is interrupted.

The mining benchmark currently has committed summary files and a recorded ZIP
checksum, but its original evidence ZIP is not available in Git. It is not among
these six assets and is not claimed to have been migrated.

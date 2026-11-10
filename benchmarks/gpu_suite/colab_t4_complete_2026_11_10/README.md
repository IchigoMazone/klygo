# Complete Tesla T4 benchmark evidence — 2026-11-10

Recovered from saved outputs of the [executed Colab notebook](https://colab.research.google.com/drive/1l2ejVBidUcgaw907giH5V6WIdD46hxMo?usp=sharing). This is the completed rerun, separate from the earlier rounded-summary import.

33 configurations, 172 supported cases, 1,204 individual measured passes (seven repeats each); every selected parity flag passes. Raw non-stream is retained for every configuration. Unsupported cases remain N/A.

| Dataset | Configurations | Supported cases | Measured passes | Report |
|---|---:|---:|---:|---|
| Controlled images/video | 25 | 124 | 868 | [Full report](controlled/REPORT.md) |
| People video | 4 | 24 | 168 | [Full report](people/REPORT.md) |
| Cars video | 4 | 24 | 168 | [Full report](cars/REPORT.md) |

Repeat-level statistics (mean, sample SD, CV, exploratory bootstrap intervals and paired elapsed overhead): [controlled](controlled/STATISTICS.md), [people](people/STATISTICS.md), [cars](cars/STATISTICS.md). Each directory also contains machine-readable raw_statistics.json.

## Original evidence

- [controlled_raw_evidence.zip](controlled_raw_evidence.zip): original full bundle.json, individual timing/latency/telemetry samples, logs, environment, job records, driver, stability and SHA-256 manifest.
- [extension_evidence.zip](extension_evidence.zip): people/cars bundles, source attribution and input hashes, logs, drivers, failed-attempt records and manifest. Its manifest also references the controlled ZIP above.
- [validation.json](validation.json): verified archive hashes and counts. Every original manifest entry was checked. ZIP bytes are unchanged from the saved Colab export.

Extract both archives into separate directories. Use the bundle.json in the controlled archive and real_people/bundle.json / real_cars/bundle.json in the extension archive with [raw_statistics.py](../raw_statistics.py). Read the full reports for latency, memory, throughput and measurement limits.

## Revisions and interpretation

The controlled rerun pins cffb4769b93947f5bbe97d27d9b6c4ebcfcbe9c2. The real-video extension uses f83352df3c45ac19bf2579383985301f80b45bc3, with unchanged core inference code and corrected observation of the executing Ultralytics AutoBackend CUDA parameters. Original factory-return device fields are preserved; they are not post-warmup execution observations.

People and cars are distinct 120-frame public Intel sample-video excerpts, not additional repeats of the controlled bus workload. [People provenance](people/source.json) and [cars provenance](cars/source.json) preserve CC-BY-4.0 credit, source revision, hashes and transcode commands. Failed device-observer attempts remain in the extension archive and are excluded from selected results.

These are recovered GPU measurements; no GPU inference was rerun during recovery. Seven repeats in one session do not establish production accuracy, a latency SLA or cross-GPU performance. For another accelerator, follow [OTHER_GPUS.md](../OTHER_GPUS.md) and retain independent raw evidence.

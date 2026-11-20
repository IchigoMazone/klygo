"""Tests for image-level dataset mining."""

from io import StringIO
import tempfile
import unittest

import numpy as np
from PIL import Image
from rich.console import Console

from klygo import files, mining
from klygo.outputs import Box, Detection, Detections

try:
    import cupy as cp

    _CUDA_READY = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, OSError, RuntimeError):
    _CUDA_READY = False


class MiningOperationsTest(unittest.TestCase):
    """Exercise mining reports, selection, duplicates, and clustering."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = files.path(self.temporary.name)
        colors = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (80, 80, 80), (160, 160, 160)]
        self.paths = []
        frames = []
        labels = [("car",), ("car",), ("person",), ("car", "person"), (), ("bike",)]
        names = {0: "car", 1: "person", 2: "bike"}
        for index, (color, frame_labels) in enumerate(zip(colors, labels)):
            path = root / f"image_{index}.png"
            pixels = np.full((48, 64, 3), color, dtype=np.uint8)
            if index >= 2:
                pixels[index * 4:index * 4 + 5, :, :] = np.asarray(colors[(index + 1) % len(colors)])
                pixels[:, index * 6:index * 6 + 3, :] = 255 - pixels[:, index * 6:index * 6 + 3, :]
            Image.fromarray(pixels).save(path)
            self.paths.append(path)
            boxes = [Box((2 + offset, 3 + offset, 20 + offset, 25 + offset), label=label,
                         class_id={value: key for key, value in names.items()}[label])
                     for offset, label in enumerate(frame_labels)]
            frames.append(Detection(boxes, source_id=str(path), source_path=str(path),
                                    frame_index=0, image_size=(64, 48), names=names))
        self.data = Detections(frames, names=names, metadata={"split": "train"})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_verbose_is_optional_and_preserves_all_results(self) -> None:
        from contextlib import redirect_stderr
        from unittest.mock import patch
        calls = [
            (mining.stats, {}), (mining.score, {}),
            (mining.find_duplicates, {}), (mining.deduplicate, {"keep": "first"}),
            (mining.cluster, {"features": ("histogram",), "clusters": 2}),
            (mining.select, {"method": "quality", "limit": 2}),
            (mining.select, {"method": "confidence", "threshold": 0.0, "limit": 2}),
            (mining.select, {"method": "diversity", "limit": 2}),
            (mining.balance, {}),
        ]
        for operation, options in calls:
            with self.subTest(operation=operation.__name__, options=options):
                with patch("klygo.utils.progress.tqdm") as bars:
                    expected = operation(self.data, **options).to_dict()
                    bars.assert_not_called()
                with redirect_stderr(StringIO()) as output:
                    actual = operation(self.data, verbose=True, **options).to_dict()
                self.assertEqual(actual, expected)
                self.assertTrue(output.getvalue())
                with self.assertRaises(TypeError):
                    operation(self.data, verbose="yes", **options)
        with redirect_stderr(StringIO()):
            actual = mining.compare(self.data, self.data, quality=True, verbose=True)
        self.assertEqual(actual.to_dict(), mining.compare(self.data, self.data, quality=True).to_dict())

    def test_stats_and_score_do_not_mutate_input(self) -> None:
        before = self.data.copy()
        summary = mining.stats(self.data)
        scored = mining.score(self.data)
        self.assertEqual((summary.images, summary.boxes, summary.empty_images), (6, 6, 1))
        self.assertEqual(summary.images_per_class, {"bike": 1, "car": 3, "person": 2})
        self.assertEqual(len(scored.values), 6)
        self.assertTrue(all(0 <= value <= 1 for value in scored.values))
        self.assertEqual([frame.labels for frame in self.data], [frame.labels for frame in before])

    def test_report_to_dict_and_save(self) -> None:
        summary = mining.stats(self.data, quality=False)
        self.assertEqual(summary.to_dict()["images"], 6)
        self.assertFalse(hasattr(summary, "show"))
        for suffix in ("json", "csv", "md"):
            path = files.path(self.temporary.name) / f"report.{suffix}"
            summary.save(path)
            self.assertTrue(files.exists(path))
        scores = mining.score(self.data, weights={"rarity": 1.0})
        scores.save(files.path(self.temporary.name) / "scores.csv")
        duplicates = mining.find_duplicates(self.data)
        duplicate_path = files.path(self.temporary.name) / "duplicates.json"
        duplicates.save(duplicate_path)
        self.assertEqual(files.load(duplicate_path, verbose=False)["duplicate_images"], 1)
        clusters = mining.cluster(self.data, features=("histogram",), clusters=2)
        cluster_path = files.path(self.temporary.name) / "clusters.json"
        clusters.save(cluster_path)
        self.assertEqual(len(files.load(cluster_path, verbose=False)["feature_vectors"]), len(self.data))

    def test_duplicate_detection_and_keep_policy(self) -> None:
        report = mining.find_duplicates(self.data, threshold=0.99)
        self.assertIn([0, 1], report.groups)
        reduced = mining.deduplicate(self.data, keep="first", threshold=0.99)
        self.assertEqual(len(reduced), 5)
        self.assertEqual(len(self.data), 6)
        reduced[0].clear()
        self.assertEqual(len(self.data[0]), 1)

    def test_cluster_is_deterministic(self) -> None:
        first = mining.cluster(self.data, features=("histogram", "boxes", "classes"), clusters=3, seed=7)
        second = mining.cluster(self.data, features=("histogram", "boxes", "classes"), clusters=3, seed=7)
        self.assertEqual(first.labels, second.labels)
        np.testing.assert_array_equal(first.feature_vectors, second.feature_vectors)
        self.assertEqual(sorted(index for group in first.groups.values() for index in group), list(range(6)))

    def test_all_selection_modes_return_whole_images(self) -> None:
        result = mining.cluster(self.data, features=("histogram", "classes"), clusters=2)
        cases = [
            mining.select(self.data, method="class", classes=["person"]),
            mining.select(self.data, method="random", limit=2, seed=1),
            mining.select(self.data, method="quality", threshold=0.0, limit=2),
            mining.select(self.data, method="boxes", min_boxes=2),
            mining.select(self.data, method="quota", per_class=1),
            mining.select(self.data, method="rare_first", limit=2),
            mining.select(self.data, method="coverage", per_class=1),
            mining.select(self.data, method="diversity", cluster_result=result),
            mining.select(self.data, method="weighted", limit=2),
            mining.select(self.data, method="confidence", threshold=0.5, limit=2),
        ]
        self.assertEqual([len(value) for value in cases[:4]], [2, 2, 2, 1])
        self.assertTrue(all(isinstance(value, Detections) for value in cases))
        self.assertTrue(all(set(value.names.values()) <= set(self.data.names.values()) for value in cases))

    def test_confidence_selection_modes_and_validation(self) -> None:
        names = {0: "car", 1: "person"}
        score_groups = [
            [0.9],          # idx 0: mean=0.9, min=0.9, max=0.9, median=0.9
            [0.2, 0.8],     # idx 1: mean=0.5, min=0.2, max=0.8, median=0.5
            [0.4, 0.6, 0.8],  # idx 2: mean=0.6, min=0.4, max=0.8, median=0.6
            [],             # idx 3: empty frame -> excluded
            [0.3],          # idx 4: mean=0.3, min=0.3, max=0.3, median=0.3
        ]
        frames = []
        for idx, frame_scores in enumerate(score_groups):
            boxes = [
                Box((2 + i, 3 + i, 20 + i, 25 + i), label="car", class_id=0, score=value)
                for i, value in enumerate(frame_scores)
            ]
            frames.append(
                Detection(
                    boxes,
                    source_id=str(self.paths[idx]),
                    source_path=str(self.paths[idx]),
                    frame_index=0,
                    image_size=(64, 48),
                    names=names,
                )
            )
        scored = Detections(frames, names=names)

        # Default desc + mean with threshold=0.55 -> idx 0 (0.9) and idx 2 (0.6)
        high = mining.select(scored, method="confidence", threshold=0.55)
        self.assertEqual([frame.source_id for frame in high], [str(self.paths[0]), str(self.paths[2])])

        # Top-1 highest confidence with limit=1
        top_one = mining.select(scored, method="confidence", threshold=0.0, limit=1)
        self.assertEqual([frame.source_id for frame in top_one], [str(self.paths[0])])

        # Asc order (lowest confidence first) excludes empty frame idx 3 -> idx 4 (0.3), idx 1 (0.5)
        low_two = mining.select(scored, method="confidence", threshold=0.0, order="asc", limit=2)
        self.assertEqual([frame.source_id for frame in low_two], [str(self.paths[1]), str(self.paths[4])])

        # Min aggregator with threshold=0.4 -> idx 0 (0.9) and idx 2 (0.4)
        min_sel = mining.select(scored, method="confidence", threshold=0.4, aggregate="min")
        self.assertEqual([frame.source_id for frame in min_sel], [str(self.paths[0]), str(self.paths[2])])

        # Max aggregator with threshold=0.8 -> idx 0 (0.9), idx 1 (0.8), idx 2 (0.8)
        max_sel = mining.select(scored, method="confidence", threshold=0.8, aggregate="max")
        self.assertEqual([frame.source_id for frame in max_sel], [str(self.paths[i]) for i in (0, 1, 2)])

        # Median aggregator with threshold=0.6 -> idx 0 (0.9), idx 2 (0.6)
        med_sel = mining.select(scored, method="confidence", threshold=0.6, aggregate="median")
        self.assertEqual([frame.source_id for frame in med_sel], [str(self.paths[0]), str(self.paths[2])])

        with self.assertRaisesRegex(ValueError, "threshold"):
            mining.select(scored, method="confidence", threshold=1.5)
        with self.assertRaisesRegex(ValueError, "aggregate"):
            mining.select(scored, method="confidence", aggregate="sum")
        with self.assertRaisesRegex(ValueError, "order"):
            mining.select(scored, method="confidence", order="invalid")

    def test_diversity_backfills_to_limit(self) -> None:
        from klygo.mining.curation import ClusterResult

        clusters = mining.cluster(self.data, features=("histogram", "classes"), clusters=2, seed=42)

        # 1. limit=None -> at most per_cluster per cluster
        no_limit = mining.select(self.data, method="diversity", cluster_result=clusters, per_cluster=1)
        self.assertEqual(len(no_limit), 2)

        # 2. limit > clusters * per_cluster -> round-robin backfill to limit
        filled = mining.select(self.data, method="diversity", cluster_result=clusters, per_cluster=1, limit=5)
        self.assertEqual(len(filled), 5)
        self.assertEqual(len({frame.source_id for frame in filled}), 5)

        # 3. limit < number of representatives -> evenly spaced across clusters (not first-cluster biased)
        custom_clusters = ClusterResult(
            labels=[0, 0, 1, 2, 3, 3],
            groups={0: [0, 1], 1: [2], 2: [], 3: [3], 4: [4, 5]},
            representatives={0: 0, 1: 2, 2: 0, 3: 3, 4: 4},
            features=("histogram",),
            clusters=5,
            seed=42,
            feature_vectors=np.zeros((6, 1), dtype=np.float32),
        )
        # Non-empty buckets: [0, 1], [2], [3], [4, 5] (4 buckets). limit=2 -> buckets 0 (idx 0) and 2 (idx 3)
        sub_rep = mining.select(self.data, method="diversity", cluster_result=custom_clusters, per_cluster=2, limit=2)
        self.assertEqual([frame.source_id for frame in sub_rep], [str(self.paths[0]), str(self.paths[3])])

        # 4. limit < clusters * per_cluster (4 buckets, per_cluster=2, limit=5) -> all 4 representatives + 1 extra
        partial = mining.select(self.data, method="diversity", cluster_result=custom_clusters, per_cluster=2, limit=5)
        partial_ids = {frame.source_id for frame in partial}
        self.assertEqual(len(partial), 5)
        for rep_idx in (0, 2, 3, 4):
            self.assertIn(str(self.paths[rep_idx]), partial_ids)

        # 5, 6, 7. Empty cluster skipped, small clusters handled, limit > len(data) capped at len(data)
        capped = mining.select(self.data, method="diversity", cluster_result=custom_clusters, per_cluster=3, limit=10)
        self.assertEqual(len(capped), len(self.data))

    def test_score_coverage_weights_validation_and_edge_cases(self) -> None:
        # 1. Coverage uses present classes rather than unused classes in data.names
        names = {0: "car", 1: "person", 2: "unused_a", 3: "unused_b"}
        frames = [
            Detection(
                [Box((2, 3, 20, 25), label="car", class_id=0),
                 Box((10, 10, 30, 30), label="person", class_id=1)],
                source_id=str(self.paths[0]), source_path=str(self.paths[0]),
                frame_index=0, image_size=(64, 48), names=names,
            ),
            Detection(
                [],
                source_id=str(self.paths[1]), source_path=str(self.paths[1]),
                frame_index=0, image_size=(64, 48), names=names,
            ),
        ]
        subset = Detections(frames, names=names)
        scored = mining.score(subset)
        self.assertEqual(scored.details[0]["coverage"], 1.0)
        self.assertEqual(scored.details[1]["coverage"], 0.0)
        self.assertEqual(scored.details[1]["boxes"], 0.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scored.values))

        # 2. Empty dataset and single-image dataset
        empty_data = Detections([], names=names)
        empty_scores = mining.score(empty_data)
        self.assertEqual((empty_scores.values, empty_scores.details, empty_scores.identities), ([], [], []))

        single_data = Detections([frames[0]], names=names)
        single_scores = mining.score(single_data)
        self.assertEqual(len(single_scores.values), 1)
        self.assertTrue(0.0 <= single_scores.values[0] <= 1.0)
        self.assertEqual(single_scores.details[0]["coverage"], 1.0)

        # 3. Strict weights validation
        for bad_weights in (
            {},
            {"unknown": 1.0},
            {"quality": -0.1, "rarity": 1.0},
            {"quality": float("nan")},
            {"quality": float("inf")},
            {"quality": 0.0, "rarity": 0.0},
        ):
            with self.assertRaises(ValueError):
                mining.score(self.data, weights=bad_weights)

        for bad_type in (["quality"], {"quality": True}, {"quality": "0.5"}):
            with self.assertRaises(TypeError):
                mining.score(self.data, weights=bad_type)  # type: ignore[arg-type]

    def test_stats_missing_classes_box_imbalance_percentiles_and_bad_image(self) -> None:
        names = {0: "car", 1: "person", 2: "truck"}
        bad_path = files.path(self.temporary.name) / "missing_image.png"
        frames = [
            Detection(
                [
                    Box((0, 0, 16, 12), label="car", class_id=0),
                    Box((2, 2, 18, 14), label="car", class_id=0),
                    Box((4, 4, 20, 16), label="car", class_id=0),
                ],
                source_id=str(self.paths[0]), source_path=str(self.paths[0]),
                frame_index=0, image_size=(64, 48), names=names,
            ),
            Detection(
                [Box((0, 0, 32, 24), label="person", class_id=1)],
                source_id=str(bad_path), source_path=str(bad_path),
                frame_index=0, image_size=(64, 48), names=names,
            ),
        ]
        dataset = Detections(frames, names=names, metadata={"names": names})
        report = mining.stats(dataset, quality=True)

        self.assertEqual(report.classes, 3)
        self.assertEqual(report.present_classes, 2)
        self.assertEqual(report.missing_classes, ["truck"])
        self.assertEqual(report.imbalance_ratio, 1.0)
        self.assertEqual(report.box_imbalance_ratio, 3.0)
        self.assertEqual(report.failed_quality_images, 1)
        for key in ("mean", "median", "min", "max", "p25", "p50", "p75", "p90"):
            self.assertIn(key, report.box_area)

        # Co-occurrence on single-class and empty datasets
        single_class = Detections([frames[0]], names={0: "car"})
        self.assertEqual(mining.stats(single_class, quality=False).cooccurrence, {})
        self.assertEqual(mining.stats(Detections([], names={}), quality=False).cooccurrence, {})

    def test_end_to_end_pipeline_stats_score_select_export_and_empty_consistency(self) -> None:
        from klygo import datasets

        empty = Detections([], names=self.data.names)
        self.assertEqual(len(mining.select(empty, method="weighted", limit=5)), 0)
        self.assertEqual(len(mining.select(empty, method="diversity", limit=5)), 0)
        self.assertEqual(len(mining.select(empty, method="random", limit=5)), 0)
        self.assertEqual(len(mining.select(empty, method="confidence", limit=5)), 0)

        before = mining.stats(self.data, quality=True)
        scores = mining.score(self.data)
        selected = mining.select(self.data, method="weighted", scores=scores, limit=3)
        after = mining.stats(selected, quality=True)
        self.assertEqual(after.images, 3)
        self.assertLessEqual(after.images, before.images)

        export_dir = files.path(self.temporary.name) / "exported_subset"
        selected.export(export_dir, format="yolo", overwrite=True)
        reloaded = datasets.load(export_dir, format="yolo", verbose=False)
        self.assertEqual(len(reloaded), 3)
        self.assertEqual(
            sorted(sum(len(frame) for frame in reloaded) for _ in [0]),
            [sum(len(frame) for frame in selected)],
        )

    def test_balance_and_compare(self) -> None:
        balanced = mining.balance(self.data, target="min")
        comparison = mining.compare(self.data, balanced)
        self.assertLessEqual(len(balanced), len(self.data))
        self.assertEqual(comparison.before.images, 6)
        self.assertEqual(comparison.after.images, len(balanced))
        self.assertIn("unmet_quotas", balanced.metadata["mining"])

    def test_invalid_requests_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown classes"):
            mining.select(self.data, method="class", classes=["plane"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            mining.select(self.data, method="class", classes=[])
        with self.assertRaisesRegex(ValueError, "strategy"):
            mining.balance(self.data, strategy="oversample")
        with self.assertRaisesRegex(ValueError, "clusters"):
            mining.cluster(self.data, clusters=99)

    @unittest.skipUnless(_CUDA_READY, "CuPy CUDA runtime is unavailable")
    def test_gpu_detections_are_preserved_across_mining(self) -> None:
        gpu = self.data.copy().cuda()
        cpu_stats = mining.stats(self.data, quality=False)
        gpu_stats = mining.stats(gpu, quality=False)
        self.assertEqual(cpu_stats.to_dict(), gpu_stats.to_dict())

        scores = mining.score(gpu, weights={"rarity": 1.0})
        duplicates = mining.find_duplicates(gpu)
        clusters = mining.cluster(gpu, features=("histogram", "boxes", "classes"), clusters=2)
        outputs = [
            mining.select(gpu, method="weighted", scores=scores, limit=2),
            mining.balance(gpu, target="min"),
            mining.deduplicate(gpu, keep="first"),
        ]
        self.assertEqual((len(scores.values), duplicates.duplicate_images, len(clusters.labels)), (6, 1, 6))
        self.assertTrue(all(result.device == "cuda:0" for result in outputs))
        self.assertTrue(all(frame.device == "cuda:0" for result in outputs for frame in result))
        self.assertEqual((gpu.device, len(gpu)), ("cuda:0", 6))

        for precision in (16, 32, 64):
            typed = self.data.copy().astype(precision).cuda()
            selected = mining.select(typed, method="random", limit=2, seed=precision)
            self.assertEqual((selected.device, selected.dtype), ("cuda:0", precision))

        packed = Detections(list(self.data.copy()), names=self.data.names, packed=True).cuda().astype(32)
        packed_selected = mining.select(packed, method="random", limit=2)
        self.assertEqual((packed_selected.device, packed_selected.dtype, packed_selected.is_packed),
                         ("cuda:0", 32, True))


if __name__ == "__main__":
    unittest.main()

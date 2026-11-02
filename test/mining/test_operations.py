"""Tests for image-level dataset mining."""

from io import StringIO
import tempfile
import unittest

import numpy as np
from PIL import Image
from rich.console import Console
from unittest.mock import patch

from klygo.mining.curation import ClusterResult

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

    def test_stats_and_score_do_not_mutate_input(self) -> None:
        before = self.data.copy()
        summary = mining.stats(self.data)
        scored = mining.score(self.data)
        self.assertEqual((summary.images, summary.boxes, summary.empty_images), (6, 6, 1))
        self.assertEqual(summary.images_per_class, {"bike": 1, "car": 3, "person": 2})
        self.assertEqual(len(scored.values), 6)
        self.assertTrue(all(0 <= value <= 1 for value in scored.values))
        self.assertEqual([frame.labels for frame in self.data], [frame.labels for frame in before])

    def test_report_show_and_save(self) -> None:
        summary = mining.stats(self.data, quality=False)
        stream = StringIO()
        summary.show(Console(file=stream, force_terminal=False, width=120))
        self.assertIn("MiningStats", stream.getvalue())
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
        ]
        self.assertEqual([len(value) for value in cases[:4]], [2, 2, 2, 1])
        self.assertTrue(all(isinstance(value, Detections) for value in cases))
        self.assertTrue(all(set(value.names.values()) <= set(self.data.names.values()) for value in cases))

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


    def test_diversity_seed_is_used_and_limit_covers_distinct_clusters(self) -> None:
        groups = {index: [index] for index in range(len(self.data))}
        result = ClusterResult(
            labels=list(range(len(self.data))),
            groups=groups,
            representatives={index: index for index in groups},
            features=("histogram",),
            clusters=len(groups),
            seed=0,
            feature_vectors=np.zeros((len(self.data), 1), dtype=np.float32),
        )
        first = mining.select(
            self.data, method="diversity", cluster_result=result, limit=3, seed=1
        )
        second = mining.select(
            self.data, method="diversity", cluster_result=result, limit=3, seed=999
        )
        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertNotEqual(
            {frame.source_id for frame in first},
            {frame.source_id for frame in second},
        )

        with patch("klygo.mining.curation.cluster", return_value=result) as cluster_mock:
            mining.select(self.data, method="diversity", limit=3, seed=17)
        self.assertEqual(cluster_mock.call_args.kwargs["seed"], 17)

    def test_diversity_limit_zero_returns_no_frames(self) -> None:
        result = mining.select(self.data, method="diversity", limit=0, seed=2)
        self.assertEqual(len(result), 0)

    def test_balance_preserves_background_and_seeded_ties(self) -> None:
        balanced = mining.balance(self.data, target="min", keep_background=True)
        self.assertTrue(any(len(frame) == 0 for frame in balanced))
        dropped = mining.balance(self.data, target="min", keep_background=False)
        self.assertFalse(any(len(frame) == 0 for frame in dropped))

        tie_frames = [
            Detection(
                [Box((1, 1, 10, 10), label="object", class_id=0)],
                source_id=f"tie-{index}",
                image_size=(32, 32),
                names={0: "object"},
            )
            for index in range(8)
        ]
        tied = Detections(tie_frames, names={0: "object"})
        first = mining.balance(tied, target=1, strategy="undersample", seed=1)
        second = mining.balance(tied, target=1, strategy="undersample", seed=999)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].source_id, second[0].source_id)

    def test_score_validates_empty_weights_and_zero_quality_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "weights"):
            mining.score(self.data, weights={})

        with patch("klygo.mining.analysis._quality", side_effect=AssertionError("quality should be skipped")):
            scored = mining.score(self.data, weights={"quality": 0.0, "rarity": 1.0})
        self.assertTrue(all(row["quality"] == 0.0 for row in scored.details))

    def test_score_rows_are_ranked_and_comparison_exports_are_tabular(self) -> None:
        scored = mining.score(self.data, weights={"rarity": 1.0})
        rows = scored._rows()
        self.assertEqual([row["rank"] for row in rows], list(range(1, len(rows) + 1)))
        self.assertEqual(
            [row["score"] for row in rows],
            sorted((row["score"] for row in rows), reverse=True),
        )

        comparison = mining.compare(self.data, self.data[:3])
        rows = comparison._rows()
        self.assertEqual(rows[0]["metric"], "images")
        self.assertEqual(set(rows[0]), {"metric", "before", "after", "delta"})
        csv_path = files.path(self.temporary.name) / "comparison.csv"
        md_path = files.path(self.temporary.name) / "comparison.md"
        comparison.save(csv_path)
        comparison.save(md_path)
        csv_text = files.load(csv_path, verbose=False)
        md_text = files.load(md_path, verbose=False)
        self.assertIn("before", str(csv_text))
        self.assertIn("delta", str(csv_text))
        self.assertIn("| metric | before | after | delta |", md_text)

    def test_duplicate_detection_supports_memory_backed_images(self) -> None:
        image = Image.new("RGB", (32, 32), (120, 80, 40))
        frames = [
            Detection(
                [Box((1, 1, 10, 10), label="object", class_id=0)],
                source_id=f"memory-{index}",
                source=image.copy(),
                image_size=(32, 32),
                names={0: "object"},
            )
            for index in range(2)
        ]
        memory_data = Detections(frames, names={0: "object"})
        report = mining.find_duplicates(memory_data, exact=True, threshold=1.0)
        self.assertEqual(report.groups, [[0, 1]])

    def test_hog_feature_dimension_is_576_and_modality_blocks_are_balanced(self) -> None:
        from klygo.mining._common import _feature_matrix

        matrix = _feature_matrix(
            self.data, ("histogram", "hog", "quality", "boxes", "classes")
        )
        self.assertEqual(matrix.shape, (len(self.data), 652))
        block_norms = [
            float(np.mean(np.linalg.norm(matrix[:, start:end], axis=1)))
            for start, end in ((0, 64), (64, 640), (640, 646), (646, 649), (649, 652))
        ]
        self.assertLess(max(block_norms) / max(min(block_norms), 1e-8), 3.0)

    def test_duplicate_report_uses_sha256_for_exact_file_hashes(self) -> None:
        with patch("klygo.mining.duplicates.files.hash", return_value="same") as hash_mock:
            mining.find_duplicates(self.data, exact=True, threshold=1.0)
        self.assertTrue(hash_mock.called)
        self.assertTrue(all(call.kwargs.get("algorithm") == "sha256" for call in hash_mock.call_args_list))

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

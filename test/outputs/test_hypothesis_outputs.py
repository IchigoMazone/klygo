"""Optional property-based and stateful tests for detection outputs."""

import math
import unittest

try:
    from hypothesis import given, settings, strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
except ImportError:
    given = settings = st = None
    RuleBasedStateMachine = invariant = precondition = rule = None

from klygo import outputs


LABELS = ("car", "person", "bus", "truck", "bicycle", "猫", "xe máy")


def make_frame(source_id="source", frame_index=0, labels=("car", "person")):
    """Build a small deterministic frame for generated operation sequences."""
    return outputs.Detection(
        [
            outputs.Box([index, index, index + 2, index + 3], label=label)
            for index, label in enumerate(labels)
        ],
        source_id=source_id,
        frame_index=frame_index,
        image_size=(128, 128),
    )


def assert_frame_consistent(frame):
    """Assert frame-local UID and label registry invariants."""
    assert frame.count == len(frame)
    assert len(frame.uids) == len(set(frame.uids))
    assert set(frame.names.values()) == set(frame.labels)
    label_ids = {}
    for box in frame:
        assert frame.names[box.class_id] == box.label
        if box.label in label_ids:
            assert label_ids[box.label] == box.class_id
        label_ids[box.label] = box.class_id


def assert_collection_consistent(results):
    """Assert collection identity, count, and global registry invariants."""
    assert len({frame.identity for frame in results}) == len(results)
    assert results.total_objects == sum(len(frame) for frame in results)
    active = {box.label for frame in results for box in frame}
    assert set(results.names.values()) == active
    label_ids = {}
    for frame in results:
        assert_frame_consistent(frame)
        for box in frame:
            assert results.names[box.class_id] == box.label
            if box.label in label_ids:
                assert label_ids[box.label] == box.class_id
            label_ids[box.label] = box.class_id


@unittest.skipIf(st is None, "Hypothesis is not installed")
class TestGeneratedBoundaries(unittest.TestCase):
    """Generate values around public validation boundaries."""

    if st is not None:
        coordinates = st.tuples(
            st.integers(-10_000, 10_000),
            st.integers(-10_000, 10_000),
            st.integers(1, 1_000),
            st.integers(1, 1_000),
        )

        @settings(max_examples=500, deadline=None)
        @given(coordinates, st.sampled_from(LABELS), st.floats(0, 1, allow_nan=False))
        def test_valid_boxes_preserve_geometry(self, values, label, score):
            x, y, width, height = values
            box = outputs.Box([x, y, x + width, y + height], label=label, score=score)
            self.assertEqual(box.size, (float(width), float(height)))
            self.assertTrue(math.isfinite(box.area))

        @settings(max_examples=100, deadline=None)
        @given(st.one_of(
            st.floats(allow_nan=False, allow_infinity=False).filter(lambda value: not value.is_integer()),
            st.text(min_size=1).filter(lambda value: value.lstrip("-").isdigit()),
            st.booleans(),
        ))
        def test_class_ids_require_actual_integers(self, invalid_id):
            with self.assertRaises((TypeError, ValueError)):
                outputs.Detection(
                    source_id="names",
                    image_size=(10, 10),
                    names={invalid_id: "car"},
                )

        @settings(max_examples=100, deadline=None)
        @given(st.one_of(
            st.floats(1, 1_000, allow_nan=False, allow_infinity=False).filter(
                lambda value: not value.is_integer()
            ),
            st.integers(1, 1_000).map(str),
            st.booleans(),
        ))
        def test_image_dimensions_require_actual_integers(self, invalid_dimension):
            with self.assertRaises((TypeError, ValueError)):
                outputs.Detection(
                    source_id="size",
                    image_size=(invalid_dimension, 10),
                )

        @given(st.booleans())
        def test_scores_reject_booleans(self, score):
            with self.assertRaises(TypeError):
                outputs.Box([0, 0, 1, 1], label="car", score=score)

        @given(st.floats(0, 1, allow_nan=False).map(str))
        def test_scores_reject_numeric_strings(self, score):
            with self.assertRaises(TypeError):
                outputs.Box([0, 0, 1, 1], label="car", score=score)

        @given(st.integers(0, 3), st.booleans())
        def test_coordinates_reject_booleans(self, index, coordinate):
            values = [0, 0, 2, 2]
            values[index] = coordinate
            with self.assertRaises(TypeError):
                outputs.Box(values, label="car")

        @given(st.integers(0, 3), st.integers(-1_000, 1_000).map(str))
        def test_coordinates_reject_numeric_strings(self, index, coordinate):
            values = [0, 0, 2, 2]
            values[index] = coordinate
            with self.assertRaises(TypeError):
                outputs.Box(values, label="car")

        @settings(max_examples=300, deadline=None)
        @given(
            st.integers(-10_000, 10_000),
            st.integers(-10_000, 10_000),
            st.integers(1, 1_000),
            st.integers(1, 1_000),
            st.floats(-1_000, 1_000, allow_nan=False, allow_infinity=False),
            st.floats(-1_000, 1_000, allow_nan=False, allow_infinity=False),
        )
        def test_translate_preserves_size(self, x, y, width, height, dx, dy):
            box = outputs.Box([x, y, x + width, y + height], label="car")
            original_size = box.size
            box.translate(dx, dy)
            self.assertAlmostEqual(box.width, original_size[0], places=7)
            self.assertAlmostEqual(box.height, original_size[1], places=7)
            self.assertAlmostEqual(box.xmin, x + dx, places=7)
            self.assertAlmostEqual(box.ymin, y + dy, places=7)

        @settings(max_examples=300, deadline=None)
        @given(
            st.integers(-1_000, 1_000),
            st.integers(-1_000, 1_000),
            st.integers(1, 500),
            st.integers(1, 500),
            st.floats(0.01, 20, allow_nan=False, allow_infinity=False),
        )
        def test_scale_preserves_center_and_scales_area(self, x, y, width, height, factor):
            box = outputs.Box([x, y, x + width, y + height], label="car")
            center, area = box.center, box.area
            box.scale(factor)
            self.assertAlmostEqual(box.center_x, center[0], places=7)
            self.assertAlmostEqual(box.center_y, center[1], places=7)
            self.assertAlmostEqual(box.area, area * factor * factor, places=5)

        @given(st.one_of(st.booleans(), st.text(min_size=1)))
        def test_transform_parameters_reject_non_numbers_atomically(self, value):
            box = outputs.Box([0, 0, 4, 6], label="car")
            before = (box.box, box.score, box.track_id, dict(box.metadata))
            with self.assertRaises((TypeError, ValueError)):
                box.translate(value, 0)
            self.assertEqual(
                (box.box, box.score, box.track_id, box.metadata), before
            )
            with self.assertRaises((TypeError, ValueError)):
                box.scale(value)
            self.assertEqual(
                (box.box, box.score, box.track_id, box.metadata), before
            )

        @settings(max_examples=200, deadline=None)
        @given(
            st.lists(st.sampled_from(LABELS), min_size=0, max_size=30),
            st.integers(0, 10),
        )
        def test_duplicate_input_uids_are_reassigned_without_reuse(self, labels, uid):
            boxes = [
                outputs.Box([0, 0, 1, 1], label=label, uid=uid)
                for label in labels
            ]
            frame = outputs.Detection(
                boxes, source_id="uids", image_size=(10, 10)
            )
            self.assertEqual(len(frame.uids), len(set(frame.uids)))
            previous = set(frame.uids)
            frame.clear()
            frame.add_box([0, 0, 1, 1], label="car")
            self.assertNotIn(frame[0].uid, previous)

        @settings(max_examples=250, deadline=None)
        @given(
            st.lists(st.sampled_from(LABELS), min_size=0, max_size=30),
            st.sets(st.sampled_from(LABELS)),
        )
        def test_filter_matches_label_oracle_and_preserves_uids(self, labels, allowed):
            frame = make_frame(labels=labels)
            expected = [(box.uid, box.label) for box in frame if box.label in allowed]
            frame.filter(lambda box: box.label in allowed)
            self.assertEqual([(box.uid, box.label) for box in frame], expected)
            assert_frame_consistent(frame)

        @settings(max_examples=200, deadline=None)
        @given(
            st.lists(st.sampled_from(LABELS), min_size=0, max_size=20),
            st.dictionaries(st.sampled_from(LABELS), st.sampled_from(LABELS), max_size=7),
        )
        def test_relabel_matches_oracle_when_labels_merge(self, labels, mapping):
            frame = make_frame(labels=labels)
            expected = [mapping.get(label, label) for label in labels]
            frame.relabel(mapping)
            self.assertEqual(frame.labels, expected)
            assert_frame_consistent(frame)

        @settings(max_examples=250, deadline=None)
        @given(
            st.lists(
                st.lists(st.sampled_from(LABELS), min_size=0, max_size=12),
                min_size=0,
                max_size=12,
            )
        )
        def test_collection_reconciles_conflicting_ids(self, frame_labels):
            frames = []
            for frame_index, labels in enumerate(frame_labels):
                boxes = [
                    outputs.Box(
                        [index, index, index + 1, index + 1],
                        label=label,
                        class_id=index % 3,
                    )
                    for index, label in enumerate(labels)
                ]
                frames.append(outputs.Detection(
                    boxes,
                    source_id="generated",
                    frame_index=frame_index,
                    image_size=(100, 100),
                ))
            results = outputs.Detections(frames)
            assert_collection_consistent(results)

        @settings(max_examples=200, deadline=None)
        @given(
            st.lists(
                st.lists(st.sampled_from(LABELS), min_size=0, max_size=8),
                min_size=1,
                max_size=8,
            ),
            st.dictionaries(st.sampled_from(LABELS), st.sampled_from(LABELS), max_size=7),
            st.sets(st.sampled_from(LABELS)),
        )
        def test_collection_relabel_and_drop_match_oracle(
            self, frame_labels, mapping, blocked
        ):
            results = outputs.Detections([
                make_frame("oracle", index, labels)
                for index, labels in enumerate(frame_labels)
            ])
            expected = [
                [mapping.get(label, label) for label in labels]
                for labels in frame_labels
            ]
            results.relabel(mapping)
            results.drop(blocked)
            expected = [
                [label for label in labels if label not in blocked]
                for labels in expected
            ]
            self.assertEqual([frame.labels for frame in results], expected)
            assert_collection_consistent(results)

        @settings(max_examples=150, deadline=None)
        @given(
            st.lists(st.integers(-5, 4), min_size=0, max_size=20),
            st.sampled_from(LABELS),
        )
        def test_repredict_normalizes_indices_and_preserves_identities(self, indices, label):
            sources = [object() for _ in range(5)]
            results = outputs.Detections([
                outputs.Detection(
                    [outputs.Box([0, 0, 2, 2], label="car")],
                    source_id="video",
                    frame_index=index,
                    image_size=(10, 10),
                    source=source,
                )
                for index, source in enumerate(sources)
            ])
            before = [frame.to_dict() for frame in results]

            class GeneratedModel:
                def predict(self, selected_sources, stream=False, **options):
                    assert stream is False
                    assert options == {"threshold": 0.25}
                    return outputs.Detections([
                        outputs.Detection(
                            [outputs.Box([1, 1, 3, 3], label=label)],
                            source_id=f"prediction-{index}",
                            image_size=(20, 20),
                        )
                        for index, _source in enumerate(selected_sources)
                    ])

            results.repredict(GeneratedModel(), index=indices, threshold=0.25)
            selected = []
            for index in indices:
                normalized = index + len(results) if index < 0 else index
                if normalized not in selected:
                    selected.append(normalized)
            for index, frame in enumerate(results):
                self.assertEqual(frame.identity, ("video", index))
                self.assertEqual(frame.image_size, (10, 10))
                if index in selected:
                    self.assertEqual(frame.labels, [label])
                    self.assertIs(frame._source, sources[index])
                else:
                    self.assertEqual(frame.to_dict(), before[index])
            assert_collection_consistent(results)

        @settings(max_examples=150, deadline=None)
        @given(st.sampled_from(LABELS), st.integers(0, 20))
        def test_failed_collection_map_is_atomic(self, label, failure_index):
            results = outputs.Detections([
                make_frame("atomic", index, ("car", label))
                for index in range(6)
            ])
            before = results.to_dict()
            calls = 0

            def fail(frame):
                nonlocal calls
                frame.drop("car")
                if calls == failure_index % len(results):
                    raise RuntimeError("generated failure")
                calls += 1
                return frame

            with self.assertRaises(RuntimeError):
                results.map(fail)
            self.assertEqual(results.to_dict(), before)
            assert_collection_consistent(results)


if RuleBasedStateMachine is not None:
    label_strategy = st.sampled_from(LABELS)
    coordinate_strategy = st.tuples(
        st.integers(-1_000, 1_000),
        st.integers(-1_000, 1_000),
        st.integers(1, 300),
        st.integers(1, 300),
    )

    def coordinates(values):
        x, y, width, height = values
        return [x, y, x + width, y + height]


    class DetectionStateMachine(RuleBasedStateMachine):
        """Generate interacting box operations on one frame."""

        def __init__(self):
            super().__init__()
            self.frame = make_frame("state", 0)

        @rule(values=coordinate_strategy, label=label_strategy)
        def add(self, values, label):
            self.frame.add_box(coordinates(values), label=label)

        @rule(index=st.integers(-100_000, 100_000), values=coordinate_strategy,
              label=label_strategy)
        def insert(self, index, values, label):
            self.frame.insert(index, outputs.Box(coordinates(values), label=label))

        @rule(values=coordinate_strategy, first=label_strategy, second=label_strategy)
        def extend(self, values, first, second):
            candidate = coordinates(values)
            self.frame.extend([
                outputs.Box(candidate, label=first),
                outputs.Box(candidate, label=second),
            ])

        @rule(label=label_strategy)
        def drop(self, label):
            self.frame.drop(label)

        @rule(old=label_strategy, new=label_strategy)
        def relabel(self, old, new):
            self.frame.relabel({old: new})

        @rule(index=st.integers(0, 100_000))
        def remove(self, index):
            if self.frame:
                self.frame.remove_box(self.frame.uids[index % len(self.frame)])

        @rule(index=st.integers(-100_000, 100_000))
        def pop(self, index):
            if self.frame:
                self.frame.pop(index % len(self.frame))

        @rule(index=st.integers(0, 100_000), values=coordinate_strategy,
              label=label_strategy)
        def replace(self, index, values, label):
            if self.frame:
                self.frame.replace(
                    index % len(self.frame),
                    outputs.Box(coordinates(values), label=label),
                )

        @rule(index=st.integers(0, 100_000), score=st.floats(0, 1, allow_nan=False))
        def mutate_score(self, index, score):
            if self.frame:
                self.frame[index % len(self.frame)].score = score

        @rule(dx=st.integers(-50, 50), dy=st.integers(-50, 50))
        def map_translate(self, dx, dy):
            self.frame.map(lambda box: box.translate(dx, dy))

        @rule(allowed=st.sets(label_strategy))
        def filter_labels(self, allowed):
            self.frame.filter(lambda box: box.label in allowed)

        @rule()
        def callback_failure_is_atomic(self):
            before = self.frame.to_dict()

            def fail(box):
                box.score = 0
                raise RuntimeError("generated failure")

            try:
                self.frame.map(fail)
            except RuntimeError:
                pass
            assert self.frame.to_dict() == before

        @invariant()
        def frame_invariants_hold(self):
            assert_frame_consistent(self.frame)


    TestDetectionStateMachine = DetectionStateMachine.TestCase
    TestDetectionStateMachine.settings = settings(
        max_examples=75,
        stateful_step_count=75,
        deadline=None,
    )


    class DetectionsStateMachine(RuleBasedStateMachine):
        """Generate frame and collection operations with registry conflicts."""

        def __init__(self):
            super().__init__()
            self.results = outputs.Detections(
                [make_frame("video", index) for index in range(3)]
            )
            self.next_source = 0

        @rule(frame_index=st.integers(0, 100_000), values=coordinate_strategy,
              label=label_strategy)
        def add_to_child(self, frame_index, values, label):
            self.results[frame_index % len(self.results)].add_box(
                coordinates(values), label=label
            )

        @rule(frame_index=st.integers(0, 100_000), label=label_strategy)
        def drop_from_child(self, frame_index, label):
            self.results[frame_index % len(self.results)].drop(label)

        @rule(old=label_strategy, new=label_strategy)
        def relabel_all(self, old, new):
            self.results.relabel({old: new})

        @rule(label=label_strategy)
        def drop_all(self, label):
            self.results.drop(label)

        @precondition(lambda self: len(self.results) < 15)
        @rule(label=label_strategy, incoming_id=st.integers(0, 5))
        def append_conflicting_frame(self, label, incoming_id):
            item = outputs.Detection(
                [outputs.Box([0, 0, 2, 2], label=label, class_id=incoming_id)],
                source_id=f"generated-{self.next_source}",
                image_size=(10, 10),
            )
            self.results.append(item)
            self.next_source += 1

        @precondition(lambda self: len(self.results) < 15)
        @rule(index=st.integers(-100_000, 100_000), label=label_strategy)
        def insert_frame(self, index, label):
            item = make_frame(f"inserted-{self.next_source}", 0, (label,))
            self.results.insert(index, item)
            self.next_source += 1

        @rule(index=st.integers(0, 100_000), label=label_strategy)
        def replace_frame(self, index, label):
            position = index % len(self.results)
            identity = self.results[position].identity
            replacement = make_frame("ignored", 999, (label,))
            self.results.replace(position, replacement)
            assert self.results[position].identity == identity

        @rule(index=st.integers(0, 100_000), score=st.floats(0, 1, allow_nan=False))
        def mutate_child_score(self, index, score):
            frame = self.results[index % len(self.results)]
            if frame:
                frame[index % len(frame)].score = score

        @rule(label=label_strategy)
        def map_children(self, label):
            def transform(frame):
                frame.drop(label)
                return frame

            self.results.map(transform)

        @rule()
        def compact_classes(self):
            identities = [frame.identity for frame in self.results]
            uids = [frame.uids for frame in self.results]
            self.results.compact()
            assert [frame.identity for frame in self.results] == identities
            assert [frame.uids for frame in self.results] == uids
            assert sorted(self.results.names) == list(range(len(self.results.names)))

        @precondition(lambda self: len(self.results) > 1)
        @rule(index=st.integers(0, 100_000))
        def remove_frame(self, index):
            self.results.pop(index % len(self.results))

        @rule(label=label_strategy)
        def copy_is_independent(self, label):
            before = self.results.to_dict()
            copied = self.results.copy()
            copied.drop(label)
            assert self.results.to_dict() == before

        @invariant()
        def collection_invariants_hold(self):
            assert_collection_consistent(self.results)

        @invariant()
        def history_sequence_is_contiguous(self):
            assert [item["sequence_number"] for item in self.results.history] == list(
                range(1, len(self.results.history) + 1)
            )

        @invariant()
        def next_class_id_is_above_active_ids(self):
            assert self.results._next_class_id > max(self.results.names, default=-1)


    TestDetectionsStateMachine = DetectionsStateMachine.TestCase
    TestDetectionsStateMachine.settings = settings(
        max_examples=75,
        stateful_step_count=75,
        deadline=None,
    )

else:
    @unittest.skip("Hypothesis is not installed")
    class TestDetectionStateMachine(unittest.TestCase):
        """Placeholder when the optional local test tool is unavailable."""

        def test_hypothesis_is_optional(self):
            pass


    @unittest.skip("Hypothesis is not installed")
    class TestDetectionsStateMachine(unittest.TestCase):
        """Placeholder when the optional local test tool is unavailable."""

        def test_hypothesis_is_optional(self):
            pass


if __name__ == "__main__":
    unittest.main()

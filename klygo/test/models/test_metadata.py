"""Behavioral tests for flat model metadata and parameter resolution."""

import importlib
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import PIL.Image

from klygo import Config, models
from klygo.models import utils
from klygo.models.detection.base import Detector
from klygo.models.load import _resolve


class RecordingDetector(Detector):
    def forward(self, images, prompt=None, **kwargs):
        self.runtime_metadata = self.metadata
        return [{"boxes": [], "scores": [], "labels": []} for _ in images]


class TestModelMetadata(unittest.TestCase):
    def setUp(self):
        self.flags = models.flags("model", "processor", "post")
        self.priority = models.priority(
            self.flags,
            model=("device",),
            processor=("size",),
            post=("threshold",),
        )
        self.details = models.details(
            name="Example detector",
            task="Object-Detection",
            backend="Custom",
            library="custom",
        )
        self.definition = models.metadata(
            details=self.details,
            flags=self.flags,
            priority=self.priority,
        )

    def test_metadata_declares_flags_without_initializing_parameters(self):
        self.assertNotIn("config", self.definition)
        self.assertIsNone(self.definition["class"])
        self.assertEqual(tuple(self.definition.flags), tuple(self.flags))
        self.assertNotIn("model", self.definition)
        self.assertNotIn("processor", self.definition)
        self.assertNotIn("post", self.definition)

    def test_metadata_rejects_loader_implementations(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'implementation'"):
            models.metadata(
                details=self.details,
                flags=self.flags,
                implementation=RecordingDetector,
            )

    def test_details_requires_core_fields_and_keeps_extra_information(self):
        details = models.details(
            name="  Example detector  ",
            task="Object-Detection",
            backend="Custom",
            library="custom",
            version="v2",
            num_params=None,
        )

        self.assertEqual(details.name, "Example detector")
        self.assertEqual(details.version, "v2")
        self.assertNotIn("num_params", details)

        with self.assertRaisesRegex(ValueError, "name must not be empty"):
            models.details(
                name=" ",
                task="Object-Detection",
                backend="Custom",
                library="custom",
            )

    def test_configure_resolves_priority_names_without_prefixes(self):
        result = models.configure(
            "weights.pt",
            metadata=self.definition,
            device="cuda",
            size=800,
            threshold=0.4,
        )

        self.assertEqual(result["model"]["device"], "cuda")
        self.assertEqual(result["processor"]["size"], 800)
        self.assertEqual(result["post"]["threshold"], 0.4)
        self.assertEqual(result.model.device, "cuda")
        self.assertEqual(result.processor.size, 800)

    def test_metadata_keeps_one_shape_with_one_two_or_three_components(self):
        only_flags = models.metadata(self.flags)
        with_priority = models.metadata(self.flags, self.priority)
        complete = models.metadata(self.flags, self.priority, self.details)

        for result in (only_flags, with_priority, complete):
            self.assertEqual(
                set(result),
                {"flags", "priority", "details", "class"},
            )
            self.assertIsNone(result["class"])

        self.assertIsNone(only_flags.priority)
        self.assertIsNone(only_flags.details)
        self.assertEqual(tuple(with_priority.priority.post), ("threshold",))
        self.assertIsNone(with_priority.details)
        self.assertEqual(complete.details.name, "Example detector")

    def test_configure_attaches_a_serializable_implementation(self):
        result = models.configure(
            "weights.pt",
            metadata=self.definition,
            implementation=RecordingDetector,
        )

        self.assertEqual(
            result["class"],
            "test.models.test_metadata.RecordingDetector",
        )
        self.assertIsInstance(models.load(result), RecordingDetector)

    def test_configure_builds_metadata_components_directly(self):
        result = models.configure(
            "weights.pt",
            flags=("encoder", "decode"),
            priority={
                "encoder": ("device",),
                "decode": ("threshold",),
            },
            details={
                "name": "Direct detector",
                "task": "Object-Detection",
                "backend": "Custom",
                "library": "custom",
            },
            implementation=RecordingDetector,
            device="cuda",
            threshold=0.4,
            encoder_compile=True,
        )

        self.assertEqual(tuple(result.flags), ("encoder", "decode"))
        self.assertEqual(result.details.name, "Direct detector")
        self.assertEqual(result.encoder.device, "cuda")
        self.assertTrue(result.encoder.compile)
        self.assertEqual(result.decode.threshold, 0.4)
        self.assertEqual(
            result["class"],
            "test.models.test_metadata.RecordingDetector",
        )

    def test_configure_uses_declared_flags(self):
        result = models.configure(
            "weights.pt",
            metadata=self.definition,
        )

        self.assertEqual(
            tuple(result.flags),
            ("model", "processor", "post"),
        )

        custom = models.configure(
            "weights.pt",
            flags=("model", "post"),
            model_device="cuda",
            post_threshold=0.5,
        )
        self.assertEqual(tuple(custom.flags), ("model", "post"))
        self.assertNotIn("processor", custom)

    def test_configure_replaces_init_in_the_public_api(self):
        self.assertTrue(callable(models.configure))
        self.assertFalse(hasattr(models, "conf"))
        self.assertTrue(callable(models.details))
        self.assertTrue(callable(models.flags))
        self.assertFalse(hasattr(models, "init"))
        self.assertFalse(hasattr(models, "profile"))

    def test_boolean_flags_control_predict_overrides_only(self):
        flags = models.flags(model=False, processor=True, post=True)
        definition = models.metadata(
            flags,
            models.priority(flags, model=("device",), post=("threshold",)),
            self.details,
        )
        configured = models.configure(
            "recording",
            metadata=definition,
            implementation=RecordingDetector,
            device="cuda",
            threshold=0.4,
        )

        self.assertIs(configured.flags.model, False)
        self.assertIs(configured.flags.processor, True)
        detector = RecordingDetector(configured)
        detector.predict(
            PIL.Image.new("RGB", (8, 8)),
            threshold=0.8,
            verbose=False,
        )
        self.assertEqual(detector.runtime_metadata.post.threshold, 0.8)

        with self.assertRaisesRegex(ValueError, "locked at predict time"):
            detector.predict(
                PIL.Image.new("RGB", (8, 8)),
                device="cpu",
                verbose=False,
            )

    def test_config_round_trips_metadata_and_conf_in_all_structured_formats(self):
        flags = models.flags(model=False, processor=True, post=True)
        definition = models.metadata(
            flags,
            models.priority(flags, post=("threshold",)),
            self.details,
        )
        configured = models.configure(
            "recording",
            metadata=definition,
            implementation=RecordingDetector,
            threshold=0.6,
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in ("json", "yaml", "toml"):
                with self.subTest(suffix=suffix):
                    definition_path = root / f"metadata.{suffix}"
                    definition_manager = Config.create_default(
                        definition_path,
                        metadata=definition,
                        verbose=False,
                    )
                    self.assertIsNone(definition_manager.to_dict()["class"])
                    self.assertIs(
                        definition_manager.to_dict()["flags"]["model"],
                        False,
                    )

                    configured_path = root / f"configured.{suffix}"
                    configured_manager = Config.create_default(
                        configured_path,
                        metadata=configured,
                        verbose=False,
                    )
                    loaded = models.load(configured_manager)
                    self.assertIsInstance(loaded, RecordingDetector)
                    self.assertIs(loaded.metadata.flags.model, False)
                    self.assertEqual(loaded.metadata.post.threshold, 0.6)

    def test_non_priority_names_require_a_group_prefix(self):
        result = models.configure(
            "weights.pt",
            metadata=self.definition,
            processor_do_resize=False,
        )
        self.assertIs(result["processor"]["do_resize"], False)

        with self.assertRaisesRegex(ValueError, "neither a priority parameter"):
            models.configure(
                "weights.pt",
                metadata=self.definition,
                do_resize=False,
            )

    def test_flags_reject_parameters_from_undeclared_groups_everywhere(self):
        definition = models.metadata(
            models.flags("processor"),
            {"processor": ("size",)},
        )
        configured = models.configure(
            "processor-only",
            metadata=definition,
            size=640,
        )
        self.assertEqual(configured.processor.size, 640)
        self.assertNotIn("model", configured)

        with self.assertRaisesRegex(ValueError, "Available groups: \\('processor',\\)"):
            models.configure(
                "processor-only",
                metadata=definition,
                model_device="cuda",
            )

        with self.assertRaisesRegex(ValueError, "Available groups: \\('processor',\\)"):
            models.configure(
                "processor-only",
                metadata=definition,
                model={"device": "cuda"},
            )

        detector = RecordingDetector(configured)
        with self.assertRaisesRegex(ValueError, "Available groups: \\('processor',\\)"):
            detector.predict(
                PIL.Image.new("RGB", (8, 8)),
                model_device="cuda",
                verbose=False,
            )

    def test_duplicate_parameter_destinations_raise(self):
        with self.assertRaisesRegex(ValueError, "provided more than once"):
            models.configure(
                "weights.pt",
                metadata=self.definition,
                threshold=0.4,
                post_threshold=0.5,
            )

    def test_priority_rejects_unknown_and_duplicate_groups(self):
        with self.assertRaisesRegex(ValueError, "not present in metadata"):
            models.priority(self.flags, tracker=("max_age",))

        with self.assertRaisesRegex(ValueError, "multiple groups"):
            models.priority(
                self.flags,
                model=("size",),
                processor=("size",),
            )

    def test_legacy_config_wrapper_is_flattened(self):
        legacy = {
            "config": {
                "model": {"device": "cpu"},
                "post": {"threshold": 0.25},
            },
            "priority": {"post": ("threshold",)},
        }

        result = models.configure(
            "weights.pt",
            metadata=legacy,
            threshold=0.5,
        )

        self.assertNotIn("config", result)
        self.assertEqual(result["post"]["threshold"], 0.5)

    def test_resolving_runtime_metadata_does_not_mutate_defaults(self):
        resolved = utils.resolve_metadata(self.definition, {"threshold": 0.9})

        self.assertEqual(resolved["post"]["threshold"], 0.9)
        self.assertNotIn("post", self.definition)

    def test_registry_uses_flat_metadata_and_model_specific_groups(self):
        grounding_dino = _resolve("grounding-dino-base")
        yolo = _resolve("yolo11n.pt")

        self.assertNotIn("config", grounding_dino)
        self.assertIn("processor", grounding_dino)
        self.assertNotIn("config", yolo)
        self.assertNotIn("processor", yolo)
        self.assertIn("threshold", yolo["priority"]["post"])
        self.assertEqual(grounding_dino["details"]["num_params"], "232M")
        self.assertEqual(
            tuple(grounding_dino["flags"]),
            ("model", "processor", "post"),
        )
        self.assertEqual(tuple(yolo["flags"]), ("model", "post"))

    def test_load_configures_registry_entries_through_the_canonical_pipeline(self):
        load_module = importlib.import_module("klygo.models.load")
        with patch.object(
            load_module,
            "_resolve_class",
            return_value=RecordingDetector,
        ):
            detector = models.load(
                "yolo11n.pt",
                device="cpu",
                threshold=0.4,
            )

        self.assertEqual(detector.model_id, "yolo11n.pt")
        self.assertEqual(detector.flags, ("model", "post"))
        self.assertEqual(detector.metadata.model.device, "cpu")
        self.assertEqual(detector.metadata.post.threshold, 0.4)
        self.assertEqual(detector.details.library, "ultralytics")

    def test_detector_resolves_metadata_once_before_forward(self):
        detector = RecordingDetector(
            models.configure("recording", metadata=self.definition)
        )

        detector.predict(
            PIL.Image.new("RGB", (8, 8)),
            threshold=0.75,
            verbose=False,
        )

        self.assertEqual(detector.runtime_metadata["post"]["threshold"], 0.75)
        self.assertEqual(detector.metadata.post, {})
        self.assertTrue(detector.has_flag("processor"))
        self.assertFalse(detector.has_flag("tracker"))
        self.assertFalse(hasattr(detector, "parse_config"))
        self.assertFalse(hasattr(detector, "split_kwargs"))
        self.assertEqual(detector.flags, ("model", "processor", "post"))

    def test_detector_call_is_an_exact_predict_alias(self):
        detector = RecordingDetector(
            models.configure("recording", metadata=self.definition)
        )
        image = PIL.Image.new("RGB", (8, 8))

        result = detector(image, threshold=0.72, verbose=False)

        self.assertEqual(result.count, 1)
        self.assertEqual(detector.runtime_metadata.post.threshold, 0.72)

        with self.assertRaisesRegex(ValueError, "neither a priority parameter"):
            detector(image, unknown=1, verbose=False)

    def test_create_default_preserves_the_same_metadata(self):
        initialized = models.configure(
            "recording",
            metadata=self.definition,
            implementation=RecordingDetector,
            device="cuda",
            threshold=0.6,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            manager = Config.create_default(
                path,
                metadata=initialized,
                verbose=False,
            )

            self.assertNotIn("default", manager.to_dict())
            self.assertEqual(manager.to_dict()["post"]["threshold"], 0.6)

            loaded = Config(path).read(verbose=False)
            self.assertEqual(loaded.model.device, "cuda")
            self.assertEqual(loaded.post.threshold, 0.6)

            from_manager = models.load(manager)
            from_path = models.load(path)
            self.assertIsInstance(from_manager, RecordingDetector)
            self.assertIsInstance(from_path, RecordingDetector)
            self.assertEqual(from_manager.metadata.model.device, "cuda")
            self.assertEqual(from_path.metadata.post.threshold, 0.6)

    def test_saved_model_folder_round_trips_through_load(self):
        configured = models.configure(
            "recording",
            metadata=self.definition,
            device="cpu",
            threshold=0.55,
        )
        detector = RecordingDetector(configured)

        with TemporaryDirectory() as directory:
            output_dir = Path(directory) / "exported"
            detector.save(output_dir)
            restored = models.load(output_dir)

        self.assertEqual(type(restored).__name__, "RecordingDetector")
        self.assertIsInstance(restored, Detector)
        self.assertEqual(restored.flags, detector.flags)
        self.assertEqual(restored.metadata.model.device, "cpu")
        self.assertEqual(restored.metadata.post.threshold, 0.55)
        self.assertEqual(
            restored.metadata["class"],
            "test.models.test_metadata.RecordingDetector",
        )

    def test_create_default_rejects_mixed_default_and_metadata_modes(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                Config.create_default(
                    Path(directory) / "invalid.yaml",
                    default_data={"default": {"root": "."}},
                    metadata=self.definition,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()

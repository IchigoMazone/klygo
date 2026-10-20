"""Contracts between klygo.config and the public model metadata builders."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from klygo import Config, config, models


class TestModelMetadataConfig(unittest.TestCase):
    def setUp(self):
        self.flags = models.flags(
            model=False,
            processor=True,
            post=True,
        )
        self.schema = models.metadata(
            flags=self.flags,
            priority=models.priority(
                self.flags,
                model=("device",),
                processor=("size",),
                post=("threshold", "iou"),
            ),
            details=models.details(
                name="Config test detector",
                task="Object-Detection",
                backend="Custom",
                library="custom",
            ),
        )
        self.configured = models.configure(
            "weights.pt",
            metadata=self.schema,
            implementation="example.detectors.CustomDetector",
            device="cuda",
            size=640,
            threshold=0.4,
            post_iou=0.7,
        )

    def test_functional_io_round_trips_both_metadata_stages(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for suffix in ("json", "yaml", "toml"):
                for name, value in (
                    ("schema", self.schema),
                    ("configured", self.configured),
                ):
                    with self.subTest(suffix=suffix, stage=name):
                        path = root / f"{name}.{suffix}"
                        config.save(path, value, verbose=False)
                        restored = config.load(path, verbose=False)
                        self.assertEqual(restored.to_dict(), value.to_dict())

    def test_create_default_writes_exact_metadata_and_returns_loaded_manager(self):
        with TemporaryDirectory() as directory:
            for suffix in ("json", "yaml", "toml"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"model.{suffix}"
                    manager = Config.create_default(
                        path,
                        metadata=self.schema,
                        verbose=False,
                    )
                    self.assertEqual(manager.to_dict(), self.schema.to_dict())
                    self.assertNotIn("default", manager.to_dict())
                    self.assertIs(manager.get("flags.model"), False)
                    self.assertIs(manager.get("flags.processor"), True)
                    self.assertIsNone(manager.get("class"))

    def test_loaded_schema_can_be_passed_back_to_models_configure(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "schema.toml"
            config.save(path, self.schema, verbose=False)
            restored = config.load(path, verbose=False)

            configured = models.configure(
                "new-weights.pt",
                metadata=restored,
                implementation="example.detectors.CustomDetector",
                device="cpu",
                size=800,
                threshold=0.55,
            )

            self.assertEqual(configured.model_id, "new-weights.pt")
            self.assertEqual(configured.model.device, "cpu")
            self.assertEqual(configured.processor.size, 800)
            self.assertEqual(configured.post.threshold, 0.55)
            self.assertIs(configured.flags.model, False)

    def test_model_metadata_inputs_are_not_mutated(self):
        schema_before = self.schema.to_dict()
        configured_before = self.configured.to_dict()

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config.save(root / "schema.toml", self.schema, verbose=False)
            Config.create_default(
                root / "configured.yaml",
                metadata=self.configured,
                verbose=False,
            )

        self.assertEqual(self.schema.to_dict(), schema_before)
        self.assertEqual(self.configured.to_dict(), configured_before)

    def test_default_data_and_metadata_modes_remain_exclusive(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                Config.create_default(
                    Path(directory) / "invalid.json",
                    default_data={"default": {"root": "."}},
                    metadata=self.schema,
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()

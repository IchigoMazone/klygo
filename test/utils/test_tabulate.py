"""Tests for klygo.utils.tabulate."""

import io
import unittest

from rich import box
from rich.console import Console
from rich.table import Table

from klygo import utils


class TabulateTestCase(unittest.TestCase):
    def test_returns_styled_rich_table_when_show_is_false(self) -> None:
        table = utils.tabulate(
            {
                "classes": ["apple", "banana"],
                "empty_list": [],
                "splits": {"train": 3, "val": 1},
                "empty_map": {},
                "none_val": None,
                "nc": 2,
            },
            title="Dataset Summary",
            key_header="Field",
            value_header="Details",
            show=False,
        )
        self.assertIsInstance(table, Table)
        self.assertEqual(table.title, "Dataset Summary")
        self.assertEqual(table.box, box.HORIZONTALS)
        self.assertTrue(table.show_lines)
        self.assertEqual([col.header for col in table.columns], ["Field", "Details"])

        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=100).print(table)
        rendered = buf.getvalue()
        self.assertIn("Dataset Summary", rendered)
        self.assertIn("apple, banana", rendered)
        self.assertIn("splits.train", rendered)
        self.assertIn("splits.val", rendered)
        self.assertIn("_", rendered)

    def test_sequence_of_mappings_renders_multi_column_with_numeric_right_align(self) -> None:
        rows = [
            {"Dessert": "Frozen yogurt", "Calories": 159, "Fat (g)": 6.0},
            {"Dessert": "Cupcake", "Calories": 305, "Fat (g)": 3.7},
        ]
        table = utils.tabulate(rows, show=False)
        self.assertEqual([col.header for col in table.columns], ["Dessert", "Calories", "Fat (g)"])
        self.assertEqual([col.justify for col in table.columns], ["left", "right", "right"])

    def test_custom_table_options_and_kwargs(self) -> None:
        table = utils.tabulate(
            {"a": 1, "b": 2},
            show=False,
            show_header=False,
            show_lines=False,
            padding=(0, 1),
            box_style=box.ROUNDED,
            border_style="cyan",
            header_style="bold",
            title_style="bold",
            caption="Footer note",
            expand=True,
        )
        self.assertFalse(table.show_header)
        self.assertFalse(table.show_lines)
        self.assertEqual(table.padding, (0, 1, 0, 1))
        self.assertEqual(table.box, box.ROUNDED)
        self.assertEqual(table.border_style, "cyan")
        self.assertEqual(table.caption, "Footer note")
        self.assertTrue(table.expand)

    def test_nested_four_level_mapping_flattens_with_dot_keys(self) -> None:
        table = utils.tabulate(
            {
                "splits": {"train": 10, "val": 5},
                "empty_nested": {},
                "config": {"model": {"backbone": {"resnet": {"layers": 50}}}},
            },
            show=False,
        )
        buf = io.StringIO()
        Console(file=buf, force_terminal=False, width=100).print(table)
        rendered = buf.getvalue()
        self.assertIn("splits.train", rendered)
        self.assertIn("splits.val", rendered)
        self.assertIn("empty_nested", rendered)
        self.assertIn("_", rendered)
        self.assertIn("config.model.backbone.resnet.layers", rendered)
        self.assertIn("50", rendered)

    def test_show_true_prints_and_returns_none(self) -> None:
        result = utils.tabulate({"status": "ready", "count": 5}, title="Status")
        self.assertIsNone(result)

    def test_invalid_arguments_raise_expected_errors(self) -> None:
        with self.assertRaises(TypeError):
            utils.tabulate([("a", 1)])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            utils.tabulate([])
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, title=123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, key_header=123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, value_header=123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, show="yes")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, show_header="yes")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, padding=(1, 2, 3))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, box_style="ROUNDED")
        with self.assertRaises(TypeError):
            utils.tabulate({"a": 1}, orient=123)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            utils.tabulate({"a": 1}, orient="diagonal")

        t_horiz = utils.tabulate({"gpu": "L4", "vram": "24GB"}, orient="horizontal", show=False)
        self.assertIsNotNone(t_horiz)
        self.assertEqual([c.header for c in t_horiz.columns], ["gpu", "vram"])


if __name__ == "__main__":
    unittest.main()

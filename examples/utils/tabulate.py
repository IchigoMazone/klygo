"""Executable examples for klygo.utils.tabulate."""

from rich import box

from klygo import utils


summary = {
    "dataset": "coco15",
    "classes": ["person", "bicycle", "car"],
    "splits": {"train": 10, "val": 5},
    "notes": None,
}
utils.tabulate(summary, title="Dataset Summary")

rows = [
    {"Dessert (100g serving)": "Frozen yogurt", "Calories": 159, "Fat (g)": 6.0, "Carbs (g)": 24},
    {"Dessert (100g serving)": "Ice cream sandwich", "Calories": 237, "Fat (g)": 9.0, "Carbs (g)": 37},
    {"Dessert (100g serving)": "Eclair", "Calories": 262, "Fat (g)": 16.0, "Carbs (g)": 24},
]
table = utils.tabulate(
    rows,
    box_style=box.ROUNDED,
    show_lines=False,
    padding=(0, 1),
    caption="Nutrition facts",
    show=False,
)
assert len(table.columns) == 4
assert table.caption == "Nutrition facts"

try:
    utils.tabulate("not-a-mapping")  # type: ignore[arg-type]
except TypeError:
    print("non-mapping inputs are rejected")

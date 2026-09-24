"""Executable example for klygo.files.list_entries."""

from klygo import files

for entry in files.list_entries("klygo", pattern="*.py"):
    print(entry)


# Additional cases
# Recursive listing includes matching entries in descendants.
recursive = files.list_entries("klygo", pattern="*.py", recursive=True)
assert all(item.suffix == ".py" for item in recursive)


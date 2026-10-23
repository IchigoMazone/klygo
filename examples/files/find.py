"""Executable example for klygo.files.find."""

from klygo import files

for source in files.find("klygo", pattern="*.py"):
    print(source)


# Additional cases
# Disable recursion when only direct children are wanted.
direct = files.find("klygo", pattern="*.py", recursive=False)
assert all(item.parent.name == "klygo" for item in direct)


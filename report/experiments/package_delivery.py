"""Package the active LaTeX project and compact reproducibility evidence."""
from pathlib import Path
import subprocess
import sys
import zipfile
import re

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"
EVIDENCE = ROOT / "klygo/artifacts/report"


def archive(destination, paths, root):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as bundle:
        for path in sorted(set(paths)):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as bundle:
        assert bundle.testzip() is None
    print(f"{destination.name}: {destination.stat().st_size:,} bytes")


if __name__ == "__main__":
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    portable = [line for line in frozen.splitlines() if not line.startswith(("-e ", "#", "klygo @"))]
    (REPORT / "experiments/requirements.lock.txt").write_text("\n".join(portable) + "\n", encoding="utf-8")
    source = [REPORT / name for name in ("main.tex", "main.bbl", "klygo_references.bib", "build.ps1", "README_KLYGO.md")]
    source += list((REPORT / "chapters").glob("klygo*.tex"))
    source += list((REPORT / "generated").glob("*.tex"))
    active_tex = list((REPORT / "chapters").glob("klygo*.tex")) + list((REPORT / "generated").glob("*.tex"))
    for tex in active_tex:
        for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text(encoding="utf-8")):
            source.append(REPORT / "images" / name)
    source += [REPORT / "images/logo.png"]
    archive(REPORT / "klygo_latex.zip", source, REPORT)
    proof = list((EVIDENCE / "main").rglob("*.json")) + list((EVIDENCE / "main").rglob("*.txt"))
    proof += list((EVIDENCE / "logs").glob("*")) + list((EVIDENCE / "demo").glob("*"))
    proof += [EVIDENCE / "data/manifest.json", EVIDENCE / "data/instances_sample.json"]
    proof += list((REPORT / "experiments").glob("*.py")) + [REPORT / "experiments/requirements.lock.txt"]
    proof += list((REPORT / "planning").glob("*.md"))
    extension = EVIDENCE / "extension"
    proof += [p for p in extension.rglob("*") if p.is_file() and p.suffix in {".json", ".yaml", ".csv", ".log"}
              and "ultralytics" not in p.parts and "matplotlib" not in p.parts]
    proof += [ROOT / "tmp/klygo-c4388f1.zip"]
    proof += list(REPORT.glob("*extension*.log")) + [REPORT / "execution_native_probe.log"]
    proof += [REPORT / "scheduling_study.log"]
    archive(REPORT / "klygo_evidence.zip", proof, ROOT)

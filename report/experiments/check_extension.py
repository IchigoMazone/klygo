"""Run the pinned snapshot's existing tests and audit; do not modify library code."""
from extension_common import *
import contextlib
import unittest
import subprocess
import time
import hashlib

if __name__ == "__main__":
    destination = OUT / "checks"
    destination.mkdir(parents=True, exist_ok=True)
    manifest=load(DATA / "manifest.json")
    annotations=load(DATA / "instances_sample.json")
    image_names={item["id"]:item["file_name"] for item in annotations["images"]}
    for image_id, expected in manifest["image_sha256"].items():
        actual=hashlib.sha256((DATA / "images" / image_names[int(image_id)]).read_bytes()).hexdigest()
        assert actual==expected, image_id
    save(destination / "provenance.json", {"revision":REVISION,"verified_images":len(manifest["image_sha256"]),
         "source_archive_sha256":hashlib.sha256((ROOT / "tmp/klygo-c4388f1.zip").read_bytes()).hexdigest(),
         "script_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "report/experiments").glob("*.py")}})
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(SOURCE)
    with (destination / "unittest.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            suite = unittest.defaultTestLoader.discover("test", pattern="test*.py", top_level_dir=".")
            # Match unittest's CLI warning policy; imported frameworks may globally ignore UserWarning.
            result = unittest.TextTestRunner(stream=log, verbosity=2, warnings="default").run(suite)
    save(destination / "summary.json", {"revision": REVISION,"run":result.testsRun,
         "failures":[[case.id(),reason] for case,reason in result.failures],
         "errors":[[case.id(),reason] for case,reason in result.errors],
         "skipped":[[case.id(),reason] for case,reason in result.skipped],
         "success":result.wasSuccessful()})
    script = SOURCE / ".agents/skills/klygo-module-guides/scripts/audit_public_api.py"
    with (destination / "audit.log").open("w",encoding="utf-8") as log:
        audit = subprocess.run([sys.executable,"-X","utf8",str(script),"--all"],stdout=log,stderr=subprocess.STDOUT)
    save(destination / "audit.json",{"returncode":audit.returncode,"revision":REVISION})
    print("Tests",result.testsRun,"errors",len(result.errors),"failures",len(result.failures),"skips",len(result.skipped),"audit",audit.returncode)

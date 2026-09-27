"""Wait for per-country outputs, then merge, validate, and package."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"
NEEDED = [OUTD / f"matching_results_{c}.tsv" for c in ("france", "us", "india")]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("watching for per-country outputs...")
t0 = time.time()
while True:
    if all(p.exists() and p.stat().st_size > 1000 for p in NEEDED):
        break
    if time.time() - t0 > 4 * 3600:
        log("TIMEOUT waiting for outputs")
        sys.exit(2)
    time.sleep(30)
log("all three country outputs present")
time.sleep(20)  # let the writers finish their final flush/prune

for attempt in range(3):
    r = subprocess.run([sys.executable, str(V4 / "merge_validate.py")],
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode == 0:
        log("VALIDATION PASS")
        break
    log(f"validation failed (attempt {attempt+1}); retrying in 60s")
    time.sleep(60)
else:
    sys.exit(3)

pkg = subprocess.run([sys.executable,
                      str(ROOT / "code" / "business_entity_resolution" / "build_submission.py"),
                      "--team-name", "Gemz"],
                     capture_output=True, text=True)
print(pkg.stdout[-1000:])
log("PACKAGE DONE")

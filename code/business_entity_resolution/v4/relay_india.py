"""Relay: when France completes, launch the India v5 test run."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("waiting for france completion...")
while True:
    p = OUTD / "matching_results_france.tsv"
    if p.exists():
        n = sum(1 for _ in open(p, encoding="utf-8")) - 1
        if n >= 259452:
            break
    time.sleep(45)
log("france complete - launching india v5 test")
r = subprocess.run(
    [sys.executable, "code/business_entity_resolution/v4/run_test.py",
     "india", "v5_config.json"], capture_output=True, text=True)
print((r.stdout or "")[-1500:])
if r.returncode != 0:
    print((r.stderr or "")[-800:])
log(f"india test exit: {r.returncode}")

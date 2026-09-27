"""Wait for v5 India completion, then merge all-v5 files and validate."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"
INDIA_ROWS = 809986


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("waiting for v5 india completion...")
t0 = time.time()
p = OUTD / "matching_results_india.tsv"
while True:
    n = (sum(1 for _ in open(p, encoding="utf-8")) - 1) if p.exists() else -1
    if n >= INDIA_ROWS:
        break
    if time.time() - t0 > 3 * 3600:
        log("TIMEOUT")
        sys.exit(2)
    time.sleep(60)
    print(f"  [{time.strftime('%H:%M:%S')}] india rows: {n:,}", flush=True)

time.sleep(60)  # prune rewrite grace
log("india complete - merging all-v5 files + validating")
r = subprocess.run([sys.executable, str(V4 / "merge_validate.py")],
                   capture_output=True, text=True)
print((r.stdout or "")[-2500:])
log(f"merge_validate exit: {r.returncode}")
sys.exit(r.returncode)

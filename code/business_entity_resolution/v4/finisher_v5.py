"""Wait for complete per-country v5 outputs (row-count verified), merge, validate."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"
EXPECT = {"france": 259452, "us": 663106, "india": 809986}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rows_done(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    return n >= expected


log("watching for complete per-country outputs...")
t0 = time.time()
while True:
    done = {c: rows_done(OUTD / f"matching_results_{c}.tsv", e) for c, e in EXPECT.items()}
    if all(done.values()):
        break
    if time.time() - t0 > 3 * 3600:
        log(f"TIMEOUT; state: {done}")
        sys.exit(2)
    time.sleep(45)
    print(f"  [{time.strftime('%H:%M:%S')}] " +
          " | ".join(f"{c}: {'DONE' if d else 'pending'}" for c, d in done.items()), flush=True)

time.sleep(30)  # let writers finish prune/replace
log("all three complete")

r = subprocess.run([sys.executable, str(V4 / "merge_validate.py")],
                   capture_output=True, text=True)
print((r.stdout or "")[-2500:])
log(f"merge_validate exit: {r.returncode}")
sys.exit(r.returncode)

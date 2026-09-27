"""Wait for france+us completion, then run the hybrid merge (v4 india rows)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
V4 = ROOT / "code" / "business_entity_resolution" / "v4"
EXPECT = {"france": 259452, "us": 663106}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("watching for france+us completion...")
t0 = time.time()
while True:
    done = {}
    for c, e in EXPECT.items():
        p = OUTD / f"matching_results_{c}.tsv"
        n = (sum(1 for _ in open(p, encoding="utf-8")) - 1) if p.exists() else -1
        done[c] = n >= e
    if all(done.values()):
        break
    if time.time() - t0 > 3 * 3600:
        log(f"TIMEOUT: {done}")
        sys.exit(2)
    time.sleep(60)
    print(f"  [{time.strftime('%H:%M:%S')}] " +
          " | ".join(f"{c}: {'DONE' if d else 'pending'}" for c, d in done.items()), flush=True)

time.sleep(30)
log("france+us complete - running hybrid merge + validate")
r = subprocess.run([sys.executable, str(V4 / "merge_hybrid.py")],
                   capture_output=True, text=True)
print((r.stdout or "")[-2500:])
if r.returncode != 0:
    print((r.stderr or "")[-800:])
log(f"merge_hybrid exit: {r.returncode}")

"""Phase D parallel: staggered per-country inference (memory-safe overlap),
then wait for all three. Launch AFTER best rules exist.
Usage: python phase_d_parallel.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCR8 = ROOT / "scratch" / "v8"
ORDER = ("us", "india", "france")
STAGGER = (0, 1200, 2400)  # seconds between launches (build peaks ~20 min)


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main():
    procs = {}
    t0 = time.time()
    for ctry, delay in zip(ORDER, STAGGER):
        while time.time() - t0 < delay:
            time.sleep(15)
        p = subprocess.Popen(
            [sys.executable, "scratch/v8/run_test_v8.py", ctry],
            stdout=open(SCR8 / f"phase_d_{ctry}.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT)
        procs[ctry] = p
        log(f"{ctry} launched pid={p.pid}")
        try:
            hp = __import__("psutil").Process(p.pid)
            hp.nice(20)
        except Exception:
            pass
    rc = {}
    for ctry, p in procs.items():
        rc[ctry] = p.wait()
        log(f"{ctry} finished rc={rc[ctry]}")
    if any(rc.values()):
        log(f"PHASE D FAILURES: {rc}")
        sys.exit(2)
    log("PHASE D COMPLETE")


if __name__ == "__main__":
    main()

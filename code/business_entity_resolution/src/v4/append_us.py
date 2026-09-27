"""Append US rows when its run completes, then LF/subset checks + validator."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
US_ROWS = 663106


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


log("waiting for US completion...")
while True:
    p = OUTD / "matching_results_us.tsv"
    n = (sum(1 for _ in open(p, encoding="utf-8")) - 1) if p.exists() else -1
    if n >= US_ROWS:
        break
    time.sleep(30)
log("US rows complete; waiting 90s for 1-to-1 prune rewrite...")
time.sleep(90)

n = 0
with open(OUTD / "matching_results.tsv", "a", encoding="utf-8", newline="\n") as fm, \
     open(OUTD / "candidate_pairs.tsv", "a", encoding="utf-8", newline="\n") as fc, \
     open(OUTD / "matching_results_us.tsv", encoding="utf-8") as a, \
     open(OUTD / "candidate_pairs_us.tsv", encoding="utf-8") as b:
    next(a)
    next(b)
    for la in a:
        lb = b.readline()
        fm.write(la if la.endswith("\n") else la + "\n")
        fc.write(lb if lb.endswith("\n") else lb + "\n")
        n += 1
log(f"appended {n:,} US rows")

# LF purity
for p in (OUTD / "matching_results.tsv", OUTD / "candidate_pairs.tsv"):
    cr = 0
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            cr += chunk.count(b"\r")
    assert cr == 0, f"{p.name}: {cr} CR bytes"
log("LF purity OK")

# subset integrity
bad = 0
with open(OUTD / "matching_results.tsv", encoding="utf-8") as fm, \
     open(OUTD / "candidate_pairs.tsv", encoding="utf-8") as fc:
    next(fm)
    next(fc)
    for lm in fm:
        lc = fc.readline()
        pm = lm.rstrip("\n").split("\t")
        pc = lc.rstrip("\n").split("\t")
        ms = set(pm[1].split(",")) if len(pm) > 1 and pm[1] else set()
        cs = set(pc[1].split(",")) if len(pc) > 1 and pc[1] else set()
        if not ms.issubset(cs):
            bad += 1
assert bad == 0, f"{bad} subset violations"
log("subset integrity OK")

r = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                    "--matching", str(OUTD / "matching_results.tsv"),
                    "--candidate", str(OUTD / "candidate_pairs.tsv"),
                    "--test-dir", str(ROOT / "dataset" / "test")],
                   capture_output=True, text=True)
tail = (r.stdout or r.stderr).strip().splitlines()
log("validator: " + (tail[-1] if tail else "no output"))
log(f"validator exit: {r.returncode}")
sys.exit(r.returncode)

"""Merge per-country outputs into final submission files and validate.

Usage: python merge_validate.py
Combines matching_results_{france,us,india}.tsv (already 1-to-1 pruned)
into output/matching_results.tsv, same for candidate pairs, then checks
LF purity, subset integrity, and runs the official validator.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ctries = ("france", "us", "india")
    m_final = OUTD / "matching_results.tsv"
    c_final = OUTD / "candidate_pairs.tsv"

    n_rows = 0
    with open(m_final, "w", encoding="utf-8", newline="\n") as fm, \
         open(c_final, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for ctry in ctries:
            m_part = OUTD / f"matching_results_{ctry}.tsv"
            c_part = OUTD / f"candidate_pairs_{ctry}.tsv"
            with open(m_part, encoding="utf-8") as a, \
                 open(c_part, encoding="utf-8") as b:
                next(a)
                next(b)
                for la, lb in zip(a, b):
                    assert la.split("\t", 1)[0] == lb.split("\t", 1)[0], "row mismatch"
                    fm.write(la if la.endswith("\n") else la + "\n")
                    fc.write(lb if lb.endswith("\n") else lb + "\n")
                    n_rows += 1
    log(f"merged {n_rows:,} rows from {len(ctries)} countries")

    # LF purity
    for p in (m_final, c_final):
        n_cr = 0
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4 << 20), b""):
                n_cr += chunk.count(b"\r")
        assert n_cr == 0, f"{p.name} contains {n_cr} CR bytes"
    log("LF purity verified (0 CR bytes)")

    # subset integrity (matches ⊆ candidates)
    bad = 0
    with open(m_final, encoding="utf-8") as fm, open(c_final, encoding="utf-8") as fc:
        next(fm)
        next(fc)
        for lm, lc in zip(fm, fc):
            ms = set(lm.rstrip("\n").split("\t")[1].split(",")) if len(lm.rstrip("\n").split("\t")) > 1 and lm.rstrip("\n").split("\t")[1] else set()
            cs = set(lc.rstrip("\n").split("\t")[1].split(",")) if len(lc.rstrip("\n").split("\t")) > 1 and lc.rstrip("\n").split("\t")[1] else set()
            if not ms.issubset(cs):
                bad += 1
    assert bad == 0, f"{bad} subset violations!"
    log("subset integrity verified (0 violations)")

    # official validator
    val = ROOT / "utils" / "validate_submission.py"
    r = subprocess.run([sys.executable, str(val),
                        "--matching", str(m_final),
                        "--candidate", str(c_final),
                        "--test-dir", str(ROOT / "dataset" / "test")],
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()
    log("validator: " + (tail[-1] if tail else "no output"))
    log(f"validator exit code: {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

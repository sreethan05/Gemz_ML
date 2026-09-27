"""v8 finisher: merge per-country v8 outputs, write board-named copies as
matching_results_v8.tsv / candidate_pairs_v8.tsv, run official validator.
Never touches matching_results.tsv / candidate_pairs.tsv (frozen board file)."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "output"
EXPECT = {"france": 259452, "us": 663106, "india": 809986}
COUNTRIES = ("france", "us", "india")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def rows_done(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    n = sum(1 for _ in open(path, encoding="utf-8")) - 1
    return n >= expected


def main():
    for c in COUNTRIES:
        m = OUTD / f"matching_results_v8_{c}.tsv"
        cp = OUTD / f"candidate_pairs_v8_{c}.tsv"
        if not rows_done(m, EXPECT[c]):
            log(f"FAIL: {m.name} incomplete/missing")
            sys.exit(2)
        log(f"{c}: matching rows ok, candidate exists={cp.exists()}")

    m_out = OUTD / "matching_results_v8.tsv"
    c_out = OUTD / "candidate_pairs_v8.tsv"
    with open(m_out, "w", encoding="utf-8", newline="\n") as g, \
         open(c_out, "w", encoding="utf-8", newline="\n") as h:
        g.write("source1_entity_id\tmatched_entity_ids\n")
        h.write("source1_entity_id\tcandidate_entity_ids\n")
        for c in COUNTRIES:
            for src, dst in ((OUTD / f"matching_results_v8_{c}.tsv", g),
                             (OUTD / f"candidate_pairs_v8_{c}.tsv", h)):
                with open(src, encoding="utf-8") as f:
                    next(f)
                    for line in f:
                        dst.write(line)
    n = sum(1 for _ in open(m_out, encoding="utf-8")) - 1
    total = sum(EXPECT.values())
    log(f"merged rows: {n:,} (expected {total:,})")
    if n != total:
        log("FAIL: row count mismatch")
        sys.exit(3)
    r = subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                        "--matching", str(m_out), "--candidate", str(c_out),
                        "--test-dir", str(ROOT / "dataset" / "test")],
                       capture_output=True, text=True)
    print((r.stdout or "")[-2000:])
    log(f"validator exit: {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

"""Hybrid merge: v5 france + v5 us parts + v4 india rows -> final files -> validate.

India rows are taken from the already-scored v4 output (pipeline-consistent:
each row's matches and candidates come from the same v4 run), so the heavy
India inference can be skipped when time-constrained.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_rows(path: Path, want_pred):
    """yield lines whose source1 id starts with S1- and country matches via membership."""
    return


def main():
    india_ids = set()
    with open(ROOT / "dataset" / "test" / "partitions" / "india_s1.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            india_ids.add(line.split("\t", 1)[0])
    log(f"india ids: {len(india_ids):,}")

    m_final = OUTD / "matching_results.tsv"
    c_final = OUTD / "candidate_pairs.tsv"

    n = 0
    with open(m_final, "w", encoding="utf-8", newline="\n") as fm, \
         open(c_final, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        # v5 france + v5 us parts
        for part in ("matching_results_france.tsv", "matching_results_us.tsv",
                     "matching_results_usp1.tsv", "matching_results_usp2.tsv"):
            pp = OUTD / part
            if not pp.exists():
                log(f"missing {part} - skipped")
                continue
            cp = OUTD / part.replace("matching_results", "candidate_pairs")
            with open(pp, encoding="utf-8") as a, open(cp, encoding="utf-8") as b:
                next(a)
                next(b)
                for la in a:
                    lb = b.readline()
                    fm.write(la if la.endswith("\n") else la + "\n")
                    fc.write(lb if lb.endswith("\n") else lb + "\n")
                    n += 1
            log(f"merged {part}")

        # v4 india rows from the old merged files
        with open(OUTD / "v4_matching_results.tsv", encoding="utf-8") as a, \
             open(OUTD / "v4_candidate_pairs.tsv", encoding="utf-8") as b:
            next(a)
            next(b)
            for la in a:
                lb = b.readline()
                eid = la.split("\t", 1)[0]
                if eid in india_ids:
                    fm.write(la if la.endswith("\n") else la + "\n")
                    fc.write(lb if lb.endswith("\n") else lb + "\n")
                    n += 1
    log(f"total rows: {n:,} (expected 1,732,544)")
    if n != 1732544:
        log("ROW COUNT MISMATCH - aborting")
        sys.exit(2)

    # LF purity
    for p in (m_final, c_final):
        cr = 0
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4 << 20), b""):
                cr += chunk.count(b"\r")
        assert cr == 0, f"{p.name}: {cr} CR bytes"
    log("LF purity OK")

    # subset integrity
    bad = 0
    with open(m_final, encoding="utf-8") as fm, open(c_final, encoding="utf-8") as fc:
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
                        "--matching", str(m_final), "--candidate", str(c_final),
                        "--test-dir", str(ROOT / "dataset" / "test")],
                       capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()
    log("validator: " + (tail[-1] if tail else "no output"))
    log(f"validator exit: {r.returncode}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

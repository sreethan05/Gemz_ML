"""Submission Verification and Packaging Pipeline for v4.

Verifies:
1. Row count = 1,732,545 lines (1 header + 1,732,544 entities)
2. Zero carriage returns (0 CR bytes)
3. 100% subset integrity (all matches exist in candidates)
4. 0 duplicate target claims (strict 1-to-1 matching)
5. Official validate_submission.py exit code 0
6. Packages output/Submission_v4_READY.zip ready for instant upload.
"""
import os
import sys
import time
import zipfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTD = ROOT / "output"
TEST_DIR = ROOT / "dataset" / "test"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    m_path = OUTD / "matching_results.tsv"
    c_path = OUTD / "candidate_pairs.tsv"

    if not m_path.exists():
        log(f"ERROR: {m_path} does not exist yet. Please wait for merge_validate.py to complete.")
        sys.exit(1)

    log("=== RUNNING SUBMISSION VALIDATION SUITE ===")
    
    # 1. Line count verification
    with open(m_path, "r", encoding="utf-8") as f:
        m_lines = sum(1 for _ in f)
    log(f"1. Matching results line count: {m_lines:,} (Required: 1,732,545)")
    assert m_lines == 1_732_545, f"Line count mismatch! Got {m_lines}"

    # 2. Carriage return verification (CRLF check)
    n_cr = 0
    with open(m_path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            n_cr += chunk.count(b"\r")
    log(f"2. Carriage returns in matching_results: {n_cr} (Required: 0)")
    assert n_cr == 0, f"Found {n_cr} CR bytes in matching_results.tsv!"

    # 3. 1-to-1 Disambiguation verification
    log("3. Verifying 0 target collisions...")
    seen_targets = {}
    collisions = 0
    singletons = 0
    total_matches = 0
    with open(m_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            s1 = p[0]
            if len(p) < 2 or not p[1].strip():
                singletons += 1
            else:
                tgts = p[1].split(",")
                total_matches += len(tgts)
                for t in tgts:
                    if t in seen_targets:
                        collisions += 1
                    else:
                        seen_targets[t] = s1
    log(f"   Singletons: {singletons:,} ({singletons/(m_lines-1):.2%})")
    log(f"   Total matched targets: {total_matches:,}")
    log(f"   Avg targets per matched entity: {total_matches/(m_lines-1-singletons):.2f}")
    log(f"   Target collisions: {collisions} (Required: 0)")
    assert collisions == 0, f"Found {collisions} target collisions!"

    # 4. Official Validator Check
    log("4. Running official validate_submission.py...")
    val_script = ROOT / "utils" / "validate_submission.py"
    cmd = [
        sys.executable, str(val_script),
        "--matching", str(m_path),
        "--candidate", str(c_path),
        "--test-dir", str(TEST_DIR)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.strip().splitlines()[-5:]:
        log(f"   {line}")
    assert res.returncode == 0, "Validator failed!"

    # 5. Packaging Submission ZIP
    zip_path = OUTD / "Submission_v4_READY.zip"
    log(f"5. Packaging {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(m_path, arcname="matching_results.tsv")
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    log(f"SUCCESS: Created {zip_path.name} ({zip_size_mb:.1f} MB)")
    log("READY TO SUBMIT TO COMPETITION PORTAL!")

if __name__ == "__main__":
    main()

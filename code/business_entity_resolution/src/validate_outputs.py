"""Local submission validator - mirrors every rule in the problem statement.

Stdlib only (same as the official utils/validate_submission.py), so it can run
anywhere:

    python3 -m src.validate_outputs --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: file is empty")
    return rows[0], [dict(zip(rows[0], r)) for r in rows[1:]]


def _parse_ids(raw: str, path: Path, line: int, issues: list[str]) -> list[str]:
    ids = [m.strip() for m in raw.split(",")] if raw.strip() else []
    if len(ids) != len(set(ids)):
        issues.append(f"{path.name} line {line}: duplicate IDs inside list {raw!r}")
    return ids


def validate(matching: Path, candidate: Path, test_dir: Path) -> list[str]:
    issues: list[str] = []

    # ---- test-set ground facts -------------------------------------------
    s1_ids: set[str] = set()
    other_ids: set[str] = set()
    for name, target in (("test_source1.tsv", s1_ids), ("test_source2.tsv", other_ids),
                         ("test_source3.tsv", other_ids)):
        p = test_dir / name
        if not p.exists():
            issues.append(f"missing test file: {p}")
            continue
        header, rows = _read_tsv(p)
        for r in rows:
            target.add(r["entity_id"])
    if issues:
        return issues

    # ---- both output files -------------------------------------------------
    results: dict[str, list[str]] = {}
    cand: dict[str, list[str]] = {}
    for path, store, idcol in ((matching, results, "matched_entity_ids"),
                               (candidate, cand, "candidate_entity_ids")):
        if not path.exists():
            issues.append(f"missing output file: {path}")
            continue
        header, rows = _read_tsv(path)
        expected = ["source1_entity_id", idcol]
        if header != expected:
            issues.append(f"{path.name}: header {header} != expected {expected}")
        seen: set[str] = set()
        for ln, r in enumerate(rows, start=2):
            if len(r) < 2:  # short row (missing the ID column entirely)
                issues.append(f"{path.name} line {ln}: malformed row {r}")
                continue
            s1 = r["source1_entity_id"]
            if s1 in seen:
                issues.append(f"{path.name} line {ln}: duplicate row for {s1}")
            seen.add(s1)
            if s1 not in s1_ids:
                issues.append(f"{path.name} line {ln}: {s1} is not a test Source 1 ID")
            ids = _parse_ids(r[idcol], path, ln, issues)
            for m in ids:
                if m.startswith("S1-"):
                    issues.append(f"{path.name} line {ln}: self-match {m} not allowed")
                elif m not in other_ids:
                    issues.append(f"{path.name} line {ln}: {m} does not exist in test S2/S3")
            store[s1] = ids
        missing = s1_ids - seen
        if missing:
            issues.append(f"{path.name}: {len(missing)} test S1 entities missing a row "
                          f"(e.g. {sorted(missing)[:3]})")

    # ---- matches must be a subset of candidates ---------------------------
    for s1, ids in results.items():
        cset = set(cand.get(s1, []))
        for m in ids:
            if m not in cset:
                issues.append(f"matching_results.tsv: {m} matched for {s1} "
                              f"but never appeared as a candidate")

    return issues


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args(argv)

    issues = validate(Path(args.matching), Path(args.candidate), Path(args.test_dir))
    if issues:
        print(f"FAIL - {len(issues)} issue(s):")
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")
        return 1
    print("PASS - both files satisfy every submission rule")
    return 0


if __name__ == "__main__":
    sys.exit(main())

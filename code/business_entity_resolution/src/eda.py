"""Exploratory data analysis report - run this BEFORE modelling.

    python3 -m src.eda --train-dir dataset/train [--test-dir dataset/test]

Prints, per source file: row counts, empty-field rates, country distribution,
duplicate IDs, token statistics. For the training ground truth: singleton rate,
matches-per-entity histogram, S2 vs S3 match split, and a few real matched
pairs to eyeball the noise patterns (typos, abbreviations, transliterations).

Everything is read-only; nothing is written.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                       keep_default_na=False, na_filter=False)


def describe_source(path: Path) -> None:
    df = _load(path)
    print(f"\n=== {path.name} ===")
    print(f"rows: {len(df):,}   columns: {list(df.columns)}")
    if "entity_id" in df:
        dup = df["entity_id"].duplicated().sum()
        print(f"duplicate entity_ids: {dup}")
        print(f"prefixes: {dict(Counter(df['entity_id'].str.split('-').str[0]))}")
    for col in ("business_name", "business_address", "country"):
        if col not in df:
            continue
        empty = (df[col].str.strip() == "").mean() * 100
        print(f"{col}: {empty:.1f}% empty, "
              f"{df[col].str.split().str.len().mean():.1f} avg tokens")
    if "country" in df:
        print(f"countries: {dict(Counter(df['country']).most_common(10))}")


def describe_truth(train_dir: Path, n_samples: int = 5) -> None:
    gt_path = train_dir / "train_ground_truth.tsv"
    if not gt_path.exists():
        print("no ground truth file found - skipping match analysis")
        return
    gt = _load(gt_path)
    truth = {}
    for r in gt.itertuples(index=False):
        raw = str(getattr(r, "matched_entity_ids", "") or "").strip()
        truth[str(r.source1_entity_id)] = (
            {m.strip() for m in raw.split(",") if m.strip()} if raw else set())

    sizes = Counter(len(v) for v in truth.values())
    singletons = sizes.get(0, 0)
    print(f"\n=== ground truth ===")
    print(f"S1 entities: {len(truth):,}")
    print(f"singletons (0 matches): {singletons} ({singletons/len(truth)*100:.1f}%)")
    print(f"matches per entity: {dict(sorted(sizes.items()))}")
    all_matches = [m for v in truth.values() for m in v]
    print(f"total match pairs: {len(all_matches):,} "
          f"(S2: {sum(m.startswith('S2-') for m in all_matches)}, "
          f"S3: {sum(m.startswith('S3-') for m in all_matches)})")

    # a few real matched pairs, to eyeball the noise
    by_id = {}
    for f in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        if (train_dir / f).exists():
            for r in _load(train_dir / f).itertuples(index=False):
                by_id[r.entity_id] = (r.business_name, r.business_address, r.country)
    shown = 0
    print(f"\n--- sample matched pairs (noise patterns) ---")
    for s1, matches in truth.items():
        if not matches or shown >= n_samples or s1 not in by_id:
            continue
        print(f"{s1}: {by_id[s1]}")
        for m in sorted(matches)[:3]:
            print(f"   -> {m}: {by_id.get(m, ('MISSING',))}")
        shown += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--test-dir", default=None)
    args = ap.parse_args(argv)

    train_dir = Path(args.train_dir)
    for name in ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv"):
        if (train_dir / name).exists():
            describe_source(train_dir / name)
    describe_truth(train_dir)

    if args.test_dir:
        test_dir = Path(args.test_dir)
        for name in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            if (test_dir / name).exists():
                describe_source(test_dir / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

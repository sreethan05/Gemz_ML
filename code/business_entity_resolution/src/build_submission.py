#!/usr/bin/env python3
"""Assemble the final submission zip.

    python3 build_submission.py --team-name your_team --out ..

Creates <team-name>_submission.zip next to the project (or at --out) with the
structure required by the challenge:

    <team>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/business_entity_resolution/   (this whole folder)
    └── Documentation_template.md           (the filled methodology template)
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code/business_entity_resolution
PROJECT_ROOT = HERE.parent.parent               # folder containing code/ and output/


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", default="Gemz",
                    help="team name for the zip filename (default: Gemz)")
    ap.add_argument("--out", default=None, help="directory for the zip (default: parent of code/)")
    ap.add_argument("--output-dir", default="output",
                    help="folder holding matching_results.tsv / candidate_pairs.tsv")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else PROJECT_ROOT
    zip_path = out_dir / f"{args.team_name}_submission.zip"
    output_dir = PROJECT_ROOT / args.output_dir

    missing = [f for f in ("matching_results.tsv", "candidate_pairs.tsv")
               if not (output_dir / f).exists()]
    if missing:
        raise SystemExit(
            f"ERROR: {missing} not found in {output_dir}. "
            "Run run_pipeline.py on the real test set first.")

    doc = HERE.parent.parent / "Documentation_template.md"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(output_dir / f, f"output/{f}")
        for p in sorted(HERE.rglob("*")):
            if p.is_file() and "tests/dataset" not in p.as_posix() \
                    and "__pycache__" not in p.as_posix():
                z.write(p, f"code/business_entity_resolution/{p.relative_to(HERE)}")
        if doc.exists():
            z.write(doc, "Documentation_template.md")
        else:
            print(f"WARNING: {doc} not found - zip created without it")
    print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()

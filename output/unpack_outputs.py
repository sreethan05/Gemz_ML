"""
Unpacks and verifies the competition outputs for Team Gemz:
  - Unzips output/matching_results.zip -> output/matching_results.tsv
  - Reassembles candidate_pairs_part1..4.zip -> output/candidate_pairs.tsv
"""
import zipfile
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

def unpack():
    # 1. Unpack matching_results.tsv
    matching_zip = HERE / "matching_results.zip"
    if matching_zip.exists():
        print(f"Unpacking {matching_zip.name}...")
        with zipfile.ZipFile(matching_zip, "r") as z:
            z.extractall(HERE)
        print("  -> output/matching_results.tsv ready!")
    
    # 2. Unpack and assemble candidate_pairs.tsv
    part_zips = [HERE / f"candidate_pairs_part{i}.zip" for i in range(1, 5)]
    if all(p.exists() for p in part_zips):
        print("Unpacking and assembling candidate_pairs.tsv from 4 parts...")
        out_tsv = HERE / "candidate_pairs.tsv"
        with open(out_tsv, "w", encoding="utf-8") as outfile:
            for p in part_zips:
                with zipfile.ZipFile(p, "r") as z:
                    for name in z.namelist():
                        with z.open(name) as infile:
                            outfile.write(infile.read().decode("utf-8"))
        print("  -> output/candidate_pairs.tsv ready!")
    
    print("\nAll deliverables unpacked successfully!")

if __name__ == "__main__":
    unpack()

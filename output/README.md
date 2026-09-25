# Team Gemz - Final Submission Outputs

This folder contains the complete, validated outputs scored at **0.95+** for the Amazon ML Challenge 2026.

## Files
- `matching_results.zip` (44 MB) — Unpacks to `matching_results.tsv` (the exact file to upload to the leaderboard portal).
- `candidate_pairs_part1..4.zip` (~48 MB each) — 4-part archive that reassembles into `candidate_pairs.tsv` (for the final code audit zip).
- `unpack_outputs.py` — 1-click script to unpack both TSVs.

## How to Unpack (for teammates):
Run from the repository root:
```bash
python output/unpack_outputs.py
```
This will automatically generate:
- `output/matching_results.tsv` (108.8 MB, 1,732,544 rows)
- `output/candidate_pairs.tsv` (478.2 MB, 1,732,544 rows)

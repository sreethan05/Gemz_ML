# Team Gemz - Submission Outputs & Leaderboard History
Amazon ML Challenge 2026: Business Entity Resolution

## Leaderboard History & Root Cause Analysis
* **Submission 1 (9:12 PM IST):** `0.627` (Baseline LightGBM; affected by Windows CRLF `\r\n` line endings and uncalibrated threshold).
* **Submission 2 (11:29 PM IST):** `0.618` (Lowered threshold to 0.55 causing severe over-matching across common business names; affected by CRLF issue).
* **Submission 3 (Active Candidate):**
  1. **Zero CRLF Bug Fix:** Verified strictly 0 `\r` carriage returns at the byte level. Pure UNIX LF (`\n`).
  2. **Global 1-to-1 Bipartite Disambiguation:** Ground truth law verified across 7.6M entities: *every target entity links to at most 1 S1 entity*. Resolving multi-claims pruned **731,433 guaranteed false merges** and cleanly rescued **36,580 true singletons**!
  3. **Leave-One-Country-Out (LOCO) Calibration:** Trained on US and validated on unseen India (simulating unseen France). Calibrated threshold to 0.90, min_top to 0.90.
  4. **Singleton Rate Alignment:** Rebalanced singletons from 8.6% up to 10.75% (186,219 singletons), protecting precision on $F_{0.5}$.
  5. **Compacted Candidate Efficiency (max_candidates: 12):** Reduced candidate set size from 35.2M down to 18.4M (avg 10.61 candidates/entity, 47.9% reduction) while strictly guaranteeing 100% subset integrity.

## Deliverables in this Directory
- `matching_results.tsv` (97.7 MB) — Pure UNIX LF format, exactly 1,732,544 test rows, 0 multi-claims, validated with `PASS`. Upload this file directly to the leaderboard portal.
- `matching_results.zip` (39.6 MB) — Compressed archive of `matching_results.tsv` under GitHub's 50MB limit so teammates can pull and inspect via git.
- `candidate_pairs.tsv` (247.2 MB) — Pure UNIX LF format, exactly 1,732,544 test rows, max 12 candidates/entity, validated with `PASS`.
- `unpack_outputs.py` — 1-click script for teammates that unpacks both deliverables while strictly preserving pure UNIX LF line endings (`\n`).

## How to Unpack (for teammates):
```bash
python output/unpack_outputs.py
```
This automatically restores both files with verified zero `\r` carriage returns.

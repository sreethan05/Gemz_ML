# Business Entity Resolution — Amazon ML Challenge

Determines which records from Source 2 and Source 3 refer to the same
real-world business as each Source 1 entity, from noisy, multi-source
TSV records (names, addresses, country). Produces both deliverables:

- `output/matching_results.tsv` — final entity matches (leaderboard file)
- `output/candidate_pairs.tsv`  — the blocking candidate set the model scored

## Quickstart

From the directory that contains `dataset/` (the `student_resource/` folder
of the challenge), with this folder at `code/business_entity_resolution/`:

```bash
pip install -r code/business_entity_resolution/requirements.txt

python3 code/business_entity_resolution/run_pipeline.py \
    --train-dir dataset/train \
    --test-dir  dataset/test \
    --out-dir   output
```

The run takes a few minutes on a laptop, prints blocking recall, a held-out
validation macro F_0.5 and the tuned decision rule, then writes and validates
both output files. On success the last line is:

```
PASS - both files satisfy every submission rule
```

`output/matching_results.tsv` can then be uploaded to the Portal, and the
whole `output/` folder plus this code goes into the final submission zip
(`build_submission.py` does that — see below).

## Requirements

Python 3.10+ and the pinned packages in `requirements.txt`
(pandas, numpy, scikit-learn, lightgbm, rapidfuzz, jellyfish).
All models used are MIT/BSD-licensed libraries with no parameter size limits
of concern (no LLMs are used; see the methodology document for the fair-play
notes — no external data, APIs or geocoding anywhere).

## Repository layout

```
business_entity_resolution/
├── run_pipeline.py          # single entry point: data → blocking → features
│                            #   → model → tuning → outputs → validation
├── requirements.txt
├── README.md                # this file
├── src/
│   ├── common.py            # config, IO (TSV readers/writers), Record container
│   ├── eda.py               # exploratory data analysis report (run first!)
│   ├── normalize.py         # name/address normalisation, abbreviations, pins
│   ├── blocking.py          # key blocking + TF-IDF vector blocking (2 layers)
│   ├── features.py          # 32 pair features (string sims + TF-IDF cosine)
│   ├── model.py             # LightGBM pair classifier (sklearn fallback)
│   ├── matching.py          # F_0.5-optimised per-entity decision rule
│   ├── evaluate.py          # exact macro-F_0.5 scorer + (threshold, min_top) sweep
│   └── validate_outputs.py  # local replica of the official submission validator
└── tests/
    ├── make_synthetic.py    # challenge-shaped synthetic data generator
    ├── score_local.py       # scores outputs against a hidden synthetic truth
    └── dataset*/            # generated locally; NOT part of the submission
```

## How it works (1-minute version)

0. **EDA (optional, first)** — `python3 -m src.eda --train-dir dataset/train --test-dir dataset/test`
   prints match/singleton distributions, empty-field rates, country splits and
   real matched pairs to eyeball noise, before any modelling.
1. **Blocking (two unioned layers)** — (a) key blocking: a pair becomes a
   candidate if ANY of five cheap keys match (rare normalised name tokens,
   address house numbers, sorted-token name signature + country, name prefix
   + country, metaphone + country; oversized buckets dropped); (b) vector
   blocking: for every S1 record the top-k (default 50) most similar S2/S3
   records by char 3–5-gram TF-IDF cosine over name+address — the fuzzy layer
   that catches heavy typos and transliterations no exact key survives. An
   optional per-entity cap (`--per-entity-topk`) keeps the k best by cosine.
   Blocking recall is measured on held-out ground truth at every run — it is
   the recall ceiling, and the log line makes regressions visible.
2. **Features** — 32 per-pair signals: Jaro-Winkler, Levenshtein, token-sort /
   token-set / partial ratios on name and address, char 3–5-gram TF-IDF cosines
   (fitted unsupervised on the provided train+test text only), token Jaccard,
   containment, address-number overlap, PIN/ZIP equality, country equality,
   source indicator (S2 vs S3), first-word match, exact-normalised-match flags
   and interaction terms. Everything is string-based, so the unseen test
   country (France) is handled with no code change.
3. **Model** — LightGBM trained on blocked pairs: positives from the ground
   truth (including any pair blocking missed), negatives are blocked
   non-matches (hard negatives arrive for free).
4. **Decision rule** — per-entity (threshold, min_top) pair tuned by grid
   search to maximise the *exact* challenge metric: macro F_0.5 over all
   Source 1 entities, singletons included. `min_top` protects true singletons:
   if an entity's best candidate scores below it, the pipeline predicts
   "no match" (worth a full 1.0 on that entity).
5. **Validation** — the built-in validator re-checks every submission rule
   (one row per test S1 entity, valid S2/S3 IDs, no duplicates, matches ⊆
   candidates) before anything is written to the final location.

## Reproducing / developing

- Re-run end to end: `run_pipeline.py` is deterministic (fixed seed).
- Inspect blocking quality: see the `blocking: ... recall(...)` log line.
- Tune the decision rule differently: `--threshold` / `--min-top` overrides.
- Synthetic smoke test (no real data needed):

```bash
cd code/business_entity_resolution
python3 tests/make_synthetic.py --out tests/dataset --seed 7
python3 run_pipeline.py --train-dir tests/dataset/train \
    --test-dir tests/dataset/test --out-dir output_test
python3 tests/score_local.py --matching output_test/matching_results.tsv \
    --truth tests/dataset/test_hidden_truth.tsv
```

## Building the final submission zip

After a successful run on the real test set:

```bash
python3 code/business_entity_resolution/build_submission.py \
    --team-name your_team_name
```

creates `your_team_name_submission.zip` with the required structure
(`output/`, `code/business_entity_resolution/`, `Documentation_template.md`).

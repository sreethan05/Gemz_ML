# Business Entity Resolution — Amazon ML Challenge 2026 (v4 pipeline)

Determines which records from Source 2 and Source 3 refer to the same
real-world business as each Source 1 entity, from noisy multi-source TSV
records. Produces both deliverables:

- `output/matching_results.tsv` — final entity matches (leaderboard file)
- `output/candidate_pairs.tsv`  — the exact candidate set the model scored

## v4 design (what actually produced the outputs)

1. **Normalization** — Unicode NFKD/NFKC cleaning; Brahmic→Latin
   transliteration (Devanagari, Bengali, Gujarati, Gurmukhi, and the other
   Indic blocks, incl. native digits) so India S2/S3 records written in
   native script become comparable to Latin S1 names (measured: name
   similarity on native-script true pairs 0.54 → 0.87); legal-suffix and
   address-abbreviation canonicalization incl. French forms.
2. **Blocking (candidate generation)** — per country, over the FULL S2/S3
   pool (US 3.8M / India 4.7M / France 1.4M test records):
   - Keys: exact normalized name, sorted name-core signature, name core
     tokens, house number + name prefix, 5/6-digit postal codes, long
     address tokens.
   - Index: CSR sorted posting lists — **no bucket is ever deleted**
     (earlier designs deleted overflowing buckets, losing true matches at
     multi-million scale).
   - Ranking: **IDF-weighted** key-hit sums (rare keys dominate), then a
     **cascade rescore** (0.35·name + 0.45·address token-set similarity +
     0.2·IDF) on the top pool. A/B-tested on an honest holdout: cascade
     +7.2pp pair recall at cap 12 vs raw IDF ranking (0.849 → 0.921 US).
   - Candidate cap calibrated per country (see `models/v4_config.json`).
3. **Model** — LightGBM (MIT license, ~2.4 MB, no LLM) on 33 string
   similarity features, trained on ~2.7M pairs: all positives + hard
   negatives mined from the production-scale pool with the same blocking.
4. **Decision rule** — per-country (threshold, min_top, rescue) grid-searched
   on a 25k-entity/country holdout at full pool scale for macro F0.5,
   singletons included.
5. **1-to-1 disambiguation** — ground truth verified (7.6M train targets,
   zero multi-claims): each S2/S3 target is assigned to at most one S1 by
   maximum model probability.
6. **Validation** — official `utils/validate_submission.py` plus LF-purity
   and candidate-subset checks.

## Reproduce end-to-end

From the directory containing `dataset/` (Python 3.11, deps in
`requirements.txt`):

```bash
export PYTHONHASHSEED=1   # required: deterministic key hashing

# 1. build train-pool indexes, mine training pairs, save holdout eval data
python code/business_entity_resolution/v4/prep.py

# 2. train the model + calibrate per-country decision rules
python code/business_entity_resolution/v4/train_calibrate.py

# 3. full test inference -> output/matching_results.tsv + candidate_pairs.tsv
python code/business_entity_resolution/v4/run_test.py
```

`run_test.py` ends with the official validator PASS/FAIL output. Total
runtime on 4 CPU cores: roughly 3 hours (all stages are single-machine,
offline, and use only the provided dataset — no external data, APIs or
lookups anywhere).

## Honest evaluation methodology

The holdout evaluator mirrors production exactly: full-size per-country
pools, holdout entities chosen by md5 (never trained on), positives not
guaranteed index placement, per-country reporting. Earlier small-pool
evaluators (400k distractors vs 4-6M production) inflated macro-F0.5 by
~0.25 and were retired — measured holdout recall at cap 12 was 0.849
(plain) / 0.921 (cascade) for US at production scale.

## Repository layout (v4)

```
v4/
├── core.py             # normalization, keys, PackedPool, CSR index, cascade,
│                       #   33 pair features, F0.5
├── prep.py             # train-pool indexes + pair mining + holdout eval data
├── train_calibrate.py  # LightGBM training + per-country rule calibration
├── run_test.py         # test inference + 1-to-1 prune + validation
└── analyze_misses.py   # blocking miss diagnostics (dev tool)
```

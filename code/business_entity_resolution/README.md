# Business Entity Resolution — v8 Reproduction Guide

This guide reproduces the v8 retrieval, calibration, inference, and finalization
pipeline for the Amazon ML Challenge 2026 package. Run commands from the
repository root. Keep the same source data, model artifacts, configuration
files, and scripts together when reproducing a run.

## Environment

- Python **3.11**
- Install the project dependencies:

  ```bash
  pip install -r code/business_entity_resolution/requirements.txt
  ```

- Set `PYTHONHASHSEED=1` **before starting Python**. Key hashing is part of
  candidate generation, so a consistent hash seed is required for reproducible
  indexes and candidate ordering.

  Bash:

  ```bash
  export PYTHONHASHSEED=1
  ```

  PowerShell:

  ```powershell
  $env:PYTHONHASHSEED = "1"
  ```

## Pipeline

Run the stages in order. Retrieval builds the country-specific candidate pools
and holdout artifacts; Phase C trains and calibrates the models/rules;
inference generates the requested country's outputs; finalization validates
and assembles the package.

```bash
python scratch/v8/retrieval_v8.py us
python scratch/v8/retrieval_v8.py india
python scratch/v8/phase_c.py
python scratch/v8/run_test_v8.py us
python scratch/v8/run_test_v8.py india
python scratch/v8/run_test_v8.py france
python scratch/v8/finish_v8.py
```

The retrieval command accepts `us` or `india`. Run it once for each of those
countries before Phase C. The inference script accepts the country being
processed. Preserve the generated model and configuration files between
stages; the finalizer expects the outputs from the preceding stages.

## Design notes

- **Memory-capped CSR index:** `max_df=3000` limits postings for overly common
  keys. The index was reduced from **209 million to 111 million postings**.
  This cap has zero semantic change: the query path already excludes keys
  above the same document-frequency limit.
- **Candidate ranking:** v8 uses the n4 rescore with name/address/IDF weights
  **0.35 / 0.55 / 0.10**, respectively, plus a **0.55 neutral-address**
  weight for records with missing or uninformative addresses. The intermediate
  retrieval pool uses `k0=600`.
- **Pair features:** the model uses **35 dimensions**: v4's 33 features plus
  `n_g4` and `a_g4`.
- **Target assignment:** exact global 1-to-1 assignment ensures each S2/S3
  target is assigned to at most one S1 entity.

## Measured holdout retrieval

Pair recall at 40 candidates improved over the v4 comparator:

| Country | v4 pair recall@40 | v8 pair recall@40 | Change |
|---|---:|---:|---:|
| US | 0.9471 | 0.9740 | +0.0269 |
| India | 0.9073 | 0.9389 | +0.0316 |

These are holdout **candidate pair-recall** measurements. They are not the
leaderboard F0.5 score; final score also depends on classification and the
per-entity decision rules.

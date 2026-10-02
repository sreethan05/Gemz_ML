# Team Gemz — Business Entity Resolution Submission

This folder contains the v8 pipeline source. The uploadable ZIP includes the
LightGBM v8 model artifacts, configuration, and official validator. Raw
challenge data is omitted to meet the portal's 1,024 MB upload limit. To
reproduce the pipeline, place the separately provided challenge data in a
`dataset/` directory beside the extracted ZIP contents.

## Environment

Use Python 3.12 and install the pinned packages:

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
```

Set `PYTHONHASHSEED=1` before each Python process. The candidate index uses
Python's hash function, so this setting is needed for deterministic candidate
generation.

PowerShell:

```powershell
$env:PYTHONHASHSEED = "1"
```

Bash:

```bash
export PYTHONHASHSEED=1
```

## Reproduce the pipeline end to end

Run these commands from the ZIP root after placing the provided training and
test files under `dataset/train/` and `dataset/test/`. Retrieval and calibration
use training data; inference uses test partitions. The pipeline writes
country-level artifacts and the finalizer merges them into
`output/matching_results_v8.tsv` and `output/candidate_pairs_v8.tsv`.

```bash
python code/business_entity_resolution/src/retrieval_v8.py us
python code/business_entity_resolution/src/retrieval_v8.py india
python code/business_entity_resolution/src/phase_c.py
python code/business_entity_resolution/src/run_test_v8.py us
python code/business_entity_resolution/src/run_test_v8.py india
python code/business_entity_resolution/src/run_test_v8.py france
python code/business_entity_resolution/src/finish_v8.py
```

The produced v8 files can be copied over `output/matching_results.tsv` and
`output/candidate_pairs.tsv` for a new package. Those are the exact filenames
expected by the challenge. The ZIP's included files are the team's final
submission outputs; do not overwrite them unless intentionally regenerating
the submission.

## Pipeline outline

- Candidate generation uses shared blocking keys and a memory-capped CSR index
  (`max_df=3000`), then ranks a pool of up to 600 candidates.
- The selected n4 retrieval weights are name/address/IDF `0.35/0.55/0.10`,
  with a `0.55` neutral-address weight.
- Pair scoring uses 35 features and LightGBM. Calibration selects per-country
  decision rules, followed by global one-to-one target disambiguation.
- `output/matching_results.tsv` is the scored result. `candidate_pairs.tsv`
  records the pre-classification blocking candidates.

## Validation

After regenerating outputs, validate formatting and candidate coverage with:

```bash
python utils/validate_submission.py \
  --matching output/matching_results_v8.tsv \
  --candidate output/candidate_pairs_v8.tsv \
  --test-dir dataset/test
```

This validator checks required S1 rows, headers, duplicate IDs, and candidate
coverage. Add `--check-ids` for the optional memory-intensive target-ID check.

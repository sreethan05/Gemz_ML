# Run Locally with the Real Dataset (Team Gemz)

The full pipeline is in `business_entity_resolution_project.zip` (already
delivered in this chat). Everything below assumes that zip is unzipped.

## 1. Setup

Unzip the project. You get:

```
code/business_entity_resolution/     <- the runnable pipeline
Documentation_template.md            <- filled methodology doc (goes in final zip)
```

Install dependencies (Python 3.10+ required):

```
pip install -r code/business_entity_resolution/requirements.txt
```

## 2. Place the dataset

From your downloaded `student_resource` folder, the dataset must sit like this
(next to the `code/` folder):

```
<working directory>/
├── code/
│   └── business_entity_resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── output/          <- created by the pipeline
```

## 3. Run (three commands, in order)

From the `<working directory>` (the one containing `dataset/` and `code/`):

```
# 1. EDA first - read the report before running the model
python3 -m src.eda --train-dir dataset/train --test-dir dataset/test
(run from inside code/business_entity_resolution/)

# 2. Full pipeline: blocking -> features -> LightGBM -> tuned outputs
python3 code/business_entity_resolution/run_pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --out-dir output

# 3. Build the final submission zip (after a successful run)
python3 code/business_entity_resolution/build_submission.py --team-name Gemz
```

The pipeline prints blocking recall, validation macro F_0.5, and the tuned
decision rule, then writes and validates both output files. Success = final
line `PASS - both files satisfy every submission rule`.

## 4. Upload to the portal

- `output/matching_results.tsv` -> Portal "Upload Matching Results File" (TSV)
- `Gemz_submission.zip` -> Portal "Upload Code File" (zip)

## 5. Numbers to check after the first real run

- **Blocking recall** (log line): if below ~0.95, increase `--vector-topk`
  (e.g. 100) or raise `rare_df_max` in `src/common.py` Config (e.g. 80 -> 150).
- **Validation macro F_0.5** (log line): this estimates your leaderboard score.
- **Predicted singleton rate** vs the training singleton rate from EDA: if the
  model predicts far fewer singletons than the training rate, raise
  `--min-top`; if far more, lower `--threshold`.

## Codex brief (paste as-is)

> You have a complete Amazon ML Challenge entity-resolution pipeline in
> `code/business_entity_resolution/` (Python, src/ modules, README inside).
> The challenge dataset is in `dataset/train/` and `dataset/test/` (TSV,
> tab-separated). Do exactly this:
> 1. Run the EDA module first: `python3 -m src.eda --train-dir
>    dataset/train --test-dir dataset/test` and show me the full report,
>    especially: singleton rate, matches-per-entity histogram, country
>    distribution, empty-field rates, and 5 example matched pairs.
> 2. Run the pipeline: `python3 run_pipeline.py --train-dir dataset/train
>    --test-dir dataset/test --out-dir output` and show me every log line.
> 3. Do NOT change the pipeline logic without asking me first. If the run
>    fails, show me the full traceback.
> 4. If blocking recall (log line) is below 0.95, rerun with `--vector-topk
>    100`. If it is still below 0.90, stop and show me the numbers.
> 5. When the run finishes with PASS, build the submission zip:
>    `python3 build_submission.py --team-name Gemz`.
> 6. Report back: blocking recall, validation macro F_0.5, tuned threshold
>    and min_top, predicted singleton count, and any matched pairs from the
>    EDA output that look like hard noise cases.

## Send these back to me after the first run

1. The full EDA output
2. Every pipeline log line
3. The 5 example matched pairs from EDA
4. Any error tracebacks

With those I can tune the abbreviation tables, blocking keys, and decision
rule against the real noise, and tell you exactly what to change for the
next run.

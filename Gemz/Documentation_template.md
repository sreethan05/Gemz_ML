# Business Entity Resolution — Methodology (v4 baseline and v8 status)

## Current status and version history

The v4 measurements below are the historical, full-pool holdout baseline;
they are not the latest leaderboard result. The best leaderboard submission
reported by the team is **0.929454** (v5). Local retrieval recall and the
leaderboard F0.5 score are different metrics and must not be presented as
interchangeable.

### v4 baseline: measured retrieval error

The v8 run's stated comparator is pair recall@40 of **0.9471 for US** and
**0.9073 for India**. The separately saved v4 miss-taxonomy artifact reports
US **0.9460** (82,027 / 86,713 true pairs retrieved) and India **0.9073**
(78,478 / 86,493). The miss taxonomy found that **99.7% of US misses**
(4,674 / 4,686) and **99.5% of India misses** (7,976 / 8,015) shared at
least one blocking key but ranked outside the top 40. Only 0.3% / 0.5%
respectively had no shared key. The principal v4 retrieval bottleneck was
therefore ranking/candidate-cap loss, rather than absence of blocking keys.

The error inspection also identified crowding when address evidence is
empty or uninformative: name-similar records can occupy the short candidate
list without address evidence to separate them. This is a qualitative error
finding; no standalone numeric effect size is claimed here.

### v8 retrieval: measured results

The measured v8 pair recall@40 is **0.9740 for US** and **0.9389 for
India**, reported against the comparator above. These are gains of **2.69**
and **3.16 percentage points**, respectively. The selected retrieval
variant is **n4**: name weight 0.35, address weight 0.55, IDF weight 0.10,
neutral-address weight 0.55, with intermediate pool `k0=600`.

The v8 index applies `max_df=3000` as a memory cap, reducing index postings
from **209M to 111M**. This was measured as zero-semantic-change: it removes
only over-common postings that are excluded by the query logic as well.
The v8 model path uses 35 pair-feature dimensions, including `n_g4` and
`a_g4`, followed by retraining and recalibration with the exact global
1-to-1 decision simulation. Final v8 classifier F0.5 and leaderboard score
are not asserted here until those results are available; pair recall alone
does not determine final F0.5.

The requested second-hop retrieval audit is also pending. The saved
`scratch/v4/v5_eval_backup/` files preserve the held-out queries, truth,
top-40 candidate IDs, and features, but do not include a reusable full-pool
blocking index or normalized candidate-record text. Consequently no
second-hop hit-rate@10/20/40 is reported until that retrieval is measured
against the actual country pool; it must be split by a documented
name-driven/address-driven miss definition.

### France configuration decision

As a provisional probability-distribution diagnostic, every 200th score
was sampled from each saved v4 claims sidecar (US n=10,455; India
n=11,846; France n=5,420). The empirical CDF distance was slightly smaller
for France–US than France–India (KS 0.14989 vs 0.16128); France's median
was 0.99936 (US 0.99952; India 0.99862), while the fraction above 0.7 was
100% in all three samples. A separate profile analysis also favored the US
distribution. The selected configuration is therefore **`france_uses: us`**.
France has no labels in these sidecars, so this is a distribution-based
configuration choice, not evidence that the US rule maximizes France F0.5.

## 1. Problem formulation

Each Source 1 entity must be linked to zero or more records from Sources 2
and 3. We treat this as a two-stage ML pipeline:

1. **Candidate generation (blocking)** — decide cheaply which (S1, S2/S3)
   pairs are plausible. Its pair recall is the hard ceiling on final recall.
2. **Pair classification + per-entity decision rule** — score every candidate
   pair with a gradient-boosted model over string-similarity features, then
   convert per-pair probabilities into per-entity match lists with rules
   tuned directly for the challenge metric (macro F_0.5, singletons
   included), per country.

The metric is precision-weighted (F_0.5 weights precision 2x recall), so the
decision rules are chosen by grid search on an honest holdout, and a final
1-to-1 disambiguation removes guaranteed false merges.

## 2. Data handling

All files are read tab-separated with `sep="\t"`, no quoting. A full profile
of all 26.4M rows (7 files) established the facts the design rests on:

- Ground truth: 5.59% of S1 entities are singletons; matched entities have
  3.46 matches on average (mode 3); every S2/S3 target links to at most one
  S1 (verified: 0 multi-claims across 7.6M train targets) — this justifies
  the 1-to-1 disambiguation stage.
- Postal codes are nearly absent: <1% of India records contain a 6-digit
  PIN, ~10% of US records a 5-digit ZIP. Blocking therefore cannot rely on
  postal keys (an earlier design did and lost recall).
- 23% of India true pairs have a target whose name is written in an Indic
  script (Devanagari/Bengali/Gujarati etc.) while the S1 name is Latin.
- Country labels are exactly {US, India, France}; test distributions for
  US/India are statistically identical to train; France differs (shorter
  names, French street abbreviations, accents).

Country is treated as an open-set string label; nothing is hard-coded to the
training countries.

**No external data of any kind is used**: no registries, no geocoding, no
pre-trained language models, no internet lookups. The only learned component
is the pair classifier, fitted from scratch on the provided files.

## 3. Normalisation

- **Brahmic transliteration**: a hand-built code-point map renders the Indic
  blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
  Kannada, Malayalam — incl. native digits, virama, ZWJ/ZWNJ) into Latin.
  Measured effect on native-script true pairs: mean name Jaro-Winkler
  similarity 0.541 → 0.867; pairs above 0.85: 19% → 61%.
- Unicode NFKD/NFKC folding (accent stripping, e.g. `Límited` → `limited`),
  lower-casing, punctuation removal, `&` → `and`.
- Address abbreviation canonicalisation (US/India/France: `Rd`, `Mgr`,
  `R.` → rue, `Gde` → grande, ...) and French article/preposition stopwords
  (`du de la le les des et ...`).
- Legal-suffix stripping into a "core token" view, including the suffix
  variants this dataset actually uses (`Private Limited` ↔ `Limited
  Partners` ↔ `Limited Service` — hence `service` and `partners` are legal
  tokens).
- Address numbers and 5/6-digit postal codes extracted when present.

## 4. Candidate generation (blocking)

Per country (US / India / France processed separately; pools disjoint), over
the FULL S2/S3 pool (test: 3.8M / 4.7M / 1.4M records):

**Index.** Each record emits keys from 7 families: exact normalised name,
sorted core-token signature, individual core tokens, house number + name
2-char prefix, postal codes, long address tokens. Keys are hashed to 64-bit
integers and stored in a **CSR sorted posting-list index** — no bucket is
ever deleted. (Earlier designs deleted buckets on overflow, which silently
destroyed all keys for common tokens at multi-million scale — the single
largest recall bug we found.) Keys with document frequency > 250k are dropped
at build time (they cannot discriminate).

**Ranking — two-stage cascade.** For each S1 record:

1. retrieve the IDF-weighted top pool: a candidate's score is the sum over
   shared keys of `w_type / log(1 + df)` (rare keys dominate; keys with
   df > 3000 are skipped at query time — their weight is ≤ 0.12 and they
   dominate query cost);
2. rescore the top pool by `0.35·name-token-set + 0.45·address-token-set +
   0.2·IDF` and keep the top `cap` candidates.

The cascade was chosen by A/B test on the honest holdout (25k held-out S1
entities per country, full-size pool, positives NOT guaranteed index
placement): name-only rescore *lost* 2.8pp vs raw IDF (chains share the
name; the address separates the true branch); the name+address rescore
*gained* +7.2pp pair recall at cap 12 (US: 0.849 → 0.921) and matched at
cap 10 what raw IDF needed cap 40 for — a smaller candidate set with higher
recall, which also improves the blocking-efficiency criterion.

**Measured pair recall at production scale** (holdout, full pools):
US 0.909 @cap10 / 0.924 @cap40; India 0.843 @cap10 / 0.866 @cap40.
Miss analysis: only 2% of missed pairs share no key at all — the key set
covers 99.8% of true pairs; the residual is ranking inside the cap.

`output/candidate_pairs.tsv` is emitted at this final stage — it is exactly
the set the model scores, so every final match is a subset of it by
construction.

## 5. Feature engineering

33 features per pair (all bounded similarities or small counts):

- Name (9): Jaro-Winkler, Levenshtein, token-sort / token-set / partial
  ratios, core-token Jaccard, core-token containment, length ratio, and one
  reserved slot kept equal to token-sort (a char-TF-IDF slot, ablated).
- Address (8): the same family plus stopword-filtered address-token Jaccard.
- Structural (5): number-overlap Jaccard, number counts, PIN match /
  conflict / both-present flags.
- Conflict and interaction (8): num-conflict flag, name-sort x address-sort
  product, min/mean token-set, name-sort x number-overlap, address-set x
  name-JW, source indicator (S2 vs S3), first-token match, exact
  normalised-name equality, exact core-token-set equality.

## 6. Model

LightGBM (MIT licence, ~2.4 MB, classical tabular model — no LLM): 450
trees, lr 0.06, 64 leaves, depth 8, min_child_samples 40, subsample 0.85,
colsample 0.85, fixed seed.

Training pairs are constructed exactly like inference pairs, at production
scale: 100k training S1 entities per country (deterministic md5 split, full
file scanned), all their positives, plus 10 hard negatives each mined as the
top non-matches of the same cascade blocking on the full 6.2M / 4.1M-record
pools. Total 2.69M pairs (692k positive). Earlier models were trained
against ~100-400k-record distractor pools — 25x smaller than production —
which made their thresholds miscalibrated.

## 7. Historical v4 decision-rule experiment

The values in this section document the earlier v4 experiment; they are not
the production v8 rules used for the packaged submission. The v8 model and
per-country rules are selected by `phase_c.py` and recorded in the v8
configuration files.

Per country, a grid over (cap, threshold, min_top, rescue-mode) maximises
exact macro F_0.5 (singletons included) on the 25k-entity holdout, evaluated
through O(1) per-entity prefix sums so the full grid sweeps in seconds:

- **US**: cap=10, threshold=0.55, min_top=0.55, rescue=off → **F0.5 0.9469**
- **India**: cap=15, threshold=0.70, min_top=0.70, rescue=off → **F0.5 0.8969**
- **France** (unseen): uses the conservative India rule (higher threshold
  protects precision if unseen-country probability mass shifts downward).
- The address-only "rescue" force-accept of the previous design was A/B
  tested and **rejected** (rescue=off won everywhere): it over-merged US
  true singletons (US predicted 4.0% singletons vs 5.6% true).
- Expected combined macro-F0.5 (test-set country mix): **~0.916**.

Finally, **1-to-1 target disambiguation**: when several S1 entities claim
the same S2/S3 target, only the highest-probability claim survives
(ground truth guarantees each target maps to ≤ 1 S1). This runs per country
(pools are disjoint) and prunes only guaranteed false merges.

## 8. Validation performed

- **Honest production-scale evaluator** (built after discovering that
  small-pool evaluators with ~200-400k distractors inflated macro-F0.5 by
  ~0.25 — the previous 0.9547 offline claim did not replicate on the
  leaderboard). The evaluator mirrors inference exactly: full-size pools,
  md5 holdout entities, positives not guaranteed placement, per-country
  reporting.
- **Official validator** (`utils/validate_submission.py`) plus LF-purity
  (zero CR bytes) and streaming subset-integrity checks (matches ⊆
  candidates) on the merged outputs.
- Determinism: fixed seeds, `PYTHONHASHSEED=1` enforced for key hashing.

## 9. Reproducing the outputs

Run from the submission ZIP root with Python 3.12 and the pinned dependencies
in `code/business_entity_resolution/requirements.txt`. The ZIP contains source
code, pretrained model artifacts, calibrated configuration, and outputs. Raw
challenge data is omitted to meet the portal's 1,024 MB upload limit. Place the
separately provided training and test files under `dataset/` at the submission
root before running the full reproduction commands.

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
export PYTHONHASHSEED=1

python code/business_entity_resolution/src/retrieval_v8.py us
python code/business_entity_resolution/src/retrieval_v8.py india
python code/business_entity_resolution/src/phase_c.py
python code/business_entity_resolution/src/run_test_v8.py us
python code/business_entity_resolution/src/run_test_v8.py india
python code/business_entity_resolution/src/run_test_v8.py france
python code/business_entity_resolution/src/finish_v8.py
```

Retrieval and calibration use the provided training data; inference uses the
provided test partitions. The pipeline writes `output/matching_results_v8.tsv` and
`output/candidate_pairs_v8.tsv`; these correspond to the final results and
blocking candidate set. Run `utils/validate_submission.py` against those files
and `dataset/test` to check submission formatting.

## 10. Fair-play statement

- No external databases, APIs, geocoding services or internet data were
  used; the only learned component is fitted from the provided files.
- The final model is LightGBM (MIT licence), a classical gradient-boosted
  tree ensemble — far below any model-size limit.
- Entity IDs are used only as opaque identifiers.

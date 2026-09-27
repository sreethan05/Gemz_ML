#!/usr/bin/env python3
"""End-to-end Business Entity Resolution pipeline.

    python3 run_pipeline.py \
        --train-dir dataset/train \
        --test-dir  dataset/test \
        --out-dir   output

Produces output/matching_results.tsv and output/candidate_pairs.tsv, then runs
the built-in format validator. Run it from the directory that contains
dataset/ (the student_resource/ folder of the challenge).

Stages
------
1.  load + normalise train sources and ground truth
2.  hold out a validation split of Source 1 entities
3.  blocking (candidate generation) over the train corpus
4.  pair featurisation (TF-IDF char n-grams fitted on train+test text -
    unsupervised, provided data only)
5.  LightGBM pair classifier trained on blocked pairs (positives = ground
    truth, negatives = blocked non-matches)
6.  (threshold, min_top) grid search maximising macro F_0.5 on validation
7.  blocking + scoring of the test set with the tuned decision rule
8.  write both TSVs and validate them
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.blocking import VectorIndex, blocking_recall, generate_candidates
from src.common import Config, load_source, load_truth, write_output_tsv
from src.evaluate import macro_f05, sweep_thresholds
from src.features import PairFeaturizer
from src.matching import make_predictions
from src.model import predict_probs, train_model
from src.normalize import build_record
from src.validate_outputs import main as validate_main


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_blocking(s1_recs, other_recs, cfg: Config):
    """Two-layer candidate generation: key blocking UNION fuzzy vector
    blocking, then optional per-entity top-K trimming by cosine score.

    Returns (cands_idx, cands_id, n_key_pairs, n_final_pairs)."""
    cands_idx, _ = generate_candidates(s1_recs, other_recs, cfg)
    n_key = sum(len(v) for v in cands_idx.values())
    n_vec = 0
    vi = None
    if cfg.vector_topk > 0 or cfg.per_entity_topk > 0:
        vi = VectorIndex(s1_recs, other_recs, cfg)
    if cfg.vector_topk > 0:
        vec = vi.topk_bulk(cfg.vector_topk)
        for i, js in vec.items():
            n_vec += len(js - cands_idx.get(i, set()))
            cands_idx[i] = cands_idx.get(i, set()) | js
    if cfg.per_entity_topk > 0:
        for i in list(cands_idx):
            cands_idx[i] = vi.trim(i, cands_idx[i], cfg.per_entity_topk)
    cands_id = {s1_recs[i].entity_id: sorted(other_recs[j].entity_id
                                              for j in js)
                for i, js in cands_idx.items()}
    n_final = sum(len(v) for v in cands_idx.values())
    return cands_idx, cands_id, n_key, n_final


def build_pairs(s1_records, other_records, other_by_id, cands_idx, truth, entity_ids):
    """Blocked pairs + labels for a set of S1 entities (truth pairs are added
    even when blocking missed them, so the model still learns them)."""
    pairs, labels = [], []
    id_pairs = []  # (s1_entity_id, other_entity_id) parallel to pairs
    for i, rec in enumerate(s1_records):
        if rec.entity_id not in entity_ids:
            continue
        matches = truth.get(rec.entity_id, set())
        got = cands_idx.get(i, set())
        for j in got:
            pairs.append((i, j))
            labels.append(1 if other_records[j].entity_id in matches else 0)
            id_pairs.append((rec.entity_id, other_records[j].entity_id))
        for mid in matches:  # truth pairs missed by blocking
            j = other_by_id.get(mid)
            if j is not None and j not in got:
                pairs.append((i, j))
                labels.append(1)
                id_pairs.append((rec.entity_id, mid))
    return pairs, labels, id_pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--val-fraction", type=float, default=None,
                    help="override the validation split fraction")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the tuned pair threshold")
    ap.add_argument("--min-top", type=float, default=None,
                    help="override the tuned entity-level min-top rule")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vector-topk", type=int, default=None,
                    help="top-k TF-IDF cosine neighbours per S1 entity unioned "
                         "into the key-blocking candidates (0 disables)")
    ap.add_argument("--per-entity-topk", type=int, default=None,
                    help="optional cap on candidates per entity, kept by cosine "
                         "score (0 = keep everything)")
    args = ap.parse_args()

    cfg = Config(seed=args.seed)
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.vector_topk is not None:
        cfg.vector_topk = args.vector_topk
    if args.per_entity_topk is not None:
        cfg.per_entity_topk = args.per_entity_topk

    # ------------------------------------------------------------------ load
    log("loading training data ...")
    train = Path(args.train_dir)
    test = Path(args.test_dir)

    s1_raw = load_source(train / "train_source1.tsv")
    s2_raw = load_source(train / "train_source2.tsv")
    s3_raw = load_source(train / "train_source3.tsv")
    truth = load_truth(train / "train_ground_truth.tsv")

    s1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s1_raw]
    s2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s2_raw]
    s3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in s3_raw]
    other = s2 + s3
    other_by_id = {r.entity_id: j for j, r in enumerate(other)}
    log(f"train: |S1|={len(s1)} |S2|={len(s2)} |S3|={len(s3)} "
        f"truth entities={len(truth)}")

    # -------------------------------------------------- validation split
    rng = random.Random(cfg.seed)
    all_s1_ids = [r.entity_id for r in s1]
    shuffled = all_s1_ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * cfg.val_fraction))
    val_ids = set(shuffled[:n_val])
    fit_ids = set(shuffled[n_val:])
    log(f"split: {len(fit_ids)} fit / {len(val_ids)} validation S1 entities")

    # ------------------------------------------------------------- blocking
    log("blocking over train corpus (keys + vector layer) ...")
    cands_idx, _, n_key, n_cand = run_blocking(s1, other, cfg)
    rec_all, f_all, t_all = blocking_recall(cands_idx, s1, other_by_id, truth)
    rec_val, f_val, t_val = blocking_recall(cands_idx, s1, other_by_id, truth,
                                            entity_ids=val_ids)
    log(f"blocking: {n_key} key pairs + vector layer -> {n_cand} candidate "
        f"pairs; recall(all)={rec_all:.4f} ({f_all}/{t_all}); "
        f"recall(val)={rec_val:.4f}")

    # ---------------------------------------------------- featurise + train
    log("loading test sources for TF-IDF fitting + scoring ...")
    t1_raw = load_source(test / "test_source1.tsv")
    t2_raw = load_source(test / "test_source2.tsv")
    t3_raw = load_source(test / "test_source3.tsv")
    t1 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t1_raw]
    t2 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t2_raw]
    t3 = [build_record(r.entity_id, r.name, r.address, r.country) for r in t3_raw]
    log(f"test: |S1|={len(t1)} |S2|={len(t2)} |S3|={len(t3)}")

    # the char n-gram TF-IDF spaces are fitted on train+test text jointly
    # (unsupervised; uses only the provided files)
    fe = PairFeaturizer(cfg).fit(s1 + other + t1 + t2 + t3)

    fit_pairs, fit_labels, _ = build_pairs(s1, other, other_by_id, cands_idx,
                                           truth, fit_ids)
    val_pairs, val_labels, val_id_pairs = build_pairs(
        s1, other, other_by_id, cands_idx, truth, val_ids)
    log(f"training pairs: {len(fit_pairs)} ({sum(fit_labels)} positive); "
        f"validation pairs: {len(val_pairs)}")

    X_fit = fe.transform(s1 + other, [(i, len(s1) + j) for i, j in fit_pairs])
    X_val = fe.transform(s1 + other, [(i, len(s1) + j) for i, j in val_pairs])

    log("training the pair classifier ...")
    model = train_model(X_fit, np_labels(fit_labels), cfg)

    val_p = predict_probs(model, X_val)
    val_scores: dict[str, list[tuple[str, float]]] = {e: [] for e in val_ids}
    for (s1_id, other_id), p in zip(val_id_pairs, val_p):
        val_scores[s1_id].append((other_id, float(p)))
    val_truth = {e: truth.get(e, set()) for e in val_ids}

    t, mt, val_f05 = sweep_thresholds(val_scores, val_truth)
    if args.threshold is not None:
        t = args.threshold
    if args.min_top is not None:
        mt = args.min_top
    log(f"tuned decision rule: threshold={t:.2f} min_top={mt:.2f} "
        f"-> validation macro F_0.5 = {val_f05:.4f}")

    # -------------------------------------------------------------- predict
    log("blocking over the test corpus (keys + vector layer) ...")
    test_other = t2 + t3
    t_cands_idx, t_cands_id, t_key, n_tcand = run_blocking(t1, test_other, cfg)
    log(f"test blocking: {t_key} key pairs + vector layer -> {n_tcand} "
        f"candidate pairs")

    t_pairs = [(i, len(t1) + j) for i in range(len(t1))
              for j in t_cands_idx.get(i, ())]
    X_test = fe.transform(t1 + test_other, t_pairs)
    log("scoring test candidates ...")
    t_probs = predict_probs(model, X_test)

    t_scores: dict[str, list[tuple[str, float]]] = {r.entity_id: [] for r in t1}
    for (i, j), p in zip(t_pairs, t_probs):
        s1_id = t1[i].entity_id
        t_scores[s1_id].append((test_other[j - len(t1)].entity_id, float(p)))

    preds = make_predictions(t_scores, t, mt)
    t1_ids = [r.entity_id for r in t1]

    # --------------------------------------------------------------- output
    out = Path(args.out_dir)
    write_output_tsv(out / "matching_results.tsv", t1_ids, preds)
    write_output_tsv(out / "candidate_pairs.tsv", t1_ids, t_cands_id,
                    id_column="candidate_entity_ids")
    log(f"wrote {out/'matching_results.tsv'} and {out/'candidate_pairs.tsv'}")

    n_pred = sum(len(v) for v in preds.values())
    n_singletons = sum(1 for v in preds.values() if not v)
    log(f"predicted {n_pred} matches; {n_singletons}/{len(t1)} singletons")

    log("validating output files ...")
    rc = validate_main(["--matching", str(out / "matching_results.tsv"),
                        "--candidate", str(out / "candidate_pairs.tsv"),
                        "--test-dir", str(test)])
    return rc


def np_labels(labels):
    import numpy as np
    return np.asarray(labels, dtype=np.int8)


if __name__ == "__main__":
    sys.exit(main())
